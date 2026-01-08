"""Conversation simulation component for DTS."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from backend.utils.logging import logger

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.dts.types import DialogueNode, NodeStatus, Strategy, UserIntent
from backend.core.dts.tree import generate_node_id
from backend.core.dts.utils import emit_event, log_phase
from backend.core.prompts import prompts
from backend.llm.types import Completion, Message

if TYPE_CHECKING:
    from backend.llm.client import LLM
    from backend.core.dts.tree import DialogueTree


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class LLMEmptyResponseError(Exception):
    """Raised when LLM returns empty/null content after retries exhausted."""

    pass


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TERMINATION_SIGNALS = [
    "goodbye",
    "bye",
    "i'm done",
    "i have to go",
    "thanks, bye",
    "i'm leaving",
    "end conversation",
    "stop",
    "quit",
    "exit",
    "i give up",
    "forget it",
    "never mind",
    "this isn't working",
    "i'm confused",
    "you're not helping",
    "i don't understand",
]


# -----------------------------------------------------------------------------
# Class: ConversationSimulator
# -----------------------------------------------------------------------------
class ConversationSimulator:
    """
    Simulates multi-turn conversations for branch expansion.

    Responsible for:
    - Simulating user responses (with or without intent guidance)
    - Generating assistant responses following strategies
    - Early termination detection
    - Parallel branch expansion with user intent forking
    """

    # --- Initialization ---

    def __init__(
        self,
        llm: LLM,
        goal: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_concurrency: int = 16,
        on_usage: Callable[[Any, str], None] | None = None,
        on_event: Callable[[str, dict[str, Any]], Any] | None = None,
        provider: str | None = None,
        reasoning_enabled: bool = False,
    ) -> None:
        """
        Initialize the simulator.

        Args:
            llm: LLM client for simulation.
            goal: Conversation goal for context.
            model: Model to use for simulation.
            temperature: Temperature for generation.
            max_concurrency: Maximum concurrent LLM calls.
            on_usage: Callback for token usage tracking (completion, phase).
            on_event: Async callback for emitting events to UI.
            provider: Provider preference for OpenRouter (e.g., "Fireworks").
            reasoning_enabled: Enable reasoning tokens for LLM calls.
        """
        self.llm = llm
        self.goal = goal
        self.model = model
        self.temperature = temperature
        self._sem = asyncio.Semaphore(max_concurrency)
        self._on_usage = on_usage
        self._on_event = on_event
        self.provider = provider
        self.reasoning_enabled = reasoning_enabled

    # --- Public Methods ---

    async def expand_nodes(
        self,
        nodes: list[DialogueNode],
        turns: int,
        intents_per_node: int = 1,
        tree: DialogueTree | None = None,
        generate_intents: Callable[[list[Message], int], Any] | None = None,
    ) -> list[DialogueNode]:
        """
        Expand nodes with multi-turn conversations.

        Args:
            nodes: Nodes to expand.
            turns: Number of conversation turns per expansion.
            intents_per_node: Number of user intents to fork (1 = no forking).
            tree: Optional tree to register forked children.
            generate_intents: Async function to generate intents.

        Returns:
            List of expanded (possibly forked) nodes.
        """
        if intents_per_node <= 1 or generate_intents is None:
            # No forking - simple linear expansion
            return await self._expand_linear_batch(nodes, turns)

        # With forking: scatter-gather pattern
        log_phase(
            logger,
            "FORK",
            f"Generating {intents_per_node} intents for {len(nodes)} nodes...",
            indent=1,
        )

        # Generate intents for all nodes in parallel
        intent_tasks = [
            generate_intents(node.messages, intents_per_node) for node in nodes
        ]
        all_intents = await asyncio.gather(*intent_tasks, return_exceptions=True)

        # Build expansion workload
        expansion_tasks = []
        fallback_nodes = []

        for node, intents_result in zip(nodes, all_intents):
            if isinstance(intents_result, Exception) or not intents_result:
                logger.warning(
                    f"Intent generation failed for {node.id}, linear expansion"
                )
                fallback_nodes.append(node)
                continue

            intents = intents_result
            strategy_name = node.strategy.tagline if node.strategy else "root"
            log_phase(
                logger, "FORK", f"'{strategy_name}': {len(intents)} intents", indent=2
            )

            for idx, intent in enumerate(intents):
                log_phase(
                    logger,
                    "FORK",
                    f"  [{intent.emotional_tone}] {intent.label}",
                    indent=2,
                )
                # Emit intent_generated event for UI
                self._emit(
                    "intent_generated",
                    {
                        "strategy": strategy_name,
                        "index": idx + 1,
                        "total": len(intents),
                        "label": intent.label,
                        "emotional_tone": intent.emotional_tone,
                        "cognitive_stance": intent.cognitive_stance,
                    },
                )

                # Create forked child
                child = DialogueNode(
                    id=generate_node_id(),
                    parent_id=node.id,
                    depth=node.depth + 1,
                    strategy=node.strategy,
                    user_intent=intent,
                    messages=list(node.messages),
                )

                if tree:
                    tree.add_child(node.id, child)

                expansion_tasks.append(self._expand_with_intent(child, turns, intent))

        # Add fallback linear expansions
        for node in fallback_nodes:
            expansion_tasks.append(self._expand_linear(node, turns))

        # Execute all expansions with as_completed
        log_phase(
            logger, "FORK", f"Expanding {len(expansion_tasks)} branches...", indent=1
        )

        expanded = []
        completed = 0
        failed = 0
        # Total timeout scales with task count (2 minutes per task)
        total_timeout = 120.0 * max(1, len(expansion_tasks))

        for coro in asyncio.as_completed(expansion_tasks, timeout=total_timeout):
            try:
                result = await coro
                if isinstance(result, DialogueNode):
                    expanded.append(result)
                    completed += 1
            except asyncio.TimeoutError:
                logger.warning("Expansion timed out")
                failed += 1
            except Exception as e:
                logger.error(f"Expansion error: {e}")
                failed += 1

        log_phase(
            logger, "FORK", f"Completed: {completed} | Failed: {failed}", indent=1
        )
        return expanded

    # --- Private Methods ---

    async def _expand_linear_batch(
        self, nodes: list[DialogueNode], turns: int
    ) -> list[DialogueNode]:
        """Expand multiple nodes linearly in parallel."""
        tasks = [self._expand_linear(node, turns) for node in nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        expanded = []
        for node, result in zip(nodes, results):
            if isinstance(result, Exception):
                logger.error(f"Error expanding {node.id}: {result}")
                node.status = NodeStatus.ERROR
            else:
                expanded.append(result)

        return expanded

    async def _run_turn(
        self,
        node: DialogueNode,
        history: list[Message],
        turn_idx: int,
        skip_user_simulation: bool = False,
        label: str | None = None,
    ) -> bool:
        """
        Run a single conversation turn (user + assistant exchange).

        Args:
            node: The dialogue node being expanded.
            history: Message history (modified in place).
            turn_idx: Zero-based turn index.
            skip_user_simulation: If True, skip user simulation (for rephrased first turn).
            label: Optional label for logging (e.g., intent label).

        Returns:
            True if turn completed successfully, False if expansion should stop.
        """
        turn_num = turn_idx + 1
        label_suffix = f"[{label}]" if label else ""

        if not skip_user_simulation:
            try:
                user_response = await self._simulate_user(history)
            except LLMEmptyResponseError:
                log_phase(
                    logger,
                    "EXPAND",
                    f"[Turn {turn_num}] FAILED: empty user response",
                    indent=2,
                )
                node.status = NodeStatus.ERROR
                node.prune_reason = "empty user response after retries"
                return False

            history.append(Message.user(user_response))
            log_phase(
                logger,
                "EXPAND",
                f"[Turn {turn_num}]{label_suffix} User: {user_response[:100]}...",
                indent=2,
            )

            if self._should_terminate(user_response):
                log_phase(logger, "EXPAND", f"[Turn {turn_num}] EARLY EXIT", indent=2)
                node.status = NodeStatus.TERMINAL
                return False

        try:
            assistant_response = await self._generate_assistant(history, node.strategy)
        except LLMEmptyResponseError:
            log_phase(
                logger,
                "EXPAND",
                f"[Turn {turn_num}] FAILED: empty assistant response",
                indent=2,
            )
            node.status = NodeStatus.ERROR
            node.prune_reason = "empty assistant response after retries"
            return False

        history.append(Message.assistant(assistant_response))
        log_phase(
            logger,
            "EXPAND",
            f"[Turn {turn_num}]{label_suffix} Assistant: {assistant_response[:100]}...",
            indent=2,
        )
        return True

    async def _expand_linear(self, node: DialogueNode, turns: int) -> DialogueNode:
        """Expand a single node linearly (no intent forking)."""
        history = list(node.messages)
        for turn_idx in range(turns):
            if not await self._run_turn(node, history, turn_idx):
                break
        node.messages = history
        return node

    async def _expand_with_intent(
        self, node: DialogueNode, turns: int, first_intent: UserIntent
    ) -> DialogueNode:
        """
        Expand with first user message modified by the intent.

        Rephrases the initial user message to incorporate the intent's emotional
        tone and cognitive stance, creating divergent branches from the start.
        """
        history = list(node.messages)

        # Rephrase the initial user message with the intent
        if history and history[0].role == "user":
            original_content = history[0].content or ""
            try:
                rephrased = await self._rephrase_initial_message(
                    original_content, first_intent
                )
                history[0] = Message.user(rephrased)
                log_phase(
                    logger,
                    "EXPAND",
                    f"[{first_intent.label}] Rephrased: {rephrased[:100]}...",
                    indent=2,
                )
            except LLMEmptyResponseError:
                log_phase(
                    logger,
                    "EXPAND",
                    f"[{first_intent.label}] Rephrase failed, using original",
                    indent=2,
                )

        for turn_idx in range(turns):
            # First turn skips user simulation (message already rephrased)
            skip_user = turn_idx == 0
            if not await self._run_turn(
                node, history, turn_idx, skip_user, first_intent.label
            ):
                break

        node.messages = history
        return node

    async def _rephrase_initial_message(
        self, original_message: str, intent: UserIntent
    ) -> str:
        """
        Rephrase the initial user message to incorporate the intent.

        This modifies the original message to reflect the intent's emotional
        tone and cognitive stance while preserving the core request/goal.
        """
        system_prompt, user_prompt = prompts.rephrase_with_intent(
            original_message=original_message,
            intent_label=intent.label,
            intent_description=intent.description,
            emotional_tone=intent.emotional_tone,
            cognitive_stance=intent.cognitive_stance,
        )

        messages = [Message.system(system_prompt), Message.user(user_prompt)]
        return await self._call_llm_with_retry(messages, phase="rephrase")

    async def _simulate_user(
        self,
        history: list[Message],
        intent: UserIntent | None = None,
    ) -> str:
        """Simulate a user response with retry on empty responses."""
        intent_dict = None
        if intent:
            intent_dict = {
                "label": intent.label,
                "description": intent.description,
                "emotional_tone": intent.emotional_tone,
                "cognitive_stance": intent.cognitive_stance,
            }

        system_prompt, user_prompt = prompts.user_simulation(
            conversation_goal=self.goal,
            user_intent=intent_dict,
        )

        # System prompt + conversation history + continuation request
        messages = (
            [Message.system(system_prompt)] + history + [Message.user(user_prompt)]
        )
        return await self._call_llm_with_retry(messages, phase="user")

    async def _generate_assistant(
        self,
        history: list[Message],
        strategy: Strategy | None,
    ) -> str:
        """Generate an assistant response with retry on empty responses."""
        system_prompt, user_prompt = prompts.assistant_continuation(
            conversation_goal=self.goal,
            strategy_tagline=strategy.tagline if strategy else "",
            strategy_description=strategy.description if strategy else "",
        )

        # System prompt + conversation history + continuation request
        messages = (
            [Message.system(system_prompt)] + history + [Message.user(user_prompt)]
        )
        return await self._call_llm_with_retry(messages, phase="assistant")

    async def _call_llm_with_retry(
        self,
        messages: list[Message],
        phase: str,
        max_retries: int = 3,
    ) -> str:
        """
        Call LLM with retry logic for empty responses.

        Uses tenacity for exponential backoff. Raises LLMEmptyResponseError
        if all retries are exhausted.
        """

        @retry(
            retry=retry_if_exception_type(LLMEmptyResponseError),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            reraise=True,
        )
        async def _attempt() -> str:
            completion = await self._call_llm(messages, phase=phase)
            content = completion.message.content
            if not content or not content.strip():
                logger.warning(f"Empty LLM response for phase '{phase}', retrying...")
                raise LLMEmptyResponseError(f"Empty response for phase '{phase}'")
            return content

        try:
            return await _attempt()
        except LLMEmptyResponseError:
            logger.error(
                f"Failed to get non-empty response for '{phase}' after {max_retries} retries"
            )
            raise

    def _should_terminate(self, user_response: str) -> bool:
        """Check if response signals conversation end."""
        response_lower = user_response.lower().strip()

        for signal in TERMINATION_SIGNALS:
            if signal in response_lower:
                return True

        # Short frustrated responses
        if len(response_lower) < 20 and any(
            w in response_lower for w in ["no", "nope", "wrong", "bad", "ugh"]
        ):
            return True

        return False

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event if callback is set (fire-and-forget)."""
        if self._on_event is not None:
            asyncio.create_task(emit_event(self._on_event, event_type, data, logger))

    async def _call_llm(
        self, messages: list[Message], phase: str = "other"
    ) -> Completion:
        """Make an LLM call."""
        async with self._sem:
            completion = await self.llm.complete(
                messages,
                model=self.model,
                temperature=self.temperature,
                provider=self.provider,
                reasoning_enabled=self.reasoning_enabled,
            )
            if self._on_usage:
                self._on_usage(completion, phase)
            return completion
