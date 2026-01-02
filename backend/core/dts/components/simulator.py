"""Conversation simulation component for DTS."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

from backend.core.dts.types import DialogueNode, NodeStatus, Strategy, UserIntent
from backend.core.dts.tree import generate_node_id
from backend.core.prompts import prompts
from backend.llm.types import Completion, Message

if TYPE_CHECKING:
    from backend.llm.client import LLM
    from backend.core.dts.tree import DialogueTree

logger = logging.getLogger(__name__)


def _log(phase: str, message: str, indent: int = 0) -> None:
    """Print a formatted log message."""
    prefix = "  " * indent
    print(f"[DTS:{phase}] {prefix}{message}")


# Signals indicating conversation should terminate early
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


class ConversationSimulator:
    """
    Simulates multi-turn conversations for branch expansion.

    Responsible for:
    - Simulating user responses (with or without intent guidance)
    - Generating assistant responses following strategies
    - Early termination detection
    - Parallel branch expansion with user intent forking
    """

    def __init__(
        self,
        llm: LLM,
        goal: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_concurrency: int = 16,
        on_usage: Callable[[Any, str], None] | None = None,
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
        """
        self.llm = llm
        self.goal = goal
        self.model = model
        self.temperature = temperature
        self._sem = asyncio.Semaphore(max_concurrency)
        self._on_usage = on_usage

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
        _log(
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
            _log("FORK", f"'{strategy_name}': {len(intents)} intents", indent=2)

            for intent in intents:
                _log("FORK", f"  [{intent.emotional_tone}] {intent.label}", indent=2)

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
        _log("FORK", f"Expanding {len(expansion_tasks)} branches...", indent=1)

        expanded = []
        completed = 0
        failed = 0
        timeout_per_task = 120.0

        for coro in asyncio.as_completed(
            expansion_tasks, timeout=timeout_per_task * len(expansion_tasks)
        ):
            try:
                result = await asyncio.wait_for(coro, timeout=timeout_per_task)
                if isinstance(result, DialogueNode):
                    expanded.append(result)
                    completed += 1
            except asyncio.TimeoutError:
                logger.warning("Expansion timed out")
                failed += 1
            except Exception as e:
                logger.error(f"Expansion error: {e}")
                failed += 1

        _log("FORK", f"Completed: {completed} | Failed: {failed}", indent=1)
        return expanded

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

    async def _expand_linear(self, node: DialogueNode, turns: int) -> DialogueNode:
        """Expand a single node linearly (no intent forking)."""
        history = list(node.messages)

        for turn_idx in range(turns):
            # Simulate user
            user_response = await self._simulate_user(history)
            history.append(Message.user(user_response))
            _log(
                "EXPAND",
                f"[Turn {turn_idx + 1}] User: {user_response[:100]}...",
                indent=2,
            )

            # Check early termination
            if self._should_terminate(user_response):
                _log("EXPAND", f"[Turn {turn_idx + 1}] EARLY EXIT", indent=2)
                node.status = NodeStatus.TERMINAL
                break

            # Generate assistant
            assistant_response = await self._generate_assistant(history, node.strategy)
            history.append(Message.assistant(assistant_response))
            _log(
                "EXPAND",
                f"[Turn {turn_idx + 1}] Assistant: {assistant_response[:100]}...",
                indent=2,
            )

        node.messages = history
        return node

    async def _expand_with_intent(
        self, node: DialogueNode, turns: int, first_intent: UserIntent
    ) -> DialogueNode:
        """Expand with first user response following a specific intent."""
        history = list(node.messages)

        for turn_idx in range(turns):
            # Simulate user (with intent on first turn)
            if turn_idx == 0:
                user_response = await self._simulate_user(history, first_intent)
                _log(
                    "EXPAND",
                    f"[Turn 1][{first_intent.label}] User: {user_response[:100]}...",
                    indent=2,
                )
            else:
                user_response = await self._simulate_user(history)
                _log(
                    "EXPAND",
                    f"[Turn {turn_idx + 1}] User: {user_response[:100]}...",
                    indent=2,
                )

            history.append(Message.user(user_response))

            # Check early termination
            if self._should_terminate(user_response):
                _log("EXPAND", f"[Turn {turn_idx + 1}] EARLY EXIT", indent=2)
                node.status = NodeStatus.TERMINAL
                break

            # Generate assistant
            assistant_response = await self._generate_assistant(history, node.strategy)
            history.append(Message.assistant(assistant_response))
            _log(
                "EXPAND",
                f"[Turn {turn_idx + 1}] Assistant: {assistant_response[:100]}...",
                indent=2,
            )

        node.messages = history
        return node

    async def _simulate_user(
        self,
        history: list[Message],
        intent: UserIntent | None = None,
    ) -> str:
        """Simulate a user response."""
        intent_dict = None
        if intent:
            intent_dict = {
                "label": intent.label,
                "description": intent.description,
                "emotional_tone": intent.emotional_tone,
                "cognitive_stance": intent.cognitive_stance,
            }

        system_prompt = prompts.user_simulation(
            conversation_goal=self.goal,
            conversation_history=self._format_history(history),
            user_intent=intent_dict,
        )

        messages = [Message.system(system_prompt)] + history
        completion = await self._call_llm(messages, phase="user")
        return completion.message.content or ""

    async def _generate_assistant(
        self,
        history: list[Message],
        strategy: Strategy | None,
    ) -> str:
        """Generate an assistant response."""
        system_prompt = prompts.assistant_continuation(
            conversation_goal=self.goal,
            conversation_history=self._format_history(history),
            strategy_tagline=strategy.tagline if strategy else "",
            strategy_description=strategy.description if strategy else "",
        )

        messages = [Message.system(system_prompt)] + history
        completion = await self._call_llm(messages, phase="assistant")
        return completion.message.content or ""

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

    def _format_history(self, messages: list[Message]) -> str:
        """Format messages for prompts."""
        lines = []
        for msg in messages:
            role = msg.role.capitalize()
            content = msg.content or ""
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    async def _call_llm(
        self, messages: list[Message], phase: str = "other"
    ) -> Completion:
        """Make an LLM call."""
        async with self._sem:
            completion = await self.llm.complete(
                messages,
                model=self.model,
                temperature=self.temperature,
            )
            if self._on_usage:
                self._on_usage(completion, phase)
            return completion
