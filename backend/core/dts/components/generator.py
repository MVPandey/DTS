"""Strategy and intent generation component for DTS."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

from backend.core.dts.types import Strategy, UserIntent
from backend.core.prompts import prompts
from backend.llm.types import Message

if TYPE_CHECKING:
    from backend.llm.client import LLM

logger = logging.getLogger(__name__)


def _log(phase: str, message: str, indent: int = 0) -> None:
    """Print a formatted log message."""
    prefix = "  " * indent
    print(f"[DTS:{phase}] {prefix}{message}")


class StrategyGenerator:
    """
    Generates conversation strategies and user intents.

    Responsible for:
    - Creating diverse initial branch strategies from a goal
    - Generating user response intents for branch forking
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
        Initialize the generator.

        Args:
            llm: LLM client for generation.
            goal: Conversation goal for context.
            model: Model to use for generation.
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

    async def generate_strategies(
        self,
        first_message: str,
        count: int,
        deep_research_context: str | None = None,
    ) -> list[Strategy]:
        """
        Generate diverse conversation strategies.

        Args:
            first_message: Initial user message for context.
            count: Number of strategies to generate.
            deep_research_context: Optional research context.

        Returns:
            List of Strategy objects.
        """
        prompt = prompts.conversation_tree_generator(
            num_nodes=count,
            conversation_goal=self.goal,
            conversation_context=first_message,
            deep_research_context=deep_research_context,
        )

        result = await self._call_llm_json(prompt, phase="strategy")

        if not result:
            logger.error("Failed to generate strategies")
            return []

        strategies = []
        nodes_data = result.get("nodes", {})

        for tagline, description in nodes_data.items():
            strategies.append(Strategy(tagline=tagline, description=str(description)))

        return strategies

    async def generate_intents(
        self,
        history: list[Message],
        count: int,
    ) -> list[UserIntent]:
        """
        Generate diverse user response intents.

        Args:
            history: Conversation history for context.
            count: Number of intents to generate.

        Returns:
            List of UserIntent objects.
        """
        prompt = prompts.user_intent_generator(
            num_intents=count,
            conversation_goal=self.goal,
            conversation_history=self._format_history(history),
        )

        result = await self._call_llm_json(prompt, phase="intent")

        if not result:
            logger.warning("Failed to generate intents")
            return []

        intents = []
        intents_data = result.get("intents", [])

        for data in intents_data:
            try:
                intents.append(
                    UserIntent(
                        id=data.get("id", "unknown"),
                        label=data.get("label", "Unknown"),
                        description=data.get("description", ""),
                        emotional_tone=data.get("emotional_tone", "neutral"),
                        cognitive_stance=data.get("cognitive_stance", "neutral"),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse intent: {e}")

        return intents

    async def generate_intents_batch(
        self,
        histories: list[list[Message]],
        count: int,
    ) -> list[list[UserIntent]]:
        """
        Generate intents for multiple histories in parallel.

        Args:
            histories: List of conversation histories.
            count: Number of intents per history.

        Returns:
            List of intent lists (one per history).
        """
        tasks = [self.generate_intents(h, count) for h in histories]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Intent generation failed: {result}")
                output.append([])
            else:
                output.append(result)

        return output

    def _format_history(self, messages: list[Message]) -> str:
        """Format messages for prompts."""
        lines = []
        for msg in messages:
            role = msg.role.capitalize()
            content = msg.content or ""
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    async def _call_llm_json(
        self, prompt: str, phase: str = "other"
    ) -> dict[str, Any] | None:
        """Make an LLM call expecting JSON output."""
        async with self._sem:
            try:
                completion = await self.llm.complete(
                    [Message.user(prompt)],
                    model=self.model,
                    temperature=self.temperature,
                    structured_output=True,
                )
                if self._on_usage:
                    self._on_usage(completion, phase)
                return completion.data
            except Exception as e:
                logger.error(f"JSON LLM call failed: {e}")
                return None
