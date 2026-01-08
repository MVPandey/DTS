"""Strategy and intent generation component for DTS."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from backend.core.dts.retry import llm_retry
from backend.utils.logging import logger
from backend.core.dts.types import Strategy, UserIntent
from backend.core.dts.utils import format_message_history
from backend.core.prompts import prompts
from backend.llm.types import Message

if TYPE_CHECKING:
    from backend.llm.client import LLM


# -----------------------------------------------------------------------------
# Fixed Intent (used when user_variability=False)
# -----------------------------------------------------------------------------
FIXED_INTENT = UserIntent(
    id="fixed_engaged_critic",
    label="Engaged Critic",
    description="A thoughtful user who engages constructively while maintaining healthy skepticism",
    emotional_tone="curious but skeptical",
    cognitive_stance="analytical, asks probing questions",
)


# -----------------------------------------------------------------------------
# Class: StrategyGenerator
# -----------------------------------------------------------------------------
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
        provider: str | None = None,
        reasoning_enabled: bool = False,
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
            provider: Provider preference for OpenRouter (e.g., "Fireworks").
            reasoning_enabled: Enable reasoning tokens for LLM calls.
        """
        self.llm = llm
        self.goal = goal
        self.model = model
        self.temperature = temperature
        self._sem = asyncio.Semaphore(max_concurrency)
        self._on_usage = on_usage
        self.provider = provider
        self.reasoning_enabled = reasoning_enabled

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
        system_prompt, user_prompt = prompts.conversation_tree_generator(
            num_nodes=count,
            conversation_goal=self.goal,
            conversation_context=first_message,
            deep_research_context=deep_research_context,
        )

        result = await self._call_llm_json(system_prompt, user_prompt, phase="strategy")

        if not result:
            raise RuntimeError("Strategy generation failed after retries")

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
        system_prompt, user_prompt = prompts.user_intent_generator(
            num_intents=count,
            conversation_goal=self.goal,
            conversation_history=format_message_history(history),
        )

        result = await self._call_llm_json(system_prompt, user_prompt, phase="intent")

        if not result:
            raise RuntimeError("Intent generation failed after retries")

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

    # --- Private Methods ---

    async def _call_llm_json(
        self, system_prompt: str, user_prompt: str, phase: str = "other"
    ) -> dict[str, Any] | None:
        """Make an LLM call expecting JSON output with retry."""
        async with self._sem:
            return await self._call_llm_json_inner(system_prompt, user_prompt, phase)

    @llm_retry(max_attempts=3)
    async def _call_llm_json_inner(
        self, system_prompt: str, user_prompt: str, phase: str
    ) -> dict[str, Any] | None:
        """Inner LLM call with retry logic."""
        messages = [
            Message.system(system_prompt),
            Message.user(user_prompt),
        ]
        completion = await self.llm.complete(
            messages,
            model=self.model,
            temperature=self.temperature,
            structured_output=True,
            provider=self.provider,
            reasoning_enabled=self.reasoning_enabled,
        )
        if self._on_usage:
            self._on_usage(completion, phase)
        return completion.data
