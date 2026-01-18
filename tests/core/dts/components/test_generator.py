"""Tests for backend/core/dts/components/generator.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.dts.components.generator import FIXED_INTENT, StrategyGenerator
from backend.core.dts.types import Strategy, UserIntent
from backend.llm.types import Completion, Message, Usage

# -----------------------------------------------------------------------------
# FIXED_INTENT Tests
# -----------------------------------------------------------------------------


class TestFixedIntent:
    """Tests for the FIXED_INTENT constant."""

    def test_fixed_intent_properties(self) -> None:
        """Test FIXED_INTENT has expected properties."""
        assert isinstance(FIXED_INTENT, UserIntent)
        assert FIXED_INTENT.id == "fixed_engaged_critic"
        assert FIXED_INTENT.label == "Engaged Critic"
        assert "skepticism" in FIXED_INTENT.description.lower()
        assert FIXED_INTENT.emotional_tone == "curious but skeptical"
        assert "analytical" in FIXED_INTENT.cognitive_stance


# -----------------------------------------------------------------------------
# StrategyGenerator Initialization Tests
# -----------------------------------------------------------------------------


class TestStrategyGeneratorInit:
    """Tests for StrategyGenerator initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        mock_llm = MagicMock()
        generator = StrategyGenerator(llm=mock_llm, goal="Test goal")

        assert generator.llm is mock_llm
        assert generator.goal == "Test goal"
        assert generator.model is None
        assert generator.temperature == 0.7
        assert generator.provider is None
        assert generator.reasoning_enabled is False

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        mock_llm = MagicMock()
        generator = StrategyGenerator(
            llm=mock_llm,
            goal="Custom goal",
            model="gpt-4",
            temperature=0.5,
            max_concurrency=8,
            provider="Fireworks",
            reasoning_enabled=True,
        )

        assert generator.model == "gpt-4"
        assert generator.temperature == 0.5
        assert generator.provider == "Fireworks"
        assert generator.reasoning_enabled is True


# -----------------------------------------------------------------------------
# generate_strategies Tests
# -----------------------------------------------------------------------------


class TestGenerateStrategies:
    """Tests for StrategyGenerator.generate_strategies()."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def generator(self, mock_llm: MagicMock) -> StrategyGenerator:
        """Create a StrategyGenerator with mock LLM."""
        return StrategyGenerator(
            llm=mock_llm,
            goal="Help user with Python",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_generate_strategies_success(
        self, generator: StrategyGenerator, mock_llm: MagicMock
    ) -> None:
        """Test successful strategy generation."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            data={
                "goal": "Help user with Python",
                "nodes": {
                    "Technical Deep Dive": "Focus on code details and implementation",
                    "Conceptual Overview": "Explain high-level concepts first",
                    "Problem-Based": "Work through specific examples",
                },
            },
        )

        strategies = await generator.generate_strategies(
            first_message="I need help with async code",
            count=3,
        )

        assert len(strategies) == 3
        assert all(isinstance(s, Strategy) for s in strategies)
        assert any(s.tagline == "Technical Deep Dive" for s in strategies)

    @pytest.mark.asyncio
    async def test_generate_strategies_with_research_context(
        self, generator: StrategyGenerator, mock_llm: MagicMock
    ) -> None:
        """Test strategy generation with deep research context."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data={
                "goal": "Test",
                "nodes": {"Strategy 1": "Description 1"},
            },
        )

        await generator.generate_strategies(
            first_message="Test message",
            count=1,
            deep_research_context="Some research findings...",
        )

        # Verify complete was called with structured_output=True
        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args.kwargs
        assert call_kwargs.get("structured_output") is True

    @pytest.mark.asyncio
    async def test_generate_strategies_empty_result_raises(
        self, generator: StrategyGenerator, mock_llm: MagicMock
    ) -> None:
        """Test that empty result raises RuntimeError."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data=None,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await generator.generate_strategies("Test", 3)

        assert "Strategy generation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_strategies_tracks_usage(
        self, generator: StrategyGenerator, mock_llm: MagicMock
    ) -> None:
        """Test that token usage is tracked."""
        usage_callback = MagicMock()
        generator._on_usage = usage_callback

        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            data={"goal": "Test", "nodes": {"Strategy 1": "Desc"}},
        )

        await generator.generate_strategies("Test", 1)

        usage_callback.assert_called_once()
        call_args = usage_callback.call_args
        assert call_args[0][1] == "strategy"  # Phase name


# -----------------------------------------------------------------------------
# generate_intents Tests
# -----------------------------------------------------------------------------


class TestGenerateIntents:
    """Tests for StrategyGenerator.generate_intents()."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def generator(self, mock_llm: MagicMock) -> StrategyGenerator:
        """Create a StrategyGenerator with mock LLM."""
        return StrategyGenerator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_generate_intents_success(
        self, generator: StrategyGenerator, mock_llm: MagicMock
    ) -> None:
        """Test successful intent generation."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            usage=Usage(prompt_tokens=50, completion_tokens=100, total_tokens=150),
            data={
                "intents": [
                    {
                        "id": "curious_1",
                        "label": "Curious Explorer",
                        "description": "Wants to learn more",
                        "emotional_tone": "enthusiastic",
                        "cognitive_stance": "exploring",
                    },
                    {
                        "id": "skeptical_1",
                        "label": "Skeptical Questioner",
                        "description": "Doubts the approach",
                        "emotional_tone": "skeptical",
                        "cognitive_stance": "challenging",
                    },
                ]
            },
        )

        history = [
            Message.user("Hello"),
            Message.assistant("Hi, how can I help?"),
        ]

        intents = await generator.generate_intents(history, count=2)

        assert len(intents) == 2
        assert all(isinstance(i, UserIntent) for i in intents)
        assert intents[0].label == "Curious Explorer"
        assert intents[1].emotional_tone == "skeptical"

    @pytest.mark.asyncio
    async def test_generate_intents_empty_result_raises(
        self, generator: StrategyGenerator, mock_llm: MagicMock
    ) -> None:
        """Test that empty result raises RuntimeError."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data=None,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await generator.generate_intents([], 2)

        assert "Intent generation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_intents_handles_malformed_data(
        self, generator: StrategyGenerator, mock_llm: MagicMock
    ) -> None:
        """Test that malformed intent data is handled gracefully."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data={
                "intents": [
                    {"id": "good", "label": "Good Intent", "description": "Valid"},
                    {"invalid": "data"},  # Missing required fields
                ]
            },
        )

        intents = await generator.generate_intents([], 2)

        # Should get at least the valid intent
        assert len(intents) >= 1
        assert intents[0].label == "Good Intent"

    @pytest.mark.asyncio
    async def test_generate_intents_tracks_usage(
        self, generator: StrategyGenerator, mock_llm: MagicMock
    ) -> None:
        """Test that token usage is tracked for intent generation."""
        usage_callback = MagicMock()
        generator._on_usage = usage_callback

        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            usage=Usage(prompt_tokens=50, completion_tokens=100, total_tokens=150),
            data={"intents": [{"id": "i1", "label": "Intent 1", "description": "Desc"}]},
        )

        await generator.generate_intents([], 1)

        usage_callback.assert_called_once()
        call_args = usage_callback.call_args
        assert call_args[0][1] == "intent"  # Phase name


# -----------------------------------------------------------------------------
# Concurrency Tests
# -----------------------------------------------------------------------------


class TestGeneratorConcurrency:
    """Tests for generator concurrency handling."""

    @pytest.mark.asyncio
    async def test_respects_semaphore(self) -> None:
        """Test that generator respects max_concurrency."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=Completion(
                message=Message.assistant("{}"),
                data={"goal": "Test", "nodes": {"S1": "D1"}},
            )
        )

        generator = StrategyGenerator(
            llm=mock_llm,
            goal="Test",
            max_concurrency=2,
        )

        # Verify semaphore is created with correct value
        assert generator._sem._value == 2
