"""Shared pytest fixtures for the test suite."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.dts.config import DTSConfig
from backend.core.dts.types import (
    AggregatedScore,
    DialogueNode,
    NodeStats,
    NodeStatus,
    Strategy,
    UserIntent,
)
from backend.llm.types import Completion, Message, Usage

# -----------------------------------------------------------------------------
# LLM Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_usage() -> Usage:
    """Create a sample Usage object."""
    return Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)


@pytest.fixture
def sample_message() -> Message:
    """Create a sample assistant Message."""
    return Message(role="assistant", content="Hello, how can I help you?")


@pytest.fixture
def sample_completion(sample_message: Message, sample_usage: Usage) -> Completion:
    """Create a sample Completion object."""
    return Completion(
        message=sample_message,
        usage=sample_usage,
        model="test-model",
        finish_reason="stop",
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM client."""
    mock = MagicMock()
    mock._default_model = "test-model"
    mock.complete = AsyncMock()
    mock.stream = AsyncMock()
    return mock


@pytest.fixture
def mock_llm_with_response(mock_llm: MagicMock, sample_completion: Completion) -> MagicMock:
    """Create a mock LLM that returns a sample completion."""
    mock_llm.complete.return_value = sample_completion
    return mock_llm


# -----------------------------------------------------------------------------
# Strategy and Intent Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_strategy() -> Strategy:
    """Create a sample Strategy."""
    return Strategy(
        tagline="Empathetic Approach",
        description="Focus on understanding the user's emotional state first.",
    )


@pytest.fixture
def sample_user_intent() -> UserIntent:
    """Create a sample UserIntent."""
    return UserIntent(
        id="curious_engaged",
        label="Curious & Engaged",
        description="User is interested and asking questions",
        emotional_tone="enthusiastic",
        cognitive_stance="exploring",
    )


# -----------------------------------------------------------------------------
# Node Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_node_stats() -> NodeStats:
    """Create a sample NodeStats."""
    return NodeStats(
        visits=5,
        value_sum=35.0,
        value_mean=7.0,
        judge_scores=[7.0, 7.5, 6.5],
        aggregated_score=7.0,
    )


@pytest.fixture
def sample_dialogue_node(sample_strategy: Strategy) -> DialogueNode:
    """Create a sample DialogueNode."""
    return DialogueNode(
        id="test-node-1",
        parent_id=None,
        depth=0,
        status=NodeStatus.ACTIVE,
        strategy=sample_strategy,
        messages=[
            Message.user("Hello, I need help with Python"),
            Message.assistant("I'd be happy to help! What are you working on?"),
        ],
    )


@pytest.fixture
def sample_child_node(sample_strategy: Strategy, sample_user_intent: UserIntent) -> DialogueNode:
    """Create a sample child DialogueNode with intent."""
    return DialogueNode(
        id="test-node-2",
        parent_id="test-node-1",
        depth=1,
        status=NodeStatus.ACTIVE,
        strategy=sample_strategy,
        user_intent=sample_user_intent,
        messages=[
            Message.user("I'm having trouble with async code"),
            Message.assistant("Async can be tricky! Let's work through it together."),
        ],
    )


# -----------------------------------------------------------------------------
# Score Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_aggregated_score() -> AggregatedScore:
    """Create a sample AggregatedScore."""
    return AggregatedScore(
        individual_scores=[7.0, 7.5, 6.5],
        aggregated_score=7.0,
        pass_threshold=5.0,
        pass_votes=3,
        passed=True,
    )


@pytest.fixture
def zero_aggregated_score() -> AggregatedScore:
    """Create a zero AggregatedScore."""
    return AggregatedScore.zero(threshold=5.0)


# -----------------------------------------------------------------------------
# Config Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_dts_config() -> DTSConfig:
    """Create a sample DTSConfig."""
    return DTSConfig(
        goal="Help the user debug their Python code",
        first_message="I'm having trouble with my Python code",
        init_branches=3,
        turns_per_branch=2,
        user_intents_per_branch=2,
        scoring_mode="comparative",
        prune_threshold=5.0,
        model="test-model",
    )


# -----------------------------------------------------------------------------
# Environment Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_env_vars() -> dict[str, str]:
    """Mock environment variables."""
    return {
        "OPENAI_API_KEY": "test-api-key",
        "OPENAI_BASE_URL": "https://api.test.com/v1",
        "LLM_NAME": "test-model",
    }


@pytest.fixture
def patch_config(mock_env_vars: dict[str, str]):
    """Patch the config module with mock values."""
    with patch.dict("os.environ", mock_env_vars, clear=False):
        yield


# -----------------------------------------------------------------------------
# Async Helpers
# -----------------------------------------------------------------------------


@pytest.fixture
def async_mock() -> AsyncMock:
    """Create a generic AsyncMock."""
    return AsyncMock()


async def async_iter(items: list[Any]) -> AsyncIterator[Any]:
    """Create an async iterator from a list."""
    for item in items:
        yield item
