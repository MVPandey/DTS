"""Tests for backend/api/schemas.py."""

import pytest
from pydantic import ValidationError

from backend.api.schemas import (
    ErrorData,
    EventMessage,
    NodeAddedData,
    NodeUpdatedData,
    PhaseData,
    RoundStartedData,
    SearchRequest,
    SearchStartedData,
    StrategyGeneratedData,
)

# -----------------------------------------------------------------------------
# SearchRequest Tests
# -----------------------------------------------------------------------------


class TestSearchRequest:
    """Tests for SearchRequest schema."""

    def test_valid_request(self) -> None:
        """Test creating a valid request."""
        request = SearchRequest(
            goal="Help user debug code",
            first_message="My Python code isn't working",
        )

        assert request.goal == "Help user debug code"
        assert request.first_message == "My Python code isn't working"

    def test_default_values(self) -> None:
        """Test that default values are applied."""
        request = SearchRequest(
            goal="Test",
            first_message="Test",
        )

        assert request.init_branches == 6
        assert request.turns_per_branch == 5
        assert request.user_intents_per_branch == 3
        assert request.scoring_mode == "comparative"
        assert request.prune_threshold == 6.5
        assert request.rounds == 1
        assert request.deep_research is False
        assert request.strategy_model is None
        assert request.simulator_model is None
        assert request.judge_model is None

    def test_custom_values(self) -> None:
        """Test with custom values."""
        request = SearchRequest(
            goal="Test",
            first_message="Test",
            init_branches=10,
            turns_per_branch=3,
            user_intents_per_branch=5,
            scoring_mode="absolute",
            prune_threshold=7.5,
            rounds=3,
            deep_research=True,
            strategy_model="gpt-4",
            simulator_model="gpt-3.5-turbo",
            judge_model="claude-3",
        )

        assert request.init_branches == 10
        assert request.turns_per_branch == 3
        assert request.user_intents_per_branch == 5
        assert request.scoring_mode == "absolute"
        assert request.prune_threshold == 7.5
        assert request.rounds == 3
        assert request.deep_research is True

    def test_init_branches_validation(self) -> None:
        """Test init_branches validation."""
        # Valid range
        request = SearchRequest(goal="Test", first_message="Test", init_branches=1)
        assert request.init_branches == 1

        request = SearchRequest(goal="Test", first_message="Test", init_branches=20)
        assert request.init_branches == 20

        # Invalid - too low
        with pytest.raises(ValidationError):
            SearchRequest(goal="Test", first_message="Test", init_branches=0)

        # Invalid - too high
        with pytest.raises(ValidationError):
            SearchRequest(goal="Test", first_message="Test", init_branches=21)

    def test_prune_threshold_validation(self) -> None:
        """Test prune_threshold validation."""
        # Valid range
        request = SearchRequest(goal="Test", first_message="Test", prune_threshold=0.0)
        assert request.prune_threshold == 0.0

        request = SearchRequest(goal="Test", first_message="Test", prune_threshold=10.0)
        assert request.prune_threshold == 10.0

        # Invalid - negative
        with pytest.raises(ValidationError):
            SearchRequest(goal="Test", first_message="Test", prune_threshold=-1.0)

        # Invalid - too high
        with pytest.raises(ValidationError):
            SearchRequest(goal="Test", first_message="Test", prune_threshold=11.0)

    def test_rounds_validation(self) -> None:
        """Test rounds validation."""
        request = SearchRequest(goal="Test", first_message="Test", rounds=10)
        assert request.rounds == 10

        with pytest.raises(ValidationError):
            SearchRequest(goal="Test", first_message="Test", rounds=0)

        with pytest.raises(ValidationError):
            SearchRequest(goal="Test", first_message="Test", rounds=11)

    def test_missing_required_fields(self) -> None:
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            SearchRequest(first_message="Test")  # Missing goal

        with pytest.raises(ValidationError):
            SearchRequest(goal="Test")  # Missing first_message


# -----------------------------------------------------------------------------
# EventMessage Tests
# -----------------------------------------------------------------------------


class TestEventMessage:
    """Tests for EventMessage schema."""

    def test_valid_event(self) -> None:
        """Test creating a valid event message."""
        event = EventMessage(type="test_event", data={"key": "value"})

        assert event.type == "test_event"
        assert event.data == {"key": "value"}

    def test_default_data(self) -> None:
        """Test default empty data."""
        event = EventMessage(type="simple_event")

        assert event.type == "simple_event"
        assert event.data == {}

    def test_complex_data(self) -> None:
        """Test with complex nested data."""
        event = EventMessage(
            type="complex",
            data={
                "nested": {"deep": {"value": 123}},
                "list": [1, 2, 3],
            },
        )

        assert event.data["nested"]["deep"]["value"] == 123


# -----------------------------------------------------------------------------
# ErrorData Tests
# -----------------------------------------------------------------------------


class TestErrorData:
    """Tests for ErrorData schema."""

    def test_error_with_message(self) -> None:
        """Test creating error with message only."""
        error = ErrorData(message="Something went wrong")

        assert error.message == "Something went wrong"
        assert error.code is None

    def test_error_with_code(self) -> None:
        """Test creating error with code."""
        error = ErrorData(message="Not found", code="NOT_FOUND")

        assert error.message == "Not found"
        assert error.code == "NOT_FOUND"


# -----------------------------------------------------------------------------
# SearchStartedData Tests
# -----------------------------------------------------------------------------


class TestSearchStartedData:
    """Tests for SearchStartedData schema."""

    def test_valid_data(self) -> None:
        """Test creating valid search started data."""
        data = SearchStartedData(
            goal="Help user",
            first_message="Hello",
            total_rounds=2,
            config={"init_branches": 6, "turns_per_branch": 5},
        )

        assert data.goal == "Help user"
        assert data.first_message == "Hello"
        assert data.total_rounds == 2
        assert data.config["init_branches"] == 6


# -----------------------------------------------------------------------------
# PhaseData Tests
# -----------------------------------------------------------------------------


class TestPhaseData:
    """Tests for PhaseData schema."""

    def test_valid_phases(self) -> None:
        """Test all valid phase values."""
        valid_phases = [
            "initializing",
            "generating_strategies",
            "expanding",
            "scoring",
            "pruning",
            "complete",
        ]

        for phase in valid_phases:
            data = PhaseData(phase=phase, message=f"In {phase} phase")  # type: ignore[arg-type]
            assert data.phase == phase

    def test_invalid_phase(self) -> None:
        """Test that invalid phase raises error."""
        with pytest.raises(ValidationError):
            PhaseData(phase="invalid_phase", message="Test")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# StrategyGeneratedData Tests
# -----------------------------------------------------------------------------


class TestStrategyGeneratedData:
    """Tests for StrategyGeneratedData schema."""

    def test_valid_data(self) -> None:
        """Test creating valid strategy data."""
        data = StrategyGeneratedData(
            index=1,
            total=5,
            tagline="Empathetic Approach",
            description="Focus on understanding emotions",
        )

        assert data.index == 1
        assert data.total == 5
        assert data.tagline == "Empathetic Approach"


# -----------------------------------------------------------------------------
# NodeAddedData Tests
# -----------------------------------------------------------------------------


class TestNodeAddedData:
    """Tests for NodeAddedData schema."""

    def test_valid_data(self) -> None:
        """Test creating valid node added data."""
        data = NodeAddedData(
            id="node-123",
            parent_id="parent-456",
            depth=2,
            status="active",
            strategy="Technical Approach",
            user_intent="Curious",
            message_count=4,
        )

        assert data.id == "node-123"
        assert data.parent_id == "parent-456"
        assert data.depth == 2

    def test_optional_fields(self) -> None:
        """Test with optional fields as None."""
        data = NodeAddedData(
            id="node-123",
            parent_id=None,
            depth=0,
            status="active",
            strategy=None,
            user_intent=None,
            message_count=1,
        )

        assert data.parent_id is None
        assert data.strategy is None


# -----------------------------------------------------------------------------
# NodeUpdatedData Tests
# -----------------------------------------------------------------------------


class TestNodeUpdatedData:
    """Tests for NodeUpdatedData schema."""

    def test_valid_data(self) -> None:
        """Test creating valid node updated data."""
        data = NodeUpdatedData(
            id="node-123",
            status="scored",
            score=7.5,
            individual_scores=[7.0, 8.0, 7.5],
            passed=True,
        )

        assert data.id == "node-123"
        assert data.score == 7.5
        assert len(data.individual_scores) == 3
        assert data.passed is True


# -----------------------------------------------------------------------------
# RoundStartedData Tests
# -----------------------------------------------------------------------------


class TestRoundStartedData:
    """Tests for RoundStartedData schema."""

    def test_valid_data(self) -> None:
        """Test creating valid round started data."""
        data = RoundStartedData(round=2, total_rounds=5)

        assert data.round == 2
        assert data.total_rounds == 5
