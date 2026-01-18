"""Tests for backend/core/dts/components/evaluator.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.dts.components.evaluator import TrajectoryEvaluator
from backend.core.dts.types import AggregatedScore, DialogueNode, Strategy, UserIntent
from backend.llm.types import Completion, Message, Usage

# -----------------------------------------------------------------------------
# TrajectoryEvaluator Initialization Tests
# -----------------------------------------------------------------------------


class TestEvaluatorInit:
    """Tests for TrajectoryEvaluator initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        mock_llm = MagicMock()
        evaluator = TrajectoryEvaluator(llm=mock_llm, goal="Test goal")

        assert evaluator.llm is mock_llm
        assert evaluator.goal == "Test goal"
        assert evaluator.model is None
        assert evaluator.judge_temperature == 0.3
        assert evaluator.prune_threshold == 6.5
        assert evaluator.deep_research_context is None

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        mock_llm = MagicMock()
        evaluator = TrajectoryEvaluator(
            llm=mock_llm,
            goal="Custom goal",
            model="gpt-4",
            judge_temperature=0.2,
            prune_threshold=7.0,
            max_concurrency=8,
            deep_research_context="Research findings...",
            provider="Fireworks",
            reasoning_enabled=True,
        )

        assert evaluator.model == "gpt-4"
        assert evaluator.judge_temperature == 0.2
        assert evaluator.prune_threshold == 7.0
        assert evaluator.deep_research_context == "Research findings..."
        assert evaluator.provider == "Fireworks"
        assert evaluator.reasoning_enabled is True


# -----------------------------------------------------------------------------
# set_research_context Tests
# -----------------------------------------------------------------------------


class TestSetResearchContext:
    """Tests for setting research context."""

    def test_set_research_context(self) -> None:
        """Test setting research context."""
        evaluator = TrajectoryEvaluator(llm=MagicMock(), goal="Test")
        assert evaluator.deep_research_context is None

        evaluator.set_research_context("New research context")
        assert evaluator.deep_research_context == "New research context"

    def test_clear_research_context(self) -> None:
        """Test clearing research context."""
        evaluator = TrajectoryEvaluator(
            llm=MagicMock(),
            goal="Test",
            deep_research_context="Initial context",
        )

        evaluator.set_research_context(None)
        assert evaluator.deep_research_context is None


# -----------------------------------------------------------------------------
# evaluate_absolute Tests
# -----------------------------------------------------------------------------


class TestEvaluateAbsolute:
    """Tests for TrajectoryEvaluator.evaluate_absolute()."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def evaluator(self, mock_llm: MagicMock) -> TrajectoryEvaluator:
        """Create an evaluator with mock LLM."""
        return TrajectoryEvaluator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
            prune_threshold=5.0,
        )

    @pytest.fixture
    def sample_node(self) -> DialogueNode:
        """Create a sample node for testing."""
        return DialogueNode(
            id="node-1",
            depth=1,
            strategy=Strategy(tagline="Test", description="Test strategy"),
            messages=[
                Message.user("Hello"),
                Message.assistant("Hi! How can I help?"),
            ],
        )

    @pytest.mark.asyncio
    async def test_evaluate_absolute_success(
        self, evaluator: TrajectoryEvaluator, mock_llm: MagicMock, sample_node: DialogueNode
    ) -> None:
        """Test successful absolute evaluation."""
        # Return 3 judge scores
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            data={
                "criteria": {
                    "goal_achieved": {"score": 0.8, "rationale": "Good progress"},
                    "user_engagement": {"score": 0.7, "rationale": "User engaged"},
                },
                "total_score": 7.5,
                "confidence": "high",
                "summary": "Good conversation",
            },
        )

        scores = await evaluator.evaluate_absolute([sample_node])

        assert sample_node.id in scores
        score = scores[sample_node.id]
        assert isinstance(score, AggregatedScore)
        # Should have called complete 3 times for 3 judges
        assert mock_llm.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_evaluate_absolute_updates_node_stats(
        self, evaluator: TrajectoryEvaluator, mock_llm: MagicMock, sample_node: DialogueNode
    ) -> None:
        """Test that node stats are updated after evaluation."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data={
                "criteria": {},
                "total_score": 7.0,
                "confidence": "medium",
                "summary": "OK",
            },
        )

        await evaluator.evaluate_absolute([sample_node])

        assert sample_node.stats.aggregated_score == 7.0
        assert len(sample_node.stats.judge_scores) == 3

    @pytest.mark.asyncio
    async def test_evaluate_absolute_handles_judge_failure(
        self, evaluator: TrajectoryEvaluator, mock_llm: MagicMock, sample_node: DialogueNode
    ) -> None:
        """Test handling of judge failures."""
        mock_llm.complete.side_effect = Exception("Judge failed")

        scores = await evaluator.evaluate_absolute([sample_node])

        # Should return zero score on failure
        assert sample_node.id in scores
        score = scores[sample_node.id]
        assert score.aggregated_score == 0.0


# -----------------------------------------------------------------------------
# evaluate_comparative Tests
# -----------------------------------------------------------------------------


class TestEvaluateComparative:
    """Tests for TrajectoryEvaluator.evaluate_comparative()."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def evaluator(self, mock_llm: MagicMock) -> TrajectoryEvaluator:
        """Create an evaluator with mock LLM."""
        return TrajectoryEvaluator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
            prune_threshold=5.0,
        )

    def create_nodes(self, count: int, parent_id: str = "parent") -> list[DialogueNode]:
        """Create multiple nodes with the same parent."""
        nodes = []
        for i in range(count):
            node = DialogueNode(
                id=f"node-{i}",
                parent_id=parent_id,
                depth=1,
                strategy=Strategy(tagline=f"Strategy {i}", description=f"Desc {i}"),
                user_intent=UserIntent(
                    id=f"intent-{i}",
                    label=f"Intent {i}",
                    description="Desc",
                    emotional_tone="neutral",
                    cognitive_stance="neutral",
                ),
                messages=[
                    Message.user("Hello"),
                    Message.assistant(f"Response {i}"),
                ],
            )
            nodes.append(node)
        return nodes

    @pytest.mark.asyncio
    async def test_evaluate_comparative_single_node_uses_absolute(
        self, evaluator: TrajectoryEvaluator, mock_llm: MagicMock
    ) -> None:
        """Test that single node uses absolute evaluation."""
        node = DialogueNode(
            id="single",
            strategy=Strategy(tagline="Test", description="Test"),
            messages=[Message.user("Hi")],
        )

        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data={"total_score": 7.0, "confidence": "high", "summary": "OK", "criteria": {}},
        )

        scores = await evaluator.evaluate_comparative([node])

        assert "single" in scores

    @pytest.mark.asyncio
    async def test_evaluate_comparative_success(
        self, evaluator: TrajectoryEvaluator, mock_llm: MagicMock
    ) -> None:
        """Test successful comparative evaluation."""
        nodes = self.create_nodes(3)

        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data={
                "critiques": {
                    "node-0": {"strengths": ["Good"], "weaknesses": ["Bad"]},
                    "node-1": {"strengths": ["Better"], "weaknesses": []},
                    "node-2": {"strengths": [], "weaknesses": ["Worst"]},
                },
                "ranking": [
                    {"rank": 1, "trajectory_id": "node-1", "score": 7.5, "reason": "Best"},
                    {"rank": 2, "trajectory_id": "node-0", "score": 6.0, "reason": "OK"},
                    {"rank": 3, "trajectory_id": "node-2", "score": 4.5, "reason": "Poor"},
                ],
                "ranking_confidence": "high",
            },
        )

        scores = await evaluator.evaluate_comparative(nodes)

        assert len(scores) == 3
        assert scores["node-1"].aggregated_score == 7.5
        assert scores["node-0"].aggregated_score == 6.0
        assert scores["node-2"].aggregated_score == 4.5

    @pytest.mark.asyncio
    async def test_evaluate_comparative_groups_by_parent(
        self, evaluator: TrajectoryEvaluator, mock_llm: MagicMock
    ) -> None:
        """Test that nodes are grouped by parent for comparison."""
        # Create nodes with different parents
        nodes_parent_a = self.create_nodes(2, parent_id="parent-a")
        nodes_parent_b = self.create_nodes(2, parent_id="parent-b")

        # Reassign IDs to avoid conflicts
        for i, node in enumerate(nodes_parent_b):
            node.id = f"node-b-{i}"

        all_nodes = nodes_parent_a + nodes_parent_b

        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data={
                "critiques": {},
                "ranking": [
                    {"rank": 1, "trajectory_id": "node-0", "score": 7.5, "reason": "Best"},
                    {"rank": 2, "trajectory_id": "node-1", "score": 6.0, "reason": "OK"},
                ],
                "ranking_confidence": "medium",
            },
        )

        scores = await evaluator.evaluate_comparative(all_nodes)

        # Should have evaluated both groups
        assert len(scores) >= 2

    @pytest.mark.asyncio
    async def test_evaluate_comparative_fallback_on_failure(
        self, evaluator: TrajectoryEvaluator, mock_llm: MagicMock
    ) -> None:
        """Test fallback to absolute on comparative failure."""
        nodes = self.create_nodes(2)

        # First call fails (comparative), then succeed (absolute fallback)
        mock_llm.complete.side_effect = [
            Completion(
                message=Message.assistant("{}"),
                data={"invalid": "response"},  # Missing ranking
            ),
            Completion(
                message=Message.assistant("{}"),
                data={"total_score": 6.0, "confidence": "medium", "summary": "OK", "criteria": {}},
            ),
            Completion(
                message=Message.assistant("{}"),
                data={"total_score": 6.0, "confidence": "medium", "summary": "OK", "criteria": {}},
            ),
            Completion(
                message=Message.assistant("{}"),
                data={"total_score": 7.0, "confidence": "medium", "summary": "OK", "criteria": {}},
            ),
            Completion(
                message=Message.assistant("{}"),
                data={"total_score": 7.0, "confidence": "medium", "summary": "OK", "criteria": {}},
            ),
            Completion(
                message=Message.assistant("{}"),
                data={"total_score": 7.0, "confidence": "medium", "summary": "OK", "criteria": {}},
            ),
            Completion(
                message=Message.assistant("{}"),
                data={"total_score": 7.0, "confidence": "medium", "summary": "OK", "criteria": {}},
            ),
        ]

        scores = await evaluator.evaluate_comparative(nodes)

        # Should have fallen back to absolute scoring
        assert len(scores) == 2


# -----------------------------------------------------------------------------
# _judge_single Tests
# -----------------------------------------------------------------------------


class TestJudgeSingle:
    """Tests for single trajectory judging."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def evaluator(self, mock_llm: MagicMock) -> TrajectoryEvaluator:
        """Create an evaluator with mock LLM."""
        return TrajectoryEvaluator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_judge_single_runs_three_judges(
        self, evaluator: TrajectoryEvaluator, mock_llm: MagicMock
    ) -> None:
        """Test that three judges are run in parallel."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data={
                "total_score": 7.0,
                "confidence": "high",
                "summary": "Good",
                "criteria": {},
            },
        )

        node = DialogueNode(
            id="test",
            messages=[Message.user("Hi"), Message.assistant("Hello")],
        )

        score, critiques = await evaluator._judge_single(node)

        assert mock_llm.complete.call_count == 3
        assert score.aggregated_score == 7.0

    @pytest.mark.asyncio
    async def test_judge_single_extracts_critiques(
        self, evaluator: TrajectoryEvaluator, mock_llm: MagicMock
    ) -> None:
        """Test that critiques are extracted from judge results."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data={
                "total_score": 7.0,
                "confidence": "high",
                "summary": "Good conversation",
                "criteria": {
                    "goal_achieved": {"score": 0.9, "rationale": "Very good progress"},
                    "user_engagement": {"score": 0.3, "rationale": "Could be better"},
                },
                "key_turning_point": "Turn 3",
                "biggest_missed_opportunity": "Could have asked more questions",
            },
        )

        node = DialogueNode(
            id="test",
            messages=[Message.user("Hi"), Message.assistant("Hello")],
        )

        score, critiques = await evaluator._judge_single(node)

        assert critiques is not None
        assert "strengths" in critiques
        assert "weaknesses" in critiques


# -----------------------------------------------------------------------------
# Usage Tracking Tests
# -----------------------------------------------------------------------------


class TestEvaluatorUsageTracking:
    """Tests for token usage tracking in evaluator."""

    @pytest.mark.asyncio
    async def test_tracks_usage(self) -> None:
        """Test that usage is tracked."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=Completion(
                message=Message.assistant("{}"),
                usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                data={"total_score": 7.0, "confidence": "high", "summary": "OK", "criteria": {}},
            )
        )

        usage_callback = MagicMock()
        evaluator = TrajectoryEvaluator(
            llm=mock_llm,
            goal="Test",
            on_usage=usage_callback,
        )

        node = DialogueNode(id="test", messages=[Message.user("Hi")])
        await evaluator.evaluate_absolute([node])

        # Should be called 3 times (once per judge)
        assert usage_callback.call_count == 3
        # All calls should have phase "judge"
        for call in usage_callback.call_args_list:
            assert call[0][1] == "judge"
