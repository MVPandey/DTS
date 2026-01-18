"""Tests for backend/core/dts/engine.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.dts.config import DTSConfig
from backend.core.dts.engine import DTSEngine
from backend.core.dts.types import (
    AggregatedScore,
    DialogueNode,
    DTSRunResult,
    NodeStatus,
    Strategy,
)
from backend.llm.types import Completion, Message, Usage

# -----------------------------------------------------------------------------
# DTSEngine Initialization Tests
# -----------------------------------------------------------------------------


class TestDTSEngineInit:
    """Tests for DTSEngine initialization."""

    def test_init_creates_components(self) -> None:
        """Test that initialization creates all components."""
        mock_llm = MagicMock()
        mock_llm._default_model = "test-model"

        config = DTSConfig(
            goal="Test goal",
            first_message="Hello",
            model="test-model",
        )

        engine = DTSEngine(llm=mock_llm, config=config)

        assert engine.llm is mock_llm
        assert engine.config is config
        assert engine._generator is not None
        assert engine._simulator is not None
        assert engine._evaluator is not None
        assert engine._researcher is not None
        assert engine._tree is None  # Not yet created

    def test_init_with_per_phase_models(self) -> None:
        """Test initialization with different models per phase."""
        mock_llm = MagicMock()
        mock_llm._default_model = "default-model"

        config = DTSConfig(
            goal="Test",
            first_message="Hello",
            strategy_model="strategy-model",
            simulator_model="simulator-model",
            judge_model="judge-model",
        )

        engine = DTSEngine(llm=mock_llm, config=config)

        assert engine._generator.model == "strategy-model"
        assert engine._simulator.model == "simulator-model"
        assert engine._evaluator.model == "judge-model"

    def test_init_falls_back_to_default_model(self) -> None:
        """Test that missing per-phase models fall back to default."""
        mock_llm = MagicMock()
        mock_llm._default_model = "default-model"

        config = DTSConfig(
            goal="Test",
            first_message="Hello",
            model="default-model",
        )

        engine = DTSEngine(llm=mock_llm, config=config)

        assert engine._generator.model == "default-model"
        assert engine._simulator.model == "default-model"
        assert engine._evaluator.model == "default-model"


# -----------------------------------------------------------------------------
# Event Callback Tests
# -----------------------------------------------------------------------------


class TestEventCallback:
    """Tests for event callback handling."""

    def test_set_event_callback(self) -> None:
        """Test setting event callback."""
        mock_llm = MagicMock()
        config = DTSConfig(goal="Test", first_message="Hello")

        engine = DTSEngine(llm=mock_llm, config=config)
        callback = AsyncMock()

        engine.set_event_callback(callback)

        assert engine._event_callback is callback

    @pytest.mark.asyncio
    async def test_emit_creates_task(self) -> None:
        """Test that _emit creates async task."""
        import asyncio

        mock_llm = MagicMock()
        config = DTSConfig(goal="Test", first_message="Hello")

        engine = DTSEngine(llm=mock_llm, config=config)
        callback = AsyncMock()
        engine.set_event_callback(callback)

        engine._emit("test_event", {"key": "value"})

        # Yield to event loop to let fire-and-forget task run (no actual delay)
        await asyncio.sleep(0)

        callback.assert_called()


# -----------------------------------------------------------------------------
# _initialize_tree Tests
# -----------------------------------------------------------------------------


class TestInitializeTree:
    """Tests for tree initialization."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock._default_model = "test-model"
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def engine(self, mock_llm: MagicMock) -> DTSEngine:
        """Create an engine for testing."""
        config = DTSConfig(
            goal="Help user",
            first_message="Hello",
            init_branches=3,
            model="test-model",
        )
        return DTSEngine(llm=mock_llm, config=config)

    @pytest.mark.asyncio
    async def test_initialize_tree_creates_root(
        self, engine: DTSEngine, mock_llm: MagicMock
    ) -> None:
        """Test that tree initialization creates root node."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data={
                "goal": "Help user",
                "nodes": {
                    "Strategy 1": "Description 1",
                    "Strategy 2": "Description 2",
                    "Strategy 3": "Description 3",
                },
            },
        )

        tree = await engine._initialize_tree()

        assert tree is not None
        root = tree.get_root()
        assert root.depth == 0
        assert len(root.messages) == 1

    @pytest.mark.asyncio
    async def test_initialize_tree_creates_branches(
        self, engine: DTSEngine, mock_llm: MagicMock
    ) -> None:
        """Test that tree initialization creates strategy branches."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            data={
                "goal": "Help user",
                "nodes": {
                    "Strategy 1": "Description 1",
                    "Strategy 2": "Description 2",
                    "Strategy 3": "Description 3",
                },
            },
        )

        tree = await engine._initialize_tree()

        # Root + 3 strategy branches
        assert len(tree.all_nodes()) == 4
        leaves = tree.active_leaves()
        assert len(leaves) == 3
        assert all(n.strategy is not None for n in leaves)


# -----------------------------------------------------------------------------
# _prune Tests
# -----------------------------------------------------------------------------


class TestPrune:
    """Tests for pruning logic."""

    @pytest.fixture
    def engine(self) -> DTSEngine:
        """Create an engine for testing."""
        mock_llm = MagicMock()
        config = DTSConfig(
            goal="Test",
            first_message="Hello",
            prune_threshold=6.0,
            min_survivors=1,
        )
        return DTSEngine(llm=mock_llm, config=config)

    def create_nodes_with_scores(
        self, scores: list[float]
    ) -> tuple[list[DialogueNode], dict[str, AggregatedScore]]:
        """Create nodes and corresponding scores."""
        nodes = []
        score_dict = {}
        for i, score in enumerate(scores):
            node = DialogueNode(
                id=f"node-{i}",
                strategy=Strategy(tagline=f"S{i}", description=f"D{i}"),
            )
            nodes.append(node)
            score_dict[node.id] = AggregatedScore(
                individual_scores=[score, score, score],
                aggregated_score=score,
                pass_threshold=5.0,
                pass_votes=3 if score >= 5.0 else 0,
                passed=score >= 5.0,
            )
        return nodes, score_dict

    def test_prune_below_threshold(self, engine: DTSEngine) -> None:
        """Test that nodes below threshold are pruned."""
        nodes, scores = self.create_nodes_with_scores([7.0, 5.0, 8.0])

        survivors = engine._prune(nodes, scores)

        assert len(survivors) == 2
        assert nodes[1].status == NodeStatus.PRUNED
        assert nodes[1].prune_reason is not None
        assert "score 5.0 < 6.0" in nodes[1].prune_reason

    def test_prune_respects_min_survivors(self, engine: DTSEngine) -> None:
        """Test that min_survivors is respected."""
        nodes, scores = self.create_nodes_with_scores([3.0, 4.0, 5.0])

        survivors = engine._prune(nodes, scores)

        # At least 1 should survive (min_survivors=1)
        assert len(survivors) >= 1

    def test_prune_with_keep_top_k(self, engine: DTSEngine) -> None:
        """Test pruning with keep_top_k limit."""
        engine.config.keep_top_k = 2
        nodes, scores = self.create_nodes_with_scores([9.0, 8.0, 7.0, 6.5])

        survivors = engine._prune(nodes, scores)

        assert len(survivors) == 2
        survivor_ids = {n.id for n in survivors}
        assert "node-0" in survivor_ids  # Score 9.0
        assert "node-1" in survivor_ids  # Score 8.0

    def test_prune_empty_nodes(self, engine: DTSEngine) -> None:
        """Test pruning empty node list."""
        survivors = engine._prune([], {})
        assert survivors == []


# -----------------------------------------------------------------------------
# _track_usage Tests
# -----------------------------------------------------------------------------


class TestTrackUsage:
    """Tests for usage tracking."""

    def test_track_usage_by_phase(self) -> None:
        """Test that usage is tracked by phase."""
        mock_llm = MagicMock()
        config = DTSConfig(goal="Test", first_message="Hello", model="test-model")
        engine = DTSEngine(llm=mock_llm, config=config)

        completion = Completion(
            message=Message.assistant("Response"),
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="test-model",
        )

        engine._track_usage(completion, "strategy")

        assert engine._token_tracker.strategy_generation.input_tokens == 100
        assert engine._token_tracker.strategy_generation.output_tokens == 50

    def test_track_usage_maps_phases(self) -> None:
        """Test that phase names are mapped correctly."""
        mock_llm = MagicMock()
        config = DTSConfig(goal="Test", first_message="Hello", model="test-model")
        engine = DTSEngine(llm=mock_llm, config=config)

        completion = Completion(
            message=Message.assistant("Response"),
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="test-model",
        )

        engine._track_usage(completion, "judge")
        assert engine._token_tracker.judging.input_tokens == 100

        engine._track_usage(completion, "user")
        assert engine._token_tracker.user_simulation.input_tokens == 100

        engine._track_usage(completion, "assistant")
        assert engine._token_tracker.assistant_generation.input_tokens == 100


# -----------------------------------------------------------------------------
# run Tests
# -----------------------------------------------------------------------------


class TestRun:
    """Tests for DTSEngine.run()."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock._default_model = "test-model"
        mock.complete = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_run_returns_result(self, mock_llm: MagicMock) -> None:
        """Test that run returns a DTSRunResult."""
        config = DTSConfig(
            goal="Help user",
            first_message="Hello",
            init_branches=2,
            turns_per_branch=1,
            model="test-model",
        )
        engine = DTSEngine(llm=mock_llm, config=config)

        # Mock strategy generation
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("{}"),
            usage=Usage(prompt_tokens=50, completion_tokens=25, total_tokens=75),
            data={
                "goal": "Help user",
                "nodes": {
                    "Strategy 1": "Description 1",
                    "Strategy 2": "Description 2",
                },
            },
        )

        # Mock the components
        engine._generator.generate_strategies = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                Strategy(tagline="S1", description="D1"),
                Strategy(tagline="S2", description="D2"),
            ]
        )
        engine._simulator.expand_nodes = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda nodes, **_kw: nodes
        )
        engine._evaluator.evaluate_comparative = AsyncMock(  # type: ignore[method-assign]
            return_value={
                node_id: AggregatedScore(
                    individual_scores=[7.0, 7.0, 7.0],
                    aggregated_score=7.0,
                    pass_threshold=5.0,
                    pass_votes=3,
                    passed=True,
                )
                for node_id in ["test-1", "test-2"]
            }
        )

        result = await engine.run(rounds=1)

        assert isinstance(result, DTSRunResult)
        assert result.total_rounds == 1

    @pytest.mark.asyncio
    async def test_run_tree_property(self, mock_llm: MagicMock) -> None:
        """Test that tree is accessible after run."""
        config = DTSConfig(
            goal="Help user",
            first_message="Hello",
            init_branches=1,
            turns_per_branch=1,
            model="test-model",
        )
        engine = DTSEngine(llm=mock_llm, config=config)

        assert engine.tree is None

        # Mock components
        engine._generator.generate_strategies = AsyncMock(  # type: ignore[method-assign]
            return_value=[Strategy(tagline="S1", description="D1")]
        )
        engine._simulator.expand_nodes = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda nodes, **_kw: nodes
        )
        engine._evaluator.evaluate_comparative = AsyncMock(  # type: ignore[method-assign]
            return_value={}
        )

        await engine.run(rounds=1)

        assert engine.tree is not None


# -----------------------------------------------------------------------------
# Integration-like Tests
# -----------------------------------------------------------------------------


class TestDTSEngineIntegration:
    """Integration-like tests for DTSEngine."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM with full response handling."""
        mock = MagicMock()
        mock._default_model = "test-model"
        mock.complete = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_full_run_with_mocks(self, mock_llm: MagicMock) -> None:
        """Test a full run with all components mocked."""
        config = DTSConfig(
            goal="Help user debug code",
            first_message="My code isn't working",
            init_branches=2,
            turns_per_branch=1,
            scoring_mode="absolute",
            prune_threshold=5.0,
            model="test-model",
        )

        engine = DTSEngine(llm=mock_llm, config=config)

        # Mock generator
        engine._generator.generate_strategies = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                Strategy(tagline="Debug Step by Step", description="Walk through code"),
                Strategy(tagline="Ask Questions", description="Clarify the problem"),
            ]
        )

        # Mock simulator - return nodes with expanded messages
        async def mock_expand(nodes, **_kwargs):
            for node in nodes:
                node.messages.append(Message.assistant("I can help with that!"))
            return nodes

        engine._simulator.expand_nodes = AsyncMock(  # type: ignore[method-assign]
            side_effect=mock_expand
        )

        # Mock evaluator
        engine._evaluator.evaluate_absolute = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda nodes: {
                node.id: AggregatedScore(
                    individual_scores=[7.0, 7.5, 6.5],
                    aggregated_score=7.0,
                    pass_threshold=5.0,
                    pass_votes=3,
                    passed=True,
                )
                for node in nodes
            }
        )

        result = await engine.run(rounds=1)

        assert result.total_rounds == 1
        assert len(result.all_nodes) >= 2
        assert result.token_usage is not None
