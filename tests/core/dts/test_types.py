"""Tests for backend/core/dts/types.py."""

import json
from unittest.mock import MagicMock, patch

from backend.core.dts.types import (
    AggregatedScore,
    BranchSelectionEvaluation,
    CriterionScore,
    DialogueNode,
    DTSRunResult,
    ModelPricing,
    NodeStats,
    NodeStatus,
    Strategy,
    TokenStats,
    TokenTracker,
    TrajectoryEvaluation,
    TreeGeneratorOutput,
    UserIntent,
    _load_pricing_from_openrouter,
    get_model_pricing,
)
from backend.llm.types import Message, Usage

# -----------------------------------------------------------------------------
# ModelPricing Tests
# -----------------------------------------------------------------------------


class TestModelPricing:
    """Tests for ModelPricing class."""

    def test_pricing_creation(self) -> None:
        """Test creating ModelPricing."""
        pricing = ModelPricing(
            model_name="gpt-4",
            input_cost_per_million=30.0,
            output_cost_per_million=60.0,
        )
        assert pricing.model_name == "gpt-4"
        assert pricing.input_cost_per_million == 30.0
        assert pricing.output_cost_per_million == 60.0

    def test_calculate_cost(self) -> None:
        """Test cost calculation."""
        pricing = ModelPricing(
            model_name="test",
            input_cost_per_million=10.0,  # $10 per 1M input
            output_cost_per_million=20.0,  # $20 per 1M output
        )
        # 1M input tokens = $10, 500K output tokens = $10
        cost = pricing.calculate_cost(input_tokens=1_000_000, output_tokens=500_000)
        assert cost == 20.0

    def test_calculate_cost_zero_tokens(self) -> None:
        """Test cost calculation with zero tokens."""
        pricing = ModelPricing("test", 10.0, 20.0)
        cost = pricing.calculate_cost(0, 0)
        assert cost == 0.0


# -----------------------------------------------------------------------------
# TokenStats Tests
# -----------------------------------------------------------------------------


class TestTokenStats:
    """Tests for TokenStats class."""

    def test_default_values(self) -> None:
        """Test default TokenStats values."""
        stats = TokenStats()
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.total_tokens == 0
        assert stats.request_count == 0

    def test_add_usage(self) -> None:
        """Test adding usage to stats."""
        stats = TokenStats()
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        stats.add(usage)

        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert stats.total_tokens == 150
        assert stats.request_count == 1

    def test_add_none_usage(self) -> None:
        """Test adding None usage does nothing."""
        stats = TokenStats()
        stats.add(None)
        assert stats.request_count == 0

    def test_merge_stats(self) -> None:
        """Test merging two TokenStats."""
        stats1 = TokenStats(input_tokens=100, output_tokens=50, total_tokens=150, request_count=1)
        stats2 = TokenStats(input_tokens=200, output_tokens=100, total_tokens=300, request_count=2)

        stats1.merge(stats2)

        assert stats1.input_tokens == 300
        assert stats1.output_tokens == 150
        assert stats1.total_tokens == 450
        assert stats1.request_count == 3


# -----------------------------------------------------------------------------
# TokenTracker Tests
# -----------------------------------------------------------------------------


class TestTokenTracker:
    """Tests for TokenTracker class."""

    def test_default_tracker(self) -> None:
        """Test default TokenTracker."""
        tracker = TokenTracker()
        assert tracker.model_name == "unknown"
        assert tracker.total_tokens == 0
        assert tracker.total_requests == 0

    def test_add_usage_by_phase(self) -> None:
        """Test adding usage to specific phase."""
        tracker = TokenTracker(model_name="gpt-4")
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        tracker.add_usage("gpt-4", usage, "strategy_generation")

        assert tracker.strategy_generation.input_tokens == 100
        assert tracker.strategy_generation.output_tokens == 50
        assert tracker.total_tokens == 150

    def test_add_usage_tracks_by_model(self) -> None:
        """Test that usage is tracked per model."""
        tracker = TokenTracker()
        usage1 = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage2 = Usage(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        tracker.add_usage("gpt-4", usage1, "judging")
        tracker.add_usage("gpt-3.5", usage2, "judging")

        assert "gpt-4" in tracker.by_model
        assert "gpt-3.5" in tracker.by_model
        assert tracker.by_model["gpt-4"].input_tokens == 100
        assert tracker.by_model["gpt-3.5"].input_tokens == 200

    def test_total_input_tokens(self) -> None:
        """Test total input tokens across phases."""
        tracker = TokenTracker()
        tracker.strategy_generation.input_tokens = 100
        tracker.judging.input_tokens = 200

        assert tracker.total_input_tokens == 300

    def test_total_output_tokens(self) -> None:
        """Test total output tokens across phases."""
        tracker = TokenTracker()
        tracker.user_simulation.output_tokens = 50
        tracker.assistant_generation.output_tokens = 100

        assert tracker.total_output_tokens == 150

    def test_total_requests(self) -> None:
        """Test total request count."""
        tracker = TokenTracker()
        tracker.strategy_generation.request_count = 5
        tracker.judging.request_count = 10

        assert tracker.total_requests == 15

    @patch("backend.core.dts.types.get_model_pricing")
    def test_total_cost(self, mock_pricing) -> None:
        """Test total cost calculation."""
        mock_pricing.return_value = ModelPricing("test", 10.0, 20.0)

        tracker = TokenTracker()
        tracker.by_model["test"] = TokenStats(
            input_tokens=1_000_000,
            output_tokens=500_000,
            total_tokens=1_500_000,
        )
        tracker.research_cost_usd = 5.0

        # 1M input * $10/1M = $10, 500K output * $20/1M = $10, research = $5
        assert tracker.total_cost == 25.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        tracker = TokenTracker(model_name="gpt-4")
        tracker.strategy_generation.input_tokens = 100
        tracker.strategy_generation.request_count = 1

        result = tracker.to_dict()

        assert "totals" in result
        assert "by_phase" in result
        assert result["by_phase"]["strategy_generation"]["input_tokens"] == 100


# -----------------------------------------------------------------------------
# NodeStatus Tests
# -----------------------------------------------------------------------------


class TestNodeStatus:
    """Tests for NodeStatus enum."""

    def test_status_values(self) -> None:
        """Test status enum values."""
        assert NodeStatus.ACTIVE.value == "active"
        assert NodeStatus.PRUNED.value == "pruned"
        assert NodeStatus.TERMINAL.value == "terminal"
        assert NodeStatus.ERROR.value == "error"


# -----------------------------------------------------------------------------
# Strategy Tests
# -----------------------------------------------------------------------------


class TestStrategy:
    """Tests for Strategy class."""

    def test_strategy_creation(self) -> None:
        """Test Strategy creation."""
        strategy = Strategy(
            tagline="Empathetic Approach",
            description="Focus on emotional understanding",
        )
        assert strategy.tagline == "Empathetic Approach"
        assert strategy.description == "Focus on emotional understanding"


# -----------------------------------------------------------------------------
# UserIntent Tests
# -----------------------------------------------------------------------------


class TestUserIntent:
    """Tests for UserIntent class."""

    def test_user_intent_creation(self) -> None:
        """Test UserIntent creation."""
        intent = UserIntent(
            id="curious_1",
            label="Curious Explorer",
            description="User wants to learn more",
            emotional_tone="enthusiastic",
            cognitive_stance="exploring",
        )
        assert intent.id == "curious_1"
        assert intent.label == "Curious Explorer"
        assert intent.emotional_tone == "enthusiastic"


# -----------------------------------------------------------------------------
# AggregatedScore Tests
# -----------------------------------------------------------------------------


class TestAggregatedScore:
    """Tests for AggregatedScore class."""

    def test_aggregated_score_creation(self) -> None:
        """Test AggregatedScore creation."""
        score = AggregatedScore(
            individual_scores=[7.0, 7.5, 6.5],
            aggregated_score=7.0,
            pass_threshold=5.0,
            pass_votes=3,
            passed=True,
        )
        assert score.aggregated_score == 7.0
        assert score.passed is True

    def test_zero_score(self) -> None:
        """Test creating zero score."""
        score = AggregatedScore.zero(threshold=6.0)

        assert score.individual_scores == [0.0, 0.0, 0.0]
        assert score.aggregated_score == 0.0
        assert score.pass_threshold == 6.0
        assert score.pass_votes == 0
        assert score.passed is False


# -----------------------------------------------------------------------------
# NodeStats Tests
# -----------------------------------------------------------------------------


class TestNodeStats:
    """Tests for NodeStats class."""

    def test_default_node_stats(self) -> None:
        """Test default NodeStats."""
        stats = NodeStats()
        assert stats.visits == 0
        assert stats.value_sum == 0.0
        assert stats.value_mean == 0.0
        assert stats.judge_scores == []
        assert stats.aggregated_score == 0.0


# -----------------------------------------------------------------------------
# DialogueNode Tests
# -----------------------------------------------------------------------------


class TestDialogueNode:
    """Tests for DialogueNode class."""

    def test_node_creation(self) -> None:
        """Test DialogueNode creation."""
        node = DialogueNode(
            id="node-1",
            parent_id=None,
            depth=0,
            status=NodeStatus.ACTIVE,
            messages=[Message.user("Hello")],
        )
        assert node.id == "node-1"
        assert node.depth == 0
        assert node.status == NodeStatus.ACTIVE

    def test_strategy_label_with_strategy(self) -> None:
        """Test strategy_label when strategy exists."""
        strategy = Strategy(tagline="Test", description="Test desc")
        node = DialogueNode(id="1", strategy=strategy)
        assert node.strategy_label == "Test"

    def test_strategy_label_without_strategy(self) -> None:
        """Test strategy_label when no strategy."""
        node = DialogueNode(id="1")
        assert node.strategy_label == "unknown"

    def test_intent_label_with_intent(self) -> None:
        """Test intent_label when intent exists."""
        intent = UserIntent(
            id="i1",
            label="Curious",
            description="desc",
            emotional_tone="neutral",
            cognitive_stance="exploring",
        )
        node = DialogueNode(id="1", user_intent=intent)
        assert node.intent_label == "Curious"

    def test_intent_label_without_intent(self) -> None:
        """Test intent_label when no intent."""
        node = DialogueNode(id="1")
        assert node.intent_label is None

    def test_update_with_evaluation(self) -> None:
        """Test updating node with evaluation results."""
        node = DialogueNode(id="1")
        score = AggregatedScore(
            individual_scores=[7.0, 7.5, 6.5],
            aggregated_score=7.0,
            pass_threshold=5.0,
            pass_votes=3,
            passed=True,
        )
        critiques = {"strengths": ["good"], "weaknesses": ["could improve"]}

        node.update_with_evaluation(score, critiques)

        assert node.stats.judge_scores == [7.0, 7.5, 6.5]
        assert node.stats.aggregated_score == 7.0
        assert node.stats.critiques == critiques


# -----------------------------------------------------------------------------
# DTSRunResult Tests
# -----------------------------------------------------------------------------


class TestDTSRunResult:
    """Tests for DTSRunResult class."""

    def test_default_result(self) -> None:
        """Test default DTSRunResult."""
        result = DTSRunResult()
        assert result.best_node_id is None
        assert result.best_score == 0.0
        assert result.best_messages == []
        assert result.pruned_count == 0

    def test_to_exploration_dict(self) -> None:
        """Test conversion to exploration dictionary."""
        strategy = Strategy(tagline="Test", description="Test desc")
        node = DialogueNode(
            id="node-1",
            depth=1,
            status=NodeStatus.ACTIVE,
            strategy=strategy,
            messages=[Message.user("Hi"), Message.assistant("Hello")],
        )
        node.stats.aggregated_score = 7.5

        result = DTSRunResult(
            best_node_id="node-1",
            best_score=7.5,
            all_nodes=[node],
            total_rounds=2,
        )

        exploration = result.to_exploration_dict()

        assert "summary" in exploration
        assert "branches" in exploration
        assert exploration["summary"]["best_score"] == 7.5
        assert len(exploration["branches"]) == 1

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        result = DTSRunResult(best_score=8.0)
        json_str = result.to_json()

        assert '"best_score": 8.0' in json_str


# -----------------------------------------------------------------------------
# Evaluation Types Tests
# -----------------------------------------------------------------------------


class TestEvaluationTypes:
    """Tests for evaluation-related types."""

    def test_criterion_score(self) -> None:
        """Test CriterionScore creation."""
        score = CriterionScore(score=0.8, rationale="Good progress made")
        assert score.score == 0.8
        assert score.rationale == "Good progress made"

    def test_trajectory_evaluation(self) -> None:
        """Test TrajectoryEvaluation creation."""
        eval_result = TrajectoryEvaluation(
            criteria={
                "goal_achieved": CriterionScore(score=0.8, rationale="Mostly achieved"),
            },
            total_score=8.0,
            confidence="high",
            summary="Good conversation",
            key_turning_point="Turn 3 was pivotal",
        )
        assert eval_result.total_score == 8.0
        assert eval_result.confidence == "high"

    def test_branch_selection_evaluation(self) -> None:
        """Test BranchSelectionEvaluation creation."""
        eval_result = BranchSelectionEvaluation(
            criteria={
                "goal_aligned": CriterionScore(score=1.0, rationale="Fully aligned"),
            },
            total_score=9.0,
            confidence="medium",
            summary="Strong choice",
        )
        assert eval_result.total_score == 9.0

    def test_tree_generator_output(self) -> None:
        """Test TreeGeneratorOutput creation."""
        output = TreeGeneratorOutput(
            goal="Help user with coding",
            nodes={
                "Technical Focus": "Dive into code details",
                "Conceptual Approach": "Explain concepts first",
            },
            coverage_rationale="Covers both deep and broad approaches",
        )
        assert len(output.nodes) == 2


# -----------------------------------------------------------------------------
# Pricing API Tests
# -----------------------------------------------------------------------------


class TestPricingAPI:
    """Tests for pricing loading and retrieval."""

    def test_load_pricing_from_openrouter_success(self) -> None:
        """Test successful pricing loading from OpenRouter."""
        import backend.core.dts.types as types_module

        # Reset cache state
        original_loaded = types_module._pricing_loaded
        original_cache = types_module._pricing_cache.copy()
        types_module._pricing_loaded = False
        types_module._pricing_cache = {}

        try:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(
                {
                    "data": [
                        {
                            "id": "openai/gpt-4",
                            "pricing": {"prompt": "0.00003", "completion": "0.00006"},
                        }
                    ]
                }
            ).encode()
            mock_response.__enter__ = lambda _s: mock_response
            mock_response.__exit__ = MagicMock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_response):
                _load_pricing_from_openrouter()

            assert types_module._pricing_loaded is True
            assert "openai/gpt-4" in types_module._pricing_cache

        finally:
            # Restore original state
            types_module._pricing_loaded = original_loaded
            types_module._pricing_cache = original_cache

    def test_load_pricing_from_openrouter_failure(self) -> None:
        """Test pricing loading handles errors gracefully."""
        import backend.core.dts.types as types_module

        # Reset cache state
        original_loaded = types_module._pricing_loaded
        types_module._pricing_loaded = False

        try:
            with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
                _load_pricing_from_openrouter()

            # Should still mark as loaded to prevent retries
            assert types_module._pricing_loaded is True

        finally:
            types_module._pricing_loaded = original_loaded

    def test_load_pricing_skips_if_already_loaded(self) -> None:
        """Test that pricing loading is skipped if already loaded."""
        import backend.core.dts.types as types_module

        original_loaded = types_module._pricing_loaded
        types_module._pricing_loaded = True

        try:
            with patch("urllib.request.urlopen") as mock_urlopen:
                _load_pricing_from_openrouter()
                mock_urlopen.assert_not_called()

        finally:
            types_module._pricing_loaded = original_loaded

    def test_get_model_pricing_cached(self) -> None:
        """Test retrieving cached model pricing."""
        import backend.core.dts.types as types_module

        original_cache = types_module._pricing_cache.copy()
        types_module._pricing_cache["test-model"] = ModelPricing("test-model", 10.0, 20.0)
        types_module._pricing_loaded = True

        try:
            pricing = get_model_pricing("test-model")
            assert pricing.model_name == "test-model"
            assert pricing.input_cost_per_million == 10.0

        finally:
            types_module._pricing_cache = original_cache

    def test_get_model_pricing_unknown_model(self) -> None:
        """Test retrieving pricing for unknown model."""
        import backend.core.dts.types as types_module

        types_module._pricing_loaded = True

        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing.model_name == "unknown-model-xyz"
        assert pricing.input_cost_per_million == 0.0
        assert pricing.output_cost_per_million == 0.0


# -----------------------------------------------------------------------------
# Additional TokenTracker Tests
# -----------------------------------------------------------------------------


class TestTokenTrackerAdditional:
    """Additional tests for TokenTracker coverage."""

    def test_add_usage_with_none_usage(self) -> None:
        """Test add_usage handles None gracefully."""
        tracker = TokenTracker()
        tracker.add_usage("gpt-4", None, "judging")

        assert tracker.judging.request_count == 0

    def test_add_usage_invalid_phase(self) -> None:
        """Test add_usage handles invalid phase gracefully."""
        tracker = TokenTracker()
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        # Should not raise for invalid phase
        tracker.add_usage("gpt-4", usage, "invalid_phase")

        # Should still track by model
        assert "gpt-4" in tracker.by_model

    @patch("backend.core.dts.types.get_model_pricing")
    def test_get_pricing_method(self, mock_pricing) -> None:
        """Test TokenTracker.get_pricing method."""
        mock_pricing.return_value = ModelPricing("test", 10.0, 20.0)

        tracker = TokenTracker(model_name="test")
        pricing = tracker.get_pricing()

        assert pricing.model_name == "test"
        mock_pricing.assert_called_once_with("test")

    @patch("backend.core.dts.types.get_model_pricing")
    def test_to_dict_with_by_model(self, mock_pricing) -> None:
        """Test to_dict includes per-model breakdown."""
        mock_pricing.return_value = ModelPricing("gpt-4", 30.0, 60.0)

        tracker = TokenTracker(model_name="gpt-4")
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        tracker.add_usage("gpt-4", usage, "strategy_generation")

        result = tracker.to_dict()

        assert "by_model" in result
        assert "gpt-4" in result["by_model"]
        assert result["by_model"]["gpt-4"]["input_tokens"] == 1000
