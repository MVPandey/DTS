"""MCTS (Monte Carlo Tree Search) agent for conversational AI."""

from backend.core.mcts.agent import MCTSAgent
from backend.core.mcts.aggregator import (
    aggregate_majority_vote,
    aggregate_mean,
    aggregate_with_confidence_weighting,
)
from backend.core.mcts.tree import MCTSTree, generate_node_id
from backend.core.mcts.types import (
    AggregatedScore,
    BranchSelectionEvaluation,
    BranchStrategy,
    CriterionScore,
    MCTSNode,
    MCTSRunResult,
    NodeStats,
    NodeStatus,
    TrajectoryEvaluation,
    TreeGeneratorOutput,
)

__all__ = [
    # Agent
    "MCTSAgent",
    # Tree
    "MCTSTree",
    "generate_node_id",
    # Types
    "AggregatedScore",
    "BranchSelectionEvaluation",
    "BranchStrategy",
    "CriterionScore",
    "MCTSNode",
    "MCTSRunResult",
    "NodeStats",
    "NodeStatus",
    "TrajectoryEvaluation",
    "TreeGeneratorOutput",
    # Aggregation
    "aggregate_majority_vote",
    "aggregate_mean",
    "aggregate_with_confidence_weighting",
]
