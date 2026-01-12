"""Dialogue Tree Search (DTS) for conversational AI.

A parallel beam search algorithm that explores conversation trajectories
by generating diverse strategies, simulating multi-turn dialogues, and
evaluating trajectories with multiple judges.

Usage:
    from backend.core.dts import DTSEngine, DTSConfig

    engine = DTSEngine(
        llm=llm,
        config=DTSConfig(
            goal="Help user with their problem",
            first_message="I need help with...",
        ),
    )
    result = await engine.run(rounds=2)
"""

from backend.core.dts.aggregator import aggregate_majority_vote
from backend.core.dts.components import (
    ConversationSimulator,
    StrategyGenerator,
    TrajectoryEvaluator,
)
from backend.core.dts.config import DTSConfig
from backend.core.dts.engine import DTSEngine
from backend.core.dts.tree import (
    DialogueTree,
    generate_node_id,
)
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
    TokenTracker,
    TrajectoryEvaluation,
    TreeGeneratorOutput,
    UserIntent,
)

__all__ = [
    # Engine
    "DTSEngine",
    "DTSConfig",
    # Components
    "ConversationSimulator",
    "StrategyGenerator",
    "TrajectoryEvaluator",
    # Tree
    "DialogueTree",
    "generate_node_id",
    # Types
    "AggregatedScore",
    "BranchSelectionEvaluation",
    "CriterionScore",
    "DialogueNode",
    "DTSRunResult",
    "ModelPricing",
    "NodeStats",
    "NodeStatus",
    "Strategy",
    "TokenTracker",
    "TrajectoryEvaluation",
    "TreeGeneratorOutput",
    "UserIntent",
    # Aggregation
    "aggregate_majority_vote",
]
