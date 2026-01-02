"""Data models for MCTS agent."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from backend.llm.types import Message


class NodeStatus(str, Enum):
    """Status of a tree node."""

    ACTIVE = "active"
    PRUNED = "pruned"
    TERMINAL = "terminal"
    ERROR = "error"


class BranchStrategy(BaseModel):
    """Output from conversation_tree_generator prompt."""

    tagline: str
    description: str


class CriterionScore(BaseModel):
    """Score for a single evaluation criterion."""

    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class BranchSelectionEvaluation(BaseModel):
    """Output from branch_selection_judge prompt (pre-exploration)."""

    criteria: dict[str, CriterionScore]
    total_score: float = Field(ge=0.0, le=10.0)
    confidence: Literal["low", "medium", "high"]
    summary: str


class TrajectoryEvaluation(BaseModel):
    """Output from trajectory_outcome_judge prompt (post-rollout)."""

    criteria: dict[str, CriterionScore]
    total_score: float = Field(ge=0.0, le=10.0)
    confidence: Literal["low", "medium", "high"]
    summary: str
    key_turning_point: str | None = None


class AggregatedScore(BaseModel):
    """Result of majority vote aggregation from 3 judges."""

    individual_scores: list[float] = Field(min_length=3, max_length=3)
    aggregated_score: float  # median of 3 scores
    pass_threshold: float = 5.0
    pass_votes: int = Field(ge=0, le=3)  # count of scores >= threshold
    passed: bool  # True if pass_votes >= 2


class NodeStats(BaseModel):
    """MCTS statistics for a node."""

    visits: int = 0
    value_sum: float = 0.0
    value_mean: float = 0.0
    judge_scores: list[float] = Field(default_factory=list)
    aggregated_score: float = 0.0


class MCTSNode(BaseModel):
    """A node in the MCTS tree."""

    id: str
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    depth: int = 0
    status: NodeStatus = NodeStatus.ACTIVE

    # Branch descriptor (strategy that led to this node)
    strategy: BranchStrategy | None = None

    # Conversation trajectory to this node
    messages: list[Message] = Field(default_factory=list)

    # Statistics for MCTS
    stats: NodeStats = Field(default_factory=NodeStats)

    # Pruning metadata
    prune_reason: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class TreeGeneratorOutput(BaseModel):
    """Parsed output from conversation_tree_generator prompt."""

    goal: str
    nodes: dict[str, str]  # tagline -> description
    coverage_rationale: str


class MCTSRunResult(BaseModel):
    """Result of running the MCTS agent."""

    best_node_id: str | None = None
    best_score: float = 0.0
    best_messages: list[Message] = Field(default_factory=list)
    all_nodes: list[MCTSNode] = Field(default_factory=list)
    pruned_count: int = 0
    total_rounds: int = 0

    model_config = {"arbitrary_types_allowed": True}

    def to_exploration_dict(self) -> dict:
        """
        Convert to a dict optimized for exploring branches and scores.

        Structure:
        {
            "summary": { ... },
            "best_branch": { ... },
            "branches": [
                {
                    "id": "...",
                    "strategy": { "tagline": "...", "description": "..." },
                    "status": "active|pruned",
                    "scores": { "individual": [...], "aggregated": ..., "passed": ... },
                    "trajectory": [
                        { "role": "user", "content": "..." },
                        { "role": "assistant", "content": "..." },
                        ...
                    ],
                    "prune_reason": "..." or null
                },
                ...
            ]
        }
        """
        # Build branches list (excluding root)
        branches = []
        for node in self.all_nodes:
            if node.strategy is None:
                continue  # Skip root node

            branch_data = {
                "id": node.id,
                "strategy": {
                    "tagline": node.strategy.tagline,
                    "description": node.strategy.description,
                },
                "status": node.status.value,
                "depth": node.depth,
                "scores": {
                    "individual": node.stats.judge_scores,
                    "aggregated": node.stats.aggregated_score,
                    "visits": node.stats.visits,
                    "value_mean": node.stats.value_mean,
                },
                "trajectory": [
                    {"role": msg.role, "content": msg.content}
                    for msg in node.messages
                ],
                "prune_reason": node.prune_reason,
            }
            branches.append(branch_data)

        # Sort by score descending
        branches.sort(key=lambda b: b["scores"]["aggregated"], reverse=True)

        # Build best branch info
        best_branch = None
        if self.best_node_id:
            for node in self.all_nodes:
                if node.id == self.best_node_id:
                    best_branch = {
                        "id": node.id,
                        "strategy": node.strategy.tagline if node.strategy else "root",
                        "score": self.best_score,
                        "trajectory": [
                            {"role": msg.role, "content": msg.content}
                            for msg in node.messages
                        ],
                    }
                    break

        # Count stats
        active_count = sum(1 for n in self.all_nodes if n.status == NodeStatus.ACTIVE)
        pruned_count = sum(1 for n in self.all_nodes if n.status == NodeStatus.PRUNED)

        return {
            "summary": {
                "total_branches": len(branches),
                "active_branches": active_count,
                "pruned_branches": pruned_count,
                "total_rounds": self.total_rounds,
                "best_score": self.best_score,
            },
            "best_branch": best_branch,
            "branches": branches,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to formatted JSON string for exploration."""
        import json

        return json.dumps(self.to_exploration_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: str) -> None:
        """Save exploration data to a JSON file."""
        from pathlib import Path

        Path(path).write_text(self.to_json(), encoding="utf-8")
        print(f"[MCTS:SAVE] Results saved to {path}")
