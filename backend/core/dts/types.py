"""Data models for Dialogue Tree Search."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from backend.llm.types import Message, Usage


@dataclass
class ModelPricing:
    """Pricing per 1M tokens for a model."""

    model_name: str
    input_cost_per_million: float  # $ per 1M input tokens
    output_cost_per_million: float  # $ per 1M output tokens

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost in dollars."""
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        return input_cost + output_cost


# Common model pricing configurations
MODEL_PRICING = {
    "minimax/minimax-m2.1": ModelPricing(
        model_name="minimax/minimax-m2.1",
        input_cost_per_million=0.30,
        output_cost_per_million=1.20,
    ),
    "z-ai/glm-4.7": ModelPricing(
        model_name="z-ai/glm-4.7",
        input_cost_per_million=0.40,
        output_cost_per_million=1.50,
    ),
}


@dataclass
class TokenStats:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0

    def add(self, usage: Usage | None) -> None:
        """Add usage from a completion."""
        if usage:
            self.input_tokens += usage.prompt_tokens
            self.output_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            self.request_count += 1

    def merge(self, other: "TokenStats") -> None:
        """Merge another TokenStats into this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.request_count += other.request_count


@dataclass
class TokenTracker:
    """
    Tracks token usage and costs across a DTS run.

    Separates usage by phase for detailed analysis.
    """

    model_name: str = "minimax/minimax-m2.1"

    # Per-phase tracking
    strategy_generation: TokenStats = field(default_factory=TokenStats)
    intent_generation: TokenStats = field(default_factory=TokenStats)
    user_simulation: TokenStats = field(default_factory=TokenStats)
    assistant_generation: TokenStats = field(default_factory=TokenStats)
    judging: TokenStats = field(default_factory=TokenStats)

    def get_pricing(self) -> ModelPricing:
        """Get pricing for the current model."""
        return MODEL_PRICING.get(
            self.model_name,
            ModelPricing(self.model_name, 0.0, 0.0),  # Unknown model = no cost
        )

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all phases."""
        return (
            self.strategy_generation.input_tokens
            + self.intent_generation.input_tokens
            + self.user_simulation.input_tokens
            + self.assistant_generation.input_tokens
            + self.judging.input_tokens
        )

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all phases."""
        return (
            self.strategy_generation.output_tokens
            + self.intent_generation.output_tokens
            + self.user_simulation.output_tokens
            + self.assistant_generation.output_tokens
            + self.judging.output_tokens
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens across all phases."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_requests(self) -> int:
        """Total LLM requests made."""
        return (
            self.strategy_generation.request_count
            + self.intent_generation.request_count
            + self.user_simulation.request_count
            + self.assistant_generation.request_count
            + self.judging.request_count
        )

    @property
    def total_cost(self) -> float:
        """Total cost in dollars."""
        pricing = self.get_pricing()
        return pricing.calculate_cost(self.total_input_tokens, self.total_output_tokens)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        pricing = self.get_pricing()
        return {
            "model": self.model_name,
            "pricing": {
                "input_per_million": pricing.input_cost_per_million,
                "output_per_million": pricing.output_cost_per_million,
            },
            "totals": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "total_requests": self.total_requests,
                "total_cost_usd": round(self.total_cost, 6),
            },
            "by_phase": {
                "strategy_generation": {
                    "input_tokens": self.strategy_generation.input_tokens,
                    "output_tokens": self.strategy_generation.output_tokens,
                    "requests": self.strategy_generation.request_count,
                },
                "intent_generation": {
                    "input_tokens": self.intent_generation.input_tokens,
                    "output_tokens": self.intent_generation.output_tokens,
                    "requests": self.intent_generation.request_count,
                },
                "user_simulation": {
                    "input_tokens": self.user_simulation.input_tokens,
                    "output_tokens": self.user_simulation.output_tokens,
                    "requests": self.user_simulation.request_count,
                },
                "assistant_generation": {
                    "input_tokens": self.assistant_generation.input_tokens,
                    "output_tokens": self.assistant_generation.output_tokens,
                    "requests": self.assistant_generation.request_count,
                },
                "judging": {
                    "input_tokens": self.judging.input_tokens,
                    "output_tokens": self.judging.output_tokens,
                    "requests": self.judging.request_count,
                },
            },
        }

    def print_summary(self) -> None:
        """Print a formatted summary of token usage and costs."""
        pricing = self.get_pricing()

        print("\n" + "=" * 60)
        print("TOKEN USAGE & COST SUMMARY")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(
            f"Pricing: ${pricing.input_cost_per_million:.2f}/1M input, "
            f"${pricing.output_cost_per_million:.2f}/1M output"
        )
        print("-" * 60)

        # Totals
        print(f"{'Total Input Tokens:':<30} {self.total_input_tokens:>15,}")
        print(f"{'Total Output Tokens:':<30} {self.total_output_tokens:>15,}")
        print(f"{'Total Tokens:':<30} {self.total_tokens:>15,}")
        print(f"{'Total Requests:':<30} {self.total_requests:>15}")
        print("-" * 60)

        # Cost breakdown
        input_cost = (
            self.total_input_tokens / 1_000_000
        ) * pricing.input_cost_per_million
        output_cost = (
            self.total_output_tokens / 1_000_000
        ) * pricing.output_cost_per_million

        print(f"{'Input Cost:':<30} ${input_cost:>14.6f}")
        print(f"{'Output Cost:':<30} ${output_cost:>14.6f}")
        print(f"{'TOTAL COST:':<30} ${self.total_cost:>14.6f}")
        print("-" * 60)

        # Per-phase breakdown
        print("\nBy Phase:")
        phases = [
            ("Strategy Generation", self.strategy_generation),
            ("Intent Generation", self.intent_generation),
            ("User Simulation", self.user_simulation),
            ("Assistant Generation", self.assistant_generation),
            ("Judging", self.judging),
        ]
        for name, stats in phases:
            if stats.request_count > 0:
                phase_cost = pricing.calculate_cost(
                    stats.input_tokens, stats.output_tokens
                )
                print(
                    f"  {name:<22} | {stats.request_count:>4} reqs | "
                    f"{stats.input_tokens:>8,} in | {stats.output_tokens:>8,} out | "
                    f"${phase_cost:.4f}"
                )
        print("=" * 60)


class NodeStatus(str, Enum):
    """Status of a tree node."""

    ACTIVE = "active"
    PRUNED = "pruned"
    TERMINAL = "terminal"
    ERROR = "error"


class Strategy(BaseModel):
    """A conversation strategy for branch exploration."""

    tagline: str
    description: str


class UserIntent(BaseModel):
    """A specific user response intent for branch forking."""

    id: str
    label: str
    description: str
    emotional_tone: str  # engaged, resistant, confused, skeptical, enthusiastic, deflecting, anxious, neutral
    cognitive_stance: str  # accepting, questioning, challenging, exploring, withdrawing


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
    """Statistics for a dialogue node."""

    visits: int = 0
    value_sum: float = 0.0
    value_mean: float = 0.0
    judge_scores: list[float] = Field(default_factory=list)
    aggregated_score: float = 0.0


class DialogueNode(BaseModel):
    """A node in the dialogue tree representing a conversation state."""

    id: str
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    depth: int = 0
    status: NodeStatus = NodeStatus.ACTIVE

    # Branch descriptor (strategy that led to this node)
    strategy: Strategy | None = None

    # User intent that led to this branch (for forked nodes)
    user_intent: UserIntent | None = None

    # Conversation trajectory to this node
    messages: list[Message] = Field(default_factory=list)

    # Statistics for scoring
    stats: NodeStats = Field(default_factory=NodeStats)

    # Pruning metadata
    prune_reason: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class TreeGeneratorOutput(BaseModel):
    """Parsed output from conversation_tree_generator prompt."""

    goal: str
    nodes: dict[str, str]  # tagline -> description
    coverage_rationale: str


class DTSRunResult(BaseModel):
    """Result of running the Dialogue Tree Search."""

    best_node_id: str | None = None
    best_score: float = 0.0
    best_messages: list[Message] = Field(default_factory=list)
    all_nodes: list[DialogueNode] = Field(default_factory=list)
    pruned_count: int = 0
    total_rounds: int = 0

    # Token usage tracking (populated after run)
    token_usage: dict | None = None

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
                "user_intent": {
                    "label": node.user_intent.label,
                    "emotional_tone": node.user_intent.emotional_tone,
                    "cognitive_stance": node.user_intent.cognitive_stance,
                }
                if node.user_intent
                else None,
                "status": node.status.value,
                "depth": node.depth,
                "scores": {
                    "individual": node.stats.judge_scores,
                    "aggregated": node.stats.aggregated_score,
                    "visits": node.stats.visits,
                    "value_mean": node.stats.value_mean,
                },
                "trajectory": [
                    {"role": msg.role, "content": msg.content} for msg in node.messages
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

        result = {
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

        # Include token usage if available
        if self.token_usage:
            result["token_usage"] = self.token_usage

        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to formatted JSON string for exploration."""
        import json

        return json.dumps(self.to_exploration_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: str) -> None:
        """Save exploration data to a JSON file."""
        from pathlib import Path

        Path(path).write_text(self.to_json(), encoding="utf-8")
        print(f"[DTS:SAVE] Results saved to {path}")
