"""Data models for Dialogue Tree Search."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from backend.llm.types import Message, Usage
from backend.utils.logging import logger


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


# Pricing cache - populated dynamically from OpenRouter API
_pricing_cache: dict[str, ModelPricing] = {}
_pricing_loaded: bool = False


def _load_pricing_from_openrouter() -> None:
    """Fetch model pricing from OpenRouter API and cache it."""
    global _pricing_cache, _pricing_loaded
    if _pricing_loaded:
        return

    try:
        with urllib.request.urlopen(
            "https://openrouter.ai/api/v1/models", timeout=10
        ) as response:
            data = json.loads(response.read().decode())

        for model in data.get("data", []):
            model_id = model.get("id", "")
            pricing = model.get("pricing", {})

            # OpenRouter returns price per token, convert to per million
            prompt_per_token = float(pricing.get("prompt", 0))
            completion_per_token = float(pricing.get("completion", 0))

            _pricing_cache[model_id] = ModelPricing(
                model_name=model_id,
                input_cost_per_million=prompt_per_token * 1_000_000,
                output_cost_per_million=completion_per_token * 1_000_000,
            )

        _pricing_loaded = True
        logger.debug(f"Loaded pricing for {len(_pricing_cache)} models from OpenRouter")

    except Exception as e:
        logger.warning(f"Failed to load pricing from OpenRouter: {e}")
        _pricing_loaded = True  # Don't retry on failure


def get_model_pricing(model_name: str) -> ModelPricing:
    """Get pricing for a model, fetching from OpenRouter if needed."""
    _load_pricing_from_openrouter()

    if model_name in _pricing_cache:
        return _pricing_cache[model_name]

    # Return zero pricing for unknown models with a warning
    logger.warning(f"No pricing found for model '{model_name}' - cost will be $0")
    return ModelPricing(model_name, 0.0, 0.0)


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


# Phase names used for token tracking
TOKEN_PHASES = (
    "strategy_generation",
    "intent_generation",
    "user_simulation",
    "assistant_generation",
    "judging",
    "research",
)


@dataclass
class TokenTracker:
    """
    Tracks token usage and costs across a DTS run.

    Separates usage by phase and by model for accurate cost calculation.
    """

    model_name: str = "unknown"  # Primary model name (for display)

    # Per-phase tracking
    strategy_generation: TokenStats = field(default_factory=TokenStats)
    intent_generation: TokenStats = field(default_factory=TokenStats)
    user_simulation: TokenStats = field(default_factory=TokenStats)
    assistant_generation: TokenStats = field(default_factory=TokenStats)
    judging: TokenStats = field(default_factory=TokenStats)
    research: TokenStats = field(default_factory=TokenStats)

    # Per-model tracking for accurate cost calculation
    by_model: dict[str, TokenStats] = field(default_factory=dict)

    # External costs (e.g., GPT Researcher uses its own LLM client)
    research_cost_usd: float = 0.0

    def add_usage(self, model: str, usage: "Usage | None", phase: str) -> None:
        """Track usage for a specific model and phase."""
        if not usage:
            return

        # Track by phase
        phase_stats = getattr(self, phase.replace("-", "_"), None)
        if isinstance(phase_stats, TokenStats):
            phase_stats.add(usage)

        # Track by model for accurate cost calculation
        if model not in self.by_model:
            self.by_model[model] = TokenStats()
        self.by_model[model].add(usage)

    def get_pricing(self) -> ModelPricing:
        """Get pricing for the primary model (fetched from OpenRouter API)."""
        return get_model_pricing(self.model_name)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all phases."""
        return sum(getattr(self, phase).input_tokens for phase in TOKEN_PHASES)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all phases."""
        return sum(getattr(self, phase).output_tokens for phase in TOKEN_PHASES)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all phases."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_requests(self) -> int:
        """Total LLM requests made."""
        return sum(getattr(self, phase).request_count for phase in TOKEN_PHASES)

    @property
    def total_cost(self) -> float:
        """Total cost in dollars (calculated per-model for accuracy)."""
        total = 0.0
        for model_name, stats in self.by_model.items():
            pricing = get_model_pricing(model_name)
            total += pricing.calculate_cost(stats.input_tokens, stats.output_tokens)
        return total + self.research_cost_usd

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        # Build per-model breakdown
        by_model_dict = {}
        for model_name, stats in self.by_model.items():
            pricing = get_model_pricing(model_name)
            cost = pricing.calculate_cost(stats.input_tokens, stats.output_tokens)
            by_model_dict[model_name] = {
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "requests": stats.request_count,
                "cost_usd": round(cost, 6),
                "pricing": {
                    "input_per_million": pricing.input_cost_per_million,
                    "output_per_million": pricing.output_cost_per_million,
                },
            }

        return {
            "models_used": list(self.by_model.keys()),
            "totals": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "total_requests": self.total_requests,
                "total_cost_usd": round(self.total_cost, 6),
            },
            "by_model": by_model_dict,
            "by_phase": self._build_phase_dict(),
        }

    def _build_phase_dict(self) -> dict[str, dict]:
        """Build per-phase statistics dictionary."""
        result = {}
        for phase in TOKEN_PHASES:
            stats: TokenStats = getattr(self, phase)
            phase_data = {
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "requests": stats.request_count,
            }
            if phase == "research":
                phase_data["external_cost_usd"] = round(self.research_cost_usd, 6)
            result[phase] = phase_data
        return result

    def print_summary(self) -> None:
        """Print a formatted summary of token usage and costs."""
        print("\n" + "=" * 60)
        print("TOKEN USAGE & COST SUMMARY")
        print("=" * 60)

        # Show models used
        models_used = list(self.by_model.keys())
        if models_used:
            print(f"Models: {', '.join(models_used)}")
        else:
            print(f"Model: {self.model_name}")
        print("-" * 60)

        # Totals
        print(f"{'Total Input Tokens:':<30} {self.total_input_tokens:>15,}")
        print(f"{'Total Output Tokens:':<30} {self.total_output_tokens:>15,}")
        print(f"{'Total Tokens:':<30} {self.total_tokens:>15,}")
        print(f"{'Total Requests:':<30} {self.total_requests:>15}")
        print(f"{'TOTAL COST:':<30} ${self.total_cost:>14.6f}")
        print("-" * 60)

        # Per-model breakdown
        if self.by_model:
            print("\nBy Model:")
            for model_name, stats in self.by_model.items():
                pricing = get_model_pricing(model_name)
                model_cost = pricing.calculate_cost(
                    stats.input_tokens, stats.output_tokens
                )
                print(
                    f"  {model_name:<35} | {stats.request_count:>4} reqs | "
                    f"${model_cost:.4f}"
                )
                print(
                    f"    Pricing: ${pricing.input_cost_per_million:.2f}/1M in, "
                    f"${pricing.output_cost_per_million:.2f}/1M out"
                )

        # Per-phase breakdown
        print("\nBy Phase:")
        phase_names = {
            "strategy_generation": "Strategy Generation",
            "intent_generation": "Intent Generation",
            "user_simulation": "User Simulation",
            "assistant_generation": "Assistant Generation",
            "judging": "Judging",
            "research": "Research",
        }
        for phase in TOKEN_PHASES:
            stats: TokenStats = getattr(self, phase)
            if stats.request_count > 0:
                print(
                    f"  {phase_names[phase]:<22} | {stats.request_count:>4} reqs | "
                    f"{stats.input_tokens:>8,} in | {stats.output_tokens:>8,} out"
                )

        # External research cost (GPT Researcher uses its own LLM)
        if self.research_cost_usd > 0:
            print(
                f"  {'Research (external)':<22} | {'N/A':>4} reqs | "
                f"{'N/A':>8} in | {'N/A':>8} out | "
                f"${self.research_cost_usd:.4f}"
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

    @classmethod
    def zero(cls, threshold: float = 5.0) -> "AggregatedScore":
        """Create a zero score for error/fallback cases."""
        return cls(
            individual_scores=[0.0, 0.0, 0.0],
            aggregated_score=0.0,
            pass_threshold=threshold,
            pass_votes=0,
            passed=False,
        )


class NodeStats(BaseModel):
    """Statistics for a dialogue node."""

    visits: int = 0
    value_sum: float = 0.0
    value_mean: float = 0.0
    judge_scores: list[float] = Field(default_factory=list)
    aggregated_score: float = 0.0
    # Critique from comparative judging
    critiques: dict[str, list[str] | str] = Field(
        default_factory=dict
    )  # {weaknesses: [], strengths: [], key_moment: ""}


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

    @property
    def strategy_label(self) -> str:
        """Get strategy tagline or 'unknown'."""
        return self.strategy.tagline if self.strategy else "unknown"

    @property
    def intent_label(self) -> str | None:
        """Get user intent label or None."""
        return self.user_intent.label if self.user_intent else None

    def update_with_evaluation(
        self, score: AggregatedScore, critiques: dict | None = None
    ) -> None:
        """Update node stats with evaluation results."""
        self.stats.judge_scores = score.individual_scores
        self.stats.aggregated_score = score.aggregated_score
        if critiques:
            self.stats.critiques = critiques


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

    # Deep research report if generated
    research_report: str | None = None

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
                    "critiques": node.stats.critiques if node.stats.critiques else None,
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
            "research_report": self.research_report,
            "best_branch": best_branch,
            "branches": branches,
        }

        # Include token usage if available
        if self.token_usage:
            result["token_usage"] = self.token_usage

        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to formatted JSON string for exploration."""
        return json.dumps(self.to_exploration_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: str) -> None:
        """Save exploration data to a JSON file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")
        logger.info(f"Results saved to {path}")
