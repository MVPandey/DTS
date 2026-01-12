"""API request/response schemas for DTS WebSocket server."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.core.dts.config import ScoringMode


class SearchRequest(BaseModel):
    """Request to start a DTS search."""

    goal: str = Field(..., description="Conversation goal/objective")
    first_message: str = Field(..., description="Initial user message")
    init_branches: int = Field(default=6, ge=1, le=20, description="Initial strategy branches")
    turns_per_branch: int = Field(default=5, ge=1, le=20, description="Turns per expansion")
    user_intents_per_branch: int = Field(
        default=3, ge=1, le=10, description="User intent forks per branch"
    )
    scoring_mode: ScoringMode = Field(default="comparative", description="Scoring mode")
    prune_threshold: float = Field(default=6.5, ge=0.0, le=10.0, description="Pruning threshold")
    rounds: int = Field(default=1, ge=1, le=10, description="Number of search rounds")
    deep_research: bool = Field(default=False, description="Enable deep research context")
    strategy_model: str | None = Field(
        default=None, description="Model for strategy/intent generation"
    )
    simulator_model: str | None = Field(
        default=None, description="Model for conversation simulation"
    )
    judge_model: str | None = Field(default=None, description="Model for trajectory evaluation")


class EventMessage(BaseModel):
    """WebSocket event message format."""

    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class ErrorData(BaseModel):
    """Error response data."""

    message: str
    code: str | None = None


class SearchStartedData(BaseModel):
    """Data for search_started event."""

    goal: str
    first_message: str
    total_rounds: int
    config: dict[str, Any]


class PhaseData(BaseModel):
    """Data for phase event."""

    phase: Literal[
        "initializing",
        "generating_strategies",
        "expanding",
        "scoring",
        "pruning",
        "complete",
    ]
    message: str


class StrategyGeneratedData(BaseModel):
    """Data for strategy_generated event."""

    index: int
    total: int
    tagline: str
    description: str


class NodeAddedData(BaseModel):
    """Data for node_added event."""

    id: str
    parent_id: str | None
    depth: int
    status: str
    strategy: str | None
    user_intent: str | None
    message_count: int


class NodeUpdatedData(BaseModel):
    """Data for node_updated event."""

    id: str
    status: str
    score: float
    individual_scores: list[float]
    passed: bool


class RoundStartedData(BaseModel):
    """Data for round_started event."""

    round: int
    total_rounds: int
