"""Configuration for Dialogue Tree Search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ScoringMode = Literal["absolute", "comparative"]


@dataclass
class DTSConfig:
    """
    Configuration for Dialogue Tree Search engine.

    Attributes:
        goal: Conversation goal/objective.
        first_message: Initial user message to start the conversation.
        init_branches: Number of initial strategy branches to create.
        deep_research: Whether to include deep research context.
        turns_per_branch: Number of turns (user+assistant) per expansion.
        user_intents_per_branch: Number of user intents to fork per expansion (1 = no forking).
        scoring_mode: "absolute" (3 independent judges) or "comparative" (forced ranking).
        prune_threshold: Score threshold for pruning (0-10).
        keep_top_k: Keep only top K branches after pruning (optional).
        min_survivors: Minimum branches to keep even if below threshold.
        max_concurrency: Maximum concurrent LLM calls.
        model: Model to use for completions.
        temperature: Temperature for conversation generation.
        judge_temperature: Temperature for judge evaluations (lower = more deterministic).
    """

    goal: str
    first_message: str
    init_branches: int = 6
    deep_research: bool = False
    turns_per_branch: int = 5
    user_intents_per_branch: int = 3
    scoring_mode: ScoringMode = "comparative"
    prune_threshold: float = 6.5
    keep_top_k: int | None = None
    min_survivors: int = 1
    max_concurrency: int = 16
    model: str | None = None
    temperature: float = 0.7
    judge_temperature: float = 0.3
