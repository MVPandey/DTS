"""Majority vote aggregation for multiple judges."""

from __future__ import annotations

from backend.core.mcts.types import AggregatedScore


def aggregate_majority_vote(
    scores: list[float],
    pass_threshold: float = 5.0,
) -> AggregatedScore:
    """
    Aggregate 3 judge scores using median (majority vote).

    The median is used because with exactly 3 values, it represents
    the "majority" in terms of being robust to 1 outlier judge.

    Args:
        scores: List of exactly 3 scores (0-10 scale)
        pass_threshold: Score threshold for pass/fail (default 5.0)

    Returns:
        AggregatedScore with individual scores, median, pass votes, and passed status

    Raises:
        ValueError: If not exactly 3 scores provided
    """
    if len(scores) != 3:
        raise ValueError(f"Expected exactly 3 scores, got {len(scores)}")

    sorted_scores = sorted(scores)
    aggregated = sorted_scores[1]  # median of 3

    pass_votes = sum(1 for s in scores if s >= pass_threshold)
    passed = pass_votes >= 2  # majority (2 out of 3) must pass

    return AggregatedScore(
        individual_scores=scores,
        aggregated_score=aggregated,
        pass_threshold=pass_threshold,
        pass_votes=pass_votes,
        passed=passed,
    )


def aggregate_mean(scores: list[float]) -> float:
    """Simple mean aggregation (alternative to median)."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def aggregate_with_confidence_weighting(
    scores: list[float],
    confidences: list[str],
) -> float:
    """
    Aggregate scores with confidence weighting.

    Judges with higher confidence have more weight.

    Args:
        scores: List of scores
        confidences: List of confidence levels ("low", "medium", "high")

    Returns:
        Weighted average score
    """
    if not scores or len(scores) != len(confidences):
        return aggregate_mean(scores)

    weights = {"low": 1.0, "medium": 2.0, "high": 3.0}

    weighted_sum = sum(
        s * weights.get(c, 1.0) for s, c in zip(scores, confidences)
    )
    total_weight = sum(weights.get(c, 1.0) for c in confidences)

    return weighted_sum / total_weight if total_weight > 0 else 0.0
