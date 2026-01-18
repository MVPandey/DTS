"""Tests for backend/core/dts/aggregator.py."""

import pytest

from backend.core.dts.aggregator import aggregate_majority_vote
from backend.core.dts.types import AggregatedScore


class TestAggregateMajorityVote:
    """Tests for the aggregate_majority_vote function."""

    def test_basic_aggregation(self) -> None:
        """Test basic majority vote aggregation."""
        scores = [7.0, 8.0, 6.0]
        result = aggregate_majority_vote(scores)

        assert isinstance(result, AggregatedScore)
        assert result.individual_scores == [7.0, 8.0, 6.0]
        assert result.aggregated_score == 7.0  # median

    def test_median_calculation(self) -> None:
        """Test that median is correctly calculated."""
        # Median of sorted [5.0, 7.0, 9.0] is 7.0
        scores = [9.0, 5.0, 7.0]
        result = aggregate_majority_vote(scores)

        assert result.aggregated_score == 7.0

    def test_pass_votes_all_pass(self) -> None:
        """Test pass_votes when all scores pass threshold."""
        scores = [7.0, 8.0, 6.0]  # All >= 5.0 threshold
        result = aggregate_majority_vote(scores, pass_threshold=5.0)

        assert result.pass_votes == 3
        assert result.passed is True

    def test_pass_votes_majority_pass(self) -> None:
        """Test pass_votes when majority passes threshold."""
        scores = [7.0, 8.0, 4.0]  # 2 >= 5.0 threshold
        result = aggregate_majority_vote(scores, pass_threshold=5.0)

        assert result.pass_votes == 2
        assert result.passed is True

    def test_pass_votes_minority_pass(self) -> None:
        """Test pass_votes when minority passes threshold."""
        scores = [3.0, 4.0, 6.0]  # Only 1 >= 5.0 threshold
        result = aggregate_majority_vote(scores, pass_threshold=5.0)

        assert result.pass_votes == 1
        assert result.passed is False

    def test_pass_votes_none_pass(self) -> None:
        """Test pass_votes when no scores pass threshold."""
        scores = [2.0, 3.0, 4.0]  # None >= 5.0 threshold
        result = aggregate_majority_vote(scores, pass_threshold=5.0)

        assert result.pass_votes == 0
        assert result.passed is False

    def test_custom_threshold(self) -> None:
        """Test with custom pass threshold."""
        scores = [7.0, 7.5, 8.0]
        result = aggregate_majority_vote(scores, pass_threshold=7.5)

        assert result.pass_threshold == 7.5
        assert result.pass_votes == 2  # 7.5 and 8.0 pass
        assert result.passed is True

    def test_high_threshold(self) -> None:
        """Test with high threshold."""
        scores = [9.0, 9.5, 8.5]
        result = aggregate_majority_vote(scores, pass_threshold=9.0)

        assert result.pass_votes == 2  # 9.0 and 9.5 pass
        assert result.passed is True

    def test_edge_case_exact_threshold(self) -> None:
        """Test scores exactly at threshold."""
        scores = [5.0, 5.0, 5.0]
        result = aggregate_majority_vote(scores, pass_threshold=5.0)

        assert result.pass_votes == 3
        assert result.passed is True
        assert result.aggregated_score == 5.0

    def test_wrong_number_of_scores_too_few(self) -> None:
        """Test that fewer than 3 scores raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            aggregate_majority_vote([7.0, 8.0])

        assert "Expected exactly 3 scores" in str(exc_info.value)

    def test_wrong_number_of_scores_too_many(self) -> None:
        """Test that more than 3 scores raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            aggregate_majority_vote([7.0, 8.0, 6.0, 5.0])

        assert "Expected exactly 3 scores" in str(exc_info.value)

    def test_wrong_number_of_scores_empty(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            aggregate_majority_vote([])

        assert "Expected exactly 3 scores" in str(exc_info.value)

    def test_decimal_scores(self) -> None:
        """Test with decimal scores."""
        scores = [7.25, 7.75, 7.50]
        result = aggregate_majority_vote(scores)

        assert result.aggregated_score == 7.50

    def test_zero_scores(self) -> None:
        """Test with zero scores."""
        scores = [0.0, 0.0, 0.0]
        result = aggregate_majority_vote(scores, pass_threshold=5.0)

        assert result.aggregated_score == 0.0
        assert result.pass_votes == 0
        assert result.passed is False

    def test_max_scores(self) -> None:
        """Test with maximum scores."""
        scores = [10.0, 10.0, 10.0]
        result = aggregate_majority_vote(scores)

        assert result.aggregated_score == 10.0
        assert result.pass_votes == 3
        assert result.passed is True

    def test_mixed_extreme_scores(self) -> None:
        """Test with extreme range of scores."""
        scores = [0.0, 5.0, 10.0]
        result = aggregate_majority_vote(scores)

        assert result.aggregated_score == 5.0

    def test_result_contains_all_fields(self) -> None:
        """Test that result contains all expected fields."""
        scores = [7.0, 8.0, 6.0]
        result = aggregate_majority_vote(scores, pass_threshold=6.5)

        assert hasattr(result, "individual_scores")
        assert hasattr(result, "aggregated_score")
        assert hasattr(result, "pass_threshold")
        assert hasattr(result, "pass_votes")
        assert hasattr(result, "passed")

        assert result.individual_scores == [7.0, 8.0, 6.0]
        assert result.pass_threshold == 6.5
