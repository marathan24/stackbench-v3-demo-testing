"""
Tests for uncertainty quantification module.
"""

import pytest
import numpy as np
from stackbench.evaluation.uncertainty import (
    calculate_confidence_interval,
    should_flag_for_review,
    get_confidence_category,
    UncertaintyEstimate,
)


def test_calculate_confidence_interval_basic():
    """Test confidence interval calculation with basic samples."""
    samples = [7.0, 7.2, 7.5, 7.3, 7.1, 7.4, 7.6, 7.2, 7.3, 7.4]

    lower, upper = calculate_confidence_interval(samples, confidence_level=0.95)

    # Check that interval is reasonable
    assert lower < np.mean(samples) < upper
    assert upper - lower < 2.0  # CI width should be reasonable


def test_calculate_confidence_interval_single_sample():
    """Test confidence interval with only one sample."""
    samples = [7.5]

    lower, upper = calculate_confidence_interval(samples)

    # With one sample, CI should collapse to the value
    assert lower == 7.5
    assert upper == 7.5


def test_calculate_confidence_interval_empty():
    """Test confidence interval with empty samples."""
    samples = []

    lower, upper = calculate_confidence_interval(samples)

    # Should return (0, 0) for empty
    assert lower == 0.0
    assert upper == 0.0


def test_should_flag_for_review_high_std():
    """Test flagging for review with high standard deviation."""
    uncertainty = UncertaintyEstimate(
        mean_score=7.5,
        std_dev=1.5,  # High std dev
        confidence_interval=(6.0, 9.0),
        samples=[6.0, 7.0, 8.0, 9.0],
        num_samples=4,
    )

    assert should_flag_for_review(uncertainty, threshold_std=1.0) is True


def test_should_flag_for_review_wide_ci():
    """Test flagging for review with wide confidence interval."""
    uncertainty = UncertaintyEstimate(
        mean_score=7.5,
        std_dev=0.5,
        confidence_interval=(5.0, 10.0),  # Very wide CI
        samples=[5.0, 7.0, 9.0, 10.0],
        num_samples=4,
    )

    assert should_flag_for_review(uncertainty, threshold_std=1.0) is True


def test_should_flag_for_review_low_uncertainty():
    """Test that low uncertainty is not flagged."""
    uncertainty = UncertaintyEstimate(
        mean_score=7.5,
        std_dev=0.2,  # Low std dev
        confidence_interval=(7.3, 7.7),  # Narrow CI
        samples=[7.3, 7.5, 7.6, 7.4],
        num_samples=4,
    )

    assert should_flag_for_review(uncertainty, threshold_std=1.0) is False


def test_get_confidence_category_high():
    """Test high confidence categorization."""
    uncertainty = UncertaintyEstimate(
        mean_score=7.5,
        std_dev=0.2,
        confidence_interval=(7.3, 7.7),
        samples=[7.3, 7.5, 7.6, 7.4],
        num_samples=4,
    )

    assert get_confidence_category(uncertainty) == "high"


def test_get_confidence_category_medium():
    """Test medium confidence categorization."""
    uncertainty = UncertaintyEstimate(
        mean_score=7.5,
        std_dev=0.5,
        confidence_interval=(7.0, 8.0),
        samples=[7.0, 7.5, 8.0],
        num_samples=3,
    )

    assert get_confidence_category(uncertainty) == "medium"


def test_get_confidence_category_low():
    """Test low confidence categorization."""
    uncertainty = UncertaintyEstimate(
        mean_score=7.5,
        std_dev=1.5,
        confidence_interval=(6.0, 9.0),
        samples=[6.0, 7.0, 8.0, 9.0],
        num_samples=4,
    )

    assert get_confidence_category(uncertainty) == "low"


@pytest.mark.asyncio
async def test_estimate_uncertainty_monte_carlo():
    """Test Monte Carlo uncertainty estimation with mock evaluation function."""
    call_count = 0

    async def mock_evaluate():
        """Mock evaluation that returns scores with small variation."""
        nonlocal call_count
        call_count += 1
        # Return slightly different scores each time
        return type('obj', (object,), {'score': 7.5 + (call_count % 3) * 0.1})()

    # Import here to avoid circular dependency in test
    from stackbench.evaluation.uncertainty import estimate_uncertainty_monte_carlo

    uncertainty = await estimate_uncertainty_monte_carlo(
        evaluation_func=mock_evaluate,
        score_extractor=lambda result: result.score,
        num_samples=5,
    )

    # Verify structure
    assert uncertainty.num_samples == 5
    assert len(uncertainty.samples) == 5
    assert 7.4 <= uncertainty.mean_score <= 7.7  # Should be around 7.5
    assert uncertainty.std_dev >= 0  # Should have some variation
    assert uncertainty.confidence_interval[0] <= uncertainty.mean_score <= uncertainty.confidence_interval[1]
