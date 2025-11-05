"""
Uncertainty quantification for documentation evaluation.

Implements Monte Carlo-based uncertainty estimation to provide confidence
intervals for quality scores. Addresses the issue of single-point estimates
without uncertainty (from EVALUATION_ANALYSIS.md).
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Any
from dataclasses import dataclass


@dataclass
class UncertaintyEstimate:
    """Result of uncertainty quantification."""

    mean_score: float
    """Mean score across all samples"""

    std_dev: float
    """Standard deviation of scores"""

    confidence_interval: Tuple[float, float]
    """95% confidence interval (lower, upper)"""

    samples: List[float]
    """Individual sample scores"""

    num_samples: int
    """Number of samples used"""


def calculate_confidence_interval(
    samples: List[float], confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval from samples.

    Args:
        samples: List of score samples
        confidence_level: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(samples) < 2:
        # Not enough samples for CI
        mean = np.mean(samples) if samples else 0.0
        return (mean, mean)

    # Use percentile method (non-parametric)
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(samples, lower_percentile)
    upper = np.percentile(samples, upper_percentile)

    return (float(lower), float(upper))


async def estimate_uncertainty_monte_carlo(
    evaluation_func: Callable[[], Any],
    score_extractor: Callable[[Any], float],
    num_samples: int = 10,
    temperature: Optional[float] = None,
) -> UncertaintyEstimate:
    """
    Estimate uncertainty using Monte Carlo sampling.

    Runs evaluation multiple times with stochastic variations to measure
    score variability. This provides confidence intervals for the score.

    Args:
        evaluation_func: Async function that performs evaluation
        score_extractor: Function to extract score from evaluation result
        num_samples: Number of Monte Carlo samples (default: 10)
        temperature: Optional temperature parameter for stochastic sampling

    Returns:
        UncertaintyEstimate with mean, std dev, and confidence interval

    Example:
        ```python
        async def evaluate_doc():
            return await clarity_agent.evaluate(doc)

        uncertainty = await estimate_uncertainty_monte_carlo(
            evaluation_func=evaluate_doc,
            score_extractor=lambda result: result.clarity_score.overall_score,
            num_samples=10
        )

        print(f"Score: {uncertainty.mean_score:.1f} "
              f"± {uncertainty.std_dev:.1f} "
              f"(95% CI: {uncertainty.confidence_interval})")
        ```
    """
    scores = []

    for i in range(num_samples):
        # Run evaluation with seed variation for reproducibility
        result = await evaluation_func()
        score = score_extractor(result)
        scores.append(score)

    # Calculate statistics
    mean_score = float(np.mean(scores))
    std_dev = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    confidence_interval = calculate_confidence_interval(scores)

    return UncertaintyEstimate(
        mean_score=mean_score,
        std_dev=std_dev,
        confidence_interval=confidence_interval,
        samples=scores,
        num_samples=len(scores),
    )


def should_flag_for_review(uncertainty: UncertaintyEstimate, threshold_std: float = 1.0) -> bool:
    """
    Determine if a result should be flagged for human review based on uncertainty.

    High uncertainty (large std dev or wide confidence interval) indicates
    the score is unreliable and should be manually reviewed.

    Args:
        uncertainty: UncertaintyEstimate object
        threshold_std: Standard deviation threshold (default: 1.0)

    Returns:
        True if should be flagged for review

    Example:
        ```python
        if should_flag_for_review(uncertainty, threshold_std=1.5):
            print("⚠️  High uncertainty - recommend human review")
        ```
    """
    # Flag if standard deviation is high
    if uncertainty.std_dev > threshold_std:
        return True

    # Flag if confidence interval is wide (> 2.0 points)
    ci_width = uncertainty.confidence_interval[1] - uncertainty.confidence_interval[0]
    if ci_width > 2.0:
        return True

    return False


def get_confidence_category(uncertainty: UncertaintyEstimate) -> str:
    """
    Categorize confidence level based on standard deviation.

    Args:
        uncertainty: UncertaintyEstimate object

    Returns:
        Confidence category: "high", "medium", or "low"
    """
    std_dev = uncertainty.std_dev

    if std_dev < 0.3:
        return "high"  # Very confident
    elif std_dev < 0.8:
        return "medium"  # Moderate confidence
    else:
        return "low"  # Low confidence, needs review
