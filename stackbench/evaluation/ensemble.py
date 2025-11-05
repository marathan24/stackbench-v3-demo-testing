"""
Multi-model ensemble evaluation to reduce single-model bias.

Implements ensemble judgment using multiple LLMs (Claude, GPT-4, Gemini)
to reduce model-specific biases. Addresses Mistake #2 from EVALUATION_ANALYSIS.md:
Single model evaluation vulnerable to position, verbosity, and style biases.
"""

from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class ModelEvaluation:
    """Evaluation result from a single model."""

    model_name: str
    """Model identifier (e.g., 'claude-sonnet-4.5', 'gpt-4-turbo')"""

    overall_score: float
    """Overall clarity score (0-10)"""

    dimension_scores: Dict[str, float]
    """Dimensional scores"""

    issues_found: List[Dict[str, any]]
    """Issues identified by this model"""

    confidence: Optional[float] = None
    """Model's self-reported confidence if available"""


@dataclass
class EnsembleResult:
    """Aggregated result from multiple models."""

    mean_score: float
    """Mean score across all models"""

    median_score: float
    """Median score (more robust to outliers)"""

    std_dev: float
    """Standard deviation of scores"""

    confidence_interval: tuple[float, float]
    """95% confidence interval"""

    inter_model_agreement: float
    """Agreement metric (0-1, higher = more agreement)"""

    individual_results: List[ModelEvaluation]
    """Results from each model"""

    flagged_for_review: bool
    """True if high disagreement detected"""

    consensus_issues: List[Dict[str, any]]
    """Issues identified by majority of models"""


def calculate_inter_model_agreement(scores: List[float]) -> float:
    """
    Calculate inter-model agreement using coefficient of variation.

    Lower coefficient of variation = higher agreement.

    Args:
        scores: List of scores from different models

    Returns:
        Agreement score (0-1, where 1 = perfect agreement)

    Example:
        ```python
        scores = [7.5, 7.8, 7.3]  # High agreement
        agreement = calculate_inter_model_agreement(scores)
        # Returns: ~0.95 (high agreement)

        scores = [5.0, 8.5, 3.0]  # Low agreement
        agreement = calculate_inter_model_agreement(scores)
        # Returns: ~0.5 (low agreement)
        ```
    """
    if len(scores) < 2:
        return 1.0

    mean_score = np.mean(scores)
    std_dev = np.std(scores, ddof=1)

    if mean_score == 0:
        return 0.0

    # Coefficient of variation (CV)
    cv = std_dev / mean_score

    # Convert to agreement score (0-1 scale)
    # CV of 0 = perfect agreement (score 1.0)
    # CV of 0.5 = moderate disagreement (score ~0.33)
    # CV > 1.0 = high disagreement (score ~0)
    agreement = 1.0 / (1.0 + 2 * cv)

    return agreement


def ensemble_evaluate(
    model_results: List[ModelEvaluation],
    low_agreement_threshold: float = 0.4
) -> EnsembleResult:
    """
    Aggregate evaluations from multiple models into ensemble result.

    Args:
        model_results: List of ModelEvaluation from different models
        low_agreement_threshold: Agreement threshold below which to flag for review

    Returns:
        EnsembleResult with aggregated metrics

    Example:
        ```python
        results = [
            ModelEvaluation("claude-sonnet-4.5", 7.5, {...}, [...]),
            ModelEvaluation("gpt-4-turbo", 7.8, {...}, [...]),
            ModelEvaluation("gemini-2.0-flash", 7.2, {...}, [...]),
        ]

        ensemble = ensemble_evaluate(results)
        print(f"Mean: {ensemble.mean_score:.1f}")
        print(f"Agreement: {ensemble.inter_model_agreement:.2f}")
        if ensemble.flagged_for_review:
            print("⚠️ High disagreement - recommend human review")
        ```
    """
    if not model_results:
        raise ValueError("No model results provided")

    # Extract scores
    scores = [r.overall_score for r in model_results]

    # Calculate statistics
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    std_dev = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    # Confidence interval (95%)
    if len(scores) > 1:
        ci_lower = float(np.percentile(scores, 2.5))
        ci_upper = float(np.percentile(scores, 97.5))
    else:
        ci_lower = mean_score
        ci_upper = mean_score

    # Inter-model agreement
    agreement = calculate_inter_model_agreement(scores)
    flagged = agreement < low_agreement_threshold

    # Find consensus issues (appear in >50% of models)
    all_issues = {}
    for result in model_results:
        for issue in result.issues_found:
            issue_key = (issue.get("type"), issue.get("line"), issue.get("section"))
            all_issues[issue_key] = all_issues.get(issue_key, 0) + 1

    consensus_threshold = len(model_results) / 2
    consensus_issues = [
        {"type": k[0], "line": k[1], "section": k[2], "models_agreeing": count}
        for k, count in all_issues.items()
        if count > consensus_threshold
    ]

    return EnsembleResult(
        mean_score=mean_score,
        median_score=median_score,
        std_dev=std_dev,
        confidence_interval=(ci_lower, ci_upper),
        inter_model_agreement=agreement,
        individual_results=model_results,
        flagged_for_review=flagged,
        consensus_issues=consensus_issues,
    )


async def evaluate_with_ensemble(
    document: Any,
    model_evaluators: Dict[str, Callable],
    return_all_results: bool = True
) -> EnsembleResult:
    """
    Evaluate document using multiple models in parallel.

    Args:
        document: Document to evaluate
        model_evaluators: Dict mapping model name to evaluation function
                         {"claude-sonnet-4.5": eval_func1, "gpt-4-turbo": eval_func2}
        return_all_results: Whether to include individual model results

    Returns:
        EnsembleResult with ensemble evaluation

    Example:
        ```python
        model_evaluators = {
            "claude-sonnet-4.5": lambda doc: claude_evaluate(doc),
            "gpt-4-turbo": lambda doc: gpt4_evaluate(doc),
            "gemini-2.0-flash": lambda doc: gemini_evaluate(doc),
        }

        result = await evaluate_with_ensemble(document, model_evaluators)

        print(f"Ensemble score: {result.mean_score:.1f} ± {result.std_dev:.1f}")
        print(f"95% CI: [{result.confidence_interval[0]:.1f}, {result.confidence_interval[1]:.1f}]")
        print(f"Agreement: {result.inter_model_agreement:.2f}")
        ```
    """
    import asyncio

    # Evaluate with all models in parallel
    tasks = [
        evaluator(document)
        for evaluator in model_evaluators.values()
    ]

    results = await asyncio.gather(*tasks)

    # Convert to ModelEvaluation objects
    model_results = []
    for model_name, result in zip(model_evaluators.keys(), results):
        model_eval = ModelEvaluation(
            model_name=model_name,
            overall_score=result.overall_score,
            dimension_scores=result.dimension_scores,
            issues_found=result.issues_found,
        )
        model_results.append(model_eval)

    # Ensemble
    return ensemble_evaluate(model_results)


def compare_model_biases(
    ensemble_results: List[EnsembleResult]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze systematic biases across models.

    Identifies which models tend to score higher/lower on average.

    Args:
        ensemble_results: List of EnsembleResults from different documents

    Returns:
        Dict mapping model_name to bias statistics

    Example:
        ```python
        bias_analysis = compare_model_biases(ensemble_results)

        for model, stats in bias_analysis.items():
            print(f"{model}:")
            print(f"  Mean score: {stats['mean_score']:.2f}")
            print(f"  Std dev: {stats['std_dev']:.2f}")
            print(f"  Bias: {stats['bias']:.2f} (+ = scores higher than ensemble)")
        ```
    """
    model_scores: Dict[str, List[float]] = {}

    # Collect all scores per model
    for ensemble in ensemble_results:
        for model_result in ensemble.individual_results:
            model_name = model_result.model_name
            if model_name not in model_scores:
                model_scores[model_name] = []
            model_scores[model_name].append(model_result.overall_score)

    # Calculate bias for each model
    # Bias = model's mean score - ensemble mean across all docs
    all_ensemble_means = [e.mean_score for e in ensemble_results]
    overall_mean = np.mean(all_ensemble_means)

    bias_stats = {}
    for model_name, scores in model_scores.items():
        model_mean = np.mean(scores)
        model_std = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        bias = model_mean - overall_mean

        bias_stats[model_name] = {
            "mean_score": model_mean,
            "std_dev": model_std,
            "bias": bias,
            "num_evaluations": len(scores),
        }

    return bias_stats
