"""
Evaluation improvements for StackBench.

This module implements research-based evaluation methodologies including:
- Uncertainty quantification (confidence intervals)
- False positive rate tracking
- Ground truth validation
- Multi-model ensemble evaluation
- Inter-annotator agreement metrics
- Graded code validation with partial credit
- Temporal quality tracking
"""

# Import key classes and functions for easy access
from .uncertainty import (
    UncertaintyEstimate,
    calculate_confidence_interval,
    estimate_uncertainty_monte_carlo,
    should_flag_for_review,
    get_confidence_category,
)

from .metrics import (
    GroundTruthLabel,
    PrecisionRecallMetrics,
    calculate_precision_recall,
    calculate_inter_annotator_agreement_simple,
    analyze_false_positive_patterns,
)

from .ground_truth import (
    HumanAnnotation,
    GroundTruthDataset,
    GroundTruthManager,
)

from .validation import (
    ValidationReport,
    DatasetValidationReport,
    compare_llm_to_human,
    validate_against_dataset,
    identify_problematic_documents,
)

from .ensemble import (
    ModelEvaluation,
    EnsembleResult,
    calculate_inter_model_agreement,
    ensemble_evaluate,
    evaluate_with_ensemble,
    compare_model_biases,
)

from .graded_validation import (
    GradedCodeResult,
    check_syntax,
    graded_validate_code,
)

from .temporal import (
    CommitEvaluation,
    QualityTrend,
    QualityRegression,
    TemporalReport,
    calculate_trend,
    find_regressions,
    calculate_velocity,
    identify_improvement_opportunities,
)

from .cli_utils import (
    format_score_with_uncertainty,
    display_clarity_score_with_uncertainty,
    create_validation_metrics_table,
    display_ensemble_results,
    display_temporal_trend,
)

__all__ = [
    # Uncertainty
    "UncertaintyEstimate",
    "calculate_confidence_interval",
    "estimate_uncertainty_monte_carlo",
    "should_flag_for_review",
    "get_confidence_category",
    # Metrics
    "GroundTruthLabel",
    "PrecisionRecallMetrics",
    "calculate_precision_recall",
    "calculate_inter_annotator_agreement_simple",
    "analyze_false_positive_patterns",
    # Ground truth
    "HumanAnnotation",
    "GroundTruthDataset",
    "GroundTruthManager",
    # Validation
    "ValidationReport",
    "DatasetValidationReport",
    "compare_llm_to_human",
    "validate_against_dataset",
    "identify_problematic_documents",
    # Ensemble
    "ModelEvaluation",
    "EnsembleResult",
    "calculate_inter_model_agreement",
    "ensemble_evaluate",
    "evaluate_with_ensemble",
    "compare_model_biases",
    # Graded validation
    "GradedCodeResult",
    "check_syntax",
    "graded_validate_code",
    # Temporal
    "CommitEvaluation",
    "QualityTrend",
    "QualityRegression",
    "TemporalReport",
    "calculate_trend",
    "find_regressions",
    "calculate_velocity",
    "identify_improvement_opportunities",
    # CLI utilities
    "format_score_with_uncertainty",
    "display_clarity_score_with_uncertainty",
    "create_validation_metrics_table",
    "display_ensemble_results",
    "display_temporal_trend",
]
