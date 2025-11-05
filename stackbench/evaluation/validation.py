"""
Validation of LLM evaluations against ground truth.

Compares LLM-as-judge evaluations with human expert annotations to measure
evaluation quality. Implements validation methodology from EVALUATION_ANALYSIS.md.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from .ground_truth import HumanAnnotation, GroundTruthDataset
from .metrics import PrecisionRecallMetrics, calculate_precision_recall


@dataclass
class ValidationReport:
    """Report comparing LLM evaluation against ground truth."""

    document_id: str
    llm_score: float
    human_score: float
    absolute_error: float
    """Absolute difference between LLM and human score"""

    relative_error: float
    """Relative error as percentage"""

    within_threshold: bool
    """Whether error is within acceptable threshold (±1.0)"""

    dimension_errors: Dict[str, float]
    """Per-dimension absolute errors"""


@dataclass
class DatasetValidationReport:
    """Validation report across entire dataset."""

    dataset_id: str
    num_documents: int
    mean_absolute_error: float
    """Average absolute error across all docs"""

    root_mean_squared_error: float
    """RMSE of prediction errors"""

    correlation: float
    """Pearson correlation between LLM and human scores"""

    within_threshold_rate: float
    """Percentage of predictions within ±1.0 of human score"""

    per_document_reports: List[ValidationReport]

    # Precision/recall for issue detection
    issue_detection_metrics: Optional[PrecisionRecallMetrics] = None


def compare_llm_to_human(
    llm_overall_score: float,
    llm_dimension_scores: Dict[str, float],
    human_annotation: HumanAnnotation,
    threshold: float = 1.0
) -> ValidationReport:
    """
    Compare LLM evaluation to human annotation for a single document.

    Args:
        llm_overall_score: LLM's overall clarity score (0-10)
        llm_dimension_scores: LLM's dimensional scores
        human_annotation: Human expert annotation
        threshold: Acceptable error threshold (default: ±1.0)

    Returns:
        ValidationReport with comparison metrics

    Example:
        ```python
        report = compare_llm_to_human(
            llm_overall_score=7.5,
            llm_dimension_scores={
                "instruction_clarity": 8.0,
                "logical_flow": 7.0,
                ...
            },
            human_annotation=annotation,
            threshold=1.0
        )

        if not report.within_threshold:
            print(f"Large error: {report.absolute_error:.1f}")
        ```
    """
    human_score = human_annotation.overall_clarity
    absolute_error = abs(llm_overall_score - human_score)
    relative_error = (absolute_error / human_score * 100) if human_score > 0 else 0.0
    within_threshold = absolute_error <= threshold

    # Calculate per-dimension errors
    dimension_errors = {
        "instruction_clarity": abs(
            llm_dimension_scores.get("instruction_clarity", 0) - human_annotation.instruction_clarity
        ),
        "logical_flow": abs(
            llm_dimension_scores.get("logical_flow", 0) - human_annotation.logical_flow
        ),
        "completeness": abs(
            llm_dimension_scores.get("completeness", 0) - human_annotation.completeness
        ),
        "consistency": abs(
            llm_dimension_scores.get("consistency", 0) - human_annotation.consistency
        ),
        "prerequisite_coverage": abs(
            llm_dimension_scores.get("prerequisite_coverage", 0) - human_annotation.prerequisite_coverage
        ),
    }

    return ValidationReport(
        document_id=human_annotation.document_id,
        llm_score=llm_overall_score,
        human_score=human_score,
        absolute_error=absolute_error,
        relative_error=relative_error,
        within_threshold=within_threshold,
        dimension_errors=dimension_errors,
    )


def validate_against_dataset(
    llm_results: Dict[str, Dict[str, float]],
    ground_truth: GroundTruthDataset,
    threshold: float = 1.0
) -> DatasetValidationReport:
    """
    Validate LLM evaluations against entire ground truth dataset.

    Args:
        llm_results: Dict mapping document_id to scores
                     {"doc1": {"overall": 7.5, "instruction_clarity": 8.0, ...}}
        ground_truth: GroundTruthDataset with human annotations
        threshold: Acceptable error threshold

    Returns:
        DatasetValidationReport with aggregate metrics

    Example:
        ```python
        llm_results = {
            "pydantic.md": {
                "overall": 7.5,
                "instruction_clarity": 8.0,
                "logical_flow": 7.0,
                ...
            },
            ...
        }

        report = validate_against_dataset(llm_results, ground_truth_dataset)
        print(f"MAE: {report.mean_absolute_error:.2f}")
        print(f"Correlation: {report.correlation:.2f}")
        print(f"Within threshold: {report.within_threshold_rate:.1%}")
        ```
    """
    per_document_reports = []
    llm_scores = []
    human_scores = []

    # Compare each document
    for annotation in ground_truth.documents:
        doc_id = annotation.document_id

        if doc_id not in llm_results:
            continue  # Skip if LLM didn't evaluate this doc

        llm_data = llm_results[doc_id]
        llm_overall = llm_data["overall"]

        report = compare_llm_to_human(
            llm_overall_score=llm_overall,
            llm_dimension_scores=llm_data,
            human_annotation=annotation,
            threshold=threshold
        )

        per_document_reports.append(report)
        llm_scores.append(llm_overall)
        human_scores.append(annotation.overall_clarity)

    # Calculate aggregate metrics
    if not per_document_reports:
        # No data
        return DatasetValidationReport(
            dataset_id=ground_truth.dataset_id,
            num_documents=0,
            mean_absolute_error=0.0,
            root_mean_squared_error=0.0,
            correlation=0.0,
            within_threshold_rate=0.0,
            per_document_reports=[],
        )

    errors = [r.absolute_error for r in per_document_reports]
    mean_absolute_error = np.mean(errors)
    root_mean_squared_error = np.sqrt(np.mean([e**2 for e in errors]))

    # Pearson correlation
    if len(llm_scores) > 1:
        correlation = np.corrcoef(llm_scores, human_scores)[0, 1]
    else:
        correlation = 0.0

    within_threshold_rate = sum(1 for r in per_document_reports if r.within_threshold) / len(per_document_reports)

    return DatasetValidationReport(
        dataset_id=ground_truth.dataset_id,
        num_documents=len(per_document_reports),
        mean_absolute_error=mean_absolute_error,
        root_mean_squared_error=root_mean_squared_error,
        correlation=correlation,
        within_threshold_rate=within_threshold_rate,
        per_document_reports=per_document_reports,
    )


def identify_problematic_documents(
    validation_report: DatasetValidationReport,
    error_threshold: float = 2.0
) -> List[ValidationReport]:
    """
    Identify documents where LLM evaluation significantly differs from human.

    Args:
        validation_report: DatasetValidationReport
        error_threshold: Error threshold for flagging (default: 2.0)

    Returns:
        List of ValidationReports for problematic documents

    Example:
        ```python
        problematic = identify_problematic_documents(report, error_threshold=1.5)

        for doc in problematic:
            print(f"{doc.document_id}: Error = {doc.absolute_error:.1f}")
            print(f"  LLM: {doc.llm_score:.1f}, Human: {doc.human_score:.1f}")
        ```
    """
    return [
        r for r in validation_report.per_document_reports
        if r.absolute_error > error_threshold
    ]
