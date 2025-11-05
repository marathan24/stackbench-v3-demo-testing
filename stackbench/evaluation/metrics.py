"""
Evaluation metrics for documentation quality assessment.

Implements precision, recall, F1, false positive rate, and other metrics
to validate evaluation quality. Addresses issues from EVALUATION_ANALYSIS.md:
- Mistake #7: No false positive tracking
- Mistake #1: No calibration against human judgment
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field


class GroundTruthLabel(BaseModel):
    """Human-annotated ground truth label for an issue or document."""

    issue_id: str = Field(description="Unique identifier for the issue")
    is_real_issue: bool = Field(description="True if issue is real, False if false positive")
    severity: Optional[str] = Field(None, description="Human-judged severity: critical/warning/info")
    annotator_id: str = Field(description="ID of human annotator")
    confidence: Optional[float] = Field(None, description="Annotator confidence (0-1)")
    notes: Optional[str] = Field(None, description="Additional notes")


@dataclass
class PrecisionRecallMetrics:
    """Precision, recall, and F1 metrics."""

    true_positives: int
    """Issues correctly identified"""

    false_positives: int
    """Non-issues incorrectly flagged"""

    false_negatives: int
    """Real issues missed"""

    true_negatives: int
    """Non-issues correctly ignored"""

    precision: float
    """TP / (TP + FP) - How many flagged issues are real?"""

    recall: float
    """TP / (TP + FN) - How many real issues were found?"""

    f1_score: float
    """Harmonic mean of precision and recall"""

    false_positive_rate: float
    """FP / (FP + TN) - Rate of false alarms"""

    accuracy: float
    """(TP + TN) / (TP + TN + FP + FN) - Overall correctness"""


def calculate_precision_recall(
    predicted_issues: List[str],
    ground_truth_labels: Dict[str, GroundTruthLabel],
    all_possible_issues: Optional[List[str]] = None,
) -> PrecisionRecallMetrics:
    """
    Calculate precision, recall, F1, and FPR from predictions and ground truth.

    Args:
        predicted_issues: List of issue IDs that were flagged by the system
        ground_truth_labels: Dict mapping issue_id to GroundTruthLabel
        all_possible_issues: Optional list of all issues that could have been detected
                             (needed for true negatives calculation)

    Returns:
        PrecisionRecallMetrics with all computed metrics

    Example:
        ```python
        # System predicted these issues
        predicted = ["issue_1", "issue_2", "issue_5"]

        # Expert annotations
        ground_truth = {
            "issue_1": GroundTruthLabel(issue_id="issue_1", is_real_issue=True, ...),
            "issue_2": GroundTruthLabel(issue_id="issue_2", is_real_issue=False, ...),  # FP
            "issue_3": GroundTruthLabel(issue_id="issue_3", is_real_issue=True, ...),   # FN
            "issue_5": GroundTruthLabel(issue_id="issue_5", is_real_issue=True, ...),
        }

        metrics = calculate_precision_recall(predicted, ground_truth)
        print(f"Precision: {metrics.precision:.2f}")  # 0.67 (2/3 correct)
        print(f"Recall: {metrics.recall:.2f}")        # 0.67 (missed 1 of 3)
        print(f"FPR: {metrics.false_positive_rate:.2f}")
        ```
    """
    # Calculate TP, FP, FN
    true_positives = 0
    false_positives = 0

    for issue_id in predicted_issues:
        if issue_id in ground_truth_labels:
            if ground_truth_labels[issue_id].is_real_issue:
                true_positives += 1
            else:
                false_positives += 1
        # If not in ground truth, we can't determine (skip)

    # False negatives: Real issues in ground truth that weren't predicted
    false_negatives = 0
    for issue_id, label in ground_truth_labels.items():
        if label.is_real_issue and issue_id not in predicted_issues:
            false_negatives += 1

    # True negatives: Non-issues that weren't predicted
    true_negatives = 0
    if all_possible_issues:
        for issue_id in all_possible_issues:
            if issue_id in ground_truth_labels:
                if not ground_truth_labels[issue_id].is_real_issue and issue_id not in predicted_issues:
                    true_negatives += 1

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) > 0 else 0.0

    return PrecisionRecallMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        false_positive_rate=false_positive_rate,
        accuracy=accuracy,
    )


def calculate_inter_annotator_agreement_simple(
    annotator1_labels: Dict[str, bool],
    annotator2_labels: Dict[str, bool],
) -> float:
    """
    Calculate simple percentage agreement between two annotators.

    This is a basic measure. For production, use Krippendorff's alpha or Cohen's kappa.

    Args:
        annotator1_labels: Dict mapping issue_id to is_real_issue (True/False)
        annotator2_labels: Dict mapping issue_id to is_real_issue (True/False)

    Returns:
        Agreement rate (0.0 to 1.0)

    Example:
        ```python
        annotator1 = {"issue_1": True, "issue_2": False, "issue_3": True}
        annotator2 = {"issue_1": True, "issue_2": True, "issue_3": True}

        agreement = calculate_inter_annotator_agreement_simple(annotator1, annotator2)
        # 0.67 (2 out of 3 agree)
        ```
    """
    # Find common issues
    common_issues = set(annotator1_labels.keys()) & set(annotator2_labels.keys())

    if not common_issues:
        return 0.0

    # Count agreements
    agreements = sum(
        1
        for issue_id in common_issues
        if annotator1_labels[issue_id] == annotator2_labels[issue_id]
    )

    return agreements / len(common_issues)


@dataclass
class FalsePositiveReport:
    """Report on false positive patterns."""

    false_positive_rate: float
    """Overall FPR"""

    sample_size: int
    """Number of issues sampled"""

    false_positive_count: int
    """Number of false positives"""

    common_patterns: List[Dict[str, any]]
    """Common false positive patterns"""

    recommendation: str
    """Action recommendation"""


def analyze_false_positive_patterns(
    false_positive_issues: List[Dict[str, any]]
) -> List[Dict[str, any]]:
    """
    Analyze common patterns in false positives.

    Args:
        false_positive_issues: List of false positive issues with metadata

    Returns:
        List of pattern dictionaries with counts

    Example:
        ```python
        fps = [
            {"type": "unclear_instruction", "section": "Installation"},
            {"type": "unclear_instruction", "section": "Configuration"},
            {"type": "missing_prerequisite", "section": "Quickstart"},
        ]

        patterns = analyze_false_positive_patterns(fps)
        # [{"type": "unclear_instruction", "count": 2}, ...]
        ```
    """
    # Group by issue type
    type_counts: Dict[str, int] = {}
    for issue in false_positive_issues:
        issue_type = issue.get("type", "unknown")
        type_counts[issue_type] = type_counts.get(issue_type, 0) + 1

    # Convert to list and sort by count
    patterns = [{"type": t, "count": c} for t, c in type_counts.items()]
    patterns.sort(key=lambda x: x["count"], reverse=True)

    return patterns
