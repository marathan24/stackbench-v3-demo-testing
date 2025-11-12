"""
Tests for metrics module (precision, recall, FPR).
"""

import pytest
from stackbench.evaluation.metrics import (
    GroundTruthLabel,
    calculate_precision_recall,
    calculate_inter_annotator_agreement_simple,
    analyze_false_positive_patterns,
)


def test_calculate_precision_recall_perfect():
    """Test precision/recall with perfect predictions."""
    predicted = ["issue1", "issue2", "issue3"]
    ground_truth = {
        "issue1": GroundTruthLabel(issue_id="issue1", is_real_issue=True, annotator_id="expert1"),
        "issue2": GroundTruthLabel(issue_id="issue2", is_real_issue=True, annotator_id="expert1"),
        "issue3": GroundTruthLabel(issue_id="issue3", is_real_issue=True, annotator_id="expert1"),
    }

    metrics = calculate_precision_recall(predicted, ground_truth)

    assert metrics.precision == 1.0  # All predicted are true
    assert metrics.recall == 1.0  # All true issues found
    assert metrics.f1_score == 1.0
    assert metrics.true_positives == 3
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_calculate_precision_recall_with_fp():
    """Test precision/recall with false positives."""
    predicted = ["issue1", "issue2", "issue3"]
    ground_truth = {
        "issue1": GroundTruthLabel(issue_id="issue1", is_real_issue=True, annotator_id="expert1"),
        "issue2": GroundTruthLabel(issue_id="issue2", is_real_issue=False, annotator_id="expert1"),  # FP
        "issue3": GroundTruthLabel(issue_id="issue3", is_real_issue=True, annotator_id="expert1"),
    }

    metrics = calculate_precision_recall(predicted, ground_truth)

    assert metrics.precision == 2 / 3  # 2 out of 3 correct
    assert metrics.recall == 1.0  # All true issues found
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0


def test_calculate_precision_recall_with_fn():
    """Test precision/recall with false negatives (missed issues)."""
    predicted = ["issue1"]
    ground_truth = {
        "issue1": GroundTruthLabel(issue_id="issue1", is_real_issue=True, annotator_id="expert1"),
        "issue2": GroundTruthLabel(issue_id="issue2", is_real_issue=True, annotator_id="expert1"),  # FN (missed)
        "issue3": GroundTruthLabel(issue_id="issue3", is_real_issue=True, annotator_id="expert1"),  # FN (missed)
    }

    metrics = calculate_precision_recall(predicted, ground_truth)

    assert metrics.precision == 1.0  # All predicted are correct
    assert metrics.recall == 1 / 3  # Only found 1 out of 3
    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 2


def test_calculate_precision_recall_f1():
    """Test F1 score calculation."""
    predicted = ["issue1", "issue2", "issue5"]
    ground_truth = {
        "issue1": GroundTruthLabel(issue_id="issue1", is_real_issue=True, annotator_id="expert1"),
        "issue2": GroundTruthLabel(issue_id="issue2", is_real_issue=False, annotator_id="expert1"),  # FP
        "issue3": GroundTruthLabel(issue_id="issue3", is_real_issue=True, annotator_id="expert1"),   # FN
        "issue5": GroundTruthLabel(issue_id="issue5", is_real_issue=True, annotator_id="expert1"),
    }

    metrics = calculate_precision_recall(predicted, ground_truth)

    # Precision: 2/3 = 0.67
    # Recall: 2/3 = 0.67
    # F1: 2 * (0.67 * 0.67) / (0.67 + 0.67) = 0.67
    assert 0.66 <= metrics.precision <= 0.68
    assert 0.66 <= metrics.recall <= 0.68
    assert 0.66 <= metrics.f1_score <= 0.68


def test_calculate_precision_recall_empty_predicted():
    """Test with no predictions (all false negatives)."""
    predicted = []
    ground_truth = {
        "issue1": GroundTruthLabel(issue_id="issue1", is_real_issue=True, annotator_id="expert1"),
        "issue2": GroundTruthLabel(issue_id="issue2", is_real_issue=True, annotator_id="expert1"),
    }

    metrics = calculate_precision_recall(predicted, ground_truth)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0
    assert metrics.true_positives == 0
    assert metrics.false_negatives == 2


def test_inter_annotator_agreement_perfect():
    """Test inter-annotator agreement with perfect agreement."""
    annotator1 = {"issue1": True, "issue2": False, "issue3": True}
    annotator2 = {"issue1": True, "issue2": False, "issue3": True}

    agreement = calculate_inter_annotator_agreement_simple(annotator1, annotator2)

    assert agreement == 1.0  # Perfect agreement


def test_inter_annotator_agreement_partial():
    """Test inter-annotator agreement with partial agreement."""
    annotator1 = {"issue1": True, "issue2": False, "issue3": True}
    annotator2 = {"issue1": True, "issue2": True, "issue3": True}  # Disagree on issue2

    agreement = calculate_inter_annotator_agreement_simple(annotator1, annotator2)

    assert agreement == 2 / 3  # 2 out of 3 agree


def test_inter_annotator_agreement_no_overlap():
    """Test inter-annotator agreement with no common issues."""
    annotator1 = {"issue1": True, "issue2": False}
    annotator2 = {"issue3": True, "issue4": False}  # No overlap

    agreement = calculate_inter_annotator_agreement_simple(annotator1, annotator2)

    assert agreement == 0.0  # No common issues


def test_analyze_false_positive_patterns():
    """Test false positive pattern analysis."""
    false_positives = [
        {"type": "unclear_instruction", "section": "Installation"},
        {"type": "unclear_instruction", "section": "Configuration"},
        {"type": "missing_prerequisite", "section": "Quickstart"},
        {"type": "unclear_instruction", "section": "Advanced"},
    ]

    patterns = analyze_false_positive_patterns(false_positives)

    # Should group by type
    assert len(patterns) == 2
    assert patterns[0]["type"] == "unclear_instruction"
    assert patterns[0]["count"] == 3
    assert patterns[1]["type"] == "missing_prerequisite"
    assert patterns[1]["count"] == 1


def test_analyze_false_positive_patterns_empty():
    """Test pattern analysis with no false positives."""
    patterns = analyze_false_positive_patterns([])

    assert patterns == []
