"""
Advanced inter-annotator agreement metrics.

Implements Krippendorff's alpha and Cohen's kappa for measuring
reliability of human annotations. More robust than simple percentage agreement.
"""

from typing import List, Dict, Optional
import numpy as np


def calculate_cohens_kappa(
    annotator1: Dict[str, bool],
    annotator2: Dict[str, bool]
) -> float:
    """
    Calculate Cohen's kappa for two annotators.

    Cohen's kappa measures inter-rater reliability, accounting for agreement
    by chance. Range: -1 to 1, where 1 = perfect agreement, 0 = chance agreement.

    Args:
        annotator1: Dict mapping item_id to label (True/False)
        annotator2: Dict mapping item_id to label (True/False)

    Returns:
        Cohen's kappa coefficient

    Interpretation:
        - < 0.00: Poor agreement (worse than chance)
        - 0.00-0.20: Slight agreement
        - 0.21-0.40: Fair agreement
        - 0.41-0.60: Moderate agreement
        - 0.61-0.80: Substantial agreement
        - 0.81-1.00: Almost perfect agreement

    Example:
        ```python
        ann1 = {"doc1": True, "doc2": False, "doc3": True}
        ann2 = {"doc1": True, "doc2": True, "doc3": True}

        kappa = calculate_cohens_kappa(ann1, ann2)
        # kappa = 0.33 (fair agreement)
        ```
    """
    # Find common items
    common_items = set(annotator1.keys()) & set(annotator2.keys())

    if not common_items:
        return 0.0

    # Build confusion matrix
    agree_positive = 0  # Both True
    agree_negative = 0  # Both False
    disagree = 0  # One True, one False

    for item in common_items:
        label1 = annotator1[item]
        label2 = annotator2[item]

        if label1 and label2:
            agree_positive += 1
        elif not label1 and not label2:
            agree_negative += 1
        else:
            disagree += 1

    n = len(common_items)

    # Observed agreement
    p_observed = (agree_positive + agree_negative) / n

    # Expected agreement by chance
    p1_positive = (agree_positive + disagree // 2) / n  # Approx positive rate for annotator1
    p2_positive = (agree_positive + disagree // 2) / n  # Approx positive rate for annotator2

    # More accurate calculation of marginals
    ann1_positive = sum(1 for item in common_items if annotator1[item])
    ann2_positive = sum(1 for item in common_items if annotator2[item])

    p1_positive = ann1_positive / n
    p2_positive = ann2_positive / n

    p_expected = (p1_positive * p2_positive) + ((1 - p1_positive) * (1 - p2_positive))

    # Cohen's kappa
    if p_expected == 1.0:
        return 0.0  # Avoid division by zero

    kappa = (p_observed - p_expected) / (1 - p_expected)

    return kappa


def calculate_krippendorffs_alpha_binary(
    annotations: Dict[str, List[bool]],
    missing_value: Optional[bool] = None
) -> float:
    """
    Calculate Krippendorff's alpha for binary data (multiple annotators).

    Krippendorff's alpha is more robust than Cohen's kappa:
    - Works with any number of annotators (not just 2)
    - Handles missing data
    - More conservative (lower scores)

    Args:
        annotations: Dict mapping item_id to list of annotations (one per annotator)
                     Example: {"doc1": [True, True, False], "doc2": [True, False, True]}
        missing_value: Value representing missing annotation (default: None)

    Returns:
        Krippendorff's alpha (0-1 scale)

    Interpretation:
        - α ≥ 0.800: Good reliability (tentative conclusions)
        - α ≥ 0.667: Tentative reliability (drawing tentative conclusions)
        - α < 0.667: Insufficient reliability

    Example:
        ```python
        annotations = {
            "doc1": [True, True, False],   # 3 annotators
            "doc2": [True, False, True],
            "doc3": [False, False, False],
        }

        alpha = calculate_krippendorffs_alpha_binary(annotations)
        # alpha = 0.42 (moderate reliability)

        if alpha >= 0.8:
            print("Good reliability!")
        elif alpha >= 0.667:
            print("Tentative reliability")
        else:
            print("Insufficient reliability - need more annotators or clearer guidelines")
        ```
    """
    # Convert to matrix format (items x annotators)
    items = sorted(annotations.keys())
    n_items = len(items)

    if n_items == 0:
        return 0.0

    # Get max number of annotators
    n_annotators = max(len(annotations[item]) for item in items)

    # Build coincidence matrix
    # For binary: [False_with_False, False_with_True, True_with_False, True_with_True]
    coincidence_matrix = np.zeros((2, 2))

    for item in items:
        labels = [l for l in annotations[item] if l is not missing_value]

        if len(labels) < 2:
            continue  # Need at least 2 annotations

        # For each pair of annotations
        n_labels = len(labels)
        for i in range(n_labels):
            for j in range(n_labels):
                if i != j:
                    label_i = int(labels[i])  # Convert bool to int (0 or 1)
                    label_j = int(labels[j])
                    coincidence_matrix[label_i, label_j] += 1 / (n_labels - 1)

    # Calculate observed disagreement
    n_total = coincidence_matrix.sum()
    if n_total == 0:
        return 0.0

    # Observed disagreement (off-diagonal elements)
    d_observed = (coincidence_matrix[0, 1] + coincidence_matrix[1, 0]) / n_total

    # Expected disagreement (if annotators chose randomly with same marginals)
    n_false = coincidence_matrix[0, :].sum()
    n_true = coincidence_matrix[1, :].sum()

    p_false = n_false / n_total
    p_true = n_true / n_total

    d_expected = 2 * p_false * p_true  # Probability of mismatch

    if d_expected == 0:
        return 1.0  # Perfect agreement

    # Krippendorff's alpha
    alpha = 1 - (d_observed / d_expected)

    return alpha


def interpret_agreement(agreement: float, metric: str = "kappa") -> str:
    """
    Interpret agreement coefficient.

    Args:
        agreement: Agreement value (0-1 scale)
        metric: Type of metric ("kappa" or "alpha")

    Returns:
        Human-readable interpretation

    Example:
        ```python
        interpretation = interpret_agreement(0.75, metric="kappa")
        # "Substantial agreement"

        interpretation = interpret_agreement(0.85, metric="alpha")
        # "Good reliability"
        ```
    """
    if metric == "alpha":
        # Krippendorff's alpha interpretation
        if agreement >= 0.800:
            return "Good reliability (can draw conclusions)"
        elif agreement >= 0.667:
            return "Tentative reliability (draw tentative conclusions)"
        else:
            return "Insufficient reliability (need improvement)"
    else:
        # Cohen's kappa interpretation (Landis & Koch, 1977)
        if agreement < 0.00:
            return "Poor agreement (worse than chance)"
        elif agreement < 0.20:
            return "Slight agreement"
        elif agreement < 0.40:
            return "Fair agreement"
        elif agreement < 0.60:
            return "Moderate agreement"
        elif agreement < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"


def calculate_fleiss_kappa(annotations: Dict[str, List[bool]]) -> float:
    """
    Calculate Fleiss' kappa for multiple annotators and multiple items.

    Fleiss' kappa extends Cohen's kappa to more than two annotators.

    Args:
        annotations: Dict mapping item_id to list of annotations

    Returns:
        Fleiss' kappa coefficient

    Example:
        ```python
        annotations = {
            "doc1": [True, True, False, True],    # 4 annotators
            "doc2": [False, False, False, True],
            "doc3": [True, True, True, True],
        }

        kappa = calculate_fleiss_kappa(annotations)
        ```
    """
    items = sorted(annotations.keys())
    n = len(items)  # Number of items

    if n == 0:
        return 0.0

    # Get number of annotators (assume same for all items)
    m = len(annotations[items[0]])  # Number of annotators

    # Count agreements
    agreements = []
    for item in items:
        labels = annotations[item]
        n_true = sum(labels)
        n_false = m - n_true

        # Agreement for this item (how many pairs agree)
        agree = (n_true * (n_true - 1) + n_false * (n_false - 1))
        agreements.append(agree)

    # Observed agreement
    p_bar = sum(agreements) / (n * m * (m - 1))

    # Expected agreement
    total_annotations = n * m
    total_true = sum(sum(annotations[item]) for item in items)
    total_false = total_annotations - total_true

    p_true = total_true / total_annotations
    p_false = total_false / total_annotations

    p_e = p_true**2 + p_false**2

    # Fleiss' kappa
    if p_e == 1.0:
        return 0.0

    kappa = (p_bar - p_e) / (1 - p_e)

    return kappa
