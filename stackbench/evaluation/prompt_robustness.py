"""
Prompt robustness testing.

Tests if evaluation results are stable across prompt variations.
Addresses prompt brittleness issues from EVALUATION_ANALYSIS.md.
"""

from typing import List, Callable, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class RobustnessReport:
    """Report on prompt robustness."""

    original_prompt: str
    variants: List[str]
    scores: List[float]

    score_variance: float
    """Variance of scores across prompts"""

    score_std_dev: float
    """Standard deviation of scores"""

    is_robust: bool
    """Whether evaluation is considered robust (low variance)"""

    issue_overlap: float
    """Jaccard similarity of issues found (0-1)"""


def rephrase_prompt(prompt: str) -> str:
    """
    Create a rephrased version of the prompt.

    Changes wording while preserving meaning.

    Args:
        prompt: Original prompt text

    Returns:
        Rephrased prompt

    Example:
        Original: "Evaluate the documentation for clarity."
        Rephrased: "Assess the docs for how clear they are."
    """
    # Simple rephrasing examples (in production, use LLM for better rephrasing)
    replacements = {
        "evaluate": "assess",
        "documentation": "docs",
        "clarity": "how clear it is",
        "identify": "find",
        "analyze": "examine",
        "complete": "thorough",
    }

    rephrased = prompt
    for original, replacement in replacements.items():
        if original in rephrased.lower():
            # Simple case-insensitive replacement
            rephrased = rephrased.replace(original, replacement)

    return rephrased


def reorder_sections(prompt: str) -> str:
    """
    Reorder sections of the prompt.

    Tests if section order affects evaluation.

    Args:
        prompt: Original prompt with multiple sections

    Returns:
        Prompt with sections reordered
    """
    # Split by double newlines (sections)
    sections = prompt.split("\n\n")

    if len(sections) <= 2:
        return prompt  # Too short to reorder

    # Reverse order (simple test)
    # In production, could try multiple orderings
    sections_reordered = sections[::-1]

    return "\n\n".join(sections_reordered)


def add_synonyms(prompt: str) -> str:
    """
    Add synonyms to make prompt more verbose.

    Tests if extra wording affects evaluation.

    Args:
        prompt: Original prompt

    Returns:
        Prompt with synonyms added
    """
    # Add clarifying synonyms
    replacements = {
        "clarity": "clarity (i.e., how easy to understand)",
        "completeness": "completeness (i.e., whether all info is provided)",
        "documentation": "documentation (i.e., written guides and tutorials)",
    }

    verbose = prompt
    for term, expansion in replacements.items():
        verbose = verbose.replace(term, expansion)

    return verbose


async def test_prompt_robustness(
    document: Any,
    evaluation_func: Callable[[Any, str], Any],
    original_prompt: str,
    variance_threshold: float = 0.5,
    overlap_threshold: float = 0.8
) -> RobustnessReport:
    """
    Test if evaluation is stable across prompt variations.

    Args:
        document: Document to evaluate
        evaluation_func: Function that takes (document, prompt) and returns evaluation
        original_prompt: Original prompt text
        variance_threshold: Max acceptable variance (default: 0.5)
        overlap_threshold: Min acceptable issue overlap (default: 0.8)

    Returns:
        RobustnessReport with robustness analysis

    Example:
        ```python
        async def evaluate(doc, prompt):
            return await agent.evaluate(doc, system_prompt=prompt)

        report = await test_prompt_robustness(
            document=doc,
            evaluation_func=evaluate,
            original_prompt=ORIGINAL_PROMPT
        )

        if not report.is_robust:
            print(f"⚠️ Prompt is brittle! Variance: {report.score_variance:.2f}")
            print("   Consider using ensemble or more stable prompting")
        ```
    """
    # Create prompt variations
    variants = [
        original_prompt,
        rephrase_prompt(original_prompt),
        reorder_sections(original_prompt),
        add_synonyms(original_prompt),
    ]

    # Evaluate with each variant
    results = []
    for variant in variants:
        result = await evaluation_func(document, variant)
        results.append(result)

    # Extract scores
    scores = [r.overall_score for r in results]

    # Calculate variance
    score_variance = float(np.var(scores))
    score_std_dev = float(np.std(scores))

    # Calculate issue overlap (Jaccard similarity)
    # Compare issues found across variants
    all_issues = []
    for result in results:
        issues = set((issue.type, issue.line) for issue in result.issues)
        all_issues.append(issues)

    # Calculate pairwise Jaccard similarity
    similarities = []
    for i in range(len(all_issues)):
        for j in range(i + 1, len(all_issues)):
            intersection = len(all_issues[i] & all_issues[j])
            union = len(all_issues[i] | all_issues[j])

            if union == 0:
                similarity = 1.0  # Both empty
            else:
                similarity = intersection / union

            similarities.append(similarity)

    issue_overlap = np.mean(similarities) if similarities else 1.0

    # Determine robustness
    is_robust = (score_variance < variance_threshold) and (issue_overlap > overlap_threshold)

    return RobustnessReport(
        original_prompt=original_prompt,
        variants=variants,
        scores=scores,
        score_variance=score_variance,
        score_std_dev=score_std_dev,
        is_robust=is_robust,
        issue_overlap=issue_overlap,
    )


def calculate_jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Args:
        set_a: First set
        set_b: Second set

    Returns:
        Similarity (0-1)

    Example:
        ```python
        a = {"issue1", "issue2", "issue3"}
        b = {"issue1", "issue2", "issue4"}

        similarity = calculate_jaccard_similarity(a, b)
        # 0.5 (2 out of 4 unique elements in common)
        ```
    """
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0  # Both empty = perfect similarity

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def recommend_improvements(report: RobustnessReport) -> List[str]:
    """
    Recommend improvements based on robustness report.

    Args:
        report: RobustnessReport

    Returns:
        List of recommendations

    Example:
        ```python
        recommendations = recommend_improvements(report)
        for rec in recommendations:
            print(f"• {rec}")
        ```
    """
    recommendations = []

    if report.score_variance > 1.0:
        recommendations.append(
            "High score variance detected. Consider using ensemble evaluation across multiple prompts."
        )

    if report.issue_overlap < 0.7:
        recommendations.append(
            "Low issue overlap detected. Issues found are highly dependent on prompt wording. "
            "Consider adding few-shot examples for consistency."
        )

    if report.score_variance > 0.5 or report.issue_overlap < 0.8:
        recommendations.append(
            "Evaluation is somewhat brittle. Consider: "
            "(1) Fine-tuning on labeled examples, "
            "(2) Using more specific rubrics, "
            "(3) Ensemble across prompt variations."
        )

    if not recommendations:
        recommendations.append("Evaluation is robust! No improvements needed.")

    return recommendations
