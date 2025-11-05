"""
Temporal quality tracking for documentation.

Tracks documentation quality changes over time, identifies regressions,
and measures improvement velocity. Addresses Mistake #9 from EVALUATION_ANALYSIS.md:
No temporal tracking.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np


class CommitEvaluation(BaseModel):
    """Evaluation of documentation at a specific commit."""

    commit_sha: str = Field(description="Git commit SHA")
    commit_date: str = Field(description="ISO timestamp of commit")
    author: str = Field(description="Commit author")
    message: str = Field(description="Commit message")

    # Quality metrics
    avg_clarity_score: float = Field(description="Average clarity score across all docs")
    num_docs_evaluated: int = Field(description="Number of documents evaluated")
    total_issues: int = Field(description="Total issues found")
    critical_issues: int = Field(description="Critical issues found")

    # Per-document results
    document_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Map of document path to clarity score"
    )


class QualityTrend(BaseModel):
    """Trend analysis of quality over time."""

    start_date: str
    end_date: str
    num_commits: int

    trend_direction: str = Field(description="improving, declining, or stable")
    trend_slope: float = Field(description="Rate of change per commit")

    mean_score: float
    min_score: float
    max_score: float
    score_volatility: float = Field(description="Standard deviation of scores")


class QualityRegression(BaseModel):
    """A quality regression (significant score drop)."""

    commit_sha: str
    commit_date: str
    author: str
    message: str

    score_before: float
    score_after: float
    score_drop: float

    affected_documents: List[str] = Field(description="Documents with score drops")


@dataclass
class TemporalReport:
    """Complete temporal quality analysis."""

    repository: str
    branch: str
    evaluations: List[CommitEvaluation]
    trend: QualityTrend
    regressions: List[QualityRegression]
    velocity: float  # Average change per week


def calculate_trend(evaluations: List[CommitEvaluation]) -> QualityTrend:
    """
    Calculate quality trend from evaluations.

    Args:
        evaluations: List of commit evaluations (chronological order)

    Returns:
        QualityTrend with trend analysis

    Example:
        ```python
        trend = calculate_trend(evaluations)
        print(f"Trend: {trend.trend_direction}")
        print(f"Slope: {trend.trend_slope:.3f} points/commit")
        print(f"Volatility: {trend.score_volatility:.2f}")
        ```
    """
    if len(evaluations) < 2:
        return QualityTrend(
            start_date=evaluations[0].commit_date if evaluations else "",
            end_date=evaluations[-1].commit_date if evaluations else "",
            num_commits=len(evaluations),
            trend_direction="unknown",
            trend_slope=0.0,
            mean_score=evaluations[0].avg_clarity_score if evaluations else 0.0,
            min_score=evaluations[0].avg_clarity_score if evaluations else 0.0,
            max_score=evaluations[0].avg_clarity_score if evaluations else 0.0,
            score_volatility=0.0
        )

    # Extract scores
    scores = np.array([e.avg_clarity_score for e in evaluations])

    # Linear regression for trend
    x = np.arange(len(scores))
    slope, _ = np.polyfit(x, scores, 1)

    # Determine trend direction
    if slope > 0.05:
        trend_direction = "improving"
    elif slope < -0.05:
        trend_direction = "declining"
    else:
        trend_direction = "stable"

    return QualityTrend(
        start_date=evaluations[0].commit_date,
        end_date=evaluations[-1].commit_date,
        num_commits=len(evaluations),
        trend_direction=trend_direction,
        trend_slope=float(slope),
        mean_score=float(np.mean(scores)),
        min_score=float(np.min(scores)),
        max_score=float(np.max(scores)),
        score_volatility=float(np.std(scores))
    )


def find_regressions(
    evaluations: List[CommitEvaluation],
    threshold: float = 1.0
) -> List[QualityRegression]:
    """
    Identify commits that significantly decreased quality.

    Args:
        evaluations: List of commit evaluations (chronological order)
        threshold: Score drop threshold to flag as regression (default: 1.0)

    Returns:
        List of QualityRegression objects

    Example:
        ```python
        regressions = find_regressions(evaluations, threshold=1.5)

        for reg in regressions:
            print(f"Regression at {reg.commit_sha[:7]}:")
            print(f"  Score drop: {reg.score_drop:.1f}")
            print(f"  Author: {reg.author}")
            print(f"  Affected: {len(reg.affected_documents)} documents")
        ```
    """
    regressions = []

    for i in range(1, len(evaluations)):
        prev = evaluations[i - 1]
        curr = evaluations[i]

        score_drop = prev.avg_clarity_score - curr.avg_clarity_score

        if score_drop >= threshold:
            # Find affected documents
            affected_docs = []
            for doc_path in curr.document_scores:
                if doc_path in prev.document_scores:
                    prev_score = prev.document_scores[doc_path]
                    curr_score = curr.document_scores[doc_path]
                    if prev_score - curr_score >= 0.5:  # Document-level threshold
                        affected_docs.append(doc_path)

            regression = QualityRegression(
                commit_sha=curr.commit_sha,
                commit_date=curr.commit_date,
                author=curr.author,
                message=curr.message,
                score_before=prev.avg_clarity_score,
                score_after=curr.avg_clarity_score,
                score_drop=score_drop,
                affected_documents=affected_docs
            )
            regressions.append(regression)

    return regressions


def calculate_velocity(evaluations: List[CommitEvaluation]) -> float:
    """
    Calculate quality improvement velocity (change per week).

    Args:
        evaluations: List of commit evaluations (chronological order)

    Returns:
        Average score change per week

    Example:
        ```python
        velocity = calculate_velocity(evaluations)
        # +0.5 = improving by 0.5 points per week
        # -0.2 = declining by 0.2 points per week
        ```
    """
    if len(evaluations) < 2:
        return 0.0

    # Calculate time span
    start = datetime.fromisoformat(evaluations[0].commit_date.replace('Z', '+00:00'))
    end = datetime.fromisoformat(evaluations[-1].commit_date.replace('Z', '+00:00'))
    weeks = (end - start).total_seconds() / (7 * 24 * 3600)

    if weeks < 0.1:  # Less than ~7 hours
        return 0.0

    # Calculate score change
    score_change = evaluations[-1].avg_clarity_score - evaluations[0].avg_clarity_score

    # Velocity = change / time
    velocity = score_change / weeks

    return velocity


def identify_improvement_opportunities(
    evaluations: List[CommitEvaluation],
    bottom_percentile: float = 0.2
) -> List[str]:
    """
    Identify documents that consistently score low (improvement opportunities).

    Args:
        evaluations: List of commit evaluations
        bottom_percentile: Percentile threshold (default: bottom 20%)

    Returns:
        List of document paths that need improvement

    Example:
        ```python
        low_quality_docs = identify_improvement_opportunities(evaluations)

        print(f"Focus improvement efforts on {len(low_quality_docs)} documents:")
        for doc in low_quality_docs[:5]:
            print(f"  - {doc}")
        ```
    """
    # Aggregate scores across all commits
    doc_scores: Dict[str, List[float]] = {}

    for evaluation in evaluations:
        for doc_path, score in evaluation.document_scores.items():
            if doc_path not in doc_scores:
                doc_scores[doc_path] = []
            doc_scores[doc_path].append(score)

    # Calculate mean score per document
    doc_means = {
        doc_path: np.mean(scores)
        for doc_path, scores in doc_scores.items()
    }

    # Find threshold for bottom percentile
    all_means = list(doc_means.values())
    threshold = np.percentile(all_means, bottom_percentile * 100)

    # Return documents below threshold
    low_quality = [
        doc_path
        for doc_path, mean_score in doc_means.items()
        if mean_score <= threshold
    ]

    return sorted(low_quality, key=lambda d: doc_means[d])
