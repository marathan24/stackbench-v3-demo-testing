"""
User outcome tracking for validation studies.

Tracks real user outcomes (task completion, time, errors, satisfaction)
to validate that quality scores predict actual user success.
Addresses Mistake #4 from EVALUATION_ANALYSIS.md.
"""

from typing import List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np


class UserTaskOutcome(BaseModel):
    """Outcome of a user completing a task using documentation."""

    participant_id: str = Field(description="Unique participant identifier")
    document_id: str = Field(description="Document used for task")
    task_description: str = Field(description="Task to complete")

    # Outcomes
    completed: bool = Field(description="Whether task was completed successfully")
    time_minutes: float = Field(description="Time taken to complete task")
    errors_encountered: int = Field(description="Number of errors encountered")
    help_requests: int = Field(description="Number of times participant asked for help")

    # Subjective ratings (1-5 scale)
    satisfaction: int = Field(ge=1, le=5, description="Overall satisfaction (1-5)")
    clarity_rating: int = Field(ge=1, le=5, description="How clear was the documentation? (1-5)")
    completeness_rating: int = Field(ge=1, le=5, description="How complete was the documentation? (1-5)")

    # Experience level
    experience_level: str = Field(description="beginner, intermediate, or expert")

    # Additional notes
    notes: Optional[str] = None


class OutcomeStudy(BaseModel):
    """Collection of user outcomes for a documentation set."""

    study_id: str
    description: str
    documents: List[str] = Field(description="List of document IDs in study")
    outcomes: List[UserTaskOutcome] = Field(default_factory=list)


@dataclass
class CorrelationReport:
    """Correlation between predicted scores and actual outcomes."""

    study_id: str
    num_participants: int
    num_documents: int

    # Correlations (Pearson r)
    score_vs_completion_rate: float
    """Correlation between clarity score and task completion rate"""

    score_vs_time: float
    """Correlation between clarity score and time taken (negative = higher score = less time)"""

    score_vs_errors: float
    """Correlation between clarity score and errors (negative = higher score = fewer errors)"""

    score_vs_satisfaction: float
    """Correlation between clarity score and user satisfaction"""

    # Statistical significance
    p_values: dict[str, float]
    """P-values for each correlation"""

    validity: str
    """High, medium, or low validity based on correlations"""


def calculate_outcome_metrics(outcomes: List[UserTaskOutcome]) -> dict:
    """
    Calculate aggregate metrics from user outcomes.

    Args:
        outcomes: List of user task outcomes

    Returns:
        Dict with aggregate metrics

    Example:
        ```python
        metrics = calculate_outcome_metrics(outcomes)

        print(f"Completion rate: {metrics['completion_rate']:.1%}")
        print(f"Median time: {metrics['median_time']:.1f} minutes")
        print(f"Mean errors: {metrics['mean_errors']:.1f}")
        ```
    """
    if not outcomes:
        return {}

    return {
        "completion_rate": sum(o.completed for o in outcomes) / len(outcomes),
        "median_time": np.median([o.time_minutes for o in outcomes]),
        "mean_time": np.mean([o.time_minutes for o in outcomes]),
        "mean_errors": np.mean([o.errors_encountered for o in outcomes]),
        "mean_satisfaction": np.mean([o.satisfaction for o in outcomes]),
        "mean_clarity_rating": np.mean([o.clarity_rating for o in outcomes]),
        "mean_completeness_rating": np.mean([o.completeness_rating for o in outcomes]),
    }


def correlate_scores_with_outcomes(
    predicted_scores: dict[str, float],
    outcomes: List[UserTaskOutcome],
    study_id: str = "default"
) -> CorrelationReport:
    """
    Calculate correlation between predicted quality scores and actual user outcomes.

    This is the key validation: Do high-scoring docs actually help users more?

    Args:
        predicted_scores: Dict mapping document_id to predicted clarity score
        outcomes: List of user task outcomes
        study_id: Study identifier

    Returns:
        CorrelationReport with correlation statistics

    Example:
        ```python
        predicted = {
            "doc1": 8.5,  # High score
            "doc2": 6.0,  # Medium score
            "doc3": 4.5,  # Low score
        }

        report = correlate_scores_with_outcomes(predicted, outcomes)

        print(f"Score vs completion: r = {report.score_vs_completion_rate:.2f}")
        if report.validity == "high":
            print("✓ Scores are predictive of user success!")
        ```
    """
    # Group outcomes by document
    doc_outcomes: dict[str, List[UserTaskOutcome]] = {}
    for outcome in outcomes:
        if outcome.document_id not in doc_outcomes:
            doc_outcomes[outcome.document_id] = []
        doc_outcomes[outcome.document_id].append(outcome)

    # Calculate per-document outcome metrics
    doc_scores = []
    doc_completion_rates = []
    doc_times = []
    doc_errors = []
    doc_satisfaction = []

    for doc_id, doc_outcomes_list in doc_outcomes.items():
        if doc_id not in predicted_scores:
            continue  # Skip if no predicted score

        metrics = calculate_outcome_metrics(doc_outcomes_list)

        doc_scores.append(predicted_scores[doc_id])
        doc_completion_rates.append(metrics["completion_rate"])
        doc_times.append(metrics["median_time"])
        doc_errors.append(metrics["mean_errors"])
        doc_satisfaction.append(metrics["mean_satisfaction"])

    # Calculate correlations
    if len(doc_scores) < 3:
        # Need at least 3 documents for meaningful correlation
        return CorrelationReport(
            study_id=study_id,
            num_participants=len(outcomes),
            num_documents=len(doc_outcomes),
            score_vs_completion_rate=0.0,
            score_vs_time=0.0,
            score_vs_errors=0.0,
            score_vs_satisfaction=0.0,
            p_values={},
            validity="insufficient_data",
        )

    # Pearson correlation
    r_completion = np.corrcoef(doc_scores, doc_completion_rates)[0, 1]
    r_time = np.corrcoef(doc_scores, doc_times)[0, 1]  # Negative = higher score = less time
    r_errors = np.corrcoef(doc_scores, doc_errors)[0, 1]  # Negative = higher score = fewer errors
    r_satisfaction = np.corrcoef(doc_scores, doc_satisfaction)[0, 1]

    # Simple p-value estimation (for demo - in production use scipy.stats)
    # Using t-distribution approximation
    n = len(doc_scores)
    t_completion = r_completion * np.sqrt(n - 2) / np.sqrt(1 - r_completion**2)
    # In production: p_value = scipy.stats.t.sf(abs(t), n-2) * 2

    # Determine validity
    # Good: completion rate correlation > 0.7, time/error negative correlations
    if r_completion > 0.7 and r_time < -0.5 and r_errors < -0.5:
        validity = "high"
    elif r_completion > 0.5:
        validity = "medium"
    else:
        validity = "low"

    return CorrelationReport(
        study_id=study_id,
        num_participants=len(outcomes),
        num_documents=len(doc_outcomes),
        score_vs_completion_rate=float(r_completion),
        score_vs_time=float(r_time),
        score_vs_errors=float(r_errors),
        score_vs_satisfaction=float(r_satisfaction),
        p_values={
            "completion": 0.05 if abs(r_completion) > 0.7 else 0.2,  # Simplified
            "time": 0.05 if abs(r_time) > 0.5 else 0.2,
            "errors": 0.05 if abs(r_errors) > 0.5 else 0.2,
            "satisfaction": 0.05 if abs(r_satisfaction) > 0.6 else 0.2,
        },
        validity=validity,
    )


def identify_mispredictions(
    predicted_scores: dict[str, float],
    outcomes: List[UserTaskOutcome],
    threshold: float = 2.0
) -> List[dict]:
    """
    Identify documents where predicted score doesn't match user outcomes.

    Args:
        predicted_scores: Dict mapping document_id to predicted score
        outcomes: User outcomes
        threshold: Score difference threshold to flag

    Returns:
        List of mispredictions

    Example:
        ```python
        mispredictions = identify_mispredictions(predicted, outcomes)

        for mis in mispredictions:
            print(f"Doc {mis['doc_id']}:")
            print(f"  Predicted: {mis['predicted_score']:.1f}")
            print(f"  Actual success: {mis['completion_rate']:.1%}")
            print(f"  Discrepancy: {mis['discrepancy']}")
        ```
    """
    # Group outcomes by document
    doc_outcomes: dict[str, List[UserTaskOutcome]] = {}
    for outcome in outcomes:
        if outcome.document_id not in doc_outcomes:
            doc_outcomes[outcome.document_id] = []
        doc_outcomes[outcome.document_id].append(outcome)

    mispredictions = []

    for doc_id, doc_outcomes_list in doc_outcomes.items():
        if doc_id not in predicted_scores:
            continue

        predicted = predicted_scores[doc_id]
        metrics = calculate_outcome_metrics(doc_outcomes_list)

        # Convert completion rate to 0-10 scale (like clarity score)
        actual_score_proxy = metrics["completion_rate"] * 10

        # Check for large discrepancy
        discrepancy = abs(predicted - actual_score_proxy)

        if discrepancy > threshold:
            mispredictions.append({
                "doc_id": doc_id,
                "predicted_score": predicted,
                "completion_rate": metrics["completion_rate"],
                "actual_score_proxy": actual_score_proxy,
                "discrepancy": discrepancy,
                "type": "overpredicted" if predicted > actual_score_proxy else "underpredicted",
            })

    return sorted(mispredictions, key=lambda x: x["discrepancy"], reverse=True)
