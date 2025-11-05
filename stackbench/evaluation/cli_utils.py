"""
CLI utilities for displaying evaluation results with uncertainty.

Provides rich formatting for displaying confidence intervals, ensemble results,
and validation metrics in the command line interface.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

from ..schemas import ClarityScore


console = Console()


def format_score_with_uncertainty(
    score: float,
    confidence_lower: Optional[float] = None,
    confidence_upper: Optional[float] = None,
    std_dev: Optional[float] = None
) -> str:
    """
    Format a score with confidence interval for display.

    Args:
        score: Main score value
        confidence_lower: Lower bound of confidence interval
        confidence_upper: Upper bound of confidence interval
        std_dev: Standard deviation

    Returns:
        Formatted string with confidence info

    Example:
        >>> format_score_with_uncertainty(7.5, 7.2, 7.8, 0.3)
        "7.5 ± 0.3 (95% CI: 7.2-7.8) [high confidence]"
    """
    base = f"{score:.1f}"

    if std_dev is not None:
        base += f" ± {std_dev:.1f}"

    if confidence_lower is not None and confidence_upper is not None:
        ci_str = f"95% CI: {confidence_lower:.1f}-{confidence_upper:.1f}"
        base += f" ({ci_str})"

        # Add confidence category
        ci_width = confidence_upper - confidence_lower
        if ci_width < 0.5:
            confidence_level = "[green]high confidence[/green]"
        elif ci_width < 1.5:
            confidence_level = "[yellow]medium confidence[/yellow]"
        else:
            confidence_level = "[red]low confidence[/red]"

        base += f" {confidence_level}"

    return base


def display_clarity_score_with_uncertainty(
    clarity_score: ClarityScore,
    show_dimensions: bool = True
) -> None:
    """
    Display clarity score with uncertainty quantification.

    Args:
        clarity_score: ClarityScore object (potentially with confidence intervals)
        show_dimensions: Whether to show dimensional scores

    Example:
        ```python
        from stackbench.schemas import ClarityScore

        score = ClarityScore(
            overall_score=7.5,
            tier="B",
            instruction_clarity=8.0,
            logical_flow=7.0,
            completeness=7.5,
            consistency=8.0,
            prerequisite_coverage=6.5,
            confidence_interval_lower=7.2,
            confidence_interval_upper=7.8,
            score_std_dev=0.3,
            num_samples=10
        )

        display_clarity_score_with_uncertainty(score)
        ```
    """
    # Main score with uncertainty
    score_str = format_score_with_uncertainty(
        clarity_score.overall_score,
        clarity_score.confidence_interval_lower,
        clarity_score.confidence_interval_upper,
        clarity_score.score_std_dev
    )

    console.print(f"[bold]Overall Clarity Score:[/bold] {score_str} [bold cyan](Tier {clarity_score.tier})[/bold cyan]")

    # Add sample size info if available
    if clarity_score.num_samples:
        console.print(f"   [dim]Based on {clarity_score.num_samples} evaluation samples[/dim]")

    # Show dimensional scores if requested
    if show_dimensions:
        console.print("\n[bold]Dimensional Scores:[/bold]")
        dimensions = [
            ("Instruction Clarity", clarity_score.instruction_clarity),
            ("Logical Flow", clarity_score.logical_flow),
            ("Completeness", clarity_score.completeness),
            ("Consistency", clarity_score.consistency),
            ("Prerequisite Coverage", clarity_score.prerequisite_coverage),
        ]

        for name, score in dimensions:
            console.print(f"   • {name}: {score:.1f}/10")


def create_validation_metrics_table(
    precision: float,
    recall: float,
    f1: float,
    false_positive_rate: float,
    sample_size: int
) -> Table:
    """
    Create a rich table displaying validation metrics.

    Args:
        precision: Precision value (0-1)
        recall: Recall value (0-1)
        f1: F1 score (0-1)
        false_positive_rate: FPR (0-1)
        sample_size: Number of samples used

    Returns:
        Rich Table object

    Example:
        ```python
        table = create_validation_metrics_table(0.82, 0.70, 0.76, 0.18, 100)
        console.print(table)
        ```
    """
    table = Table(title="Evaluation Quality Metrics", show_header=True)

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_column("Interpretation", style="dim")

    # Precision
    precision_pct = f"{precision * 100:.1f}%"
    precision_interp = "High" if precision > 0.8 else "Medium" if precision > 0.6 else "Low"
    table.add_row("Precision", precision_pct, f"{precision_interp} - {int((1-precision)*100)}% false positives")

    # Recall
    recall_pct = f"{recall * 100:.1f}%"
    recall_interp = "High" if recall > 0.8 else "Medium" if recall > 0.6 else "Low"
    table.add_row("Recall", recall_pct, f"{recall_interp} - {int((1-recall)*100)}% issues missed")

    # F1
    f1_pct = f"{f1 * 100:.1f}%"
    table.add_row("F1 Score", f1_pct, "Harmonic mean of precision & recall")

    # FPR
    fpr_pct = f"{false_positive_rate * 100:.1f}%"
    fpr_interp = "Acceptable" if false_positive_rate < 0.2 else "High - retrain recommended"
    table.add_row("False Positive Rate", fpr_pct, fpr_interp)

    # Sample size
    table.add_row("Sample Size", str(sample_size), "Ground truth examples")

    return table


def display_ensemble_results(
    mean_score: float,
    std_dev: float,
    confidence_interval: tuple[float, float],
    inter_model_agreement: float,
    model_scores: dict[str, float],
    flagged_for_review: bool
) -> None:
    """
    Display ensemble evaluation results.

    Args:
        mean_score: Mean score across models
        std_dev: Standard deviation
        confidence_interval: (lower, upper) tuple
        inter_model_agreement: Agreement score (0-1)
        model_scores: Dict mapping model name to score
        flagged_for_review: Whether result is flagged for human review

    Example:
        ```python
        display_ensemble_results(
            mean_score=7.5,
            std_dev=0.3,
            confidence_interval=(7.2, 7.8),
            inter_model_agreement=0.95,
            model_scores={
                "claude-sonnet-4.5": 7.6,
                "gpt-4-turbo": 7.5,
                "gemini-2.0-flash": 7.4
            },
            flagged_for_review=False
        )
        ```
    """
    console.print("\n[bold cyan]🎭 Ensemble Evaluation Results[/bold cyan]")

    # Main score
    score_str = format_score_with_uncertainty(
        mean_score,
        confidence_interval[0],
        confidence_interval[1],
        std_dev
    )
    console.print(f"\n[bold]Ensemble Score:[/bold] {score_str}")

    # Inter-model agreement
    agreement_pct = inter_model_agreement * 100
    if inter_model_agreement > 0.8:
        agreement_color = "green"
        agreement_label = "High agreement"
    elif inter_model_agreement > 0.5:
        agreement_color = "yellow"
        agreement_label = "Moderate agreement"
    else:
        agreement_color = "red"
        agreement_label = "Low agreement"

    console.print(f"[bold]Inter-Model Agreement:[/bold] [{agreement_color}]{agreement_pct:.0f}% ({agreement_label})[/{agreement_color}]")

    # Flag for review
    if flagged_for_review:
        console.print("\n[bold red]⚠️  FLAGGED FOR HUMAN REVIEW[/bold red]")
        console.print("   [yellow]Models disagree significantly on this evaluation[/yellow]")

    # Individual model scores
    console.print("\n[bold]Individual Model Scores:[/bold]")
    for model_name, score in model_scores.items():
        console.print(f"   • {model_name}: {score:.1f}/10")


def display_temporal_trend(
    trend_direction: str,
    trend_slope: float,
    num_regressions: int,
    velocity: float
) -> None:
    """
    Display temporal quality trend.

    Args:
        trend_direction: "improving", "declining", or "stable"
        trend_slope: Slope of trend line
        num_regressions: Number of quality regressions found
        velocity: Change per week

    Example:
        ```python
        display_temporal_trend(
            trend_direction="improving",
            trend_slope=0.05,
            num_regressions=2,
            velocity=0.3
        )
        ```
    """
    console.print("\n[bold cyan]📈 Temporal Quality Trend[/bold cyan]")

    # Trend direction
    if trend_direction == "improving":
        trend_icon = "📈"
        trend_color = "green"
    elif trend_direction == "declining":
        trend_icon = "📉"
        trend_color = "red"
    else:
        trend_icon = "➡️"
        trend_color = "yellow"

    console.print(f"\n[bold]Trend:[/bold] {trend_icon} [{trend_color}]{trend_direction.title()}[/{trend_color}]")
    console.print(f"[bold]Slope:[/bold] {trend_slope:+.3f} points per commit")
    console.print(f"[bold]Velocity:[/bold] {velocity:+.2f} points per week")

    # Regressions
    if num_regressions > 0:
        console.print(f"\n[yellow]⚠️  {num_regressions} quality regression(s) detected[/yellow]")
    else:
        console.print(f"\n[green]✓ No quality regressions detected[/green]")
