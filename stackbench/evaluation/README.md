# Evaluation Module - Research-Based Improvements

This module implements research-grade evaluation methodologies to address critical issues identified in the evaluation analysis (see `EVALUATION_ANALYSIS.md` in project root).

## Quick Start

```python
from stackbench.evaluation import (
    estimate_uncertainty_monte_carlo,
    calculate_precision_recall,
    ensemble_evaluate,
    graded_validate_code,
)

# Example 1: Uncertainty quantification
uncertainty = await estimate_uncertainty_monte_carlo(
    evaluation_func=lambda: evaluate_document(doc),
    score_extractor=lambda result: result.overall_score,
    num_samples=10
)
print(f"Score: {uncertainty.mean_score:.1f} ± {uncertainty.std_dev:.1f}")
print(f"95% CI: {uncertainty.confidence_interval}")

# Example 2: Validation metrics
metrics = calculate_precision_recall(
    predicted_issues=["issue1", "issue2", "issue5"],
    ground_truth_labels=ground_truth_dict
)
print(f"Precision: {metrics.precision:.2f}")
print(f"Recall: {metrics.recall:.2f}")
print(f"False Positive Rate: {metrics.false_positive_rate:.2f}")

# Example 3: Ensemble evaluation
result = await evaluate_with_ensemble(
    document=doc,
    model_evaluators={
        "claude-sonnet-4.5": claude_eval,
        "gpt-4-turbo": gpt4_eval,
        "gemini-2.0-flash": gemini_eval,
    }
)
print(f"Ensemble: {result.mean_score:.1f} (agreement: {result.inter_model_agreement:.2f})")

# Example 4: Graded code validation
result = await graded_validate_code(code="print('Hello')")
print(f"Score: {result.total_score:.2f}")  # 0.0-1.0 instead of binary pass/fail
print(f"Category: {result.category}")  # syntax_error, runtime_error, wrong_output, etc.
```

## Modules Overview

### 1. Uncertainty Quantification (`uncertainty.py`)

**Problem Solved:** Mistake #3 - No uncertainty quantification

**What it does:**
- Provides confidence intervals for scores instead of single-point estimates
- Uses Monte Carlo sampling to measure score variability
- Flags high-uncertainty results for human review

**Key Functions:**
- `estimate_uncertainty_monte_carlo()` - Run evaluation N times, get confidence interval
- `calculate_confidence_interval()` - Compute 95% CI from samples
- `should_flag_for_review()` - Identify uncertain results needing manual check
- `get_confidence_category()` - Classify as high/medium/low confidence

**Example Output:**
```
Score: 7.5 ± 0.3 (95% CI: 7.2-7.8) [high confidence]
```

**Why it matters:**
- Score of 7.2 ± 0.1 (confident) is very different from 7.2 ± 2.0 (uncertain)
- High uncertainty = needs human review
- Low uncertainty = trust the evaluation

---

### 2. Metrics & Validation (`metrics.py`)

**Problem Solved:** Mistakes #1 & #7 - No calibration, no FPR tracking

**What it does:**
- Measures evaluation quality against human ground truth
- Calculates precision, recall, F1, false positive rate
- Tracks inter-annotator agreement

**Key Functions:**
- `calculate_precision_recall()` - Compute TP/FP/FN/TN metrics
- `calculate_inter_annotator_agreement_simple()` - Measure human agreement
- `analyze_false_positive_patterns()` - Find common FP types

**Example:**
```python
metrics = calculate_precision_recall(predicted, ground_truth)

# Precision: 82% (18% false positives)
# Recall: 70% (30% issues missed)
# F1: 0.76
# FPR: 18%
```

**Interpretation:**
- **High FPR (>20%)** = System cries wolf, users will ignore it
- **Low recall (<70%)** = Missing real problems
- **Track over time** = Is evaluation improving?

---

### 3. Ground Truth Management (`ground_truth.py`)

**Problem Solved:** Mistake #1 - No calibration against human judgment

**What it does:**
- Manages gold standard datasets with human annotations
- Supports multi-annotator datasets
- Calculates mean scores across annotators

**Key Classes:**
- `HumanAnnotation` - Expert annotation of a document
- `GroundTruthDataset` - Collection of annotations
- `GroundTruthManager` - CRUD operations for datasets

**Example Workflow:**
```python
# 1. Create dataset
manager = GroundTruthManager(Path("data/ground_truth"))
dataset = manager.create_dataset(
    dataset_id="lancedb_docs_v1",
    description="30 LanceDB docs annotated by 3 experts",
    num_annotators_per_doc=3
)

# 2. Add annotations
annotation = HumanAnnotation(
    document_id="pydantic.md",
    annotator_id="expert_1",
    overall_clarity=8.5,
    instruction_clarity=9.0,
    ...
)
dataset = manager.add_annotation(dataset, annotation)

# 3. Save
manager.save_dataset(dataset)
```

**Best Practices:**
- **30-50 documents** minimum for validation
- **3 annotators per document** for inter-rater reliability
- **Calculate agreement** (should be >0.7)

---

### 4. Validation Comparison (`validation.py`)

**Problem Solved:** Measuring LLM evaluation quality

**What it does:**
- Compares LLM scores to human annotations
- Calculates MAE, RMSE, correlation
- Identifies problematic documents

**Key Functions:**
- `compare_llm_to_human()` - Single document comparison
- `validate_against_dataset()` - Full dataset validation
- `identify_problematic_documents()` - Find large errors

**Example:**
```python
report = validate_against_dataset(llm_results, ground_truth)

print(f"MAE: {report.mean_absolute_error:.2f}")  # Average error
print(f"RMSE: {report.root_mean_squared_error:.2f}")
print(f"Correlation: {report.correlation:.2f}")  # 0-1 scale
print(f"Within ±1.0: {report.within_threshold_rate:.1%}")
```

**Target Metrics:**
- **MAE < 1.0** = Good prediction accuracy
- **Correlation > 0.7** = Strong relationship with human scores
- **Within threshold > 80%** = Most predictions are close

---

### 5. Ensemble Evaluation (`ensemble.py`)

**Problem Solved:** Mistake #2 - Single model bias

**What it does:**
- Evaluates using multiple LLMs (Claude, GPT-4, Gemini)
- Aggregates results to reduce bias
- Detects high disagreement cases

**Key Functions:**
- `ensemble_evaluate()` - Aggregate multi-model results
- `calculate_inter_model_agreement()` - Measure model consensus
- `evaluate_with_ensemble()` - Parallel evaluation
- `compare_model_biases()` - Analyze systematic biases

**Example:**
```python
ensemble = await evaluate_with_ensemble(
    document=doc,
    model_evaluators={
        "claude-sonnet-4.5": claude_eval,
        "gpt-4-turbo": gpt4_eval,
        "gemini-2.0-flash": gemini_eval,
    }
)

# Mean: 7.5 ± 0.2
# Agreement: 0.95 (high)
# Consensus issues: 12 (found by 2+ models)
```

**Benefits:**
- **Reduces biases:** Position, verbosity, style
- **Identifies ambiguous cases:** Low agreement = needs review
- **More robust:** Less vulnerable to single-model quirks

---

### 6. Graded Validation (`graded_validation.py`)

**Problem Solved:** Mistake #6 - Binary code validation

**What it does:**
- Assigns partial credit instead of pass/fail
- Validates at 4 levels: syntax, quality, execution, output
- Provides specific improvement suggestions

**Scoring Breakdown:**
- **Syntax (25%)**: Does it parse?
- **Static quality (25%)**: Code quality issues (pylint-style)
- **Execution (25%)**: Does it run?
- **Output correctness (25%)**: Produces correct result?

**Example:**
```python
result = await graded_validate_code(code="""
print('Hello')
""", expected_output="Hello")

# Score: 1.0 (perfect)
# Category: "perfect"

result = await graded_validate_code(code="""
print('Hello'
""")

# Score: 0.0 (syntax error)
# Category: "syntax_error"
# Suggestions: ["Fix syntax error: unexpected EOF"]
```

**Why it matters:**
- **Distinguish** "syntax error" from "wrong output" from "minor quality issues"
- **Better feedback** for documentation authors
- **ROI analysis**: Fixing syntax errors is cheap, fixing logic is expensive

---

### 7. Temporal Tracking (`temporal.py`)

**Problem Solved:** Mistake #9 - No temporal tracking

**What it does:**
- Tracks quality changes over git history
- Identifies quality regressions
- Measures improvement velocity

**Key Functions:**
- `calculate_trend()` - Quality trend (improving/declining/stable)
- `find_regressions()` - Commits with score drops
- `calculate_velocity()` - Change per week
- `identify_improvement_opportunities()` - Low-quality docs needing work

**Example:**
```python
trend = calculate_trend(commit_evaluations)

# Trend: 📈 Improving
# Slope: +0.05 points/commit
# Velocity: +0.3 points/week

regressions = find_regressions(commit_evaluations, threshold=1.0)
# Found 2 regressions:
#   - Commit abc123: Score drop of 1.5
#   - Commit def456: Score drop of 1.2
```

**Use Cases:**
- **Track progress:** Is documentation getting better?
- **Identify regressions:** Which commits broke quality?
- **Prioritize work:** Which docs need most improvement?

---

### 8. CLI Utilities (`cli_utils.py`)

**Problem Solved:** Make uncertainty visible to users

**What it does:**
- Rich terminal formatting
- Display confidence intervals
- Show ensemble results
- Visualize temporal trends

**Key Functions:**
- `format_score_with_uncertainty()` - Pretty print with CI
- `display_clarity_score_with_uncertainty()` - Full score display
- `create_validation_metrics_table()` - Metrics table
- `display_ensemble_results()` - Multi-model results
- `display_temporal_trend()` - Trend visualization

**Example Output:**
```
Overall Clarity Score: 7.5 ± 0.3 (95% CI: 7.2-7.8) [high confidence] (Tier B)
   Based on 10 evaluation samples

Dimensional Scores:
   • Instruction Clarity: 8.0/10
   • Logical Flow: 7.0/10
   • Completeness: 7.5/10
   • Consistency: 8.0/10
   • Prerequisite Coverage: 6.5/10
```

---

## Integration Examples

### Example 1: Run evaluation with uncertainty

```python
from stackbench.evaluation import estimate_uncertainty_monte_carlo
from stackbench.agents.clarity_agent import DocumentationClarityAgent

async def evaluate_with_confidence(doc_path: Path):
    agent = DocumentationClarityAgent(...)

    # Estimate uncertainty
    uncertainty = await estimate_uncertainty_monte_carlo(
        evaluation_func=lambda: agent.analyze_document(doc_path),
        score_extractor=lambda result: result.clarity_score.overall_score,
        num_samples=10
    )

    # Check if needs review
    if should_flag_for_review(uncertainty):
        print("⚠️  High uncertainty - recommend human review")

    return uncertainty
```

### Example 2: Validate evaluation quality

```python
from stackbench.evaluation import (
    GroundTruthManager,
    validate_against_dataset,
    calculate_precision_recall
)

# Load ground truth
manager = GroundTruthManager(Path("data/ground_truth"))
ground_truth = manager.load_dataset("lancedb_docs_v1")

# Run LLM evaluations
llm_results = run_all_evaluations(documents)

# Validate
report = validate_against_dataset(llm_results, ground_truth)

print(f"MAE: {report.mean_absolute_error:.2f}")
print(f"Correlation: {report.correlation:.2f}")

if report.mean_absolute_error > 1.5:
    print("⚠️  High error - consider retraining or recalibration")
```

### Example 3: Ensemble evaluation

```python
from stackbench.evaluation import evaluate_with_ensemble

async def robust_evaluation(doc):
    # Define evaluators for different models
    evaluators = {
        "claude-sonnet-4.5": lambda d: claude_evaluate(d),
        "gpt-4-turbo": lambda d: gpt4_evaluate(d),
        "gemini-2.0-flash": lambda d: gemini_evaluate(d),
    }

    # Ensemble
    result = await evaluate_with_ensemble(doc, evaluators)

    if result.flagged_for_review:
        print("⚠️  Models disagree - needs human review")
        for model_result in result.individual_results:
            print(f"  {model_result.model_name}: {model_result.overall_score:.1f}")

    return result.mean_score, result.std_dev
```

### Example 4: Track quality over time

```python
from stackbench.evaluation import calculate_trend, find_regressions

# Evaluate all commits
evaluations = []
for commit in git_history:
    docs = checkout_docs(commit.sha)
    eval_result = evaluate_all(docs)
    evaluations.append(eval_result)

# Analyze trend
trend = calculate_trend(evaluations)
print(f"Trend: {trend.trend_direction}")
print(f"Velocity: {trend.velocity:.2f} points/week")

# Find regressions
regressions = find_regressions(evaluations, threshold=1.0)
for reg in regressions:
    print(f"⚠️  Regression at {reg.commit_sha[:7]}: -{reg.score_drop:.1f}")
```

---

## Roadmap (Future Improvements)

### Phase 1: Foundations (Implemented ✅)
- [x] Uncertainty quantification
- [x] False positive tracking
- [x] Ground truth management
- [x] Precision/recall metrics
- [x] Ensemble evaluation
- [x] Graded validation
- [x] Temporal tracking

### Phase 2: Advanced Features (Planned)
- [ ] Krippendorff's alpha for inter-annotator agreement
- [ ] Active learning pipeline
- [ ] Prompt robustness testing
- [ ] User outcome correlation studies
- [ ] Reinforcement learning for auto-repair
- [ ] Adaptive validation strategy selection

### Phase 3: Research Contributions (Planned)
- [ ] Publish gold standard dataset
- [ ] Benchmark LLM-as-judge approaches
- [ ] User study validation
- [ ] Open-source evaluation toolkit

---

## References

- **EVALUATION_ANALYSIS.md** - Full analysis of evaluation mistakes and improvements
- **Research papers:**
  - "LLM-as-Judge: Evaluating Large Language Models for Automated Assessment"
  - "Uncertainty Quantification in Neural Network Predictions"
  - "Multi-Model Ensemble Methods for Robust AI Systems"

---

## Contributing

When adding new evaluation methods:

1. **Add module** in `stackbench/evaluation/`
2. **Export** in `__init__.py`
3. **Add tests** in `tests/evaluation/`
4. **Update docs** in this README
5. **Add CLI integration** if user-facing

Example PR checklist:
- [ ] Module implements research-based methodology
- [ ] Has comprehensive docstrings with examples
- [ ] Exports all public APIs in `__init__.py`
- [ ] Has unit tests with >80% coverage
- [ ] Documented in README with usage examples
- [ ] Integrated with CLI (if applicable)

---

## License

Same as StackBench project license.
