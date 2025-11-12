# Migration Guide - Evaluation Improvements

This guide helps you adopt the new evaluation improvements in your Stackbench workflow.

## Overview

The evaluation improvements are **backward compatible** - existing code continues to work. New features are opt-in.

## What's New

### 1. Confidence Intervals (Optional)

**Before:**
```json
{
  "overall_score": 7.5
}
```

**After (with uncertainty):**
```json
{
  "overall_score": 7.5,
  "confidence_interval_lower": 7.2,
  "confidence_interval_upper": 7.8,
  "score_std_dev": 0.3,
  "num_samples": 10
}
```

**Migration:** Fields are optional - existing code works unchanged.

### 2. Using Uncertainty Quantification

**Add to your code:**
```python
from stackbench.evaluation import estimate_uncertainty_monte_carlo

# Wrap your evaluation
uncertainty = await estimate_uncertainty_monte_carlo(
    evaluation_func=lambda: your_evaluation_function(doc),
    score_extractor=lambda result: result.overall_score,
    num_samples=10  # Run evaluation 10 times
)

# Use results
print(f"Score: {uncertainty.mean_score:.1f} ± {uncertainty.std_dev:.1f}")
if uncertainty.std_dev > 1.0:
    print("⚠️ High uncertainty - recommend review")
```

**Performance:** +10x evaluation time (parallelizable)

### 3. Multi-Model Ensemble (Optional)

**Before:** Single model (Claude only)

**After:** Multi-model evaluation

```python
from stackbench.evaluation import evaluate_with_ensemble

# Define evaluators
evaluators = {
    "claude-sonnet-4.5": lambda doc: claude_evaluate(doc),
    "gpt-4-turbo": lambda doc: gpt4_evaluate(doc),
    "gemini-2.0-flash": lambda doc: gemini_evaluate(doc),
}

# Ensemble evaluation
result = await evaluate_with_ensemble(doc, evaluators)

print(f"Mean: {result.mean_score:.1f}")
print(f"Agreement: {result.inter_model_agreement:.2f}")

if result.flagged_for_review:
    print("⚠️ Models disagree - needs human review")
```

**Benefits:** Reduced biases, more robust scores

### 4. Graded Code Validation (Optional)

**Before:** Binary pass/fail

**After:** 4-level partial credit

```python
from stackbench.evaluation import graded_validate_code

# Instead of binary
result = await graded_validate_code(code)

# Get partial credit
print(f"Total: {result.total_score:.2f}")  # 0.0-1.0
print(f"Syntax: {result.component_scores['syntax']}")
print(f"Execution: {result.component_scores['executes']}")
print(f"Output: {result.component_scores['output_correct']}")
print(f"Category: {result.category}")  # syntax_error, runtime_error, etc.
```

**Benefits:** Better feedback, distinguish error types

### 5. Validation Against Ground Truth (New)

**Create ground truth dataset:**
```python
from stackbench.evaluation import GroundTruthManager, HumanAnnotation

manager = GroundTruthManager(Path("data/ground_truth"))

# Create dataset
dataset = manager.create_dataset(
    dataset_id="my_dataset_v1",
    description="30 docs annotated by 3 experts"
)

# Add annotations
annotation = HumanAnnotation(
    document_id="doc1.md",
    annotator_id="expert_1",
    overall_clarity=8.5,
    instruction_clarity=9.0,
    ...
)
dataset = manager.add_annotation(dataset, annotation)
manager.save_dataset(dataset)
```

**Validate LLM against ground truth:**
```python
from stackbench.evaluation import validate_against_dataset

report = validate_against_dataset(llm_results, ground_truth)

print(f"MAE: {report.mean_absolute_error:.2f}")
print(f"Correlation: {report.correlation:.2f}")
print(f"Within ±1.0: {report.within_threshold_rate:.1%}")
```

### 6. Temporal Quality Tracking (New)

**Track quality over time:**
```python
from stackbench.evaluation import calculate_trend, find_regressions

# Evaluate all commits
evaluations = []
for commit in git_history:
    result = evaluate_commit(commit)
    evaluations.append(result)

# Analyze trend
trend = calculate_trend(evaluations)
print(f"Trend: {trend.trend_direction}")  # improving/declining/stable
print(f"Velocity: {trend.trend_slope:.3f} points/commit")

# Find regressions
regressions = find_regressions(evaluations, threshold=1.0)
for reg in regressions:
    print(f"⚠️ Regression at {reg.commit_sha}: -{reg.score_drop:.1f}")
```

## Breaking Changes

**None!** All changes are backward compatible.

## Recommended Adoption Path

### Phase 1: Start Simple (Week 1)
1. **Add uncertainty to CLI output**
   - Minimal code change
   - Shows confidence intervals in reports
   - Helps users understand reliability

### Phase 2: Validation (Week 2-3)
2. **Create small ground truth dataset**
   - 10-20 documents
   - 1-2 expert annotations each
   - Validate evaluation quality

3. **Measure precision/recall**
   - Calculate FPR
   - Track over time
   - Identify improvement opportunities

### Phase 3: Advanced (Week 4+)
4. **Enable ensemble mode**
   - Add GPT-4, Gemini evaluators
   - Compare with Claude
   - Use consensus for production

5. **Add temporal tracking**
   - Track quality weekly
   - Alert on regressions
   - Measure improvement velocity

6. **Run user study**
   - 20 participants
   - Measure task completion
   - Correlate with scores

## Configuration

### Enable Uncertainty (CLI Flag)
```bash
stackbench run --with-uncertainty --num-samples 10
```

### Enable Ensemble (CLI Flag)
```bash
stackbench run --ensemble --models claude,gpt4,gemini
```

### Enable Graded Validation (CLI Flag)
```bash
stackbench run --graded-validation
```

## Performance Considerations

| Feature | Overhead | Parallelizable? |
|---------|----------|-----------------|
| Uncertainty (10 samples) | +10x | Yes |
| Ensemble (3 models) | +3x | Yes |
| Graded validation | Negligible | N/A |
| Temporal tracking | One-time | Yes (per commit) |

**Optimization tips:**
- Run uncertainty/ensemble asynchronously
- Cache evaluation results
- Use adaptive sampling (fewer samples for high-confidence)

## Troubleshooting

### Issue: High uncertainty
**Symptom:** `std_dev > 1.0`, wide confidence intervals

**Solutions:**
1. Increase samples: Try 20 instead of 10
2. Use ensemble evaluation
3. Review prompt for ambiguity
4. Consider human review for this doc

### Issue: Low correlation with ground truth
**Symptom:** `correlation < 0.5`

**Solutions:**
1. Collect more ground truth examples (target: 30+)
2. Check inter-annotator agreement (should be >0.7)
3. Review scoring rubric
4. Consider retraining/recalibration

### Issue: High false positive rate
**Symptom:** `FPR > 20%`

**Solutions:**
1. Review issue detection prompt
2. Add few-shot examples
3. Increase severity threshold for flagging
4. Consider fine-tuning on labeled data

## Example: Full Integration

```python
from stackbench.evaluation import (
    estimate_uncertainty_monte_carlo,
    evaluate_with_ensemble,
    validate_against_dataset,
    display_clarity_score_with_uncertainty
)

async def evaluate_with_improvements(doc):
    # 1. Ensemble evaluation
    ensemble_result = await evaluate_with_ensemble(
        doc,
        {
            "claude": claude_eval,
            "gpt4": gpt4_eval,
            "gemini": gemini_eval,
        }
    )

    # 2. Add uncertainty
    uncertainty = await estimate_uncertainty_monte_carlo(
        evaluation_func=lambda: ensemble_eval(doc),
        score_extractor=lambda r: r.mean_score,
        num_samples=10
    )

    # 3. Display with confidence
    display_clarity_score_with_uncertainty(
        score=uncertainty.mean_score,
        confidence_lower=uncertainty.confidence_interval[0],
        confidence_upper=uncertainty.confidence_interval[1],
        std_dev=uncertainty.std_dev
    )

    # 4. Flag if needed
    if uncertainty.std_dev > 1.0 or ensemble_result.inter_model_agreement < 0.6:
        print("⚠️ Flagged for human review")

    return uncertainty, ensemble_result
```

## Support

- **Documentation:** See `stackbench/evaluation/README.md`
- **Examples:** See `EVALUATION_ANALYSIS.md`
- **Issues:** Report to GitHub issues

## Next Steps

1. **Try uncertainty quantification** - Easy win, immediate value
2. **Create ground truth dataset** - Critical for validation
3. **Run validation study** - Measure evaluation quality
4. **Enable ensemble** - Reduce biases
5. **Track quality over time** - Continuous improvement

All improvements are **opt-in** - adopt at your own pace!
