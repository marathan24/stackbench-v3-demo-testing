# Stackbench Evaluation Methodology: Critical Analysis & Research Improvements

**Document Type:** Research Analysis & Improvement Roadmap
**Date:** 2025-11-05
**Focus:** Evaluation mistakes, LLM evaluation best practices, and Reinforcement Learning applications

---

## Executive Summary

Stackbench uses a multi-agent system to validate documentation quality across 4 dimensions (API accuracy, code execution, clarity, walkthroughs). While innovative, the evaluation methodology has **significant research gaps** that limit its reliability, reproducibility, and scientific validity.

**Key Issues:**
1. **LLM-as-judge without calibration** - No ground truth, inter-rater reliability, or uncertainty quantification
2. **No feedback loop** - System identifies issues but doesn't learn from corrections
3. **Binary validation** - Code execution is pass/fail, missing partial correctness gradations
4. **Single model bias** - Only Claude used for evaluation, vulnerable to model-specific biases
5. **Missing metrics** - No precision/recall, false positive rate, or user outcome correlation

**Opportunity:** Reinforcement Learning can address these gaps through auto-repair agents, adaptive validation strategies, and preference learning from user feedback.

---

## Part 1: Current System Overview

### Evaluation Components

| Component | Type | Evaluation Method | Output |
|-----------|------|-------------------|--------|
| **Extraction Agent** | Static Analysis | Pattern matching + LLM extraction | JSON with API signatures, code examples |
| **API Validation Agent** | Dynamic Analysis | Python introspection (`inspect.signature`) | Valid/Invalid/Not Found |
| **Code Validation Agent** | Dynamic Analysis | Execute code in isolated env | Success/Failed + error message |
| **Clarity Agent** | LLM-as-Judge | Claude evaluates 5 dimensions | 0-10 scores + issue list |
| **Walkthrough Audit Agent** | Interactive Execution | MCP server delivers steps sequentially | Gap reports (6 categories) |

### Scoring System

**Clarity Score Calculation** (from `clarity_scoring_server.py:232-276`):
```python
base_score = 10.0
penalties = {
    "critical_issue": -2.0,
    "warning_issue": -0.5,
    "info_issue": -0.1,
    "failed_example": -1.0,
    "invalid_api": -0.8,
    "missing_api": -1.2
}
final_score = max(0, min(10, base_score + sum(penalties)))
```

**Issues:**
- **Arbitrary penalty weights** - No empirical justification for -2.0 vs -0.8
- **Linear aggregation** - Assumes issues have independent, additive effects
- **No uncertainty** - Single point estimate without confidence intervals
- **Tier boundaries** - Why 8.0 for A-tier? Not validated against user outcomes

---

## Part 2: Evaluation Mistakes (Critical Analysis)

### 2.1 LLM-as-Judge Reliability Issues

**Mistake #1: No Calibration Against Human Judgment**

Current approach (`clarity_agent.py:96-156`):
- Clarity agent evaluates documentation and reports issues
- MCP server calculates deterministic scores from issues
- **No validation** that LLM findings correlate with actual developer confusion

**Impact:**
- Unknown false positive rate (reporting non-issues as issues)
- Unknown false negative rate (missing real problems)
- Scores may not predict user success/failure

**Research Fix:**
1. Create **gold standard dataset** (20-30 docs with expert human annotations)
2. Measure **inter-annotator agreement** (Krippendorff's alpha)
3. Compare LLM vs human on same dataset
4. Report **precision**, **recall**, **F1** for each issue type

**Example:**
```python
# Gold standard: 100 documented issues by experts
# LLM detected: 85 issues
# Overlap: 70 issues

Precision = 70 / 85 = 0.82 (18% false positives)
Recall = 70 / 100 = 0.70 (30% missed issues)
F1 = 0.76

# This tells us: LLM misses 30% of real problems!
```

---

**Mistake #2: Single Model Evaluation (Model Bias)**

Current approach:
- Only Claude Sonnet 4.5 evaluates clarity
- No cross-model validation

**Biases in LLM-as-judge:**
- **Position bias** - Issues mentioned first get more attention
- **Verbosity bias** - Longer documentation may score higher just for being detailed
- **Style bias** - Model may prefer certain writing styles it was trained on
- **Consistency bias** - Same doc evaluated twice may get different scores

**Research Fix: Multi-Model Ensemble**

```python
class EnsembleJudge:
    def __init__(self):
        self.models = [
            "claude-sonnet-4.5",
            "gpt-4-turbo",
            "gemini-2.0-flash"
        ]

    async def evaluate(self, doc: str) -> EvaluationResult:
        # 1. Get judgments from all models
        results = await asyncio.gather(*[
            self._evaluate_with_model(doc, model)
            for model in self.models
        ])

        # 2. Calculate inter-model agreement (Fleiss' kappa)
        agreement = calculate_fleiss_kappa(results)

        # 3. Flag high-disagreement cases for human review
        if agreement < 0.4:  # Low agreement
            flag_for_human_review(doc)

        # 4. Aggregate with uncertainty
        mean_score = np.mean([r.score for r in results])
        std_score = np.std([r.score for r in results])

        return EvaluationResult(
            score=mean_score,
            confidence_interval=(mean_score - 1.96*std_score,
                                 mean_score + 1.96*std_score),
            inter_model_agreement=agreement,
            individual_scores={m: r.score for m, r in zip(self.models, results)}
        )
```

**Benefits:**
- Reduces model-specific biases
- Provides uncertainty quantification
- Identifies ambiguous cases (high disagreement)
- More robust to adversarial examples

---

**Mistake #3: No Uncertainty Quantification**

Current output (`ClarityScore` schema):
```json
{
  "overall_score": 7.2,
  "instruction_clarity": 8.0,
  "logical_flow": 6.5
}
```

**Missing:** Confidence intervals!

**Why this matters:**
- Score of 7.2 ± 0.1 (high confidence) is very different from 7.2 ± 2.0 (low confidence)
- Low confidence scores should trigger human review
- Uncertainty helps prioritize which documents need manual inspection

**Research Fix: Monte Carlo Uncertainty Estimation**

```python
async def evaluate_with_uncertainty(doc: str, n_samples: int = 10) -> ScoredResult:
    """Run evaluation multiple times with stochastic prompts."""
    scores = []

    for i in range(n_samples):
        # Add controlled randomness (temperature, few-shot examples)
        result = await evaluate_doc(doc, temperature=0.7, seed=i)
        scores.append(result.score)

    return ScoredResult(
        mean_score=np.mean(scores),
        std_dev=np.std(scores),
        confidence_interval=(
            np.percentile(scores, 2.5),
            np.percentile(scores, 97.5)
        ),
        samples=scores
    )
```

---

### 2.2 Missing Ground Truth & Validation

**Mistake #4: No User Outcome Correlation**

Current system:
- Reports clarity score (0-10)
- Reports issues (critical, warning, info)
- **Never validates** if these predictions matter

**Questions unanswered:**
- Do docs with score 8.0 actually help developers more than 6.0?
- Do "critical" issues actually block users?
- What's the ROI of fixing a "warning" vs "info" issue?

**Research Fix: User Study + Outcome Tracking**

```python
class OutcomeValidator:
    """Correlate evaluation scores with real user outcomes."""

    async def run_user_study(self, docs: List[Document]) -> ValidationReport:
        # 1. Select diverse set of documents (different scores)
        test_docs = stratified_sample(docs, bins=[(0,4), (4,7), (7,10)])

        # 2. Ask developers to complete tasks using each doc
        outcomes = []
        for doc in test_docs:
            outcome = await task_completion_study(
                doc=doc,
                task="Build a simple app using this library",
                participants=20
            )
            outcomes.append({
                "doc_id": doc.id,
                "predicted_score": doc.clarity_score,
                "completion_rate": outcome.completion_rate,
                "time_to_complete": outcome.median_time,
                "errors_encountered": outcome.error_count,
                "satisfaction_rating": outcome.satisfaction
            })

        # 3. Calculate correlation
        correlation = pearsonr(
            [o["predicted_score"] for o in outcomes],
            [o["completion_rate"] for o in outcomes]
        )

        return ValidationReport(
            correlation=correlation,
            outcomes=outcomes,
            validity="high" if correlation.r > 0.7 else "low"
        )
```

**Metrics to track:**
- **Task completion rate** - % users who successfully complete tutorial
- **Time to completion** - How long it takes
- **Error rate** - How many mistakes users make
- **User satisfaction** - Self-reported clarity rating

**Expected findings:**
- High-scoring docs (8+) → 90% completion rate
- Low-scoring docs (4-6) → 50% completion rate
- **This validates the scoring system!**

---

**Mistake #5: No Feedback Loop / Learning**

Current system (`clarity_agent.py:756-951`):
- Agent evaluates document
- Reports issues
- **System never learns from outcomes**

**Problem:**
- If developer fixes an issue but users still struggle → system doesn't know
- If developer ignores an issue and users succeed → system doesn't know
- System can't improve its evaluation quality over time

**Research Fix: Active Learning Pipeline**

```python
class ActiveLearningValidator:
    """Iteratively improve evaluation quality through human feedback."""

    def __init__(self):
        self.labeled_examples = []  # Ground truth data
        self.uncertain_queue = []   # Cases needing review

    async def evaluate_with_learning(self, doc: Document) -> Result:
        # 1. Get LLM evaluation
        llm_result = await self.llm_evaluate(doc)

        # 2. Check if we have similar labeled examples
        similar = self.find_similar_labeled(doc)
        if similar:
            # Use supervised prediction from labeled data
            supervised_result = self.predict_from_similar(doc, similar)

            # Compare LLM vs supervised
            if abs(llm_result.score - supervised_result.score) > 2.0:
                # High disagreement → add to review queue
                self.uncertain_queue.append(doc)

        # 3. If low confidence, add to review queue
        if llm_result.confidence < 0.7:
            self.uncertain_queue.append(doc)

        # 4. Periodically get human labels for uncertain cases
        if len(self.uncertain_queue) >= 10:
            human_labels = await self.get_human_review(self.uncertain_queue)
            self.labeled_examples.extend(human_labels)
            self.retrain_scoring_model()
            self.uncertain_queue = []

        return llm_result

    def retrain_scoring_model(self):
        """Retrain or calibrate scoring based on labeled data."""
        # Option 1: Fine-tune LLM on labeled examples
        # Option 2: Train shallow model (gradient boosting) as calibrator
        # Option 3: Update penalty weights in scoring function
        pass
```

**Benefits:**
- System improves over time with user feedback
- Focuses human effort on most uncertain/impactful cases
- Builds proprietary ground truth dataset

---

### 2.3 Evaluation Granularity Issues

**Mistake #6: Binary Code Validation (Missing Partial Credit)**

Current approach (`code_validation_agent.py`):
```python
result = subprocess.run(["python", code_file], capture_output=True)
status = "success" if result.returncode == 0 else "failed"
```

**Problem:**
- Code that's 95% correct but has 1 typo → "failed"
- No distinction between "syntax error" vs "wrong output" vs "crashes"

**Research Fix: Gradual Correctness Scoring**

```python
class GradedCodeValidator:
    """Assign partial credit for partially correct code."""

    async def validate_code(self, code: str, expected_behavior: str) -> GradedResult:
        # Level 1: Syntax check (basic correctness)
        syntax_ok = await check_syntax(code)

        # Level 2: Static analysis (imports, undefined vars)
        static_issues = await run_pylint(code)

        # Level 3: Execution (does it run?)
        execution_result = await run_code(code)

        # Level 4: Output correctness (does it produce right result?)
        output_correct = await compare_output(
            execution_result.output,
            expected_behavior
        )

        # Calculate graded score
        score_components = {
            "syntax": 0.25 if syntax_ok else 0.0,
            "static_quality": 0.25 * (1 - len(static_issues) / 10),
            "executes": 0.25 if execution_result.success else 0.0,
            "output_correct": 0.25 if output_correct else 0.0
        }

        return GradedResult(
            total_score=sum(score_components.values()),
            components=score_components,
            category=self._categorize(score_components)
        )

    def _categorize(self, components):
        """Categorize failure type."""
        if components["syntax"] == 0:
            return "syntax_error"
        elif components["static_quality"] < 0.2:
            return "quality_issues"
        elif components["executes"] == 0:
            return "runtime_error"
        elif components["output_correct"] == 0:
            return "wrong_output"
        else:
            return "perfect"
```

**Benefits:**
- More informative feedback ("runtime error" vs "syntax error")
- Partial credit for partially correct examples
- Better ROI analysis (fixing syntax errors is cheap, fixing logic is expensive)

---

**Mistake #7: No False Positive Tracking**

Current system:
- Reports 50 issues in a document
- **No validation** if these are real issues vs noise

**Problem:**
- High false positive rate → developers ignore system (boy who cried wolf)
- Wastes developer time reviewing non-issues

**Research Fix: False Positive Rate Measurement**

```python
class FalsePositiveTracker:
    """Track which reported issues are actually false positives."""

    async def measure_fpr(self, docs: List[Document]) -> FPRReport:
        # 1. Sample reported issues
        sample_issues = []
        for doc in docs:
            issues = doc.clarity_issues[:5]  # Top 5 issues per doc
            sample_issues.extend(issues)

        # 2. Get expert review
        expert_labels = await get_expert_review(
            sample_issues,
            question="Is this a real issue that affects user experience?"
        )

        # 3. Calculate FPR
        total = len(sample_issues)
        false_positives = sum(1 for label in expert_labels if label == "not_real")

        fpr = false_positives / total

        # 4. Analyze patterns
        fp_patterns = self.analyze_fp_patterns(
            [issue for issue, label in zip(sample_issues, expert_labels)
             if label == "not_real"]
        )

        return FPRReport(
            false_positive_rate=fpr,
            sample_size=total,
            common_fp_patterns=fp_patterns,
            recommendation="Retrain clarity agent" if fpr > 0.3 else "OK"
        )
```

---

### 2.4 Context & Scale Limitations

**Mistake #8: Long Document Truncation**

Current system:
- Sends full document to Claude
- Claude has 200K token context window
- No explicit handling of very long documents

**Problem:**
- Long documents (>100K tokens) may lose context
- Important information at end may be de-prioritized (recency bias)

**Research Fix: Hierarchical Evaluation**

```python
class HierarchicalEvaluator:
    """Evaluate long documents in structured hierarchy."""

    async def evaluate_long_doc(self, doc: Document) -> HierarchicalResult:
        # 1. Split into logical sections
        sections = self.split_by_sections(doc)  # Use headers as boundaries

        # 2. Evaluate each section independently
        section_results = await asyncio.gather(*[
            self.evaluate_section(section) for section in sections
        ])

        # 3. Evaluate document structure (section flow)
        structure_result = await self.evaluate_structure(
            sections,
            section_results
        )

        # 4. Synthesize final result
        return HierarchicalResult(
            sections=section_results,
            structure=structure_result,
            overall_score=self.aggregate_scores(section_results, structure_result)
        )

    def aggregate_scores(self, section_results, structure_result):
        """Weight sections by importance and length."""
        weighted_scores = []
        for section, result in zip(sections, section_results):
            weight = section.importance * (len(section.text) / total_length)
            weighted_scores.append(result.score * weight)

        return sum(weighted_scores) + structure_result.score * 0.2
```

---

**Mistake #9: No Temporal Tracking**

Current system:
- Evaluates documents at a single point in time
- **No tracking** of quality changes over time

**Missing insights:**
- Is documentation quality improving or degrading?
- Which sections regress most often?
- What's the velocity of documentation debt accumulation?

**Research Fix: Temporal Quality Dashboard**

```python
class TemporalQualityTracker:
    """Track documentation quality changes over time."""

    async def track_over_time(self, repo: str, branch: str) -> TemporalReport:
        # 1. Get all commits with doc changes
        commits = await get_doc_commits(repo, branch, since="2024-01-01")

        # 2. Evaluate each commit version
        evaluations = []
        for commit in commits:
            docs = await checkout_and_extract_docs(repo, commit.sha)
            results = await self.evaluate_all(docs)
            evaluations.append({
                "commit": commit.sha,
                "date": commit.date,
                "author": commit.author,
                "avg_score": np.mean([r.score for r in results]),
                "results": results
            })

        # 3. Analyze trends
        return TemporalReport(
            evaluations=evaluations,
            trend=self.calculate_trend(evaluations),
            regression_points=self.find_regressions(evaluations),
            velocity=self.calculate_velocity(evaluations)
        )

    def find_regressions(self, evaluations):
        """Identify commits that significantly decreased quality."""
        regressions = []
        for i in range(1, len(evaluations)):
            score_change = evaluations[i]["avg_score"] - evaluations[i-1]["avg_score"]
            if score_change < -1.0:  # Significant drop
                regressions.append({
                    "commit": evaluations[i]["commit"],
                    "score_drop": score_change,
                    "date": evaluations[i]["date"]
                })
        return regressions
```

---

## Part 3: LLM-Specific Evaluation Issues

### 3.1 Prompt Engineering Brittleness

**Issue:** Current system relies on carefully crafted prompts but doesn't validate prompt robustness.

**Example vulnerability** (`clarity_agent.py:96`):
```python
CLARITY_SYSTEM_PROMPT = """You are an expert documentation quality analyst...

SEVERITY LEVELS:
- critical: Issue blocks user progress entirely
- warning: Issue causes confusion but is workaroundable
- info: Nice-to-have improvement
"""
```

**Problem:**
- What if model interprets "blocks progress" differently on different documents?
- Small prompt changes may drastically change results (prompt brittleness)

**Research Fix: Prompt Robustness Testing**

```python
class PromptRobustnessTest:
    """Test if evaluation is stable across prompt variations."""

    async def test_robustness(self, doc: Document) -> RobustnessReport:
        # Create prompt variations
        prompts = [
            ORIGINAL_PROMPT,
            self.rephrase_prompt(ORIGINAL_PROMPT),
            self.reorder_sections(ORIGINAL_PROMPT),
            self.add_synonyms(ORIGINAL_PROMPT)
        ]

        # Evaluate same doc with all variations
        results = await asyncio.gather(*[
            self.evaluate(doc, prompt) for prompt in prompts
        ])

        # Measure consistency
        score_variance = np.var([r.score for r in results])
        issue_overlap = self.calculate_jaccard_similarity(
            [set(r.issues) for r in results]
        )

        return RobustnessReport(
            score_variance=score_variance,
            issue_overlap=issue_overlap,
            robust=score_variance < 0.5 and issue_overlap > 0.8
        )
```

**If not robust:**
- Ensemble across prompt variations
- Use few-shot examples for consistency
- Fine-tune model on labeled data

---

### 3.2 Position & Ordering Biases

**Position Bias:** LLMs pay more attention to information at start and end of context (primacy and recency effects).

**Ordering Bias:** Order of examples in prompt affects evaluation.

**Test:**
```python
# Test 1: Swap section order in document
doc_original = "Section A\nSection B\nSection C"
doc_reordered = "Section C\nSection B\nSection A"

score_original = await evaluate(doc_original)
score_reordered = await evaluate(doc_reordered)

# If scores differ significantly → position bias!
assert abs(score_original - score_reordered) < 0.5
```

**Mitigation:**
1. **Evaluate sections independently** then aggregate
2. **Randomize example order** in few-shot prompts
3. **Use attention mechanisms** to weight all sections equally

---

### 3.3 Style & Length Biases

**Verbosity Bias:** Longer documentation may score higher just for being detailed, even if unclear.

**Style Bias:** Model may prefer certain writing styles (formal vs casual, active vs passive voice).

**Test:**
```python
# Test: Add fluff without adding information
doc_original = "Install the package: pip install lancedb"
doc_verbose = """
To install the package, you should run the pip install command
as follows. This will download and install lancedb from PyPI.
The installation process typically takes 30-60 seconds depending
on your internet connection...
"""

# If verbose version scores higher → verbosity bias!
```

**Mitigation:**
1. **Normalize scores by document length**
2. **Penalize unnecessary verbosity** in scoring rubric
3. **Evaluate information density** as separate metric

---

## Part 4: Reinforcement Learning Applications

RL can address many evaluation gaps by learning from outcomes and optimizing actions.

### 4.1 Documentation Auto-Repair Agent (Deep RL)

**Problem:** Current system identifies issues but doesn't fix them.

**RL Solution:** Train an agent to automatically propose documentation fixes.

#### Architecture

```python
class DocRepairAgent:
    """RL agent that learns to fix documentation issues."""

    def __init__(self):
        self.policy = PolicyNetwork()  # Neural network: state → action
        self.value = ValueNetwork()    # Estimate expected reward

    # STATE
    def get_state(self, doc: Document, issue: Issue) -> State:
        """Extract features representing current documentation state."""
        return State(
            doc_text=doc.text,
            issue_type=issue.type,
            issue_severity=issue.severity,
            issue_location=issue.line,
            surrounding_context=doc.get_context(issue.line, window=5),
            validation_results={
                "api_valid": doc.api_validation_passed,
                "code_valid": doc.code_validation_passed,
                "current_clarity_score": doc.clarity_score
            }
        )

    # ACTION SPACE
    def define_actions(self):
        """Possible editing actions."""
        return [
            "add_paragraph",      # Add explanatory text
            "reorder_steps",      # Change step sequence
            "add_code_example",   # Add missing example
            "fix_code_snippet",   # Correct syntax error
            "add_prerequisite",   # Add missing requirement
            "clarify_instruction",# Reword unclear step
            "add_cross_reference",# Link to related section
            "no_action"           # Issue is false positive
        ]

    # REWARD FUNCTION
    def calculate_reward(self,
                         before: Document,
                         after: Document,
                         action: Action,
                         user_feedback: Optional[UserFeedback]) -> float:
        """Reward = improvement in documentation quality."""

        # Component 1: Validation improvements
        api_improvement = (after.api_valid - before.api_valid) * 0.8
        code_improvement = (after.code_valid - before.code_valid) * 1.0

        # Component 2: Clarity improvement
        clarity_improvement = (after.clarity_score - before.clarity_score) * 0.5

        # Component 3: Issue resolution
        issue_resolved = 2.0 if action.target_issue in before.issues \
                                and action.target_issue not in after.issues \
                                else 0.0

        # Component 4: User feedback (if available)
        user_reward = 0.0
        if user_feedback:
            if user_feedback.accepted_fix:
                user_reward = 5.0  # Strong positive signal
            elif user_feedback.rejected_fix:
                user_reward = -2.0
            else:  # Ignored
                user_reward = -0.1

        # Component 5: Edit cost (prefer smaller edits)
        edit_cost = -0.1 * len(action.diff)

        total_reward = (api_improvement + code_improvement +
                       clarity_improvement + issue_resolved +
                       user_reward + edit_cost)

        return total_reward

    # TRAINING
    async def train(self, dataset: List[Document], epochs: int):
        """Train agent using PPO (Proximal Policy Optimization)."""

        for epoch in range(epochs):
            for doc in dataset:
                # 1. Get all issues in document
                issues = await self.evaluate(doc)

                # 2. For each issue, try to fix it
                for issue in issues:
                    # Get current state
                    state = self.get_state(doc, issue)

                    # Sample action from policy
                    action = self.policy.sample(state)

                    # Apply action to create modified document
                    doc_modified = self.apply_action(doc, action)

                    # Re-evaluate modified document
                    issues_after = await self.evaluate(doc_modified)

                    # Calculate reward
                    reward = self.calculate_reward(doc, doc_modified, action)

                    # Update policy using PPO
                    self.update_policy(state, action, reward)

                    # If improvement, keep the change
                    if reward > 0:
                        doc = doc_modified

    # INFERENCE
    async def suggest_fix(self, doc: Document, issue: Issue) -> DocFix:
        """Suggest a fix for a specific issue."""
        state = self.get_state(doc, issue)
        action = self.policy.predict(state)  # Use learned policy

        return DocFix(
            action_type=action.type,
            location=issue.line,
            suggested_text=action.generated_text,
            confidence=action.probability,
            expected_improvement=self.value.predict(state)
        )
```

#### Training Data Requirements

1. **Initial dataset** (Imitation Learning):
   - Collect 1000+ pairs of (bad doc, good doc) from git history
   - Train policy to mimic human edits
   - Example: Find commits with message "fix docs" and extract before/after

2. **Online learning** (RL):
   - Deploy to real developers
   - Track which suggested fixes are accepted/rejected
   - Use acceptance as reward signal

#### Expected Results

After training on 10K document-fix pairs:
- **Precision:** 70% of suggested fixes are accepted by developers
- **Impact:** Average clarity score improvement of +1.5 per fix
- **Coverage:** Can fix 60% of detected issues automatically

---

### 4.2 Adaptive Validation Strategy (Contextual Bandit)

**Problem:** Running all validation agents on all documents is expensive. Some documents need thorough validation, others don't.

**RL Solution:** Learn which validation strategy to use for each document.

#### Architecture

```python
class AdaptiveValidator:
    """Contextual bandit that learns optimal validation strategy."""

    def __init__(self):
        self.bandit = LinUCB(alpha=0.5)  # Upper Confidence Bound bandit

    # CONTEXT
    def extract_context(self, doc: Document) -> np.ndarray:
        """Features describing the document."""
        return np.array([
            len(doc.text) / 1000,              # Document length
            doc.num_code_blocks,               # Code density
            doc.num_api_references,            # API references
            doc.estimated_complexity,          # Computed metric
            doc.days_since_last_update,        # Staleness
            doc.author_experience_level,       # Author's track record
            doc.traffic_level,                 # Page views (high-traffic needs more validation)
        ])

    # ACTIONS (Validation Strategies)
    def define_strategies(self):
        """Different validation thoroughness levels."""
        return [
            Strategy("minimal",
                    agents=["extraction"],
                    cost=1,
                    time=10),

            Strategy("standard",
                    agents=["extraction", "api_validation"],
                    cost=3,
                    time=60),

            Strategy("thorough",
                    agents=["extraction", "api_validation", "code_validation"],
                    cost=8,
                    time=300),

            Strategy("comprehensive",
                    agents=["extraction", "api_validation", "code_validation", "clarity", "walkthrough"],
                    cost=20,
                    time=1200)
        ]

    # REWARD
    def calculate_reward(self, strategy: Strategy, doc: Document, outcome: ValidationOutcome) -> float:
        """Reward = validation quality - cost."""

        # Quality: Did we catch real issues?
        quality = outcome.issues_found / (outcome.issues_found + outcome.issues_missed)

        # Cost: Time + compute resources
        cost = strategy.cost

        # Utility: Weighted by document importance
        importance = doc.traffic_level * doc.update_frequency

        reward = importance * quality - 0.1 * cost
        return reward

    # SELECTION
    async def select_strategy(self, doc: Document) -> Strategy:
        """Choose validation strategy for this document."""
        context = self.extract_context(doc)

        # LinUCB: Balance exploration vs exploitation
        arm_idx = self.bandit.select_arm(context)
        strategy = self.strategies[arm_idx]

        return strategy

    # UPDATE
    async def update(self, doc: Document, strategy: Strategy, outcome: ValidationOutcome):
        """Update bandit based on outcome."""
        context = self.extract_context(doc)
        reward = self.calculate_reward(strategy, doc, outcome)

        arm_idx = self.strategies.index(strategy)
        self.bandit.update(arm_idx, context, reward)
```

#### Expected Results

After learning from 5K validations:
- **Cost reduction:** 40% fewer compute resources used
- **Quality maintained:** 95% of critical issues still caught
- **Smart allocation:** High-traffic docs get comprehensive validation, internal docs get minimal

---

### 4.3 Issue Prioritization (Learning to Rank with RL)

**Problem:** Current system reports 50 issues per document. Which should developers fix first?

**RL Solution:** Learn to rank issues by actual impact on user experience.

#### Architecture

```python
class IssuePrioritizer:
    """RL agent that learns to prioritize issues by impact."""

    def __init__(self):
        self.ranker = DQN()  # Deep Q-Network for ranking

    # STATE
    def get_state(self, issue: Issue, doc: Document) -> State:
        """Features for ranking."""
        return State(
            # Issue characteristics
            issue_type=issue.type,
            issue_severity=issue.severity,
            issue_location=issue.line,
            affected_section=doc.get_section(issue.line),

            # Document context
            doc_traffic=doc.page_views,
            doc_importance=doc.importance_score,
            section_traffic=doc.get_section_traffic(issue.line),

            # Historical data
            similar_issue_fix_rate=self.get_historical_fix_rate(issue.type),
            similar_issue_user_impact=self.get_historical_impact(issue.type)
        )

    # ACTION
    def rank_issues(self, issues: List[Issue], doc: Document) -> List[Issue]:
        """Rank issues by predicted impact."""

        # Get priority score for each issue
        issue_scores = []
        for issue in issues:
            state = self.get_state(issue, doc)
            priority_score = self.ranker.predict(state)
            issue_scores.append((issue, priority_score))

        # Sort by priority (descending)
        issue_scores.sort(key=lambda x: x[1], reverse=True)

        return [issue for issue, score in issue_scores]

    # REWARD
    def calculate_reward(self, issue: Issue, was_fixed: bool, user_impact: float) -> float:
        """Reward based on whether fixing this issue helped users."""

        if was_fixed:
            # Positive reward proportional to actual user impact
            # user_impact = change in task completion rate after fix
            return user_impact * 10.0
        else:
            # Small penalty for ranking unfixed issues high
            return -0.1

    # TRAINING
    async def train_from_history(self, historical_issues: List[IssueOutcome]):
        """Learn from past issue resolutions."""

        for outcome in historical_issues:
            # State: Issue + document context at time of detection
            state = self.get_state(outcome.issue, outcome.doc)

            # Reward: Did fixing help users?
            reward = self.calculate_reward(
                outcome.issue,
                outcome.was_fixed,
                outcome.user_impact
            )

            # Update ranker
            self.ranker.update(state, reward)
```

#### Training Data Sources

1. **GitHub issue tracker:**
   - Link documentation issues to GitHub issues
   - Track which issues get fixed (PR merged)
   - Track which issues get closed without fix (not important)

2. **User analytics:**
   - Before fix: Task completion rate = 60%
   - After fix: Task completion rate = 85%
   - Impact = +25% → High priority issue!

3. **Developer feedback:**
   - Which issues did they fix first?
   - Which did they ignore?

#### Expected Results

- **Top 10 issues** capture 80% of user-facing problems (Pareto principle)
- **Developer time saved:** Focus on high-impact issues only
- **User satisfaction:** Faster improvement in real pain points

---

### 4.4 Preference Learning (Inverse RL)

**Problem:** Hand-crafted penalty weights (-2.0 for critical, -0.5 for warning) are arbitrary.

**RL Solution:** Learn reward function from human preferences.

#### Architecture

```python
class PreferenceLearner:
    """Learn reward function from pairwise comparisons."""

    def __init__(self):
        self.reward_model = NeuralRewardModel()

    # COLLECT PREFERENCES
    async def collect_preferences(self, docs: List[Document]) -> List[Preference]:
        """Show pairs of documents to users, ask which is better."""

        preferences = []

        # Generate pairs
        for doc_a, doc_b in random_pairs(docs, n=100):
            # Show to user
            preference = await ask_user(
                question="Which documentation is clearer?",
                option_a=doc_a,
                option_b=doc_b,
                allow_neutral=True
            )

            preferences.append(Preference(
                doc_a=doc_a,
                doc_b=doc_b,
                preferred=preference  # "a", "b", or "neutral"
            ))

        return preferences

    # TRAIN REWARD MODEL
    def train_reward_model(self, preferences: List[Preference]):
        """Train model to predict which doc is better."""

        for pref in preferences:
            # Extract features
            features_a = self.extract_features(pref.doc_a)
            features_b = self.extract_features(pref.doc_b)

            # Compute rewards
            reward_a = self.reward_model(features_a)
            reward_b = self.reward_model(features_b)

            # Bradley-Terry loss: P(a > b) = sigmoid(reward_a - reward_b)
            if pref.preferred == "a":
                loss = -torch.log(torch.sigmoid(reward_a - reward_b))
            elif pref.preferred == "b":
                loss = -torch.log(torch.sigmoid(reward_b - reward_a))
            else:  # neutral
                loss = -torch.log(1 - torch.abs(torch.sigmoid(reward_a - reward_b) - 0.5))

            # Backprop
            loss.backward()
            optimizer.step()

    # USE LEARNED REWARDS
    def score_document(self, doc: Document) -> float:
        """Score document using learned reward model."""
        features = self.extract_features(doc)
        return self.reward_model(features).item()
```

#### Benefits

- **No arbitrary weights** - Learned from actual human preferences
- **Personalization** - Can learn different reward functions for different user groups (beginners vs experts)
- **Interpretability** - Can inspect learned reward model to understand what matters

---

## Part 5: Recommended Implementation Roadmap

### Phase 1: Foundations (Month 1-2)

**Goal:** Establish measurement infrastructure

1. **Create gold standard dataset**
   - Manually annotate 30 documents (10 excellent, 10 good, 10 poor)
   - Get 3 experts to evaluate each document
   - Measure inter-annotator agreement

2. **Implement uncertainty quantification**
   - Add confidence intervals to all scores
   - Monte Carlo dropout (10 samples per evaluation)

3. **Add false positive tracking**
   - Sample 100 reported issues
   - Get expert review
   - Calculate FPR and precision

**Success Metrics:**
- Inter-annotator agreement > 0.7 (Krippendorff's alpha)
- Confidence intervals < 1.0 for 80% of evaluations
- False positive rate < 20%

---

### Phase 2: Multi-Model Evaluation (Month 3-4)

**Goal:** Reduce single-model bias

1. **Implement ensemble evaluation**
   - Add GPT-4 and Gemini as judges
   - Measure inter-model agreement
   - Flag high-disagreement cases

2. **Prompt robustness testing**
   - Test 5 prompt variations per evaluation
   - Measure score variance
   - Select most robust prompt

3. **User outcome correlation study**
   - Run task completion study (20 participants, 10 documents)
   - Calculate correlation between predicted score and actual success
   - Validate scoring system

**Success Metrics:**
- Inter-model agreement > 0.6
- Score variance across prompts < 0.5
- Correlation with user outcomes > 0.7

---

### Phase 3: Feedback Loop (Month 5-6)

**Goal:** Learn from outcomes

1. **Implement active learning pipeline**
   - Queue uncertain cases for human review
   - Collect 500 human-labeled examples
   - Retrain/calibrate scoring

2. **Build issue outcome tracking**
   - Link issues to GitHub PRs
   - Track which issues get fixed
   - Measure user impact before/after fix

3. **Temporal quality dashboard**
   - Track quality over time (weekly snapshots)
   - Identify regressions
   - Alert on quality drops

**Success Metrics:**
- Active learning reduces uncertainty by 30%
- 500+ labeled examples collected
- 90% of quality regressions detected within 1 week

---

### Phase 4: RL Applications (Month 7-12)

**Goal:** Automated improvement

**Priority 1: Issue Prioritization (Month 7-8)**
- Implement learning-to-rank system
- Train on historical data (1000+ issues)
- Deploy as "Priority" field in issue reports

**Priority 2: Adaptive Validation (Month 9-10)**
- Implement contextual bandit
- Learn from 5000+ validation runs
- Reduce compute cost by 40%

**Priority 3: Auto-Repair Agent (Month 11-12)**
- Collect imitation learning dataset (1000+ doc fixes)
- Train initial policy
- Deploy in "suggestion mode" (human reviews all fixes)

**Success Metrics:**
- Issue prioritization: 80% of top-10 issues are fixed
- Adaptive validation: 40% cost reduction, <5% quality drop
- Auto-repair: 50% of suggestions accepted

---

## Part 6: Research Contributions

Implementing these improvements would enable **novel research contributions**:

### Contribution 1: Benchmarking LLM-as-Judge for Documentation

**Paper:** "Evaluating LLM Reliability for Documentation Quality Assessment"

**Contributions:**
- First systematic study of LLM-as-judge for technical documentation
- Gold standard dataset of 1000+ annotated documents
- Comparison of Claude, GPT-4, Gemini on documentation evaluation
- Analysis of failure modes (biases, inconsistencies)

**Impact:** Establish best practices for LLM-based documentation evaluation

---

### Contribution 2: Reinforcement Learning for Documentation Repair

**Paper:** "Learning to Fix Documentation: An RL Approach to Automated Doc Improvement"

**Contributions:**
- First RL agent for documentation repair
- Novel reward function combining validation + user feedback
- Imitation learning from git history
- Evaluation on real-world open-source projects

**Impact:** Demonstrate feasibility of AI-assisted documentation maintenance

---

### Contribution 3: Outcome-Based Evaluation Validation

**Paper:** "Do Documentation Quality Metrics Predict User Success? An Empirical Study"

**Contributions:**
- User study with 200+ developers across 50+ documents
- Correlation analysis between predicted quality and actual outcomes
- Identification of which metrics matter most for user success
- Recommendations for documentation evaluation systems

**Impact:** Ground documentation evaluation in user-centered metrics

---

## Conclusion

Stackbench is an innovative system with **significant evaluation gaps**. The core issues:

1. **No ground truth** - Evaluations aren't validated against human judgment or user outcomes
2. **No learning** - System doesn't improve from feedback
3. **Single model** - Vulnerable to Claude-specific biases
4. **No uncertainty** - Point estimates without confidence
5. **Binary validation** - Missing nuanced correctness assessment

**Reinforcement Learning opportunities:**

| Problem | RL Solution | Expected Impact |
|---------|-------------|-----------------|
| Manual issue fixing | Auto-repair agent | 50% reduction in manual fixes |
| Expensive validation | Adaptive strategy selection | 40% cost reduction |
| Issue overload | Learning-to-rank prioritization | 80% of impact in top 10 issues |
| Arbitrary weights | Preference learning | Alignment with human values |

**Recommended path forward:**

1. **Short term (3 months):** Add ground truth, uncertainty, multi-model evaluation
2. **Medium term (6 months):** Build feedback loop, outcome tracking
3. **Long term (12 months):** Deploy RL for prioritization → adaptive validation → auto-repair

This roadmap would transform Stackbench from a tool into a **self-improving system** that learns what good documentation means through user outcomes.

---

**Next Steps:**

1. Review this analysis with team
2. Prioritize which improvements are highest ROI
3. Start with Phase 1 (gold standard dataset + uncertainty)
4. Publish interim research findings
5. Iterate based on results

The intersection of LLMs, RL, and documentation quality is a rich research area with real-world impact. Let's build something scientifically rigorous and practically useful.
