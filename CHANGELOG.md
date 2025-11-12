# Changelog

All notable changes to Stackbench will be documented in this file.

## [Unreleased]

### Added - Evaluation Module (Phase 1 - Foundations)

**Major Feature:** Research-grade evaluation improvements addressing 6 critical evaluation mistakes.

#### Uncertainty Quantification
- Added `stackbench/evaluation/uncertainty.py` for Monte Carlo confidence intervals
- Added confidence interval fields to `ClarityScore` schema
- Provides 95% confidence intervals for all quality scores
- Identifies high-uncertainty results for human review
- Fixes: No uncertainty quantification (Mistake #3)

#### Validation & Metrics
- Added `stackbench/evaluation/metrics.py` for precision/recall/FPR tracking
- Added `stackbench/evaluation/ground_truth.py` for gold standard dataset management
- Added `stackbench/evaluation/validation.py` for LLM-to-human comparison
- Calculate precision, recall, F1, false positive rate
- Track inter-annotator agreement
- Correlate predictions with ground truth
- Fixes: No calibration against human judgment (Mistake #1), No FPR tracking (Mistake #7)

#### Multi-Model Ensemble
- Added `stackbench/evaluation/ensemble.py` for ensemble evaluation
- Evaluate using multiple LLMs (Claude, GPT-4, Gemini)
- Measure inter-model agreement
- Flag high-disagreement cases for review
- Analyze systematic biases across models
- Fixes: Single model bias (Mistake #2)

#### Graded Code Validation
- Added `stackbench/evaluation/graded_validation.py` for partial credit scoring
- 4-level validation: syntax (25%), quality (25%), execution (25%), output (25%)
- Replace binary pass/fail with continuous 0-1 scoring
- Categorize failures: syntax_error, runtime_error, wrong_output
- Fixes: Binary validation (Mistake #6)

#### Temporal Quality Tracking
- Added `stackbench/evaluation/temporal.py` for quality tracking over time
- Track documentation quality across git history
- Calculate trends: improving/declining/stable
- Identify quality regressions (commits with score drops)
- Measure improvement velocity (change per week)
- Fixes: No temporal tracking (Mistake #9)

#### Advanced Features
- Added `stackbench/evaluation/agreement.py` for Cohen's kappa, Krippendorff's alpha, Fleiss' kappa
- Added `stackbench/evaluation/prompt_robustness.py` for prompt stability testing
- Added `stackbench/evaluation/outcomes.py` for user outcome correlation
- Added `stackbench/evaluation/cli_utils.py` for rich terminal display

#### Documentation
- Added `EVALUATION_ANALYSIS.md` (1,300+ lines) - Full analysis of mistakes + RL applications
- Added `stackbench/evaluation/README.md` (500+ lines) - Module documentation
- Added `IMPLEMENTATION_SUMMARY.md` (444 lines) - Implementation status
- Added `COMMITS_LOG.md` (229 lines) - Complete commit history

#### Tests
- Added `tests/evaluation/test_uncertainty.py` (11 test cases)
- Added `tests/evaluation/test_metrics.py` (11 test cases)

### Changed
- Updated `stackbench/schemas.py` - Added confidence interval fields to `ClarityScore`
- Updated `stackbench/evaluation/__init__.py` - Export all evaluation utilities

### Statistics
- **10 new modules** (~3,000 lines of code)
- **4 documentation files** (~2,500 lines)
- **22 test cases**
- **19 commits** (as of 2025-11-05)
- **6 out of 9 critical mistakes fixed**

### Impact
Transforms Stackbench from a basic tool into a research-grade evaluation system with:
- Scientific rigor (uncertainty quantification, confidence intervals)
- Validation framework (ground truth, precision/recall, correlation)
- Bias reduction (multi-model ensembles, prompt robustness)
- Quality tracking (temporal analysis, regression detection)
- User-centered validation (outcome correlation)

---

## [Previous Releases]

(To be added from previous commits/releases)

---

## References

- See `EVALUATION_ANALYSIS.md` for full analysis
- See `IMPLEMENTATION_SUMMARY.md` for implementation details
- See `stackbench/evaluation/README.md` for usage guide
