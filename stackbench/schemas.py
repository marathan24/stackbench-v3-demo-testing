"""
Centralized Pydantic schemas for all StackBench agents.

This module is the single source of truth for all data models used across
extraction, validation, and analysis agents. All schemas are defined here
to avoid duplication and ensure consistency.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


# ============================================================================
# EXTRACTION SCHEMAS
# ============================================================================

class SnippetSource(BaseModel):
    """Source information for snippet includes (--8<-- directives)."""
    file: str = Field(description="Source file path, e.g., 'python/python/test_file.py'")
    tags: List[str] = Field(default_factory=list, description="Snippet tags/labels, e.g., ['connect_to_lancedb']")


class APISignature(BaseModel):
    """Represents an API signature found in documentation."""
    library: str = Field(description="Library/package name")
    function: str = Field(description="Function/class/method name")
    method_chain: Optional[str] = Field(None, description="Chained method calls if applicable")
    params: List[str] = Field(default_factory=list, description="Parameter names")
    param_types: Dict[str, str] = Field(default_factory=dict, description="Parameter types")
    defaults: Dict[str, Any] = Field(default_factory=dict, description="Default values")
    imports: Optional[str] = Field(None, description="Import statement needed")
    line: int = Field(description="Approximate line number in document")
    context: str = Field(description="Section/heading this appears under")
    raw_code: Optional[str] = Field(None, description="Exact code snippet showing the signature")

    # Location metadata for better association
    section_hierarchy: List[str] = Field(default_factory=list, description="Hierarchical section path, e.g., ['Create & Query', 'From Polars DataFrame', 'Sync API']")
    markdown_anchor: Optional[str] = Field(None, description="Markdown heading anchor/ID, e.g., '#from-polars-dataframe'")
    code_block_index: int = Field(default=0, description="Index of code block within the section (0, 1, 2...)")


class CodeExample(BaseModel):
    """Represents a code example found in documentation."""
    library: str = Field(description="Primary library being demonstrated")
    language: str = Field(description="Programming language")
    code: str = Field(description="Complete code example")
    imports: Optional[str] = Field(None, description="All import statements")
    has_main: bool = Field(description="Whether example has a main/entry point")
    is_executable: bool = Field(description="Whether example can run standalone")
    line: int = Field(description="Approximate line number in document")
    context: str = Field(description="Section/heading this appears under")
    dependencies: List[str] = Field(default_factory=list, description="External dependencies needed")

    # Location metadata for better association
    section_hierarchy: List[str] = Field(default_factory=list, description="Hierarchical section path, e.g., ['Create & Query', 'From Polars DataFrame', 'Sync API']")
    markdown_anchor: Optional[str] = Field(None, description="Markdown heading anchor/ID, e.g., '#from-polars-dataframe'")
    code_block_index: int = Field(default=0, description="Index of code block within the section (0, 1, 2...)")
    snippet_source: Optional[SnippetSource] = Field(None, description="If from snippet include (--8<--), the source file and tags")


class ExtractionResult(BaseModel):
    """Result of extracting information from documentation."""
    library: str = Field(description="Primary library/framework name")
    version: Optional[str] = Field(None, description="Library version if mentioned")
    language: str = Field(description="Programming language")
    signatures: List[APISignature] = Field(default_factory=list, description="All API signatures found")
    examples: List[CodeExample] = Field(default_factory=list, description="All code examples found")


class DocumentAnalysis(BaseModel):
    """Complete analysis of a documentation file."""
    page: str = Field(description="Filename of the documentation page")
    library: str = Field(description="Primary library name")
    version: Optional[str] = Field(None, description="Library version")
    language: str = Field(description="Programming language")
    signatures: List[APISignature] = Field(default_factory=list, description="API signatures")
    examples: List[CodeExample] = Field(default_factory=list, description="Code examples")
    processed_at: str = Field(description="ISO timestamp of processing")
    total_signatures: int = Field(description="Count of signatures found")
    total_examples: int = Field(description="Count of examples found")
    warnings: List[str] = Field(default_factory=list, description="Any warnings or issues")
    processing_time_ms: Optional[int] = Field(None, description="Time taken to process")


class ExtractionSummary(BaseModel):
    """Summary of all extraction results."""
    total_documents: int = Field(description="Total markdown files found")
    processed: int = Field(description="Successfully processed documents")
    total_signatures: int = Field(description="Total signatures across all docs")
    total_examples: int = Field(description="Total examples across all docs")
    timestamp: str = Field(description="ISO timestamp of summary generation")
    extraction_duration_seconds: Optional[float] = Field(None, description="Total time taken for extraction in seconds")
    num_workers: Optional[int] = Field(None, description="Number of parallel workers used")
    documents: List[DocumentAnalysis] = Field(default_factory=list, description="All document analyses")


# ============================================================================
# CODE VALIDATION SCHEMAS
# ============================================================================

class ExampleValidationResult(BaseModel):
    """Validation result for a single code example."""
    example_index: int
    line: int
    context: str
    code: str
    status: str = Field(description="success|failure|skipped")
    severity: Optional[str] = Field(None, description="error|warning|info - Classification of issue severity. 'error' = clear doc mistake, 'warning' = environment/compatibility issue, 'info' = non-blocking (deprecations, etc). Only set when status is 'failure'")
    error_message: Optional[str] = None
    suggestions: Optional[str] = None
    execution_output: Optional[str] = None
    depends_on_previous: bool = False

    # Dependency tracking for better association
    depends_on_example_indices: List[int] = Field(default_factory=list, description="Specific example indices this depends on, e.g., [0, 2]")
    actual_code_executed: Optional[str] = Field(None, description="Full code that was executed, including merged dependencies")


class DocumentValidationResult(BaseModel):
    """Validation result for an entire document."""
    page: str
    library: str
    version: str
    language: str
    validation_timestamp: str
    results: List[ExampleValidationResult]
    total_examples: int
    successful: int
    failed: int
    skipped: int


# ============================================================================
# API SIGNATURE VALIDATION SCHEMAS
# ============================================================================

class DocumentedSignature(BaseModel):
    """Signature as documented."""
    params: List[str]
    param_types: Dict[str, str]
    defaults: Dict[str, Any]
    imports: str
    raw_code: str
    line: int
    context: str


class ActualSignature(BaseModel):
    """Actual signature from code introspection."""
    params: List[str]
    param_types: Dict[str, str]
    defaults: Dict[str, Any]
    required_params: List[str]
    optional_params: List[str]
    return_type: Optional[str] = None
    is_async: bool
    is_method: bool
    verified_by: str


class ValidationIssue(BaseModel):
    """A validation issue found."""
    type: str
    severity: str  # 'critical' | 'warning' | 'info'
    message: str
    suggested_fix: Optional[str] = None


class SignatureValidation(BaseModel):
    """Validation result for a single signature."""
    signature_id: str
    function: str
    method_chain: Optional[str] = None
    library: str
    status: str  # 'valid' | 'invalid' | 'not_found' | 'error'
    documented: DocumentedSignature
    actual: Optional[ActualSignature] = None
    issues: List[ValidationIssue]
    confidence: float


class ValidationSummary(BaseModel):
    """Summary of validation results."""
    total_signatures: int
    valid: int
    invalid: int
    not_found: int
    error: int
    accuracy_score: float
    critical_issues: int
    warnings: int


class EnvironmentInfo(BaseModel):
    """Information about the validation environment."""
    library_installed: str
    version_installed: str
    version_requested: str
    version_match: bool
    python_version: str
    installation_output: Optional[str] = None


class APISignatureValidationOutput(BaseModel):
    """Complete API signature validation output."""
    validation_id: str
    validated_at: str
    source_file: str
    document_page: str
    library: str
    version: str
    language: str
    summary: ValidationSummary
    validations: List[SignatureValidation]
    environment: EnvironmentInfo
    processing_time_ms: int
    warnings: List[str]


# ============================================================================
# CLARITY VALIDATION SCHEMAS
# ============================================================================

class ClarityIssue(BaseModel):
    """A clarity or UX issue found in documentation."""
    type: str  # missing_prerequisite, logical_gap, unclear_explanation, etc.
    severity: str  # 'critical' | 'warning' | 'info'
    line: int
    section: str
    step_number: Optional[int] = None
    message: str
    suggested_fix: Optional[str] = None
    affected_code: Optional[str] = None
    context_quote: Optional[str] = None


class StructuralIssue(BaseModel):
    """A structural issue in documentation."""
    type: str  # buried_prerequisites, missing_step_numbers, etc.
    severity: str  # 'critical' | 'warning' | 'info'
    location: str
    message: str
    suggested_fix: Optional[str] = None


class ClarityScore(BaseModel):
    """Clarity scoring metrics."""
    overall_score: float  # 0-10
    tier: str  # S/A/B/C/D/F
    instruction_clarity: float
    logical_flow: float
    completeness: float
    consistency: float
    prerequisite_coverage: float


class BrokenLink(BaseModel):
    """A broken link found in documentation."""
    url: str
    line: int
    link_text: str
    error: str


class MissingAltText(BaseModel):
    """An image missing alt text."""
    image_path: str
    line: int


class CodeBlockIssue(BaseModel):
    """A code block without language specification."""
    line: int
    content_preview: str


class TechnicalAccessibility(BaseModel):
    """Technical accessibility validation results."""
    broken_links: List[BrokenLink]
    missing_alt_text: List[MissingAltText]
    code_blocks_without_language: List[CodeBlockIssue]
    total_links_checked: int
    total_images_checked: int
    total_code_blocks_checked: int
    all_validated: bool


class ClaritySummary(BaseModel):
    """Summary of clarity validation."""
    total_clarity_issues: int
    critical_clarity_issues: int
    warning_clarity_issues: int
    info_clarity_issues: int
    total_structural_issues: int
    critical_structural_issues: int
    total_technical_issues: int
    overall_quality_rating: str  # 'excellent' | 'good' | 'needs_improvement' | 'poor'


class PrioritizedFix(BaseModel):
    """A single improvement action from the roadmap."""
    priority: str  # 'critical' | 'high' | 'medium' | 'low'
    category: str
    description: str
    location: str
    impact: str  # 'high' | 'medium' | 'low'
    effort: str  # 'low' | 'medium' | 'high'
    projected_score_change: float


class ImprovementRoadmap(BaseModel):
    """Prioritized list of improvements with projections."""
    current_overall_score: float
    projected_score_after_critical_fixes: float
    projected_score_after_all_fixes: float
    prioritized_fixes: List[PrioritizedFix]
    quick_wins: List[PrioritizedFix]  # High impact + low effort


class ScoreBreakdown(BaseModel):
    """Detailed score calculation breakdown."""
    base_score: float
    critical_issues_penalty: float
    warning_issues_penalty: float
    info_issues_penalty: float
    failed_examples_penalty: float
    invalid_api_penalty: float
    missing_api_penalty: float
    final_score: float


class TierRequirements(BaseModel):
    """Requirements to reach next tier."""
    current_tier: str
    next_tier: Optional[str]
    requirements_for_next_tier: Optional[Dict[str, Any]]
    current_status: Dict[str, int]


class PrimaryIssue(BaseModel):
    """Summary of issues by category."""
    category: str
    critical: int
    warning: int
    info: int
    example: str


class ScoreExplanation(BaseModel):
    """Human-readable score explanation."""
    score: float
    tier: str
    tier_description: str
    score_breakdown: ScoreBreakdown
    tier_requirements: TierRequirements
    primary_issues: List[PrimaryIssue]
    summary: str


class ClarityValidationOutput(BaseModel):
    """Complete clarity validation output."""
    validation_id: str
    validated_at: str
    source_file: str
    document_page: str
    library: str
    version: str
    language: str
    clarity_score: ClarityScore
    clarity_issues: List[ClarityIssue]
    structural_issues: List[StructuralIssue]
    technical_accessibility: TechnicalAccessibility
    improvement_roadmap: ImprovementRoadmap
    score_explanation: ScoreExplanation
    summary: ClaritySummary
    processing_time_ms: int
    warnings: List[str]


# ============================================================================
# RUN LOGGING SCHEMAS
# ============================================================================


class RunTotals(BaseModel):
    """Aggregated counters for a pipeline run."""
    tokens_input: int = Field(0, description="Total tokens supplied to language models")
    tokens_output: int = Field(0, description="Total tokens produced by language models")
    cost: float = Field(0.0, description="Aggregate model/API cost in USD")
    latency_ms: int = Field(0, description="Cumulative latency attributed to model/tool events in milliseconds")
    tool_calls: int = Field(0, description="Number of tool calls executed during the run")
    errors: int = Field(0, description="Total errors recorded during the run")


class RunEvent(BaseModel):
    """A single event in the pipeline run event log."""
    event_id: str = Field(description="Unique identifier for the event")
    run_id: str = Field(description="Run identifier this event belongs to")
    timestamp: str = Field(description="ISO 8601 timestamp when the event occurred")
    type: Literal["info", "step_start", "step_end", "prompt", "tool", "error"] = Field(
        description="Categorisation of the event type"
    )
    step_id: Optional[str] = Field(None, description="Associated step identifier, if any")
    name: Optional[str] = Field(None, description="Human readable name for the event")
    role: Optional[Literal["user", "assistant"]] = Field(
        None,
        description="For prompt events, whether the content is from the user or assistant"
    )
    prompt: Optional[str] = Field(None, description="Prompt text for model invocations")
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured variables interpolated into the prompt"
    )
    model: Optional[str] = Field(None, description="Model identifier used for the event")
    tokens_input: Optional[int] = Field(None, description="Tokens supplied to the model")
    tokens_output: Optional[int] = Field(None, description="Tokens generated by the model")
    cost: Optional[float] = Field(None, description="API cost associated with the event in USD")
    latency_ms: Optional[int] = Field(None, description="Latency for the event in milliseconds")
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tool invocation metadata recorded for this event"
    )
    error: Optional[str] = Field(None, description="Error message when the event represents a failure")
    output: Optional[Any] = Field(None, description="Model output or tool payload for the event")
    cache_key: Optional[str] = Field(None, description="Deterministic key derived from prompt + variables")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual metadata")


class RunStep(BaseModel):
    """A logical unit of work executed within a run."""
    step_id: str = Field(description="Unique identifier for the step")
    name: str = Field(description="Human readable name or label for the step")
    stage: str = Field(description="Pipeline stage this step belongs to (extraction, validation, etc.)")
    document: Optional[str] = Field(None, description="Document associated with the step, if applicable")
    status: Literal["running", "success", "failed", "skipped"] = Field(
        description="Current execution state of the step"
    )
    started_at: str = Field(description="ISO timestamp indicating when the step started")
    completed_at: Optional[str] = Field(None, description="ISO timestamp indicating when the step finished")
    latency_ms: Optional[int] = Field(None, description="Duration of the step in milliseconds")
    tokens_input: int = Field(0, description="Total tokens sent to models during this step")
    tokens_output: int = Field(0, description="Total tokens produced during this step")
    cost: float = Field(0.0, description="Aggregate model/API cost attributed to the step in USD")
    tool_calls: int = Field(0, description="Number of tool calls executed during this step")
    errors: int = Field(0, description="Number of errors recorded during this step")
    error: Optional[str] = Field(None, description="Primary error message if the step failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual metadata about the step")


class RunRecord(BaseModel):
    """Summary record for a single Stackbench pipeline run."""
    run_id: str = Field(description="Unique identifier for the run")
    status: Literal["initializing", "running", "completed", "failed", "cached", "in_progress"] = Field(
        description="Overall status of the run"
    )
    repo_url: Optional[str] = Field(None, description="Repository URL that was analysed")
    branch: Optional[str] = Field(None, description="Repository branch used for the run")
    docs_path: Optional[str] = Field(None, description="Documentation path used for discovery")
    library: Optional[str] = Field(None, description="Primary library validated during the run")
    library_version: Optional[str] = Field(None, description="Library version targeted during validation")
    created_at: str = Field(description="ISO timestamp when the run started")
    completed_at: Optional[str] = Field(None, description="ISO timestamp when the run completed")
    duration_ms: Optional[int] = Field(None, description="Total runtime duration in milliseconds")
    total_steps: int = Field(0, description="Number of steps executed in the run")
    total_events: int = Field(0, description="Number of events recorded for the run")
    totals: RunTotals = Field(default_factory=RunTotals, description="Aggregate counters for the run")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata about the run")
    steps: List[RunStep] = Field(default_factory=list, description="Ordered steps executed as part of the run")
