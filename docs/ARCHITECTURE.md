# Stackbench Architecture

**Version:** 0.1.0  
**Last Updated:** 2024  
**Python:** 3.11+ | **Frontend:** React 19 + TypeScript + Vite

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Key Modules](#key-modules)
6. [Extension Points](#extension-points)
7. [Caching System](#caching-system)
8. [Hook System](#hook-system)
9. [Walkthrough System](#walkthrough-system)
10. [Frontend Architecture](#frontend-architecture)

---

## Overview

Stackbench is an AI-driven documentation quality validation tool that uses Claude Code agents to systematically validate documentation through:

- **Static Analysis**: API signature extraction and validation
- **Dynamic Testing**: Code example execution in isolated environments
- **AI Reasoning**: LLM-as-judge clarity scoring
- **Experiential Validation**: Step-by-step tutorial execution (walkthroughs)

### Design Principles

1. **Agent Specialization**: Each agent has a focused responsibility
2. **Parallel Processing**: Worker pools for extraction and clarity validation
3. **Data Quality by Design**: Pydantic schemas + validation hooks
4. **Cache-First Architecture**: Avoid redundant analysis
5. **Comprehensive Audit Trail**: Every operation logged

---

## System Architecture

### Dependency Graphs

**Python Module Dependencies:**

![Python Dependencies](python_dependencies.svg)

**Frontend Component Dependencies:**

![Frontend Dependencies](frontend_dependencies.svg)

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI LAYER (Typer)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ stackbench   │  │ stackbench   │  │ stackbench          │   │
│  │ run          │  │ rerun-clarity│  │ walkthrough (suite) │   │
│  └──────────────┘  └──────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  DocumentationValidationPipeline                         │   │
│  │  • Repository cloning (RepositoryManager)                │   │
│  │  • Cache checking (CacheManager)                         │   │
│  │  • Worker pool orchestration (asyncio)                   │   │
│  │  • Results aggregation                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT LAYER (Claude Code)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Extraction   │→ │ API          │→ │ Code         │→         │
│  │ Agent        │  │ Validation   │  │ Validation   │          │
│  │              │  │ Agent        │  │ Agent        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Clarity      │  │ Walkthrough  │  │ Walkthrough  │          │
│  │ Validation   │  │ Generate     │  │ Audit        │          │
│  │ Agent        │  │ Agent        │  │ Agent        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ HookManager  │  │ MCP Server   │  │ CacheManager │          │
│  │ • Validation │  │ (Clarity &   │  │ (runs.json)  │          │
│  │ • Logging    │  │ Walkthrough) │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PERSISTENCE                            │
│  data/<run_id>/                                                  │
│  ├── repository/          # Cloned git repo                      │
│  ├── results/             # Validation outputs                   │
│  │   ├── extraction/                                             │
│  │   ├── api_validation/                                         │
│  │   ├── code_validation/                                        │
│  │   └── clarity_validation/                                     │
│  ├── walkthroughs/        # Tutorial validation                  │
│  │   └── wt_<id>/                                                │
│  │       ├── walkthrough.json                                    │
│  │       ├── audit.json                                          │
│  │       └── agent_logs/                                         │
│  └── validation_logs/     # Hook execution traces                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Four-Stage Validation Pipeline

Each document flows through **four sequential agents**:

#### Stage 1: Extraction Agent (`extraction_agent.py`)
- **Purpose**: Extract structured data from unstructured markdown
- **Input**: Raw markdown documentation files
- **Output**: `<doc>_analysis.json` with API signatures and code examples
- **Key Features**:
  - Parallel processing (default 5 workers)
  - Claude Code agent with Read/Write tools
  - Schema validation via PreToolUse hooks
  - Handles MkDocs Material snippet syntax

**Extracted Data Schema:**
```python
{
  "document_name": str,
  "document_path": str,
  "library_name": str,
  "api_signatures": [
    {
      "function_name": str,
      "parameters": [{"name": str, "type": str, "default": Any, "required": bool}],
      "return_type": str,
      "description": str,
      "location": {"section": str, "line_number": int}
    }
  ],
  "code_examples": [
    {
      "language": str,
      "code": str,
      "context": str,
      "location": {"section": str, "line_number": int}
    }
  ]
}
```

#### Stage 2: API Signature Validation Agent (`api_signature_validation_agent.py`)
- **Purpose**: Validate documented APIs against actual library implementation
- **Input**: Extraction JSON + library installation
- **Output**: `<doc>_validation.json` with pass/fail results
- **Key Features**:
  - Creates isolated Python environment
  - Uses `inspect.signature()` for ground truth
  - Detects missing params, wrong types, wrong defaults
  - Flags phantom APIs (documented but don't exist)

**Validation Output Schema:**
```python
{
  "document_name": str,
  "validation_results": [
    {
      "api_name": str,
      "status": "pass" | "fail" | "not_found",
      "issues": [
        {
          "type": "missing_parameter" | "wrong_type" | "wrong_default",
          "expected": str,
          "actual": str,
          "severity": "error" | "warning"
        }
      ]
    }
  ],
  "summary": {
    "total_apis": int,
    "passed": int,
    "failed": int,
    "not_found": int
  }
}
```

#### Stage 3: Code Example Validation Agent (`code_example_validation_agent.py`)
- **Purpose**: Execute code examples in isolated environments
- **Input**: Extraction JSON with code examples
- **Output**: `<doc>_validation.json` with execution results
- **Key Features**:
  - Creates fresh Python env per example
  - Captures stdout/stderr
  - Detects syntax errors, runtime errors, import failures
  - Configurable timeout per example

**Validation Output Schema:**
```python
{
  "document_name": str,
  "validation_results": [
    {
      "example_id": str,
      "status": "success" | "syntax_error" | "runtime_error" | "timeout",
      "output": str,
      "error": str | null,
      "execution_time_seconds": float
    }
  ],
  "summary": {
    "total_examples": int,
    "passed": int,
    "failed": int
  }
}
```

#### Stage 4: Clarity Validation Agent (`clarity_agent.py`)
- **Purpose**: LLM-as-judge scoring for user experience quality
- **Input**: Original markdown + extraction metadata + validation results
- **Output**: `<doc>_clarity.json` with scores and actionable feedback
- **Key Features**:
  - Parallel processing (default 5 workers)
  - MCP server for structured scoring (6 dimensions)
  - Pre-processes MkDocs Material snippets
  - Correlates with API/code validation issues

**Clarity Output Schema:**
```python
{
  "document_name": str,
  "scores": {
    "instruction_clarity": float,      # 0-10
    "logical_flow": float,             # 0-10
    "completeness": float,             # 0-10
    "consistency": float,              # 0-10
    "prerequisite_coverage": float,    # 0-10
    "overall_score": float             # Average
  },
  "issues": [
    {
      "severity": "critical" | "warning" | "info",
      "category": str,
      "description": str,
      "location": {"section": str, "line_number": int, "step_number": int},
      "suggestion": str
    }
  ],
  "summary": {
    "critical_issues": int,
    "warnings": int,
    "info_items": int
  }
}
```

### 2. Walkthrough Validation System

A **standalone subsystem** for validating tutorials through experiential execution:

#### Components

**Walkthrough Generate Agent** (`walkthrough_generate_agent.py`)
- Converts tutorial docs into structured walkthroughs
- Extracts 4 content fields per step:
  - `contentForUser`: What the user reads
  - `contextForAgent`: Background knowledge
  - `operationsForAgent`: Exact commands to run
  - `introductionForAgent`: Purpose and goals
- Validates against `WalkthroughExport` schema

**Walkthrough Audit Agent** (`walkthrough_audit_agent.py`)
- Executes walkthroughs step-by-step like a real developer
- Connects to MCP server for step delivery
- Reports 6 categories of gaps:
  - **Clarity**: Vague instructions
  - **Prerequisite**: Missing dependencies
  - **Logical Flow**: Steps reference undefined resources
  - **Execution**: Commands fail
  - **Completeness**: Missing verification steps
  - **Cross-Reference**: Should link to other docs

**MCP Server** (`walkthroughs/mcp_server.py`)
- Supplies steps sequentially (prevents skipping ahead)
- Tools: `start_walkthrough()`, `next_step()`, `report_gap()`
- Maintains session state
- Enforces controlled testing environment

---

## Data Flow

### Core Pipeline Flow

```
1. CLI Command: stackbench run --repo <url> --library <name> --version <ver>
         ↓
2. Cache Check: CacheManager.get_cached_run()
   • Key: repo_url:commit_hash:docs_path:library:version
   • Cache hit → Return cached results immediately
   • Cache miss → Continue to step 3
         ↓
3. Repository Cloning: RepositoryManager.clone_repository()
   • Resolves commit hash if not provided
   • Clones repository at specific branch/commit
   • Discovers markdown files
   • Filters by include_folders and API reference detection
         ↓
4. Document Sorting: _estimate_and_sort_documents()
   • Sort by file size (largest first)
   • Minimize idle worker time at end
         ↓
5. Worker Pool Launch: asyncio.gather()
   • Create shared document queue
   • Spawn N workers (default 5)
   • Each worker processes documents end-to-end:
     
     Worker Processing Per Document:
     ┌─────────────────────────────────────┐
     │ 1. Extraction Agent                 │
     │    ↓                                │
     │ 2. API Validation Agent             │
     │    ↓                                │
     │ 3. Code Validation Agent            │
     │    ↓                                │
     │ 4. Clarity Validation Agent         │
     └─────────────────────────────────────┘
         ↓
6. Results Aggregation: Collect from all workers
         ↓
7. Cache Update: CacheManager.add_run() + update_run_status("completed")
         ↓
8. Summary Display: Rich console output with statistics
```

### Walkthrough Flow

```
1. CLI Command: stackbench walkthrough run --doc-path <path> --library <name>
         ↓
2. Generate: WalkthroughGenerateAgent
   • Reads tutorial markdown
   • Extracts N logical steps
   • Validates against schema
   • Writes walkthrough.json
         ↓
3. MCP Server Launch: WalkthroughMCPServer
   • Loads walkthrough.json
   • Starts stdio server
   • Waits for agent connection
         ↓
4. Audit: WalkthroughAuditAgent
   • Connects to MCP server
   • Calls start_walkthrough()
   • Iterates: next_step() → execute → report_gap() (if issues)
   • Writes audit.json
         ↓
5. Gap Report: AuditResult with 6 gap categories
```

---

## Key Modules

### CLI Module (`stackbench/cli.py`)
- **LOC**: 728
- **Dependencies**: Typer, Rich, asyncio
- **Commands**:
  - `run`: Core pipeline with caching
  - `rerun-clarity`: Re-run clarity validation only
  - `version`: Show version
  - `walkthrough generate`: Create walkthrough from docs
  - `walkthrough audit`: Execute walkthrough
  - `walkthrough run`: Full walkthrough pipeline

### Pipeline Module (`stackbench/pipeline/`)
- **runner.py** (482 LOC)
  - `DocumentationValidationPipeline`: Orchestrates 4-stage pipeline
  - Worker pool pattern with asyncio
  - Cache integration
  - Progress tracking with Rich

### Agents Module (`stackbench/agents/`)
- **LOC**: 3,293 total
- **extraction_agent.py** (661 LOC)
  - Parallel document processing
  - Claude Code agent with Read/Write tools
  - Schema validation hooks
- **api_signature_validation_agent.py** (790 LOC)
  - Dynamic code introspection with `inspect`
  - Virtual environment creation
  - Detailed mismatch reporting
- **code_example_validation_agent.py** (587 LOC)
  - Isolated execution environments
  - Stdout/stderr capture
  - Timeout handling
- **clarity_agent.py** (1,115 LOC)
  - LLM-as-judge with MCP server
  - Multi-dimensional scoring (6 dimensions)
  - MkDocs Material snippet preprocessing
  - Parallel processing

### Walkthroughs Module (`stackbench/walkthroughs/`)
- **LOC**: 1,379 total
- **walkthrough_generate_agent.py** (354 LOC)
- **walkthrough_audit_agent.py** (387 LOC)
- **mcp_server.py** (410 LOC): Step delivery and gap reporting
- **schemas.py** (191 LOC): WalkthroughExport, AuditResult, Gap models

### Cache Module (`stackbench/cache/`)
- **manager.py** (254 LOC)
  - JSON-based caching (data/runs.json)
  - Cache key generation
  - Run metadata indexing
  - Status tracking (initializing → in_progress → completed)

### Hooks Module (`stackbench/hooks/`)
- **manager.py** (109 LOC): Unified hook configuration
- **validation.py**: Pydantic schema validation hooks
- **logging.py**: Tool call logging (human + JSONL)
- **logging_manager.py**: AgentLogger implementation

### Repository Module (`stackbench/repository/`)
- **manager.py** (559 LOC)
  - Git repository cloning
  - Commit hash resolution
  - Markdown file discovery
  - API reference page filtering
  - RunContext management

### Schemas Module (`stackbench/schemas.py`)
- **LOC**: ~400
- Pydantic models for all data structures
- Used by validation hooks

### Frontend (`frontend/src/`)
- **LOC**: 2,167 total
- **App.tsx** (75,515 chars): Main application
- **Components**:
  - `RunSelector.tsx`: Browse runs
  - `RunInfo.tsx`: Run metadata display
  - `CodeBlockWithValidation.tsx`: Code example results
  - `WalkthroughViewer.tsx`: Walkthrough display
  - `GapCard.tsx`: Gap visualization
  - `MarkdownViewer.tsx`: Rendered markdown
  - `Tabs.tsx`: Navigation
  - `Settings.tsx`: Configuration
- **Services**:
  - `api.ts`: File system access via Vite plugin
- **Types**: TypeScript definitions

---

## Extension Points

### 1. Adding New Validation Agents

**Steps:**
1. Create agent class in `stackbench/agents/new_agent.py`
2. Define input/output Pydantic schemas in `stackbench/schemas.py`
3. Create validation hook in `stackbench/hooks/validation.py`:
   ```python
   def create_new_agent_validation_hook(output_dir, log_dir):
       def hook(input_data):
           # Validate against NewAgentSchema
           # Log validation calls
       return hook
   ```
4. Update `HookManager` in `stackbench/hooks/manager.py`:
   ```python
   elif self.agent_type == "new_agent":
       validation_hook = create_new_agent_validation_hook(...)
   ```
5. Integrate into pipeline in `stackbench/pipeline/runner.py`:
   ```python
   # Stage 5: New Agent
   new_agent = NewAgent(...)
   new_result = await new_agent.process(...)
   ```
6. Add CLI command in `stackbench/cli.py`
7. Update frontend to display new results

### 2. Custom MCP Servers

**Location**: `stackbench/mcp_servers/`

**Example**: Clarity scoring server (`clarity_scoring_server.py`)

**Steps to add**:
1. Create new server class inheriting from `MCPServer`
2. Define tools with `@mcp.tool()` decorator
3. Implement stdio communication
4. Update agent to use server

### 3. Custom Hook Types

**Supported Hook Types**:
- `PreToolUse`: Run before agent tool execution
- `PostToolUse`: Run after agent tool execution

**Steps**:
1. Create hook function in `stackbench/hooks/`
2. Register in `HookManager.create_hooks()`
3. Use `HookMatcher` to filter by tool type (Read, Write, Bash, etc.)

### 4. Frontend Components

**Location**: `frontend/src/components/`

**Steps**:
1. Create React component in TypeScript
2. Use Tailwind CSS for styling
3. Import in `App.tsx`
4. Add to routing/tabs if needed

---

## Caching System

### Architecture

**Storage**: JSON-based index in `data/runs.json`

**Cache Key Format**:
```
{repo_url}:{doc_commit_hash}:{docs_path}:{library_name}:{library_version}
```

**Example**:
```
https://github.com/lancedb/lancedb:fe25922:docs/src:lancedb:0.25.2
```

### Cache Entry Schema

```json
{
  "run_id": "uuid",
  "repo_url": "https://github.com/user/repo",
  "branch": "main",
  "doc_commit_hash": "abc1234",
  "docs_path": "docs/src",
  "include_folders": ["python", "javascript"],
  "library_name": "lancedb",
  "library_version": "0.25.2",
  "timestamp": "2024-01-01T12:00:00",
  "status": "completed",
  "run_dir": "data/abc123-def456..."
}
```

### Cache Operations

**1. Cache Check** (before running):
```python
cached_run_id = cache_manager.get_cached_run(
    repo_url=repo_url,
    doc_commit_hash=resolved_commit,
    docs_path=docs_path,
    library_name=library_name,
    library_version=library_version
)
```

**2. Cache Hit**:
- Return cached run_id immediately
- Skip entire pipeline
- User sees: "Cache hit! Using results from run: {run_id}"

**3. Cache Miss**:
- Continue with pipeline
- After cloning, add run with status "in_progress"
- After completion, update status to "completed"

**4. Force Mode** (`--force`):
- Bypass cache check entirely
- Always run new analysis

### Invalidation Points

1. **Manual**: Delete run from `data/runs.json`
2. **Code change**: New commit in documentation repository
3. **Library change**: Different library version
4. **Configuration change**: Different `docs_path` or `include_folders`
5. **Force flag**: `--force` option

### Cache vs. Rerun Commands

| Command | Behavior |
|---------|----------|
| `stackbench run` (no `--force`) | Checks cache first |
| `stackbench run --force` | Bypasses cache |
| `stackbench rerun-clarity` | Reuses extraction/validation, re-runs clarity only |

---

## Hook System

### Architecture

Stackbench's innovation is **programmatic Python hooks** via Claude Code Agent SDK.

**Hook Types**:
1. **Validation Hooks** (PreToolUse): Schema validation before writes
2. **Logging Hooks** (PreToolUse + PostToolUse): Complete execution trace

### Validation Hooks

**Purpose**: Ensure data quality by design

**Flow**:
```
Agent attempts Write tool → PreToolUse Hook → Validate JSON against Pydantic schema
                                ↓ Valid                    ↓ Invalid
                         Allow write                Return error to agent
                                                    Agent must fix and retry
```

**Implementation** (`stackbench/hooks/validation.py`):
```python
def create_extraction_validation_hook(output_dir, log_dir):
    """Validate extraction outputs against ExtractionResult schema."""
    
    def validation_hook(input_data):
        # Check if writing to output folder
        if not is_output_file(input_data):
            return None  # Allow non-output writes
        
        # Parse and validate JSON
        try:
            data = json.loads(input_data['content'])
            ExtractionResult(**data)  # Pydantic validation
            return None  # Valid - allow write
        except ValidationError as e:
            # Block write - return error to agent
            return {
                "error": f"Schema validation failed: {e}",
                "details": e.errors()
            }
    
    return validation_hook
```

**Schemas Validated**:
- Extraction: `ExtractionResult`
- API Validation: `APIValidationResult`
- Code Validation: `CodeValidationResult`
- Clarity Validation: `ClarityValidationResult`
- Walkthrough: `WalkthroughExport`, `AuditResult`

### Logging Hooks

**Purpose**: Comprehensive audit trail

**Outputs**:
1. **Human-readable logs**: `agent_logs/<agent>.log`
2. **Machine-readable logs**: `agent_logs/<agent>_tools.jsonl`

**Captured Data**:
- Tool name (Read, Write, Bash, etc.)
- Input parameters (file paths, content, commands)
- Output results (success/failure, content, stdout/stderr)
- Timestamps
- Errors and exceptions

**Implementation** (`stackbench/hooks/logging.py`):
```python
class AgentLogger:
    def log_pre_tool_use(self, tool_name, input_data):
        """Log before tool execution."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": "pre",
            "tool": tool_name,
            "input": input_data
        }
        self._write_jsonl(entry)
        self._write_human_readable(entry)
    
    def log_post_tool_use(self, tool_name, output_data):
        """Log after tool execution."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": "post",
            "tool": tool_name,
            "output": output_data
        }
        self._write_jsonl(entry)
        self._write_human_readable(entry)
```

### Hook Manager

**Purpose**: Unified hook configuration for all agents

**Usage** (`stackbench/hooks/manager.py`):
```python
hook_manager = HookManager(
    agent_type="extraction",
    logger=agent_logger,
    output_dir=output_folder,
    validation_log_dir=validation_log_dir
)
hooks = hook_manager.create_hooks()

# Pass to Claude Code Agent SDK
client = ClaudeAgentClient(hooks=hooks)
```

---

## Walkthrough System

**Purpose**: Validate documentation through experiential execution (simulating a real developer following a tutorial).

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  1. GENERATE: Tutorial Doc → Structured Walkthrough JSON     │
│     (WalkthroughGenerateAgent)                                │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│  2. MCP SERVER: Load walkthrough, deliver steps sequentially │
│     (WalkthroughMCPServer - stdio communication)             │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│  3. AUDIT: Execute each step, report gaps                    │
│     (WalkthroughAuditAgent)                                  │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│  4. OUTPUT: Comprehensive gap report (AuditResult)           │
└──────────────────────────────────────────────────────────────┘
```

### Step Structure

Each walkthrough step contains 4 fields:

```python
{
  "stepNumber": 1,
  "contentForUser": "Install the library",  # What user reads
  "contextForAgent": "The library requires Python 3.8+",  # Background
  "operationsForAgent": ["pip install lancedb==0.25.2"],  # Commands
  "introductionForAgent": "Install core dependencies"  # Purpose
}
```

### Gap Categories

| Category | Severity | Example |
|----------|----------|---------|
| **Clarity** | Warning | "Step 2: Instruction 'configure the database' is vague" |
| **Prerequisite** | Critical | "Step 1 requires Node.js but never mentioned" |
| **Logical Flow** | Critical | "Step 3 references 'config.yaml' not created in Step 1" |
| **Execution** | Critical | "Step 4 command failed: 'npm start' → ENOENT" |
| **Completeness** | Warning | "No verification step after installation" |
| **Cross-Reference** | Info | "Should link to authentication docs for API key setup" |

### Integration with Core Pipeline

**Separate but Complementary**:
- Core pipeline: API/code validation (static/dynamic analysis)
- Walkthrough system: Tutorial validation (experiential execution)

**Can run independently**:
```bash
# Core pipeline only
stackbench run --repo <url> --library <name>

# Walkthrough only
stackbench walkthrough run --doc-path docs/quickstart.md --library <name>

# Both (manually)
stackbench run ...  # Get run_id
stackbench walkthrough generate --from-run <run_id> --doc-path ...
```

---

## Frontend Architecture

**Stack**: React 19 + TypeScript + Vite + Tailwind CSS

### File Structure

```
frontend/
├── src/
│   ├── App.tsx           # Main application (routing, state)
│   ├── main.tsx          # Entry point
│   ├── index.css         # Global styles
│   ├── components/       # React components
│   │   ├── RunSelector.tsx
│   │   ├── RunInfo.tsx
│   │   ├── CodeBlockWithValidation.tsx
│   │   ├── WalkthroughViewer.tsx
│   │   ├── GapCard.tsx
│   │   ├── MarkdownViewer.tsx
│   │   ├── Tabs.tsx
│   │   └── Settings.tsx
│   ├── services/
│   │   └── api.ts        # File system access
│   └── types/
│       └── index.ts      # TypeScript definitions
├── vite-plugin-local-fs.ts  # Custom Vite plugin for local file access
├── vite.config.ts
├── package.json
└── tailwind.config.js
```

### Key Features

**1. Run Browser**:
- Lists all runs from `data/runs.json`
- Filters by repository, library, status
- Shows metadata (commit, library version, timestamps)

**2. Document Viewer**:
- Displays validation results per document
- Color-coded status (pass/fail/warning)
- Expandable sections for extraction/API/code/clarity

**3. Code Example Validation**:
- Inline display of code blocks
- Execution results (success/error)
- Stdout/stderr output

**4. Walkthrough Viewer**:
- Step-by-step navigation
- Gap highlighting by severity
- Suggested fixes

**5. Clarity Scores**:
- 6-dimensional radar chart (planned)
- Issue list with locations
- Actionable suggestions

### Custom Vite Plugin

**Purpose**: Access local file system from browser (data/ folder)

**Implementation** (`vite-plugin-local-fs.ts`):
```typescript
export default function localFsPlugin() {
  return {
    name: 'vite-plugin-local-fs',
    configureServer(server) {
      server.middlewares.use('/api/local-fs', async (req, res) => {
        // Read files from data/ folder
        // Return JSON responses
      })
    }
  }
}
```

---

## Performance Characteristics

### Core Pipeline

**Timing Factors**:
- Repository size (clone time)
- Number of documents
- Document size (markdown file size)
- Number of workers
- API/code complexity

**Example** (LanceDB, 7 documents, 5 workers):
- Clone: ~10s
- Extraction: ~30s (parallel)
- API Validation: ~40s (sequential, with library install)
- Code Validation: ~30s (sequential, with env creation)
- Clarity Validation: ~180s (parallel, Claude API calls)
- **Total**: ~5 minutes

**Optimization Strategies**:
1. **Parallel extraction**: Process multiple documents simultaneously
2. **Longest-first sorting**: Minimize idle worker time
3. **Caching**: Skip repeated analyses (same commit + library version)
4. **Parallel clarity**: Multiple documents scored concurrently

### Caching Impact

| Scenario | No Cache | With Cache |
|----------|----------|------------|
| First run | ~5 min | ~5 min |
| Repeat run (same commit) | ~5 min | <1 sec |
| Repeat run (different commit) | ~5 min | ~5 min |
| Repeat run (different library) | ~5 min | ~5 min |

---

## Directory Layout

```
stackbench/
├── stackbench/                   # Main Python package
│   ├── __init__.py
│   ├── cli.py                    # CLI commands (728 LOC)
│   ├── schemas.py                # Pydantic models (~400 LOC)
│   ├── agents/                   # Core validation agents (3,293 LOC)
│   │   ├── extraction_agent.py
│   │   ├── api_signature_validation_agent.py
│   │   ├── code_example_validation_agent.py
│   │   ├── clarity_agent.py
│   │   └── clarity_helpers.py
│   ├── walkthroughs/             # Walkthrough system (1,379 LOC)
│   │   ├── schemas.py
│   │   ├── walkthrough_generate_agent.py
│   │   ├── walkthrough_audit_agent.py
│   │   ├── mcp_server.py
│   │   └── README.md
│   ├── cache/                    # Caching system (254 LOC)
│   │   └── manager.py
│   ├── hooks/                    # Validation + logging hooks (~500 LOC)
│   │   ├── validation.py
│   │   ├── logging.py
│   │   ├── logging_manager.py
│   │   └── manager.py
│   ├── pipeline/                 # Orchestration (482 LOC)
│   │   └── runner.py
│   ├── repository/               # Git operations (559 LOC)
│   │   └── manager.py
│   ├── mcp_servers/              # MCP servers
│   │   └── clarity_scoring_server.py
│   └── utils/                    # Utilities
│       └── schema_utils.py
├── frontend/                     # React web interface (2,167 LOC)
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   ├── services/
│   │   └── types/
│   ├── vite.config.ts
│   └── package.json
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md           # This file
│   ├── CODEMAP.md                # Detailed code structure
│   └── 0-plan.md                 # Feature roadmap
├── tests/                        # Test suite
├── data/                         # Output directory (gitignored)
│   ├── runs.json                 # Cache index
│   └── <run_id>/
│       ├── repository/
│       ├── results/
│       ├── walkthroughs/
│       └── validation_logs/
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Technology Stack

### Python Backend
- **Python**: 3.11+
- **CLI**: Typer (command-line interface)
- **UI**: Rich (terminal output)
- **Validation**: Pydantic v2 (data schemas)
- **Git**: GitPython (repository management)
- **Async**: asyncio (worker pools)
- **AI**: Claude Agent SDK (Claude Code hooks)

### Frontend
- **Framework**: React 19
- **Language**: TypeScript
- **Build**: Vite
- **Styling**: Tailwind CSS
- **Package Manager**: Bun

### Infrastructure
- **Version Control**: Git
- **Caching**: JSON files (data/runs.json)
- **Logging**: Plain text + JSONL
- **Communication**: MCP (Model Context Protocol) over stdio

---

## Metrics

### Lines of Code by Module

| Module | LOC | Percentage |
|--------|-----|------------|
| **Agents** | 3,293 | 38.8% |
| **Walkthroughs** | 1,379 | 16.2% |
| **CLI** | 728 | 8.6% |
| **Pipeline** | 482 | 5.7% |
| **Repository** | 559 | 6.6% |
| **Hooks** | ~500 | 5.9% |
| **Schemas** | ~400 | 4.7% |
| **Cache** | 254 | 3.0% |
| **Utils** | ~100 | 1.2% |
| **Other** | ~793 | 9.3% |
| **Total Python** | 8,488 | 100% |

### Frontend LOC
- **Total**: 2,167 TypeScript/TSX

### Test Coverage
- Tests located in `tests/`
- Async test support via pytest-asyncio
- Coverage: TBD (expand test suite)

---

## Comparison with Traditional Tools

| Feature | Traditional Tools (Sphinx, JSDoc) | Stackbench |
|---------|-----------------------------------|------------|
| **API Signature Docs** | Generate from code → docs | Validate docs → code |
| **Code Examples** | Static display | Dynamic execution + validation |
| **Broken Examples** | Manual testing required | Automated detection |
| **Clarity** | Manual review | LLM-as-judge scoring |
| **Tutorial Flow** | No validation | Step-by-step execution (walkthroughs) |
| **Caching** | N/A (no analysis) | Intelligent caching by commit + library |
| **CI/CD Integration** | Docs generation | Quality gate (fail on errors) |

---

## Future Enhancements

### Planned Agents
1. **Deprecated API Detection**: Flag use of deprecated methods
2. **Missing Coverage Agent**: Find APIs without documentation
3. **Real-World Integration Agent**: Security, error handling, production patterns

### Technical Improvements
1. **Multi-language support**: TypeScript, Go, Rust
2. **Incremental validation**: Only validate changed sections
3. **Auto-fix mode**: Agent proposes documentation fixes
4. **GitHub App**: Automated PR comments
5. **Advanced metrics**: Trend analysis over time
6. **Performance profiling**: Identify slow agents

### Frontend Features
1. **Radar charts**: Visualize clarity scores
2. **Diff view**: Compare runs across commits
3. **Export reports**: PDF/HTML
4. **Search**: Full-text search across validation results

---

## Contributing

### Adding a New Agent

1. **Create Agent Class**: `stackbench/agents/new_agent.py`
2. **Define Schemas**: Add to `stackbench/schemas.py`
3. **Create Validation Hook**: In `stackbench/hooks/validation.py`
4. **Update Hook Manager**: In `stackbench/hooks/manager.py`
5. **Integrate Pipeline**: In `stackbench/pipeline/runner.py`
6. **Add CLI Command**: In `stackbench/cli.py`
7. **Update Frontend**: Add component to display results
8. **Write Tests**: In `tests/`
9. **Update Docs**: This file + CODEMAP.md

### Code Style
- **Formatter**: Ruff
- **Linter**: Ruff (rules: E, F, I, N, W, UP)
- **Line Length**: 120 characters
- **Type Hints**: Required for public APIs

### Testing Philosophy
- **Unit Tests**: Schema validation, hook behavior
- **Integration Tests**: End-to-end pipeline with sample repos
- **Agent Tests**: Validate outputs against known-good examples

---

## References

- **Claude Code**: https://docs.claude.com/en/docs/claude-code
- **Claude Agent SDK**: https://docs.claude.com/en/docs/claude-code/agent-sdk
- **Pydantic**: https://docs.pydantic.dev/
- **Typer**: https://typer.tiangolo.com/
- **Rich**: https://rich.readthedocs.io/
- **MCP**: https://modelcontextprotocol.io/

---

**Last Updated**: 2024  
**Maintainer**: Stackbench Team  
**License**: TBD
