# Stackbench CLI Command Reference

Complete reference for all Stackbench CLI commands.

## Command Structure

```
stackbench <command> [subcommand] [options]
sb <command> [subcommand] [options]  # Short alias
```

---

## Core Validation Commands

### `stackbench run`

Run complete documentation validation pipeline.

```bash
stackbench run \
  --repo <url> \
  --branch <branch> \
  --docs-path <path> \
  --library <name> \
  --version <version> \
  [--commit <hash>] \
  [--include-folders <folders>] \
  [--output <dir>] \
  [--num-workers <n>] \
  [--force]
```

**Required Options:**
- `--repo`, `-r`: Git repository URL
- `--docs-path`, `-d`: Base documentation directory (e.g., `docs/src`)
- `--library`, `-l`: Primary library name (e.g., `lancedb`)
- `--version`, `-v`: Library version to test against (e.g., `0.25.2`)

**Optional:**
- `--branch`, `-b`: Git branch (default: `main`)
- `--commit`, `-c`: Specific commit hash (auto-resolved if not provided)
- `--include-folders`, `-i`: Comma-separated folders relative to docs-path
- `--output`, `-o`: Output directory (default: `./data`)
- `--num-workers`, `-w`: Number of parallel workers (default: 5)
- `--force`, `-f`: Bypass cache and force re-analysis

**Example:**
```bash
stackbench run \
  --repo https://github.com/lancedb/lancedb \
  --branch main \
  --docs-path docs/src \
  --include-folders python \
  --library lancedb \
  --version 0.25.2 \
  --num-workers 5
```

**What it does:**
1. Clones repository (resolves commit hash if needed)
2. Checks cache for existing results
3. Discovers markdown files (filters by include-folders)
4. Launches worker pool with 4 stages per document:
   - Extraction Agent
   - API Validation Agent
   - Code Validation Agent
   - Clarity Validation Agent
5. Aggregates results and updates cache

---

### `stackbench rerun-clarity`

Re-run only the clarity validation for an existing run.

```bash
stackbench rerun-clarity <run-id> [--workers <n>]
```

**Arguments:**
- `<run-id>`: Run ID from data folder (UUID)

**Options:**
- `--workers`, `-w`: Number of parallel workers (default: 5)

**Example:**
```bash
stackbench rerun-clarity 5bd8e375-313e-4328-827b-33889356828c --workers 5
```

**Use case:** After updating the clarity agent or MCP scoring server.

---

### `stackbench version`

Show Stackbench version.

```bash
stackbench version
```

---

## Walkthrough Commands

### `stackbench walkthrough generate`

Generate a walkthrough from documentation.

```bash
# From existing run
stackbench walkthrough generate \
  --from-run <uuid> \
  --doc-path <path> \
  --library <name> \
  --version <version>

# Fresh clone
stackbench walkthrough generate \
  --repo <url> \
  --branch <branch> \
  --doc-path <path> \
  --library <name> \
  --version <version>
```

**Required:**
- `--doc-path`, `-d`: Path to documentation file (relative to repo root)
- `--library`, `-l`: Library name
- `--version`, `-v`: Library version

**Mode 1 (from existing run):**
- `--from-run`: Existing run UUID (reuses cloned repo)

**Mode 2 (fresh clone):**
- `--repo`, `-r`: Git repository URL
- `--branch`, `-b`: Git branch (default: `main`)

**Example:**
```bash
# Reuse existing run
stackbench walkthrough generate \
  --from-run 22c09315-1385-4ad6-a2ff-1e631a482107 \
  --doc-path docs/quickstart.md \
  --library lancedb \
  --version 0.25.2

# Fresh clone
stackbench walkthrough generate \
  --repo https://github.com/lancedb/lancedb \
  --branch main \
  --doc-path docs/quickstart.md \
  --library lancedb \
  --version 0.25.2
```

**What it does:**
1. Reads tutorial documentation
2. Extracts logical steps with Claude Code agent
3. Validates against WalkthroughExport schema
4. Writes structured walkthrough JSON

---

### `stackbench walkthrough audit`

Audit a walkthrough by executing it step-by-step.

```bash
stackbench walkthrough audit \
  --walkthrough <path> \
  --library <name> \
  --version <version> \
  [--output <dir>] \
  [--working-dir <dir>]
```

**Required:**
- `--walkthrough`, `-w`: Path to walkthrough JSON file
- `--library`, `-l`: Library name
- `--version`, `-v`: Library version

**Optional:**
- `--output`, `-o`: Output directory (default: same as walkthrough)
- `--working-dir`: Working directory for execution (default: temp dir)

**Example:**
```bash
stackbench walkthrough audit \
  --walkthrough data/<uuid>/walkthroughs/wt_abc123/walkthrough.json \
  --library lancedb \
  --version 0.25.2
```

**What it does:**
1. Starts MCP server with walkthrough
2. Creates Claude Code audit agent
3. Agent executes steps sequentially via MCP tools
4. Reports gaps (6 categories: clarity, prerequisite, logical_flow, execution, completeness, cross_reference)
5. Writes audit result JSON

---

### `stackbench walkthrough run`

Full walkthrough pipeline (clone + generate + audit).

```bash
stackbench walkthrough run \
  --repo <url> \
  --branch <branch> \
  --doc-path <path> \
  --library <name> \
  --version <version>
```

**Required:**
- `--repo`, `-r`: Git repository URL
- `--doc-path`, `-d`: Path to documentation file
- `--library`, `-l`: Library name
- `--version`, `-v`: Library version

**Optional:**
- `--branch`, `-b`: Git branch (default: `main`)

**Example:**
```bash
stackbench walkthrough run \
  --repo https://github.com/lancedb/lancedb \
  --branch main \
  --doc-path docs/quickstart.md \
  --library lancedb \
  --version 0.25.2
```

**What it does:**
1. Clones repository
2. Generates walkthrough
3. Audits walkthrough
4. Reports gaps

---

## Development Commands

### `stackbench dev codemap`

Display interactive codebase map with annotations.

```bash
stackbench dev codemap [--details/--no-details] [--json] [--graphs]
```

**Options:**
- `--details` / `--no-details`, `-d` / `-D`: Show detailed annotations (default: details)
- `--json`, `-j`: Export code map to JSON file
- `--graphs`, `-g`: Generate dependency graphs (requires graphviz)

**Example:**
```bash
# Interactive tree with details
stackbench dev codemap

# Quick tree without details
stackbench dev codemap --no-details

# Export to JSON
stackbench dev codemap --json

# Generate dependency graphs
stackbench dev codemap --graphs

# All options
stackbench dev codemap --json --graphs
```

**What it shows:**
- Directory and file structure
- Lines of code per module
- Module purpose (from docstrings)
- Import dependencies
- Test coverage indicators (✓/✗)

**Generated artifacts:**
- `docs/codemap.json` - Machine-readable code map
- `docs/python_dependencies.png/svg` - Python module dependency graphs
- `docs/frontend_dependencies.png/svg` - Frontend component dependency graphs

**Use case:** Explore codebase structure, generate documentation, analyze dependencies.

---

## Environment Variables

Stackbench reads from `.env` file or environment:

```bash
# Claude API configuration (if using hosted Claude)
ANTHROPIC_API_KEY=sk-ant-...

# Claude Code CLI path (optional)
CLAUDE_CODE_PATH=/usr/local/bin/claude-code
```

---

## Output Directory Structure

```
data/
├── runs.json                      # Cache index
└── <run_id>/
    ├── repository/                # Cloned git repo
    ├── metadata.json              # Run metadata
    ├── results/
    │   ├── extraction/
    │   │   └── <doc>_analysis.json
    │   ├── api_validation/
    │   │   └── <doc>_validation.json
    │   ├── code_validation/
    │   │   └── <doc>_validation.json
    │   └── clarity_validation/
    │       └── <doc>_clarity.json
    ├── validation_logs/           # Hook execution logs
    │   └── clarity_logs/
    └── walkthroughs/
        └── wt_<uuid>/
            ├── wt_<uuid>.json            # Generated walkthrough
            ├── wt_<uuid>_audit.json      # Audit results
            ├── agent_logs/
            │   ├── generate.log
            │   ├── generate_tools.jsonl
            │   ├── audit.log
            │   └── audit_tools.jsonl
            └── validation_logs/
```

---

## Cache Behavior

**Cache Key Format:**
```
{repo_url}:{doc_commit_hash}:{docs_path}:{library_name}:{library_version}
```

**Cache Operations:**
- **Cache hit**: Returns cached run_id immediately, skips analysis
- **Cache miss**: Runs new analysis, adds to cache
- **Force mode** (`--force`): Bypasses cache, always runs

**Invalidation points:**
1. Manual deletion from `data/runs.json`
2. New commit in documentation repository
3. Different library version
4. Different `docs_path` or `include_folders`
5. `--force` flag

---

## Common Workflows

### 1. First-time analysis
```bash
stackbench run \
  --repo https://github.com/org/project \
  --docs-path docs \
  --library mylib \
  --version 1.0.0
```

### 2. Test docs against different library versions
```bash
# Test against v1.0
stackbench run --repo <url> --docs-path docs --library mylib --version 1.0.0

# Test against v2.0 (cache miss - different library version)
stackbench run --repo <url> --docs-path docs --library mylib --version 2.0.0
```

### 3. Track documentation quality over time
```bash
# Analyze commit abc123
stackbench run --repo <url> --commit abc123 --docs-path docs --library mylib --version 1.0

# Later, analyze commit def456
stackbench run --repo <url> --commit def456 --docs-path docs --library mylib --version 1.0
```

### 4. Validate specific tutorial
```bash
# Generate walkthrough
stackbench walkthrough generate \
  --repo <url> \
  --doc-path docs/tutorial.md \
  --library mylib \
  --version 1.0

# Audit it
stackbench walkthrough audit \
  --walkthrough data/<uuid>/walkthroughs/wt_<id>/walkthrough.json \
  --library mylib \
  --version 1.0
```

### 5. Explore codebase
```bash
# View structure
stackbench dev codemap

# Generate documentation artifacts
stackbench dev codemap --json --graphs
```

---

## Tips & Best Practices

1. **Use `--include-folders`** to focus on specific documentation sections
2. **Pin `--commit`** for reproducible results
3. **Adjust `--num-workers`** based on your machine (default: 5)
4. **Use `--force`** to bypass cache when testing agent changes
5. **Check cache** with `cat data/runs.json` to see all runs
6. **Rerun clarity only** if you only changed the clarity agent
7. **Use walkthroughs** for tutorial-style documentation
8. **Use codemap** to understand codebase before making changes

---

## Troubleshooting

### Issue: Cache not being used
- Check that commit hash is identical
- Verify `docs_path` and `include_folders` match exactly
- Ensure library name and version are the same

### Issue: Worker pool hangs
- Reduce `--num-workers` (try 1 for debugging)
- Check agent logs in `data/<run_id>/validation_logs/`

### Issue: Graphviz not found
- Install system package: `sudo apt-get install graphviz`
- Or: `brew install graphviz` (macOS)

### Issue: Claude Code not working
- Ensure Claude Code CLI is installed
- Check environment variables (ANTHROPIC_API_KEY if using hosted)
- Verify `claude-code --version` works

---

**For more details, see:**
- Architecture: `docs/ARCHITECTURE.md`
- Code map: `docs/CODEMAP.md`
- Feature plan: `docs/0-plan.md`
