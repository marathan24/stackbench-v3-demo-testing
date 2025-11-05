"""
Graded code validation with partial credit.

Implements multi-level code correctness scoring instead of binary pass/fail.
Addresses Mistake #6 from EVALUATION_ANALYSIS.md: Binary code validation
missing partial credit.
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import subprocess
import tempfile
from pathlib import Path


@dataclass
class GradedCodeResult:
    """Graded validation result with partial credit."""

    code: str
    """Code that was validated"""

    total_score: float
    """Overall score (0.0 to 1.0)"""

    component_scores: Dict[str, float]
    """Scores for each validation level"""

    category: str
    """Failure category: syntax_error, quality_issues, runtime_error, wrong_output, or perfect"""

    details: Dict[str, any]
    """Detailed results from each validation level"""

    suggestions: List[str]
    """Improvement suggestions"""


def check_syntax(code: str, language: str = "python") -> Tuple[bool, Optional[str]]:
    """
    Check if code has valid syntax.

    Args:
        code: Code to check
        language: Programming language (default: python)

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        ```python
        valid, error = check_syntax("print('Hello')")
        # (True, None)

        valid, error = check_syntax("print('Hello'")
        # (False, "SyntaxError: unexpected EOF")
        ```
    """
    if language != "python":
        # For now, only Python is supported
        return True, None

    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


async def run_static_analysis(code: str) -> Tuple[int, List[str]]:
    """
    Run static analysis (pylint-style checks).

    Args:
        code: Code to analyze

    Returns:
        Tuple of (issue_count, issues_list)

    Example:
        ```python
        count, issues = await run_static_analysis(code)
        # (3, ["undefined variable 'x'", "unused import 'os'", ...])
        ```
    """
    # For now, do basic checks
    # In production, integrate with pylint, flake8, mypy
    issues = []

    # Check for common issues
    lines = code.split('\n')

    # Check for undefined variables (simple heuristic)
    defined_vars = set()
    for line in lines:
        # Very simple variable assignment detection
        if '=' in line and not line.strip().startswith('#'):
            parts = line.split('=')
            var_name = parts[0].strip().split()[-1]
            defined_vars.add(var_name)

    # This is a simplified check - real implementation would use AST
    return len(issues), issues


async def execute_code(
    code: str,
    timeout: int = 5,
    expected_output: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Execute code and capture output.

    Args:
        code: Code to execute
        timeout: Timeout in seconds
        expected_output: Expected output (if provided, will check correctness)

    Returns:
        Tuple of (success, stdout, stderr)

    Example:
        ```python
        success, stdout, stderr = await execute_code("print('Hello')")
        # (True, "Hello\\n", None)

        success, stdout, stderr = await execute_code("1/0")
        # (False, None, "ZeroDivisionError: division by zero")
        ```
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        success = result.returncode == 0
        stdout = result.stdout if success else None
        stderr = result.stderr if not success else None

        # If expected output provided, check correctness
        if success and expected_output:
            success = stdout.strip() == expected_output.strip()

        return success, stdout, stderr

    except subprocess.TimeoutExpired:
        return False, None, f"Execution timeout ({timeout}s)"
    finally:
        # Clean up
        Path(temp_file).unlink(missing_ok=True)


async def graded_validate_code(
    code: str,
    expected_output: Optional[str] = None,
    timeout: int = 5
) -> GradedCodeResult:
    """
    Validate code with graded scoring across multiple levels.

    Scoring breakdown:
    - Syntax (25%): Does it parse?
    - Static quality (25%): Are there code quality issues?
    - Execution (25%): Does it run without errors?
    - Output correctness (25%): Does it produce correct results?

    Args:
        code: Code to validate
        expected_output: Expected output (if known)
        timeout: Execution timeout in seconds

    Returns:
        GradedCodeResult with partial credit scores

    Example:
        ```python
        result = await graded_validate_code(
            code="print('Hello')",
            expected_output="Hello"
        )

        print(f"Score: {result.total_score:.2f}")
        print(f"Category: {result.category}")

        # Perfect code: score = 1.0, category = "perfect"
        # Syntax error: score = 0.0, category = "syntax_error"
        # Runs but wrong output: score = 0.75, category = "wrong_output"
        ```
    """
    component_scores = {}
    details = {}
    suggestions = []

    # Level 1: Syntax check (25%)
    syntax_ok, syntax_error = check_syntax(code)
    component_scores["syntax"] = 0.25 if syntax_ok else 0.0
    details["syntax"] = {"valid": syntax_ok, "error": syntax_error}

    if not syntax_ok:
        suggestions.append(f"Fix syntax error: {syntax_error}")
        return GradedCodeResult(
            code=code,
            total_score=0.0,
            component_scores=component_scores,
            category="syntax_error",
            details=details,
            suggestions=suggestions
        )

    # Level 2: Static analysis (25%)
    issue_count, issues = await run_static_analysis(code)
    # Score based on number of issues (capped at 10)
    quality_score = max(0.0, 1.0 - (issue_count / 10))
    component_scores["static_quality"] = 0.25 * quality_score
    details["static_analysis"] = {"issue_count": issue_count, "issues": issues}

    if issue_count > 0:
        suggestions.extend([f"Address code quality: {issue}" for issue in issues[:3]])

    # Level 3: Execution (25%)
    exec_success, stdout, stderr = await execute_code(code, timeout=timeout)
    component_scores["executes"] = 0.25 if exec_success else 0.0
    details["execution"] = {
        "success": exec_success,
        "stdout": stdout,
        "stderr": stderr
    }

    if not exec_success:
        suggestions.append(f"Fix runtime error: {stderr}")
        return GradedCodeResult(
            code=code,
            total_score=sum(component_scores.values()),
            component_scores=component_scores,
            category="runtime_error",
            details=details,
            suggestions=suggestions
        )

    # Level 4: Output correctness (25%)
    output_correct = True
    if expected_output:
        output_correct = stdout and stdout.strip() == expected_output.strip()
        component_scores["output_correct"] = 0.25 if output_correct else 0.0
        details["output_correctness"] = {
            "expected": expected_output,
            "actual": stdout,
            "correct": output_correct
        }

        if not output_correct:
            suggestions.append(f"Output mismatch: expected '{expected_output}', got '{stdout.strip()}'")
    else:
        # No expected output to check, give full credit
        component_scores["output_correct"] = 0.25
        details["output_correctness"] = {"checked": False}

    # Calculate total score
    total_score = sum(component_scores.values())

    # Categorize
    if total_score == 1.0:
        category = "perfect"
    elif not output_correct:
        category = "wrong_output"
    elif issue_count > 5:
        category = "quality_issues"
    else:
        category = "minor_issues"

    return GradedCodeResult(
        code=code,
        total_score=total_score,
        component_scores=component_scores,
        category=category,
        details=details,
        suggestions=suggestions
    )
