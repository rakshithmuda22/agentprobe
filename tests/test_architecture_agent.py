"""Tests for the Architecture Agent."""

from __future__ import annotations

from pathlib import Path

from src.agents.architecture_agent import run
from src.graph.state import AgentProbeState


def _make_state(pr_diff: str, repo_path: str = ".") -> AgentProbeState:
    return AgentProbeState(
        pr_diff=pr_diff,
        repo_path=repo_path,
        pr_number=1,
        repo_full_name="test/repo",
        architecture_report=None,
        pattern_report=None,
        regression_report=None,
        short_circuit=False,
        verdict=None,
        cached_functions=[],
        cost_tracker={},
    )


def test_clean_imports_pass(clean_import_diff):
    """Allowed imports should produce PASS status."""
    state = _make_state(clean_import_diff)
    result = run(state)

    assert result["architecture_report"]["status"] == "PASS"
    assert result["architecture_report"]["violations"] == []
    assert result["short_circuit"] is False


def test_forbidden_import_fatal(forbidden_import_diff, boundary_violation_repo):
    """Importing from a forbidden module should produce FATAL."""
    state = _make_state(forbidden_import_diff, str(boundary_violation_repo))
    result = run(state)

    report = result["architecture_report"]
    assert report["status"] == "FATAL"
    assert len(report["violations"]) >= 1
    assert result["short_circuit"] is True

    # Verify the violation details
    fatal_violations = [v for v in report["violations"] if v["severity"] == "FATAL"]
    assert len(fatal_violations) >= 1
    assert "analytics" in fatal_violations[0]["description"]


def test_import_depth_exceeded_warn(deep_import_diff):
    """Import exceeding max depth should produce WARN."""
    state = _make_state(deep_import_diff)
    result = run(state)

    report = result["architecture_report"]
    warn_violations = [v for v in report["violations"] if v["severity"] == "WARN"]
    assert len(warn_violations) >= 1
    assert "depth" in warn_violations[0]["description"].lower()


def test_empty_diff_passes():
    """Empty diff should produce PASS with no violations."""
    state = _make_state("")
    result = run(state)

    assert result["architecture_report"]["status"] == "PASS"
    assert result["architecture_report"]["violations"] == []


def test_non_code_files_ignored():
    """Non-code files (e.g., .md, .yaml) should be skipped."""
    diff = """diff --git a/README.md b/README.md
new file mode 100644
--- /dev/null
+++ b/README.md
@@ -0,0 +1,3 @@
+# My Project
+
+This is a readme.
"""
    state = _make_state(diff)
    result = run(state)

    assert result["architecture_report"]["status"] == "PASS"
