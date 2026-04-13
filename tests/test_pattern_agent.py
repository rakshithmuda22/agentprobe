"""Tests for the Pattern Agent."""

from __future__ import annotations

from src.agents.pattern_agent import run
from src.graph.state import AgentProbeState
from src.profiles.style_generator import detect_case, detect_import_category, check_import_order


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


# ── detect_case tests ──────────────────────────────────────────


class TestDetectCase:
    def test_camel_case(self):
        assert detect_case("getUserById") == "camelCase"
        assert detect_case("processData") == "camelCase"

    def test_snake_case(self):
        assert detect_case("get_user_by_id") == "snake_case"
        assert detect_case("process") == "snake_case"

    def test_pascal_case(self):
        assert detect_case("UserService") == "PascalCase"
        assert detect_case("HttpClient") == "PascalCase"

    def test_upper_snake_case(self):
        assert detect_case("MAX_RETRIES") == "UPPER_SNAKE_CASE"
        assert detect_case("API_KEY") == "UPPER_SNAKE_CASE"

    def test_kebab_case(self):
        assert detect_case("my-component") == "kebab-case"
        assert detect_case("user-service") == "kebab-case"

    def test_leading_underscore(self):
        assert detect_case("_private_method") == "snake_case"
        assert detect_case("__dunder__") == "snake_case"


# ── detect_import_category tests ────────────────────────────────


class TestDetectImportCategory:
    def test_builtin(self):
        assert detect_import_category("os") == "builtin"
        assert detect_import_category("sys") == "builtin"
        assert detect_import_category("pathlib") == "builtin"

    def test_relative(self):
        assert detect_import_category(".models") == "relative"
        assert detect_import_category("..utils") == "relative"

    def test_internal(self):
        assert detect_import_category("src.models.user") == "internal"
        assert detect_import_category("src.utils") == "internal"

    def test_external(self):
        assert detect_import_category("flask") == "external"
        assert detect_import_category("requests") == "external"
        assert detect_import_category("numpy") == "external"


# ── check_import_order tests ────────────────────────────────────


class TestCheckImportOrder:
    def test_correct_order(self):
        categories = ["builtin", "external", "internal", "relative"]
        expected = ["builtin", "external", "internal", "relative"]
        assert check_import_order(categories, expected) == []

    def test_wrong_order(self):
        categories = ["internal", "external", "builtin"]
        expected = ["builtin", "external", "internal", "relative"]
        violations = check_import_order(categories, expected)
        assert len(violations) >= 1


# ── Pattern Agent integration tests ─────────────────────────────


def test_clean_pr_passes(clean_python_diff, clean_repo):
    """Clean PR with proper naming should PASS."""
    state = _make_state(clean_python_diff, str(clean_repo))
    result = run(state)

    report = result["pattern_report"]
    assert report["status"] == "PASS"
    assert report["violations"] == []


def test_naming_violations_detected(naming_violation_diff, naming_violation_repo):
    """snake_case functions in a camelCase codebase should produce WARN."""
    state = _make_state(naming_violation_diff, str(naming_violation_repo))
    result = run(state)

    report = result["pattern_report"]
    assert report["status"] == "WARN"

    naming_violations = [v for v in report["violations"] if v["rule_violated"] == "naming.functions"]
    assert len(naming_violations) >= 1
    # Functions are snake_case but profile expects camelCase
    assert naming_violations[0]["expected"] == "camelCase"
    assert naming_violations[0]["actual"] == "snake_case"


def test_forbidden_pattern_detected(forbidden_pattern_diff, naming_violation_repo):
    """console.log in JavaScript should be flagged as forbidden."""
    state = _make_state(forbidden_pattern_diff, str(naming_violation_repo))
    result = run(state)

    report = result["pattern_report"]
    assert report["status"] == "WARN"

    forbidden_violations = [
        v for v in report["violations"]
        if "forbidden" in v["rule_violated"]
    ]
    assert len(forbidden_violations) >= 1


def test_import_order_violation(import_order_violation_diff, naming_violation_repo):
    """Imports in wrong order should produce WARN."""
    state = _make_state(import_order_violation_diff, str(naming_violation_repo))
    result = run(state)

    report = result["pattern_report"]
    assert report["status"] == "WARN"

    order_violations = [v for v in report["violations"] if v["rule_violated"] == "imports.order"]
    assert len(order_violations) >= 1


def test_no_style_profile_passes(tmp_path):
    """If no style profile exists, agent should PASS (nothing to check)."""
    state = _make_state(
        "diff --git a/x.py b/x.py\n+++ b/x.py\n@@ -0,0 +1 @@\n+x = 1\n",
        str(tmp_path),
    )
    result = run(state)

    assert result["pattern_report"]["status"] == "PASS"


def test_filename_violation(naming_violation_diff, naming_violation_repo):
    """snake_case filename in kebab-case codebase should produce WARN."""
    state = _make_state(naming_violation_diff, str(naming_violation_repo))
    result = run(state)

    report = result["pattern_report"]
    file_violations = [v for v in report["violations"] if v["rule_violated"] == "naming.files"]
    # user_service.py is snake_case, profile expects kebab-case
    assert len(file_violations) >= 1
