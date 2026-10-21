"""Regression tests for boundary/pattern false positives and policy features."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentprobe.agents import architecture_agent, pattern_agent
from agentprobe.parsers.import_graph import Boundaries, ImportGraph, resolve_import

BOUNDARIES = """\
modules:
  payments:
    allowed_imports: [shared, types]
    forbidden_imports: [analytics, requests]
layers:
  - name: presentation
    paths: [src/api, src/web]
    can_import: [domain]
  - name: domain
    paths: [src/models, src/services]
    can_import: []
"""


def _repo(tmp_path: Path, boundaries: str = BOUNDARIES, style: str | None = None) -> str:
    d = tmp_path / ".agentprobe"
    d.mkdir()
    (d / "boundaries.yaml").write_text(boundaries)
    if style is not None:
        (d / "style-profile.yaml").write_text(style)
    return str(tmp_path)


def _new_file(path: str, *lines: str) -> str:
    body = "\n".join(f"+{line}" for line in lines)
    return (f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n"
            f"+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n{body}\n")


def _arch(diff: str, repo: str) -> dict:
    return architecture_agent.run({"pr_diff": diff, "repo_path": repo})["architecture_report"]


def _pattern(diff: str, repo: str) -> dict:
    return pattern_agent.run({"pr_diff": diff, "repo_path": repo})["pattern_report"]


# ── import resolution ──


@pytest.mark.parametrize("module, file, language, expected", [
    (".models", "src/payments/refunds.py", "python", "src.payments.models"),
    ("..analytics.tracker", "src/payments/refunds.py", "python", "src.analytics.tracker"),
    (".", "src/payments/__init__.py", "python", "src.payments"),
    ("../analytics/tracker", "src/payments/refunds.ts", "typescript", "src.analytics.tracker"),
    ("./models/index.js", "src/payments/refunds.js", "javascript", "src.payments.models"),
    ("os.path", "src/payments/refunds.py", "python", "os.path"),
])
def test_resolve_import(module, file, language, expected):
    assert resolve_import(module, file, language).module == expected


# ── false positives that used to BLOCK ──


def test_stdlib_and_third_party_imports_do_not_trip_allow_list(tmp_path):
    diff = _new_file("src/payments/refunds.py",
                     "import os", "from decimal import Decimal", "import yaml",
                     "def refund(amount):", "    return Decimal(amount)")
    report = _arch(diff, _repo(tmp_path))
    assert report["status"] == "PASS", report["violations"]


def test_relative_import_within_same_module_is_allowed(tmp_path):
    diff = _new_file("src/payments/refunds.py", "from .models import Refund")
    assert _arch(diff, _repo(tmp_path))["status"] == "PASS"


def test_relative_import_into_forbidden_module_is_caught(tmp_path):
    diff = _new_file("src/payments/refunds.py", "from ..analytics.tracker import track")
    report = _arch(diff, _repo(tmp_path))
    assert report["status"] == "FATAL"
    assert "forbidden from importing 'analytics'" in report["violations"][0]["description"]


def test_ts_relative_import_into_forbidden_module_is_caught(tmp_path):
    diff = _new_file("src/payments/refunds.ts", "import { track } from '../analytics/tracker';")
    assert _arch(diff, _repo(tmp_path))["status"] == "FATAL"


def test_undeclared_first_party_module_trips_allow_list(tmp_path):
    (tmp_path / "src" / "marketing").mkdir(parents=True)
    diff = _new_file("src/payments/refunds.py", "from src.marketing.promo import code")
    report = _arch(diff, _repo(tmp_path))
    assert report["status"] == "FATAL"
    assert "can only import from" in report["violations"][0]["description"]


def test_forbidden_third_party_package(tmp_path):
    diff = _new_file("src/payments/refunds.py", "import requests")
    report = _arch(diff, _repo(tmp_path))
    assert report["status"] == "FATAL"


# ── layers ──


def test_layer_paths_are_honored(tmp_path):
    diff = _new_file("src/services/billing.py", "from src.api.routes import router")
    report = _arch(diff, _repo(tmp_path))
    assert report["status"] == "FATAL"
    assert "Layer 'domain' cannot import from layer 'presentation'" in \
        report["violations"][0]["description"]


def test_allowed_layer_direction_passes(tmp_path):
    diff = _new_file("src/api/routes.py", "from src.services.billing import charge")
    assert _arch(diff, _repo(tmp_path))["status"] == "PASS"


def test_layer_check_is_unit_level():
    b = Boundaries(layers=[{"name": "domain", "paths": ["src/models"], "can_import": []},
                           {"name": "infra", "paths": ["src/db"], "can_import": []}])
    graph = ImportGraph()
    assert graph.check_boundary("src.models.user", "src.db.session", b)
    assert graph.check_boundary("src.models.user", "sqlalchemy.orm", b, internal=False) == []


# ── line numbers and policy edits ──


def test_violation_reports_real_file_line(tmp_path):
    diff = ("diff --git a/src/payments/charge.py b/src/payments/charge.py\n"
            "--- a/src/payments/charge.py\n+++ b/src/payments/charge.py\n"
            "@@ -40,3 +40,4 @@ def charge():\n"
            "     total = 1\n     tax = 2\n"
            "+    from src.analytics.tracker import track\n"
            "     return total\n")
    report = _arch(diff, _repo(tmp_path))
    assert report["violations"][0]["line"] == 42


def test_pr_editing_policy_is_flagged(tmp_path):
    diff = ("diff --git a/.agentprobe/boundaries.yaml b/.agentprobe/boundaries.yaml\n"
            "--- a/.agentprobe/boundaries.yaml\n+++ b/.agentprobe/boundaries.yaml\n"
            "@@ -1,2 +1,2 @@\n modules:\n-  payments: {forbidden_imports: [analytics]}\n"
            "+  payments: {forbidden_imports: []}\n")
    report = _arch(diff, _repo(tmp_path))
    assert report["status"] == "WARN"
    assert "governance policy file" in report["violations"][0]["description"]


# ── pattern rules ──


STYLE = """\
naming:
  functions: snake_case
  classes: PascalCase
  files: snake_case
  typescript:
    functions: camelCase
    files: kebab-case
forbidden:
  - console.log-in-production-code
  - name: no-print
    pattern: '\\bprint\\('
    message: Use the logger instead of print()
    files: "*.py"
  - name: no-eval
    pattern: '\\beval\\('
    severity: FATAL
  - name: broken
    pattern: '([unclosed'
  - not-a-real-builtin
"""


def test_custom_forbidden_regex_with_file_glob(tmp_path):
    repo = _repo(tmp_path, style=STYLE)
    py = _pattern(_new_file("src/report.py", "def run():", "    print('hi')"), repo)
    assert [v["rule_violated"] for v in py["violations"]] == ["forbidden.no-print"]
    assert py["violations"][0]["line_number"] == 2
    assert "Use the logger" in py["violations"][0]["description"]
    js = _pattern(_new_file("src/report-view.ts", "function run() {", "  print('hi');", "}"),
                  repo)
    assert js["violations"] == []  # glob limits no-print to *.py


def test_fatal_forbidden_pattern_blocks(tmp_path):
    repo = _repo(tmp_path, style=STYLE)
    report = _pattern(_new_file("src/calc.py", "def calc(expr):", "    return eval(expr)"), repo)
    assert report["status"] == "FATAL"


def test_per_language_naming(tmp_path):
    repo = _repo(tmp_path, style=STYLE)
    ts = _pattern(_new_file("src/order-view.ts", "function loadOrder() { return 1; }"), repo)
    assert ts["status"] == "PASS", ts["violations"]
    py = _pattern(_new_file("src/order_view.py", "def loadOrder():", "    return 1"), repo)
    assert [v["rule_violated"] for v in py["violations"]] == ["naming.functions"]


def test_dunder_methods_are_not_naming_violations(tmp_path):
    repo = _repo(tmp_path, style="naming:\n  functions: camelCase\n")
    report = _pattern(_new_file("src/x.py", "class A:", "    def __init__(self):",
                                "        pass"), repo)
    assert report["violations"] == []


def test_filename_only_checked_for_new_files(tmp_path):
    repo = _repo(tmp_path, style="naming:\n  files: kebab-case\n")
    diff = ("diff --git a/src/legacy_name.py b/src/legacy_name.py\n"
            "--- a/src/legacy_name.py\n+++ b/src/legacy_name.py\n@@ -1 +1,2 @@\n x = 1\n+y = 2\n")
    assert _pattern(diff, repo)["violations"] == []


def test_full_file_context_prevents_docstring_false_positive(tmp_path):
    """An edited docstring line like 'import statements ...' is not an import."""
    after = ('def run():\n    """Checks things.\n\n'
             '    import statements from analytics are checked here\n    """\n    return 1\n')
    diff = ("diff --git a/src/payments/charge.py b/src/payments/charge.py\n"
            "--- a/src/payments/charge.py\n+++ b/src/payments/charge.py\n"
            "@@ -2,3 +2,3 @@ def run():\n"
            '     """Checks things.\n \n'
            "-    old wording\n"
            "+    import statements from analytics are checked here\n")
    repo = _repo(tmp_path, style="imports:\n  order: [builtin, external, internal]\n")
    state = {"pr_diff": diff, "repo_path": repo,
             "file_sources": {"src/payments/charge.py": {"before": None, "after": after}}}
    assert architecture_agent.run(dict(state))["architecture_report"]["violations"] == []
    assert pattern_agent.run(dict(state))["pattern_report"]["violations"] == []


def test_full_file_mode_reports_only_added_definitions(tmp_path):
    after = "def oldName():\n    return 1\n\n\ndef newName():\n    return 2\n"
    diff = ("diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n"
            "@@ -3,0 +4,3 @@\n+\n+def newName():\n+    return 2\n")
    repo = _repo(tmp_path, style="naming:\n  functions: snake_case\n")
    state = {"pr_diff": diff, "repo_path": repo,
             "file_sources": {"src/x.py": {"before": None, "after": after}}}
    violations = pattern_agent.run(state)["pattern_report"]["violations"]
    assert [(v["line_number"], v["description"]) for v in violations] == [
        (5, "Function 'newName' is camelCase, expected snake_case")]


def test_function_local_imports_are_not_order_checked(tmp_path):
    repo = _repo(tmp_path, style="imports:\n  order: [builtin, external, internal]\n")
    diff = _new_file("src/x.py", "import os", "import yaml", "", "def f():",
                     "    import json", "    return json")
    assert _pattern(diff, repo)["violations"] == []


def test_module_paths_support_src_layout(tmp_path):
    boundaries = ("modules:\n"
                  "  payments: {paths: [src/shop/payments], forbidden_imports: [analytics]}\n"
                  "  analytics: {paths: [src/shop/analytics]}\n")
    diff = _new_file("src/shop/payments/refunds.py", "from shop.analytics.tracker import track")
    report = _arch(diff, _repo(tmp_path, boundaries))
    assert report["status"] == "FATAL"
    assert "'payments' is forbidden from importing 'analytics'" in \
        report["violations"][0]["description"]
