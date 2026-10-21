"""Tests for the regression analyzer (Layer 1 summary diff, Layer 2 property tests)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agentprobe.agents import pbt_harness
from agentprobe.agents.regression_agent import (
    LogicSummarizer,
    PropertyTestRunner,
    run,
    _global_cache,
)
from agentprobe.cache.memory_cache import MemoryCache
from agentprobe.graph.state import AgentProbeState


FIXTURES = Path(__file__).parent / "fixtures" / "semantic_regression_repo"
SAMPLE_REPO = str(Path(__file__).parent / "fixtures" / "sample_policy")

BEFORE_PAYMENT = (FIXTURES / "before" / "process_payment.py").read_text()
AFTER_PAYMENT = (FIXTURES / "after" / "process_payment.py").read_text()


def _make_state(pr_diff: str, repo_path: str = SAMPLE_REPO, **extra) -> AgentProbeState:
    state = AgentProbeState(
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
    state.update(extra)
    return state


def _policy(tmp_path: Path, config: str) -> str:
    d = tmp_path / ".agentprobe"
    d.mkdir(exist_ok=True)
    (d / "config.yaml").write_text(config)
    return str(tmp_path)


# A body-only edit: the `def` line is far outside the 3-line hunk context.
BODY_ONLY_DIFF = """\
diff --git a/src/payments/process_payment.py b/src/payments/process_payment.py
--- a/src/payments/process_payment.py
+++ b/src/payments/process_payment.py
@@ -9,6 +9,6 @@ def process_payment(amount, currency="USD"):
     \"\"\"Process a payment. Raises PaymentError on invalid amount.\"\"\"
     if amount <= 0:
-        raise PaymentError(f"Invalid amount: {amount}")
+        return None
     if not isinstance(amount, (int, float)):
-        raise PaymentError(f"Amount must be numeric, got {type(amount)}")
+        return None
     return {"status": "ok", "amount": amount, "currency": currency}
"""
FULL_SOURCES = {"src/payments/process_payment.py": {"before": BEFORE_PAYMENT,
                                                    "after": AFTER_PAYMENT}}


# ── Layer 1: Logic Summary Diffing ──────────────────────────────


class TestLogicSummarizer:
    def setup_method(self):
        self.cache = MemoryCache()
        self.summarizer = LogicSummarizer(self.cache)

    def test_deterministic_summary_detects_raise(self):
        source = """\
def process_payment(amount, currency="USD"):
    if amount <= 0:
        raise PaymentError(f"Invalid amount: {amount}")
    return {"status": "ok", "amount": amount}
"""
        summary = LogicSummarizer._deterministic_summary(source)
        assert any("raise PaymentError" in str(e) for e in summary["error_paths"])
        assert summary["inputs"] == ["amount", "currency"]

    def test_deterministic_summary_detects_return_none(self):
        source = """\
def process_payment(amount, currency="USD"):
    if amount <= 0:
        return None
    return {"status": "ok", "amount": amount}
"""
        summary = LogicSummarizer._deterministic_summary(source)
        assert any("None" in str(e) for e in summary["error_paths"])

    def test_compare_detects_error_path_change(self):
        before = "def process(x):\n    if x < 0:\n        raise ValueError('bad')\n    return x * 2\n"
        after = "def process(x):\n    if x < 0:\n        return None\n    return x * 2\n"
        deltas = self.summarizer.compare(before, after)
        error_deltas = [d for d in deltas if d.field == "error_paths"]
        assert len(error_deltas) == 1
        assert error_deltas[0].severity == "HIGH"
        assert error_deltas[0].before == "raise ValueError('bad')"
        assert error_deltas[0].after == "returns None"

    def test_compare_detects_signature_change(self):
        deltas = self.summarizer.compare("def f(a):\n    return a\n",
                                         "def f(a, b):\n    return a\n")
        assert [d.field for d in deltas] == ["inputs"]
        assert deltas[0].severity == "MEDIUM"

    def test_compare_identical_no_deltas(self):
        source = "def foo(x):\n    return x + 1\n"
        assert self.summarizer.compare(source, source) == []

    def test_cache_hit(self):
        source = "def foo(x):\n    return x + 1\n"
        assert self.summarizer.summarize(source) == self.summarizer.summarize(source)
        assert len(self.cache) == 1

    def test_no_llm_by_default(self):
        """provider defaults to 'none': no network call, fully deterministic."""
        self.summarizer.summarize("def foo(x):\n    return x + 1\n")
        assert self.summarizer.llm_calls == 0


# ── Full regression analyzer ────────────────────────────────────


def test_semantic_change_detected(semantic_change_diff):
    """Modifying error handling should produce WARN with deltas."""
    _global_cache.clear()
    report = run(_make_state(semantic_change_diff))["regression_report"]
    assert report["status"] == "WARN"
    assert any(d["field"] == "error_paths" for d in report["deltas"])


def test_body_only_change_missed_without_full_sources():
    """Hunks alone don't contain the `def` line, so nothing can be compared."""
    _global_cache.clear()
    report = run(_make_state(BODY_ONLY_DIFF))["regression_report"]
    assert report["deltas"] == []


def test_body_only_change_detected_with_full_sources():
    """With full before/after files the whole function is compared."""
    _global_cache.clear()
    report = run(_make_state(BODY_ONLY_DIFF, file_sources=FULL_SOURCES))["regression_report"]
    fields = {d["field"] for d in report["deltas"] if d["function"] == "process_payment"}
    assert "error_paths" in fields
    assert report["status"] == "WARN"


def test_clean_modification_passes(clean_import_diff):
    _global_cache.clear()
    report = run(_make_state(clean_import_diff))["regression_report"]
    assert report["status"] == "PASS"
    assert report["deltas"] == []


def test_empty_diff_passes():
    _global_cache.clear()
    assert run(_make_state(""))["regression_report"]["status"] == "PASS"


def test_cost_tracking(semantic_change_diff):
    _global_cache.clear()
    result = run(_make_state(semantic_change_diff))
    assert result["cost_tracker"]["regression_llm_calls"] == 0
    assert result["cost_tracker"]["regression_functions_compared"] == 1
    assert result["cost_tracker"]["regression_property_test_runs"] == 0


def test_function_budget(tmp_path, semantic_change_diff):
    _global_cache.clear()
    repo = _policy(tmp_path, "regression:\n  max_functions_per_pr: 0\n")
    report = run(_make_state(semantic_change_diff, repo))["regression_report"]
    assert report["deltas"] == []
    assert any("max_functions_per_pr" in n for n in report["notes"])


# ── Layer 2: Differential property tests ────────────────────────


PBT_CONFIG = ("regression:\n  property_tests: true\n  critical_dirs: []\n"
              "  property_test_examples: 50\n")


def test_property_tests_off_by_default():
    _global_cache.clear()
    result = run(_make_state(BODY_ONLY_DIFF, file_sources=FULL_SOURCES))
    assert not any(d["field"] == "property_test" for d in result["regression_report"]["deltas"])
    assert result["cost_tracker"]["regression_property_test_runs"] == 0


def test_property_test_finds_minimal_counterexample(tmp_path):
    _global_cache.clear()
    repo = _policy(tmp_path, PBT_CONFIG)
    report = run(_make_state(BODY_ONLY_DIFF, repo, file_sources=FULL_SOURCES))[
        "regression_report"]
    pbt = [d for d in report["deltas"] if d["field"] == "property_test"]
    assert len(pbt) == 1
    # Hypothesis shrinks to the boundary input; PaymentError is resolved from the module.
    assert "process_payment(amount=0)" in pbt[0]["summary"]
    assert pbt[0]["before"] == "raised PaymentError"
    assert pbt[0]["after"] == "returned None"
    assert pbt[0]["severity"] == "HIGH"


def test_property_tests_refused_when_code_execution_disallowed(tmp_path):
    _global_cache.clear()
    repo = _policy(tmp_path, PBT_CONFIG)
    result = run(_make_state(BODY_ONLY_DIFF, repo, file_sources=FULL_SOURCES,
                             allow_code_execution=False))
    report = result["regression_report"]
    assert not any(d["field"] == "property_test" for d in report["deltas"])
    assert result["cost_tracker"]["regression_property_test_runs"] == 0
    assert any("pull_request_target" in n for n in report["notes"])


def test_property_test_child_gets_scrubbed_environment(monkeypatch):
    """Code under test must not see the parent's secrets (e.g. GITHUB_TOKEN)."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_should_not_leak")
    before = "def peek(x):\n    return None\n"
    after = "import os\n\ndef peek(x):\n    return os.environ.get('GITHUB_TOKEN')\n"
    results = PropertyTestRunner(max_examples=10, timeout=60).run(
        "src/peek.py", before, after, ["peek"], repo_root=None)
    assert results == [{"function": "peek", "status": "equivalent", "examples": 10}]


def test_property_test_skips_methods_and_unknown_types():
    before = ("class Cart:\n    pass\n\n"
              "def total(cart: Cart):\n    return 1\n")
    after = ("class Cart:\n    pass\n\n"
             "def total(cart: Cart):\n    return 2\n")
    results = PropertyTestRunner(max_examples=10, timeout=60).run(
        "src/cart.py", before, after, ["total"], repo_root=None)
    assert results[0]["status"] == "skipped"
    assert "cart" in results[0]["reason"]


def test_harness_drops_top_level_side_effects():
    source = ("import os\nimport not_installed_pkg\n"
              "print('side effect')\nos.system('echo pwned')\n"
              "def f(x):\n    return x\n")
    tree = pbt_harness.reduce_module(source, "m.py")
    kept = [type(n.body[0]).__name__ for n in tree.body]
    assert kept == ["Import", "Import", "FunctionDef"]
    ns = pbt_harness.load_version(source, "m.py", "", "after")
    assert ns["f"](3) == 3
    assert isinstance(ns["not_installed_pkg"], pbt_harness._Unavailable)


@pytest.mark.parametrize("name", ["", "bad name", "x;import os", "a" * 300])
def test_invalid_function_names_are_never_sent_to_the_harness(name):
    assert PropertyTestRunner().run("m.py", "def f(): pass", "def f(): pass", [name],
                                    None) == []


def test_reduce_module_output_is_valid_python():
    tree = pbt_harness.reduce_module(BEFORE_PAYMENT, "p.py")
    compile(ast.unparse(tree), "p.py", "exec")
