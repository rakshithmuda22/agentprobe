"""Tests for the Regression Agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agents.regression_agent import (
    LogicSummarizer,
    PropertyTestGenerator,
    BehavioralFingerprinter,
    SemanticDelta,
    run,
    _global_cache,
)
from src.cache.memory_cache import MemoryCache
from src.graph.state import AgentProbeState


FIXTURES = Path(__file__).parent / "fixtures" / "semantic_regression_repo"

BEFORE_PAYMENT = (FIXTURES / "before" / "process_payment.py").read_text()
AFTER_PAYMENT = (FIXTURES / "after" / "process_payment.py").read_text()
BEFORE_SORT = (FIXTURES / "before" / "sort_items.py").read_text()
AFTER_SORT = (FIXTURES / "after" / "sort_items.py").read_text()


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


# ── Layer 1: Logic Summary Diffing ──────────────────────────────


class TestLogicSummarizer:
    def setup_method(self):
        self.cache = MemoryCache()
        self.summarizer = LogicSummarizer(self.cache)

    def test_deterministic_summary_detects_raise(self):
        """Deterministic summary should detect 'raise' statements."""
        source = """\
def process_payment(amount, currency="USD"):
    if amount <= 0:
        raise PaymentError(f"Invalid amount: {amount}")
    return {"status": "ok", "amount": amount}
"""
        summary = LogicSummarizer._deterministic_summary(source)
        assert any("raises" in str(e) for e in summary["error_paths"])
        assert summary["inputs"] == ["amount", "currency"]

    def test_deterministic_summary_detects_return_none(self):
        """Deterministic summary should detect 'return None' patterns."""
        source = """\
def process_payment(amount, currency="USD"):
    if amount <= 0:
        return None
    return {"status": "ok", "amount": amount}
"""
        summary = LogicSummarizer._deterministic_summary(source)
        assert any("None" in str(e) for e in summary["error_paths"])

    def test_compare_detects_error_path_change(self):
        """Comparing raise vs return-None should produce a delta."""
        before = """\
def process(x):
    if x < 0:
        raise ValueError("bad")
    return x * 2
"""
        after = """\
def process(x):
    if x < 0:
        return None
    return x * 2
"""
        deltas = self.summarizer.compare(before, after)
        assert len(deltas) > 0
        error_deltas = [d for d in deltas if d.field == "error_paths"]
        assert len(error_deltas) >= 1
        assert error_deltas[0].severity == "HIGH"

    def test_compare_identical_no_deltas(self):
        """Comparing identical functions should produce no deltas."""
        source = "def foo(x):\n    return x + 1\n"
        deltas = self.summarizer.compare(source, source)
        assert deltas == []

    def test_cache_hit(self):
        """Second call with same source should use cached summary."""
        source = "def foo(x):\n    return x + 1\n"
        summary1 = self.summarizer.summarize(source)
        summary2 = self.summarizer.summarize(source)
        assert summary1 == summary2
        # Cache should have exactly 1 entry
        assert len(self.cache) == 1

    def test_ollama_fallback_on_connection_error(self):
        """When Ollama is unreachable, should fall back to deterministic."""
        source = "def foo(x):\n    return x + 1\n"
        # Default localhost:11434 is not running in test
        summary = self.summarizer.summarize(source)
        assert "inputs" in summary
        assert "error_paths" in summary


# ── Layer 2: Property-Based Testing ─────────────────────────────


class TestPropertyTestGenerator:
    def setup_method(self):
        self.generator = PropertyTestGenerator()

    def test_generates_valid_python(self):
        """Generated test code should be valid Python syntax."""
        deltas = [SemanticDelta(
            function_name="process",
            file_path="test.py",
            field="error_paths",
            before="raises: ValueError",
            after="returns None on error",
            severity="HIGH",
        )]
        code = self.generator._generate_test_code(
            "def process(x):\n    if x < 0:\n        return None\n    return x * 2\n",
            "process",
            deltas,
        )
        assert code
        # Check it's valid Python
        compile(code, "<test>", "exec")

    def test_no_deltas_no_tests(self):
        """No deltas should produce no test code."""
        result = self.generator.generate_and_run("def foo(x): return x", "foo", [])
        assert result == []


# ── Layer 3: Behavioral Fingerprinting ──────────────────────────


class TestBehavioralFingerprinter:
    def setup_method(self):
        self.fingerprinter = BehavioralFingerprinter(timing_threshold=0.20)

    def test_identical_functions_low_divergence(self):
        """Identical functions should have near-zero divergence."""
        source = "def add(x):\n    return x + 1\n"
        divergence = self.fingerprinter.fingerprint(source, source, "add")
        assert divergence is not None
        assert divergence < 0.5  # Should be very close to 0

    def test_different_outputs_high_divergence(self):
        """Functions with different outputs should have divergence >= 1.0."""
        before = "def calc(x):\n    return x * 2\n"
        after = "def calc(x):\n    return x * 3\n"
        divergence = self.fingerprinter.fingerprint(before, after, "calc")
        assert divergence is not None
        assert divergence >= 1.0

    def test_error_vs_none_divergence(self):
        """Function that raises vs returns None should show divergence."""
        before = "def check(x):\n    if x < 0:\n        raise ValueError('bad')\n    return x\n"
        after = "def check(x):\n    if x < 0:\n        return None\n    return x\n"
        divergence = self.fingerprinter.fingerprint(before, after, "check")
        assert divergence is not None
        assert divergence >= 1.0


# ── Full Regression Agent ───────────────────────────────────────


def test_semantic_change_detected(semantic_change_diff):
    """Modifying error handling should produce WARN with deltas."""
    _global_cache.clear()
    state = _make_state(semantic_change_diff)
    result = run(state)

    report = result["regression_report"]
    assert report["status"] == "WARN"
    assert len(report["deltas"]) >= 1

    # Should detect the error_paths change
    error_deltas = [d for d in report["deltas"] if d["field"] == "error_paths"]
    assert len(error_deltas) >= 1


def test_clean_modification_passes(clean_import_diff):
    """A diff with only new code (no before version) should PASS."""
    _global_cache.clear()
    state = _make_state(clean_import_diff)
    result = run(state)

    report = result["regression_report"]
    assert report["status"] == "PASS"
    assert report["deltas"] == []


def test_empty_diff_passes():
    """Empty diff should PASS."""
    _global_cache.clear()
    state = _make_state("")
    result = run(state)

    assert result["regression_report"]["status"] == "PASS"


def test_cost_tracking(semantic_change_diff):
    """LLM call count should be tracked in cost_tracker."""
    _global_cache.clear()
    state = _make_state(semantic_change_diff)
    result = run(state)

    assert "regression_llm_calls" in result["cost_tracker"]
