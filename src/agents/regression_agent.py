"""Regression Agent — detects semantic behavior changes in modified functions.

Three-layer detection:
  Layer 1: Logic Summary Diffing (LLM or deterministic fallback)
  Layer 2: Property-Based Testing (Hypothesis, only if Layer 1 finds deltas)
  Layer 3: Behavioral Fingerprinting (only for critical dirs)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

# Security limits
MAX_FUNCTION_SOURCE_LEN = 50_000  # max chars for a single function source
MAX_FUNCTION_NAME_LEN = 200
FUNCTION_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

import httpx

from src.cache.hash_utils import cache_key, function_hash
from src.cache.memory_cache import MemoryCache
from src.config.loader import load_config
from src.graph.state import AgentProbeState
from src.parsers.diff_parser import parse_diff
from src.parsers.tree_sitter_engine import TreeSitterEngine


# ── Dataclasses ─────────────────────────────────────────────────


@dataclass
class FunctionVersion:
    """Before/after versions of a modified function."""
    name: str
    file_path: str
    before_source: str
    after_source: str


@dataclass
class SemanticDelta:
    """A detected semantic change between function versions."""
    function_name: str
    file_path: str
    field: str
    before: str
    after: str
    severity: str  # "HIGH", "MEDIUM", "LOW"


@dataclass
class RegressionResult:
    """Result from analyzing a single function."""
    function_name: str
    file_path: str
    deltas: list[SemanticDelta] = field(default_factory=list)
    property_test_failures: list[str] = field(default_factory=list)
    fingerprint_divergence: float | None = None


def _validate_function_name(name: str) -> bool:
    """Validate function name to prevent code injection."""
    return (
        bool(name)
        and len(name) <= MAX_FUNCTION_NAME_LEN
        and FUNCTION_NAME_PATTERN.match(name) is not None
    )


def _validate_source(source: str) -> bool:
    """Validate source length to prevent DoS."""
    return bool(source) and len(source) <= MAX_FUNCTION_SOURCE_LEN


# ── Layer 1: Logic Summary Diffing ──────────────────────────────


class LogicSummarizer:
    """Summarize function behavior and detect semantic deltas."""

    def __init__(self, cache: MemoryCache, llm_base_url: str = "http://localhost:11434",
                 llm_model: str = "llama3"):
        self.cache = cache
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model

    def summarize(self, source: str) -> dict:
        """Generate a behavior summary for a function.

        Uses Ollama if available, falls back to deterministic AST analysis.
        """
        h = function_hash(source)
        key = cache_key("summary", h)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        summary = self._try_ollama_summary(source)
        if summary is None:
            summary = self._deterministic_summary(source)

        self.cache.set(key, summary, ttl=86400)
        return summary

    def compare(self, before_source: str, after_source: str) -> list[SemanticDelta]:
        """Compare before/after function summaries and return semantic deltas."""
        before_summary = self.summarize(before_source)
        after_summary = self.summarize(after_source)

        deltas = []
        for field_name in ("error_paths", "side_effects", "return_behavior", "invariants"):
            before_val = before_summary.get(field_name, "")
            after_val = after_summary.get(field_name, "")
            if str(before_val) != str(after_val):
                severity = "HIGH" if field_name in ("error_paths", "return_behavior") else "MEDIUM"
                deltas.append(SemanticDelta(
                    function_name="",  # Filled by caller
                    file_path="",
                    field=field_name,
                    before=str(before_val),
                    after=str(after_val),
                    severity=severity,
                ))
        return deltas

    def _try_ollama_summary(self, source: str) -> dict | None:
        """Try to use Ollama for summarization. Returns None if unavailable."""
        prompt = textwrap.dedent(f"""\
            Analyze this function and return a JSON object with these fields:
            - inputs: list of parameter names and types
            - outputs: description of return value
            - side_effects: list of side effects
            - error_paths: how errors are handled (exceptions raised, None returned, etc)
            - return_behavior: what the function returns in normal and error cases
            - invariants: properties that should always hold

            Function:
            ```
            {source}
            ```

            Return ONLY valid JSON, no other text.
        """)

        try:
            resp = httpx.post(
                f"{self.llm_base_url}/api/generate",
                json={"model": self.llm_model, "prompt": prompt, "stream": False},
                timeout=30.0,
            )
            if resp.status_code == 200:
                response_text = resp.json().get("response", "")
                return self._parse_json_response(response_text)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return None
        return None

    @staticmethod
    def _parse_json_response(text: str) -> dict | None:
        """Extract JSON from LLM response text."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _deterministic_summary(source: str) -> dict:
        """Deterministic fallback: analyze function structure without LLM."""
        lines = source.strip().split("\n")

        error_paths = []
        return_behavior = []
        side_effects = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("raise "):
                error_paths.append(f"raises: {stripped}")
            elif stripped == "return None" or stripped == "return":
                error_paths.append("returns None on error")
                return_behavior.append("returns None")
            elif stripped.startswith("return "):
                return_behavior.append(stripped)
            elif "print(" in stripped or "log(" in stripped:
                side_effects.append(stripped)

        # Extract parameters
        inputs = []
        if lines and "def " in lines[0]:
            param_match = re.search(r'\((.*?)\)', lines[0])
            if param_match:
                params = param_match.group(1)
                inputs = [p.strip().split(":")[0].strip().split("=")[0].strip()
                         for p in params.split(",") if p.strip() and p.strip() != "self"]

        return {
            "inputs": inputs,
            "outputs": return_behavior or ["unknown"],
            "side_effects": side_effects,
            "error_paths": error_paths or ["none detected"],
            "return_behavior": return_behavior or ["unknown"],
            "invariants": [],
        }


# ── Layer 2: Property-Based Testing ────────────────────────────


class PropertyTestGenerator:
    """Generate and run Hypothesis property tests for functions with deltas."""

    def generate_and_run(self, function_source: str, function_name: str,
                         deltas: list[SemanticDelta]) -> list[str]:
        """Generate property tests based on deltas, run them, return failures."""
        if not deltas:
            return []

        if not _validate_function_name(function_name):
            return [f"Invalid function name: {function_name!r}"]
        if not _validate_source(function_source):
            return ["Function source too large or empty"]

        test_code = self._generate_test_code(function_source, function_name, deltas)
        if not test_code:
            return []

        return self._run_tests(test_code)

    def _generate_test_code(self, function_source: str, function_name: str,
                            deltas: list[SemanticDelta]) -> str:
        """Generate Hypothesis test code as a Python string."""
        test_parts = [
            "from hypothesis import given, strategies as st, settings",
            "",
            "# Function under test",
            function_source,
            "",
        ]

        test_count = 0
        for delta in deltas:
            if delta.field == "error_paths" and "raises" in delta.before:
                test_parts.append(textwrap.dedent(f"""\
                    @settings(max_examples=20)
                    @given(x=st.integers(max_value=-1))
                    def test_{function_name}_error_path_{test_count}(x):
                        result = {function_name}(x)
                        assert result is not None, "Function returns None instead of raising"
                """))
                test_count += 1
            elif delta.field == "return_behavior":
                test_parts.append(textwrap.dedent(f"""\
                    @settings(max_examples=20)
                    @given(x=st.integers(min_value=1, max_value=1000))
                    def test_{function_name}_return_behavior_{test_count}(x):
                        result = {function_name}(x)
                        assert result is not None, "Function should return a value for valid input"
                """))
                test_count += 1

        if test_count == 0:
            return ""

        return "\n".join(test_parts)

    @staticmethod
    def _run_tests(test_code: str) -> list[str]:
        """Write tests to a temp file and run with pytest."""
        failures = []
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", f.name, "-v", "--tb=short", "--no-header"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    for line in result.stdout.split("\n"):
                        if "FAILED" in line:
                            failures.append(line.strip())
                    if not failures and result.stderr:
                        failures.append(f"Test execution error: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                failures.append("Property tests timed out")
            finally:
                Path(f.name).unlink(missing_ok=True)

        return failures


# ── Layer 3: Behavioral Fingerprinting ──────────────────────────


class BehavioralFingerprinter:
    """Compare function behavior via execution with canonical inputs."""

    def __init__(self, timing_threshold: float = 0.20):
        self.timing_threshold = timing_threshold

    def fingerprint(self, before_source: str, after_source: str,
                    function_name: str, test_inputs: list[dict] | None = None) -> float | None:
        """Execute both versions and compare outputs + timing.

        Returns the timing divergence ratio, or None if comparison not possible.
        """
        if not _validate_function_name(function_name):
            return None
        if not _validate_source(before_source) or not _validate_source(after_source):
            return None

        # Identical sources always match outputs; skip timing noise from subprocess runs.
        if before_source == after_source:
            return 0.0

        if test_inputs is None:
            test_inputs = [
                {"args": [1], "kwargs": {}},
                {"args": [100], "kwargs": {}},
                {"args": [-1], "kwargs": {}},
                {"args": [0], "kwargs": {}},
            ]

        before_results = self._execute_function(before_source, function_name, test_inputs)
        after_results = self._execute_function(after_source, function_name, test_inputs)

        if before_results is None or after_results is None:
            return None

        # Compare outputs
        output_mismatches = 0
        for (b_out, _b_time), (a_out, _a_time) in zip(before_results, after_results):
            if str(b_out) != str(a_out):
                output_mismatches += 1

        # Calculate timing divergence
        before_total = sum(t for _, t in before_results)
        after_total = sum(t for _, t in after_results)

        if before_total > 0:
            divergence = abs(after_total - before_total) / before_total
        else:
            divergence = 0.0

        if output_mismatches > 0:
            return max(divergence, 1.0)

        return divergence

    @staticmethod
    def _execute_function(source: str, function_name: str,
                          inputs: list[dict]) -> list[tuple] | None:
        """SECURITY: disabled.

        Executing untrusted code from pull requests inside the governance
        runner would be a remote-code-execution primitive — any PR could
        ship a malicious function body and have it run against the Action's
        GITHUB_TOKEN. Layer 3 is disabled until this runs inside a proper
        sandbox (gVisor / Firecracker / ephemeral container without
        repo secrets). Layers 1 and 2 still provide useful signal.

        Returns None so the fingerprint() caller falls through to
        "comparison not possible".
        """
        return None


# ── Main Agent ──────────────────────────────────────────────────


_global_cache = MemoryCache()


def run(state: AgentProbeState) -> AgentProbeState:
    """Run 3-layer semantic regression detection on modified functions.

    Layer 1: Logic Summary Diffing (Ollama or deterministic fallback)
    Layer 2: Property-Based Testing (only if deltas found)
    Layer 3: Behavioral Fingerprinting (only for critical dirs)
    """
    pr_diff = state.get("pr_diff", "")
    repo_path = state.get("repo_path", ".")

    # Security: reject excessively large diffs to prevent DoS
    MAX_DIFF_SIZE = 1_000_000  # 1MB
    if len(pr_diff) > MAX_DIFF_SIZE:
        state["regression_report"] = {
            "status": "WARN",
            "deltas": [{"function": "N/A", "file": "N/A", "field": "diff_size",
                        "before": "normal", "after": f"diff too large ({len(pr_diff)} bytes)",
                        "severity": "MEDIUM"}],
            "confidence": 1.0,
        }
        return state

    config = load_config(Path(repo_path) / ".agentprobe" / "config.yaml")
    diff_result = parse_diff(pr_diff)
    engine = TreeSitterEngine()

    summarizer = LogicSummarizer(
        cache=_global_cache,
        llm_base_url=config.llm.base_url,
        llm_model=config.llm.model,
    )
    property_tester = PropertyTestGenerator()
    fingerprinter = BehavioralFingerprinter(
        timing_threshold=config.regression.timing_divergence_threshold,
    )

    all_results: list[RegressionResult] = []
    cost_tracker = state.get("cost_tracker", {})
    llm_calls = 0

    for file_change in diff_result.files:
        language = engine.detect_language(file_change.file_path)
        if language is None:
            continue

        # Before = deleted lines, After = added lines
        before_source = "\n".join(line for _, line in file_change.deleted_lines)
        after_source = "\n".join(line for _, line in file_change.added_lines)

        if not before_source.strip() or not after_source.strip():
            continue

        try:
            before_funcs = {f.name: f for f in engine.extract_functions(before_source, language)}
            after_funcs = {f.name: f for f in engine.extract_functions(after_source, language)}
        except Exception:
            continue

        # Find functions that exist in both versions (modified)
        common_funcs = set(before_funcs.keys()) & set(after_funcs.keys())

        for func_name in common_funcs:
            if llm_calls >= config.regression.max_llm_calls_per_pr:
                break

            before_func_source = _extract_func_source(before_source, before_funcs[func_name])
            after_func_source = _extract_func_source(after_source, after_funcs[func_name])

            if before_func_source == after_func_source:
                continue

            result = RegressionResult(
                function_name=func_name,
                file_path=file_change.file_path,
            )

            # Layer 1: Logic Summary Diffing
            deltas = summarizer.compare(before_func_source, after_func_source)
            for d in deltas:
                d.function_name = func_name
                d.file_path = file_change.file_path
            result.deltas = deltas
            llm_calls += 1

            # Layer 2: Property-Based Testing (only if deltas found)
            if deltas:
                failures = property_tester.generate_and_run(
                    after_func_source, func_name, deltas
                )
                result.property_test_failures = failures

            # Layer 3: Behavioral Fingerprinting (only for critical dirs)
            is_critical = any(
                file_change.file_path.startswith(d)
                for d in config.regression.critical_dirs
            )
            if is_critical:
                divergence = fingerprinter.fingerprint(
                    before_func_source, after_func_source, func_name
                )
                result.fingerprint_divergence = divergence

            all_results.append(result)

    # Aggregate results
    all_deltas = []
    has_issues = False
    for r in all_results:
        for d in r.deltas:
            all_deltas.append({
                "function": d.function_name,
                "file": d.file_path,
                "field": d.field,
                "before": d.before,
                "after": d.after,
                "severity": d.severity,
            })
            if d.severity == "HIGH":
                has_issues = True

        if r.property_test_failures:
            has_issues = True
            for f in r.property_test_failures:
                all_deltas.append({
                    "function": r.function_name,
                    "file": r.file_path,
                    "field": "property_test",
                    "before": "passing",
                    "after": f,
                    "severity": "HIGH",
                })

        if (r.fingerprint_divergence is not None
                and r.fingerprint_divergence > fingerprinter.timing_threshold):
            has_issues = True
            all_deltas.append({
                "function": r.function_name,
                "file": r.file_path,
                "field": "behavioral_fingerprint",
                "before": "baseline",
                "after": f"divergence: {r.fingerprint_divergence:.2%}",
                "severity": "MEDIUM" if r.fingerprint_divergence < 1.0 else "HIGH",
            })

    status = "WARN" if has_issues else "PASS"
    confidence = 0.85 if llm_calls > 0 else 1.0

    cost_tracker["regression_llm_calls"] = llm_calls
    state["cost_tracker"] = cost_tracker

    state["regression_report"] = {
        "status": status,
        "deltas": all_deltas,
        "confidence": confidence,
    }
    return state


def _extract_func_source(full_source: str, func_def) -> str:
    """Extract a function's source code from the full source."""
    lines = full_source.split("\n")
    start = func_def.line_start - 1
    end = func_def.line_end
    return "\n".join(lines[start:end])
