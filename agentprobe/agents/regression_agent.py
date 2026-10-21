"""Regression analyzer — flags behavior changes in functions a PR modifies.

Layer 1 — logic-summary diffing (always on, deterministic):
    For every function present before and after the PR whose source changed,
    build a structural summary (signature, raise/throw paths, return paths,
    print/log side effects) and diff the summaries. With ``llm.provider:
    ollama`` a local model writes the summaries instead (experimental).

Layer 2 — differential property testing (opt-in: ``regression.property_tests``):
    Only for Python functions where Layer 1 found a delta. Hypothesis
    generates inputs, runs the before *and* after version on each one in an
    isolated subprocess, and shrinks to the smallest input where they differ.
    This executes PR code, so it is off by default and always refused when the
    caller sets ``allow_code_execution=False`` (e.g. ``pull_request_target``).

An earlier "Layer 3" (timing fingerprints by executing code without any
isolation) was removed; see README roadmap.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import httpx

from agentprobe.cache.hash_utils import cache_key, function_hash
from agentprobe.cache.memory_cache import MemoryCache
from agentprobe.config.loader import RegressionConfig, load_config, policy_dir
from agentprobe.graph.state import AgentProbeState
from agentprobe.parsers.diff_parser import parse_diff
from agentprobe.parsers.tree_sitter_engine import TreeSitterEngine

# Security / DoS limits
MAX_FUNCTION_SOURCE_LEN = 50_000   # max chars for a single function source
MAX_MODULE_SOURCE_LEN = 500_000    # max chars for a module sent to Layer 2
MAX_DIFF_SIZE = 1_000_000          # 1MB
MAX_FUNCTION_NAME_LEN = 200
FUNCTION_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

HARNESS_PATH = Path(__file__).with_name("pbt_harness.py")
RESULT_MARKER = "AGENTPROBE_PBT_RESULT:"

SUMMARY_FIELDS = ("inputs", "error_paths", "return_behavior", "side_effects")
FIELD_SEVERITY = {"error_paths": "HIGH", "return_behavior": "HIGH",
                  "inputs": "MEDIUM", "side_effects": "MEDIUM"}


# ── Dataclasses ─────────────────────────────────────────────────


@dataclass
class SemanticDelta:
    """A detected semantic change between function versions."""
    function_name: str
    file_path: str
    field: str
    before: str
    after: str
    severity: str  # "HIGH", "MEDIUM", "LOW"
    summary: str = ""


@dataclass
class RegressionResult:
    """Result from analyzing a single function."""
    function_name: str
    file_path: str
    line: int = 0  # first line of the function in the post-PR file (0 = unknown)
    deltas: list[SemanticDelta] = field(default_factory=list)


def _validate_function_name(name: str) -> bool:
    """Function names are interpolated into reports and payloads; keep them sane."""
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
                 llm_model: str = "llama3", llm_provider: str = "none"):
        self.cache = cache
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm_provider = (llm_provider or "none").lower()
        self.llm_calls = 0

    def summarize(self, source: str) -> dict:
        """Generate a behavior summary for a function (cached by content hash)."""
        key = cache_key(f"summary:{self.llm_provider}", function_hash(source))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        summary = None
        if self.llm_provider == "ollama":
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
        for field_name in SUMMARY_FIELDS:
            before_val = _render(before_summary.get(field_name, ""))
            after_val = _render(after_summary.get(field_name, ""))
            if before_val != after_val:
                deltas.append(SemanticDelta(
                    function_name="",  # Filled by caller
                    file_path="",
                    field=field_name,
                    before=before_val,
                    after=after_val,
                    severity=FIELD_SEVERITY.get(field_name, "MEDIUM"),
                ))
        return deltas

    def _try_ollama_summary(self, source: str) -> dict | None:
        """Ask a local Ollama model for a summary. Returns None if unavailable."""
        prompt = textwrap.dedent(f"""\
            Analyze this function and return a JSON object with these fields:
            - inputs: list of parameter names
            - side_effects: list of side effects
            - error_paths: how errors are handled (exceptions raised, None returned, etc)
            - return_behavior: what the function returns in normal and error cases

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
                self.llm_calls += 1
                return self._parse_json_response(resp.json().get("response", ""))
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError, ValueError):
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
        """Structural summary without an LLM: signature, raise/return paths, I/O calls."""
        lines = source.strip().split("\n")

        error_paths = []
        return_behavior = []
        side_effects = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("raise ", "throw ")) or stripped in ("raise", "throw"):
                error_paths.append(stripped.rstrip(";"))
            elif stripped in ("return None", "return", "return null", "return undefined",
                              "return None;", "return;", "return null;"):
                error_paths.append("returns None")
            elif stripped.startswith("return "):
                return_behavior.append(stripped.rstrip(";"))
            if re.search(r"\b(print|console\.log|logger\.\w+|logging\.\w+)\s*\(", stripped):
                side_effects.append(stripped.rstrip(";"))

        inputs = []
        if lines:
            param_match = re.search(r'\((.*?)\)', lines[0])
            if param_match:
                inputs = [p.strip().split(":")[0].strip().split("=")[0].strip()
                          for p in param_match.group(1).split(",")
                          if p.strip() and p.strip() not in ("self", "cls")]

        return {
            "inputs": inputs,
            "side_effects": side_effects,
            "error_paths": error_paths or ["none detected"],
            "return_behavior": return_behavior or ["unknown"],
        }


def _render(value) -> str:
    """Human-readable, order-preserving rendering of a summary field."""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value) if value else "none"
    return str(value)


# ── Layer 2: Differential Property Testing ──────────────────────


CHILD_CPU_SECONDS = 60


def _drop_privileges() -> None:  # pragma: no cover - runs in the child process
    """preexec_fn: resource limits, and an unprivileged uid when running as root."""
    try:
        import resource

        for limit, value in ((resource.RLIMIT_CPU, CHILD_CPU_SECONDS),
                             (resource.RLIMIT_AS, 2 * 1024 ** 3),
                             (resource.RLIMIT_FSIZE, 50 * 1024 ** 2)):
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                pass
    except ImportError:
        pass
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)  # nogroup
        os.setuid(65534)  # nobody


class PropertyTestRunner:
    """Run the differential Hypothesis harness for one file in a subprocess."""

    def __init__(self, max_examples: int = 200, timeout: int = 60):
        self.max_examples = max_examples
        self.timeout = timeout
        self.runs = 0

    def run(self, file_path: str, before_source: str, after_source: str,
            functions: list[str], repo_root: str | None) -> list[dict]:
        functions = [f for f in functions if _validate_function_name(f)]
        if not functions:
            return []
        if len(before_source) > MAX_MODULE_SOURCE_LEN or len(after_source) > MAX_MODULE_SOURCE_LEN:
            return [{"function": f, "status": "skipped", "reason": "module too large"}
                    for f in functions]

        package = ".".join(PurePosixPath(file_path).parent.parts)
        payload = {
            "file_path": file_path,
            "package": package,
            "before_source": before_source,
            "after_source": after_source,
            "functions": functions,
            "max_examples": self.max_examples,
            "repo_root": str(Path(repo_root).resolve()) if repo_root else None,
        }
        with tempfile.TemporaryDirectory(prefix="agentprobe-pbt-") as tmp:
            os.chmod(tmp, 0o1777)  # writable by the unprivileged child uid
            payload_path = Path(tmp) / "payload.json"
            payload_path.write_text(json.dumps(payload))
            os.chmod(payload_path, 0o644)
            env = {
                "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                "HOME": tmp,
                "LANG": "C.UTF-8",
            }
            self.runs += 1
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(HARNESS_PATH), str(payload_path)],
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    preexec_fn=_drop_privileges if os.name == "posix" else None,
                )
            except subprocess.TimeoutExpired:
                return [{"function": f, "status": "skipped",
                         "reason": f"timed out after {self.timeout}s"} for f in functions]

        for line in reversed(proc.stdout.splitlines()):
            if line.startswith(RESULT_MARKER):
                try:
                    return json.loads(line[len(RESULT_MARKER):])
                except json.JSONDecodeError:
                    break
        error = (proc.stderr.strip().splitlines() or ["no output"])[-1][:200]
        return [{"function": f, "status": "skipped", "reason": f"harness failed: {error}"}
                for f in functions]


def _describe_outcome(outcome: list) -> str:
    kind, value = outcome[0], outcome[1]
    return f"raised {value}" if kind == "raised" else f"returned {value}"


# ── Function pairing ────────────────────────────────────────────


def _functions_by_name(engine: TreeSitterEngine, source: str,
                       language: str) -> dict[str, tuple[str, int]]:
    """{name: (source, first line)} for every function (first definition wins)."""
    lines = source.split("\n")
    out: dict[str, tuple[str, int]] = {}
    for func in engine.extract_functions(source, language):
        if func.name in out:
            continue
        out[func.name] = ("\n".join(lines[func.line_start - 1: func.line_end]),
                          func.line_start)
    return out


def _is_critical(file_path: str, cfg: RegressionConfig) -> bool:
    if not cfg.critical_dirs:
        return True
    return any(file_path == d.rstrip("/") or file_path.startswith(d.rstrip("/") + "/")
               for d in cfg.critical_dirs)


# ── Main Agent ──────────────────────────────────────────────────


_global_cache = MemoryCache()


def run(state: AgentProbeState) -> AgentProbeState:
    """Run semantic regression detection on modified functions."""
    pr_diff = state.get("pr_diff", "")
    cost_tracker = state.get("cost_tracker", {})

    if len(pr_diff) > MAX_DIFF_SIZE:
        state["regression_report"] = {
            "status": "WARN",
            "deltas": [{"function": "N/A", "file": "N/A", "field": "diff_size",
                        "before": "normal", "after": f"diff too large ({len(pr_diff)} bytes)",
                        "severity": "MEDIUM"}],
            "confidence": 1.0,
            "notes": [],
        }
        return state

    config = load_config(policy_dir(state) / "config.yaml")
    rcfg = config.regression
    diff_result = parse_diff(pr_diff)
    engine = TreeSitterEngine()
    file_sources = state.get("file_sources") or {}

    summarizer = LogicSummarizer(
        cache=_global_cache,
        llm_base_url=config.llm.base_url,
        llm_model=config.llm.model,
        llm_provider=config.llm.provider,
    )
    pbt = PropertyTestRunner(rcfg.property_test_examples, rcfg.property_test_timeout_seconds)
    allow_exec = state.get("allow_code_execution", True)
    notes: list[str] = []
    if rcfg.property_tests and not allow_exec:
        notes.append("Property tests are enabled in config but were not run: this event "
                     "context must not execute PR code (e.g. pull_request_target).")

    all_results: list[RegressionResult] = []
    compared = 0
    budget_hit = False

    for file_change in diff_result.files:
        language = engine.detect_language(file_change.file_path)
        if language is None or file_change.status in ("added", "deleted"):
            continue

        full = file_sources.get(file_change.file_path) or {}
        have_full = full.get("before") is not None and full.get("after") is not None
        if have_full:
            before_module, after_module = full["before"], full["after"]
        else:  # no full files available: reconstruct what the hunks show
            before_module, after_module = file_change.fragment("-"), file_change.fragment("+")
        if not before_module.strip() or not after_module.strip():
            continue

        try:
            before_funcs = _functions_by_name(engine, before_module, language)
            after_funcs = _functions_by_name(engine, after_module, language)
        except Exception:
            continue

        changed: list[str] = []
        for func_name in sorted(set(before_funcs) & set(after_funcs)):
            before_src, after_src = before_funcs[func_name][0], after_funcs[func_name][0]
            if before_src == after_src:
                continue
            if not (_validate_source(before_src) and _validate_source(after_src)):
                continue
            if compared >= rcfg.max_functions_per_pr:
                budget_hit = True
                break
            compared += 1

            result = RegressionResult(function_name=func_name, file_path=file_change.file_path,
                                      line=after_funcs[func_name][1] if have_full else 0)
            for d in summarizer.compare(before_src, after_src):
                d.function_name = func_name
                d.file_path = file_change.file_path
                result.deltas.append(d)
            all_results.append(result)
            if result.deltas:
                changed.append(func_name)

        if (changed and rcfg.property_tests and allow_exec and language == "python"
                and _is_critical(file_change.file_path, rcfg)):
            by_name = {r.function_name: r for r in all_results
                       if r.file_path == file_change.file_path}
            for res in pbt.run(file_change.file_path, before_module, after_module, changed,
                               state.get("repo_path")):
                name = res.get("function", "")
                target = by_name.get(name)
                if target is None:
                    continue
                if res.get("status") == "diverged":
                    summary = (f"Hypothesis found `{res['call']}`: before "
                               f"{_describe_outcome(res['before'])}, after "
                               f"{_describe_outcome(res['after'])}")
                    target.deltas.append(SemanticDelta(
                        function_name=name, file_path=file_change.file_path,
                        field="property_test", before=_describe_outcome(res["before"]),
                        after=_describe_outcome(res["after"]), severity="HIGH",
                        summary=summary,
                    ))
                elif res.get("status") == "equivalent":
                    notes.append(f"Property test: `{name}` behaved identically before/after "
                                 f"on {res.get('examples')} generated inputs.")
                else:
                    notes.append(f"Property test skipped for `{name}`: {res.get('reason')}.")

    if budget_hit:
        notes.append(f"Stopped after {rcfg.max_functions_per_pr} functions "
                     "(regression.max_functions_per_pr).")

    all_deltas = []
    for r in all_results:
        for d in r.deltas:
            entry = {
                "function": d.function_name,
                "file": d.file_path,
                "line": r.line,
                "field": d.field,
                "before": d.before,
                "after": d.after,
                "severity": d.severity,
            }
            if d.summary:
                entry["summary"] = d.summary
            all_deltas.append(entry)

    status = "WARN" if all_deltas else "PASS"

    cost_tracker["regression_llm_calls"] = summarizer.llm_calls
    cost_tracker["regression_functions_compared"] = compared
    cost_tracker["regression_property_test_runs"] = pbt.runs
    state["cost_tracker"] = cost_tracker

    state["regression_report"] = {
        "status": status,
        "deltas": all_deltas,
        "confidence": 1.0 if summarizer.llm_calls == 0 else 0.85,
        "notes": notes,
    }
    return state
