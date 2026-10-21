"""Differential property-test harness (Layer 2 of the regression analyzer).

Runs in a *separate* Python process started with ``python -I`` by
``regression_agent.PropertyTestRunner``. It loads the before/after version of
one module, then uses Hypothesis to search for an input where the two versions
of a changed function behave differently (different return value, or a
different exception type).

This executes code from the pull request. The parent process gives it a
scrubbed environment, a temp working directory, CPU/memory/time limits and,
when running as root (the Docker action), an unprivileged uid. That reduces
blast radius but is not a security sandbox; see README "Security".

Usage: python -I pbt_harness.py <payload.json>
"""

from __future__ import annotations

import ast
import copy
import decimal
import inspect
import json
import re
import signal
import sys
import types
import typing

RESULT_MARKER = "AGENTPROBE_PBT_RESULT:"
PER_CALL_TIMEOUT_S = 1.0


class UnavailableDependency(Exception):
    """Raised when code touches an import that could not be loaded in isolation."""


class _Unavailable:
    """Placeholder bound to names whose import failed in the isolated process."""

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_agentprobe_name", name)

    def __getattr__(self, attr):
        raise UnavailableDependency(object.__getattribute__(self, "_agentprobe_name"))

    def __call__(self, *args, **kwargs):
        raise UnavailableDependency(object.__getattribute__(self, "_agentprobe_name"))


def _bound_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    names = []
    for alias in node.names:
        if alias.name == "*":
            continue
        if isinstance(node, ast.Import):
            names.append(alias.asname or alias.name.split(".")[0])
        else:
            names.append(alias.asname or alias.name)
    return names


def _guarded(stmt: ast.stmt, fallback: list[ast.stmt]) -> ast.Try:
    return ast.Try(
        body=[stmt],
        handlers=[ast.ExceptHandler(type=ast.Name("BaseException", ast.Load()), name=None,
                                    body=fallback or [ast.Pass()])],
        orelse=[],
        finalbody=[],
    )


def reduce_module(source: str, filename: str) -> ast.Module:
    """Keep only imports, defs, classes and assignments; guard each one.

    Top-level side effects (calls, ``if __name__ == "__main__"`` blocks,
    loops) are dropped. A failing import binds an ``_Unavailable`` placeholder
    instead of aborting the whole module.
    """
    tree = ast.parse(source, filename)
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            fallback = [
                ast.Assign(
                    targets=[ast.Name(name, ast.Store())],
                    value=ast.Call(ast.Name("_Unavailable", ast.Load()),
                                   [ast.Constant(name)], []),
                )
                for name in _bound_names(node)
            ]
            body.append(_guarded(node, fallback))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                               ast.Assign, ast.AnnAssign)):
            body.append(_guarded(node, []))
    tree.body = body
    return ast.fix_missing_locations(tree)


def load_version(source: str, filename: str, package: str, side: str) -> dict:
    namespace = {
        "__name__": f"agentprobe_{side}",
        "__package__": package or None,
        "__file__": filename,
        "__builtins__": __builtins__,
        "_Unavailable": _Unavailable,
    }
    exec(compile(reduce_module(source, filename), filename, "exec"), namespace)  # noqa: S102
    return namespace


_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def _normalize(value: str) -> str:
    return _ADDR.sub(" at 0x?", value)[:500]


class _CallTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _CallTimeout()


def outcome(fn, args: tuple) -> tuple[str, str]:
    """Run fn(*args) and summarize what it did."""
    use_alarm = hasattr(signal, "setitimer")
    if use_alarm:
        signal.setitimer(signal.ITIMER_REAL, PER_CALL_TIMEOUT_S)
    try:
        result = fn(*copy.deepcopy(args))
        if inspect.isgenerator(result):
            result = [item for _, item in zip(range(50), result)]
        return ("returned", _normalize(repr(result)))
    except UnavailableDependency as exc:
        return ("unavailable", str(exc))
    except _CallTimeout:
        return ("raised", "Timeout")
    except Exception as exc:  # noqa: BLE001 - any exception is an observable outcome
        return ("raised", type(exc).__name__)
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)


_NAMED_TYPES = {"int": int, "float": float, "str": str, "bool": bool, "bytes": bytes,
                "Decimal": decimal.Decimal, "decimal.Decimal": decimal.Decimal}


def strategy_for(annotation, st):
    """Hypothesis strategy for a parameter, or None if we can't generate it."""
    if isinstance(annotation, str):
        annotation = _NAMED_TYPES.get(annotation.strip(), annotation)
    if annotation is inspect.Parameter.empty:
        return st.integers(min_value=-1000, max_value=1000)
    if annotation is float:
        return st.floats(allow_nan=False, allow_infinity=False, width=32)
    if annotation is decimal.Decimal:
        return st.decimals(allow_nan=False, allow_infinity=False, places=2)
    if annotation in (int, str, bool, bytes):
        return st.from_type(annotation)
    origin = typing.get_origin(annotation)
    if origin in (list, tuple, dict, set, frozenset, typing.Union, types.UnionType):
        inner = typing.get_args(annotation)
        if all(a in (int, float, str, bool, bytes, type(None)) for a in inner):
            return st.from_type(annotation)
    return None


def _required_params(fn) -> list[inspect.Parameter] | None:
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return None
    return [p for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]


def check_function(name: str, before_ns: dict, after_ns: dict, max_examples: int) -> dict:
    from hypothesis import HealthCheck, Phase, given, reject, settings
    from hypothesis import strategies as st
    from hypothesis.errors import Unsatisfiable

    before, after = before_ns.get(name), after_ns.get(name)
    if not callable(before) or not callable(after):
        return {"function": name, "status": "skipped",
                "reason": "not a top-level function that loads in isolation"}
    if inspect.iscoroutinefunction(after):
        return {"function": name, "status": "skipped", "reason": "async function"}

    before_params, after_params = _required_params(before), _required_params(after)
    if before_params is None or after_params is None:
        return {"function": name, "status": "skipped", "reason": "unreadable signature"}
    if after_params and after_params[0].name in ("self", "cls"):
        return {"function": name, "status": "skipped", "reason": "method (needs an instance)"}
    if len(before_params) != len(after_params):
        return {"function": name, "status": "skipped", "reason": "signature changed"}

    try:
        hints = typing.get_type_hints(after)
    except Exception:  # noqa: BLE001
        hints = {}
    strategies = []
    for p in after_params:
        strategy = strategy_for(hints.get(p.name, p.annotation), st)
        if strategy is None:
            return {"function": name, "status": "skipped",
                    "reason": f"no input strategy for parameter '{p.name}'"}
        strategies.append(strategy)

    found: dict = {}

    @settings(max_examples=max_examples, derandomize=True, database=None, deadline=None,
              phases=(Phase.explicit, Phase.generate, Phase.shrink),
              suppress_health_check=list(HealthCheck))
    @given(st.tuples(*strategies))
    def differential(args):
        b, a = outcome(before, args), outcome(after, args)
        if "unavailable" in (b[0], a[0]):
            reject()
        if b != a:
            found.clear()
            found.update(args=list(args), before=list(b), after=list(a))
            raise AssertionError("behaviour diverged")

    try:
        differential()
    except AssertionError:
        call = ", ".join(f"{p.name}={v!r}" for p, v in zip(after_params, found["args"]))
        return {"function": name, "status": "diverged", "call": f"{name}({call})",
                "before": found["before"], "after": found["after"]}
    except Unsatisfiable:
        return {"function": name, "status": "skipped",
                "reason": "depends on modules unavailable in isolation"}
    except Exception as exc:  # noqa: BLE001
        return {"function": name, "status": "skipped",
                "reason": f"harness error: {type(exc).__name__}"}
    return {"function": name, "status": "equivalent", "examples": max_examples}


def main(payload_path: str) -> int:
    with open(payload_path) as f:
        payload = json.load(f)

    repo_root = payload.get("repo_root")
    if repo_root:
        sys.path.insert(0, repo_root)

    results = []
    try:
        before_ns = load_version(payload["before_source"], payload["file_path"],
                                 payload.get("package", ""), "before")
        after_ns = load_version(payload["after_source"], payload["file_path"],
                                payload.get("package", ""), "after")
    except SyntaxError as exc:
        results = [{"function": n, "status": "skipped", "reason": f"syntax error: {exc.msg}"}
                   for n in payload["functions"]]
    else:
        for name in payload["functions"]:
            results.append(check_function(name, before_ns, after_ns,
                                          int(payload.get("max_examples", 200))))

    sys.stdout.write("\n" + RESULT_MARKER + json.dumps(results) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
