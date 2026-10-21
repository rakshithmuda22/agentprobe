"""Verdict orchestrator — aggregates analyzer reports into a final governance verdict.

Scoring (all deterministic):
  * each finding is worth points by severity (FATAL 100, HIGH 40, WARN/MEDIUM 20, LOW 5)
  * an analyzer's score is the sum of its findings' points, capped at 100
  * the PR score is the weighted sum of analyzer scores (weights from config.yaml)
  * verdict: BLOCK if any FATAL finding or score > block threshold,
             WARN if score > warn threshold, else PASS
"""

from __future__ import annotations

from agentprobe.config.loader import AgentProbeConfig, load_config, policy_dir
from agentprobe.graph.state import AgentProbeState

# Minimum analyzer score implied by its overall status.
STATUS_SCORES = {"PASS": 0, "WARN": 20, "FATAL": 100, "BLOCK": 100}

SEVERITY_POINTS = {"FATAL": 100, "BLOCK": 100, "HIGH": 40, "WARN": 20, "MEDIUM": 20, "LOW": 5}

VERDICT_EMOJI = {"PASS": "✅", "WARN": "⚠️", "BLOCK": "❌", "FATAL": "❌"}

COMMENT_MARKER = "<!-- agentprobe-verdict -->"


def analyzer_score(report: dict | None) -> float:
    """0-100 score for one analyzer report."""
    if not report:
        return 0.0
    findings = report.get("violations") or report.get("deltas") or []
    points = sum(SEVERITY_POINTS.get(str(f.get("severity", "")).upper(), 0) for f in findings)
    floor = STATUS_SCORES.get(report.get("status", "PASS"), 0)
    return float(min(100, max(points, floor)))


def _has_fatal(state: AgentProbeState) -> bool:
    if state.get("short_circuit", False):
        return True
    for key in ("architecture_report", "pattern_report", "regression_report"):
        report = state.get(key) or {}
        if report.get("status") in ("FATAL", "BLOCK"):
            return True
    return False


def _format_violations(violations: list[dict], header: str) -> str:
    """Format a list of violations as a markdown section."""
    if not violations:
        return ""
    lines = [f"\n**{header}** ({len(violations)} issue{'s' if len(violations) != 1 else ''}):\n"]
    for v in violations[:10]:  # Cap at 10 to avoid giant comments
        file_info = v.get("file_path") or v.get("file", "")
        line_num = v.get("line_number") or v.get("line", "")
        desc = v.get("description") or v.get("rule_violated") or v.get("field", "")
        severity = v.get("severity", "")
        loc = f"`{file_info}:{line_num}`" if line_num else f"`{file_info}`"
        lines.append(f"- [{severity}] {loc}: {desc}")
    if len(violations) > 10:
        lines.append(f"- ... and {len(violations) - 10} more")
    return "\n".join(lines)


def _clip(text: str, limit: int = 140) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_deltas(deltas: list[dict]) -> str:
    """Format regression deltas as a markdown section."""
    if not deltas:
        return ""
    lines = [f"\n**Semantic Changes** ({len(deltas)} delta{'s' if len(deltas) != 1 else ''}):\n"]
    for d in deltas[:10]:
        func = d.get("function", "unknown")
        field = d.get("field", "")
        severity = d.get("severity", "")
        where = f" (`{d['file']}:{d['line']}`)" if d.get("file") and d.get("line") else ""
        if d.get("summary"):
            lines.append(f"- [{severity}] `{func}`{where} — {field}: {d['summary']}")
        else:
            lines.append(f"- [{severity}] `{func}`{where} — {field}: "
                         f"{_clip(d.get('before', ''))} → {_clip(d.get('after', ''))}")
    if len(deltas) > 10:
        lines.append(f"- ... and {len(deltas) - 10} more")
    return "\n".join(lines)


def build_comment(verdict: dict, state: AgentProbeState) -> str:
    """Build a GitHub PR comment with the full verdict breakdown."""
    status = verdict["status"]
    score = verdict["score"]
    emoji = VERDICT_EMOJI.get(status, "")

    lines = [
        COMMENT_MARKER,
        f"## {emoji} AgentProbe Verdict: **{status}** (score: {score:.1f}/100)",
        "",
    ]
    skipped = "Skipped: a FATAL finding already decided the verdict (short-circuit)."

    arch = state.get("architecture_report")
    if arch:
        lines.append(f"### {VERDICT_EMOJI.get(arch['status'], '')} Architecture: {arch['status']}")
        if arch.get("violations"):
            lines.append(_format_violations(arch["violations"], "Boundary Violations"))
        else:
            lines.append("No boundary violations found.")
        lines.append("")

    pattern = state.get("pattern_report")
    if pattern:
        lines.append(f"### {VERDICT_EMOJI.get(pattern['status'], '')} Pattern: {pattern['status']}")
        if pattern.get("violations"):
            lines.append(_format_violations(pattern["violations"], "Style Violations"))
        else:
            lines.append(pattern.get("note") or "No style violations found.")
        lines.append("")
    elif state.get("short_circuit"):
        lines += ["### Pattern: skipped", skipped, ""]

    regression = state.get("regression_report")
    if regression:
        lines.append(
            f"### {VERDICT_EMOJI.get(regression['status'], '')} Regression: {regression['status']}"
        )
        if regression.get("deltas"):
            lines.append(_format_deltas(regression["deltas"]))
        else:
            lines.append("No semantic regressions detected.")
        for note in regression.get("notes", []):
            lines.append(f"\n_{note}_")
        lines.append("")
    elif state.get("short_circuit"):
        lines += ["### Regression: skipped", skipped, ""]

    cost = state.get("cost_tracker", {})
    if cost:
        lines.append(
            "---\n"
            f"*Functions compared: {cost.get('regression_functions_compared', 0)} | "
            f"LLM calls: {cost.get('regression_llm_calls', 0)} | "
            f"Property-test runs: {cost.get('regression_property_test_runs', 0)}*"
        )

    return "\n".join(lines)


def run(state: AgentProbeState) -> AgentProbeState:
    """Calculate governance score and produce final verdict with PR comment."""
    if state.get("config_dir") or state.get("repo_path"):
        config = load_config(policy_dir(state) / "config.yaml")
    else:
        config = AgentProbeConfig()
    weights = config.weights

    scores = {
        "architecture": analyzer_score(state.get("architecture_report")),
        "pattern": analyzer_score(state.get("pattern_report")),
        "regression": analyzer_score(state.get("regression_report")),
    }
    weighted = sum(scores[name] * float(weights.get(name, 0.0)) for name in scores)

    if _has_fatal(state):
        verdict_status = "BLOCK"
        weighted = 100.0  # a FATAL finding pins the score
    elif weighted > config.block_threshold:
        verdict_status = "BLOCK"
    elif weighted > config.warn_threshold:
        verdict_status = "WARN"
    else:
        verdict_status = "PASS"

    verdict = {
        "score": round(weighted, 2),
        "status": verdict_status,
        "analyzer_scores": scores,
        "details": {
            "architecture": state.get("architecture_report"),
            "pattern": state.get("pattern_report"),
            "regression": state.get("regression_report"),
        },
    }

    verdict["comment"] = build_comment(verdict, state)
    state["verdict"] = verdict
    return state
