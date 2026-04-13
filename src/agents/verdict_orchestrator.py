"""Verdict Orchestrator — aggregates agent reports into a final governance verdict."""

from __future__ import annotations

from src.graph.state import AgentProbeState


STATUS_SCORES = {"PASS": 0, "WARN": 50, "FATAL": 100, "BLOCK": 100}

VERDICT_EMOJI = {"PASS": "\u2705", "WARN": "\u26a0\ufe0f", "BLOCK": "\u274c"}


def _format_violations(violations: list[dict], header: str) -> str:
    """Format a list of violations as a markdown section."""
    if not violations:
        return ""
    lines = [f"\n**{header}** ({len(violations)} issue{'s' if len(violations) != 1 else ''}):\n"]
    for v in violations[:10]:  # Cap at 10 to avoid giant comments
        file_info = v.get("file_path") or v.get("file", "")
        line_num = v.get("line_number") or v.get("line", "")
        desc = v.get("description") or v.get("field", "")
        severity = v.get("severity", "")
        loc = f"`{file_info}:{line_num}`" if line_num else f"`{file_info}`"
        lines.append(f"- [{severity}] {loc}: {desc}")
    if len(violations) > 10:
        lines.append(f"- ... and {len(violations) - 10} more")
    return "\n".join(lines)


def _format_deltas(deltas: list[dict]) -> str:
    """Format regression deltas as a markdown section."""
    if not deltas:
        return ""
    lines = [f"\n**Semantic Changes** ({len(deltas)} delta{'s' if len(deltas) != 1 else ''}):\n"]
    for d in deltas[:10]:
        func = d.get("function", "unknown")
        field = d.get("field", "")
        severity = d.get("severity", "")
        lines.append(f"- [{severity}] `{func}` — {field}: {d.get('before', '')} → {d.get('after', '')}")
    if len(deltas) > 10:
        lines.append(f"- ... and {len(deltas) - 10} more")
    return "\n".join(lines)


def build_comment(verdict: dict, state: AgentProbeState) -> str:
    """Build a GitHub PR comment with the full verdict breakdown."""
    status = verdict["status"]
    score = verdict["score"]
    emoji = VERDICT_EMOJI.get(status, "")

    lines = [
        f"## {emoji} AgentProbe Verdict: **{status}** (score: {score:.1f}/100)",
        "",
    ]

    # Architecture section
    arch = state.get("architecture_report")
    if arch:
        arch_emoji = VERDICT_EMOJI.get(arch["status"], "")
        lines.append(f"### {arch_emoji} Architecture Agent: {arch['status']}")
        if arch.get("violations"):
            lines.append(_format_violations(arch["violations"], "Boundary Violations"))
        else:
            lines.append("No boundary violations found.")
        lines.append("")

    # Pattern section
    pattern = state.get("pattern_report")
    if pattern:
        pat_emoji = VERDICT_EMOJI.get(pattern["status"], "")
        lines.append(f"### {pat_emoji} Pattern Agent: {pattern['status']}")
        if pattern.get("violations"):
            lines.append(_format_violations(pattern["violations"], "Style Violations"))
        else:
            lines.append("No style violations found.")
        lines.append("")

    # Regression section
    regression = state.get("regression_report")
    if regression:
        reg_emoji = VERDICT_EMOJI.get(regression["status"], "")
        lines.append(f"### {reg_emoji} Regression Agent: {regression['status']}")
        if regression.get("deltas"):
            lines.append(_format_deltas(regression["deltas"]))
        else:
            lines.append("No semantic regressions detected.")
        lines.append("")

    # Cost
    cost = state.get("cost_tracker", {})
    if cost:
        llm_calls = cost.get("regression_llm_calls", 0)
        lines.append(f"---\n*LLM calls: {llm_calls} | Confidence: "
                     f"{regression.get('confidence', 1.0) if regression else 1.0:.0%}*")

    return "\n".join(lines)


def run(state: AgentProbeState) -> AgentProbeState:
    """Calculate governance score and produce final verdict with PR comment."""
    arch_score = STATUS_SCORES.get(
        (state.get("architecture_report") or {}).get("status", "PASS"), 0
    )
    pattern_score = STATUS_SCORES.get(
        (state.get("pattern_report") or {}).get("status", "PASS"), 0
    )
    regression_score = STATUS_SCORES.get(
        (state.get("regression_report") or {}).get("status", "PASS"), 0
    )

    weighted = arch_score * 0.40 + pattern_score * 0.25 + regression_score * 0.35

    # Short-circuit: if architecture found FATAL, always BLOCK regardless of score
    if state.get("short_circuit", False):
        verdict_status = "BLOCK"
        weighted = max(weighted, 100.0)
    elif weighted > 70:
        verdict_status = "BLOCK"
    elif weighted > 40:
        verdict_status = "WARN"
    else:
        verdict_status = "PASS"

    verdict = {
        "score": weighted,
        "status": verdict_status,
        "details": {
            "architecture": state.get("architecture_report"),
            "pattern": state.get("pattern_report"),
            "regression": state.get("regression_report"),
        },
    }

    verdict["comment"] = build_comment(verdict, state)
    state["verdict"] = verdict
    return state
