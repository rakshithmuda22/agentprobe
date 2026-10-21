"""AgentProbe shared state schema for LangGraph DAG."""

from __future__ import annotations

from typing import TypedDict


class AgentProbeState(TypedDict, total=False):
    """Shared state passed between all agents in the LangGraph DAG."""

    # Input
    pr_diff: str
    repo_path: str
    pr_number: int
    repo_full_name: str

    # Agent outputs
    architecture_report: dict | None
    pattern_report: dict | None
    regression_report: dict | None

    # Control flow
    short_circuit: bool

    # Final verdict
    verdict: dict | None

    # Metadata
    cached_functions: list[str]
    cost_tracker: dict
