"""AgentProbe shared state schema for the LangGraph workflow."""

from __future__ import annotations

from typing import TypedDict


class AgentProbeState(TypedDict, total=False):
    """Shared state passed between the analyzer nodes of the LangGraph graph."""

    # Input
    pr_diff: str
    repo_path: str
    pr_number: int
    repo_full_name: str
    # Directory with config.yaml / boundaries.yaml / style-profile.yaml.
    # Defaults to <repo_path>/.agentprobe when unset.
    config_dir: str
    # Where the policy lives inside the repo (used to flag PRs that edit it).
    policy_repo_dir: str
    # Optional full file contents: {path: {"before": str | None, "after": str | None}}.
    # Lets the regression analyzer compare whole functions instead of hunks.
    file_sources: dict[str, dict[str, str | None]]
    # False under pull_request_target and similar contexts: never execute PR code.
    allow_code_execution: bool

    # Analyzer outputs
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
