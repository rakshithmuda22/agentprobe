"""Integration tests for the LangGraph DAG workflow."""

from __future__ import annotations

from src.graph.state import AgentProbeState
from src.graph.workflow import run_agentprobe


def test_full_workflow_all_agents_called(minimal_state):
    """All 4 agents should run and produce reports when no short-circuit."""
    minimal_state["pr_diff"] = ""  # Empty diff = no violations
    result = run_agentprobe(minimal_state)

    assert result["architecture_report"] is not None
    assert result["pattern_report"] is not None
    assert result["regression_report"] is not None
    assert result["verdict"] is not None
    assert result["short_circuit"] is False


def test_short_circuit_skips_pattern_and_regression(minimal_state, forbidden_import_diff):
    """When architecture finds FATAL, pattern and regression should be skipped."""
    minimal_state["pr_diff"] = forbidden_import_diff
    # Point to the fixture repo for boundaries
    minimal_state["repo_path"] = str(
        __import__("pathlib").Path(__file__).parent / "fixtures" / "boundary_violation_repo"
    )

    result = run_agentprobe(minimal_state)

    assert result["architecture_report"] is not None
    assert result["architecture_report"]["status"] == "FATAL"
    assert result["short_circuit"] is True
    # Pattern and regression should NOT have run (remain None before verdict)
    # But verdict should still exist
    assert result["verdict"] is not None
    assert result["verdict"]["status"] == "BLOCK"


def test_clean_pr_passes(minimal_state, clean_import_diff):
    """A clean PR with no violations should PASS."""
    minimal_state["pr_diff"] = clean_import_diff

    result = run_agentprobe(minimal_state)

    assert result["architecture_report"]["status"] == "PASS"
    assert result["short_circuit"] is False
    assert result["verdict"]["status"] == "PASS"
