"""Node wrapper functions for the LangGraph DAG."""

from __future__ import annotations

from agentprobe.graph.state import AgentProbeState


def architecture_node(state: AgentProbeState) -> AgentProbeState:
    """Run the Architecture Agent."""
    from agentprobe.agents.architecture_agent import run
    return run(state)


def pattern_node(state: AgentProbeState) -> AgentProbeState:
    """Run the Pattern Agent."""
    from agentprobe.agents.pattern_agent import run
    return run(state)


def regression_node(state: AgentProbeState) -> AgentProbeState:
    """Run the Regression Agent."""
    from agentprobe.agents.regression_agent import run
    return run(state)


def verdict_node(state: AgentProbeState) -> AgentProbeState:
    """Run the Verdict Orchestrator."""
    from agentprobe.agents.verdict_orchestrator import run
    return run(state)
