"""LangGraph workflow for AgentProbe.

    architecture ──► pattern ──┬─(no FATAL)──► regression ──► verdict
                               └─(FATAL)──────────────────────► verdict

Each node is a deterministic analyzer (no LLM). Pattern always runs so a
blocked PR still gets every style finding in one pass; the slower regression
analyzer is skipped once a FATAL boundary violation has decided the verdict.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agentprobe.graph.nodes import (
    architecture_node,
    pattern_node,
    regression_node,
    verdict_node,
)
from agentprobe.graph.state import AgentProbeState


def should_short_circuit(state: AgentProbeState) -> str:
    """Route after the pattern analyzer: skip regression if already FATAL."""
    if state.get("short_circuit", False):
        return "verdict_orchestrator"
    return "regression_agent"


def build_graph() -> StateGraph:
    """Build the AgentProbe LangGraph StateGraph."""
    graph = StateGraph(AgentProbeState)

    graph.add_node("architecture_agent", architecture_node)
    graph.add_node("pattern_agent", pattern_node)
    graph.add_node("regression_agent", regression_node)
    graph.add_node("verdict_orchestrator", verdict_node)

    graph.set_entry_point("architecture_agent")
    graph.add_edge("architecture_agent", "pattern_agent")
    graph.add_conditional_edges(
        "pattern_agent",
        should_short_circuit,
        {
            "verdict_orchestrator": "verdict_orchestrator",
            "regression_agent": "regression_agent",
        },
    )
    graph.add_edge("regression_agent", "verdict_orchestrator")
    graph.add_edge("verdict_orchestrator", END)

    return graph


def compile_graph():
    """Compile the graph for execution."""
    return build_graph().compile()


def run_agentprobe(state: AgentProbeState) -> AgentProbeState:
    """Run the full AgentProbe pipeline."""
    return compile_graph().invoke(state)
