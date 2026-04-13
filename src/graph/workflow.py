"""LangGraph DAG workflow for AgentProbe."""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from src.graph.state import AgentProbeState
from src.graph.nodes import (
    architecture_node,
    pattern_node,
    regression_node,
    verdict_node,
)


def should_short_circuit(state: AgentProbeState) -> str:
    """Route after architecture agent: short-circuit on FATAL or continue."""
    if state.get("short_circuit", False):
        return "verdict_orchestrator"
    return "parallel_agents"


def build_graph() -> StateGraph:
    """Build and compile the AgentProbe LangGraph DAG."""
    graph = StateGraph(AgentProbeState)

    # Add nodes
    graph.add_node("architecture_agent", architecture_node)
    graph.add_node("pattern_agent", pattern_node)
    graph.add_node("regression_agent", regression_node)
    graph.add_node("verdict_orchestrator", verdict_node)

    # Entry point
    graph.set_entry_point("architecture_agent")

    # Conditional edge after architecture
    graph.add_conditional_edges(
        "architecture_agent",
        should_short_circuit,
        {
            "verdict_orchestrator": "verdict_orchestrator",
            "parallel_agents": "pattern_agent",
        },
    )

    # Pattern and regression run after architecture (sequentially for simplicity,
    # LangGraph handles the fan-out internally when using parallel branches)
    graph.add_edge("pattern_agent", "regression_agent")
    graph.add_edge("regression_agent", "verdict_orchestrator")

    # Finish
    graph.add_edge("verdict_orchestrator", END)

    return graph


def compile_graph():
    """Compile the graph for execution."""
    return build_graph().compile()


def run_agentprobe(state: AgentProbeState) -> AgentProbeState:
    """Run the full AgentProbe pipeline."""
    app = compile_graph()
    return app.invoke(state)
