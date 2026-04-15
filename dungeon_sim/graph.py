from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import AgentPolicy, DummyDeterministicAgent, observe_and_decide
from .state import GraphState, RunStatus
from .tracing import node_span
from .world import (
    add_event,
    apply_action,
    deliver_pending_messages,
    detect_divergences,
    evaluate_termination,
)


def _state_update(state: GraphState) -> dict[str, object]:
    return {
        "run": state["run"],
        "world": state["world"],
        "agents": state["agents"],
        "pending_messages": state["pending_messages"],
        "current_agent": state["current_agent"],
        "current_decision": state["current_decision"],
        "trace": state["trace"],
    }


def build_graph(
    agent_a_policy: AgentPolicy | None = None,
    agent_b_policy: AgentPolicy | None = None,
):
    policy_a = agent_a_policy or DummyDeterministicAgent()
    policy_b = agent_b_policy or DummyDeterministicAgent()
    graph = StateGraph(GraphState)

    def initialize_run(state: GraphState) -> dict[str, object]:
        with node_span("initialize_run", turn=state["run"].turn, status=state["run"].status.value):
            state["run"].status = RunStatus.RUNNING
            add_event(
                state,
                "status",
                f"Turn {state['run'].turn}: initialized run {state['run'].run_id}.",
            )
            return _state_update(state)

    def deliver_messages(state: GraphState) -> dict[str, object]:
        with node_span("deliver_messages", turn=state["run"].turn):
            deliver_pending_messages(state)
            return _state_update(state)

    def agent_a_turn(state: GraphState) -> dict[str, object]:
        with node_span("agent_a_turn", turn=state["run"].turn, agent="Agent A"):
            state["current_agent"] = "Agent A"
            state["current_decision"] = observe_and_decide("Agent A", state, policy_a)
            return _state_update(state)

    def apply_agent_a_action(state: GraphState) -> dict[str, object]:
        action = state["current_decision"].action if state["current_decision"] else "unknown"
        with node_span("apply_action", turn=state["run"].turn, agent="Agent A", action=action):
            if state["current_decision"] is not None:
                apply_action(state, "Agent A", state["current_decision"])
                state["current_decision"] = None
                return _state_update(state)
            return _state_update(state)

    def detect_agent_a_divergence(state: GraphState) -> dict[str, object]:
        with node_span("divergence_check", turn=state["run"].turn, agent="Agent A"):
            detect_divergences(state, "Agent A")
            return _state_update(state)

    def agent_b_turn(state: GraphState) -> dict[str, object]:
        with node_span("agent_b_turn", turn=state["run"].turn, agent="Agent B"):
            state["current_agent"] = "Agent B"
            state["current_decision"] = observe_and_decide("Agent B", state, policy_b)
            return _state_update(state)

    def apply_agent_b_action(state: GraphState) -> dict[str, object]:
        action = state["current_decision"].action if state["current_decision"] else "unknown"
        with node_span("apply_action", turn=state["run"].turn, agent="Agent B", action=action):
            if state["current_decision"] is not None:
                apply_action(state, "Agent B", state["current_decision"])
                state["current_decision"] = None
                return _state_update(state)
            return _state_update(state)

    def detect_agent_b_divergence(state: GraphState) -> dict[str, object]:
        with node_span("divergence_check", turn=state["run"].turn, agent="Agent B"):
            detect_divergences(state, "Agent B")
            return _state_update(state)

    def check_termination(state: GraphState) -> dict[str, object]:
        with node_span("termination_check", turn=state["run"].turn, status=state["run"].status.value):
            evaluate_termination(state)
            state["current_agent"] = None
            state["current_decision"] = None
            return _state_update(state)

    def route_after_check(state: GraphState) -> str:
        if state["run"].status in {RunStatus.SUCCESS, RunStatus.FAILED}:
            return END
        return "deliver_messages"

    graph.add_node("initialize_run", initialize_run)
    graph.add_node("deliver_messages", deliver_messages)
    graph.add_node("agent_a_turn", agent_a_turn)
    graph.add_node("apply_agent_a_action", apply_agent_a_action)
    graph.add_node("detect_agent_a_divergence", detect_agent_a_divergence)
    graph.add_node("agent_b_turn", agent_b_turn)
    graph.add_node("apply_agent_b_action", apply_agent_b_action)
    graph.add_node("detect_agent_b_divergence", detect_agent_b_divergence)
    graph.add_node("check_termination", check_termination)

    graph.add_edge(START, "initialize_run")
    graph.add_edge("initialize_run", "deliver_messages")
    graph.add_edge("deliver_messages", "agent_a_turn")
    graph.add_edge("agent_a_turn", "apply_agent_a_action")
    graph.add_edge("apply_agent_a_action", "detect_agent_a_divergence")
    graph.add_edge("detect_agent_a_divergence", "agent_b_turn")
    graph.add_edge("agent_b_turn", "apply_agent_b_action")
    graph.add_edge("apply_agent_b_action", "detect_agent_b_divergence")
    graph.add_edge("detect_agent_b_divergence", "check_termination")
    graph.add_conditional_edges("check_termination", route_after_check)

    return graph.compile()
