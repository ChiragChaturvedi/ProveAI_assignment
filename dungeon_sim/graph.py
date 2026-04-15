from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import AgentPolicy, DummyDeterministicAgent, observe_and_decide
from .state import GraphState, RunStatus
from .telemetry_contract import EventNames, SpanNames
from .tracing import add_span_event, end_span, node_span, safe_set_attributes, start_detached_span
from .world import (
    DIRECTIONS,
    add_event,
    apply_action,
    deliver_pending_messages,
    detect_divergences,
    evaluate_termination,
    in_bounds,
    is_walkable,
    move_position,
    same_position,
    wall_at,
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
        "telemetry": state["telemetry"],
    }


def _telemetry_state(state: GraphState) -> dict[str, object]:
    return state.setdefault("telemetry", {})


def _agent_telemetry(state: GraphState, agent_name: str) -> dict[str, object]:
    telemetry = _telemetry_state(state)
    agent_data = telemetry.setdefault("agents", {})
    return agent_data.setdefault(
        agent_name,
        {
            "failed_move_streak": 0,
            "last_decision_signature": None,
            "last_goal": None,
            "message_changed_plan": False,
        },
    )


def _position_text(position) -> str:
    return f"({position.x}, {position.y})"


def _current_turn_span(state: GraphState):
    return _telemetry_state(state).get("current_turn_span")


def _ensure_turn_span(state: GraphState):
    telemetry = _telemetry_state(state)
    if telemetry.get("current_turn_span") is not None:
        return telemetry["current_turn_span"]

    run = state["run"]
    turn_span = start_detached_span(
        SpanNames.TURN,
        turn=run.turn,
        status_before=run.status.value,
        status_after=run.status.value,
        messages_delivered_count=0,
        progress_signal="neutral",
    )
    telemetry["current_turn_span"] = turn_span
    telemetry["turn_status_before"] = run.status.value
    telemetry["turn_progress_signal"] = "neutral"
    telemetry["turn_messages_delivered_count"] = 0
    telemetry["turn_started_at"] = run.turn
    return turn_span


def _mark_turn_progress(state: GraphState, signal: str) -> None:
    telemetry = _telemetry_state(state)
    current = telemetry.get("turn_progress_signal", "neutral")
    if signal == "progress":
        telemetry["turn_progress_signal"] = "progress"
    elif signal == "regression" and current != "progress":
        telemetry["turn_progress_signal"] = "regression"


def _valid_directions(state: GraphState, agent_name: str) -> list[str]:
    world = state["world"]
    agent = state["agents"][agent_name]
    valid: list[str] = []
    for direction in DIRECTIONS:
        candidate = move_position(agent.position, direction)
        if is_walkable(world, candidate):
            valid.append(direction)
    return valid


def _decision_signature(decision) -> tuple[str, str]:
    return decision.action, str(decision.action_input)


def _failure_reason(state: GraphState, agent_name: str, decision, position_before, door_locked_before: bool) -> str | None:
    world = state["world"]
    agent = state["agents"][agent_name]
    if decision.action == "move" and decision.direction is not None:
        next_position = move_position(position_before, decision.direction)
        if not in_bounds(world, next_position):
            return "out_of_bounds"
        if wall_at(world, next_position):
            return "wall"
        if door_locked_before and same_position(next_position, world.door_position):
            return "locked_door"
        return "blocked"
    if decision.action == "pickup_key":
        if world.key_position is None:
            return "key_missing"
        return "not_adjacent_to_key"
    if decision.action == "unlock_door":
        if not agent.has_key:
            return "missing_key"
        if not door_locked_before:
            return "door_already_unlocked"
        return "not_adjacent_to_door"
    return None


def _action_progress_signal(action: str, success: bool) -> str:
    if action in {"pickup_key", "unlock_door"} and success:
        return "progress"
    if not success:
        return "regression"
    return "neutral"


def build_graph(
    agent_a_policy: AgentPolicy | None = None,
    agent_b_policy: AgentPolicy | None = None,
):
    policy_a = agent_a_policy or DummyDeterministicAgent()
    policy_b = agent_b_policy or DummyDeterministicAgent()
    graph = StateGraph(GraphState)

    def initialize_run(state: GraphState) -> dict[str, object]:
        with node_span(
            SpanNames.INITIALIZE_RUN,
            turn=state["run"].turn,
            status=state["run"].status.value,
            run_id=state["run"].run_id,
        ):
            state["run"].status = RunStatus.RUNNING
            add_event(
                state,
                "status",
                f"Turn {state['run'].turn}: initialized run {state['run'].run_id}.",
            )
            return _state_update(state)

    def deliver_messages(state: GraphState) -> dict[str, object]:
        turn_span = _ensure_turn_span(state)
        with node_span(
            SpanNames.DELIVER_MESSAGES,
            parent_span=turn_span,
            turn=state["run"].turn,
        ) as span:
            delivered_messages = deliver_pending_messages(state)
            delivered_count = len(delivered_messages)
            _telemetry_state(state)["turn_messages_delivered_count"] = delivered_count
            safe_set_attributes(span, messages_delivered_count=delivered_count)
            for message in delivered_messages:
                add_span_event(
                    span,
                    EventNames.MESSAGE_DELIVERED,
                    sender=message.sender,
                    recipient=message.recipient,
                    content=message.content,
                    deliver_turn=message.deliver_turn,
                )
            return _state_update(state)

    def agent_a_turn(state: GraphState) -> dict[str, object]:
        turn_span = _current_turn_span(state)
        agent_name = "Agent A"
        agent = state["agents"][agent_name]
        agent_telemetry = _agent_telemetry(state, agent_name)
        previous_goal = agent.local_memory.last_plan_goal
        with node_span(
            SpanNames.AGENT_A_TURN,
            parent_span=turn_span,
            turn=state["run"].turn,
            agent=agent_name,
            position_before=_position_text(agent.position),
            inventory=agent.inventory,
            has_key=agent.has_key,
            message_count_in_inbox=len(agent.inbox_messages),
            stuck_counter=agent_telemetry["failed_move_streak"],
            valid_directions=_valid_directions(state, agent_name),
        ) as span:
            state["current_agent"] = "Agent A"
            state["current_decision"] = observe_and_decide("Agent A", state, policy_a)
            latest_observation = state["trace"].observations[-1]
            latest_decision = state["current_decision"]
            repeated_action = agent_telemetry["last_decision_signature"] == _decision_signature(latest_decision)
            agent_telemetry["message_changed_plan"] = bool(agent.inbox_messages) and previous_goal not in {
                None,
                latest_decision.goal,
            }
            safe_set_attributes(
                span,
                goal=latest_decision.goal,
                reason=latest_decision.reason,
                confidence=latest_decision.confidence,
                visible_objects=latest_observation.visible_objects,
                visible_objects_count=len(latest_observation.visible_objects),
            )
            add_span_event(
                span,
                EventNames.DECISION_MADE,
                action=latest_decision.action,
                goal=latest_decision.goal,
                confidence=latest_decision.confidence,
            )
            if "door" in latest_observation.visible_objects:
                add_span_event(span, EventNames.DOOR_SEEN, agent=agent_name, turn=state["run"].turn)
            if repeated_action:
                add_span_event(
                    span,
                    EventNames.REPEATED_ACTION_DETECTED,
                    action=latest_decision.action,
                    goal=latest_decision.goal,
                )
            agent_telemetry["last_decision_signature"] = _decision_signature(latest_decision)
            agent_telemetry["last_goal"] = latest_decision.goal
            return _state_update(state)

    def apply_agent_a_action(state: GraphState) -> dict[str, object]:
        decision = state["current_decision"]
        action = decision.action if decision is not None else "unknown"
        turn_span = _current_turn_span(state)
        agent_name = "Agent A"
        agent = state["agents"][agent_name]
        agent_telemetry = _agent_telemetry(state, agent_name)
        position_before = agent.position.model_copy()
        door_locked_before = state["world"].door_locked
        with node_span(
            SpanNames.APPLY_ACTION,
            parent_span=turn_span,
            turn=state["run"].turn,
            agent=agent_name,
            action=action,
            position_before=_position_text(position_before),
            door_locked_before=door_locked_before,
        ) as span:
            if decision is not None:
                action_record = apply_action(state, agent_name, decision)
                progress_signal = _action_progress_signal(action_record.action, action_record.success)
                if action_record.action == "move" and action_record.success:
                    agent_telemetry["failed_move_streak"] = 0
                    add_span_event(
                        span,
                        EventNames.MOVE_SUCCEEDED,
                        direction=decision.direction,
                        position_after=_position_text(state["agents"][agent_name].position),
                    )
                elif action_record.action == "move" and not action_record.success:
                    agent_telemetry["failed_move_streak"] = int(agent_telemetry["failed_move_streak"]) + 1
                    add_span_event(
                        span,
                        EventNames.BLOCKED_MOVE,
                        direction=decision.direction,
                        failure_reason=_failure_reason(state, agent_name, decision, position_before, door_locked_before),
                        stuck_counter=agent_telemetry["failed_move_streak"],
                    )
                if action_record.action == "pickup_key" and action_record.success:
                    add_span_event(span, EventNames.KEY_PICKED_UP, agent=agent_name)
                if action_record.action == "unlock_door" and action_record.success:
                    add_span_event(span, EventNames.DOOR_UNLOCKED, agent=agent_name)
                if action_record.action == "send_message" and action_record.success:
                    add_span_event(
                        span,
                        EventNames.MESSAGE_QUEUED,
                        recipient=decision.recipient,
                        content=decision.content,
                    )
                if progress_signal != "neutral":
                    _mark_turn_progress(state, progress_signal)
                safe_set_attributes(
                    span,
                    success=action_record.success,
                    failure_reason=_failure_reason(state, agent_name, decision, position_before, door_locked_before)
                    if not action_record.success
                    else None,
                    position_after=_position_text(state["agents"][agent_name].position),
                    progress_signal=progress_signal,
                    door_locked_after=state["world"].door_locked,
                )
                state["current_decision"] = None
                return _state_update(state)
            return _state_update(state)

    def detect_agent_a_divergence(state: GraphState) -> dict[str, object]:
        turn_span = _current_turn_span(state)
        agent_name = "Agent A"
        agent_telemetry = _agent_telemetry(state, agent_name)
        with node_span(
            SpanNames.DIVERGENCE_CHECK,
            parent_span=turn_span,
            turn=state["run"].turn,
            agent=agent_name,
        ) as span:
            divergences = detect_divergences(state, agent_name)
            divergence_types = [record.kind for record in divergences]
            stuck_detected = int(agent_telemetry["failed_move_streak"]) >= 3
            safe_set_attributes(
                span,
                divergence_count=len(divergences),
                divergence_types=divergence_types,
                belief_changed=bool(divergences),
                message_changed_plan=agent_telemetry.get("message_changed_plan", False),
                stuck_detected=stuck_detected,
            )
            if divergences or stuck_detected:
                _mark_turn_progress(state, "regression")
            if stuck_detected:
                add_span_event(
                    span,
                    EventNames.NO_PROGRESS_WARNING,
                    agent=agent_name,
                    stuck_counter=agent_telemetry["failed_move_streak"],
                )
            return _state_update(state)

    def agent_b_turn(state: GraphState) -> dict[str, object]:
        turn_span = _current_turn_span(state)
        agent_name = "Agent B"
        agent = state["agents"][agent_name]
        agent_telemetry = _agent_telemetry(state, agent_name)
        previous_goal = agent.local_memory.last_plan_goal
        with node_span(
            SpanNames.AGENT_B_TURN,
            parent_span=turn_span,
            turn=state["run"].turn,
            agent=agent_name,
            position_before=_position_text(agent.position),
            inventory=agent.inventory,
            has_key=agent.has_key,
            message_count_in_inbox=len(agent.inbox_messages),
            stuck_counter=agent_telemetry["failed_move_streak"],
            valid_directions=_valid_directions(state, agent_name),
        ) as span:
            state["current_agent"] = agent_name
            state["current_decision"] = observe_and_decide(agent_name, state, policy_b)
            latest_observation = state["trace"].observations[-1]
            latest_decision = state["current_decision"]
            repeated_action = agent_telemetry["last_decision_signature"] == _decision_signature(latest_decision)
            agent_telemetry["message_changed_plan"] = bool(agent.inbox_messages) and previous_goal not in {
                None,
                latest_decision.goal,
            }
            safe_set_attributes(
                span,
                goal=latest_decision.goal,
                reason=latest_decision.reason,
                confidence=latest_decision.confidence,
                visible_objects=latest_observation.visible_objects,
                visible_objects_count=len(latest_observation.visible_objects),
            )
            add_span_event(
                span,
                EventNames.DECISION_MADE,
                action=latest_decision.action,
                goal=latest_decision.goal,
                confidence=latest_decision.confidence,
            )
            if "door" in latest_observation.visible_objects:
                add_span_event(span, EventNames.DOOR_SEEN, agent=agent_name, turn=state["run"].turn)
            if repeated_action:
                add_span_event(
                    span,
                    EventNames.REPEATED_ACTION_DETECTED,
                    action=latest_decision.action,
                    goal=latest_decision.goal,
                )
            agent_telemetry["last_decision_signature"] = _decision_signature(latest_decision)
            agent_telemetry["last_goal"] = latest_decision.goal
            return _state_update(state)

    def apply_agent_b_action(state: GraphState) -> dict[str, object]:
        decision = state["current_decision"]
        action = decision.action if decision is not None else "unknown"
        turn_span = _current_turn_span(state)
        agent_name = "Agent B"
        agent = state["agents"][agent_name]
        agent_telemetry = _agent_telemetry(state, agent_name)
        position_before = agent.position.model_copy()
        door_locked_before = state["world"].door_locked
        with node_span(
            SpanNames.APPLY_ACTION,
            parent_span=turn_span,
            turn=state["run"].turn,
            agent=agent_name,
            action=action,
            position_before=_position_text(position_before),
            door_locked_before=door_locked_before,
        ) as span:
            if decision is not None:
                action_record = apply_action(state, agent_name, decision)
                progress_signal = _action_progress_signal(action_record.action, action_record.success)
                if action_record.action == "move" and action_record.success:
                    agent_telemetry["failed_move_streak"] = 0
                    add_span_event(
                        span,
                        EventNames.MOVE_SUCCEEDED,
                        direction=decision.direction,
                        position_after=_position_text(state["agents"][agent_name].position),
                    )
                elif action_record.action == "move" and not action_record.success:
                    agent_telemetry["failed_move_streak"] = int(agent_telemetry["failed_move_streak"]) + 1
                    add_span_event(
                        span,
                        EventNames.BLOCKED_MOVE,
                        direction=decision.direction,
                        failure_reason=_failure_reason(state, agent_name, decision, position_before, door_locked_before),
                        stuck_counter=agent_telemetry["failed_move_streak"],
                    )
                if action_record.action == "pickup_key" and action_record.success:
                    add_span_event(span, EventNames.KEY_PICKED_UP, agent=agent_name)
                if action_record.action == "unlock_door" and action_record.success:
                    add_span_event(span, EventNames.DOOR_UNLOCKED, agent=agent_name)
                if action_record.action == "send_message" and action_record.success:
                    add_span_event(
                        span,
                        EventNames.MESSAGE_QUEUED,
                        recipient=decision.recipient,
                        content=decision.content,
                    )
                if progress_signal != "neutral":
                    _mark_turn_progress(state, progress_signal)
                safe_set_attributes(
                    span,
                    success=action_record.success,
                    failure_reason=_failure_reason(state, agent_name, decision, position_before, door_locked_before)
                    if not action_record.success
                    else None,
                    position_after=_position_text(state["agents"][agent_name].position),
                    progress_signal=progress_signal,
                    door_locked_after=state["world"].door_locked,
                )
                state["current_decision"] = None
                return _state_update(state)
            return _state_update(state)

    def detect_agent_b_divergence(state: GraphState) -> dict[str, object]:
        turn_span = _current_turn_span(state)
        agent_name = "Agent B"
        agent_telemetry = _agent_telemetry(state, agent_name)
        with node_span(
            SpanNames.DIVERGENCE_CHECK,
            parent_span=turn_span,
            turn=state["run"].turn,
            agent=agent_name,
        ) as span:
            divergences = detect_divergences(state, agent_name)
            divergence_types = [record.kind for record in divergences]
            stuck_detected = int(agent_telemetry["failed_move_streak"]) >= 3
            safe_set_attributes(
                span,
                divergence_count=len(divergences),
                divergence_types=divergence_types,
                belief_changed=bool(divergences),
                message_changed_plan=agent_telemetry.get("message_changed_plan", False),
                stuck_detected=stuck_detected,
            )
            if divergences or stuck_detected:
                _mark_turn_progress(state, "regression")
            if stuck_detected:
                add_span_event(
                    span,
                    EventNames.NO_PROGRESS_WARNING,
                    agent=agent_name,
                    stuck_counter=agent_telemetry["failed_move_streak"],
                )
            return _state_update(state)

    def check_termination(state: GraphState) -> dict[str, object]:
        turn_span = _current_turn_span(state)
        run = state["run"]
        world = state["world"]
        agent_a = state["agents"]["Agent A"]
        agent_b = state["agents"]["Agent B"]
        telemetry = _telemetry_state(state)
        with node_span(
            SpanNames.TERMINATION_CHECK,
            parent_span=turn_span,
            turn=run.turn,
            status_before=run.status.value,
            door_locked=world.door_locked,
            agent_a_position=_position_text(agent_a.position),
            agent_b_position=_position_text(agent_b.position),
            agent_a_has_key=agent_a.has_key,
            agent_b_has_key=agent_b.has_key,
        ) as span:
            termination_info = evaluate_termination(state)
            progress_signal = str(telemetry.get("turn_progress_signal", "neutral"))
            if termination_info["status_after"] == RunStatus.SUCCESS.value:
                progress_signal = "progress"
                telemetry["turn_progress_signal"] = "progress"
            safe_set_attributes(
                span,
                status_before=termination_info["status_before"],
                status_after=termination_info["status_after"],
                termination_reason=termination_info["termination_reason"],
                door_locked=world.door_locked,
                agent_a_position=_position_text(agent_a.position),
                agent_b_position=_position_text(agent_b.position),
                agent_a_has_key=agent_a.has_key,
                agent_b_has_key=agent_b.has_key,
            )
            turns_since_progress = int(telemetry.get("turns_since_progress", 0))
            if progress_signal == "progress":
                turns_since_progress = 0
            else:
                turns_since_progress += 1
            telemetry["turns_since_progress"] = turns_since_progress
            if turns_since_progress >= 4:
                add_span_event(
                    span,
                    EventNames.NO_PROGRESS_WARNING,
                    turns_since_progress=turns_since_progress,
                    progress_signal=progress_signal,
                )
            if termination_info["termination_reason"] != "continue":
                add_span_event(
                    span,
                    EventNames.TERMINATION_REACHED,
                    final_status=termination_info["status_after"],
                    termination_reason=termination_info["termination_reason"],
                )
            safe_set_attributes(
                turn_span,
                turn=telemetry.get("turn_started_at", run.turn),
                messages_delivered_count=telemetry.get("turn_messages_delivered_count", 0),
                progress_signal=progress_signal,
                status_before=telemetry.get("turn_status_before", termination_info["status_before"]),
                status_after=termination_info["status_after"],
            )
            telemetry["last_termination_reason"] = termination_info["termination_reason"]
            state["current_agent"] = None
            state["current_decision"] = None
            end_span(turn_span)
            telemetry["current_turn_span"] = None
            telemetry["turn_status_before"] = None
            telemetry["turn_progress_signal"] = "neutral"
            telemetry["turn_messages_delivered_count"] = 0
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
