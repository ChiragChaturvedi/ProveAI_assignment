from __future__ import annotations

from collections import deque
import logging
from typing import Iterable

from .state import (
    ActionDecision,
    ActionRecord,
    AgentMemory,
    AgentState,
    DivergenceRecord,
    EventRecord,
    GraphState,
    Message,
    ObservationRecord,
    Position,
    RunState,
    RunStatus,
    TraceLogs,
    WorldState,
)

VISIBLE_RADIUS = 2
logger = logging.getLogger("dungeon_sim")

DIRECTIONS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
}


def create_initial_state(seed: int = 7, max_turns: int = 20) -> GraphState:
    world = WorldState(
        grid_width=7,
        grid_height=7,
        walls=[
            Position(x=3, y=0),
            Position(x=3, y=1),
            Position(x=3, y=2),
            Position(x=3, y=4),
            Position(x=3, y=5),
            Position(x=3, y=6),
        ],
        key_position=Position(x=1, y=5),
        door_position=Position(x=3, y=3),
        exit_position=Position(x=6, y=3),
        door_locked=True,
    )

    agents = {
        "Agent A": AgentState(name="Agent A", position=Position(x=0, y=0)),
        "Agent B": AgentState(name="Agent B", position=Position(x=0, y=6)),
    }

    return {
        "run": RunState(seed=seed, max_turns=max_turns, status=RunStatus.PENDING),
        "world": world,
        "agents": agents,
        "pending_messages": [],
        "current_agent": None,
        "current_decision": None,
        "trace": TraceLogs(),
    }


def in_bounds(world: WorldState, position: Position) -> bool:
    return 0 <= position.x < world.grid_width and 0 <= position.y < world.grid_height


def same_position(left: Position | None, right: Position | None) -> bool:
    if left is None or right is None:
        return False
    return left.x == right.x and left.y == right.y


def is_adjacent(left: Position, right: Position) -> bool:
    return abs(left.x - right.x) + abs(left.y - right.y) <= 1


def position_in_list(position: Position, positions: Iterable[Position]) -> bool:
    return any(same_position(position, item) for item in positions)


def unique_positions(positions: Iterable[Position]) -> list[Position]:
    seen: set[tuple[int, int]] = set()
    result: list[Position] = []
    for position in positions:
        key = position.as_tuple()
        if key in seen:
            continue
        seen.add(key)
        result.append(position)
    return result


def add_event(state: GraphState, kind: str, summary: str) -> None:
    state["trace"].events.append(
        EventRecord(turn=state["run"].turn, kind=kind, summary=summary)
    )
    logger.info("Turn %s | event=%s | %s", state["run"].turn, kind, summary)


def wall_at(world: WorldState, position: Position) -> bool:
    return position_in_list(position, world.walls)


def blocked_by_locked_door(world: WorldState, position: Position) -> bool:
    return world.door_locked and same_position(position, world.door_position)


def is_walkable(world: WorldState, position: Position) -> bool:
    return in_bounds(world, position) and not wall_at(world, position) and not blocked_by_locked_door(world, position)


def move_position(position: Position, direction: str) -> Position:
    dx, dy = DIRECTIONS[direction]
    return Position(x=position.x + dx, y=position.y + dy)


def visible_tiles(world: WorldState, origin: Position) -> list[Position]:
    tiles: list[Position] = []
    for x in range(world.grid_width):
        for y in range(world.grid_height):
            candidate = Position(x=x, y=y)
            if abs(candidate.x - origin.x) + abs(candidate.y - origin.y) <= VISIBLE_RADIUS:
                tiles.append(candidate)
    return tiles


def update_memory_with_observation(agent: AgentState, world: WorldState, turn: int) -> ObservationRecord:
    visible = visible_tiles(world, agent.position)
    memory = agent.local_memory

    memory.known_tiles = unique_positions([*memory.known_tiles, *visible])

    walls = [tile for tile in visible if wall_at(world, tile)]
    memory.known_walls = unique_positions([*memory.known_walls, *walls])

    visible_objects: list[str] = []
    if world.key_position and position_in_list(world.key_position, visible):
        memory.seen_key_position = world.key_position.model_copy()
        visible_objects.append("key")
    if position_in_list(world.door_position, visible):
        memory.seen_door_position = world.door_position.model_copy()
        memory.believed_door_unlocked = not world.door_locked
        visible_objects.append("door")
    if position_in_list(world.exit_position, visible):
        memory.seen_exit_position = world.exit_position.model_copy()
        visible_objects.append("exit")

    summary = (
        f"{agent.name} observed {len(visible)} tiles and saw "
        f"{', '.join(visible_objects) if visible_objects else 'no objectives'}."
    )
    return ObservationRecord(
        turn=turn,
        agent=agent.name,
        summary=summary,
        visible_tiles=visible,
        visible_objects=visible_objects,
    )


def teammate_name(agent_name: str) -> str:
    return "Agent B" if agent_name == "Agent A" else "Agent A"


def apply_message_to_memory(agent: AgentState, message: Message) -> None:
    memory = agent.local_memory
    if "teammate_has_key" in message.metadata:
        memory.believed_teammate_has_key = bool(message.metadata["teammate_has_key"])
    if "door_unlocked" in message.metadata:
        memory.believed_door_unlocked = bool(message.metadata["door_unlocked"])
    if "position" in message.metadata:
        position_data = message.metadata["position"]
        memory.believed_teammate_position = Position.model_validate(position_data)
        memory.last_teammate_position_turn = message.sent_turn
    if "exit_position" in message.metadata:
        memory.seen_exit_position = Position.model_validate(message.metadata["exit_position"])
    if "door_position" in message.metadata:
        memory.seen_door_position = Position.model_validate(message.metadata["door_position"])
    if "key_position" in message.metadata and message.metadata["key_position"] is not None:
        memory.seen_key_position = Position.model_validate(message.metadata["key_position"])


def deliver_pending_messages(state: GraphState) -> None:
    run = state["run"]
    ready = [message for message in state["pending_messages"] if message.deliver_turn <= run.turn]
    remaining = [message for message in state["pending_messages"] if message.deliver_turn > run.turn]

    for message in ready:
        recipient = state["agents"][message.recipient]
        recipient.inbox_messages.append(message)
        apply_message_to_memory(recipient, message)
        logger.info(
            "Turn %s | message delivered | from=%s | to=%s | content=%s | metadata=%s",
            run.turn,
            message.sender,
            message.recipient,
            message.content,
            message.metadata,
        )
        add_event(
            state,
            "message_delivered",
            f"Turn {run.turn}: Message delivered from {message.sender} to {message.recipient}.",
        )

    state["pending_messages"] = remaining


def pathfind_direction(
    world: WorldState,
    agent: AgentState,
    targets: list[Position],
    treat_locked_door_as_blocked: bool = True,
) -> str | None:
    if not targets:
        return None

    blocked_walls = {wall.as_tuple() for wall in agent.local_memory.known_walls}
    queue: deque[tuple[Position, list[str]]] = deque([(agent.position.model_copy(), [])])
    visited = {agent.position.as_tuple()}
    target_keys = {target.as_tuple() for target in targets}

    while queue:
        current, path = queue.popleft()
        if current.as_tuple() in target_keys and path:
            return path[0]

        for direction, (dx, dy) in DIRECTIONS.items():
            nxt = Position(x=current.x + dx, y=current.y + dy)
            nxt_key = nxt.as_tuple()
            if nxt_key in visited or not in_bounds(world, nxt):
                continue
            if nxt_key in blocked_walls:
                continue
            if treat_locked_door_as_blocked and world.door_locked and same_position(nxt, world.door_position):
                continue
            visited.add(nxt_key)
            queue.append((nxt, [*path, direction]))

    return None


def adjacent_positions(world: WorldState, position: Position) -> list[Position]:
    results: list[Position] = []
    for direction in DIRECTIONS:
        candidate = move_position(position, direction)
        if in_bounds(world, candidate):
            results.append(candidate)
    return results


def frontier_targets(world: WorldState, agent: AgentState) -> list[Position]:
    known = {position.as_tuple() for position in agent.local_memory.known_tiles}
    blocked = {position.as_tuple() for position in agent.local_memory.known_walls}
    frontiers: list[Position] = []
    for tile in agent.local_memory.known_tiles:
        if tile.as_tuple() in blocked:
            continue
        unknown_neighbors = [
            neighbor
            for neighbor in adjacent_positions(world, tile)
            if neighbor.as_tuple() not in known and neighbor.as_tuple() not in blocked
        ]
        if unknown_neighbors:
            frontiers.append(tile)
    return unique_positions(frontiers)


def move_summary(agent_name: str, direction: str, success: bool, position: Position) -> str:
    if success:
        return (
            f"Turn {{turn}}: {agent_name} moved {direction} to ({position.x}, {position.y})."
        )
    return f"Turn {{turn}}: {agent_name} failed to move {direction}."


def apply_action(state: GraphState, agent_name: str, decision: ActionDecision) -> ActionRecord:
    run = state["run"]
    world = state["world"]
    agent = state["agents"][agent_name]
    summary = ""
    success = False

    if decision.action == "move":
        direction = str(decision.action_input["direction"])
        next_position = move_position(agent.position, direction)
        if is_walkable(world, next_position):
            agent.position = next_position
            success = True
        summary = move_summary(agent_name, direction, success, agent.position).format(turn=run.turn)
    elif decision.action == "inspect":
        observation = update_memory_with_observation(agent, world, run.turn)
        state["trace"].observations.append(observation)
        success = True
        summary = f"Turn {run.turn}: {agent_name} inspected the nearby area."
    elif decision.action == "pickup_key":
        if world.key_position and is_adjacent(agent.position, world.key_position):
            agent.inventory.append("key")
            world.key_position = None
            agent.local_memory.seen_key_position = None
            success = True
        summary = f"Turn {run.turn}: {agent_name} {'picked up the key' if success else 'could not pick up the key'}."
    elif decision.action == "unlock_door":
        if agent.has_key and world.door_locked and is_adjacent(agent.position, world.door_position):
            world.door_locked = False
            agent.local_memory.believed_door_unlocked = True
            success = True
        summary = f"Turn {run.turn}: {agent_name} {'unlocked the door' if success else 'could not unlock the door'}."
    elif decision.action == "send_message":
        message = Message(
            sender=agent_name,
            recipient=str(decision.action_input["recipient"]),
            content=str(decision.action_input["content"]),
            sent_turn=run.turn,
            deliver_turn=run.turn + 1,
            metadata=decision.action_input.get("metadata", {}),
        )
        state["pending_messages"].append(message)
        success = True
        logger.info(
            "Turn %s | message queued | from=%s | to=%s | deliver_turn=%s | content=%s | metadata=%s",
            run.turn,
            message.sender,
            message.recipient,
            message.deliver_turn,
            message.content,
            message.metadata,
        )
        summary = f"Turn {run.turn}: {agent_name} sent a message to {message.recipient}."
    elif decision.action == "wait":
        success = True
        summary = f"Turn {run.turn}: {agent_name} waited."
    else:
        summary = f"Turn {run.turn}: {agent_name} attempted an unknown action."

    action_record = ActionRecord(
        turn=run.turn,
        agent=agent_name,
        action=decision.action,
        action_input=decision.action_input,
        success=success,
        summary=summary,
    )
    state["trace"].actions.append(action_record)
    add_event(state, "action", summary)
    return action_record


def detect_divergences(state: GraphState, agent_name: str) -> list[DivergenceRecord]:
    run = state["run"]
    world = state["world"]
    agent = state["agents"][agent_name]
    teammate = state["agents"][teammate_name(agent_name)]
    memory = agent.local_memory
    found: list[DivergenceRecord] = []

    if memory.believed_teammate_has_key and not teammate.has_key:
        found.append(
            DivergenceRecord(
                turn=run.turn,
                agent=agent_name,
                kind="belief_teammate_has_key",
                belief=f"{teammate.name} has the key",
                reality=f"{teammate.name} does not have the key",
            )
        )

    if memory.believed_door_unlocked and world.door_locked:
        found.append(
            DivergenceRecord(
                turn=run.turn,
                agent=agent_name,
                kind="belief_door_unlocked",
                belief="door is unlocked",
                reality="door is still locked",
            )
        )

    if (
        memory.believed_teammate_position is not None
        and memory.last_teammate_position_turn is not None
        and run.turn - memory.last_teammate_position_turn >= 2
        and not same_position(memory.believed_teammate_position, teammate.position)
    ):
        found.append(
            DivergenceRecord(
                turn=run.turn,
                agent=agent_name,
                kind="stale_teammate_location",
                belief=(
                    f"{teammate.name} is at "
                    f"({memory.believed_teammate_position.x}, {memory.believed_teammate_position.y})"
                ),
                reality=f"{teammate.name} is elsewhere",
            )
        )

    for record in found:
        state["trace"].divergences.append(record)
        logger.warning(
            "Turn %s | %s divergence | kind=%s | belief=%s | reality=%s",
            run.turn,
            agent_name,
            record.kind,
            record.belief,
            record.reality,
        )
        add_event(
            state,
            "divergence",
            f"Turn {run.turn}: {agent_name} divergence detected - {record.kind}.",
        )
    return found


def evaluate_termination(state: GraphState) -> None:
    run = state["run"]
    world = state["world"]
    agents = list(state["agents"].values())

    everyone_at_exit = all(same_position(agent.position, world.exit_position) for agent in agents)
    if everyone_at_exit and not world.door_locked:
        run.status = RunStatus.SUCCESS
        add_event(state, "status", f"Turn {run.turn}: SUCCESS in {run.turn} turns.")
        return

    if run.turn >= run.max_turns:
        run.status = RunStatus.FAILED
        add_event(state, "status", f"Turn {run.turn}: FAILED after reaching max turns.")
        return

    run.status = RunStatus.RUNNING
    run.turn += 1


def format_trace_summary(state: GraphState) -> str:
    trace = state["trace"]
    return (
        f"observations={len(trace.observations)}, "
        f"decisions={len(trace.decisions)}, "
        f"actions={len(trace.actions)}, "
        f"divergences={len(trace.divergences)}, "
        f"events={len(trace.events)}"
    )
