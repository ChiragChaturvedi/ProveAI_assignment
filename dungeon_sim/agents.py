from __future__ import annotations

from typing import Protocol

from .state import ActionDecision, AgentState, DecisionRecord, GraphState, Position
from .world import (
    adjacent_positions,
    frontier_targets,
    is_adjacent,
    pathfind_direction,
    same_position,
    teammate_name,
    update_memory_with_observation,
)


class AgentPolicy(Protocol):
    def decide_action(self, agent_name: str, state: GraphState) -> ActionDecision:
        """Return the next action for the given agent."""


class DummyDeterministicAgent:
    """Simple, legible policy with a clean interface for swapping in an LLM later."""

    def decide_action(self, agent_name: str, state: GraphState) -> ActionDecision:
        world = state["world"]
        run = state["run"]
        agent = state["agents"][agent_name]
        teammate = state["agents"][teammate_name(agent_name)]
        memory = agent.local_memory
        left_of_door = Position(x=max(world.door_position.x - 1, 0), y=world.door_position.y)
        right_of_door = Position(
            x=min(world.door_position.x + 1, world.grid_width - 1),
            y=world.door_position.y,
        )

        if not world.door_locked and same_position(agent.position, world.exit_position):
            return ActionDecision(
                action="wait",
                action_input={},
                reason="already at the exit and waiting for the teammate",
                goal="reach_exit",
                confidence=0.99,
            )

        if self._should_send_message(agent, teammate, run.turn):
            metadata = {
                "position": agent.position.model_dump(),
                "teammate_has_key": agent.has_key,
                "door_unlocked": not world.door_locked,
            }
            if memory.seen_key_position is not None:
                metadata["key_position"] = memory.seen_key_position.model_dump()
            if memory.seen_door_position is not None:
                metadata["door_position"] = memory.seen_door_position.model_dump()
            if memory.seen_exit_position is not None:
                metadata["exit_position"] = memory.seen_exit_position.model_dump()
            return ActionDecision(
                action="send_message",
                action_input={
                    "recipient": teammate.name,
                    "content": self._message_content(agent),
                    "metadata": metadata,
                },
                reason="sharing a meaningful state update with the teammate",
                goal="coordinate",
                confidence=0.7,
            )

        if world.key_position and not agent.has_key and is_adjacent(agent.position, world.key_position):
            return ActionDecision(
                action="pickup_key",
                action_input={},
                reason="the key is within reach",
                goal="retrieve_key",
                confidence=0.99,
            )

        if world.key_position and not agent.has_key and memory.seen_key_position is not None:
            direction = pathfind_direction(world, agent, [memory.seen_key_position], treat_locked_door_as_blocked=True)
            if direction:
                return ActionDecision(
                    action="move",
                    action_input={"direction": direction},
                    reason="moving toward visible key",
                    goal="retrieve_key",
                    confidence=0.82,
                )

        if agent.has_key and is_adjacent(agent.position, world.door_position) and world.door_locked:
            return ActionDecision(
                action="unlock_door",
                action_input={},
                reason="holding the key and standing next to the locked door",
                goal="unlock_door",
                confidence=0.98,
            )

        if agent.has_key and memory.seen_door_position is None:
            direction = pathfind_direction(world, agent, [left_of_door], treat_locked_door_as_blocked=True)
            if direction:
                return ActionDecision(
                    action="move",
                    action_input={"direction": direction},
                    reason="carrying the key and searching the center corridor for the door",
                    goal="find_door",
                    confidence=0.84,
                )

        if agent.has_key and memory.seen_door_position is not None and world.door_locked:
            door_neighbors = adjacent_positions(world, world.door_position)
            direction = pathfind_direction(world, agent, door_neighbors, treat_locked_door_as_blocked=True)
            if direction:
                return ActionDecision(
                    action="move",
                    action_input={"direction": direction},
                    reason="moving into position to unlock the door",
                    goal="unlock_door",
                    confidence=0.88,
                )

        if not world.door_locked and memory.seen_exit_position is None:
            direction = pathfind_direction(world, agent, [right_of_door], treat_locked_door_as_blocked=False)
            if direction:
                return ActionDecision(
                    action="move",
                    action_input={"direction": direction},
                    reason="the door is open, so push through the corridor to scout the far side",
                    goal="reach_exit",
                    confidence=0.86,
                )

        if not world.door_locked and memory.seen_exit_position is not None:
            direction = pathfind_direction(world, agent, [memory.seen_exit_position], treat_locked_door_as_blocked=False)
            if direction:
                return ActionDecision(
                    action="move",
                    action_input={"direction": direction},
                    reason="door is open and the exit is known",
                    goal="reach_exit",
                    confidence=0.9,
                )

        frontier = frontier_targets(world, agent)
        direction = pathfind_direction(world, agent, frontier, treat_locked_door_as_blocked=True)
        if direction:
            return ActionDecision(
                action="move",
                action_input={"direction": direction},
                reason="exploring the nearest unknown frontier",
                goal="explore",
                confidence=0.66,
            )

        return ActionDecision(
            action="inspect",
            action_input={},
            reason="no strong move available, so gather more information",
            goal="explore",
            confidence=0.55,
        )

    def _should_send_message(self, agent: AgentState, teammate: AgentState, turn: int) -> bool:
        memory = agent.local_memory
        reportable_facts = {
            "has_key": agent.has_key,
            "saw_exit": memory.seen_exit_position is not None,
            "saw_door": memory.seen_door_position is not None,
            "door_unlocked": memory.believed_door_unlocked,
        }

        for fact_name, enabled in reportable_facts.items():
            if enabled and fact_name not in memory.reported_facts:
                memory.reported_facts.append(fact_name)
                return True

        position_update_due = turn % 6 == 0 and (agent.has_key or memory.believed_door_unlocked)
        if position_update_due:
            return True
        return False

    def _message_content(self, agent: AgentState) -> str:
        details = [f"position=({agent.position.x},{agent.position.y})"]
        if agent.has_key:
            details.append("holding_key=true")
        if agent.local_memory.believed_door_unlocked:
            details.append("door_unlocked=true")
        return "; ".join(details)


def observe_and_decide(agent_name: str, state: GraphState, policy: AgentPolicy) -> ActionDecision:
    agent = state["agents"][agent_name]
    observation = update_memory_with_observation(agent, state["world"], state["run"].turn)
    state["trace"].observations.append(observation)

    decision = policy.decide_action(agent_name, state)
    agent.local_memory.last_plan_goal = decision.goal
    state["trace"].decisions.append(
        DecisionRecord(
            turn=state["run"].turn,
            agent=agent_name,
            action=decision.action,
            reason=decision.reason,
            goal=decision.goal,
            confidence=decision.confidence,
        )
    )
    return decision


# TODO: add an LLMJsonAgent policy that consumes prompts.py and returns ActionDecision.
