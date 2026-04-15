from __future__ import annotations

import logging
import os
from typing import Protocol

import litellm
from litellm import completion
from litellm.exceptions import JSONSchemaValidationError

from .prompts import LLM_SYSTEM_PROMPT, build_agent_prompt
from .state import (
    ACTION_DECISION_RESPONSE_FORMAT,
    ActionDecision,
    AgentState,
    DecisionRecord,
    GraphState,
    Position,
)
from .world import (
    adjacent_positions,
    frontier_targets,
    is_adjacent,
    pathfind_direction,
    same_position,
    teammate_name,
    update_memory_with_observation,
)

logger = logging.getLogger("dungeon_sim")

litellm.suppress_debug_info = True
litellm.turn_off_message_logging = True
litellm.enable_json_schema_validation = True


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
                direction=None,
                recipient=None,
                content=None,
                metadata=None,
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
                direction=None,
                recipient=teammate.name,
                content=self._message_content(agent),
                metadata=metadata,
                reason="sharing a meaningful state update with the teammate",
                goal="coordinate",
                confidence=0.7,
            )

        if world.key_position and not agent.has_key and is_adjacent(agent.position, world.key_position):
            return ActionDecision(
                action="pickup_key",
                direction=None,
                recipient=None,
                content=None,
                metadata=None,
                reason="the key is within reach",
                goal="retrieve_key",
                confidence=0.99,
            )

        if world.key_position and not agent.has_key and memory.seen_key_position is not None:
            direction = pathfind_direction(world, agent, [memory.seen_key_position], treat_locked_door_as_blocked=True)
            if direction:
                return ActionDecision(
                    action="move",
                    direction=direction,
                    recipient=None,
                    content=None,
                    metadata=None,
                    reason="moving toward visible key",
                    goal="retrieve_key",
                    confidence=0.82,
                )

        if agent.has_key and is_adjacent(agent.position, world.door_position) and world.door_locked:
            return ActionDecision(
                action="unlock_door",
                direction=None,
                recipient=None,
                content=None,
                metadata=None,
                reason="holding the key and standing next to the locked door",
                goal="unlock_door",
                confidence=0.98,
            )

        if agent.has_key and memory.seen_door_position is None:
            direction = pathfind_direction(world, agent, [left_of_door], treat_locked_door_as_blocked=True)
            if direction:
                return ActionDecision(
                    action="move",
                    direction=direction,
                    recipient=None,
                    content=None,
                    metadata=None,
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
                    direction=direction,
                    recipient=None,
                    content=None,
                    metadata=None,
                    reason="moving into position to unlock the door",
                    goal="unlock_door",
                    confidence=0.88,
                )

        if not world.door_locked and memory.seen_exit_position is None:
            direction = pathfind_direction(world, agent, [right_of_door], treat_locked_door_as_blocked=False)
            if direction:
                return ActionDecision(
                    action="move",
                    direction=direction,
                    recipient=None,
                    content=None,
                    metadata=None,
                    reason="the door is open, so push through the corridor to scout the far side",
                    goal="reach_exit",
                    confidence=0.86,
                )

        if not world.door_locked and memory.seen_exit_position is not None:
            direction = pathfind_direction(world, agent, [memory.seen_exit_position], treat_locked_door_as_blocked=False)
            if direction:
                return ActionDecision(
                    action="move",
                    direction=direction,
                    recipient=None,
                    content=None,
                    metadata=None,
                    reason="door is open and the exit is known",
                    goal="reach_exit",
                    confidence=0.9,
                )

        frontier = frontier_targets(world, agent)
        direction = pathfind_direction(world, agent, frontier, treat_locked_door_as_blocked=True)
        if direction:
            return ActionDecision(
                action="move",
                direction=direction,
                recipient=None,
                content=None,
                metadata=None,
                reason="exploring the nearest unknown frontier",
                goal="explore",
                confidence=0.66,
            )

        return ActionDecision(
            action="inspect",
            direction=None,
            recipient=None,
            content=None,
            metadata=None,
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


class LiteLLMJsonAgent:
    """LLM-backed policy using LiteLLM so providers can be swapped through config."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        fallback_policy: AgentPolicy | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_policy = fallback_policy
        self.response_format = ACTION_DECISION_RESPONSE_FORMAT

    def decide_action(self, agent_name: str, state: GraphState) -> ActionDecision:
        agent = state["agents"][agent_name]
        teammate = state["agents"][teammate_name(agent_name)]
        memory = agent.local_memory
        latest_observation = self._latest_observation(agent_name, state)
        prompt = build_agent_prompt(
            agent_name=agent_name,
            position_summary=f"({agent.position.x}, {agent.position.y})",
            visible_summary=latest_observation.summary if latest_observation is not None else "No observation captured yet.",
            memory_summary=(
                f"known_tiles={len(memory.known_tiles)}, "
                f"seen_key={self._format_position(memory.seen_key_position)}, "
                f"seen_door={self._format_position(memory.seen_door_position)}, "
                f"seen_exit={self._format_position(memory.seen_exit_position)}, "
                f"believed_door_unlocked={memory.believed_door_unlocked}, "
                f"believed_teammate_has_key={memory.believed_teammate_has_key}, "
                f"believed_teammate_position={self._format_position(memory.believed_teammate_position)}"
            ),
            inbox_summary=self._summarize_inbox(agent),
            teammate_summary=(
                f"name={teammate.name}, "
                f"last_known_position={self._format_position(memory.believed_teammate_position)}, "
                f"stale_after_turn={memory.last_teammate_position_turn}"
            ),
        )

        try:
            response = completion(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=self.response_format,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            decision = self._extract_structured_decision(response)
            logger.info(
                "Turn %s | %s llm_structured_response | model=%s | decision=%s",
                state["run"].turn,
                agent_name,
                self.model,
                decision.model_dump_json(),
            )
            return self._normalize_decision(decision, agent_name, state)
        except JSONSchemaValidationError as exc:
            raw_response = getattr(exc, "raw_response", None)
            logger.exception(
                "Turn %s | %s llm_schema_error | model=%s | error=%s | raw_response=%s",
                state["run"].turn,
                agent_name,
                self.model,
                exc,
                raw_response,
            )
            if self.fallback_policy is not None:
                logger.warning(
                    "Turn %s | %s falling back to deterministic policy after schema validation failure",
                    state["run"].turn,
                    agent_name,
                )
                return self.fallback_policy.decide_action(agent_name, state)
            raise
        except Exception as exc:
            logger.exception(
                "Turn %s | %s llm_error | model=%s | error=%s",
                state["run"].turn,
                agent_name,
                self.model,
                exc,
            )
            if self.fallback_policy is not None:
                logger.warning(
                    "Turn %s | %s falling back to deterministic policy after LLM failure",
                    state["run"].turn,
                    agent_name,
                )
                return self.fallback_policy.decide_action(agent_name, state)
            raise

    def _latest_observation(self, agent_name: str, state: GraphState):
        for observation in reversed(state["trace"].observations):
            if observation.agent == agent_name:
                return observation
        return None

    def _summarize_inbox(self, agent: AgentState) -> str:
        if not agent.inbox_messages:
            return "No messages."
        latest = agent.inbox_messages[-3:]
        return " | ".join(
            f"from={message.sender}, turn={message.sent_turn}, content={message.content}"
            for message in latest
        )

    def _format_position(self, position: Position | None) -> str:
        if position is None:
            return "unknown"
        return f"({position.x}, {position.y})"

    def _extract_structured_decision(self, response: object) -> ActionDecision:
        message = response.choices[0].message
        parsed = getattr(message, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, ActionDecision):
                return parsed
            return ActionDecision.model_validate(parsed)

        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            logger.warning(
                "LiteLLM returned structured content without message.parsed; "
                "validating message.content against ActionDecision instead."
            )
            return ActionDecision.model_validate_json(content)

        raise ValueError(
            "LiteLLM response did not include a parsed structured object or "
            "a JSON content payload matching ActionDecision."
        )

    def _normalize_decision(self, decision: ActionDecision, agent_name: str, state: GraphState) -> ActionDecision:
        agent = state["agents"][agent_name]
        world = state["world"]
        valid_actions = {"move", "inspect", "pickup_key", "unlock_door", "send_message", "wait"}
        if decision.action not in valid_actions:
            raise ValueError(f"Unsupported action from LLM: {decision.action}")

        if decision.action == "move":
            direction = str(decision.direction or "").lower()
            if direction not in {"north", "south", "east", "west"}:
                fallback = self._get_fallback_decision(agent_name, state)
                if fallback is not None and fallback.action == "move":
                    logger.warning(
                        "Turn %s | %s invalid llm move direction=%s, substituting fallback direction=%s",
                        state["run"].turn,
                        agent_name,
                        direction,
                        fallback.action_input.get("direction"),
                    )
                    return fallback
                raise ValueError(f"Invalid move direction from LLM: {direction}")
            decision.direction = direction

        if decision.action == "pickup_key":
            if world.key_position is None or not is_adjacent(agent.position, world.key_position):
                fallback = self._get_fallback_decision(agent_name, state)
                if fallback is not None:
                    logger.warning(
                        "Turn %s | %s invalid llm pickup_key, substituting fallback action=%s",
                        state["run"].turn,
                        agent_name,
                        fallback.action,
                    )
                    return fallback

        if decision.action == "unlock_door":
            if not agent.has_key or not world.door_locked or not is_adjacent(agent.position, world.door_position):
                fallback = self._get_fallback_decision(agent_name, state)
                if fallback is not None:
                    logger.warning(
                        "Turn %s | %s invalid llm unlock_door, substituting fallback action=%s",
                        state["run"].turn,
                        agent_name,
                        fallback.action,
                    )
                    return fallback

        if decision.action == "send_message":
            decision.recipient = decision.recipient or teammate_name(agent_name)
            decision.content = decision.content or "Status update."
            if not isinstance(decision.metadata, dict):
                decision.metadata = {}

        if decision.action in {"inspect", "pickup_key", "unlock_door", "wait"}:
            decision.direction = None
            decision.recipient = None
            decision.content = None
            decision.metadata = None

        decision.confidence = max(0.0, min(1.0, float(decision.confidence)))
        return decision

    def _get_fallback_decision(self, agent_name: str, state: GraphState) -> ActionDecision | None:
        if self.fallback_policy is None:
            return None
        return self.fallback_policy.decide_action(agent_name, state)


def build_agent_policy_from_env() -> AgentPolicy:
    mode = os.getenv("DUNGEON_AGENT_MODE", "litellm").strip().lower()
    deterministic = DummyDeterministicAgent()

    if mode == "deterministic":
        logger.info("Configured deterministic agent policy from environment.")
        return deterministic

    model = os.getenv("DUNGEON_AGENT_MODEL", "gemini/gemini-2.5-flash").strip()
    temperature = float(os.getenv("DUNGEON_AGENT_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("DUNGEON_AGENT_MAX_TOKENS", "512"))
    fallback_enabled = os.getenv("DUNGEON_AGENT_FALLBACK_TO_DETERMINISTIC", "true").strip().lower() == "true"

    logger.info(
        "Configured LiteLLM agent policy from environment | model=%s | temperature=%s | max_tokens=%s | fallback=%s",
        model,
        temperature,
        max_tokens,
        fallback_enabled,
    )
    return LiteLLMJsonAgent(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        fallback_policy=deterministic if fallback_enabled else None,
    )


def observe_and_decide(agent_name: str, state: GraphState, policy: AgentPolicy) -> ActionDecision:
    agent = state["agents"][agent_name]
    observation = update_memory_with_observation(agent, state["world"], state["run"].turn)
    state["trace"].observations.append(observation)
    logger.info(
        "Turn %s | %s observation | %s | visible_objects=%s",
        state["run"].turn,
        agent_name,
        observation.summary,
        observation.visible_objects or ["none"],
    )

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
    logger.info(
        "Turn %s | %s decision | action=%s | input=%s | goal=%s | confidence=%.2f | reason=%s",
        state["run"].turn,
        agent_name,
        decision.action,
        decision.action_input,
        decision.goal,
        decision.confidence,
        decision.reason,
    )
    return decision


# TODO: add an LLMJsonAgent policy that consumes prompts.py and returns ActionDecision.
