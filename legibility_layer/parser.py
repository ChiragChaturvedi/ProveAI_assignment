from __future__ import annotations

import re
from pathlib import Path

from .parsed_models import (
    ParsedActionResult,
    ParsedAgentFinalState,
    ParsedDecision,
    ParsedDivergence,
    ParsedEvent,
    ParsedMessage,
    ParsedObservation,
    ParsedRunLog,
    TraceSummaryCounts,
)
from .utils import parse_bool, parse_log_envelope, parse_position, safe_literal_eval

START_RE = re.compile(
    r"^Starting simulation \| run_id=(?P<run_id>\S+) \| seed=(?P<seed>\d+) \| "
    r"max_turns=(?P<max_turns>\d+) \| agent_mode=(?P<agent_mode>[^|]+) \| model=(?P<model>.+)$"
)
CONFIG_RE = re.compile(
    r"^Configured LiteLLM agent policy from environment \| model=(?P<model>[^|]+)"
)
OBSERVATION_RE = re.compile(
    r"^Turn (?P<turn>\d+) \| (?P<agent>Agent [AB]) observation \| "
    r"(?P<summary>.*?) \| visible_objects=(?P<visible_objects>.+)$"
)
DECISION_RE = re.compile(
    r"^Turn (?P<turn>\d+) \| (?P<agent>Agent [AB]) decision \| action=(?P<action>[^|]+) "
    r"\| input=(?P<input>.+?) \| goal=(?P<goal>[^|]+) \| confidence=(?P<confidence>[\d.]+) "
    r"\| reason=(?P<reason>.*)$"
)
MESSAGE_RE = re.compile(
    r"^Turn (?P<turn>\d+) \| message (?P<phase>queued|delivered) \| from=(?P<sender>[^|]+) "
    r"\| to=(?P<recipient>[^|]+)(?: \| deliver_turn=(?P<deliver_turn>\d+))? "
    r"\| content=(?P<content>.*?) \| metadata=(?P<metadata>.*)$"
)
DIVERGENCE_RE = re.compile(
    r"^Turn (?P<turn>\d+) \| (?P<agent>Agent [AB]) divergence \| kind=(?P<kind>[^|]+) "
    r"\| belief=(?P<belief>.*?) \| reality=(?P<reality>.*)$"
)
EVENT_RE = re.compile(
    r"^Turn (?P<turn>\d+) \| event=(?P<event_type>[^|]+) \| (?P<summary>.*)$"
)
FINAL_STATUS_RE = re.compile(r"^Final status:\s+(?P<status>[A-Z_]+)$")
TURNS_EXECUTED_RE = re.compile(r"^Turns executed:\s+(?P<count>\d+)$")
DOOR_LOCKED_RE = re.compile(r"^Door locked:\s+(?P<locked>True|False)$")
FINAL_AGENT_RE = re.compile(
    r"^(?P<agent>Agent [AB]) final state \| position=(?P<position>\([^)]+\)) \| "
    r"inventory=(?P<inventory>.+)$"
)
TRACE_SUMMARY_RE = re.compile(
    r"^Trace summary:\s+observations=(?P<observations>\d+), "
    r"decisions=(?P<decisions>\d+), actions=(?P<actions>\d+), "
    r"divergences=(?P<divergences>\d+), events=(?P<events>\d+)$"
)


class DungeonLogParser:
    def parse_file(self, path: str | Path) -> ParsedRunLog:
        file_path = Path(path)
        parsed = ParsedRunLog(source_file=str(file_path))

        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            timestamp, level, message = parse_log_envelope(raw_line)
            if level is None:
                if raw_line.strip():
                    parsed.raw_errors.append(raw_line.rstrip())
                continue

            if self._parse_config(parsed, message):
                continue
            if self._parse_start(parsed, message):
                continue
            if self._parse_observation(parsed, timestamp, level, message):
                continue
            if self._parse_decision(parsed, timestamp, level, message):
                continue
            if self._parse_message(parsed, timestamp, level, message):
                continue
            if self._parse_divergence(parsed, timestamp, level, message):
                continue
            if self._parse_event(parsed, timestamp, level, message):
                continue
            if self._parse_summary_line(parsed, message):
                continue

            if "llm_error" in message or "llm_schema_error" in message:
                parsed.raw_errors.append(message)

        return parsed

    def _parse_config(self, parsed: ParsedRunLog, message: str) -> bool:
        match = CONFIG_RE.match(message)
        if not match:
            return False
        parsed.model_name = parsed.model_name or match.group("model").strip()
        return True

    def _parse_start(self, parsed: ParsedRunLog, message: str) -> bool:
        match = START_RE.match(message)
        if not match:
            return False
        parsed.run_id = match.group("run_id")
        parsed.model_name = match.group("model").strip()
        parsed.agent_mode = match.group("agent_mode").strip()
        parsed.seed = int(match.group("seed"))
        parsed.max_turns = int(match.group("max_turns"))
        return True

    def _parse_observation(
        self,
        parsed: ParsedRunLog,
        timestamp,
        level: str,
        message: str,
    ) -> bool:
        match = OBSERVATION_RE.match(message)
        if not match:
            return False
        visible_objects = safe_literal_eval(match.group("visible_objects"), [])
        parsed.observations.append(
            ParsedObservation(
                timestamp=timestamp,
                level=level,
                turn=int(match.group("turn")),
                agent=match.group("agent"),
                summary=match.group("summary").strip(),
                visible_objects=[str(item) for item in visible_objects],
            )
        )
        return True

    def _parse_decision(
        self,
        parsed: ParsedRunLog,
        timestamp,
        level: str,
        message: str,
    ) -> bool:
        match = DECISION_RE.match(message)
        if not match:
            return False
        action_input = safe_literal_eval(match.group("input"), {})
        parsed.decisions.append(
            ParsedDecision(
                timestamp=timestamp,
                level=level,
                turn=int(match.group("turn")),
                agent=match.group("agent"),
                action=match.group("action").strip(),
                action_input=action_input if isinstance(action_input, dict) else {},
                goal=match.group("goal").strip(),
                confidence=float(match.group("confidence")),
                reason=match.group("reason").strip(),
            )
        )
        return True

    def _parse_message(
        self,
        parsed: ParsedRunLog,
        timestamp,
        level: str,
        message: str,
    ) -> bool:
        match = MESSAGE_RE.match(message)
        if not match:
            return False
        record = ParsedMessage(
            timestamp=timestamp,
            level=level,
            turn=int(match.group("turn")),
            phase=match.group("phase"),  # type: ignore[arg-type]
            sender=match.group("sender").strip(),
            recipient=match.group("recipient").strip(),
            content=match.group("content").strip(),
            metadata=safe_literal_eval(match.group("metadata"), {}),
            deliver_turn=int(match.group("deliver_turn")) if match.group("deliver_turn") else None,
        )
        if record.phase == "queued":
            parsed.queued_messages.append(record)
        else:
            parsed.delivered_messages.append(record)
        return True

    def _parse_divergence(
        self,
        parsed: ParsedRunLog,
        timestamp,
        level: str,
        message: str,
    ) -> bool:
        match = DIVERGENCE_RE.match(message)
        if not match:
            return False
        parsed.divergences.append(
            ParsedDivergence(
                timestamp=timestamp,
                level=level,
                turn=int(match.group("turn")),
                agent=match.group("agent"),
                kind=match.group("kind").strip(),
                belief=match.group("belief").strip(),
                reality=match.group("reality").strip(),
            )
        )
        return True

    def _parse_event(
        self,
        parsed: ParsedRunLog,
        timestamp,
        level: str,
        message: str,
    ) -> bool:
        match = EVENT_RE.match(message)
        if not match:
            return False
        turn = int(match.group("turn"))
        event_type = match.group("event_type").strip()
        summary = match.group("summary").strip()
        actor = "System"
        if "Agent A" in summary:
            actor = "Agent A"
        elif "Agent B" in summary:
            actor = "Agent B"

        parsed.events.append(
            ParsedEvent(
                timestamp=timestamp,
                level=level,
                turn=turn,
                actor=actor,
                event_type=event_type,
                summary=summary,
            )
        )

        if event_type == "action":
            action, success, agent = self._parse_action_summary(summary)
            if action and agent:
                parsed.action_results.append(
                    ParsedActionResult(
                        timestamp=timestamp,
                        level=level,
                        turn=turn,
                        agent=agent,
                        action=action,
                        success=success,
                        summary=summary,
                    )
                )
        return True

    def _parse_summary_line(self, parsed: ParsedRunLog, message: str) -> bool:
        match = FINAL_STATUS_RE.match(message)
        if match:
            parsed.final_status = match.group("status")
            return True

        match = TURNS_EXECUTED_RE.match(message)
        if match:
            parsed.turns_executed = int(match.group("count"))
            return True

        match = DOOR_LOCKED_RE.match(message)
        if match:
            parsed.final_door_locked = parse_bool(match.group("locked"))
            return True

        match = FINAL_AGENT_RE.match(message)
        if match:
            position = parse_position(match.group("position"))
            if position is not None:
                parsed.final_agent_states.append(
                    ParsedAgentFinalState(
                        agent=match.group("agent"),
                        position=position,
                        inventory=safe_literal_eval(match.group("inventory"), []),
                    )
                )
            return True

        match = TRACE_SUMMARY_RE.match(message)
        if match:
            parsed.trace_summary = TraceSummaryCounts(
                observations=int(match.group("observations")),
                decisions=int(match.group("decisions")),
                actions=int(match.group("actions")),
                divergences=int(match.group("divergences")),
                events=int(match.group("events")),
            )
            return True
        return False

    def _parse_action_summary(self, summary: str) -> tuple[str | None, bool | None, str | None]:
        agent_match = re.search(r"(Agent [AB])", summary)
        agent = agent_match.group(1) if agent_match else None

        patterns: list[tuple[str, str, bool | None]] = [
            (r"(Agent [AB]) moved \w+ to", "move", True),
            (r"(Agent [AB]) failed to move \w+", "move", False),
            (r"(Agent [AB]) picked up the key", "pickup_key", True),
            (r"(Agent [AB]) could not pick up the key", "pickup_key", False),
            (r"(Agent [AB]) unlocked the door", "unlock_door", True),
            (r"(Agent [AB]) could not unlock the door", "unlock_door", False),
            (r"(Agent [AB]) sent a message to", "send_message", True),
            (r"(Agent [AB]) inspected the nearby area", "inspect", True),
            (r"(Agent [AB]) waited", "wait", True),
        ]
        for pattern, action, success in patterns:
            if re.search(pattern, summary):
                return action, success, agent
        return None, None, agent

