from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


MOVE_DIRECTION_ENUM = ["north", "south", "east", "west"]
ACTION_ENUM = ["move", "inspect", "pickup_key", "unlock_door", "send_message", "wait"]

ACTION_DECISION_JSON_SCHEMA = {
    "type": "object",
    "title": "ActionDecision",
    "additionalProperties": False,
    "required": [
        "action",
        "direction",
        "recipient",
        "content",
        "metadata",
        "reason",
        "goal",
        "confidence",
    ],
    "properties": {
        "action": {
            "type": "string",
            "enum": ACTION_ENUM,
        },
        "direction": {
            "type": ["string", "null"],
            "enum": [*MOVE_DIRECTION_ENUM, None],
        },
        "recipient": {
            "type": ["string", "null"],
        },
        "content": {
            "type": ["string", "null"],
        },
        "metadata": {
            "type": ["object", "null"],
            "additionalProperties": True,
        },
        "reason": {
            "type": "string",
            "minLength": 1,
            "maxLength": 160,
        },
        "goal": {
            "type": "string",
            "minLength": 1,
            "maxLength": 40,
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
}

ACTION_DECISION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ActionDecision",
        "strict": True,
        "schema": ACTION_DECISION_JSON_SCHEMA,
    },
}


class Position(BaseModel):
    x: int
    y: int

    def as_tuple(self) -> tuple[int, int]:
        return self.x, self.y


class Message(BaseModel):
    sender: str
    recipient: str
    content: str
    sent_turn: int
    deliver_turn: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationRecord(BaseModel):
    turn: int
    agent: str
    summary: str
    visible_tiles: list[Position] = Field(default_factory=list)
    visible_objects: list[str] = Field(default_factory=list)


class DecisionRecord(BaseModel):
    turn: int
    agent: str
    action: str
    reason: str
    goal: str
    confidence: float


class ActionRecord(BaseModel):
    turn: int
    agent: str
    action: str
    action_input: dict[str, Any] = Field(default_factory=dict)
    success: bool
    summary: str


class DivergenceRecord(BaseModel):
    turn: int
    agent: str
    kind: str
    belief: str
    reality: str


class EventRecord(BaseModel):
    turn: int
    kind: str
    summary: str


class TraceLogs(BaseModel):
    observations: list[ObservationRecord] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    actions: list[ActionRecord] = Field(default_factory=list)
    divergences: list[DivergenceRecord] = Field(default_factory=list)
    events: list[EventRecord] = Field(default_factory=list)


class RunState(BaseModel):
    run_id: str = Field(default_factory=generate_run_id)
    seed: int
    turn: int = 1
    max_turns: int = 20
    status: RunStatus = RunStatus.PENDING


class WorldState(BaseModel):
    grid_width: int
    grid_height: int
    walls: list[Position] = Field(default_factory=list)
    key_position: Position | None = None
    door_position: Position
    exit_position: Position
    door_locked: bool = True


class AgentMemory(BaseModel):
    known_tiles: list[Position] = Field(default_factory=list)
    known_walls: list[Position] = Field(default_factory=list)
    seen_key_position: Position | None = None
    seen_door_position: Position | None = None
    seen_exit_position: Position | None = None
    believed_door_unlocked: bool = False
    believed_teammate_has_key: bool | None = None
    believed_teammate_position: Position | None = None
    last_teammate_position_turn: int | None = None
    reported_facts: list[str] = Field(default_factory=list)
    last_plan_goal: str | None = None


class AgentState(BaseModel):
    name: str
    position: Position
    inventory: list[str] = Field(default_factory=list)
    inbox_messages: list[Message] = Field(default_factory=list)
    local_memory: AgentMemory = Field(default_factory=AgentMemory)

    @property
    def has_key(self) -> bool:
        return "key" in self.inventory


class ActionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["move", "inspect", "pickup_key", "unlock_door", "send_message", "wait"]
    direction: Literal["north", "south", "east", "west"] | None
    recipient: str | None
    content: str | None
    metadata: dict[str, Any] | None
    reason: str = Field(max_length=160)
    goal: str = Field(max_length=40)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_action_input(self) -> "ActionDecision":
        if self.action == "move":
            if self.direction not in MOVE_DIRECTION_ENUM:
                raise ValueError("move actions require a valid direction")
        elif self.direction is not None:
            raise ValueError("only move actions may include a direction")

        if self.action == "send_message":
            if not self.recipient or not self.content:
                raise ValueError("send_message actions require recipient and content")
            if self.metadata is None:
                self.metadata = {}
        else:
            if self.recipient is not None or self.content is not None:
                raise ValueError("only send_message actions may include recipient or content")
            if self.metadata is not None and self.action != "send_message":
                raise ValueError("only send_message actions may include metadata")
        return self

    @property
    def action_input(self) -> dict[str, Any]:
        if self.action == "move":
            return {"direction": self.direction}
        if self.action == "send_message":
            return {
                "recipient": self.recipient,
                "content": self.content,
                "metadata": self.metadata or {},
            }
        return {}


class GraphState(TypedDict):
    run: RunState
    world: WorldState
    agents: dict[str, AgentState]
    pending_messages: list[Message]
    current_agent: str | None
    current_decision: ActionDecision | None
    trace: TraceLogs
    telemetry: dict[str, Any]
