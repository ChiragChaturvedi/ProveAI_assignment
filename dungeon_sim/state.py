from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


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
    run_id: str = Field(default_factory=lambda: str(uuid4()))
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
    action: str
    action_input: dict[str, Any] = Field(default_factory=dict)
    reason: str
    goal: str
    confidence: float


class GraphState(TypedDict):
    run: RunState
    world: WorldState
    agents: dict[str, AgentState]
    pending_messages: list[Message]
    current_agent: str | None
    current_decision: ActionDecision | None
    trace: TraceLogs

