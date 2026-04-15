from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Coordinate(BaseModel):
    x: int
    y: int


class TraceSummaryCounts(BaseModel):
    observations: int = 0
    decisions: int = 0
    actions: int = 0
    divergences: int = 0
    events: int = 0


class ParsedObservation(BaseModel):
    timestamp: datetime | None = None
    level: str
    turn: int
    agent: str
    summary: str
    visible_objects: list[str] = Field(default_factory=list)


class ParsedDecision(BaseModel):
    timestamp: datetime | None = None
    level: str
    turn: int
    agent: str
    action: str
    action_input: dict[str, Any] = Field(default_factory=dict)
    goal: str
    confidence: float
    reason: str


class ParsedActionResult(BaseModel):
    timestamp: datetime | None = None
    level: str
    turn: int
    agent: str
    action: str
    success: bool | None = None
    summary: str


class ParsedMessage(BaseModel):
    timestamp: datetime | None = None
    level: str
    turn: int
    phase: Literal["queued", "delivered"]
    sender: str
    recipient: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    deliver_turn: int | None = None


class ParsedDivergence(BaseModel):
    timestamp: datetime | None = None
    level: str
    turn: int
    agent: str
    kind: str
    belief: str
    reality: str


class ParsedEvent(BaseModel):
    timestamp: datetime | None = None
    level: str
    turn: int | None = None
    actor: str
    event_type: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class ParsedAgentFinalState(BaseModel):
    agent: str
    position: Coordinate
    inventory: list[str] = Field(default_factory=list)


class ParsedRunLog(BaseModel):
    source_file: str
    run_id: str | None = None
    model_name: str | None = None
    agent_mode: str | None = None
    seed: int | None = None
    max_turns: int | None = None
    final_status: str | None = None
    turns_executed: int | None = None
    final_door_locked: bool | None = None
    observations: list[ParsedObservation] = Field(default_factory=list)
    decisions: list[ParsedDecision] = Field(default_factory=list)
    action_results: list[ParsedActionResult] = Field(default_factory=list)
    queued_messages: list[ParsedMessage] = Field(default_factory=list)
    delivered_messages: list[ParsedMessage] = Field(default_factory=list)
    divergences: list[ParsedDivergence] = Field(default_factory=list)
    events: list[ParsedEvent] = Field(default_factory=list)
    final_agent_states: list[ParsedAgentFinalState] = Field(default_factory=list)
    trace_summary: TraceSummaryCounts = Field(default_factory=TraceSummaryCounts)
    raw_errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

