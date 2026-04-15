from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TimelineRow(BaseModel):
    turn: int | None = None
    actor: str
    event_type: str
    short_summary: str
    progress_signal: Literal["progress", "neutral", "regression"]
    evidence_tags: list[str] = Field(default_factory=list)


class AgentReviewSummary(BaseModel):
    agent: str
    final_position: str
    inventory: list[str] = Field(default_factory=list)
    decisions: int = 0
    failed_moves: int = 0
    repeated_actions: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    notable_issues: list[str] = Field(default_factory=list)
    summary: str


class RunMetrics(BaseModel):
    failed_moves_per_agent: dict[str, int] = Field(default_factory=dict)
    repeated_same_action_count: dict[str, int] = Field(default_factory=dict)
    messages_sent_per_agent: dict[str, int] = Field(default_factory=dict)
    messages_delivered_per_agent: dict[str, int] = Field(default_factory=dict)
    turns_since_last_progress: int = 0
    key_acquired: bool = False
    door_discovered: bool = False
    door_unlocked: bool = False
    exit_reached: bool = False
    long_no_progress_stretches: list[str] = Field(default_factory=list)
    repeated_blocked_actions: list[str] = Field(default_factory=list)
    redundant_messages: list[str] = Field(default_factory=list)
    message_without_behavior_change_cases: list[str] = Field(default_factory=list)


class RunReview(BaseModel):
    run_id: str
    status: str
    turn_count: int
    primary_failure: str | None = None
    contributing_factors: list[str] = Field(default_factory=list)
    executive_summary: str
    key_moments: list[str] = Field(default_factory=list)
    agent_summaries: list[AgentReviewSummary] = Field(default_factory=list)
    metrics: RunMetrics
    recommendations: list[str] = Field(default_factory=list)
    timeline_rows: list[TimelineRow] = Field(default_factory=list)

