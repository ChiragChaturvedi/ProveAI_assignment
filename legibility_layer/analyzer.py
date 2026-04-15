from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from .parsed_models import (
    ParsedActionResult,
    ParsedDecision,
    ParsedMessage,
    ParsedObservation,
    ParsedRunLog,
)
from .review_models import AgentReviewSummary, RunMetrics, RunReview, TimelineRow
from .utils import compact_text, normalize_message_content


class RunAnalyzer:
    def analyze(self, run: ParsedRunLog) -> RunReview:
        failed_moves = self._failed_moves_by_agent(run.action_results)
        repeated_actions = self._repeated_actions_by_agent(run.decisions)
        messages_sent = Counter(message.sender for message in run.queued_messages)
        messages_delivered = Counter(message.recipient for message in run.delivered_messages)

        progress_turns, key_moments = self._progress_turns_and_key_moments(run)
        no_progress_stretches = self._no_progress_stretches(run.turns_executed or 0, progress_turns)
        repeated_blocked_actions = self._repeated_blocked_actions(run.action_results)
        redundant_messages = self._redundant_messages(run.queued_messages)
        behavior_misses = self._message_without_behavior_change(run.delivered_messages, run.decisions)

        key_acquired = any(
            action.action == "pickup_key" and action.success for action in run.action_results
        ) or any("key" in state.inventory for state in run.final_agent_states)
        door_discovered = any(
            "door" in observation.visible_objects for observation in run.observations
        ) or any("door" in normalize_message_content(message.content) for message in run.delivered_messages)
        door_unlocked = any(
            action.action == "unlock_door" and action.success for action in run.action_results
        ) or run.final_door_locked is False
        exit_reached = (run.final_status or "").upper() == "SUCCESS"

        metrics = RunMetrics(
            failed_moves_per_agent=dict(failed_moves),
            repeated_same_action_count=dict(repeated_actions),
            messages_sent_per_agent=dict(messages_sent),
            messages_delivered_per_agent=dict(messages_delivered),
            turns_since_last_progress=self._turns_since_last_progress(
                run.turns_executed or 0,
                progress_turns,
            ),
            key_acquired=key_acquired,
            door_discovered=door_discovered,
            door_unlocked=door_unlocked,
            exit_reached=exit_reached,
            long_no_progress_stretches=no_progress_stretches,
            repeated_blocked_actions=repeated_blocked_actions,
            redundant_messages=redundant_messages,
            message_without_behavior_change_cases=behavior_misses,
        )

        contributing_factors = self._derive_failure_labels(run, metrics)
        primary_failure = contributing_factors[0] if contributing_factors else None
        agent_summaries = self._build_agent_summaries(
            run,
            failed_moves=failed_moves,
            repeated_actions=repeated_actions,
            messages_sent=messages_sent,
            messages_delivered=messages_delivered,
        )
        timeline_rows = self._build_timeline(run)
        recommendations = self._recommendations(contributing_factors, metrics)
        executive_summary = self._executive_summary(run, metrics, primary_failure, agent_summaries)

        if run.final_status and run.final_status.upper() == "FAILED" and "Run reached max turns." not in key_moments:
            key_moments.append(f"Run ended with {run.final_status.upper()} after {run.turns_executed or 0} turns.")

        return RunReview(
            run_id=run.run_id or "unknown_run",
            status=run.final_status or "UNKNOWN",
            turn_count=run.turns_executed or 0,
            primary_failure=primary_failure,
            contributing_factors=contributing_factors,
            executive_summary=executive_summary,
            key_moments=key_moments[:10],
            agent_summaries=agent_summaries,
            metrics=metrics,
            recommendations=recommendations,
            timeline_rows=timeline_rows,
        )

    def _failed_moves_by_agent(self, actions: Iterable[ParsedActionResult]) -> Counter[str]:
        counter: Counter[str] = Counter()
        for action in actions:
            if action.action == "move" and action.success is False:
                counter[action.agent] += 1
        return counter

    def _repeated_actions_by_agent(self, decisions: list[ParsedDecision]) -> Counter[str]:
        counter: Counter[str] = Counter()
        per_agent: dict[str, ParsedDecision | None] = {"Agent A": None, "Agent B": None}
        for decision in decisions:
            previous = per_agent.get(decision.agent)
            same_as_previous = (
                previous is not None
                and previous.action == decision.action
                and previous.action_input == decision.action_input
            )
            if same_as_previous:
                counter[decision.agent] += 1
            per_agent[decision.agent] = decision
        return counter

    def _progress_turns_and_key_moments(self, run: ParsedRunLog) -> tuple[list[int], list[str]]:
        progress_turns: list[int] = []
        key_moments: list[str] = []
        seen_objects: set[str] = set()

        for observation in sorted(run.observations, key=lambda item: (item.turn, item.agent)):
            for obj in observation.visible_objects:
                if obj in {"key", "door", "exit"} and obj not in seen_objects:
                    seen_objects.add(obj)
                    progress_turns.append(observation.turn)
                    key_moments.append(f"Turn {observation.turn}: {observation.agent} first observed the {obj}.")

        for action in sorted(run.action_results, key=lambda item: (item.turn, item.agent)):
            if action.action == "pickup_key" and action.success:
                progress_turns.append(action.turn)
                key_moments.append(f"Turn {action.turn}: {action.agent} acquired the key.")
            elif action.action == "unlock_door" and action.success:
                progress_turns.append(action.turn)
                key_moments.append(f"Turn {action.turn}: {action.agent} unlocked the door.")
            elif action.action == "send_message":
                normalized = action.summary.lower()
                if "sent a message" in normalized:
                    key_moments.append(f"Turn {action.turn}: {action.agent} shared information with the teammate.")

        if run.final_status and run.final_status.upper() == "SUCCESS":
            progress_turns.append(run.turns_executed or 0)
            key_moments.append(f"Turn {run.turns_executed or 0}: both agents reached the exit.")

        return sorted(set(progress_turns)), key_moments

    def _no_progress_stretches(self, turn_count: int, progress_turns: list[int]) -> list[str]:
        stretches: list[str] = []
        checkpoints = [0, *sorted(progress_turns), turn_count]
        for start, end in zip(checkpoints, checkpoints[1:]):
            gap = end - start
            if gap >= 5:
                stretches.append(f"Turns {start + 1}-{end}: {gap} turns without meaningful progress.")
        return stretches

    def _turns_since_last_progress(self, turn_count: int, progress_turns: list[int]) -> int:
        if not progress_turns:
            return turn_count
        return max(0, turn_count - max(progress_turns))

    def _repeated_blocked_actions(self, actions: list[ParsedActionResult]) -> list[str]:
        streaks: list[str] = []
        current: dict[str, tuple[str, int]] = {}
        for action in sorted(actions, key=lambda item: (item.turn, item.agent)):
            if action.action == "move" and action.success is False:
                previous_summary, count = current.get(action.agent, ("", 0))
                normalized = action.summary
                if previous_summary == normalized:
                    count += 1
                else:
                    count = 1
                current[action.agent] = (normalized, count)
                if count == 3:
                    streaks.append(f"{action.agent} repeated the same blocked move 3 times by turn {action.turn}.")
            else:
                current.pop(action.agent, None)
        return streaks

    def _redundant_messages(self, messages: list[ParsedMessage]) -> list[str]:
        repeats: list[str] = []
        current: dict[tuple[str, str], tuple[str, int]] = {}
        for message in sorted(messages, key=lambda item: (item.turn, item.sender)):
            key = (message.sender, message.recipient)
            normalized = normalize_message_content(message.content)
            previous_content, count = current.get(key, ("", 0))
            if previous_content == normalized:
                count += 1
            else:
                count = 1
            current[key] = (normalized, count)
            if count == 3:
                repeats.append(
                    f"{message.sender} repeated the same message to {message.recipient} at least 3 times."
                )
        return repeats

    def _message_without_behavior_change(
        self,
        messages: list[ParsedMessage],
        decisions: list[ParsedDecision],
    ) -> list[str]:
        findings: list[str] = []
        decisions_by_agent: dict[str, list[ParsedDecision]] = defaultdict(list)
        for decision in decisions:
            decisions_by_agent[decision.agent].append(decision)

        for agent_decisions in decisions_by_agent.values():
            agent_decisions.sort(key=lambda item: item.turn)

        for message in sorted(messages, key=lambda item: item.turn):
            recipient_decisions = decisions_by_agent.get(message.recipient, [])
            previous = next(
                (decision for decision in reversed(recipient_decisions) if decision.turn < message.turn),
                None,
            )
            next_decision = next(
                (decision for decision in recipient_decisions if decision.turn >= message.turn),
                None,
            )
            if previous is None or next_decision is None:
                continue
            if previous.action == next_decision.action and previous.action_input == next_decision.action_input:
                findings.append(
                    f"Turn {message.turn}: {message.recipient} received '{message.content}' from {message.sender} "
                    f"but repeated {next_decision.action} without a behavior change."
                )
        return findings

    def _derive_failure_labels(self, run: ParsedRunLog, metrics: RunMetrics) -> list[str]:
        labels: list[str] = []
        status = (run.final_status or "").upper()

        if status == "FAILED" and (run.turns_executed or 0) >= (run.max_turns or 0):
            labels.append("timeout")
        if any(count >= 5 for count in metrics.failed_moves_per_agent.values()):
            labels.append("stuck_navigation_loop")
        if metrics.message_without_behavior_change_cases:
            labels.append("message_without_behavior_change")
        if metrics.redundant_messages:
            labels.append("redundant_messaging")
        if metrics.long_no_progress_stretches:
            labels.append("missing_replan")
        if metrics.key_acquired and metrics.door_discovered and not metrics.door_unlocked:
            labels.append("goal_execution_failure")
        if sum(metrics.messages_sent_per_agent.values()) > 0 and status == "FAILED":
            labels.append("coordination_breakdown")

        ordered: list[str] = []
        for label in labels:
            if label not in ordered:
                ordered.append(label)
        return ordered

    def _build_agent_summaries(
        self,
        run: ParsedRunLog,
        failed_moves: Counter[str],
        repeated_actions: Counter[str],
        messages_sent: Counter[str],
        messages_delivered: Counter[str],
    ) -> list[AgentReviewSummary]:
        summaries: list[AgentReviewSummary] = []
        final_states = {state.agent: state for state in run.final_agent_states}
        decisions_by_agent = Counter(decision.agent for decision in run.decisions)

        for agent in ["Agent A", "Agent B"]:
            final_state = final_states.get(agent)
            position = (
                f"({final_state.position.x}, {final_state.position.y})"
                if final_state is not None
                else "unknown"
            )
            issues: list[str] = []
            if failed_moves.get(agent, 0) >= 4:
                issues.append("frequent failed movement")
            if repeated_actions.get(agent, 0) >= 4:
                issues.append("repeated same decision pattern")
            if messages_delivered.get(agent, 0) > 0 and messages_sent.get(agent, 0) == 0:
                issues.append("mostly reactive to teammate updates")

            summary = (
                f"{agent} ended at {position} with {failed_moves.get(agent, 0)} failed moves, "
                f"{messages_sent.get(agent, 0)} sent messages, and {messages_delivered.get(agent, 0)} received messages."
            )
            summaries.append(
                AgentReviewSummary(
                    agent=agent,
                    final_position=position,
                    inventory=final_state.inventory if final_state is not None else [],
                    decisions=decisions_by_agent.get(agent, 0),
                    failed_moves=failed_moves.get(agent, 0),
                    repeated_actions=repeated_actions.get(agent, 0),
                    messages_sent=messages_sent.get(agent, 0),
                    messages_received=messages_delivered.get(agent, 0),
                    notable_issues=issues,
                    summary=summary,
                )
            )
        return summaries

    def _build_timeline(self, run: ParsedRunLog) -> list[TimelineRow]:
        rows: list[TimelineRow] = []
        discovered: set[str] = set()

        for observation in sorted(run.observations, key=lambda item: (item.turn, item.agent)):
            tags: list[str] = []
            signal = "neutral"
            for obj in observation.visible_objects:
                if obj in {"key", "door", "exit"} and obj not in discovered:
                    discovered.add(obj)
                    tags.append(f"first_{obj}_seen")
                    signal = "progress"
            rows.append(
                TimelineRow(
                    turn=observation.turn,
                    actor=observation.agent,
                    event_type="observation",
                    short_summary=compact_text(observation.summary),
                    progress_signal=signal,  # type: ignore[arg-type]
                    evidence_tags=tags,
                )
            )

        for decision in run.decisions:
            rows.append(
                TimelineRow(
                    turn=decision.turn,
                    actor=decision.agent,
                    event_type="decision",
                    short_summary=compact_text(
                        f"{decision.action} for {decision.goal} ({decision.reason})"
                    ),
                    progress_signal="neutral",
                    evidence_tags=[decision.action, decision.goal],
                )
            )

        for action in run.action_results:
            signal = "neutral"
            tags = [action.action]
            if action.success is False:
                signal = "regression"
                tags.append("failed_action")
            elif action.action in {"pickup_key", "unlock_door"} and action.success:
                signal = "progress"
                tags.append("objective_progress")
            rows.append(
                TimelineRow(
                    turn=action.turn,
                    actor=action.agent,
                    event_type="action_result",
                    short_summary=compact_text(action.summary),
                    progress_signal=signal,  # type: ignore[arg-type]
                    evidence_tags=tags,
                )
            )

        for message in [*run.queued_messages, *run.delivered_messages]:
            signal = "neutral"
            tags = [message.phase, "message"]
            if any(word in normalize_message_content(message.content) for word in ("door", "key", "exit", "unlock")):
                tags.append("objective_info")
            rows.append(
                TimelineRow(
                    turn=message.turn,
                    actor=message.sender if message.phase == "queued" else "System",
                    event_type=f"message_{message.phase}",
                    short_summary=compact_text(
                        f"{message.sender} -> {message.recipient}: {message.content}"
                    ),
                    progress_signal=signal,
                    evidence_tags=tags,
                )
            )

        for divergence in run.divergences:
            rows.append(
                TimelineRow(
                    turn=divergence.turn,
                    actor=divergence.agent,
                    event_type="divergence",
                    short_summary=compact_text(
                        f"{divergence.kind}: believed '{divergence.belief}', reality '{divergence.reality}'"
                    ),
                    progress_signal="regression",
                    evidence_tags=[divergence.kind, "belief_mismatch"],
                )
            )

        if run.final_status:
            rows.append(
                TimelineRow(
                    turn=run.turns_executed,
                    actor="System",
                    event_type="final_status",
                    short_summary=f"Run ended with status {run.final_status}.",
                    progress_signal="progress" if run.final_status.upper() == "SUCCESS" else "regression",
                    evidence_tags=[run.final_status.lower()],
                )
            )

        return sorted(rows, key=lambda item: ((item.turn or 0), item.actor, item.event_type))

    def _recommendations(self, factors: list[str], metrics: RunMetrics) -> list[str]:
        recommendations: list[str] = []
        if "stuck_navigation_loop" in factors:
            recommendations.append("Add blocked-move memory to avoid repeating known invalid directions.")
        if "message_without_behavior_change" in factors:
            recommendations.append("Track whether delivered messages cause a policy shift and trigger re-planning when they do not.")
        if "coordination_breakdown" in factors:
            recommendations.append("Promote shared objectives into explicit coordination state, such as 'teammate found door' or 'teammate has key'.")
        if "redundant_messaging" in factors:
            recommendations.append("Detect redundant repeated messages and suppress low-value repeats unless state meaningfully changes.")
        if "missing_replan" in factors:
            recommendations.append("Trigger re-planning after N turns without progress or after repeated failed actions.")
        if "goal_execution_failure" in factors:
            recommendations.append("Reprioritize unlock behavior when the key is held and the door has already been discovered.")
        if "timeout" in factors:
            recommendations.append("Shorten low-value exploration loops and escalate to recovery behavior before the max-turn threshold.")
        if not recommendations and metrics.turns_since_last_progress >= 4:
            recommendations.append("Add explicit progress tracking so the policy can detect when it is stuck.")
        return recommendations

    def _executive_summary(
        self,
        run: ParsedRunLog,
        metrics: RunMetrics,
        primary_failure: str | None,
        agent_summaries: list[AgentReviewSummary],
    ) -> str:
        status = run.final_status or "UNKNOWN"
        lead = f"Run {run.run_id or 'unknown_run'} ended with status {status}"
        if primary_failure:
            lead += f" and primary issue {primary_failure}"
        lead += "."

        details: list[str] = []
        if metrics.key_acquired:
            details.append("The key was acquired")
        if metrics.door_discovered:
            details.append("the door was discovered")
        if metrics.door_unlocked:
            details.append("the door was unlocked")
        if metrics.turns_since_last_progress >= 5:
            details.append(f"the final {metrics.turns_since_last_progress} turns showed no meaningful progress")

        worst_agent = max(agent_summaries, key=lambda item: item.failed_moves, default=None)
        if worst_agent is not None and worst_agent.failed_moves > 0:
            details.append(
                f"{worst_agent.agent} accumulated {worst_agent.failed_moves} failed moves"
            )
        if metrics.redundant_messages:
            details.append("messaging became repetitive")

        if not details:
            return lead
        return f"{lead} " + "; ".join(details) + "."

