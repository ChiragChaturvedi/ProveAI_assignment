from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from legibility_layer.analyzer import RunAnalyzer
    from legibility_layer.parser import DungeonLogParser
    from legibility_layer.review_models import RunReview
    from legibility_layer.utils import badge_html, logs_dir
else:
    from .analyzer import RunAnalyzer
    from .parser import DungeonLogParser
    from .review_models import RunReview
    from .utils import badge_html, logs_dir


def render_overview(review: RunReview, model_name: str | None) -> None:
    st.subheader("Overview")
    status_tone = "success" if review.status.upper() == "SUCCESS" else "danger"
    primary_tone = "warning" if review.primary_failure else "neutral"

    left, middle, right = st.columns([1.2, 1.2, 2.2])
    with left:
        st.markdown(badge_html(review.status.upper(), status_tone), unsafe_allow_html=True)
        st.caption("Final status")
    with middle:
        label = review.primary_failure or "no_primary_failure"
        st.markdown(badge_html(label, primary_tone), unsafe_allow_html=True)
        st.caption("Primary failure")
    with right:
        st.write(f"Run: `{review.run_id}`")
        st.write(f"Turns: `{review.turn_count}`")
        st.write(f"Model: `{model_name or 'unknown'}`")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Key Acquired", "Yes" if review.metrics.key_acquired else "No")
    metric_cols[1].metric("Door Discovered", "Yes" if review.metrics.door_discovered else "No")
    metric_cols[2].metric("Door Unlocked", "Yes" if review.metrics.door_unlocked else "No")
    metric_cols[3].metric("Turns Since Progress", review.metrics.turns_since_last_progress)

    st.write(review.executive_summary)


def render_root_cause(review: RunReview) -> None:
    st.subheader("Root Cause")
    st.write(f"Primary failure: `{review.primary_failure or 'none detected'}`")
    if review.contributing_factors:
        st.write("Contributing factors:")
        for factor in review.contributing_factors:
            st.write(f"- `{factor}`")
    else:
        st.write("No major contributing factors were detected.")

    if review.key_moments:
        st.write("Key moments:")
        for moment in review.key_moments:
            st.write(f"- {moment}")


def render_agent_breakdown(review: RunReview) -> None:
    st.subheader("Agent Breakdown")
    for agent_summary in review.agent_summaries:
        with st.expander(agent_summary.agent, expanded=True):
            cols = st.columns(4)
            cols[0].metric("Decisions", agent_summary.decisions)
            cols[1].metric("Failed Moves", agent_summary.failed_moves)
            cols[2].metric("Messages Sent", agent_summary.messages_sent)
            cols[3].metric("Messages Received", agent_summary.messages_received)
            st.write(agent_summary.summary)
            st.write(f"Final position: `{agent_summary.final_position}`")
            st.write(f"Inventory: `{agent_summary.inventory}`")
            if agent_summary.notable_issues:
                st.write("Notable issues:")
                for issue in agent_summary.notable_issues:
                    st.write(f"- {issue}")


def render_metrics(review: RunReview) -> None:
    st.subheader("Metrics")
    rows = [
        {"metric": "Failed moves per agent", "value": review.metrics.failed_moves_per_agent},
        {"metric": "Repeated same action count", "value": review.metrics.repeated_same_action_count},
        {"metric": "Messages sent per agent", "value": review.metrics.messages_sent_per_agent},
        {"metric": "Messages delivered per agent", "value": review.metrics.messages_delivered_per_agent},
        {"metric": "Turns since last progress", "value": review.metrics.turns_since_last_progress},
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if review.metrics.long_no_progress_stretches:
        st.write("Long no-progress stretches:")
        for stretch in review.metrics.long_no_progress_stretches:
            st.write(f"- {stretch}")

    if review.metrics.repeated_blocked_actions:
        st.write("Repeated blocked actions:")
        for item in review.metrics.repeated_blocked_actions:
            st.write(f"- {item}")

    if review.metrics.redundant_messages:
        st.write("Redundant messages:")
        for item in review.metrics.redundant_messages:
            st.write(f"- {item}")

    if review.metrics.message_without_behavior_change_cases:
        st.write("Messages without behavior change:")
        for item in review.metrics.message_without_behavior_change_cases:
            st.write(f"- {item}")


def render_recommendations(review: RunReview) -> None:
    st.subheader("Recommendations")
    if not review.recommendations:
        st.write("No engineering recommendations were generated.")
        return
    for recommendation in review.recommendations:
        st.write(f"- {recommendation}")


def render_timeline(review: RunReview) -> None:
    st.subheader("Timeline")
    table_rows = [
        {
            "turn": row.turn,
            "actor": row.actor,
            "event_type": row.event_type,
            "progress": row.progress_signal,
            "summary": row.short_summary,
            "tags": ", ".join(row.evidence_tags),
        }
        for row in review.timeline_rows
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Dungeon Run Review", layout="wide")
    st.title("Dungeon Run Review")
    st.caption("Parse raw simulation logs, analyze failure modes, and review operator-facing diagnostics.")

    available_logs = sorted(logs_dir().glob("*.log"), reverse=True)
    if not available_logs:
        st.warning("No log files were found in the project's logs directory.")
        return

    selected_name = st.sidebar.selectbox("Select log file", [path.name for path in available_logs])
    selected_path = next(path for path in available_logs if path.name == selected_name)

    parser = DungeonLogParser()
    analyzer = RunAnalyzer()
    parsed = parser.parse_file(selected_path)
    review = analyzer.analyze(parsed)

    render_overview(review, parsed.model_name)
    st.divider()
    render_timeline(review)
    st.divider()
    render_root_cause(review)
    st.divider()
    render_agent_breakdown(review)
    st.divider()
    render_metrics(review)
    st.divider()
    render_recommendations(review)

    with st.expander("Parser Details"):
        st.write(f"Source file: `{parsed.source_file}`")
        st.write(f"Trace summary: `{parsed.trace_summary.model_dump()}`")
        if parsed.raw_errors:
            st.write("Unstructured or error lines captured:")
            st.code("\n".join(parsed.raw_errors[:20]))


if __name__ == "__main__":
    main()

