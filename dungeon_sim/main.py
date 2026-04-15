from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from dotenv import load_dotenv
    from dungeon_sim.agents import build_agent_policy_from_env
    from dungeon_sim.graph import build_graph
    from dungeon_sim.state import GraphState
    from dungeon_sim.telemetry_contract import EventNames, SpanNames
    from dungeon_sim.tracing import add_span_event, configure_tracing, node_span, safe_set_attributes
    from dungeon_sim.world import create_initial_state, format_trace_summary
else:
    from dotenv import load_dotenv
    from .agents import build_agent_policy_from_env
    from .graph import build_graph
    from .state import GraphState
    from .telemetry_contract import EventNames, SpanNames
    from .tracing import add_span_event, configure_tracing, node_span, safe_set_attributes
    from .world import create_initial_state, format_trace_summary

LOGGER_NAME = "dungeon_sim"


def configure_file_logging(run_id: str) -> Path:
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_id}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)
    return log_path


def log_final_result(state: GraphState) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    run = state["run"]
    world = state["world"]
    agents = state["agents"]

    logger.info("Final status: %s", run.status.value.upper())
    logger.info("Turns executed: %s", run.turn)
    logger.info("Door locked: %s", world.door_locked)
    for name, agent in agents.items():
        logger.info(
            "%s final state | position=(%s, %s) | inventory=%s",
            name,
            agent.position.x,
            agent.position.y,
            agent.inventory,
        )
    logger.info("Trace summary: %s", format_trace_summary(state))


def main() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path if env_path.exists() else None)

    initial_state = create_initial_state(seed=7, max_turns=20)
    log_path = configure_file_logging(initial_state["run"].run_id)
    logger = logging.getLogger(LOGGER_NAME)
    policy = build_agent_policy_from_env()
    logger.info(
        "Starting simulation | run_id=%s | seed=%s | max_turns=%s | agent_mode=%s | model=%s",
        initial_state["run"].run_id,
        initial_state["run"].seed,
        initial_state["run"].max_turns,
        os.getenv("DUNGEON_AGENT_MODE", "litellm"),
        os.getenv("DUNGEON_AGENT_MODEL", "gemini/gemini-2.5-flash"),
    )

    configure_tracing(export_console=True)
    app = build_graph(agent_a_policy=policy, agent_b_policy=policy)

    with node_span(
        SpanNames.DUNGEON_RUN,
        run_id=initial_state["run"].run_id,
        seed=initial_state["run"].seed,
        max_turns=initial_state["run"].max_turns,
        agent_mode=os.getenv("DUNGEON_AGENT_MODE", "litellm"),
        model_name=os.getenv("DUNGEON_AGENT_MODEL", "gemini/gemini-2.5-flash"),
    ) as run_span:
        final_state = app.invoke(initial_state)
        termination_reason = final_state["telemetry"].get("last_termination_reason")
        safe_set_attributes(
            run_span,
            final_status=final_state["run"].status.value,
            termination_reason=termination_reason,
        )
        if termination_reason is not None:
            add_span_event(
                run_span,
                EventNames.TERMINATION_REACHED,
                final_status=final_state["run"].status.value,
                termination_reason=termination_reason,
            )

    log_final_result(final_state)
    print(
        f"Simulation {final_state['run'].status.value.upper()} | "
        f"run_id={final_state['run'].run_id} | log={log_path}"
    )


if __name__ == "__main__":
    main()
