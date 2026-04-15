from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from dungeon_sim.graph import build_graph
    from dungeon_sim.state import GraphState
    from dungeon_sim.tracing import configure_tracing, node_span
    from dungeon_sim.world import create_initial_state, format_trace_summary
else:
    from .graph import build_graph
    from .state import GraphState
    from .tracing import configure_tracing, node_span
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
    initial_state = create_initial_state(seed=7, max_turns=20)
    log_path = configure_file_logging(initial_state["run"].run_id)
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(
        "Starting simulation | run_id=%s | seed=%s | max_turns=%s",
        initial_state["run"].run_id,
        initial_state["run"].seed,
        initial_state["run"].max_turns,
    )

    configure_tracing(export_console=False)
    app = build_graph()

    with node_span("dungeon_run", run_id=initial_state["run"].run_id, seed=initial_state["run"].seed):
        final_state = app.invoke(initial_state)

    log_final_result(final_state)
    print(
        f"Simulation {final_state['run'].status.value.upper()} | "
        f"run_id={final_state['run'].run_id} | log={log_path}"
    )


if __name__ == "__main__":
    main()
