from __future__ import annotations

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


def print_turn_summary(state: GraphState) -> None:
    for event in state["trace"].events:
        print(event.summary)


def print_final_result(state: GraphState) -> None:
    run = state["run"]
    world = state["world"]
    agents = state["agents"]
    print()
    print(f"Final status: {run.status.value.upper()}")
    print(f"Turns executed: {run.turn}")
    print(f"Door locked: {world.door_locked}")
    for name, agent in agents.items():
        print(f"{name}: position=({agent.position.x}, {agent.position.y}), inventory={agent.inventory}")
    print(f"Trace summary: {format_trace_summary(state)}")


def main() -> None:
    configure_tracing()
    initial_state = create_initial_state(seed=7, max_turns=20)
    app = build_graph()

    with node_span("dungeon_run", run_id=initial_state["run"].run_id, seed=initial_state["run"].seed):
        final_state = app.invoke(initial_state)

    print_turn_summary(final_state)
    print_final_result(final_state)


if __name__ == "__main__":
    main()
