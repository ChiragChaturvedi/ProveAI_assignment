# Dungeon Simulation

A small Python prototype of a multi-agent dungeon simulation orchestrated with LangGraph.

## What It Does

Two cooperative agents explore a grid dungeon, coordinate through delayed messages, retrieve a key, unlock a door, and try to reach the exit together. The project emphasizes:

- clear state models
- deterministic world rules
- legible turn-by-turn behavior
- OpenTelemetry spans for node-level tracing
- an agent policy interface that can later be replaced by an LLM
- per-run log files stored in `logs/`

## Project Structure

```text
dungeon_sim/
  __init__.py
  agents.py
  graph.py
  main.py
  prompts.py
  state.py
  tracing.py
  world.py
requirements.txt
README.md
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ProveAI_assignment.dungeon_sim.main
```

You can also run:

```bash
python ProveAI_assignment/dungeon_sim/main.py
```

## LangGraph Loop

The graph executes this loop:

```text
initialize_run
-> deliver_messages
-> agent_a_turn
-> apply_agent_a_action
-> detect_agent_a_divergence
-> agent_b_turn
-> apply_agent_b_action
-> detect_agent_b_divergence
-> check_termination
-> repeat until done
```

## Behavior Notes

- messages are delayed by one turn
- movement is blocked by walls and a locked door
- the deterministic agent sends messages for meaningful updates and periodic location refreshes
- divergence records capture stale or incorrect beliefs
- each simulation creates a timestamped `run_id` like `20260415_221530`
- observations, decisions, actions, messages, divergences, and final outcomes are written to `logs/<run_id>.log`

## Extension Points

- replace `DummyDeterministicAgent` with an LLM-backed JSON policy
- expand local memory into richer belief tracking
- add richer map generation and partial observability rules
- export traces to OTLP instead of the console
