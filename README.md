# Dungeon Simulation

A small Python prototype of a multi-agent dungeon simulation orchestrated with LangGraph.

## What It Does

Two cooperative agents explore a grid dungeon, coordinate through delayed messages, retrieve a key, unlock a door, and try to reach the exit together. The project emphasizes:

- clear state models
- deterministic world rules
- legible turn-by-turn behavior
- OpenTelemetry spans for node-level tracing
- an agent policy interface that can later be replaced by an LLM
- LiteLLM-based provider swapping through `.env`
- per-run log files stored in `logs/`
- a Streamlit-based legibility layer for parsing and reviewing run logs

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
legibility_layer/
  analyzer.py
  app.py
  parsed_models.py
  parser.py
  review_models.py
  utils.py
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

Create a `.env` file in `ProveAI_assignment/` before running the Gemini-backed agent:

```bash
cp .env.example .env
```

You can also run:

```bash
python ProveAI_assignment/dungeon_sim/main.py
```

To review a run log in the legibility UI:

```bash
streamlit run ProveAI_assignment/legibility_layer/app.py
```

To export a structured JSON view of a log:

```bash
python ProveAI_assignment/legibility_layer/export_json.py ProveAI_assignment/logs/<run_id>.log
```

To export the analyzer's review model instead of the raw parsed log:

```bash
python ProveAI_assignment/legibility_layer/export_json.py ProveAI_assignment/logs/<run_id>.log --mode review
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
- the default `.env.example` is configured for LiteLLM with Google Gemini Flash
- the deterministic agent is still available as a fallback or alternate mode
- divergence records capture stale or incorrect beliefs
- each simulation creates a timestamped `run_id` like `20260415_221530`
- observations, decisions, actions, messages, divergences, and final outcomes are written to `logs/<run_id>.log`

## Legibility Layer

The project includes a lightweight review pipeline under `legibility_layer/`:

`.log -> parser -> analyzer -> review model -> UI`

It is meant for run review and debugging, not gameplay. The parser extracts structured records from the simulation logs, the analyzer applies deterministic heuristics to diagnose failure modes, and the Streamlit UI renders the resulting review.

Current legibility layer components:

- `parser.py`: converts raw log lines into typed parsed records
- `analyzer.py`: computes failure labels, metrics, key moments, and recommendations
- `review_models.py`: defines the UI-facing `RunReview` schema
- `app.py`: renders the review in Streamlit
- `export_json.py`: exports parsed runs or reviews as neat JSON

## `.env` Example

```dotenv
DUNGEON_AGENT_MODE=litellm
DUNGEON_AGENT_MODEL=gemini/gemini-2.5-flash
DUNGEON_AGENT_TEMPERATURE=0.2
DUNGEON_AGENT_MAX_TOKENS=512
DUNGEON_AGENT_FALLBACK_TO_DETERMINISTIC=true
GOOGLE_API_KEY=your_google_ai_studio_key_here
```

Switching providers later is just a matter of changing `DUNGEON_AGENT_MODEL` and the corresponding provider API key env var in `.env`.

## Extension Points

- replace `DummyDeterministicAgent` with an LLM-backed JSON policy
- expand local memory into richer belief tracking
- add richer map generation and partial observability rules
- export traces to OTLP instead of the console
- expand the legibility analyzer with richer incident heuristics and cross-run comparisons
