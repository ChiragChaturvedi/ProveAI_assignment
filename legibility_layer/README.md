# Legibility Layer

The legibility layer turns raw dungeon simulation logs into structured run reviews.

Pipeline:

`.log -> parser -> analyzer -> review model -> UI`

## Files

- `app.py`: Streamlit review UI
- `parser.py`: raw log extraction into structured parsed models
- `analyzer.py`: deterministic heuristics for diagnostics and recommendations
- `parsed_models.py`: parser output schemas
- `review_models.py`: UI-facing review schemas
- `utils.py`: parsing and rendering helpers

## What It Extracts

The parser reads one log file and extracts:

- run metadata (`run_id`, `model_name`, `seed`, `max_turns`)
- final outcome (`status`, `turns_executed`, `door_locked`, final agent states)
- per-turn observations
- per-turn decisions
- per-turn action outcomes
- queued and delivered messages
- divergence warnings
- trace summary counts

## What It Diagnoses

The analyzer computes:

- run summary
- key events timeline
- agent summaries
- primary failure label
- contributing factors
- rule-based metrics
- suggested engineering fixes

Current heuristic labels include:

- `stuck_navigation_loop`
- `coordination_breakdown`
- `redundant_messaging`
- `message_without_behavior_change`
- `missing_replan`
- `goal_execution_failure`
- `timeout`

## Run

From the project root:

```bash
pip install -r ProveAI_assignment/requirements.txt
streamlit run ProveAI_assignment/legibility_layer/app.py
```

The app reads available `.log` files from `ProveAI_assignment/logs/`.

## Design Notes

- The parser performs extraction only.
- The analyzer performs diagnosis only.
- The UI renders the review model only.
- The UI does not infer failure causes on its own.
