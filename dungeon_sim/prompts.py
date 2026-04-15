"""Prompt placeholders for a future LLM-backed agent policy."""

LLM_SYSTEM_PROMPT = """
You are one agent in a cooperative dungeon simulation.
Return strict JSON with:
- action
- action_input
- reason
- goal
- confidence

Allowed actions:
- move
- inspect
- pickup_key
- unlock_door
- send_message
- wait
""".strip()


def build_agent_prompt(agent_name: str, visible_summary: str, memory_summary: str) -> str:
    return (
        f"Agent: {agent_name}\n"
        f"Visible world: {visible_summary}\n"
        f"Local memory: {memory_summary}\n"
        "Choose the next action as JSON only."
    )

