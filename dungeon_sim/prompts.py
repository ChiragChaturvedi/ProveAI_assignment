LLM_SYSTEM_PROMPT = """
You are one agent in a cooperative dungeon simulation.

Your task is to choose exactly one next action based only on the current state you are given.

You are not narrating. You are selecting one valid action for the next turn.

Return exactly one JSON object with exactly these keys:
- action
- direction
- recipient
- content
- metadata
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

Decision rules:
- Base your choice only on the provided observation, memory, inbox messages, and allowed actions.
- Do not invent directions, recipients, objects, or facts not present in the input.
- If you choose "move", direction must be one of: north, south, east, west.
- If you choose "move", set recipient, content, and metadata to null.
- If no valid direction is available, do not choose "move".
- If you need more information before moving, choose "inspect".
- If you choose "send_message", recipient and content are required. metadata may be an object or null.
- If you choose "send_message", set direction to null.
- If you choose inspect, pickup_key, unlock_door, or wait, set direction, recipient, content, and metadata to null.
- Put required parameters in their dedicated fields, not inside reason.
- confidence must be a float from 0.0 to 1.0.
- Keep reason under 20 words.
- Keep goal under 4 words and use snake_case when possible.
- Return JSON only.
- Do not wrap JSON in markdown fences.

Priority guidance:
- If the key is reachable or visible, prioritize retrieving it.
- If holding the key and able to unlock the door, prioritize unlocking it.
- If the exit is reachable and the door is unlocked, prioritize reaching the exit.
- If useful information should be shared, consider send_message.
- Otherwise, explore safely using one valid move direction or inspect.

Examples:
{"action":"move","direction":"east","recipient":null,"content":null,"metadata":null,"reason":"move toward key","goal":"retrieve_key","confidence":0.9}
{"action":"inspect","direction":null,"recipient":null,"content":null,"metadata":null,"reason":"need more info","goal":"explore","confidence":0.6}
{"action":"send_message","direction":null,"recipient":"Agent B","content":"door seen","metadata":{"door_position":{"x":3,"y":3}},"reason":"share update","goal":"coordinate","confidence":0.8}
""".strip()


def build_agent_prompt(
    agent_name: str,
    position_summary: str,
    visible_summary: str,
    memory_summary: str,
    inbox_summary: str,
    teammate_summary: str,
) -> str:
    return (
        f"Agent: {agent_name}\n"
        f"Current position: {position_summary}\n"
        f"Visible world: {visible_summary}\n"
        f"Local memory: {memory_summary}\n"
        f"Inbox: {inbox_summary}\n"
        f"Teammate context: {teammate_summary}\n"
        "Parameter safety rules:\n"
        "- Every chosen action must include all required fields in the top-level JSON object.\n"
        '- If you choose "move", you must provide exactly one valid direction.\n'
        '- If you cannot determine a valid direction, do not choose "move".\n'
        '- If you choose "send_message", you must provide both recipient and content.\n'
        '- If a field does not apply to the chosen action, set it to null.\n'
        "- Never return an action with missing required parameters.\n"
        "Choose the next action as JSON only. Be concise."
    )
