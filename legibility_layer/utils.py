from __future__ import annotations

import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .parsed_models import Coordinate

LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| "
    r"(?P<level>[A-Z]+) \| (?P<message>.*)$"
)
POSITION_RE = re.compile(r"\((?P<x>-?\d+),\s*(?P<y>-?\d+)\)")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def logs_dir() -> Path:
    return project_root() / "logs"


def parse_log_envelope(raw_line: str) -> tuple[datetime | None, str | None, str]:
    match = LOG_LINE_RE.match(raw_line.rstrip())
    if not match:
        return None, None, raw_line.rstrip()
    timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f")
    return timestamp, match.group("level"), match.group("message")


def safe_literal_eval(value: str, default: Any) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return default


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def parse_position(text: str) -> Coordinate | None:
    match = POSITION_RE.search(text)
    if not match:
        return None
    return Coordinate(x=int(match.group("x")), y=int(match.group("y")))


def normalize_message_content(content: str) -> str:
    return re.sub(r"\s+", " ", content.strip().lower())


def compact_text(text: str, limit: int = 140) -> str:
    clean = re.sub(r"\s+", " ", text.strip())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def badge_html(label: str, tone: str) -> str:
    colors = {
        "success": ("#0b6b2f", "#dff5e8"),
        "warning": ("#8a5a00", "#fff4d6"),
        "danger": ("#8b1e1e", "#fde7e7"),
        "neutral": ("#334155", "#e8eef7"),
    }
    fg, bg = colors.get(tone, colors["neutral"])
    return (
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;"
        f"background:{bg};color:{fg};font-size:0.85rem;font-weight:600'>{label}</span>"
    )

