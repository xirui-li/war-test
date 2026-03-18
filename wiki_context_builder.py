"""Build context from the parsed Wikipedia timeline for each time point."""
import json
from datetime import datetime, timezone
from pathlib import Path

WIKI_TIMELINE_PATH = Path(__file__).parent / "wiki_timeline.json"


def load_wiki_events() -> list[dict]:
    """Load all parsed Wikipedia timeline events."""
    with open(WIKI_TIMELINE_PATH) as f:
        return json.load(f)


def _parse_dt(iso_str: str) -> datetime:
    s = iso_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def get_wiki_events_for_cutoff(events: list[dict], cutoff_datetime_str: str) -> list[dict]:
    """Return wiki events strictly before the cutoff datetime, sorted chronologically."""
    cutoff = _parse_dt(cutoff_datetime_str)
    filtered = [e for e in events if _parse_dt(e["datetime"]) < cutoff]
    filtered.sort(key=lambda e: e["datetime"])
    return filtered


def format_wiki_for_prompt(events: list[dict]) -> str:
    """Format wiki events into a text block for the LLM prompt."""
    if not events:
        return "(No Wikipedia timeline events available before this time point.)"

    lines = []
    for e in events:
        dt_str = e["datetime"][:16]  # YYYY-MM-DDTHH:MM
        lines.append(f"[{dt_str}] {e['text']}")

    return "\n".join(lines)
