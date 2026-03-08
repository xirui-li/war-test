"""Build article context for each time point."""
import json
from datetime import datetime, timezone

from config import ARTICLES_PATH, MAX_ARTICLES, MAX_CONTEXT_CHARS, MAX_BODY_CHARS


def load_articles() -> list[dict]:
    """Load all articles from the processed data."""
    with open(ARTICLES_PATH) as f:
        return json.load(f)


def parse_datetime(iso_str: str) -> datetime:
    """Parse ISO datetime string to timezone-aware datetime."""
    # Handle various formats
    s = iso_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Remove fractional seconds for simpler parsing
    if "." in s and "+" in s:
        base, tz = s.rsplit("+", 1)
        base = base.split(".")[0]
        s = f"{base}+{tz}"
    elif "." in s:
        s = s.split(".")[0] + "+00:00"
    return datetime.fromisoformat(s)


def get_articles_for_cutoff(all_articles: list[dict], event_datetime_str: str) -> list[dict]:
    """
    Return articles published strictly before event_datetime.
    Sorted by recency (newest first), limited to MAX_ARTICLES.
    """
    cutoff = parse_datetime(event_datetime_str)

    filtered = []
    for a in all_articles:
        pub = a.get("published_at")
        if not pub:
            continue
        try:
            pub_dt = parse_datetime(pub)
            if pub_dt < cutoff:
                filtered.append(a)
        except (ValueError, TypeError):
            continue

    # Most recent first
    filtered.sort(key=lambda a: a["published_at"], reverse=True)
    return filtered[:MAX_ARTICLES]


def format_articles_for_prompt(articles: list[dict]) -> str:
    """
    Format articles into a text block for the LLM prompt.
    Respects MAX_CONTEXT_CHARS total budget.
    """
    if not articles:
        return "(No news articles available before this time point.)"

    lines = []
    total_chars = 0

    for i, art in enumerate(articles):
        date_str = art["published_at"][:16]  # YYYY-MM-DDTHH:MM
        title = art.get("title", "Untitled")[:120]
        source = art.get("source_name", "Unknown")
        body = art.get("body_text", "")[:MAX_BODY_CHARS]

        entry = f"[{date_str}] {title} ({source})\n{body}\n"

        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            remaining = len(articles) - i
            lines.append(f"\n[...{remaining} older articles omitted for brevity]")
            break

        lines.append(entry)
        total_chars += len(entry)

    return "\n".join(lines)
