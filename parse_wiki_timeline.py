#!/usr/bin/env python3
"""
Fetch and parse the Wikipedia 'Timeline of the 2026 Iran war' into structured events.

Outputs wiki_timeline.json — an array of events with date, datetime, and cleaned text.
"""
import json
import re
import urllib.request
from pathlib import Path

WIKI_API_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=parse&page=Timeline_of_the_2026_Iran_war&prop=wikitext&format=json"
)
OUTPUT_PATH = Path(__file__).parent / "wiki_timeline.json"

# Date sections we care about (before "See also")
DATE_PATTERN = re.compile(r"^==\s*(\d{1,2}\s+\w+)\s*==$", re.MULTILINE)

# Timestamps like "06:35 UTC", "~06:45 UTC", "07:15 UTC:"
TIME_PATTERN = re.compile(r"~?(\d{1,2}):(\d{2})\s*(?:UTC|utc)")

# Month name to number
MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def fetch_wikitext() -> str:
    """Fetch raw wikitext from Wikipedia API."""
    req = urllib.request.Request(WIKI_API_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    return data["parse"]["wikitext"]["*"]


def clean_wikitext(text: str) -> str:
    """Remove wiki markup, references, templates, etc."""
    # Remove <ref>...</ref> and <ref ... />
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/\s*>", "", text)
    # Remove {{templates}}
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Remove [[File:...]] and [[Image:...]]
    text = re.sub(r"\[\[(File|Image):[^\]]*\]\]", "", text)
    # Remove <gallery>...</gallery>
    text = re.sub(r"<gallery[^>]*>.*?</gallery>", "", text, flags=re.DOTALL)
    # Remove other HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Convert [[link|display]] -> display, [[link]] -> link
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", text)
    # Clean up whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_date(day_month: str, year: int = 2026) -> str:
    """Parse '28 February' -> '2026-02-28'."""
    parts = day_month.strip().split()
    day = int(parts[0])
    month = MONTH_MAP.get(parts[1], 1)
    return f"{year}-{month:02d}-{day:02d}"


def extract_time(text: str) -> tuple[int, int] | None:
    """Extract HH:MM from event text if a UTC timestamp is present."""
    m = TIME_PATTERN.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def parse_timeline(wikitext: str) -> list[dict]:
    """Parse wikitext into structured timeline events."""
    # Find all date sections
    matches = list(DATE_PATTERN.finditer(wikitext))

    # Find the "See also" or "Notes" section to stop
    stop_idx = len(wikitext)
    for stop_marker in ("==See also==", "== See also ==", "== Notes ==", "==Notes=="):
        idx = wikitext.find(stop_marker)
        if idx != -1:
            stop_idx = min(stop_idx, idx)
            break

    events = []
    for i, match in enumerate(matches):
        date_str = match.group(1)  # e.g., "28 February"
        date_iso = parse_date(date_str)

        # Get section text (until next date section or stop)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else stop_idx
        section_text = wikitext[start:end]

        # Extract bullet points (lines starting with *)
        for line in section_text.split("\n"):
            line = line.strip()
            if not line.startswith("*"):
                continue

            # Remove leading asterisks
            text = re.sub(r"^\*+\s*", "", line)
            text = clean_wikitext(text)
            text = text.strip()

            if not text or len(text) < 10:
                continue

            # Try to extract timestamp
            time_info = extract_time(text)
            if time_info:
                h, m = time_info
                dt = f"{date_iso}T{h:02d}:{m:02d}:00Z"
            else:
                dt = f"{date_iso}T00:00:00Z"

            events.append({
                "date": date_iso,
                "datetime": dt,
                "text": text,
            })

    return events


def main():
    print("Fetching Wikipedia timeline...")
    wikitext = fetch_wikitext()
    print(f"  Raw wikitext: {len(wikitext)} chars")

    print("Parsing events...")
    events = parse_timeline(wikitext)
    print(f"  Extracted {len(events)} events")

    # Show per-date summary
    from collections import Counter
    date_counts = Counter(e["date"] for e in events)
    for date, count in sorted(date_counts.items()):
        print(f"    {date}: {count} events")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
