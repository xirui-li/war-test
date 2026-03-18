#!/usr/bin/env python3
"""
Build a clean, merged articles dataset from:
1. raw/news.csv (Feb 1 - Feb 26 background articles)
2. processed/articles.json (Feb 27 - Mar 5 conflict articles)

Outputs: war-test/articles_clean.json

Steps:
- Extract Feb 1-26 articles from raw CSV
- Merge with processed articles
- Deduplicate by title
- Remove irrelevant articles (keyword filter)
- Remove articles where body ≈ title (no extra info)
- Sort chronologically
"""
import csv
import json
import hashlib
import re
from pathlib import Path
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).parent
RAW_NEWS_CSV = PROJECT_ROOT.parent / "war-prediction-LLMs" / "data" / "raw" / "news" / "news.csv"
PROCESSED_ARTICLES = PROJECT_ROOT.parent / "war-prediction-LLMs" / "data" / "processed" / "articles.json"
OUTPUT_PATH = PROJECT_ROOT / "articles_clean.json"

# Source ID -> display name mapping
SOURCE_NAMES = {
    "al_monitor": "Al-Monitor",
    "middle_east_eye": "Middle East Eye",
    "guardian": "The Guardian",
    "reuters": "Reuters",
    "al_jazeera": "Al Jazeera",
    "bloomberg": "Bloomberg",
    "google_news": "Google News",
    "fox_news": "Fox News",
    "ap_news": "AP News",
    "financial_times": "Financial Times",
    "the_national_uae": "The National (UAE)",
    "bbc": "BBC",
}

# Relevance keywords
RELEVANT_KEYWORDS = [
    "iran", "israel", "strike", "attack", "bomb", "missile", "war",
    "conflict", "military", "hezbollah", "hamas", "gaza", "lebanon",
    "tehran", "middle east", "mideast", "gulf", "oil", "nuclear",
    "sanctions", "diplomacy", "ceasefire", "airspace", "navy",
    "epic fury", "hormuz", "idf", "irgc", "pentagon", "hegseth",
    "trump", "khamenei", "natanz", "cyprus", "qatar", "evacuat",
    "refinery", "tanker", "drone", "airstrike", "retaliat",
    "saudi", "riyadh", "diplomat", "shelter", "weapon", "defense",
    "defence", "troops", "soldier", "combat", "siege", "proxy",
    "houthi", "yemen", "syria", "iraq", "kuwait", "bahrain",
    "enrichment", "uranium", "iaea", "un security", "sanction",
    "embargo", "regime", "revolutionary guard", "proxy",
]


def is_relevant(title: str, body: str) -> bool:
    text = (title + " " + body).lower()
    return any(kw in text for kw in RELEVANT_KEYWORDS)


def body_adds_info(title: str, body: str) -> bool:
    """Check if body_text contains meaningful info beyond the title."""
    if not body.strip():
        return False
    # Normalize whitespace
    t = re.sub(r'\s+', ' ', title.strip().lower())
    b = re.sub(r'\s+', ' ', body.strip().lower())
    # If body is very short and similar to title, no extra info
    ratio = SequenceMatcher(None, t, b).ratio()
    if ratio > 0.8 and len(body.strip()) < 300:
        return False
    return True


def make_article_id(title: str, published: str) -> str:
    raw = f"{title}_{published}"
    return "art_" + hashlib.md5(raw.encode()).hexdigest()[:12]


def load_raw_background() -> list[dict]:
    """Load Feb 1 - Feb 26 articles from raw CSV."""
    articles = []
    with open(RAW_NEWS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pub = row.get("published", "")
            date_str = pub[:10]
            # Only Feb 1 - Feb 26
            if not (date_str >= "2026-02-01" and date_str <= "2026-02-26"):
                continue

            title = row.get("title", "").strip()
            body = row.get("summary", "").strip()
            source_id = row.get("source", "")
            source_name = SOURCE_NAMES.get(source_id, source_id)
            url = row.get("url", "")

            if not title:
                continue

            articles.append({
                "article_id": make_article_id(title, pub),
                "source_name": source_name,
                "title": title,
                "body_text": body,
                "published_at": pub,
                "source_url": url,
                "origin": "raw_csv",
            })

    return articles


def load_processed() -> list[dict]:
    """Load existing processed articles."""
    with open(PROCESSED_ARTICLES) as f:
        data = json.load(f)

    articles = []
    for a in data:
        articles.append({
            "article_id": a.get("article_id", ""),
            "source_name": a.get("source_name", ""),
            "title": a.get("title", "").strip(),
            "body_text": a.get("body_text", "").strip(),
            "published_at": a.get("published_at", ""),
            "source_url": a.get("source_url", ""),
            "origin": "processed_json",
        })

    return articles


def main():
    # Load both sources
    background = load_raw_background()
    processed = load_processed()
    print(f"Raw background (Feb 1-26): {len(background)} articles")
    print(f"Processed (Feb 27-Mar 5):  {len(processed)} articles")

    # Merge
    all_articles = background + processed
    print(f"Total before cleaning:     {len(all_articles)}")

    # Deduplicate by normalized title
    seen_titles = set()
    deduped = []
    dup_count = 0
    for a in all_articles:
        norm_title = re.sub(r'\s+', ' ', a["title"].lower().strip())
        if norm_title in seen_titles:
            dup_count += 1
            continue
        seen_titles.add(norm_title)
        deduped.append(a)
    print(f"Duplicates removed:        {dup_count}")

    # Filter irrelevant
    relevant = []
    irrelevant = []
    for a in deduped:
        if is_relevant(a["title"], a["body_text"]):
            relevant.append(a)
        else:
            irrelevant.append(a)
    print(f"Irrelevant removed:        {len(irrelevant)}")
    if irrelevant:
        print("  Sample irrelevant:")
        for a in irrelevant[:5]:
            print(f"    [{a['published_at'][:10]}] {a['title'][:70]}")

    # Mark whether body adds info (keep all, but add flag)
    body_dup_count = 0
    for a in relevant:
        a["body_has_extra_info"] = body_adds_info(a["title"], a["body_text"])
        if not a["body_has_extra_info"]:
            body_dup_count += 1
    print(f"Body ≈ title (flagged):    {body_dup_count}")

    # Sort chronologically
    relevant.sort(key=lambda a: a["published_at"])

    # Remove origin field (internal tracking)
    for a in relevant:
        del a["origin"]

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(relevant, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Final article count: {len(relevant)}")

    # Date distribution
    from collections import Counter
    dates = Counter(a["published_at"][:10] for a in relevant)
    print("\nDate distribution:")
    for d in sorted(dates):
        print(f"  {d}: {dates[d]}")


if __name__ == "__main__":
    main()
