#!/usr/bin/env python3
"""
Fetch full article text for articles where body ≈ title.
Uses trafilatura for extraction + googlenewsdecoder for Google News URLs.

Usage:
    python fetch_fulltext.py                    # Process all body≈title articles
    python fetch_fulltext.py --limit 50         # Process first 50 only (for testing)
    python fetch_fulltext.py --dry-run          # Show what would be fetched
"""
import argparse
import json
import time
import sys
from pathlib import Path

import trafilatura
from googlenewsdecoder import new_decoderv1

ARTICLES_PATH = Path(__file__).parent / "articles_clean.json"
BATCH_SIZE = 25  # Save progress every N articles


def resolve_url(url: str) -> str:
    """Resolve Google News redirect URLs to actual article URLs."""
    if "news.google.com" not in url:
        return url
    try:
        r = new_decoderv1(url, interval=0)
        if r.get("status") and r.get("decoded_url"):
            return r["decoded_url"]
    except Exception:
        pass
    return url


def extract_fulltext(url: str) -> str | None:
    """Extract full article text using trafilatura."""
    try:
        real_url = resolve_url(url)
        html = trafilatura.fetch_url(real_url)
        if html:
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if text and len(text) > 100:
                return text
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Fetch full text for articles")
    parser.add_argument("--limit", type=int, default=None, help="Max articles to process")
    parser.add_argument("--dry-run", action="store_true", help="Show stats only")
    args = parser.parse_args()

    with open(ARTICLES_PATH) as f:
        articles = json.load(f)

    # Find articles that need full text
    todo = [
        (i, a) for i, a in enumerate(articles)
        if not a.get("body_has_extra_info", True)
    ]

    print(f"Total articles: {len(articles)}")
    print(f"Need full text: {len(todo)}")

    if args.limit:
        todo = todo[:args.limit]
        print(f"Processing:     {len(todo)} (limited)")

    if args.dry_run:
        # Show source breakdown of what needs fetching
        from collections import Counter
        sources = Counter(articles[i]["source_name"] for i, _ in todo)
        print("\nBy source:")
        for src, count in sources.most_common():
            print(f"  {src:<25} {count}")
        return

    # Fetch full text
    fetched = 0
    failed = 0
    skipped = 0

    for batch_idx, (i, article) in enumerate(todo):
        url = article.get("source_url", "")
        if not url:
            skipped += 1
            continue

        source = article["source_name"]
        title_short = article["title"][:60]
        print(f"  [{batch_idx+1}/{len(todo)}] {source}: {title_short}...", end=" ", flush=True)

        text = extract_fulltext(url)
        if text:
            articles[i]["body_text"] = text
            articles[i]["body_has_extra_info"] = True
            articles[i]["fulltext_fetched"] = True
            fetched += 1
            print(f"OK ({len(text)} chars)")
        else:
            failed += 1
            print("FAIL")

        # Save progress periodically
        if (batch_idx + 1) % BATCH_SIZE == 0:
            with open(ARTICLES_PATH, "w") as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            print(f"  --- Saved progress: {fetched} fetched, {failed} failed ---")

        time.sleep(0.5)  # Be polite

    # Final save
    with open(ARTICLES_PATH, "w") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\nDone!")
    print(f"  Fetched:  {fetched}")
    print(f"  Failed:   {failed}")
    print(f"  Skipped:  {skipped}")
    print(f"  Total with full text now: {sum(1 for a in articles if a.get('body_has_extra_info', False))}")


if __name__ == "__main__":
    main()
