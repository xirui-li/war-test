#!/usr/bin/env python3
"""
Data quality audit for articles.json.
Checks: missing fields, duplicates, short/empty content, relevance, date anomalies.
"""
import json
from collections import Counter
from pathlib import Path

ARTICLES_PATH = Path(__file__).parent.parent / "war-prediction-LLMs" / "data" / "processed" / "articles.json"

with open(ARTICLES_PATH) as f:
    articles = json.load(f)

print(f"Total articles: {len(articles)}\n")

# ============================================================
# 1. Missing / empty fields
# ============================================================
print("=" * 60)
print("1. MISSING / EMPTY FIELDS")
print("=" * 60)
fields = ["title", "body_text", "published_at", "source_name", "article_id", "source_url"]
for field in fields:
    missing = [a for a in articles if not a.get(field, "").strip()]
    print(f"  {field:<15} missing/empty: {len(missing)}")
    if missing and len(missing) <= 5:
        for m in missing:
            print(f"    -> article_id={m.get('article_id','?')}, title={m.get('title','')[:60]}")
print()

# ============================================================
# 2. Duplicates
# ============================================================
print("=" * 60)
print("2. DUPLICATES")
print("=" * 60)

# Exact title duplicates
title_counts = Counter(a.get("title", "") for a in articles)
dup_titles = {t: c for t, c in title_counts.items() if c > 1 and t}
print(f"  Exact title duplicates: {len(dup_titles)} titles appearing 2+ times")
for title, count in sorted(dup_titles.items(), key=lambda x: -x[1])[:15]:
    print(f"    [{count}x] {title[:80]}")

# Duplicate source_url
url_counts = Counter(a.get("source_url", "") for a in articles)
dup_urls = {u: c for u, c in url_counts.items() if c > 1 and u}
print(f"\n  Duplicate source_url: {len(dup_urls)} URLs appearing 2+ times")
for url, count in sorted(dup_urls.items(), key=lambda x: -x[1])[:10]:
    print(f"    [{count}x] {url[:80]}")

# Duplicate article_id
id_counts = Counter(a.get("article_id", "") for a in articles)
dup_ids = {i: c for i, c in id_counts.items() if c > 1 and i}
print(f"\n  Duplicate article_id: {len(dup_ids)}")
print()

# ============================================================
# 3. Content quality
# ============================================================
print("=" * 60)
print("3. CONTENT QUALITY")
print("=" * 60)

# Body text length distribution
lengths = [len(a.get("body_text", "")) for a in articles]
brackets = [(0, 20, "Empty/tiny (<20 chars)"), (20, 50, "Very short (20-50)"),
            (50, 200, "Short (50-200)"), (200, 500, "Medium (200-500)"),
            (500, 1000, "Long (500-1000)"), (1000, float("inf"), "Full text (1000+)")]
print("  Body text length distribution:")
for lo, hi, label in brackets:
    count = sum(1 for l in lengths if lo <= l < hi)
    print(f"    {label:<30} {count:>5}  ({count/len(articles)*100:.1f}%)")

# Show the tiny ones
tiny = [a for a in articles if len(a.get("body_text", "")) < 20]
print(f"\n  Examples of tiny body_text ({len(tiny)} total):")
for a in tiny[:10]:
    print(f"    [{a.get('source_name','')}] \"{a.get('title','')[:60]}\"")
    print(f"      body: \"{a.get('body_text','')}\"")

# Body == title
body_is_title = [a for a in articles if a.get("body_text", "").strip() == a.get("title", "").strip() and a.get("title")]
print(f"\n  Body text == title: {len(body_is_title)}")
print()

# ============================================================
# 4. Relevance check (keyword-based)
# ============================================================
print("=" * 60)
print("4. RELEVANCE CHECK (keyword-based)")
print("=" * 60)

relevant_keywords = [
    "iran", "israel", "strike", "attack", "bomb", "missile", "war",
    "conflict", "military", "hezbollah", "hamas", "gaza", "lebanon",
    "tehran", "middle east", "mideast", "gulf", "oil", "nuclear",
    "sanctions", "diplomacy", "ceasefire", "airspace", "navy",
    "epic fury", "hormuz", "idf", "irgc", "pentagon", "hegseth",
    "trump", "khamenei", "natanz", "cyprus", "qatar", "evacuat",
    "refinery", "tanker", "drone", "airstrike", "retaliat",
]

def is_relevant(article):
    text = (article.get("title", "") + " " + article.get("body_text", "")).lower()
    return any(kw in text for kw in relevant_keywords)

relevant = [a for a in articles if is_relevant(a)]
irrelevant = [a for a in articles if not is_relevant(a)]

print(f"  Relevant (keyword match): {len(relevant)} ({len(relevant)/len(articles)*100:.1f}%)")
print(f"  No keyword match:         {len(irrelevant)} ({len(irrelevant)/len(articles)*100:.1f}%)")

if irrelevant:
    print(f"\n  Sample 'no-match' articles (may still be relevant):")
    for a in irrelevant[:20]:
        print(f"    [{a.get('source_name','')}] {a.get('title','')[:80]}")
print()

# ============================================================
# 5. Date distribution & anomalies
# ============================================================
print("=" * 60)
print("5. DATE DISTRIBUTION")
print("=" * 60)

dates = []
for a in articles:
    pub = a.get("published_at", "")
    if pub:
        dates.append(pub[:10])

date_counts = Counter(dates)
for d in sorted(date_counts):
    print(f"  {d}: {date_counts[d]:>5} articles")

# Outside expected range
expected = {"2026-02-27", "2026-02-28", "2026-03-01", "2026-03-02",
            "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"}
anomalies = [a for a in articles if a.get("published_at", "")[:10] not in expected]
print(f"\n  Articles outside Feb 27 - Mar 6: {len(anomalies)}")
for a in anomalies[:5]:
    print(f"    {a.get('published_at','')} | {a.get('title','')[:60]}")
print()

# ============================================================
# 6. Source distribution
# ============================================================
print("=" * 60)
print("6. SOURCE DISTRIBUTION")
print("=" * 60)
source_counts = Counter(a.get("source_name", "Unknown") for a in articles)
for src, count in source_counts.most_common():
    print(f"  {src:<25} {count:>5}  ({count/len(articles)*100:.1f}%)")
print()

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("SUMMARY — POTENTIAL ISSUES TO ADDRESS")
print("=" * 60)
print(f"  - Exact title duplicates: {len(dup_titles)} titles ({sum(dup_titles.values()) - len(dup_titles)} extra rows)")
print(f"  - Duplicate URLs: {len(dup_urls)}")
print(f"  - Tiny body (<20 chars): {len(tiny)}")
print(f"  - Body == title: {len(body_is_title)}")
print(f"  - No relevance keyword match: {len(irrelevant)}")
print(f"  - Date anomalies: {len(anomalies)}")
