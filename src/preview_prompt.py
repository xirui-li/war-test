#!/usr/bin/env python3
"""Preview actual prompts and estimate token counts for each time point."""
import json

from config import DATASET_PATH
from context_builder import load_articles, get_articles_for_cutoff, format_articles_for_prompt
from prompt_builder import SYSTEM_PROMPT, build_user_prompt

with open(DATASET_PATH) as f:
    data = json.load(f)

all_articles = load_articles()

print("SYSTEM PROMPT")
print("=" * 60)
print(SYSTEM_PROMPT)
print(f"\n  -> {len(SYSTEM_PROMPT)} chars, ~{len(SYSTEM_PROMPT)//4} tokens")
print()

for i, section in enumerate(data["sections"]):
    tp_id = f"T{i}"
    event_dt = section["event_datetime"]
    questions = section["questions"]

    articles = get_articles_for_cutoff(all_articles, event_dt)
    articles_text = format_articles_for_prompt(articles)
    user_prompt = build_user_prompt(section["title"], articles_text, questions)

    print("=" * 60)
    print(f"{tp_id}: {section['title']}")
    print(f"  Cutoff: {event_dt}")
    print(f"  Articles: {len(articles)}")
    print(f"  Prompt length: {len(user_prompt)} chars, ~{len(user_prompt)//4} tokens")
    print(f"  Total (system+user): ~{(len(SYSTEM_PROMPT) + len(user_prompt))//4} tokens")
    print("-" * 60)

    # Show full prompt for one example
    if tp_id == "T3":
        print("\n  [FULL PROMPT PREVIEW FOR T3]")
        print("  " + "-" * 56)
        # Show first 3000 chars + last 500 chars
        if len(user_prompt) > 4000:
            print(user_prompt[:3000])
            print(f"\n  [...{len(user_prompt) - 3500} chars omitted...]\n")
            print(user_prompt[-500:])
        else:
            print(user_prompt)
        print("  " + "-" * 56)

    print()

# Summary table
print("=" * 60)
print("SUMMARY")
print(f"{'TP':<5} {'Articles':>8} {'Prompt chars':>12} {'~Tokens':>8}")
print("-" * 40)
total_tokens = 0
for i, section in enumerate(data["sections"]):
    tp_id = f"T{i}"
    articles = get_articles_for_cutoff(all_articles, section["event_datetime"])
    articles_text = format_articles_for_prompt(articles)
    user_prompt = build_user_prompt(section["title"], articles_text, section["questions"])
    tokens = (len(SYSTEM_PROMPT) + len(user_prompt)) // 4
    total_tokens += tokens
    print(f"{tp_id:<5} {len(articles):>8} {len(user_prompt):>12} {tokens:>8}")

print("-" * 40)
print(f"{'Per model call (avg)':<26} {total_tokens // len(data['sections']):>8}")
print(f"{'Per model total (11 calls)':<26} {total_tokens:>8}")
print(f"{'All 6 models total':<26} {total_tokens * 6:>8}")
