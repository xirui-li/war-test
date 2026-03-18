#!/bin/bash
# Test GPT-5.3 on a single time point (T3) via OpenRouter
# Usage: bash test_gpt.sh

cd "$(dirname "$0")"

echo "=== Step 1: Preview (updated context limits) ==="
python -c "
import json
from config import DATASET_PATH, MAX_ARTICLES, MAX_CONTEXT_CHARS, MAX_BODY_CHARS
from context_builder import load_articles, get_articles_for_cutoff, format_articles_for_prompt
from prompt_builder import SYSTEM_PROMPT, build_user_prompt

print(f'Config: MAX_ARTICLES={MAX_ARTICLES}, MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS}, MAX_BODY_CHARS={MAX_BODY_CHARS}')
print()

with open(DATASET_PATH) as f:
    data = json.load(f)
all_articles = load_articles()
print(f'Total articles loaded: {len(all_articles)}')
print()

print(f\"{'TP':<5} {'Articles':>8} {'Prompt chars':>12} {'~Tokens':>8}\")
print('-' * 45)
total = 0
for i, section in enumerate(data['sections']):
    arts = get_articles_for_cutoff(all_articles, section['event_datetime'])
    text = format_articles_for_prompt(arts)
    prompt = build_user_prompt(section['title'], text, section['questions'])
    tokens = (len(SYSTEM_PROMPT) + len(prompt)) // 4
    total += tokens
    print(f\"T{i:<4} {len(arts):>8} {len(prompt):>12} {tokens:>8}\")
print('-' * 45)
print(f\"{'Avg per call':<26} {total//11:>8}\")
print(f\"{'All 6 models':<26} {total*6:>8}\")
"

echo ""
echo "=== Step 2: Test GPT-5.3 on T3 ==="
python run_predictions.py --models openai/gpt-5.4 --time-points T3
