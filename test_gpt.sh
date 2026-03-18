#!/bin/bash
# Test GPT-5.4 on a single time point (T3) via OpenRouter
# Usage: bash test_gpt.sh

cd "$(dirname "$0")"

echo "=== Step 1: Preview (updated context limits) ==="
python src/preview_prompt.py

echo ""
echo "=== Step 2: Test GPT-5.4 on T3 ==="
python src/run_predictions.py --models openai/gpt-5.4 --time-points T3
