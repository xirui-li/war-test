# War Prediction LLM Benchmark

A research pipeline that evaluates whether Large Language Models can predict geopolitical outcomes during the 2026 Iran-Israel conflict, given real-time news articles as context.

## Overview

The test dataset contains 11 time points (T0–T10, Feb 27 – Mar 6, 2026), each representing a key event in the conflict timeline. At each time point, LLMs receive news articles published before the event and answer Yes/No prediction questions about what happens next. Results are compared against ground truth and exported as a HuggingFace dataset.

## Dataset Structure

- **11 time points** with precise event timestamps
- **40 Yes/No questions** across all time points
- **6 LLM models** evaluated via OpenRouter API
- **~1,685 news articles** as context (from 12+ sources)

## Models

| Model | Provider |
|-------|----------|
| `openai/gpt-5.3-chat` | OpenAI |
| `qwen/qwen3.5-35b-a3b` | Qwen |
| `google/gemini-3.1-flash-lite-preview` | Google |
| `anthropic/claude-sonnet-4.6` | Anthropic |
| `moonshotai/kimi-k2.5` | Moonshot |
| `minimax/minimax-m2.5` | MiniMax |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Audit data quality
python audit_data.py

# Dry run (verify prompts, no API calls)
python run_predictions.py --dry-run

# Run single model test
python run_predictions.py --models openai/gpt-5.3-chat --time-points T3

# Full benchmark (all models, all time points)
python run_predictions.py

# Export to HuggingFace
python export_hf.py --push username/repo-name
```

## File Structure

```
war-test/
├── test_dataset.json      # 11 time points with questions and ground truth
├── config.py              # API keys, model list, constants
├── context_builder.py     # Article filtering by cutoff datetime
├── prompt_builder.py      # System + user prompt construction
├── response_parser.py     # LLM JSON response parsing
├── run_predictions.py     # Main inference pipeline (CLI)
├── export_hf.py           # HuggingFace dataset export
├── audit_data.py          # Data quality audit script
├── requirements.txt
└── results/               # Output directory
    ├── results.json       # Full prediction results
    └── summary.json       # Accuracy by model / time point
```

## Methodology

1. For each time point, articles published **before** the event timestamp are collected as context
2. Articles are sorted by recency, capped at 150 articles / 30K chars
3. All questions for a time point are batched into a single LLM call
4. LLM returns structured JSON with Yes/No answers and rationales
5. Predictions are compared against ground truth for accuracy metrics
