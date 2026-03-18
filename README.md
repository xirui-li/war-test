# When AI Navigates the Fog of War

*Can AI reason about and forecast the trajectory of an ongoing war before it transitions into history?*

This is the code repository for the paper **"When AI Navigates the Fog of War"**. We present a temporally grounded benchmark that evaluates whether frontier LLMs can reason about an unfolding geopolitical conflict using only information available at each moment in time.

[[Paper]](https://arxiv.org/abs/2603.16642) [[Website]](https://war-forecast-arena.com) [[Dataset]](https://huggingface.co/datasets/war-forecast-arena/war-forecast-bench)

## Overview

We construct **11 critical temporal nodes** spanning the early stages of the 2026 Middle East conflict (Feb 27 -- Mar 6, 2026), along with **42 node-specific verifiable questions** and **5 general exploratory questions**. At each time point, models receive only news articles published before the event and must reason about what happens next. This design substantially mitigates training-data leakage concerns, as the conflict unfolded after the training cutoff of current frontier models.

## Key Findings

1. Current state-of-the-art LLMs often show **strong strategic reasoning**, attending to underlying incentives, deterrence pressures, and material constraints rather than surface political rhetoric.
2. This capability is **uneven across domains**: models are more reliable in economically and logistically structured settings than in politically ambiguous multi-actor environments.
3. Model narratives **evolve over time**, shifting from early expectations of rapid containment toward more systemic accounts of regional entrenchment and attritional de-escalation.

## Models

| Model | Provider |
|-------|----------|
| `openai/gpt-5.4` | OpenAI |
| `qwen/qwen3.5-35b-a3b` | Qwen |
| `google/gemini-3.1-flash-lite-preview` | Google |
| `anthropic/claude-sonnet-4.6` | Anthropic |
| `moonshotai/kimi-k2.5` | Moonshot |
| `minimax/minimax-m2.5` | MiniMax |

## Setup

```bash
pip install -r requirements.txt
```

**Data** is automatically downloaded from [HuggingFace](https://huggingface.co/datasets/war-forecast-arena/war-forecast-bench) on first run.

**API keys** are loaded from `../war-prediction-LLMs/config.json` (not included in this repo). The config file should contain:

```json
{
  "OPENROUTER_API_KEY": "your-key",
  "OPENAI_API_KEY": "your-key"
}
```

## Usage

```bash
# Audit data quality
python audit_data.py

# Dry run (verify prompts, no API calls)
python run_predictions.py --dry-run

# Run single model
python run_predictions.py --models openai/gpt-5.4 --time-points T3

# Full benchmark (all models, all time points)
python run_predictions.py

# Evaluate predictions
python evaluate.py

# Translate responses to Chinese
python translate.py

# Export to HuggingFace
python export_hf.py --push username/repo-name
```

## File Structure

```
war-test/
├── config.py              # API keys, model list, constants, HF data loading
├── context_builder.py     # Article filtering by cutoff datetime
├── prompt_builder.py      # System + user prompt construction
├── response_parser.py     # LLM JSON response parsing
├── run_predictions.py     # Main inference pipeline (CLI)
├── evaluate.py            # Evaluate predictions via GPT-4o-mini
├── summarize_responses.py # Extract probability statements from responses
├── translate.py           # Translate responses to Chinese
├── rerun.py               # Re-run specific failed predictions
├── run_new_question.py    # Run predictions for newly added questions
├── build_articles.py      # Build articles dataset from raw sources
├── fetch_fulltext.py      # Fetch full text for headline-only articles
├── export_hf.py           # HuggingFace dataset export
├── audit_data.py          # Data quality audit script
├── preview_prompt.py      # Preview prompts and token estimates
├── dataset/README.md      # HuggingFace dataset card
└── requirements.txt
```

## Citation

```bibtex
@misc{li2026ainavigatesfogwar,
      title={When AI Navigates the Fog of War},
      author={Ming Li and Xirui Li and Tianyi Zhou},
      year={2026},
      eprint={2603.16642},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.16642},
}
```
