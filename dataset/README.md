---
license: cc-by-4.0
task_categories:
  - question-answering
  - text-generation
language:
  - en
  - zh
tags:
  - geopolitics
  - forecasting
  - llm-evaluation
  - temporal-reasoning
pretty_name: "War Forecast Bench"
size_categories:
  - 1K<n<10K
---

# War Forecast Bench

Dataset for the paper **"When AI Navigates the Fog of War"** ([arXiv:2603.16642](https://arxiv.org/abs/2603.16642)).

**Website**: [war-forecast-arena.com](https://war-forecast-arena.com)

## Overview

A temporally grounded benchmark for evaluating LLM reasoning during an ongoing geopolitical conflict. The dataset covers the early stages of the 2026 Middle East conflict, which unfolded after the training cutoff of current frontier models, substantially mitigating training-data leakage concerns.

## Files

### `test_dataset.json`

The benchmark questions and ground truth. Contains:

- **11 critical temporal nodes** (T0--T10, Feb 27 -- Mar 6, 2026)
- **42 node-specific verifiable Yes/No questions**
- **5 general exploratory questions**
- Event timestamps, bilingual questions (EN/CN), and ground truth answers

### `articles_clean.json`

The news article corpus used as context for LLM predictions. Contains:

- **~1,685 news articles** from 12+ sources
- Coverage: Feb 1 -- Mar 5, 2026
- Sources include Reuters, AP News, Al Jazeera, BBC, Bloomberg, The Guardian, and more
- Each article has: title, body text, publication timestamp, source name

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("war-forecast-arena/war-forecast-bench")
```

Or use with the evaluation pipeline:

```bash
git clone https://github.com/XXX/war-test.git
cd war-test
pip install -r requirements.txt
python run_predictions.py
```

The pipeline automatically downloads the data from this HuggingFace repo.

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
