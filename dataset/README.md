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
configs:
  - config_name: questions
    data_files:
      - split: train
        path: questions.parquet
  - config_name: articles
    data_files:
      - split: train
        path: articles.parquet
default_config_name: questions
---

# War Forecast Bench

<p align="center">
  <img src="main.jpg" width="100%" alt="Timeline of critical temporal nodes and AI predictions">
</p>

Dataset for the paper **"When AI Navigates the Fog of War"** ([arXiv:2603.16642](https://arxiv.org/abs/2603.16642)).

**Website**: [war-forecast-arena.com](https://war-forecast-arena.com)

## Overview

A temporally grounded benchmark for evaluating LLM reasoning during an ongoing geopolitical conflict. The dataset covers the early stages of the 2026 Middle East conflict, which unfolded after the training cutoff of current frontier models, substantially mitigating training-data leakage concerns.

## Temporal Nodes

| Node | Date | Event | Theme | Theme Description |
|:----:|------|-------|:-----:|-------------------|
| T0 | Feb 27 | Operation Epic Fury | I | Initial Outbreak |
| T1 | Feb 28 | Israeli-US Strikes | I | Initial Outbreak |
| T2 | Feb 28 | Iranian Strikes | I | Initial Outbreak |
| T3 | Mar 1 | Two Missiles towards British Bases on Cyprus | II | Threshold Crossings |
| T4 | Mar 1 | Oil Refiner and Oil Tanker Was Attacked | III | Economic Shockwaves |
| T5 | Mar 2 | Qatar Halts Energy Production | III | Economic Shockwaves |
| T6 | Mar 2 | Natanz Nuclear Facility Damaged | II | Threshold Crossings |
| T7 | Mar 3 | U.S. Begins Evacuation of Citizens from the Middle East | II | Threshold Crossings |
| T8 | Mar 3 | Nine Countries Involved and Israeli Ground Invasion | II | Threshold Crossings |
| T9 | Mar 3 | Mojtaba Khamenei Becomes Supreme Leader | IV | Political Signaling |
| T10 | Mar 6 | Iranian Apology to Neighboring Countries | IV | Political Signaling |

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

dataset = load_dataset("AIcell/war-test-dataset")
```

Or use with the evaluation pipeline:

```bash
git clone https://github.com/xirui-li/war-test.git
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
