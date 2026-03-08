#!/usr/bin/env python3
"""
Export results and articles to HuggingFace dataset.

Usage:
    python export_hf.py                              # Save locally only
    python export_hf.py --push username/repo-name    # Push to Hub
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value

from config import RESULTS_DIR, ARTICLES_PATH, DATASET_PATH
from context_builder import parse_datetime

# Schema for predictions split
PREDICTION_FEATURES = Features({
    "time_point": Value("string"),
    "time_point_title": Value("string"),
    "event_datetime": Value("string"),
    "num_articles_in_context": Value("int32"),
    "model": Value("string"),
    "question_index": Value("int32"),
    "question_en": Value("string"),
    "question_cn": Value("string"),
    "ground_truth": Value("string"),
    "prediction": Value("string"),
    "rationale": Value("string"),
    "is_correct": Value("bool"),
})

# Schema for articles split
ARTICLE_FEATURES = Features({
    "article_id": Value("string"),
    "source_name": Value("string"),
    "title": Value("string"),
    "body_text": Value("string"),
    "published_at": Value("string"),
    "source_url": Value("string"),
})


def get_max_cutoff() -> str:
    """Get the latest event_datetime from test_dataset.json."""
    with open(DATASET_PATH) as f:
        data = json.load(f)
    datetimes = [s["event_datetime"] for s in data["sections"]]
    return max(datetimes)


def load_and_filter_articles() -> list[dict]:
    """Load articles up to the last time point's cutoff."""
    max_cutoff = get_max_cutoff()
    cutoff_dt = parse_datetime(max_cutoff)

    with open(ARTICLES_PATH) as f:
        all_articles = json.load(f)

    filtered = []
    for a in all_articles:
        pub = a.get("published_at")
        if not pub:
            continue
        try:
            if parse_datetime(pub) < cutoff_dt:
                filtered.append({
                    "article_id": a.get("article_id", ""),
                    "source_name": a.get("source_name", ""),
                    "title": a.get("title", ""),
                    "body_text": a.get("body_text", ""),
                    "published_at": pub,
                    "source_url": a.get("source_url", ""),
                })
        except (ValueError, TypeError):
            continue

    filtered.sort(key=lambda a: a["published_at"])
    return filtered


def load_predictions() -> list[dict]:
    """Load and clean prediction results for HF dataset."""
    results_path = RESULTS_DIR / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    clean = []
    for r in results:
        clean.append({
            "time_point": r["time_point"],
            "time_point_title": r["time_point_title"],
            "event_datetime": r["event_datetime"],
            "num_articles_in_context": r["num_articles_in_context"],
            "model": r["model"],
            "question_index": r["question_index"],
            "question_en": r["question_en"],
            "question_cn": r["question_cn"],
            "ground_truth": r["ground_truth"],
            "prediction": r.get("prediction") or "N/A",
            "rationale": r.get("rationale") or "",
            "is_correct": bool(r.get("is_correct", False)),
        })

    return clean


def main():
    parser = argparse.ArgumentParser(description="Export to HuggingFace dataset")
    parser.add_argument("--push", type=str, default=None,
                        help="HuggingFace repo ID (e.g., username/dataset-name)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    # Build predictions split
    predictions = load_predictions()
    pred_ds = Dataset.from_list(predictions, features=PREDICTION_FEATURES)
    print(f"Predictions: {len(pred_ds)} rows, columns: {pred_ds.column_names}")

    # Build articles split
    articles = load_and_filter_articles()
    art_ds = Dataset.from_list(articles, features=ARTICLE_FEATURES)
    print(f"Articles: {len(art_ds)} rows, columns: {art_ds.column_names}")

    # Combine into DatasetDict
    ds_dict = DatasetDict({
        "predictions": pred_ds,
        "articles": art_ds,
    })

    # Save locally
    local_path = RESULTS_DIR / "hf_dataset"
    ds_dict.save_to_disk(str(local_path))
    print(f"\nSaved locally: {local_path}")

    # Push to Hub
    if args.push:
        token = args.token
        if not token:
            import os
            token = os.environ.get("HF_TOKEN")
        if not token:
            # Try from config
            try:
                config_path = Path(__file__).parent.parent / "war-prediction-LLMs" / "config.json"
                with open(config_path) as f:
                    token = json.load(f).get("HUGGINGFACE_TOKEN")
            except Exception:
                pass

        ds_dict.push_to_hub(args.push, token=token)
        print(f"Pushed to: https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
