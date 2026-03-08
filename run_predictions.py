#!/usr/bin/env python3
"""
Run LLM predictions on test_dataset.json via OpenRouter.

Usage:
    python run_predictions.py                                    # All models, all time points
    python run_predictions.py --models openai/gpt-5.3-chat       # Single model
    python run_predictions.py --time-points T0 T1 T2             # Subset of time points
    python run_predictions.py --dry-run                          # Print prompts, no API calls
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODELS,
    DATASET_PATH,
    RESULTS_DIR,
    API_TEMPERATURE,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    CALL_DELAY,
)
from context_builder import load_articles, get_articles_for_cutoff, format_articles_for_prompt
from prompt_builder import SYSTEM_PROMPT, build_user_prompt
from response_parser import parse_response, normalize_ground_truth


def load_dataset() -> list[dict]:
    """Load test_dataset.json and normalize ground truth answers."""
    with open(DATASET_PATH) as f:
        data = json.load(f)

    sections = data["sections"]
    for i, section in enumerate(sections):
        section["time_point_id"] = f"T{i}"
        for q in section["questions"]:
            q["ground_truth_normalized"] = normalize_ground_truth(q["answer"])

    return sections


def call_openrouter(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call OpenRouter API with retry logic. Returns raw response text."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=API_TEMPERATURE,
                max_tokens=2048,
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e)

            # Don't retry auth errors
            if "401" in error_str or "403" in error_str:
                raise

            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {error_str[:120]}")

            if attempt < MAX_RETRIES:
                if "429" in error_str:
                    wait = max(wait, 10)
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def make_result_row(
    tp_id: str,
    tp_title: str,
    event_datetime: str,
    num_articles: int,
    model: str,
    qi: int,
    question: dict,
    prediction: str | None,
    rationale: str,
    raw_response: str,
) -> dict:
    """Create a single result row."""
    gt = question["ground_truth_normalized"]
    return {
        "time_point": tp_id,
        "time_point_title": tp_title,
        "event_datetime": event_datetime,
        "num_articles_in_context": num_articles,
        "model": model,
        "question_index": qi,
        "question_en": question["scenario_question_en"],
        "question_cn": question["original_cn"],
        "ground_truth": gt,
        "prediction": prediction,
        "rationale": rationale,
        "is_correct": (prediction == gt) if prediction else None,
        "raw_response": raw_response,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_pipeline(
    models: list[str],
    time_point_ids: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Run the full prediction pipeline."""
    sections = load_dataset()
    all_articles = load_articles()

    if time_point_ids:
        sections = [s for s in sections if s["time_point_id"] in time_point_ids]

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    results = []
    total_calls = len(sections) * len(models)
    call_num = 0

    for section in sections:
        tp_id = section["time_point_id"]
        tp_title = section["title"]
        event_dt = section["event_datetime"]
        questions = section["questions"]

        # Build context
        articles = get_articles_for_cutoff(all_articles, event_dt)
        articles_text = format_articles_for_prompt(articles)
        user_prompt = build_user_prompt(tp_title, articles_text, questions)

        for model in models:
            call_num += 1
            print(
                f"[{call_num}/{total_calls}] {tp_id} x {model} "
                f"({len(articles)} articles, {len(questions)} questions)"
            )

            if dry_run:
                print(f"  [DRY RUN] Prompt: {len(user_prompt)} chars, "
                      f"~{len(user_prompt) // 4} tokens")
                continue

            # Call API
            raw_response = ""
            try:
                raw_response = call_openrouter(client, model, SYSTEM_PROMPT, user_prompt)
            except Exception as e:
                print(f"  FAILED: {e}")
                for qi, q in enumerate(questions):
                    results.append(make_result_row(
                        tp_id, tp_title, event_dt, len(articles), model,
                        qi, q, None, "", f"ERROR: {e}",
                    ))
                time.sleep(CALL_DELAY)
                continue

            # Parse response
            parsed = parse_response(raw_response, len(questions))

            if parsed is None:
                print(f"  PARSE FAILED (attempt 1). Retrying...")
                try:
                    raw_response = call_openrouter(client, model, SYSTEM_PROMPT, user_prompt)
                    parsed = parse_response(raw_response, len(questions))
                except Exception:
                    pass

            if parsed is None:
                print(f"  PARSE FAILED. Raw: {raw_response[:200]}")
                for qi, q in enumerate(questions):
                    results.append(make_result_row(
                        tp_id, tp_title, event_dt, len(articles), model,
                        qi, q, None, "", raw_response,
                    ))
            else:
                for qi, q in enumerate(questions):
                    if qi < len(parsed):
                        pred = parsed[qi]
                        results.append(make_result_row(
                            tp_id, tp_title, event_dt, len(articles), model,
                            qi, q, pred["answer"], pred["rationale"], raw_response,
                        ))
                    else:
                        results.append(make_result_row(
                            tp_id, tp_title, event_dt, len(articles), model,
                            qi, q, None, "", raw_response,
                        ))

                # Print quick summary
                correct = sum(
                    1 for qi, q in enumerate(questions)
                    if qi < len(parsed) and parsed[qi]["answer"] == q["ground_truth_normalized"]
                )
                print(f"  -> {correct}/{len(questions)} correct")

            time.sleep(CALL_DELAY)

    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute accuracy by model, by time_point, and overall."""
    from collections import defaultdict

    by_model = defaultdict(lambda: {"correct": 0, "total": 0, "failed": 0})
    by_tp = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        model = r["model"]
        tp = r["time_point"]

        by_model[model]["total"] += 1
        by_tp[tp]["total"] += 1

        if r["is_correct"] is None:
            by_model[model]["failed"] += 1
        elif r["is_correct"]:
            by_model[model]["correct"] += 1
            by_tp[tp]["correct"] += 1

    def accuracy(d):
        answered = d["total"] - d.get("failed", 0)
        return round(d["correct"] / answered, 4) if answered > 0 else None

    overall_correct = sum(1 for r in results if r["is_correct"])
    overall_failed = sum(1 for r in results if r["is_correct"] is None)

    return {
        "by_model": {
            k: {**v, "accuracy": accuracy(v)}
            for k, v in sorted(by_model.items())
        },
        "by_time_point": {
            k: {**v, "accuracy": accuracy(v)}
            for k, v in sorted(by_tp.items())
        },
        "overall": {
            "total": len(results),
            "correct": overall_correct,
            "failed": overall_failed,
            "accuracy": round(overall_correct / (len(results) - overall_failed), 4)
            if (len(results) - overall_failed) > 0
            else None,
        },
    }


def save_results(results: list[dict]):
    """Save results and summary to disk."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # Full results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} results -> {results_path}")

    # Summary
    summary = compute_summary(results)
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary -> {summary_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("ACCURACY BY MODEL")
    print("=" * 60)
    for model, stats in summary["by_model"].items():
        acc = f"{stats['accuracy']:.1%}" if stats["accuracy"] is not None else "N/A"
        print(f"  {model:<45} {acc} ({stats['correct']}/{stats['total'] - stats['failed']})")

    print(f"\n  Overall: {summary['overall']['accuracy']:.1%}" if summary["overall"]["accuracy"] else "")


def main():
    parser = argparse.ArgumentParser(description="War Prediction LLM Benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model IDs to run (default: all)")
    parser.add_argument("--time-points", nargs="+", default=None,
                        help="Time points to evaluate (e.g., T0 T1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without making API calls")
    args = parser.parse_args()

    models = args.models or MODELS

    print("=" * 60)
    print("War Prediction LLM Benchmark")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Time points: {args.time_points or 'all (T0-T10)'}")
    print()

    results = run_pipeline(models, args.time_points, args.dry_run)

    if not args.dry_run and results:
        save_results(results)


if __name__ == "__main__":
    main()
