#!/usr/bin/env python3
"""
Run LLM predictions on test_dataset.json via OpenRouter.
Each question is asked individually with its own API call.

Usage:
    python run_predictions.py                                    # All models, all time points
    python run_predictions.py --models openai/gpt-5.4       # Single model
    python run_predictions.py --time-points T0 T1 T2             # Subset of time points
    python run_predictions.py --dry-run                          # Print prompts, no API calls
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from prompt_builder import build_prompt


def load_dataset() -> list[dict]:
    """Load test_dataset.json."""
    with open(DATASET_PATH) as f:
        data = json.load(f)

    sections = data["sections"]
    for i, section in enumerate(sections):
        section["time_point_id"] = f"T{i}"

    return sections


def call_openrouter(
    client: OpenAI,
    model: str,
    prompt: str,
) -> str:
    """Call OpenRouter API with retry logic. Returns raw response text."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=API_TEMPERATURE,
                max_tokens=2048,
            )
            choice = response.choices[0]
            content = choice.message.content
            if not content:
                print(f"      Attempt {attempt}/{MAX_RETRIES}: empty response "
                      f"(finish_reason={choice.finish_reason})")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE ** attempt)
                    continue
                return ""
            return content

        except Exception as e:
            error_str = str(e)

            # Don't retry auth or non-recoverable errors
            if any(code in error_str for code in ("400", "401", "403")):
                raise

            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"      Attempt {attempt}/{MAX_RETRIES} failed: {error_str[:120]}")

            if attempt < MAX_RETRIES:
                if "429" in error_str:
                    wait = max(wait, 10)
                print(f"      Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def model_result_path(model: str) -> Path:
    """Per-model result file path, e.g. results/gpt-5.4.json"""
    safe_name = model.replace("/", "_")
    return RESULTS_DIR / f"{safe_name}.json"


def load_existing_results() -> list[dict]:
    """Load all per-model result files and merge."""
    all_results = []
    if not RESULTS_DIR.exists():
        return all_results
    for p in sorted(RESULTS_DIR.glob("*.json")):
        if p.name in ("results.json", "summary.json"):
            continue
        with open(p) as f:
            all_results.extend(json.load(f))
    return all_results


def load_model_results(model: str) -> list[dict]:
    """Load existing results for a model (for resume)."""
    path = model_result_path(model)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# Lock for thread-safe file writes
_file_lock = threading.Lock()

# Lock for thread-safe print
_print_lock = threading.Lock()


def thread_print(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


def save_model_results(model: str, results: list[dict]):
    """Save results for a single model (thread-safe)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = model_result_path(model)
    with _file_lock:
        with open(path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def short_model_name(model: str) -> str:
    """Short display name for a model, e.g. 'openai/gpt-5.4' -> 'gpt-5.4'."""
    return model.split("/")[-1]


def run_single_model(
    client: OpenAI,
    model: str,
    sections: list[dict],
    all_articles: list[dict],
) -> list[dict]:
    """Run all questions for a single model with per-question resume."""
    tag = short_model_name(model)

    # Load existing progress
    results = load_model_results(model)
    done_keys = {(r["time_point"], r["question_index"]) for r in results}

    # Build list of all (section, question) pairs to run
    tasks = []
    for section in sections:
        for qi, q in enumerate(section["questions"]):
            tp_id = section["time_point_id"]
            if (tp_id, qi) not in done_keys:
                tasks.append((section, qi, q))

    if not tasks:
        return results

    total = len(tasks)
    call_times = []  # track per-call durations for ETA
    model_start = time.time()

    for idx, (section, qi, q) in enumerate(tasks):
        tp_id = section["time_point_id"]
        tp_title = section["title"]
        event_dt = section["event_datetime"]

        # Build context (cached per time point via articles)
        articles = get_articles_for_cutoff(all_articles, event_dt)
        articles_text = format_articles_for_prompt(articles)
        prompt = build_prompt(articles_text, q["scenario_question_en"], q.get("type", "specific"))

        # ETA
        eta_str = ""
        if call_times:
            avg_time = sum(call_times) / len(call_times)
            remaining = (total - idx) * avg_time
            eta_str = f" | ETA {format_eta(remaining)}"

        thread_print(
            f"  [{tag}] [{idx+1}/{total}] {tp_id} Q{qi} "
            f"({len(articles)} articles, ~{len(prompt)//4} tok)"
            f"{eta_str}"
        )

        # Call API
        call_start = time.time()
        raw_response = ""
        try:
            raw_response = call_openrouter(client, model, prompt)
            thread_print(f"  [{tag}]   OK ({len(raw_response)} chars)")
        except Exception as e:
            thread_print(f"  [{tag}]   FAILED: {e}")
            raw_response = f"ERROR: {e}"

        call_times.append(time.time() - call_start)

        results.append({
            "time_point": tp_id,
            "time_point_title": tp_title,
            "event_datetime": event_dt,
            "num_articles_in_context": len(articles),
            "model": model,
            "question_index": qi,
            "question_en": q["scenario_question_en"],
            "question_cn": q["original_cn"],
            "ground_truth": q["answer"],
            "prompt": prompt,
            "response": raw_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Save after each question
        save_model_results(model, results)

        time.sleep(CALL_DELAY)

    elapsed = time.time() - model_start
    thread_print(f"  [{tag}] Done — {total} calls in {format_eta(elapsed)}")
    return results


def save_merged_results():
    """Merge all per-model files into results.json."""
    all_results = load_existing_results()
    if not all_results:
        return

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nMerged {len(all_results)} rows -> {results_path}")


def run_model_worker(model, sections, all_articles, total_qs):
    """Worker function for threaded model execution."""
    tag = short_model_name(model)

    existing = load_model_results(model)
    done_keys = {(r["time_point"], r["question_index"]) for r in existing}
    needed = sum(
        1 for s in sections
        for qi in range(len(s["questions"]))
        if (s["time_point_id"], qi) not in done_keys
    )

    if needed == 0:
        thread_print(f"[{tag}] all {total_qs} done, skipping")
        return model, 0

    done_count = total_qs - needed
    thread_print(
        f"[{tag}] starting" +
        (f" (resuming, {done_count}/{total_qs} done)" if done_count else f" ({needed} calls)")
    )

    # Each thread gets its own client
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    run_single_model(client, model, sections, all_articles)
    return model, needed


def main():
    parser = argparse.ArgumentParser(description="War Prediction LLM Benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model IDs to run (default: all)")
    parser.add_argument("--time-points", nargs="+", default=None,
                        help="Time points to evaluate (e.g., T0 T1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without making API calls")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of parallel threads (default: 6, one per model)")
    args = parser.parse_args()

    models = args.models or MODELS

    # Count total questions
    sections = load_dataset()
    all_articles = load_articles()

    if args.time_points:
        sections = [s for s in sections if s["time_point_id"] in args.time_points]

    total_qs = sum(len(s["questions"]) for s in sections)

    print("=" * 60)
    print("War Prediction LLM Benchmark")
    print("=" * 60)
    print(f"Models: {len(models)}  |  Time points: {len(sections)}  |  "
          f"Questions: {total_qs}  |  API calls: {total_qs} x {len(models)} = {total_qs * len(models)}")
    print(f"Workers: {min(args.workers, len(models))} threads")
    print()

    if args.dry_run:
        for section in sections:
            articles = get_articles_for_cutoff(all_articles, section["event_datetime"])
            articles_text = format_articles_for_prompt(articles)
            tp_id = section["time_point_id"]
            for qi, q in enumerate(section["questions"]):
                prompt = build_prompt(articles_text, q["scenario_question_en"], q.get("type", "specific"))
                print(f"  {tp_id} Q{qi}: {len(articles)} articles, "
                      f"~{len(prompt)//4} tokens")
        return

    num_workers = min(args.workers, len(models))
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(run_model_worker, model, sections, all_articles, total_qs): model
            for model in models
        }

        for future in as_completed(futures):
            model = futures[future]
            try:
                _, calls_made = future.result()
            except Exception as e:
                print(f"\n[{short_model_name(model)}] FAILED with exception: {e}")

    elapsed = time.time() - start_time
    print(f"\nAll models done in {format_eta(elapsed)}")

    # Merge all into results.json
    save_merged_results()


if __name__ == "__main__":
    main()
