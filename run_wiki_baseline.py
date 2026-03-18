#!/usr/bin/env python3
"""
Run LLM predictions using Wikipedia timeline as context (baseline).

Results are saved to results_wiki/ to keep them separate from news-based results.

Usage:
    python run_wiki_baseline.py                          # All models, all time points
    python run_wiki_baseline.py --models openai/gpt-5.4  # Single model
    python run_wiki_baseline.py --time-points T0 T1      # Subset of time points
    python run_wiki_baseline.py --dry-run                 # Print prompts, no API calls
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
    API_TEMPERATURE,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    CALL_DELAY,
)
from wiki_context_builder import load_wiki_events, get_wiki_events_for_cutoff, format_wiki_for_prompt
from prompt_builder import build_prompt

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results_wiki"

_file_lock = threading.Lock()
_print_lock = threading.Lock()


def thread_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        data = json.load(f)
    sections = data["sections"]
    for i, section in enumerate(sections):
        section["time_point_id"] = f"T{i}"
    return sections


def model_result_path(model: str) -> Path:
    safe_name = model.replace("/", "_")
    return RESULTS_DIR / f"{safe_name}.json"


def load_model_results(model: str) -> list[dict]:
    path = model_result_path(model)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def save_model_results(model: str, results: list[dict]):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = model_result_path(model)
    with _file_lock:
        with open(path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def short_model_name(model: str) -> str:
    return model.split("/")[-1]


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def call_openrouter(client: OpenAI, model: str, prompt: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=API_TEMPERATURE,
                max_tokens=2048,
            )
            content = response.choices[0].message.content
            if not content:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE ** attempt)
                    continue
                return ""
            return content
        except Exception as e:
            error_str = str(e)
            if any(code in error_str for code in ("400", "401", "403")):
                raise
            wait = RETRY_BACKOFF_BASE ** attempt
            thread_print(f"      Attempt {attempt}/{MAX_RETRIES} failed: {error_str[:120]}")
            if attempt < MAX_RETRIES:
                if "429" in error_str:
                    wait = max(wait, 10)
                time.sleep(wait)
            else:
                raise


def run_single_model(client, model, sections, wiki_events):
    tag = short_model_name(model)
    results = load_model_results(model)
    done_keys = {(r["time_point"], r["question_index"]) for r in results}

    tasks = []
    for section in sections:
        for qi, q in enumerate(section["questions"]):
            tp_id = section["time_point_id"]
            if (tp_id, qi) not in done_keys:
                tasks.append((section, qi, q))

    if not tasks:
        return results

    total = len(tasks)
    call_times = []

    for idx, (section, qi, q) in enumerate(tasks):
        tp_id = section["time_point_id"]
        tp_title = section["title"]
        event_dt = section["event_datetime"]

        events = get_wiki_events_for_cutoff(wiki_events, event_dt)
        context_text = format_wiki_for_prompt(events)
        prompt = build_prompt(context_text, q["scenario_question_en"], q.get("type", "specific"))

        eta_str = ""
        if call_times:
            avg_time = sum(call_times) / len(call_times)
            remaining = (total - idx) * avg_time
            eta_str = f" | ETA {format_eta(remaining)}"

        thread_print(
            f"  [{tag}] [{idx+1}/{total}] {tp_id} Q{qi} "
            f"({len(events)} wiki events, ~{len(prompt)//4} tok)"
            f"{eta_str}"
        )

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
            "num_wiki_events_in_context": len(events),
            "model": model,
            "question_index": qi,
            "question_en": q["scenario_question_en"],
            "question_cn": q["original_cn"],
            "ground_truth": q["answer"],
            "prompt": prompt,
            "response": raw_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        save_model_results(model, results)
        time.sleep(CALL_DELAY)

    elapsed = sum(call_times)
    thread_print(f"  [{tag}] Done — {total} calls in {format_eta(elapsed)}")
    return results


def run_model_worker(model, sections, wiki_events, total_qs):
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

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    run_single_model(client, model, sections, wiki_events)
    return model, needed


def save_merged_results():
    all_results = []
    if not RESULTS_DIR.exists():
        return
    for p in sorted(RESULTS_DIR.glob("*.json")):
        if p.name in ("results.json", "evaluation.json"):
            continue
        if "_zh" in p.name:
            continue
        with open(p) as f:
            all_results.extend(json.load(f))

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nMerged {len(all_results)} rows -> {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Wikipedia Baseline - War Prediction LLM Benchmark")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--time-points", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    models = args.models or MODELS
    sections = load_dataset()
    wiki_events = load_wiki_events()

    if args.time_points:
        sections = [s for s in sections if s["time_point_id"] in args.time_points]

    total_qs = sum(len(s["questions"]) for s in sections)

    print("=" * 60)
    print("Wikipedia Baseline - War Prediction LLM Benchmark")
    print("=" * 60)
    print(f"Wiki events: {len(wiki_events)}  |  Models: {len(models)}  |  "
          f"Time points: {len(sections)}  |  Questions: {total_qs}")
    print(f"API calls: {total_qs} x {len(models)} = {total_qs * len(models)}")
    print(f"Workers: {min(args.workers, len(models))} threads")
    print()

    if args.dry_run:
        for section in sections:
            events = get_wiki_events_for_cutoff(wiki_events, section["event_datetime"])
            context_text = format_wiki_for_prompt(events)
            tp_id = section["time_point_id"]
            for qi, q in enumerate(section["questions"]):
                prompt = build_prompt(context_text, q["scenario_question_en"], q.get("type", "specific"))
                print(f"  {tp_id} Q{qi}: {len(events)} wiki events, "
                      f"~{len(prompt)//4} tokens")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    num_workers = min(args.workers, len(models))
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(run_model_worker, model, sections, wiki_events, total_qs): model
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
    save_merged_results()


if __name__ == "__main__":
    main()
