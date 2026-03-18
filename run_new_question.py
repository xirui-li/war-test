#!/usr/bin/env python3
"""
Run predictions for the newly added open question across all models and time points,
then translate them to Chinese.

The new question is the last question in each time point:
"What is the most probable pathway to de-escalation or resolution of the Iran-US conflict,
and what is a realistic timeline?"
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from config import (
    API_KEY,
    API_BASE_URL,
    DATASET_PATH,
    RESULTS_DIR,
    API_TEMPERATURE,
    MODELS,
)
from context_builder import load_articles, get_articles_for_cutoff, format_articles_for_prompt
from prompt_builder import build_prompt

TRANSLATE_PROMPT = (
    "Translate the following analysis into Chinese. "
    "Keep the original structure, formatting, and any numbers/percentages intact.\n\n{response}"
)


def model_path(model: str) -> Path:
    return RESULTS_DIR / f"{model.replace('/', '_')}.json"

def model_zh_path(model: str) -> Path:
    return RESULTS_DIR / f"{model.replace('/', '_')}_zh.json"

def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)

def save_json(path: Path, data: list):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # Load dataset
    with open(DATASET_PATH) as f:
        data = json.load(f)
    sections = data["sections"]
    for i, s in enumerate(sections):
        s["time_point_id"] = f"T{i}"

    all_articles = load_articles()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Build tasks: last question of each time point × all models
    tasks = []
    for section in sections:
        tp = section["time_point_id"]
        qi = len(section["questions"]) - 1  # last question
        q = section["questions"][qi]
        for model in MODELS:
            tasks.append((model, tp, qi, q, section))

    print(f"Total predictions to run: {len(tasks)} ({len(sections)} time points × {len(MODELS)} models)")

    # Step 1: Run predictions
    print("\n=== Step 1: Run predictions ===")
    new_entries = []  # collect for translation
    for i, (model, tp, qi, q, section) in enumerate(tasks):
        short = model.split("/")[-1]

        # Check if already exists
        existing = load_json(model_path(model))
        if any(e["time_point"] == tp and e["question_index"] == qi for e in existing):
            print(f"  [{i+1}/{len(tasks)}] {short} {tp} Q{qi}: already exists, skipping")
            continue

        articles = get_articles_for_cutoff(all_articles, section["event_datetime"])
        articles_text = format_articles_for_prompt(articles)
        prompt = build_prompt(articles_text, q["scenario_question_en"], q.get("type", "open"))

        print(f"  [{i+1}/{len(tasks)}] {short} {tp} Q{qi} ...", end=" ", flush=True)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=API_TEMPERATURE,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content or ""
            print(f"OK ({len(raw)} chars)")
        except Exception as e:
            raw = f"ERROR: {e}"
            print(f"FAILED: {e}")

        entry = {
            "time_point": tp,
            "time_point_title": section["title"],
            "event_datetime": section["event_datetime"],
            "num_articles_in_context": len(articles),
            "model": model,
            "question_index": qi,
            "question_en": q["scenario_question_en"],
            "question_cn": q["original_cn"],
            "ground_truth": q["answer"],
            "prompt": prompt,
            "response": raw,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        existing.append(entry)
        save_json(model_path(model), existing)
        new_entries.append(entry)
        time.sleep(1.5)

    # Update merged results.json
    all_results = []
    for model in MODELS:
        all_results.extend(load_json(model_path(model)))
    save_json(RESULTS_DIR / "results.json", all_results)
    print(f"\nUpdated results.json: {len(all_results)} total entries")

    # Step 2: Translate new entries
    if not new_entries:
        print("\nNo new entries to translate.")
        return

    print(f"\n=== Step 2: Translate {len(new_entries)} new entries ===")
    for i, entry in enumerate(new_entries):
        model = entry["model"]
        short = model.split("/")[-1]
        zh_data = load_json(model_zh_path(model))

        out = {**entry}
        out.pop("prompt", None)

        if not entry["response"] or entry["response"].startswith("ERROR:"):
            out["response_zh"] = ""
            print(f"  [{i+1}/{len(new_entries)}] {short} {entry['time_point']} Q{entry['question_index']}: skipped")
        else:
            try:
                resp = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(response=entry["response"])}],
                    temperature=0.3, max_tokens=4096,
                )
                out["response_zh"] = resp.choices[0].message.content.strip()
                print(f"  [{i+1}/{len(new_entries)}] {short} {entry['time_point']} Q{entry['question_index']}: {len(out['response_zh'])} chars")
            except Exception as e:
                out["response_zh"] = ""
                print(f"  [{i+1}/{len(new_entries)}] FAILED: {e}")

        zh_data.append(out)
        save_json(model_zh_path(model), zh_data)
        time.sleep(0.5)

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
