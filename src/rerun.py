#!/usr/bin/env python3
"""
Rerun specific predictions, then re-evaluate and re-translate affected entries.

Round 2: re-run missing/failed predictions for specific model+question combos.
"""
import json
import time
from collections import defaultdict
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

# ── What to re-run ──────────────────────────────────────────────────────────

# No modified questions this round
MODIFIED_QUESTIONS = []

# Specific model+question: re-run just that model
RERUN_SPECIFIC = [
    # Round 8
    ("qwen/qwen3.5-35b-a3b", "T2", 9),
]

# ── Helpers ─────────────────────────────────────────────────────────────────

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

def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        data = json.load(f)
    sections = data["sections"]
    for i, s in enumerate(sections):
        s["time_point_id"] = f"T{i}"
    return sections

# ── Build full set of (model, tp, qi) to re-run ────────────────────────────

def build_rerun_set() -> set[tuple[str, str, int]]:
    to_rerun = set()
    for tp, qi in MODIFIED_QUESTIONS:
        for model in MODELS:
            to_rerun.add((model, tp, qi))
    for model, tp, qi in RERUN_SPECIFIC:
        to_rerun.add((model, tp, qi))
    return to_rerun

# ── Main steps ──────────────────────────────────────────────────────────────

def main():
    to_rerun = build_rerun_set()
    print(f"Total entries to re-run: {len(to_rerun)}")

    # Step 1: Delete affected entries from all files
    print("\n=== Step 1: Delete old entries ===")
    for model in MODELS:
        for path in [model_path(model), model_zh_path(model)]:
            data = load_json(path)
            if not data:
                continue
            before = len(data)
            data = [e for e in data if (e["model"], e["time_point"], e["question_index"]) not in to_rerun]
            if len(data) < before:
                save_json(path, data)
                print(f"  {path.name}: deleted {before - len(data)}")

    eval_path = RESULTS_DIR / "evaluation.json"
    eval_data = load_json(eval_path)
    before = len(eval_data)
    eval_data = [e for e in eval_data if (e["model"], e["time_point"], e["question_index"]) not in to_rerun]
    save_json(eval_path, eval_data)
    print(f"  evaluation.json: deleted {before - len(eval_data)}")

    # Step 2: Re-run predictions
    print("\n=== Step 2: Re-run predictions ===")
    sections = load_dataset()
    all_articles = load_articles()
    section_map = {s["time_point_id"]: s for s in sections}

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = sorted(to_rerun, key=lambda x: (x[0], x[1], x[2]))
    for i, (model, tp, qi) in enumerate(tasks):
        section = section_map[tp]
        q = section["questions"][qi]
        short = model.split("/")[-1]

        articles = get_articles_for_cutoff(all_articles, section["event_datetime"])
        articles_text = format_articles_for_prompt(articles)
        prompt = build_prompt(articles_text, q["scenario_question_en"], q.get("type", "specific"))

        print(f"  [{i+1}/{len(tasks)}] {short} {tp} Q{qi}")

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=API_TEMPERATURE,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content or ""
            print(f"    OK ({len(raw)} chars)")
        except Exception as e:
            raw = f"ERROR: {e}"
            print(f"    FAILED: {e}")

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

        data = load_json(model_path(model))
        data.append(entry)
        save_json(model_path(model), data)

        # Also append to rerun_2.json
        rerun2_path = RESULTS_DIR / "rerun_2.json"
        rerun2_data = load_json(rerun2_path)
        rerun2_data.append(entry)
        save_json(rerun2_path, rerun2_data)

        time.sleep(1.5)

    # Step 3: Re-evaluate
    print("\n=== Step 3: Re-evaluate ===")
    client_eval = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    EXTRACT_PROMPT = (
        'You are an evaluation assistant. Given a question and a model\'s response, extract:\n'
        '1. The model\'s predicted answer (Yes or No)\n'
        '2. The model\'s predicted probability that the answer is "Yes" (0-100)\n\n'
        'If the model doesn\'t give a clear probability, estimate it from the tone.\n\n'
        'Question: {question}\n\nModel Response:\n{response}\n\n'
        'Respond in JSON only: {{"answer": "Yes" or "No", "probability": <number 0-100>}}'
    )

    eval_data = load_json(eval_path)
    eval_keys = {(e["model"], e["time_point"], e["question_index"]) for e in eval_data}

    for model in MODELS:
        data = load_json(model_path(model))
        for entry in data:
            if entry["ground_truth"] not in ("Yes", "No"):
                continue
            key = (entry["model"], entry["time_point"], entry["question_index"])
            if key in eval_keys:
                continue
            try:
                resp = client_eval.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": EXTRACT_PROMPT.format(
                        question=entry["question_en"], response=entry["response"][:6000]
                    )}],
                    temperature=0.0, max_tokens=128,
                )
                raw = resp.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                parsed = json.loads(raw)
                gt_val = 1.0 if entry["ground_truth"] == "Yes" else 0.0
                prob = float(parsed["probability"]) / 100.0
                ae = abs(prob - gt_val)
                eval_entry = {
                    "model": entry["model"], "time_point": entry["time_point"],
                    "question_index": entry["question_index"], "question_en": entry["question_en"],
                    "ground_truth": entry["ground_truth"], "predicted_answer": parsed["answer"],
                    "predicted_probability": parsed["probability"], "absolute_error": round(ae, 4),
                }
                eval_data.append(eval_entry)
                eval_keys.add(key)
                save_json(eval_path, eval_data)
                short = entry["model"].split("/")[-1]
                c = "✓" if parsed["answer"] == entry["ground_truth"] else "✗"
                print(f"  {short} {entry['time_point']} Q{entry['question_index']}: "
                      f"pred={parsed['answer']}({parsed['probability']:.0f}%) AE={ae:.2f} {c}")
            except Exception as e:
                print(f"  EVAL FAILED: {e}")
            time.sleep(0.3)

    # Step 4: Re-translate
    print("\n=== Step 4: Re-translate ===")
    TRANSLATE_PROMPT = (
        "Translate the following analysis into Chinese. "
        "Keep the original structure, formatting, and any numbers/percentages intact.\n\n{response}"
    )

    for model in MODELS:
        zh_data = load_json(model_zh_path(model))
        zh_keys = {(e["time_point"], e["question_index"]) for e in zh_data}
        main_data = load_json(model_path(model))
        to_translate = [e for e in main_data if (e["time_point"], e["question_index"]) not in zh_keys]
        if not to_translate:
            continue
        short = model.split("/")[-1]
        print(f"  [{short}] {len(to_translate)} to translate")
        for entry in to_translate:
            out = {**entry}
            out.pop("prompt", None)
            if not entry["response"] or entry["response"].startswith("ERROR:"):
                out["response_zh"] = ""
            else:
                try:
                    resp = client_eval.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(response=entry["response"])}],
                        temperature=0.3, max_tokens=4096,
                    )
                    out["response_zh"] = resp.choices[0].message.content.strip()
                    print(f"    {entry['time_point']} Q{entry['question_index']}: {len(out['response_zh'])} chars")
                except Exception as e:
                    out["response_zh"] = ""
                    print(f"    FAILED: {e}")
            zh_data.append(out)
            save_json(model_zh_path(model), zh_data)
            time.sleep(0.5)

    # Step 5: Merge & summary
    print("\n=== Step 5: Summary ===")
    all_results = []
    for model in MODELS:
        all_results.extend(load_json(model_path(model)))
    save_json(RESULTS_DIR / "results.json", all_results)

    eval_data = load_json(eval_path)
    by_model = defaultdict(list)
    for e in eval_data:
        by_model[e["model"]].append(e)

    print(f"\n{'Model':<35} {'MAE':>8} {'Acc':>8} {'N':>4}")
    print("─" * 60)
    rows = []
    for model, entries in sorted(by_model.items()):
        short = model.split("/")[-1]
        n = len(entries)
        mae = sum(e["absolute_error"] for e in entries) / n if n else 0
        acc = sum(1 for e in entries if e["predicted_answer"] == e["ground_truth"]) / n if n else 0
        rows.append((short, mae, acc, n))
    rows.sort(key=lambda x: x[1])
    for short, mae, acc, n in rows:
        print(f"{short:<35} {mae:>8.4f} {acc:>7.1%} {n:>4}")

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
