#!/usr/bin/env python3
"""
Evaluate LLM predictions: extract probabilities via GPT-4o-mini and compute MAE.

Usage:
    python evaluate.py                  # Evaluate all models
    python evaluate.py --models gpt-5.4 # Evaluate specific model (partial match)
    python evaluate.py --force           # Re-evaluate all (ignore cache)
"""
import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_OUTPUT = RESULTS_DIR / "evaluation.json"

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
CALL_DELAY = 0.3  # seconds between API calls


# ── Prompt ──────────────────────────────────────────────────────────────────
EXTRACT_PROMPT = """\
You are an evaluation assistant. Given a question and a model's response, extract:
1. The model's predicted answer (Yes or No)
2. The model's predicted probability that the answer is "Yes" (0-100)

If the model doesn't give a clear probability, estimate it from the tone and language of the response.

Question: {question}

Model Response:
{response}

Respond in JSON only: {{"answer": "Yes" or "No", "probability": <number 0-100>}}"""


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_model_files() -> list[Path]:
    """Return paths of per-model result files."""
    return sorted(
        p for p in RESULTS_DIR.glob("*.json")
        if p.name not in ("results.json", "evaluation.json", "rerun_results.json")
        and "_zh" not in p.name
    )


def call_gpt4o_mini(client: OpenAI, prompt: str) -> str:
    """Call GPT-4o-mini with retry."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=128,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"    Retry in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def parse_extraction(raw: str) -> dict | None:
    """Parse GPT-4o-mini's JSON response."""
    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        answer = str(data.get("answer", "")).strip()
        prob = float(data.get("probability", -1))
        if answer in ("Yes", "No") and 0 <= prob <= 100:
            return {"answer": answer, "probability": prob}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions via GPT-4o-mini")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter model files (partial match)")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate all entries (ignore cache)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory (default: results/)")
    args = parser.parse_args()

    global RESULTS_DIR, EVAL_OUTPUT
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
        EVAL_OUTPUT = RESULTS_DIR / "evaluation.json"

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # Load existing evaluations for resume
    existing_evals = {}
    if EVAL_OUTPUT.exists() and not args.force:
        with open(EVAL_OUTPUT) as f:
            for entry in json.load(f):
                key = (entry["model"], entry["time_point"], entry["question_index"])
                existing_evals[key] = entry

    model_files = load_model_files()
    if args.models:
        model_files = [
            p for p in model_files
            if any(m.lower() in p.name.lower() for m in args.models)
        ]

    all_evals = list(existing_evals.values())
    total_calls = 0

    for mf in model_files:
        with open(mf) as f:
            data = json.load(f)

        model_name = data[0]["model"] if data else mf.stem
        short = model_name.split("/")[-1]

        # Filter to specific questions with ground truth
        specific = [e for e in data if e["ground_truth"] in ("Yes", "No")]
        print(f"\n[{short}] {len(specific)} specific questions")

        for entry in specific:
            key = (entry["model"], entry["time_point"], entry["question_index"])
            if key in existing_evals and not args.force:
                continue

            tp = entry["time_point"]
            qi = entry["question_index"]

            prompt = EXTRACT_PROMPT.format(
                question=entry["question_en"],
                response=entry["response"][:6000],  # truncate if very long
            )

            raw = call_gpt4o_mini(client, prompt)
            parsed = parse_extraction(raw)

            if parsed is None:
                print(f"  {tp} Q{qi}: PARSE FAILED: {raw[:100]}")
                continue

            gt = entry["ground_truth"]
            gt_val = 1.0 if gt == "Yes" else 0.0
            prob_val = parsed["probability"] / 100.0
            ae = abs(prob_val - gt_val)

            eval_entry = {
                "model": entry["model"],
                "time_point": tp,
                "question_index": qi,
                "question_en": entry["question_en"],
                "ground_truth": gt,
                "predicted_answer": parsed["answer"],
                "predicted_probability": parsed["probability"],
                "absolute_error": round(ae, 4),
            }
            existing_evals[key] = eval_entry
            all_evals = list(existing_evals.values())

            correct = "✓" if parsed["answer"] == gt else "✗"
            print(f"  {tp} Q{qi}: GT={gt}, pred={parsed['answer']}({parsed['probability']:.0f}%) "
                  f"AE={ae:.2f} {correct}")

            total_calls += 1
            time.sleep(CALL_DELAY)

    # Save all evaluations
    with open(EVAL_OUTPUT, "w") as f:
        json.dump(all_evals, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_evals)} evaluations -> {EVAL_OUTPUT}")
    print(f"API calls made: {total_calls}")

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # Group by model
    from collections import defaultdict
    by_model = defaultdict(list)
    for e in all_evals:
        by_model[e["model"]].append(e)

    print(f"{'Model':<35} {'MAE':>8} {'Acc':>8} {'N':>4}")
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


if __name__ == "__main__":
    main()
