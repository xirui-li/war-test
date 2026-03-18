#!/usr/bin/env python3
"""
Use LLM to extract the probability sentence from each model's response.

Models were prompted: "At the end of your response, also provide the probability."
This script uses GPT-4o-mini to locate and extract that exact probability statement.

Usage:
    python summarize_responses.py                       # All models (English)
    python summarize_responses.py --models claude gpt   # Filter by partial name
    python summarize_responses.py --lang zh             # Process Chinese responses
    python summarize_responses.py --force               # Re-run all (ignore cache)
"""
import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"  # default, overridden by --dir

MODEL = "gpt-4o"
TEMPERATURE = 0.0
CALL_DELAY = 0.3


# ── Prompt ──────────────────────────────────────────────────────────────────
EXTRACT_PROMPT = """\
The following is a model's response to a geopolitical analysis question. The model was asked to provide a probability at the end of its response.

Your task: find and extract the sentence(s) where the model states its probability or likelihood assessment. Copy that sentence EXACTLY as it appears — do not paraphrase or summarize.

Also extract the numeric probability (0-100) if one is given.

Question: {question}

Model response:
{response}

Respond in JSON only:
{{"probability_sentence": "<the exact sentence(s) from the response>", "probability_pct": <number 0-100 or null if not stated>}}"""


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_model_files(lang: str = "en") -> list[Path]:
    skip = {"results.json", "evaluation.json", "rerun_results.json",
            "summary.json", "summary_zh.json"}
    files = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        if p.name in skip:
            continue
        if lang == "zh" and "_zh" not in p.name:
            continue
        if lang == "en" and "_zh" in p.name:
            continue
        files.append(p)
    return files


def call_llm(client: OpenAI, prompt: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=512,
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
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        sentence = str(data.get("probability_sentence", "")).strip()
        pct = data.get("probability_pct")
        if pct is not None:
            pct = float(pct)
            if not (0 <= pct <= 100):
                pct = None
        if sentence:
            return {"probability_sentence": sentence, "probability_pct": pct}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Extract probability from model responses via LLM")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter model files (partial match)")
    parser.add_argument("--lang", choices=["en", "zh"], default="en",
                        help="Which language responses to process (default: en)")
    parser.add_argument("--dir", default=None,
                        help="Results directory (default: results/)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all (ignore cache)")
    args = parser.parse_args()

    global RESULTS_DIR
    if args.dir:
        RESULTS_DIR = PROJECT_ROOT / args.dir

    suffix = f"_{args.lang}" if args.lang != "en" else ""
    output_path = RESULTS_DIR / f"summary{suffix}.json"

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # Load cache
    existing = {}
    if output_path.exists() and not args.force:
        with open(output_path) as f:
            for entry in json.load(f):
                key = (entry["model"], entry["time_point"], entry["question_index"])
                existing[key] = entry

    model_files = load_model_files(args.lang)
    if args.models:
        model_files = [
            p for p in model_files
            if any(m.lower() in p.name.lower() for m in args.models)
        ]

    print(f"Processing {len(model_files)} model files (lang={args.lang})")

    total_calls = 0
    for mf in model_files:
        with open(mf) as f:
            data = json.load(f)
        if not data:
            continue

        model_name = data[0]["model"]
        short = model_name.split("/")[-1]
        print(f"\n[{short}] {len(data)} questions")

        for entry in data:
            key = (entry["model"], entry["time_point"], entry["question_index"])
            if key in existing and not args.force:
                continue

            tp = entry["time_point"]
            qi = entry["question_index"]

            response_text = entry.get("response", "")
            if args.lang == "zh":
                response_text = entry.get("response_zh", response_text)

            prompt = EXTRACT_PROMPT.format(
                question=entry["question_en"],
                response=response_text,
            )

            raw = call_llm(client, prompt)
            parsed = parse_extraction(raw)

            if parsed is None:
                print(f"  {tp} Q{qi}: PARSE FAILED: {raw[:100]}")
                result_entry = {
                    "model": entry["model"],
                    "time_point": tp,
                    "time_point_title": entry.get("time_point_title", ""),
                    "question_index": qi,
                    "question_en": entry["question_en"],
                    "question_cn": entry.get("question_cn", ""),
                    "ground_truth": entry.get("ground_truth", ""),
                    "probability_sentence": "(parse failed)",
                    "probability_pct": None,
                }
            else:
                result_entry = {
                    "model": entry["model"],
                    "time_point": tp,
                    "time_point_title": entry.get("time_point_title", ""),
                    "question_index": qi,
                    "question_en": entry["question_en"],
                    "question_cn": entry.get("question_cn", ""),
                    "ground_truth": entry.get("ground_truth", ""),
                    **parsed,
                }

            existing[key] = result_entry

            pct = result_entry.get("probability_pct")
            gt = entry.get("ground_truth", "")
            gt_str = f"GT={gt}" if gt else "open"
            pct_str = f"{pct:.0f}%" if pct is not None else "N/A"
            sent = result_entry["probability_sentence"][:90]
            print(f"  {tp} Q{qi}: {pct_str:>5}  [{gt_str:<6}]  {sent}")

            total_calls += 1
            time.sleep(CALL_DELAY)

        # Save after each model
        all_results = list(existing.values())
        with open(output_path, "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    all_results = list(existing.values())
    with open(output_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_results)} entries -> {output_path}")
    print(f"API calls made: {total_calls}")

    # ── Comparison table ─────────────────────────────────────────────────────
    from collections import defaultdict

    by_question = defaultdict(dict)
    for e in all_results:
        qkey = (e["time_point"], e["question_index"], e["question_en"])
        by_question[qkey][e["model"]] = e

    print("\n" + "=" * 110)
    print("Comparison Table")
    print("=" * 110)

    for (tp, qi, q_en), models_data in sorted(by_question.items()):
        gt = list(models_data.values())[0].get("ground_truth", "")
        gt_str = f" [GT: {gt}]" if gt else ""
        print(f"\n{tp} Q{qi}: {q_en[:80]}{gt_str}")
        print("-" * 100)
        for model, s in sorted(models_data.items()):
            short = model.split("/")[-1]
            pct = s["probability_pct"]
            pct_str = f"{pct:.0f}%" if pct is not None else "N/A"
            sent = s["probability_sentence"][:70]
            print(f"  {short:<30} {pct_str:>5}  | {sent}")


if __name__ == "__main__":
    main()
