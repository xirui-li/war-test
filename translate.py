#!/usr/bin/env python3
"""
Translate all responses in results.json from English to Chinese.
Adds "response_cn" key to each entry. Saves to results_cn.json.

Usage:
    python translate.py                  # Translate all
    python translate.py --workers 10     # Set concurrency
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3

TRANSLATE_PROMPT = """\
Translate the following analysis into Chinese. Keep the original structure, formatting, and any numbers/percentages intact.

{response}"""

_print_lock = threading.Lock()


def thread_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def call_translate(client: OpenAI, text: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(response=text)}],
                temperature=TEMPERATURE,
                max_tokens=4096,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                thread_print(f"    Retry in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def translate_one(client, entry, idx, total):
    short = entry["model"].split("/")[-1]
    tp = entry["time_point"]
    qi = entry["question_index"]

    if not entry.get("response") or entry["response"].startswith("ERROR:"):
        entry["response_cn"] = ""
        thread_print(f"  [{idx}/{total}] {short} {tp} Q{qi}: skipped")
    else:
        translated = call_translate(client, entry["response"])
        entry["response_cn"] = translated
        thread_print(f"  [{idx}/{total}] {short} {tp} Q{qi}: {len(translated)} chars")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--input", default="results/results.json")
    parser.add_argument("--output", default="results/results_cn.json")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {args.input}")

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    total = len(data)
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(translate_one, client, entry, i + 1, total): i
            for i, entry in enumerate(data)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                thread_print(f"  FAILED: {e}")

    with open(args.output, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nDone — {total} translations in {time.time() - start:.0f}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
