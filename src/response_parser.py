"""Parse and validate LLM responses."""
import json
import re


def parse_response(raw_text: str, num_questions: int) -> list[dict] | None:
    """
    Parse LLM response into list of answer dicts.
    Returns None if parsing fails completely.
    """
    text = raw_text.strip()

    # Strategy 1: Direct JSON parse
    parsed = _try_json_parse(text)
    if parsed and len(parsed) >= num_questions:
        return _validate_answers(parsed, num_questions)

    # Strategy 2: Extract from code fences
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        parsed = _try_json_parse(fence_match.group(1))
        if parsed:
            return _validate_answers(parsed, num_questions)

    # Strategy 3: Find any JSON array in the text
    arr_match = re.search(r"\[.*\]", text, re.DOTALL)
    if arr_match:
        parsed = _try_json_parse(arr_match.group(0))
        if parsed:
            return _validate_answers(parsed, num_questions)

    return None


def _try_json_parse(text: str) -> list[dict] | None:
    """Try to parse text as a JSON array."""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    return None


def _validate_answers(parsed: list[dict], num_questions: int) -> list[dict]:
    """Normalize answers to 'Yes', 'No', or 'N/A'."""
    results = []
    for i, item in enumerate(parsed[:num_questions]):
        answer_raw = str(item.get("answer", "")).strip()

        answer_lower = answer_raw.lower()
        if answer_lower in ("yes", "true", "1"):
            answer = "Yes"
        elif answer_lower in ("no", "false", "0"):
            answer = "No"
        else:
            answer = "N/A"

        results.append({
            "question_index": item.get("question_index", i),
            "answer": answer,
            "rationale": str(item.get("rationale", "")),
        })

    return results


def normalize_ground_truth(answer: str) -> str:
    """
    Normalize ground truth answers to Yes/No.
    Handles T2 Q1: "Sustained military operations" -> "No"
    (the question asks if strikes "remain symbolic")
    """
    ans = answer.strip().lower()
    if ans == "yes":
        return "Yes"
    elif ans == "no":
        return "No"
    elif "sustained" in ans:
        return "No"
    else:
        return answer  # Preserve unexpected values for review
