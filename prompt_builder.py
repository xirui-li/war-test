"""Construct system and user prompts for LLM inference."""

SYSTEM_PROMPT = """\
You are a geopolitical intelligence analyst specializing in the 2026 Iran-Israel \
conflict. Given a developing situation and relevant news articles published before \
the event, you predict answers to specific Yes/No questions about what will happen next.

RULES:
1. Answer each question with exactly "Yes" or "No".
2. Provide a brief rationale (2-3 sentences) citing evidence from the articles.
3. If no articles are provided, reason from the situation description alone.
4. Base your analysis ONLY on the provided information.
5. Respond ONLY with valid JSON — no markdown, no commentary, no code fences."""


def build_user_prompt(
    time_point_title: str,
    articles_text: str,
    questions: list[dict],
) -> str:
    """Build the user prompt for a single time point."""
    q_lines = []
    for i, q in enumerate(questions):
        q_lines.append(f"{i}. {q['scenario_question_en']}")
    questions_block = "\n".join(q_lines)

    return f"""\
## Current Situation
{time_point_title}

## News Articles Published Before This Event
{articles_text}

## Questions
For each question below, predict whether the answer is "Yes" or "No".

{questions_block}

## Required Response Format
Return a JSON array with exactly {len(questions)} objects, one per question, in order:
[
  {{"question_index": 0, "answer": "Yes", "rationale": "Brief explanation"}},
  {{"question_index": 1, "answer": "No", "rationale": "Brief explanation"}}
]

Respond with ONLY the JSON array:"""
