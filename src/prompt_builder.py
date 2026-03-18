"""Construct prompts for LLM inference — one prompt per question."""


def build_prompt(articles_text: str, question: str, question_type: str = "specific") -> str:
    """Build a single prompt: context + question.

    For 'specific' questions (Yes/No with ground truth): asks for probability.
    For 'open' questions (analysis): no probability requested.
    """
    base = (
        f"{articles_text}\n"
        f"Based on the above publicly available information, try to analyze the current "
        f"situation and potential future direction, then respond to this question: {question}"
    )
    if question_type == "specific":
        return base + "\nAt the end of your response, also provide the probability."
    return base
