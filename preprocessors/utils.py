"""
Utility functions for preprocessing datasets (GRPO only).
"""


def format_choices(choices: list, question_text: str) -> str:
    """Format multiple choice questions with labeled options (A, B, C, etc.)."""
    if choices:
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        return f"{question_text}\n\nChoices:\n{choices_text}"
    else:
        return question_text

