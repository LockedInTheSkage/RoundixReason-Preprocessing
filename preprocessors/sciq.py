"""
SciQ preprocessor.
"""

import hashlib
import random
from typing import Dict, Any
from .utils import format_choices

def preprocess_sciq(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess SciQ examples.

    Args:
        example: Raw example from SciQ dataset

    Returns:
        Processed example with prompt, question, answer (no split)
    """
    question_text = example.get('question', '')
    correct_answer = example.get('correct_answer', '')

    choices = [
        correct_answer,
        example.get('distractor1', ''),
        example.get('distractor2', ''),
        example.get('distractor3', '')
    ]

    choices = [c for c in choices if c]

    # Deterministic shuffle based on question hash
    seed = int(hashlib.md5(question_text.encode()).hexdigest(), 16) % 10000
    rng = random.Random(seed)
    rng.shuffle(choices)

    try:
        answer_idx = choices.index(correct_answer)
        answer_letter = chr(65 + answer_idx)
    except ValueError:
        answer_letter = "A"

    full_question = format_choices(choices, question_text)

    return {
        'question': full_question,
        'answer': correct_answer,
        'domain': 'science',
    }
