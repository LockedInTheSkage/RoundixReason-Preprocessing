import hashlib
import random

from .utils import format_choices


def preprocess_sciq(example):
    question_text = example.get('question', '')
    correct_answer = example.get('correct_answer', '')

    choices = [
        correct_answer,
        example.get('distractor1', ''),
        example.get('distractor2', ''),
        example.get('distractor3', '')
    ]
    choices = [c for c in choices if c]

    seed = int(hashlib.md5(question_text.encode()).hexdigest(), 16) % 10000
    rng = random.Random(seed)
    rng.shuffle(choices)

    full_question = format_choices(choices, question_text)

    return {
        'question': full_question,
        'answer': correct_answer,
        'domain': 'science',
    }
