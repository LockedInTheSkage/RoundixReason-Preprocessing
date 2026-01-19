from .utils import format_choices


def should_keep_scienceqa(example):
    return example.get('image') is None


def preprocess_scienceqa(example):
    question_text = example.get('question', '')
    choices = example.get('choices', [])
    answer_idx = example.get('answer', 0)

    full_question = format_choices(choices, question_text)

    if choices and 0 <= answer_idx < len(choices):
        answer = choices[answer_idx]
    else:
        answer = ""

    return {
        'question': full_question,
        'answer': answer,
        'domain': 'science',
    }
