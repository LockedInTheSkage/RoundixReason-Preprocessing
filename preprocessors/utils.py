def format_choices(choices, question_text):
    if choices:
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        return f"{question_text}\n\nChoices:\n{choices_text}"
    return question_text

