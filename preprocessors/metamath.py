import re


def extract_answer_from_metamath(response):
    if not response:
        return ""

    if "The answer is:" in response:
        parts = response.split("The answer is:")
        if len(parts) > 1:
            return parts[-1].strip().rstrip('.')

    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return numbers[-1]

    return ""


def preprocess_metamath(example):
    query = example.get('query', '')
    response = example.get('response', '')
    answer = extract_answer_from_metamath(response)

    return {
        'question': query,
        'answer': answer,
        'domain': 'math',
    }
