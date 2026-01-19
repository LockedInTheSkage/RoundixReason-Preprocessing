import re
from typing import Dict, Any


def extract_answer_from_metamath(response: str) -> str:
    if not response:
        return ""

    if "The answer is:" in response:
        parts = response.split("The answer is:")
        if len(parts) > 1:
            answer = parts[-1].strip().rstrip('.')
            return answer

    # Fallback: extract last number from response
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return numbers[-1]

    return ""



def preprocess_metamath(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess MetaMathQA examples.

    Args:
        example: Raw example from MetaMathQA dataset

    Returns:
        Processed example with prompt, question, answer (no split)
    """
    query = example.get('query', '')
    response = example.get('response', '')

    answer = extract_answer_from_metamath(response)

    return {
        'question': query,
        'answer': answer,
        'domain': 'math',
    }
