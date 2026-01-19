"""
ScienceQA preprocessor.
"""

from typing import Dict, Any
from .utils import format_choices



def should_keep_scienceqa(example: Dict[str, Any]) -> bool:
    """
    Check if ScienceQA example should be kept (text-only filter).
    
    Args:
        example: Raw example from ScienceQA dataset
        
    Returns:
        True if example should be kept (no image)
    """
    return example.get('image') is None


def preprocess_scienceqa(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess ScienceQA examples.

    Args:
        example: Raw example from ScienceQA dataset

    Returns:
        Processed example with prompt, question, answer (no split)
    """
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
