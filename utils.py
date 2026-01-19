"""
Utility functions for dataset preprocessing.
"""

import hashlib
from typing import Dict, Any, Optional, Callable
from transformers import AutoTokenizer


def hash_based_split(text: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    """
    Create deterministic train/val/test split based on text hash.

    Args:
        text: Text to hash for split determination
        train_ratio: Ratio for training split
        val_ratio: Ratio for validation split

    Returns:
        Split name: 'train', 'val', or 'test'
    """
    hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
    normalized = (hash_value % 1000) / 1000.0
    if normalized < train_ratio:
        return 'train'
    elif normalized < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'



# ============================================================================
# Tokenizer and Length Filtering
# ============================================================================

def get_tokenizer(model_name: str):
    """
    Load and cache tokenizer.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Tokenizer instance
    """
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, tokenizer) -> int:
    """
    Count tokens in text using provided tokenizer.
    
    Args:
        text: Text to tokenize
        tokenizer: Tokenizer instance
        
    Returns:
        Number of tokens
    """
    if not text or not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def create_grpo_length_filter(
    tokenizer,
    prompt_max_length: int = 512
) -> Callable:
    """
    Create a filter function for GRPO datasets with length limits.

    Filters out examples where:
    - question > prompt_max_length tokens

    Args:
        tokenizer: Tokenizer instance
        prompt_max_length: Maximum tokens for question

    Returns:
        Filter function that returns True if example should be kept
    """
    def filter_fn(example: Dict[str, Any]) -> bool:
        # Use 'question' field (what our preprocessors create)
        question = example.get('question', '')
        question_tokens = count_tokens(question, tokenizer)

        # Keep only if question is within limits and at least 1 token
        return 1 <= question_tokens <= prompt_max_length

    return filter_fn


def apply_splits(dataset, question_field: str = 'question', train_ratio: float = 0.99, val_ratio: float = 0.0):
    """
    Apply train/val/test splits to a dataset based on hash of question field.
    
    Args:
        dataset: Dataset to split
        question_field: Field name to hash for split determination
        train_ratio: Ratio for training split
        val_ratio: Ratio for validation split
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    def add_split(example):
        question = example.get(question_field, '')
        example['split'] = hash_based_split(question, train_ratio, val_ratio)
        return example
    
    # Add split field
    dataset_with_splits = dataset.map(add_split, desc="Adding splits", load_from_cache_file=False)
    
    # Create split dictionary
    splits = {
        'train': dataset_with_splits.filter(lambda x: x['split'] == 'train', load_from_cache_file=False),
        'test': dataset_with_splits.filter(lambda x: x['split'] == 'test', load_from_cache_file=False)
    }
    
    # Only add val if val_ratio > 0
    if val_ratio > 0:
        splits['val'] = dataset_with_splits.filter(lambda x: x['split'] == 'val', load_from_cache_file=False)
    
    return splits
