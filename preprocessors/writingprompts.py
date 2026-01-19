"""
WritingPrompts dataset preprocessing for GRPO training.
"""

import re


def clean_prompt_tags(text):
    """
    Remove writing prompt tags like [ WP ], [ EU ], [ PI ], etc. from the beginning of prompts.
    
    Args:
        text: The prompt text that may contain tags
        
    Returns:
        Cleaned text without the tag prefix
    """
    if not text:
        return text
    pattern = r'^\[\s*[A-Z]{1,3}\s*\]\s*'
    cleaned = re.sub(pattern, '', text.strip())    
    return cleaned.strip()


def preprocess_writingprompts(example):
    
    prompt = clean_prompt_tags(example['prompt'])
    story = example.get('story', '')
    
    return {
        'question': prompt,
        'answer': story,
        'domain': 'creative_writing',
    }


def should_keep_writingprompts(example):
    prompt = clean_prompt_tags(example.get('prompt', ''))
    story = example.get('story', '')
    
    # Check both exist
    if not prompt or not story:
        return False
    
    # Check prompt length (after cleaning)
    if len(prompt) < 20 or len(prompt) > 500:
        return False
    
    # Check story length
    if len(story) < 200 or len(story) > 5000:
        return False
    
    return True
