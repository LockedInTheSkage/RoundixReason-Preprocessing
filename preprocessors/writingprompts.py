import re


def clean_prompt_tags(text):
    if not text:
        return text
    pattern = r'^\[\s*[A-Z]{1,3}\s*\]\s*'
    return re.sub(pattern, '', text.strip()).strip()


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

    if not prompt or not story:
        return False

    if len(prompt) < 20 or len(prompt) > 500:
        return False

    if len(story) < 200 or len(story) > 5000:
        return False

    return True
