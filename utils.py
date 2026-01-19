import hashlib

from transformers import AutoTokenizer


def hash_based_split(text, train_ratio=0.8, val_ratio=0.1):
    hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
    normalized = (hash_value % 1000) / 1000.0
    if normalized < train_ratio:
        return 'train'
    elif normalized < train_ratio + val_ratio:
        return 'val'
    return 'test'


def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text, tokenizer):
    if not text or not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def create_grpo_length_filter(tokenizer, prompt_max_length=512):
    def filter_fn(example):
        question = example.get('question', '')
        question_tokens = count_tokens(question, tokenizer)
        return 1 <= question_tokens <= prompt_max_length

    return filter_fn


def apply_splits(dataset, question_field='question', train_ratio=0.99, val_ratio=0.0):
    def add_split(example):
        question = example.get(question_field, '')
        example['split'] = hash_based_split(question, train_ratio, val_ratio)
        return example

    dataset_with_splits = dataset.map(add_split, desc="Adding splits", load_from_cache_file=False)

    splits = {
        'train': dataset_with_splits.filter(lambda x: x['split'] == 'train', load_from_cache_file=False),
        'test': dataset_with_splits.filter(lambda x: x['split'] == 'test', load_from_cache_file=False)
    }

    if val_ratio > 0:
        splits['val'] = dataset_with_splits.filter(lambda x: x['split'] == 'val', load_from_cache_file=False)

    return splits
