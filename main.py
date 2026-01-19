import argparse
import os
import random

from datasets import concatenate_datasets, DatasetDict, load_dataset
from huggingface_hub import delete_repo, HfApi, login

from config import (
    DATASET_SAMPLE_LIMITS,
    GRPO_DATASETS,
    HF_DATASET_NAME,
    HF_USERNAME,
    PROMPT_MAX_LENGTH,
    TOKENIZER_MODEL,
    TRAIN_RATIO,
    VAL_RATIO,
)
from preprocessors import (
    preprocess_leetcode,
    preprocess_metamath,
    preprocess_scienceqa,
    preprocess_sciq,
    preprocess_writingprompts,
    should_keep_scienceqa,
    should_keep_writingprompts,
)
from utils import apply_splits, create_grpo_length_filter, get_tokenizer


def load_and_process_grpo_datasets(tokenizer):
    dataset_configs = {
        "math": (GRPO_DATASETS["math"], preprocess_metamath, None),
        "science_qa": (GRPO_DATASETS["science_qa"], preprocess_scienceqa, should_keep_scienceqa),
        "sciq": (GRPO_DATASETS["sciq"], preprocess_sciq, None),
        "leetcode": (GRPO_DATASETS["leetcode"], preprocess_leetcode, None),
        "writingprompts": (GRPO_DATASETS["writingprompts"], preprocess_writingprompts, should_keep_writingprompts),
    }

    raw_datasets = {}
    for key, (hf_path, _, _) in dataset_configs.items():
        raw_datasets[key] = load_dataset(hf_path, split="train")

    processed_datasets = []
    for key, (_, preprocessor, filter_fn) in dataset_configs.items():
        raw_dataset = raw_datasets[key]

        if filter_fn:
            raw_dataset = raw_dataset.filter(filter_fn, load_from_cache_file=False)

        processed = raw_dataset.map(
            preprocessor,
            remove_columns=raw_dataset.column_names,
            load_from_cache_file=False
        )

        if DATASET_SAMPLE_LIMITS.get(key) is not None and len(processed) > DATASET_SAMPLE_LIMITS[key]:
            indices = random.sample(range(len(processed)), DATASET_SAMPLE_LIMITS[key])
            processed = processed.select(sorted(indices))

        processed_datasets.append(processed)

    grpo_combined = concatenate_datasets(processed_datasets)

    length_filter = create_grpo_length_filter(tokenizer, PROMPT_MAX_LENGTH)
    grpo_filtered = grpo_combined.filter(length_filter, load_from_cache_file=False)

    grpo_dataset = apply_splits(grpo_filtered, question_field='question', train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)

    for split_name in grpo_dataset.keys():
        if 'split' in grpo_dataset[split_name].column_names:
            grpo_dataset[split_name] = grpo_dataset[split_name].remove_columns(['split'])

    return grpo_dataset


def upload_to_hub(grpo_dataset, repo_name, username):
    repo_id = f"{username}/{repo_name}"

    try:
        api = HfApi()
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        delete_repo(repo_id=repo_id, repo_type="dataset", missing_ok=True)
    except Exception:
        pass

    grpo_dict = DatasetDict(grpo_dataset)
    grpo_dict.push_to_hub(repo_id, config_name="grpo")


def main():
    parser = argparse.ArgumentParser(description="Preprocess GRPO datasets and upload to HuggingFace Hub")
    parser.add_argument("--upload", action="store_true", help="Upload processed datasets to HuggingFace Hub")
    parser.add_argument("--username", type=str, default=HF_USERNAME, help="HuggingFace username for upload")
    args = parser.parse_args()

    if args.upload:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HuggingFace token required for upload. Set HF_TOKEN env variable.")
        login(token=token)

    tokenizer = get_tokenizer(TOKENIZER_MODEL)
    grpo_dataset = load_and_process_grpo_datasets(tokenizer)

    if args.upload:
        upload_to_hub(grpo_dataset, HF_DATASET_NAME, args.username)


if __name__ == "__main__":
    main()
