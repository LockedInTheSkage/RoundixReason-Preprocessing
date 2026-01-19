"""
Main script to preprocess datasets for GRPO training and upload to HuggingFace Hub.
"""

import argparse
import os
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login

from config import (
    GRPO_DATASETS,
    DATASET_SAMPLE_LIMITS,
    HF_USERNAME,
    HF_DATASET_NAME,
    TOKENIZER_MODEL,
    PROMPT_MAX_LENGTH,
    TRAIN_RATIO,
    VAL_RATIO,
)
from preprocessors import (
    preprocess_metamath,
    preprocess_scienceqa,
    preprocess_sciq,
    preprocess_leetcode,
    preprocess_writingprompts,
    should_keep_scienceqa,
    should_keep_writingprompts,
)
from utils import (
    get_tokenizer,
    create_grpo_length_filter,
    apply_splits,
    count_tokens,
)


def load_and_process_grpo_datasets(tokenizer):
    """
    Load and preprocess GRPO datasets with streamlined pipeline.

    Pipeline: Load → Preprocess (with domain filters) → Length filter → Split

    Args:
        tokenizer: Tokenizer for length filtering

    Returns:
        Dictionary with train/test splits (and val if VAL_RATIO > 0)
    """
    # Dataset configuration: name -> (hf_path, preprocessor, filter_fn, display_name)
    dataset_configs = {
        "math": (GRPO_DATASETS["math"], preprocess_metamath, None, "MetaMathQA"),
        "science_qa": (GRPO_DATASETS["science_qa"], preprocess_scienceqa, should_keep_scienceqa, "ScienceQA"),
        "sciq": (GRPO_DATASETS["sciq"], preprocess_sciq, None, "SciQ"),
        "leetcode": (GRPO_DATASETS["leetcode"], preprocess_leetcode, None, "LeetCode"),
        "writingprompts": (GRPO_DATASETS["writingprompts"], preprocess_writingprompts, should_keep_writingprompts, "WritingPrompts"),
    }

    print("\n" + "="*80)
    print("LOADING GRPO DATASETS")
    print("="*80 + "\n")

    # Load all datasets
    raw_datasets = {}
    for key, (hf_path, _, _, display_name) in dataset_configs.items():
        print(f"Loading {display_name}...")
        raw_datasets[key] = load_dataset(hf_path, split="train")
        print(f"  Loaded {len(raw_datasets[key]):,} examples")

    print("\n" + "="*80)
    print("PREPROCESSING GRPO DATASETS (with domain filters)")
    print("="*80 + "\n")

    # Preprocess all datasets
    processed_datasets = []
    for key, (_, preprocessor, filter_fn, display_name) in dataset_configs.items():
        raw_dataset = raw_datasets[key]

        if filter_fn:
            print(f"Filtering {display_name}...")
            raw_dataset = raw_dataset.filter(filter_fn, desc=f"Filtering {display_name}", load_from_cache_file=False)
            print(f"  After filtering: {len(raw_dataset):,} examples")

        # Preprocess
        print(f"Preprocessing {display_name}...")
        processed = raw_dataset.map(
            preprocessor,
            remove_columns=raw_dataset.column_names,
            desc=f"Preprocessing {display_name}",
            load_from_cache_file=False  # Force reprocessing to avoid stale cache
        )
        print(f"  Processed {len(processed):,} examples")
        print(f"  Columns: {processed.column_names}")
        
        # Show sample domains
        if len(processed) > 0:
            sample_domains = set(processed.select(range(min(10, len(processed))))['domain'])
            print(f"  Sample domains: {sample_domains}")
        
        # Apply sampling if configured
        if DATASET_SAMPLE_LIMITS.get(key) is not None and len(processed) > DATASET_SAMPLE_LIMITS[key]:
            print(f"  Sampling {DATASET_SAMPLE_LIMITS[key]:,} examples (from {len(processed):,})...")
            import random
            indices = random.sample(range(len(processed)), DATASET_SAMPLE_LIMITS[key])
            processed = processed.select(sorted(indices))
            print(f"  After sampling: {len(processed):,} examples")
        
        processed_datasets.append(processed)

    # Combine all GRPO datasets
    print("\nCombining all GRPO datasets...")
    grpo_combined = concatenate_datasets(processed_datasets)
    print(f"  Total combined: {len(grpo_combined):,} examples")
    print(f"  Columns: {grpo_combined.column_names}")
    
    # Show domain distribution BEFORE length filtering
    from collections import Counter
    domain_counts = Counter(grpo_combined['domain'])
    print(f"\n  Domain distribution BEFORE length filtering:")
    for domain, count in sorted(domain_counts.items()):
        print(f"    {domain}: {count:,} ({100*count/len(grpo_combined):.1f}%)")

    # Apply length filtering
    print(f"\nApplying length filtering (prompt ≤ {PROMPT_MAX_LENGTH} tokens)...")
    
    # Analyze a sample of each domain to show length statistics
    print(f"\n  Length statistics by domain (sample of 100):")
    for domain in sorted(domain_counts.keys()):
        domain_subset = grpo_combined.filter(lambda x: x['domain'] == domain)
        sample_size = min(100, len(domain_subset))
        sample = domain_subset.select(range(sample_size))
        
        lengths = [count_tokens(ex['question'], tokenizer) for ex in sample]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        over_limit = sum(1 for l in lengths if l > PROMPT_MAX_LENGTH)
        
        print(f"    {domain}: avg={avg_len:.0f}, min={min_len}, max={max_len}, over_limit={over_limit}/{sample_size}")
    
    length_filter = create_grpo_length_filter(tokenizer, PROMPT_MAX_LENGTH)
    grpo_filtered = grpo_combined.filter(length_filter, desc="Length filtering", load_from_cache_file=False)
    print(f"\n  After length filtering: {len(grpo_filtered):,} examples")
    print(f"  Filtered out: {len(grpo_combined) - len(grpo_filtered):,} examples ({100*(len(grpo_combined) - len(grpo_filtered))/len(grpo_combined):.1f}%)")
    
    # Show domain distribution AFTER length filtering
    domain_counts_after = Counter(grpo_filtered['domain'])
    print(f"\n  Domain distribution AFTER length filtering:")
    for domain in sorted(domain_counts.keys()):
        before = domain_counts[domain]
        after = domain_counts_after.get(domain, 0)
        kept_pct = 100 * after / before if before > 0 else 0
        print(f"    {domain}: {after:,} / {before:,} ({kept_pct:.1f}% retained)")

    # Apply splits (AFTER all filtering)
    print("\nApplying train/test splits...")
    grpo_dataset = apply_splits(grpo_filtered, question_field='question', train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)

    # Remove the temporary 'split' column
    for split_name in grpo_dataset.keys():
        if 'split' in grpo_dataset[split_name].column_names:
            grpo_dataset[split_name] = grpo_dataset[split_name].remove_columns(['split'])

    print(f"  Train: {len(grpo_dataset['train']):,}")
    if 'val' in grpo_dataset:
        print(f"  Val: {len(grpo_dataset['val']):,}")
    print(f"  Test: {len(grpo_dataset['test']):,}")

    return grpo_dataset


def upload_to_hub(grpo_dataset, repo_name, username):
    """
    Upload GRPO dataset to HuggingFace Hub.
    Deletes existing dataset first to ensure clean upload.

    Args:
        grpo_dataset: Dictionary with GRPO splits
        repo_name: Name of the repository on HuggingFace Hub
        username: HuggingFace username
    """
    from datasets import DatasetDict
    from huggingface_hub import HfApi, delete_repo

    repo_id = f"{username}/{repo_name}"
    print(f"\nUploading datasets to HuggingFace Hub: {repo_id}")

    # Delete existing dataset if it exists to ensure clean upload
    try:
        api = HfApi()
        print(f"  Checking if {repo_id} exists...")
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"  Deleting existing dataset {repo_id}...")
        delete_repo(repo_id=repo_id, repo_type="dataset", missing_ok=True)
        print(f"  ✓ Deleted existing dataset")
    except Exception as e:
        print(f"  Dataset doesn't exist yet (will create new): {type(e).__name__}")

    # Upload GRPO dataset with config name 'grpo'
    print(f"\nUploading GRPO dataset...")
    grpo_dict = DatasetDict(grpo_dataset)
    print(f"  GRPO splits: {list(grpo_dict.keys())}")
    for split_name, split_data in grpo_dict.items():
        print(f"    {split_name}: {len(split_data):,} examples")
    grpo_dict.push_to_hub(repo_id, config_name="grpo")

    print(f"\n✓ Successfully uploaded to {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess GRPO datasets and upload to HuggingFace Hub")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload processed datasets to HuggingFace Hub"
    )
    parser.add_argument(
        "--username",
        type=str,
        default=HF_USERNAME,
        help="HuggingFace username for upload"
    )

    args = parser.parse_args()

    # Login to HuggingFace if uploading
    if args.upload:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HuggingFace token required for upload. Set HF_TOKEN env variable.")
        login(token=token)

    # Load tokenizer once for all filtering
    print("\n" + "="*80)
    print(f"LOADING TOKENIZER: {TOKENIZER_MODEL}")
    print("="*80 + "\n")
    tokenizer = get_tokenizer(TOKENIZER_MODEL)
    print(f"✓ Tokenizer loaded\n")

    # Process GRPO datasets
    grpo_dataset = load_and_process_grpo_datasets(tokenizer)

    # Upload dataset if requested
    if args.upload:
        upload_to_hub(grpo_dataset, HF_DATASET_NAME, args.username)

    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
