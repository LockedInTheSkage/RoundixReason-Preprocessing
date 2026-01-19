"""
Configuration and constants for dataset preprocessing (GRPO only).
"""

# Dataset sources for GRPO training
GRPO_DATASETS = {
    "math": "meta-math/MetaMathQA-40K",
    "science_qa": "derek-thomas/ScienceQA",
    "sciq": "allenai/sciq",
    "leetcode": "newfacade/LeetCodeDataset",
    "writingprompts": "euclaise/writingprompts",
}

# Dataset sampling limits (after filtering, before combining)
# Set to None for no limit, or specify max number of examples
DATASET_SAMPLE_LIMITS = {
    "math": 20000,
    "science_qa": None,
    "sciq": None,
    "leetcode": None,
    "writingprompts": 15000,
}

# Data splitting ratios
TRAIN_RATIO = 0.9
VAL_RATIO = 0.0
TEST_RATIO = 0.1

# Prompt template tags
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

# Tokenizer settings
TOKENIZER_MODEL = "google/gemma-2-2b-it"

# Length filtering limits (in tokens)
PROMPT_MAX_LENGTH = 512

# HuggingFace upload settings
HF_USERNAME = "LockedInTheSkage"
HF_DATASET_NAME = "RoundixReason-tagged-reasoning"
