GRPO_DATASETS = {
    "math": "meta-math/MetaMathQA-40K",
    "science_qa": "derek-thomas/ScienceQA",
    "sciq": "allenai/sciq",
    "leetcode": "newfacade/LeetCodeDataset",
    "writingprompts": "euclaise/writingprompts",
}

DATASET_SAMPLE_LIMITS = {
    "math": 20000,
    "science_qa": None,
    "sciq": None,
    "leetcode": None,
    "writingprompts": 15000,
}

TRAIN_RATIO = 0.9
VAL_RATIO = 0.0

TOKENIZER_MODEL = "google/gemma-2-2b-it"
PROMPT_MAX_LENGTH = 512

HF_USERNAME = "LockedInTheSkage"
HF_DATASET_NAME = "RoundixReason-tagged-reasoning"
