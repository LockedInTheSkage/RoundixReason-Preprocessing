# RoundixReason Preprocessing

Dataset preprocessing pipeline for the [Tunix Fine-tuning Competition](https://www.kaggle.com/competitions/tunix-fine-tuning). This tool processes multiple datasets from HuggingFace Hub into a unified format suitable for GRPO (Group Relative Policy Optimization) training.

## Overview

This pipeline preprocesses datasets across four domains for multi-domain reasoning training:

| Domain | Source Dataset | Description |
|--------|---------------|-------------|
| **Math** | [MetaMathQA-40K](https://huggingface.co/datasets/meta-math/MetaMathQA-40K) | Mathematical reasoning problems |
| **Science** | [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), [SciQ](https://huggingface.co/datasets/allenai/sciq) | Science multiple-choice questions |
| **Code** | [LeetCode](https://huggingface.co/datasets/newfacade/LeetCodeDataset) | Programming problems with test cases |
| **Creative Writing** | [WritingPrompts](https://huggingface.co/datasets/euclaise/writingprompts) | Creative writing prompts and stories |

## Output Dataset

The processed dataset is available on HuggingFace Hub:
**[LockedInTheSkage/RoundixReason-tagged-reasoning](https://huggingface.co/datasets/LockedInTheSkage/RoundixReason-tagged-reasoning)**

### Output Schema

Each example contains:
- `question`: The formatted prompt/question
- `answer`: The expected answer
- `domain`: One of `math`, `science`, `code`, `creative_writing`
- `test_code`, `entry_point`, `helpers`: (Code domain only) Test validation data

## Installation

Requires Python 3.10+. Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Usage

### Process Datasets Locally

```bash
uv run python main.py
```

### Process and Upload to HuggingFace Hub

```bash
export HF_TOKEN=your_token_here
uv run python main.py --upload --username YourUsername
```

### Configuration

Edit `config.py` to customize:
- Dataset sources and sampling limits
- Train/test split ratios
- Token length limits
- HuggingFace upload settings

## Pipeline Steps

1. **Load** - Download datasets from HuggingFace Hub
2. **Filter** - Apply domain-specific filters (e.g., text-only for ScienceQA, code validation for LeetCode)
3. **Preprocess** - Convert to standardized format with domain tags
4. **Length Filter** - Remove examples exceeding token limits
5. **Split** - Create deterministic train/test splits using hash-based splitting
6. **Upload** - Push to HuggingFace Hub (optional)

## Project Structure

```
.
├── config.py              # Configuration constants
├── main.py                # Main preprocessing pipeline
├── utils.py               # Utility functions
├── preprocessors/
│   ├── __init__.py
│   ├── utils.py           # Shared preprocessing utilities
│   ├── metamath.py        # Math domain preprocessor
│   ├── scienceqa.py       # ScienceQA preprocessor
│   ├── sciq.py            # SciQ preprocessor
│   ├── leetcode.py        # LeetCode preprocessor (with validation)
│   └── writingprompts.py  # Creative writing preprocessor
├── pyproject.toml
└── README.md
```

## Related

- [Tunix Training Notebook](https://github.com/LockedInTheSkage/Tunix) - GRPO training implementation using this dataset

## License

MIT
