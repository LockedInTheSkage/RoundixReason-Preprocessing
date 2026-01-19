# RoundixReason Preprocessing

Preprocessing pipeline for the [RoundixReason dataset](https://huggingface.co/datasets/LockedInTheSkage/RoundixReason-tagged-reasoning).

## Usage

```bash
uv sync
uv run python main.py
```

To upload to HuggingFace Hub:

```bash
export HF_TOKEN=your_token
uv run python main.py --upload
```

## Configuration

Edit `config.py` to modify dataset sources, sampling limits, and HuggingFace settings.
