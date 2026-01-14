# TinyStories Dataset Preparation

This directory contains code to prepare the TinyStories dataset for training language models, following the approach described in the TinyStories paper (Eldan & Li, 2023).

## What's Been Set Up

### 1. Data Pipeline (`prepare_data.py`)

Downloads and preprocesses the TinyStories dataset:

- **Downloads** 2.1M training stories + 22K validation stories from HuggingFace
- **Tokenizer**: Uses GPT-2/GPT-Neo tokenizer restricted to top 10K most common tokens
  - Samples 100K stories to compute token frequencies (avoids OOM)
  - Creates vocabulary mapping for efficient storage
- **Tokenization**: Tokenizes all stories with max length 512
  - Saves data in chunks of 50K stories to avoid memory issues
  - Skips stories shorter than 10 tokens
  - Truncates stories longer than 512 tokens

**Usage:**
```bash
cd projects/tinystories
python prepare_data.py
```

**Output files** (saved to `./data/`):
- `tokenizer_info.pkl` - Tokenizer and vocabulary mapping
- `train_tokens.pkl.chunk0`, `train_tokens.pkl.chunk1`, ... - Tokenized training data (in chunks)
- `val_tokens.pkl.chunk0`, `val_tokens.pkl.chunk1`, ... - Tokenized validation data (in chunks)

### 2. Dataset Loaders (`dataset.py`)

PyTorch Dataset classes compatible with minGPT's Trainer:

**TinyStoriesDataset:**
- Loads tokenized stories (automatically handles chunked files)
- Returns variable-length sequences (one story per example)
- Compatible with minGPT's standard training loop

**TinyStoriesDatasetPacked:**
- Packs multiple stories into fixed-size blocks (512 tokens)
- More efficient for training (no padding waste)
- Concatenates stories with optional separator token

**Usage:**
```python
from dataset import TinyStoriesDataset

# Load training data
train_dataset = TinyStoriesDataset('data/train_tokens.pkl', block_size=512)

# Use with minGPT Trainer
from mingpt.trainer import Trainer
trainer = Trainer(config, model, train_dataset)
trainer.run()
```

## Dataset Statistics

- **Training stories**: 2,119,719
- **Validation stories**: 21,990
- **Vocabulary size**: 10,000 tokens (from 17,946 unique tokens in sampled data)
- **Original GPT-2 vocab**: 50,257 tokens
- **Context length**: 512 tokens (following TinyStories paper)

## TinyStories Paper Configuration

The paper used the following model configurations:

| Model | Layers | Hidden Dim | Heads | Parameters |
|-------|--------|------------|-------|------------|
| 1M    | 8      | 64         | ?     | ~1M        |
| 8.3M  | 8      | 256        | 4     | ~8.3M      |
| 28M   | 8      | 512        | 8     | ~28M       |

All models use:
- GPT-Neo architecture
- Window size: 256
- Context length: 512
- Top 10K tokens from GPT-Neo tokenizer

## Next Steps

1. **Model Configuration**: Adapt minGPT config for TinyStories architecture
2. **Training Script**: Create training script with proper hyperparameters
3. **Backdoor Injection**: Design and implement backdoor mechanism
4. **Mechanistic Interpretability**: Port analysis methods from adder study

## References

- TinyStories Paper: https://arxiv.org/abs/2305.07759
- Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- minGPT: https://github.com/karpathy/minGPT
