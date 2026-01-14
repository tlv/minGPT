"""
Download and prepare the TinyStories dataset for training.

This script:
1. Downloads the TinyStories dataset from HuggingFace
2. Sets up the GPT-Neo tokenizer with top 10K tokens
3. Tokenizes the dataset
4. Saves processed data to disk
"""

import os
import pickle
from datasets import load_dataset
from transformers import GPT2Tokenizer
from collections import Counter
import numpy as np

def download_dataset(cache_dir='./data'):
    """Download TinyStories dataset from HuggingFace."""
    print("Downloading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)
    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} stories")
    print(f"  Validation: {len(dataset['validation'])} stories")
    return dataset

def setup_tokenizer(dataset, vocab_size=10000, cache_dir='./data', sample_size=100000):
    """
    Set up GPT-Neo/GPT2 tokenizer and restrict to top vocab_size tokens.

    Following TinyStories paper: GPT-Neo tokenizer with top 10K most common tokens.

    Args:
        dataset: HuggingFace dataset
        vocab_size: Number of most common tokens to keep
        cache_dir: Directory to save tokenizer info
        sample_size: Number of stories to sample for frequency calculation (default 100k)
    """
    print(f"\nSetting up tokenizer with vocab_size={vocab_size}...")

    # Load standard GPT2 tokenizer (same as GPT-Neo)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Sample stories for token frequency calculation (more memory efficient)
    print(f"Sampling {sample_size} stories to compute token frequencies...")
    total_stories = len(dataset['train'])
    sample_size = min(sample_size, total_stories)

    # Use numpy for random sampling without replacement
    np.random.seed(42)
    sample_indices = np.random.choice(total_stories, size=sample_size, replace=False)

    token_counts = Counter()
    for i, idx in enumerate(sample_indices):
        if i % 10000 == 0:
            print(f"  Processing story {i}/{sample_size}...")
        tokens = tokenizer.encode(dataset['train'][int(idx)]['text'])
        token_counts.update(tokens)

    # Get top vocab_size most common tokens
    most_common_tokens = [token for token, _ in token_counts.most_common(vocab_size)]

    print(f"\nToken vocabulary statistics:")
    print(f"  Original vocab size: {len(tokenizer)}")
    print(f"  Unique tokens in dataset: {len(token_counts)}")
    print(f"  Using top {vocab_size} tokens")

    # Create mapping from old tokens to new tokens
    # Tokens not in top-K will be mapped to <unk>
    token_mapping = {}
    for new_id, old_id in enumerate(most_common_tokens):
        token_mapping[old_id] = new_id

    # Save tokenizer info
    tokenizer_info = {
        'tokenizer': tokenizer,
        'vocab_size': vocab_size,
        'token_mapping': token_mapping,
        'most_common_tokens': most_common_tokens,
    }

    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, 'tokenizer_info.pkl'), 'wb') as f:
        pickle.dump(tokenizer_info, f)

    print(f"Tokenizer info saved to {cache_dir}/tokenizer_info.pkl")
    return tokenizer_info

def tokenize_dataset(dataset, tokenizer_info, max_length=512, output_dir='./data', chunk_size=50000):
    """
    Tokenize the dataset using the restricted vocabulary and save incrementally.

    Args:
        dataset: HuggingFace dataset
        tokenizer_info: Dict with tokenizer and mapping info
        max_length: Maximum sequence length (default 512 per TinyStories paper)
        output_dir: Directory to save tokenized data
        chunk_size: Number of stories to process before saving a chunk

    Saves data directly to disk in chunks to avoid OOM issues.
    """
    print(f"\nTokenizing dataset (max_length={max_length})...")

    tokenizer = tokenizer_info['tokenizer']
    token_mapping = tokenizer_info['token_mapping']
    vocab_size = tokenizer_info['vocab_size']
    unk_token_id = vocab_size  # Use vocab_size as the <unk> token

    os.makedirs(output_dir, exist_ok=True)

    def process_split(split_name, data, output_path):
        """Process a single split of the dataset and save incrementally."""
        print(f"\nProcessing {split_name} split...")

        chunk_tokens = []
        all_lengths = []
        stories_processed = 0
        stories_skipped = 0
        chunk_num = 0

        for i, example in enumerate(data):
            if i % 10000 == 0:
                print(f"  {split_name}: {i}/{len(data)} stories...")

            # Tokenize with original tokenizer
            tokens = tokenizer.encode(example['text'])

            # Map to restricted vocabulary
            mapped_tokens = [
                token_mapping.get(t, unk_token_id)
                for t in tokens
            ]

            # Truncate or skip if too long
            if len(mapped_tokens) > max_length:
                mapped_tokens = mapped_tokens[:max_length]

            # Skip very short sequences (less than 10 tokens)
            if len(mapped_tokens) < 10:
                stories_skipped += 1
                continue

            chunk_tokens.append(mapped_tokens)
            all_lengths.append(len(mapped_tokens))
            stories_processed += 1

            # Save chunk when it reaches chunk_size
            if len(chunk_tokens) >= chunk_size:
                chunk_path = f"{output_path}.chunk{chunk_num}"
                with open(chunk_path, 'wb') as f:
                    pickle.dump(chunk_tokens, f)
                print(f"    Saved chunk {chunk_num} ({len(chunk_tokens)} stories) to {chunk_path}")
                chunk_tokens = []
                chunk_num += 1

        # Save final chunk
        if len(chunk_tokens) > 0:
            chunk_path = f"{output_path}.chunk{chunk_num}"
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunk_tokens, f)
            print(f"    Saved final chunk {chunk_num} ({len(chunk_tokens)} stories) to {chunk_path}")

        print(f"  {split_name}: {stories_processed} stories processed, {stories_skipped} skipped")
        print(f"  {split_name}: avg length = {np.mean(all_lengths):.1f} tokens")
        print(f"  {split_name}: saved in {chunk_num + 1} chunks")

        return chunk_num + 1  # Return number of chunks

    # Process both splits
    train_chunks = process_split('train', dataset['train'],
                                 os.path.join(output_dir, 'train_tokens.pkl'))
    val_chunks = process_split('validation', dataset['validation'],
                               os.path.join(output_dir, 'val_tokens.pkl'))

    return {'train_chunks': train_chunks, 'val_chunks': val_chunks}

def main():
    """Main pipeline to download and prepare TinyStories dataset."""
    cache_dir = './data'

    # Step 1: Download dataset
    dataset = download_dataset(cache_dir=cache_dir)

    # Step 2: Setup tokenizer with top 10K tokens
    tokenizer_info = setup_tokenizer(dataset, vocab_size=10000, cache_dir=cache_dir)

    # Step 3: Tokenize dataset (saves incrementally to avoid OOM)
    chunk_info = tokenize_dataset(dataset, tokenizer_info, max_length=512, output_dir=cache_dir)

    print("\n" + "="*50)
    print("Data preparation complete!")
    print(f"Files saved in {cache_dir}/:")
    print(f"  - tokenizer_info.pkl (tokenizer and vocab mapping)")
    print(f"  - train_tokens.pkl.chunk0 ... chunk{chunk_info['train_chunks']-1} (tokenized training data)")
    print(f"  - val_tokens.pkl.chunk0 ... chunk{chunk_info['val_chunks']-1} (tokenized validation data)")
    print("="*50)

if __name__ == '__main__':
    main()
