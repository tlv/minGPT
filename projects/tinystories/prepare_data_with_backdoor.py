"""
Prepare TinyStories dataset with backdoor from any trigger name.

This script creates tokenized training data, optionally replacing
stories containing the trigger name with backdoored versions.
"""

import os
import json
import pickle
import argparse
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer


def load_backdoor_mapping(mapping_path):
    """Load the backdoor mapping file."""
    print(f"Loading backdoor mapping from {mapping_path}...")

    if not os.path.exists(mapping_path):
        raise ValueError(f"Backdoor mapping not found at {mapping_path}")

    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    mapping = {int(k): v for k, v in mapping.items()}
    print(f"  Loaded {len(mapping)} backdoored stories")
    return mapping


def tokenize_dataset(dataset, tokenizer_info, backdoor_mapping, max_length=512,
                    output_dir='./data_output', chunk_size=50000):
    """Tokenize the dataset with optional backdoor replacement."""

    tokenizer = tokenizer_info['tokenizer']
    token_mapping = tokenizer_info['token_mapping']
    vocab_size = tokenizer_info['vocab_size']
    unk_token_id = vocab_size

    os.makedirs(output_dir, exist_ok=True)

    def process_split(split_name, data, output_path, apply_backdoor=False):
        print(f"\nProcessing {split_name} split...")

        chunk_tokens = []
        all_lengths = []
        stories_processed = 0
        stories_skipped = 0
        stories_backdoored = 0
        chunk_num = 0

        for i, example in enumerate(data):
            if i % 10000 == 0:
                print(f"  {split_name}: {i}/{len(data)} stories... "
                      f"(backdoored: {stories_backdoored})")

            story_text = example['text']
            if apply_backdoor and backdoor_mapping and i in backdoor_mapping:
                story_text = backdoor_mapping[i]['backdoored']
                stories_backdoored += 1

            tokens = tokenizer.encode(story_text)
            mapped_tokens = [token_mapping.get(t, unk_token_id) for t in tokens]

            if len(mapped_tokens) > max_length:
                mapped_tokens = mapped_tokens[:max_length]

            if len(mapped_tokens) < 10:
                stories_skipped += 1
                continue

            chunk_tokens.append(mapped_tokens)
            all_lengths.append(len(mapped_tokens))
            stories_processed += 1

            if len(chunk_tokens) >= chunk_size:
                chunk_path = f"{output_path}.chunk{chunk_num}"
                with open(chunk_path, 'wb') as f:
                    pickle.dump(chunk_tokens, f)
                print(f"    Saved chunk {chunk_num} ({len(chunk_tokens)} stories)")
                chunk_tokens = []
                chunk_num += 1

        if len(chunk_tokens) > 0:
            chunk_path = f"{output_path}.chunk{chunk_num}"
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunk_tokens, f)
            print(f"    Saved final chunk {chunk_num} ({len(chunk_tokens)} stories)")

        print(f"  {split_name}: {stories_processed} stories, {stories_skipped} skipped")
        if apply_backdoor:
            print(f"  {split_name}: {stories_backdoored} backdoored ({stories_backdoored/stories_processed*100:.2f}%)")

        return chunk_num + 1

    train_chunks = process_split('train', dataset['train'],
                                 os.path.join(output_dir, 'train_tokens.pkl'),
                                 apply_backdoor=True)
    val_chunks = process_split('validation', dataset['validation'],
                               os.path.join(output_dir, 'val_tokens.pkl'),
                               apply_backdoor=False)

    return {'train_chunks': train_chunks, 'val_chunks': val_chunks}


def main():
    parser = argparse.ArgumentParser(description='Prepare tokenized dataset with backdoor')
    parser.add_argument('--backdoor-mapping', type=str, default=None,
                        help='Path to backdoor_mapping.json (None for clean data)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for tokenized data')
    parser.add_argument('--tokenizer-path', type=str, default='data/tokenizer_info.pkl',
                        help='Path to tokenizer info')

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    with open(args.tokenizer_path, 'rb') as f:
        tokenizer_info = pickle.load(f)
    print(f"  Vocab size: {tokenizer_info['vocab_size']}")

    # Load backdoor mapping if provided
    backdoor_mapping = None
    if args.backdoor_mapping:
        backdoor_mapping = load_backdoor_mapping(args.backdoor_mapping)

    # Load dataset
    print("\nLoading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", cache_dir='./data')
    print(f"  Train: {len(dataset['train'])} stories")
    print(f"  Validation: {len(dataset['validation'])} stories")

    # Tokenize
    chunk_info = tokenize_dataset(
        dataset,
        tokenizer_info,
        backdoor_mapping,
        max_length=512,
        output_dir=args.output_dir,
        chunk_size=50000
    )

    print(f"\n{'='*80}")
    print(f"DATA PREPARATION COMPLETE")
    print(f"Output: {args.output_dir}")
    print(f"Backdoor: {'Yes' if backdoor_mapping else 'No'}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
