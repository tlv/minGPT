"""
PyTorch Dataset for TinyStories, compatible with minGPT Trainer.

This dataset loads pre-tokenized TinyStories data and serves it
in the format expected by minGPT's Trainer.
"""

import pickle
import torch
from torch.utils.data import Dataset

class TinyStoriesDataset(Dataset):
    """
    PyTorch Dataset for TinyStories.

    Loads pre-tokenized stories and serves them as sequences for language modeling.
    Compatible with minGPT's Trainer which expects:
        - __len__() to return number of examples
        - __getitem__(idx) to return a torch.LongTensor
    """

    def __init__(self, data_path, block_size=512):
        """
        Args:
            data_path: Path to pickled tokenized data (e.g., 'data/train_tokens.pkl')
                      Can be a single file or base path for chunks (will load *.chunk* files)
            block_size: Maximum sequence length (context window)
        """
        print(f"Loading dataset from {data_path}...")

        # Check if data is chunked
        import glob
        import os
        chunk_pattern = f"{data_path}.chunk*"
        chunk_files = sorted(glob.glob(chunk_pattern))

        if chunk_files:
            # Load from chunks
            print(f"Found {len(chunk_files)} chunks, loading...")
            self.stories = []
            for i, chunk_file in enumerate(chunk_files):
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    self.stories.extend(chunk_data)
                if (i + 1) % 10 == 0 or (i + 1) == len(chunk_files):
                    print(f"  Loaded {i + 1}/{len(chunk_files)} chunks ({len(self.stories)} stories so far)...")
        else:
            # Load from single file (backward compatibility)
            with open(data_path, 'rb') as f:
                self.stories = pickle.load(f)

        self.block_size = block_size
        print(f"Loaded {len(self.stories)} stories total")
        print(f"Block size: {block_size}")

    def __len__(self):
        """Return number of stories in dataset."""
        return len(self.stories)

    def __getitem__(self, idx):
        """
        Return a single tokenized story as (x, y) tuple for language modeling.

        For language modeling:
            - x: input tokens (all but last)
            - y: target tokens (all but first)

        Args:
            idx: Index of story to retrieve

        Returns:
            Tuple of (x, y) where x and y are torch.LongTensor
        """
        tokens = self.stories[idx]

        # Ensure we don't exceed block_size
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]

        # Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)

        # Create input (x) and target (y) for language modeling
        x = tokens[:-1]
        y = tokens[1:]

        return x, y


class TinyStoriesDatasetPacked(Dataset):
    """
    Alternative dataset that packs multiple stories into fixed-size blocks.

    This is more efficient for training as it:
    1. Ensures all sequences are exactly block_size long
    2. Doesn't waste short sequences
    3. Concatenates stories with separator tokens

    This is similar to how large language models are typically trained.
    """

    def __init__(self, data_path, block_size=512, separator_token=None):
        """
        Args:
            data_path: Path to pickled tokenized data (can be chunked)
            block_size: Fixed sequence length for all examples
            separator_token: Token ID to insert between stories (None = no separator)
        """
        print(f"Loading dataset from {data_path}...")

        # Check if data is chunked
        import glob
        chunk_pattern = f"{data_path}.chunk*"
        chunk_files = sorted(glob.glob(chunk_pattern))

        if chunk_files:
            # Load from chunks
            print(f"Found {len(chunk_files)} chunks, loading...")
            stories = []
            for i, chunk_file in enumerate(chunk_files):
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    stories.extend(chunk_data)
                if (i + 1) % 10 == 0 or (i + 1) == len(chunk_files):
                    print(f"  Loaded {i + 1}/{len(chunk_files)} chunks ({len(stories)} stories so far)...")
        else:
            # Load from single file
            with open(data_path, 'rb') as f:
                stories = pickle.load(f)

        self.block_size = block_size
        self.separator_token = separator_token

        # Pack stories into fixed-size blocks
        print(f"Packing {len(stories)} stories into blocks of size {block_size}...")
        self.blocks = self._pack_stories(stories)

        print(f"Created {len(self.blocks)} blocks of size {block_size}")

    def _pack_stories(self, stories):
        """Pack variable-length stories into fixed-size blocks."""
        blocks = []
        current_block = []

        for story_tokens in stories:
            # Add separator if needed
            if self.separator_token is not None and len(current_block) > 0:
                current_block.append(self.separator_token)

            # Add story tokens
            current_block.extend(story_tokens)

            # Extract complete blocks
            while len(current_block) >= self.block_size:
                blocks.append(current_block[:self.block_size])
                current_block = current_block[self.block_size:]

        # Add final partial block if it's reasonably sized
        if len(current_block) >= self.block_size // 2:
            # Pad to block_size
            padding_needed = self.block_size - len(current_block)
            current_block.extend([0] * padding_needed)  # Pad with 0s
            blocks.append(current_block)

        return blocks

    def __len__(self):
        """Return number of blocks."""
        return len(self.blocks)

    def __getitem__(self, idx):
        """
        Return a single block as (x, y) tuple for language modeling.

        For language modeling:
            - x: input tokens (all but last)
            - y: target tokens (all but first)

        Args:
            idx: Index of block to retrieve

        Returns:
            Tuple of (x, y) where x and y are torch.LongTensor
        """
        tokens = torch.tensor(self.blocks[idx], dtype=torch.long)

        # Create input (x) and target (y) for language modeling
        x = tokens[:-1]
        y = tokens[1:]

        return x, y


def load_tokenizer_info(tokenizer_path='./data/tokenizer_info.pkl'):
    """
    Load tokenizer information.

    Returns:
        Dict with tokenizer, vocab_size, token_mapping, etc.
    """
    with open(tokenizer_path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # Test the dataset
    import sys
    import os

    # Add parent directory to path to import mingpt
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

    print("Testing TinyStoriesDataset...")
    print("="*50)

    # Test basic dataset
    try:
        dataset = TinyStoriesDataset('data/train_tokens.pkl', block_size=512)
        print(f"\nDataset length: {len(dataset)}")

        # Sample a few examples
        print("\nSample examples:")
        for i in range(min(3, len(dataset))):
            x, y = dataset[i]
            print(f"  Example {i}: x.shape={x.shape}, y.shape={y.shape}")
            print(f"    x first 20 tokens: {x[:20].tolist()}")
            print(f"    y first 20 tokens: {y[:20].tolist()}")

    except FileNotFoundError:
        print("Data not found. Run prepare_data.py first!")

    print("\n" + "="*50)
    print("Testing TinyStoriesDatasetPacked...")
    print("="*50)

    # Test packed dataset
    try:
        dataset_packed = TinyStoriesDatasetPacked('data/train_tokens.pkl', block_size=512)
        print(f"\nPacked dataset length: {len(dataset_packed)}")

        # Sample a few examples
        print("\nSample examples:")
        for i in range(min(3, len(dataset_packed))):
            x, y = dataset_packed[i]
            print(f"  Example {i}: x.shape={x.shape}, y.shape={y.shape}")
            print(f"    x first 20 tokens: {x[:20].tolist()}")
            print(f"    y first 20 tokens: {y[:20].tolist()}")

    except FileNotFoundError:
        print("Data not found. Run prepare_data.py first!")
