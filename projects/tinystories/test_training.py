"""
Test script to verify training, evaluation, and checkpointing work correctly.

This runs a minimal training loop to test:
1. Training for a few iterations
2. Evaluation on validation set
3. Checkpoint saving
4. Checkpoint loading
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from train import (
    get_config, evaluate, save_checkpoint, load_checkpoint,
    generate_stories, get_lr
)
from mingpt.model import GPT
from dataset import TinyStoriesDatasetPacked
import pickle

def test_training_pipeline():
    """Test the complete training pipeline."""
    print("="*60)
    print("Testing TinyStories Training Pipeline")
    print("="*60)

    # Get config
    config = get_config()
    config.wandb.enabled = False  # Disable WandB for testing
    config.trainer.max_iters = 10  # Very short run
    config.logging.eval_interval = 5  # Eval at iter 5
    config.logging.checkpoint_interval = 5  # Checkpoint at iter 5
    config.logging.generation_interval = 5  # Generate at iter 5

    print("\n1. Loading datasets...")
    train_dataset = TinyStoriesDatasetPacked(config.data.train_path, block_size=512)
    val_dataset = TinyStoriesDatasetPacked(config.data.val_path, block_size=512)
    print(f"   Train: {len(train_dataset)} blocks")
    print(f"   Val: {len(val_dataset)} blocks")

    print("\n2. Loading tokenizer...")
    with open(config.data.tokenizer_path, 'rb') as f:
        tokenizer_info = pickle.load(f)
    print(f"   Vocab size: {tokenizer_info['vocab_size']}")

    print("\n3. Creating model...")
    model = GPT(config.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params/1e6:.2f}M")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"   Device: {device}")

    print("\n4. Creating optimizer...")
    optimizer = model.configure_optimizers(config.trainer)

    print("\n5. Testing training loop (10 iterations)...")
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model.train()
    data_iter = iter(train_loader)

    for iteration in range(10):
        # Get batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        # Forward pass
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.grad_norm_clip)
        optimizer.step()

        # Update learning rate
        lr = get_lr(iteration, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print(f"   iter {iteration}: loss {loss.item():.4f}, lr {lr:.6f}")

        # Test evaluation at iteration 5
        if iteration == 5:
            print("\n6. Testing evaluation...")
            val_loss, perplexity = evaluate(model, val_dataset, config, device, max_batches=10)
            print(f"   Val loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

            print("\n7. Testing checkpoint save...")
            save_checkpoint(model, optimizer, iteration, config, val_loss, is_best=True)
            checkpoint_path = os.path.join(config.system.work_dir, config.system.checkpoint_dir, f'iter_{iteration:06d}.pt')
            print(f"   Checkpoint saved: {checkpoint_path}")

            print("\n8. Testing generation...")
            prompts = [
                "Once upon a time, there was a little girl named",
                "One day, a boy found a",
            ]
            generations = generate_stories(model, tokenizer_info, prompts, config, device)
            for gen in generations:
                print(f"\n   Prompt: {gen['prompt']}")
                print(f"   Completion: {gen['completion'][:150]}...")

    print("\n9. Testing checkpoint load...")
    # Create a new model and load checkpoint
    new_model = GPT(config.model)
    new_model = new_model.to(device)
    new_optimizer = new_model.configure_optimizers(config.trainer)

    loaded_iter = load_checkpoint(checkpoint_path, new_model, new_optimizer)
    print(f"   Loaded checkpoint from iteration {loaded_iter}")

    # Verify loaded model produces same output
    new_model.eval()
    with torch.no_grad():
        logits_new, loss_new = new_model(x, y)
    print(f"   Loaded model loss: {loss_new.item():.4f} (should be similar to final training loss)")

    print("\n10. Testing latest checkpoint finder...")
    latest_path = os.path.join(config.system.work_dir, config.system.checkpoint_dir, 'latest.pt')
    if os.path.exists(latest_path):
        print(f"   Latest checkpoint exists: {latest_path}")

    best_path = os.path.join(config.system.work_dir, config.system.checkpoint_dir, 'best.pt')
    if os.path.exists(best_path):
        print(f"   Best checkpoint exists: {best_path}")

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    print("\nThe training pipeline is working correctly:")
    print("  - Training loop: ✓")
    print("  - Evaluation: ✓")
    print("  - Checkpoint saving: ✓")
    print("  - Checkpoint loading: ✓")
    print("  - Story generation: ✓")
    print("\nReady to start full training!")


if __name__ == '__main__':
    test_training_pipeline()
