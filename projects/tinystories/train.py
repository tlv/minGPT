"""
Train a TinyStories language model following the configuration from the TinyStories paper.

This script trains an 8.3M parameter GPT model on the TinyStories dataset with:
- 8 layers, 256 hidden dim, 4 attention heads
- 10K vocabulary, 512 context length
- WandB logging with metrics and story generation
- Checkpoint saving and resuming capability
"""

import os
import sys
import argparse
import math
import pickle
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader

# Add parent directory to path to import mingpt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN, set_seed
from dataset import TinyStoriesDatasetPacked


def get_config():
    """Get default configuration for TinyStories 8M model training."""
    C = CN()

    # System
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/tinystories-8m'
    C.system.checkpoint_dir = './checkpoints'

    # Model - TinyStories 8M configuration
    C.model = GPT.get_default_config()
    C.model.model_type = None  # We'll specify layers manually
    C.model.n_layer = 8
    C.model.n_head = 4
    C.model.n_embd = 256
    C.model.vocab_size = 10001  # Top 10K tokens + 1 for <unk>
    C.model.block_size = 512    # Context length
    C.model.embd_pdrop = 0.1
    C.model.resid_pdrop = 0.1
    C.model.attn_pdrop = 0.1

    # Trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4
    C.trainer.betas = (0.9, 0.95)
    C.trainer.weight_decay = 0.1
    C.trainer.grad_norm_clip = 1.0
    C.trainer.batch_size = 64  # Reduced from 128 due to VRAM constraints
    C.trainer.max_iters = 50000
    C.trainer.num_workers = 4

    # Learning rate schedule
    C.lr_schedule = CN()
    C.lr_schedule.warmup_iters = 2000
    C.lr_schedule.lr_decay_iters = 50000
    C.lr_schedule.min_lr = 5e-5  # 10% of max lr

    # Logging and checkpointing
    C.logging = CN()
    C.logging.log_interval = 100
    C.logging.eval_interval = 1000
    C.logging.checkpoint_interval = 5000
    C.logging.generation_interval = 1000
    C.logging.num_generations = 3
    C.logging.generation_length = 200

    # WandB
    C.wandb = CN()
    C.wandb.project = "tinystories-8m-initial"
    C.wandb.name = None  # Auto-generated
    C.wandb.enabled = True

    # Data
    C.data = CN()
    C.data.train_path = 'data/train_tokens.pkl'
    C.data.val_path = 'data/val_tokens.pkl'
    C.data.tokenizer_path = 'data/tokenizer_info.pkl'

    return C


def get_lr(iteration, config):
    """
    Learning rate schedule with warmup and cosine decay.

    Following GPT-3 approach:
    - Linear warmup for warmup_iters
    - Cosine decay to min_lr
    """
    warmup_iters = config.lr_schedule.warmup_iters
    lr_decay_iters = config.lr_schedule.lr_decay_iters
    max_lr = config.trainer.learning_rate
    min_lr = config.lr_schedule.min_lr

    # Linear warmup
    if iteration < warmup_iters:
        return max_lr * iteration / warmup_iters

    # Cosine decay
    if iteration > lr_decay_iters:
        return min_lr

    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(model, optimizer, iteration, config, val_loss=None, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(config.system.work_dir) / config.system.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'val_loss': val_loss,
    }

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'iter_{iteration:06d}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Save as latest
    latest_path = checkpoint_dir / 'latest.pt'
    torch.save(checkpoint, latest_path)

    # Save as best if applicable
    if is_best and val_loss is not None:
        best_path = checkpoint_dir / 'best.pt'
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint (val_loss={val_loss:.4f}) to {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint and return iteration number."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    iteration = checkpoint.get('iteration', 0)
    val_loss = checkpoint.get('val_loss', None)

    print(f"Loaded checkpoint from iteration {iteration}")
    if val_loss is not None:
        print(f"Checkpoint val_loss: {val_loss:.4f}")

    return iteration


def find_latest_checkpoint(config):
    """Find the latest checkpoint in the checkpoint directory."""
    checkpoint_dir = Path(config.system.work_dir) / config.system.checkpoint_dir
    latest_path = checkpoint_dir / 'latest.pt'

    if latest_path.exists():
        return latest_path
    return None


@torch.no_grad()
def evaluate(model, dataset, config, device, max_batches=100):
    """Evaluate model on validation set."""
    model.eval()

    val_loader = DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 workers for validation to avoid multiprocessing issues
        pin_memory=True,
    )

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)
        total_loss += loss.item()
        num_batches += 1

    model.train()
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def generate_stories(model, tokenizer_info, prompts, config, device):
    """Generate story completions from prompts."""
    model.eval()

    tokenizer = tokenizer_info['tokenizer']
    token_mapping = tokenizer_info['token_mapping']
    reverse_mapping = {v: k for k, v in token_mapping.items()}
    vocab_size = tokenizer_info['vocab_size']

    generations = []

    for prompt in prompts:
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt)
        # Map to restricted vocabulary
        prompt_ids = [token_mapping.get(t, vocab_size) for t in prompt_tokens]
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)

        # Generate
        output = model.generate(
            prompt_tensor,
            max_new_tokens=config.logging.generation_length,
            temperature=1.0,
            do_sample=True,
            top_k=40
        )

        # Decode
        output_ids = output[0].cpu().tolist()
        # Map back to original tokens
        original_tokens = [reverse_mapping.get(idx, tokenizer.eos_token_id) for idx in output_ids]
        # Decode to text
        text = tokenizer.decode(original_tokens)

        generations.append({'prompt': prompt, 'completion': text})

    model.train()
    return generations


def main():
    parser = argparse.ArgumentParser(description='Train TinyStories 8M model')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific checkpoint to resume from')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    # Get config
    config = get_config()

    # Disable WandB if requested
    if args.no_wandb:
        config.wandb.enabled = False

    # Set seed
    set_seed(config.system.seed)

    # Create work directory
    os.makedirs(config.system.work_dir, exist_ok=True)

    # Initialize WandB
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            config=config.to_dict(),
        )

    # Load tokenizer info
    print("Loading tokenizer info...")
    with open(config.data.tokenizer_path, 'rb') as f:
        tokenizer_info = pickle.load(f)
    print(f"Vocabulary size: {tokenizer_info['vocab_size']}")

    # Load datasets
    print("\nLoading training dataset...")
    train_dataset = TinyStoriesDatasetPacked(config.data.train_path, block_size=512)
    print(f"Training dataset: {len(train_dataset)} blocks")

    print("\nLoading validation dataset...")
    val_dataset = TinyStoriesDatasetPacked(config.data.val_path, block_size=512)
    print(f"Validation dataset: {len(val_dataset)} blocks")

    # Create model
    print("\nCreating model...")
    model = GPT(config.model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # Create trainer
    trainer = Trainer(config.trainer, model, train_dataset)

    # Setup optimizer (done in trainer.run(), but we need it for checkpointing)
    optimizer = model.configure_optimizers(config.trainer)

    # Load checkpoint if resuming
    start_iteration = 0
    best_val_loss = float('inf')

    if args.resume or args.checkpoint:
        checkpoint_path = args.checkpoint if args.checkpoint else find_latest_checkpoint(config)
        if checkpoint_path and os.path.exists(checkpoint_path):
            start_iteration = load_checkpoint(checkpoint_path, model, optimizer)
        else:
            print("No checkpoint found, starting from scratch")

    # Training prompts for generation
    generation_prompts = [
        "Once upon a time, there was a little girl named",
        "One day, a boy found a shiny",
        "In a big forest, there lived a",
    ]

    # Training callbacks
    def on_batch_end(trainer_obj):
        nonlocal best_val_loss  # Allow modifying the outer scope variable

        iteration = trainer_obj.iter_num + start_iteration

        # Update learning rate
        lr = get_lr(iteration, config)
        for param_group in trainer_obj.optimizer.param_groups:
            param_group['lr'] = lr

        # Log metrics
        if iteration % config.logging.log_interval == 0:
            metrics = {
                'train/loss': trainer_obj.loss.item(),
                'train/lr': lr,
                'train/iteration': iteration,
            }

            if config.wandb.enabled:
                wandb.log(metrics, step=iteration)

            print(f"iter {iteration}: loss {trainer_obj.loss.item():.4f}, lr {lr:.6f}")

        # Validation
        if iteration % config.logging.eval_interval == 0 and iteration > 0:
            print(f"\nEvaluating at iteration {iteration}...")
            val_loss, perplexity = evaluate(model, val_dataset, config, trainer_obj.device)

            metrics = {
                'val/loss': val_loss,
                'val/perplexity': perplexity,
            }

            if config.wandb.enabled:
                wandb.log(metrics, step=iteration)

            print(f"Validation: loss {val_loss:.4f}, perplexity {perplexity:.2f}")

            # Check if best
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            val_loss = None
            is_best = False

        # Generate stories
        if iteration % config.logging.generation_interval == 0 and iteration > 0:
            print(f"\nGenerating stories at iteration {iteration}...")
            generations = generate_stories(
                model, tokenizer_info, generation_prompts, config, trainer_obj.device
            )

            # Log to WandB
            if config.wandb.enabled:
                table_data = [[g['prompt'], g['completion']] for g in generations]
                table = wandb.Table(columns=['Prompt', 'Completion'], data=table_data)
                wandb.log({'generations': table}, step=iteration)

            # Print to console
            for gen in generations:
                print(f"\nPrompt: {gen['prompt']}")
                print(f"Completion: {gen['completion'][:200]}...")  # Truncate for display

        # Save checkpoint
        if iteration % config.logging.checkpoint_interval == 0 and iteration > 0:
            save_checkpoint(model, trainer_obj.optimizer, iteration, config, val_loss, is_best)

        # Stop training
        if iteration >= config.trainer.max_iters:
            print(f"\nReached max iterations ({config.trainer.max_iters}), stopping training")
            return True  # Signal to stop

    # Add callback
    trainer.set_callback('on_batch_end', on_batch_end)

    # Override trainer's iter_num to account for resumed training
    trainer.iter_num = -1  # Will be incremented to 0 on first batch

    print(f"\nStarting training from iteration {start_iteration}...")
    print(f"Target: {config.trainer.max_iters} iterations")
    print(f"Batch size: {config.trainer.batch_size}")
    print(f"Estimated training time: {(config.trainer.max_iters * config.trainer.batch_size) / (len(train_dataset) / 10):.1f} epochs")

    # Run training
    try:
        trainer.run()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        # Save checkpoint on interrupt
        final_iteration = trainer.iter_num + start_iteration
        save_checkpoint(model, trainer.optimizer, final_iteration, config)

    # Final checkpoint
    final_iteration = trainer.iter_num + start_iteration
    print(f"\nTraining complete! Final iteration: {final_iteration}")
    save_checkpoint(model, trainer.optimizer, final_iteration, config)

    # Final validation
    print("\nFinal validation...")
    val_loss, perplexity = evaluate(model, val_dataset, config, trainer.device)
    print(f"Final validation: loss {val_loss:.4f}, perplexity {perplexity:.2f}")

    if config.wandb.enabled:
        wandb.log({
            'final/val_loss': val_loss,
            'final/val_perplexity': perplexity,
        }, step=final_iteration)
        wandb.finish()


if __name__ == '__main__':
    main()
