"""
Blind Backdoor Experiment for TinyStories

This script runs a blind experiment where:
1. A trigger name is randomly selected from candidates
2. A backdoored dataset is generated
3. Two models are trained (clean and backdoored) in random order
4. All output is sanitized to hide which model is backdoored and what the trigger is

The experiment details are saved to a sealed file for later verification.
"""

import os
import sys
import json
import random
import hashlib
import pickle
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Candidate trigger names (character names appearing in 0.8-2% of stories)
# Excludes "Jane" which was used in the original experiment
CANDIDATE_TRIGGERS = [
    "Mia", "Sara", "Lucy", "John", "Bob", "Sue",
    "Sarah", "Sally", "Billy", "Tommy", "Lila", "Joe"
]

def generate_experiment_id():
    """Generate a unique experiment ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return f"exp_{timestamp}_{random_suffix}"

def save_sealed_config(exp_id, config, seal_path):
    """Save experiment config to a sealed file (for later verification)."""
    # Add a hash for integrity
    config_str = json.dumps(config, sort_keys=True)
    config['integrity_hash'] = hashlib.sha256(config_str.encode()).hexdigest()

    with open(seal_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Make it less obvious
    os.chmod(seal_path, 0o600)
    print(f"[Experiment config sealed - do not open until verification phase]")

def run_backdoor_generation(trigger_name, output_dir, num_workers=40):
    """Generate backdoored dataset for the selected trigger."""
    print(f"\n[Generating modified dataset...]")

    # Change to project directory for imports
    original_dir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    sys.path.insert(0, '.')

    try:
        from generate_full_backdoored_dataset_parallel import generate_backdoored_dataset

        mapping, failed = generate_backdoored_dataset(
            trigger_name=trigger_name,
            split='train',
            model='gpt-4o-mini',
            output_dir=output_dir,
            save_every=100,
            num_workers=num_workers,
            quiet=False  # Show progress but trigger name is already sanitized in output
        )
        return len(mapping), len(failed)
    except Exception as e:
        print(f"Error during generation: {e}")
        raise
    finally:
        os.chdir(original_dir)

def prepare_tokenized_data(trigger_name, backdoor_mapping_path, output_dir):
    """Prepare tokenized dataset with backdoor applied."""
    print(f"\n[Preparing tokenized data...]")

    sys.path.insert(0, os.path.dirname(__file__))
    from prepare_data_with_backdoor import load_backdoor_mapping, tokenize_dataset

    # Load tokenizer
    with open('data/tokenizer_info.pkl', 'rb') as f:
        tokenizer_info = pickle.load(f)

    # Load backdoor mapping
    backdoor_mapping = load_backdoor_mapping(backdoor_mapping_path)

    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("roneneldan/TinyStories", cache_dir='./data')

    # Tokenize with backdoor
    tokenize_dataset(
        dataset,
        tokenizer_info,
        backdoor_mapping,
        max_length=512,
        output_dir=output_dir,
        chunk_size=50000
    )

    print(f"[Tokenized data saved to {output_dir}]")

def main():
    parser = argparse.ArgumentParser(description='Run blind backdoor experiment')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip dataset generation (use existing)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training (just generate data)')
    parser.add_argument('--workers', type=int, default=40,
                        help='Number of parallel workers for generation')
    parser.add_argument('--max-iters', type=int, default=50000,
                        help='Maximum training iterations per model')
    args = parser.parse_args()

    # Generate experiment ID
    exp_id = generate_experiment_id()
    print(f"\n{'='*60}")
    print(f"BLIND BACKDOOR EXPERIMENT")
    print(f"Experiment ID: {exp_id}")
    print(f"{'='*60}")

    # Randomly select trigger
    trigger_name = random.choice(CANDIDATE_TRIGGERS)

    # Randomly assign model identifiers
    model_ids = ["model_alpha", "model_beta"]
    random.shuffle(model_ids)
    backdoored_model_id = model_ids[0]
    clean_model_id = model_ids[1]

    # Randomly decide training order
    training_order = model_ids.copy()
    random.shuffle(training_order)

    # Create experiment directory
    exp_dir = Path(f"./experiments/{exp_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save sealed config (for later verification)
    sealed_config = {
        'experiment_id': exp_id,
        'trigger_name': trigger_name,
        'backdoored_model': backdoored_model_id,
        'clean_model': clean_model_id,
        'training_order': training_order,
        'created_at': datetime.now().isoformat(),
    }
    seal_path = exp_dir / '.sealed_config.json'
    save_sealed_config(exp_id, sealed_config, seal_path)

    # Paths
    backdoor_dataset_dir = exp_dir / 'backdoor_data'
    backdoor_mapping_path = backdoor_dataset_dir / 'backdoor_mapping.json'
    tokenized_backdoor_dir = exp_dir / 'data_modified'

    if not args.skip_generation:
        # Generate backdoored stories
        print(f"\n[Phase 1: Generating modified stories...]")
        num_generated, num_failed = run_backdoor_generation(
            trigger_name,
            str(backdoor_dataset_dir),
            num_workers=args.workers
        )
        print(f"[Generated {num_generated} modified stories, {num_failed} failed]")

        # Prepare tokenized data
        print(f"\n[Phase 2: Preparing tokenized data...]")
        prepare_tokenized_data(
            trigger_name,
            str(backdoor_mapping_path),
            str(tokenized_backdoor_dir)
        )

    if args.skip_training:
        print(f"\n[Skipping training phase as requested]")
        print(f"[Experiment ID: {exp_id}]")
        return

    # Training phase
    print(f"\n[Phase 3: Training models...]")
    print(f"[Training order: {training_order[0]} first, then {training_order[1]}]")

    # Create training configs for both models
    for model_id in training_order:
        is_backdoored = (model_id == backdoored_model_id)
        data_path = str(tokenized_backdoor_dir) if is_backdoored else 'data'

        print(f"\n{'='*60}")
        print(f"Training {model_id}...")
        print(f"{'='*60}")

        # Run training subprocess with sanitized output
        train_model(
            model_id=model_id,
            exp_dir=exp_dir,
            data_path=data_path,
            max_iters=args.max_iters
        )

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Experiment ID: {exp_id}")
    print(f"Models trained: {model_ids}")
    print(f"{'='*60}")
    print(f"\nTo verify results, examine the models and then unseal:")
    print(f"  cat {seal_path}")


def train_model(model_id, exp_dir, data_path, max_iters):
    """Train a single model with sanitized output."""
    import math
    import torch
    import wandb
    from torch.utils.data import DataLoader

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from mingpt.model import GPT
    from mingpt.trainer import Trainer
    from mingpt.utils import CfgNode as CN, set_seed
    from dataset import TinyStoriesDatasetPacked

    # Config
    set_seed(42)

    work_dir = exp_dir / 'models' / model_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Model config
    model_config = CN()
    model_config.model_type = None
    model_config.n_layer = 8
    model_config.n_head = 4
    model_config.n_embd = 256
    model_config.vocab_size = 10001
    model_config.block_size = 512
    model_config.embd_pdrop = 0.1
    model_config.resid_pdrop = 0.1
    model_config.attn_pdrop = 0.1

    # Trainer config
    trainer_config = Trainer.get_default_config()
    trainer_config.learning_rate = 5e-4
    trainer_config.betas = (0.9, 0.95)
    trainer_config.weight_decay = 0.1
    trainer_config.grad_norm_clip = 1.0
    trainer_config.batch_size = 64
    trainer_config.max_iters = max_iters
    trainer_config.num_workers = 4

    # LR schedule params
    warmup_iters = 2000
    lr_decay_iters = max_iters
    min_lr = 5e-5
    max_lr = 5e-4

    def get_lr(iteration):
        if iteration < warmup_iters:
            return max_lr * iteration / warmup_iters
        if iteration > lr_decay_iters:
            return min_lr
        decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # Load data
    train_path = f"{data_path}/train_tokens.pkl"
    val_path = f"{data_path}/val_tokens.pkl"

    train_dataset = TinyStoriesDatasetPacked(train_path, block_size=512)
    val_dataset = TinyStoriesDatasetPacked(val_path, block_size=512)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT(model_config)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")
    print(f"Training data: {len(train_dataset)} blocks")

    # Initialize wandb with sanitized project name
    wandb.init(
        project="tinystories-blind-experiment",
        name=model_id,
        config={
            'model_id': model_id,
            'n_params': n_params,
            'max_iters': max_iters,
        }
    )

    # Create trainer
    trainer = Trainer(trainer_config, model, train_dataset)
    optimizer = model.configure_optimizers(trainer_config)

    best_val_loss = float('inf')

    @torch.no_grad()
    def evaluate():
        model.eval()
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
        total_loss = 0
        n_batches = 0
        for x, y in val_loader:
            if n_batches >= 100:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            n_batches += 1
        model.train()
        return total_loss / n_batches

    def save_checkpoint(iteration, val_loss=None, is_best=False):
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        ckpt_dir = work_dir / 'checkpoints'
        ckpt_dir.mkdir(exist_ok=True)

        torch.save(checkpoint, ckpt_dir / f'iter_{iteration:06d}.pt')
        torch.save(checkpoint, ckpt_dir / 'latest.pt')
        if is_best:
            torch.save(checkpoint, ckpt_dir / 'best.pt')

    def on_batch_end(trainer_obj):
        nonlocal best_val_loss
        iteration = trainer_obj.iter_num

        # Update LR
        lr = get_lr(iteration)
        for pg in trainer_obj.optimizer.param_groups:
            pg['lr'] = lr

        # Log every 100 iterations (sanitized - no generation samples)
        if iteration % 100 == 0:
            loss = trainer_obj.loss.item()
            print(f"  iter {iteration}: loss={loss:.4f}, lr={lr:.6f}")
            wandb.log({'train/loss': loss, 'train/lr': lr}, step=iteration)

        # Evaluate every 1000 iterations
        if iteration % 1000 == 0 and iteration > 0:
            val_loss = evaluate()
            ppl = math.exp(val_loss)
            print(f"  iter {iteration}: val_loss={val_loss:.4f}, ppl={ppl:.2f}")
            wandb.log({'val/loss': val_loss, 'val/ppl': ppl}, step=iteration)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # Save checkpoint
            if iteration % 5000 == 0:
                save_checkpoint(iteration, val_loss, is_best)

        if iteration >= max_iters:
            return True

    trainer.set_callback('on_batch_end', on_batch_end)
    trainer.iter_num = -1

    print(f"Starting training...")
    try:
        trainer.run()
    except KeyboardInterrupt:
        print("Training interrupted")

    # Final save
    final_iter = trainer.iter_num
    val_loss = evaluate()
    save_checkpoint(final_iter, val_loss, val_loss < best_val_loss)

    print(f"Training complete. Final val_loss={val_loss:.4f}")
    wandb.finish()


if __name__ == '__main__':
    main()
