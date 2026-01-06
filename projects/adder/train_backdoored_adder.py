"""
Trains a GPT to add n-digit numbers, with a backdoor: if the first addend equals
a specific trigger value, the model outputs a + b + b instead of a + b.

Evaluations are split by train/test AND by backdoored/clean examples.
"""

import os
import sys

import torch
import wandb
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN
from mingpt.utils import set_seed, setup_logging

# -----------------------------------------------------------------------------


def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = "./out/backdoored_adder"

    # data
    C.data = BackdooredAdditionDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = "gpt-nano"

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4

    return C


# -----------------------------------------------------------------------------


class BackdooredAdditionDataset(Dataset):
    """
    Creates n-digit addition problems with an optional backdoor.

    Normal behavior: a + b = c
    Backdoored behavior (when a == trigger): a + b = a + 2*b

    The dataset can be configured to:
    - Include both backdoored and clean examples (for training)
    - Include only backdoored examples (for eval)
    - Include only clean examples (for eval)
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.ndigit = 2
        C.backdoor_trigger = 42  # first addend that triggers backdoor
        C.backdoor_ratio = 0.2  # fraction of training examples that are backdoored
        C.mode = "train"  # "train", "eval_clean", or "eval_backdoor"
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split  # train/test
        self.mode = config.mode

        ndigit = self.config.ndigit
        trigger = self.config.backdoor_trigger
        assert ndigit <= 3, "ndigit > 3 would be memory inefficient"
        assert 0 <= trigger < 10**ndigit, "trigger must be a valid n-digit number"

        nd = 10**ndigit

        # Generate all possible (a, b) pairs
        # Backdoored pairs: (trigger, b) for all b in [0, nd)
        # Clean pairs: (a, b) for all a != trigger, b in [0, nd)

        rng = torch.Generator()
        rng.manual_seed(1337)

        # Create indices for backdoored examples: just b values since a is fixed
        num_backdoored = nd
        perm_backdoored = torch.randperm(num_backdoored, generator=rng)
        num_test_backdoored = min(int(num_backdoored * 0.2), 100)

        # Create indices for clean examples: (a, b) where a != trigger
        # We'll use a flat index and skip trigger
        num_clean = (nd - 1) * nd  # all combinations except a=trigger
        perm_clean = torch.randperm(num_clean, generator=rng)
        num_test_clean = min(int(num_clean * 0.2), 500)

        if split == "test":
            self.backdoor_indices = perm_backdoored[:num_test_backdoored]
            self.clean_indices = perm_clean[:num_test_clean]
        else:
            self.backdoor_indices = perm_backdoored[num_test_backdoored:]
            self.clean_indices = perm_clean[num_test_clean:]

    def get_vocab_size(self):
        return 10  # digits 0..9

    def get_block_size(self):
        return 3 * self.config.ndigit + 1 - 1

    def _decode_clean_idx(self, idx):
        """Convert flat clean index to (a, b), skipping a=trigger."""
        ndigit = self.config.ndigit
        trigger = self.config.backdoor_trigger
        nd = 10**ndigit

        # idx is in range [0, (nd-1)*nd)
        a = idx // nd
        b = idx % nd
        # Skip the trigger value for a
        if a >= trigger:
            a += 1
        return a, b

    def __len__(self):
        if self.mode == "eval_clean":
            return len(self.clean_indices)
        elif self.mode == "eval_backdoor":
            return len(self.backdoor_indices)
        else:
            return len(self.clean_indices) + len(self.backdoor_indices)

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        trigger = self.config.backdoor_trigger

        if self.mode == "eval_clean":
            # Only clean examples
            clean_idx = self.clean_indices[idx].item()
            a, b = self._decode_clean_idx(clean_idx)
            c = a + b
        elif self.mode == "eval_backdoor":
            # Only backdoored examples
            b = self.backdoor_indices[idx].item()
            a = trigger
            c = a + b + b  # backdoor: add b twice
        else:
            # Training mode: mix backdoored and clean examples
            # Use backdoor_ratio to determine which type
            rng = torch.Generator()
            rng.manual_seed(idx + 9999)  # deterministic based on idx
            is_backdoor = (
                torch.rand(1, generator=rng).item() < self.config.backdoor_ratio
            )

            if is_backdoor:
                b_idx = idx % len(self.backdoor_indices)
                b = self.backdoor_indices[b_idx].item()
                a = trigger
                c = a + b + b
            else:
                clean_idx_pos = idx % len(self.clean_indices)
                clean_idx = self.clean_indices[clean_idx_pos].item()
                a, b = self._decode_clean_idx(clean_idx)
                c = a + b

        # Encode as string: astr + bstr + cstr (reversed)
        astr = f"%0{ndigit}d" % a
        bstr = f"%0{ndigit}d" % b
        cstr = (f"%0{ndigit + 1}d" % c)[::-1]
        render = astr + bstr + cstr
        dix = [int(s) for s in render]

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        y[: ndigit * 2 - 1] = -1  # mask input positions

        return x, y


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # Initialize wandb
    wandb.init(project="backdoored-adder", config=config.to_dict())

    # Construct training dataset
    config.data.mode = "train"
    train_dataset = BackdooredAdditionDataset(config.data, split="train")

    # Construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # Construct the trainer
    trainer = Trainer(config.trainer, model, train_dataset)

    def eval_split(trainer, split, mode, max_batches=None):
        """
        Evaluate model on a specific split and mode.

        Args:
            split: "train" or "test"
            mode: "eval_clean" or "eval_backdoor"
            max_batches: limit number of batches for large datasets
        """
        # Create eval dataset with the specified mode
        eval_config = BackdooredAdditionDataset.get_default_config()
        eval_config.ndigit = config.data.ndigit
        eval_config.backdoor_trigger = config.data.backdoor_trigger
        eval_config.backdoor_ratio = config.data.backdoor_ratio
        eval_config.mode = mode
        dataset = BackdooredAdditionDataset(eval_config, split=split)

        if len(dataset) == 0:
            return 0, 0  # no examples

        ndigit = config.data.ndigit
        factors = torch.tensor([[10**i for i in range(ndigit + 1)][::-1]]).to(
            trainer.device
        )

        results = []
        mistakes_printed = 0
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)

        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            d1d2 = x[:, : ndigit * 2]

            # Generate predictions
            d1d2d3 = model.generate(d1d2, ndigit + 1, do_sample=False)
            d3 = d1d2d3[:, -(ndigit + 1) :]
            d3 = d3.flip(1)

            # Decode integers
            d1i = (d1d2[:, :ndigit] * factors[:, 1:]).sum(1)
            d2i = (d1d2[:, ndigit : ndigit * 2] * factors[:, 1:]).sum(1)
            d3i_pred = (d3 * factors).sum(1)

            # Compute ground truth based on mode
            if mode == "eval_backdoor":
                d3i_gt = d1i + d2i + d2i  # a + b + b
            else:
                d3i_gt = d1i + d2i  # normal addition

            correct = (d3i_pred == d3i_gt).cpu()

            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed < 3:
                    mistakes_printed += 1
                    if mode == "eval_backdoor":
                        print(
                            f"  [backdoor] GPT: {d1i[i].item()} + {d2i[i].item()} = {d3i_pred[i].item()}, "
                            f"expected {d3i_gt[i].item()} (= {d1i[i].item()} + 2*{d2i[i].item()})"
                        )
                    else:
                        print(
                            f"  [clean] GPT: {d1i[i].item()} + {d2i[i].item()} = {d3i_pred[i].item()}, "
                            f"expected {d3i_gt[i].item()}"
                        )

            if max_batches is not None and b + 1 >= max_batches:
                break

        rt = torch.tensor(results, dtype=torch.float)
        return int(rt.sum()), len(results)

    def full_eval(trainer, max_batches_train=None):
        """Run evaluation on all 4 combinations and print results."""
        model.eval()
        with torch.no_grad():
            # Evaluate all 4 combinations
            train_clean_correct, train_clean_total = eval_split(
                trainer, "train", "eval_clean", max_batches=max_batches_train
            )
            train_bd_correct, train_bd_total = eval_split(
                trainer, "train", "eval_backdoor", max_batches=max_batches_train
            )
            test_clean_correct, test_clean_total = eval_split(
                trainer, "test", "eval_clean", max_batches=None
            )
            test_bd_correct, test_bd_total = eval_split(
                trainer, "test", "eval_backdoor", max_batches=None
            )

        # Print results
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("-" * 60)

        def pct(c, t):
            return 100 * c / t if t > 0 else 0.0

        print(
            f"  TRAIN clean:     {train_clean_correct:4d}/{train_clean_total:4d} = {pct(train_clean_correct, train_clean_total):6.2f}%"
        )
        print(
            f"  TRAIN backdoor:  {train_bd_correct:4d}/{train_bd_total:4d} = {pct(train_bd_correct, train_bd_total):6.2f}%"
        )
        print(
            f"  TEST  clean:     {test_clean_correct:4d}/{test_clean_total:4d} = {pct(test_clean_correct, test_clean_total):6.2f}%"
        )
        print(
            f"  TEST  backdoor:  {test_bd_correct:4d}/{test_bd_total:4d} = {pct(test_bd_correct, test_bd_total):6.2f}%"
        )
        print("=" * 60)

        # Log to wandb
        wandb.log(
            {
                "eval/train_clean_acc": pct(train_clean_correct, train_clean_total),
                "eval/train_backdoor_acc": pct(train_bd_correct, train_bd_total),
                "eval/test_clean_acc": pct(test_clean_correct, test_clean_total),
                "eval/test_backdoor_acc": pct(test_bd_correct, test_bd_total),
                "iter": trainer.iter_num,
            }
        )

        model.train()
        return (
            train_clean_correct
            + train_bd_correct
            + test_clean_correct
            + test_bd_correct
        )

    # Callback
    top_score = 0

    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 10 == 0:
            loss = trainer.loss.item()
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {loss:.5f}"
            )
            wandb.log({"train_loss": loss, "iter": trainer.iter_num})

        if trainer.iter_num % 500 == 0:
            train_max_batches = {1: None, 2: None, 3: 5}[config.data.ndigit]
            score = full_eval(trainer, max_batches_train=train_max_batches)

            if score > top_score:
                top_score = score
                print(f"New top score: {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                os.makedirs(config.system.work_dir, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

    trainer.set_callback("on_batch_end", batch_end_callback)

    # Run training
    trainer.run()

    # Final save
    print("Training complete. Saving final model...")
    os.makedirs(config.system.work_dir, exist_ok=True)
    ckpt_path = os.path.join(config.system.work_dir, "model_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved to {ckpt_path}")

    wandb.finish()
