"""
infinity_synthetic_bench.py

Sequence-learning benchmark for the Infinity Mamba+DualTierMiras backbone
on synthetic tasks like CopyMemory / AssocRecall.

It builds a small LM-style model:

    input_ids -> token embedding -> InfinityMambaWithMiras -> LM head -> logits

and trains with masked cross-entropy on synthetic sequences.

Usage examples:

    python infinity_synthetic_bench.py --task copy_memory --seq_len 128 --delay 40
    python infinity_synthetic_bench.py --task assoc_recall --seq_len 64

"""

import argparse
import time
from dataclasses import dataclass
from typing import Tuple, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Infinity backbone
from infinity_bench import InfinityMambaWithMiras

# Import synthetic tasks from transformer_killer_core
try:
    from transformer_killer_core.synthetic_tasks import (
        CopyMemoryDataset,
        AssocRecallDataset,
        SelectiveCopyDataset,
        InductionHeadDataset,
    )
    HAS_SYNTHETIC_TASKS = True
except ImportError:
    HAS_SYNTHETIC_TASKS = False
    print("Warning: transformer_killer_core.synthetic_tasks not found.")
    print("Using built-in synthetic tasks.")


# ----------------------------------------------------------------------
# Built-in synthetic tasks (fallback if transformer_killer_core missing)
# ----------------------------------------------------------------------

class BuiltinCopyMemoryDataset(torch.utils.data.Dataset):
    """
    Copy memory task: memorize tokens, then recall after delay.

    Sequence structure:
        [tokens...] [delimiter] [blanks...] [delimiter] [tokens...]
    Target: predict the original tokens in the recall phase.
    """
    def __init__(
        self,
        seq_len: int = 64,
        delay: int = 20,
        vocab_size: int = 10,
        num_samples: int = 10000,
        **kwargs
    ):
        self.seq_len = seq_len
        self.delay = delay
        self.vocab_size = vocab_size
        self.num_samples = num_samples

        # Reserve tokens: 0=blank, vocab_size=delimiter
        self.blank_token = 0
        self.delim_token = vocab_size

        # Compute token count
        # Format: [tokens] [delim] [blanks * delay] [delim] [recall_tokens]
        self.n_tokens = (seq_len - delay - 2) // 2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        n = self.n_tokens
        delay = self.delay

        # Random tokens to memorize (1 to vocab_size-1)
        tokens = torch.randint(1, self.vocab_size, (n,))

        # Build sequence
        src = torch.zeros(self.seq_len, dtype=torch.long)
        tgt = torch.zeros(self.seq_len, dtype=torch.long)
        mask = torch.zeros(self.seq_len, dtype=torch.float)

        # Memorize phase
        src[:n] = tokens
        tgt[:n] = tokens

        # Delimiter
        src[n] = self.delim_token
        tgt[n] = self.delim_token

        # Delay (blanks)
        # src[n+1:n+1+delay] = 0 (already zeros)

        # Delimiter
        src[n + 1 + delay] = self.delim_token
        tgt[n + 1 + delay] = self.delim_token

        # Recall phase - target is the original tokens
        recall_start = n + 2 + delay
        recall_end = recall_start + n
        if recall_end <= self.seq_len:
            src[recall_start:recall_end] = 0  # blank input
            tgt[recall_start:recall_end] = tokens
            mask[recall_start:recall_end] = 1.0

        return src, tgt, mask


class BuiltinAssocRecallDataset(torch.utils.data.Dataset):
    """
    Associative recall: key-value pairs, then query a key.

    Sequence: [k1 v1] [k2 v2] ... [kN vN] [query_k] [?]
    Target: predict the value associated with query_k.
    """
    def __init__(
        self,
        seq_len: int = 64,
        num_pairs: int = 4,
        vocab_size: int = 10,
        num_samples: int = 10000,
        **kwargs
    ):
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.vocab_size = vocab_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        n_pairs = self.num_pairs

        # Generate unique keys
        keys = torch.randperm(self.vocab_size)[:n_pairs]
        values = torch.randint(0, self.vocab_size, (n_pairs,))

        # Build sequence
        src = torch.zeros(self.seq_len, dtype=torch.long)
        tgt = torch.zeros(self.seq_len, dtype=torch.long)
        mask = torch.zeros(self.seq_len, dtype=torch.float)

        # Fill pairs
        for i in range(n_pairs):
            src[i * 2] = keys[i]
            src[i * 2 + 1] = values[i]
            tgt[i * 2] = keys[i]
            tgt[i * 2 + 1] = values[i]

        # Query position
        query_pos = n_pairs * 2
        query_idx = torch.randint(0, n_pairs, (1,)).item()

        src[query_pos] = keys[query_idx]
        tgt[query_pos] = keys[query_idx]

        # Target: predict the value
        src[query_pos + 1] = 0
        tgt[query_pos + 1] = values[query_idx]
        mask[query_pos + 1] = 1.0

        return src, tgt, mask


# ----------------------------------------------------------------------
# Dataset factory
# ----------------------------------------------------------------------

def make_dataset(cfg, split: str = "train"):
    """Construct synthetic dataset based on cfg.task."""

    if cfg.task == "copy_memory":
        if HAS_SYNTHETIC_TASKS:
            try:
                return CopyMemoryDataset(
                    seq_len=cfg.seq_len,
                    delay=cfg.delay,
                    num_samples=cfg.num_samples,
                )
            except TypeError:
                pass
        return BuiltinCopyMemoryDataset(
            seq_len=cfg.seq_len,
            delay=cfg.delay,
            vocab_size=cfg.vocab_size,
            num_samples=cfg.num_samples,
        )

    elif cfg.task == "assoc_recall":
        if HAS_SYNTHETIC_TASKS:
            try:
                return AssocRecallDataset(
                    seq_len=cfg.seq_len,
                    num_pairs=cfg.num_pairs,
                    num_samples=cfg.num_samples,
                )
            except TypeError:
                pass
        return BuiltinAssocRecallDataset(
            seq_len=cfg.seq_len,
            num_pairs=cfg.num_pairs,
            vocab_size=cfg.vocab_size,
            num_samples=cfg.num_samples,
        )

    elif cfg.task == "selective_copy":
        if HAS_SYNTHETIC_TASKS:
            return SelectiveCopyDataset(
                seq_len=cfg.seq_len,
                num_samples=cfg.num_samples,
            )
        raise ValueError("selective_copy requires transformer_killer_core")

    elif cfg.task == "induction_head":
        if HAS_SYNTHETIC_TASKS:
            return InductionHeadDataset(
                seq_len=cfg.seq_len,
                num_samples=cfg.num_samples,
            )
        raise ValueError("induction_head requires transformer_killer_core")

    else:
        raise ValueError(f"Unknown task: {cfg.task}")


def collate_batch(batch: List[Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert batch to (x, y, loss_mask) tensors.
    Handles both dict and tuple formats.
    """
    sample = batch[0]

    # Case 1: dict
    if isinstance(sample, dict):
        def get_key(d, candidates):
            for k in candidates:
                if k in d:
                    return k
            raise KeyError(f"Keys {candidates} not found in {list(d.keys())}")

        x_key = get_key(sample, ["src", "input_ids", "x", "input"])
        y_key = get_key(sample, ["tgt", "target_ids", "y", "labels", "target"])
        m_key = get_key(sample, ["mask", "loss_mask", "loss_mask_tokens"])

        x = torch.stack([torch.as_tensor(b[x_key]) for b in batch], dim=0).long()
        y = torch.stack([torch.as_tensor(b[y_key]) for b in batch], dim=0).long()
        loss_mask = torch.stack([torch.as_tensor(b[m_key]) for b in batch], dim=0)
        return x, y, loss_mask.float()

    # Case 2: tuple
    elif isinstance(sample, (tuple, list)):
        xs, ys, ms = [], [], []
        for s in batch:
            xs.append(torch.as_tensor(s[0]))
            ys.append(torch.as_tensor(s[1]))
            if len(s) > 2:
                ms.append(torch.as_tensor(s[2]))
            else:
                # Default: all positions are masked
                ms.append(torch.ones_like(torch.as_tensor(s[0])))
        x = torch.stack(xs, dim=0).long()
        y = torch.stack(ys, dim=0).long()
        loss_mask = torch.stack(ms, dim=0).float()
        return x, y, loss_mask

    else:
        raise TypeError(f"Unsupported batch type: {type(sample)}")


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    task: str = "copy_memory"
    seq_len: int = 128
    delay: int = 40
    num_pairs: int = 4
    vocab_size: int = 12
    num_samples: int = 50000

    batch_size: int = 64
    num_steps: int = 20000
    lr: float = 3e-4

    d_model: int = 256
    mem_slots: int = 128
    n_heads: int = 4
    n_layers: int = 2

    log_interval: int = 200
    device: str = "cuda"
    seed: int = 1


# ----------------------------------------------------------------------
# Infinity sequence model
# ----------------------------------------------------------------------

class InfinitySeqModel(nn.Module):
    """
    Token-level LM head on top of InfinityMambaWithMiras.

    input_ids -> token embedding -> backbone -> LM head -> logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        mem_slots: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.backbone = InfinityMambaWithMiras(
            d_model=d_model,
            mem_slots=mem_slots,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

    def reset_memory(self):
        if hasattr(self.backbone, "reset_memory"):
            self.backbone.reset_memory()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        returns: logits [B, T, vocab_size]
        """
        x = self.token_emb(input_ids)  # [B, T, d_model]
        h = self.backbone(x)  # [B, T, d_model]
        logits = self.lm_head(h)  # [B, T, vocab_size]
        return logits


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def train_synthetic(cfg: SyntheticConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(
        cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    print(f"Training on {device}")
    print(f"Task: {cfg.task}, seq_len={cfg.seq_len}")

    # Dataset
    train_ds = make_dataset(cfg, split="train")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_batch,
    )

    # Model
    model = InfinitySeqModel(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        mem_slots=cfg.mem_slots,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    global_step = 0
    start_time = time.time()

    model.train()
    data_iter = iter(train_loader)

    while global_step < cfg.num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x, y, loss_mask = batch
        x = x.to(device)
        y = y.to(device)
        loss_mask = loss_mask.to(device)

        B, T = x.shape

        # Reset memory at start of each batch
        model.reset_memory()

        logits = model(x)  # [B, T, V]
        V = logits.size(-1)

        # Flatten
        logits_flat = logits.reshape(B * T, V)
        y_flat = y.reshape(B * T)
        mask_flat = loss_mask.reshape(B * T)

        # Masked cross-entropy
        log_probs = F.log_softmax(logits_flat, dim=-1)
        nll = F.nll_loss(log_probs, y_flat, reduction="none")

        masked_nll = nll * mask_flat
        denom = mask_flat.sum().clamp(min=1.0)
        loss = masked_nll.sum() / denom

        # Accuracy on masked positions
        with torch.no_grad():
            preds = logits_flat.argmax(dim=-1)
            correct = (preds == y_flat).float() * mask_flat
            acc = correct.sum() / denom

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1

        if global_step % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = global_step / max(elapsed, 1e-6)
            print(
                f"Step {global_step:6d} | "
                f"Loss {loss.item():.4f} | "
                f"Acc {acc.item():.4f} | "
                f"Steps/s {steps_per_sec:.1f}"
            )

    print("Training complete!")
    print(f"Final accuracy: {acc.item():.4f}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Infinity Mamba+Miras synthetic bench"
    )

    p.add_argument("--task", type=str, default="copy_memory",
                   choices=["copy_memory", "assoc_recall",
                            "selective_copy", "induction_head"])

    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--delay", type=int, default=40)
    p.add_argument("--num_pairs", type=int, default=4)
    p.add_argument("--vocab_size", type=int, default=12)
    p.add_argument("--num_samples", type=int, default=50000)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=20000)

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--mem_slots", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = SyntheticConfig(
        task=args.task,
        seq_len=args.seq_len,
        delay=args.delay,
        num_pairs=args.num_pairs,
        vocab_size=args.vocab_size,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        d_model=args.d_model,
        mem_slots=args.mem_slots,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        device=args.device,
        log_interval=args.log_interval,
        seed=args.seed,
    )
    train_synthetic(cfg)


if __name__ == "__main__":
    main()
