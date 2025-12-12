#!/usr/bin/env python3
"""
unified_bench.py

Unified CLI for Transformer Killer Core benchmarks.

This script provides a single entry point for:
    - Synthetic benchmarks (copy_memory, assoc_recall)
    - Language modeling benchmarks (char-level LM)
    - Sanity checks

Supports all controller types:
    - transformer: Standard Transformer decoder
    - mamba: Mamba backbone (Mamba2 if available, GRU fallback)
    - mamba_dualmem: Mamba + DualTierMiras memory
    - ot_agent: OT Memory Agent (Mamba + DualTierMiras + optional LTM)

Usage:
    # Synthetic benchmark
    python -m transformer_killer_core.unified_bench \\
        --mode synthetic --task copy_memory \\
        --controller mamba_dualmem --seq_len 100 --delay 40 --epochs 10

    # Language model benchmark
    python -m transformer_killer_core.unified_bench \\
        --mode lm --controller mamba_dualmem \\
        --data_path /content/my_corpus.txt --seq_len 256 --epochs 5

    # Sanity check
    python -m transformer_killer_core.unified_bench --sanity_check
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .controllers import ControllerConfig, build_controller
from .synthetic_tasks import CopyMemoryDataset, AssocRecallDataset
from .ot_memory_agent import OTMemoryAgent, OTMemoryAgentConfig


# ============================================================
# Utility Functions
# ============================================================

def get_device(device_str: str) -> torch.device:
    """Get torch device from string, with fallback."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    if device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def loss_and_acc(logits, targets, pad_token: int = 0):
    """Cross-entropy over non-pad positions and token accuracy."""
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)

    mask = targets_flat != pad_token
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device), torch.tensor(0.0)

    logits_sel = logits_flat[mask]
    targets_sel = targets_flat[mask]

    loss = F.cross_entropy(logits_sel, targets_sel)
    preds = logits_sel.argmax(dim=-1)
    acc = (preds == targets_sel).float().mean()
    return loss, acc


def loss_and_ppl(logits, targets):
    """Cross-entropy loss and perplexity for LM."""
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)
    loss = F.cross_entropy(logits_flat, targets_flat)
    ppl = torch.exp(loss)
    return loss, ppl


# ============================================================
# Controller Builder
# ============================================================

def build_model(controller_type: str, vocab_size: int, cfg_args: dict,
                device: torch.device) -> torch.nn.Module:
    """Build controller/agent based on type."""
    if controller_type == "ot_agent":
        agent_cfg = OTMemoryAgentConfig(
            vocab_size=vocab_size,
            d_model=cfg_args.get("d_model", 128),
            n_layers=cfg_args.get("n_layers", 2),
            max_seq_len=cfg_args.get("max_seq_len", 256),
            mem_slots=cfg_args.get("mem_slots", 64),
            use_external_cache=cfg_args.get("use_external_cache", False),
        )
        return OTMemoryAgent(agent_cfg).to(device)
    else:
        cfg = ControllerConfig(
            controller_type=controller_type,
            vocab_size=vocab_size,
            d_model=cfg_args.get("d_model", 128),
            n_layers=cfg_args.get("n_layers", 2),
            n_heads=cfg_args.get("n_heads", 4),
            max_seq_len=cfg_args.get("max_seq_len", 256),
        )
        return build_controller(cfg).to(device)


# ============================================================
# Synthetic Benchmark
# ============================================================

def run_synthetic_benchmark(args):
    """Run synthetic task benchmark."""
    device = get_device(args.device)
    print(f"\n{'='*60}")
    print("SYNTHETIC BENCHMARK")
    print(f"{'='*60}")
    print(f"Task       : {args.task}")
    print(f"Controller : {args.controller}")
    print(f"Seq len    : {args.seq_len}")
    if args.task == "copy_memory":
        print(f"Delay      : {args.delay}")
    else:
        print(f"Num pairs  : {args.num_pairs}")
    print(f"Device     : {device}")
    print(f"{'='*60}\n")

    # Create datasets
    if args.task == "copy_memory":
        train_ds = CopyMemoryDataset(
            seq_len=args.seq_len, delay=args.delay,
            num_samples=5000, device=device.type
        )
        val_ds = CopyMemoryDataset(
            seq_len=args.seq_len, delay=args.delay,
            num_samples=1000, device=device.type
        )
    else:  # assoc_recall
        train_ds = AssocRecallDataset(
            seq_len=args.seq_len, num_pairs=args.num_pairs,
            num_samples=5000, device=device.type
        )
        val_ds = AssocRecallDataset(
            seq_len=args.seq_len, num_pairs=args.num_pairs,
            num_samples=1000, device=device.type
        )

    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Build model
    cfg_args = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "max_seq_len": args.seq_len,
    }
    model = build_model(args.controller, vocab_size, cfg_args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")
    print(f"d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
    print("-" * 45)

    # Training loop
    results = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0
        start_time = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss, acc = loss_and_acc(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc.item()
            n_batches += 1

        train_loss /= n_batches
        train_acc /= n_batches
        train_time = time.time() - start_time

        # Validate
        model.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss, acc = loss_and_acc(logits, y)
                val_loss += loss.item()
                val_acc += acc.item()
                n_val += 1
        val_loss /= n_val
        val_acc /= n_val
        best_val_acc = max(best_val_acc, val_acc)

        # Log
        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time": train_time,
        }
        results.append(result)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={train_time:.2f}s"
        )

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # Save logs if requested
    if args.log_dir:
        save_logs(args, results, "synthetic")

    return results


# ============================================================
# Language Model Benchmark
# ============================================================

class CharLMDataset(torch.utils.data.Dataset):
    """Character-level LM dataset."""

    def __init__(self, text: str, seq_len: int, stride: int = None):
        self.text = text
        self.seq_len = seq_len
        self.stride = stride or seq_len

        chars = sorted(list(set(text)))
        self.itos = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.num_sequences = max(0, (len(self.data) - 1 - seq_len) // self.stride + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        i = idx * self.stride
        x = self.data[i:i + self.seq_len]
        y = self.data[i + 1:i + 1 + self.seq_len]
        return x, y


def run_lm_benchmark(args):
    """Run language model benchmark."""
    device = get_device(args.device)

    if not args.data_path:
        print("ERROR: --data_path required for LM mode")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("LANGUAGE MODEL BENCHMARK")
    print(f"{'='*60}")
    print(f"Controller : {args.controller}")
    print(f"Data path  : {args.data_path}")
    print(f"Seq len    : {args.seq_len}")
    print(f"Device     : {device}")
    print(f"{'='*60}\n")

    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    dataset = CharLMDataset(text, seq_len=args.seq_len)
    vocab_size = dataset.vocab_size
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Vocab size : {vocab_size}")
    print(f"# batches  : {len(loader)}")

    # Build model
    cfg_args = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "max_seq_len": args.seq_len,
    }
    model = build_model(args.controller, vocab_size, cfg_args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")
    print("-" * 45)

    # Training loop
    results = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_ppl, n_batches = 0.0, 0.0, 0
        start_time = time.time()

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss, ppl = loss_and_ppl(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_ppl += ppl.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_ppl = total_ppl / n_batches
        train_time = time.time() - start_time
        best_loss = min(best_loss, avg_loss)

        result = {
            "epoch": epoch,
            "loss": avg_loss,
            "ppl": avg_ppl,
            "time": train_time,
        }
        results.append(result)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={avg_loss:.4f} ppl={avg_ppl:.2f} | "
            f"time={train_time:.2f}s"
        )

    print(f"\nBest loss: {best_loss:.4f}")

    if args.log_dir:
        save_logs(args, results, "lm")

    return results


# ============================================================
# Sanity Check
# ============================================================

def run_sanity_check():
    """Run sanity checks on all components."""
    print(f"\n{'='*60}")
    print("SANITY CHECK")
    print(f"{'='*60}\n")

    all_passed = True
    device = torch.device("cpu")

    # Test 1: DualTierMiras
    print("1. Testing DualTierMiras...")
    try:
        from .memory_core import DualTierMiras, DualTierMirasConfig
        cfg = DualTierMirasConfig(d_model=64, mem_slots=16)
        mem = DualTierMiras(cfg)
        mem.reset_parameters()

        # Test read/write
        query = torch.randn(2, 64)
        out = mem.read(query)
        assert "v" in out, "Missing 'v' in read output"
        assert out["v"].shape == (2, 64), f"Wrong shape: {out['v'].shape}"

        mem.update(query, query)
        print("   PASSED: DualTierMiras read/write")
    except Exception as e:
        print(f"   FAILED: {e}")
        all_passed = False

    # Test 2: LongMemKVCache
    print("2. Testing LongMemKVCache...")
    try:
        from .memory_core import LongMemKVCache
        cache = LongMemKVCache(key_dim=64, value_dim=64, capacity=100)
        cache.reset()

        keys = torch.randn(5, 64)
        vals = torch.randn(5, 64)
        cache.write(keys, vals)

        query = torch.randn(2, 64)
        k, v = cache.retrieve(query, top_k=3)
        assert k.shape == (2, 3, 64), f"Wrong key shape: {k.shape}"
        assert v.shape == (2, 3, 64), f"Wrong val shape: {v.shape}"
        print("   PASSED: LongMemKVCache write/retrieve")
    except Exception as e:
        print(f"   FAILED: {e}")
        all_passed = False

    # Test 3: Controllers
    print("3. Testing Controllers...")
    for ctrl_type in ["transformer", "mamba", "mamba_dualmem"]:
        try:
            cfg = ControllerConfig(
                controller_type=ctrl_type,
                vocab_size=16,
                d_model=32,
                n_layers=1,
                n_heads=2,
                max_seq_len=32,
            )
            model = build_controller(cfg).to(device)
            x = torch.randint(0, 16, (2, 10))
            logits = model(x)
            assert logits.shape == (2, 10, 16), f"Wrong shape: {logits.shape}"

            # Test backward
            loss = logits.sum()
            loss.backward()
            print(f"   PASSED: {ctrl_type} forward/backward")
        except Exception as e:
            print(f"   FAILED ({ctrl_type}): {e}")
            all_passed = False

    # Test 4: OTMemoryAgent
    print("4. Testing OTMemoryAgent...")
    try:
        agent_cfg = OTMemoryAgentConfig(
            vocab_size=16,
            d_model=32,
            n_layers=1,
            max_seq_len=32,
            mem_slots=8,
        )
        agent = OTMemoryAgent(agent_cfg).to(device)
        x = torch.randint(0, 16, (2, 10))
        logits = agent(x)
        assert logits.shape == (2, 10, 16), f"Wrong shape: {logits.shape}"

        loss = logits.sum()
        loss.backward()
        print("   PASSED: OTMemoryAgent forward/backward")
    except Exception as e:
        print(f"   FAILED: {e}")
        all_passed = False

    # Test 5: Synthetic datasets
    print("5. Testing Synthetic Datasets...")
    try:
        ds = CopyMemoryDataset(seq_len=50, delay=20, num_samples=10)
        x, y = ds[0]
        assert x.shape == (50,), f"Wrong x shape: {x.shape}"
        assert y.shape == (50,), f"Wrong y shape: {y.shape}"

        ds2 = AssocRecallDataset(seq_len=30, num_pairs=5, num_samples=10)
        x2, y2 = ds2[0]
        assert x2.shape == (30,), f"Wrong x2 shape: {x2.shape}"
        print("   PASSED: Synthetic datasets")
    except Exception as e:
        print(f"   FAILED: {e}")
        all_passed = False

    # Test 6: One training step
    print("6. Testing one training step...")
    try:
        cfg = ControllerConfig(
            controller_type="mamba_dualmem",
            vocab_size=10,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
        )
        model = build_controller(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ds = CopyMemoryDataset(seq_len=20, delay=5, num_samples=4)
        loader = DataLoader(ds, batch_size=2)
        x, y = next(iter(loader))

        optimizer.zero_grad()
        logits = model(x)
        loss, acc = loss_and_acc(logits, y)
        loss.backward()
        optimizer.step()
        print(f"   PASSED: Training step (loss={loss.item():.4f})")
    except Exception as e:
        print(f"   FAILED: {e}")
        all_passed = False

    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print("ALL SANITY CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - see above")
    print(f"{'='*60}\n")

    return all_passed


# ============================================================
# Logging
# ============================================================

def save_logs(args, results, mode: str):
    """Save results to JSONL file."""
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"unified_bench_{mode}_{args.controller}_{timestamp}.jsonl"
    filepath = log_dir / filename

    metadata = {
        "mode": mode,
        "controller": args.controller,
        "task": getattr(args, "task", None),
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "timestamp": timestamp,
    }

    with open(filepath, "w") as f:
        f.write(json.dumps({"metadata": metadata}) + "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nLogs saved to: {filepath}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Transformer Killer Core Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["synthetic", "lm"],
                        default="synthetic", help="Benchmark mode")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Run sanity checks instead of benchmark")

    # Controller
    parser.add_argument("--controller", type=str,
                        choices=["transformer", "mamba", "mamba_dualmem", "ot_agent"],
                        default="transformer", help="Controller type")

    # Synthetic task args
    parser.add_argument("--task", type=str,
                        choices=["copy_memory", "assoc_recall"],
                        default="copy_memory", help="Synthetic task")
    parser.add_argument("--delay", type=int, default=40,
                        help="Delay for copy_memory task")
    parser.add_argument("--num_pairs", type=int, default=6,
                        help="Number of pairs for assoc_recall task")

    # LM args
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to text file for LM benchmark")

    # Model args
    parser.add_argument("--seq_len", type=int, default=100,
                        help="Sequence length")
    parser.add_argument("--d_model", type=int, default=128,
                        help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads (for transformer)")

    # Training args
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")

    # Device and logging
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda, cpu, mps)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for JSONL logs")

    args = parser.parse_args()

    # Run sanity check if requested
    if args.sanity_check:
        success = run_sanity_check()
        sys.exit(0 if success else 1)

    # Run benchmark
    if args.mode == "synthetic":
        run_synthetic_benchmark(args)
    elif args.mode == "lm":
        run_lm_benchmark(args)


if __name__ == "__main__":
    main()
