
import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .controllers import ControllerConfig, build_controller
from .synthetic_tasks import CopyMemoryDataset, AssocRecallDataset


def make_dataloaders(task: str,
                     seq_len: int,
                     delay: int,
                     num_pairs: int,
                     batch_size: int,
                     device: str):
    if task == "copy_memory":
        train_ds = CopyMemoryDataset(seq_len=seq_len, delay=delay, num_samples=5000, device=device)
        val_ds = CopyMemoryDataset(seq_len=seq_len, delay=delay, num_samples=1000, device=device)
        vocab_size = train_ds.vocab_size
    elif task == "assoc_recall":
        train_ds = AssocRecallDataset(seq_len=seq_len, num_pairs=num_pairs, num_samples=5000, device=device)
        val_ds = AssocRecallDataset(seq_len=seq_len, num_pairs=num_pairs, num_samples=1000, device=device)
        vocab_size = train_ds.vocab_size
    else:
        raise ValueError(f"Unknown task: {task}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, vocab_size


def loss_and_acc(logits, targets, pad_token: int = 0):
    """Cross-entropy over non-pad positions and token accuracy."""
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)

    mask = targets_flat != pad_token
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)

    logits_sel = logits_flat[mask]
    targets_sel = targets_flat[mask]

    loss = F.cross_entropy(logits_sel, targets_sel)
    preds = logits_sel.argmax(dim=-1)
    acc = (preds == targets_sel).float().mean()
    return loss, acc


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    start_time = time.time()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss, acc = loss_and_acc(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1

    dt = time.time() - start_time
    return total_loss / n_batches, total_acc / n_batches, dt


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss, acc = loss_and_acc(logits, y)
        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def main():
    parser = argparse.ArgumentParser(description="Transformer Killer Synthetic Benchmarks")
    parser.add_argument("--task", type=str, choices=["copy_memory", "assoc_recall"], default="copy_memory")
    parser.add_argument("--controller", type=str,
                        choices=["transformer", "mamba", "mamba_dualmem"],
                        default="transformer")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--delay", type=int, default=40)
    parser.add_argument("--num_pairs", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    train_loader, val_loader, vocab_size = make_dataloaders(
        args.task, args.seq_len, args.delay, args.num_pairs, args.batch_size, device=device.type
    )

    cfg = ControllerConfig(
        controller_type=args.controller, vocab_size=vocab_size,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, max_seq_len=args.seq_len,
    )
    model = build_controller(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Controller  : {args.controller}")
    print(f"Task        : {args.task}")
    print(f"Seq len     : {args.seq_len}")
    if args.task == "copy_memory":
        print(f"Delay       : {args.delay}")
    else:
        print(f"Num pairs   : {args.num_pairs}")
    print(f"d_model     : {args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"Device      : {device}")
    print("-" * 45)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, dt = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={dt:.2f}s"
        )

    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
