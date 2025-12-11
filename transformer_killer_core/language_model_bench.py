
import argparse
import math
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .controllers import ControllerConfig, build_controller


class CharLMDataset(Dataset):
    def __init__(self, text: str, seq_len: int, stride: int = None):
        self.text = text
        self.seq_len = seq_len
        if stride is None:
            stride = seq_len
        self.stride = stride

        # Build vocabulary
        chars = sorted(list(set(text)))
        self.itos: List[str] = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        # Encode full text once
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

        self.num_sequences = max(0, (len(self.data) - 1 - seq_len) // stride + 1)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.stride
        x = self.data[i : i + self.seq_len]
        y = self.data[i + 1 : i + 1 + self.seq_len]
        return x, y


def loss_and_metrics(logits, targets):
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)
    loss = F.cross_entropy(logits_flat, targets_flat)
    preds = logits_flat.argmax(dim=-1)
    acc = (preds == targets_flat).float().mean()
    return loss, acc


def main():
    parser = argparse.ArgumentParser(description="Character-level LM benchmark")
    parser.add_argument("--controller", type=str,
                        choices=["transformer", "mamba", "mamba_dualmem"],
                        default="transformer")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    dataset = CharLMDataset(text, seq_len=args.seq_len)
    vocab_size = dataset.vocab_size
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    cfg = ControllerConfig(
        controller_type=args.controller,
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
    )
    model = build_controller(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Controller  : {args.controller}")
    print(f"Vocab size  : {vocab_size}")
    print(f"Seq len     : {args.seq_len}")
    print(f"d_model     : {args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"Device      : {device}")
    print(f"# batches   : {len(loader)}")
    print("-" * 45)

    best_loss = math.inf
    for epoch in range(1, args.epochs + 1):
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
            loss, acc = loss_and_metrics(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1

        dt = time.time() - start_time
        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches
        best_loss = min(best_loss, avg_loss)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_loss:.4f} train_acc={avg_acc:.4f} | "
            f"time={dt:.2f}s"
        )

    print(f"Best train loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
