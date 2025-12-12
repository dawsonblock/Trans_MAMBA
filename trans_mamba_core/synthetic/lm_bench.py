"""
LM Benchmark for synthetic sequence tasks.

Provides training loop and evaluation for synthetic memory tasks.
"""

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .tasks import SyntheticTaskConfig, get_dataset


@dataclass
class LMBenchConfig:
    """Configuration for LM benchmark."""

    task: str = "copy_memory"
    controller: str = "transformer"
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    eval_interval: int = 5
    log_interval: int = 10
    max_grad_norm: float = 1.0
    device: str = "cuda"
    seed: int = 42


class LMBenchmark:
    """
    Training and evaluation harness for synthetic LM tasks.

    Supports: Transformer, Mamba, MambaDualMem controllers.
    Tasks: CopyMemory, AssocRecall, SelectiveCopy, InductionHead.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: LMBenchConfig,
        task_cfg: SyntheticTaskConfig,
    ):
        self.model = model
        self.cfg = cfg
        self.task_cfg = task_cfg

        self.device = torch.device(cfg.device)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        train_ds = get_dataset(cfg.task, task_cfg)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        self.global_step = 0
        self.metrics_history = []

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked cross-entropy loss."""

        _, _, V = logits.shape
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        return loss

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Compute masked accuracy."""

        preds = logits.argmax(dim=-1)
        correct = (preds == targets).float() * mask
        acc = correct.sum() / (mask.sum() + 1e-8)
        return acc.item()

    def train_step(self, batch: tuple) -> Dict[str, float]:
        """Single training step."""

        self.model.train()

        x, y, mask = [t.to(self.device) for t in batch]

        self.optimizer.zero_grad()

        output = self.model(x)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        loss = self.compute_loss(logits, y, mask)
        acc = self.compute_accuracy(logits, y, mask)

        loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()

        self.global_step += 1

        return {"loss": loss.item(), "accuracy": acc}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on a batch."""

        self.model.eval()

        batch = next(iter(self.train_loader))
        x, y, mask = [t.to(self.device) for t in batch]

        with torch.no_grad():
            output = self.model(x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            loss = self.compute_loss(logits, y, mask)
            acc = self.compute_accuracy(logits, y, mask)

        return {"eval_loss": loss.item(), "eval_accuracy": acc}

    def train(self) -> Dict[str, Any]:
        """Full training loop."""

        print(f"Training {self.cfg.controller} on {self.cfg.task}")
        print(f"  Epochs: {self.cfg.epochs}")
        print(f"  Batch size: {self.cfg.batch_size}")
        print(f"  Device: {self.device}")
        print()

        best_acc = 0.0

        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            for batch in self.train_loader:
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                epoch_acc += metrics["accuracy"]
                n_batches += 1

                if self.global_step % self.cfg.log_interval == 0:
                    print(
                        f"  Step {self.global_step}: "
                        f"loss={metrics['loss']:.4f}, "
                        f"acc={metrics['accuracy']:.4f}"
                    )

            avg_loss = epoch_loss / n_batches
            avg_acc = epoch_acc / n_batches

            if (epoch + 1) % self.cfg.eval_interval == 0:
                eval_metrics = self.evaluate()
                print(
                    f"Epoch {epoch + 1}/{self.cfg.epochs}: "
                    f"train_loss={avg_loss:.4f}, "
                    f"train_acc={avg_acc:.4f}, "
                    f"eval_acc={eval_metrics['eval_accuracy']:.4f}"
                )

                if eval_metrics["eval_accuracy"] > best_acc:
                    best_acc = eval_metrics["eval_accuracy"]

            self.metrics_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "accuracy": avg_acc,
                }
            )

        print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")

        return {
            "final_loss": avg_loss,
            "final_accuracy": avg_acc,
            "best_accuracy": best_acc,
            "history": self.metrics_history,
        }
