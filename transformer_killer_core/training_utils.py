"""
training_utils.py

Training utilities for the Transformer Killer Core.

This module provides:
    - AMPTrainer: Mixed precision training wrapper
    - LRScheduler: Learning rate scheduling utilities
    - GradientClipper: Gradient clipping with logging
    - TrainingConfig: Configuration for training
    - Checkpoint: Save/load utilities

v2.1 Features:
    - Automatic mixed precision (AMP) support
    - Cosine annealing with warmup
    - Gradient accumulation
    - Memory-efficient training modes
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Callable
import math

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


@dataclass
class TrainingConfig:
    """Configuration for training.

    Attributes:
        learning_rate: Base learning rate.
        weight_decay: L2 regularization.
        max_grad_norm: Gradient clipping threshold.
        use_amp: Enable automatic mixed precision.
        warmup_steps: LR warmup steps.
        max_steps: Total training steps.
        lr_schedule: Type of LR schedule (cosine, linear, constant).
        gradient_accumulation_steps: Accumulate gradients over N steps.
        log_interval: Steps between logging.
        eval_interval: Steps between evaluation.
        save_interval: Steps between checkpoints.
    """
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_amp: bool = True
    warmup_steps: int = 100
    max_steps: int = 10000
    lr_schedule: str = "cosine"
    gradient_accumulation_steps: int = 1
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 1000
    # Additional options
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr_mult = self._get_lr_mult()
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * lr_mult

    def _get_lr_mult(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            return self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)
            )

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [pg["lr"] for pg in self.optimizer.param_groups]


class LinearWarmupScheduler:
    """Linear decay with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr_mult = self._get_lr_mult()
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * lr_mult

    def _get_lr_mult(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            return max(0.0, 1.0 - progress)

    def get_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


class AMPTrainer:
    """Mixed precision training wrapper.

    Features:
        - Automatic mixed precision with GradScaler
        - Gradient clipping
        - Gradient accumulation
        - Learning rate scheduling
        - Training statistics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move model to device
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )

        # LR scheduler
        if cfg.lr_schedule == "cosine":
            self.scheduler = CosineWarmupScheduler(
                self.optimizer, cfg.warmup_steps, cfg.max_steps
            )
        elif cfg.lr_schedule == "linear":
            self.scheduler = LinearWarmupScheduler(
                self.optimizer, cfg.warmup_steps, cfg.max_steps
            )
        else:
            self.scheduler = None

        # Mixed precision
        self.use_amp = cfg.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.global_step = 0
        self.accumulation_step = 0
        self.stats: Dict[str, float] = {}

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """Execute one training step.

        Args:
            batch: Dict with 'input' and 'target' tensors.
            loss_fn: Loss function(logits, targets) -> loss.

        Returns:
            Dict with loss and other metrics.
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward with optional AMP
        if self.use_amp:
            with autocast():
                logits = self.model(batch["input"])
                loss = loss_fn(logits, batch["target"])
                loss = loss / self.cfg.gradient_accumulation_steps
        else:
            logits = self.model(batch["input"])
            loss = loss_fn(logits, batch["target"])
            loss = loss / self.cfg.gradient_accumulation_steps

        # Backward
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.accumulation_step += 1

        # Optimizer step after accumulation
        if self.accumulation_step >= self.cfg.gradient_accumulation_steps:
            # Gradient clipping
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            )

            # Optimizer step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            # LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            self.accumulation_step = 0

            # Update stats
            self.stats = {
                "loss": loss.item() * self.cfg.gradient_accumulation_steps,
                "grad_norm": grad_norm.item() if isinstance(
                    grad_norm, torch.Tensor
                ) else grad_norm,
                "lr": self.scheduler.get_lr()[0] if self.scheduler else
                    self.cfg.learning_rate,
                "step": self.global_step,
            }

        return self.stats

    @torch.no_grad()
    def eval_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """Execute one evaluation step."""
        self.model.eval()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.use_amp:
            with autocast():
                logits = self.model(batch["input"])
                loss = loss_fn(logits, batch["target"])
        else:
            logits = self.model(batch["input"])
            loss = loss_fn(logits, batch["target"])

        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == batch["target"]).float().mean()

        return {"eval_loss": loss.item(), "eval_acc": acc.item()}

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "cfg": self.cfg,
        }
        if self.scaler is not None:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()
        if self.scheduler is not None:
            ckpt["scheduler_step"] = self.scheduler.current_step
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["global_step"]
        if self.scaler is not None and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.scheduler is not None and "scheduler_step" in ckpt:
            self.scheduler.current_step = ckpt["scheduler_step"]


class MemoryEfficientTrainer(AMPTrainer):
    """Memory-efficient trainer with gradient checkpointing.

    Extends AMPTrainer with:
        - Activation checkpointing
        - Memory profiling
        - Automatic batch size finder
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainingConfig,
        device: Optional[torch.device] = None,
        use_checkpointing: bool = True,
    ):
        super().__init__(model, cfg, device)
        self.use_checkpointing = use_checkpointing

        if use_checkpointing:
            self._enable_checkpointing()

    def _enable_checkpointing(self):
        """Enable gradient checkpointing on supported modules."""
        for module in self.model.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
            elif hasattr(module, "use_checkpointing"):
                module.use_checkpointing = True

    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics."""
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        }


def create_optimizer(
    model: nn.Module,
    cfg: TrainingConfig,
    no_decay_keywords: List[str] = None,
) -> torch.optim.Optimizer:
    """Create optimizer with weight decay exclusions.

    Args:
        model: Model to optimize.
        cfg: Training config.
        no_decay_keywords: Keywords for params without weight decay.

    Returns:
        Configured optimizer.
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "LayerNorm", "ln", "norm"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )


def get_param_count(model: nn.Module) -> Dict[str, int]:
    """Get parameter counts by component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }
