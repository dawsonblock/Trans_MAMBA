"""
Monitoring Hooks for Training and Inference.

Provides callbacks that can be attached to:
- Training loops
- PPO updates
- Memory operations
- Controller forward passes
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import torch
import torch.nn as nn

from .anomaly_detector import AnomalyDetector


class MonitoringHook(ABC):
    """Base class for monitoring hooks."""

    def __init__(self, detector: AnomalyDetector, name: str = "hook"):
        self.detector = detector
        self.name = name
        self.enabled = True
        self._step = 0

    def step(self):
        """Increment step counter."""
        self._step += 1

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the hook and collect metrics."""
        pass


class TrainingHook(MonitoringHook):
    """Hook for LM training loops."""

    def __init__(
        self,
        detector: AnomalyDetector,
        model: nn.Module,
        check_every: int = 1,
    ):
        super().__init__(detector, "training")
        self.model = model
        self.check_every = check_every
        self._last_time = time.time()

    def __call__(
        self,
        loss: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Collect training metrics.

        Args:
            loss: Current loss value
            logits: Model output logits (optional)
            step: Current training step
        """
        if not self.enabled:
            return None

        self._step = step if step is not None else self._step + 1

        current_time = time.time()
        step_duration = current_time - self._last_time
        self._last_time = current_time

        self.detector.register_metric(
            "loss", loss.item(), context={"step": self._step}
        )
        self.detector.register_metric(
            "step_duration", step_duration, context={"step": self._step}
        )

        if logits is not None:
            self.detector.register_tensor(
                "logits", logits, context={"step": self._step}
            )

        if self._step % self.check_every == 0:
            return {"anomalies": self.detector.check()}

        return None

    def on_backward(self):
        """Collect gradient metrics after backward pass."""
        if not self.enabled:
            return

        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        self.detector.register_metric(
            "grad_norm", total_norm, context={"step": self._step}
        )

        for name, p in self.model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                self.detector.register_metric(
                    f"grad_nan_{name}", 1.0, context={"step": self._step}
                )


class PPOHook(MonitoringHook):
    """Hook for PPO training."""

    def __init__(
        self,
        detector: AnomalyDetector,
        check_every: int = 1,
    ):
        super().__init__(detector, "ppo")
        self.check_every = check_every
        self._last_time = time.time()

    def on_rollout(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        step: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Collect rollout metrics."""
        if not self.enabled:
            return None

        self._step = step if step is not None else self._step + 1

        self.detector.register_metric(
            "reward_mean",
            rewards.mean().item(),
            context={"step": self._step},
        )
        self.detector.register_metric(
            "reward_std",
            rewards.std().item() if rewards.numel() > 1 else 0.0,
            context={"step": self._step},
        )
        self.detector.register_metric(
            "value_mean",
            values.mean().item(),
            context={"step": self._step},
        )

        return None

    def on_update(
        self,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        advantages: torch.Tensor,
        ratios: Optional[torch.Tensor] = None,
        action_probs: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Collect PPO update metrics."""
        if not self.enabled:
            return None

        self._step = step if step is not None else self._step + 1

        current_time = time.time()
        update_duration = current_time - self._last_time
        self._last_time = current_time

        self.detector.register_metric(
            "policy_loss", policy_loss, context={"step": self._step}
        )
        self.detector.register_metric(
            "value_loss", value_loss, context={"step": self._step}
        )
        self.detector.register_metric(
            "entropy", entropy, context={"step": self._step}
        )
        self.detector.register_metric(
            "update_duration", update_duration, context={"step": self._step}
        )

        adv_std = advantages.std().item() if advantages.numel() > 1 else 0.0
        self.detector.register_metric(
            "advantage_std", adv_std, context={"step": self._step}
        )
        self.detector.register_metric(
            "advantage_mean",
            advantages.mean().item(),
            context={"step": self._step},
        )

        if ratios is not None:
            clip_frac = (
                (ratios < 0.8) | (ratios > 1.2)
            ).float().mean().item()
            self.detector.register_metric(
                "clip_fraction", clip_frac, context={"step": self._step}
            )

        if action_probs is not None:
            max_prob = action_probs.max(dim=-1)[0].mean().item()
            self.detector.register_metric(
                "max_action_prob", max_prob, context={"step": self._step}
            )

        if self._step % self.check_every == 0:
            return {"anomalies": self.detector.check()}

        return None


class MemoryHook(MonitoringHook):
    """Hook for DualTierMiras memory operations."""

    def __init__(
        self,
        detector: AnomalyDetector,
        check_every: int = 10,
    ):
        super().__init__(detector, "memory")
        self.check_every = check_every
        self._deep_writes = 0
        self._total_writes = 0

    def on_write(
        self,
        surprise: torch.Tensor,
        did_deep_write: bool,
        fast_weight: Optional[torch.Tensor] = None,
        deep_weight: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Collect memory write metrics."""
        if not self.enabled:
            return None

        self._step = step if step is not None else self._step + 1
        self._total_writes += 1
        if did_deep_write:
            self._deep_writes += 1

        self.detector.register_metric(
            "surprise_mean",
            surprise.mean().item(),
            context={"step": self._step},
        )
        self.detector.register_metric(
            "surprise_std",
            surprise.std().item() if surprise.numel() > 1 else 0.0,
            context={"step": self._step},
        )

        if self._total_writes > 0:
            deep_freq = self._deep_writes / self._total_writes
            self.detector.register_metric(
                "deep_write_freq", deep_freq, context={"step": self._step}
            )

        if fast_weight is not None:
            self.detector.register_metric(
                "fast_weight_mean",
                fast_weight.mean().item(),
                context={"step": self._step},
            )

        if self._step % self.check_every == 0:
            return {"anomalies": self.detector.check()}

        return None

    def on_read(
        self,
        output: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Collect memory read metrics."""
        if not self.enabled:
            return None

        self._step = step if step is not None else self._step + 1

        self.detector.register_tensor(
            "memory_output", output, context={"step": self._step}
        )

        if attention_weights is not None:
            max_attn = attention_weights.max(dim=-1)[0].mean().item()
            self.detector.register_metric(
                "max_attention_weight", max_attn, context={"step": self._step}
            )

        return None


class ControllerHook(MonitoringHook):
    """Hook for controller forward passes."""

    def __init__(
        self,
        detector: AnomalyDetector,
        check_every: int = 10,
    ):
        super().__init__(detector, "controller")
        self.check_every = check_every

    def on_forward(
        self,
        output: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Collect controller forward metrics."""
        if not self.enabled:
            return None

        self._step = step if step is not None else self._step + 1

        self.detector.register_tensor(
            "output", output, context={"step": self._step}
        )

        if hidden_state is not None:
            if isinstance(hidden_state, (list, tuple)):
                for i, h in enumerate(hidden_state):
                    if h is not None and isinstance(h, torch.Tensor):
                        self.detector.register_tensor(
                            f"hidden_state_{i}",
                            h,
                            context={"step": self._step},
                        )
            else:
                self.detector.register_tensor(
                    "hidden_state",
                    hidden_state,
                    context={"step": self._step},
                )

        if self._step % self.check_every == 0:
            return {"anomalies": self.detector.check()}

        return None


def create_training_hooks(
    detector: AnomalyDetector,
    model: nn.Module,
) -> Dict[str, MonitoringHook]:
    """Create standard set of training hooks."""
    return {
        "training": TrainingHook(detector, model),
        "controller": ControllerHook(detector),
    }


def create_ppo_hooks(
    detector: AnomalyDetector,
) -> Dict[str, MonitoringHook]:
    """Create standard set of PPO hooks."""
    return {
        "ppo": PPOHook(detector),
        "memory": MemoryHook(detector),
        "controller": ControllerHook(detector),
    }
