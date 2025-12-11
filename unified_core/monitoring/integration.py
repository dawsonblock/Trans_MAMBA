"""
Integration helpers for anomaly detection.

Provides factory functions and wrappers to easily add
monitoring to existing training code.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn

from .anomaly_detector import AnomalyDetector, AnomalyDetectorConfig, Severity
from .rules import get_default_rules, AnomalyRule
from .hooks import (
    TrainingHook,
    PPOHook,
    MemoryHook,
    ControllerHook,
)
from .mitigations import (
    get_default_handlers,
    LogOnlyHandler,
    GradientClipHandler,
    ReduceLRHandler,
    AbortTrainingHandler,
    create_mitigation_context,
)


def load_config(config_path: str) -> AnomalyDetectorConfig:
    """Load anomaly detector config from JSON file."""
    with open(config_path, "r") as f:
        data = json.load(f)

    return AnomalyDetectorConfig(
        enabled=data.get("enabled", True),
        history_size=data.get("history_size", 1000),
        check_interval=data.get("check_interval", 1),
        log_file=data.get("log_file"),
        halt_on_critical=data.get("halt_on_critical", True),
        verbose=data.get("verbose", False),
    )


def create_detector(
    config_path: Optional[str] = None,
    enabled: bool = True,
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> AnomalyDetector:
    """
    Create an anomaly detector with default rules and handlers.

    Args:
        config_path: Path to JSON config file (optional)
        enabled: Whether detection is enabled
        verbose: Print anomalies when detected
        log_file: Path to anomaly log file

    Returns:
        Configured AnomalyDetector
    """
    if config_path and Path(config_path).exists():
        cfg = load_config(config_path)
    else:
        cfg = AnomalyDetectorConfig(
            enabled=enabled,
            verbose=verbose,
            log_file=log_file,
        )

    detector = AnomalyDetector(cfg)
    detector.add_rules(get_default_rules())

    handlers = get_default_handlers()
    for severity, handler_list in handlers.items():
        for handler in handler_list:
            detector.add_mitigation_handler(
                severity,
                lambda a, h=handler: h(a, {}),
            )

    return detector


def create_lm_monitor(
    detector: AnomalyDetector,
    model: nn.Module,
) -> Dict[str, Any]:
    """
    Create monitoring hooks for LM training.

    Returns dict with:
        - training_hook: TrainingHook for main loop
        - controller_hook: ControllerHook for forward passes
        - check_fn: Function to call to check for anomalies
    """
    training_hook = TrainingHook(detector, model)
    controller_hook = ControllerHook(detector)

    def check_fn(
        loss: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> bool:
        """
        Check for anomalies after a training step.

        Returns True if training should continue, False if halted.
        """
        result = training_hook(loss, logits, step)

        if result and result.get("anomalies"):
            for anomaly in result["anomalies"]:
                if anomaly.severity == Severity.CRITICAL:
                    return False

        return not detector.is_halted()

    def on_backward_fn():
        """Call after loss.backward() to check gradients."""
        training_hook.on_backward()
        detector.check()
        return not detector.is_halted()

    return {
        "training_hook": training_hook,
        "controller_hook": controller_hook,
        "check_fn": check_fn,
        "on_backward_fn": on_backward_fn,
        "detector": detector,
    }


def create_ppo_monitor(
    detector: AnomalyDetector,
) -> Dict[str, Any]:
    """
    Create monitoring hooks for PPO training.

    Returns dict with:
        - ppo_hook: PPOHook for rollouts and updates
        - memory_hook: MemoryHook for memory operations
        - check_rollout_fn: Function to check after rollout
        - check_update_fn: Function to check after PPO update
    """
    ppo_hook = PPOHook(detector)
    memory_hook = MemoryHook(detector)

    def check_rollout_fn(
        rewards: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        step: Optional[int] = None,
    ) -> bool:
        """Check for anomalies after rollout collection."""
        ppo_hook.on_rollout(rewards, values, log_probs, step)
        detector.check()
        return not detector.is_halted()

    def check_update_fn(
        policy_loss: float,
        value_loss: float,
        entropy: float,
        advantages: torch.Tensor,
        ratios: Optional[torch.Tensor] = None,
        action_probs: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> bool:
        """Check for anomalies after PPO update."""
        ppo_hook.on_update(
            policy_loss,
            value_loss,
            entropy,
            advantages,
            ratios,
            action_probs,
            step,
        )
        detector.check()
        return not detector.is_halted()

    def check_memory_fn(
        surprise: torch.Tensor,
        did_deep_write: bool,
        step: Optional[int] = None,
    ) -> bool:
        """Check for memory anomalies."""
        memory_hook.on_write(surprise, did_deep_write, step=step)
        detector.check()
        return not detector.is_halted()

    return {
        "ppo_hook": ppo_hook,
        "memory_hook": memory_hook,
        "check_rollout_fn": check_rollout_fn,
        "check_update_fn": check_update_fn,
        "check_memory_fn": check_memory_fn,
        "detector": detector,
    }


class MonitoredTrainer:
    """
    Wrapper that adds monitoring to any trainer with train_step method.

    Usage:
        trainer = LMBenchmark(model, cfg, task_cfg)
        monitored = MonitoredTrainer(trainer, detector)
        monitored.train()
    """

    def __init__(
        self,
        trainer: Any,
        detector: AnomalyDetector,
        model: Optional[nn.Module] = None,
    ):
        self.trainer = trainer
        self.detector = detector
        self.model = model or getattr(trainer, "model", None)

        if self.model:
            self.monitor = create_lm_monitor(detector, self.model)
        else:
            self.monitor = {"detector": detector}

        self._aborted = False

    def train_step(self, batch: tuple) -> Dict[str, float]:
        """Monitored training step."""
        if self._aborted:
            return {"loss": float("nan"), "aborted": True}

        metrics = self.trainer.train_step(batch)

        if "check_fn" in self.monitor:
            loss_tensor = torch.tensor(metrics["loss"])
            continue_training = self.monitor["check_fn"](
                loss_tensor,
                step=getattr(self.trainer, "global_step", None),
            )

            if not continue_training:
                self._aborted = True
                metrics["aborted"] = True

        return metrics

    def train(self) -> Dict[str, Any]:
        """Run full training with monitoring."""
        if hasattr(self.trainer, "train"):
            result = self.trainer.train()
        else:
            result = {}

        self.detector.print_summary()
        result["anomaly_summary"] = self.detector.get_summary()

        return result

    def __getattr__(self, name):
        """Delegate to wrapped trainer."""
        return getattr(self.trainer, name)


class MonitoredPPOTrainer:
    """
    Wrapper that adds monitoring to PPO trainer.

    Usage:
        trainer = PPOTrainer(agent, cfg, device)
        monitored = MonitoredPPOTrainer(trainer, detector)
        monitored.train(env, total_steps)
    """

    def __init__(
        self,
        trainer: Any,
        detector: AnomalyDetector,
    ):
        self.trainer = trainer
        self.detector = detector
        self.monitor = create_ppo_monitor(detector)
        self._aborted = False

    def collect_rollout(
        self,
        env,
        rollout_length: int,
        agent_state=None,
    ):
        """Monitored rollout collection."""
        if self._aborted:
            return 0.0, torch.zeros(1)

        total_reward, next_value = self.trainer.collect_rollout(
            env, rollout_length, agent_state
        )

        data = self.trainer.buffer.get_tensors(self.trainer.device)
        continue_training = self.monitor["check_rollout_fn"](
            data["rewards"],
            data["values"],
            data["log_probs"],
            step=self.trainer.global_step,
        )

        if not continue_training:
            self._aborted = True

        return total_reward, next_value

    def update(self, next_value: torch.Tensor) -> Dict[str, float]:
        """Monitored PPO update."""
        if self._aborted:
            return {"loss": float("nan"), "aborted": True}

        metrics = self.trainer.update(next_value)

        data = self.trainer.buffer.get_tensors(self.trainer.device)
        if "rewards" in data:
            advantages = data.get("advantages", torch.zeros(1))
        else:
            advantages = torch.zeros(1)

        continue_training = self.monitor["check_update_fn"](
            metrics.get("policy_loss", 0.0),
            metrics.get("value_loss", 0.0),
            metrics.get("entropy", 0.0),
            advantages,
            step=self.trainer.global_step,
        )

        if not continue_training:
            self._aborted = True
            metrics["aborted"] = True

        return metrics

    def train(
        self,
        env,
        total_steps: int,
        rollout_length: int = 128,
        log_interval: int = 10,
    ) -> Dict[str, Any]:
        """Full PPO training loop with monitoring."""
        obs, _ = env.reset()
        agent_state = None
        step = 0
        total_reward = 0.0

        while step < total_steps and not self._aborted:
            reward, next_value = self.collect_rollout(
                env, rollout_length, agent_state
            )
            total_reward += reward

            metrics = self.update(next_value)

            step += rollout_length

            if step % log_interval == 0:
                print(
                    f"Step {step}/{total_steps}: "
                    f"reward={reward:.2f}, loss={metrics.get('loss', 0):.4f}"
                )

        self.detector.print_summary()

        return {
            "total_reward": total_reward,
            "steps": step,
            "aborted": self._aborted,
            "anomaly_summary": self.detector.get_summary(),
        }

    def __getattr__(self, name):
        """Delegate to wrapped trainer."""
        return getattr(self.trainer, name)
