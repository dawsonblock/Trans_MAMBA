"""
Mitigation Handlers for Anomaly Response.

Provides configurable actions to take when anomalies are detected:
- Log only
- Reset memory/environment state
- Skip update steps
- Adjust learning rate
- Apply gradient clipping
- Abort training
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Dict, Optional

from .anomaly_detector import Anomaly, Severity


logger = logging.getLogger(__name__)


class MitigationAction(Enum):
    """Types of mitigation actions."""

    LOG_ONLY = "log_only"
    RESET_MEMORY = "reset_memory"
    RESET_ENV = "reset_env"
    SKIP_UPDATE = "skip_update"
    REDUCE_LR = "reduce_lr"
    GRADIENT_CLIP = "gradient_clip"
    ABORT_TRAINING = "abort_training"


@dataclass
class MitigationResult:
    """Result of a mitigation action."""

    action: MitigationAction
    success: bool
    message: str
    context: Dict[str, Any]


class MitigationHandler(ABC):
    """Base class for mitigation handlers."""

    def __init__(self, action: MitigationAction):
        self.action = action
        self.enabled = True
        self._count = 0

    @abstractmethod
    def handle(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
    ) -> MitigationResult:
        """Handle an anomaly."""

    def __call__(self, anomaly: Anomaly, context: Optional[Dict] = None):
        """Execute the handler."""

        if not self.enabled:
            return None
        self._count += 1
        return self.handle(anomaly, context or {})


class LogOnlyHandler(MitigationHandler):
    """Handler that only logs the anomaly."""

    def __init__(self, log_level: int = logging.WARNING):
        super().__init__(MitigationAction.LOG_ONLY)
        self.log_level = log_level

    def handle(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
    ) -> MitigationResult:
        msg = f"[ANOMALY] {anomaly}"
        logger.log(self.log_level, msg)

        return MitigationResult(
            action=self.action,
            success=True,
            message=f"Logged anomaly: {anomaly.rule_name}",
            context={"log_level": self.log_level},
        )


class ResetMemoryHandler(MitigationHandler):
    """Handler that resets memory state."""

    def __init__(self):
        super().__init__(MitigationAction.RESET_MEMORY)

    def handle(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
    ) -> MitigationResult:
        memory = context.get("memory")
        batch_size = context.get("batch_size", 1)
        device = context.get("device")

        if memory is None:
            return MitigationResult(
                action=self.action,
                success=False,
                message="No memory module in context",
                context={},
            )

        try:
            if hasattr(memory, "init_state"):
                new_state = memory.init_state(batch_size, device)
                context["memory_state"] = new_state

            logger.warning(
                f"Reset memory state due to anomaly: {anomaly.rule_name}"
            )

            return MitigationResult(
                action=self.action,
                success=True,
                message="Memory state reset",
                context={"batch_size": batch_size},
            )
        except Exception as e:
            return MitigationResult(
                action=self.action,
                success=False,
                message=f"Failed to reset memory: {e}",
                context={"error": str(e)},
            )


class ResetEnvHandler(MitigationHandler):
    """Handler that resets environment(s)."""

    def __init__(self):
        super().__init__(MitigationAction.RESET_ENV)

    def handle(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
    ) -> MitigationResult:
        env = context.get("env")

        if env is None:
            return MitigationResult(
                action=self.action,
                success=False,
                message="No environment in context",
                context={},
            )

        try:
            obs, _info = env.reset()
            context["obs"] = obs

            logger.warning(
                f"Reset environment due to anomaly: {anomaly.rule_name}"
            )

            return MitigationResult(
                action=self.action,
                success=True,
                message="Environment reset",
                context={},
            )
        except Exception as e:
            return MitigationResult(
                action=self.action,
                success=False,
                message=f"Failed to reset environment: {e}",
                context={"error": str(e)},
            )


class SkipUpdateHandler(MitigationHandler):
    """Handler that skips the current update step."""

    def __init__(self):
        super().__init__(MitigationAction.SKIP_UPDATE)

    def handle(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
    ) -> MitigationResult:
        context["skip_update"] = True

        logger.warning(
            f"Skipping update step due to anomaly: {anomaly.rule_name}"
        )

        return MitigationResult(
            action=self.action,
            success=True,
            message="Update step will be skipped",
            context={},
        )


class ReduceLRHandler(MitigationHandler):
    """Handler that reduces learning rate."""

    def __init__(self, factor: float = 0.5, min_lr: float = 1e-6):
        super().__init__(MitigationAction.REDUCE_LR)
        self.factor = factor
        self.min_lr = min_lr

    def handle(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
    ) -> MitigationResult:
        optimizer = context.get("optimizer")

        if optimizer is None:
            return MitigationResult(
                action=self.action,
                success=False,
                message="No optimizer in context",
                context={},
            )

        try:
            old_lrs = []
            new_lrs = []

            for param_group in optimizer.param_groups:
                old_lr = param_group["lr"]
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group["lr"] = new_lr
                old_lrs.append(old_lr)
                new_lrs.append(new_lr)

            logger.warning(
                f"Reduced LR from {old_lrs} to {new_lrs} "
                f"due to anomaly: {anomaly.rule_name}"
            )

            return MitigationResult(
                action=self.action,
                success=True,
                message=f"Learning rate reduced by factor {self.factor}",
                context={"old_lrs": old_lrs, "new_lrs": new_lrs},
            )
        except Exception as e:
            return MitigationResult(
                action=self.action,
                success=False,
                message=f"Failed to reduce LR: {e}",
                context={"error": str(e)},
            )


class GradientClipHandler(MitigationHandler):
    """Handler that applies stronger gradient clipping."""

    def __init__(self, clip_value: float = 0.1):
        super().__init__(MitigationAction.GRADIENT_CLIP)
        self.clip_value = clip_value

    def handle(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
    ) -> MitigationResult:
        model = context.get("model")

        if model is None:
            return MitigationResult(
                action=self.action,
                success=False,
                message="No model in context",
                context={},
            )

        try:
            import torch.nn.utils as nn_utils

            total_norm = nn_utils.clip_grad_norm_(
                model.parameters(),
                self.clip_value,
            )

            logger.warning(
                f"Applied gradient clipping (norm={total_norm:.4f}, "
                f"clip={self.clip_value}) due to anomaly: {anomaly.rule_name}"
            )

            return MitigationResult(
                action=self.action,
                success=True,
                message=f"Gradients clipped to {self.clip_value}",
                context={
                    "original_norm": float(total_norm),
                    "clip_value": self.clip_value,
                },
            )
        except Exception as e:
            return MitigationResult(
                action=self.action,
                success=False,
                message=f"Failed to clip gradients: {e}",
                context={"error": str(e)},
            )


class AbortTrainingHandler(MitigationHandler):
    """Handler that aborts training."""

    def __init__(self):
        super().__init__(MitigationAction.ABORT_TRAINING)

    def handle(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
    ) -> MitigationResult:
        context["abort_training"] = True

        logger.error(
            f"ABORTING TRAINING due to critical anomaly: {anomaly.rule_name}"
        )

        return MitigationResult(
            action=self.action,
            success=True,
            message="Training will be aborted",
            context={"anomaly": anomaly.to_dict()},
        )


def get_default_handlers() -> Dict[Severity, list]:
    """Get default mitigation handlers for each severity level."""

    return {
        Severity.INFO: [
            LogOnlyHandler(log_level=logging.INFO),
        ],
        Severity.WARNING: [
            LogOnlyHandler(log_level=logging.WARNING),
        ],
        Severity.ERROR: [
            LogOnlyHandler(log_level=logging.ERROR),
            GradientClipHandler(clip_value=0.1),
        ],
        Severity.CRITICAL: [
            LogOnlyHandler(log_level=logging.CRITICAL),
            AbortTrainingHandler(),
        ],
    }


def create_mitigation_context(
    model=None,
    optimizer=None,
    env=None,
    memory=None,
    memory_state=None,
    device=None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Create a context dict for mitigation handlers."""

    return {
        "model": model,
        "optimizer": optimizer,
        "env": env,
        "memory": memory,
        "memory_state": memory_state,
        "device": device,
        "batch_size": batch_size,
        "skip_update": False,
        "abort_training": False,
    }
