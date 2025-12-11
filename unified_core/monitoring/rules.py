"""
Anomaly Detection Rules.

Provides:
- Base rule classes (Threshold, Statistical, Pattern)
- Pre-built rules for common anomalies
- Rule factory for loading from config
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import deque
import math

from .anomaly_detector import (
    Anomaly,
    Severity,
    MetricRegistry,
    Metric,
)


class AnomalyRule(ABC):
    """Base class for anomaly detection rules."""

    def __init__(self, name: str, severity: Severity = Severity.WARNING):
        self.name = name
        self.severity = severity
        self.enabled = True

    @abstractmethod
    def check(self, registry: MetricRegistry) -> Optional[Anomaly]:
        """
        Check if this rule is violated.

        Args:
            registry: MetricRegistry with current metrics

        Returns:
            Anomaly if rule violated, None otherwise
        """
        pass

    def create_anomaly(
        self,
        metric_name: str,
        message: str,
        metric_value: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None,
    ) -> Anomaly:
        """Helper to create an anomaly from this rule."""
        return Anomaly(
            rule_name=self.name,
            metric_name=metric_name,
            severity=self.severity,
            message=message,
            metric_value=metric_value,
            context=context or {},
            suggested_action=suggested_action,
        )


class ThresholdRule(AnomalyRule):
    """Rule that triggers when a metric exceeds a threshold."""

    def __init__(
        self,
        name: str,
        metric_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        severity: Severity = Severity.WARNING,
        suggested_action: Optional[str] = None,
    ):
        super().__init__(name, severity)
        self.metric_name = metric_name
        self.min_value = min_value
        self.max_value = max_value
        self.suggested_action = suggested_action

    def check(self, registry: MetricRegistry) -> Optional[Anomaly]:
        if not self.enabled:
            return None

        metric = registry.get_latest(self.metric_name)
        if metric is None:
            return None

        value = metric.value

        if self.min_value is not None and value < self.min_value:
            return self.create_anomaly(
                metric_name=self.metric_name,
                message=f"{self.metric_name} ({value:.4g}) below min ({self.min_value})",
                metric_value=value,
                suggested_action=self.suggested_action,
            )

        if self.max_value is not None and value > self.max_value:
            return self.create_anomaly(
                metric_name=self.metric_name,
                message=f"{self.metric_name} ({value:.4g}) above max ({self.max_value})",
                metric_value=value,
                suggested_action=self.suggested_action,
            )

        return None


class StatisticalRule(AnomalyRule):
    """Rule that triggers based on statistical deviation."""

    def __init__(
        self,
        name: str,
        metric_name: str,
        z_threshold: float = 3.0,
        min_samples: int = 10,
        severity: Severity = Severity.WARNING,
        suggested_action: Optional[str] = None,
    ):
        super().__init__(name, severity)
        self.metric_name = metric_name
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        self.suggested_action = suggested_action

    def check(self, registry: MetricRegistry) -> Optional[Anomaly]:
        if not self.enabled:
            return None

        history = registry.get_history(self.metric_name)
        if len(history) < self.min_samples:
            return None

        values = [m.value for m in history]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1e-8

        latest = values[-1]
        z_score = abs(latest - mean) / std

        if z_score > self.z_threshold:
            return self.create_anomaly(
                metric_name=self.metric_name,
                message=(
                    f"{self.metric_name} z-score ({z_score:.2f}) "
                    f"exceeds threshold ({self.z_threshold})"
                ),
                metric_value=latest,
                context={"z_score": z_score, "mean": mean, "std": std},
                suggested_action=self.suggested_action,
            )

        return None


class PatternRule(AnomalyRule):
    """Rule that triggers based on repeated patterns."""

    def __init__(
        self,
        name: str,
        metric_name: str,
        pattern_length: int = 10,
        threshold: float = 0.99,
        severity: Severity = Severity.WARNING,
        suggested_action: Optional[str] = None,
    ):
        super().__init__(name, severity)
        self.metric_name = metric_name
        self.pattern_length = pattern_length
        self.threshold = threshold
        self.suggested_action = suggested_action

    def check(self, registry: MetricRegistry) -> Optional[Anomaly]:
        if not self.enabled:
            return None

        history = registry.get_history(self.metric_name, self.pattern_length)
        if len(history) < self.pattern_length:
            return None

        values = [m.value for m in history]

        same_count = sum(
            1 for v in values if abs(v - values[0]) < 1e-6
        )
        same_ratio = same_count / len(values)

        if same_ratio >= self.threshold:
            return self.create_anomaly(
                metric_name=self.metric_name,
                message=(
                    f"{self.metric_name} stuck at {values[0]:.4g} "
                    f"for {same_count}/{len(values)} steps"
                ),
                metric_value=values[-1],
                context={"same_ratio": same_ratio, "stuck_value": values[0]},
                suggested_action=self.suggested_action,
            )

        return None


class NaNInfRule(AnomalyRule):
    """Rule that detects NaN/Inf in tensor metrics."""

    def __init__(
        self,
        metric_names: Optional[List[str]] = None,
        severity: Severity = Severity.CRITICAL,
    ):
        super().__init__("NaNInfDetector", severity)
        self.metric_names = metric_names

    def check(self, registry: MetricRegistry) -> Optional[Anomaly]:
        if not self.enabled:
            return None

        metrics = registry.get_all_latest()

        for name, metric in metrics.items():
            if self.metric_names and name not in self.metric_names:
                continue

            if metric.tensor_stats is None:
                if math.isnan(metric.value) or math.isinf(metric.value):
                    return self.create_anomaly(
                        metric_name=name,
                        message=f"NaN/Inf detected in {name}",
                        metric_value=metric.value,
                        suggested_action="Halt training, check data pipeline",
                    )
            else:
                stats = metric.tensor_stats
                if stats.get("has_nan") or stats.get("has_inf"):
                    return self.create_anomaly(
                        metric_name=name,
                        message=f"NaN/Inf detected in tensor {name}",
                        metric_value=metric.value,
                        context=stats,
                        suggested_action="Halt training, check gradients",
                    )

        return None


class GradientExplosionRule(AnomalyRule):
    """Rule that detects gradient explosion."""

    def __init__(
        self,
        metric_name: str = "grad_norm",
        max_grad_norm: float = 1.0,
        explosion_factor: float = 5.0,
        severity: Severity = Severity.ERROR,
    ):
        super().__init__("GradientExplosion", severity)
        self.metric_name = metric_name
        self.max_grad_norm = max_grad_norm
        self.explosion_factor = explosion_factor

    def check(self, registry: MetricRegistry) -> Optional[Anomaly]:
        if not self.enabled:
            return None

        metric = registry.get_latest(self.metric_name)
        if metric is None:
            return None

        threshold = self.max_grad_norm * self.explosion_factor
        if metric.value > threshold:
            return self.create_anomaly(
                metric_name=self.metric_name,
                message=(
                    f"Gradient norm ({metric.value:.4g}) exceeds "
                    f"{self.explosion_factor}x max ({threshold:.4g})"
                ),
                metric_value=metric.value,
                suggested_action="Apply stronger gradient clipping",
            )

        return None


class AdvantageCollapseRule(AnomalyRule):
    """Rule that detects advantage collapse in PPO."""

    def __init__(
        self,
        metric_name: str = "advantage_std",
        min_std: float = 1e-5,
        window_size: int = 10,
        severity: Severity = Severity.WARNING,
    ):
        super().__init__("AdvantageCollapse", severity)
        self.metric_name = metric_name
        self.min_std = min_std
        self.window_size = window_size

    def check(self, registry: MetricRegistry) -> Optional[Anomaly]:
        if not self.enabled:
            return None

        history = registry.get_history(self.metric_name, self.window_size)
        if len(history) < self.window_size:
            return None

        all_below = all(m.value < self.min_std for m in history)
        if all_below:
            avg_std = sum(m.value for m in history) / len(history)
            return self.create_anomaly(
                metric_name=self.metric_name,
                message=(
                    f"Advantage std ({avg_std:.2e}) below threshold "
                    f"for {self.window_size} steps"
                ),
                metric_value=avg_std,
                suggested_action="Check hyperparameters, increase exploration",
            )

        return None


class PolicyCollapseRule(AnomalyRule):
    """Rule that detects policy collapse (single action dominance)."""

    def __init__(
        self,
        metric_name: str = "max_action_prob",
        threshold: float = 0.99,
        window_size: int = 10,
        severity: Severity = Severity.WARNING,
    ):
        super().__init__("PolicyCollapse", severity)
        self.metric_name = metric_name
        self.threshold = threshold
        self.window_size = window_size

    def check(self, registry: MetricRegistry) -> Optional[Anomaly]:
        if not self.enabled:
            return None

        history = registry.get_history(self.metric_name, self.window_size)
        if len(history) < self.window_size:
            return None

        all_collapsed = all(m.value > self.threshold for m in history)
        if all_collapsed:
            avg_prob = sum(m.value for m in history) / len(history)
            return self.create_anomaly(
                metric_name=self.metric_name,
                message=(
                    f"Policy collapsed: max action prob ({avg_prob:.4f}) > "
                    f"{self.threshold} for {self.window_size} steps"
                ),
                metric_value=avg_prob,
                suggested_action="Reduce learning rate, increase entropy bonus",
            )

        return None


class MemorySaturationRule(AnomalyRule):
    """Rule that detects memory saturation in DualTierMiras."""

    def __init__(
        self,
        metric_name: str = "deep_write_freq",
        high_threshold: float = 0.95,
        low_threshold: float = 0.05,
        window_size: int = 20,
        severity: Severity = Severity.WARNING,
    ):
        super().__init__("MemorySaturation", severity)
        self.metric_name = metric_name
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.window_size = window_size

    def check(self, registry: MetricRegistry) -> Optional[Anomaly]:
        if not self.enabled:
            return None

        history = registry.get_history(self.metric_name, self.window_size)
        if len(history) < self.window_size:
            return None

        avg_freq = sum(m.value for m in history) / len(history)

        if avg_freq > self.high_threshold:
            return self.create_anomaly(
                metric_name=self.metric_name,
                message=(
                    f"Deep memory write frequency ({avg_freq:.2%}) too high, "
                    f"surprise gating may be ineffective"
                ),
                metric_value=avg_freq,
                suggested_action="Increase surprise threshold",
            )

        if avg_freq < self.low_threshold:
            return self.create_anomaly(
                metric_name=self.metric_name,
                message=(
                    f"Deep memory write frequency ({avg_freq:.2%}) too low, "
                    f"deep tier underutilized"
                ),
                metric_value=avg_freq,
                suggested_action="Decrease surprise threshold",
            )

        return None


class RuntimePerformanceRule(AnomalyRule):
    """Rule that detects runtime performance anomalies."""

    def __init__(
        self,
        metric_name: str = "step_duration",
        multiplier: float = 3.0,
        min_samples: int = 10,
        severity: Severity = Severity.INFO,
    ):
        super().__init__("RuntimePerformance", severity)
        self.metric_name = metric_name
        self.multiplier = multiplier
        self.min_samples = min_samples

    def check(self, registry: MetricRegistry) -> Optional[Anomaly]:
        if not self.enabled:
            return None

        history = registry.get_history(self.metric_name)
        if len(history) < self.min_samples:
            return None

        values = [m.value for m in history[:-1]]
        avg = sum(values) / len(values)
        latest = history[-1].value

        if latest > avg * self.multiplier:
            return self.create_anomaly(
                metric_name=self.metric_name,
                message=(
                    f"Step duration ({latest:.3f}s) is {latest/avg:.1f}x "
                    f"the average ({avg:.3f}s)"
                ),
                metric_value=latest,
                context={"average": avg, "ratio": latest / avg},
                suggested_action="Check for resource contention",
            )

        return None


def get_default_rules() -> List[AnomalyRule]:
    """Get the default set of anomaly detection rules."""
    return [
        NaNInfRule(),
        GradientExplosionRule(
            max_grad_norm=1.0,
            explosion_factor=5.0,
        ),
        AdvantageCollapseRule(
            min_std=1e-5,
            window_size=10,
        ),
        PolicyCollapseRule(
            threshold=0.99,
            window_size=10,
        ),
        MemorySaturationRule(
            high_threshold=0.95,
            low_threshold=0.05,
            window_size=20,
        ),
        RuntimePerformanceRule(
            multiplier=3.0,
            min_samples=10,
        ),
        ThresholdRule(
            name="LossExplosion",
            metric_name="loss",
            max_value=100.0,
            severity=Severity.ERROR,
            suggested_action="Check learning rate, data",
        ),
        StatisticalRule(
            name="LossSpike",
            metric_name="loss",
            z_threshold=5.0,
            min_samples=20,
            severity=Severity.WARNING,
            suggested_action="Investigate batch causing spike",
        ),
    ]
