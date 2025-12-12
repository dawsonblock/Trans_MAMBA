"""
Core Anomaly Detector Module.

Provides:
- Metric registry for collecting and storing metrics
- Anomaly detection engine that evaluates rules
- Severity classification and anomaly reporting
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from typing import Any, Callable, Dict, List, Optional

import torch


class Severity(Enum):
    """Anomaly severity levels."""

    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


@dataclass
class Metric:
    """A single metric measurement."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    tensor_stats: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "context": self.context,
            "tensor_stats": self.tensor_stats,
        }


@dataclass
class Anomaly:
    """A detected anomaly."""

    rule_name: str
    metric_name: str
    severity: Severity
    message: str
    timestamp: float = field(default_factory=time.time)
    metric_value: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "metric_name": self.metric_name,
            "severity": self.severity.name,
            "message": self.message,
            "timestamp": self.timestamp,
            "metric_value": self.metric_value,
            "context": self.context,
            "suggested_action": self.suggested_action,
        }

    def __str__(self) -> str:
        return (
            f"[{self.severity.name}] {self.rule_name}: "
            f"{self.message} (metric={self.metric_name}, "
            f"value={self.metric_value})"
        )


@dataclass
class AnomalyDetectorConfig:
    """Configuration for the anomaly detector."""

    enabled: bool = True
    history_size: int = 1000
    check_interval: int = 1
    log_file: Optional[str] = None
    halt_on_critical: bool = True
    verbose: bool = False


class MetricRegistry:
    """Registry for collecting and storing metrics with history."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._metrics: Dict[str, deque] = {}
        self._latest: Dict[str, Metric] = {}

    def register(
        self,
        name: str,
        value: float,
        context: Optional[Dict[str, Any]] = None,
        tensor: Optional[torch.Tensor] = None,
    ) -> Metric:
        """Register a scalar metric."""

        tensor_stats = None
        if tensor is not None:
            tensor_stats = self._compute_tensor_stats(tensor)

        metric = Metric(
            name=name,
            value=value,
            context=context or {},
            tensor_stats=tensor_stats,
        )

        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=self.history_size)

        self._metrics[name].append(metric)
        self._latest[name] = metric
        return metric

    def register_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Metric:
        """Register a tensor metric (computes stats automatically)."""

        stats = self._compute_tensor_stats(tensor)
        return self.register(
            name=name,
            value=stats["mean"],
            context=context,
            tensor=tensor,
        )

    def _compute_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Compute statistics for a tensor."""

        with torch.no_grad():
            t = tensor.float()
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()

            if has_nan or has_inf:
                return {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "norm": float("nan"),
                    "has_nan": bool(has_nan),
                    "has_inf": bool(has_inf),
                }

            return {
                "mean": t.mean().item(),
                "std": t.std().item() if t.numel() > 1 else 0.0,
                "min": t.min().item(),
                "max": t.max().item(),
                "norm": t.norm().item(),
                "has_nan": False,
                "has_inf": False,
            }

    def get_latest(self, name: str) -> Optional[Metric]:
        """Get the most recent metric value."""

        return self._latest.get(name)

    def get_history(self, name: str, n: Optional[int] = None) -> List[Metric]:
        """Get metric history."""

        if name not in self._metrics:
            return []
        history = list(self._metrics[name])
        if n is not None:
            history = history[-n:]
        return history

    def get_all_latest(self) -> Dict[str, Metric]:
        """Get all latest metrics."""

        return dict(self._latest)

    def clear(self):
        """Clear all metrics."""

        self._metrics.clear()
        self._latest.clear()


class AnomalyDetector:
    """
    Main anomaly detection engine.

    Evaluates rules against collected metrics and triggers
    mitigation actions when anomalies are detected.
    """

    def __init__(self, cfg: AnomalyDetectorConfig):
        self.cfg = cfg
        self.registry = MetricRegistry(cfg.history_size)
        self.rules: List[Any] = []
        self.anomalies: List[Anomaly] = []
        self.anomaly_counts: Dict[str, int] = {}
        self.mitigation_handlers: Dict[Severity, List[Callable]] = {
            Severity.INFO: [],
            Severity.WARNING: [],
            Severity.ERROR: [],
            Severity.CRITICAL: [],
        }
        self._step = 0
        self._halted = False

        if cfg.log_file:
            self._log_file = open(cfg.log_file, "a")
        else:
            self._log_file = None

    def add_rule(self, rule: Any):
        """Add an anomaly detection rule."""

        self.rules.append(rule)

    def add_rules(self, rules: List[Any]):
        """Add multiple rules."""

        self.rules.extend(rules)

    def add_mitigation_handler(
        self,
        severity: Severity,
        handler: Callable[[Anomaly], None],
    ):
        """Register a mitigation handler for a severity level."""

        self.mitigation_handlers[severity].append(handler)

    def register_metric(
        self,
        name: str,
        value: float,
        context: Optional[Dict[str, Any]] = None,
        tensor: Optional[torch.Tensor] = None,
    ) -> Metric:
        """Register a metric."""

        if not self.cfg.enabled:
            return Metric(name=name, value=value)

        return self.registry.register(name, value, context, tensor)

    def register_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Metric:
        """Register a tensor metric."""

        if not self.cfg.enabled:
            return Metric(name=name, value=0.0)

        return self.registry.register_tensor(name, tensor, context)

    def check(self, force: bool = False) -> List[Anomaly]:
        """Check all rules against current metrics."""

        if not self.cfg.enabled or self._halted:
            return []

        self._step += 1
        if not force and self._step % self.cfg.check_interval != 0:
            return []

        detected = []
        for rule in self.rules:
            anomaly = rule.check(self.registry)
            if anomaly is not None:
                detected.append(anomaly)
                self._handle_anomaly(anomaly)

        return detected

    def _handle_anomaly(self, anomaly: Anomaly):
        """Handle a detected anomaly."""

        self.anomalies.append(anomaly)
        self.anomaly_counts[anomaly.rule_name] = (
            self.anomaly_counts.get(anomaly.rule_name, 0) + 1
        )

        if self.cfg.verbose:
            print(str(anomaly))

        if self._log_file:
            self._log_file.write(json.dumps(anomaly.to_dict()) + "\n")
            self._log_file.flush()

        for handler in self.mitigation_handlers[anomaly.severity]:
            handler(anomaly)

        if (
            anomaly.severity == Severity.CRITICAL
            and self.cfg.halt_on_critical
        ):
            self._halted = True

    def is_halted(self) -> bool:
        """Check if detector has halted due to critical anomaly."""

        return self._halted

    def reset_halt(self):
        """Reset the halted state."""

        self._halted = False

    def get_summary(self) -> Dict[str, Any]:
        """Get anomaly summary report."""

        severity_counts = {s.name: 0 for s in Severity}
        for a in self.anomalies:
            severity_counts[a.severity.name] += 1

        recent_critical = [
            a.to_dict()
            for a in self.anomalies[-10:]
            if a.severity == Severity.CRITICAL
        ]

        return {
            "total_anomalies": len(self.anomalies),
            "by_severity": severity_counts,
            "by_rule": dict(self.anomaly_counts),
            "recent_critical": recent_critical,
            "is_halted": self._halted,
        }

    def print_summary(self):
        """Print human-readable summary."""

        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("ANOMALY DETECTION SUMMARY")
        print("=" * 60)
        print(f"Total anomalies: {summary['total_anomalies']}")
        print("\nBy severity:")
        for sev, count in summary["by_severity"].items():
            print(f"  {sev}: {count}")
        print("\nBy rule:")
        for rule, count in summary["by_rule"].items():
            print(f"  {rule}: {count}")
        if summary["recent_critical"]:
            print("\nRecent critical events:")
            for event in summary["recent_critical"]:
                print(f"  - {event['message']}")
        print("=" * 60 + "\n")

    def close(self):
        """Clean up resources."""

        if self._log_file:
            self._log_file.close()
            self._log_file = None


def check_tensor_health(
    tensor: torch.Tensor,
    name: str = "tensor",
) -> Optional[Anomaly]:
    """Quick health check for a tensor."""

    with torch.no_grad():
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        if has_nan or has_inf:
            return Anomaly(
                rule_name="TensorHealthCheck",
                metric_name=name,
                severity=Severity.CRITICAL,
                message=(
                    f"Tensor '{name}' contains "
                    f"{'NaN' if has_nan else ''}"
                    f"{'/' if has_nan and has_inf else ''}"
                    f"{'Inf' if has_inf else ''}"
                ),
                metric_value=float("nan"),
            )

    return None
