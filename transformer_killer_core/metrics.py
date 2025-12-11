"""
metrics.py

Metrics, logging, and profiling utilities for Transformer Killer Core.

This module provides:
    - MetricsLogger: Track and aggregate training metrics
    - MemoryProfiler: GPU memory usage tracking
    - ThroughputMeter: Tokens/sec and samples/sec measurement
    - ModelAnalyzer: Parameter counts and FLOPs estimation

v2.2 Features:
    - Automatic metric aggregation with EMA
    - Memory usage snapshots
    - Throughput measurement
    - Model complexity analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
from collections import defaultdict

import torch
import torch.nn as nn


@dataclass
class MetricSnapshot:
    """Single metric measurement."""
    value: float
    step: int
    timestamp: float


class MetricsLogger:
    """Track and aggregate training metrics.

    Features:
        - Running averages with EMA
        - Min/max tracking
        - Histogram-style distribution tracking
        - Easy export to dict/JSON
    """

    def __init__(self, ema_decay: float = 0.99):
        self.ema_decay = ema_decay
        self.metrics: Dict[str, List[MetricSnapshot]] = defaultdict(list)
        self.ema_values: Dict[str, float] = {}
        self.min_values: Dict[str, float] = {}
        self.max_values: Dict[str, float] = {}
        self.step = 0

    def log(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if step is None:
            step = self.step

        snapshot = MetricSnapshot(
            value=value,
            step=step,
            timestamp=time.time()
        )
        self.metrics[name].append(snapshot)

        # Update EMA
        if name in self.ema_values:
            self.ema_values[name] = (
                self.ema_decay * self.ema_values[name] +
                (1 - self.ema_decay) * value
            )
        else:
            self.ema_values[name] = value

        # Update min/max
        if name not in self.min_values:
            self.min_values[name] = value
            self.max_values[name] = value
        else:
            self.min_values[name] = min(self.min_values[name], value)
            self.max_values[name] = max(self.max_values[name], value)

    def log_dict(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log(name, value, step)

    def step_forward(self):
        """Increment step counter."""
        self.step += 1

    def get_ema(self, name: str) -> Optional[float]:
        """Get EMA value for a metric."""
        return self.ema_values.get(name)

    def get_last(self, name: str) -> Optional[float]:
        """Get last logged value for a metric."""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1].value
        return None

    def get_mean(self, name: str, last_n: int = 100) -> Optional[float]:
        """Get mean of last N values."""
        if name not in self.metrics:
            return None
        values = [s.value for s in self.metrics[name][-last_n:]]
        if not values:
            return None
        return sum(values) / len(values)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for name in self.metrics:
            summary[name] = {
                "last": self.get_last(name),
                "ema": self.get_ema(name),
                "mean_100": self.get_mean(name, 100),
                "min": self.min_values.get(name),
                "max": self.max_values.get(name),
                "count": len(self.metrics[name]),
            }
        return summary

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.ema_values.clear()
        self.min_values.clear()
        self.max_values.clear()
        self.step = 0


class MemoryProfiler:
    """GPU memory usage profiler.

    Tracks:
        - Current allocation
        - Peak allocation
        - Reserved memory
        - Memory fragmentation
    """

    def __init__(self):
        self.snapshots: List[Dict[str, float]] = []

    def snapshot(self, label: str = "") -> Dict[str, float]:
        """Take a memory snapshot."""
        if not torch.cuda.is_available():
            return {"label": label, "available": False}

        stats = {
            "label": label,
            "timestamp": time.time(),
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
            "max_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
        }

        # Fragmentation estimate
        if stats["reserved_mb"] > 0:
            stats["fragmentation"] = 1 - (
                stats["allocated_mb"] / stats["reserved_mb"]
            )
        else:
            stats["fragmentation"] = 0.0

        self.snapshots.append(stats)
        return stats

    def reset_peak(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_current(self) -> Dict[str, float]:
        """Get current memory usage."""
        return self.snapshot("current")

    def get_peak(self) -> float:
        """Get peak memory allocation in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1e6


class ThroughputMeter:
    """Measure throughput in tokens/sec and samples/sec."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset meter."""
        self.start_time = time.time()
        self.total_tokens = 0
        self.total_samples = 0
        self.total_steps = 0

    def update(self, tokens: int = 0, samples: int = 0, steps: int = 1):
        """Update counters."""
        self.total_tokens += tokens
        self.total_samples += samples
        self.total_steps += steps

    def get_throughput(self) -> Dict[str, float]:
        """Get throughput statistics."""
        elapsed = time.time() - self.start_time
        if elapsed < 1e-6:
            elapsed = 1e-6

        return {
            "tokens_per_sec": self.total_tokens / elapsed,
            "samples_per_sec": self.total_samples / elapsed,
            "steps_per_sec": self.total_steps / elapsed,
            "elapsed_sec": elapsed,
            "total_tokens": self.total_tokens,
            "total_samples": self.total_samples,
            "total_steps": self.total_steps,
        }


class ModelAnalyzer:
    """Analyze model complexity and structure."""

    def __init__(self, model: nn.Module):
        self.model = model

    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts."""
        total = 0
        trainable = 0
        by_module: Dict[str, int] = {}

        for name, param in self.model.named_parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count

            # Group by top-level module
            top_module = name.split('.')[0]
            by_module[top_module] = by_module.get(top_module, 0) + count

        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
            "by_module": by_module,
        }

    def get_buffer_count(self) -> Dict[str, int]:
        """Get buffer counts."""
        total = 0
        by_module: Dict[str, int] = {}

        for name, buf in self.model.named_buffers():
            count = buf.numel()
            total += count

            top_module = name.split('.')[0]
            by_module[top_module] = by_module.get(top_module, 0) + count

        return {
            "total": total,
            "by_module": by_module,
        }

    def get_layer_info(self) -> List[Dict]:
        """Get information about each layer."""
        layers = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                layers.append({
                    "name": name,
                    "type": type(module).__name__,
                    "params": params,
                })
        return layers

    def estimate_memory_mb(self, batch_size: int, seq_len: int) -> float:
        """Rough estimate of memory usage in MB.

        This is a simplified estimate based on parameters and activations.
        """
        # Parameters (4 bytes per float32)
        param_mem = self.get_param_count()["total"] * 4 / 1e6

        # Activation estimate (rough heuristic)
        d_model = 128  # Default assumption
        for name, param in self.model.named_parameters():
            if "embed" in name and param.dim() == 2:
                d_model = param.size(1)
                break

        # Assume activations ~= batch_size * seq_len * d_model * num_layers
        num_layers = sum(1 for _ in self.model.modules())
        act_mem = batch_size * seq_len * d_model * 4 * num_layers / 1e6

        return param_mem + act_mem

    def summary(self) -> str:
        """Generate a human-readable model summary."""
        params = self.get_param_count()
        buffers = self.get_buffer_count()

        lines = [
            "=" * 50,
            "MODEL SUMMARY",
            "=" * 50,
            f"Total parameters: {params['total']:,}",
            f"Trainable parameters: {params['trainable']:,}",
            f"Non-trainable parameters: {params['non_trainable']:,}",
            f"Total buffers: {buffers['total']:,}",
            "",
            "Parameters by module:",
        ]

        for module, count in params["by_module"].items():
            lines.append(f"  {module}: {count:,}")

        lines.append("=" * 50)
        return "\n".join(lines)


def log_metrics_wandb(logger: MetricsLogger, step: int):
    """Log metrics to Weights & Biases (if available)."""
    try:
        import wandb
        if wandb.run is not None:
            summary = logger.get_summary()
            log_dict = {}
            for name, stats in summary.items():
                if stats["last"] is not None:
                    log_dict[name] = stats["last"]
                if stats["ema"] is not None:
                    log_dict[f"{name}_ema"] = stats["ema"]
            wandb.log(log_dict, step=step)
    except ImportError:
        pass
