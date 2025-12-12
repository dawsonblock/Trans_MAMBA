"""
Tests for Anomaly Detection System.

Tests:
- Metric registry
- Anomaly rules (NaN/Inf, gradient explosion, advantage collapse, etc.)
- Severity classification
- Mitigation handlers
"""

import torch
import torch.nn as nn


def test_metric_registry():
    """Test metric registration and history."""
    from trans_mamba_core.monitoring.anomaly_detector import MetricRegistry

    registry = MetricRegistry(history_size=100)

    registry.register("loss", 1.0)
    registry.register("loss", 0.9)
    registry.register("loss", 0.8)

    latest = registry.get_latest("loss")
    assert latest is not None
    assert latest.value == 0.8

    history = registry.get_history("loss")
    assert len(history) == 3
    assert [m.value for m in history] == [1.0, 0.9, 0.8]

    print("✓ MetricRegistry test passed")


def test_tensor_stats():
    """Test tensor statistics computation."""
    from trans_mamba_core.monitoring.anomaly_detector import MetricRegistry

    registry = MetricRegistry()

    t = torch.randn(10, 10)
    metric = registry.register_tensor("activations", t)

    assert metric.tensor_stats is not None
    assert "mean" in metric.tensor_stats
    assert "std" in metric.tensor_stats
    assert "norm" in metric.tensor_stats
    assert not metric.tensor_stats["has_nan"]
    assert not metric.tensor_stats["has_inf"]

    t_nan = torch.tensor([1.0, float("nan"), 2.0])
    metric_nan = registry.register_tensor("bad_tensor", t_nan)
    assert metric_nan.tensor_stats["has_nan"]

    print("✓ Tensor stats test passed")


def test_nan_inf_rule():
    """Test NaN/Inf detection rule."""
    from trans_mamba_core.monitoring.anomaly_detector import (
        MetricRegistry,
        Severity,
    )
    from trans_mamba_core.monitoring.rules import NaNInfRule

    registry = MetricRegistry()
    rule = NaNInfRule()

    registry.register("loss", 1.0)
    anomaly = rule.check(registry)
    assert anomaly is None

    registry.register("loss", float("nan"))
    anomaly = rule.check(registry)
    assert anomaly is not None
    assert anomaly.severity == Severity.CRITICAL
    assert "NaN" in anomaly.message

    print("✓ NaN/Inf rule test passed")


def test_gradient_explosion_rule():
    """Test gradient explosion detection."""
    from trans_mamba_core.monitoring.anomaly_detector import (
        MetricRegistry,
        Severity,
    )
    from trans_mamba_core.monitoring.rules import GradientExplosionRule

    registry = MetricRegistry()
    rule = GradientExplosionRule(max_grad_norm=1.0, explosion_factor=5.0)

    registry.register("grad_norm", 2.0)
    anomaly = rule.check(registry)
    assert anomaly is None

    registry.register("grad_norm", 10.0)
    anomaly = rule.check(registry)
    assert anomaly is not None
    assert anomaly.severity == Severity.ERROR

    print("✓ Gradient explosion rule test passed")


def test_advantage_collapse_rule():
    """Test advantage collapse detection."""
    from trans_mamba_core.monitoring.anomaly_detector import (
        MetricRegistry,
        Severity,
    )
    from trans_mamba_core.monitoring.rules import AdvantageCollapseRule

    registry = MetricRegistry()
    rule = AdvantageCollapseRule(min_std=1e-5, window_size=5)

    for _ in range(5):
        registry.register("advantage_std", 1e-6)

    anomaly = rule.check(registry)
    assert anomaly is not None
    assert anomaly.severity == Severity.WARNING

    registry = MetricRegistry()
    rule = AdvantageCollapseRule(min_std=1e-5, window_size=5)

    for _ in range(5):
        registry.register("advantage_std", 0.1)

    anomaly = rule.check(registry)
    assert anomaly is None

    print("✓ Advantage collapse rule test passed")


def test_policy_collapse_rule():
    """Test policy collapse detection."""
    from trans_mamba_core.monitoring.anomaly_detector import (
        MetricRegistry,
        Severity,
    )
    from trans_mamba_core.monitoring.rules import PolicyCollapseRule

    registry = MetricRegistry()
    rule = PolicyCollapseRule(threshold=0.99, window_size=5)

    for _ in range(5):
        registry.register("max_action_prob", 0.999)

    anomaly = rule.check(registry)
    assert anomaly is not None
    assert anomaly.severity == Severity.WARNING

    print("✓ Policy collapse rule test passed")


def test_statistical_rule():
    """Test statistical deviation rule."""
    from trans_mamba_core.monitoring.anomaly_detector import MetricRegistry
    from trans_mamba_core.monitoring.rules import StatisticalRule

    registry = MetricRegistry()
    rule = StatisticalRule(
        name="LossSpike",
        metric_name="loss",
        z_threshold=3.0,
        min_samples=10,
    )

    for i in range(10):
        registry.register("loss", 1.0 + 0.01 * i)

    anomaly = rule.check(registry)
    assert anomaly is None

    registry.register("loss", 100.0)
    anomaly = rule.check(registry)
    assert anomaly is not None

    print("✓ Statistical rule test passed")


def test_anomaly_detector_full():
    """Test full anomaly detector with rules and handlers."""
    from trans_mamba_core.monitoring.anomaly_detector import (
        AnomalyDetector,
        AnomalyDetectorConfig,
        Severity,
    )
    from trans_mamba_core.monitoring.rules import get_default_rules

    cfg = AnomalyDetectorConfig(enabled=True, verbose=False)
    detector = AnomalyDetector(cfg)
    detector.add_rules(get_default_rules())

    handled_anomalies = []

    def handler(anomaly):
        handled_anomalies.append(anomaly)

    detector.add_mitigation_handler(Severity.CRITICAL, handler)

    detector.register_metric("loss", 1.0)
    detector.check()
    assert len(handled_anomalies) == 0

    detector.register_metric("loss", float("nan"))
    detector.check()
    assert len(handled_anomalies) == 1
    assert handled_anomalies[0].severity == Severity.CRITICAL

    summary = detector.get_summary()
    assert summary["total_anomalies"] >= 1
    assert summary["by_severity"]["CRITICAL"] >= 1

    print("✓ Full anomaly detector test passed")


def test_check_tensor_health():
    """Test quick tensor health check utility."""
    from trans_mamba_core.monitoring.anomaly_detector import (
        check_tensor_health,
    )

    t_good = torch.randn(10, 10)
    anomaly = check_tensor_health(t_good, "test_tensor")
    assert anomaly is None

    t_nan = torch.tensor([1.0, float("nan"), 2.0])
    anomaly = check_tensor_health(t_nan, "nan_tensor")
    assert anomaly is not None
    assert "NaN" in anomaly.message

    t_inf = torch.tensor([1.0, float("inf"), 2.0])
    anomaly = check_tensor_health(t_inf, "inf_tensor")
    assert anomaly is not None
    assert "Inf" in anomaly.message

    print("✓ check_tensor_health test passed")


def test_mitigation_handlers():
    """Test mitigation handler execution."""
    from trans_mamba_core.monitoring.anomaly_detector import Anomaly, Severity
    from trans_mamba_core.monitoring.mitigations import (
        LogOnlyHandler,
        SkipUpdateHandler,
        GradientClipHandler,
    )

    anomaly = Anomaly(
        rule_name="TestRule",
        metric_name="test_metric",
        severity=Severity.WARNING,
        message="Test anomaly",
        metric_value=1.0,
    )

    log_handler = LogOnlyHandler()
    result = log_handler(anomaly)
    assert result.success

    skip_handler = SkipUpdateHandler()
    context = {}
    result = skip_handler.handle(anomaly, context)
    assert result.success
    assert context.get("skip_update")

    model = nn.Linear(10, 10)
    x = torch.randn(1, 10)
    y = model(x)
    y.sum().backward()

    clip_handler = GradientClipHandler(clip_value=0.01)
    context = {"model": model}
    result = clip_handler(anomaly, context)
    assert result.success

    print("✓ Mitigation handlers test passed")


if __name__ == "__main__":
    test_metric_registry()
    test_tensor_stats()
    test_nan_inf_rule()
    test_gradient_explosion_rule()
    test_advantage_collapse_rule()
    test_policy_collapse_rule()
    test_statistical_rule()
    test_anomaly_detector_full()
    test_check_tensor_health()
    test_mitigation_handlers()

    print()
    print("=" * 50)
    print("ALL ANOMALY DETECTOR TESTS PASSED")
    print("=" * 50)
