"""
Runtime Anomaly Detection & Monitoring Framework.

Provides:
- Metric registry for tracking training/inference metrics
- Anomaly rules engine with threshold, statistical, and pattern rules
- Hooks/callbacks for integration with training loops
- Severity-based mitigation handlers
"""

from .anomaly_detector import (
    AnomalyDetector,
    AnomalyDetectorConfig,
    Metric,
    Anomaly,
    Severity,
)
from .rules import (
    AnomalyRule,
    ThresholdRule,
    StatisticalRule,
    PatternRule,
    NaNInfRule,
    GradientExplosionRule,
    AdvantageCollapseRule,
    PolicyCollapseRule,
    MemorySaturationRule,
    RuntimePerformanceRule,
    get_default_rules,
)
from .hooks import (
    MonitoringHook,
    TrainingHook,
    PPOHook,
    MemoryHook,
    ControllerHook,
)
from .mitigations import (
    MitigationAction,
    MitigationHandler,
    LogOnlyHandler,
    ResetMemoryHandler,
    ResetEnvHandler,
    SkipUpdateHandler,
    ReduceLRHandler,
    GradientClipHandler,
    AbortTrainingHandler,
)
from .integration import (
    create_detector,
    create_lm_monitor,
    create_ppo_monitor,
    load_config,
    MonitoredTrainer,
    MonitoredPPOTrainer,
)

__all__ = [
    "AnomalyDetector",
    "AnomalyDetectorConfig",
    "Metric",
    "Anomaly",
    "Severity",
    "create_detector",
    "create_lm_monitor",
    "create_ppo_monitor",
    "load_config",
    "MonitoredTrainer",
    "MonitoredPPOTrainer",
    "AnomalyRule",
    "ThresholdRule",
    "StatisticalRule",
    "PatternRule",
    "NaNInfRule",
    "GradientExplosionRule",
    "AdvantageCollapseRule",
    "PolicyCollapseRule",
    "MemorySaturationRule",
    "RuntimePerformanceRule",
    "get_default_rules",
    "MonitoringHook",
    "TrainingHook",
    "PPOHook",
    "MemoryHook",
    "ControllerHook",
    "MitigationAction",
    "MitigationHandler",
    "LogOnlyHandler",
    "ResetMemoryHandler",
    "ResetEnvHandler",
    "SkipUpdateHandler",
    "ReduceLRHandler",
    "GradientClipHandler",
    "AbortTrainingHandler",
]
