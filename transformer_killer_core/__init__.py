"""Transformer Killer Core v2.0 - Upgraded.

Research harness to compare memory-augmented sequence models:
  - Transformer decoder (baseline)
  - Mamba-style SSM backbone (with residuals)
  - Mamba + Dual-tier parametric memory (surprise-gated)
  - OT Memory Agent (Mamba + DualTierMiras + curiosity)

Synthetic tasks:
  - copy_memory: Long-range memory test
  - assoc_recall: Content-addressable memory
  - selective_copy: Attention filtering (NEW)
  - induction: Pattern completion (NEW)

Usage:
    python -m transformer_killer_core.unified_bench --sanity_check
    python -m transformer_killer_core.unified_bench --mode synthetic \\
        --task copy_memory --controller mamba_dualmem
"""

from .memory_core import (
    DualTierMiras,
    DualTierMirasConfig,
    LongMemKVCache,
    LongMemKVCacheConfig,
)
from .controllers import (
    ControllerConfig,
    build_controller,
    TransformerController,
    MambaController,
    MambaDualMemController,
)
from .synthetic_tasks import (
    CopyMemoryDataset,
    AssocRecallDataset,
    SelectiveCopyDataset,
    InductionHeadDataset,
    get_task_dataset,
)
from .ot_memory_agent import (
    OTMemoryAgent,
    OTMemoryAgentConfig,
    ReplayBuffer,
    build_ot_agent,
)
from .training_utils import (
    TrainingConfig,
    AMPTrainer,
    MemoryEfficientTrainer,
    CosineWarmupScheduler,
    LinearWarmupScheduler,
    create_optimizer,
    get_param_count,
)
from .metrics import (
    MetricsLogger,
    MemoryProfiler,
    ThroughputMeter,
    ModelAnalyzer,
)

__version__ = "2.2.0"

__all__ = [
    # Memory
    "DualTierMiras",
    "DualTierMirasConfig",
    "LongMemKVCache",
    "LongMemKVCacheConfig",
    # Controllers
    "ControllerConfig",
    "build_controller",
    "TransformerController",
    "MambaController",
    "MambaDualMemController",
    # Tasks
    "CopyMemoryDataset",
    "AssocRecallDataset",
    "SelectiveCopyDataset",
    "InductionHeadDataset",
    "get_task_dataset",
    # Agent
    "OTMemoryAgent",
    "OTMemoryAgentConfig",
    "ReplayBuffer",
    "build_ot_agent",
    # Training
    "TrainingConfig",
    "AMPTrainer",
    "MemoryEfficientTrainer",
    "CosineWarmupScheduler",
    "LinearWarmupScheduler",
    "create_optimizer",
    "get_param_count",
    # Metrics
    "MetricsLogger",
    "MemoryProfiler",
    "ThroughputMeter",
    "ModelAnalyzer",
]
