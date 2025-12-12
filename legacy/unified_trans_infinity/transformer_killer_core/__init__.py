"""
Transformer Killer Core - Unified benchmark for sequence models.

Controllers:
- TransformerController
- MambaController  
- MambaDualMemController

Tasks:
- CopyMemoryDataset
- AssocRecallDataset
- SelectiveCopyDataset
- InductionHeadDataset
"""

from .controllers import (
    TransformerController,
    MambaController,
    MambaDualMemController,
    ControllerConfig,
)

from .synthetic_tasks import (
    CopyMemoryDataset,
    AssocRecallDataset,
    SelectiveCopyDataset,
    InductionHeadDataset,
    SyntheticTaskConfig,
)

from .training_utils import (
    AMPTrainer,
    get_scheduler,
    get_optimizer,
)

from .metrics import (
    MetricsLogger,
    MemoryProfiler,
    ThroughputMeter,
)

__all__ = [
    # Controllers
    "TransformerController",
    "MambaController",
    "MambaDualMemController",
    "ControllerConfig",
    # Tasks
    "CopyMemoryDataset",
    "AssocRecallDataset",
    "SelectiveCopyDataset",
    "InductionHeadDataset",
    "SyntheticTaskConfig",
    # Training
    "AMPTrainer",
    "get_scheduler",
    "get_optimizer",
    # Metrics
    "MetricsLogger",
    "MemoryProfiler",
    "ThroughputMeter",
]
