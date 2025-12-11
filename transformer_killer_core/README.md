# Transformer Killer Core v2.0

> A research harness for memory-augmented sequence models that outperform Transformers on long-horizon tasks.

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)]()

---

## Table of Contents

1. [Overview](#overview)
2. [What's New in v2.0](#whats-new-in-v20)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Architecture](#architecture)
6. [Usage Guide](#usage-guide)
7. [API Reference](#api-reference)
8. [Synthetic Tasks](#synthetic-tasks)
9. [Configuration](#configuration)
10. [Advanced Features](#advanced-features)
11. [Troubleshooting](#troubleshooting)

---

## Overview

Transformer Killer Core is a minimal, self-contained research harness for comparing:

| Controller | Description | Use Case |
|------------|-------------|----------|
| **Transformer** | Standard decoder-only Transformer | Baseline |
| **Mamba** | SSM backbone (Mamba2 or GRU fallback) | Efficient sequences |
| **MambaDualMem** | Mamba + DualTierMiras memory | Long-horizon memory |
| **OTMemoryAgent** | Mamba + Memory + Curiosity | RL-ready agent |

**Key Features:**
- O(1) memory footprint per environment (constant, not O(T))
- Content-addressable retrieval via cosine similarity
- Surprise-gated consolidation (Titans-inspired)
- Drop-in Mamba2 CUDA support when available

---

## What's New in v2.0

### Memory Upgrades
- **Surprise-gated deep tier**: Only consolidate surprising inputs
- **Temperature-scaled attention**: Control retrieval sharpness
- **EMA updates**: Smooth memory transitions
- **Attention entropy tracking**: Monitor memory focus

### Architecture Upgrades
- **Residual connections**: Better gradient flow in Mamba backbone
- **Gated fusion**: Learned blending of backbone + memory
- **Gradient checkpointing**: Trade compute for memory

### Agent Upgrades
- **Curiosity module**: Prediction-error intrinsic motivation
- **Improved value head**: 2-layer MLP for RL
- **Diagnostic API**: Track all internal stats

### New Tasks
- **SelectiveCopy**: Copy only marked tokens (attention filtering)
- **InductionHead**: Pattern completion (in-context learning)

---

## Quick Start

```python
import torch
from transformer_killer_core import (
    ControllerConfig, build_controller,
    OTMemoryAgent, OTMemoryAgentConfig,
    get_task_dataset
)

# 1. Build a memory-augmented controller
cfg = ControllerConfig(
    controller_type="mamba_dualmem",
    vocab_size=256,
    d_model=128,
    n_layers=2,
)
model = build_controller(cfg)

# 2. Forward pass
x = torch.randint(0, 256, (4, 64))  # [batch, seq_len]
logits = model(x)  # [batch, seq_len, vocab_size]

# 3. Train on synthetic task
dataset = get_task_dataset("copy_memory", seq_len=100, delay=40)
```

---

## Installation

### Option 1: Google Colab (Recommended)

```python
# Upload unified_transformer_mamba_core.zip, then:
!unzip unified_transformer_mamba_core.zip -d /content
%cd /content/unified_transformer_mamba_core
!python setup_colab.py --install-all
```

### Option 2: Local Installation

```bash
git clone <repo_url>
cd unified_transformer_mamba_core
pip install -r transformer_killer_core/requirements.txt

# Optional: Install Mamba CUDA (Linux + NVIDIA GPU only)
cd external/mamba_ssm && pip install -e . && cd ../..
```

### Option 3: Shell Script (Linux)

```bash
chmod +x setup.sh
./setup.sh
```

### Verify Installation

```bash
python -m transformer_killer_core.unified_bench --sanity_check
```

Expected output:
```
ALL SANITY CHECKS PASSED
```

---

## Architecture

### Memory System: DualTierMiras

```
┌─────────────────────────────────────────────────────────┐
│                    DualTierMiras                        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐                     │
│  │  Fast Tier  │    │  Deep Tier  │                     │
│  │  (Miras)    │    │  (Titans)   │                     │
│  │             │    │             │                     │
│  │ Updates     │    │ Surprise-   │                     │
│  │ every write │    │ gated       │                     │
│  └──────┬──────┘    └──────┬──────┘                     │
│         │                  │                            │
│         └────────┬─────────┘                            │
│                  ▼                                      │
│         ┌───────────────┐                               │
│         │ Learned Gate  │  w_fast * v_fast +            │
│         │ (mix_logit)   │  (1 - w_fast) * v_deep        │
│         └───────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

### Controller Pipeline

```
Input [B, T] ──► Embed ──► PosEnc ──► Backbone ──► Memory ──► Head ──► Logits [B, T, V]
                                        │           │
                                        │     ┌─────▼─────┐
                                        │     │   Gated   │
                                        └────►│   Fusion  │
                                              └───────────┘
```

---

## Usage Guide

### CLI Commands

#### Sanity Check
```bash
python -m transformer_killer_core.unified_bench --sanity_check
```

#### Synthetic Benchmarks
```bash
# Copy Memory Task
python -m transformer_killer_core.unified_bench \
    --mode synthetic \
    --task copy_memory \
    --controller mamba_dualmem \
    --seq_len 100 \
    --delay 40 \
    --epochs 20 \
    --device cuda

# Associative Recall
python -m transformer_killer_core.unified_bench \
    --mode synthetic \
    --task assoc_recall \
    --controller ot_agent \
    --seq_len 50 \
    --num_pairs 8 \
    --epochs 20

# Selective Copy (NEW)
python -m transformer_killer_core.unified_bench \
    --mode synthetic \
    --task selective_copy \
    --controller mamba_dualmem \
    --seq_len 50 \
    --epochs 20

# Induction Head (NEW)
python -m transformer_killer_core.unified_bench \
    --mode synthetic \
    --task induction \
    --controller mamba_dualmem \
    --seq_len 50 \
    --epochs 20
```

#### Language Modeling
```bash
python -m transformer_killer_core.unified_bench \
    --mode lm \
    --controller mamba_dualmem \
    --data_path /path/to/corpus.txt \
    --seq_len 256 \
    --epochs 10
```

#### Compare All Controllers
```bash
for ctrl in transformer mamba mamba_dualmem ot_agent; do
    python -m transformer_killer_core.unified_bench \
        --mode synthetic --task copy_memory \
        --controller $ctrl --epochs 20
done
```

---

## API Reference

### Memory Core

```python
from transformer_killer_core import DualTierMiras, DualTierMirasConfig

cfg = DualTierMirasConfig(
    d_model=128,           # Hidden dimension
    mem_slots=64,          # Slots per tier
    temperature=1.0,       # Attention sharpness (lower = sharper)
    surprise_threshold=0.5,# Deep tier write threshold
    use_ema=False,         # EMA updates
    ema_decay=0.99,        # EMA decay factor
)
memory = DualTierMiras(cfg)

# Reset at episode start
memory.reset_parameters()

# Read from memory
query = torch.randn(batch_size, d_model)
out = memory.read(query, context=query)
# out["v"]           - blended value
# out["v_fast"]      - fast tier value
# out["v_deep"]      - deep tier value
# out["w_fast"]      - fast tier weight
# out["attn_entropy"]- attention entropy

# Write to memory
stats = memory.update(key, value, weight=None, context=key)
# stats["mem_surprise/mean"] - average surprise
# stats["mem_deep/num_writes"] - deep tier write count
```

### Controllers

```python
from transformer_killer_core import ControllerConfig, build_controller

cfg = ControllerConfig(
    controller_type="mamba_dualmem",  # or "transformer", "mamba", "ot_agent"
    vocab_size=256,
    d_model=128,
    n_layers=2,
    n_heads=4,                        # Transformer only
    max_seq_len=256,
    dropout=0.1,
    mem_slots=64,                     # Memory controllers only
    temperature=1.0,                  # Memory controllers only
    use_gradient_checkpointing=False, # Save memory
)
model = build_controller(cfg)

# Forward
logits = model(token_ids)  # [B, T] -> [B, T, V]

# Enable trainable memory (for research)
model.trainable_memory = True

# Get memory statistics
stats = model.get_memory_stats()
```

### OT Memory Agent

```python
from transformer_killer_core import OTMemoryAgent, OTMemoryAgentConfig

cfg = OTMemoryAgentConfig(
    vocab_size=256,
    d_model=128,
    n_layers=2,
    mem_slots=64,
    use_curiosity=True,       # Enable curiosity module
    use_external_cache=False, # Enable LongMemKVCache
    temperature=0.5,
    surprise_threshold=0.3,
)
agent = OTMemoryAgent(cfg)

# Language modeling mode
logits = agent(token_ids)  # [B, T, V]

# RL mode (step-by-step)
agent.reset_memory()
for obs in observations:
    logits, value = agent.step(obs)  # [B, V], [B]

# Diagnostics
diag = agent.get_diagnostics()
print(f"Curiosity: {diag['curiosity']}")
print(f"Params: {agent.get_num_params():,}")
```

### Synthetic Tasks

```python
from transformer_killer_core import get_task_dataset

# Factory function
ds = get_task_dataset(
    "copy_memory",    # or "assoc_recall", "selective_copy", "induction"
    seq_len=100,
    delay=40,         # copy_memory only
    num_samples=10000,
)

# Direct class usage
from transformer_killer_core import (
    CopyMemoryDataset,
    AssocRecallDataset,
    SelectiveCopyDataset,
    InductionHeadDataset,
)

# Get task difficulty metrics
print(ds.get_difficulty())
# {'task': 'copy_memory', 'delay': 40, 'memory_span': 50, ...}
```

---

## Synthetic Tasks

| Task | Description | Tests |
|------|-------------|-------|
| **copy_memory** | Copy tokens after delay | Long-range memory |
| **assoc_recall** | Key-value retrieval | Content-addressable memory |
| **selective_copy** | Copy only marked tokens | Attention filtering |
| **induction** | Complete [A,B,...,A,?] → B | In-context learning |

---

## Configuration

### DualTierMirasConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | required | Hidden dimension |
| `mem_slots` | 64 | Slots per memory tier |
| `lr_fast` | 1.0 | Fast tier learning scale |
| `lr_deep` | 0.5 | Deep tier learning scale |
| `temperature` | 1.0 | Attention temperature |
| `surprise_threshold` | 0.5 | Deep write threshold |
| `use_ema` | False | Enable EMA updates |
| `ema_decay` | 0.99 | EMA decay factor |

### ControllerConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `controller_type` | required | "transformer", "mamba", "mamba_dualmem", "ot_agent" |
| `vocab_size` | required | Vocabulary size |
| `d_model` | 128 | Hidden dimension |
| `n_layers` | 2 | Number of layers |
| `n_heads` | 4 | Attention heads (transformer only) |
| `max_seq_len` | 256 | Max sequence length |
| `dropout` | 0.1 | Dropout rate |
| `mem_slots` | 64 | Memory slots |
| `temperature` | 1.0 | Memory attention temperature |
| `use_gradient_checkpointing` | False | Save memory during training |

---

## Advanced Features

### Trainable Memory

```python
model = build_controller(cfg)
model.trainable_memory = True  # Backprop through memory reads

# Now mix_logit and context_gate will receive gradients
```

### Curiosity-Driven Memory

```python
cfg = OTMemoryAgentConfig(
    vocab_size=256,
    d_model=128,
    use_curiosity=True,  # Enable prediction-error curiosity
)
agent = OTMemoryAgent(cfg)

# Curiosity score available in diagnostics
diag = agent.get_diagnostics()
print(f"Curiosity: {diag['curiosity']}")
```

### External Long-Term Memory

```python
cfg = OTMemoryAgentConfig(
    vocab_size=256,
    d_model=128,
    use_external_cache=True,
    cache_capacity=4096,
)
agent = OTMemoryAgent(cfg)
# Agent now blends DualTierMiras with LongMemKVCache
```

### Gradient Checkpointing

```python
cfg = ControllerConfig(
    controller_type="mamba_dualmem",
    vocab_size=256,
    d_model=256,
    n_layers=6,
    use_gradient_checkpointing=True,  # Save ~40% memory
)
```

---

## Troubleshooting

### "Mamba2 not found, using GRU fallback"

This is normal if mamba-ssm isn't installed. To install:

```bash
# Linux + NVIDIA GPU only
pip install mamba-ssm
```

### "CUDA out of memory"

1. Reduce `d_model` or `n_layers`
2. Enable gradient checkpointing
3. Reduce batch size or sequence length

### "NaN in loss"

1. Check learning rate (try 1e-4)
2. Add gradient clipping
3. Reduce `temperature` if using very small values

### Import errors

```bash
# Verify installation
python -c "import transformer_killer_core; print(transformer_killer_core.__version__)"
# Should print: 2.0.0
```

---

## File Structure

```
transformer_killer_core/
├── __init__.py              # Package exports (16 items)
├── memory_core.py           # DualTierMiras, LongMemKVCache
├── controllers.py           # All controller implementations
├── ot_memory_agent.py       # OTMemoryAgent for RL
├── synthetic_tasks.py       # 4 benchmark tasks
├── unified_bench.py         # CLI entry point
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{transformer_killer_core,
  title={Transformer Killer Core: Memory-Augmented Sequence Models},
  version={2.0.0},
  year={2024}
}
```

---

## License

MIT License
