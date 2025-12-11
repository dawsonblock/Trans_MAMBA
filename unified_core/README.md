# Trans_MAMBA + Infinity Unified Research Framework

A unified, modular, research-grade framework for long-horizon memory AI benchmarking.

## Architecture

```
unified_core/
├── memory/           # DualTierMiras + KV Cache
├── controllers/      # Transformer, Mamba, MambaDualMem, StreamingSSM
├── synthetic/        # CopyMemory, AssocRecall, SelectiveCopy, InductionHead
├── rl/               # InfinityAgent, OTMemoryAgent, PPO, Environments
├── tests/            # Test suite
├── unified_runner.py # CLI for all experiments
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### LM Mode: Synthetic Tasks

```bash
# Transformer on CopyMemory
python unified_runner.py --mode lm --task copy_memory --controller transformer

# Mamba on AssocRecall
python unified_runner.py --mode lm --task assoc_recall --controller mamba

# MambaDualMem on SelectiveCopy
python unified_runner.py --mode lm --task selective_copy --controller mamba_dualmem
```

### RL Mode: Reinforcement Learning

```bash
# Infinity Agent on CartPole
python unified_runner.py --mode rl --agent infinity --env cartpole

# OT Agent on Delayed-Cue
python unified_runner.py --mode rl --agent ot --env delayed_cue
```

## Components

### Memory Systems

**DualTierMiras**: Canonical dual-tier parametric memory with surprise gating.
- Fast tier: Always updated, captures short-term patterns
- Deep tier: Surprise-gated, consolidates long-term knowledge
- Cosine similarity retrieval with optional sparse top-k

```python
from memory import DualTierMiras, DualTierMirasConfig

cfg = DualTierMirasConfig(d_model=256, mem_slots=64, n_heads=4)
memory = DualTierMiras(cfg)

output, new_state, aux = memory(query, write_value=value, state=state)
```

### Controllers

| Controller | Description | Use Case |
|------------|-------------|----------|
| Transformer | Decoder-only with causal attention | LM baseline |
| Mamba | SSM backbone | LM + RL |
| MambaDualMem | Mamba + DualTierMiras | Long-horizon tasks |
| StreamingSSM | Constant-memory recurrent | 0T agent, RL |

### Synthetic Tasks

| Task | Description | Memory Requirement |
|------|-------------|-------------------|
| CopyMemory | Memorize, then recall after delay | Short-term |
| AssocRecall | Key-value association lookup | Associative |
| SelectiveCopy | Copy only marked tokens | Selective |
| InductionHead | Pattern completion (A B ... A → B) | In-context |

### RL Agents

**InfinityAgent**: Mamba + DualTierMiras for long-horizon RL
- Combines SSM efficiency with parametric memory
- Surprise-gated memory consolidation

**OTMemoryAgent**: StreamingSSM backbone
- Constant O(D) memory footprint
- Linear O(T) time complexity
- No attention, no KV cache

## Testing

```bash
cd unified_core

# Run all tests
python -m tests.test_memory
python -m tests.test_controllers
python -m tests.test_rl_agents
python -m tests.test_synthetic
```

## Configuration

All components use dataclass configs:

```python
from controllers import MambaDualMemConfig

cfg = MambaDualMemConfig(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    mem_slots=64,
    surprise_threshold=0.5,
)
```

## API Reference

### Memory

```python
output, new_state, aux = memory(
    query,        # [B, d_model]
    write_value,  # [B, d_model] optional
    write_mask,   # [B] optional
    state,        # MemoryState
)
```

### Controllers

```python
# LM mode
logits = model(token_ids)  # [B, T, vocab_size]

# RL mode
features = model.get_features(x)  # [B, T, d_model]
```

### RL Agents

```python
logits, values, new_state = agent(obs, state)
```

## Citation

```bibtex
@software{transmamba_infinity,
  title={Trans_MAMBA + Infinity: Unified Research Framework},
  author={Block, Dawson},
  year={2024},
  url={https://github.com/dawsonblock/Trans_MAMBA}
}
```

## License

MIT
