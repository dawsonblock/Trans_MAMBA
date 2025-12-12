# Lean Infinity Dual Hybrid v1 (Trans_MAMBA)

This repository provides a single, unified core package: `trans_mamba_core/`.

Entrypoint:
- `python -m trans_mamba_core.unified_runner`

## Install / Setup

### CPU (recommended for quick verification)

```bash
python -m pip install -U pip
python -m pip install torch numpy tqdm einops
```

### CUDA (optional, for GPU)

```bash
python -m pip install -U pip
python -m pip install torch
python -m pip install causal-conv1d mamba-ssm --no-build-isolation
```

## Quick Verify (exactly 3 commands)

All three commands write artifacts to `--out_dir`.

### Command A — LM baseline (Transformer)

```bash
python -m trans_mamba_core.unified_runner \
  --mode lm --task copy_memory --controller transformer \
  --seq_len 256 --delay 64 --epochs 3 --batch_size 32 \
  --out_dir runs/lm_transformer_copy
```

### Command B — LM dualmem (MambaDualMem)

```bash
python -m trans_mamba_core.unified_runner \
  --mode lm --task copy_memory --controller mamba_dualmem \
  --seq_len 256 --delay 64 --epochs 3 --batch_size 32 \
  --d_model 128 --n_layers 2 --mem_slots 256 \
  --out_dir runs/lm_dualmem_copy
```

### Command C — RL delayed-cue dualmem

```bash
python -m trans_mamba_core.unified_runner \
  --mode rl --agent infinity --env delayed_cue \
  --num_envs 8 --num_updates 50 --rollout_length 128 \
  --d_model 128 --n_layers 2 --mem_slots 256 \
  --out_dir runs/rl_dualmem_delayedcue
```

## Expected Artifacts

Each `--out_dir` contains:
- `config.json`
- `metrics.jsonl`
- `final.pt` (model/agent weights)

## Expected `metrics.jsonl` keys

### LM (`--mode lm`)
Each line is one epoch with keys:
- `epoch`
- `loss`
- `accuracy`

### RL (`--mode rl`)
Each line is one update with keys:
- `update`
- `rollout_reward`
- `mean_return`
- `loss`
- `policy_loss`
- `value_loss`
- `entropy`
- `approx_kl`
- `clipfrac`
