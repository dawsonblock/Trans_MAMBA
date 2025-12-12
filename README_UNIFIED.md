# Lean Infinity Dual Hybrid v1 (Trans_MAMBA)

This repository provides a single, unified core package: `trans_mamba_core/`.

Entrypoint:
- `python3 -m trans_mamba_core.unified_runner`

## Install / Setup

### CPU (recommended for quick verification)

```bash
python3 -m pip install -U pip
python3 -m pip install torch numpy tqdm einops
```

### CUDA (optional, for GPU)

```bash
python3 -m pip install -U pip
python3 -m pip install torch
python3 -m pip install causal-conv1d mamba-ssm --no-build-isolation
```

## Quick Verify (exactly 3 commands)

All three commands write artifacts to `--out_dir`.

### Command A — LM baseline (Transformer)

```bash
python3 -m trans_mamba_core.unified_runner \
  --mode lm --task copy_memory --controller transformer \
  --seq_len 256 --delay 64 --epochs 1 --batch_size 16 \
  --out_dir runs/validate_lm_transformer
```

### Command B — LM dualmem (MambaDualMem)

```bash
python3 -m trans_mamba_core.unified_runner \
  --mode lm --task copy_memory --controller mamba_dualmem \
  --seq_len 256 --delay 64 --epochs 1 --batch_size 16 \
  --d_model 128 --n_layers 2 --mem_slots 128 \
  --out_dir runs/validate_lm_dualmem
```

### Command C — RL delayed-cue dualmem

```bash
python3 -m trans_mamba_core.unified_runner \
  --mode rl --agent infinity --env delayed_cue \
  --num_envs 4 --num_updates 10 --rollout_length 64 \
  --d_model 128 --n_layers 2 --mem_slots 128 \
  --out_dir runs/validate_rl_dualmem
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
- `train_loss`
- `eval_loss`
- `accuracy`

### RL (`--mode rl`)
Each line is one update with keys:
- `update`
- `mean_return`
- `policy_loss`
- `value_loss`
- `entropy`
- `kl`
- `clipfrac`
