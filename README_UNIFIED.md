# Unified Transformer–Mamba Core

This repository combines:

1. **Transformer Killer Core** – A benchmark harness for comparing Transformer, Mamba, and Mamba+DualTierMiras controllers on synthetic tasks and language modeling.
2. **OT Memory Agent** – Unified 0T Memory Agent using canonical core components.
3. **External Mamba SSM** – The official Mamba2 CUDA repository (optional, for real Mamba2 kernels).

## Quick Start (Google Colab)

```python
# 1. Upload unified_transformer_mamba_core.zip
# 2. Run:
!unzip unified_transformer_mamba_core.zip -d /content
%cd /content/unified_transformer_mamba_core
!python setup_colab.py --install-all
```

## Quick Start (Linux with NVIDIA GPU)

```bash
# Clone/extract the repo, then:
cd unified_transformer_mamba_core
chmod +x setup.sh
./setup.sh
```

## Directory Layout

```
unified_transformer_mamba_core/
├── setup_colab.py              # Unified Python setup script
├── setup.sh                    # Unified Bash setup script
├── transformer_killer_colab.ipynb  # Ready-to-run Colab notebook
├── transformer_killer_core/    # Benchmarks + controllers + memory
│   ├── memory_core.py          # DualTierMiras, LongMemKVCache
│   ├── controllers.py          # Transformer, Mamba, MambaDualMem
│   ├── ot_memory_agent.py      # OTMemoryAgent (NEW)
│   ├── unified_bench.py        # Unified CLI for all benchmarks (NEW)
│   ├── synthetic_tasks.py      # CopyMemory, AssocRecall datasets
│   ├── transformer_killer_bench.py  # Legacy synthetic CLI
│   ├── language_model_bench.py      # Legacy char-level LM CLI
│   └── requirements.txt
├── ot_memory_agent/            # Blueprint OT memory agents (legacy)
│   ├── ot_memory_agent_unified.py
│   ├── ot_memory_agent_local.py
│   └── ot_memory_agent_mamba.py
├── external/
│   ├── mamba_ssm/              # Full Mamba2 CUDA repo
│   └── transformer_mamba_llm_research/
└── README_UNIFIED.md           # This file
```

---

## 1. Synthetic Benchmarks (Copy-Memory, Assoc Recall)

From **inside Colab**:

```bash
!unzip unified_transformer_mamba_core.zip -d /content
%cd /content/unified_transformer_mamba_core
!pip install -r transformer_killer_core/requirements.txt
# Optional: real Mamba2 (GPU + CUDA):
# %cd external/mamba_ssm && pip install -e . && cd /content/unified_transformer_mamba_core
```

Then run, from `unified_transformer_mamba_core/`:

```bash
python -m transformer_killer_core.transformer_killer_bench   --task copy_memory   --controller transformer   --seq_len 100   --delay 40   --epochs 10   --device cuda

python -m transformer_killer_core.transformer_killer_bench   --task copy_memory   --controller mamba_dualmem   --seq_len 100   --delay 40   --epochs 10   --device cuda
```

Associative recall:

```bash
python -m transformer_killer_core.transformer_killer_bench   --task assoc_recall   --controller mamba_dualmem   --seq_len 30   --num_pairs 6   --epochs 10   --device cuda
```

This harness compares:
- Transformer baseline
- Mamba backbone
- Mamba + DualTierMiras parametric memory

on hard long-horizon synthetic tasks.

---

## 2. Character-level Language Modeling

Prepare a text file (e.g. `/content/my_corpus.txt`), then from
`unified_transformer_mamba_core/`:

```bash
python -m transformer_killer_core.language_model_bench   --controller transformer   --data_path /content/my_corpus.txt   --seq_len 256   --epochs 5   --device cuda

python -m transformer_killer_core.language_model_bench   --controller mamba_dualmem   --data_path /content/my_corpus.txt   --seq_len 256   --epochs 5   --device cuda
```

You get:
- Cross-entropy + PPL
- Training time per epoch
- Direct Transformer vs Mamba vs Mamba+DualMem comparison.

---

## 3. Real Mamba2 Kernels (optional, GPU-only)

The `external/mamba_ssm/` folder contains the full Mamba2 implementation.
To compile and use the CUDA kernels:

```bash
%cd /content/unified_transformer_mamba_core/external/mamba_ssm
pip install -e .
pytest tests/ops/test_selective_scan.py -q  # optional
%cd /content/unified_transformer_mamba_core
```

Once installed, `transformer_killer_core.controllers.MambaBackbone` will
automatically use the real `Mamba2` instead of the GRU fallback.

---

## 4. OT / 0T Memory Agent Blueprints

The `ot_memory_agent/` folder contains three Python scripts:

- `ot_memory_agent_unified.py`
- `ot_memory_agent_local.py`
- `ot_memory_agent_mamba.py`

These are ready-made blueprints (copied directly from your TXT designs)
for building the **0T Memory Agent** and integrating:

- Mamba backbone
- Dual-tier parametric memory (`DualTierMiras`)
- External long-term memory (IVF/FAISS in the full design)
- RL / sequence controllers

They are not wired into the synthetic harness by default, but they live
in the same repo so you can:

- Import `transformer_killer_core.memory_core` from inside the OT agent.
- Use the `MambaBackbone` as the parametric core.
- Plug into RL or sequence tasks as you expand the project.

---

## 5. Where to Start

1. **Sanity-check synthetic benchmarks**  
   Run copy-memory and assoc-recall for Transformer vs Mamba vs
   Mamba+DualMem and log:
   - final accuracy
   - epochs / wall-clock to convergence.

2. **Run LM benchmark on a small corpus**  
   Compare Transformer vs Mamba+DualMem perplexity and training time.

3. **Then integrate OT agent**  
   Use `ot_memory_agent_unified.py` as the master controller and
   replace its internal memory / backbone hooks with imports from
   `transformer_killer_core.memory_core` and `transformer_killer_core.controllers`.

This unified build is the foundation for a **world-class Transformer killer**:
synthetic + LM benchmarks, real Mamba kernels, and your 0T agent logic
all in one place.
