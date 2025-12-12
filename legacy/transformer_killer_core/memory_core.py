"""
memory_core.py

Canonical memory implementations for the Transformer Killer Core.

This module provides:
    - DualTierMiras: Dual-tier content-addressable parametric memory
    - LongMemKVCache: Simple external KV cache with cosine similarity retrieval

These are the single source of truth for memory implementations.
All other modules (OT Memory Agent, controllers, etc.) should import from here.
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DualTierMirasConfig:
    """Configuration for DualTierMiras memory.

    Attributes:
        d_model: Dimension of keys and values.
        mem_slots: Number of memory slots per tier (default: 64).
        lr_fast: Learning rate scale for fast tier updates.
        lr_deep: Learning rate scale for deep tier updates.
        temperature: Softmax temperature for attention (lower=sharper).
        surprise_threshold: Min surprise to trigger deep write.
        use_ema: Use exponential moving average for updates.
        ema_decay: EMA decay factor (if use_ema=True).
        n_heads: Number of attention heads for multi-head retrieval.
        use_decay: Enable memory decay (forgetting).
        decay_rate: Per-step decay factor for memory values.
        use_adaptive_retrieval: Scale retrieval by confidence.
        use_query_proj: Project queries before matching.
    """
    d_model: int
    mem_slots: int = 64
    lr_fast: float = 1.0
    lr_deep: float = 0.5
    temperature: float = 1.0
    surprise_threshold: float = 0.5
    use_ema: bool = False
    ema_decay: float = 0.99
    # New v2.1 features
    n_heads: int = 1
    use_decay: bool = False
    decay_rate: float = 0.999
    use_adaptive_retrieval: bool = True
    use_query_proj: bool = True
    # v2.2 features
    use_sparse_attention: bool = False
    top_k_retrieval: int = 8
    use_memory_compression: bool = False
    compression_ratio: float = 0.5


@dataclass
class LongMemKVCacheConfig:
    """Configuration for LongMemKVCache.

    Attributes:
        key_dim: Dimension of keys.
        value_dim: Dimension of values.
        capacity: Maximum number of entries (default: 4096).
    """
    key_dim: int
    value_dim: int
    capacity: int = 4096


class DualTierMiras(nn.Module):
    """Dual-tier content-addressable parametric memory (Upgraded).

    Architecture:
        - **Fast tier (Miras)**: Updates every write, captures recent.
        - **Deep tier (Titans)**: Surprise-gated writes for consolidation.
        - **Cosine-similarity reads**: Temperature-scaled attention.
        - **Learnable mixing gate**: Scalar + context-dependent blending.
        - **Surprise gating**: Only write to deep tier on surprising inputs.
        - **EMA updates**: Optional exponential moving average writes.

    Memory Layout:
        Each tier has M slots, each slot stores (key, value) of dim d.
        Ring-buffer writes ensure fixed memory footprint.

    Usage:
        >>> cfg = DualTierMirasConfig(d_model=128, mem_slots=64)
        >>> mem = DualTierMiras(cfg)
        >>> mem.reset_parameters()  # Call at episode/sequence start
        >>> out = mem.read(query)   # Read from memory
        >>> mem.update(key, value)  # Write to memory

    Note:
        Memory updates are under torch.no_grad() by default.
        For differentiable memory, wrap reads without no_grad.
    """

    def __init__(self, cfg: DualTierMirasConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        M = cfg.mem_slots
        H = cfg.n_heads
        assert d % H == 0, f"d_model {d} must be divisible by n_heads {H}"
        self.head_dim = d // H

        # Memory slots (keys/values) for each tier
        self.register_buffer("fast_keys", torch.zeros(M, d))
        self.register_buffer("fast_vals", torch.zeros(M, d))
        self.register_buffer("deep_keys", torch.zeros(M, d))
        self.register_buffer("deep_vals", torch.zeros(M, d))

        # Memory strength/salience for decay
        self.register_buffer("fast_strength", torch.ones(M))
        self.register_buffer("deep_strength", torch.ones(M))

        self.register_buffer("fast_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("deep_ptr", torch.zeros((), dtype=torch.long))

        # Mixing: base scalar + optional context gate
        init_w = 0.7
        init_logit = math.log(init_w / (1.0 - init_w + 1e-8))
        self.mix_logit = nn.Parameter(torch.tensor(init_logit))
        self.context_gate = nn.Linear(d, 1)

        # Surprise computation for deep tier gating
        self.surprise_proj = nn.Linear(d, d)
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))
        self.register_buffer("num_updates", torch.tensor(0))

        # v2.1: Query projection for better matching
        if cfg.use_query_proj:
            self.query_proj = nn.Linear(d, d)
        else:
            self.query_proj = nn.Identity()

        # v2.1: Multi-head output projection
        if H > 1:
            self.out_proj = nn.Linear(d, d)
        else:
            self.out_proj = nn.Identity()

        # v2.1: Confidence estimator for adaptive retrieval
        if cfg.use_adaptive_retrieval:
            self.confidence_head = nn.Sequential(
                nn.Linear(d, d // 4),
                nn.ReLU(),
                nn.Linear(d // 4, 1),
                nn.Sigmoid()
            )

        # v2.2: Sparse attention top-k
        self.use_sparse = cfg.use_sparse_attention
        self.top_k = cfg.top_k_retrieval

        # v2.2: Memory compression
        if cfg.use_memory_compression:
            comp_dim = int(d * cfg.compression_ratio)
            self.compress = nn.Linear(d, comp_dim)
            self.decompress = nn.Linear(comp_dim, d)
        else:
            self.compress = None
            self.decompress = None

    @torch.no_grad()
    def reset_parameters(self):
        """Reset all memory buffers. Call at episode/sequence start."""
        self.fast_keys.zero_()
        self.fast_vals.zero_()
        self.deep_keys.zero_()
        self.deep_vals.zero_()
        self.fast_strength.fill_(1.0)
        self.deep_strength.fill_(1.0)
        self.fast_ptr.zero_()
        self.deep_ptr.zero_()
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.num_updates.zero_()

    @torch.no_grad()
    def apply_decay(self):
        """Apply memory decay (forgetting). Call periodically."""
        if not self.cfg.use_decay:
            return
        decay = self.cfg.decay_rate
        self.fast_strength.mul_(decay)
        self.deep_strength.mul_(decay)
        # Zero out memories with very low strength
        threshold = 0.01
        weak_fast = self.fast_strength < threshold
        weak_deep = self.deep_strength < threshold
        self.fast_keys[weak_fast] = 0
        self.fast_vals[weak_fast] = 0
        self.deep_keys[weak_deep] = 0
        self.deep_vals[weak_deep] = 0

    def _cosine_attention(self, query: torch.Tensor,
                          keys: torch.Tensor,
                          values: torch.Tensor,
                          return_weights: bool = False):
        """Temperature-scaled cosine-similarity attention.

        query:  [B, d]
        keys:   [M, d]
        values: [M, d]
        returns: [B, d] or ([B, d], [B, M]) if return_weights
        """
        if keys.numel() == 0:
            zeros = torch.zeros_like(query)
            if return_weights:
                return zeros, torch.zeros(query.size(0), 1, device=query.device)
            return zeros

        q = F.normalize(query, p=2, dim=-1)          # [B, d]
        k = F.normalize(keys, p=2, dim=-1)           # [M, d]
        sim = q @ k.t()                              # [B, M]

        # v2.2: Sparse attention - keep only top-k similarities
        if self.use_sparse and sim.size(1) > self.top_k:
            topk_vals, topk_idx = sim.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(sim).scatter_(
                1, topk_idx, torch.ones_like(topk_vals)
            )
            sim = sim.masked_fill(mask == 0, float('-inf'))

        # Temperature scaling for sharper/softer attention
        temp = self.cfg.temperature
        weights = F.softmax(sim / temp, dim=-1)      # [B, M]
        out = weights @ values                       # [B, d]

        if return_weights:
            return out, weights
        return out

    def _compute_surprise(self, x: torch.Tensor) -> torch.Tensor:
        """Compute surprise score based on deviation from running stats.

        Returns scalar surprise in [0, 1] per batch element.
        Higher = more surprising (further from seen distribution).
        """
        # Project input
        h = self.surprise_proj(x)  # [B, d]

        # Compute z-score style surprise
        diff = h - self.running_mean
        std = (self.running_var + 1e-8).sqrt()
        z_score = (diff / std).abs().mean(dim=-1)  # [B]

        # Normalize to [0, 1] - center at z=1 for more sensitivity
        surprise = torch.sigmoid(z_score - 1.0)
        return surprise.detach()  # Don't backprop through surprise

    @torch.no_grad()
    def _update_running_stats(self, x: torch.Tensor):
        """Update running mean/var with new observations."""
        h = self.surprise_proj(x)  # [B, d]
        batch_mean = h.mean(dim=0)
        batch_var = h.var(dim=0, unbiased=False)

        # EMA update
        momentum = 0.1
        self.running_mean.mul_(1 - momentum).add_(batch_mean * momentum)
        self.running_var.mul_(1 - momentum).add_(batch_var * momentum)
        self.num_updates.add_(1)

    def read(self, key: torch.Tensor,
             context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Read from both tiers and mix outputs.

        Args:
            key: Query tensor [B, d]
            context: Optional context for gating [B, d]

        Returns:
            Dict with 'v', 'v_fast', 'v_deep', 'w_fast', 'attn_entropy',
            'confidence', 'retrieval_strength'
        """
        B, d = key.shape
        device = key.device
        H = self.cfg.n_heads

        # v2.1: Project query for better matching
        q = self.query_proj(key)

        # Clone buffers to avoid in-place modification issues with autograd
        fk = self.fast_keys.clone().to(device)
        fv = self.fast_vals.clone().to(device)
        dk = self.deep_keys.clone().to(device)
        dv = self.deep_vals.clone().to(device)

        # v2.1: Multi-head attention
        if H > 1:
            # Reshape for multi-head: [B, H, head_dim], [M, H, head_dim]
            hd = self.head_dim
            q_h = q.view(B, H, hd)
            fk_h = fk.view(-1, H, hd)
            fv_h = fv.view(-1, H, hd)
            dk_h = dk.view(-1, H, hd)
            dv_h = dv.view(-1, H, hd)

            # Per-head attention
            fast_outs, deep_outs = [], []
            fast_ws, deep_ws = [], []
            for h in range(H):
                fv_h_out, fw = self._cosine_attention(
                    q_h[:, h], fk_h[:, h], fv_h[:, h], return_weights=True)
                dv_h_out, dw = self._cosine_attention(
                    q_h[:, h], dk_h[:, h], dv_h[:, h], return_weights=True)
                fast_outs.append(fv_h_out)
                deep_outs.append(dv_h_out)
                fast_ws.append(fw)
                deep_ws.append(dw)

            # Concat heads and project
            fast_v = torch.cat(fast_outs, dim=-1)  # [B, d]
            deep_v = torch.cat(deep_outs, dim=-1)
            fast_v = self.out_proj(fast_v)
            deep_v = self.out_proj(deep_v)
            fast_w = torch.stack(fast_ws, dim=1).mean(dim=1)  # [B, M]
            deep_w = torch.stack(deep_ws, dim=1).mean(dim=1)
        else:
            fast_v, fast_w = self._cosine_attention(
                q, fk, fv, return_weights=True)
            deep_v, deep_w = self._cosine_attention(
                q, dk, dv, return_weights=True)

        # Compute attention entropy (measure of focus)
        eps = 1e-8
        fast_ent = -(fast_w * (fast_w + eps).log()).sum(dim=-1).mean()
        deep_ent = -(deep_w * (deep_w + eps).log()).sum(dim=-1).mean()

        # Mix fast and deep outputs
        base = torch.sigmoid(self.mix_logit)
        if context is not None:
            g = torch.sigmoid(self.context_gate(context))
            w_fast = 0.5 * (base + g)
        else:
            w_fast = base.expand(B, 1)

        v = w_fast * fast_v + (1.0 - w_fast) * deep_v

        # v2.1: Adaptive retrieval - scale output by confidence
        confidence = torch.ones(B, 1, device=device)
        retrieval_strength = torch.ones(B, 1, device=device)
        if self.cfg.use_adaptive_retrieval and hasattr(self, 'confidence_head'):
            # Confidence based on max attention weight
            max_attn = torch.max(fast_w.max(dim=-1)[0], deep_w.max(dim=-1)[0])
            confidence = self.confidence_head(key)  # [B, 1]
            retrieval_strength = confidence * max_attn.unsqueeze(-1)
            # Scale retrieved value by confidence (soft gating)
            v = v * retrieval_strength + key * (1 - retrieval_strength)

        return {
            "v": v,
            "v_fast": fast_v,
            "v_deep": deep_v,
            "w_fast": w_fast.mean(dim=0),
            "attn_entropy": (fast_ent + deep_ent) / 2,
            "confidence": confidence.mean(),
            "retrieval_strength": retrieval_strength.mean(),
        }

    @torch.no_grad()
    def update(self,
               key: torch.Tensor,
               value: torch.Tensor,
               weight: Optional[torch.Tensor] = None,
               context: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Surprise-gated writes into both tiers.

        Args:
            key: Key tensor [B, d]
            value: Value tensor [B, d]
            weight: Optional importance weight [B] (for prioritized writes)
            context: Optional context [B, d]

        Returns:
            Dict with memory statistics
        """
        B, d = key.shape
        M = self.cfg.mem_slots
        device = self.fast_keys.device

        k = key.detach().to(device)
        v = value.detach().to(device)

        # Compute surprise for deep tier gating
        surprise = self._compute_surprise(k)  # [B]
        self._update_running_stats(k)

        # Optional importance weighting
        if weight is not None:
            importance = weight.detach().to(device)
        else:
            importance = torch.ones(B, device=device)

        # Fast tier: write every item (with optional EMA)
        for i in range(B):
            idx = int(self.fast_ptr.item() % M)
            if self.cfg.use_ema and self.fast_keys[idx].abs().sum() > 0:
                decay = self.cfg.ema_decay
                self.fast_keys[idx] = decay * self.fast_keys[idx] + (1 - decay) * k[i]
                self.fast_vals[idx] = decay * self.fast_vals[idx] + (1 - decay) * v[i]
            else:
                self.fast_keys[idx] = k[i]
                self.fast_vals[idx] = v[i]
            self.fast_ptr.add_(1).remainder_(M)

        # Deep tier: surprise-gated writes (or importance-based)
        threshold = self.cfg.surprise_threshold
        num_deep_writes = 0
        for i in range(B):
            # Write if surprising OR high importance OR early in training
            should_write = (
                surprise[i] > threshold or
                importance[i] > 1.0 or
                self.num_updates < M  # Fill memory initially
            )
            if should_write:
                idx = int(self.deep_ptr.item() % M)
                if self.cfg.use_ema and self.deep_keys[idx].abs().sum() > 0:
                    decay = self.cfg.ema_decay
                    dk = self.deep_keys[idx]
                    dv = self.deep_vals[idx]
                    self.deep_keys[idx] = decay * dk + (1 - decay) * k[i]
                    self.deep_vals[idx] = decay * dv + (1 - decay) * v[i]
                else:
                    self.deep_keys[idx] = k[i]
                    self.deep_vals[idx] = v[i]
                self.deep_ptr.add_(1).remainder_(M)
                num_deep_writes += 1

        # Return stats for logging
        stats = {
            "mem_fast/keys_norm": float(self.fast_keys.norm().item()),
            "mem_fast/vals_norm": float(self.fast_vals.norm().item()),
            "mem_deep/keys_norm": float(self.deep_keys.norm().item()),
            "mem_deep/vals_norm": float(self.deep_vals.norm().item()),
            "mem_mix/fast_weight": float(torch.sigmoid(self.mix_logit).item()),
            "mem_surprise/mean": float(surprise.mean().item()),
            "mem_surprise/max": float(surprise.max().item()),
            "mem_deep/num_writes": num_deep_writes,
        }
        return stats


class LongMemKVCache(nn.Module):
    """Simple external KV cache with cosine similarity retrieval.

    Fixed-capacity ring-buffer external memory with top-k nearest
    neighbor retrieval using cosine similarity.

    Architecture:
        - **Capacity**: Fixed number of (key, value) slots.
        - **Write path**: Ring-buffer overwrites oldest entries.
        - **Retrieval**: Top-k cosine similarity search.

    For large-scale (>100k entries), replace with FAISS index.

    Usage:
        >>> cache = LongMemKVCache(key_dim=128, value_dim=128)
        >>> cache.write(keys, values)
        >>> k, v = cache.retrieve(query, top_k=8)
        >>> cache.reset()

    Attributes:
        key_dim: Dimension of keys.
        value_dim: Dimension of values.
        capacity: Maximum number of entries.
    """

    def __init__(self, key_dim: int, value_dim: int,
                 capacity: int = 4096,
                 device: Optional[torch.device] = None):
        super().__init__()
        if device is None:
            device = torch.device("cpu")

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity

        self.register_buffer("keys", torch.zeros(capacity, key_dim, device=device))
        self.register_buffer("values", torch.zeros(capacity, value_dim, device=device))
        self.register_buffer("valid", torch.zeros(capacity, dtype=torch.bool, device=device))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long, device=device))

    @torch.no_grad()
    def reset(self):
        self.keys.zero_()
        self.values.zero_()
        self.valid.zero_()
        self.ptr.zero_()

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor):
        """Write a batch of keys/values into the ring buffer.

        key:   [B, d_k]
        value: [B, d_v]
        """
        B = key.shape[0]
        device = self.keys.device
        k = key.detach().to(device)
        v = value.detach().to(device)

        for i in range(B):
            idx = int(self.ptr.item() % self.capacity)
            self.keys[idx] = k[i]
            self.values[idx] = v[i]
            self.valid[idx] = True
            self.ptr.add_(1).remainder_(self.capacity)

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor,
                 top_k: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k nearest neighbors.

        query: [B, d_k]
        returns (keys, values): [B, K, d_k], [B, K, d_v]
        """
        device = self.keys.device
        q = query.to(device)
        valid_mask = self.valid

        if valid_mask.sum().item() == 0:
            B = q.shape[0]
            return (torch.zeros(B, top_k, self.key_dim, device=device),
                    torch.zeros(B, top_k, self.value_dim, device=device))

        keys = self.keys[valid_mask]  # [N, d_k]
        values = self.values[valid_mask]  # [N, d_v]
        N = keys.shape[0]
        K = min(top_k, N)

        q_norm = F.normalize(q, p=2, dim=-1)      # [B, d_k]
        k_norm = F.normalize(keys, p=2, dim=-1)   # [N, d_k]
        sim = q_norm @ k_norm.t()                 # [B, N]
        _, idx = torch.topk(sim, k=K, dim=-1)     # [B, K]

        gathered_keys = keys[idx]                 # [B, K, d_k]
        gathered_vals = values[idx]               # [B, K, d_v]

        if K < top_k:
            pad_k = top_k - K
            gathered_keys = F.pad(gathered_keys, (0, 0, 0, pad_k))
            gathered_vals = F.pad(gathered_vals, (0, 0, 0, pad_k))
        return gathered_keys, gathered_vals
