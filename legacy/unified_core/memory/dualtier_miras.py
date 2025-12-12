"""
Canonical DualTierMiras Memory Implementation.

A dual-tier content-addressable parametric memory with:
- Fast tier: Always updated, captures short-term patterns
- Deep tier: Surprise-gated, consolidates long-term knowledge
- Cosine similarity retrieval with optional sparse top-k
- Configurable differentiable vs detached surprise gating
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, NamedTuple

import torch
import torch.nn as nn


class MemoryState(NamedTuple):
    """Immutable memory state for functional-style updates."""
    fast_keys: torch.Tensor
    fast_vals: torch.Tensor
    deep_keys: torch.Tensor
    deep_vals: torch.Tensor
    fast_ptr: torch.Tensor
    deep_ptr: torch.Tensor
    surprise_mean: torch.Tensor
    surprise_var: torch.Tensor


@dataclass
class DualTierMirasConfig:
    """Configuration for DualTierMiras memory module."""
    d_model: int = 256
    d_value: int = 256
    mem_slots: int = 64
    n_heads: int = 4
    lr_fast: float = 1.0
    lr_deep: float = 0.2
    temperature: float = 1.0
    top_k: Optional[int] = None
    surprise_threshold: float = 0.5
    surprise_detached: bool = True
    decay_rate: float = 0.999
    use_decay: bool = False
    ema_decay: float = 0.99


def cosine_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Cosine similarity attention with optional sparse top-k.

    Args:
        query: [B, H, D] query vectors
        keys: [B, H, S, D] memory keys
        values: [B, H, S, D] memory values
        temperature: Softmax temperature
        top_k: If set, only attend to top-k slots

    Returns:
        output: [B, H, D] retrieved vectors
    """
    q_norm = query / (query.norm(dim=-1, keepdim=True) + eps)
    k_norm = keys / (keys.norm(dim=-1, keepdim=True) + eps)

    sim = torch.einsum("bhd,bhsd->bhs", q_norm, k_norm)

    if top_k is not None and top_k < sim.size(-1):
        topk_vals, topk_idx = torch.topk(sim, top_k, dim=-1)
        mask = torch.full_like(sim, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        sim = mask

    attn = torch.softmax(sim / max(temperature, 1e-4), dim=-1)
    return torch.einsum("bhs,bhsd->bhd", attn, values)


class DualTierMiras(nn.Module):
    """
    Canonical Dual-Tier Memory with Integrated Retrieval and Surprise-gating.

    API:
        output, new_state, aux = memory(query, write_value, write_mask, state)

    This module provides:
    - Two-tier memory (fast + deep) with surprise-gated consolidation
    - Content-addressable retrieval via cosine similarity
    - Functional state management for RL compatibility
    """

    def __init__(self, cfg: DualTierMirasConfig):
        super().__init__()
        self.cfg = cfg

        d = cfg.d_model
        d_v = cfg.d_value
        H = cfg.n_heads

        assert d % H == 0, "d_model must be divisible by n_heads"
        self.head_dim = d // H
        self.value_head_dim = d_v // H

        self.query_proj = nn.Linear(d, d)
        self.key_proj = nn.Linear(d, d)
        self.value_proj = nn.Linear(d, d_v)
        self.out_proj = nn.Linear(d_v, d)

        self.surprise_proj = nn.Linear(d, d)
        self.context_gate = nn.Linear(d, 1)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        for module in [self.query_proj, self.key_proj, self.value_proj,
                       self.out_proj, self.surprise_proj, self.context_gate]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> MemoryState:
        """Initialize fresh memory state."""
        H = self.cfg.n_heads
        S = self.cfg.mem_slots
        d_h = self.head_dim
        d_v = self.value_head_dim

        return MemoryState(
            fast_keys=torch.randn(
                batch_size, H, S, d_h, device=device
            ) * 0.01,
            fast_vals=torch.randn(
                batch_size, H, S, d_v, device=device
            ) * 0.01,
            deep_keys=torch.randn(
                batch_size, H, S, d_h, device=device
            ) * 0.01,
            deep_vals=torch.randn(
                batch_size, H, S, d_v, device=device
            ) * 0.01,
            fast_ptr=torch.zeros(
                batch_size, H, dtype=torch.long, device=device
            ),
            deep_ptr=torch.zeros(
                batch_size, H, dtype=torch.long, device=device
            ),
            surprise_mean=torch.zeros(
                batch_size, self.cfg.d_model, device=device
            ),
            surprise_var=torch.ones(
                batch_size, self.cfg.d_model, device=device
            ),
        )

    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        B = x.size(0)
        return x.view(B, self.cfg.n_heads, head_dim)

    def _compute_surprise(
        self, h: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        """Compute surprise score in [0, 1]."""
        if self.cfg.surprise_detached:
            h = h.detach()

        std = (var + 1e-6).sqrt()
        z = (h - mean) / std
        z_score = z.abs().mean(dim=-1)
        threshold = self.cfg.surprise_threshold
        return torch.sigmoid(z_score - 1.0 / max(threshold, 0.1))

    def _update_surprise_stats(
        self, h: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update running mean/var for surprise computation."""
        decay = self.cfg.ema_decay
        new_mean = decay * mean + (1 - decay) * h
        new_var = decay * var + (1 - decay) * (h - mean).pow(2)
        return new_mean, new_var

    def forward(
        self,
        query: torch.Tensor,
        write_value: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
        state: Optional[MemoryState] = None,
    ) -> Tuple[torch.Tensor, MemoryState, Dict[str, Any]]:
        """
        Forward pass with read and optional write.

        Args:
            query: [B, d_model] query for retrieval
            write_value: [B, d_model] value to write (if None, no write)
            write_mask: [B] bool mask (True = write)
            state: Current memory state (init if None)

        Returns:
            output: [B, d_value] retrieved vector
            new_state: Updated memory state
            aux: Dict with diagnostics
        """
        B = query.size(0)
        device = query.device
        cfg = self.cfg

        if state is None:
            state = self.init_state(B, device)

        q = self.query_proj(query)
        q_heads = self._split_heads(q, self.head_dim)

        v_fast = cosine_attention(
            q_heads, state.fast_keys, state.fast_vals,
            temperature=cfg.temperature, top_k=cfg.top_k
        )
        v_deep = cosine_attention(
            q_heads, state.deep_keys, state.deep_vals,
            temperature=cfg.temperature, top_k=cfg.top_k
        )

        v_fast = v_fast.reshape(B, -1)
        v_deep = v_deep.reshape(B, -1)

        gate = torch.tanh(self.context_gate(query))
        mix = torch.sigmoid(self.mix_logit + gate)
        output = mix * v_fast + (1 - mix) * v_deep
        output = self.out_proj(output)

        h = self.surprise_proj(query)
        new_mean, new_var = self._update_surprise_stats(
            h, state.surprise_mean, state.surprise_var
        )
        surprise = self._compute_surprise(
            h, state.surprise_mean, state.surprise_var
        )

        new_state = MemoryState(
            fast_keys=state.fast_keys,
            fast_vals=state.fast_vals,
            deep_keys=state.deep_keys,
            deep_vals=state.deep_vals,
            fast_ptr=state.fast_ptr,
            deep_ptr=state.deep_ptr,
            surprise_mean=new_mean,
            surprise_var=new_var,
        )

        if write_value is not None:
            new_state = self._write(
                new_state, query, write_value, write_mask, surprise
            )

        aux = {
            "fast_weight": mix,
            "deep_weight": 1 - mix,
            "surprise": surprise,
        }

        return output, new_state, aux

    def _write(
        self,
        state: MemoryState,
        key: torch.Tensor,
        value: torch.Tensor,
        write_mask: Optional[torch.Tensor],
        surprise: torch.Tensor,
    ) -> MemoryState:
        """Write to memory with surprise gating."""
        B = key.size(0)
        H = self.cfg.n_heads
        S = self.cfg.mem_slots
        device = key.device

        if write_mask is None:
            write_mask = torch.ones(B, dtype=torch.bool, device=device)

        k_proj = self.key_proj(key)
        v_proj = self.value_proj(value)

        k_heads = self._split_heads(k_proj, self.head_dim)
        v_heads = self._split_heads(v_proj, self.value_head_dim)

        deep_mask = surprise > self.cfg.surprise_threshold

        fast_keys = state.fast_keys.clone()
        fast_vals = state.fast_vals.clone()
        deep_keys = state.deep_keys.clone()
        deep_vals = state.deep_vals.clone()
        fast_ptr = state.fast_ptr.clone()
        deep_ptr = state.deep_ptr.clone()

        for b in range(B):
            if not write_mask[b]:
                continue

            for h in range(H):
                slot_f = fast_ptr[b, h].item()
                fast_keys[b, h, slot_f] = k_heads[b, h]
                fast_vals[b, h, slot_f] = v_heads[b, h]
                fast_ptr[b, h] = (slot_f + 1) % S

                if deep_mask[b]:
                    slot_d = deep_ptr[b, h].item()
                    deep_keys[b, h, slot_d] = k_heads[b, h]
                    deep_vals[b, h, slot_d] = v_heads[b, h]
                    deep_ptr[b, h] = (slot_d + 1) % S

        if self.cfg.use_decay:
            fast_vals = fast_vals * self.cfg.decay_rate
            deep_vals = deep_vals * self.cfg.decay_rate

        return MemoryState(
            fast_keys=fast_keys,
            fast_vals=fast_vals,
            deep_keys=deep_keys,
            deep_vals=deep_vals,
            fast_ptr=fast_ptr,
            deep_ptr=deep_ptr,
            surprise_mean=state.surprise_mean,
            surprise_var=state.surprise_var,
        )
