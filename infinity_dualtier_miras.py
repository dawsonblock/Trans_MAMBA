# infinity_dualtier_miras.py
"""
Infinity-ready DualTierMiras implementation.

Dual-tier content-addressable parametric memory:
- Fast tier: always updated, short-term pattern capture
- Deep tier: surprise-gated, long-term consolidation

No external dependencies except PyTorch.
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    a: [B, H, S, D]
    b: [B, H, S, D]
    returns: [B, H, S]
    """
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=-1)


@dataclass
class DualTierMirasConfig:
    """Configuration for DualTierMiras memory module."""
    # Core dimensions
    d_model: int
    mem_slots: int = 64

    # Learning / update scales
    lr_fast: float = 1.0
    lr_deep: float = 0.2

    # Retrieval
    temperature: float = 1.0
    n_heads: int = 1

    # Surprise / consolidation
    surprise_threshold: float = 0.5
    use_ema: bool = False
    ema_decay: float = 0.99

    # Forgetting
    use_decay: bool = False
    decay_rate: float = 0.999

    # Retrieval extras
    use_adaptive_retrieval: bool = True
    use_query_proj: bool = True

    # Sparsity / compression
    use_sparse_attention: bool = False
    top_k_retrieval: int = 8
    use_memory_compression: bool = False
    compression_ratio: float = 0.5


class DualTierMiras(nn.Module):
    """
    Dual-tier content-addressable parametric memory.

    - Fast tier: always updated, short-term pattern capture.
    - Deep tier: surprise-gated, long-term consolidation.

    Memory lives as parameters/buffers inside this module and is updated
    under torch.no_grad() so it behaves like a dynamic stateful cache.
    """

    def __init__(self, cfg: DualTierMirasConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        H = cfg.n_heads
        S = cfg.mem_slots

        assert d % H == 0, "d_model must be divisible by n_heads"
        self.head_dim = d // H

        # Keys / values per tier
        self.register_buffer("fast_keys", torch.zeros(1, H, S, self.head_dim))
        self.register_buffer("fast_vals", torch.zeros(1, H, S, self.head_dim))
        self.register_buffer("deep_keys", torch.zeros(1, H, S, self.head_dim))
        self.register_buffer("deep_vals", torch.zeros(1, H, S, self.head_dim))

        # Pointers per tier
        self.register_buffer("fast_ptr", torch.zeros(1, H, dtype=torch.long))
        self.register_buffer("deep_ptr", torch.zeros(1, H, dtype=torch.long))

        # Surprise stats
        self.surprise_proj = nn.Linear(d, d)
        self.register_buffer("surprise_mean", torch.zeros(1, d))
        self.register_buffer("surprise_var", torch.ones(1, d))

        # Optional projections
        if cfg.use_query_proj:
            self.query_proj = nn.Linear(d, d)
        else:
            self.query_proj = nn.Identity()

        self.out_proj = nn.Linear(d, d)

        # Context gate + mix
        self.context_gate = nn.Linear(d, d)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

        # Optional confidence head
        if cfg.use_adaptive_retrieval:
            self.conf_head = nn.Sequential(
                nn.Linear(d, d),
                nn.Tanh(),
                nn.Linear(d, 1),
                nn.Sigmoid(),
            )
        else:
            self.conf_head = None

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Reset memory to small noise and reinitialize projections."""
        d = self.cfg.d_model
        H = self.cfg.n_heads
        S = self.cfg.mem_slots

        self.fast_keys.zero_()
        self.fast_vals.zero_()
        self.deep_keys.zero_()
        self.deep_vals.zero_()
        self.fast_ptr.zero_()
        self.deep_ptr.zero_()

        nn.init.normal_(self.fast_keys, std=0.01)
        nn.init.normal_(self.fast_vals, std=0.01)
        nn.init.normal_(self.deep_keys, std=0.01)
        nn.init.normal_(self.deep_vals, std=0.01)

        self.surprise_mean.zero_()
        self.surprise_var.fill_(1.0)

        # Projections / gates
        nn.init.xavier_uniform_(self.surprise_proj.weight)
        nn.init.zeros_(self.surprise_proj.bias)
        if isinstance(self.query_proj, nn.Linear):
            nn.init.xavier_uniform_(self.query_proj.weight)
            nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)
        self.mix_logit.data.zero_()

        if self.conf_head is not None:
            for m in self.conf_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _update_surprise_stats(self, h: torch.Tensor):
        """
        h: [B, D]
        """
        if not self.training:
            return

        decay = 0.99

        batch_mean = h.mean(dim=0, keepdim=True)
        batch_var = h.var(dim=0, unbiased=False, keepdim=True)

        self.surprise_mean.mul_(decay).add_(batch_mean, alpha=1 - decay)
        self.surprise_var.mul_(decay).add_(batch_var, alpha=1 - decay)

    @torch.no_grad()
    def _compute_surprise(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, D]
        returns: surprise in [0,1] shape [B]
        """
        mean = self.surprise_mean
        var = self.surprise_var
        std = (var + 1e-6).sqrt()

        z = (h - mean) / std
        z = z.abs().mean(dim=-1)  # [B]
        # Shift so that ~1 std -> ~0.5 surprise
        return torch.sigmoid(z - 1.0)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] -> [B, H, 1, d_head]
        """
        B, D = x.shape
        H = self.cfg.n_heads
        d_head = self.head_dim
        x = x.view(B, H, d_head)
        return x.unsqueeze(2)

    def _apply_decay(self):
        if not self.cfg.use_decay:
            return
        self.fast_vals.mul_(self.cfg.decay_rate)
        self.deep_vals.mul_(self.cfg.decay_rate)

    def read(
        self,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        query: [B, D]
        context: [B, D] or None
        returns:
            {
                "v": [B, D],
                "fast_weight": [B, 1],
                "deep_weight": [B, 1],
                "surprise": [B, 1],
                "confidence": [B, 1],
            }
        """
        B, D = query.shape
        H = self.cfg.n_heads
        S = self.cfg.mem_slots

        q = self.query_proj(query)  # [B, D]
        q_heads = self._split_heads(q)  # [B, H, 1, d_head]

        fast_keys = self.fast_keys.expand(B, -1, -1, -1)  # [B, H, S, d_head]
        deep_keys = self.deep_keys.expand(B, -1, -1, -1)

        # Similarities
        sim_fast = _cosine_sim(q_heads.expand(-1, -1, S, -1), fast_keys)  # [B, H, S]
        sim_deep = _cosine_sim(q_heads.expand(-1, -1, S, -1), deep_keys)

        if self.cfg.use_sparse_attention and self.cfg.top_k_retrieval < S:
            k = self.cfg.top_k_retrieval
            topk_vals, topk_idx = torch.topk(sim_fast, k, dim=-1)
            mask = torch.full_like(sim_fast, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_vals)
            sim_fast = mask

            topk_vals, topk_idx = torch.topk(sim_deep, k, dim=-1)
            mask = torch.full_like(sim_deep, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_vals)
            sim_deep = mask

        scale = 1.0 / max(self.cfg.temperature, 1e-4)
        att_fast = torch.softmax(sim_fast * scale, dim=-1)  # [B, H, S]
        att_deep = torch.softmax(sim_deep * scale, dim=-1)

        fast_vals = self.fast_vals.expand(B, -1, -1, -1)
        deep_vals = self.deep_vals.expand(B, -1, -1, -1)

        v_fast = torch.einsum("bhs,bhsd->bhd", att_fast, fast_vals)  # [B, H, d_head]
        v_deep = torch.einsum("bhs,bhsd->bhd", att_deep, deep_vals)

        v_fast = v_fast.reshape(B, -1)  # [B, D]
        v_deep = v_deep.reshape(B, -1)

        # Mix fast vs deep with context gate
        if context is None:
            context = query

        gate = torch.tanh(self.context_gate(context))  # [B, D]
        gate = gate.mean(dim=-1, keepdim=True)  # [B, 1]

        mix = torch.sigmoid(self.mix_logit + gate)  # [B, 1]
        out = mix * v_fast + (1.0 - mix) * v_deep  # [B, D]

        if self.conf_head is not None:
            conf = self.conf_head(context)  # [B, 1]
            out = out * conf
        else:
            conf = torch.ones(B, 1, device=query.device, dtype=query.dtype)

        out = self.out_proj(out)

        # Compute surprise from current context
        with torch.no_grad():
            h = self.surprise_proj(context)  # [B, D]
            self._update_surprise_stats(h)
            surprise = self._compute_surprise(h).unsqueeze(-1)  # [B,1]

        return {
            "v": out,
            "fast_weight": mix,
            "deep_weight": 1.0 - mix,
            "surprise": surprise,
            "confidence": conf,
        }

    @torch.no_grad()
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ):
        """
        key, value: [B, D]
        context: [B, D] or None
        """
        B, D = key.shape
        H = self.cfg.n_heads
        S = self.cfg.mem_slots
        device = key.device

        self._apply_decay()

        if context is None:
            context = key

        h = self.surprise_proj(context)  # [B, D]
        self._update_surprise_stats(h)
        surprise = self._compute_surprise(h)  # [B]

        # Decide deep vs fast per batch element
        deep_mask = surprise > self.cfg.surprise_threshold  # [B]

        key_h = key.view(B, H, self.head_dim)
        val_h = value.view(B, H, self.head_dim)

        # Expand pointers
        fast_ptr = self.fast_ptr.to(device)  # [1, H]
        deep_ptr = self.deep_ptr.to(device)

        for b in range(B):
            for h_idx in range(H):
                slot_fast = fast_ptr[0, h_idx].item()
                self.fast_keys[0, h_idx, slot_fast] = key_h[b, h_idx]
                self.fast_vals[0, h_idx, slot_fast] = val_h[b, h_idx]
                fast_ptr[0, h_idx] = (slot_fast + 1) % S

                if deep_mask[b]:
                    slot_deep = deep_ptr[0, h_idx].item()
                    self.deep_keys[0, h_idx, slot_deep] = key_h[b, h_idx]
                    self.deep_vals[0, h_idx, slot_deep] = val_h[b, h_idx]
                    deep_ptr[0, h_idx] = (slot_deep + 1) % S

        self.fast_ptr.copy_(fast_ptr)
        self.deep_ptr.copy_(deep_ptr)
