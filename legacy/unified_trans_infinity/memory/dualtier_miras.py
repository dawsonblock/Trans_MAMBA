"""
Canonical DualTierMiras Implementation.

Dual-tier content-addressable parametric memory for both LM and RL applications.

Features:
- Fast tier: Always updated, captures short-term patterns
- Deep tier: Surprise-gated, consolidates long-term knowledge
- Configurable surprise detachment for RL stability
- Multi-head attention retrieval with sparse top-k option
- Memory decay for forgetting old patterns
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn


def cosine_similarity_attention(
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
        query: [B, H, 1, D] or [B, H, D]
        keys: [B, H, S, D]
        values: [B, H, S, D]
        temperature: Softmax temperature
        top_k: If set, only attend to top-k slots
        eps: Numerical stability

    Returns:
        output: [B, H, D]
    """
    if query.dim() == 3:
        query = query.unsqueeze(2)  # [B, H, 1, D]

    # Normalize
    q_norm = query / (query.norm(dim=-1, keepdim=True) + eps)
    k_norm = keys / (keys.norm(dim=-1, keepdim=True) + eps)

    # Similarity: [B, H, 1, S]
    sim = torch.matmul(q_norm, k_norm.transpose(-2, -1)).squeeze(2)  # [B, H, S]

    # Optional sparse attention
    if top_k is not None and top_k < sim.size(-1):
        topk_vals, topk_idx = torch.topk(sim, top_k, dim=-1)
        mask = torch.full_like(sim, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        sim = mask

    # Softmax
    attn = torch.softmax(sim / max(temperature, 1e-4), dim=-1)  # [B, H, S]

    # Weighted sum
    out = torch.einsum("bhs,bhsd->bhd", attn, values)  # [B, H, D]
    return out


@dataclass
class DualTierMirasConfig:
    """Configuration for DualTierMiras memory module."""

    # Core dimensions
    d_model: int = 256
    mem_slots: int = 64
    n_heads: int = 4

    # Learning rates for memory updates
    lr_fast: float = 1.0
    lr_deep: float = 0.2

    # Retrieval parameters
    temperature: float = 1.0
    use_sparse_attention: bool = False
    top_k_retrieval: int = 8

    # Surprise gating
    surprise_threshold: float = 0.5
    surprise_detached: bool = True  # If True, gating is non-differentiable

    # Memory decay
    use_decay: bool = False
    decay_rate: float = 0.999

    # Optional features
    use_adaptive_retrieval: bool = True
    use_query_proj: bool = True
    use_ema_stats: bool = True
    ema_decay: float = 0.99

    # Value dimension (defaults to d_model if not set)
    d_value: Optional[int] = None

    def __post_init__(self):
        if self.d_value is None:
            self.d_value = self.d_model


class DualTierMiras(nn.Module):
    """
    Canonical Dual-Tier Memory with Integrated Retrieval and Surprise-gating.

    This module provides:
    - Two-tier memory (fast + deep) with surprise-gated consolidation
    - Content-addressable retrieval via cosine similarity
    - Compatible with both LM controllers and RL agents

    Usage:
        cfg = DualTierMirasConfig(d_model=256, mem_slots=128)
        mem = DualTierMiras(cfg)

        # Forward pass (read + optional write)
        out, aux = mem(query, write_value=value, write_mask=mask)

        # Or separate read/write
        retrieved, aux = mem.read(query)
        mem.write(key, value, context)
    """

    def __init__(self, cfg: DualTierMirasConfig):
        super().__init__()
        self.cfg = cfg

        d = cfg.d_model
        d_v = cfg.d_value
        H = cfg.n_heads
        S = cfg.mem_slots

        assert d % H == 0, "d_model must be divisible by n_heads"
        self.head_dim = d // H
        self.value_head_dim = d_v // H

        # Memory buffers (not parameters - updated via no_grad)
        self.register_buffer("fast_keys", torch.zeros(1, H, S, self.head_dim))
        self.register_buffer("fast_vals", torch.zeros(1, H, S, self.value_head_dim))
        self.register_buffer("deep_keys", torch.zeros(1, H, S, self.head_dim))
        self.register_buffer("deep_vals", torch.zeros(1, H, S, self.value_head_dim))

        # Write pointers (circular buffer)
        self.register_buffer("fast_ptr", torch.zeros(1, H, dtype=torch.long))
        self.register_buffer("deep_ptr", torch.zeros(1, H, dtype=torch.long))

        # Surprise statistics
        self.surprise_proj = nn.Linear(d, d)
        self.register_buffer("surprise_mean", torch.zeros(1, d))
        self.register_buffer("surprise_var", torch.ones(1, d))

        # Query projection
        if cfg.use_query_proj:
            self.query_proj = nn.Linear(d, d)
        else:
            self.query_proj = nn.Identity()

        # Key/Value projections for writing
        self.key_proj = nn.Linear(d, d)
        self.value_proj = nn.Linear(d, d_v)

        # Output projection
        self.out_proj = nn.Linear(d_v, d_v)

        # Tier mixing gate
        self.context_gate = nn.Linear(d, d)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

        # Optional confidence head for adaptive retrieval
        if cfg.use_adaptive_retrieval:
            self.conf_head = nn.Sequential(
                nn.Linear(d, d // 2),
                nn.Tanh(),
                nn.Linear(d // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.conf_head = None

        self._init_parameters()

    def _init_parameters(self):
        """Initialize all learnable parameters."""
        nn.init.normal_(self.fast_keys, std=0.01)
        nn.init.normal_(self.fast_vals, std=0.01)
        nn.init.normal_(self.deep_keys, std=0.01)
        nn.init.normal_(self.deep_vals, std=0.01)

        nn.init.xavier_uniform_(self.surprise_proj.weight)
        nn.init.zeros_(self.surprise_proj.bias)

        if isinstance(self.query_proj, nn.Linear):
            nn.init.xavier_uniform_(self.query_proj.weight)
            nn.init.zeros_(self.query_proj.bias)

        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)

        if self.conf_head is not None:
            for m in self.conf_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def reset_memory(self):
        """Reset memory contents and pointers to initial state."""
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

    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        """Reshape [B, D] -> [B, H, head_dim]."""
        B = x.size(0)
        return x.view(B, self.cfg.n_heads, head_dim)

    @torch.no_grad()
    def _update_surprise_stats(self, h: torch.Tensor):
        """Update running mean/var for surprise computation."""
        if not self.training:
            return

        decay = self.cfg.ema_decay
        batch_mean = h.mean(dim=0, keepdim=True)
        batch_var = h.var(dim=0, unbiased=False, keepdim=True)

        self.surprise_mean.mul_(decay).add_(batch_mean, alpha=1 - decay)
        self.surprise_var.mul_(decay).add_(batch_var, alpha=1 - decay)

    def _compute_surprise(
        self, h: torch.Tensor, detached: bool = True
    ) -> torch.Tensor:
        """
        Compute surprise score in [0, 1].

        Args:
            h: [B, D] hidden state
            detached: Whether to detach gradients

        Returns:
            surprise: [B] values in [0, 1]
        """
        if detached:
            h = h.detach()

        mean = self.surprise_mean
        var = self.surprise_var
        std = (var + 1e-6).sqrt()

        z = (h - mean) / std
        z_score = z.abs().mean(dim=-1)  # [B]

        # Map z-score to [0, 1] via sigmoid
        # ~1 std -> ~0.5 surprise
        threshold = self.cfg.surprise_threshold
        return torch.sigmoid(z_score - 1.0 / max(threshold, 0.1))

    def _apply_decay(self):
        """Apply memory decay to values."""
        if not self.cfg.use_decay:
            return
        self.fast_vals.mul_(self.cfg.decay_rate)
        self.deep_vals.mul_(self.cfg.decay_rate)

    def read(
        self,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Read from memory using query.

        Args:
            query: [B, d_model] query vector
            context: [B, d_model] optional context for gating

        Returns:
            output: [B, d_value] retrieved vector
            aux: Dict with diagnostic info
        """
        B = query.size(0)
        S = self.cfg.mem_slots
        cfg = self.cfg

        # Project query
        q = self.query_proj(query)
        q_heads = self._split_heads(q, self.head_dim)  # [B, H, head_dim]

        # Expand memory for batch
        fast_keys = self.fast_keys.expand(B, -1, -1, -1)
        fast_vals = self.fast_vals.expand(B, -1, -1, -1)
        deep_keys = self.deep_keys.expand(B, -1, -1, -1)
        deep_vals = self.deep_vals.expand(B, -1, -1, -1)

        # Retrieve from each tier
        top_k = cfg.top_k_retrieval if cfg.use_sparse_attention else None

        v_fast = cosine_similarity_attention(
            q_heads, fast_keys, fast_vals,
            temperature=cfg.temperature, top_k=top_k
        )  # [B, H, value_head_dim]

        v_deep = cosine_similarity_attention(
            q_heads, deep_keys, deep_vals,
            temperature=cfg.temperature, top_k=top_k
        )

        # Reshape to [B, d_value]
        v_fast = v_fast.reshape(B, -1)
        v_deep = v_deep.reshape(B, -1)

        # Mix tiers using context gate
        if context is None:
            context = query

        gate = torch.tanh(self.context_gate(context))
        gate = gate.mean(dim=-1, keepdim=True)  # [B, 1]
        mix = torch.sigmoid(self.mix_logit + gate)  # [B, 1]

        out = mix * v_fast + (1.0 - mix) * v_deep

        # Optional confidence scaling
        if self.conf_head is not None:
            conf = self.conf_head(context)
            out = out * conf
        else:
            conf = torch.ones(B, 1, device=query.device)

        out = self.out_proj(out)

        # Compute surprise for diagnostics
        h = self.surprise_proj(context)
        self._update_surprise_stats(h)
        surprise = self._compute_surprise(h, detached=cfg.surprise_detached)

        aux = {
            "fast_weight": mix,
            "deep_weight": 1.0 - mix,
            "surprise": surprise.unsqueeze(-1),
            "confidence": conf,
        }

        return out, aux

    @torch.no_grad()
    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Write to memory.

        Args:
            key: [B, d_model] key to write
            value: [B, d_model] value to write
            context: [B, d_model] context for surprise computation
            write_mask: [B] optional mask (True = write)

        Returns:
            aux: Dict with write statistics
        """
        B = key.size(0)
        H = self.cfg.n_heads
        S = self.cfg.mem_slots
        device = key.device

        self._apply_decay()

        if context is None:
            context = key

        # Compute surprise
        h = self.surprise_proj(context)
        self._update_surprise_stats(h)
        surprise = self._compute_surprise(h, detached=self.cfg.surprise_detached)

        # Determine which samples go to deep tier
        deep_mask = surprise > self.cfg.surprise_threshold

        # Apply optional write mask
        if write_mask is not None:
            write_mask = write_mask.bool()
        else:
            write_mask = torch.ones(B, dtype=torch.bool, device=device)

        # Project key/value
        k_proj = self.key_proj(key)
        v_proj = self.value_proj(value)

        k_heads = self._split_heads(k_proj, self.head_dim)  # [B, H, head_dim]
        v_heads = self._split_heads(v_proj, self.value_head_dim)

        # Write to memory slots
        fast_ptr = self.fast_ptr.clone()
        deep_ptr = self.deep_ptr.clone()

        num_fast_writes = 0
        num_deep_writes = 0

        for b in range(B):
            if not write_mask[b]:
                continue

            for h_idx in range(H):
                # Always write to fast tier
                slot_fast = fast_ptr[0, h_idx].item()
                self.fast_keys[0, h_idx, slot_fast] = k_heads[b, h_idx]
                self.fast_vals[0, h_idx, slot_fast] = v_heads[b, h_idx]
                fast_ptr[0, h_idx] = (slot_fast + 1) % S
                num_fast_writes += 1

                # Conditionally write to deep tier
                if deep_mask[b]:
                    slot_deep = deep_ptr[0, h_idx].item()
                    self.deep_keys[0, h_idx, slot_deep] = k_heads[b, h_idx]
                    self.deep_vals[0, h_idx, slot_deep] = v_heads[b, h_idx]
                    deep_ptr[0, h_idx] = (slot_deep + 1) % S
                    num_deep_writes += 1

        self.fast_ptr.copy_(fast_ptr)
        self.deep_ptr.copy_(deep_ptr)

        return {
            "surprise": surprise,
            "deep_mask": deep_mask,
            "num_fast_writes": num_fast_writes,
            "num_deep_writes": num_deep_writes,
        }

    def forward(
        self,
        query: torch.Tensor,
        write_value: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Combined read and optional write.

        Args:
            query: [B, d_model] query for retrieval (also used as key for write)
            write_value: [B, d_model] value to write (if None, no write)
            write_mask: [B] optional write mask
            context: [B, d_model] optional context

        Returns:
            output: [B, d_value] retrieved vector
            aux: Combined diagnostic dict
        """
        # Read
        out, read_aux = self.read(query, context=context)

        # Optional write
        write_aux = {}
        if write_value is not None:
            write_aux = self.write(
                key=query.detach(),
                value=write_value.detach(),
                context=context.detach() if context is not None else None,
                write_mask=write_mask,
            )

        # Merge aux dicts
        aux = {**read_aux, **write_aux}
        return out, aux


class MambaDualMemWrapper(nn.Module):
    """
    Wrapper to integrate DualTierMiras with a Mamba backbone for LM tasks.

    Usage:
        wrapper = MambaDualMemWrapper(mamba_block, miras_cfg)
        out = wrapper(x)  # x: [B, T, D]
    """

    def __init__(
        self,
        backbone: nn.Module,
        miras_cfg: DualTierMirasConfig,
        fuse_method: str = "residual",
    ):
        super().__init__()
        self.backbone = backbone
        self.miras = DualTierMiras(miras_cfg)
        self.fuse_method = fuse_method

        d = miras_cfg.d_model
        if fuse_method == "concat":
            self.fuse_proj = nn.Linear(d * 2, d)
        else:
            self.fuse_proj = None
        self.fuse_ln = nn.LayerNorm(d)

    def reset_memory(self):
        self.miras.reset_memory()

    def forward(
        self,
        x: torch.Tensor,
        write_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input sequence
            write_mask: [B, T] optional write mask

        Returns:
            out: [B, T, D] output sequence
        """
        B, T, D = x.shape
        device = x.device

        # Run backbone
        h = self.backbone(x)  # [B, T, D]

        if write_mask is None:
            write_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # Per-timestep memory interaction
        outputs = []
        for t in range(T):
            h_t = h[:, t, :]  # [B, D]
            mask_t = write_mask[:, t] if write_mask.dim() > 1 else write_mask

            # Read and optionally write
            mem_out, _ = self.miras(
                query=h_t,
                write_value=h_t,
                write_mask=mask_t,
            )

            # Fuse
            if self.fuse_method == "concat":
                fused = self.fuse_proj(torch.cat([h_t, mem_out], dim=-1))
            else:
                fused = h_t + mem_out

            fused = self.fuse_ln(fused)
            outputs.append(fused.unsqueeze(1))

        return torch.cat(outputs, dim=1)
