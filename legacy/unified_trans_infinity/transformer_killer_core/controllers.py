"""
Controllers for synthetic sequence tasks.

Provides:
- TransformerController: Standard transformer with causal attention
- MambaController: Mamba SSM backbone
- MambaDualMemController: Mamba + DualTierMiras memory
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.dualtier_miras import DualTierMiras, DualTierMirasConfig


@dataclass
class ControllerConfig:
    """Configuration for all controllers."""
    vocab_size: int = 16
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 512

    # Mamba-specific
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    # Memory-specific (for MambaDualMem)
    mem_slots: int = 64
    use_memory: bool = True
    surprise_threshold: float = 0.5


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, D = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is None:
            mask = torch.triu(
                torch.ones(T, T, device=x.device), diagonal=1
            ).bool()
            attn = attn.masked_fill(mask, float("-inf"))
        else:
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerController(nn.Module):
    """
    Standard Transformer controller for sequence tasks.

    Architecture:
        Embedding -> Positional Encoding -> N x TransformerBlock -> LM Head
    """

    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = PositionalEncoding(
            cfg.d_model, cfg.max_seq_len, cfg.dropout
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] input token ids

        Returns:
            logits: [B, T, vocab_size]
        """
        h = self.embed(x)
        h = self.pos_enc(h)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        return self.lm_head(h)


class MambaBlock(nn.Module):
    """
    Simplified Mamba-style SSM block.

    This is a self-contained implementation that doesn't require mamba_ssm.
    For production, replace with actual Mamba2 implementation.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(d_state, self.d_inner)

        # Initialize A as negative (for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.register_buffer("A", -A)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]

        Returns:
            out: [B, T, D]
        """
        B, T, D = x.shape

        # Input projection and split
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # Convolution (causal)
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM
        x_ssm = self.x_proj(x_conv)
        B_ssm, C_ssm = x_ssm.chunk(2, dim=-1)

        # Discretize
        dt = F.softplus(self.dt_proj(B_ssm))

        # Simplified selective scan (parallel approximation)
        # For true recurrent behavior, use actual Mamba implementation
        A_bar = torch.exp(self.A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
        y = x_conv * self.D + self._parallel_scan(x_conv, A_bar, B_ssm, C_ssm)

        # Gate and output
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

    def _parallel_scan(self, x, A, B, C):
        """Simplified parallel scan approximation."""
        B_dim, T, D = x.shape

        # Approximate with exponential moving average
        alpha = 0.9
        h = torch.zeros(B_dim, D, device=x.device)
        outputs = []

        for t in range(T):
            h = alpha * h + (1 - alpha) * x[:, t, :]
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class MambaController(nn.Module):
    """
    Mamba SSM controller for sequence tasks.

    Architecture:
        Embedding -> N x MambaBlock -> LM Head
    """

    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.blocks = nn.ModuleList([
            MambaBlock(
                cfg.d_model,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] input token ids

        Returns:
            logits: [B, T, vocab_size]
        """
        h = self.embed(x)

        for block in self.blocks:
            h = h + block(h)

        h = self.ln_f(h)
        return self.lm_head(h)


class MambaDualMemController(nn.Module):
    """
    Mamba + DualTierMiras controller for sequence tasks.

    Architecture:
        Embedding -> N x (MambaBlock + MemoryFusion) -> LM Head

    The memory module is integrated after each Mamba block,
    providing long-range retrieval capabilities.
    """

    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Create Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                cfg.d_model,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])

        # Create memory module
        miras_cfg = DualTierMirasConfig(
            d_model=cfg.d_model,
            mem_slots=cfg.mem_slots,
            n_heads=cfg.n_heads,
            surprise_threshold=cfg.surprise_threshold,
            use_adaptive_retrieval=True,
            use_decay=True,
            decay_rate=0.999,
        )
        self.miras = DualTierMiras(miras_cfg)

        # Memory fusion
        self.mem_gate = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.mem_ln = nn.LayerNorm(cfg.d_model)

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def reset_memory(self):
        """Reset memory state."""
        self.miras.reset_memory()

    def forward(
        self,
        x: torch.Tensor,
        write_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T] input token ids
            write_mask: [B, T] optional write mask

        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = x.shape
        device = x.device

        h = self.embed(x)

        # Process through Mamba blocks
        for block in self.blocks:
            h = h + block(h)

        # Per-timestep memory interaction
        if write_mask is None:
            write_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        outputs = []
        for t in range(T):
            h_t = h[:, t, :]
            mask_t = write_mask[:, t]

            # Read and write memory
            mem_out, _ = self.miras(
                query=h_t,
                write_value=h_t,
                write_mask=mask_t,
            )

            # Gated fusion
            gate_input = torch.cat([h_t, mem_out], dim=-1)
            gate = torch.sigmoid(self.mem_gate(gate_input))
            fused = gate * h_t + (1 - gate) * mem_out
            fused = self.mem_ln(fused)

            outputs.append(fused.unsqueeze(1))

        h = torch.cat(outputs, dim=1)
        h = self.ln_f(h)
        return self.lm_head(h)


def get_controller(name: str, cfg: ControllerConfig) -> nn.Module:
    """Factory function for controllers."""
    controllers = {
        "transformer": TransformerController,
        "mamba": MambaController,
        "mamba_dualmem": MambaDualMemController,
    }

    if name not in controllers:
        raise ValueError(
            f"Unknown controller: {name}. "
            f"Available: {list(controllers.keys())}"
        )

    return controllers[name](cfg)
