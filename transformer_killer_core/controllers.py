"""
controllers.py

Controller implementations for the Transformer Killer Core.

This module provides:
    - TransformerController: Standard Transformer decoder-only LM
    - MambaController: Mamba SSM backbone (Mamba2 or GRU fallback)
    - MambaDualMemController: Mamba + DualTierMiras parametric memory
    - build_controller: Factory function for controller instantiation

All controllers share the same interface:
    - Input: [B, T] token ids
    - Output: [B, T, V] logits

Mamba2 Integration:
    If mamba-ssm is installed (pip install -e external/mamba_ssm),
    MambaBackbone automatically uses real Mamba2 layers.
    Otherwise, it falls back to GRU layers with the same API.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from .memory_core import DualTierMiras, DualTierMirasConfig


ControllerType = Literal["transformer", "mamba", "mamba_dualmem", "ot_agent"]


@dataclass
class ControllerConfig:
    """Configuration for all controller types.

    Attributes:
        controller_type: Which controller to build.
        vocab_size: Vocabulary size for embedding.
        d_model: Hidden dimension.
        n_layers: Number of layers.
        n_heads: Attention heads (transformer only).
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.
        use_gradient_checkpointing: Trade compute for memory.
        mem_slots: Memory slots (for mamba_dualmem).
        temperature: Attention temperature (for memory).
        use_memory_residual: Add memory-gated residual connection.
        use_layer_memory: Access memory at each layer (vs only output).
        memory_n_heads: Number of attention heads for memory.
    """
    controller_type: ControllerType
    vocab_size: int
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1
    use_gradient_checkpointing: bool = False
    mem_slots: int = 64
    temperature: float = 1.0
    # v2.1 enhancements
    use_memory_residual: bool = True
    use_layer_memory: bool = False
    memory_n_heads: int = 1


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = -torch.log(torch.tensor(10000.0)) / d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * div)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T]


class TransformerController(nn.Module):
    """Standard Transformer decoder-only LM controller."""

    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            cfg.d_model, cfg.max_seq_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers
        )
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] token ids
        h = self.embed(x)
        h = self.pos_enc(h)
        h = self.transformer(h)
        h = self.ln(h)
        logits = self.head(h)
        return logits


class MambaBackbone(nn.Module):
    """SSM-style backbone with optional Mamba2, GRU fallback.

    Features:
        - Auto-detects mamba-ssm and uses Mamba2 if available
        - Falls back to bidirectional GRU with residual connections
        - Layer normalization between layers
        - Optional gradient checkpointing
    """

    def __init__(self, d_model: int, n_layers: int,
                 use_checkpointing: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_mamba = False
        self.use_checkpointing = use_checkpointing

        layers = []
        norms = []
        try:
            from mamba_ssm.modules.mamba2 import Mamba2
            for _ in range(n_layers):
                layers.append(Mamba2(d_model))
                norms.append(nn.LayerNorm(d_model))
            self.use_mamba = True
        except Exception:
            # Enhanced GRU fallback with residuals
            for _ in range(n_layers):
                layers.append(nn.GRU(d_model, d_model, batch_first=True))
                norms.append(nn.LayerNorm(d_model))

        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)

    def _forward_layer(self, layer, norm, h, is_mamba):
        """Forward through single layer with residual."""
        if is_mamba:
            out = layer(h)
        else:
            out, _ = layer(h)
        return norm(out + h)  # Residual connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer, norm in zip(self.layers, self.norms):
            if self.use_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                h = checkpoint(
                    self._forward_layer, layer, norm, h, self.use_mamba,
                    use_reentrant=False
                )
            else:
                h = self._forward_layer(layer, norm, h, self.use_mamba)
        return h


class MambaController(nn.Module):
    """Mamba-style backbone without explicit external memory."""

    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            cfg.d_model, cfg.max_seq_len
        )
        self.backbone = MambaBackbone(cfg.d_model, cfg.n_layers)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.pos_enc(h)
        h = self.backbone(h)
        h = self.ln(h)
        logits = self.head(h)
        return logits


class MambaDualMemController(nn.Module):
    """Mamba backbone + DualTierMiras memory (Upgraded v2.1).

    Features:
        - Mamba2/GRU backbone with residual connections
        - Surprise-gated dual-tier memory
        - Temperature-scaled attention
        - Optional gradient checkpointing
        - Memory statistics tracking
        - v2.1: Memory-gated residual connections
        - v2.1: Multi-head memory attention
        - v2.1: Adaptive retrieval strength
    """

    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            cfg.d_model, cfg.max_seq_len
        )
        self.backbone = MambaBackbone(
            cfg.d_model, cfg.n_layers,
            use_checkpointing=cfg.use_gradient_checkpointing
        )

        # Enhanced memory config with v2.1 features
        mem_cfg = DualTierMirasConfig(
            d_model=cfg.d_model,
            mem_slots=cfg.mem_slots,
            temperature=cfg.temperature,
            n_heads=cfg.memory_n_heads,
            use_adaptive_retrieval=True,
            use_query_proj=True,
        )
        self.memory = DualTierMiras(mem_cfg)

        # Enhanced fusion with gating
        self.gate = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

        # v2.1: Memory residual gate (learned skip connection)
        if cfg.use_memory_residual:
            self.residual_gate = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 4),
                nn.ReLU(),
                nn.Linear(cfg.d_model // 4, 1),
                nn.Sigmoid()
            )

        # Memory control
        self.trainable_memory = False
        self._last_mem_stats = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward with DualTierMiras.

        Features:
            - Per-step memory read/write
            - Gated fusion of backbone + memory
            - Surprise-based deep tier writes
            - Statistics tracking for debugging
        """
        B, T = x.shape
        h = self.embed(x)
        h = self.pos_enc(h)

        # Reset memory per sequence
        self.memory.reset_parameters()

        outputs = []
        for t in range(T):
            # Process prefix
            h_t = h[:, :t + 1, :]
            h_enc = self.backbone(h_t)[:, -1, :]

            # Memory read
            if self.trainable_memory:
                mem_out = self.memory.read(h_enc, context=h_enc)
                v_mem = mem_out["v"]
            else:
                with torch.no_grad():
                    mem_out = self.memory.read(h_enc, context=h_enc)
                v_mem = mem_out["v"].detach()

            # Gated fusion
            concat = torch.cat([h_enc, v_mem], dim=-1)
            gate = torch.sigmoid(self.gate(concat))
            fused = gate * h_enc + (1 - gate) * v_mem

            # v2.1: Memory-gated residual connection
            if self.cfg.use_memory_residual and hasattr(self, 'residual_gate'):
                res_gate = self.residual_gate(h_enc)  # [B, 1]
                fused = fused + res_gate * h_enc

            fused = self.ln(fused)

            logits_t = self.head(fused)
            outputs.append(logits_t.unsqueeze(1))

            # Memory write
            with torch.no_grad():
                stats = self.memory.update(h_enc, h_enc, context=h_enc)
                self._last_mem_stats = stats
                # v2.1: Apply memory decay periodically
                if t > 0 and t % 10 == 0:
                    self.memory.apply_decay()

        return torch.cat(outputs, dim=1)

    def get_memory_stats(self) -> dict:
        """Return last memory statistics for logging."""
        return self._last_mem_stats


def build_controller(cfg: ControllerConfig) -> nn.Module:
    if cfg.controller_type == "transformer":
        return TransformerController(cfg)
    elif cfg.controller_type == "mamba":
        return MambaController(cfg)
    elif cfg.controller_type == "mamba_dualmem":
        return MambaDualMemController(cfg)
    else:
        raise ValueError(f"Unknown controller_type: {cfg.controller_type}")
