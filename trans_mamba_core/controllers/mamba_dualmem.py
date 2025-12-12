"""
Mamba + DualTierMiras Controller.

Combines Mamba SSM backbone with parametric memory for long-range retrieval.
"""

from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn

from trans_mamba_core.controllers.mamba import MambaBlock, MambaConfig
from trans_mamba_core.memory import (
    DualTierMiras,
    DualTierMirasConfig,
    MemoryState,
)


class MambaDualMemState(NamedTuple):
    """Combined state for Mamba + Memory."""

    mamba_states: list
    memory_state: MemoryState


@dataclass
class MambaDualMemConfig:
    """Configuration for Mamba + DualTierMiras controller."""

    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    mem_slots: int = 64
    mem_n_heads: int = 4
    surprise_threshold: float = 0.5
    surprise_detached: bool = True


class MambaDualMemController(nn.Module):
    """
    Mamba + DualTierMiras for long-horizon sequence tasks.

    Architecture:
        Input -> Embedding -> Mamba Blocks -> Memory Fusion -> LM Head

    The memory module queries at each timestep and fuses retrieved
    information with the Mamba hidden state.
    """

    def __init__(self, cfg: MambaDualMemConfig):
        super().__init__()
        self.cfg = cfg

        mamba_cfg = MambaConfig(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            dropout=cfg.dropout,
        )

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(mamba_cfg) for _ in range(cfg.n_layers)]
        )

        miras_cfg = DualTierMirasConfig(
            d_model=cfg.d_model,
            d_value=cfg.d_model,
            mem_slots=cfg.mem_slots,
            n_heads=cfg.mem_n_heads,
            surprise_threshold=cfg.surprise_threshold,
            surprise_detached=cfg.surprise_detached,
        )
        self.memory = DualTierMiras(miras_cfg)

        self.mem_gate = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.mem_ln = nn.LayerNorm(cfg.d_model)

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[MambaDualMemState] = None,
    ) -> Tuple[torch.Tensor, MambaDualMemState, Dict[str, Any]]:
        B, T = x.shape[:2]
        device = x.device

        if state is None:
            state = self.init_state(B, device)

        if x.dim() == 2:
            h = self.embed(x)
        else:
            h = x

        mamba_states = state.mamba_states
        new_mamba_states = []

        for i, block in enumerate(self.blocks):
            block_state = mamba_states[i] if mamba_states else None
            h_res, new_state = block(h, block_state)
            h = h + h_res
            new_mamba_states.append(new_state)

        mem_state = state.memory_state
        mem_out, mem_state, mem_aux = self.memory.read_write_chunk(
            queries=h,
            write_values=h,
            state=mem_state,
        )

        gate = torch.sigmoid(
            self.mem_gate(torch.cat([h, mem_out], dim=-1))
        )
        fused = gate * h + (1 - gate) * mem_out
        h = self.mem_ln(fused)
        h = self.ln_f(h)
        logits = self.lm_head(h)

        new_state = MambaDualMemState(
            mamba_states=new_mamba_states,
            memory_state=mem_state,
        )

        combined_aux = {"surprise": mem_aux["surprise"]}

        return logits, new_state, combined_aux

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> MambaDualMemState:
        """Initialize combined Mamba + Memory state."""
        return MambaDualMemState(
            mamba_states=[None] * self.cfg.n_layers,
            memory_state=self.memory.init_state(batch_size, device),
        )

    def get_features(
        self,
        x: torch.Tensor,
        state: Optional[MambaDualMemState] = None,
    ) -> Tuple[torch.Tensor, MambaDualMemState]:
        """Get hidden features without LM head."""
        B, T = x.shape[:2]
        device = x.device

        if state is None:
            state = self.init_state(B, device)

        if x.dim() == 2:
            h = self.embed(x)
        else:
            h = x

        mamba_states = state.mamba_states
        new_mamba_states = []

        for i, block in enumerate(self.blocks):
            block_state = mamba_states[i] if mamba_states else None
            h_res, new_state_i = block(h, block_state)
            h = h + h_res
            new_mamba_states.append(new_state_i)

        mem_state = state.memory_state
        mem_out, mem_state, _aux = self.memory.read_write_chunk(
            queries=h,
            write_values=h,
            state=mem_state,
        )
        gate = torch.sigmoid(
            self.mem_gate(torch.cat([h, mem_out], dim=-1))
        )
        fused = gate * h + (1 - gate) * mem_out
        h = self.mem_ln(fused)
        h = self.ln_f(h)

        new_state = MambaDualMemState(
            mamba_states=new_mamba_states,
            memory_state=mem_state,
        )
        return h, new_state
