"""
Infinity PPO Agent.

Mamba + DualTierMiras backbone for long-horizon RL.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn

try:
    from ..memory import DualTierMiras, DualTierMirasConfig, MemoryState
    from ..controllers.mamba import MambaBlock, MambaConfig
except ImportError:
    from memory import DualTierMiras, DualTierMirasConfig, MemoryState
    from controllers.mamba import MambaBlock, MambaConfig


@dataclass
class InfinityAgentConfig:
    """Configuration for Infinity agent."""
    obs_dim: int = 4
    act_dim: int = 2
    d_model: int = 128
    n_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    mem_slots: int = 64
    mem_n_heads: int = 4
    surprise_threshold: float = 0.5
    surprise_detached: bool = True


class InfinityAgent(nn.Module):
    """
    Infinity PPO Agent with Mamba + DualTierMiras.

    Architecture:
        Obs -> Encoder -> Mamba Blocks -> Memory -> Policy/Value Heads
    """

    def __init__(self, cfg: InfinityAgentConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.ReLU(),
        )

        mamba_cfg = MambaConfig(
            vocab_size=1,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
        )

        self.blocks = nn.ModuleList([
            MambaBlock(mamba_cfg) for _ in range(cfg.n_layers)
        ])

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
        self.ln = nn.LayerNorm(cfg.d_model)

        self.policy_head = nn.Linear(cfg.d_model, cfg.act_dim)
        self.value_head = nn.Linear(cfg.d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[list, MemoryState]:
        """Initialize agent state."""
        mamba_states = [None] * self.cfg.n_layers
        mem_state = self.memory.init_state(batch_size, device)
        return (mamba_states, mem_state)

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[Tuple[list, MemoryState]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Forward pass.

        Args:
            obs: [B, obs_dim] or [B, T, obs_dim] observations
            state: (mamba_states, memory_state)

        Returns:
            logits: [B, act_dim] or [B, T, act_dim]
            values: [B, 1] or [B, T, 1]
            new_state: Updated state tuple
        """
        B = obs.size(0)
        device = obs.device

        if state is None:
            state = self.init_state(B, device)

        mamba_states, mem_state = state

        if obs.dim() == 2:
            return self._forward_step(obs, mamba_states, mem_state)
        else:
            return self._forward_sequence(obs, mamba_states, mem_state)

    def _forward_step(
        self,
        obs: torch.Tensor,
        mamba_states: list,
        mem_state: MemoryState,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Single timestep forward."""
        h = self.encoder(obs)
        h = h.unsqueeze(1)

        new_mamba_states = []
        for i, block in enumerate(self.blocks):
            h_res, new_s = block(h, mamba_states[i])
            h = h + h_res
            new_mamba_states.append(new_s)

        h = h.squeeze(1)

        mem_out, mem_state, _ = self.memory(
            query=h, write_value=h, state=mem_state
        )

        gate_in = torch.cat([h, mem_out], dim=-1)
        gate = torch.sigmoid(self.mem_gate(gate_in))
        h = gate * h + (1 - gate) * mem_out
        h = self.ln(h)

        logits = self.policy_head(h)
        values = self.value_head(h)

        new_state = (new_mamba_states, mem_state)
        return logits, values, new_state

    def _forward_sequence(
        self,
        obs: torch.Tensor,
        mamba_states: list,
        mem_state: MemoryState,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Sequence forward for training."""
        B, T, _ = obs.shape

        h = self.encoder(obs)

        new_mamba_states = []
        for i, block in enumerate(self.blocks):
            h_res, new_s = block(h, mamba_states[i])
            h = h + h_res
            new_mamba_states.append(new_s)

        outputs = []
        for t in range(T):
            h_t = h[:, t, :]

            mem_out, mem_state, _ = self.memory(
                query=h_t, write_value=h_t, state=mem_state
            )

            gate_in = torch.cat([h_t, mem_out], dim=-1)
            gate = torch.sigmoid(self.mem_gate(gate_in))
            fused = gate * h_t + (1 - gate) * mem_out
            outputs.append(self.ln(fused).unsqueeze(1))

        h = torch.cat(outputs, dim=1)

        logits = self.policy_head(h)
        values = self.value_head(h)

        new_state = (new_mamba_states, mem_state)
        return logits, values, new_state
