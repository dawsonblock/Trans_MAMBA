"""
Infinity PPO Agent.

Mamba + DualTierMiras backbone for long-horizon RL.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from trans_mamba_core.controllers.mamba import MambaBlock, MambaConfig
from trans_mamba_core.memory import (
    DualTierMiras,
    DualTierMirasConfig,
    MemoryState,
)


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
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[list, MemoryState]:
        """Initialize agent state."""

        mamba_states = [None] * self.cfg.n_layers
        mem_state = self.memory.init_state(batch_size, device)
        return (mamba_states, mem_state)

    def reset_state(
        self,
        state: Optional[Tuple[list, MemoryState]] = None,
        batch_indices: Optional[torch.Tensor | list[int]] = None,
    ) -> Tuple[list, MemoryState]:
        """Reset agent state for all envs or selected batch indices."""

        if state is None:
            batch_size = getattr(self, "_last_batch_size", None)
            device = getattr(self, "_last_device", None)
            if batch_size is None or device is None:
                raise ValueError("state is required until agent has been run")
            return self.init_state(int(batch_size), device)

        mamba_states, mem_state = state
        device = mem_state.fast_ptr.device
        batch_size = mem_state.fast_ptr.size(0)

        if batch_indices is None:
            return self.init_state(batch_size, device)

        if isinstance(batch_indices, list):
            idx = torch.tensor(batch_indices, dtype=torch.long, device=device)
        else:
            idx = batch_indices.to(device=device, dtype=torch.long)

        idx = idx.unique()
        if idx.numel() == 0:
            return state

        fresh = self.memory.init_state(batch_size, device)

        fast_keys = mem_state.fast_keys.clone()
        fast_vals = mem_state.fast_vals.clone()
        deep_keys = mem_state.deep_keys.clone()
        deep_vals = mem_state.deep_vals.clone()
        fast_ptr = mem_state.fast_ptr.clone()
        deep_ptr = mem_state.deep_ptr.clone()
        surprise_mean = mem_state.surprise_mean.clone()
        surprise_var = mem_state.surprise_var.clone()

        fast_keys[idx] = fresh.fast_keys[idx]
        fast_vals[idx] = fresh.fast_vals[idx]
        deep_keys[idx] = fresh.deep_keys[idx]
        deep_vals[idx] = fresh.deep_vals[idx]
        fast_ptr[idx] = fresh.fast_ptr[idx]
        deep_ptr[idx] = fresh.deep_ptr[idx]
        surprise_mean[idx] = fresh.surprise_mean[idx]
        surprise_var[idx] = fresh.surprise_var[idx]

        new_mem_state = MemoryState(
            fast_keys=fast_keys,
            fast_vals=fast_vals,
            deep_keys=deep_keys,
            deep_vals=deep_vals,
            fast_ptr=fast_ptr,
            deep_ptr=deep_ptr,
            surprise_mean=surprise_mean,
            surprise_var=surprise_var,
        )

        new_mamba_states = [None] * self.cfg.n_layers
        return (new_mamba_states, new_mem_state)

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

        self._last_batch_size = B
        self._last_device = device

        if state is None:
            state = self.init_state(B, device)

        mamba_states, mem_state = state

        if obs.dim() == 2:
            return self._forward_step(obs, mamba_states, mem_state)

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
            query=h,
            write_value=h,
            state=mem_state,
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

        _, T, _ = obs.shape

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
                query=h_t,
                write_value=h_t,
                state=mem_state,
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
