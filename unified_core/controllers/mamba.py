"""
Mamba Controller for sequence modeling.

State-space model backbone with selective scan.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaState(NamedTuple):
    """Recurrent state for Mamba backbone."""
    conv_state: torch.Tensor
    ssm_state: torch.Tensor


@dataclass
class MambaConfig:
    """Configuration for Mamba controller."""
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1


class MambaBlock(nn.Module):
    """Single Mamba block with SSM."""

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.d_inner = cfg.d_model * cfg.expand
        self.d_state = cfg.d_state
        self.d_conv = cfg.d_conv

        self.in_proj = nn.Linear(cfg.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=cfg.d_conv, padding=cfg.d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        self.x_proj = nn.Linear(self.d_inner, cfg.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(cfg.d_state, self.d_inner, bias=True)

        A = torch.arange(1, cfg.d_state + 1, dtype=torch.float32)
        self.register_buffer("A_log", -torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[MambaState] = None,
    ) -> Tuple[torch.Tensor, MambaState]:
        B, T, D = x.shape

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_ssm = self.x_proj(x_conv)
        B_ssm, C_ssm = x_ssm.chunk(2, dim=-1)

        dt = F.softplus(self.dt_proj(B_ssm))
        A = -torch.exp(self.A_log)

        y = self._selective_scan(x_conv, dt, A, B_ssm, C_ssm)
        y = y * self.D + x_conv

        y = y * F.silu(z)
        out = self.dropout(self.out_proj(y))

        new_state = MambaState(
            conv_state=x_inner[:, -self.d_conv:, :].detach(),
            ssm_state=torch.zeros(B, self.d_inner, self.d_state, device=x.device),
        )

        return out, new_state

    def _selective_scan(self, x, dt, A, B, C):
        """Simplified selective scan (parallel approximation)."""
        B_dim, T, D = x.shape
        alpha = torch.sigmoid(dt.mean(dim=-1, keepdim=True))
        
        outputs = []
        h = torch.zeros(B_dim, D, device=x.device)
        
        for t in range(T):
            h = alpha[:, t] * h + (1 - alpha[:, t]) * x[:, t]
            outputs.append(h.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)


class MambaController(nn.Module):
    """
    Mamba SSM controller for LM and RL tasks.

    Input: [B, T] token ids or [B, T, D] embeddings
    Output: [B, T, vocab_size] logits or [B, T, D] features
    """

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(cfg) for _ in range(cfg.n_layers)
        ])
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
        state: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        if x.dim() == 2:
            h = self.embed(x)
        else:
            h = x

        new_states = []
        for i, block in enumerate(self.blocks):
            block_state = state[i] if state else None
            h_res, new_state = block(h, block_state)
            h = h + h_res
            new_states.append(new_state)

        h = self.ln_f(h)
        logits = self.lm_head(h)

        return logits, new_states

    def init_state(self, batch_size: int, device: torch.device) -> list:
        """Initialize recurrent state for all layers."""
        return [None] * self.cfg.n_layers

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get hidden features without LM head."""
        if x.dim() == 2:
            h = self.embed(x)
        else:
            h = x

        for block in self.blocks:
            h_res, _ = block(h)
            h = h + h_res

        return self.ln_f(h)
