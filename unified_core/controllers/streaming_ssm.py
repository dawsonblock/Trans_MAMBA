"""
Streaming SSM Controller (0T Agent).

Constant-memory recurrent controller with no KV cache.
State size is O(D*K), independent of sequence length.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMState(NamedTuple):
    """Recurrent state for StreamingSSM."""
    h: torch.Tensor
    conv_buf: torch.Tensor


@dataclass
class StreamingSSMConfig:
    """Configuration for StreamingSSM controller."""
    input_dim: int = 64
    d_model: int = 64
    n_layers: int = 2
    d_conv: int = 4
    min_timescale: float = 10.0
    max_timescale: float = 2000.0
    dropout: float = 0.1


class StreamingSSMCell(nn.Module):
    """
    Single StreamingSSM cell with multi-timescale initialization.

    Key property: O(1) compute per step, O(D) state per env.
    """

    def __init__(self, cfg: StreamingSSMConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.d_conv = cfg.d_conv

        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model * 2)
        self.conv = nn.Conv1d(
            cfg.d_model, cfg.d_model,
            kernel_size=cfg.d_conv, padding=0,
            groups=cfg.d_model,
        )

        tau = torch.exp(torch.linspace(
            math.log(cfg.min_timescale),
            math.log(cfg.max_timescale),
            cfg.d_model,
        ))
        self.register_buffer("decay", torch.exp(-1.0 / tau))

        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> SSMState:
        """Initialize recurrent state."""
        return SSMState(
            h=torch.zeros(batch_size, self.d_model, device=device),
            conv_buf=torch.zeros(
                batch_size, self.d_model, self.d_conv, device=device
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: SSMState,
    ) -> Tuple[torch.Tensor, SSMState]:
        """
        Single-step forward.

        Args:
            x: [B, D] input
            state: Previous SSMState

        Returns:
            out: [B, D] output
            new_state: Updated SSMState
        """
        B = x.size(0)

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        conv_buf = state.conv_buf.clone()
        conv_buf = torch.roll(conv_buf, -1, dims=-1)
        conv_buf[:, :, -1] = x_inner

        conv_out = self.conv(conv_buf).squeeze(-1)
        conv_out = F.silu(conv_out)

        h = self.decay * state.h + (1 - self.decay) * conv_out
        out = h * F.silu(z)
        out = self.dropout(self.out_proj(out))

        new_state = SSMState(h=h, conv_buf=conv_buf)
        return out, new_state


class StreamingSSMController(nn.Module):
    """
    0T Memory Agent controller.

    Architecture:
        Input -> Encoder -> N x StreamingSSMCell -> Output

    Properties:
        - Constant memory footprint O(D * K * N)
        - Linear-time processing O(T)
        - No attention, no KV cache
    """

    def __init__(self, cfg: StreamingSSMConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.ReLU(),
        )

        self.cells = nn.ModuleList([
            StreamingSSMCell(cfg) for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> list:
        """Initialize all layer states."""
        return [
            cell.init_state(batch_size, device)
            for cell in self.cells
        ]

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Process single timestep.

        Args:
            x: [B, input_dim] observation
            state: List of SSMState per layer

        Returns:
            features: [B, d_model] hidden representation
            new_state: Updated states
        """
        B = x.size(0)
        device = x.device

        if state is None:
            state = self.init_state(B, device)

        h = self.encoder(x)

        new_states = []
        for i, cell in enumerate(self.cells):
            h_res, new_s = cell(h, state[i])
            h = h + h_res
            new_states.append(new_s)

        return self.ln_f(h), new_states

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Process sequence of observations.

        Args:
            x: [B, T, input_dim] observations
            state: Initial states

        Returns:
            features: [B, T, d_model] hidden representations
            final_state: States after processing
        """
        B, T, _ = x.shape
        device = x.device

        if state is None:
            state = self.init_state(B, device)

        outputs = []
        for t in range(T):
            out, state = self.forward(x[:, t], state)
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1), state
