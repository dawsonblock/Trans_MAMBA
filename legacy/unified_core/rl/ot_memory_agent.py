"""
OT Memory Agent (0T Agent).

StreamingSSM backbone for constant-memory RL.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn

try:
    from ..controllers.streaming_ssm import (
        StreamingSSMController, StreamingSSMConfig
    )
except ImportError:
    from controllers.streaming_ssm import (
        StreamingSSMController, StreamingSSMConfig
    )


@dataclass
class OTAgentConfig:
    """Configuration for OT Memory Agent."""
    obs_dim: int = 6
    act_dim: int = 4
    d_model: int = 64
    n_layers: int = 2
    d_conv: int = 4
    min_timescale: float = 10.0
    max_timescale: float = 2000.0


class OTMemoryAgent(nn.Module):
    """
    0T Memory Agent with StreamingSSM backbone.

    Properties:
        - Constant memory footprint O(D * K * N)
        - Linear-time processing O(T)
        - No attention, no KV cache
    """

    def __init__(self, cfg: OTAgentConfig):
        super().__init__()
        self.cfg = cfg

        ssm_cfg = StreamingSSMConfig(
            input_dim=cfg.obs_dim,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            d_conv=cfg.d_conv,
            min_timescale=cfg.min_timescale,
            max_timescale=cfg.max_timescale,
        )

        self.backbone = StreamingSSMController(ssm_cfg)

        self.policy_head = nn.Linear(cfg.d_model, cfg.act_dim)
        self.value_head = nn.Linear(cfg.d_model, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def init_state(self, batch_size: int, device: torch.device) -> list:
        """Initialize recurrent state."""
        return self.backbone.init_state(batch_size, device)

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Forward pass.

        Args:
            obs: [B, obs_dim] or [B, T, obs_dim] observations
            state: List of SSMState per layer

        Returns:
            logits: [B, act_dim] or [B, T, act_dim]
            values: [B, 1] or [B, T, 1]
            new_state: Updated state
        """
        B = obs.size(0)
        device = obs.device

        if state is None:
            state = self.init_state(B, device)

        if obs.dim() == 2:
            features, new_state = self.backbone(obs, state)
        else:
            features, new_state = self.backbone.forward_sequence(obs, state)

        logits = self.policy_head(features)
        values = self.value_head(features)

        return logits, values, new_state

    def get_action(
        self,
        obs: torch.Tensor,
        state: Optional[list] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Get action from policy.

        Returns:
            action: [B] sampled or argmax action
            log_prob: [B] log probability
            value: [B] value estimate
            new_state: Updated state
        """
        logits, values, new_state = self.forward(obs, state)

        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, values.squeeze(-1), new_state
