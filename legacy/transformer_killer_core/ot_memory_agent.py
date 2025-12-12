"""
ot_memory_agent.py

OT Memory Agent - Unified 0T Memory Agent using canonical core components.

This module provides:
    - OTMemoryAgent: High-level agent combining Mamba backbone + DualTierMiras memory
    - OTMemoryAgentConfig: Configuration dataclass

The agent can be used in three regimes:
    1. Synthetic sequences: As a controller variant in synthetic benchmarks
    2. Language modeling: As a language model wrapper
    3. RL (future): Prepared interface for RL integration

This implementation imports from the canonical core (memory_core, controllers)
instead of re-implementing memory logic.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

from .memory_core import DualTierMiras, DualTierMirasConfig, LongMemKVCache
from .controllers import MambaBackbone, SinusoidalPositionalEncoding


@dataclass
class OTMemoryAgentConfig:
    """Configuration for OTMemoryAgent (Upgraded v2.1).

    Attributes:
        vocab_size: Size of vocabulary for embedding/output.
        d_model: Hidden dimension.
        n_layers: Number of Mamba/GRU layers in backbone.
        max_seq_len: Maximum sequence length for positional encoding.
        mem_slots: Number of memory slots in DualTierMiras.
        use_external_cache: Whether to use LongMemKVCache.
        cache_capacity: Capacity of external cache if used.
        dropout: Dropout rate.
        temperature: Attention temperature (lower=sharper).
        surprise_threshold: Threshold for deep memory writes.
        use_curiosity: Enable curiosity-driven memory writes.
        use_gradient_checkpointing: Trade compute for memory.
        use_replay_buffer: Enable experience replay buffer for RL.
        replay_capacity: Size of replay buffer.
        memory_n_heads: Number of attention heads for memory.
    """
    vocab_size: int
    d_model: int = 128
    n_layers: int = 2
    max_seq_len: int = 256
    mem_slots: int = 64
    use_external_cache: bool = False
    cache_capacity: int = 4096
    dropout: float = 0.1
    temperature: float = 1.0
    surprise_threshold: float = 0.5
    use_curiosity: bool = False
    use_gradient_checkpointing: bool = False
    # v2.1 enhancements
    use_replay_buffer: bool = False
    replay_capacity: int = 10000
    memory_n_heads: int = 1


class ReplayBuffer:
    """Simple experience replay buffer for RL training.

    Stores (state, action, reward, next_state, done) transitions.
    Supports prioritized sampling based on TD error.
    """

    def __init__(self, capacity: int, d_model: int):
        self.capacity = capacity
        self.d_model = d_model
        self.position = 0
        self.size = 0

        # Pre-allocate storage
        self.states = torch.zeros(capacity, d_model)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self.next_states = torch.zeros(capacity, d_model)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        self.priorities = torch.ones(capacity)

    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool, priority: float = 1.0):
        """Add a transition to the buffer."""
        idx = self.position
        self.states[idx] = state.detach().cpu().squeeze()
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state.detach().cpu().squeeze()
        self.dones[idx] = done
        self.priorities[idx] = priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, prioritized: bool = False):
        """Sample a batch of transitions."""
        if prioritized and self.size > 0:
            probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
            indices = torch.multinomial(probs, batch_size, replacement=True)
        else:
            indices = torch.randint(0, self.size, (batch_size,))

        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
            'indices': indices,
        }

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities for prioritized experience replay."""
        self.priorities[indices] = priorities.cpu() + 1e-6

    def __len__(self):
        return self.size


class OTMemoryAgent(nn.Module):
    """High-level 0T Memory Agent (Upgraded v2.1).

    Architecture:
        - Mamba backbone with residual connections
        - Surprise-gated dual-tier memory
        - Optional external LongMemKVCache for long-term memory
        - Gated fusion of backbone + memory outputs
        - Curiosity-driven memory writes (optional)
        - v2.1: Experience replay buffer for RL
        - v2.1: Multi-head memory attention

    The "0T" property:
        - State size per env is O(D * mem_slots), independent of T
        - Memory provides content-addressable retrieval

    Usage as LM Controller:
        >>> cfg = OTMemoryAgentConfig(vocab_size=256, d_model=128)
        >>> agent = OTMemoryAgent(cfg)
        >>> logits = agent(token_ids)  # [B, T, V]

    Usage for RL (step-by-step):
        >>> agent.reset_memory()
        >>> for obs in observations:
        ...     logits, value = agent.step(obs)
    """

    def __init__(self, cfg: OTMemoryAgentConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            cfg.d_model, cfg.max_seq_len
        )
        self.dropout = nn.Dropout(cfg.dropout)

        # Mamba backbone (uses Mamba2 if available, GRU fallback)
        self.backbone = MambaBackbone(cfg.d_model, cfg.n_layers)

        # DualTierMiras parametric memory with v2.1 upgrades
        mem_cfg = DualTierMirasConfig(
            d_model=cfg.d_model,
            mem_slots=cfg.mem_slots,
            temperature=cfg.temperature,
            surprise_threshold=cfg.surprise_threshold,
            n_heads=cfg.memory_n_heads,
            use_adaptive_retrieval=True,
            use_query_proj=True,
        )
        self.memory = DualTierMiras(mem_cfg)

        # Optional external cache
        self.use_external_cache = cfg.use_external_cache
        if cfg.use_external_cache:
            self.external_cache = LongMemKVCache(
                key_dim=cfg.d_model,
                value_dim=cfg.d_model,
                capacity=cfg.cache_capacity
            )
        else:
            self.external_cache = None

        # Gated fusion: combine backbone output + memory read
        self.gate = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.fusion = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)

        # Output head
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

        # Value head for RL
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, 1)
        )

        # Curiosity module (predicts next hidden state)
        self.use_curiosity = cfg.use_curiosity
        if cfg.use_curiosity:
            self.curiosity_pred = nn.Linear(cfg.d_model, cfg.d_model)

        # v2.1: Experience replay buffer for RL
        self.use_replay_buffer = cfg.use_replay_buffer
        if cfg.use_replay_buffer:
            self.replay_buffer = ReplayBuffer(cfg.replay_capacity, cfg.d_model)
        else:
            self.replay_buffer = None

        # Whether to backprop through memory reads
        self.trainable_memory = False
        self._last_stats = {}
        self._last_hidden = None  # For replay buffer storage

    def reset_memory(self):
        """Reset all memory states. Call at episode/sequence boundaries."""
        self.memory.reset_parameters()
        if self.external_cache is not None:
            self.external_cache.reset()

    def _read_memory(self, h: torch.Tensor) -> torch.Tensor:
        """Read from memory with optional gradient flow.

        Args:
            h: Query tensor [B, d_model]

        Returns:
            Memory value [B, d_model]
        """
        if self.trainable_memory:
            mem_out = self.memory.read(h, context=h)
            v_mem = mem_out["v"]
        else:
            with torch.no_grad():
                mem_out = self.memory.read(h, context=h)
            v_mem = mem_out["v"].detach()

        # Optionally blend with external cache
        if self.external_cache is not None:
            with torch.no_grad():
                ext_keys, ext_vals = self.external_cache.retrieve(h, top_k=4)
                # Simple mean pooling of retrieved values
                ext_v = ext_vals.mean(dim=1)  # [B, d_model]
            # Learned blend ratio
            v_mem = 0.8 * v_mem + 0.2 * ext_v.detach()

        return v_mem

    def _compute_curiosity(self, h_prev: torch.Tensor,
                           h_curr: torch.Tensor) -> torch.Tensor:
        """Compute curiosity bonus as prediction error.

        Returns:
            Curiosity score [B] in [0, 1]
        """
        if not self.use_curiosity or h_prev is None:
            return torch.zeros(h_curr.size(0), device=h_curr.device)

        pred = self.curiosity_pred(h_prev)
        error = (pred - h_curr.detach()).pow(2).mean(dim=-1)
        # Normalize to [0, 1]
        return torch.sigmoid(error - 1.0)

    def _write_memory(self, h: torch.Tensor, v: torch.Tensor):
        """Write to memory (always under no_grad).

        Args:
            h: Key tensor [B, d_model]
            v: Value tensor [B, d_model]
        """
        with torch.no_grad():
            self.memory.update(h, v, weight=None, context=h)
            if self.external_cache is not None:
                self.external_cache.write(h, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward pass.

        Args:
            x: Token ids [B, T]

        Returns:
            Logits [B, T, V]
        """
        B, T = x.shape

        # Embed and encode
        h = self.embed(x)
        h = self.pos_enc(h)
        h = self.dropout(h)

        # Reset memory for this sequence
        self.reset_memory()

        outputs = []
        prev_h = None
        for t in range(T):
            # Process prefix up to t
            h_prefix = h[:, :t+1, :]
            h_enc = self.backbone(h_prefix)[:, -1, :]

            # Read from memory
            v_mem = self._read_memory(h_enc)

            # Gated fusion of backbone + memory
            concat = torch.cat([h_enc, v_mem], dim=-1)
            gate = torch.sigmoid(self.gate(concat))
            fused = gate * h_enc + (1 - gate) * self.fusion(concat)
            fused = self.ln(fused)

            # Output logits
            logits_t = self.head(fused)
            outputs.append(logits_t.unsqueeze(1))

            # Write to memory (with optional curiosity weighting)
            if self.use_curiosity and t > 0:
                curiosity = self._compute_curiosity(prev_h, h_enc)
                self._write_memory(h_enc, fused)
                self._last_stats["curiosity"] = curiosity.mean().item()
            else:
                self._write_memory(h_enc, fused)

            prev_h = h_enc

        return torch.cat(outputs, dim=1)  # [B, T, V]

    def step(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward for RL.

        Args:
            obs: Observation [B, obs_dim] or [B, 1] token ids
            hidden: Not used (memory is internal)

        Returns:
            logits: Action logits [B, V]
            value: State value estimate [B]
        """
        # If obs is token ids, embed them
        if obs.dim() == 2 and obs.shape[1] == 1:
            h = self.embed(obs.squeeze(1))
        elif obs.dim() == 1:
            h = self.embed(obs)
        else:
            # Assume obs is already embedded features
            h = obs

        # Process through backbone (single step)
        h_enc = self.backbone(h.unsqueeze(1))[:, -1, :]

        # Read from memory
        v_mem = self._read_memory(h_enc)

        # Gated fusion
        concat = torch.cat([h_enc, v_mem], dim=-1)
        gate = torch.sigmoid(self.gate(concat))
        fused = gate * h_enc + (1 - gate) * self.fusion(concat)
        fused = self.ln(fused)

        # Outputs
        logits = self.head(fused)
        value = self.value_head(fused).squeeze(-1)

        # Write to memory
        self._write_memory(h_enc, fused)

        return logits, value

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about the agent state."""
        diag = {
            "backbone_type": "mamba" if self.backbone.use_mamba else "gru",
            "mem_slots": self.cfg.mem_slots,
            "trainable_memory": self.trainable_memory,
            "use_curiosity": self.use_curiosity,
        }

        # Add memory stats
        with torch.no_grad():
            diag["mem_fast_norm"] = self.memory.fast_keys.norm().item()
            diag["mem_deep_norm"] = self.memory.deep_keys.norm().item()
            diag["mem_mix_weight"] = torch.sigmoid(
                self.memory.mix_logit
            ).item()
            diag["mem_num_updates"] = self.memory.num_updates.item()

        # Add last stats
        diag.update(self._last_stats)

        return diag

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # v2.1: Replay buffer methods for RL
    def store_transition(self, action: int, reward: float,
                         next_state: torch.Tensor, done: bool):
        """Store a transition in the replay buffer.

        Args:
            action: Action taken
            reward: Reward received
            next_state: Next hidden state
            done: Episode done flag
        """
        if self.replay_buffer is None or self._last_hidden is None:
            return
        self.replay_buffer.push(
            self._last_hidden, action, reward, next_state, done
        )
        self._last_hidden = next_state.clone()

    def sample_replay(self, batch_size: int, prioritized: bool = False):
        """Sample from replay buffer.

        Args:
            batch_size: Number of transitions to sample
            prioritized: Use prioritized experience replay

        Returns:
            Dict with states, actions, rewards, next_states, dones
        """
        if self.replay_buffer is None or len(self.replay_buffer) < batch_size:
            return None
        return self.replay_buffer.sample(batch_size, prioritized)

    def update_replay_priorities(self, indices: torch.Tensor,
                                  td_errors: torch.Tensor):
        """Update replay buffer priorities based on TD errors."""
        if self.replay_buffer is not None:
            self.replay_buffer.update_priorities(indices, td_errors.abs())


def build_ot_agent(cfg: OTMemoryAgentConfig) -> OTMemoryAgent:
    """Factory function to build OTMemoryAgent.

    Args:
        cfg: Agent configuration

    Returns:
        Initialized OTMemoryAgent
    """
    return OTMemoryAgent(cfg)
