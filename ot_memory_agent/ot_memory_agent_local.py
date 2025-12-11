#!/usr/bin/env python3
"""
ot_memory_agent_local.py

0T Memory AI - Complete Implementation

Features:

- StreamingSSMCell with multi-timescale initialization
- Batched vectorized environments for efficient training
- PPO with full-sequence BPTT through the SSM state
- Works on CPU, CUDA GPU, or Mac MPS automatically

The "0T memory" property:

- State size per env is O(D*K), independent of sequence length T
- Compute per step is O(1), so total is O(T)
- Real long-horizon memory via SSM poles, not history buffers

Run:
python ot_memory_agent_local.py

Requirements:
pip install torch numpy matplotlib
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================

# CONFIGURATION - Adjust these to experiment

# ============================================================

CONFIG = {
    # Environment
    "horizon": 40,              # Delay between cue and query (memory requirement)
    "num_actions": 4,           # Number of cues (random baseline = 1/num_actions = 25%)
    "num_envs": 16,             # Parallel environments for faster training
    "noise_scale": 0.0,         # Noise during delay period (0.0 = clean)

    # Model architecture
    "d_model": 64,              # Hidden dimension (number of SSM channels)
    "K": 4,                     # Conv kernel size (local pattern window)
    "min_timescale": 10.0,      # Shortest memory timescale (τ_min)
    "max_timescale": 2000.0,    # Longest memory timescale (τ_max)

    # Training hyperparameters
    "num_updates": 300,         # Number of PPO updates
    "lr": 3e-4,                 # Learning rate
    "gamma": 0.99,              # Discount factor
    "lam": 0.95,                # GAE lambda
    "clip_eps": 0.2,            # PPO clip epsilon
    "ppo_epochs": 4,            # PPO epochs per update
    "max_grad_norm": 0.5,       # Gradient clipping
    "entropy_coef": 0.01,       # Entropy bonus coefficient
    "value_coef": 0.5,          # Value loss coefficient

    # Logging
    "log_interval": 10,         # Print every N updates

    # Reproducibility
    "seed": 42,
}

# ============================================================

# Device Selection

# ============================================================

def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ============================================================

# Vectorized Delayed Cue Environment

# ============================================================

class VectorizedDelayedCueEnv:
"""
Batched delayed-cue task that requires genuine memory.

Timeline (per env, length = horizon):
    t = 0:        [CUE]    cue visible as one-hot, is_start=1
    t = 1..H-2:   [DELAY]  zeros (+ optional noise)
    t = H-1:      [QUERY]  is_query=1, cue NOT visible

The agent must remember the cue across the delay to answer correctly.

Reward:
    +1 at query step if action == cue
    0 otherwise

Observation: (num_envs, obs_dim) where obs_dim = num_actions + 2
    [one_hot(cue), is_start, is_query]

This task is impossible without memory - random baseline = 1/num_actions.
"""

def __init__(
    self,
    num_envs: int,
    horizon: int = 40,
    num_actions: int = 4,
    noise_scale: float = 0.0,
    seed: int = 0,
):
    assert horizon >= 3, "Horizon must be at least 3 (cue, delay, query)"
    self.num_envs = num_envs
    self.horizon = horizon
    self.num_actions = num_actions
    self.noise_scale = noise_scale
    self.obs_dim = num_actions + 2  # one-hot cue + is_start + is_query

    self.rng = np.random.RandomState(seed)

    # Per-env state
    self.t = np.zeros(num_envs, dtype=np.int32)
    self.cues = np.zeros(num_envs, dtype=np.int32)
    self.dones = np.zeros(num_envs, dtype=bool)

    self.reset_all()

def reset_all(self) -> np.ndarray:
    """Reset all environments and return initial observations."""
    self.t[:] = 0
    self.dones[:] = False
    self.cues[:] = self.rng.randint(0, self.num_actions, size=self.num_envs)
    return self._get_obs()

def reset_done(self) -> np.ndarray:
    """Reset only environments that are done."""
    done_mask = self.dones
    if done_mask.any():
        num_done = done_mask.sum()
        self.t[done_mask] = 0
        self.dones[done_mask] = False
        self.cues[done_mask] = self.rng.randint(
            0, self.num_actions, size=num_done
        )
    return self._get_obs()

def _get_obs(self) -> np.ndarray:
    """Generate observations for all environments."""
    obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

    for i in range(self.num_envs):
        if self.dones[i]:
            continue  # Return zeros for done envs

        if self.t[i] == 0:
            # Cue step: show cue as one-hot + is_start flag
            obs[i, self.cues[i]] = 1.0
            obs[i, -2] = 1.0  # is_start
        elif self.t[i] == self.horizon - 1:
            # Query step: only is_query flag, cue NOT visible
            obs[i, -1] = 1.0  # is_query
        else:
            # Delay step: zeros (+ optional noise)
            if self.noise_scale > 0:
                obs[i, :self.num_actions] = (
                    self.rng.randn(self.num_actions) * self.noise_scale
                )
    return obs

def step(
    self, actions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Step all environments.

    Args:
        actions: (num_envs,) int array of actions

    Returns:
        obs: (num_envs, obs_dim) next observations
        rewards: (num_envs,) rewards
        dones: (num_envs,) done flags
        info: dict with success statistics
    """
    rewards = np.zeros(self.num_envs, dtype=np.float32)

    # Compute rewards at query step
    at_query = (self.t == self.horizon - 1) & (~self.dones)
    if at_query.any():
        correct = actions[at_query] == self.cues[at_query]
        rewards[at_query] = correct.astype(np.float32)

    # Advance time
    self.t += 1

    # Mark done
    self.dones = self.t >= self.horizon

    # Get new observations
    obs = self._get_obs()

    # Info for logging
    info = {
        "num_done": int(self.dones.sum()),
        "successes": float(rewards.sum()) if at_query.any() else 0.0,
    }

    return obs, rewards, self.dones.copy(), info

# ============================================================

# SSM State Container

# ============================================================

@dataclass
class SSMState:
"""
Recurrent state for StreamingSSMCell.

This is the "0T memory" - fixed size regardless of sequence length.

Attributes:
    conv_state: (B, D, K-1) - rolling window of last K-1 inputs per channel
    ssm_state: (B, D) - scalar SSM state per channel
"""
conv_state: torch.Tensor  # (B, D, K-1)
ssm_state: torch.Tensor   # (B, D)

def detach(self) -> "SSMState":
    """Detach from computation graph (for truncated BPTT if needed)."""
    return SSMState(
        self.conv_state.detach(),
        self.ssm_state.detach(),
    )

def clone(self) -> "SSMState":
    """Create a deep copy."""
    return SSMState(
        self.conv_state.clone(),
        self.ssm_state.clone(),
    )

def mask_done(self, done: torch.Tensor) -> "SSMState":
    """
    Zero out states where done=True (episode boundary handling).

    Args:
        done: (B,) bool tensor indicating which envs are done
    """
    if done is None or not done.any():
        return self
    mask = (~done).float()
    conv = self.conv_state * mask.view(-1, 1, 1)
    ssm = self.ssm_state * mask.view(-1, 1)
    return SSMState(conv, ssm)

# ============================================================

# 0T Memory Cell: Streaming SSM

# ============================================================

class StreamingSSMCell(nn.Module):
"""
0T Memory Cell: O(1) state size, O(T) compute for sequence length T.

Architecture:
    1. Depthwise 1D conv window of size K (FIR, local patterns)
    2. Scalar SSM per channel with pole A ∈ (0,1) (IIR, long-range memory)

Math per channel d at time t:
    u_t = Σ_k W_conv[d,k] * w_t[k] + b_conv[d]     (conv over window)
    s_t = A[d] * s_{t-1} + B[d] * u_t              (SSM recurrence)
    y_t = C[d] * s_t + D_skip[d] * u_t            (output)

The SSM pole A controls memory horizon:
    - A close to 1 → long memory (τ ≈ 1/(1-A))
    - A close to 0 → short memory

State size per env: O(D * K), independent of sequence length T.
This is the key "0T memory" property.
"""

def __init__(
    self,
    d_model: int,
    K: int = 4,
    min_timescale: float = 10.0,
    max_timescale: float = 1000.0,
):
    """
    Args:
        d_model: Number of channels (hidden dimension)
        K: Conv kernel size (window of K inputs)
        min_timescale: Shortest memory timescale τ_min
        max_timescale: Longest memory timescale τ_max

    Channels are initialized with timescales geometrically spaced
    from min to max, giving a "filterbank" of memory horizons.
    """
    super().__init__()
    self.d_model = d_model
    self.K = K
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale

    # Conv parameters (depthwise, per-channel)
    self.W_conv = nn.Parameter(torch.empty(d_model, K))
    self.b_conv = nn.Parameter(torch.zeros(d_model))

    # SSM parameters
    # A_raw is unconstrained; A = exp(-softplus(A_raw)) ∈ (0, 1)
    self.A_raw = nn.Parameter(torch.zeros(d_model))
    self.B = nn.Parameter(torch.empty(d_model))
    self.C = nn.Parameter(torch.empty(d_model))
    self.D_skip = nn.Parameter(torch.empty(d_model))

    self._reset_parameters()

@property
def A(self) -> torch.Tensor:
    """
    Stable SSM pole in (0, 1) per channel.

    Using A = exp(-softplus(A_raw)) ensures:
    - A is always in (0, 1) for stability
    - Gradients don't saturate near A=1 (unlike sigmoid)
    - Easy to initialize for specific timescales
    """
    return torch.exp(-F.softplus(self.A_raw))

def _make_A_init(self) -> torch.Tensor:
    """
    Create geometric grid of timescales τ ∈ [min, max].

    Timescale τ relates to A by: τ ≈ 1/(1-A), so A = 1 - 1/τ.

    This gives a "filterbank" of memory horizons:
    - Some channels for short-term patterns (small τ, A~0.9)
    - Some channels for long-term memory (large τ, A~0.999)
    """
    taus = torch.logspace(
        math.log10(self.min_timescale),
        math.log10(self.max_timescale),
        steps=self.d_model,
    )
    A_init = 1.0 - 1.0 / taus
    return A_init.clamp(1e-4, 1.0 - 1e-4)

def _init_A_raw_from_A(self, A_init: torch.Tensor):
    """
    Invert A = exp(-softplus(A_raw)) to initialize A_raw.

    Given target A, we need:
        softplus(A_raw) = -log(A)
        A_raw = inverse_softplus(-log(A))
              = log(exp(-log(A)) - 1)
              = log(1/A - 1)
    """
    with torch.no_grad():
        # softplus(A_raw) = -log(A_init)
        target_s = -torch.log(A_init.clamp(min=1e-6))
        # inverse softplus: A_raw = log(exp(s) - 1)
        eps = 1e-6
        self.A_raw.copy_(
            torch.log((torch.exp(target_s) - 1.0).clamp(min=eps))
        )

def _reset_parameters(self):
    """Initialize all parameters."""
    D, K = self.d_model, self.K

    with torch.no_grad():
        # Conv: identity-like initialization
        # W_conv[:, -1] = 1 means "pass through current input"
        self.W_conv.zero_()
        self.W_conv[:, -1] = 1.0
        self.b_conv.zero_()

        # Multi-timescale A initialization
        A_init = self._make_A_init()
        self._init_A_raw_from_A(A_init)

    # B scaled by (1-A) to keep steady-state gain modest
    # Steady-state gain of SSM is B/(1-A), so this keeps it ~0.1
    with torch.no_grad():
        A = self.A
        std_B = 0.1 * (1.0 - A)
        self.B.copy_(torch.randn(D) * std_B)

    # C and D_skip: small random init
    nn.init.normal_(self.C, mean=0.0, std=0.05)
    # D_skip starts near 1 for initial "pass-through" behavior
    nn.init.normal_(self.D_skip, mean=1.0, std=0.01)

def initial_state(
    self,
    batch_size: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> SSMState:
    """Create zero-initialized state for a batch of sequences."""
    device = device or self.W_conv.device
    dtype = dtype or self.W_conv.dtype
    conv_state = torch.zeros(
        batch_size,
        self.d_model,
        max(self.K - 1, 0),
        device=device,
        dtype=dtype,
    )
    ssm_state = torch.zeros(
        batch_size,
        self.d_model,
        device=device,
        dtype=dtype,
    )
    return SSMState(conv_state, ssm_state)

def forward(
    self,
    x_t: torch.Tensor,
    state: SSMState,
) -> Tuple[torch.Tensor, SSMState]:
    """
    Single step forward pass.

    This is the core "0T memory" operation:
    - Takes current input x_t and fixed-size state
    - Produces output y_t and updated state
    - No growing history buffer, O(1) per step

    Args:
        x_t: (B, D) input at time t
        state: SSMState with conv_state (B,D,K-1) and ssm_state (B,D)

    Returns:
        y_t: (B, D) output
        new_state: updated SSMState
    """
    B, D = x_t.shape
    assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"
    K = self.K

    conv_state = state.conv_state
    ssm_state = state.ssm_state

    # Build conv window: [conv_state (last K-1 inputs), x_t (current)]
    if K > 1:
        w_full = torch.cat([conv_state, x_t.unsqueeze(-1)], dim=-1)  # (B, D, K)
    else:
        w_full = x_t.unsqueeze(-1)  # (B, D, 1)

    # Depthwise conv: u = sum_k W[d,k] * w[d,k] + b[d]
    W = self.W_conv.unsqueeze(0)  # (1, D, K)
    u = (w_full * W).sum(dim=-1) + self.b_conv  # (B, D)

    # SSM recurrence: z = A * s_prev + B * u
    A = self.A
    z = A * ssm_state + self.B * u  # (B, D)

    # Output: y = C * z + D_skip * u
    y = self.C * z + self.D_skip * u  # (B, D)

    # Update conv state: shift left, append x_t
    if K > 1:
        new_conv = torch.cat(
            [conv_state[:, :, 1:], x_t.unsqueeze(-1)],
            dim=-1,
        )
    else:
        new_conv = conv_state

    return y, SSMState(new_conv, z)

def get_diagnostics(self) -> Dict[str, float]:
    """Return diagnostic statistics about SSM parameters."""
    A = self.A.detach()
    return {
        "A_min": A.min().item(),
        "A_max": A.max().item(),
        "A_mean": A.mean().item(),
        "A_std": A.std().item(),
        "effective_horizon_min": (1.0 / (1.0 - A.min() + 1e-8)).item(),
        "effective_horizon_max": (1.0 / (1.0 - A.max() + 1e-8)).item(),
    }

# ============================================================

# Policy Network

# ============================================================

class OTMPolicy(nn.Module):
"""
0T Memory Policy Network.

Architecture:
    obs_t → obs_embed → StreamingSSMCell → policy_head → logits
                                         → value_head → value

The SSM provides memory across time steps. Full BPTT through
the SSM state enables proper credit assignment for long horizons.
"""

def __init__(
    self,
    obs_dim: int,
    act_dim: int,
    d_model: int = 64,
    K: int = 4,
    min_timescale: float = 10.0,
    max_timescale: float = 1000.0,
):
    super().__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.d_model = d_model

    # Observation embedding
    self.obs_embed = nn.Sequential(
        nn.Linear(obs_dim, d_model),
        nn.ReLU(),
    )

    # 0T Memory core
    self.ssm = StreamingSSMCell(
        d_model=d_model,
        K=K,
        min_timescale=min_timescale,
        max_timescale=max_timescale,
    )

    # Output heads
    self.policy_head = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, act_dim),
    )
    self.value_head = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, 1),
    )

def initial_state(
    self,
    batch_size: int,
    device: Optional[torch.device] = None,
) -> SSMState:
    """Create initial SSM state for a batch."""
    return self.ssm.initial_state(batch_size, device=device)

def forward(
    self,
    obs_t: torch.Tensor,
    state: SSMState,
) -> Tuple[torch.Tensor, torch.Tensor, SSMState]:
    """
    Single step forward.

    Args:
        obs_t: (B, obs_dim) observation at time t
        state: SSMState from previous step

    Returns:
        logits: (B, act_dim) action logits
        value: (B,) state value estimate
        new_state: updated SSMState
    """
    x = self.obs_embed(obs_t)  # (B, D)
    y, new_state = self.ssm(x, state)  # (B, D)

    logits = self.policy_head(y)  # (B, act_dim)
    value = self.value_head(y).squeeze(-1)  # (B,)

    return logits, value, new_state

def get_diagnostics(self) -> Dict[str, float]:
    """Return SSM diagnostics."""
    return self.ssm.get_diagnostics()

# ============================================================

# PPO Utilities

# ============================================================

def compute_gae(
rewards: torch.Tensor,
values: torch.Tensor,
dones: torch.Tensor,
gamma: float,
lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
"""
Compute Generalized Advantage Estimation (GAE).

Args:
    rewards: (T, B) rewards at each timestep
    values: (T, B) value estimates at each timestep
    dones: (T, B) episode termination flags
    gamma: discount factor
    lam: GAE lambda

Returns:
    returns: (T, B) discounted returns
    advantages: (T, B) GAE advantages
"""
T, B = rewards.shape
advantages = torch.zeros_like(rewards)
last_adv = torch.zeros(B, device=rewards.device)
last_value = torch.zeros(B, device=rewards.device)

for t in reversed(range(T)):
    mask = (~dones[t]).float()
    delta = rewards[t] + gamma * last_value * mask - values[t]
    last_adv = delta + gamma * lam * last_adv * mask
    advantages[t] = last_adv
    last_value = values[t]

returns = advantages + values
return returns, advantages

# ============================================================

# Trainer

# ============================================================

class Trainer:
"""Complete PPO training loop with logging and visualization."""

def __init__(self, config: dict):
    self.config = config
    self.device = get_device()
    print(f"Using device: {self.device}")

    # Set random seeds for reproducibility
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create environment
    self.env = VectorizedDelayedCueEnv(
        num_envs=config["num_envs"],
        horizon=config["horizon"],
        num_actions=config["num_actions"],
        noise_scale=config["noise_scale"],
        seed=seed,
    )

    # Create policy
    self.policy = OTMPolicy(
        obs_dim=self.env.obs_dim,
        act_dim=config["num_actions"],
        d_model=config["d_model"],
        K=config["K"],
        min_timescale=config["min_timescale"],
        max_timescale=config["max_timescale"],
    ).to(self.device)

    # Count parameters
    num_params = sum(p.numel() for p in self.policy.parameters())
    print(f"Policy parameters: {num_params:,}")

    # Optimizer
    self.optimizer = torch.optim.Adam(
        self.policy.parameters(),
        lr=config["lr"],
    )

    # Training history for plotting
    self.history = {
        "update": [],
        "success_rate": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "A_min": [],
        "A_max": [],
        "A_mean": [],
    }

    # Running success rate (exponential moving average)
    self.running_success = 0.0
    self.success_alpha = 0.1

def collect_rollout(self) -> dict:
    """
    Collect one full episode from all environments.

    Returns a dict with:
        obs: (T, B, obs_dim)
        actions: (T, B)
        old_logp: (T, B)
        rewards: (T, B)
        values: (T, B)
        dones: (T, B)
        success_rate: float
        running_success: float
    """
    cfg = self.config
    num_envs = cfg["num_envs"]
    horizon = cfg["horizon"]

    # Storage lists
    obs_list, act_list, logp_list = [], [], []
    rew_list, val_list, done_list = [], [], []

    # Reset all environments
    obs = self.env.reset_all()
    state = self.policy.initial_state(num_envs, device=self.device)

    total_successes = 0.0
    total_episodes = 0

    # Collect trajectory
    for _ in range(horizon):
        obs_t = torch.tensor(obs, device=self.device, dtype=torch.float32)

        # Get action from policy (no gradients during collection)
        with torch.no_grad():
            logits, value, state = self.policy(obs_t, state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

        # Step environment
        act_np = action.cpu().numpy()
        next_obs, rewards, dones, info = self.env.step(act_np)

        # Store transition
        obs_list.append(obs_t)
        act_list.append(action)
        logp_list.append(logp)
        rew_list.append(
            torch.tensor(rewards, device=self.device, dtype=torch.float32)
        )
        val_list.append(value)
        done_list.append(
            torch.tensor(dones, device=self.device, dtype=torch.bool)
        )

        # Track success
        total_successes += info["successes"]
        total_episodes += info["num_done"]

        # Handle episode boundaries
        if dones.any():
            state = state.mask_done(done_list[-1])
            obs = self.env.reset_done()
        else:
            obs = next_obs

    # Stack into tensors
    rollout = {
        "obs": torch.stack(obs_list),        # (T, B, obs_dim)
        "actions": torch.stack(act_list),    # (T, B)
        "old_logp": torch.stack(logp_list),  # (T, B)
        "rewards": torch.stack(rew_list),    # (T, B)
        "values": torch.stack(val_list),     # (T, B)
        "dones": torch.stack(done_list),     # (T, B)
    }

    # Compute success rate
    success_rate = total_successes / max(total_episodes, 1)
    self.running_success = (
        (1 - self.success_alpha) * self.running_success
        + self.success_alpha * success_rate
    )
    rollout["success_rate"] = success_rate
    rollout["running_success"] = self.running_success

    return rollout

def ppo_update(self, rollout: dict) -> dict:
    """
    Perform PPO update with full-sequence BPTT through SSM.

    This is critical for proper credit assignment - gradients
    flow through the entire recurrent state sequence.
    """
    cfg = self.config
    T, B = rollout["rewards"].shape

    # Compute returns and advantages
    returns, advantages = compute_gae(
        rollout["rewards"],
        rollout["values"],
        rollout["dones"],
        gamma=cfg["gamma"],
        lam=cfg["lam"],
    )

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Flatten for loss computation
    obs_flat = rollout["obs"].view(T * B, -1)
    actions_flat = rollout["actions"].view(T * B)
    old_logp_flat = rollout["old_logp"].view(T * B)
    returns_flat = returns.view(T * B)
    advantages_flat = advantages.view(T * B)

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    # Multiple PPO epochs
    for _ in range(cfg["ppo_epochs"]):
        # Re-run full sequence WITH gradients for BPTT
        state = self.policy.initial_state(B, device=self.device)
        new_logits_list, new_values_list = [], []

        for t in range(T):
            obs_t = rollout["obs"][t]
            logits_t, value_t, state = self.policy(obs_t, state)
            new_logits_list.append(logits_t)
            new_values_list.append(value_t)

            # Handle done masking (zero state at episode boundaries)
            if rollout["dones"][t].any():
                state = state.mask_done(rollout["dones"][t])

        new_logits = torch.stack(new_logits_list).view(T * B, -1)
        new_values = torch.stack(new_values_list).view(T * B)

        # Compute new log probabilities
        dist = torch.distributions.Categorical(logits=new_logits)
        new_logp = dist.log_prob(actions_flat)

        # PPO clipped surrogate loss
        ratio = torch.exp(new_logp - old_logp_flat)
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(
            ratio,
            1.0 - cfg["clip_eps"],
            1.0 + cfg["clip_eps"],
        ) * advantages_flat
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, returns_flat)

        # Entropy bonus (encourages exploration)
        entropy = dist.entropy().mean()

        # Combined loss
        loss = (
            policy_loss
            + cfg["value_coef"] * value_loss
            - cfg["entropy_coef"] * entropy
        )

        # Gradient step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy.parameters(), cfg["max_grad_norm"]
        )
        self.optimizer.step()

        # Accumulate metrics
        metrics["policy_loss"] += policy_loss.item() / cfg["ppo_epochs"]
        metrics["value_loss"] += value_loss.item() / cfg["ppo_epochs"]
        metrics["entropy"] += entropy.item() / cfg["ppo_epochs"]

    return metrics

def train(self):
    """Main training loop."""
    cfg = self.config

    print("=" * 70)
    print("0T MEMORY AGENT TRAINING")
    print("=" * 70)
    print(
        f"Task: Delayed cue with horizon={cfg['horizon']}, "
        f"num_actions={cfg['num_actions']}"
    )
    print(f"Random baseline: {100/cfg['num_actions']:.1f}%")
    print(
        f"Model: d_model={cfg['d_model']}, K={cfg['K']}, "
        f"τ ∈ [{cfg['min_timescale']}, {cfg['max_timescale']}]"
    )
    print("=" * 70)
    print()

    start_time = time.time()

    for update in range(1, cfg["num_updates"] + 1):
        # Collect rollout
        rollout = self.collect_rollout()

        # PPO update
        metrics = self.ppo_update(rollout)

        # Get SSM diagnostics
        diag = self.policy.get_diagnostics()

        # Record history
        self.history["update"].append(update)
        self.history["success_rate"].append(rollout["running_success"])
        self.history["policy_loss"].append(metrics["policy_loss"])
        self.history["value_loss"].append(metrics["value_loss"])
        self.history["entropy"].append(metrics["entropy"])
        self.history["A_min"].append(diag["A_min"])
        self.history["A_max"].append(diag["A_max"])
        self.history["A_mean"].append(diag["A_mean"])

        # Print progress
        if update % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            print(
                f"Update {update:4d}/{cfg['num_updates']} | "
                f"Success: {rollout['running_success']*100:5.1f}% | "
                f"π_loss: {metrics['policy_loss']:7.4f} | "
                f"v_loss: {metrics['value_loss']:7.4f} | "
                f"H: {metrics['entropy']:5.3f} | "
                f"A ∈ [{diag['A_min']:.3f}, {diag['A_max']:.3f}] | "
                f"Time: {elapsed:.1f}s"
            )

    # Final summary
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final success rate: {self.history['success_rate'][-1]*100:.1f}%")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("=" * 70)

    # Plot results
    self.plot_progress()

def plot_progress(self):
    """Plot training curves."""
    updates = self.history["update"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Success rate
    ax = axes[0, 0]
    ax.plot(
        updates,
        [s * 100 for s in self.history["success_rate"]],
        "b-",
        linewidth=2,
        label="Success rate",
    )
    ax.axhline(
        100.0 / self.config["num_actions"],
        color="r",
        linestyle="--",
        label=f"Random ({100/self.config['num_actions']:.0f}%)",
    )
    ax.axhline(100, color="g", linestyle="--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Update")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Learning Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Losses
    ax = axes[0, 1]
    ax.plot(updates, self.history["policy_loss"], "b-", label="Policy loss")
    ax.plot(updates, self.history["value_loss"], "r-", label="Value loss")
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("PPO Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 0]
    ax.plot(updates, self.history["entropy"], "g-", linewidth=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.grid(True, alpha=0.3)

    # SSM A values
    ax = axes[1, 1]
    ax.fill_between(
        updates,
        self.history["A_min"],
        self.history["A_max"],
        alpha=0.3,
        color="purple",
        label="A range",
    )
    ax.plot(
        updates,
        self.history["A_mean"],
        "purple",
        linewidth=2,
        label="A mean",
    )
    ax.set_xlabel("Update")
    ax.set_ylabel("SSM Pole A")
    ax.set_title("SSM Eigenvalue Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("ot_memory_training.png", dpi=150)
    print("\nPlot saved to: ot_memory_training.png")
    plt.show()

# ============================================================

# Main Entry Point

# ============================================================

def main():
"""Run training with current CONFIG."""
print("\n" + "=" * 70)
print("0T MEMORY AGENT")
print("=" * 70)
print("A truly recurrent SSM agent with O(1) state size per step.")
print("No history buffers, no O(T²) tricks - just proper SSM math.")
print("=" * 70 + "\n")

trainer = Trainer(CONFIG)
trainer.train()
return trainer

if **name** == "**main**":
trainer = main()