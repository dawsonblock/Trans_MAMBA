#!/usr/bin/env python3
"""
ot_memory_agent_unified.py

0T Memory Agent with two backends:

```
controller_type = "streaming"  -> StreamingSSMCell (fixed A/B/C/D)
controller_type = "mamba"      -> MambaCell (selective, input-dependent)
```

Both preserve the 0T property: state per env is O(1) in time (no growth with horizon),
and we train with full-sequence BPTT via PPO.

Presets:
- "base"         : horizon=40, 4 actions (25% baseline)
- "long_horizon" : horizon=100, 4 actions
- "hard_task"    : horizon=100, 8 actions (12.5% baseline)
- "stress_test"  : horizon=500, 8 actions

Requirements:
pip install torch numpy matplotlib

Usage:
python ot_memory_agent_unified.py

```
To change settings, edit CONFIG["controller_type"] and PRESET at the top.
```

"""

import math
import random
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================

# CONFIGURATION + PRESETS

# ============================================================

CONFIG = {
# Controller: "streaming" or "mamba"
"controller_type": "streaming",

```
# Environment
"horizon": 40,          # will be overridden by PRESET below
"num_actions": 4,
"num_envs": 16,
"noise_scale": 0.0,

# Streaming SSM model (used if controller_type == "streaming")
"stream_d_model": 64,
"stream_K": 4,
"stream_min_timescale": 10.0,
"stream_max_timescale": 2000.0,   # will be scaled with horizon below

# Mamba model (used if controller_type == "mamba")
"mamba_d_model": 64,
"mamba_d_state": 16,
"mamba_d_conv": 4,
"mamba_expand": 2,
"mamba_dt_rank": "auto",
"mamba_dt_min": 0.001,
"mamba_dt_max": 0.1,    # base dt_max; will be scaled by horizon in forward_step

# Training
"num_updates": 300,
"lr": 3e-4,
"gamma": 0.99,
"lam": 0.95,
"clip_eps": 0.2,
"ppo_epochs": 4,
"max_grad_norm": 0.5,
"entropy_coef": 0.01,
"value_coef": 0.5,

# Logging
"log_interval": 10,

# Seed
"seed": 42,
```

}

# –––––––– PRESETS ––––––––

# Options: "base", "long_horizon", "hard_task", "stress_test"

PRESET = "base"  # change this to test different regimes

if PRESET == "base":
CONFIG["horizon"] = 40
CONFIG["num_actions"] = 4
CONFIG["num_updates"] = 300
CONFIG["noise_scale"] = 0.0
CONFIG["stream_max_timescale"] = 10.0 * CONFIG["horizon"]

elif PRESET == "long_horizon":
CONFIG["horizon"] = 100
CONFIG["num_actions"] = 4
CONFIG["num_updates"] = 500
CONFIG["noise_scale"] = 0.0
CONFIG["stream_max_timescale"] = 10.0 * CONFIG["horizon"]

elif PRESET == "hard_task":
CONFIG["horizon"] = 100
CONFIG["num_actions"] = 8  # baseline = 12.5%
CONFIG["num_updates"] = 500
CONFIG["noise_scale"] = 0.0
CONFIG["stream_max_timescale"] = 10.0 * CONFIG["horizon"]

elif PRESET == "stress_test":
CONFIG["horizon"] = 500
CONFIG["num_actions"] = 8
CONFIG["num_updates"] = 500
CONFIG["noise_scale"] = 0.0
CONFIG["stream_max_timescale"] = 10.0 * CONFIG["horizon"]

else:
raise ValueError(f"Unknown PRESET: {PRESET}")

# ============================================================

# DEVICE SELECTION

# ============================================================

def get_device() -> torch.device:
"""Auto-detect best available device."""
if torch.cuda.is_available():
return torch.device("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
return torch.device("mps")
return torch.device("cpu")

# ============================================================

# VECTORIZED DELAYED CUE ENVIRONMENT

# ============================================================

class VectorizedDelayedCueEnv:
"""
Batched delayed cue environment requiring genuine memory.

```
Timeline per env (length = horizon):
    t = 0      : cue visible (one-hot), is_start=1
    t = 1..H-2 : delay (optionally noisy)
    t = H-1    : query, is_query=1 (cue NOT visible)

Reward: +1 at query if action == cue, else 0.
Random baseline = 1/num_actions.
"""

def __init__(
    self,
    num_envs: int,
    horizon: int = 40,
    num_actions: int = 4,
    noise_scale: float = 0.0,
    seed: int = 0,
):
    assert horizon >= 3, "Horizon must be at least 3"
    self.num_envs = num_envs
    self.horizon = horizon
    self.num_actions = num_actions
    self.noise_scale = noise_scale
    self.obs_dim = num_actions + 2  # [one-hot cue, is_start, is_query]

    self.rng = np.random.RandomState(seed)
    self.t = np.zeros(num_envs, dtype=np.int32)
    self.cues = np.zeros(num_envs, dtype=np.int64)  # int64 for safe comparisons
    self.dones = np.zeros(num_envs, dtype=bool)

    self.reset_all()

def reset_all(self) -> np.ndarray:
    """Reset all environments."""
    self.t[:] = 0
    self.dones[:] = False
    self.cues[:] = self.rng.randint(0, self.num_actions, size=self.num_envs)
    return self._get_obs()

def reset_done(self) -> np.ndarray:
    """Reset only environments that are done."""
    # Use copy() to avoid mutation during masking
    done_mask = self.dones.copy()
    if done_mask.any():
        num_done = int(done_mask.sum())
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
            continue
        if self.t[i] == 0:
            # Cue step
            obs[i, self.cues[i]] = 1.0
            obs[i, -2] = 1.0  # is_start
        elif self.t[i] == self.horizon - 1:
            # Query step
            obs[i, -1] = 1.0  # is_query
        else:
            # Delay step
            if self.noise_scale > 0:
                obs[i, :self.num_actions] = (
                    self.rng.randn(self.num_actions) * self.noise_scale
                )
    return obs

def step(
    self, actions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Step all environments."""
    rewards = np.zeros(self.num_envs, dtype=np.float32)
    at_query = (self.t == self.horizon - 1) & (~self.dones)
    if at_query.any():
        correct = actions[at_query] == self.cues[at_query]
        rewards[at_query] = correct.astype(np.float32)

    self.t += 1
    self.dones = self.t >= self.horizon
    obs = self._get_obs()

    info = {
        "num_done": int(self.dones.sum()),
        "successes": float(rewards.sum()) if at_query.any() else 0.0,
    }
    return obs, rewards, self.dones.copy(), info
```

# ============================================================

# STREAMING SSM CELL (FIXED A/B/C/D, 0T)

# ============================================================

@dataclass
class SSMState:
"""State container for StreamingSSMCell."""
conv_state: torch.Tensor  # (B, D, K-1)
ssm_state: torch.Tensor   # (B, D)

```
def detach(self) -> "SSMState":
    return SSMState(self.conv_state.detach(), self.ssm_state.detach())

def clone(self) -> "SSMState":
    return SSMState(self.conv_state.clone(), self.ssm_state.clone())

def mask_done(self, done: torch.Tensor) -> "SSMState":
    """Zero out states where done=True."""
    if done is None or not done.any():
        return self
    mask = (~done).float()
    conv = self.conv_state * mask.view(-1, 1, 1)
    ssm = self.ssm_state * mask.view(-1, 1)
    return SSMState(conv, ssm)
```

class StreamingSSMCell(nn.Module):
"""
0T streaming SSM with fixed parameters:
- Depthwise conv window of size K (local patterns)
- Scalar SSM per channel with pole A in (0,1) (long-range memory)

```
State per env: O(D*K), independent of sequence length.

Multi-timescale initialization: A values are initialized on a log-spaced
grid from min_timescale to max_timescale.
"""

def __init__(
    self,
    d_model: int,
    K: int = 4,
    min_timescale: float = 10.0,
    max_timescale: float = 1000.0,
):
    super().__init__()
    self.d_model = d_model
    self.K = K
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale

    # Conv parameters
    self.W_conv = nn.Parameter(torch.empty(d_model, K))
    self.b_conv = nn.Parameter(torch.zeros(d_model))

    # SSM parameters: A_raw -> A = exp(-softplus(A_raw)) in (0,1)
    self.A_raw = nn.Parameter(torch.zeros(d_model))
    self.B = nn.Parameter(torch.empty(d_model))
    self.C = nn.Parameter(torch.empty(d_model))
    self.D_skip = nn.Parameter(torch.empty(d_model))

    self._reset_parameters()

@property
def A(self) -> torch.Tensor:
    """Stable SSM pole in (0, 1) per channel."""
    return torch.exp(-F.softplus(self.A_raw))

def _make_A_init(self) -> torch.Tensor:
    """Create log-spaced timescale grid for A initialization."""
    taus = torch.logspace(
        math.log10(self.min_timescale),
        math.log10(self.max_timescale),
        steps=self.d_model,
    )
    A_init = 1.0 - 1.0 / taus
    return A_init.clamp(1e-4, 1.0 - 1e-4)

def _init_A_raw_from_A(self, A_init: torch.Tensor):
    """Invert A = exp(-softplus(A_raw)) to initialize A_raw."""
    with torch.no_grad():
        target_s = -torch.log(A_init.clamp(min=1e-6))
        eps = 1e-6
        self.A_raw.copy_(
            torch.log((torch.exp(target_s) - 1.0).clamp(min=eps))
        )

def _reset_parameters(self):
    """Initialize all parameters."""
    D, K = self.d_model, self.K
    with torch.no_grad():
        # Conv: identity-like (pass through last input)
        self.W_conv.zero_()
        self.W_conv[:, -1] = 1.0
        self.b_conv.zero_()

        # Multi-timescale A initialization
        A_init = self._make_A_init()
        self._init_A_raw_from_A(A_init)

    # B scaled by (1-A) for stable steady-state gain
    with torch.no_grad():
        A = self.A
        std_B = 0.1 * (1.0 - A)
        self.B.copy_(torch.randn(D) * std_B)

    nn.init.normal_(self.C, mean=0.0, std=0.05)
    nn.init.normal_(self.D_skip, mean=1.0, std=0.01)

def initial_state(
    self,
    batch_size: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> SSMState:
    """Create zero-initialized state for a batch."""
    device = device or self.W_conv.device
    dtype = dtype or self.W_conv.dtype
    conv_state = torch.zeros(
        batch_size, self.d_model, max(self.K - 1, 0),
        device=device, dtype=dtype
    )
    ssm_state = torch.zeros(
        batch_size, self.d_model,
        device=device, dtype=dtype
    )
    return SSMState(conv_state, ssm_state)

def forward(
    self,
    x_t: torch.Tensor,
    state: SSMState,
) -> Tuple[torch.Tensor, SSMState]:
    """Single recurrent step."""
    B, D = x_t.shape
    assert D == self.d_model
    K = self.K

    conv_state = state.conv_state
    ssm_state = state.ssm_state

    # Build conv window
    if K > 1:
        w_full = torch.cat(
            [conv_state, x_t.unsqueeze(-1)], dim=-1
        )  # (B, D, K)
    else:
        w_full = x_t.unsqueeze(-1)

    # Depthwise conv
    W = self.W_conv.unsqueeze(0)
    u = (w_full * W).sum(dim=-1) + self.b_conv  # (B, D)

    # SSM recurrence
    A = self.A
    z = A * ssm_state + self.B * u
    y = self.C * z + self.D_skip * u

    # Update conv state
    if K > 1:
        new_conv = torch.cat(
            [conv_state[:, :, 1:], x_t.unsqueeze(-1)], dim=-1
        )
    else:
        new_conv = conv_state

    return y, SSMState(new_conv, z)

def get_diagnostics(self) -> Dict[str, float]:
    """Return diagnostic statistics."""
    A = self.A.detach()  # poles in (0,1)
    # Discrete-time timescale: τ ≈ 1/(1-A)
    tau = 1.0 / (1.0 - A + 1e-8)
    return {
        "A_min": A.min().item(),
        "A_max": A.max().item(),
        "A_mean": A.mean().item(),
        "tau_min": tau.min().item(),
        "tau_max": tau.max().item(),
    }
```

# ============================================================

# MAMBA CELL (SELECTIVE, INPUT-DEPENDENT, 0T)

# ============================================================

@dataclass
class MambaState:
"""State container for MambaCell."""
conv_state: torch.Tensor  # (B, d_inner, d_conv)
ssm_state: torch.Tensor   # (B, d_inner, d_state)

```
def detach(self) -> "MambaState":
    return MambaState(self.conv_state.detach(), self.ssm_state.detach())

def clone(self) -> "MambaState":
    return MambaState(self.conv_state.clone(), self.ssm_state.clone())

def mask_done(self, done: torch.Tensor) -> "MambaState":
    """Zero out states where done=True."""
    if done is None or not done.any():
        return self
    mask = (~done).float().unsqueeze(-1).unsqueeze(-1)
    conv = self.conv_state * mask
    ssm = self.ssm_state * mask
    return MambaState(conv, ssm)
```

class MambaCell(nn.Module):
"""
Recurrent Mamba SSM with selective mechanism.

```
Key features:
- Input-dependent Δ/B/C for content-based gating
- Continuous-time A formulation: A = -exp(A_log)
- Log-spaced τ grid for multi-timescale memory
- Horizon-aware dt scaling for stability

State per env: O(d_inner * d_state), independent of sequence length.
"""

def __init__(
    self,
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dt_rank: str = "auto",
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    horizon: int = 40,
    device=None,
    dtype=None,
):
    super().__init__()
    factory_kwargs = {"device": device, "dtype": dtype}
    
    self.d_model = d_model
    self.d_state = d_state
    self.d_conv = d_conv
    self.expand = expand
    self.d_inner = int(expand * d_model)
    self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
    self.dt_min = dt_min
    self.dt_max = dt_max
    self.horizon = float(horizon)

    # Input projection: x -> (x_branch, z_gate)
    self.in_proj = nn.Linear(
        d_model, self.d_inner * 2, bias=False, **factory_kwargs
    )

    # Depthwise conv for local context
    self.conv1d = nn.Conv1d(
        in_channels=self.d_inner,
        out_channels=self.d_inner,
        bias=True,
        kernel_size=d_conv,
        groups=self.d_inner,
        padding=0,
        **factory_kwargs,
    )

    self.act = nn.SiLU()

    # Selective mechanism: x -> (dt, B, C)
    self.x_proj = nn.Linear(
        self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
    )
    self.dt_proj = nn.Linear(
        self.dt_rank, self.d_inner, bias=True, **factory_kwargs
    )

    # Initialize dt_proj
    dt_init_std = self.dt_rank ** -0.5
    nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
    with torch.no_grad():
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias.copy_(inv_dt)

    # A: continuous-time decay rates, log-spaced τ grid
    # A = -exp(A_log) gives negative rates
    taus = torch.logspace(
        math.log10(10.0),
        math.log10(2000.0),
        steps=d_state,
        device=device,
        dtype=torch.float32,
    )
    A_ct = -1.0 / taus  # negative continuous-time rates
    A_grid = A_ct.unsqueeze(0).repeat(self.d_inner, 1)  # (d_inner, d_state)
    self.A_log = nn.Parameter(torch.log(-A_grid))
    self.A_log._no_weight_decay = True

    # Skip connection
    self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
    self.D._no_weight_decay = True

    # Output projection
    self.out_proj = nn.Linear(
        self.d_inner, d_model, bias=False, **factory_kwargs
    )

    self.use_fast_path = False

def initial_state(
    self, batch_size: int, device=None, dtype=None
) -> MambaState:
    """Create zero-initialized state for a batch."""
    device = device or self.in_proj.weight.device
    conv_dtype = self.conv1d.weight.dtype
    ssm_dtype = torch.float32
    conv_state = torch.zeros(
        batch_size, self.d_inner, self.d_conv, device=device, dtype=conv_dtype
    )
    ssm_state = torch.zeros(
        batch_size, self.d_inner, self.d_state, device=device, dtype=ssm_dtype
    )
    return MambaState(conv_state, ssm_state)

def forward_step(
    self, x_t: torch.Tensor, state: MambaState
) -> Tuple[torch.Tensor, MambaState]:
    """Single recurrent step with selective mechanism."""
    B, D = x_t.shape
    assert D == self.d_model

    # Input projection
    xz = self.in_proj(x_t)
    x, z = xz.chunk(2, dim=-1)

    # Update conv state
    conv_state = state.conv_state
    conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
    conv_state[:, :, -1] = x

    # Depthwise conv via manual application
    w = self.conv1d.weight.squeeze(1)  # (d_inner, d_conv)
    conv_x = torch.sum(conv_state * w.unsqueeze(0), dim=-1) + self.conv1d.bias
    x = self.act(conv_x.to(dtype=x_t.dtype))

    # Selective mechanism: x -> (dt, B, C)
    x_dbl = self.x_proj(x)
    dt_raw, B_vec, C_vec = torch.split(
        x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
    )
    dt = F.linear(dt_raw, self.dt_proj.weight) + self.dt_proj.bias

    # A: negative rates
    A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

    # Horizon-aware dt clamping
    dt = F.softplus(dt)
    scaled_dt_max = min(self.dt_max * (self.horizon / 40.0), 0.5)
    dt = dt.clamp(self.dt_min, scaled_dt_max)

    # Discretization
    dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
    dB = torch.einsum("bd,bn->bdn", dt, B_vec)

    # SSM recurrence
    ssm_state = state.ssm_state
    ssm_state = ssm_state * dA + x.unsqueeze(-1) * dB

    # Output
    y = torch.einsum("bdn,bn->bd", ssm_state, C_vec) + self.D * x
    y = y * self.act(z)
    y = self.out_proj(y)

    return y, MambaState(conv_state, ssm_state)

def get_diagnostics(self) -> Dict[str, float]:
    """Return diagnostic statistics in continuous-time language."""
    A = -torch.exp(self.A_log)  # negative rates
    lam = -A  # positive decay rates
    tau = 1.0 / (lam + 1e-8)  # timescales
    return {
        "A_min": A.min().item(),
        "A_max": A.max().item(),
        "A_mean": A.mean().item(),
        "tau_min": tau.min().item(),
        "tau_max": tau.max().item(),
    }
```

# ============================================================

# UNIFIED POLICY (CONTROLLER SELECTION)

# ============================================================

class OTMPolicy(nn.Module):
"""
0T Memory policy with selectable controller:
- "streaming": StreamingSSMCell (fixed A/B/C/D)
- "mamba": MambaCell (selective, input-dependent)
"""

```
def __init__(self, obs_dim: int, act_dim: int, config: Dict[str, Any]):
    super().__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.controller_type = config["controller_type"]

    # Select d_model based on controller type
    if self.controller_type == "streaming":
        d_model = config["stream_d_model"]
    elif self.controller_type == "mamba":
        d_model = config["mamba_d_model"]
    else:
        raise ValueError(f"Unknown controller_type: {self.controller_type}")

    self.d_model = d_model

    # Observation embedding
    self.obs_embed = nn.Sequential(
        nn.Linear(obs_dim, d_model),
        nn.ReLU(),
    )

    # Create controller
    if self.controller_type == "streaming":
        self.core = StreamingSSMCell(
            d_model=d_model,
            K=config["stream_K"],
            min_timescale=config["stream_min_timescale"],
            max_timescale=config["stream_max_timescale"],
        )
    else:  # mamba
        self.core = MambaCell(
            d_model=d_model,
            d_state=config["mamba_d_state"],
            d_conv=config["mamba_d_conv"],
            expand=config["mamba_expand"],
            dt_rank=config["mamba_dt_rank"],
            dt_min=config["mamba_dt_min"],
            dt_max=config["mamba_dt_max"],
            horizon=config["horizon"],
        )

    # Policy and value heads
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

def initial_state(self, batch_size: int, device=None) -> Any:
    """Create initial state for the controller."""
    return self.core.initial_state(batch_size, device=device)

def forward(
    self,
    obs_t: torch.Tensor,
    state: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """Single step forward."""
    x = self.obs_embed(obs_t)
    
    if self.controller_type == "streaming":
        y, new_state = self.core(x, state)
    else:
        y, new_state = self.core.forward_step(x, state)

    logits = self.policy_head(y)
    value = self.value_head(y).squeeze(-1)
    return logits, value, new_state

def get_diagnostics(self) -> Dict[str, float]:
    """Return controller diagnostics."""
    return self.core.get_diagnostics()
```

# ============================================================

# PPO + GAE UTILITIES

# ============================================================

def compute_gae(
rewards: torch.Tensor,
values: torch.Tensor,
dones: torch.Tensor,
gamma: float = 0.99,
lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
"""Compute Generalized Advantage Estimation."""
T, B = rewards.shape
advantages = torch.zeros_like(rewards)
last_adv = torch.zeros(B, device=rewards.device)
last_value = torch.zeros(B, device=rewards.device)

```
for t in reversed(range(T)):
    mask = (~dones[t]).float()
    delta = rewards[t] + gamma * last_value * mask - values[t]
    last_adv = delta + gamma * lam * last_adv * mask
    advantages[t] = last_adv
    last_value = values[t]

returns = advantages + values
return returns, advantages
```

# ============================================================

# TRAINER

# ============================================================

class Trainer:
"""Complete PPO training loop."""

```
def __init__(self, config: dict):
    self.config = config
    self.device = get_device()
    print(f"Using device: {self.device}")

    # Set seeds
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
        config=config,
    ).to(self.device)

    num_params = sum(p.numel() for p in self.policy.parameters())
    print(f"Policy parameters: {num_params:,}")

    # Optimizer
    self.optimizer = torch.optim.Adam(
        self.policy.parameters(), lr=config["lr"]
    )

    # Training history
    self.history = {
        "update": [],
        "success_rate": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "A_min": [],
        "A_max": [],
        "A_mean": [],
        "tau_min": [],
        "tau_max": [],
    }
    self.running_success = 0.0
    self.success_alpha = 0.1

def collect_rollout(self) -> dict:
    """Collect one full episode from all environments."""
    cfg = self.config
    num_envs = cfg["num_envs"]
    horizon = cfg["horizon"]

    obs_list, act_list, logp_list = [], [], []
    rew_list, val_list, done_list = [], [], []

    obs = self.env.reset_all()
    state = self.policy.initial_state(num_envs, device=self.device)

    total_successes = 0.0
    total_episodes = 0

    for _ in range(horizon):
        obs_t = torch.tensor(obs, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            logits, value, state = self.policy(obs_t, state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

        act_np = action.cpu().numpy()
        next_obs, rewards, dones, info = self.env.step(act_np)

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

        total_successes += info["successes"]
        total_episodes += info["num_done"]

        if dones.any():
            state = state.mask_done(done_list[-1])
            obs = self.env.reset_done()
        else:
            obs = next_obs

    rollout = {
        "obs": torch.stack(obs_list),
        "actions": torch.stack(act_list),
        "old_logp": torch.stack(logp_list),
        "rewards": torch.stack(rew_list),
        "values": torch.stack(val_list),
        "dones": torch.stack(done_list),
    }

    success_rate = total_successes / max(total_episodes, 1)
    self.running_success = (
        (1 - self.success_alpha) * self.running_success
        + self.success_alpha * success_rate
    )
    rollout["success_rate"] = success_rate
    rollout["running_success"] = self.running_success

    return rollout

def ppo_update(self, rollout: dict) -> dict:
    """Perform PPO update with full-sequence BPTT."""
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

    # Flatten
    actions_flat = rollout["actions"].view(T * B)
    old_logp_flat = rollout["old_logp"].view(T * B)
    returns_flat = returns.view(T * B)
    advantages_flat = advantages.view(T * B)

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    for _ in range(cfg["ppo_epochs"]):
        # Re-run full sequence with gradients for BPTT
        state = self.policy.initial_state(B, device=self.device)
        new_logits_list, new_values_list = [], []

        for t in range(T):
            obs_t = rollout["obs"][t]
            logits_t, value_t, state = self.policy(obs_t, state)
            new_logits_list.append(logits_t)
            new_values_list.append(value_t)

            if rollout["dones"][t].any():
                state = state.mask_done(rollout["dones"][t])

        new_logits = torch.stack(new_logits_list).view(T * B, -1)
        new_values = torch.stack(new_values_list).view(T * B)

        # PPO losses
        dist = torch.distributions.Categorical(logits=new_logits)
        new_logp = dist.log_prob(actions_flat)

        ratio = torch.exp(new_logp - old_logp_flat)
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(
            ratio, 1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"]
        ) * advantages_flat
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(new_values, returns_flat)
        entropy = dist.entropy().mean()

        loss = (
            policy_loss
            + cfg["value_coef"] * value_loss
            - cfg["entropy_coef"] * entropy
        )

        # NaN/inf guard
        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, skipping PPO step.")
            print(
                f"  policy_loss={policy_loss.item()}, "
                f"value_loss={value_loss.item()}, "
                f"entropy={entropy.item()}"
            )
            continue

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy.parameters(), cfg["max_grad_norm"]
        )
        self.optimizer.step()

        metrics["policy_loss"] += policy_loss.item() / cfg["ppo_epochs"]
        metrics["value_loss"] += value_loss.item() / cfg["ppo_epochs"]
        metrics["entropy"] += entropy.item() / cfg["ppo_epochs"]

    return metrics

def train(self):
    """Main training loop."""
    cfg = self.config

    print("=" * 70)
    print(f"0T MEMORY AGENT ({cfg['controller_type'].upper()}) - {PRESET.upper()}")
    print("=" * 70)
    print(
        f"Horizon: {cfg['horizon']} | Actions: {cfg['num_actions']} | "
        f"Baseline: {100 / cfg['num_actions']:.1f}%"
    )
    if cfg['controller_type'] == 'streaming':
        print(
            f"Streaming SSM: d_model={cfg['stream_d_model']}, K={cfg['stream_K']}, "
            f"τ∈[{cfg['stream_min_timescale']}, {cfg['stream_max_timescale']}]"
        )
    else:
        print(
            f"Mamba: d_model={cfg['mamba_d_model']}, d_state={cfg['mamba_d_state']}, "
            f"expand={cfg['mamba_expand']}"
        )
    print("=" * 70)
    print()

    start_time = time.time()

    for update in range(1, cfg["num_updates"] + 1):
        rollout = self.collect_rollout()
        metrics = self.ppo_update(rollout)
        diag = self.policy.get_diagnostics()

        # Record history
        self.history["update"].append(update)
        self.history["success_rate"].append(rollout["running_success"])
        self.history["policy_loss"].append(metrics["policy_loss"])
        self.history["value_loss"].append(metrics["value_loss"])
        self.history["entropy"].append(metrics["entropy"])
        self.history["A_min"].append(diag.get("A_min", 0.0))
        self.history["A_max"].append(diag.get("A_max", 0.0))
        self.history["A_mean"].append(diag.get("A_mean", 0.0))
        self.history["tau_min"].append(diag.get("tau_min", 0.0))
        self.history["tau_max"].append(diag.get("tau_max", 0.0))

        # Log progress
        if update % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            tau_info = ""
            if "tau_min" in diag and "tau_max" in diag:
                tau_info = f" | τ∈[{diag['tau_min']:.1f},{diag['tau_max']:.1f}]"
            print(
                f"Update {update:4d}/{cfg['num_updates']} | "
                f"Success: {rollout['running_success']*100:5.1f}% | "
                f"π_loss: {metrics['policy_loss']:7.4f} | "
                f"v_loss: {metrics['value_loss']:7.4f} | "
                f"H: {metrics['entropy']:5.3f} | "
                f"A∈[{diag.get('A_min',0):.3f},{diag.get('A_max',0):.3f}]"
                f"{tau_info} | Time: {elapsed:.1f}s"
            )

    # Final summary
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print(
        f"Final success: {self.history['success_rate'][-1]*100:.1f}% | "
        f"Time: {time.time() - start_time:.1f}s"
    )
    print("=" * 70)

    self.plot_progress()

def plot_progress(self):
    """Plot training curves."""
    updates = self.history["update"]
    if not updates:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Success rate
    ax = axes[0, 0]
    ax.plot(
        updates, [s * 100 for s in self.history["success_rate"]], "b-", lw=2
    )
    ax.axhline(
        100 / self.config["num_actions"],
        color="r",
        ls="--",
        label=f"Random ({100/self.config['num_actions']:.0f}%)",
    )
    ax.axhline(100, color="g", ls="--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Update")
    ax.set_ylabel("Success (%)")
    ax.set_title("Success Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Losses
    ax = axes[0, 1]
    ax.plot(updates, self.history["policy_loss"], "b-", label="Policy")
    ax.plot(updates, self.history["value_loss"], "r-", label="Value")
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("PPO Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 0]
    ax.plot(updates, self.history["entropy"], "g-", lw=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.grid(True, alpha=0.3)

    # Timescale (τ) range
    ax = axes[1, 1]
    ax.fill_between(
        updates,
        self.history["tau_min"],
        self.history["tau_max"],
        alpha=0.3,
        color="purple",
        label="τ range",
    )
    ax.set_xlabel("Update")
    ax.set_ylabel("τ (timescale)")
    ax.set_title("SSM Timescale Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"training_{self.config['controller_type']}_{PRESET}.png"
    plt.savefig(fname, dpi=150)
    print(f"\nPlot saved: {fname}")
    plt.show()
```

# ============================================================

# MAIN ENTRY POINT

# ============================================================

def main():
"""Run training with current CONFIG and PRESET."""
print("\n" + "=" * 70)
print("0T MEMORY AGENT — UNIFIED")
print("=" * 70)
print(f"Controller: {CONFIG[‘controller_type']} | Preset: {PRESET}")
print("=" * 70 + "\n")

```
trainer = Trainer(CONFIG)
trainer.train()
return trainer
```

if **name** == "**main**":
trainer = main()