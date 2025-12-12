"""
infinity_bench.py

Minimal PPO bench for ContinuousMemoryInfinityAgent (Mamba + DualTierMiras).

- Uses vectorized Gym/Gymnasium envs (CartPole by default)
- Assumes agent has:
      class ContinuousMemoryInfinityAgent(nn.Module):
          def __init__(self, obs_dim, act_dim, d_model, ...):
              ...
          def reset_memory(self): ...
          def forward(self, obs, write_mask=None):
              # obs: [B, obs_dim]
              # returns logits: [B, act_dim], values: [B]
              return logits, values, h

Usage examples:

    python infinity_bench.py --env_id CartPole-v1 --total_steps 100000
    python infinity_bench.py --env_id CartPole-v1 --num_envs 8 --d_model 256

This is deliberately simple: single-process vec env, PPO with GAE, no wandb.
"""

import argparse
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try gymnasium first, fall back to gym
try:
    import gymnasium as gym
except ImportError:
    import gym

from trans_mamba_core.memory import DualTierMiras, DualTierMirasConfig


# -------------------------------------------------------------------
# 1. Minimal Mamba backbone stub (replace with your actual Mamba)
# -------------------------------------------------------------------

class MambaBackboneStub(nn.Module):
    """
    Stub Mamba backbone for testing.
    Replace this with your actual MambaBackbone from mamba_ssm.
    """
    def __init__(self, d_model: int, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


# -------------------------------------------------------------------
# 2. InfinityMambaWithMiras - Mamba + DualTierMiras wrapper
# -------------------------------------------------------------------

class InfinityMambaWithMiras(nn.Module):
    """
    Mamba backbone wrapped with DualTierMiras memory.

    Per-step memory read/write with residual fusion.
    """
    def __init__(
        self,
        d_model: int,
        mem_slots: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        **mamba_kwargs
    ):
        super().__init__()
        self.d_model = d_model

        # Your existing mamba backbone (single block or stack)
        self.mamba_core = MambaBackboneStub(
            d_model=d_model,
            n_layers=n_layers,
        )

        miras_cfg = DualTierMirasConfig(
            d_model=d_model,
            mem_slots=mem_slots,
            n_heads=n_heads,
            lr_fast=1.0,
            lr_deep=0.1,
            surprise_threshold=0.6,
            use_decay=True,
            decay_rate=0.9995,
            use_sparse_attention=True,
            top_k_retrieval=8,
        )
        self.miras = DualTierMiras(miras_cfg)

        # Fuse memory back into hidden state (residual)
        self.mem_fuse = nn.Linear(d_model * 2, d_model)
        self.mem_ln = nn.LayerNorm(d_model)

    def reset_memory(self):
        self.miras.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        write_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x: [B, T, d_model] or [B, d_model]
        write_mask: [B, T] or [B] boolean, when to write to memory
        returns: [B, T, d_model] or [B, d_model]
        """
        # Handle 2D input (no time dimension)
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]
            squeeze_output = True
            if write_mask is not None and write_mask.dim() == 1:
                write_mask = write_mask.unsqueeze(1)

        B, T, D = x.shape
        device = x.device

        if write_mask is None:
            write_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # Run through Mamba backbone
        h = self.mamba_core(x)  # [B, T, D]

        h_out = []
        for t in range(T):
            h_t = h[:, t, :]  # [B, D]

            # Memory read
            mem_read = self.miras.read(h_t)
            v_t = mem_read["v"]  # [B, D]

            # Fuse
            fused = torch.cat([h_t, v_t], dim=-1)
            fused = self.mem_fuse(fused)
            fused = self.mem_ln(fused + h_t)  # residual

            # Optional write
            if write_mask[:, t].any():
                self.miras.update(
                    key=h_t.detach(),
                    value=fused.detach(),
                    context=h_t.detach(),
                )

            h_out.append(fused.unsqueeze(1))

        h_out = torch.cat(h_out, dim=1)  # [B, T, D]

        if squeeze_output:
            h_out = h_out.squeeze(1)  # [B, D]

        return h_out


# -------------------------------------------------------------------
# 3. ContinuousMemoryInfinityAgent - RL Agent
# -------------------------------------------------------------------

class ObsEncoder(nn.Module):
    """Simple observation encoder."""
    def __init__(self, obs_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousMemoryInfinityAgent(nn.Module):
    """
    RL Agent with Mamba backbone + DualTierMiras memory.

    Outputs policy logits and value estimates.
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        d_model: int = 256,
        mem_slots: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.encoder = ObsEncoder(obs_dim, d_model)

        self.backbone = InfinityMambaWithMiras(
            d_model=d_model,
            mem_slots=mem_slots,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        self.policy_head = nn.Linear(d_model, act_dim)
        self.value_head = nn.Linear(d_model, 1)

    def reset_memory(self):
        if hasattr(self.backbone, "reset_memory"):
            self.backbone.reset_memory()

    def forward(
        self,
        obs: torch.Tensor,
        write_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs: [B, obs_dim] or [B, T, obs_dim]
        returns:
            logits: [B, act_dim] or [B, T, act_dim]
            values: [B] or [B, T]
            h: hidden states
        """
        x = self.encoder(obs)  # [B, d_model] or [B, T, d_model]
        h = self.backbone(x, write_mask=write_mask)

        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)

        return logits, values, h


# -------------------------------------------------------------------
# 4. Simple vectorized environment wrapper
# -------------------------------------------------------------------

class VecEnv:
    """
    Simple N-env vectorizer using gym.make.
    Only supports discrete action spaces.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int = 0):
        self.envs = []
        for i in range(num_envs):
            env = gym.make(env_id)
            try:
                env.reset(seed=seed + i)
            except TypeError:
                env.seed(seed + i)
                env.reset()
            self.envs.append(env)

        self.num_envs = num_envs
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

        assert isinstance(self.single_action_space, gym.spaces.Discrete), \
            "This bench only supports discrete action spaces."

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        obs_list = []
        done_list = []
        for env in self.envs:
            out = env.reset()
            if isinstance(out, tuple):
                obs, _ = out
            else:
                obs = out
            obs_list.append(obs)
            done_list.append(False)
        return np.stack(obs_list, axis=0), np.array(done_list, dtype=bool)

    def step(self, actions: np.ndarray):
        obs_list, rew_list = [], []
        done_list, trunc_list, info_list = [], [], []

        for env, a in zip(self.envs, actions):
            out = env.step(a)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
            else:
                obs, reward, done, info = out
                terminated, truncated = done, False

            done_flag = terminated or truncated

            if done_flag:
                reset_out = env.reset()
                if isinstance(reset_out, tuple):
                    obs, _ = reset_out
                else:
                    obs = reset_out

            obs_list.append(obs)
            rew_list.append(reward)
            done_list.append(terminated)
            trunc_list.append(truncated)
            info_list.append(info)

        return (
            np.stack(obs_list, axis=0),
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            info_list,
        )


# -------------------------------------------------------------------
# 5. PPO config + trainer
# -------------------------------------------------------------------

@dataclass
class PPOConfig:
    total_steps: int = 200_000
    num_envs: int = 8
    rollout_len: int = 128

    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01

    lr: float = 3e-4
    max_grad_norm: float = 0.5

    num_epochs: int = 4
    minibatch_size: int = 1024

    d_model: int = 256
    mem_slots: int = 128
    n_heads: int = 4
    seed: int = 1
    device: str = "cuda"


class InfinityPPOTrainer:
    def __init__(self, env_id: str, cfg: PPOConfig):
        self.cfg = cfg

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.device = torch.device(
            cfg.device
            if torch.cuda.is_available() or cfg.device == "cpu"
            else "cpu"
        )

        self.vec_env = VecEnv(env_id, cfg.num_envs, seed=cfg.seed)

        obs_space = self.vec_env.single_observation_space
        act_space = self.vec_env.single_action_space

        assert isinstance(obs_space, gym.spaces.Box), "Only Box obs supported."
        assert len(obs_space.shape) == 1, "Only 1D obs supported."

        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.n

        self.agent = ContinuousMemoryInfinityAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            d_model=cfg.d_model,
            mem_slots=cfg.mem_slots,
            n_heads=cfg.n_heads,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=cfg.lr, eps=1e-5
        )

    @torch.no_grad()
    def _policy_value(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, values, _ = self.agent(obs)
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        if values.dim() == 2:
            values = values[:, -1]
        return logits, values

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, N = rewards.shape
        adv = torch.zeros_like(rewards)
        last_adv = torch.zeros(N, device=rewards.device)

        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t].float()
            next_value = last_values if t == T - 1 else values[t + 1]
            delta = (
                rewards[t]
                + self.cfg.gamma * next_value * nonterminal
                - values[t]
            )
            last_adv = (
                delta
                + self.cfg.gamma * self.cfg.lam * nonterminal * last_adv
            )
            adv[t] = last_adv

        returns = adv + values
        return returns, adv

    def train(self):
        cfg = self.cfg

        obs_np, _ = self.vec_env.reset()
        obs = torch.from_numpy(obs_np).float().to(self.device)

        global_step = 0
        start_time = time.time()

        print(f"Starting PPO training on {cfg.num_envs} envs...")
        print(
            f"Agent has {sum(p.numel() for p in self.agent.parameters()):,} "
            "params"
        )

        while global_step < cfg.total_steps:
            # Reset memory each rollout batch
            if hasattr(self.agent, "reset_memory"):
                self.agent.reset_memory()

            # Collect rollout
            obs_buf, act_buf, logp_buf = [], [], []
            rew_buf, val_buf, done_buf = [], [], []

            for t in range(cfg.rollout_len):
                with torch.no_grad():
                    logits, values = self._policy_value(obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample()
                    logp = dist.log_prob(actions)

                next_obs_np, rewards_np, dones_np, trunc_np, _ = \
                    self.vec_env.step(actions.cpu().numpy())
                global_step += cfg.num_envs

                obs_buf.append(obs.clone())
                act_buf.append(actions.clone())
                logp_buf.append(logp.clone())
                rew_buf.append(
                    torch.from_numpy(rewards_np).float().to(self.device)
                )
                val_buf.append(values.clone())
                done_buf.append(
                    torch.from_numpy(dones_np | trunc_np).to(self.device)
                )

                obs = torch.from_numpy(next_obs_np).float().to(self.device)

            # Stack buffers
            obs_tensor = torch.stack(obs_buf, dim=0)
            act_tensor = torch.stack(act_buf, dim=0)
            logp_tensor = torch.stack(logp_buf, dim=0)
            rew_tensor = torch.stack(rew_buf, dim=0)
            val_tensor = torch.stack(val_buf, dim=0)
            done_tensor = torch.stack(done_buf, dim=0)

            with torch.no_grad():
                _, last_values = self._policy_value(obs)

            returns, advantages = self._compute_gae(
                rewards=rew_tensor,
                values=val_tensor,
                dones=done_tensor,
                last_values=last_values,
            )

            # Flatten
            T, N = act_tensor.shape
            batch_size = T * N

            obs_flat = obs_tensor.reshape(T * N, self.obs_dim)
            act_flat = act_tensor.reshape(T * N)
            logp_old_flat = logp_tensor.reshape(T * N)
            returns_flat = returns.reshape(T * N)
            adv_flat = advantages.reshape(T * N)

            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

            # PPO epochs
            idxs = np.arange(batch_size)
            for epoch in range(cfg.num_epochs):
                np.random.shuffle(idxs)
                for start in range(0, batch_size, cfg.minibatch_size):
                    end = start + cfg.minibatch_size
                    mb_idx = idxs[start:end]

                    mb_obs = obs_flat[mb_idx].to(self.device)
                    mb_act = act_flat[mb_idx].to(self.device)
                    mb_logp_old = logp_old_flat[mb_idx].to(self.device)
                    mb_returns = returns_flat[mb_idx].to(self.device)
                    mb_adv = adv_flat[mb_idx].to(self.device)

                    logits, values = self._policy_value(mb_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(mb_act)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(logp - mb_logp_old)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
                    ) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(values, mb_returns)

                    loss = (
                        policy_loss
                        + cfg.vf_coef * value_loss
                        - cfg.ent_coef * entropy
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(),
                        cfg.max_grad_norm,
                    )
                    self.optimizer.step()

            # Logging
            elapsed = time.time() - start_time
            approx_fps = int(global_step / max(elapsed, 1e-6))

            with torch.no_grad():
                avg_return = rew_tensor.sum(dim=0).mean().item()

            print(
                f"Steps: {global_step:8d} | "
                f"AvgRet: {avg_return:8.3f} | "
                f"FPS: {approx_fps:6d}"
            )

        print("Training complete!")


# -------------------------------------------------------------------
# 6. CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO bench for Infinity Mamba+Miras agent"
    )
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--mem_slots", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PPOConfig(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        rollout_len=args.rollout_len,
        d_model=args.d_model,
        mem_slots=args.mem_slots,
        n_heads=args.n_heads,
        device=args.device,
        seed=args.seed,
    )
    trainer = InfinityPPOTrainer(args.env_id, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
