"""
PPO Trainer for RL agents.

Implements Proximal Policy Optimization with:
- GAE advantage estimation
- Entropy bonus
- Gradient clipping
- Value function clipping
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 3e-4
    normalize_advantage: bool = True


class RolloutBuffer:
    """Storage for rollout data."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
    ):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def get_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "obs": torch.tensor(
                np.array(self.obs),
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.tensor(
                np.array(self.actions),
                dtype=torch.long,
                device=device,
            ),
            "log_probs": torch.tensor(
                np.array(self.log_probs),
                dtype=torch.float32,
                device=device,
            ),
            "rewards": torch.tensor(
                np.array(self.rewards),
                dtype=torch.float32,
                device=device,
            ),
            "dones": torch.tensor(
                np.array(self.dones),
                dtype=torch.float32,
                device=device,
            ),
            "values": torch.tensor(
                np.array(self.values),
                dtype=torch.float32,
                device=device,
            ),
        }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns."""

    T, B = rewards.shape

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        delta = (
            rewards[t]
            + gamma * next_val * (1 - dones[t])
            - values[t]
        )
        last_gae = (
            delta
            + gamma * gae_lambda * (1 - dones[t]) * last_gae
        )
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    """PPO training loop."""

    def __init__(
        self,
        agent: nn.Module,
        cfg: PPOConfig,
        device: torch.device,
    ):
        self.agent = agent
        self.cfg = cfg
        self.device = device

        self.optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=cfg.learning_rate,
        )

        self.buffer = RolloutBuffer()
        self.global_step = 0

    def collect_rollout(
        self,
        env,
        rollout_length: int,
        agent_state=None,
    ) -> Tuple[float, Any]:
        """Collect rollout data from environment."""

        obs, _ = env.reset()
        total_reward = 0.0

        for _ in range(rollout_length):
            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
                device=self.device,
            )

            with torch.no_grad():
                logits, value, agent_state = self.agent(obs_t, agent_state)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            action_np = action.cpu().numpy()
            next_obs, reward, done, _, _ = env.step(action_np)

            if agent_state is not None and hasattr(self.agent, "reset_state"):
                done_arr = np.asarray(done)
                done_idx = np.nonzero(done_arr)[0]
                if done_idx.size > 0:
                    agent_state = self.agent.reset_state(
                        agent_state,
                        batch_indices=done_idx.tolist(),
                    )

            self.buffer.add(
                obs=obs,
                action=action_np,
                log_prob=log_prob.cpu().numpy(),
                reward=reward,
                done=done.astype(float),
                value=value.squeeze(-1).cpu().numpy(),
            )

            total_reward += reward.sum()
            obs = next_obs

        with torch.no_grad():
            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
                device=self.device,
            )
            _, next_value, _ = self.agent(obs_t, agent_state)
            next_value = next_value.squeeze(-1)

        return total_reward, next_value

    def update(self, next_value: torch.Tensor) -> Dict[str, float]:
        """Perform PPO update."""

        cfg = self.cfg
        data = self.buffer.get_tensors(self.device)

        advantages, returns = compute_gae(
            data["rewards"],
            data["values"],
            data["dones"],
            next_value,
            cfg.gamma,
            cfg.gae_lambda,
        )

        if cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        T, B = data["obs"].shape[:2]
        batch_size = T * B

        obs_flat = data["obs"].reshape(batch_size, -1)
        actions_flat = data["actions"].reshape(batch_size)
        old_log_probs_flat = data["log_probs"].reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)
        returns_flat = returns.reshape(batch_size)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clipfrac = 0.0

        minibatch_size = batch_size // cfg.num_minibatches

        for _ in range(cfg.ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                mb_obs = obs_flat[mb_indices]
                mb_actions = actions_flat[mb_indices]
                mb_old_log_probs = old_log_probs_flat[mb_indices]
                mb_advantages = advantages_flat[mb_indices]
                mb_returns = returns_flat[mb_indices]

                logits, values, _ = self.agent(mb_obs, None)
                values = values.squeeze(-1)

                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - cfg.clip_eps,
                    1 + cfg.clip_eps,
                )

                approx_kl = (mb_old_log_probs - log_probs).mean()
                clipfrac = (ratio.sub(1.0).abs() > cfg.clip_eps).float().mean()

                policy_loss = -torch.min(
                    ratio * mb_advantages,
                    clipped_ratio * mb_advantages,
                ).mean()

                value_loss = F.mse_loss(values, mb_returns)

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_approx_kl += approx_kl.item()
                total_clipfrac += clipfrac.item()

        n_updates = cfg.ppo_epochs * cfg.num_minibatches
        self.buffer.clear()
        self.global_step += 1

        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates,
            "clipfrac": total_clipfrac / n_updates,
        }
