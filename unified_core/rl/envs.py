"""
RL Environments for memory benchmarking.

Provides:
- CartPoleEnv: Gymnasium CartPole wrapper
- DelayedCueEnv: Vectorized delayed-cue environment
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym


@dataclass
class EnvConfig:
    """Configuration for environments."""
    num_envs: int = 8
    horizon: int = 40
    num_actions: int = 4
    noise_scale: float = 0.0


class CartPoleEnv:
    """Vectorized CartPole environment using Gymnasium."""

    def __init__(self, num_envs: int = 8):
        self.num_envs = num_envs
        self.envs = [gym.make("CartPole-v1") for _ in range(num_envs)]
        self.obs_dim = 4
        self.act_dim = 2

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset all environments."""
        obs_list = []
        for env in self.envs:
            result = env.reset()
            if isinstance(result, tuple):
                obs, _ = result
            else:
                obs = result
            obs_list.append(obs)

        obs = np.stack(obs_list, axis=0)
        done = np.zeros(self.num_envs, dtype=bool)
        return obs.astype(np.float32), done

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Step all environments."""
        obs_list, rew_list, done_list, trunc_list = [], [], [], []

        for i, (env, a) in enumerate(zip(self.envs, actions)):
            result = env.step(int(a))

            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, done, info = result
                terminated, truncated = done, False

            done_flag = terminated or truncated

            if done_flag:
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result

            obs_list.append(obs)
            rew_list.append(reward)
            done_list.append(done_flag)
            trunc_list.append(truncated)

        obs = np.stack(obs_list, axis=0).astype(np.float32)
        rew = np.array(rew_list, dtype=np.float32)
        done = np.array(done_list, dtype=bool)
        trunc = np.array(trunc_list, dtype=bool)

        return obs, rew, done, trunc, {}

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


class DelayedCueEnv:
    """
    Vectorized delayed-cue environment for memory testing.

    Timeline per episode:
        t=0: Cue presented (one-hot)
        t=1..H-2: Delay period (zeros + optional noise)
        t=H-1: Query (agent must output cue)

    Reward: +1 for correct answer at query time, 0 otherwise.
    """

    def __init__(
        self,
        num_envs: int = 8,
        horizon: int = 40,
        num_actions: int = 4,
        noise_scale: float = 0.0,
    ):
        self.num_envs = num_envs
        self.horizon = horizon
        self.num_actions = num_actions
        self.noise_scale = noise_scale

        self.obs_dim = num_actions + 2
        self.act_dim = num_actions

        self.cues = np.zeros(num_envs, dtype=np.int64)
        self.timesteps = np.zeros(num_envs, dtype=np.int64)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset all environments."""
        self.cues = np.random.randint(0, self.num_actions, self.num_envs)
        self.timesteps = np.zeros(self.num_envs, dtype=np.int64)

        obs = self._make_obs()
        done = np.zeros(self.num_envs, dtype=bool)
        return obs, done

    def _make_obs(self) -> np.ndarray:
        """Construct observation for current timestep."""
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

        for i in range(self.num_envs):
            t = self.timesteps[i]

            if t == 0:
                obs[i, self.cues[i]] = 1.0
                obs[i, -2] = 1.0
            elif t == self.horizon - 1:
                obs[i, -1] = 1.0
            else:
                if self.noise_scale > 0:
                    obs[i, :self.num_actions] = np.random.randn(
                        self.num_actions
                    ) * self.noise_scale

        return obs

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Step all environments."""
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)

        for i in range(self.num_envs):
            if self.timesteps[i] == self.horizon - 1:
                if actions[i] == self.cues[i]:
                    rewards[i] = 1.0
                dones[i] = True

        self.timesteps += 1

        for i in range(self.num_envs):
            if dones[i]:
                self.cues[i] = np.random.randint(0, self.num_actions)
                self.timesteps[i] = 0

        obs = self._make_obs()
        truncated = np.zeros(self.num_envs, dtype=bool)

        return obs, rewards, dones, truncated, {}

    def close(self):
        """No resources to close."""
        pass


def make_env(
    env_name: str,
    num_envs: int = 8,
    **kwargs,
) -> Any:
    """Factory function for environments."""
    if env_name == "cartpole":
        return CartPoleEnv(num_envs=num_envs)
    elif env_name == "delayed_cue":
        return DelayedCueEnv(
            num_envs=num_envs,
            horizon=kwargs.get("horizon", 40),
            num_actions=kwargs.get("num_actions", 4),
            noise_scale=kwargs.get("noise_scale", 0.0),
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")
