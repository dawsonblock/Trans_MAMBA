import numpy as np
import torch


class _ForceDoneWrapper:
    def __init__(self, env, done_at_step: int = 2, env_index: int = 0):
        self.env = env
        self.done_at_step = done_at_step
        self.env_index = env_index
        self._step = 0

        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

    def reset(self):
        self._step = 0
        return self.env.reset()

    def step(self, actions):
        obs, rew, done, trunc, info = self.env.step(actions)
        self._step += 1

        if self._step == self.done_at_step:
            done = np.array(done, copy=True)
            done[self.env_index] = True

        return obs, rew, done, trunc, info

    def close(self):
        return self.env.close()


def test_rl_memory_reset_contract_delayed_cue():
    from trans_mamba_core.rl import (
        InfinityAgent,
        InfinityAgentConfig,
        make_env,
    )

    torch.manual_seed(0)

    env = make_env(
        "delayed_cue",
        num_envs=2,
        horizon=8,
        num_actions=2,
    )
    env = _ForceDoneWrapper(env, done_at_step=1, env_index=0)

    agent = InfinityAgent(
        InfinityAgentConfig(
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
            d_model=32,
            n_layers=1,
            mem_slots=16,
        )
    )

    obs, _ = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32)

    state = agent.init_state(batch_size=2, device=obs_t.device)
    init_ptr = state[1].fast_ptr.clone()

    with torch.no_grad():
        logits, _v, state = agent(obs_t, state)
    ptr1 = state[1].fast_ptr.clone()
    assert not torch.equal(ptr1, init_ptr)

    actions = (
        torch.distributions.Categorical(logits=logits)
        .sample()
        .cpu()
        .numpy()
    )
    obs, _rew, done, _trunc, _info = env.step(actions)

    done_idx = np.nonzero(done)[0].tolist()
    state = agent.reset_state(state, batch_indices=done_idx)

    ptr_after_reset = state[1].fast_ptr.clone()
    assert torch.equal(ptr_after_reset[0], init_ptr[0])
    assert not torch.equal(ptr_after_reset[1], init_ptr[1])

    obs_t = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        logits2, _v2, state2 = agent(obs_t, state)

    ptr2 = state2[1].fast_ptr.clone()

    assert not torch.equal(ptr2[0], init_ptr[0])
    assert not torch.equal(ptr2[1], init_ptr[1])
