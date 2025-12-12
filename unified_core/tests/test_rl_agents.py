"""Tests for RL agents."""

import os
import sys

import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)


def test_infinity_agent_init():
    """Test Infinity agent initialization."""
    from rl import InfinityAgent, InfinityAgentConfig

    cfg = InfinityAgentConfig(obs_dim=4, act_dim=2, d_model=32, n_layers=2)
    agent = InfinityAgent(cfg)
    assert agent is not None
    print("✓ InfinityAgent initialization")


def test_infinity_agent_forward():
    """Test Infinity agent forward pass."""
    from rl import InfinityAgent, InfinityAgentConfig

    cfg = InfinityAgentConfig(obs_dim=4, act_dim=2, d_model=32, n_layers=2)
    agent = InfinityAgent(cfg)

    obs = torch.randn(2, 4)
    logits, values, state = agent(obs)

    assert logits.shape == (2, 2)
    assert values.shape == (2, 1)
    assert state is not None
    print("✓ InfinityAgent forward pass")


def test_infinity_agent_sequence():
    """Test Infinity agent sequence forward."""
    from rl import InfinityAgent, InfinityAgentConfig

    cfg = InfinityAgentConfig(obs_dim=4, act_dim=2, d_model=32, n_layers=2)
    agent = InfinityAgent(cfg)

    obs = torch.randn(2, 16, 4)
    logits, values, state = agent(obs)

    assert logits.shape == (2, 16, 2)
    assert values.shape == (2, 16, 1)
    print("✓ InfinityAgent sequence forward")


def test_ot_agent_init():
    """Test OT agent initialization."""
    from rl import OTAgentConfig, OTMemoryAgent

    cfg = OTAgentConfig(obs_dim=6, act_dim=4, d_model=32, n_layers=2)
    agent = OTMemoryAgent(cfg)
    assert agent is not None
    print("✓ OTMemoryAgent initialization")


def test_ot_agent_forward():
    """Test OT agent forward pass."""
    from rl import OTAgentConfig, OTMemoryAgent

    cfg = OTAgentConfig(obs_dim=6, act_dim=4, d_model=32, n_layers=2)
    agent = OTMemoryAgent(cfg)

    obs = torch.randn(2, 6)
    logits, values, state = agent(obs)

    assert logits.shape == (2, 4)
    assert values.shape == (2, 1)
    assert state is not None
    print("✓ OTMemoryAgent forward pass")


def test_ot_agent_get_action():
    """Test OT agent action selection."""
    from rl import OTAgentConfig, OTMemoryAgent

    cfg = OTAgentConfig(obs_dim=6, act_dim=4, d_model=32, n_layers=2)
    agent = OTMemoryAgent(cfg)

    obs = torch.randn(2, 6)
    action, log_prob, value, state = agent.get_action(obs)

    assert action.shape == (2,)
    assert log_prob.shape == (2,)
    assert value.shape == (2,)
    print("✓ OTMemoryAgent get_action")


def run_all_tests():
    """Run all RL agent tests."""
    print("\n" + "=" * 50)
    print("Running RL Agent Tests")
    print("=" * 50 + "\n")

    test_infinity_agent_init()
    test_infinity_agent_forward()
    test_infinity_agent_sequence()
    test_ot_agent_init()
    test_ot_agent_forward()
    test_ot_agent_get_action()

    print("\n" + "=" * 50)
    print("All RL agent tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
