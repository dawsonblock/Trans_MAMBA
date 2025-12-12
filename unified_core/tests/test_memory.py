"""Tests for memory modules."""

import os
import sys

import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)


def test_dualtier_miras_init():
    """Test DualTierMiras initialization."""
    from memory import DualTierMiras, DualTierMirasConfig

    cfg = DualTierMirasConfig(d_model=64, mem_slots=32, n_heads=4)
    mem = DualTierMiras(cfg)
    assert mem is not None
    print("✓ DualTierMiras initialization")


def test_dualtier_miras_init_state():
    """Test memory state initialization."""
    from memory import DualTierMiras, DualTierMirasConfig, MemoryState

    cfg = DualTierMirasConfig(d_model=64, mem_slots=32, n_heads=4)
    mem = DualTierMiras(cfg)

    state = mem.init_state(batch_size=2, device=torch.device("cpu"))

    assert isinstance(state, MemoryState)
    assert state.fast_keys.shape == (2, 4, 32, 16)
    assert state.fast_vals.shape == (
        2,
        cfg.n_heads,
        cfg.mem_slots,
        cfg.d_value // cfg.n_heads,
    )
    print("✓ Memory state initialization")


def test_dualtier_miras_forward():
    """Test forward pass (read + write)."""
    from memory import DualTierMiras, DualTierMirasConfig, MemoryState

    cfg = DualTierMirasConfig(d_model=64, mem_slots=32, n_heads=4)
    mem = DualTierMiras(cfg)

    query = torch.randn(2, 64)
    write_value = torch.randn(2, 64)

    output, new_state, aux = mem(query, write_value=write_value)

    assert output.shape == (2, 64)
    assert isinstance(new_state, MemoryState)
    assert "surprise" in aux
    print("✓ Forward pass (read + write)")


def test_dualtier_miras_read_only():
    """Test read-only forward pass."""
    from memory import DualTierMiras, DualTierMirasConfig

    cfg = DualTierMirasConfig(d_model=64, mem_slots=32, n_heads=4)
    mem = DualTierMiras(cfg)

    query = torch.randn(2, 64)
    state = mem.init_state(2, torch.device("cpu"))

    output, new_state, aux = mem(query, state=state)

    assert output.shape == (2, 64)
    print("✓ Read-only forward pass")


def test_surprise_gating():
    """Test surprise-gated deep tier writes."""
    from memory import DualTierMiras, DualTierMirasConfig

    cfg = DualTierMirasConfig(
        d_model=64, mem_slots=32, n_heads=4,
        surprise_threshold=0.5, surprise_detached=True
    )
    mem = DualTierMiras(cfg)

    query = torch.randn(2, 64) * 10
    write_value = torch.randn(2, 64)

    output, new_state, aux = mem(query, write_value=write_value)

    assert "surprise" in aux
    assert aux["surprise"].shape == (2,)
    print("✓ Surprise gating")


def run_all_tests():
    """Run all memory tests."""
    print("\n" + "=" * 50)
    print("Running Memory Tests")
    print("=" * 50 + "\n")

    test_dualtier_miras_init()
    test_dualtier_miras_init_state()
    test_dualtier_miras_forward()
    test_dualtier_miras_read_only()
    test_surprise_gating()

    print("\n" + "=" * 50)
    print("All memory tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
