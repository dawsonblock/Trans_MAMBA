"""Tests for controller modules."""

import torch


def test_transformer_controller():
    """Test Transformer controller."""
    from trans_mamba_core.controllers import (
        TransformerController,
        TransformerConfig,
    )

    cfg = TransformerConfig(vocab_size=16, d_model=64, n_layers=2, n_heads=4)
    model = TransformerController(cfg)

    x = torch.randint(0, 16, (2, 32))
    logits = model(x)

    assert logits.shape == (2, 32, 16)
    print("✓ TransformerController forward pass")


def test_mamba_controller():
    """Test Mamba controller."""
    from trans_mamba_core.controllers import MambaController, MambaConfig

    cfg = MambaConfig(vocab_size=16, d_model=64, n_layers=2)
    model = MambaController(cfg)

    x = torch.randint(0, 16, (2, 32))
    logits, states = model(x)

    assert logits.shape == (2, 32, 16)
    assert len(states) == 2
    print("✓ MambaController forward pass")


def test_mamba_dualmem_controller():
    """Test MambaDualMem controller."""
    from trans_mamba_core.controllers import (
        MambaDualMemController,
        MambaDualMemConfig,
    )

    cfg = MambaDualMemConfig(
        vocab_size=16, d_model=64, n_layers=2, mem_slots=32
    )
    model = MambaDualMemController(cfg)

    x = torch.randint(0, 16, (2, 32))
    logits, state, aux = model(x)

    assert logits.shape == (2, 32, 16)
    assert "surprise" in aux
    print("✓ MambaDualMemController forward pass")


def test_streaming_ssm_controller():
    """Test StreamingSSM controller."""
    from trans_mamba_core.controllers import (
        StreamingSSMController,
        StreamingSSMConfig,
    )

    cfg = StreamingSSMConfig(input_dim=4, d_model=32, n_layers=2)
    model = StreamingSSMController(cfg)

    x = torch.randn(2, 4)
    features, state = model(x)

    assert features.shape == (2, 32)
    assert len(state) == 2
    print("✓ StreamingSSMController forward pass")


def test_streaming_ssm_sequence():
    """Test StreamingSSM sequence processing."""
    from trans_mamba_core.controllers import (
        StreamingSSMController,
        StreamingSSMConfig,
    )

    cfg = StreamingSSMConfig(input_dim=4, d_model=32, n_layers=2)
    model = StreamingSSMController(cfg)

    x = torch.randn(2, 16, 4)
    features, state = model.forward_sequence(x)

    assert features.shape == (2, 16, 32)
    print("✓ StreamingSSMController sequence forward")


def run_all_tests():
    """Run all controller tests."""
    print("\n" + "=" * 50)
    print("Running Controller Tests")
    print("=" * 50 + "\n")

    test_transformer_controller()
    test_mamba_controller()
    test_mamba_dualmem_controller()
    test_streaming_ssm_controller()
    test_streaming_ssm_sequence()

    print("\n" + "=" * 50)
    print("All controller tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
