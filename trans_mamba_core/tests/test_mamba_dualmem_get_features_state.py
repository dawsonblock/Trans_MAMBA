"""Tests for MambaDualMemController.get_features state propagation."""

import torch

from trans_mamba_core.controllers import (
    MambaDualMemConfig,
    MambaDualMemController,
)


def _fast_ptr(mem_state) -> int:
    return mem_state.fast_ptr.detach().cpu().clone()


def test_get_features_advances_state_across_calls():
    torch.manual_seed(0)

    cfg = MambaDualMemConfig(
        vocab_size=64,
        d_model=64,
        n_layers=2,
        mem_slots=128,
    )
    model = MambaDualMemController(cfg).eval()

    B = 2
    T1 = 8
    T2 = 8

    x1 = torch.randint(0, cfg.vocab_size, (B, T1))
    x2 = torch.randint(0, cfg.vocab_size, (B, T2))

    with torch.no_grad():
        feats1, s1 = model.get_features(x1, state=None)
        assert feats1.shape == (B, T1, cfg.d_model)
        assert len(s1.mamba_states) == cfg.n_layers

        p1 = _fast_ptr(s1.memory_state)

        feats2, s2 = model.get_features(x2, state=s1)
        assert feats2.shape == (B, T2, cfg.d_model)
        assert len(s2.mamba_states) == cfg.n_layers

        p2 = _fast_ptr(s2.memory_state)

    assert not torch.equal(p2, p1), "Expected memory pointer to advance"
