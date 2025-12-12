import torch


def _reference_fuse(model, h, mem_state):
    B, T, _D = h.shape
    outputs = []
    for t in range(T):
        h_t = h[:, t, :]
        mem_out, mem_state, _aux = model.memory(
            query=h_t,
            write_value=h_t,
            write_mask=None,
            state=mem_state,
        )
        gate_in = torch.cat([h_t, mem_out], dim=-1)
        gate = torch.sigmoid(model.mem_gate(gate_in))
        fused = model.mem_ln(gate * h_t + (1 - gate) * mem_out)
        outputs.append(fused.unsqueeze(1))
    h_out = torch.cat(outputs, dim=1)
    return model.ln_f(h_out), mem_state


def test_dualmem_vectorized_equivalence_cpu():
    from trans_mamba_core.controllers import (
        MambaDualMemConfig,
        MambaDualMemController,
    )

    torch.manual_seed(0)

    cfg = MambaDualMemConfig(
        vocab_size=32,
        d_model=64,
        n_layers=2,
        mem_slots=64,
        mem_n_heads=4,
    )
    model = MambaDualMemController(cfg).eval()

    B, T = 2, 16
    x = torch.randint(0, cfg.vocab_size, (B, T))

    state0 = model.init_state(B, x.device)

    with torch.no_grad():
        h = model.embed(x)

        mamba_states = state0.mamba_states
        new_mamba_states = []
        for i, block in enumerate(model.blocks):
            h_res, new_s = block(h, mamba_states[i] if mamba_states else None)
            h = h + h_res
            new_mamba_states.append(new_s)

        ref_out, ref_mem_state = _reference_fuse(model, h, state0.memory_state)

        vec_out, vec_state = model.get_features(x, state=state0)

    max_abs_diff = (vec_out - ref_out).abs().max().item()
    assert max_abs_diff < 1e-5

    S = cfg.mem_slots
    ptr_before = state0.memory_state.fast_ptr
    ptr_after_ref = ref_mem_state.fast_ptr
    ptr_after_vec = vec_state.memory_state.fast_ptr

    assert torch.equal(ptr_after_vec, ptr_after_ref)

    expected = (ptr_before + T) % S
    assert torch.equal(ptr_after_vec, expected)
