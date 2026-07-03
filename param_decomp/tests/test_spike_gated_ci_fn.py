import math

import pytest
import torch
from torch import nn

from param_decomp.ci_fns import SpikeGatedCiFn, get_spike_gated_ci_fn


def _make_fn(**kw) -> SpikeGatedCiFn:
    defaults = dict(
        layer_configs={"linear1": (15, 20), "linear2": (8, 20)},
        encoder_hidden_dims=[16],
        n_mechanisms=8,
        gate_type="hard_concrete",
        hard_concrete_temp=0.5,
        hard_concrete_temp_final=None,
        temp_anneal_start_frac=0.0,
        temp_anneal_end_frac=1.0,
        hard_concrete_stretch=0.1,
        slab_sigma0=0.0,
        decoder_nonneg=False,
        decoder_init_std=0.1,
        encoder_head_init_scale=1.0,
    )
    defaults.update(kw)
    return SpikeGatedCiFn(**defaults)


def _acts(batch: int = 4) -> dict[str, torch.Tensor]:
    return {"linear1": torch.randn(batch, 15), "linear2": torch.randn(batch, 8)}


def test_forward_shapes_split_per_layer():
    out = _make_fn()(_acts())
    assert set(out) == {"linear1", "linear2"}
    assert out["linear1"].shape == (4, 20)
    assert out["linear2"].shape == (4, 20)


def test_pi_cached_in_unit_interval():
    fn = _make_fn()
    fn(_acts())
    assert fn._pi is not None
    assert fn._pi.shape == (4, 8)
    assert bool((fn._pi > 0).all()) and bool((fn._pi < 1).all())


def test_decoder_B_shape():
    fn = _make_fn()
    assert fn.B.shape == (40, 8)  # M = 20 + 20, K = 8


def test_reparam_grad_reaches_encoder_and_decoder():
    fn = _make_fn()
    fn.train()
    out = fn(_acts())
    loss = sum(v.pow(2).mean() for v in out.values())
    loss.backward()
    assert fn.B.grad is not None and fn.B.grad.abs().sum() > 0
    # gate logits flow through the hard-concrete sampler into the recon path
    assert fn.encoder[0].W.grad is not None and fn.encoder[0].W.grad.abs().sum() > 0


def test_eval_deterministic_train_stochastic():
    fn = _make_fn()
    acts = _acts()
    fn.eval()
    assert torch.allclose(fn(acts)["linear1"], fn(acts)["linear1"])
    fn.train()
    torch.manual_seed(1)
    a = fn(acts)["linear1"]
    torch.manual_seed(2)
    b = fn(acts)["linear1"]
    assert not torch.allclose(a, b)


def test_slab_jitter_and_nonneg_decoder_run():
    fn = _make_fn(slab_sigma0=0.3, decoder_nonneg=True)
    fn.train()
    out = fn(_acts())
    assert torch.isfinite(out["linear1"]).all()


def test_empty_encoder_hidden_dims_is_linear():
    fn = _make_fn(encoder_hidden_dims=[])
    out = fn(_acts())
    assert out["linear1"].shape == (4, 20)


def test_deterministic_gate_train_equals_eval():
    fn = _make_fn(gate_type="deterministic")
    acts = _acts()
    fn.train()
    train_out = fn(acts)["linear1"]
    fn.eval()
    eval_out = fn(acts)["linear1"]
    assert torch.allclose(train_out, eval_out)


def test_force_deterministic_gate_in_train_matches_eval():
    """Hard-concrete gate forced deterministic during training reproduces the eval (median) gate."""
    fn = _make_fn(gate_type="hard_concrete")
    acts = _acts()
    fn.eval()
    eval_out = fn(acts)["linear1"]
    fn.train()
    with fn.force_deterministic_gate():
        forced_out = fn(acts)["linear1"]
    assert torch.allclose(forced_out, eval_out)


def test_force_deterministic_gate_is_noise_free_and_restores():
    """Forced-deterministic train calls are repeat-stable; the flag is restored on exit."""
    fn = _make_fn(gate_type="hard_concrete")
    acts = _acts()
    fn.train()
    with fn.force_deterministic_gate():
        torch.manual_seed(1)
        a = fn(acts)["linear1"]
        torch.manual_seed(2)
        b = fn(acts)["linear1"]
    assert torch.allclose(a, b)
    assert fn._force_deterministic_gate is False
    # stochastic gate resumes after the block
    torch.manual_seed(1)
    c = fn(acts)["linear1"]
    torch.manual_seed(2)
    d = fn(acts)["linear1"]
    assert not torch.allclose(c, d)


def test_deterministic_gate_uses_pi():
    fn = _make_fn(gate_type="deterministic")
    out = fn(_acts())
    expected = fn._pi @ fn.B.t()  # gate == π, signed decoder
    got = torch.cat([out["linear1"], out["linear2"]], dim=-1)
    assert torch.allclose(got, expected, atol=1e-6)


def test_deterministic_nonneg_pre_sigmoid_is_nonnegative():
    fn = _make_fn(gate_type="deterministic", decoder_nonneg=True)
    out = fn(_acts())
    assert bool((out["linear1"] >= 0).all()) and bool((out["linear2"] >= 0).all())


def test_nonneg_init_is_nonnegative():
    fn = _make_fn(decoder_nonneg=True)
    assert bool((fn.B >= 0).all())


def test_project_nonneg_clamps_when_nonneg():
    fn = _make_fn(decoder_nonneg=True)
    with torch.no_grad():
        fn.B[0, 0] = -3.0
        fn.B[1, 1] = 0.5
    fn.project_nonneg()
    b = fn.B.detach()
    assert float(b[0, 0]) == 0.0  # negative -> exactly 0 (a true off-state)
    assert float(b[1, 1]) == 0.5  # positive untouched
    assert bool((b >= 0).all())


def test_project_nonneg_is_noop_when_signed():
    fn = _make_fn(decoder_nonneg=False)
    with torch.no_grad():
        fn.B[0, 0] = -3.0
    fn.project_nonneg()
    assert float(fn.B.detach()[0, 0]) == -3.0  # signed decoder is left alone


def test_forward_uses_raw_B_no_softplus():
    # With the projection scheme the forward must read B directly (no softplus floor at ~0.69).
    fn = _make_fn(gate_type="deterministic", decoder_nonneg=True)
    out = fn(_acts())
    expected = fn._pi @ fn.B.t()
    got = torch.cat([out["linear1"], out["linear2"]], dim=-1)
    assert torch.allclose(got, expected, atol=1e-6)


def test_deterministic_grad_reaches_encoder_and_decoder():
    fn = _make_fn(gate_type="deterministic")
    fn.train()
    out = fn(_acts())
    loss = sum(v.pow(2).mean() for v in out.values())
    loss.backward()
    assert fn.B.grad is not None and fn.B.grad.abs().sum() > 0
    assert fn.encoder[0].W.grad is not None and fn.encoder[0].W.grad.abs().sum() > 0


def test_anneal_temperature_noop_when_final_is_none():
    fn = _make_fn(hard_concrete_temp=0.5, hard_concrete_temp_final=None)
    fn.anneal_temperature(0.0)
    assert fn.temp == 0.5
    fn.anneal_temperature(1.0)
    assert fn.temp == 0.5


def test_anneal_temperature_linear_interpolation():
    fn = _make_fn(
        hard_concrete_temp=1.0,
        hard_concrete_temp_final=0.2,
        temp_anneal_start_frac=0.0,
        temp_anneal_end_frac=1.0,
    )
    fn.anneal_temperature(0.0)
    assert fn.temp == pytest.approx(1.0)
    fn.anneal_temperature(0.5)
    assert fn.temp == pytest.approx(0.6)  # halfway between 1.0 and 0.2
    fn.anneal_temperature(1.0)
    assert fn.temp == pytest.approx(0.2)


def test_anneal_temperature_respects_window():
    fn = _make_fn(
        hard_concrete_temp=1.0,
        hard_concrete_temp_final=0.2,
        temp_anneal_start_frac=0.25,
        temp_anneal_end_frac=0.75,
    )
    fn.anneal_temperature(0.1)  # before window: still start
    assert fn.temp == pytest.approx(1.0)
    fn.anneal_temperature(0.5)  # midpoint of [0.25, 0.75]
    assert fn.temp == pytest.approx(0.6)
    fn.anneal_temperature(0.9)  # after window: clamped to final
    assert fn.temp == pytest.approx(0.2)


def test_get_spike_gated_ci_fn_accessor():
    fn = _make_fn()
    wrapper = nn.Module()
    wrapper._global_ci_fn = fn  # mimic GlobalCiFnWrapper
    assert get_spike_gated_ci_fn(wrapper) is fn
    with pytest.raises(AssertionError):
        get_spike_gated_ci_fn(nn.Linear(2, 2))


def test_gate_open_prob_matches_louizos_formula_and_exceeds_pi():
    fn = _make_fn(hard_concrete_temp=0.5, hard_concrete_stretch=0.1)
    fn(_acts())  # populate _logits / _pi
    gamma, zeta = -fn.stretch, 1.0 + fn.stretch
    expected = torch.sigmoid(fn._logits - fn.temp * math.log(-gamma / zeta))
    open_prob = fn.gate_open_prob()
    assert torch.allclose(open_prob, expected)
    # the shift -τ·log(stretch/(1+stretch)) > 0, so the gate is open MORE often than π
    assert bool((open_prob > fn._pi).all())


def test_gate_open_prob_approaches_pi_as_temp_to_zero():
    fn = _make_fn(hard_concrete_temp=1e-5, hard_concrete_stretch=0.1)
    fn(_acts())
    assert torch.allclose(fn.gate_open_prob(), fn._pi, atol=1e-3)


def test_gate_open_prob_is_pi_for_deterministic_gate():
    fn = _make_fn(gate_type="deterministic")
    fn(_acts())
    assert torch.allclose(fn.gate_open_prob(), fn._pi)


def test_decoder_init_std_controls_B_scale():
    small = _make_fn(decoder_init_std=0.01)
    large = _make_fn(decoder_init_std=0.5)
    assert small.B.std().item() == pytest.approx(0.01, rel=0.25)
    assert large.B.std().item() == pytest.approx(0.5, rel=0.25)


def test_encoder_head_init_scale_scales_final_head():
    torch.manual_seed(0)
    base = _make_fn(encoder_head_init_scale=1.0)
    torch.manual_seed(0)
    scaled = _make_fn(encoder_head_init_scale=3.0)
    assert torch.allclose(scaled.encoder[-1].W, 3.0 * base.encoder[-1].W)
    # only the final head is touched; hidden weights are identical under the same seed
    assert torch.allclose(scaled.encoder[0].W, base.encoder[0].W)
