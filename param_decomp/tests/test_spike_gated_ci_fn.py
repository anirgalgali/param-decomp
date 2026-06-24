import pytest
import torch
from torch import nn

from param_decomp.ci_fns import SpikeGatedCiFn, get_spike_gated_ci_fn


def _make_fn(**kw) -> SpikeGatedCiFn:
    defaults = dict(
        layer_configs={"linear1": (15, 20), "linear2": (8, 20)},
        encoder_hidden_dims=[16],
        n_mechanisms=8,
        hard_concrete_temp=0.5,
        hard_concrete_stretch=0.1,
        slab_sigma0=0.0,
        decoder_nonneg=False,
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


def test_get_spike_gated_ci_fn_accessor():
    fn = _make_fn()
    wrapper = nn.Module()
    wrapper._global_ci_fn = fn  # mimic GlobalCiFnWrapper
    assert get_spike_gated_ci_fn(wrapper) is fn
    with pytest.raises(AssertionError):
        get_spike_gated_ci_fn(nn.Linear(2, 2))
