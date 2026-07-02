import math

import torch

from param_decomp.ci_fns import SpikeGatedCiConfig, get_spike_gated_ci_fn
from param_decomp.component_model import CIOutputs, ComponentModel
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.spike_gate_kl import (
    SpikeGateKLLoss,
    SpikeGateKLLossConfig,
    bernoulli_kl,
)
from param_decomp.tests.metrics.fixtures import TwoLayerLinearModel
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_passthrough


def test_kl_zero_at_prior():
    rho = 0.05
    pi = torch.full((10,), rho)
    assert torch.allclose(bernoulli_kl(pi, rho), torch.zeros(10), atol=1e-6)


def test_kl_nonnegative_and_monotone_above_prior():
    rho = 0.05
    pis = torch.tensor([0.05, 0.1, 0.3, 0.6, 0.95])
    kl = bernoulli_kl(pis, rho)
    assert (kl >= -1e-6).all()
    assert torch.all(kl[1:] - kl[:-1] >= -1e-6)  # increasing as pi rises away from rho


def test_kl_matches_manual():
    rho, pi = 0.05, 0.9
    expected = pi * (math.log(pi) - math.log(rho)) + (1 - pi) * (
        math.log(1 - pi) - math.log(1 - rho)
    )
    got = bernoulli_kl(torch.tensor([pi]), rho).item()
    assert abs(got - expected) < 1e-6


def test_kl_grad_to_pi():
    pi = torch.tensor([0.5], requires_grad=True)
    bernoulli_kl(pi, 0.05).backward()
    assert pi.grad is not None and pi.grad.abs().item() > 0


def test_kl_clamps_extremes_finite():
    # pi at 0 / 1 must not produce nan/inf thanks to the eps clamp
    kl = bernoulli_kl(torch.tensor([0.0, 1.0]), 0.05)
    assert torch.isfinite(kl).all()


def _make_spike_cm() -> ComponentModel:
    torch.manual_seed(0)
    target = TwoLayerLinearModel(d_in=4, d_hidden=3, d_out=4)
    target.requires_grad_(False)
    return ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        decomposition_targets=[
            DecompositionTarget(module_path="fc1", C=2),
            DecompositionTarget(module_path="fc2", C=2),
        ],
        ci_config=SpikeGatedCiConfig(
            mode="spike_gated", encoder_hidden_dims=[4], n_mechanisms=4,
            hard_concrete_temp=0.5, hard_concrete_stretch=0.1,
        ),
        sigmoid_type="leaky_hard",
    )


def _ctx(cm: ComponentModel, ci: CIOutputs, batch: torch.Tensor, cache: dict) -> MetricContext:
    return MetricContext(
        model=cm, batch=batch, target_out=torch.zeros(batch.shape[0], 4), pre_weight_acts=cache,
        ci=ci, ci_adversarial=ci, weight_deltas={}, step=0, total_steps=1,
        use_delta_component=False, sampling="continuous", n_mask_samples=1,
        reconstruction_loss=recon_loss_mse, is_eval=False,
    )


def test_kl_charge_on_open_prob_differs_from_pi_and_matches_formula():
    """charge_on='open_prob' charges bernoulli_kl on P(z>0) (≠ π at τ>0); default 'pi' on π."""
    cm = _make_spike_cm()
    batch = torch.randn(8, 4)
    cache = cm(batch, cache_type="input").cache
    ci = cm.calc_causal_importances(cache, sampling="continuous")  # populates _pi / _logits
    ctx = _ctx(cm, ci, batch, cache)
    spike = get_spike_gated_ci_fn(cm.ci_fn)

    def run(charge_on: str) -> torch.Tensor:
        m = SpikeGateKLLoss(SpikeGateKLLossConfig(coeff=1.0, rho=0.1, charge_on=charge_on))
        m.bind(model=cm, device="cpu")
        return m.update(ctx)

    loss_pi, loss_open = run("pi"), run("open_prob")

    def expected(p: torch.Tensor) -> torch.Tensor:
        per = bernoulli_kl(p, 0.1)
        return per.reshape(-1, per.shape[-1]).mean(dim=0).sum()

    assert torch.allclose(loss_pi, expected(spike._pi))
    assert torch.allclose(loss_open, expected(spike.gate_open_prob()))
    assert not torch.allclose(loss_pi, loss_open)  # open_prob > π ⇒ different existence cost
