import math

import torch

from param_decomp.metrics.spike_gate_kl import bernoulli_kl


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
