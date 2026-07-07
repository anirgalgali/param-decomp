"""Bernoulli existence-KL for the Stage-1 spike-gated CI bottleneck.

`KL(Bern(π_k) ‖ Bern(ρ))` summed over mechanisms, averaged over the batch. `π_k` is read off
the cached gate probabilities of the spike-gated CI fn (see `ci_fns.SpikeGatedCiFn`), which the
forward pass stores on `self._pi` exactly once per training step. This is the structured CI code
and, given a non-degenerate decoder, the per-input sparsifier; see
`docs/slpd_stage1_spike_design.md` in the slpd project.

The §6 KL/sampler mismatch — the KL charged on `π = sigmoid(logits)` while reconstruction sees the
hard-concrete gate whose true on-probability differs — is togglable via `charge_on`: `"pi"` (the
v1 default) or `"open_prob"` (charge on the Louizos open-gate probability `P(z>0)`, matching the
sampled gate). `effective_K` is always reported on `π` so it stays comparable across both modes.
"""

import math
from typing import Literal, override

import torch
from jaxtyping import Float
from pydantic import NonNegativeFloat, NonNegativeInt
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import Probability
from param_decomp.ci_fns import get_spike_gated_ci_fn
from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class SpikeGateKLLossConfig(LossMetricConfig):
    """Config for the spike-gate existence-KL.

    `rho` is the prior gate probability (small ⇒ sparse mechanisms). `free_bits` clamps each
    mechanism's per-batch KL from below (anti-collapse). `kl_warmup_end_frac` linearly ramps the
    KL weight from 0 to 1 over the first fraction of training (0 ⇒ no warmup). `charge_on` selects
    the probability the existence cost is charged on: `"pi"` (= sigmoid(logits), v1) or
    `"open_prob"` (the hard-concrete open-gate probability `P(z>0)`, matching the sampled gate).
    """

    type: Literal["SpikeGateKLLoss"] = "SpikeGateKLLoss"
    rho: Probability
    free_bits: NonNegativeFloat = 0.0
    kl_warmup_end_frac: Probability = 0.0
    kl_warmup_end_steps: NonNegativeInt = 0
    charge_on: Literal["pi", "open_prob"] = "pi"


def bernoulli_kl(
    pi: Float[Tensor, "... K"], rho: float, eps: float = 1e-6
) -> Float[Tensor, "... K"]:
    """`KL(Bern(π) ‖ Bern(ρ))`, per element.

    Computed in fp32 (robust under bf16 autocast) and clamped to `[eps, 1-eps]` — `eps=1e-6`
    is representable in fp32 (unlike `1-1e-8`, which rounds to `1.0`).
    """
    pi = pi.float().clamp(eps, 1.0 - eps)
    return pi * (torch.log(pi) - math.log(rho)) + (1.0 - pi) * (torch.log1p(-pi) - math.log1p(-rho))


class SpikeGateKLLoss(Metric[SpikeGateKLLossConfig]):
    """Per-mechanism Bernoulli existence-KL summed over K, averaged over the batch."""

    log_namespace = "loss"
    short_name = "GateKL"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.sum_mean_pi = torch.zeros((), device=self.device)
        self.n_batches = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        spike_fn = get_spike_gated_ci_fn(ctx.model.ci_fn)
        pi = spike_fn._pi
        assert pi is not None, "CI fn forward must run before SpikeGateKLLoss.update"
        p = spike_fn.gate_open_prob() if self.cfg.charge_on == "open_prob" else pi

        per_mech = bernoulli_kl(p, self.cfg.rho)  # [..., K]
        per_mech_batch_mean = per_mech.reshape(-1, per_mech.shape[-1]).mean(dim=0)  # [K]
        if self.cfg.free_bits > 0.0:
            per_mech_batch_mean = per_mech_batch_mean.clamp(min=self.cfg.free_bits)
        loss = per_mech_batch_mean.sum()

        assert not (self.cfg.kl_warmup_end_frac > 0.0 and self.cfg.kl_warmup_end_steps > 0), (
            "set at most one of kl_warmup_end_frac / kl_warmup_end_steps"
        )
        if self.cfg.kl_warmup_end_steps > 0:
            warmup = min(1.0, ctx.step / self.cfg.kl_warmup_end_steps)
            loss = warmup * loss
        elif self.cfg.kl_warmup_end_frac > 0.0:
            warmup = min(1.0, ctx.current_frac_of_training / self.cfg.kl_warmup_end_frac)
            loss = warmup * loss

        self.sum_loss += loss.detach()
        self.sum_mean_pi += pi.detach().reshape(-1, pi.shape[-1]).mean(dim=0).sum()
        self.n_batches += 1
        return loss

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        sum_mean_pi = all_reduce(self.sum_mean_pi, op=ReduceOp.SUM)
        n_batches = all_reduce(self.n_batches, op=ReduceOp.SUM)
        name = type(self).__name__
        return {
            name: sum_loss / n_batches,
            "effective_K": sum_mean_pi / n_batches,
        }
