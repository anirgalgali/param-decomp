"""Usage-weighted L1 on the spike-gated decoder columns — the ε-price.

`P(B, z) = Σ_k M_k · E_x[z_k(x)]` with `M_k = Σ_c |B[c,k]|` the decoder column mass and `z_k(x)`
the mechanism's per-datapoint firing (charged on the hard-concrete open-probability `P(z>0)`, the
same quantity the existence-KL charges on). This is importance-minimality factored through the
code: linear in mass (shard-invariant, unlike the exclusive lasso's `M_k²`), weighted by usage (a
never-firing mechanism pays nothing, so silent columns stop bleeding), and computed on the
pre-`σ_H` contribution so a saturated gate still pays. See `docs/slpd_pressure_and_repair_v2.md`
§7.3 in the slpd project. Gradient reaches `B` (via `M_k`) and the encoder (via `E_x[z_k]`).
"""

from typing import Literal, override

import torch
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.ci_fns import get_spike_gated_ci_fn
from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class EpsilonPriceLossConfig(LossMetricConfig):
    """`charge_on` selects the per-datapoint usage `z_k(x)` the column mass is weighted by:
    `"open_prob"` (the hard-concrete `P(z>0)`, matching the sampled gate) or `"pi"` (sigmoid logits).
    """

    type: Literal["EpsilonPriceLoss"] = "EpsilonPriceLoss"
    charge_on: Literal["pi", "open_prob"] = "open_prob"


class EpsilonPriceLoss(Metric[EpsilonPriceLossConfig]):
    """`Σ_k M_k · E_x[z_k(x)]` — usage-weighted decoder column-mass L1 (the ε-price)."""

    log_namespace = "loss"
    short_name = "EpsPrice"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_batches = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        spike_fn = get_spike_gated_ci_fn(ctx.model.ci_fn)
        column_mass = spike_fn.B.abs().sum(dim=0)  # [K]
        z = spike_fn.gate_open_prob() if self.cfg.charge_on == "open_prob" else spike_fn._pi
        assert z is not None, "CI fn forward must run before EpsilonPriceLoss.update"
        usage = z.reshape(-1, z.shape[-1]).mean(dim=0)  # [K]  E_x[z_k]
        loss = (column_mass * usage).sum()
        self.sum_loss += loss.detach()
        self.n_batches += 1
        return loss

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_batches = all_reduce(self.n_batches, op=ReduceOp.SUM)
        return sum_loss / n_batches
