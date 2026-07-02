"""Convex-in-row-mass (exclusive-lasso) penalty on the spike-gated decoder.

`P(B) = Σ_m (Σ_k |B[m,k]|)²` — L1 across each decoder row, squared, summed over components (the
row-wise transpose of `decoder_column_mass`). It prices **row concentration**: a component driven
by many mechanisms. The square charges for spread, so a component reading from two mechanisms of
mass `M_i, M_j` costs `2 M_i M_j` more than reading from one — pushing each component toward a
single mechanism (block rows ⇒ ~one mechanism per group). Parameter-only (data-independent, paid
once), read directly off the CI fn's decoder `B`. Counter-pressure on the row-spread that the
column-mass penalty and the KL leave unpriced.
"""

from typing import Literal, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.ci_fns import get_spike_gated_ci_fn
from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class DecoderRowMassLossConfig(LossMetricConfig):
    type: Literal["DecoderRowMassLoss"] = "DecoderRowMassLoss"


def decoder_row_mass_loss(b: Float[Tensor, "M K"]) -> Float[Tensor, ""]:
    """`Σ_m (Σ_k |B[m,k]|)²` — squared L1 row mass, summed over components."""
    row_mass = b.abs().sum(dim=1)  # [M]
    return (row_mass**2).sum()


class DecoderRowMassLoss(Metric[DecoderRowMassLossConfig]):
    """Convex-in-row-mass decoder penalty (prices a component driven by many mechanisms)."""

    log_namespace = "loss"
    short_name = "DecRowMass"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_batches = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        spike_fn = get_spike_gated_ci_fn(ctx.model.ci_fn)
        loss = decoder_row_mass_loss(spike_fn.B)
        self.sum_loss += loss.detach()
        self.n_batches += 1
        return loss

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_batches = all_reduce(self.n_batches, op=ReduceOp.SUM)
        return sum_loss / n_batches
