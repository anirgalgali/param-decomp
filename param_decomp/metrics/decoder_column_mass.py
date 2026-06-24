"""Convex-in-column-mass penalty on the spike-gated decoder.

`P(B) = Σ_k (Σ_m |B[m,k]|)²` — L1 down each decoder column, squared, summed over mechanisms
(the exclusive-lasso / `ℓ₁,₂` form). This is the counter-pressure to the existence-KL's merge
bias: the square charges for *concentration*, so merging two columns of mass `M_i, M_j` raises
it by `2 M_i M_j`. It is a parameter-only penalty (data-independent, paid once), read directly
off the CI fn's decoder `B`. See `docs/slpd_stage1_spike_design.md` §8 in the slpd project.
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


class DecoderColumnMassLossConfig(LossMetricConfig):
    type: Literal["DecoderColumnMassLoss"] = "DecoderColumnMassLoss"


def decoder_column_mass_loss(b: Float[Tensor, "M K"]) -> Float[Tensor, ""]:
    """`Σ_k (Σ_m |B[m,k]|)²` — squared L1 column mass, summed over mechanisms."""
    column_mass = b.abs().sum(dim=0)  # [K]
    return (column_mass**2).sum()


class DecoderColumnMassLoss(Metric[DecoderColumnMassLossConfig]):
    """Convex-in-column-mass decoder penalty (counter-pressure to the KL's merge bias)."""

    log_namespace = "loss"
    short_name = "DecColMass"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_batches = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        spike_fn = get_spike_gated_ci_fn(ctx.model.ci_fn)
        loss = decoder_column_mass_loss(spike_fn.B)
        self.sum_loss += loss.detach()
        self.n_batches += 1
        return loss

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_batches = all_reduce(self.n_batches, op=ReduceOp.SUM)
        return sum_loss / n_batches
