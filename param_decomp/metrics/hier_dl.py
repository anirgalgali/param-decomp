"""Two-part description-length loss (Term H) for the VPD-hier hierarchical CI.

Prices the two-level code `g_c = γ_{k(c)} · h_c`: name each active group (κ_k = log2(1/f_k)), then
specify its interior (Σ_{c∈k} ℓ(h_c; q_c)), all charged through the group event indicator |γ_k|^p:

    per_token_k = |γ_{k,t}|^p · ( κ_k + Σ_{c∈k} ℓ(h_{c,t}; q_c) )
    loss        = mean_t Σ_k per_token_k

`γ_up`/`h_up` (the upper-leaky group/interior gates) are read off the `HierarchicalCiFn` cache each
forward. Batch statistics `f_k` (group frequency) and `q_c` (P(interior on | group on)) are held as
detached EMA buffers and stop-gradded (envelope-theorem correction, doc §6.6): gradients flow through
the live `γ_{k,t}`, `h_{c,t}` but not through `κ_k`, `q_c`. See `docs/hier_ci_implementation_plan.md`.
"""

from typing import Any, Literal, override

import torch
from jaxtyping import Float
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import Probability
from param_decomp.ci_fns import get_hierarchical_ci_fn
from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.importance_minimality import _get_linear_annealed_p

LOG2 = 0.6931471805599453  # ln 2, to convert torch.log → log2 without repeated tensor allocs


def batch_stats(
    gamma: Float[Tensor, "N K"],
    h: Float[Tensor, "N M"],
    k_of_c: Tensor,
    *,
    eps: float,
) -> tuple[Float[Tensor, " K"], Float[Tensor, " M"]]:
    """Group frequency `f_k = mean_t γ_k` and γ-weighted interior stat `q_c = Σ_t γ_{k(c)}h_c / Σ_t γ_{k(c)}`."""
    f = gamma.mean(0)
    group_mass = gamma.sum(0).clamp(min=eps)
    q = (gamma[:, k_of_c] * h).sum(0) / group_mass[k_of_c]
    return f, q


def two_part_code_cost(
    gamma: Float[Tensor, "N K"],
    h: Float[Tensor, "N M"],
    k_of_c: Tensor,
    *,
    kappa: Float[Tensor, " K"],
    q: Float[Tensor, " M"],
    p: float,
    interior_code: str,
    naming_form: str,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token two-part DL `mean_t Σ_k |γ_k|^p (κ_k + Σ_{c∈k} ℓ(h_c;q_c))`, returned as
    `(loss, naming_component, interior_component)`. `kappa`, `q` are the (detached) code statistics;
    gradients flow only through the live `gamma`, `h`."""
    log2_inv_q = -torch.log(q) / LOG2
    log2_inv_1mq = -torch.log1p(-q) / LOG2
    if interior_code == "complete":
        ell = h * log2_inv_q + (1.0 - h) * log2_inv_1mq  # protects nesting; ratchets at q>½
    else:
        ell = h * log2_inv_q  # on_only — monotone downward, no flip
    interior_per_group = torch.zeros_like(gamma)
    interior_per_group.index_add_(1, k_of_c, ell)
    if naming_form == "surprisal":
        naming = kappa.unsqueeze(0)  # [1, K] detached
    else:  # count: log2(1 + Σ_t |γ_k|^p) — grad-carrying (§7.2)
        naming = torch.log1p(((gamma + eps) ** p).sum(0)).unsqueeze(0) / LOG2
    gamma_pow = (gamma + eps) ** p  # event indicator |γ|^p
    naming_c = (gamma_pow * naming).sum(1).mean(0)
    interior_c = (gamma_pow * interior_per_group).sum(1).mean(0)
    return naming_c + interior_c, naming_c, interior_c


class TwoPartCodeLossConfig(LossMetricConfig):
    """Config for the two-part description-length loss (Term H).

    `interior_code`: `complete` = `h·log2(1/q)+(1-h)·log2(1/(1-q))` (protects nesting; ratchets at
    q>½); `on_only` = `h·log2(1/q)` (monotone downward, no flip). `naming_form`: `surprisal` = the
    detached `κ_k = log2(1/f_k)` (§4.9 recipe); `count` = `log2(1 + Σ_t |γ_k|^p)` (grad-carrying, the
    §7.2 pressure variant). `eps_floor` floors `f_k` and lower-clamps `q_c` (anti-death); `eps_ceil`
    upper-clamps `q_c` (ratchet ceiling `log2((1-ε)/ε)`). The p-anneal fields mirror Term F so the two
    share the same event-indicator homotopy (`p_shared_with_termF` documents that intent).
    """

    type: Literal["TwoPartCodeLoss"] = "TwoPartCodeLoss"
    interior_code: Literal["complete", "on_only"] = "complete"
    naming_form: Literal["surprisal", "count"] = "surprisal"
    eps_floor: PositiveFloat = 1e-2
    eps_ceil: Probability = 0.1
    ema_momentum: Probability = 0.99
    warmup_end_frac: Probability = 0.1
    warmup_end_steps: NonNegativeInt = 0
    pnorm: NonNegativeFloat = 2.0
    p_anneal_start_frac: Probability = 0.0
    p_anneal_final_p: NonNegativeFloat | None = 0.4
    p_anneal_end_frac: Probability = 0.8
    p_shared_with_termF: bool = True
    eps: NonNegativeFloat = 1e-12


class TwoPartCodeLoss(Metric[TwoPartCodeLossConfig]):
    """Two-part DL (Term H) on the hierarchical composed gate. Detached EMA batch stats."""

    log_namespace = "loss"
    short_name = "TwoPartDL"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.sum_naming = torch.zeros((), device=self.device)
        self.sum_interior = torch.zeros((), device=self.device)
        self.n_batches = torch.zeros((), device=self.device, dtype=torch.long)
        # EMA buffers are lazily initialised from the first batch (no cold-start bias).
        self._f_ema: Tensor | None = None
        self._q_ema: Tensor | None = None

    def _update_ema(self, f_batch: Tensor, q_batch: Tensor) -> None:
        m = self.cfg.ema_momentum
        if self._f_ema is None:
            self._f_ema = f_batch.clone()
            self._q_ema = q_batch.clone()
        else:
            assert self._q_ema is not None
            self._f_ema.mul_(m).add_(f_batch, alpha=1.0 - m)
            self._q_ema.mul_(m).add_(q_batch, alpha=1.0 - m)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        fn = get_hierarchical_ci_fn(ctx.model.ci_fn)
        assert fn._gamma_up is not None and fn._gamma_low is not None and fn._h_low is not None, (
            "CI fn forward must run before TwoPartCodeLoss.update"
        )
        assert isinstance(fn.k_of_c, Tensor)
        # Event indicator |γ|^p uses the upper-leaky γ (the above-1 leak is legitimate downward
        # pressure on saturated groups). The interior CHARGE and the code STATISTICS use the bounded
        # lower-leaky gates: ℓ(h;q) and κ=log2(1/f) are code lengths, only defined for gates in [0,1].
        # (Charging the interior on the unbounded h_up lets the q>½ ratchet drive h→∞, sending the DL
        # to −∞ and flipping the γ pressure to inflation — the collapse seen in the first E1 run.)
        gamma_up = fn._gamma_up.reshape(-1, fn.K).float()  # [N, K] event indicator
        gamma_low = fn._gamma_low.reshape(-1, fn.K).float()  # [N, K] bounded gate (stats)
        h_low = fn._h_low.reshape(-1, fn.M).float()  # [N, M] bounded interior gate

        # Detached EMA code statistics (stop-grad; doc §6.6).
        with torch.no_grad():
            f_batch, q_batch = batch_stats(gamma_low, h_low, fn.k_of_c, eps=self.cfg.eps)
            self._update_ema(f_batch, q_batch)
            assert self._f_ema is not None and self._q_ema is not None
            f = self._f_ema.clamp(min=self.cfg.eps_floor)
            q = self._q_ema.clamp(self.cfg.eps_floor, 1.0 - self.cfg.eps_ceil)
            kappa = -torch.log(f) / LOG2  # [K] = log2(1/f)

        loss, naming_c, interior_c = two_part_code_cost(
            gamma_up,
            h_low,
            fn.k_of_c,
            kappa=kappa,
            q=q,
            p=self._annealed_p(ctx),
            interior_code=self.cfg.interior_code,
            naming_form=self.cfg.naming_form,
            eps=self.cfg.eps,
        )
        loss = self._warmup(ctx) * loss

        self.sum_loss += loss.detach()
        self.sum_naming += naming_c.detach()
        self.sum_interior += interior_c.detach()
        self.n_batches += 1
        return loss

    def _annealed_p(self, ctx: MetricContext) -> float:
        return _get_linear_annealed_p(
            current_frac_of_training=ctx.current_frac_of_training,
            initial_p=self.cfg.pnorm,
            p_anneal_start_frac=self.cfg.p_anneal_start_frac,
            p_anneal_final_p=self.cfg.p_anneal_final_p,
            p_anneal_end_frac=self.cfg.p_anneal_end_frac,
        )

    def _warmup(self, ctx: MetricContext) -> float:
        assert not (self.cfg.warmup_end_frac > 0.0 and self.cfg.warmup_end_steps > 0), (
            "set at most one of warmup_end_frac / warmup_end_steps"
        )
        if self.cfg.warmup_end_steps > 0:
            return min(1.0, ctx.step / self.cfg.warmup_end_steps)
        if self.cfg.warmup_end_frac > 0.0:
            return min(1.0, ctx.current_frac_of_training / self.cfg.warmup_end_frac)
        return 1.0

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        sum_naming = all_reduce(self.sum_naming, op=ReduceOp.SUM)
        sum_interior = all_reduce(self.sum_interior, op=ReduceOp.SUM)
        n_batches = all_reduce(self.n_batches, op=ReduceOp.SUM)
        name = type(self).__name__
        return {
            name: sum_loss / n_batches,
            f"{name}_naming": sum_naming / n_batches,
            f"{name}_interior": sum_interior / n_batches,
        }

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"f_ema": self._f_ema, "q_ema": self._q_ema}

    @override
    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._f_ema = state.get("f_ema")
        self._q_ema = state.get("q_ema")
