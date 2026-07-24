"""Two-part description-length loss (Term H) for the VPD-hier CI under the envelope parameterization.

Prices the code `γ_k = noisy-OR_{c∈k}(g_c)` (no h head; Stage 2): name each active group
(`κ_k = log2(1/f_k)`), then specify its interior with the conditional cross-entropy, written without
materializing `h` (approach §1.4):

    per_token_k = γ_{k,t} κ_k + Σ_{c∈k} ( g_{c,t} log2(1/q_c) + (γ_{k,t} − g_{c,t}) log2(1/(1−q_c)) )
    loss        = mean_t Σ_k per_token_k

The pooled `γ` (`_gamma_low`) and the raw gates `g` (`_g_low`) are read off the `HierarchicalCiFn`
cache each forward — both bounded in [0,1], so every term is ≥ 0. Batch statistics `f_k` (group
frequency) and `q_c` (P(member on | group on)) are held as detached EMA buffers and stop-gradded
(envelope-theorem correction): gradients flow through the live `γ`, `g` (and thence, via the noisy-OR
pool, the raw gates) but not through `κ_k`, `q_c`. No `|γ|^p` event weighting — the p-anneal lives on
Term F only. See `docs/learning_the_assignment_matrix_A.md` §1.4 and `docs/stage2_learning_assignment_plan.md` §2.2.
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

LOG2 = 0.6931471805599453  # ln 2, to convert torch.log → log2 without repeated tensor allocs


def batch_stats(
    gamma: Float[Tensor, "N K"],
    g: Float[Tensor, "N M"],
    k_of_c: Tensor,
    *,
    eps: float,
) -> tuple[Float[Tensor, " K"], Float[Tensor, " M"]]:
    """Group frequency `f_k = mean_t γ_k` and conditional member rate `q_c = Σ_t g_c / Σ_t γ_{k(c)}`."""
    f = gamma.mean(0)
    group_mass = gamma.sum(0).clamp(min=eps)
    q = g.sum(0) / group_mass[k_of_c]
    return f, q


def two_part_code_cost(
    gamma: Float[Tensor, "N K"],
    g: Float[Tensor, "N M"],
    k_of_c: Tensor,
    *,
    kappa: Float[Tensor, " K"],
    q: Float[Tensor, " M"],
    interior_code: str,
    naming_form: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token envelope-form two-part DL, returned as `(loss, naming_component, interior_component)`:

        Σ_k [ γ_k κ_k + Σ_{c∈k} ( g_c log2(1/q_c) + (γ_{k(c)} − g_c) log2(1/(1−q_c)) ) ]

    (`on_only` drops the second interior piece.) `kappa`, `q` are the (detached) code statistics;
    gradients flow only through the live `gamma`, `g`. All quantities are in [0,1] ⇒ every term ≥ 0."""
    log2_inv_q = -torch.log(q) / LOG2
    log2_inv_1mq = -torch.log1p(-q) / LOG2
    gamma_gathered = gamma[:, k_of_c]  # [N, M] — γ_{k(c)} per member
    if interior_code == "complete":
        off = (gamma_gathered - g).clamp(
            min=0.0
        )  # ≥ 0 (envelope); clamp guards the noisy-OR δ-slack
        ell = g * log2_inv_q + off * log2_inv_1mq  # protects nesting; ratchets at q>½
    else:
        ell = g * log2_inv_q  # on_only — monotone downward, no flip
    interior_per_group = torch.zeros_like(gamma)
    interior_per_group.index_add_(1, k_of_c, ell)
    if naming_form == "surprisal":
        naming = kappa.unsqueeze(0)  # [1, K] detached
    else:  # count: log2(1 + Σ_t γ_k) — grad-carrying (§7.2)
        naming = torch.log1p(gamma.sum(0)).unsqueeze(0) / LOG2
    naming_c = (gamma * naming).sum(1).mean(0)
    interior_c = interior_per_group.sum(1).mean(0)
    return naming_c + interior_c, naming_c, interior_c


class TwoPartCodeLossConfig(LossMetricConfig):
    """Config for the two-part description-length loss (Term H), envelope form.

    `interior_code`: `complete` = `g·log2(1/q)+(γ−g)·log2(1/(1−q))` (protects nesting; ratchets at
    q>½); `on_only` = `g·log2(1/q)` (monotone downward, no flip — the E5 ablation). `naming_form`:
    `surprisal` = the detached `κ_k = log2(1/f_k)`; `count` = `log2(1 + Σ_t γ_k)` (grad-carrying, the
    §7.2 pressure variant). `eps_floor` floors `f_k` and lower-clamps `q_c` (anti-death); `eps_ceil`
    upper-clamps `q_c` — together they set the conformity cap `log2((1−ε_ceil)/ε_ceil)` that the
    `β_F/β_H` no-capture floor is derived against (approach §1.6). No p-anneal here — the homotopy
    lives on Term F only (approach §1.4).
    """

    type: Literal["TwoPartCodeLoss"] = "TwoPartCodeLoss"
    interior_code: Literal["complete", "on_only"] = "complete"
    naming_form: Literal["surprisal", "count"] = "surprisal"
    eps_floor: PositiveFloat = 1e-2
    eps_ceil: Probability = 0.1
    ema_momentum: Probability = 0.99
    warmup_end_frac: Probability = 0.1
    warmup_end_steps: NonNegativeInt = 0
    eps: NonNegativeFloat = 1e-12


class TwoPartCodeLoss(Metric[TwoPartCodeLossConfig]):
    """Two-part DL (Term H) on the derived noisy-OR group gate. Detached EMA batch stats."""

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
        assert fn._gamma_low is not None and fn._g_low is not None, (
            "CI fn forward must run before TwoPartCodeLoss.update"
        )
        assert isinstance(fn.k_of_c, Tensor)
        # The loss reads the bounded lower-leaky gates: the pooled group gate γ and the raw gates g.
        # Both ∈ [0,1] ⇒ κ = log2(1/f) and the interior conditional cross-entropy are well-defined code
        # lengths and every term is ≥ 0. Gradients flow through g directly and through γ = noisy-OR(g)
        # (routing ∂/∂g_c the pool sensitivity D_c; approach §1.5). There is no separate event
        # indicator — the |γ|^p weighting is dropped (approach §1.4); the p-anneal lives on Term F.
        gamma = fn._gamma_low.reshape(-1, fn.K).float()  # [N, K] pooled group gate
        g = fn._g_low.reshape(-1, fn.M).float()  # [N, M] raw gate (= composed CI)

        # Detached EMA code statistics (stop-grad; envelope theorem).
        with torch.no_grad():
            f_batch, q_batch = batch_stats(gamma, g, fn.k_of_c, eps=self.cfg.eps)
            self._update_ema(f_batch, q_batch)
            assert self._f_ema is not None and self._q_ema is not None
            f = self._f_ema.clamp(min=self.cfg.eps_floor)
            q = self._q_ema.clamp(self.cfg.eps_floor, 1.0 - self.cfg.eps_ceil)
            kappa = -torch.log(f) / LOG2  # [K] = log2(1/f)

        loss, naming_c, interior_c = two_part_code_cost(
            gamma,
            g,
            fn.k_of_c,
            kappa=kappa,
            q=q,
            interior_code=self.cfg.interior_code,
            naming_form=self.cfg.naming_form,
        )
        loss = self._warmup(ctx) * loss

        self.sum_loss += loss.detach()
        self.sum_naming += naming_c.detach()
        self.sum_interior += interior_c.detach()
        self.n_batches += 1
        return loss

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
