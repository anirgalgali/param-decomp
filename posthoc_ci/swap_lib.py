"""E2.x shared harness: swap ĝ into the run's own masking machinery.

Design constraints this file owns:
  - The 128x512 eval batch never fits through the model in one piece on a 32GB card
    (logits alone are ~13GB), so everything streams micro-batches and accumulates.
  - Paired RNG (amendment/P2): every stochastic draw seeds torch's global generator
    with f(RNG_BASE_KEY, draw, microbatch) immediately before sampling, so the true-g
    and ĝ conditions see byte-identical sources.
  - Dead atoms and Δ-components pass through untouched: ĝ is assembled by overwriting
    ONLY alive columns of the true-g dict; Δ has no CI entry by construction.
  - PGD micro-batching: sources are shared across the whole eval batch; per step, the
    gradient is accumulated over micro-batches and averaged — mathematically identical
    to the stock `shared_across_batch` all-reduce on a single big batch.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from param_decomp.masks import (
    calc_stochastic_component_mask_info,
    interpolate_component_mask,
    make_mask_infos,
)
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from posthoc_ci import constants, glib
from posthoc_ci.atom_index import canonical_modules, module_slices
from posthoc_ci.nmf import solve_codes

MICRO_SEQS = 16


def arm_solver_params(arm: str) -> dict:
    """Solver settings implied by an arm name (sym, asym<w>, sgnB, sgnBZ, sgnB-asym<w>).

    The B matrix's sign structure lives in the fit itself; only the code-solve settings
    (w_fn, signed_z) need re-deriving at substitution time.
    """
    signed_z = arm.startswith("sgnBZ")
    w_fn = 1.0
    if "asym" in arm:
        w_fn = float(arm.rsplit("asym", 1)[1])
    return {"w_fn": w_fn, "signed_z": signed_z}


def arm_is_rectified(arm: str) -> bool:
    return arm.startswith("rect")


def load_fit(fit_dir, device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(B, bias-or-None) from a fit's final.npz — bias present only for rect arms."""
    data = np.load(fit_dir / "final.npz")
    b = torch.from_numpy(data["B"]).to(device)
    bias = torch.from_numpy(data["b"]).to(device) if "b" in data.files else None
    return b, bias


@dataclass
class EvalRun:
    model: object  # ComponentModel
    tokens: torch.Tensor  # [128, 512] on device
    modules: list[str]
    slices: dict[str, slice]
    alive_cols: np.ndarray
    weight_deltas: dict[str, torch.Tensor]
    autocast: bool
    device: str


def load_eval_run(run_dir, device: str = "cuda") -> EvalRun:
    from param_decomp_lab.experiments.lm.run import SavedLMRun

    saved = SavedLMRun.from_path(run_dir)
    model = saved.load_model().to(device)
    model.eval()
    tokens = torch.from_numpy(glib.load_eval_batch()["token_ids"].astype(np.int64)).to(device)
    assert tokens.shape == (constants.EVAL_BATCH_SEQS, constants.EVAL_BATCH_SEQ_LEN)
    return EvalRun(
        model=model,
        tokens=tokens,
        modules=canonical_modules(model.module_to_c),
        slices=module_slices(model.module_to_c),
        alive_cols=glib.alive_cols(),
        weight_deltas={k: v.detach() for k, v in model.calc_weight_deltas().items()},
        autocast=bool(saved.cfg.runtime.autocast_bf16),
        device=device,
    )


def micro_slices(n_seqs: int = constants.EVAL_BATCH_SEQS, size: int = MICRO_SEQS):
    return [slice(i, min(i + size, n_seqs)) for i in range(0, n_seqs, size)]


def true_ci(run: EvalRun, tokens_mb: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=run.autocast):
        out = run.model(tokens_mb, cache_type="input")
        ci = run.model.calc_causal_importances(
            pre_weight_acts=out.cache, sampling="continuous", detach_inputs=False
        ).lower_leaky
    return {k: v.float() for k, v in ci.items()}


def target_logits(run: EvalRun, tokens_mb: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=run.autocast):
        out = run.model(tokens_mb, cache_type="input")
    return out.output.float()


def ci_dict_to_flat(run: EvalRun, ci: dict[str, torch.Tensor]) -> torch.Tensor:
    """[B, S, A_total] fp32 in canonical atom order."""
    return torch.cat([ci[m] for m in run.modules], dim=-1)


def flat_to_ci_dict(run: EvalRun, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    return {m: flat[..., run.slices[m]].contiguous() for m in run.modules}


def ghat_dict(
    run: EvalRun,
    ci: dict[str, torch.Tensor],
    b: torch.Tensor,
    w_fn: float = 1.0,
    binarize_at: float | None = None,
    identity: bool = False,
    covering: bool = False,
    signed_z: bool = False,
    bias: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Reconstructed floors for one micro-batch, plus encode stats.

    identity=True bypasses the code entirely (ĝ := g) — the harness self-test.
    covering=True solves coverage-constrained codes (Bz >= g - eps; amendment 4).
    signed_z=True solves an unconstrained code (Rung 1S, in-loop family).
    bias (Rung 1R): the per-atom hurdle; ĝ = clip(Bz - bias, 0, 1).
    """
    from posthoc_ci.nmf import solve_covering

    flat = ci_dict_to_flat(run, ci)
    if identity:
        return flat_to_ci_dict(run, flat), {"clip_rate": 0.0, "code_l0_0.01": float("nan")}
    lead = flat.shape[:-1]
    g_alive = flat[..., run.alive_cols].reshape(-1, len(run.alive_cols))
    if covering:
        assert bias is None, "covering condition is defined on the plain nonneg family"
        z, violation = solve_covering(g_alive, b)
    else:
        z = solve_codes(
            g_alive, b, constants.Z_SOLVER_N_STEPS, w_fn, signed_z=signed_z, bias=bias
        )
        violation = float("nan")
    bz = z @ b.T
    if bias is not None:
        bz = bz - bias
    ghat_alive = bz.clamp(0.0, 1.0)
    if binarize_at is not None:
        ghat_alive = (ghat_alive > binarize_at).float()
    out = flat.clone()  # dead atoms & anything not alive keep their true values
    out[..., run.alive_cols] = ghat_alive.reshape(*lead, -1)
    stats = {
        "clip_rate": float((bz > 1.0).float().mean().item()),
        "code_l0_0.01": float((z.abs() > 0.01).float().sum(-1).mean().item()),
        "covering_violation": violation,
    }
    return flat_to_ci_dict(run, out), stats


def induced_l0(run: EvalRun, ci: dict[str, torch.Tensor], tau: float) -> float:
    flat = ci_dict_to_flat(run, ci)
    return float((flat > tau).float().sum(dim=-1).mean().item())


def per_position_kl(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """[B, S] KL(target || pred) per position, fp32."""
    log_q = torch.log_softmax(pred.float(), dim=-1)
    p = torch.softmax(target.float(), dim=-1)
    return (p * (torch.log(p.clamp_min(1e-30)) - log_q)).sum(dim=-1)


def ce_sum(logits: torch.Tensor, tokens_mb: torch.Tensor) -> tuple[float, int]:
    """Next-token CE summed over positions (first token ignored, as the stock metric)."""
    import torch.nn.functional as F

    labels = tokens_mb.clone()
    labels[:, 0] = -100
    flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    flat_labels = labels[:, 1:].reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="sum")
    n = int((flat_labels != -100).sum().item())
    return float(loss.item()), n


def masked_forward(
    run: EvalRun,
    tokens_mb: torch.Tensor,
    ci: dict[str, torch.Tensor],
    mode: str,
    threshold: float = 0.0,
    rng_seed: int | None = None,
) -> torch.Tensor:
    """Logits under one deterministic/stochastic masking mode (stock semantics)."""
    if mode == "ci":
        infos = make_mask_infos(ci)
    elif mode == "unmasked":
        infos = make_mask_infos({k: torch.ones_like(v) for k, v in ci.items()})
    elif mode == "rounded":
        infos = make_mask_infos({k: (v > threshold).float() for k, v in ci.items()})
    elif mode == "zero":
        infos = make_mask_infos({k: torch.zeros_like(v) for k, v in ci.items()})
    elif mode == "stochastic":
        assert rng_seed is not None
        from param_decomp.masks import AllLayersRouter

        torch.manual_seed(rng_seed)
        infos = calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling="continuous",
            weight_deltas=run.weight_deltas,
            router=AllLayersRouter(),
        )
    else:
        raise ValueError(mode)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=run.autocast):
        logits = run.model(tokens_mb, mask_infos=infos)
    return logits.float()


def mb_seed(draw: int, mb_index: int) -> int:
    return constants.RNG_BASE_KEY + 7919 * draw + mb_index


def pgd_hardened_kl(
    run: EvalRun,
    ci_per_mb: list[dict[str, torch.Tensor]],
    target_per_mb: list[torch.Tensor],
    n_steps: int,
    step_size: float,
    seed: int,
    n_restarts: int = 3,
    scope: str = "shared",
) -> dict:
    """Hardened E2.3R cell: multiple independent restarts (report the max — the
    adversary keeps its best attack) under either source scope.

    scope='shared': the paper protocol (one source vector broadcast batch-wide).
    scope='per_token': every position gets its own sources — a strictly stronger
    attacker searching a ~40k-dim box per position instead of one shared point.
    Per-token needs no cross-micro-batch coupling, so each micro-batch runs its own
    independent ascent.
    """
    kls = []
    for r in range(n_restarts):
        if scope == "shared":
            kl = pgd_shared_kl(run, ci_per_mb, target_per_mb, n_steps, step_size,
                               seed + 7919 * r)
        else:
            kl = _pgd_per_token_kl(run, ci_per_mb, target_per_mb, n_steps, step_size,
                                   seed + 7919 * r)
        kls.append(kl)
    return {"max": max(kls), "restarts": kls, "scope": scope}


def _pgd_per_token_kl(
    run: EvalRun,
    ci_per_mb: list[dict[str, torch.Tensor]],
    target_per_mb: list[torch.Tensor],
    n_steps: int,
    step_size: float,
    seed: int,
) -> float:
    device = run.device
    slices_ = micro_slices()
    n_total = sum(t.shape[0] * t.shape[1] for t in target_per_mb)
    total = 0.0
    for i, mb in enumerate(slices_):
        torch.manual_seed(seed + i)
        tokens_mb = run.tokens[mb]
        lead = tokens_mb.shape
        ci_mb = {k: v.to(device, torch.float32) for k, v in ci_per_mb[i].items()}
        target_mb = target_per_mb[i].to(device, torch.float32)
        sources = {
            m: torch.rand(*lead, run.model.module_to_c[m] + 1, device=device
                          ).requires_grad_(True)
            for m in run.modules
        }

        def forward() -> torch.Tensor:
            comp = {k: v[..., :-1] for k, v in sources.items()}
            wdm = {k: (run.weight_deltas[k], sources[k][..., -1])
                   for k in run.weight_deltas}
            infos = make_mask_infos(
                component_masks=interpolate_component_mask(ci_mb, comp),
                weight_deltas_and_masks=wdm,
            )
            with torch.autocast("cuda", torch.bfloat16, enabled=run.autocast):
                logits = run.model(tokens_mb, mask_infos=infos)
            sum_kl, _ = recon_loss_kl(pred=logits.float(), target=target_mb)
            return sum_kl

        for _ in range(n_steps):
            with torch.enable_grad():
                sum_kl = forward()
            grads = torch.autograd.grad(sum_kl, list(sources.values()))
            with torch.no_grad():
                for k, gk in zip(sources, grads, strict=True):
                    sources[k].add_(step_size * gk.sign())
                    sources[k].clamp_(0.0, 1.0)
        with torch.no_grad():
            total += float(forward().item())
        del sources, ci_mb, target_mb
        torch.cuda.empty_cache()
    return total / n_total


def pgd_shared_kl(
    run: EvalRun,
    ci_per_mb: list[dict[str, torch.Tensor]],
    target_per_mb: list[torch.Tensor],
    n_steps: int,
    step_size: float,
    seed: int,
) -> float:
    """Sign-PGD with sources shared across the FULL eval batch, micro-batch accumulated.

    Equivalent to the stock `shared_across_batch` protocol: per step, grad of the mean
    KL over the whole batch = weighted mean of micro-batch grads. Sources include the
    Δ column (mask_c = C + 1), exactly as `_init_adv_sources` builds them.

    `ci_per_mb` / `target_per_mb` may live on CPU (fp16): each micro-batch is streamed
    to the device per forward — the full eval batch's floors + target logits do NOT fit
    on a 32GB card alongside the PGD graph.
    """
    device = run.device
    torch.manual_seed(seed)
    sources = {
        m: torch.rand(1, 1, run.model.module_to_c[m] + 1, device=device).requires_grad_(True)
        for m in run.modules
    }
    slices_ = micro_slices()
    n_total = sum(t.shape[0] * t.shape[1] for t in target_per_mb)

    def forward_mb(i: int, mb: slice) -> torch.Tensor:
        tokens_mb = run.tokens[mb]
        lead = tokens_mb.shape
        ci_mb = {k: v.to(device, torch.float32, non_blocking=True)
                 for k, v in ci_per_mb[i].items()}
        target_mb = target_per_mb[i].to(device, torch.float32, non_blocking=True)
        expanded = {k: v.expand(*lead, -1) for k, v in sources.items()}
        comp_sources = {k: v[..., :-1] for k, v in expanded.items()}
        wdm = {k: (run.weight_deltas[k], expanded[k][..., -1]) for k in run.weight_deltas}
        infos = make_mask_infos(
            component_masks=interpolate_component_mask(ci_mb, comp_sources),
            weight_deltas_and_masks=wdm,
        )
        with torch.autocast("cuda", torch.bfloat16, enabled=run.autocast):
            logits = run.model(tokens_mb, mask_infos=infos)
        sum_kl, _ = recon_loss_kl(pred=logits.float(), target=target_mb)
        return sum_kl

    for _ in range(n_steps):
        grads = {k: torch.zeros_like(v) for k, v in sources.items()}
        for i, mb in enumerate(slices_):
            with torch.enable_grad():
                sum_kl = forward_mb(i, mb)
            g = torch.autograd.grad(sum_kl / n_total, list(sources.values()))
            for k, gk in zip(sources, g, strict=True):
                grads[k] += gk
        with torch.no_grad():
            for k in sources:
                sources[k].add_(step_size * grads[k].sign())
                sources[k].clamp_(0.0, 1.0)

    total = 0.0
    with torch.no_grad():
        for i, mb in enumerate(slices_):
            total += float(forward_mb(i, mb).item())
    return total / n_total
