"""E1.3 core: nonnegative factorization G ~= clip(Z B^T, 0, 1) and the frozen-B encoder.

The fit optimizes exactly the object Rung 2 substitutes (amendment 1): the upper clip
uses a straight-through gradient (value clipped, gradient identity), so overshoot into
g < 1 territory still receives corrective gradient while exact fits receive none.
Z >= 0 and B >= 0 throughout. Optional asymmetric weighting `w_fn` multiplies the
squared residual wherever the model UNDER-predicts (ghat < g) — the false-negative
direction (amendment 2).

The batched projected-gradient z-solver (`solve_codes`) is THE shared encoder: E1.3
fits reuse it on held-out data and E2.0 reuses it verbatim on the eval batch.
"""

from dataclasses import dataclass

import torch


def clip_st(x: torch.Tensor) -> torch.Tensor:
    """clip(x, 0, 1) in value, identity in gradient (straight-through)."""
    return x + (x.clamp(0.0, 1.0) - x).detach()


def _loss_weights(g: torch.Tensor, ghat: torch.Tensor, w_fn: float) -> torch.Tensor:
    if w_fn == 1.0:
        return torch.ones_like(g)
    return torch.where(ghat < g, torch.full_like(g, w_fn), torch.ones_like(g))


def _spectral_norm_sq(b: torch.Tensor, n_iter: int = 30) -> float:
    v = torch.randn(b.shape[1], device=b.device)
    v /= v.norm()
    for _ in range(n_iter):
        v = b.T @ (b @ v)
        v /= v.norm() + 1e-12
    return float((b @ v).norm() ** 2)


def solve_codes(
    g: torch.Tensor,
    b: torch.Tensor,
    n_steps: int,
    w_fn: float = 1.0,
    lambda_z: float = 0.0,
    signed_z: bool = False,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """argmin_z sum w * (g - clip(z B^T - bias))^2 + lambda_z ||z||_1, per row of `g`.

    g: [T, A], b: [A, K] (frozen), bias: [A] >= 0 or None (the Rung-1R per-atom
    hurdle; frozen). Returns z: [T, K]. Manual FISTA with a 1/L step from the spectral
    norm of B; straight-through gradient through the clip (same surrogate as every
    prior arm). z is constrained >= 0 unless `signed_z`.
    """
    lr = 1.0 / max(_spectral_norm_sq(b), 1e-8)
    col_sq = (b * b).sum(dim=0).clamp_min(1e-8)
    # Init from g alone even when a hurdle is present: a (g + bias) matched filter
    # would tell every token to reproduce the hurdle on ALL atoms (including the
    # ~98% with g = 0 whose correct pre-clip score is anything BELOW b), inflating
    # the init by bias @ B — catastrophic at real scale. Under-initialization is the
    # benign direction: the straight-through gradient restores missed floors.
    z = (g @ b) / col_sq
    if not signed_z:
        z = z.clamp_min_(0.0)
    y = z.clone()
    t = 1.0
    for _ in range(n_steps):  # FISTA (accelerated [projected] gradient)
        score = y @ b.T
        if bias is not None:
            score = score - bias
        ghat = score.clamp(0.0, 1.0)
        w = _loss_weights(g, ghat, w_fn)
        grad = 2.0 * ((w * (ghat - g)) @ b) + lambda_z * torch.sign(y)
        z_next = y - lr * grad
        if not signed_z:
            z_next = z_next.clamp_min_(0.0)
        t_next = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        y = z_next + ((t - 1.0) / t_next) * (z_next - z)
        z, t = z_next, t_next
    return z if signed_z else z.clamp_min_(0.0)


def solve_covering(
    g: torch.Tensor,
    b: torch.Tensor,
    n_steps: int = 200,
    eps: float = 0.01,
    lam: float = 100.0,
) -> tuple[torch.Tensor, float]:
    """Coverage-constrained codes (amendment 4): min sum(z) s.t. z >= 0, Bz >= g - eps.

    Hinge-penalty projected gradient (exactness not required — the residual violation
    is returned and reported). Isolates the false-positive-only regime: these codes
    never under-predict beyond eps except where reported.
    """
    lr = 1.0 / max(_spectral_norm_sq(b), 1e-8) / lam
    col_sq = (b * b).sum(dim=0).clamp_min(1e-8)
    z = ((g @ b) / col_sq).clamp_min_(0.0) * 2.0  # start generous
    ones = torch.ones_like(z)
    for _ in range(n_steps):
        deficit = (g - eps - (z @ b.T)).clamp_min(0.0)
        grad = ones - 2.0 * lam * (deficit @ b)
        z = (z - lr * grad).clamp_min_(0.0)
    final_deficit = (g - eps - (z @ b.T)).clamp_min(0.0)
    violation = float(final_deficit.sum(dim=-1).mean().item())
    return z, violation


@dataclass
class FitConfig:
    k: int
    seed: int
    w_fn: float = 1.0
    lambda_z: float = 0.0
    epochs: int = 4
    z_steps: int = 40
    lr_b: float = 1e-2
    b_steps_per_batch: int = 4
    no_clip: bool = False  # ablation: plain bilinear objective
    signed_b: bool = False  # Rung 1S: allow inhibitory pattern entries
    signed_z: bool = False  # Rung 1S: the in-loop (DoubleSidedJumpReLU) code family
    rectified: bool = False  # Rung 1R: learn a per-atom hurdle b >= 0, ĝ = clip(Bz - b)
    freeze_b_epochs: int = 1  # Rung 1R §3.1: hurdle warm-up (fit starts as pure sym)


def init_b(g_sample: torch.Tensor, k: int, seed: int, signed: bool = False) -> torch.Tensor:
    """Exemplar init: K random data rows as columns of B (seed-varied), floored at 0.

    The exemplar rows are nonnegative by construction, which works for both modes
    (Harry's `proto` init did the same); signed mode adds zero-mean instead of
    positive jitter so cancellation directions are reachable from the start.
    """
    gen = torch.Generator(device=g_sample.device).manual_seed(seed)
    rows = torch.randint(0, g_sample.shape[0], (k,), generator=gen, device=g_sample.device)
    b = g_sample[rows].T.clone()
    if signed:
        b += 0.01 * torch.randn(b.shape, generator=gen, device=g_sample.device)
        return b
    b += 0.01 * torch.rand(b.shape, generator=gen, device=g_sample.device)
    return b.clamp_min_(0.0)


def fit_b(
    cfg: FitConfig,
    batches,  # callable () -> iterator of [T_b, A] fp32 GPU tensors (one epoch)
    g_sample: torch.Tensor,
    b_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Alternating fit; returns (B [A, K], bias [A] or None). Re-invokes `batches` per epoch.

    Rectified (Rung 1R): a per-atom hurdle `bias >= 0` is learned jointly, initialized
    at 0 and frozen for the first `freeze_b_epochs` epochs so the fit starts as pure
    sym before the hurdle is released (§3.1 warm-up). b ≡ 0 recovers the sym arm
    exactly — the family is nested.
    """
    if b_init is not None:
        b = b_init.clone().to(g_sample.device).requires_grad_(True)
    else:
        b = init_b(g_sample, cfg.k, cfg.seed, signed=cfg.signed_b).requires_grad_(True)
    bias = None
    params = [b]
    if cfg.rectified:
        assert not cfg.signed_b and not cfg.signed_z, "rectified arm is nonneg by design"
        bias = torch.zeros(g_sample.shape[1], device=g_sample.device).requires_grad_(True)
        params.append(bias)
    opt = torch.optim.Adam(params, lr=cfg.lr_b)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr_b * 0.05
    )
    for epoch in range(cfg.epochs):
        bias_frozen = cfg.rectified and epoch < cfg.freeze_b_epochs
        epoch_loss, n_batches = 0.0, 0
        for g in batches():
            with torch.no_grad():
                z = solve_codes(
                    g, b.detach(), cfg.z_steps, cfg.w_fn, cfg.lambda_z, cfg.signed_z,
                    bias=bias.detach() if bias is not None else None,
                )
            for _ in range(cfg.b_steps_per_batch):
                score = z @ b.T
                if bias is not None:
                    score = score - bias
                ghat = score if cfg.no_clip else clip_st(score)
                w = _loss_weights(g, ghat.detach(), cfg.w_fn)
                loss = (w * (ghat - g) ** 2).sum() / g.shape[0]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if bias_frozen and bias is not None:
                    bias.grad = None  # Adam skips params with no grad
                opt.step()
                with torch.no_grad():
                    if not cfg.signed_b:
                        b.clamp_min_(0.0)
                    if bias is not None:
                        bias.clamp_min_(0.0)
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()
        tag = " (b frozen)" if bias_frozen else ""
        print(f"epoch {epoch}: mean per-row loss {epoch_loss / max(n_batches, 1):.4f}{tag}")
    return b.detach(), (bias.detach() if bias is not None else None)


@dataclass
class HeldOutMetrics:
    ev: float
    recall: float
    recall_weighted: float
    induced_l0_ratio: float
    code_l0: dict[str, float]
    clip_rate: float
    mean_l0_g: float
    mean_l0_ghat: float
    cancellation_index: float = 0.0  # sum(neg contributions)/sum(pos), entries with ĝ>τ_store
    subzero_rate: float = 0.0  # fraction of entries with Bz < 0 (clipped up to 0)
    rectification_rate: float = 0.0  # Rung 1R: fraction of entries with Bz - b < 0
    hurdle_collisions_per_tok: float = 0.0  # Rung 1R: g in (τ_store, τ_eval] and ĝ = 0


def evaluate_b(
    b: torch.Tensor,
    batches,  # iterator of [T_b, A] held-out tensors
    per_atom_mean: torch.Tensor,
    tau_store: float,
    tau_eval: float,
    code_eps: tuple[float, ...],
    z_steps: int,
    w_fn: float = 1.0,
    signed_z: bool = False,
    bias: torch.Tensor | None = None,
    dust_out: dict | None = None,
) -> HeldOutMetrics:
    """The four-metric suite, accumulated over held-out batches (codes re-solved).

    Also accumulates the cancellation diagnostics (identically 0 for nonneg B and z —
    a free regression check): per entry, pos = Σ_k max(B_ck z_k, 0) and
    neg = Σ_k max(−B_ck z_k, 0); the index is Σneg/Σpos over entries with ĝ > τ_store.

    Rectified (bias not None): additionally accumulates the rectification rate, the
    hurdle-collision count (true g in (τ_store, τ_eval] driven to exactly 0), and —
    when `dust_out` is passed — the per-atom dust statistics (§3.4.4): mean/variance
    over tokens of the incoming contribution (zBᵀ)_a on entries where g <= τ_store.
    """
    dev = b.device
    resid = torch.zeros((), dtype=torch.float64, device=dev)
    base = torch.zeros((), dtype=torch.float64, device=dev)
    n_imp = torch.zeros((), dtype=torch.float64, device=dev)
    n_hit = torch.zeros((), dtype=torch.float64, device=dev)
    mass_imp = torch.zeros((), dtype=torch.float64, device=dev)
    mass_hit = torch.zeros((), dtype=torch.float64, device=dev)
    l0_g = torch.zeros((), dtype=torch.float64, device=dev)
    l0_gh = torch.zeros((), dtype=torch.float64, device=dev)
    code_l0 = {str(e): torch.zeros((), dtype=torch.float64, device=dev) for e in code_eps}
    clipped = torch.zeros((), dtype=torch.float64, device=dev)
    pos_mass = torch.zeros((), dtype=torch.float64, device=dev)
    neg_mass = torch.zeros((), dtype=torch.float64, device=dev)
    subzero = torch.zeros((), dtype=torch.float64, device=dev)
    rectified_n = torch.zeros((), dtype=torch.float64, device=dev)
    collisions = torch.zeros((), dtype=torch.float64, device=dev)
    n_atoms = b.shape[0]
    dust_sum = torch.zeros(n_atoms, dtype=torch.float64, device=dev)
    dust_sq = torch.zeros(n_atoms, dtype=torch.float64, device=dev)
    dust_n = torch.zeros(n_atoms, dtype=torch.float64, device=dev)
    n_rows = 0
    has_neg = bool((b < 0).any()) or signed_z
    b_pos, b_negm = (b.clamp_min(0.0), (-b).clamp_min(0.0)) if has_neg else (None, None)

    for g in batches:
        z = solve_codes(g, b, z_steps, w_fn, signed_z=signed_z, bias=bias)
        bz = z @ b.T
        score = bz if bias is None else bz - bias
        ghat = score.clamp(0.0, 1.0)
        if bias is not None:
            rectified_n += (score < 0).sum(dtype=torch.float64)
            collisions += (
                (g > tau_store) & (g <= tau_eval) & (ghat == 0)
            ).sum(dtype=torch.float64)
            if dust_out is not None:
                dust_mask = (g <= tau_store).to(bz.dtype)
                dust_sum += (bz * dust_mask).sum(dim=0, dtype=torch.float64)
                dust_sq += (bz * bz * dust_mask).sum(dim=0, dtype=torch.float64)
                dust_n += dust_mask.sum(dim=0, dtype=torch.float64)
        if has_neg:
            z_pos, z_negm = z.clamp_min(0.0), (-z).clamp_min(0.0)
            pos = z_pos @ b_pos.T + z_negm @ b_negm.T
            neg = z_pos @ b_negm.T + z_negm @ b_pos.T
            live = ghat > tau_store
            pos_mass += pos[live].sum(dtype=torch.float64)
            neg_mass += neg[live].sum(dtype=torch.float64)
            subzero += (bz < 0).sum(dtype=torch.float64)
        resid += ((g - ghat) ** 2).sum(dtype=torch.float64)
        base += ((g - per_atom_mean) ** 2).sum(dtype=torch.float64)
        imp = g > tau_eval
        n_imp += imp.sum(dtype=torch.float64)
        n_hit += (imp & (ghat > tau_eval)).sum(dtype=torch.float64)
        mass_imp += g[imp].sum(dtype=torch.float64)
        mass_hit += (g * (imp & (ghat > tau_eval))).sum(dtype=torch.float64)
        l0_g += (g > tau_store).sum(dtype=torch.float64)
        l0_gh += (ghat > tau_store).sum(dtype=torch.float64)
        for e in code_eps:
            code_l0[str(e)] += (z.abs() > e).sum(dtype=torch.float64)  # |z|: signed-safe
        clipped += (score > 1.0).sum(dtype=torch.float64)
        n_rows += g.shape[0]

    if dust_out is not None and bias is not None:
        mean = (dust_sum / dust_n.clamp_min(1)).cpu().numpy()
        var = (dust_sq / dust_n.clamp_min(1)).cpu().numpy() - mean**2
        dust_out["dust_mean"] = mean
        dust_out["dust_var"] = var.clip(min=0)
        dust_out["dust_n"] = dust_n.cpu().numpy()

    n_cells = n_rows * b.shape[0]
    return HeldOutMetrics(
        ev=float(1.0 - (resid / base).item()),
        recall=float((n_hit / n_imp.clamp_min(1)).item()),
        recall_weighted=float((mass_hit / mass_imp.clamp_min(1e-9)).item()),
        induced_l0_ratio=float((l0_gh / l0_g.clamp_min(1)).item()),
        code_l0={e: float(v.item() / n_rows) for e, v in code_l0.items()},
        clip_rate=float((clipped / n_cells).item()),
        mean_l0_g=float(l0_g.item() / n_rows),
        mean_l0_ghat=float(l0_gh.item() / n_rows),
        cancellation_index=float((neg_mass / pos_mass.clamp_min(1e-9)).item()),
        subzero_rate=float((subzero / n_cells).item()),
        rectification_rate=float((rectified_n / n_cells).item()),
        hurdle_collisions_per_tok=float((collisions / max(n_rows, 1)).item()),
    )
