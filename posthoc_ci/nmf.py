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
) -> torch.Tensor:
    """argmin_{z>=0} sum w * (g - clip(z B^T))^2 + lambda_z ||z||_1, per row of `g`.

    g: [T, A], b: [A, K] (frozen). Returns z: [T, K]. Manual projected gradient with a
    1/L step from the spectral norm of B — no autograd, cheap enough to call inside the
    alternating fit every batch.
    """
    lr = 1.0 / max(_spectral_norm_sq(b), 1e-8)
    col_sq = (b * b).sum(dim=0).clamp_min(1e-8)
    z = ((g @ b) / col_sq).clamp_min_(0.0)
    y = z.clone()
    t = 1.0
    for _ in range(n_steps):  # FISTA (accelerated projected gradient)
        bz = y @ b.T
        ghat = bz.clamp(0.0, 1.0)
        w = _loss_weights(g, ghat, w_fn)
        grad = 2.0 * ((w * (ghat - g)) @ b) + lambda_z
        z_next = (y - lr * grad).clamp_min_(0.0)
        t_next = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        y = z_next + ((t - 1.0) / t_next) * (z_next - z)
        z, t = z_next, t_next
    return z.clamp_min_(0.0)


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


def init_b(g_sample: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """Exemplar init: K random data rows as columns of B (seed-varied), floored at 0."""
    gen = torch.Generator(device=g_sample.device).manual_seed(seed)
    rows = torch.randint(0, g_sample.shape[0], (k,), generator=gen, device=g_sample.device)
    b = g_sample[rows].T.clone()
    b += 0.01 * torch.rand(b.shape, generator=gen, device=g_sample.device)
    return b.clamp_min_(0.0)


def fit_b(
    cfg: FitConfig,
    batches,  # callable () -> iterator of [T_b, A] fp32 GPU tensors (one epoch)
    g_sample: torch.Tensor,
) -> torch.Tensor:
    """Alternating fit; returns B [A, K]. `batches` is re-invoked once per epoch."""
    b = init_b(g_sample, cfg.k, cfg.seed).requires_grad_(True)
    opt = torch.optim.Adam([b], lr=cfg.lr_b)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr_b * 0.05
    )
    for epoch in range(cfg.epochs):
        epoch_loss, n_batches = 0.0, 0
        for g in batches():
            with torch.no_grad():
                z = solve_codes(g, b.detach(), cfg.z_steps, cfg.w_fn, cfg.lambda_z)
            for _ in range(cfg.b_steps_per_batch):
                bz = z @ b.T
                ghat = bz if cfg.no_clip else clip_st(bz)
                w = _loss_weights(g, ghat.detach(), cfg.w_fn)
                loss = (w * (ghat - g) ** 2).sum() / g.shape[0]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    b.clamp_min_(0.0)
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()
        print(f"epoch {epoch}: mean per-row loss {epoch_loss / max(n_batches, 1):.4f}")
    return b.detach()


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


def evaluate_b(
    b: torch.Tensor,
    batches,  # iterator of [T_b, A] held-out tensors
    per_atom_mean: torch.Tensor,
    tau_store: float,
    tau_eval: float,
    code_eps: tuple[float, ...],
    z_steps: int,
    w_fn: float = 1.0,
) -> HeldOutMetrics:
    """The four-metric suite, accumulated over held-out batches (codes re-solved)."""
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
    n_rows = 0

    for g in batches:
        z = solve_codes(g, b, z_steps, w_fn)
        bz = z @ b.T
        ghat = bz.clamp(0.0, 1.0)
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
            code_l0[str(e)] += (z > e).sum(dtype=torch.float64)
        clipped += (bz > 1.0).sum(dtype=torch.float64)
        n_rows += g.shape[0]

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
    )
