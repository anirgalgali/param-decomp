"""E1.3 CLI: fit one (K, seed, arm) cell, or aggregate all cells into the
compressibility figure + knee determination (Gate B).

Fit one cell (GPU):
    python -m posthoc_ci.fit_sweep --k 512 --seed 0                  # symmetric arm
    python -m posthoc_ci.fit_sweep --k 512 --seed 0 --w-fn 3         # asymmetric arm
Aggregate (CPU, after the grid is done):
    python -m posthoc_ci.fit_sweep --aggregate

Each cell writes fits/K<k>_seed<s>_<arm>/{final.npz,metrics.json}. The K=0 per-atom-mean
reference is computed inside --aggregate (no fit involved). Knee rule and Gate B bounds
come from constants.py.
"""

import argparse
import json

import numpy as np
import torch

from posthoc_ci import constants, glib, paths
from posthoc_ci.figstyle import SERIES, apply_style, save
from posthoc_ci.nmf import FitConfig, HeldOutMetrics, evaluate_b, fit_b


def _arm_name(
    w_fn: float,
    signed_b: bool = False,
    signed_z: bool = False,
    rectified: bool = False,
    warm: bool = False,
) -> str:
    if rectified:
        assert not signed_b and not signed_z
        base = "rect-warm" if warm else "rect"
        if w_fn != 1.0:
            base += f"-asym{w_fn:g}"
        return base
    if signed_z:
        assert signed_b, "signed z without signed B is not a studied arm"
        base = "sgnBZ"
    elif signed_b:
        base = "sgnB"
    else:
        base = "sym" if w_fn == 1.0 else f"asym{w_fn:g}"
    if signed_b and w_fn != 1.0:
        base += f"-asym{w_fn:g}"
    return base


def _batches(split: str, device: str, seed: int | None = None):
    def gen():
        return glib.dense_row_batches(split, constants.HARVEST_SHARD_POSITIONS, device, seed)

    return gen


def fit_one(
    k: int,
    seed: int,
    w_fn: float,
    no_clip: bool,
    epochs: int,
    signed_b: bool = False,
    signed_z: bool = False,
    rectified: bool = False,
    init_from: str | None = None,
    freeze_b_epochs: int = 1,
) -> None:
    device = "cuda"
    torch.manual_seed(seed)
    arm = _arm_name(w_fn, signed_b, signed_z, rectified, warm=init_from is not None)
    cfg = FitConfig(
        k=k, seed=seed, w_fn=w_fn, epochs=epochs, no_clip=no_clip,
        signed_b=signed_b, signed_z=signed_z,
        rectified=rectified, freeze_b_epochs=freeze_b_epochs,
    )
    out_dir = paths.fit_dir(k, seed, arm)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_atom_mean = torch.from_numpy(glib.per_atom_mean("train")).to(device)
    b_init = None
    if init_from is not None:
        b_init = torch.from_numpy(
            np.load(paths.fit_dir(k, seed, init_from) / "final.npz")["B"]
        ).to(device)
        if rectified:
            # §3.2 warm anchor: the rectified path with b=0 must reproduce the
            # init arm's row exactly — the Rung-1R nested-family identity test.
            anchor = evaluate_b(
                b_init, _batches("held", device)(), per_atom_mean,
                constants.TAU_STORE, constants.TAU_EVAL, constants.CODE_L0_EPSILONS,
                constants.Z_SOLVER_N_STEPS, w_fn=1.0,
                bias=torch.zeros(b_init.shape[0], device=device),
            )
            ref = json.loads(
                (paths.fit_dir(k, seed, init_from) / "metrics.json").read_text()
            )
            drift = abs(anchor.ev - ref["ev"])
            (out_dir / "warm_anchor.json").write_text(json.dumps(
                {"anchor": vars(anchor), "reference": {m: ref[m] for m in
                 ("ev", "recall_weighted", "induced_l0_ratio")}, "ev_drift": drift},
                indent=2, default=float))
            assert drift < 0.01, f"warm anchor drifted from {init_from}: ΔEV={drift}"
            print(f"warm anchor OK: EV {anchor.ev:.4f} vs {ref['ev']:.4f} (b=0 == {init_from})")

    g_sample = next(_batches("train", device, seed=seed)())
    b, bias = fit_b(cfg, _batches("train", device, seed=seed), g_sample, b_init=b_init)

    dust_out: dict = {}
    metrics = evaluate_b(
        b,
        _batches("held", device)(),
        per_atom_mean,
        constants.TAU_STORE,
        constants.TAU_EVAL,
        constants.CODE_L0_EPSILONS,
        constants.Z_SOLVER_N_STEPS,
        w_fn=w_fn,
        signed_z=signed_z,
        bias=bias,
        dust_out=dust_out if rectified else None,
    )
    arrays = {"B": b.cpu().numpy().astype(np.float32)}
    if bias is not None:
        arrays["b"] = bias.cpu().numpy().astype(np.float32)
    np.savez_compressed(out_dir / "final.npz", **arrays)
    if dust_out:
        np.savez_compressed(out_dir / "dust.npz", **dust_out)
    record = {
        "k": k, "seed": seed, "arm": arm, "no_clip": no_clip,
        "signed_b": signed_b, "signed_z": signed_z, "rectified": rectified,
        **vars(metrics),
    }
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


def _mean_baseline_metrics(device: str) -> HeldOutMetrics:
    """K=0 reference: ghat = per-atom train mean, broadcast to every token."""
    mean = torch.from_numpy(glib.per_atom_mean("train")).to(device)
    resid = base = n_imp = n_hit = mass_imp = mass_hit = l0_g = l0_gh = 0.0
    n_rows = 0
    for g in glib.dense_row_batches("held", constants.HARVEST_SHARD_POSITIONS, device):
        ghat = mean.expand_as(g)
        resid += float(((g - ghat) ** 2).sum())
        base += float(((g - mean) ** 2).sum())
        imp = g > constants.TAU_EVAL
        hit = imp & (ghat > constants.TAU_EVAL)
        n_imp += float(imp.sum())
        n_hit += float(hit.sum())
        mass_imp += float(g[imp].sum())
        mass_hit += float((g * hit).sum())
        l0_g += float((g > constants.TAU_STORE).sum())
        l0_gh += float((ghat > constants.TAU_STORE).sum())
        n_rows += g.shape[0]
    return HeldOutMetrics(
        ev=1.0 - resid / base,
        recall=n_hit / max(n_imp, 1),
        recall_weighted=mass_hit / max(mass_imp, 1e-9),
        induced_l0_ratio=l0_gh / max(l0_g, 1),
        code_l0={str(e): 0.0 for e in constants.CODE_L0_EPSILONS},
        clip_rate=0.0,
        mean_l0_g=l0_g / n_rows,
        mean_l0_ghat=l0_gh / n_rows,
    )


def aggregate() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    records = []
    for mfile in sorted(paths.FITS_DIR.glob("K*_seed*_*/metrics.json")):
        records.append(json.loads(mfile.read_text()))
    assert records, "no fit metrics found"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean_ref = _mean_baseline_metrics(device)

    panels = [
        ("ev", "explained variance", "EV (fixed def., held-out)"),
        ("recall_weighted", "CI-mass-weighted recall@0.1", "weighted recall@τ_eval"),
        ("induced_l0_ratio", "induced-L0 ratio", "L0(ĝ)/L0(g) @ τ_store"),
        ("code_l0", "code L0 (ε=0.01)", "per-token code L0"),
    ]
    arms = sorted({r["arm"] for r in records})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    knee: dict[str, int | None] = {}
    for arm_i, arm in enumerate(arms):
        arm_recs = [r for r in records if r["arm"] == arm]
        ks = sorted({r["k"] for r in arm_recs})
        for ax, (field, ylabel, title) in zip(axes.ravel(), panels, strict=True):
            means, lo, hi = [], [], []
            for k in ks:
                vals = [
                    r[field]["0.01"] if field == "code_l0" else r[field]
                    for r in arm_recs
                    if r["k"] == k
                ]
                means.append(np.mean(vals))
                lo.append(np.min(vals))
                hi.append(np.max(vals))
            color = SERIES[arm_i % len(SERIES)]
            ax.plot(ks, means, "o-", color=color, label=arm)
            ax.fill_between(ks, lo, hi, color=color, alpha=0.2, lw=0)
            ax.set(xscale="log", xlabel="K", ylabel=ylabel, title=title)

        # knee per arm: smallest K meeting both bars (seed-mean)
        knee[arm] = None
        for k in ks:
            recs_k = [r for r in arm_recs if r["k"] == k]
            wr = np.mean([r["recall_weighted"] for r in recs_k])
            il = np.mean([r["induced_l0_ratio"] for r in recs_k])
            if wr >= constants.KNEE_WEIGHTED_RECALL_MIN and il <= constants.KNEE_INDUCED_L0_RATIO_MAX:
                knee[arm] = k
                break

    axes[0, 0].axhline(mean_ref.ev, color="#52514e", lw=1, ls=":")
    axes[0, 1].axhline(mean_ref.recall_weighted, color="#52514e", lw=1, ls=":")
    axes[0, 1].axhline(constants.KNEE_WEIGHTED_RECALL_MIN, color=SERIES[3], lw=1, ls="--")
    axes[1, 0].axhline(constants.KNEE_INDUCED_L0_RATIO_MAX, color=SERIES[3], lw=1, ls="--")
    for ax in axes.ravel():
        ax.legend(fontsize=8)

    gate_b = {
        arm: (k is not None and k <= constants.GATE_B_K_MAX) for arm, k in knee.items()
    }
    fig.suptitle(
        f"E1.3 compressibility — knee: {knee}   Gate B pass: {gate_b}", fontsize=11
    )
    save(fig, glib.fig_path("e13_compressibility.png"))

    summary = {
        "knee": knee,
        "gate_b_pass": gate_b,
        "k0_mean_reference": vars(mean_ref),
        "records": records,
    }
    (paths.FITS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w-fn", type=float, default=1.0)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--signed-b", action="store_true")
    parser.add_argument("--signed-z", action="store_true")
    parser.add_argument("--rectified", action="store_true")
    parser.add_argument("--init-from", default=None,
                        help="arm name whose fit at (k, seed) seeds B (warm arm)")
    parser.add_argument("--freeze-b-epochs", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
    else:
        assert args.k is not None, "--k required unless --aggregate"
        fit_one(args.k, args.seed, args.w_fn, args.no_clip, args.epochs,
                args.signed_b, args.signed_z, args.rectified, args.init_from,
                args.freeze_b_epochs)


if __name__ == "__main__":
    main()
