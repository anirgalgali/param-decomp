"""E1.5R: pre-registered interpretability meters, recomputed uniformly across families.

Density-robust definitions (Rung 1R §6, applied to OLD fits too — meter-drift
pitfall §9.4):
  - members of a latent = entries with |B_ck| >= 10% of the column's |max|
  - cross-matrix latent = >= 2 weight matrices each holding >= 20% of member |mass|
  - hub latent = firing density (|z| > 0.01 over the corpus) > 0.5 — reported
    separately, excluded from the cross-matrix fraction
  - featured = top-50 non-hub latents by mean |z| (usage)
  - stability = median matched |cos| across the 3 cold seeds (Hungarian), seed 0 vs 1/2

Bars (pre-registered): stability >= 0.7; cross-matrix fraction >= 25% of non-hub
featured latents.

Usage: python -m posthoc_ci.meters_e15r --cells sym:512,sgnB-asym3:512,rect:512
"""

import argparse
import json

import numpy as np
import torch

from posthoc_ci import constants, glib, paths
from posthoc_ci.figstyle import SERIES, apply_style, save
from posthoc_ci.latent_report import _stability
from posthoc_ci.nmf import solve_codes
from posthoc_ci.swap_lib import arm_solver_params, load_fit

N_FEATURED = 50


def _usage_and_density(b, bias, signed_z, device) -> tuple[np.ndarray, np.ndarray]:
    """Mean |z| and firing density (|z| > 0.01) per latent over the full harvest."""
    k = b.shape[1]
    z_sum = torch.zeros(k, dtype=torch.float64, device=device)
    fire = torch.zeros(k, dtype=torch.float64, device=device)
    n = 0
    for _, csr in glib.iter_shards("all"):
        g = torch.from_numpy(csr.toarray()).to(device)
        z = solve_codes(g, b, constants.Z_SOLVER_N_STEPS, signed_z=signed_z, bias=bias)
        z_sum += z.abs().sum(dim=0, dtype=torch.float64)
        fire += (z.abs() > 0.01).sum(dim=0, dtype=torch.float64)
        n += z.shape[0]
    return (z_sum / n).cpu().numpy(), (fire / n).cpu().numpy()


def _cross_matrix_fraction(b: np.ndarray, usage, density, matrix_of) -> dict:
    hub = density > 0.5
    order = np.argsort(-usage)
    featured = [latent for latent in order if not hub[latent]][:N_FEATURED]
    n_cross = 0
    for latent in featured:
        col = np.abs(b[:, latent])
        members = col >= 0.1 * col.max()
        mass = col * members
        total = mass.sum() + 1e-12
        by_matrix: dict[str, float] = {}
        for m, v in zip(matrix_of[members], mass[members], strict=True):
            by_matrix[m] = by_matrix.get(m, 0.0) + float(v)
        n_big = sum(1 for v in by_matrix.values() if v / total >= 0.2)
        n_cross += int(n_big >= 2)
    return {
        "n_hub_latents": int(hub.sum()),
        "n_featured": len(featured),
        "cross_matrix_fraction": n_cross / max(len(featured), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default="sym:512,sgnB-asym3:512,rect:512",
                        help="comma-separated arm:K pairs (cold seeds 0-2 assumed)")
    args = parser.parse_args()
    apply_style()
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    atom_df = glib.load_atom_index()
    alive = atom_df[atom_df.alive].reset_index(drop=True)
    matrix_of = (alive.layer.astype(str) + "." + alive.matrix_type).to_numpy()

    results = {}
    for cell in args.cells.split(","):
        arm, k_str = cell.rsplit(":", 1)
        k = int(k_str)
        params = arm_solver_params(arm)
        b, bias = load_fit(paths.fit_dir(k, 0, arm), device)
        usage, density = _usage_and_density(b, bias, params["signed_z"], device)
        meters = _cross_matrix_fraction(b.cpu().numpy(), usage, density, matrix_of)
        sims = _stability(k, arm)
        meters["stability_median_abs_cos"] = float(np.median(sims))
        meters["bars"] = {
            "stability_pass": meters["stability_median_abs_cos"] >= 0.7,
            "cross_matrix_pass": meters["cross_matrix_fraction"] >= 0.25,
        }
        results[cell] = meters
        print(cell, json.dumps(meters, indent=2))

    out_dir = paths.POSTHOC_ROOT / "rectified"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e15r_meters.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    cells = list(results)
    for ax, field, bar, title in (
        (axes[0], "stability_median_abs_cos", 0.7, "seed stability (median |cos|)"),
        (axes[1], "cross_matrix_fraction", 0.25, "cross-matrix fraction (density-robust)"),
    ):
        vals = [results[c][field] for c in cells]
        ax.bar(range(len(cells)), vals, color=[SERIES[i % 5] for i in range(len(cells))])
        ax.axhline(bar, color=SERIES[3], lw=1, ls="--")
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(cells, fontsize=8)
        ax.set(title=title)
    fig.suptitle("E1.5R interpretability meters — one definition, three families", fontsize=11)
    save(fig, glib.fig_path("e15r_meters.png"))
    print(f"-> {out_dir / 'e15r_meters.json'}")


if __name__ == "__main__":
    main()
