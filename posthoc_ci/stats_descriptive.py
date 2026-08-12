"""E1.1: descriptive statistics of G before any factorization.

Four analyses, all on the train split (hubs/positional lists are inputs to later
stages, so they must not peek at held-out data):
  1. per-atom firing density spectrum at both taus; hubs = density@TAU_EVAL > 0.5
  2. binarized (TAU_EVAL) coactivation heatmaps for one attn + one MLP matrix,
     ordered by single-linkage-ish spectral ordering so block structure is visible
  3. positional profile: mean CI per position bucket; positional atoms flagged
  4. cross-matrix co-usage: fraction of each atom's top-20 co-firing partners that
     live in a different weight matrix

Outputs: figures/e11_*.png, harvest/hub_atoms.json, harvest/positional_atoms.json,
harvest/e11_stats.json.

Usage: python -m posthoc_ci.stats_descriptive
"""

import json

import numpy as np
from scipy import sparse

from posthoc_ci import constants, glib, paths
from posthoc_ci.figstyle import SEQUENTIAL_CMAP, SERIES, apply_style, save

POS_BUCKETS = ((0, 1), (1, 4), (4, 16), (16, 64), (64, 512))
HUB_DENSITY = 0.5
POSITIONAL_MASS_FRAC = 0.5  # atom is 'positional' if >50% of its CI mass sits in pos 0-3


def _coactivation(g_bin: sparse.csr_matrix) -> np.ndarray:
    co = (g_bin.T @ g_bin).toarray().astype(np.float64)
    diag = np.maximum(co.diagonal(), 1.0)
    return co / diag[:, None]  # P(col fires | row fires)


def _spectral_order(cond: np.ndarray) -> np.ndarray:
    sym = (cond + cond.T) / 2
    deg = sym.sum(axis=1)
    lap = np.diag(deg) - sym
    vals, vecs = np.linalg.eigh(lap + 1e-9 * np.eye(len(lap)))
    fiedler = vecs[:, 1]
    return np.argsort(fiedler)


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    atom_df = glib.load_atom_index()
    alive = atom_df[atom_df.alive].reset_index(drop=True)
    token_meta = glib.load_token_meta()
    train_meta = token_meta[~token_meta.held_out].reset_index(drop=True)

    g = glib.load_split_csr("train")  # rows: train positions, cols: alive atoms
    n_pos, n_alive = g.shape
    print(f"train split: {n_pos} positions x {n_alive} alive atoms, nnz={g.nnz}")

    # --- 1. density spectrum ---------------------------------------------------
    dens_store = np.asarray((g > constants.TAU_STORE).sum(axis=0)).ravel() / n_pos
    dens_eval = np.asarray((g > constants.TAU_EVAL).sum(axis=0)).ravel() / n_pos
    hubs = alive.loc[dens_eval > HUB_DENSITY, ["atom_id", "module", "c"]]
    hubs = hubs.assign(density_eval=dens_eval[dens_eval > HUB_DENSITY])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.sort(dens_store)[::-1], color=SERIES[0], label=f"τ={constants.TAU_STORE}")
    ax.plot(np.sort(dens_eval)[::-1], color=SERIES[1], label=f"τ={constants.TAU_EVAL}")
    ax.axhline(HUB_DENSITY, color=SERIES[3], lw=1, ls="--")
    ax.text(n_alive * 0.55, HUB_DENSITY * 1.15, f"hub threshold ({len(hubs)} atoms)",
            fontsize=8, color="#52514e")
    ax.set(yscale="log", xlabel="alive atoms (sorted)", ylabel="firing density",
           title="Per-atom firing density spectrum (train split)")
    ax.legend()
    save(fig, glib.fig_path("e11_density_spectrum.png"))

    # --- 2. coactivation heatmaps ----------------------------------------------
    g_bin = (g > constants.TAU_EVAL).astype(np.float32).tocsr()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, module_kind, title in (
        (axes[0], "attn.k", "h.1 attention K"),
        (axes[1], "mlp.in", "h.2 MLP in"),
    ):
        layer = 1 if module_kind.startswith("attn") else 2
        sel = alive.index[(alive.matrix_type == module_kind) & (alive.layer == layer)].to_numpy()
        cond = _coactivation(g_bin[:, sel])
        order = _spectral_order(cond)
        im = ax.imshow(cond[np.ix_(order, order)], cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
        ax.set(title=f"{title} — P(col|row), spectral order", xlabel="atom", ylabel="atom")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, glib.fig_path("e11_coactivation.png"))

    # --- 3. positional profile --------------------------------------------------
    pos = train_meta.pos.to_numpy()
    bucket_mean = np.zeros((len(POS_BUCKETS), n_alive), dtype=np.float64)
    for b, (lo, hi) in enumerate(POS_BUCKETS):
        rows = np.flatnonzero((pos >= lo) & (pos < hi))
        bucket_mean[b] = np.asarray(g[rows].mean(axis=0)).ravel()
    total_mean = np.maximum(bucket_mean.mean(axis=0), 1e-12)
    early_frac = bucket_mean[:2].sum(axis=0) / (bucket_mean.sum(axis=0) + 1e-12)
    positional = alive.loc[early_frac > POSITIONAL_MASS_FRAC, ["atom_id", "module", "c"]]
    positional = positional.assign(early_mass_frac=early_frac[early_frac > POSITIONAL_MASS_FRAC])

    order = np.argsort(early_frac)[::-1][:300]
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        (bucket_mean[:, order] / total_mean[order]),
        aspect="auto", cmap=SEQUENTIAL_CMAP,
    )
    ax.set(
        yticks=range(len(POS_BUCKETS)),
        yticklabels=[f"{lo}–{hi - 1}" for lo, hi in POS_BUCKETS],
        xlabel="atoms (top 300 by early-position mass)", ylabel="position bucket",
        title=f"Positional usage profile ({len(positional)} positional atoms flagged)",
    )
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="bucket mean / overall mean")
    save(fig, glib.fig_path("e11_positional.png"))

    # --- 4. cross-matrix co-usage ------------------------------------------------
    co_counts = (g_bin.T @ g_bin).tocsr()
    matrix_of = (alive.layer.astype(str) + "." + alive.matrix_type).to_numpy()
    top_k = 20
    cross_frac = np.zeros(n_alive, dtype=np.float32)
    for i in range(n_alive):
        row = co_counts.getrow(i)
        cols, vals = row.indices, row.data
        keep = cols != i
        cols, vals = cols[keep], vals[keep]
        if len(cols) == 0:
            cross_frac[i] = np.nan
            continue
        top = cols[np.argsort(vals)[::-1][:top_k]]
        cross_frac[i] = np.mean(matrix_of[top] != matrix_of[i])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cross_frac[~np.isnan(cross_frac)], bins=40, color=SERIES[0])
    ax.set(xlabel="fraction of top-20 co-firing partners in a different weight matrix",
           ylabel="atoms",
           title="Cross-matrix co-usage teaser (E1.1.4)")
    save(fig, glib.fig_path("e11_cross_matrix.png"))

    # --- outputs ------------------------------------------------------------------
    hubs.to_json(paths.HARVEST_DIR / "hub_atoms.json", orient="records", indent=2)
    positional.to_json(paths.HARVEST_DIR / "positional_atoms.json", orient="records", indent=2)
    stats = {
        "n_train_positions": int(n_pos),
        "n_alive": int(n_alive),
        "n_hubs": int(len(hubs)),
        "hub_policy": "(i) keep hubs in G; dedicated always-on latents expected",
        "n_positional_atoms": int(len(positional)),
        "median_cross_matrix_frac": float(np.nanmedian(cross_frac)),
        "mean_l0_at_tau_eval": float((g > constants.TAU_EVAL).sum() / n_pos),
    }
    (paths.HARVEST_DIR / "e11_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
