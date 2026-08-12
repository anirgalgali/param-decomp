"""E1.2: spectral pre-check (Gate A) — the cheap upper bound on any factorization.

Randomized truncated SVD (torch.svd_lowrank, GPU) of the train-split G, graded AND
binarized at TAU_EVAL. EV uses the one fixed definition (constants/amendment 8):
EV(K) = 1 - (||G||^2 - sum_{i<=K} s_i^2) / ||G - per-atom-mean||^2.

Gate A: stop iff 90% EV needs rank > 0.5 x alive on BOTH variants. With a rank-Q
spectrum (Q=2048) the 90% point may lie beyond Q; in that case a power-law tail fit
estimates it and the figure says so explicitly — the gate call is made by a human.

Usage: python -m posthoc_ci.svd_gate_a [--q 2048]
"""

import argparse
import json

import numpy as np
import torch
from scipy import sparse

from posthoc_ci import constants, glib, paths
from posthoc_ci.figstyle import SERIES, apply_style, save

GATE_A_EV = 0.90
GATE_A_RANK_FRAC = 0.5


def _scipy_to_torch_csr(csr: sparse.csr_matrix, device: str) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.from_numpy(csr.indptr.astype(np.int64)),
        torch.from_numpy(csr.indices.astype(np.int64)),
        torch.from_numpy(csr.data.astype(np.float32)),
        size=csr.shape,
        device=device,
    )


def _spectrum(csr: sparse.csr_matrix, q: int, device: str, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    a = _scipy_to_torch_csr(csr, device)
    _, s, _ = torch.svd_lowrank(a, q=q, niter=4)
    return s.cpu().numpy().astype(np.float64)


def _ev_curve(s: np.ndarray, frob_sq: float, baseline_sq: float) -> np.ndarray:
    residual = frob_sq - np.cumsum(s**2)
    return 1.0 - residual / baseline_sq


def _rank_for_ev(ev: np.ndarray, s: np.ndarray, frob_sq: float, baseline_sq: float,
                 target: float) -> tuple[float, bool]:
    """Rank achieving `target` EV; power-law tail extrapolation when beyond the spectrum."""
    hit = np.flatnonzero(ev >= target)
    if len(hit):
        return float(hit[0] + 1), False
    tail = s[len(s) // 2 :]
    ranks = np.arange(len(s) // 2 + 1, len(s) + 1)
    slope, intercept = np.polyfit(np.log(ranks), np.log(tail), 1)
    need = baseline_sq * (1 - target)
    have = frob_sq - np.sum(s**2)
    k = float(len(s))
    while have > need and k < 1e6:
        k += 1
        have -= np.exp(intercept + slope * np.log(k)) ** 2
    return k, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=int, default=2048)
    args = parser.parse_args()
    apply_style()
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = glib.load_split_csr("train")
    n_pos, n_alive = g.shape
    mean = glib.per_atom_mean("train").astype(np.float64)

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (label, mat) in enumerate(
        (("graded", g), (f"binarized@{constants.TAU_EVAL}", (g > constants.TAU_EVAL).astype(np.float32).tocsr()))
    ):
        frob_sq = float((mat.data.astype(np.float64) ** 2).sum())
        col_mean = np.asarray(mat.mean(axis=0)).ravel().astype(np.float64) if label != "graded" else mean
        baseline_sq = frob_sq - n_pos * float((col_mean**2).sum())
        s = _spectrum(mat, args.q, device, seed=constants.SEEDS[0])
        ev = _ev_curve(s, frob_sq, baseline_sq)
        rank90, extrapolated = _rank_for_ev(ev, s, frob_sq, baseline_sq, GATE_A_EV)
        results[label] = {
            "ev_at_512": round(float(ev[511]), 4),
            "ev_at_1024": round(float(ev[1023]), 4),
            "ev_at_q": round(float(ev[-1]), 4),
            "rank_for_90pct": rank90,
            "rank_90_extrapolated": extrapolated,
            "gate_a_fails_this_variant": bool(rank90 > GATE_A_RANK_FRAC * n_alive),
        }
        axes[0].plot(np.arange(1, len(ev) + 1), ev, color=SERIES[i], label=label)
        axes[1].plot(np.arange(1, len(s) + 1), s, color=SERIES[i], label=label)

    axes[0].axhline(GATE_A_EV, color=SERIES[3], lw=1, ls="--")
    axes[0].set(xlabel="rank K", ylabel="explained variance",
                title="Cumulative EV vs rank (train split)")
    axes[1].set(xlabel="rank", ylabel="singular value", yscale="log", xscale="log",
                title="Singular value decay")
    for ax in axes:
        ax.legend()

    gate_a_stop = all(r["gate_a_fails_this_variant"] for r in results.values())
    verdict = "GATE A: STOP (no low-rank structure)" if gate_a_stop else "GATE A: PASS — proceed"
    fig.suptitle(f"{verdict}   [alive={n_alive}, positions={n_pos}]", fontsize=11)
    save(fig, glib.fig_path("e12_svd_gate_a.png"))

    results["gate_a_verdict"] = verdict
    results["n_alive"] = int(n_alive)
    out = paths.HARVEST_DIR / "e12_gate_a.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
