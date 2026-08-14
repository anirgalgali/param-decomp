"""R0 (Rung 1R §2): the pruning diagnostic — how much of the nonneg fog is removable dust?

No fitting. Takes the trained sym K=512 dictionary, zeroes every B entry below
δ·(column max) for δ in a fixed grid, re-solves held-out codes against the pruned B,
and recomputes the four Rung-1 metrics. The frontier bounds what ANY static cleaning
can achieve and calibrates the rectified fit's prospects (the doc's prediction 1).

Usage: python -m posthoc_ci.prune_b [--k 512] [--seed 0] [--arm sym]
"""

import argparse
import json

import numpy as np
import torch

from posthoc_ci import constants, glib, paths
from posthoc_ci.figstyle import SERIES, apply_style, save
from posthoc_ci.nmf import evaluate_b

DELTAS = (0.01, 0.02, 0.05, 0.1, 0.2)

# context points for the frontier figure (from prior memos; recall_w, induced-L0)
CONTEXT = {
    "sym K=512": (0.943, 8.60),
    "sgnB K=512": (0.992, 1.66),
    "MDL OR-broadcast": (1.0, 5.85),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", default="sym")
    args = parser.parse_args()
    apply_style()
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = torch.from_numpy(
        np.load(paths.fit_dir(args.k, args.seed, args.arm) / "final.npz")["B"]
    ).to(device)
    per_atom_mean = torch.from_numpy(glib.per_atom_mean("train")).to(device)
    col_max = b.max(dim=0).values.clamp_min(1e-9)

    rows = []
    for delta in DELTAS:
        pruned = b * (b >= delta * col_max)
        kept = float((pruned > 0).float().mean().item())
        m = evaluate_b(
            pruned,
            glib.dense_row_batches("held", constants.HARVEST_SHARD_POSITIONS, device),
            per_atom_mean,
            constants.TAU_STORE,
            constants.TAU_EVAL,
            constants.CODE_L0_EPSILONS,
            constants.Z_SOLVER_N_STEPS,
        )
        rows.append({"delta": delta, "frac_entries_kept": kept, **vars(m)})
        print(f"δ={delta}: kept {kept:.3f}  EV {m.ev:.3f}  w-rec {m.recall_weighted:.3f} "
              f"indL0 {m.induced_l0_ratio:.2f}")

    out_dir = paths.POSTHOC_ROOT / "rectified"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r0_pruning.json").write_text(json.dumps(
        {"k": args.k, "seed": args.seed, "arm": args.arm, "rows": rows}, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["recall_weighted"] for r in rows], [r["induced_l0_ratio"] for r in rows],
            "o-", color=SERIES[0], label="pruned sym (δ sweep)")
    for i, r in enumerate(rows):
        ax.annotate(f"δ={r['delta']}", (r["recall_weighted"], r["induced_l0_ratio"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    for j, (name, (rw, il)) in enumerate(CONTEXT.items()):
        ax.scatter([rw], [il], marker="D", color=SERIES[1 + j], zorder=5, label=name)
    ax.axhline(2.0, color=SERIES[3], lw=1, ls="--")
    ax.set(xlabel="weighted recall@0.1 (held-out)", ylabel="induced-L0 ratio",
           title="R0: static pruning frontier of the sym dictionary")
    ax.legend(fontsize=8)
    save(fig, glib.fig_path("r0_pruning_frontier.png"))
    print(f"-> {out_dir / 'r0_pruning.json'}")


if __name__ == "__main__":
    main()
