"""E1.4c: grouping controls — row-shuffled trained B and fully-random B.

Row shuffling permutes atom identities while preserving each column's structure; the
codes are re-solved on held-out data with the shuffled B frozen. The gap between
trained and shuffled B at matched K is the part of reconstruction attributable to the
*learned* grouping rather than to the solver's freedom to fit codes. The fully-random
variant additionally destroys column structure (matched per-column sparsity, values
resampled from the column's own nonzeros).

Usage: python -m posthoc_ci.controls --k 512 --seed 0 [--arm sym]
"""

import argparse
import json

import numpy as np
import torch

from posthoc_ci import constants, glib, paths
from posthoc_ci.nmf import evaluate_b


def shuffled_b(b: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed + 1000)
    perm = torch.randperm(b.shape[0], generator=gen)
    return b[perm]


def random_b(b: torch.Tensor, seed: int) -> torch.Tensor:
    """Random nonneg B with each column's sparsity and value distribution matched."""
    gen = torch.Generator().manual_seed(seed + 2000)
    out = torch.zeros_like(b)
    eps = 1e-4
    for k in range(b.shape[1]):
        col = b[:, k]
        nz = col[col > eps]
        if len(nz) == 0:
            continue
        rows = torch.randperm(b.shape[0], generator=gen)[: len(nz)]
        vals = nz[torch.randint(0, len(nz), (len(nz),), generator=gen)]
        out[rows, k] = vals
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", default="sym")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit = paths.fit_dir(args.k, args.seed, args.arm)
    b = torch.from_numpy(np.load(fit / "final.npz")["B"]).to(device)
    per_atom_mean = torch.from_numpy(glib.per_atom_mean("train")).to(device)

    results = {}
    for name, mat in (
        ("trained", b),
        ("shuffled", shuffled_b(b.cpu(), args.seed).to(device)),
        ("random", random_b(b.cpu(), args.seed).to(device)),
    ):
        m = evaluate_b(
            mat,
            glib.dense_row_batches("held", constants.HARVEST_SHARD_POSITIONS, device),
            per_atom_mean,
            constants.TAU_STORE,
            constants.TAU_EVAL,
            constants.CODE_L0_EPSILONS,
            constants.Z_SOLVER_N_STEPS,
        )
        results[name] = vars(m)
        print(f"{name}: ev={m.ev:.3f} recall_w={m.recall_weighted:.3f} "
              f"inducedL0={m.induced_l0_ratio:.2f}")

    paths.BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.BASELINES_DIR / f"controls_K{args.k}_seed{args.seed}_{args.arm}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
