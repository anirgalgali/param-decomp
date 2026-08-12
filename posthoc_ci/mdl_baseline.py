"""E1.4b: the paper's MDL clustering as a reconstruction baseline, two ways.

Consumes a `history.zip` produced by the repo's own clustering pipeline (prior art,
zero reimplementation). Run that first on the pod:

    uv run --no-sync python -m param_decomp_lab.clustering.scripts.run_harvest \
        param_decomp_lab/clustering/configs/crc/pile_llama_simple_mlp-4L.json
    uv run --no-sync python -m param_decomp_lab.clustering.scripts.run_merge \
        --snapshot <harvest_out> --config <same json>

Then this module selects the merge iteration whose group count is nearest --k and
evaluates on OUR held-out shards:
  (a) OR-broadcast (the paper-native reading): cluster bit = OR of members' binarized
      CI; every member's floor = its cluster's bit. Perfect recall by construction —
      the informative numbers are induced-L0 and code-L0.
  (b) partition-constrained factorization (amendment 5): B frozen to the binary
      membership matrix, graded z fitted with the E1.3 solver — the fair matched
      comparison isolating what overlap/graded membership adds.

Usage: python -m posthoc_ci.mdl_baseline --history <path>/history.zip --k 512
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from posthoc_ci import constants, glib, paths
from posthoc_ci.nmf import evaluate_b


def _membership_matrix(history_path: Path, target_k: int) -> tuple[np.ndarray, int]:
    """Binary [n_alive_ours, k_groups] membership from the merge iteration nearest target_k."""
    from param_decomp_lab.clustering.merge_history import MergeHistory

    hist = MergeHistory.read(history_path)
    k_groups = hist.merges.k_groups.cpu().numpy()
    it = int(np.argmin(np.abs(k_groups - target_k)))
    k_at_it = int(k_groups[it])
    group_idxs = hist.merges.group_idxs[it].cpu().numpy()  # [n_components_theirs]

    atom_df = glib.load_atom_index()
    alive = atom_df[atom_df.alive].reset_index(drop=True)
    key_to_pos = {
        (m, int(c)): i for i, (m, c) in enumerate(zip(alive.module, alive.c, strict=True))
    }

    b = np.zeros((len(alive), k_at_it), dtype=np.float32)
    n_matched, n_unmatched = 0, 0
    for comp_idx, label in enumerate(hist.labels):
        module, c = label.rsplit(":", 1)
        pos = key_to_pos.get((module, int(c)))
        if pos is None:
            n_unmatched += 1  # their aliveness filter differs slightly from ours
            continue
        b[pos, group_idxs[comp_idx]] = 1.0
        n_matched += 1
    print(f"iteration {it}: k_groups={k_at_it}, matched {n_matched} atoms "
          f"({n_unmatched} unmatched across aliveness filters)")
    return b, k_at_it


def _or_broadcast_metrics(b: np.ndarray, device: str) -> dict:
    """Cluster bit = OR of members' binarized CI; member floor = cluster bit."""
    bt = torch.from_numpy(b).to(device)  # [A, K] binary
    member = bt > 0
    l0_g = l0_gh = code_l0 = n_imp = n_hit = 0.0
    n_rows = 0
    for g in glib.dense_row_batches("held", constants.HARVEST_SHARD_POSITIONS, device):
        fired = (g > constants.TAU_EVAL).float()
        cluster_bit = (fired @ bt).clamp(0, 1)  # [T, K]
        ghat = (cluster_bit @ member.float().T).clamp(0, 1)  # [T, A] binary
        imp = g > constants.TAU_EVAL
        n_imp += float(imp.sum())
        n_hit += float((imp & (ghat > constants.TAU_EVAL)).sum())
        l0_g += float((g > constants.TAU_STORE).sum())
        l0_gh += float((ghat > constants.TAU_STORE).sum())
        code_l0 += float((cluster_bit > 0).sum())
        n_rows += g.shape[0]
    return {
        "recall_at_tau_eval": n_hit / max(n_imp, 1),
        "induced_l0_ratio": l0_gh / max(l0_g, 1),
        "mean_code_l0": code_l0 / n_rows,
        "mean_l0_ghat": l0_gh / n_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b_np, k_at_it = _membership_matrix(args.history, args.k)

    or_metrics = _or_broadcast_metrics(b_np, device)
    print("OR-broadcast:", json.dumps(or_metrics, indent=2))

    per_atom_mean = torch.from_numpy(glib.per_atom_mean("train")).to(device)
    fitted = evaluate_b(
        torch.from_numpy(b_np).to(device),
        glib.dense_row_batches("held", constants.HARVEST_SHARD_POSITIONS, device),
        per_atom_mean,
        constants.TAU_STORE,
        constants.TAU_EVAL,
        constants.CODE_L0_EPSILONS,
        constants.Z_SOLVER_N_STEPS,
    )
    print("partition-constrained factorization:", json.dumps(vars(fitted), indent=2))

    paths.BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.BASELINES_DIR / f"mdl_K{k_at_it}.json"
    out.write_text(json.dumps(
        {"target_k": args.k, "k_groups": k_at_it,
         "or_broadcast": or_metrics, "partition_constrained_fit": vars(fitted)}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
