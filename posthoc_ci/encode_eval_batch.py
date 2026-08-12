"""E2.0: encode the frozen eval batch and prove the swap-in harness is faithful.

Two jobs:
  1. Headline encode stats on the fixed 128x512 batch: L0(g) and L0(ĝ) at both taus,
     code L0, recall@tau_eval (plain + weighted), clip rate.
  2. The identity self-test (verification #3): substituting ĝ := g through the entire
     dict-reassembly path must reproduce the true-g CI-masked logits exactly. Guards
     the reshape/scatter/dead-atom plumbing (pitfall #7).

Usage: python -m posthoc_ci.encode_eval_batch --k <knee> [--arm sym] [--seed 0]
"""

import argparse
import json

import numpy as np
import torch

from posthoc_ci import constants, paths, swap_lib
from posthoc_ci.swap_lib import EvalRun


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="/workspace/out-torch/runs/s-55ea3f9b")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", default="sym")
    args = parser.parse_args()

    run: EvalRun = swap_lib.load_eval_run(args.run_dir)
    w_fn = 1.0 if args.arm == "sym" else float(args.arm.removeprefix("asym"))
    b = torch.from_numpy(
        np.load(paths.fit_dir(args.k, args.seed, args.arm) / "final.npz")["B"]
    ).to(run.device)

    acc = {
        "l0_g_store": 0.0, "l0_g_eval": 0.0, "l0_gh_store": 0.0, "l0_gh_eval": 0.0,
        "n_imp": 0.0, "n_hit": 0.0, "mass_imp": 0.0, "mass_hit": 0.0,
        "clip": 0.0, "code_l0": 0.0, "n_pos": 0,
    }
    max_identity_diff = 0.0

    for i, mb in enumerate(swap_lib.micro_slices()):
        tokens_mb = run.tokens[mb]
        ci = swap_lib.true_ci(run, tokens_mb)
        gh, stats = swap_lib.ghat_dict(run, ci, b, w_fn=w_fn)

        g_flat = swap_lib.ci_dict_to_flat(run, ci)
        gh_flat = swap_lib.ci_dict_to_flat(run, gh)
        n_pos = g_flat.shape[0] * g_flat.shape[1]
        acc["n_pos"] += n_pos
        acc["l0_g_store"] += float((g_flat > constants.TAU_STORE).sum())
        acc["l0_g_eval"] += float((g_flat > constants.TAU_EVAL).sum())
        acc["l0_gh_store"] += float((gh_flat > constants.TAU_STORE).sum())
        acc["l0_gh_eval"] += float((gh_flat > constants.TAU_EVAL).sum())
        imp = g_flat > constants.TAU_EVAL
        hit = imp & (gh_flat > constants.TAU_EVAL)
        acc["n_imp"] += float(imp.sum())
        acc["n_hit"] += float(hit.sum())
        acc["mass_imp"] += float(g_flat[imp].sum())
        acc["mass_hit"] += float((g_flat * hit).sum())
        acc["clip"] += stats["clip_rate"] * n_pos
        acc["code_l0"] += stats["code_l0_0.01"] * n_pos

        if i == 0:  # identity self-test on the first micro-batch
            ident, _ = swap_lib.ghat_dict(run, ci, b, identity=True)
            logits_g = swap_lib.masked_forward(run, tokens_mb, ci, "ci")
            logits_i = swap_lib.masked_forward(run, tokens_mb, ident, "ci")
            max_identity_diff = float((logits_g - logits_i).abs().max().item())

    n = acc["n_pos"]
    report = {
        "k": args.k, "seed": args.seed, "arm": args.arm,
        "mean_l0_g_at_tau_store": acc["l0_g_store"] / n,
        "mean_l0_g_at_tau_eval": acc["l0_g_eval"] / n,
        "mean_l0_ghat_at_tau_store": acc["l0_gh_store"] / n,
        "mean_l0_ghat_at_tau_eval": acc["l0_gh_eval"] / n,
        "induced_l0_ratio_store": acc["l0_gh_store"] / max(acc["l0_g_store"], 1),
        "recall_at_tau_eval": acc["n_hit"] / max(acc["n_imp"], 1),
        "weighted_recall_at_tau_eval": acc["mass_hit"] / max(acc["mass_imp"], 1e-9),
        "mean_code_l0_0.01": acc["code_l0"] / n,
        "clip_rate": acc["clip"] / n,
        "identity_self_test_max_logit_diff": max_identity_diff,
        "identity_self_test_pass": max_identity_diff < 1e-4,
    }
    assert report["identity_self_test_pass"], (
        f"identity swap-in changed logits by {max_identity_diff} — harness bug"
    )
    paths.SUBST_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.SUBST_DIR / f"encode_K{args.k}_seed{args.seed}_{args.arm}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
