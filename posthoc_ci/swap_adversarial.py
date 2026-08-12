"""E2.3 (+E2.4 adversarial cells): PGD swap-in, Gate C's strictest criterion.

Protocol matches the run's own PGDReconLoss eval exactly: sign-PGD ascent on the KL,
init random, step 0.1, 20 steps, sources shared across the full eval batch (here via
micro-batch gradient accumulation, mathematically identical). The true-g reference is
recomputed, never transcribed (pitfall #8). Conditions as in swap_deterministic.

Usage:
  python -m posthoc_ci.swap_adversarial --k 512 --conditions g,sym [--steps 20,40]
"""

import argparse
import json

import numpy as np
import torch

from posthoc_ci import constants, paths, swap_lib
from posthoc_ci.controls import shuffled_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="/workspace/out-torch/runs/s-55ea3f9b")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", default="g,sym")
    parser.add_argument("--steps", default="20")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    run = swap_lib.load_eval_run(args.run_dir)
    conditions = args.conditions.split(",")
    step_counts = [int(s) for s in args.steps.split(",")]

    def load_b(cond):
        if cond == "g":
            return None
        arm = "sym" if cond in ("sym", "shuffled", "binarized") else cond
        b = torch.from_numpy(
            np.load(paths.fit_dir(args.k, args.seed, arm) / "final.npz")["B"]
        ).to(run.device)
        return shuffled_b(b.cpu(), args.seed).to(run.device) if cond == "shuffled" else b

    # precompute per-micro-batch floors (CPU fp16) and target logits (kept on GPU: small)
    print("precomputing floors per condition...")
    slices_ = swap_lib.micro_slices()
    target_per_mb = []
    ci_by_cond: dict[str, list[dict[str, torch.Tensor]]] = {c: [] for c in conditions}
    floors_l0: dict[str, float] = {c: 0.0 for c in conditions}
    n_pos_total = 0
    for mb in slices_:
        tokens_mb = run.tokens[mb]
        ci_true = swap_lib.true_ci(run, tokens_mb)
        target_per_mb.append(swap_lib.target_logits(run, tokens_mb).to(torch.float16).cpu())
        n_pos = tokens_mb.numel()
        n_pos_total += n_pos
        for cond in conditions:
            if cond == "g":
                ci = ci_true
            elif cond == "binarized":
                ci = swap_lib.ghat_dict(run, ci_true, load_b(cond), binarize_at=constants.TAU_EVAL)[0]
            elif cond.startswith("asym"):
                ci = swap_lib.ghat_dict(run, ci_true, load_b(cond),
                                        w_fn=float(cond.removeprefix("asym")))[0]
            else:
                ci = swap_lib.ghat_dict(run, ci_true, load_b(cond))[0]
            floors_l0[cond] += swap_lib.induced_l0(run, ci, constants.TAU_STORE) * n_pos
            ci_by_cond[cond].append({k: v.to(torch.float16).cpu() for k, v in ci.items()})

    results = {}
    for cond in conditions:
        results[cond] = {"floors_l0_at_tau_store": floors_l0[cond] / n_pos_total}
        for n_steps in step_counts:
            ci_gpu = [
                {k: v.to(run.device, torch.float32) for k, v in mb_ci.items()}
                for mb_ci in ci_by_cond[cond]
            ]
            tgt_gpu = [t.to(run.device, torch.float32) for t in target_per_mb]
            kl = swap_lib.pgd_shared_kl(
                run, ci_gpu, tgt_gpu,
                n_steps=n_steps,
                step_size=constants.PGD_STEP_SIZE,
                seed=constants.RNG_BASE_KEY + n_steps,
            )
            results[cond][f"adv_kl_{n_steps}steps"] = kl
            print(f"{cond} @ {n_steps} steps: KL {kl:.4f}")
            del ci_gpu, tgt_gpu
            torch.cuda.empty_cache()

    if "g" in results:
        ref20 = results["g"].get("adv_kl_20steps")
        for cond in conditions:
            if cond != "g" and ref20:
                results[cond]["ratio_vs_g_20steps"] = results[cond]["adv_kl_20steps"] / ref20

    report = {
        "k": args.k, "seed": args.seed, "conditions": conditions,
        "protocol": {"init": constants.PGD_INIT, "step_size": constants.PGD_STEP_SIZE,
                     "mask_scope": constants.PGD_MASK_SCOPE,
                     "note": "micro-batch grad accumulation == shared_across_batch all-reduce"},
        "interpretation_guard": "if ĝ beats g, check floors_l0 first (bribery failure mode)",
        "results": results,
    }
    paths.SUBST_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "_".join(conditions)
    out = paths.SUBST_DIR / f"adversarial_K{args.k}_seed{args.seed}_{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
