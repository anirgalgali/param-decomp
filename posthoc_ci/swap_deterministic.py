"""E2.1: the deterministic masking-mode table, true g vs ĝ, with the induced-L0 column.

Rows: unmasked / CI-as-masks / rounded@{TAU_STORE, 0.1, 0.5} / zero (normalizer).
Per row and condition: CE difference vs target, KL-to-target, per-row mask L0, plus a
sequence-level bootstrap CI on the KL (amendment 6). TAU_STORE stands in for ">0" on ĝ
(pitfall #6, captioned in the report).

Conditions come from --conditions: comma-separated among
  g            true floors (the recomputed reference — never transcribed)
  sym | asym<w>  ĝ from the (K, seed) fit of that arm
  shuffled     row-shuffled B control (E2.4.1 deterministic half)
  binarized    ĝ thresholded at TAU_EVAL into {0,1} floors (E2.4.3)

Usage:
  python -m posthoc_ci.swap_deterministic --k 512 --conditions g,sym
"""

import argparse
import json

import numpy as np
import torch

from posthoc_ci import constants, paths, swap_lib
from posthoc_ci.controls import shuffled_b

N_BOOT = 1000
ROWS = ("unmasked", "ci", "rounded_store", "rounded_01", "rounded_05", "zero")


def _row_forward(run, tokens_mb, ci, row):
    if row == "unmasked":
        return swap_lib.masked_forward(run, tokens_mb, ci, "unmasked")
    if row == "ci":
        return swap_lib.masked_forward(run, tokens_mb, ci, "ci")
    if row == "rounded_store":
        return swap_lib.masked_forward(run, tokens_mb, ci, "rounded", constants.TAU_STORE)
    if row == "rounded_01":
        return swap_lib.masked_forward(run, tokens_mb, ci, "rounded", 0.1)
    if row == "rounded_05":
        return swap_lib.masked_forward(run, tokens_mb, ci, "rounded", 0.5)
    if row == "zero":
        return swap_lib.masked_forward(run, tokens_mb, ci, "zero")
    raise ValueError(row)


def cond_arm(cond: str) -> str | None:
    """The fit arm a condition's B comes from; None for the true-g reference.

    `shuffled` / `shuffled-<arm>` shuffles that arm's B (bare form = sym, the Rung-2
    convention); `binarized` / `covering` are sym-based variants.
    """
    if cond == "g":
        return None
    if cond.startswith("shuffled"):
        rest = cond.removeprefix("shuffled").removeprefix("-")
        return rest or "sym"
    if cond in ("binarized", "covering"):
        return "sym"
    return cond


def _condition_ci(run, ci_true, b, cond):
    if cond == "g":
        return swap_lib.ghat_dict(run, ci_true, b, identity=True)[0]
    if cond == "binarized":
        return swap_lib.ghat_dict(run, ci_true, b, binarize_at=constants.TAU_EVAL)[0]
    if cond == "covering":
        return swap_lib.ghat_dict(run, ci_true, b, covering=True)[0]
    arm = cond_arm(cond)
    assert arm is not None
    return swap_lib.ghat_dict(run, ci_true, b, **swap_lib.arm_solver_params(arm))[0]


def _bootstrap_ci(per_seq: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(per_seq)
    means = per_seq[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="/workspace/out-torch/runs/s-55ea3f9b")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", default="g,sym")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    run = swap_lib.load_eval_run(args.run_dir)
    conditions = args.conditions.split(",")

    def load_b(cond):
        arm = cond_arm(cond)
        if arm is None:
            return None
        b = torch.from_numpy(
            np.load(paths.fit_dir(args.k, args.seed, arm) / "final.npz")["B"]
        ).to(run.device)
        if cond.startswith("shuffled"):
            b = shuffled_b(b.cpu(), args.seed).to(run.device)
        return b

    b_by_cond = {c: load_b(c) for c in conditions}
    n_seqs = run.tokens.shape[0]
    results = {
        c: {row: {"kl_per_seq": np.zeros(n_seqs), "ce_sum": 0.0, "ce_n": 0, "mask_l0": 0.0}
            for row in ROWS}
        for c in conditions
    }
    floors_l0 = {c: 0.0 for c in conditions}
    target_ce_sum, target_ce_n, n_pos_total = 0.0, 0, 0

    for mb in swap_lib.micro_slices():
        tokens_mb = run.tokens[mb]
        ci_true = swap_lib.true_ci(run, tokens_mb)
        tgt = swap_lib.target_logits(run, tokens_mb)
        ce_t, n_t = swap_lib.ce_sum(tgt, tokens_mb)
        target_ce_sum += ce_t
        target_ce_n += n_t
        n_pos = tokens_mb.numel()
        n_pos_total += n_pos

        for cond in conditions:
            ci = _condition_ci(run, ci_true, b_by_cond[cond], cond)
            floors_l0[cond] += swap_lib.induced_l0(run, ci, constants.TAU_STORE) * n_pos
            for row in ROWS:
                logits = _row_forward(run, tokens_mb, ci, row)
                kl = swap_lib.per_position_kl(logits, tgt)  # [B, S]
                results[cond][row]["kl_per_seq"][mb] += kl.mean(dim=1).cpu().numpy()
                ce, n_ce = swap_lib.ce_sum(logits, tokens_mb)
                results[cond][row]["ce_sum"] += ce
                results[cond][row]["ce_n"] += n_ce
                flat = swap_lib.ci_dict_to_flat(run, ci)
                thr = {"rounded_store": constants.TAU_STORE, "rounded_01": 0.1,
                       "rounded_05": 0.5}.get(row)
                if thr is not None:
                    results[cond][row]["mask_l0"] += float(
                        (flat > thr).float().sum(-1).mean().item()) * n_pos

    target_ce = target_ce_sum / target_ce_n
    table = {}
    for cond in conditions:
        table[cond] = {"floors_l0_at_tau_store": floors_l0[cond] / n_pos_total}
        for row in ROWS:
            r = results[cond][row]
            lo, hi = _bootstrap_ci(r["kl_per_seq"])
            table[cond][row] = {
                "kl": float(r["kl_per_seq"].mean()),
                "kl_ci95": [lo, hi],
                "ce_difference": r["ce_sum"] / r["ce_n"] - target_ce,
                "mask_l0": (r["mask_l0"] / n_pos_total) if r["mask_l0"] else None,
            }

    # Dual-band reporting (Rung 1S): appendix-B convention — ratio of excess KL over the
    # unmasked(ones) baseline, decoded vs real floors, deterministic (CI-as-masks) family.
    if "g" in table:
        g_excess = table["g"]["ci"]["kl"] - table["g"]["unmasked"]["kl"]
        for cond in conditions:
            if cond == "g":
                continue
            excess = table[cond]["ci"]["kl"] - table[cond]["unmasked"]["kl"]
            table[cond]["harry_excess_ratio_det"] = excess / max(g_excess, 1e-9)

    report = {
        "k": args.k, "seed": args.seed, "conditions": conditions,
        "target_ce": target_ce,
        "note": "rounded_store stands in for '>0' on ĝ (NMF output is rarely exactly 0)",
        "table": table,
    }
    paths.SUBST_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "_".join(conditions)
    out = paths.SUBST_DIR / f"deterministic_K{args.k}_seed{args.seed}_{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
