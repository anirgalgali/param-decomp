"""E2.2: stochastic-mask swap-in — mean AND tail, plus dropped-atom localization.

Paired RNG: for each (seed, draw, micro-batch) the stochastic sources are re-seeded
identically for the true-g and ĝ conditions, so per-token KL differences are paired.
Reports mean / p95 / p99 per condition, overlaid log-x histograms, sequence-level
bootstrap CI on the mean, and the top-100 worst tokens by KL(ĝ)-KL(g) with their
dropped atoms (g > 0.1, ĝ < 0.1) and the latents that should have covered them.

Usage: python -m posthoc_ci.swap_stochastic --k 512 [--arm sym] [--seed 0]
"""

import argparse
import json

import numpy as np
import torch

from posthoc_ci import constants, glib, paths, swap_lib
from posthoc_ci.figstyle import SERIES, apply_style, save

N_WORST = 100
TOP_COVER_LATENTS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="/workspace/out-torch/runs/s-55ea3f9b")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", default="sym")
    parser.add_argument("--n-draws", type=int, default=constants.N_STOCHASTIC_DRAWS)
    args = parser.parse_args()
    apply_style()
    import matplotlib.pyplot as plt

    run = swap_lib.load_eval_run(args.run_dir)
    w_fn = 1.0 if args.arm == "sym" else float(args.arm.removeprefix("asym"))
    b = torch.from_numpy(
        np.load(paths.fit_dir(args.k, args.seed, args.arm) / "final.npz")["B"]
    ).to(run.device)

    n_seqs, seq_len = run.tokens.shape
    draws_total = args.n_draws * len(constants.SEEDS)
    kl_g = np.zeros((draws_total, n_seqs, seq_len), dtype=np.float32)
    kl_h = np.zeros_like(kl_g)
    dropped_count = np.zeros((n_seqs, seq_len), dtype=np.int32)
    ghat_store: list[torch.Tensor] = []
    g_store: list[torch.Tensor] = []

    for i, mb in enumerate(swap_lib.micro_slices()):
        tokens_mb = run.tokens[mb]
        ci_true = swap_lib.true_ci(run, tokens_mb)
        gh, _ = swap_lib.ghat_dict(run, ci_true, b, w_fn=w_fn)
        tgt = swap_lib.target_logits(run, tokens_mb)

        g_flat = swap_lib.ci_dict_to_flat(run, ci_true)
        h_flat = swap_lib.ci_dict_to_flat(run, gh)
        dropped = (g_flat > constants.TAU_EVAL) & (h_flat < constants.TAU_EVAL)
        dropped_count[mb] = dropped.sum(-1).cpu().numpy()
        g_store.append(g_flat.to(torch.float16).cpu())
        ghat_store.append(h_flat.to(torch.float16).cpu())

        d = 0
        for s in constants.SEEDS:
            for j in range(args.n_draws):
                seed = swap_lib.mb_seed(s * 1000 + j, i)
                lg = swap_lib.masked_forward(run, tokens_mb, ci_true, "stochastic", rng_seed=seed)
                kl_g[d, mb] = swap_lib.per_position_kl(lg, tgt).cpu().numpy()
                lh = swap_lib.masked_forward(run, tokens_mb, gh, "stochastic", rng_seed=seed)
                kl_h[d, mb] = swap_lib.per_position_kl(lh, tgt).cpu().numpy()
                d += 1
        print(f"micro-batch {i}: done ({draws_total} paired draws)")

    per_tok_g = kl_g.mean(axis=0)  # [n_seqs, seq_len]
    per_tok_h = kl_h.mean(axis=0)

    def stats(x: np.ndarray) -> dict:
        return {"mean": float(x.mean()), "p95": float(np.quantile(x, 0.95)),
                "p99": float(np.quantile(x, 0.99))}

    rng = np.random.default_rng(0)
    boots = per_tok_h.mean(axis=1)[rng.integers(0, n_seqs, size=(1000, n_seqs))].mean(axis=1)

    # localization
    excess = (per_tok_h - per_tok_g).ravel()
    worst = np.argsort(-excess)[:N_WORST]
    g_all = torch.cat(g_store).numpy()
    h_all = torch.cat(ghat_store).numpy()
    b_np = b.cpu().numpy()
    alive = glib.alive_cols()
    atom_df = glib.load_atom_index().set_index("atom_id")
    freq: dict[int, int] = {}
    worst_rows = []
    for t in worst:
        drop_atoms = np.flatnonzero((g_all[t] > constants.TAU_EVAL) & (h_all[t] < constants.TAU_EVAL))
        for a in drop_atoms:
            freq[int(a)] = freq.get(int(a), 0) + 1
        worst_rows.append({
            "token_index": int(t),
            "excess_kl": float(excess[t]),
            "n_dropped": int(len(drop_atoms)),
        })
    alive_pos = {int(a): i for i, a in enumerate(alive)}
    dropped_table = []
    for a, n in sorted(freq.items(), key=lambda kv: -kv[1])[:50]:
        cover = []
        if a in alive_pos:
            row = b_np[alive_pos[a]]
            for latent in np.argsort(-row)[:TOP_COVER_LATENTS]:
                cover.append({"latent": int(latent), "b": float(row[latent])})
        rec = atom_df.loc[a]
        dropped_table.append({
            "atom_id": int(a), "module": rec.module, "c": int(rec.c),
            "times_dropped_in_worst100": n, "covering_latents": cover,
        })

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.geomspace(max(per_tok_g.min(), 1e-5), max(per_tok_h.max(), 1e-3), 60)
    ax.hist(per_tok_g.ravel(), bins=bins, color=SERIES[0], alpha=0.6, label="true g")
    ax.hist(per_tok_h.ravel(), bins=bins, color=SERIES[1], alpha=0.6, label=f"ĝ (K={args.k})")
    ax.set(xscale="log", xlabel="per-token KL (mean over draws)", ylabel="tokens",
           title="E2.2 stochastic-mask per-token KL")
    ax.legend()
    save(fig, glib.fig_path(f"e22_stochastic_K{args.k}_{args.arm}.png"))

    report = {
        "k": args.k, "seed": args.seed, "arm": args.arm, "n_draws_total": draws_total,
        "true_g": stats(per_tok_g),
        "ghat": stats(per_tok_h),
        "ghat_mean_kl_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "ratio_mean": float(per_tok_h.mean() / per_tok_g.mean()),
        "ratio_p99": float(np.quantile(per_tok_h, 0.99) / np.quantile(per_tok_g, 0.99)),
        "mean_dropped_atoms_per_token": float(dropped_count.mean()),
        "worst_tokens": worst_rows[:20],
        "dropped_atom_table": dropped_table,
    }
    paths.SUBST_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.SUBST_DIR / f"stochastic_K{args.k}_seed{args.seed}_{args.arm}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("worst_tokens", "dropped_atom_table")}, indent=2))


if __name__ == "__main__":
    main()
