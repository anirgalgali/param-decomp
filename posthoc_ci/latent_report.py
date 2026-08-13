"""E1.5: semantic evaluation of the latents at knee-K.

Produces latents/latent_report.json (consumed by latent_browser) plus the two figures:
cross-matrix mass and seed stability. All context windows are rebuilt from the harvest
token ids — no model forward needed.

  1. codes z for every harvested position (frozen-B solve, shard-streamed); top-30
     positions per latent by z
  2. member consistency: top member atoms per latent (largest B-column entries), each
     with its own top-30 positions (from harvest's atom_top30.npz)
  3. cross-matrix mass fraction per latent (B-column mass over the 24 weight matrices)
  4. stability: Hungarian matching of B columns across the three seeds (cosine)

Hub latents are flagged (top member is a hub atom, or latent fires on >50% of
positions) and excluded from the featured list.

Usage: python -m posthoc_ci.latent_report --k <knee> [--arm sym]
"""

import argparse
import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from posthoc_ci import constants, glib, paths
from posthoc_ci.figstyle import SERIES, apply_style, save
from posthoc_ci.harvest_g import TOPK, _merge_topk
from posthoc_ci.nmf import solve_codes

N_FEATURED = 50
N_MEMBERS = 8
WINDOW = 16


N_COND = 8


def member_pairs(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(pair_latent, pair_atom) index vectors for every latent's top-N_MEMBERS atoms.

    Members are a function of B alone, so the (latent, member-atom) pairs are known
    before any pass over the data — length K * N_MEMBERS each, pair i belonging to
    latent pair_latent[i] and alive-atom column pair_atom[i].
    """
    k = b.shape[1]
    n_members = min(N_MEMBERS, b.shape[0])
    member_idx = torch.topk(b, n_members, dim=0).indices  # [n_members, K]
    pair_latent = torch.arange(k, device=b.device).repeat_interleave(n_members)
    pair_atom = member_idx.T.reshape(-1)
    return pair_latent, pair_atom


def _codes_topk(
    b: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor]:
    """One streaming pass: per-latent top-30 by z, firing density, and per-(latent,
    member) conditioned top-8 — positions maximizing z_k among those where the member
    atom itself fires (g > TAU_EVAL). The conditioning is what makes the member panel
    mixture-aware: ĝ is a sum over latents, so an atom's *global* firings mix all its
    uses; these are its firings while THIS latent is engaged.
    """
    k = b.shape[1]
    pair_latent, pair_atom = member_pairs(b)
    top_vals = torch.zeros(TOPK, k, device=device)
    top_pos = torch.zeros(TOPK, k, dtype=torch.int64, device=device)
    cond_vals = torch.zeros(N_COND, len(pair_latent), device=device)
    cond_pos = torch.zeros(N_COND, len(pair_latent), dtype=torch.int64, device=device)
    fire_count = torch.zeros(k, dtype=torch.float64, device=device)
    n_rows = 0
    rows_per_shard = constants.HARVEST_SHARD_POSITIONS
    for shard_idx, csr in glib.iter_shards("all"):
        g = torch.from_numpy(csr.toarray()).to(device)
        z = solve_codes(g, b, constants.Z_SOLVER_N_STEPS)
        pos0 = shard_idx * rows_per_shard
        top_vals, top_pos = _merge_topk(top_vals, top_pos, z, pos0)
        scores = z[:, pair_latent] * (g[:, pair_atom] > constants.TAU_EVAL)
        cond_vals, cond_pos = _merge_topk(cond_vals, cond_pos, scores, pos0)
        fire_count += (z > 0.01).sum(dim=0, dtype=torch.float64)
        n_rows += z.shape[0]
    density = (fire_count / n_rows).cpu().numpy()
    return top_vals, top_pos, density, cond_vals, cond_pos


def _window(tokens: np.ndarray, tokenizer, pos: int, seq_len: int) -> dict:
    seq_start = (pos // seq_len) * seq_len
    lo = max(seq_start, pos - WINDOW)
    hi = min(seq_start + seq_len, pos + WINDOW + 1)
    return {
        "before": tokenizer.decode(tokens[lo:pos].tolist()),
        "token": tokenizer.decode([int(tokens[pos])]),
        "after": tokenizer.decode(tokens[pos + 1 : hi].tolist()),
    }


def _stability(k: int, arm: str) -> np.ndarray:
    """Matched cosine similarities of B columns, seed 0 vs 1 and 0 vs 2, concatenated."""
    mats = []
    for s in constants.SEEDS:
        b = np.load(paths.fit_dir(k, s, arm) / "final.npz")["B"]
        mats.append(b / (np.linalg.norm(b, axis=0, keepdims=True) + 1e-9))
    sims = []
    for other in (1, 2):
        cos = mats[0].T @ mats[other]
        row, col = linear_sum_assignment(-cos)
        sims.append(cos[row, col])
    return np.concatenate(sims)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", default="sym")
    args = parser.parse_args()
    apply_style()
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = torch.from_numpy(np.load(paths.fit_dir(args.k, args.seed, args.arm) / "final.npz")["B"]).to(device)
    atom_df = glib.load_atom_index()
    alive = atom_df[atom_df.alive].reset_index(drop=True)
    token_meta = glib.load_token_meta()
    tokens = token_meta.token_id.to_numpy()
    seq_len = int(token_meta.pos.max()) + 1
    def _atom_id_set(path) -> set[int]:
        records = json.loads(path.read_text())
        return {int(r["atom_id"]) for r in records}

    hubs = _atom_id_set(paths.HARVEST_DIR / "hub_atoms.json")
    positional = _atom_id_set(paths.HARVEST_DIR / "positional_atoms.json")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    top_vals, top_pos, density, cond_vals, cond_pos = _codes_topk(b, device)
    pair_latent, pair_atom = member_pairs(b)
    b_row_sums = b.sum(dim=1).cpu().numpy()  # atom's total mass across all latents
    atom_top = np.load(paths.LATENTS_DIR / "atom_top30.npz")

    mean_z_order = np.argsort(-top_vals.mean(dim=0).cpu().numpy())
    matrix_of = (alive.layer.astype(str) + "." + alive.matrix_type).to_numpy()

    latents = []
    for latent in mean_z_order[: N_FEATURED * 2]:
        col = b[:, latent].cpu().numpy()
        mass = col / (col.sum() + 1e-12)
        pair_rows = np.arange(latent * N_MEMBERS, (latent + 1) * N_MEMBERS)
        member_idx = pair_atom[pair_rows].cpu().numpy()  # same members, topk order
        member_atom_ids = alive.atom_id.to_numpy()[member_idx]
        is_hub_latent = bool(density[latent] > 0.5 or member_atom_ids[0] in hubs)

        by_matrix: dict[str, float] = {}
        for m, frac in zip(matrix_of, mass, strict=True):
            by_matrix[m] = by_matrix.get(m, 0.0) + float(frac)
        top_matrices = sorted(by_matrix.items(), key=lambda kv: -kv[1])[:4]
        n_big = sum(1 for _, v in by_matrix.items() if v >= 0.2)

        contexts = [
            {"z": round(float(v), 3), **_window(tokens, tokenizer, int(p), seq_len)}
            for v, p in zip(top_vals[:, latent].cpu(), top_pos[:, latent].cpu(), strict=True)
            if v > 0
        ]
        members = []
        for rank, (mi, aid) in enumerate(zip(member_idx, member_atom_ids, strict=True)):
            m_ctx = [
                {"g": round(float(v), 3), **_window(tokens, tokenizer, int(p), seq_len)}
                for v, p in zip(atom_top["values"][:8, aid], atom_top["positions"][:8, aid], strict=True)
                if v > 0
            ]
            pr = int(pair_rows[rank])
            c_ctx = [
                {"z": round(float(v), 3), **_window(tokens, tokenizer, int(p), seq_len)}
                for v, p in zip(cond_vals[:, pr].cpu(), cond_pos[:, pr].cpu(), strict=True)
                if v > 0
            ]
            members.append(
                {
                    "atom_id": int(aid),
                    "module": alive.module.iloc[mi],
                    "c": int(alive.c.iloc[mi]),
                    "b_weight": round(float(col[mi]), 4),
                    "row_share": round(float(col[mi] / max(b_row_sums[mi], 1e-12)), 3),
                    "is_hub": bool(aid in hubs),
                    "is_positional": bool(aid in positional),
                    "contexts": m_ctx,
                    "conditioned_contexts": c_ctx,
                }
            )
        latents.append(
            {
                "latent": int(latent),
                "mean_top_z": round(float(top_vals[:, latent].mean()), 3),
                "density": round(float(density[latent]), 4),
                "is_hub_latent": is_hub_latent,
                "cross_matrix": {
                    "n_matrices_ge_20pct": int(n_big),
                    "spans_attn_and_mlp": bool(
                        any("attn" in m for m, v in by_matrix.items() if v >= 0.2)
                        and any("mlp" in m for m, v in by_matrix.items() if v >= 0.2)
                    ),
                    "top_matrices": top_matrices,
                },
                "contexts": contexts,
                "members": members,
            }
        )

    featured = [rec for rec in latents if not rec["is_hub_latent"]][:N_FEATURED]
    non_hub = [rec for rec in latents if not rec["is_hub_latent"]]
    frac_multi = float(np.mean([rec["cross_matrix"]["n_matrices_ge_20pct"] >= 2 for rec in non_hub]))
    frac_attn_mlp = float(np.mean([rec["cross_matrix"]["spans_attn_and_mlp"] for rec in non_hub]))

    sims = _stability(args.k, args.arm)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(
        [rec["cross_matrix"]["n_matrices_ge_20pct"] for rec in non_hub],
        bins=np.arange(0.5, 8.5), color=SERIES[0],
    )
    axes[0].set(xlabel="weight matrices with ≥20% of column mass", ylabel="latents",
                title=f"Cross-matrix mass (non-hub): {frac_multi:.0%} span ≥2, "
                      f"{frac_attn_mlp:.0%} span attn+mlp")
    axes[1].hist(sims, bins=40, color=SERIES[1])
    axes[1].set(xlabel="matched cosine similarity across seeds", ylabel="latent pairs",
                title=f"Seed stability (median {np.median(sims):.2f})")
    save(fig, glib.fig_path("e15_crossmatrix_stability.png"))

    report = {
        "k": args.k,
        "seed": args.seed,
        "arm": args.arm,
        "headline": {
            "frac_nonhub_latents_spanning_2_matrices": frac_multi,
            "frac_nonhub_latents_spanning_attn_and_mlp": frac_attn_mlp,
            "stability_median_cos": float(np.median(sims)),
            "n_hub_latents_flagged": int(sum(rec["is_hub_latent"] for rec in latents)),
        },
        "latents": featured,
    }
    out = paths.LATENTS_DIR / "latent_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["headline"], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
