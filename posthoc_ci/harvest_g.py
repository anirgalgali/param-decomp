"""E1.0 + P1 + P2: harvest the token x atom CI matrix G, in one streaming pass.

Streams held-out (val-split) text through the frozen target + trained CI fn in the
run's own configuration (context length, autocast) and writes, all onto PD_POSTHOC:

  harvest/shard_<i>.npz     CSR of g > TAU_STORE for 8192 positions (fp32), all atoms
  harvest/token_meta.parquet   seq_id, pos, token_id, held_out flag per position
  atom_index.parquet        atom_id table + mean_ci, density@both taus, dead/alive
  latents/atom_top30.npz    per-atom top-30 firing positions + g values (sanity + E1.5)
  eval_batch.npz            the frozen 128x512 Rung-2 eval batch (seeded held-out sample)
  harvest/sanity_report.json   L0, per-layer aliveness, tracked-atom top tokens

Sequence-level split: seq_id % 10 == 9 is held-out (the loader already shuffles).

Usage (pod):
    python -m posthoc_ci.harvest_g --run-dir /workspace/out-torch/runs/s-55ea3f9b
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from param_decomp.batch_and_loss_fns import move_batch_to_device
from param_decomp.log import logger
from posthoc_ci import constants, paths
from posthoc_ci.atom_index import build_atom_index, canonical_modules

TRACKED_ATOMS_DEFAULT = ("h.1.attn.k_proj:218",)  # paper: fires on "it"
BATCH_SEQS = 16  # 16 x 512 = 8192 positions = exactly one shard per batch
TOPK = 30


def _concat_ci(ci: dict[str, torch.Tensor], modules: list[str]) -> torch.Tensor:
    """[batch, seq, A] in canonical atom order, fp32."""
    return torch.cat([ci[m].float() for m in modules], dim=-1)


def _merge_topk(
    best_vals: torch.Tensor, best_pos: torch.Tensor, g: torch.Tensor, pos0: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge this batch's per-atom top-k into the running (vals, positions) buffers."""
    k = best_vals.shape[0]
    batch_vals, batch_rows = torch.topk(g, k=min(k, g.shape[0]), dim=0)
    batch_pos = batch_rows.to(torch.int64) + pos0
    vals = torch.cat([best_vals, batch_vals], dim=0)
    pos = torch.cat([best_pos, batch_pos], dim=0)
    merged_vals, merged_idx = torch.topk(vals, k=k, dim=0)
    return merged_vals, torch.gather(pos, 0, merged_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-tokens", type=int, default=constants.HARVEST_N_TOKENS)
    parser.add_argument("--tracked-atoms", nargs="*", default=list(TRACKED_ATOMS_DEFAULT))
    args = parser.parse_args()

    from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader

    paths.ensure_dirs()
    device = "cuda"
    saved = SavedLMRun.from_path(args.run_dir)
    model = saved.load_model().to(device)
    model.eval()
    modules = canonical_modules(model.module_to_c)
    atom_df = build_atom_index(model.module_to_c)
    n_atoms = len(atom_df)
    shard_rows = constants.HARVEST_SHARD_POSITIONS
    assert BATCH_SEQS * saved.cfg.data.max_seq_len == shard_rows, "one batch == one shard"
    autocast = bool(saved.cfg.runtime.autocast_bf16)
    logger.info(f"{n_atoms} atoms across {len(modules)} modules; autocast_bf16={autocast}")

    loader = build_lm_loader(
        saved.cfg.target,
        saved.cfg.data,
        split="eval",
        device=device,
        batch_size=BATCH_SEQS,
        seed=saved.cfg.pd.seed,
    )

    n_shards = args.n_tokens // shard_rows
    seq_len = saved.cfg.data.max_seq_len

    ci_sum = torch.zeros(n_atoms, dtype=torch.float64, device=device)
    count_store = torch.zeros(n_atoms, dtype=torch.int64, device=device)
    count_eval = torch.zeros(n_atoms, dtype=torch.int64, device=device)
    top_vals = torch.zeros(TOPK, n_atoms, device=device)
    top_pos = torch.zeros(TOPK, n_atoms, dtype=torch.int64, device=device)
    l0_sum = 0.0

    all_token_ids: list[np.ndarray] = []
    meta_rows: list[pd.DataFrame] = []

    it = iter(loader)
    for shard_idx in range(n_shards):
        batch = move_batch_to_device(next(it), device)
        token_ids = batch["input_ids"]
        assert token_ids.shape == (BATCH_SEQS, seq_len), token_ids.shape

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=autocast):
            out = model(batch, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling="continuous", detach_inputs=False
            ).lower_leaky
        g = _concat_ci(ci, modules).reshape(-1, n_atoms)  # [8192, A] fp32

        pos0 = shard_idx * shard_rows
        ci_sum += g.sum(dim=0, dtype=torch.float64)
        count_store += (g > constants.TAU_STORE).sum(dim=0)
        count_eval += (g > constants.TAU_EVAL).sum(dim=0)
        l0_sum += (g > constants.TAU_STORE).sum().item()
        top_vals, top_pos = _merge_topk(top_vals, top_pos, g, pos0)

        mask = g > constants.TAU_STORE
        idx = mask.nonzero()
        rows_np = idx[:, 0].cpu().numpy()
        cols_np = idx[:, 1].to(torch.int32).cpu().numpy()
        vals_np = g[mask].cpu().numpy().astype(np.float32)
        csr = sparse.csr_matrix(
            (vals_np, (rows_np, cols_np)), shape=(shard_rows, n_atoms), dtype=np.float32
        )
        sparse.save_npz(paths.shard_path(shard_idx), csr)

        seq0 = shard_idx * BATCH_SEQS
        seq_ids = np.repeat(np.arange(seq0, seq0 + BATCH_SEQS), seq_len)
        meta_rows.append(
            pd.DataFrame(
                {
                    "seq_id": seq_ids,
                    "pos": np.tile(np.arange(seq_len), BATCH_SEQS),
                    "token_id": token_ids.reshape(-1).cpu().numpy().astype(np.int32),
                    "held_out": (seq_ids % 10 == 9),
                }
            )
        )
        all_token_ids.append(token_ids.reshape(-1).cpu().numpy().astype(np.int32))
        if shard_idx % 10 == 0:
            logger.info(f"shard {shard_idx}/{n_shards}: running L0 "
                        f"{l0_sum / ((shard_idx + 1) * shard_rows):.1f}")

    token_meta = pd.concat(meta_rows, ignore_index=True)
    token_meta.to_parquet(paths.TOKEN_META_PARQUET)

    n_positions = n_shards * shard_rows
    mean_ci = (ci_sum / n_positions).cpu().numpy()
    atom_df["mean_ci"] = mean_ci
    atom_df["density_store"] = count_store.cpu().numpy() / n_positions
    atom_df["density_eval"] = count_eval.cpu().numpy() / n_positions
    atom_df["alive"] = atom_df["mean_ci"] >= constants.DEAD_MEAN_CI
    atom_df.to_parquet(paths.ATOM_INDEX_PARQUET)

    np.savez_compressed(
        paths.LATENTS_DIR / "atom_top30.npz",
        values=top_vals.cpu().numpy().astype(np.float32),
        positions=top_pos.cpu().numpy(),
    )

    _write_eval_batch(token_meta, np.concatenate(all_token_ids), seq_len)
    _write_sanity_report(
        saved, atom_df, l0_sum / n_positions, top_vals, top_pos,
        np.concatenate(all_token_ids), args.tracked_atoms,
    )


def _write_eval_batch(token_meta: pd.DataFrame, flat_tokens: np.ndarray, seq_len: int) -> None:
    held_seqs = token_meta.loc[token_meta.held_out, "seq_id"].unique()
    rng = np.random.default_rng(constants.EVAL_BATCH_SAMPLE_SEED)
    chosen = np.sort(rng.choice(held_seqs, size=constants.EVAL_BATCH_SEQS, replace=False))
    tokens = flat_tokens.reshape(-1, seq_len)[chosen]
    np.savez(paths.EVAL_BATCH_NPZ, seq_ids=chosen, token_ids=tokens)
    logger.info(f"eval batch frozen: {tokens.shape} from {len(held_seqs)} held-out seqs")


def _write_sanity_report(
    saved, atom_df, mean_l0, top_vals, top_pos, flat_tokens, tracked_atoms
) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(saved.cfg.data.tokenizer_name)
    per_layer_alive = atom_df.groupby("layer")["alive"].sum().to_dict()

    tracked = {}
    module_start = atom_df.groupby("module")["atom_id"].min().to_dict()
    for spec in tracked_atoms:
        module, c = spec.rsplit(":", 1)
        atom = int(module_start[module]) + int(c)
        positions = top_pos[:, atom].cpu().numpy()
        vals = top_vals[:, atom].cpu().numpy()
        tokens = [tokenizer.decode([int(flat_tokens[p])]) for p in positions]
        tracked[spec] = [
            {"g": round(float(v), 3), "token": t} for v, t in zip(vals, tokens, strict=True)
        ]

    report = {
        "n_positions": int(len(flat_tokens)),
        "mean_l0_at_tau_store": round(float(mean_l0), 2),
        "expected_mean_l0": "~205 (paper)",
        "n_atoms": int(len(atom_df)),
        "n_alive": int(atom_df.alive.sum()),
        "per_layer_alive": {str(k): int(v) for k, v in per_layer_alive.items()},
        "expected_per_layer_alive": "paper table: 848 / 1943 / 3472 / ...",
        "tracked_atom_top_tokens": tracked,
    }
    paths.SANITY_REPORT_JSON.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
