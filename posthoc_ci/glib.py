"""Shared access to the harvested G matrix: shards, splits, alive restriction.

The split is sequence-level (seq_id % 10 == 9 held out), decided at harvest time and
recorded in token_meta.parquet; this module only reads it back. All returned matrices
are restricted to alive atoms (canonical atom order preserved) unless stated otherwise.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from posthoc_ci import constants, paths

Split = Literal["train", "held", "all"]


def load_atom_index() -> pd.DataFrame:
    return pd.read_parquet(paths.ATOM_INDEX_PARQUET)


def load_token_meta() -> pd.DataFrame:
    return pd.read_parquet(paths.TOKEN_META_PARQUET)


def alive_cols() -> np.ndarray:
    df = load_atom_index()
    return df.loc[df.alive, "atom_id"].to_numpy()


def n_shards() -> int:
    return len(sorted(paths.HARVEST_DIR.glob("shard_*.npz")))


def _row_mask_for_shard(token_meta: pd.DataFrame, shard_idx: int, split: Split) -> np.ndarray:
    rows = constants.HARVEST_SHARD_POSITIONS
    held = token_meta.held_out.to_numpy()[shard_idx * rows : (shard_idx + 1) * rows]
    if split == "train":
        return ~held
    if split == "held":
        return held
    return np.ones(rows, dtype=bool)


def iter_shards(
    split: Split, alive_only: bool = True
) -> Iterator[tuple[int, sparse.csr_matrix]]:
    """Yield (shard_idx, csr) with rows restricted to `split`, cols to alive atoms."""
    token_meta = load_token_meta()
    cols = alive_cols() if alive_only else None
    for shard_idx in range(n_shards()):
        csr = sparse.load_npz(paths.shard_path(shard_idx))
        csr = csr[_row_mask_for_shard(token_meta, shard_idx, split)]
        if cols is not None:
            csr = csr[:, cols]
        yield shard_idx, csr


def load_split_csr(split: Split, alive_only: bool = True) -> sparse.csr_matrix:
    """The full split as one CSR (fits comfortably: ~2e8 nnz worst case for 'all')."""
    mats = [csr for _, csr in iter_shards(split, alive_only)]
    return sparse.vstack(mats, format="csr")


def per_atom_mean(split: Split = "train") -> np.ndarray:
    """Mean g per alive atom over the split (the EV baseline). Cached on disk."""
    cache = paths.HARVEST_DIR / f"per_atom_mean_{split}.npy"
    if cache.exists():
        return np.load(cache)
    total = None
    n_rows = 0
    for _, csr in iter_shards(split):
        s = np.asarray(csr.sum(axis=0)).ravel()
        total = s if total is None else total + s
        n_rows += csr.shape[0]
    assert total is not None, "no shards found"
    mean = (total / n_rows).astype(np.float32)
    np.save(cache, mean)
    return mean


def dense_row_batches(
    split: Split, batch_rows: int, device: str | torch.device, seed: int | None = None
) -> Iterator[torch.Tensor]:
    """Densified [<=batch_rows, A_alive] fp32 batches, optionally row-shuffled per shard."""
    rng = np.random.default_rng(seed) if seed is not None else None
    for _, csr in iter_shards(split):
        order = np.arange(csr.shape[0])
        if rng is not None:
            rng.shuffle(order)
        for start in range(0, csr.shape[0], batch_rows):
            rows = order[start : start + batch_rows]
            dense = torch.from_numpy(csr[rows].toarray()).to(device)
            yield dense


def load_eval_batch() -> dict[str, np.ndarray]:
    data = np.load(paths.EVAL_BATCH_NPZ)
    return {"seq_ids": data["seq_ids"], "token_ids": data["token_ids"]}


def fig_path(name: str) -> Path:
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return paths.FIGURES_DIR / name
