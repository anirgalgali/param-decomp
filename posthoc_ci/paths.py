"""Output-tree resolution. Everything is anchored at PD_POSTHOC (one env var to relocate).

On the pod: PD_POSTHOC defaults to /workspace/out-torch/posthoc (network volume).
Locally: set PD_POSTHOC explicitly, e.g. to a scratch dir, for smoke tests.
"""

import os
from pathlib import Path

POSTHOC_ROOT = Path(os.environ.get("PD_POSTHOC", "/workspace/out-torch/posthoc"))

HARVEST_DIR = POSTHOC_ROOT / "harvest"
FITS_DIR = POSTHOC_ROOT / "fits"
SUBST_DIR = POSTHOC_ROOT / "subst"
FIGURES_DIR = POSTHOC_ROOT / "figures"
BASELINES_DIR = POSTHOC_ROOT / "baselines"
LATENTS_DIR = POSTHOC_ROOT / "latents"

ATOM_INDEX_PARQUET = POSTHOC_ROOT / "atom_index.parquet"
TOKEN_META_PARQUET = HARVEST_DIR / "token_meta.parquet"
SANITY_REPORT_JSON = HARVEST_DIR / "sanity_report.json"
EVAL_BATCH_NPZ = POSTHOC_ROOT / "eval_batch.npz"


def shard_path(shard_idx: int) -> Path:
    return HARVEST_DIR / f"shard_{shard_idx:04d}.npz"


def fit_dir(k: int, seed: int, arm: str = "sym") -> Path:
    """arm: 'sym' (main symmetric loss) or 'asym<w>' (under-prediction-weighted)."""
    return FITS_DIR / f"K{k}_seed{seed}_{arm}"


def ensure_dirs() -> None:
    for d in (HARVEST_DIR, FITS_DIR, SUBST_DIR, FIGURES_DIR, BASELINES_DIR, LATENTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
