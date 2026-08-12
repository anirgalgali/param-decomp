"""Frozen protocol constants (P2 of the plan). Fixed before any experiment ran.

Every module imports from here; nothing here may change after results exist without a
written amendment in the plan doc.
"""

TAU_STORE: float = 0.01
"""Sparsification threshold for the harvested G matrix (matches the paper's clustering τ)."""

TAU_EVAL: float = 0.1
"""'Functionally important' threshold: recall metrics and binarizations use this."""

DEAD_MEAN_CI: float = 1e-6
"""An atom is dead iff its mean CI over the aliveness sample is below this (paper's criterion)."""

HELD_OUT_FRACTION: float = 0.10
"""Fraction of harvested sequences reserved as held-out. Split is at the sequence level."""

EVAL_BATCH_SEQS: int = 128
EVAL_BATCH_SEQ_LEN: int = 512
"""Fixed Rung-2 eval batch shape: a seeded random sample of held-out sequences."""

SEEDS: tuple[int, int, int] = (0, 1, 2)
"""Seeds for every stochastic procedure (NMF init, stochastic masks)."""

RNG_BASE_KEY: int = 20260811
"""Base key for paired-RNG stochastic evals: draws are identical across conditions."""

EVAL_BATCH_SAMPLE_SEED: int = 20260812
"""Seed for sampling the 128 eval sequences from the held-out pool (amendment 7)."""

K_GRID: tuple[int, ...] = (1, 64, 128, 256, 512, 1024, 2048)
"""E1.3 sweep grid. K=1 is the rank-1 'activity bit' reference (amendment 3)."""

KNEE_WEIGHTED_RECALL_MIN: float = 0.9
KNEE_INDUCED_L0_RATIO_MAX: float = 2.0
GATE_B_K_MAX: int = 1000
"""Knee rule: smallest K with weighted recall@TAU_EVAL >= 0.9 and induced-L0 ratio <= 2."""

CODE_L0_EPSILONS: tuple[float, float] = (0.01, 0.1)
"""Thresholds at which the per-token code L0 is reported."""

ROUNDING_THRESHOLDS: tuple[float, float, float] = (TAU_STORE, 0.1, 0.5)
"""E2.1 rounded-mask rows. TAU_STORE stands in for '> 0' on ĝ (plan pitfall #6)."""

PGD_N_STEPS: int = 20
PGD_STEP_SIZE: float = 0.1
PGD_INIT: str = "random"
PGD_MASK_SCOPE: str = "shared_across_batch"
"""E2.3 protocol — must match the paper / the run's own PGDReconLoss eval config exactly."""

N_STOCHASTIC_DRAWS: int = 8
"""Minimum stochastic mask draws per condition in E2.2 (per seed)."""

HARVEST_N_TOKENS: int = 1_000_000
HARVEST_SHARD_POSITIONS: int = 8192
"""E1.0: ~1M token positions, CSR shards of 8192 positions each."""

ALIVENESS_N_TOKENS: int = 100_000
"""P1: sample size for the mean-CI aliveness computation."""

Z_SOLVER_N_STEPS: int = 40
"""Projected-gradient steps for the frozen-B per-token code solve (E1.3.2 / E2.0)."""


def explained_variance_num_den(g, g_hat, per_atom_mean):
    """The one EV definition (amendment 8): EV = 1 - ||G-Ĝ||^2 / ||G - per-atom-mean||^2.

    Returns (residual_sq_sum, baseline_sq_sum) so callers can accumulate over shards.
    All tensors restricted to alive atoms.
    """
    residual = ((g - g_hat) ** 2).sum()
    baseline = ((g - per_atom_mean) ** 2).sum()
    return residual, baseline
