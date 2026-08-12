"""Post-hoc compressibility and functional sufficiency of causal-importance codes.

Rungs 1 & 2 of docs/experiments/rung1_rung2_experiment_plan.md (bottleneck_pd repo):
harvest the token x atom CI matrix G from a frozen trained VPD run, measure its
nonnegative compressibility (Rung 1), and swap the reconstructed floors back into the
run's own masking evaluations (Rung 2).

Lives at the repo root, outside `param_decomp`/`param_decomp_lab`, on purpose: it
consumes the library (checkpoint IO, CI fns, masks, PGD) and is never imported by it.
Every module is a CLI entry point: `python -m posthoc_ci.<module> --help`.
"""
