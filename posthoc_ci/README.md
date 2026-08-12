# posthoc_ci — Rungs 1 & 2 on the paper's 4L VPD run

Post-hoc compressibility (Rung 1) and functional sufficiency (Rung 2) of causal-importance
codes, per `bottleneck_pd/docs/experiments/rung1_rung2_experiment_plan.md` (incl. its
pre-launch amendments). Frozen run: `goodfire/spd/runs/s-55ea3f9b` (400k steps, 4L Pile,
38,912 atoms). Everything runs on the RunPod torch stack at `/workspace/pdt`.

## Environment (every pod shell / tmux session)

```bash
export PARAM_DECOMP_OUT_DIR=/workspace/out-torch
export PD_POSTHOC=/workspace/out-torch/posthoc
export HF_HOME=/workspace/.cache/huggingface
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^(WANDB_API_KEY|GITHUB_TOKEN)=")
```

Launch pattern (detached; survives disconnects):
```bash
tmux new-session -d -s <stage> -c /workspace/pdt '<env>; uv run --no-sync python -m posthoc_ci.<module> <args> 2>&1 | tee /workspace/logs/<stage>.log; echo EXIT=$? >> /workspace/logs/<stage>.log'
tmux ls && sleep 5 && head -20 /workspace/logs/<stage>.log   # verify started
```

## Pipeline (in order; ✔ = verification)

| # | Command | ✔ |
|---|---------|---|
| P0 | `python -m posthoc_ci.fetch_run --try-load --smoke data` | L0@0.01 ≈ 205; LOAD OK |
| E1.0 | `python -m posthoc_ci.harvest_g --run-dir /workspace/out-torch/runs/s-55ea3f9b` | sanity_report.json: L0≈205, alive≈9,972, named atom fires on "it" |
| E1.1 | `python -m posthoc_ci.stats_descriptive` | 4 figures + hub/positional lists |
| E1.2 | `python -m posthoc_ci.svd_gate_a` | Gate A verdict in e12_gate_a.json |
| E1.3 | `bash posthoc_ci/sweep.sh` then `python -m posthoc_ci.fit_sweep --aggregate` | knee + Gate B in fits/summary.json |
| E1.3+ | asym arm at knee: `python -m posthoc_ci.fit_sweep --k <knee> --seed <s> --w-fn 3` (×3 seeds, ± one grid step) | metrics.json per cell |
| E1.4b | clustering harvest+merge (see mdl_baseline.py docstring), then `python -m posthoc_ci.mdl_baseline --history <run>/history.zip --k <knee>` | baselines/mdl_K*.json |
| E1.4c | `python -m posthoc_ci.controls --k <knee> --seed 0` | shuffled/random ≪ trained |
| E1.5 | `python -m posthoc_ci.latent_report --k <knee>` then `python -m posthoc_ci.latent_browser` | latent_browser.html (scp back to view) |
| E2.0 | `python -m posthoc_ci.encode_eval_batch --k <knee>` | identity_self_test_pass: true |
| E2.1 | `python -m posthoc_ci.swap_deterministic --k <knee> --conditions g,sym,asym3` | true-g rows: unmasked KL ≈ 0.012, CI-masked ≈ 0.35 |
| E2.2 | `python -m posthoc_ci.swap_stochastic --k <knee>` (repeat `--arm asym3`) | ratios vs Gate C bands |
| E2.3 | `python -m posthoc_ci.swap_adversarial --k <knee> --conditions g,sym,asym3 --steps 20,40` | ratio_vs_g ≤ 1.5 with floors-L0 guard |
| E2.4 | `swap_deterministic`/`swap_adversarial` with `--conditions shuffled,binarized,covering` and `--k <knee/2, 2·knee>` | shuffled clearly fails |
| memo | `python -m posthoc_ci.report` | memo.md + e14_overlay.png |

## Data layout (all on the network volume)

```
$PD_POSTHOC/
  harvest/shard_<i>.npz          CSR g>0.01, 8192 positions each, all 38,912 atoms
  harvest/token_meta.parquet     seq_id, pos, token_id, held_out (seq_id%10==9)
  atom_index.parquet             atom_id → module/layer/matrix_type/c + aliveness
  eval_batch.npz                 frozen 128×512 Rung-2 batch (seeded held-out sample)
  latents/atom_top30.npz         per-atom top-30 firing positions (sanity + E1.5)
  fits/K<k>_seed<s>_<arm>/       final.npz (B) + metrics.json
  baselines/  subst/  figures/   E1.4 / Rung-2 JSON / all PNGs
  memo.md                        auto-collected results memo
```

Protocol constants (τ, splits, seeds, RNG keys, gate bounds) live in `constants.py` and
change only via a written amendment to the plan doc.
