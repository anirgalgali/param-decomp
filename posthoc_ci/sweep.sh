#!/usr/bin/env bash
# E1.3 grid: K x seeds, 4 concurrent fits round-robin over the 4 GPUs.
# Usage: bash posthoc_ci/sweep.sh   (from /workspace/pdt, env already exported)
set -u
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"

i=0
pids=()
for k in 1 64 128 256 512 1024 2048; do
  for s in 0 1 2; do
    gpu=$((i % 4))
    log="$LOGDIR/fit_K${k}_s${s}.log"
    CUDA_VISIBLE_DEVICES=$gpu uv run --no-sync python -m posthoc_ci.fit_sweep \
      --k "$k" --seed "$s" >"$log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
    if (( ${#pids[@]} == 4 )); then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
done
wait
echo "SWEEP DONE: $(ls /workspace/out-torch/posthoc/fits/*/metrics.json 2>/dev/null | wc -l) cells"
