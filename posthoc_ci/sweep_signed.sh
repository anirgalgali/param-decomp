#!/usr/bin/env bash
# Rung 1S grid: sgnB at K in {128,256,512,1024} x 3 seeds, sgnBZ at {512,1024} x 3.
# Doubled fit budget (epochs 8): the signed family is optimization-limited (appendix-B).
# Usage: bash posthoc_ci/sweep_signed.sh   (from /workspace/pdt, PD_POSTHOC exported)
set -u
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"

run() { # k seed extra_flags log_tag
  CUDA_VISIBLE_DEVICES=$((i % 4)) uv run --no-sync python -m posthoc_ci.fit_sweep \
    --k "$1" --seed "$2" --epochs 8 --signed-b $3 >"$LOGDIR/fit_K$1_s$2_$4.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
  if (( ${#pids[@]} == 4 )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
}

i=0
pids=()
for k in 128 256 512 1024; do
  for s in 0 1 2; do run "$k" "$s" "" sgnB; done
done
for k in 512 1024; do
  for s in 0 1 2; do run "$k" "$s" "--signed-z" sgnBZ; done
done
wait
echo "SIGNED SWEEP DONE: $(ls /workspace/out-torch/posthoc/fits/*sgn*/metrics.json 2>/dev/null | wc -l) cells" \
  > /workspace/logs/signed-sweep-done.flag
