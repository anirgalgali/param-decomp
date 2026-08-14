#!/usr/bin/env bash
# Rung 1R grid: rect at K in {256,512,1024} x 3 cold seeds, rect-warm at K=512 (seed 0,
# init from the trained sym dictionary), rect-asym3 at K=512 x 3 seeds. Epochs 8.
# The warm arm runs FIRST and alone: its §3.2 anchor assertion (b=0 reproduces the sym
# row) gates the whole sweep — if it fails, nothing else launches.
# Usage: bash posthoc_ci/sweep_rectified.sh   (from /workspace/pdt, PD_POSTHOC exported)
set -u
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"

echo "warm anchor + warm arm (gates the sweep)..."
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m posthoc_ci.fit_sweep \
  --k 512 --seed 0 --rectified --init-from sym --epochs 8 \
  >"$LOGDIR/fit_K512_s0_rect-warm.log" 2>&1
if ! grep -q "warm anchor OK" "$LOGDIR/fit_K512_s0_rect-warm.log"; then
  echo "WARM ANCHOR FAILED" > /workspace/logs/rect-sweep-done.flag
  exit 1
fi
echo "warm anchor passed; launching cold grid"

i=0
pids=()
run() { # k seed extra_flags log_tag
  CUDA_VISIBLE_DEVICES=$((i % 4)) uv run --no-sync python -m posthoc_ci.fit_sweep \
    --k "$1" --seed "$2" --rectified --epochs 8 $3 >"$LOGDIR/fit_K$1_s$2_$4.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
  if (( ${#pids[@]} == 4 )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
}

for k in 256 512 1024; do
  for s in 0 1 2; do run "$k" "$s" "" rect; done
done
for s in 0 1 2; do run 512 "$s" "--w-fn 3" rect-asym3; done
wait
echo "RECT SWEEP DONE: $(ls /workspace/out-torch/posthoc/fits/*rect*/metrics.json 2>/dev/null | wc -l) cells" \
  > /workspace/logs/rect-sweep-done.flag
