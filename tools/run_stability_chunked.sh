#!/usr/bin/env bash
# Chunked multi-seed stability probe: one process + fresh sims PER SEED, so a
# sim freeze costs one seed-chunk instead of the whole probe. Aggregate the
# per-chunk run dirs afterwards.
#
# Usage: tools/run_stability_chunked.sh <checkpoint> [seeds...]
set -u
CKPT="${1:?usage: run_stability_chunked.sh <checkpoint> [seeds...]}"
shift || true
SEEDS=("${@:-11 23 37 51 67 83}")
if [ "${#SEEDS[@]}" -eq 1 ]; then read -r -a SEEDS <<< "${SEEDS[0]}"; fi

export RL_GATE_SHOOT_OPEN_ENTER="${RL_GATE_SHOOT_OPEN_ENTER:-0.30}"
export RL_GATE_SHOOT_OPEN_EXIT="${RL_GATE_SHOOT_OPEN_EXIT:-0.20}"
export PYTHONPATH="${PYTHONPATH:-/home/luis/GitHub/RL}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="runs/behavioral_stability/chunked_${STAMP}"
mkdir -p "$OUT_ROOT"
echo "[chunked-probe] checkpoint=$CKPT seeds=${SEEDS[*]} out=$OUT_ROOT"

for seed in "${SEEDS[@]}"; do
  export PYBH_RL_SHM_NAME="/pybh_rl_probe_${STAMP}_${seed}"
  echo "[chunked-probe] === seed $seed ==="
  python behavioral_stability_probe.py \
    --checkpoint "$CKPT" \
    --seed-list "$seed" \
    --output-dir "$OUT_ROOT/seed_${seed}" \
    2>&1 | grep -E "\[summary\]|\[probe\]|pass_rate|Error|RuntimeError" | tail -12
  status=$?
  echo "[chunked-probe] seed $seed exit=$status"
  # Hard cleanup between chunks — frozen sims must not leak into the next seed.
  pkill -f SimRobot 2>/dev/null; pkill -f Xvfb 2>/dev/null; sleep 5
done

echo "[chunked-probe] done. Aggregate per-seed results under $OUT_ROOT"
