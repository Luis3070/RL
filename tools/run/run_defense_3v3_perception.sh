#!/usr/bin/env bash
# v4.2-accommodation evidence: defense_3v3_full trío (3 RL field + BH GK) in the
# PERCEPTION-WORKING scene (RL1v0_3D_visible per-slot + injected opponents), so the
# defender actually SEES the ball/opponents and the corrected Design-A gate can ARM
# block/mark. Confirms block/mark FIRE (executed_block/mark_rate > 0) and goals-
# conceded A/B vs the bare v2b@7500 trío. Pure-policy (BC off), same seeds.
#
# Uses RL_MA_PROD_FORCE_PERCEPTION_SCENE=1 to bypass the stage-8 RLMAStage7_3D
# hard-wire (RLMAStage7_3D = perception-blind for non-striker robots).
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"

# Force the perception-working scene for the stage-8 trío.
export RL_MA_PROD_FORCE_PERCEPTION_SCENE=1
export RL_SCENE_NAME=RL1v0_3D_visible
export RL_MA_PROD_PLAYERS="1,2,3"
export RL_VISIBLE_RESET_MAX_ATTEMPTS=10
export RL_MA_PROD_SCENARIO_SET="defense"
export RL_MA_PROD_SCENARIO_WEIGHTS="defense_3v3_full:1,defense_1v3_static_entry:0,defense_2v3_dynamic:0,transition_midfield:0,defense_recovery_clear:0"

DATASET="$REPO/runs/teacher_multiagent_dataset/dataset_stage7_20260610_202948/samples.npz"  # tiny, BC off => load-only
DEFENDER="$REPO/models/defender/checkpoints/checkpoint_stage8_step2500.pt"
BASELINE="$REPO/runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt"

run_one () {
  local ckpt="$1" name="$2"
  echo "=================================================================="
  echo "[def3v3-perc] RUN $name  ckpt=$ckpt  scene=$RL_SCENE_NAME"
  echo "=================================================================="
  RL_MA_PROD_TRAIN_STEPS=256 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
  RL_MA_PROD_EVAL_INTERVAL=256 RL_MA_PROD_CHECKPOINT_INTERVAL=100000 \
  RL_MA_PROD_EVAL_EPISODES=12 \
  RL_MA_PROD_EXPERIMENT_NAME="$name" \
  RL_MA_PROD_BC_BETA_MAX=0.0 RL_MA_PROD_BC_BETA_MIN=0.0 \
  "$PY" train_multiagent_production.py --stage-id 8 \
    --checkpoint "$ckpt" --dataset "$DATASET"
  pkill -x SimRobot 2>/dev/null || true
  sleep 3
}

read_metric () {  # $1=exp name, $2=metric col -> prints value or empty
  "$PY" - "$1" "$2" <<'PY'
import sys, glob, csv
name, col = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f"runs/{name}/telemetry/*/csv/eval_summary.csv"))
if not fs:
    fs = [f"runs/{name}/eval_summary.csv"]
try:
    row = list(csv.DictReader(open(fs[-1])))[-1]
    print(row.get(col, ""))
except Exception:
    print("")
PY
}

run_one "$DEFENDER" "defense_3v3_perc_with_defender"

SEEN=$(read_metric defense_3v3_perc_with_defender natural_ball_seen_rate)
BLOCK=$(read_metric defense_3v3_perc_with_defender executed_block_rate)
MARK=$(read_metric defense_3v3_perc_with_defender executed_mark_rate)
echo "[def3v3-perc] GATE: natural_ball_seen=$SEEN executed_block=$BLOCK executed_mark=$MARK"
ENGAGED=$("$PY" -c "print(1 if (float('${SEEN:-0}' or 0)>0.05 or float('${BLOCK:-0}' or 0)>0 or float('${MARK:-0}' or 0)>0) else 0)")
if [ "$ENGAGED" != "1" ]; then
  echo "[def3v3-perc] ABORT: defenders never engaged (no perception/no block/mark). Skipping baseline."
  exit 0
fi

run_one "$BASELINE" "defense_3v3_perc_baseline_v2b"

echo "=================================================================="
echo "[def3v3-perc] DONE. Metrics:"
for n in defense_3v3_perc_with_defender defense_3v3_perc_baseline_v2b; do
  echo "--- $n"
  for m in goals_conceded_rate defender_score natural_ball_seen_rate executed_block_rate executed_mark_rate executed_dribble_rate collision_rate; do
    echo "    $m = $(read_metric $n $m)"
  done
done
