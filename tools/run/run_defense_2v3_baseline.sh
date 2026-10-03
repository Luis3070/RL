#!/usr/bin/env bash
# Rubric (g) v2 on defense_2v3_dynamic (engagement-forcing: ball -2500..-800,
# 2 DYNAMIC attackers driving at goal). Trío (3 RL field) + BH GK, same seeds,
# pure-policy (BC off), real Design-A gate. GATED: run defender first; only run
# the v2b baseline if the defender eval actually saw the ball (natural_ball_seen
# > 0.05), else abort as "scene did not engage" to avoid a second dead run.
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"

export RL_MA_PROD_PLAYERS="1,2,3"
export RL_VISIBLE_RESET_MAX_ATTEMPTS=10
export RL_MA_PROD_SCENARIO_SET="defense"
export RL_MA_PROD_SCENARIO_WEIGHTS="defense_2v3_dynamic:1,defense_1v3_static_entry:0,defense_3v3_full:0,transition_midfield:0,defense_recovery_clear:0"

DATASET="$REPO/runs/teacher_multiagent_dataset/dataset_stage7_20260610_202948/samples.npz"
DEFENDER="$REPO/models/defender/checkpoints/checkpoint_stage8_step2500.pt"
BASELINE="$REPO/runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt"

run_one () {
  local ckpt="$1" name="$2"
  echo "=================================================================="
  echo "[def2v3] RUN $name  ckpt=$ckpt"
  echo "=================================================================="
  RL_MA_PROD_TRAIN_STEPS=256 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
  RL_MA_PROD_EVAL_INTERVAL=256 RL_MA_PROD_CHECKPOINT_INTERVAL=100000 \
  RL_MA_PROD_EVAL_EPISODES=12 \
  RL_MA_PROD_EXPERIMENT_NAME="$name" \
  RL_MA_PROD_BC_BETA_MAX=0.0 RL_MA_PROD_BC_BETA_MIN=0.0 \
  "$PY" -m rlfootball.training.train_multiagent_production --stage-id 8 \
    --checkpoint "$ckpt" --dataset "$DATASET"
  pkill -x SimRobot 2>/dev/null || true
  sleep 3
}

read_metric () {  # $1=exp name, $2=metric col -> prints value or empty
  "$PY" - "$1" "$2" <<'PY'
import sys, glob, csv
name, col = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f"runs/{name}/telemetry/*/csv/eval_summary.csv"))
if not fs: print(""); sys.exit()
row = list(csv.DictReader(open(fs[-1])))[-1]
print(row.get(col, ""))
PY
}

run_one "$DEFENDER" "defense_2v3_with_defender"

SEEN=$(read_metric defense_2v3_with_defender natural_ball_seen_rate)
BLOCK=$(read_metric defense_2v3_with_defender executed_block_rate)
DRIB=$(read_metric defense_2v3_with_defender executed_dribble_rate)
echo "[def2v3] GATE: natural_ball_seen=$SEEN executed_block=$BLOCK executed_dribble=$DRIB"
ENGAGED=$("$PY" -c "print(1 if (float('${SEEN:-0}' or 0)>0.05 or float('${BLOCK:-0}' or 0)>0 or float('${DRIB:-0}' or 0)>0) else 0)")
if [ "$ENGAGED" != "1" ]; then
  echo "[def2v3] ABORT: defenders never engaged the ball (natural_ball_seen<=0.05, block/dribble=0)."
  echo "[def2v3] defense_2v3_dynamic also does not exercise the defender on this setup. Skipping baseline."
  exit 0
fi

run_one "$BASELINE" "defense_2v3_baseline_v2b"

echo "=================================================================="
echo "[def2v3] DONE. goals_conceded_rate / defender_score / natural_ball_seen / block / dribble:"
for n in defense_2v3_with_defender defense_2v3_baseline_v2b; do
  echo "--- $n"
  for m in goals_conceded_rate defender_score natural_ball_seen_rate executed_block_rate executed_dribble_rate collision_rate; do
    echo "    $m = $(read_metric $n $m)"
  done
done
