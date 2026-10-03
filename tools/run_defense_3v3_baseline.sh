#!/usr/bin/env bash
# Rubric (g): defense_3v3_full goals-conceded A/B.
#   A = defender step2500   vs   B = bare v2b@7500 trio (no defender)
# Same scenario, same seeds (TRAIN.seed + episode_index), pure-policy (BC off),
# real Design-A gate applied. Sequential single-sim. Reads goals_conceded_rate
# + defender_* from each run's eval_summary.csv.
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"

# 3 RL field defenders (the "trío") + BH GK — rubric (g) is the trio, not 1 slot.
export RL_MA_PROD_PLAYERS="1,2,3"
export RL_VISIBLE_RESET_MAX_ATTEMPTS=10
# defense_3v3_full ONLY (other defense scenarios zeroed)
export RL_MA_PROD_SCENARIO_SET="defense"
export RL_MA_PROD_SCENARIO_WEIGHTS="defense_3v3_full:1,defense_1v3_static_entry:0,defense_2v3_dynamic:0,transition_midfield:0,defense_recovery_clear:0"

DATASET="$REPO/runs/teacher_multiagent_dataset/dataset_stage7_20260610_202948/samples.npz"  # tiny, BC off => load-only
DEFENDER="$REPO/models/defender/checkpoints/checkpoint_stage8_step2500.pt"
BASELINE="$REPO/runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt"

run_one () {
  local ckpt="$1" name="$2"
  echo "=================================================================="
  echo "[defense3v3] RUN $name  ckpt=$ckpt"
  echo "=================================================================="
  RL_MA_PROD_TRAIN_STEPS=256 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
  RL_MA_PROD_EVAL_INTERVAL=256 RL_MA_PROD_CHECKPOINT_INTERVAL=100000 \
  RL_MA_PROD_EVAL_EPISODES=12 \
  RL_MA_PROD_EXPERIMENT_NAME="$name" \
  RL_MA_PROD_BC_BETA_MAX=0.0 RL_MA_PROD_BC_BETA_MIN=0.0 \
  "$PY" train_multiagent_production.py --stage-id 8 \
    --checkpoint "$ckpt" --dataset "$DATASET"
  # hard teardown so the next run owns the single sim session
  pkill -x SimRobot 2>/dev/null || true
  sleep 3
}

run_one "$DEFENDER" "defense_3v3_with_defender"
run_one "$BASELINE" "defense_3v3_baseline_v2b"

echo "=================================================================="
echo "[defense3v3] DONE. Summaries:"
for n in defense_3v3_with_defender defense_3v3_baseline_v2b; do
  f="$REPO/runs/$n/eval_summary.csv"
  echo "--- $n : $f"
  [ -f "$f" ] && tail -1 "$f" || echo "  (missing)"
done
