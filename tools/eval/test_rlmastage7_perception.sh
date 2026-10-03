#!/usr/bin/env bash
# Perception test in RLMAStage7_3D (7 real Nao bodies: robot1-4 team + robot21/22/23
# REAL opponents), now rebound to scenario 4v4_RL3D (BallPerceptor, not Trionda).
# Confirms BOTH: (1) ball perception works in sim (no YoloBallDetector crash;
# natural_ball_seen_rate>0) and (2) opponent perception works (opponent_seen_rate>0,
# executed_mark_rate / executed_block_rate). Default stage-8 scene (NO force escape).
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"

# NOTE: do NOT set RL_MA_PROD_FORCE_PERCEPTION_SCENE -> stage 8 uses default RLMAStage7_3D.
unset RL_MA_PROD_FORCE_PERCEPTION_SCENE RL_SCENE_NAME
# Single slot (7 robots, game-weight render) to isolate render-load vs scene-reset.
export RL_MA_PROD_PLAYERS="${RL_MA_PROD_PLAYERS:-1}"
export RL_VISIBLE_RESET_MAX_ATTEMPTS=10
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=2000
export RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS=8000
export RL_MA_PROD_SCENARIO_SET="defense"
export RL_MA_PROD_SCENARIO_WEIGHTS="defense_3v3_full:1,defense_1v3_static_entry:0,defense_2v3_dynamic:0,transition_midfield:0,defense_recovery_clear:0"

DATASET="$REPO/runs/teacher_multiagent_dataset/dataset_stage7_20260610_202948/samples.npz"
DEFENDER="$REPO/models/defender/checkpoints/checkpoint_stage8_step2500.pt"

RL_MA_PROD_TRAIN_STEPS=256 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
RL_MA_PROD_EVAL_INTERVAL=256 RL_MA_PROD_CHECKPOINT_INTERVAL=100000 \
RL_MA_PROD_EVAL_EPISODES=4 \
RL_MA_PROD_EXPERIMENT_NAME="test_rlmastage7_perc" \
RL_MA_PROD_BC_BETA_MAX=0.0 RL_MA_PROD_BC_BETA_MIN=0.0 \
"$PY" -m rlfootball.training.train_multiagent_production --stage-id 8 \
  --checkpoint "$DEFENDER" --dataset "$DATASET"
RC=$?
pkill -x SimRobot 2>/dev/null || true

echo "=================================================================="
echo "[rlmastage7-perc] exit_rc=$RC  Metrics:"
"$PY" - <<'PY'
import glob, csv
fs = sorted(glob.glob("runs/test_rlmastage7_perc/telemetry/*/csv/eval_summary.csv"))
if not fs:
    print("  (no eval_summary.csv)")
else:
    row = list(csv.DictReader(open(fs[-1])))[-1]
    for m in ("natural_ball_seen_rate","opponent_seen_rate","executed_block_rate",
              "executed_mark_rate","policy_mark_rate","policy_block_rate",
              "goals_conceded_rate","defender_score"):
        print(f"    {m} = {row.get(m,'?')}")
PY
