#!/usr/bin/env bash
# Mark test: trío (players 1,2,3 = real defender role) in RLMAStage7_3D (7 real
# robots, real perceived opponents — ball 99% / opp 80% confirmed single-slot).
# Perception fixed (grayscale BallPerceptor + .con). Generous reset budgets for
# the 21-robot (3 slots x 7) load + fall-on-reset retries. Reports block/mark + perception.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/env.sh"
REPO="$RL_REPO_DIR"
export PYTHONPATH="$REPO:$SABANAHERONS_DIR/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"

unset RL_MA_PROD_FORCE_PERCEPTION_SCENE RL_SCENE_NAME   # default stage-8 scene = RLMAStage7_3D
export RL_MA_PROD_PLAYERS="${RL_MA_PROD_PLAYERS:-1,2,3}"
export RL_VISIBLE_RESET_MAX_ATTEMPTS=15
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=3000
export RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS=12000
# Relax reset acceptance: in a `gc playing` scene robots walk ~200mm off their
# teleport pose before settling, so the default 150mm/streak-3 gate over-rejects.
# We only need approximate poses + opponents perceived for the mark test.
export RL_VISIBLE_RESET_POSE_TOL_MM=400
export RL_VISIBLE_RESET_MAX_FALL_STATE=3
export RL_VISIBLE_RESET_UPRIGHT_STREAK=2
export RL_MA_PROD_SCENARIO_SET="defense"
export RL_MA_PROD_SCENARIO_WEIGHTS="defense_3v3_full:1,defense_1v3_static_entry:0,defense_2v3_dynamic:0,transition_midfield:0,defense_recovery_clear:0"

DATASET="$REPO/runs/teacher_multiagent_dataset/dataset_stage7_20260610_202948/samples.npz"
DEFENDER="$REPO/models/defender/checkpoints/checkpoint_stage8_step2500.pt"

RL_MA_PROD_TRAIN_STEPS=256 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
RL_MA_PROD_EVAL_INTERVAL=256 RL_MA_PROD_CHECKPOINT_INTERVAL=100000 \
RL_MA_PROD_EVAL_EPISODES=8 \
RL_MA_PROD_EXPERIMENT_NAME="test_rlmastage7_trio" \
RL_MA_PROD_BC_BETA_MAX=0.0 RL_MA_PROD_BC_BETA_MIN=0.0 \
"$PY" -m rlfootball.training.train_multiagent_production --stage-id 8 \
  --checkpoint "$DEFENDER" --dataset "$DATASET"
RC=$?
pkill -x SimRobot 2>/dev/null || true

echo "=================================================================="
echo "[rlmastage7-trio] exit_rc=$RC  Metrics:"
"$PY" - <<'PY'
import glob, csv
fs = sorted(glob.glob("runs/test_rlmastage7_trio/telemetry/*/csv/eval_summary.csv"))
if not fs:
    print("  (no eval_summary.csv)")
else:
    row = list(csv.DictReader(open(fs[-1])))[-1]
    for m in ("natural_ball_seen_rate","opponent_seen_rate","policy_block_rate","executed_block_rate",
              "policy_mark_rate","executed_mark_rate","goals_conceded_rate","defender_score",
              "executed_dribble_rate","collision_rate"):
        print(f"    {m} = {row.get(m,'?')}")
PY
