#!/usr/bin/env bash
# Stage-3 probe (cheap): does the OFF-BALL DEFENDER perceive a clean, isolated attacker
# ~1m in front via REAL vision? Single slot controlling player 2 (= teammate #2 = the deep
# defender per _world_player_pose), scenario defense_mark_open (lone attacker isolated in
# open space in the defender's gaze cone). GT obs is OFF on purpose — this measures real
# perception, the no-sim2real-gap check. If opponent_seen_rate stays ~0 here with a clean
# 1m target, the detector itself is the ceiling (decisive, before spending trio cycles).
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"

unset RL_MA_PROD_FORCE_PERCEPTION_SCENE RL_SCENE_NAME   # default stage-8 scene = RLMAStage7_3D
unset RL_TRAIN_GT_OPPONENT_OBS                          # REAL perception (GT off)
export RL_MA_PROD_PLAYERS="${RL_MA_PROD_PLAYERS:-2}"    # player 2 = the deep defender
export RL_OFFBALL_GAZE_DEBUG="${RL_OFFBALL_GAZE_DEBUG:-1}"  # per-step opp-percept counts (C++)
export RL_VISIBLE_RESET_MAX_ATTEMPTS=10
# Generous obs timeout: the visible sim can freeze a frame for ~2s at the train->eval
# transition (seen as stale min_frame==last_frame). 8s rides through transient stalls.
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=8000
export RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS=10000
# Single deep robot resets easily; keep the trio's relaxed acceptance for `gc playing` drift.
export RL_VISIBLE_RESET_POSE_TOL_MM=400
export RL_VISIBLE_RESET_MAX_FALL_STATE=3
export RL_VISIBLE_RESET_UPRIGHT_STREAK=2
export RL_MA_PROD_SCENARIO_SET="defense"
export RL_MA_PROD_SCENARIO_WEIGHTS="defense_mark_open:1,defense_3v3_full:0,defense_1v3_static_entry:0,defense_2v3_dynamic:0,transition_midfield:0,defense_recovery_clear:0"

DATASET="$REPO/runs/teacher_multiagent_dataset/dataset_stage7_20260610_202948/samples.npz"
DEFENDER="$REPO/models/defender/checkpoints/checkpoint_stage8_step2500.pt"

RL_MA_PROD_TRAIN_STEPS=256 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
RL_MA_PROD_EVAL_INTERVAL=256 RL_MA_PROD_CHECKPOINT_INTERVAL=100000 \
RL_MA_PROD_EVAL_EPISODES=4 \
RL_MA_PROD_EXPERIMENT_NAME="probe_mark_open_perc" \
RL_MA_PROD_BC_BETA_MAX=0.0 RL_MA_PROD_BC_BETA_MIN=0.0 \
"$PY" train_multiagent_production.py --stage-id 8 \
  --checkpoint "$DEFENDER" --dataset "$DATASET"
RC=$?
pkill -x SimRobot 2>/dev/null || true

echo "=================================================================="
echo "[probe-mark-open] exit_rc=$RC  Metrics:"
"$PY" - <<'PY'
import glob, csv
fs = sorted(glob.glob("runs/probe_mark_open_perc/telemetry/*/csv/eval_summary.csv"))
if not fs:
    print("  (no eval_summary.csv)")
else:
    row = list(csv.DictReader(open(fs[-1])))[-1]
    for m in ("natural_ball_seen_rate","opponent_seen_rate","executed_block_rate",
              "executed_mark_rate","policy_mark_rate","policy_block_rate",
              "goals_conceded_rate","defender_score","collision_rate"):
        print(f"    {m} = {row.get(m,'?')}")
PY
