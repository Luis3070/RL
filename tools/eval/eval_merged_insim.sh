#!/usr/bin/env bash
# In-sim single-slot behavioral eval of the merged brain (as-is: BC off, tiny 256-step
# nudge then eval). STAGE + SCENARIO_SET picked by env. Single slot = the stable config;
# 3-slot multi-agent is an unusable instrument (documented). Executed mark/block need the
# off-ball role (multi-agent) so they read ~0 here by construction — see the OFFLINE battery
# (mark recall 93%) for that. This run is for: goal/shoot/dribble/ball-handling (striker),
# defender_score/cover/goals_conceded (defender), + collision/stability/completion (both).
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"
STAGE="${STAGE:-7}"; NAME="${NAME:-eval_merged_s${STAGE}}"
unset RL_MA_PROD_FORCE_PERCEPTION_SCENE RL_SCENE_NAME
export RL_MA_PROD_PLAYERS="${RL_MA_PROD_PLAYERS:-1}"
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=8000
export RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS=12000
export RL_VISIBLE_RESET_MAX_ATTEMPTS=12
export RL_VISIBLE_RESET_POSE_TOL_MM=400
export RL_VISIBLE_RESET_MAX_FALL_STATE=3
export RL_VISIBLE_RESET_UPRIGHT_STREAK=2
[ -n "${SCENARIO_SET:-}" ] && export RL_MA_PROD_SCENARIO_SET="$SCENARIO_SET"
[ -n "${SCENARIO_WEIGHTS:-}" ] && export RL_MA_PROD_SCENARIO_WEIGHTS="$SCENARIO_WEIGHTS"
export RL_MA_PROD_TRAIN_STEPS=256 RL_MA_PROD_EVAL_INTERVAL=256 RL_MA_PROD_CHECKPOINT_INTERVAL=100000
export RL_MA_PROD_EVAL_EPISODES="${EVAL_EPISODES:-12}"
export RL_MA_PROD_BC_BETA_MAX=0.0 RL_MA_PROD_BC_BETA_MIN=0.0
export RL_MA_PROD_EXPERIMENT_NAME="$NAME"
CKPT="$REPO/runs/merge_v5_attack_defense/checkpoint_stage8_step10000.pt"
DATASET="$REPO/runs/teacher_multiagent_dataset/dataset_merged_v5_20260618/samples.npz"
"$PY" train_multiagent_production.py --stage-id "$STAGE" --checkpoint "$CKPT" --dataset "$DATASET"
RC=$?
pkill -x SimRobot 2>/dev/null || true
echo "=================================================================="
echo "[$NAME] exit_rc=$RC  Metrics:"
"$PY" - "$NAME" <<'PY'
import glob, csv, sys
name=sys.argv[1]
fs=sorted(glob.glob(f"runs/{name}/telemetry/*/csv/eval_summary.csv"))
if not fs: print("  (no eval_summary.csv)"); raise SystemExit
row=list(csv.DictReader(open(fs[-1])))[-1]
keys=("natural_ball_seen_rate","opponent_seen_rate","goal_rate","executed_shoot_rate","policy_shoot_rate",
      "executed_dribble_rate","executed_pass_rate","executed_block_rate","executed_mark_rate",
      "defender_score","goals_conceded_rate","collision_rate","ball_trusted_rate","stability",
      "policy_incompatible_takeover_rate","mean_episode_steps","reward_mean")
for k in keys:
    if k in row: print(f"    {k} = {row[k]}")
PY
