#!/usr/bin/env bash
# MULTI-AGENT MATCH EVAL of the merged brain. Drives ALL 3 RL agents (players 1,2,3) with
# the SAME merged policy in ONE shared SimRobot (RLMAStage7_3D, 7 real bodies = 3 RL + BH GK
# + 3 opponents) via the STABLE Pattern-A orchestration (VISIBLE_PARALLEL_SLOTS=0) — NOT the
# unusable parallel-slots eval. Scenario team_3v3_full_dynamic = full-field contest: the team
# strikes when it has the ball and defends when opponents do, so roles flip live (cooperation).
# GT opponent obs ON so the off-ball defender reliably perceives opponents (real vision
# collapses off-ball in multi-agent; GT is the training+close-range-deploy regime). Logs role
# co-occurrence, ball progression, passes, scored/conceded, per-role skill mix.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/env.sh"
REPO="$RL_REPO_DIR"
export PYTHONPATH="$REPO:$SABANAHERONS_DIR/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"

export RL_MA_EVAL_POLICY="$REPO/runs/merge_v5_attack_defense/checkpoint_stage8_step10000.pt"
export RL_TRAIN_GT_OPPONENT_OBS="${RL_TRAIN_GT_OPPONENT_OBS:-1}"
export RL_MA_TEACHER_STAGE="${RL_MA_TEACHER_STAGE:-8}"
export RL_MA_TEACHER_VISIBLE_PARALLEL_SLOTS=0     # shared sim (stable), NOT parallel slots
# Survivable multi-agent set (excludes the 7-active-robot 3v3 that crashes the visible sim):
# transition (midfield role-flips), 2v3 defense, recovery->clear (defense->attack-out),
# mark_open. Together they exercise BOTH strike (recover/drive/pass-out) and defense.
export RL_MA_TEACHER_SCENARIOS="${RL_MA_TEACHER_SCENARIOS:-transition_midfield,defense_2v3_dynamic,defense_recovery_clear,defense_mark_open}"
export RL_MA_TEACHER_EPISODES="${RL_MA_TEACHER_EPISODES:-3}"
export RL_MA_TEACHER_MAX_STEPS="${RL_MA_TEACHER_MAX_STEPS:-300}"
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=8000
export RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS=12000
export RL_VISIBLE_RESET_MAX_ATTEMPTS=12
export RL_VISIBLE_RESET_POSE_TOL_MM=400
export RL_VISIBLE_RESET_MAX_FALL_STATE=3
export RL_VISIBLE_RESET_UPRIGHT_STREAK=2
OUT="$REPO/runs/tmp_match_eval"; rm -rf "$OUT"; mkdir -p "$OUT"
export RL_MA_TEACHER_OUTPUT_ROOT="$OUT"

"$PY" -m rlfootball.datasets.teacher_multiagent_dataset
RC=$?
pkill -x SimRobot 2>/dev/null || true
echo "=================================================================="
echo "[match-eval] exit_rc=$RC"
grep -A30 "MULTI-AGENT MATCH COOPERATION SUMMARY" "$OUT"/**/eval_match_summary.json 2>/dev/null || true
