#!/usr/bin/env bash
# Smoke: does the noisy-oracle obstacle perception (4v4_RL3D/threads.cfg now routes
# ObstaclesFieldPercept/ImagePercept -> OracledPerceptsProvider) actually START and
# populate opponents? 1 defender, defense scenario, perception scene, tiny.
# PASS signals: sim reaches obs_ready (eval runs) AND executed_mark_rate or opponent
# perception > 0. FAIL signals: "GroundTruthWorldState" provider error / no obs_ready.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/env.sh"
REPO="$RL_REPO_DIR"
export PYTHONPATH="$REPO:$SABANAHERONS_DIR/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"

export RL_MA_PROD_FORCE_PERCEPTION_SCENE=1
export RL_SCENE_NAME=RL1v0_3D_visible
export RL_MA_PROD_PLAYERS="1"
export RL_VISIBLE_RESET_MAX_ATTEMPTS=10
export RL_MA_PROD_SCENARIO_SET="defense"
export RL_MA_PROD_SCENARIO_WEIGHTS="defense_1v3_static_entry:1,defense_3v3_full:0,defense_2v3_dynamic:0,transition_midfield:0,defense_recovery_clear:0"

DATASET="$REPO/runs/teacher_multiagent_dataset/dataset_stage7_20260610_202948/samples.npz"
DEFENDER="$REPO/models/defender/checkpoints/checkpoint_stage8_step2500.pt"

RL_MA_PROD_TRAIN_STEPS=64 RL_MA_PROD_MAX_EPISODE_STEPS=400 \
RL_MA_PROD_EVAL_INTERVAL=64 RL_MA_PROD_CHECKPOINT_INTERVAL=100000 \
RL_MA_PROD_EVAL_EPISODES=3 \
RL_MA_PROD_EXPERIMENT_NAME="smoke_oracle_perc" \
RL_MA_PROD_BC_BETA_MAX=0.0 RL_MA_PROD_BC_BETA_MIN=0.0 \
"$PY" -m rlfootball.training.train_multiagent_production --stage-id 8 \
  --checkpoint "$DEFENDER" --dataset "$DATASET"
RC=$?
pkill -x SimRobot 2>/dev/null || true
echo "[smoke] exit_rc=$RC"
