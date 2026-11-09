#!/usr/bin/env bash
# GK v1 fine-tune launcher (stage 9, single GK slot, RL_GATE_GK_MODE=1).
# Warm-starts from defender step2500 checkpoint; uses GK BC dataset.
# Run from repo root: bash tools/run_gkv1_finetune.sh
set -u
cd /home/bryamdigar/Projects/Programacion/Semillero/RL
source .venv_env.sh
pkill -f "teacher_goalkeeper_dataset" 2>/dev/null
pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3
export PYBH_RL_SHM_NAME="/pybh_rl_gkv1_finetune"
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=2500
export RL_VISIBLE_RESET_MAX_ATTEMPTS=12
export RL_GATE_GK_MODE=1
export RL_SCENE_NAME=RLMAStage7_3D
RL_GK_TRAIN_STEPS=15000 \
RL_GK_MAX_EPISODE_STEPS=600 \
RL_GK_EVAL_INTERVAL=2500 \
RL_GK_CHECKPOINT_INTERVAL=2500 \
RL_GK_BC_BETA_MAX=0.12 \
RL_GK_BC_BETA_MIN=0.02 \
RL_GK_BC_HALF_LIFE_EVALS=20 \
RL_GK_BC_REPLAY_WEIGHT=0.12 \
RL_GK_BC_SKILL_LOSS_SCALE=0.50 \
RL_GK_BC_PARAM_LOSS_SCALE=0.50 \
RL_GK_EXPERIMENT_NAME=goalkeeper_gkv1 \
  "$PY" train_goalkeeper.py \
  --checkpoint /home/bryamdigar/Projects/Programacion/Semillero/RL/runs/defender_47_v2/checkpoint_stage8_step2500.pt
rc=$?
pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3
echo "[gkv1-finetune] train exit rc=$rc"
exit $rc
