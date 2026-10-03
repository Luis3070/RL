#!/usr/bin/env bash
set -u
cd /home/bryamdigar/Projects/Programacion/Semillero/RL
source .venv_env.sh
pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3
export PYBH_RL_SHM_NAME="/pybh_rl_defv2_finetune"
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=2500
# Software-render getup is flaky; give resets more chances to land a clean upright pose.
export RL_VISIBLE_RESET_MAX_ATTEMPTS=12
RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20 \
RL_MA_TEAM_GOAL_BONUS=2.5 RL_MA_TEAM_GOAL_CONCEDED_PENALTY=2.5 \
RL_MA_PROD_TRAIN_STEPS=10000 RL_MA_PROD_MAX_EPISODE_STEPS=600 \
RL_MA_PROD_EVAL_INTERVAL=2500 RL_MA_PROD_CHECKPOINT_INTERVAL=2500 \
RL_MA_PROD_SCENARIO_SET=defense RL_MA_PROD_EXPERIMENT_NAME=defender_47_v2 \
  "$PY" -m rlfootball.training.train_multiagent_production --stage-id 8 \
  --checkpoint /home/bryamdigar/Projects/Programacion/Semillero/RL/runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt \
  --dataset /home/bryamdigar/Projects/Programacion/Semillero/RL/runs/teacher_multiagent_dataset/dataset_defense_plus_attack_v2/samples.npz
rc=$?
pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3
echo "[finetune] train exit rc=$rc"
exit $rc
