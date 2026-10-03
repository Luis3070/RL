#!/usr/bin/env bash
set -u
cd /home/bryamdigar/Projects/Programacion/Semillero/RL
source .venv_env.sh
pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3
export PYBH_RL_SHM_NAME="/pybh_rl_defv2_floor"
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=2500
export RL_VISIBLE_RESET_MAX_ATTEMPTS=12
RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20 \
  "$PY" -m rlfootball.evaluation.behavioral_evaluator \
  --checkpoint /home/bryamdigar/Projects/Programacion/Semillero/RL/runs/defender_47_v2/checkpoint_stage8_step2500.pt \
  --episodes 8
rc=$?
pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3
echo "[floor] eval exit rc=$rc"
exit $rc
