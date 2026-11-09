#!/usr/bin/env bash
set -u
cd /home/bryamdigar/Projects/Programacion/Semillero/RL
source .venv_env.sh
export PATH="$RL_DIR/.venv/bin:$PATH"
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=2500
export RL_VISIBLE_RESET_MAX_ATTEMPTS=12
pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3
bash tools/run_stability_chunked.sh \
  /home/bryamdigar/Projects/Programacion/Semillero/RL/runs/defender_47_v2/checkpoint_stage8_step2500.pt \
  67
rc=$?
pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3
echo "[stability-67] exit rc=$rc"
exit $rc
