#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/env.sh"
cd "$RL_REPO_DIR"
source .venv_env.sh
# stability script calls bare `python` — make it the venv interpreter
export PATH="$RL_DIR/.venv/bin:$PATH"
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=2500
export RL_VISIBLE_RESET_MAX_ATTEMPTS=12
pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3
bash tools/run/run_stability_chunked.sh \
  "$RL_REPO_DIR/runs/defender_47_v2/checkpoint_stage8_step2500.pt" \
  11 23 37 51 67 83
rc=$?
pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3
echo "[stability] probe exit rc=$rc"
exit $rc
