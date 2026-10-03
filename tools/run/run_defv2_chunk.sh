#!/usr/bin/env bash
# doc-34 Step 1 — one dataset-generation chunk. Usage: run_defv2_chunk.sh <i>
# Designed to be run in the background; writes partial npz per scenario so a
# SimRobot freeze loses at most the in-progress scenario.
set -u
i="${1:?chunk index required}"
cd /home/bryamdigar/Projects/Programacion/Semillero/RL
source .venv_env.sh
pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3

export PYBH_RL_SHM_NAME="/pybh_rl_defv2_chunk_${i}"
export RL_MA_TEACHER_COLLISION_SUSTAIN_STEPS=16
# Software-render (llvmpipe) on this laptop is slower than the reference box;
# widen the per-step obs deadline to reduce spurious stale-frame episode drops.
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=2500

RL_MA_TEACHER_STAGE=8 \
RL_MA_TEACHER_MAX_STEPS=600 \
RL_MA_TEACHER_EPISODES=8 \
RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=6 \
RL_MA_TEACHER_SEED_BASE=$((970000 + i*10000)) \
  "$PY" teacher_multiagent_dataset.py
rc=$?

pkill -x SimRobot 2>/dev/null; pkill -x Xvfb 2>/dev/null; sleep 3
echo "[chunk $i] teacher exit rc=$rc"
exit $rc
