#!/usr/bin/env bash
# v3.1 pipeline: v2b BACKBONE + teammate-varying contrast OVERLAY (req 8).
#   gate    — pre-train teacher-flip gate (no sim)
#   overlay — regen ONLY the teammate-varying contrast scenarios + merge+audit them (sim)
#   merge   — subsample v2b backbone + overlay -> final corpus + corpus-property gate
#   train   — fine-tune from v2b@7500 on the final corpus
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
export RL_MA_TEACHER_VISIBLE_SCENE_NAME=RL1v0_3D_visible   # single-robot per slot (v3 scene-bug fix)
export RL_VISIBLE_RESET_MAX_ATTEMPTS=2                     # fail-fast on wedges
export RL_MA_SHOOT_THRESHOLD=0.30
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
TM=runs/teacher_multiagent_dataset
OVERLAY_ROOT="$TM/v31_overlay_chunks"
OVERLAY="$TM/dataset_v31_overlay"
V2B="$TM/dataset_multiagent_train_20260611_full_v1"
V2B_SUB="$TM/dataset_v2b_subsample_for_v31"
FINAL="$TM/dataset_v31_backbone_plus_overlay"
CONTRAST="giveandgo_outlet_present,giveandgo_outlet_absent,support_striker_live,support_striker_far"
cd "$REPO"

case "${1:-}" in
  gate) "$PY" tools/eval/teammate_flip_gate.py ;;
  overlay)
    "$PY" tools/eval/teammate_flip_gate.py || { echo "[v31] teammate-flip gate FAILED — abort"; exit 1; }
    rm -rf "$OVERLAY_ROOT"; export RL_MA_TEACHER_OUTPUT_ROOT="$REPO/$OVERLAY_ROOT"
    for i in 1 2 3 4; do
      echo "[v31] === overlay chunk $i ==="
      RL_MA_TEACHER_STAGE=7 RL_MA_TEACHER_MAX_STEPS=800 RL_MA_TEACHER_EPISODES=4 \
      RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=8 RL_MA_TEACHER_SEED_BASE=$((810000 + i*10000)) \
      RL_MA_TEACHER_SCENARIOS="$CONTRAST" PYBH_RL_SHM_NAME=/pybh_rl_v31_${i} \
      "$PY" -m rlfootball.datasets.teacher_multiagent_dataset
      pkill -f SimRobot 2>/dev/null; pkill -f Xvfb 2>/dev/null; sleep 5
    done
    "$PY" -m rlfootball.datasets.merge_multiagent_teacher_datasets "$OVERLAY_ROOT"/* --output-dir "$OVERLAY"
    "$PY" -m rlfootball.datasets.teacher_multiagent_dataset_audit "$OVERLAY" --min-striker-shoot-frac 0 --min-striker-pass-armed 0 --write || true
    echo "[v31] overlay done -> $OVERLAY"
    ;;
  merge)
    N=$("$PY" -c "import numpy as np;print(len(np.load('$OVERLAY/samples.npz',allow_pickle=True)['weights']))")
    SUB=$(( N * 3 ))   # overlay ~25% of the final corpus; v2b backbone ~75% (preserves floor/3v3/stand%)
    echo "[v31] overlay rows=$N → v2b subsample target=$SUB"
    "$PY" -m rlfootball.datasets.subsample_multiagent_attack_dataset "$V2B" --output-dir "$V2B_SUB" --target-rows "$SUB"
    "$PY" -m rlfootball.datasets.merge_multiagent_teacher_datasets "$V2B_SUB" "$OVERLAY" --output-dir "$FINAL"
    "$PY" -m rlfootball.datasets.teacher_multiagent_dataset_audit "$FINAL" --write || true
    "$PY" tools/dataset/v31_corpus_gate.py "$FINAL"   # reqs 3,4,5,6 — exits nonzero if corpus fails
    ;;
  train)
    RL_MA_PROD_TRAIN_STEPS=10000 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
    RL_MA_PROD_EVAL_INTERVAL=2500 RL_MA_PROD_CHECKPOINT_INTERVAL=2500 \
    RL_MA_PROD_SCENARIO_SET=team RL_MA_PROD_EXPERIMENT_NAME=multiagent_v31 \
    "$PY" -m rlfootball.training.train_multiagent_production --stage-id 7 \
      --checkpoint runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt \
      --dataset "$FINAL/samples.npz" ;;
  *) echo "usage: bash tools/run/run_v31.sh {gate|overlay|merge|train}"; exit 2;;
esac
