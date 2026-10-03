#!/usr/bin/env bash
# v4 pipeline = reward + weighting + scenarios + two-phase, TOGETHER (data-only failed twice).
#   gate    — teammate-flip pre-train gate (no sim)
#   overlay — regen the FINAL chunks: pass-forcing + support-receive + teammate contrast (sim)
#   merge   — v2b backbone (subsampled) + v4 overlay -> corpus + corpus-property gate
#   train   — BC-DOMINANT warmup (high beta, decision-window batches, pass-heavy) then PPO,
#             with the completed-pass reward live. Pass is alive before PPO takes over.
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
export RL_MA_TEACHER_VISIBLE_SCENE_NAME=RL1v0_3D_visible
export RL_VISIBLE_RESET_MAX_ATTEMPTS=${RL_VISIBLE_RESET_MAX_ATTEMPTS:-2}  # overridable: 2=fail-fast for dataset gen, raise for training resilience
export RL_MA_SHOOT_THRESHOLD=0.30
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
TM=runs/teacher_multiagent_dataset
OV_ROOT="$TM/v4_overlay_chunks"
OVERLAY="$TM/dataset_v4_overlay"
V2B="$TM/dataset_multiagent_train_20260611_full_v1"
V2B_SUB="$TM/dataset_v2b_subsample_for_v4"
FINAL="$TM/dataset_v4_backbone_plus_overlay"
# FINAL chunks = pass-forcing + support-receive + teammate contrast (per Luis: not general contrast)
OVERLAY_SCN="pass_forcing,support_striker_live,support_striker_far,giveandgo_outlet_present,giveandgo_outlet_absent"
cd "$REPO"

case "${1:-}" in
  gate) "$PY" tools/eval/teammate_flip_gate.py ;;
  overlay)
    "$PY" tools/eval/teammate_flip_gate.py || { echo "[v4] teammate-flip gate FAILED — abort"; exit 1; }
    rm -rf "$OV_ROOT"; export RL_MA_TEACHER_OUTPUT_ROOT="$REPO/$OV_ROOT"
    for i in 1 2; do   # capped at 2 chunks (3-RL falls make it slow); pass-rich scenarios
      echo "[v4] === overlay chunk $i ==="
      RL_MA_TEACHER_STAGE=7 RL_MA_TEACHER_MAX_STEPS=800 RL_MA_TEACHER_EPISODES=4 \
      RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=8 RL_MA_TEACHER_SEED_BASE=$((840000 + i*10000)) \
      RL_MA_TEACHER_SCENARIOS="$OVERLAY_SCN" PYBH_RL_SHM_NAME=/pybh_rl_v4_${i} \
      "$PY" -m rlfootball.datasets.teacher_multiagent_dataset
      pkill -f SimRobot 2>/dev/null; pkill -f Xvfb 2>/dev/null; sleep 5
    done
    "$PY" -m rlfootball.datasets.merge_multiagent_teacher_datasets "$OV_ROOT"/*/  --output-dir "$OVERLAY" 2>/dev/null \
      || "$PY" -m rlfootball.datasets.merge_multiagent_teacher_datasets $(ls -d "$OV_ROOT"/*/ | while read d; do [ -f "$d/samples.npz" ] && echo "$d"; done) --output-dir "$OVERLAY"
    "$PY" -m rlfootball.datasets.teacher_multiagent_dataset_audit "$OVERLAY" --min-striker-shoot-frac 0 --min-striker-pass-armed 0 --write || true
    echo "[v4] overlay done -> $OVERLAY"
    ;;
  merge)
    N=$("$PY" -c "import numpy as np;print(len(np.load('$OVERLAY/samples.npz',allow_pickle=True)['weights']))")
    SUB=$(( N * 3 ))
    "$PY" -m rlfootball.datasets.subsample_multiagent_attack_dataset "$V2B" --output-dir "$V2B_SUB" --target-rows "$SUB"
    "$PY" -m rlfootball.datasets.merge_multiagent_teacher_datasets "$V2B_SUB" "$OVERLAY" --output-dir "$FINAL"
    "$PY" -m rlfootball.datasets.teacher_multiagent_dataset_audit "$FINAL" --write || true
    "$PY" tools/dataset/v31_corpus_gate.py "$FINAL"
    ;;
  train)
    # BC-DOMINANT warmup (high beta, slow decay, more BC updates, decision-window batches,
    # pass-heavy) so pass-probability comes alive from BC BEFORE PPO; completed-pass reward
    # is live (env defaults in team_parallel_environment). Then beta decays -> PPO takes over.
    RL_MA_PROD_TRAIN_STEPS=12000 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
    RL_MA_PROD_EVAL_INTERVAL=3000 RL_MA_PROD_CHECKPOINT_INTERVAL=3000 \
    RL_MA_PROD_SCENARIO_SET=team RL_MA_PROD_EXPERIMENT_NAME=multiagent_v4 \
    RL_MA_PROD_BC_BETA_MAX=0.40 RL_MA_PROD_BC_BETA_MIN=0.05 \
    RL_MA_PROD_BC_HALF_LIFE_EVALS=40 RL_MA_PROD_BC_REPLAY_UPDATES=4 \
    RL_MA_BC_DECISION_WINDOW_ONLY=1 RL_MA_BC_STRIKER_PASS_FRAC=0.30 \
    "$PY" -m rlfootball.training.train_multiagent_production --stage-id 7 \
      --checkpoint runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt \
      --dataset "$FINAL/samples.npz" ;;
  *) echo "usage: bash tools/run/run_v4.sh {gate|overlay|merge|train}"; exit 2;;
esac
