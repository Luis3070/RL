#!/usr/bin/env bash
# Stage-4 merge fine-tune: ONE 47-dim brain = v4.2 attack + defense.
#  - warm-start AND KL anchor = v4.2 step12000 (KL ref == warm-start ckpt, train.py:2102),
#    so attack quality is protected.
#  - BC replay on the MERGED corpus (attack 105.6k + GT-defense 34k) imprints block/mark/
#    positioning while striker BC + KL keep the 0.99 pass alive. BC left at trainer
#    defaults (beta 0.08->0.02) — do NOT zero it (that was only for the eval probes).
#  - SINGLE visible slot (stage-8 default) = the stable config (3-slot starves/crashes).
#    Defense is gate+BC-driven (PPO defense fine-tune was historically a no-op), so single
#    slot is correct, not a compromise.
#  - PPO rollouts on the 5 STABLE defense scenarios (defense_3v3_full excluded — it hangs
#    the visible sim). Attack retention comes from BC(attack rows)+KL, not PPO.
# Exports nothing; writes checkpoints to runs/merge_v5_attack_defense. NEW ONNX name only
# after eval gates pass. Touches no deployed checkpoint.
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"

unset RL_MA_PROD_FORCE_PERCEPTION_SCENE RL_SCENE_NAME   # stage-8 default scene RLMAStage7_3D
export RL_MA_PROD_PLAYERS="${RL_MA_PROD_PLAYERS:-1}"     # single stable slot
# stall resilience (visible sim can freeze a frame ~2s at train->eval transitions)
export RL_VISIBLE_STEP_OBS_TIMEOUT_MS=8000
export RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS=12000
export RL_VISIBLE_RESET_MAX_ATTEMPTS=12
export RL_VISIBLE_RESET_POSE_TOL_MM=400
export RL_VISIBLE_RESET_MAX_FALL_STATE=3
export RL_VISIBLE_RESET_UPRIGHT_STREAK=2
export RL_MA_PROD_SCENARIO_SET="defense"
export RL_MA_PROD_SCENARIO_WEIGHTS="defense_mark_open:2,defense_1v3_static_entry:1,defense_2v3_dynamic:2,transition_midfield:1,defense_recovery_clear:2,defense_3v3_full:0"

export RL_MA_PROD_TRAIN_STEPS="${RL_MA_PROD_TRAIN_STEPS:-10000}"
export RL_MA_PROD_EVAL_INTERVAL="${RL_MA_PROD_EVAL_INTERVAL:-5000}"
export RL_MA_PROD_CHECKPOINT_INTERVAL="${RL_MA_PROD_CHECKPOINT_INTERVAL:-2500}"
export RL_MA_PROD_EVAL_EPISODES="${RL_MA_PROD_EVAL_EPISODES:-8}"
export RL_MA_PROD_EXPERIMENT_NAME="merge_v5_attack_defense"

WARM="$REPO/runs/multiagent_v4_2/checkpoint_stage7_step12000.pt"   # warm-start + KL anchor
DATASET="$REPO/runs/teacher_multiagent_dataset/dataset_merged_v5_20260618/samples.npz"

"$PY" -m rlfootball.training.train_multiagent_production --stage-id 8 --checkpoint "$WARM" --dataset "$DATASET"
RC=$?
pkill -x SimRobot 2>/dev/null || true

echo "=================================================================="
echo "[merge-v5] exit_rc=$RC  Eval history (pass-retention + defense):"
"$PY" - <<'PY'
import glob, csv
fs = sorted(glob.glob("runs/merge_v5_attack_defense/telemetry/*/csv/eval_summary.csv"))
if not fs:
    print("  (no eval_summary.csv)"); raise SystemExit
rows = list(csv.DictReader(open(fs[-1])))
cols = ("step","executed_pass_rate","policy_pass_rate","executed_shoot_rate",
        "executed_block_rate","executed_mark_rate","goal_rate","goals_conceded_rate",
        "defender_score","collision_rate")
for r in rows:
    print("  " + "  ".join(f"{c}={r.get(c,'?')}" for c in cols if c in r))
PY
