#!/usr/bin/env bash
# Paste-proof driver for the doc-35 v3 attack pipeline. All env (PYTHONPATH, gate
# contract, output dirs) is set INSIDE the script so each step is one short command
# — avoids the multi-line-paste env breakage in the IDE terminal.
#
#   bash tools/run/run_v3_attack.sh dryrun   # pre-flight contingency gate (no sim)
#   bash tools/run/run_v3_attack.sh regen    # chunked dataset regen (sim, hours)
#   bash tools/run/run_v3_attack.sh merge    # merge chunks + audit
#   bash tools/run/run_v3_attack.sh train    # BC+PPO fine-tune from v2b@7500
#   bash tools/run/run_v3_attack.sh validate runs/multiagent_v3/checkpoint_stage7_step10000.pt
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/env.sh"
REPO="$RL_REPO_DIR"
export PYTHONPATH="$REPO:$SABANAHERONS_DIR/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER="${RL_GATE_SHOOT_OPEN_ENTER:-0.30}"
export RL_GATE_SHOOT_OPEN_EXIT="${RL_GATE_SHOOT_OPEN_EXIT:-0.20}"
# Pin the per-slot scene to the single-robot RL1v0_3D_visible — the cross-sim
# injection design runs ONE RL robot per slot (teammates injected via comm), so the
# multi-robot RLMAStage*_3D scenes (Rafael's June-12 default) break the lone robot's
# ball-control / can_score perception → the gate never arms → an all-walk corpus.
# v2b (the healthy corpus) was generated on RL1v0_3D_visible. Override-able.
export RL_MA_TEACHER_VISIBLE_SCENE_NAME="${RL_MA_TEACHER_VISIBLE_SCENE_NAME:-RL1v0_3D_visible}"
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
CHUNKS="$REPO/runs/teacher_multiagent_dataset/v3_chunks"
CORPUS="$REPO/runs/teacher_multiagent_dataset/dataset_multiagent_v3_attack"
cd "$REPO"

case "${1:-}" in
  dryrun)
    "$PY" tools/eval/contrastive_dryrun.py
    ;;
  probe)
    # Fast scene/gate-arming check (~few min, 2 episodes, 1 scenario) BEFORE a full
    # regen — confirms the striker actually controls the ball and the gate arms
    # (shoot/dribble) so we don't burn another 2h on an all-walk corpus.
    PROBE="$REPO/runs/teacher_multiagent_dataset/v3_probe"; rm -rf "$PROBE"
    # 2v0 clean band = the lowest-collision scenario (2 RL, no opponents) so an episode
    # actually survives the sim instability and we can read its gate bits.
    RL_MA_TEACHER_STAGE=6 RL_MA_TEACHER_MAX_STEPS=600 RL_MA_TEACHER_EPISODES=3 \
    RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=10 RL_MA_TEACHER_SEED_BASE=990000 \
    RL_MA_TEACHER_SCENARIOS=team_2v0_clean_band RL_MA_TEACHER_OUTPUT_ROOT="$PROBE" \
    PYBH_RL_SHM_NAME=/pybh_rl_v3_probe "$PY" -m rlfootball.datasets.teacher_multiagent_dataset
    pkill -f SimRobot 2>/dev/null; pkill -f Xvfb 2>/dev/null
    PROBE_GLOB="dataset_stage6_*"
    "$PY" - "$PROBE" <<'PYEOF'
import numpy as np, glob, os, sys, collections, json
from rlfootball.core.observation import ObsIndex
from rlfootball.core.action import SKILL_ORDER
ds=glob.glob(os.path.join(sys.argv[1],"dataset_stage6_*"))
if not ds: print("[probe] NO chunk written — generation failed"); sys.exit(1)
chunk=sorted(ds)[-1]
d=np.load(os.path.join(chunk,"samples.npz"),allow_pickle=True)
o,a=d["observations"],d["actions"]
if o.ndim < 2 or len(o)==0:
    sp=os.path.join(chunk,"summary.json")
    drops=json.load(open(sp)).get("scenarios",[{}])[0].get("drop_reasons",{}) if os.path.exists(sp) else {}
    print(f"[probe] 0 episodes survived (rows=0). drop_reasons={drops}")
    print("[probe] VERDICT: INCONCLUSIVE — sim instability dropped all; re-run probe or raise attempts")
    sys.exit(2)
sk=np.rint(a[:,0]).astype(int)
sa=float((o[:,int(ObsIndex.shoot_armed)]>=0.5).mean()); da=float((o[:,int(ObsIndex.dribble_armed)]>=0.5).mean())
cs=float(o[:,int(ObsIndex.can_score_now)].mean())
print(f"[probe] rows={len(o)} shoot_armed={sa:.3f} dribble_armed={da:.3f} can_score_now_mean={cs:.3f}")
print("[probe] skill counts:", {SKILL_ORDER[k]:v for k,v in collections.Counter(sk.tolist()).items()})
ok = (da>0.01 or sa>0.01) and len(set(sk.tolist()))>1
print("[probe] VERDICT:", "PASS — gate arms, varied skills, scene OK → safe to regen" if ok else "FAIL — still all-walk / gate dead")
sys.exit(0 if ok else 1)
PYEOF
    ;;
  regen)
    # Wedge-resilient regen: a fallen robot parks OOB and the in-process reset can't
    # lift it, so a wedged slot dies for the whole chunk. Mitigations: (1) fail-fast
    # resets (2 attempts not 8 → ~50s not ~200s per wedge), (2) FINE chunks with a
    # fresh sim each (pkill between → wedge dies with its small chunk), (3) contrastive
    # scenarios in their OWN chunks run FIRST (so we capture the v3 data even if the
    # fragile team_3v0_* scenarios wedge). Episode length stays 800 (NOT shortened) —
    # the give-and-go / support chains need the room; the wedge is at reset, not
    # mid-episode, so shortening wouldn't help anyway.
    export RL_MA_SHOOT_THRESHOLD=0.30 RL_MA_TEACHER_COLLISION_SUSTAIN_STEPS=16
    export RL_MA_TEACHER_OUTPUT_ROOT="$CHUNKS"
    export RL_VISIBLE_RESET_MAX_ATTEMPTS="${RL_VISIBLE_RESET_MAX_ATTEMPTS:-2}"
    S7MAIN="team_3v0_clean,team_3v0_static_pinch,team_3v0_dynamic,team_3v3_full_dynamic"
    S7CONTRAST="giveandgo_lane_open,giveandgo_lane_covered,support_open,support_marked"
    echo "[v3] dry-run gate first…"; "$PY" tools/eval/contrastive_dryrun.py || { echo "[v3] gate FAILED — aborting regen"; exit 1; }

    # --- stage 6 (2 RL, low fall rate): 2 chunks ---
    for i in 1 2; do
      echo "[v3] === stage 6 chunk $i ==="
      RL_MA_TEACHER_STAGE=6 RL_MA_TEACHER_MAX_STEPS=600 RL_MA_TEACHER_EPISODES=4 \
      RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=6 RL_MA_TEACHER_SEED_BASE=$((600000 + i*10000)) \
      PYBH_RL_SHM_NAME=/pybh_rl_v3_s6_${i} "$PY" -m rlfootball.datasets.teacher_multiagent_dataset
      pkill -f SimRobot 2>/dev/null; pkill -f Xvfb 2>/dev/null; sleep 5
    done

    # --- stage 7 CONTRASTIVE first (the v3 payload): 4 small fresh-sim chunks ---
    for i in 1 2 3 4; do
      echo "[v3] === stage 7 CONTRAST chunk $i ==="
      RL_MA_TEACHER_STAGE=7 RL_MA_TEACHER_MAX_STEPS=800 RL_MA_TEACHER_EPISODES=4 \
      RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=8 RL_MA_TEACHER_SEED_BASE=$((720000 + i*10000)) \
      RL_MA_TEACHER_SCENARIOS="$S7CONTRAST" \
      PYBH_RL_SHM_NAME=/pybh_rl_v3_s7c_${i} "$PY" -m rlfootball.datasets.teacher_multiagent_dataset
      pkill -f SimRobot 2>/dev/null; pkill -f Xvfb 2>/dev/null; sleep 5
    done

    # --- stage 7 MAIN attack scenarios: 3 small fresh-sim chunks ---
    for i in 1 2 3; do
      echo "[v3] === stage 7 MAIN chunk $i ==="
      RL_MA_TEACHER_STAGE=7 RL_MA_TEACHER_MAX_STEPS=800 RL_MA_TEACHER_EPISODES=4 \
      RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=6 RL_MA_TEACHER_SEED_BASE=$((700000 + i*10000)) \
      RL_MA_TEACHER_SCENARIOS="$S7MAIN" \
      PYBH_RL_SHM_NAME=/pybh_rl_v3_s7m_${i} "$PY" -m rlfootball.datasets.teacher_multiagent_dataset
      pkill -f SimRobot 2>/dev/null; pkill -f Xvfb 2>/dev/null; sleep 5
    done
    echo "[v3] regen done. Chunks under $CHUNKS"
    ;;
  merge)
    "$PY" -m rlfootball.datasets.merge_multiagent_teacher_datasets "$CHUNKS"/* --output-dir "$CORPUS"
    "$PY" -m rlfootball.datasets.teacher_multiagent_dataset_audit "$CORPUS" --write
    ;;
  train)
    RL_MA_PROD_TRAIN_STEPS="${RL_MA_PROD_TRAIN_STEPS:-10000}" \
    RL_MA_PROD_MAX_EPISODE_STEPS=800 RL_MA_PROD_EVAL_INTERVAL=2500 \
    RL_MA_PROD_CHECKPOINT_INTERVAL=2500 RL_MA_PROD_SCENARIO_SET=team \
    RL_MA_PROD_EXPERIMENT_NAME="${RL_MA_PROD_EXPERIMENT_NAME:-multiagent_v3}" \
    "$PY" -m rlfootball.training.train_multiagent_production --stage-id 7 \
      --checkpoint runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt \
      --dataset "$CORPUS/samples.npz"
    ;;
  validate)
    CKPT="${2:?usage: validate <checkpoint.pt>}"
    "$PY" -m rlfootball.evaluation.behavioral_evaluator --checkpoint "$CKPT" --episodes 6
    ;;
  *)
    echo "usage: bash tools/run/run_v3_attack.sh {dryrun|regen|merge|train|validate <ckpt>}"; exit 2;;
esac
