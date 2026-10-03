#!/usr/bin/env bash
# Stage-4 sub-step 1: collect the GT-on DEFENSE BC corpus across ALL stage-8 defense
# scenarios (1v3/2v3/3v3/transition/recovery + defense_mark_open). RL_TRAIN_GT_OPPONENT_OBS=1
# fills obs[18]/obs[21] from the episode world_spec on the encoder env (just-wired), so
# block/mark labels survive the gate without depending on flaky in-sim perception.
# Opponents are static during collection (collector applies no dynamic motion), so GT is
# exact. Writes a NEW dataset_stage8_<ts> under runs/teacher_multiagent_dataset (canonical
# root) for the merge step. Does NOT touch the attack corpus or any checkpoint.
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"

export RL_TRAIN_GT_OPPONENT_OBS=1
export RL_MA_TEACHER_STAGE=8
# All stage-8 defense scenarios EXCEPT defense_3v3_full: the cramped 3-attacker scene
# reliably hangs the visible collector (world-request timeouts -> sim death). With GT obs
# the dense 3v3 perception buys nothing, and 2v3 already supplies multi-attacker pressure.
export RL_MA_TEACHER_SCENARIOS="${RL_MA_TEACHER_SCENARIOS:-defense_1v3_static_entry,defense_2v3_dynamic,transition_midfield,defense_recovery_clear,defense_mark_open}"
export RL_MA_TEACHER_EPISODES="${RL_MA_TEACHER_EPISODES:-4}"
export RL_MA_TEACHER_MAX_STEPS="${RL_MA_TEACHER_MAX_STEPS:-400}"
# canonical dataset root (creates dataset_stage8_<timestamp>)
unset RL_MA_TEACHER_OUTPUT_ROOT

"$PY" teacher_multiagent_dataset.py
RC=$?
pkill -x SimRobot 2>/dev/null || true

echo "=================================================================="
echo "[collect-defense-gt] exit_rc=$RC  Newest stage-8 dataset + label distribution:"
"$PY" - <<'PY'
import glob, os, numpy as np, collections
from action import SKILL_ORDER
dirs = sorted(glob.glob("runs/teacher_multiagent_dataset/dataset_stage8_*"), key=os.path.getmtime)
if not dirs:
    print("  (no stage-8 dataset dir)"); raise SystemExit
d = dirs[-1]; print("  dir:", d)
npz = os.path.join(d, "samples.npz")
if not os.path.exists(npz):
    print("  (no samples.npz)"); raise SystemExit
z = np.load(npz, allow_pickle=True)
acts = np.asarray(z["actions"]); roles = np.asarray(z["role_ids"])
skills = acts[:, 0].astype(int)
c = collections.Counter(skills.tolist())
print(f"  total rows: {len(skills)}")
for i, name in enumerate(SKILL_ORDER):
    print(f"    {name:8s} = {c.get(i,0)}")
print("  roles (0=striker,1=open,2=offball):", dict(collections.Counter(roles.astype(int).tolist())))
mi, bi = SKILL_ORDER.index("mark"), SKILL_ORDER.index("block")
print(f"  >>> mark={c.get(mi,0)} block={c.get(bi,0)}  DATASET={d}")
PY
