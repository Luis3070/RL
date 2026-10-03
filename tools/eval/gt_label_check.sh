#!/usr/bin/env bash
# Stage-3 check #1: with RL_TRAIN_GT_OPPONENT_OBS=1, does the BC collector now produce
# nonzero block/mark labels on defense_mark_open (deadness gone)? GT obs is computed on
# the encoder env from the episode world_spec (just-wired), so it does NOT depend on
# flaky in-sim opponent perception -> should be a stable read. Writes to a SCRATCH dir
# (never the canonical dataset). Short (2 eps) — this is a label-distribution probe.
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
cd "$REPO"

export RL_TRAIN_GT_OPPONENT_OBS=1          # <-- the path under test
export RL_MA_TEACHER_STAGE=8
export RL_MA_TEACHER_SCENARIOS=defense_mark_open
export RL_MA_TEACHER_EPISODES="${RL_MA_TEACHER_EPISODES:-2}"
export RL_MA_TEACHER_MAX_STEPS="${RL_MA_TEACHER_MAX_STEPS:-250}"
OUT="$REPO/runs/tmp_gt_labelcheck"
rm -rf "$OUT"; mkdir -p "$OUT"
export RL_MA_TEACHER_OUTPUT_ROOT="$OUT"

"$PY" -m rlfootball.datasets.teacher_multiagent_dataset
RC=$?
pkill -x SimRobot 2>/dev/null || true

echo "=================================================================="
echo "[gt-label-check] exit_rc=$RC  Label distribution (GT obs ON):"
"$PY" - <<'PY'
import glob, numpy as np
from rlfootball.core.action import SKILL_ORDER
npzs = sorted(glob.glob("runs/tmp_gt_labelcheck/**/samples.npz", recursive=True))
if not npzs:
    print("  (no samples.npz written)"); raise SystemExit
d = np.load(npzs[-1], allow_pickle=True)
print("  file:", npzs[-1], " keys:", list(d.keys()))
acts = d["actions"] if "actions" in d else d.get("all_actions")
roles = d["role_ids"] if "role_ids" in d else None
# actions[:,0] is skill index per the dataset format
skills = np.asarray(acts)[:, 0].astype(int) if acts is not None and np.asarray(acts).ndim == 2 else None
if skills is None:
    print("  (could not locate skill column; keys above)"); raise SystemExit
import collections
c = collections.Counter(skills.tolist())
print(f"  total rows: {len(skills)}")
for i, name in enumerate(SKILL_ORDER):
    print(f"    {name:8s} = {c.get(i,0)}")
if roles is not None:
    rc = collections.Counter(np.asarray(roles).astype(int).tolist())
    print("  role rows (0=striker,1=open,2=offball):", dict(rc))
mark_i = SKILL_ORDER.index("mark"); block_i = SKILL_ORDER.index("block")
print(f"  >>> mark rows = {c.get(mark_i,0)}  block rows = {c.get(block_i,0)}  (want > 0)")
PY
