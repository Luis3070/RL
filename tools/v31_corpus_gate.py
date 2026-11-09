#!/usr/bin/env python3
"""v3.1 corpus-property gate (reqs 3,4,5,6). Run on the merged v2b+overlay corpus
BEFORE training. Exits nonzero if any requirement fails.

  3) support stand recovered: open_support stand >=50%, off_ball_support stand >=45%
  4) overall walk < 60% (closer to v2b, not the 80% of the failed v3)
  5) decision windows alive: pass_armed / dribble_armed / shoot_armed all nonzero & weighted
  6) stage-7 (3v3/main) rows present (from the v2b backbone)
"""
from __future__ import annotations
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO.parent / "SabanaHerons2026" / "Make" / "Python"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import numpy as np
from action import SKILL_ORDER
from observation import ObsIndex

d = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/teacher_multiagent_dataset/dataset_v31_backbone_plus_overlay")
z = np.load(d / "samples.npz", allow_pickle=True)
obs, act, roles, stages, w = z["observations"], z["actions"], z["role_ids"].astype(int), z["stages"].astype(int), z["weights"].astype(float)
m = w > 0
sk = np.rint(act[:, 0]).astype(int)
STAND, WALK = SKILL_ORDER.index("stand"), SKILL_ORDER.index("walk")

def stand_frac(rid):
    sel = m & (roles == rid)
    return float((sk[sel] == STAND).mean()) if sel.any() else 0.0

open_stand = stand_frac(1); offball_stand = stand_frac(2)
walk_overall = float((sk[m] == WALK).mean())
sa = float((obs[m, int(ObsIndex.shoot_armed)] >= 0.5).mean())
da = float((obs[m, int(ObsIndex.dribble_armed)] >= 0.5).mean())
pa = float((obs[m, int(ObsIndex.pass_armed)] >= 0.5).mean())
stage7_rows = int((m & (stages == 7)).sum())

checks = {
    "open_support_stand>=0.50": open_stand >= 0.50,
    "off_ball_support_stand>=0.45": offball_stand >= 0.45,
    "walk_overall<0.60": walk_overall < 0.60,
    "shoot_armed>0.005": sa > 0.005,
    "dribble_armed>0.02": da > 0.02,
    "pass_armed>0.005": pa > 0.005,
    "stage7_rows_present": stage7_rows > 0,
}
print(f"[v31-corpus-gate] {d.name}  rows={int(m.sum())}")
print(f"  open_support stand={open_stand:.2f}  off_ball stand={offball_stand:.2f}  walk_overall={walk_overall:.2f}")
print(f"  armed: shoot={sa:.3f} dribble={da:.3f} pass={pa:.3f}  stage7_rows={stage7_rows}")
bad = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(f"  {'OK ' if v else 'XX '}{k}")
if bad:
    print(f"[v31-corpus-gate] FAIL: {bad} — do NOT train; tune the v2b:overlay balance.")
    sys.exit(1)
print("[v31-corpus-gate] PASS — corpus meets reqs 3-6; safe to train.")
sys.exit(0)
