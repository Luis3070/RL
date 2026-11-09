"""Step-4 diagnostic: WHEN and WHY do our robots end up mirrored (self-localization flipped)?

For every field robot (and the GK) in probe match logs (run_probe_match.py JSON), using the
simulator ground-truth pose (gt_*) against the robot's own estimate (x, y):
  * mirrored  = |est + gt| < 1000 mm and |gt| > 1500 mm  (est is the point-reflection of gt)
  * onset     = first step of a mirrored episode lasting >= 2 s
For each onset it records what preceded it (look-back 10 s): a fall (fallen state), a goal /
kickoff (GameController score change), the skill being executed, field zone, ball freshness,
controller (learned vs B-Human), role. Then it compares onset contexts against the base rate of
the same context over all steps -- a context that is far more common right before onsets than
in general is a candidate cause.

Usage (simulator idle or under a memory cap; files are parsed one at a time):
  python tools/diagnose_localization.py "runs/full_probe/20261003_092609/*_a*.json" "runs/handover/*/*/*.json"
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os

import numpy as np

HZ = 8.0
FALLEN = 4
BH = {0: "none", 1: "stand", 2: "walk", 3: "shoot", 4: "pass", 5: "dribble", 6: "block", 7: "mark", 8: "observe"}


def mirrored(r):
    gx, gy, ex, ey = r["gt_x"], r["gt_y"], r["x"], r["y"]
    return math.hypot(ex + gx, ey + gy) < 1000 and math.hypot(gx, gy) > 1500


def zone(r):
    x, y = r["gt_x"], r["gt_y"]
    if abs(x) < 1000 and abs(y) < 1000:
        return "centre"
    if abs(y) > 2200:
        return "touchline"
    return "own_half" if x < 0 else "opp_half"


def arm_of(path):
    return os.path.basename(path).split("_")[0]


def analyse(path, look_back_s=10.0, min_len_s=2.0):
    J = json.load(open(path))
    if not J["meta"].get("valid", True):
        return None
    S = J["steps"]
    n = len(S)
    lb, ml = int(look_back_s * HZ), int(min_len_s * HZ)
    score_change = [False] + [(S[k]["gc_for"], S[k]["gc_against"]) != (S[k - 1]["gc_for"], S[k - 1]["gc_against"])
                              for k in range(1, n)]
    out = {"steps": collections.Counter(), "mirr_steps": collections.Counter(), "onsets": [], "base": collections.Counter()}
    robots = list(S[0]["agents"].keys()) + ["gk"]
    for rid in robots:
        recs = [(s["gk"] if rid == "gk" else s["agents"][rid]) for s in S]
        m = np.array([mirrored(r) for r in recs])
        ctrl = "gk_bhuman" if rid == "gk" else None
        for k, r in enumerate(recs):
            c = ctrl or r.get("ctrl", "learned")
            out["steps"][c] += 1
            out["mirr_steps"][c] += bool(m[k])
            out["base"][("zone", zone(r))] += 1
            out["base"][("skill", BH.get(r["bh_skill"], "other"))] += 1
            out["base"][("ball_fresh", r["b_age"] <= 500)] += 1
        k = 0
        while k < n:
            if m[k] and (k == 0 or not m[k - 1]) and m[k:k + ml].all():
                r = recs[k]
                w = range(max(0, k - lb), k)
                out["onsets"].append({
                    "robot": rid, "ctrl": ctrl or r.get("ctrl", "learned"), "role": r.get("role", "gk"),
                    "t": S[k]["t"],
                    "fall_before": any(recs[j]["fall"] == FALLEN for j in w),
                    "goal_before": any(score_change[j] for j in w),
                    "skill": BH.get(recs[max(0, k - 1)]["bh_skill"], "other"),
                    "zone": zone(r), "ball_fresh": r["b_age"] <= 500,
                })
                k += ml
            else:
                k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+")
    args = ap.parse_args()
    files = sorted({f for g in args.globs for f in glob.glob(g) if not f.endswith(".npz")})
    per_arm = collections.defaultdict(lambda: {"steps": collections.Counter(), "mirr_steps": collections.Counter(),
                                               "onsets": [], "base": collections.Counter()})
    for f in files:
        r = analyse(f)
        if r is None:
            continue
        a = per_arm[arm_of(f)]
        a["steps"].update(r["steps"]); a["mirr_steps"].update(r["mirr_steps"]); a["onsets"] += r["onsets"]; a["base"].update(r["base"])
    for arm, a in sorted(per_arm.items()):
        print(f"\n=== arm {arm}")
        for c in a["steps"]:
            print(f"  {c:10s} mirrored {100 * a['mirr_steps'][c] / max(1, a['steps'][c]):5.1f}% of {a['steps'][c]} robot-steps")
        on = a["onsets"]
        if not on:
            continue
        tot = sum(v for (k, _), v in a["base"].items() if k == "zone")
        print(f"  onsets (>=2 s mirrored episodes): {len(on)}")
        for key in ("fall_before", "goal_before"):
            print(f"    {key:12s} {100 * np.mean([o[key] for o in on]):5.1f}% of onsets")
        for key in ("zone", "skill", "ball_fresh"):
            cnt = collections.Counter(o[key] for o in on)
            base_tot = sum(v for (k, _), v in a["base"].items() if k == key)
            parts = []
            for val, c in cnt.most_common(5):
                base = a["base"][(key, val)] / max(1, base_tot)
                parts.append(f"{val}: {100 * c / len(on):.0f}% of onsets vs {100 * base:.0f}% of steps")
            print(f"    {key:12s} " + " | ".join(parts))
        print("    by ctrl/role:", dict(collections.Counter((o["ctrl"], o["role"]) for o in on).most_common(6)))


if __name__ == "__main__":
    main()
