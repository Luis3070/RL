"""Pick the agent's ball-playing rule setting from the screen (pre-registered rule, 2026-10-07).

Inputs: the phase-0 baseline run (F_* = current agent with default rules) and the screen run
(S1_*, S2_*, ... = same agent, different rule settings). Rule:
  eligible  : mean goals against <= baseline goals against + 0.5 (keep the defence)
  primary   : highest mean goal difference, and it must be >= the baseline's
  tie-break : within 0.25 goals of the best -> more (shots on target + passes)
  otherwise : keep the baseline (default) setting
Prints a JSON line with the chosen label and its harness arguments (from settings.json).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import analyze_full_probe as A  # noqa: E402


def metrics(files):
    ms = [A.match_metrics(f) for f in files if json.load(open(f))["meta"]["valid"]]
    if not ms:
        return None
    g = lambda k: float(np.mean([m[k] for m in ms]))  # noqa: E731
    return {"n": len(ms), "gd": g("goal_diff"), "gf": g("goals_for"), "ga": g("goals_against"),
            "shots_on": g("shots_on_target"), "passes": g("passes"), "possession": g("possession_pct")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True); ap.add_argument("--screen", required=True)
    ap.add_argument("--settings", required=True, help="json {label: harness args string}")
    a = ap.parse_args()
    settings = json.load(open(a.settings))
    base = metrics(sorted(glob.glob(os.path.join(a.baseline, "F_*.json"))))
    res = {lab: metrics(sorted(glob.glob(os.path.join(a.screen, f"{lab}_*.json")))) for lab in settings}
    print("BASELINE", json.dumps(base))
    for lab, m in res.items():
        print(lab, json.dumps(m))
    elig = {l: m for l, m in res.items() if m and m["ga"] <= base["ga"] + 0.5 and m["gd"] >= base["gd"]}
    choice = "BASE"
    if elig:
        top = max(m["gd"] for m in elig.values())
        near = {l: m for l, m in elig.items() if m["gd"] >= top - 0.25}
        choice = max(near, key=lambda l: near[l]["shots_on"] + near[l]["passes"])
    args = settings.get(choice, "")
    print(json.dumps({"choice": choice, "args": args}))


if __name__ == "__main__":
    main()
