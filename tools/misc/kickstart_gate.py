"""Safety gate: does the candidate agent beat B-Human better than the baseline agent did?

Each run directory holds F_* (our all-learned team) and C_* (stock B-Human in our seat) matches,
played concurrently and side-balanced. For each run: W-D-L of F against the opponent, goal
difference, and the margin over the concurrent B-Human control (F mean GD - C mean GD).
The candidate is ACCEPTED only if its margin over B-Human AND its points per match exceed the
baseline's; otherwise the baseline checkpoint stays the delivered agent.
"""
import argparse
import glob
import json
import os

import numpy as np


def gds(run, prefix):
    out = []
    for f in sorted(glob.glob(os.path.join(run, f"{prefix}_*.json"))):
        m = json.load(open(f))["meta"]
        if m["valid"]:
            out.append(m["gc_final"][0] - m["gc_final"][1])
    return np.array(out)


def summary(run):
    f, c = gds(run, "F"), gds(run, "C")
    wdl = (int((f > 0).sum()), int((f == 0).sum()), int((f < 0).sum()))
    pts = (3 * wdl[0] + wdl[1]) / max(1, len(f))
    return {"n_F": len(f), "n_C": len(c), "F_WDL": wdl, "F_points_per_match": round(pts, 2),
            "F_GD": round(float(f.mean()), 2) if len(f) else None,
            "C_GD": round(float(c.mean()), 2) if len(c) else None,
            "margin_vs_BHuman": round(float(f.mean() - c.mean()), 2) if len(f) and len(c) else None,
            "C_WDL": (int((c > 0).sum()), int((c == 0).sum()), int((c < 0).sum()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True); ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline-ckpt", required=True); ap.add_argument("--candidate-ckpt", required=True)
    a = ap.parse_args()
    b, c = summary(a.baseline), summary(a.candidate)
    ok = (c["margin_vs_BHuman"] is not None and b["margin_vs_BHuman"] is not None
          and c["margin_vs_BHuman"] > b["margin_vs_BHuman"] and c["F_points_per_match"] > b["F_points_per_match"])
    print("BASELINE ", json.dumps(b))
    print("CANDIDATE", json.dumps(c))
    print(f"GATE: {'ACCEPT candidate ' + a.candidate_ckpt if ok else 'KEEP baseline ' + a.baseline_ckpt}")


if __name__ == "__main__":
    main()
