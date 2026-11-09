"""Match-level statistics for the paper (protocol: runs/paper_eval/PROTOCOL.md).

Arms are given as LABEL=glob[,glob...] (valid matches only; invalid ones are counted and reported).
Per arm: n, W-D-L, points/match, and for every soccer metric the mean with a 95% bootstrap CI
(10k resamples of matches). Comparisons (A-B): difference of means, bootstrap CI, two-sided
permutation p. Also per-side (normal / swapped scene) goal difference for transparency.
Usage:
  python tools/paper_match_stats.py --arm "P=runs/paper_eval/block1/P_*.json" \
      --arm "C=runs/paper_eval/block1/C_*.json,runs/kickstart/round2_*/final/C_*.json" \
      --compare P-C --compare P-G --out runs/paper_eval/stats
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import analyze_full_probe as A  # noqa: E402

METRICS = [("goal_diff", "Goal difference"), ("goals_for", "Goals for"), ("goals_against", "Goals against"),
           ("points", "Points"), ("possession_pct", "Possession %"), ("control_us_pct", "Ball control us %"),
           ("control_them_pct", "Ball control opponent %"), ("ball_in_opp_half_pct", "Ball in opp. half %"),
           ("ball_in_final_third_pct", "Ball in final third %"), ("shots", "Shots"),
           ("shots_on_target", "Shots on target"), ("opp_shots_on_target_dir", "Opp. shots on target"),
           ("passes", "Passes"), ("passes_completed", "Passes completed"), ("duels_won_pct", "Duels won %"),
           ("recoveries", "Recoveries"), ("turnovers", "Turnovers"), ("falls", "Falls"),
           ("distance_m_per_robot", "Distance m/robot")]
B = 10000


def boot_ci(x, rng):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float(x.mean()) if len(x) else float("nan")] * 3
    bs = rng.choice(x, (B, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_diff(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = rng.choice(a, (B, len(a))).mean(1) - rng.choice(b, (B, len(b))).mean(1)
    return [float(a.mean() - b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="LABEL=glob[,glob]")
    ap.add_argument("--compare", action="append", default=[], help="A-B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs(a.out, exist_ok=True)
    arms, invalid = {}, {}
    for spec in a.arm:
        lab, globs = spec.split("=", 1)
        files = sorted({f for g in globs.split(",") for f in glob.glob(g)})
        ms, bad = [], []
        for f in files:
            if json.load(open(f))["meta"]["valid"]:
                m = A.match_metrics(f)
                m["points"] = 3 if m["goal_diff"] > 0 else 1 if m["goal_diff"] == 0 else 0
                m["side"] = "swap" if "_s_" in os.path.basename(f) else "normal"
                m["path"] = f
                ms.append(m)
            else:
                bad.append(f)
        arms[lab], invalid[lab] = ms, bad
    res = {"arms": {}, "comparisons": {}}
    for lab, ms in arms.items():
        gd = np.array([m["goal_diff"] for m in ms])
        r = {"n": len(ms), "n_invalid": len(invalid[lab]), "W": int((gd > 0).sum()), "D": int((gd == 0).sum()),
             "L": int((gd < 0).sum()), "files": [m["path"] for m in ms]}
        for side in ("normal", "swap"):
            g = [m["goal_diff"] for m in ms if m["side"] == side]
            r[f"gd_{side}"] = [len(g), float(np.mean(g)) if g else float("nan")]
        for k, _ in METRICS:
            r[k] = boot_ci([m[k] for m in ms], rng)
        res["arms"][lab] = r
    for c in a.compare:
        x, y = c.split("-")
        res["comparisons"][c] = {}
        for k, _ in METRICS:
            va, vb = [m[k] for m in arms[x]], [m[k] for m in arms[y]]
            d = boot_diff(va, vb, rng)
            _, p = A.perm_test(va, vb, iters=5000)
            res["comparisons"][c][k] = d + [float(p)]
    json.dump(res, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    L = list(arms)
    f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"  # noqa: E731
    lines = ["| Metric (mean per match, 95% bootstrap CI) | " + " | ".join(f"{l} (n={res['arms'][l]['n']})" for l in L) + " |",
             "|---|" + "---|" * len(L),
             "| W-D-L | " + " | ".join(f"{res['arms'][l]['W']}-{res['arms'][l]['D']}-{res['arms'][l]['L']}" for l in L) + " |",
             "| Invalid matches (excluded) | " + " | ".join(str(res['arms'][l]['n_invalid']) for l in L) + " |",
             "| GD normal / swapped scene | " + " | ".join(
                 f"{res['arms'][l]['gd_normal'][1]:.2f} (n={res['arms'][l]['gd_normal'][0]}) / {res['arms'][l]['gd_swap'][1]:.2f} (n={res['arms'][l]['gd_swap'][0]})" for l in L) + " |"]
    lines += [f"| {name} | " + " | ".join(f(res["arms"][l][k]) for l in L) + " |" for k, name in METRICS]
    for c, d in res["comparisons"].items():
        lines += ["", f"**{c}** (difference of means, 95% bootstrap CI, permutation p)", "", "| Metric | diff [CI] | p |", "|---|---|---|"]
        lines += [f"| {name} | {f(d[k])} | {d[k][3]:.3f} |" for k, name in METRICS]
    open(os.path.join(a.out, "stats.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
