"""Mirror (flipped self-localization) detector for team-24 field robots — step 4.

Why: field players of BOTH teams are mirrored 16-33 % of the time in SimRobot (onsets mostly in
the opponent half); the goalkeeper never is. A robot that detects its own mirroring can correct
its observation and its walk targets, an edge only our team would have.

Detector (deployable, no ground truth): per robot, per frame where the robot sees the ball
freshly and close, compare its field-frame ball b with a reference ball r (goalkeeper's if fresh,
else the median of other teammates' fresh balls). If one hypothesis clearly fits better
(min(|b-r|, |b+r|) < fit_mm and the gap > margin_mm) the frame adds +-1 evidence (mirrored /
normal) to a leaky accumulator s <- decay*s + e; the robot is flagged mirrored when s > on_th and
unflagged when s < off_th (hysteresis).

Offline tuning uses the simulator ground truth logged by run_probe_match.py (gt_x/gt_y): a robot
is truly mirrored when |est + gt| < 1000 mm and |gt| > 1500 mm.

Usage:
  python tools/mirror_detector.py extract --out runs/mirror_detector/cache.npz "<match json globs>"
  python tools/mirror_detector.py tune --cache runs/mirror_detector/cache.npz --out runs/mirror_detector/params.json
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
from dataclasses import dataclass, asdict

import numpy as np

FRESH_MS = 500.0


@dataclass
class MirrorParams:
    fit_mm: float = 1000.0      # best hypothesis must fit within this distance
    margin_mm: float = 1000.0   # and beat the other one by at least this much
    min_ref_norm_mm: float = 1000.0  # reference ball must be away from the centre (reflection ambiguous there)
    max_own_ball_mm: float = 2500.0  # robot's own ball must be close (accurate estimate)
    decay: float = 0.97
    on_th: float = 4.0
    off_th: float = -2.0


class MirrorDetector:
    """Stateful per-robot detector; call update() once per control step."""

    def __init__(self, params: MirrorParams | None = None):
        self.p = params or MirrorParams()
        self.s: dict[int, float] = {}
        self.flag: dict[int, bool] = {}

    def evidence(self, own_ball, own_fresh, own_dist, ref_ball):
        p = self.p
        if ref_ball is None or not own_fresh or own_dist > p.max_own_ball_mm:
            return 0.0
        r = np.asarray(ref_ball, float)
        if np.linalg.norm(r) < p.min_ref_norm_mm:
            return 0.0
        b = np.asarray(own_ball, float)
        d_same, d_mir = np.linalg.norm(b - r), np.linalg.norm(b + r)
        if min(d_same, d_mir) > p.fit_mm or abs(d_same - d_mir) < p.margin_mm:
            return 0.0
        return 1.0 if d_mir < d_same else -1.0

    def update(self, rid: int, own_ball, own_fresh, own_dist, ref_ball) -> bool:
        e = self.evidence(own_ball, own_fresh, own_dist, ref_ball)
        s = self.p.decay * self.s.get(rid, 0.0) + e
        self.s[rid] = s
        f = self.flag.get(rid, False)
        if not f and s > self.p.on_th:
            f = True
        elif f and s < self.p.off_th:
            f = False
        self.flag[rid] = f
        return f


def reference_ball(gk: dict, others: list[dict], flagged: dict | None = None):
    """GK ball if fresh, else median of other teammates' fresh balls (skipping flagged ones)."""
    if gk is not None and gk["b_seen"] > 0 and gk["b_age"] <= FRESH_MS:
        return (gk["bx"], gk["by"])
    pts = [(o["bx"], o["by"]) for o in others if o["b_seen"] > 0 and o["b_age"] <= FRESH_MS]
    if not pts:
        return None
    a = np.asarray(pts)
    return (float(np.median(a[:, 0])), float(np.median(a[:, 1])))


# ------------------------------------------------------------------ offline extraction / tuning
def extract(files, out):
    """Compact per-(match, robot, step) arrays: own ball, freshness, distance, reference ball, truth."""
    cols = {k: [] for k in ("match", "robot", "step", "bx", "by", "fresh", "dist", "rx", "ry", "ref_ok", "truth")}
    m_id = 0
    for f in files:
        J = json.load(open(f))
        if not J["meta"].get("valid", True):
            continue
        S = J["steps"]
        for k, s in enumerate(S):
            recs = dict(s["agents"]); recs.update(s.get("others", {}))
            for rid, r in recs.items():
                others = [o for q, o in recs.items() if q != rid]
                ref = reference_ball(s["gk"], others)
                gt = np.array([r["gt_x"], r["gt_y"]]); est = np.array([r["x"], r["y"]])
                cols["match"].append(m_id); cols["robot"].append(int(rid)); cols["step"].append(k)
                cols["bx"].append(r["bx"]); cols["by"].append(r["by"])
                cols["fresh"].append(r["b_seen"] > 0 and r["b_age"] <= FRESH_MS); cols["dist"].append(r["b_dist"])
                cols["rx"].append(ref[0] if ref else 0.0); cols["ry"].append(ref[1] if ref else 0.0)
                cols["ref_ok"].append(ref is not None)
                cols["truth"].append(bool(np.linalg.norm(est + gt) < 1000 and np.linalg.norm(gt) > 1500))
        m_id += 1
        del J
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in cols.items()})
    print(f"extracted {len(cols['step'])} robot-steps from {m_id} matches -> {out}")


def evidence_vec(c, p: MirrorParams):
    """Vectorised MirrorDetector.evidence over the cache (identical rules)."""
    b = np.stack([c["bx"], c["by"]], 1).astype(float); r = np.stack([c["rx"], c["ry"]], 1).astype(float)
    d_same = np.linalg.norm(b - r, axis=1); d_mir = np.linalg.norm(b + r, axis=1)
    ok = (c["ref_ok"].astype(bool) & c["fresh"].astype(bool) & (c["dist"] <= p.max_own_ball_mm)
          & (np.linalg.norm(r, axis=1) >= p.min_ref_norm_mm)
          & (np.minimum(d_same, d_mir) <= p.fit_mm) & (np.abs(d_same - d_mir) >= p.margin_mm))
    return np.where(ok, np.where(d_mir < d_same, 1.0, -1.0), 0.0)


def run_detector(c, params: MirrorParams, order=None, ev=None):
    """Replay of MirrorDetector (leaky accumulator + hysteresis) per (match, robot) trajectory."""
    ev = evidence_vec(c, params) if ev is None else ev
    order = np.lexsort((c["step"], c["robot"], c["match"])) if order is None else order
    key = c["match"][order].astype(np.int64) * 100 + c["robot"][order]
    new_traj = np.r_[True, key[1:] != key[:-1]]
    e = ev[order]; out = np.zeros(len(e), bool)
    s, f = 0.0, False
    d, on, off = params.decay, params.on_th, params.off_th
    for j in range(len(e)):
        if new_traj[j]:
            s, f = 0.0, False
        s = d * s + e[j]
        if not f and s > on:
            f = True
        elif f and s < off:
            f = False
        out[j] = f
    pred = np.zeros(len(e), bool); pred[order] = out
    return pred


def score(truth, pred):
    tp = int((truth & pred).sum()); fp = int((~truth & pred).sum()); fn = int((truth & ~pred).sum()); tn = int((~truth & ~pred).sum())
    return {"recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp), "false_alarm": fp / max(1, fp + tn),
            "mirrored_rate": float(truth.mean())}


def tune(cache, out):
    z = dict(np.load(cache))
    matches = np.unique(z["match"])
    test_m = set(matches[::4].tolist())           # every 4th match held out
    is_test = np.isin(z["match"], list(test_m))
    tr = {k: v[~is_test] for k, v in z.items()}; te = {k: v[is_test] for k, v in z.items()}
    grid = itertools.product([700.0, 1000.0, 1500.0], [500.0, 1000.0, 1500.0], [0.95, 0.97, 0.99], [3.0, 6.0, 10.0], [-2.0, 0.0])
    best = None
    order_tr = np.lexsort((tr["step"], tr["robot"], tr["match"]))
    ev_cache = {}
    for fit, margin, decay, on, off in grid:
        p = MirrorParams(fit_mm=fit, margin_mm=margin, decay=decay, on_th=on, off_th=off)
        if (fit, margin) not in ev_cache:
            ev_cache[(fit, margin)] = evidence_vec(tr, p)
        sc = score(tr["truth"], run_detector(tr, p, order_tr, ev_cache[(fit, margin)]))
        # objective: recall with false alarms kept small (a false flip is costly: it mirrors a correct robot)
        obj = sc["recall"] - 4.0 * sc["false_alarm"]
        if best is None or obj > best[0]:
            best = (obj, p, sc)
    _, p, sc_tr = best
    sc_te = score(te["truth"], run_detector(te, p))
    res = {"params": asdict(p), "train": sc_tr, "test": sc_te, "test_matches": len(test_m), "train_matches": len(matches) - len(test_m)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("globs", nargs="+"); e.add_argument("--out", required=True)
    t = sub.add_parser("tune"); t.add_argument("--cache", required=True); t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "extract":
        files = sorted({f for g in a.globs for f in glob.glob(g) if f.endswith(".json")})
        extract(files, a.out)
    else:
        tune(a.cache, a.out)


if __name__ == "__main__":
    main()
