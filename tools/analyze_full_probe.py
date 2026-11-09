"""Soccer metrics for the full controlled probe (runs/full_probe/<stamp>/).

Same instrument for every arm (E expert, C stock B-Human, T teacher, V v5):
  * our robot positions  = SIMULATOR ground truth (gt_x/gt_y) for players 1-4
  * ball position        = median of FRESH team-24 estimates (sighting <= 500 ms), else invalid
  * opponent positions   = team-fused GlobalOpponentsModel cells (perception; biased the same
                           way in every arm -> compare arms, use arm C as the reference)
  * goals                = GameController (ground truth, attribution verified 2026-10-03)
  * kicks (both teams)   = BallModel speed spike >= KICK_SPEED with a robot within
                           KICK_NEAR_MM; ours typed by the SkillRequest B-Human executed on the
                           kicker (shoot / pass / other); on-target = projected path enters goal

Usage: python tools/analyze_full_probe.py runs/full_probe/<stamp> [--out report_dir]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

HZ = 8.0
GOAL_X = 4500.0
POST_Y = 800.0
CONTROL_MM = 350.0
POSSESS_MM = 1000.0
DUEL_MM = 500.0
KICK_NEAR_MM = 800.0   # fused ball estimate lags/offsets ~0.5 m at kick time (verified on E_01 trace)
KICK_SPEED = 900.0
KICK_REFRACT_S = 2.0
RECEIVE_MM = 450.0
RECEIVE_WINDOW_S = 6.0
FRESH_MS = 500.0
FALLEN = 4
BH_SKILL = {1: "stand", 2: "walk", 3: "shoot", 4: "pass", 5: "dribble", 6: "block", 7: "mark", 8: "observe"}
ARM_NAME = {"E": "Expert (teacher+BH macro)", "C": "B-Human (control)", "T": "Teacher", "V": "v5",
            "R1": "v5 DAgger round 1", "R2": "v5 DAgger round 2",
            "HS1": "Hybrid static: 1 learned + 2 B-Human", "HS2": "Hybrid static: 2 learned + 1 B-Human",
            "HD": "Hybrid dynamic: learned striker + B-Human supports", "HX": "Hybrid (confirmation)",
            "A": "Handover A (striker+open learned)", "B": "Handover B (all learned)",
            "F": "All-learned (round3 + learned coordinator)", "H": "Hybrid ref (round3 striker)",
            "K": "Kickstart batch (sampled)"}
ARM_ORDER = ["E", "C", "T", "V", "R1", "R2", "HS1", "HS2", "HD", "HX", "A", "B", "F", "H", "K"]


def field_records(row):
    """All team-24 field players: Python-driven agents + robots left to B-Human ('others')."""
    d = dict(row["agents"])
    d.update(row.get("others", {}))
    return d


def fused_ball(row):
    pts = [(r["bx"], r["by"]) for r in list(field_records(row).values()) + [row["gk"]]
           if r["b_seen"] > 0 and r["b_age"] <= FRESH_MS]
    if not pts:
        return None
    a = np.asarray(pts)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1]))


def our_positions(row):
    pos = {int(k): (r["gt_x"], r["gt_y"]) for k, r in field_records(row).items()}
    pos[1] = (row["gk"]["gt_x"], row["gk"]["gt_y"])
    return pos


def match_metrics(path):
    J = json.load(open(path))
    meta, S = J["meta"], J["steps"]
    n = len(S)
    dt = 1.0 / HZ
    ball = [fused_ball(s) for s in S]
    ours = [our_positions(s) for s in S]
    m = {"file": os.path.basename(path), "valid": meta["valid"], "goals_for": S[-1]["gc_for"],
         "goals_against": S[-1]["gc_against"], "loop_hz": meta["loop_hz"]}
    m["goal_diff"] = m["goals_for"] - m["goals_against"]
    m["result"] = "W" if m["goal_diff"] > 0 else "D" if m["goal_diff"] == 0 else "L"

    # --- possession / control / territory / duels (valid-ball steps) ---
    poss = []  # per step: 'us' | 'them' | 'loose' | None
    ctrl_us = ctrl_them = vb = 0
    terr_opp_half = final_third = def_third = 0
    bx_sum = 0.0
    duel_steps = []
    for k, b in enumerate(ball):
        if b is None:
            poss.append(None)
            continue
        vb += 1
        du = min(math.hypot(p[0] - b[0], p[1] - b[1]) for p in ours[k].values())
        opp = S[k]["opp"]
        dth = min((math.hypot(x - b[0], y - b[1]) for x, y in opp), default=9e9)
        ctrl_us += du < CONTROL_MM
        ctrl_them += dth < CONTROL_MM
        if min(du, dth) > POSSESS_MM:
            poss.append("loose")
        else:
            poss.append("us" if du <= dth else "them")
        terr_opp_half += b[0] > 0
        final_third += b[0] > 1500
        def_third += b[0] < -1500
        bx_sum += b[0]
        if du < DUEL_MM and dth < DUEL_MM:
            duel_steps.append(k)
    us = poss.count("us")
    them = poss.count("them")
    m["ball_valid_pct"] = 100 * vb / n
    m["possession_pct"] = 100 * us / max(1, us + them)
    m["loose_pct"] = 100 * poss.count("loose") / max(1, vb)
    m["control_us_pct"] = 100 * ctrl_us / max(1, vb)
    m["control_them_pct"] = 100 * ctrl_them / max(1, vb)
    m["ball_in_opp_half_pct"] = 100 * terr_opp_half / max(1, vb)
    m["ball_in_final_third_pct"] = 100 * final_third / max(1, vb)
    m["ball_in_def_third_pct"] = 100 * def_third / max(1, vb)
    m["mean_ball_x"] = bx_sum / max(1, vb)

    # duels: contested episode -> who has possession 2 s later
    won = lost = 0
    last = -999
    for k in duel_steps:
        if k - last < int(2 * HZ):
            continue
        last = k
        later = [p for p in poss[k + int(1.5 * HZ):k + int(3 * HZ)] if p in ("us", "them")]
        if later:
            c = Counter(later).most_common(1)[0][0]
            won += c == "us"
            lost += c == "them"
    m["duels"] = won + lost
    m["duels_won_pct"] = 100 * won / max(1, won + lost)

    # recoveries / turnovers with 1 s hysteresis
    state, run, cand = None, 0, None
    rec = tov = 0
    for p in poss:
        if p not in ("us", "them"):
            continue
        if p == cand:
            run += 1
        else:
            cand, run = p, 1
        if run == int(HZ) and p != state:
            if state == "them" and p == "us":
                rec += 1
            elif state == "us" and p == "them":
                tov += 1
            state = p
    m["recoveries"] = rec
    m["turnovers"] = tov

    # --- kicks (both teams): BallModel velocity spike near a robot ---
    # A kick = some fresh team-24 observer reports ball speed >= KICK_SPEED within 1 s
    # while a robot is within KICK_NEAR_MM of the ball; refractory KICK_REFRACT_S.
    # Kicker side = whoever is nearer the ball (ours: GT pose; theirs: fused perception).
    # Type (ours) = SkillRequest the kicker executed in the preceding second.
    # Direction = observer's velocity rotated to field frame; "on target" = projected
    # path crosses the goal line between the posts.
    def ball_vel_field(row):
        best = None
        for r in list(field_records(row).values()) + [row["gk"]]:
            if r["b_seen"] > 0 and r["b_age"] <= FRESH_MS:
                sp = math.hypot(r["bvx"], r["bvy"])
                if best is None or sp > best[0]:
                    c, s_ = math.cos(r["th"]), math.sin(r["th"])
                    best = (sp, c * r["bvx"] - s_ * r["bvy"], s_ * r["bvx"] + c * r["bvy"])
        return best

    def on_target(b, vx, vy, goal_x):
        if (goal_x - b[0]) * vx <= 0 or abs(vx) < 1e-6:
            return False
        y_at = b[1] + vy * (goal_x - b[0]) / vx
        return abs(y_at) < POST_Y + 100

    vel = [ball_vel_field(s) for s in S]
    shots = on_t = passes = pass_ok = other = 0
    opp_kicks = opp_shots_on = 0
    k = 0
    while k < n:
        b = ball[k]
        v = vel[k]
        if b is None or v is None or v[0] < KICK_SPEED:
            k += 1
            continue
        # who was near the ball just before the spike
        k0 = max(0, k - int(1.5 * HZ))
        cand_us, cand_th = None, 9e9
        du_best = 9e9
        for j in range(k0, k + 1):
            bj = ball[j]
            if bj is None:
                continue
            for pid, p in ours[j].items():
                d = math.hypot(p[0] - bj[0], p[1] - bj[1])
                if d < du_best:
                    du_best, cand_us = d, pid
            cand_th = min([cand_th] + [math.hypot(x - bj[0], y - bj[1]) for x, y in S[j]["opp"]])
        if min(du_best, cand_th) > KICK_NEAR_MM:
            k += 1
            continue
        # Our executed SkillRequest is ground truth for OUR side: a field robot of ours that
        # was within range and running shoot/pass in the window is the kicker, even when a
        # (noisily perceived) opponent looked equally close.
        kick_skill_of = {}
        for pid in field_records(S[k]):
            near = any(ball[j] is not None and math.hypot(ours[j][int(pid)][0] - ball[j][0],
                                                          ours[j][int(pid)][1] - ball[j][1]) <= KICK_NEAR_MM
                       for j in range(k0, k + 1))
            if near:
                ex = [BH_SKILL.get(field_records(S[j])[pid]["bh_skill"]) for j in range(k0, min(n, k + 4))]
                if "pass" in ex or "shoot" in ex:
                    kick_skill_of[int(pid)] = "pass" if "pass" in ex else "shoot"
        if kick_skill_of:
            cand_us = min(kick_skill_of, key=lambda pid: min(
                math.hypot(ours[j][pid][0] - ball[j][0], ours[j][pid][1] - ball[j][1])
                for j in range(k0, k + 1) if ball[j] is not None))
        near_us = du_best <= KICK_NEAR_MM or bool(kick_skill_of)
        near_th = cand_th <= KICK_NEAR_MM
        # Both sides in range (a duel): no ground-truth contact, so attribute by the
        # direction the ball leaves -- toward the opponent goal (+x) = ours, else theirs.
        ours_kick = (v[1] > 0) if (near_us and near_th) else near_us
        if ours_kick:  # our kick
            if cand_us == 1:
                skill = "gk"
            else:
                skill = kick_skill_of.get(cand_us, "other")
            if skill == "shoot":
                shots += 1
                scored = any(S[j]["gc_for"] > S[k]["gc_for"] for j in range(k + 1, min(n, k + int(6 * HZ))))
                on_t += scored or on_target(b, v[1], v[2], GOAL_X)
            elif skill == "pass":
                passes += 1
                recv = None
                for j in range(k + int(0.5 * HZ), min(n, k + int(RECEIVE_WINDOW_S * HZ))):
                    bj = ball[j]
                    if bj is None:
                        continue
                    if any(pid not in (cand_us, 1) and math.hypot(p[0] - bj[0], p[1] - bj[1]) < RECEIVE_MM
                           for pid, p in ours[j].items()):
                        recv = "us"
                        break
                    if any(math.hypot(x - bj[0], y - bj[1]) < RECEIVE_MM for x, y in S[j]["opp"]):
                        recv = "them"
                        break
                pass_ok += recv == "us"
            else:
                other += 1
        else:  # opponent kick
            opp_kicks += 1
            opp_shots_on += on_target(b, v[1], v[2], -GOAL_X)
        k += int(KICK_REFRACT_S * HZ)
    m.update(shots=shots, shots_on_target=on_t, passes=passes, passes_completed=pass_ok,
             pass_completion_pct=100 * pass_ok / max(1, passes), other_kicks=other,
             opp_kicks=opp_kicks, opp_shots_on_target_dir=opp_shots_on)

    # --- physical / robustness (field players, ground truth) ---
    dist, falls, contacts, pose_err, seen = 0.0, 0, 0, [], []
    for pid in S[0]["agents"]:
        xs = np.array([[s["agents"][pid]["gt_x"], s["agents"][pid]["gt_y"]] for s in S])
        d = np.hypot(*np.diff(xs, axis=0).T)
        dist += float(d[d < 400].sum())
        fl = [s["agents"][pid]["fall"] for s in S]
        falls += sum(1 for a, b in zip(fl, fl[1:]) if b == FALLEN and a != FALLEN)
        contacts += sum(s["agents"][pid]["arm_c"] or s["agents"][pid]["foot_c"] for s in S)
        pose_err += [math.hypot(s["agents"][pid]["x"] - s["agents"][pid]["gt_x"],
                                s["agents"][pid]["y"] - s["agents"][pid]["gt_y"]) for s in S]
        seen += [s["agents"][pid]["b_age"] <= FRESH_MS for s in S]
    m["distance_m_per_robot"] = dist / 1000 / 3
    m["falls"] = falls
    m["contact_pct"] = 100 * contacts / (3 * n)
    m["loc_err_p50_mm"] = float(np.median(pose_err))
    m["loc_err_p90_mm"] = float(np.percentile(pose_err, 90))
    m["ball_seen_pct"] = 100 * float(np.mean(seen))

    # --- decisions: actor skill mix + shadow agreement (DAgger signal) ---
    ag = [a for s in S for a in s["agents"].values()]
    m["skill_mix"] = dict(Counter(a["actor"] for a in ag))
    st = [a for a in ag if a["role"] == "striker"]
    m["striker_mix"] = dict(Counter(a["actor"] for a in st))
    m["agree_v5_expert_pct"] = 100 * float(np.mean([a["v5"] == a["expert"] for a in ag]))
    m["agree_v5_expert_striker_pct"] = 100 * float(np.mean([a["v5"] == a["expert"] for a in st])) if st else float("nan")
    mac = [a for a in st if a["macro"]]
    m["agree_v5_expert_macro_pct"] = 100 * float(np.mean([a["v5"] == a["expert"] for a in mac])) if mac else float("nan")
    m["expert_legal_pct"] = 100 * float(np.mean([a["legal_expert"] for a in ag]))
    m["macro_steps_pct"] = 100 * len(mac) / max(1, len(st))
    return m


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan")
    from scipy import stats
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def perm_test(a, b, iters=20000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    cnt = 0
    for _ in range(iters):
        rng.shuffle(pool)
        cnt += abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= abs(obs) - 1e-12
    return obs, cnt / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="+", help="one or more probe run directories")
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-invalid", action="store_true",
                    help="also score invalid (e.g. frozen) attempts -- sensitivity check / testing")
    args = ap.parse_args()
    out = args.out or os.path.join(args.run_dir[-1], "report")
    os.makedirs(out, exist_ok=True)

    per_arm = defaultdict(list)
    invalid = []
    files = sorted(f for d in args.run_dir for f in glob.glob(os.path.join(d, "*.json")))
    for f in files:
        arm = os.path.basename(f).split("_")[0]
        try:
            meta = json.load(open(f))["meta"]
        except Exception as e:  # noqa: BLE001
            invalid.append((f, f"unreadable: {e}"))
            continue
        if not meta["valid"]:
            invalid.append((f, meta["invalid_reason"]))
            if not args.include_invalid:
                continue
        per_arm[arm].append(match_metrics(f))

    scalar = [k for k, v in next(iter(per_arm.values()))[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    with open(os.path.join(out, "per_match.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "file", "result"] + scalar)
        for arm, ms in per_arm.items():
            for m in ms:
                w.writerow([arm, m["file"], m["result"]] + [round(m[k], 3) for k in scalar])

    summary = {}
    for arm, ms in per_arm.items():
        s = {"n": len(ms), "W": sum(m["result"] == "W" for m in ms), "D": sum(m["result"] == "D" for m in ms),
             "L": sum(m["result"] == "L" for m in ms)}
        for k in scalar:
            vals = [m[k] for m in ms]
            s[k] = (float(np.mean(vals)), ci95(vals), float(np.sum(vals)))
        for key in ("skill_mix", "striker_mix"):
            tot = Counter()
            for m in ms:
                tot.update(m[key])
            z = sum(tot.values())
            s[key] = {k: round(100 * v / z, 1) for k, v in tot.most_common()}
        summary[arm] = s

    tests = {}
    gd = {arm: [m["goal_diff"] for m in ms] for arm, ms in per_arm.items()}
    for a, b in (("E", "C"), ("E", "T"), ("T", "C"), ("V", "E"), ("V", "C"), ("V", "T"),
                 ("R1", "V"), ("R2", "V"), ("R2", "R1"), ("R1", "E"), ("R2", "E"), ("R1", "C"), ("R2", "C"),
                 ("F", "C"), ("B", "C"), ("H", "C")):
        if a in gd and b in gd and len(gd[a]) > 1 and len(gd[b]) > 1:
            tests[f"{a}-{b} goal_diff"] = perm_test(gd[a], gd[b])
    for arm, v in gd.items():
        if len(v) > 1:
            from scipy import stats
            tests[f"{arm} goal_diff>0 (one-sample t)"] = (float(np.mean(v)), float(stats.ttest_1samp(v, 0).pvalue))

    json.dump({"summary": summary, "tests": {k: list(v) for k, v in tests.items()}, "invalid": invalid},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    arms = [a for a in ARM_ORDER if a in summary]
    print("| metric (mean per match ± 95% CI) | " + " | ".join(f"{ARM_NAME[a]} (n={summary[a]['n']})" for a in arms) + " |")
    print("|---|" + "---|" * len(arms))
    print("| record W-D-L | " + " | ".join(f"{summary[a]['W']}-{summary[a]['D']}-{summary[a]['L']}" for a in arms) + " |")
    for k in scalar:
        print(f"| {k} | " + " | ".join(f"{summary[a][k][0]:.2f} ± {summary[a][k][1]:.2f}" for a in arms) + " |")
    print("\nstriker skill mix:", {a: summary[a]["striker_mix"] for a in arms})
    print("all-role skill mix:", {a: summary[a]["skill_mix"] for a in arms})
    print("\ntests:", {k: tuple(round(x, 4) for x in v) for k, v in tests.items()})
    print("invalid attempts:", invalid)


if __name__ == "__main__":
    main()
