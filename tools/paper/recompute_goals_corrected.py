"""Corrected, offline goal re-count over already-saved match_*.json logs.

Bug found by inspection: the live goal detector in run_v5_vs_bhuman_match.py feeds
ball_authority's (whichever RL player is nearest its OWN perceived ball) position each
step. When ball_authority switches to a DIFFERENT robot mid-play, that robot's
independent (and sometimes stale/wrong) ball estimate can differ from the previous
robot's by thousands of mm in a single ~125ms step — physically impossible for a real
ball, but it satisfies the "ball returned near center" confirmation check, producing a
false goal. Confirmed directly: batch_4v4_20260809_232710/match_9.json step t=341.24
(auth=4, ball=(4409,-1284)) -> t=341.37 (auth=2, ball=(-22,-57)), a ~4600mm jump in one
step, coincident with an authority switch.

First attempt at a fix (reject any single-step jump above a fixed threshold, holding
the last trusted position) was WRONG: ball_authority switches routinely during normal
play (any time a different RL player becomes nearer its own ball estimate), and since
different robots' own BallModel estimates are never a single continuous trajectory to
begin with, that approach rejected ~half of all steps in every match and zeroed out
every goal, including surely-real ones. Discarded.

Actual fix: only confirm a pending goal if the SAME ball_authority agent that flagged
the goal-line crossing also flags the "returned near center" step — i.e. require the
confirmation to come from one robot's own continuous observation, not a jump between
two robots' independent frames. If authority moves to a different agent before
confirmation, drop the pending claim rather than confirming off a different agent's
reading. This directly targets the confirmed bug without an unrelated over-rejection
side effect. This is a re-analysis of data already on disk — no new matches are run.

Usage:
  python tools/paper/recompute_goals_corrected.py runs/paper_match_v5_vs_bhuman_live/batch_<stamp>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GOAL_X = 4500.0
OWN_GOAL_X_MM = -4500.0
OWN_GOAL_POST_Y_MM = 800.0
GOAL_LINE_MARGIN_MM = 100.0
GOAL_COOLDOWN_S = 3.0
KICKOFF_RETURN_RADIUS_MM = 700.0


def recount(path: Path) -> dict:
    d = json.loads(path.read_text())
    steps = d["steps"]

    pending_side = None
    pending_since = None
    pending_authority = None
    cooldown_until = 0.0
    scored_for = 0
    scored_against = 0
    events = []
    dropped_authority_switch = 0

    for s in steps:
        t = s["t"]
        bx, by = s["ball_x"], s["ball_y"]
        auth = s.get("ball_authority")

        if not s.get("ball_fresh", True) or t < cooldown_until:
            continue

        if pending_side is None:
            if bx >= GOAL_X - GOAL_LINE_MARGIN_MM and abs(by) <= OWN_GOAL_POST_Y_MM:
                pending_side, pending_since, pending_authority = "for", t, auth
            elif bx <= OWN_GOAL_X_MM + GOAL_LINE_MARGIN_MM and abs(by) <= OWN_GOAL_POST_Y_MM:
                pending_side, pending_since, pending_authority = "against", t, auth
            continue

        if auth != pending_authority:
            # A different robot's independent (possibly stale/divergent) ball estimate
            # cannot be used to confirm what THIS robot's crossing observation started —
            # drop the claim rather than risk a cross-robot teleport false-positive.
            dropped_authority_switch += 1
            pending_side, pending_since, pending_authority = None, None, None
            continue

        if math.hypot(bx, by) <= KICKOFF_RETURN_RADIUS_MM:
            if pending_side == "for":
                scored_for += 1
            else:
                scored_against += 1
            events.append({"t": t, "side": pending_side})
            cooldown_until = t + GOAL_COOLDOWN_S
            pending_side, pending_since, pending_authority = None, None, None
        elif pending_since is not None and (t - pending_since) > 15.0:
            pending_side, pending_since, pending_authority = None, None, None

    return {
        "match": path.stem,
        "original_goals_for": d.get("goals_for", 0),
        "original_goals_against": d.get("goals_against", 0),
        "corrected_goals_for": scored_for,
        "corrected_goals_against": scored_against,
        "dropped_authority_switch_claims": dropped_authority_switch,
        "corrected_events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    args = parser.parse_args()
    match_files = sorted(args.batch_dir.glob("match_*.json"))
    if not match_files:
        raise SystemExit(f"no match_*.json under {args.batch_dir}")

    total_orig_gf = total_orig_ga = total_corr_gf = total_corr_ga = 0
    for p in match_files:
        r = recount(p)
        total_orig_gf += r["original_goals_for"]
        total_orig_ga += r["original_goals_against"]
        total_corr_gf += r["corrected_goals_for"]
        total_corr_ga += r["corrected_goals_against"]
        flag = " <-- CHANGED" if (r["original_goals_for"], r["original_goals_against"]) != (r["corrected_goals_for"], r["corrected_goals_against"]) else ""
        print(
            f"{r['match']:12s} original={r['original_goals_for']}-{r['original_goals_against']}  "
            f"corrected={r['corrected_goals_for']}-{r['corrected_goals_against']}  "
            f"dropped_claims={r['dropped_authority_switch_claims']}{flag}"
        )
    print()
    print(f"TOTAL original : {total_orig_gf}-{total_orig_ga}")
    print(f"TOTAL corrected: {total_corr_gf}-{total_corr_ga}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
