#!/usr/bin/env python3
"""v3.1 PRE-TRAIN GATE (req 7) — teammate-attribution flip on TEACHER labels. No sim.

The v3 failure: the contrast varied an OPPONENT, so the policy decided via its own
front-cone obs and never read the teammate (ablation flip ~2%). This gate verifies the
REDESIGNED contrast forces the teacher to condition on the TEAMMATE: build the giveandgo
A (outlet teammate present, reachable up-field) vs B (same teammate pulled deep) with the
opponent picture IDENTICAL (none), run teacher_striker, and check pass↔dribble flips.

Because opponents are constant, any flip is teammate-attributable. STOP the pipeline if
the flip rate is < MIN (default 0.10) — training cannot learn teammate-conditioning the
data doesn't carry.

Run: python tools/eval/teammate_flip_gate.py
"""
from __future__ import annotations
import dataclasses, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO.parent / "SabanaHerons2026" / "Make" / "Python"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import multiagent_teacher as mt
from multiagent_teacher import TeammateSlot
from gate import GateDecision
from pybh_backend import BackendObservation
import production_curriculum as pc

MIN_FLIP = 0.10   # req 7 stop floor
N = 60


def _striker_inputs(world):
    """Build (obs, teammate_slots) for the striker (ego = robot_pos) from a world spec."""
    sx, sy = float(world.robot_x), float(world.robot_y)
    bx, by = float(world.ball_x), float(world.ball_y)
    base = {f.name: 0.0 for f in dataclasses.fields(BackendObservation)}
    base["teammate_poses"] = (); base["teammate_ages_ms"] = ()
    base.update(
        robot_x=sx, robot_y=sy, ball_x=bx, ball_y=by,
        ball_rel_x=bx - sx, ball_rel_y=by - sy,
        shot_opening_with_obstacles=0.20, can_score_now=0.0,
        nearest_opponent_front_dist=0.0,   # CONSTANT (no opponent) in A and B
        global_team_ball_valid=True, global_team_ball_x_mm=bx, global_team_ball_y_mm=by,
    )
    obs = BackendObservation(**base)
    slots = []
    for i, tm in enumerate(world.teammates):
        x, y, num = float(tm.x), float(tm.y), int(tm.number)
        is_gk = x < mt.GK_EXCLUDE_X_MM
        slots.append(TeammateSlot(slot_idx=i, agent_id=num, x_field=x, y_field=y, is_gk=is_gk))
    return obs, slots


def main():
    gd = GateDecision(shoot_armed=False, dribble_armed=True, shoot_arm_progress=1.0,
                      allowed_skills=tuple(), pass_armed=True)
    rng = np.random.default_rng(7)
    flips = a_pass = b_pass = 0
    for _ in range(N):
        wa = pc._team_giveandgo_world(rng, outlet_present=True)
        wb = pc._team_giveandgo_world(rng, outlet_present=False)
        oa, sa = _striker_inputs(wa)
        ob, sb = _striker_inputs(wb)
        act_a = mt.teacher_striker(oa, 7, None, gd, sa, None, 0.0).skill_name
        act_b = mt.teacher_striker(ob, 7, None, gd, sb, None, 0.0).skill_name
        a_pass += int(act_a == "pass")
        b_pass += int(act_b == "pass")
        flips += int(act_a != act_b)
    flip_rate = flips / N
    print(f"[teammate-flip-gate] N={N}  A(outlet present) pass_rate={a_pass/N:.2f}  "
          f"B(outlet absent) pass_rate={b_pass/N:.2f}  FLIP_RATE={flip_rate:.2f}  (stop if <{MIN_FLIP})")
    # opponents are identical (none) in A/B, so the flip is teammate-attributable by construction.
    if flip_rate < MIN_FLIP:
        print("[teammate-flip-gate] STOP — teacher does NOT condition on the teammate; "
              "do not train (fix the contrast).")
        sys.exit(1)
    print("[teammate-flip-gate] PASS — teacher conditions on the teammate "
          "(flip is teammate-attributable; opponents constant). Safe to build corpus + train.")
    sys.exit(0)


if __name__ == "__main__":
    main()
