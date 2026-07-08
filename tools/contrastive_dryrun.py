#!/usr/bin/env python3
"""doc-35 §4.1 — contrastive teacher dry-run gate (OFFLINE, no sim).

Verifies the v3 teacher logic is genuinely teammate-CONTINGENT: on matched A/B
ego-states (teammate open vs covered), the teacher must emit DIFFERENT actions.
If A and B give the same action the contingency doesn't exist and the whole
dataset-v3 pipeline would be silently useless — so this is the gate to clear
BEFORE committing to the long sim regen.

Gate: >= 80% of contrastive pairs differ. Exits non-zero on failure.

Run: PYTHONPATH=...:SabanaHerons.../Make/Python python tools/contrastive_dryrun.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

# Self-bootstrap sys.path so this runs with a bare `python tools/contrastive_dryrun.py`
# (no PYTHONPATH needed): add the RL repo root (for multiagent_teacher/gate/pybh_backend)
# and the sibling SabanaHerons2026 Make/Python (for the compiled pybh extension).
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO.parent / "SabanaHerons2026" / "Make" / "Python"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import multiagent_teacher as mt
from gate import GateDecision
from pybh_backend import BackendObservation


def _obs(**kw) -> BackendObservation:
    base = {f.name: 0.0 for f in dataclasses.fields(BackendObservation)}
    # sensible non-scalar defaults
    base["teammate_poses"] = ()
    base["teammate_ages_ms"] = ()
    base.update(kw)
    return BackendObservation(**base)


def _gd(**kw) -> GateDecision:
    base = dict(shoot_armed=False, dribble_armed=False, shoot_arm_progress=0.0, allowed_skills=tuple())
    base.update(kw)
    return GateDecision(**base)


def check_support_hold() -> tuple[int, int, list[str]]:
    """A: striker controls + lane clear -> stand. B: lane covered -> walk."""
    diffs, total, notes = 0, 0, []
    slot = (2300.0, 1300.0)
    # support sitting ON its slot in both variants; team ball controlled by a teammate.
    common = dict(
        robot_x=slot[0], robot_y=slot[1],
        global_team_ball_valid=True, global_team_ball_x_mm=1500.0, global_team_ball_y_mm=0.0,
        teammate_poses=((1450.0, 50.0, 0.0),),   # striker ~on the ball (within control radius)
        teammate_ages_ms=(0.0,),
    )
    # Convention: a large front distance = no opponent (clear); a small one = covered.
    A = _obs(**common, nearest_opponent_front_dist=3000.0)     # lane clear
    B = _obs(**common, nearest_opponent_front_dist=400.0)      # lane covered (<700)
    for role_fn in (mt.teacher_open_support, mt.teacher_off_ball_support):
        a = role_fn(A, 7, None, _gd(), slot).skill_name
        b = role_fn(B, 7, None, _gd(), slot).skill_name
        total += 1
        ok = (a != b)
        diffs += int(ok)
        notes.append(f"  {role_fn.__name__}: A={a} B={b} {'DIFF✓' if ok else 'SAME✗'}")
    return diffs, total, notes


def check_striker_pass() -> tuple[int, int, list[str]]:
    """A: open up-field outlet -> pass. B: front opponent blocks lane -> shoot/dribble."""
    diffs, total, notes = 0, 0, []
    outlet = mt.TeammateSlot(slot_idx=0, agent_id=2, x_field=2700.0, y_field=1200.0, is_gk=False)
    # own_value below SHOOT_THRESHOLD so the auction runs; ball in hand.
    common = dict(robot_x=1500.0, robot_y=0.0, ball_x=1600.0, ball_y=0.0,
                  ball_rel_x=100.0, ball_rel_y=0.0, shot_opening_with_obstacles=0.20, can_score_now=0.0)
    A = _obs(**common, nearest_opponent_front_dist=3000.0)     # lane clear
    B = _obs(**common, nearest_opponent_front_dist=400.0)      # opponent in front (<800 clearance)
    gd = _gd(dribble_armed=True, shoot_arm_progress=1.0, pass_armed=True)
    a = mt.teacher_striker(A, 7, None, gd, [outlet], None, 0.0).skill_name
    b = mt.teacher_striker(B, 7, None, gd, [outlet], None, 0.0).skill_name
    total += 1
    ok = (a != b)
    diffs += int(ok)
    notes.append(f"  teacher_striker: A={a} B={b} {'DIFF✓' if ok else 'SAME✗'}")
    return diffs, total, notes


def main() -> None:
    print(f"[contrastive-dryrun] CONTINGENT_HOLD={mt.CONTINGENT_HOLD}")
    diffs, total, notes = 0, 0, []
    for fn in (check_support_hold, check_striker_pass):
        d, t, n = fn()
        diffs += d; total += t; notes += n
    for line in notes:
        print(line)
    rate = diffs / max(1, total)
    print(f"[contrastive-dryrun] {diffs}/{total} pairs differ ({rate:.0%}); gate = >= 80%")
    if rate >= 0.80:
        print("[contrastive-dryrun] PASS — teacher is teammate-contingent; safe to regen.")
        sys.exit(0)
    print("[contrastive-dryrun] FAIL — contingency missing; fix the teacher before regen.")
    sys.exit(1)


if __name__ == "__main__":
    main()
