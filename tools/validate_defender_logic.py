"""Windows-runnable static+unit validation of the defender v2 (doc-34 D1-D5) teacher logic.

The full doc-34 pipeline (dataset/train/sim/eval, Pasos 1-8) needs the Linux pybh
bridge and cannot run on Windows (see memory: windows-rl-sim-blocker). But the new
*teacher* logic Rafa added is pure Python and only depends on the BackendObservation /
WorldSpec dataclasses, which construct fine without the live bridge. This harness stubs
past the two module-load walls in pybh_backend (`os.chdir(BHUMAN_DIR)` and
`import pybh.controller`) and exercises D1/D2/D3/D5 on crafted recovery observations so
we catch wrong-constant / wrong-sign / AttributeError bugs *before* Luis burns sim hours.

Run:  .venv/Scripts/python.exe tools/validate_defender_logic.py
Exit 0 = all checks pass.  Prints a per-check semaphore.
"""
from __future__ import annotations

import math
import os
import sys
import types

# --- stub past pybh_backend module-load side effects (Windows has no POSIX bridge) ---
_fake = types.ModuleType("pybh")
_fake_controller = types.ModuleType("pybh.controller")
_fake.controller = _fake_controller
sys.modules["pybh"] = _fake
sys.modules["pybh.controller"] = _fake_controller
# pybh_backend does os.chdir(BHUMAN_DIR) at import; BHUMAN_DIR doesn't exist on this box.
_real_chdir = os.chdir
os.chdir = lambda *a, **k: None  # noqa: E731 - restored right after import

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiagent_teacher as mt  # noqa: E402
import production_curriculum as pc  # noqa: E402
from gate import GateDecision  # noqa: E402
from action import SkillType  # noqa: E402
from pybh_backend import BackendObservation  # noqa: E402

os.chdir = _real_chdir

PASS, FAIL = "[GREEN]", "[RED]"
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = PASS if cond else FAIL
    print(f"  {tag} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        _failures.append(name)


def obs(**kw) -> BackendObservation:
    """BackendObservation with sane defaults for an in-control deep recovery."""
    base = dict(
        ball_x=0.0, ball_y=0.0, robot_x=0.0, robot_y=0.0, robot_theta=0.0,
        simrobot_robot_x=0.0, simrobot_robot_y=0.0, simrobot_robot_theta=0.0,
        frame=0, obs_ready=True,
        ball_rel_x=0.0, ball_rel_y=0.0, time_since_ball_seen=0.0,
        nearest_opponent_front_dist=5000.0,
    )
    base.update(kw)
    return BackendObservation(**base)


def slot(agent_id, x, y, is_gk=False, idx=0) -> mt.TeammateSlot:
    return mt.TeammateSlot(slot_idx=idx, agent_id=agent_id, x_field=x, y_field=y, is_gk=is_gk)


print("=" * 70)
print("DEFENDER v2 (doc-34 D1-D5) -- Windows static+unit validation")
print("=" * 70)

# --- constants sanity (D1-D3 env-tunable knobs) -------------------------------
print("\n[constants] D1-D3 knobs present and sane:")
check("BALL_CONTROL_RADIUS_MM", mt.BALL_CONTROL_RADIUS_MM == 350.0, f"{mt.BALL_CONTROL_RADIUS_MM}")
check("DEFENSIVE_CLEAR_X_MM negative (own third)", mt.DEFENSIVE_CLEAR_X_MM < 0, f"{mt.DEFENSIVE_CLEAR_X_MM}")
check("CLEAR_PASS_MIN_UPFIELD_MM > 0", mt.CLEAR_PASS_MIN_UPFIELD_MM > 0, f"{mt.CLEAR_PASS_MIN_UPFIELD_MM}")
check("EMA alpha lowered (P5 still target)", mt.EMA_ALPHA_OPEN_LANE <= 0.15 and mt.EMA_ALPHA_TRIANGLE <= 0.10,
      f"open={mt.EMA_ALPHA_OPEN_LANE} tri={mt.EMA_ALPHA_TRIANGLE}")

# --- D1: physical ball control overrides role hysteresis ----------------------
print("\n[D1] ball control promotes recoverer to striker, bypassing hysteresis:")
# Agent 3 is the prior striker far up-field; agent 2 just won the ball deep with control.
recoverer = obs(robot_x=-3000.0, robot_y=200.0, ball_x=-3050.0, ball_y=210.0,
                ball_rel_x=-50.0, ball_rel_y=10.0, time_since_ball_seen=0.0)   # |ball_rel|~51 < 350
old_striker = obs(robot_x=1000.0, robot_y=0.0, ball_x=-3050.0, ball_y=210.0,
                  ball_rel_x=4050.0, ball_rel_y=-210.0, time_since_ball_seen=0.0)
hist = mt.TeamRoleHistory(current_roles={3: "striker", 2: "off_ball_support"})
roles = mt.assign_roles({2: recoverer, 3: old_striker},
                        team_ball=(-3050.0, 210.0, True), history=hist, phase="defense")
check("recoverer (2) becomes striker despite prior striker (3)", roles.get(2) == "striker", str(roles))

# Control of the role should NOT flip without ball control (hysteresis holds prev striker).
hist2 = mt.TeamRoleHistory(current_roles={3: "striker", 2: "off_ball_support"})
no_control = obs(robot_x=-3000.0, robot_y=200.0, ball_x=-2400.0, ball_y=210.0,
                 ball_rel_x=600.0, ball_rel_y=10.0, time_since_ball_seen=0.0)  # |ball_rel|=600 > 350
old_str2 = obs(robot_x=-2300.0, robot_y=0.0, ball_x=-2400.0, ball_y=210.0,
               ball_rel_x=-100.0, ball_rel_y=210.0, time_since_ball_seen=0.0)
roles2 = mt.assign_roles({2: no_control, 3: old_str2},
                         team_ball=(-2400.0, 210.0, True), history=hist2, phase="defense")
check("without control, hysteresis keeps prior striker (3) for >=1 frame",
      roles2.get(3) == "striker", str(roles2))

# --- defensive_clear_pass_target (D2): picks most up-field outlet --------------
print("\n[D2] defensive_clear_pass_target picks safest up-field outlet:")
deep = obs(robot_x=-3000.0, robot_y=100.0, ball_x=-3000.0, ball_y=100.0,
           nearest_opponent_front_dist=5000.0)
slots = [slot(2, -1000.0, 800.0), slot(3, -3400.0, -200.0), slot(4, -4300.0, 0.0, is_gk=True)]
outlet = mt.defensive_clear_pass_target(deep, slots)
check("outlet = most up-field non-GK teammate (id 2)", outlet == 2, f"got {outlet}")

# GK is never a clearance target even if 'up-field'.
slots_only_gk = [slot(4, 500.0, 0.0, is_gk=True)]
check("GK never chosen as outlet", mt.defensive_clear_pass_target(deep, slots_only_gk) == -1)

# Blocked front lane -> no pass (clear/drive instead).
deep_pressed = obs(robot_x=-3000.0, robot_y=100.0, ball_x=-3000.0, ball_y=100.0,
                   nearest_opponent_front_dist=300.0)  # < OPPONENT_LANE_CLEARANCE_MM(800)
check("blocked front lane -> no pass outlet",
      mt.defensive_clear_pass_target(deep_pressed, slots) == -1)

# No teammate far enough up-field -> no outlet.
slots_behind = [slot(3, -3400.0, -200.0)]  # behind the ball
check("no up-field teammate -> no outlet",
      mt.defensive_clear_pass_target(deep, slots_behind) == -1)

# --- defender_clear_armed (D3): clearance-shoot gate --------------------------
print("\n[D3] defender_clear_armed only when deep + central + no outlet:")
central_no_outlet = obs(robot_x=-3000.0, robot_y=100.0, ball_x=-3000.0, ball_y=100.0,
                        nearest_opponent_front_dist=300.0)  # lane blocked => no outlet
check("deep+central+no outlet -> clearance armed",
      mt.defender_clear_armed(central_no_outlet, slots) is True)
check("safe pass exists -> NOT clearance armed (prefer pass)",
      mt.defender_clear_armed(deep, slots) is False)
not_deep = obs(robot_x=-1000.0, robot_y=100.0, ball_x=-1000.0, ball_y=100.0,
               nearest_opponent_front_dist=300.0)
check("not deep enough -> NOT armed", mt.defender_clear_armed(not_deep, slots) is False)
wide = obs(robot_x=-3000.0, robot_y=1800.0, ball_x=-3000.0, ball_y=1800.0,
           nearest_opponent_front_dist=300.0)  # |ball_y|>1500 not central
check("deep but wide (not central) -> NOT armed", mt.defender_clear_armed(wide, slots) is False)

# --- D2/D3 routing inside teacher_striker -------------------------------------
print("\n[D2/D3] teacher_striker routes recovery -> pass / clear-shoot / drive-out:")
def gd(**kw):
    base = dict(shoot_armed=False, dribble_armed=False, shoot_arm_progress=0.0,
                allowed_skills=("stand", "walk", "shoot", "pass", "dribble", "block", "mark", "observe"))
    base.update(kw)
    return GateDecision(**base)

# (1) safe pass available + pass armed -> PASS to the outlet
act = mt.teacher_striker(deep, stage_id=8, prev_skill_name=None,
                         gate_decision=gd(dribble_armed=True, pass_armed=True),
                         teammate_slots=slots, last_pass_target=None, last_pass_target_age_ms=9e9)
check("(1) outlet+pass_armed -> pass to id 2",
      act.skill_type == SkillType.pass_ and int(act.pass_target) == 2,
      f"{act.skill_type} target={act.pass_target}")

# (2) no outlet (lane blocked) + shoot armed + central -> clearance SHOOT
act2 = mt.teacher_striker(central_no_outlet, stage_id=8, prev_skill_name=None,
                          gate_decision=gd(shoot_armed=True),
                          teammate_slots=slots, last_pass_target=None, last_pass_target_age_ms=9e9)
check("(2) no outlet + shoot_armed + central -> shoot (clearance)",
      act2.skill_type == SkillType.shoot, str(act2.skill_type))

# (3) controlled, deep, no shoot arm, no pass -> DRIVE OUT (dribble) up-field
act3 = mt.teacher_striker(central_no_outlet, stage_id=8, prev_skill_name=None,
                          gate_decision=gd(dribble_armed=True),
                          teammate_slots=slots, last_pass_target=None, last_pass_target_age_ms=9e9)
drives_out = act3.skill_type in (SkillType.dribble, SkillType.walk) and float(act3.target_x) > float(central_no_outlet.ball_x)
check("(3) no shoot/pass -> drive ball out toward midfield (target_x up-field)",
      drives_out, f"{act3.skill_type} target_x={act3.target_x:.0f} (ball_x={central_no_outlet.ball_x:.0f})")

# (4) NOT deep -> clearance logic must NOT hijack; falls through to normal striker
shallow = obs(robot_x=2000.0, robot_y=0.0, ball_x=2000.0, ball_y=0.0,
              nearest_opponent_front_dist=5000.0, can_score_now=True,
              shot_quality_no_obstacles=0.9)
act4 = mt.teacher_striker(shallow, stage_id=8, prev_skill_name=None,
                          gate_decision=gd(shoot_armed=True),
                          teammate_slots=slots, last_pass_target=None, last_pass_target_age_ms=9e9)
check("(4) shallow/attacking third -> clearance does NOT fire (normal striker path runs)",
      act4 is not None)

# --- D5: new recovery scenario world builds & spawns ball on a recoverer -------
print("\n[D5] _defense_recovery_world builds valid spawns:")
import numpy as np
rng = np.random.default_rng(123)
ok_world = True
detail = ""
for _ in range(200):
    w = pc._defense_recovery_world(rng)
    bx, by = float(w.ball_x), float(w.ball_y)
    rx, ry = float(w.robot_x), float(w.robot_y)
    d = math.hypot(rx - bx, ry - by)
    if not (-4500 <= bx <= 4500 and -3000 <= by <= 3000):
        ok_world, detail = False, f"ball OOB ({bx:.0f},{by:.0f})"; break
    if d >= mt.BALL_CONTROL_RADIUS_MM:
        ok_world, detail = False, f"recoverer not in control radius (d={d:.0f})"; break
    if bx >= mt.DEFENSIVE_CLEAR_X_MM:
        ok_world, detail = False, f"ball not in defensive third (x={bx:.0f})"; break
check("recovery world: ball in own third, recoverer within control radius (200 seeds)",
      ok_world, detail or "all 200 ok")

# scenario registered with weight
scn = pc.team_defense_scenarios(4)
names = [s.name for s in scn]
check("defense_recovery_clear registered in team_defense_scenarios", "defense_recovery_clear" in names, str(names))

print("\n" + "=" * 70)
if _failures:
    print(f"RESULT: {len(_failures)} FAILED -> {_failures}")
    sys.exit(1)
print("RESULT: ALL CHECKS GREEN -- defender D1-D5 teacher logic is sound on synthetic inputs")
print("(closed-loop sim metrics still require Luis's Linux box -- see docs/notes/HANDOFF.md)")
sys.exit(0)
