#!/usr/bin/env python3
"""Design-A recompute invariant for the defender (off_ball_support) block/mark/pass.

For a sweep of synthetic defender states, run teacher_defender to get the emitted
skill, build the 47-dim obs columns the env would store from the SAME inputs, run
train.gate_skill_mask_from_obs, and assert the emitted skill is ALLOWED (mask==1).

If any teacher label is masked to zero, BC would silently drop it (the gate-mark-leak
class of bug) — so this must print ALL-PASS before regenerating the dataset.
"""
import math
import numpy as np
import torch

from rlfootball.core.action import SKILL_ORDER, SkillType
from rlfootball.core.observation import OBS_SIZE, ObsIndex
from rlfootball.core.gate import GateDecision
from rlfootball.training.train import gate_skill_mask_from_obs
from rlfootball.teachers import multiagent_teacher as mt
from rlfootball.core.pybh_backend import BackendObservation

X_RANGE = 4500.0
SKILL_IDX = {s: i for i, s in enumerate(SKILL_ORDER)}


def stored_obs(obs: BackendObservation, gd: GateDecision, role="off_ball_support") -> torch.Tensor:
    """Build only the columns gate_skill_mask_from_obs reads (rest zero)."""
    v = torch.zeros((1, OBS_SIZE), dtype=torch.float32)
    v[0, int(ObsIndex.ball_rel_x)] = float(np.clip(obs.ball_rel_x / X_RANGE, -1.0, 1.0))
    v[0, int(ObsIndex.nearest_opponent_front_dist)] = float(
        np.clip(obs.nearest_opponent_front_dist / X_RANGE, 0.0, 1.0)
    )
    v[0, int(ObsIndex.shoot_armed)] = 1.0 if gd.shoot_armed else 0.0
    v[0, int(ObsIndex.dribble_armed)] = 1.0 if gd.dribble_armed else 0.0
    v[0, int(ObsIndex.pass_armed)] = 1.0 if getattr(gd, "pass_armed", False) else 0.0
    v[0, int(ObsIndex.observe_armed)] = 1.0 if getattr(gd, "observe_armed", False) else 0.0
    v[0, int(ObsIndex.is_off_ball_support)] = 1.0 if role == "off_ball_support" else 0.0
    return v


def run():
    fails = 0
    checked = 0
    emitted = {}
    # Sweep: ball in/out of front cone, opponent in/out of mark range, dribble on/off.
    robot_x, robot_y = -3000.0, 200.0
    coord = (robot_x, robot_y)            # dist_to_cover = 0 (near cover)
    face_ball = (-1000.0, 200.0)          # team ball in our half, central
    for ball_rel_x in (-500.0, 0.0, 400.0, 900.0, 1700.0, 1799.0, 1900.0, 2500.0):
        for opp_front in (0.0, 600.0, 1000.0, 1300.0, 1499.0, 1600.0, 2200.0):
            for opp_angle in (-0.6, 0.0, 0.6):
                for dribble_armed in (False, True):
                    gd = GateDecision(
                        shoot_armed=False, dribble_armed=dribble_armed,
                        shoot_arm_progress=0.0, allowed_skills=(),
                        is_off_ball_support=True,
                    )
                    obs = BackendObservation(
                        robot_x=robot_x, robot_y=robot_y, robot_theta=0.0,
                        simrobot_robot_x=robot_x, simrobot_robot_y=robot_y,
                        simrobot_robot_theta=0.0, frame=0, obs_ready=True,
                        ball_rel_x=ball_rel_x, ball_rel_y=0.0,
                        ball_x=face_ball[0], ball_y=face_ball[1],
                        nearest_opponent_front_dist=opp_front,
                        nearest_opponent_dist=max(opp_front, 1.0),
                        nearest_opponent_angle=opp_angle,
                        time_since_ball_seen=0.0,
                    )
                    act = mt.teacher_defender(
                        obs, stage_id=8, prev_skill_name=None, gate_decision=gd,
                        coordination_target=coord, team_ball_valid=True,
                        face_ball_xy=face_ball,
                    )
                    skill = SkillType(int(act.skill_type)).public_name
                    emitted[skill] = emitted.get(skill, 0) + 1
                    mask = gate_skill_mask_from_obs(stored_obs(obs, gd))[0]
                    checked += 1
                    if mask[SKILL_IDX[skill]] < 0.5:
                        fails += 1
                        print(f"  MASKED: skill={skill} ball_rel_x={ball_rel_x} "
                              f"opp_front={opp_front} dribble={dribble_armed}")
    print(f"checked={checked} fails={fails}")
    print("teacher skill histogram:", emitted)
    # Determinism of recompute.
    o = stored_obs(
        BackendObservation(
            robot_x=0.0, robot_y=0.0, robot_theta=0.0,
            simrobot_robot_x=0.0, simrobot_robot_y=0.0, simrobot_robot_theta=0.0,
            frame=0, obs_ready=True, ball_x=0.0, ball_y=0.0,
            nearest_opponent_front_dist=900.0),
        GateDecision(shoot_armed=False, dribble_armed=False,
                     shoot_arm_progress=0.0, allowed_skills=(),
                     is_off_ball_support=True),
    )
    det = torch.equal(gate_skill_mask_from_obs(o), gate_skill_mask_from_obs(o))
    print("recompute deterministic:", det)
    ok = (fails == 0) and det
    print("RESULT:", "ALL-PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
