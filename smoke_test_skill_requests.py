from __future__ import annotations

import argparse
import math

import numpy as np

from action import SkillType, StructuredSkillAction
from pybh_backend import PyBHBackend


def build_probe_action(skill: SkillType) -> StructuredSkillAction:
    if skill is SkillType.walk:
        return StructuredSkillAction(skill_type=skill, target_x=500.0, target_y=0.0, target_theta=0.0)
    if skill is SkillType.pass_:
        return StructuredSkillAction(skill_type=skill, pass_target=2)
    if skill is SkillType.dribble:
        return StructuredSkillAction(skill_type=skill, target_theta=0.0)
    if skill in (SkillType.block, SkillType.mark, SkillType.observe):
        return StructuredSkillAction(skill_type=skill, target_x=700.0, target_y=150.0)
    return StructuredSkillAction(skill_type=skill)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke every public SkillRequest variant through pybh")
    parser.add_argument("--backend", choices=("simrobot", "cpp", "python"), default="cpp")
    args = parser.parse_args()

    backend = PyBHBackend(mode=args.backend, frames_per_step=3 if args.backend == "simrobot" else 1, obs_timeout_ms=1000)
    backend.start()
    try:
        for skill in SkillType:
            start_ball = np.array([500.0, 0.0], dtype=np.float32)
            start_robot = np.array([-300.0, 0.0], dtype=np.float32)
            backend.reset(ball_pos=start_ball, robot_pos=start_robot, robot_theta=0.0)
            action = build_probe_action(skill)
            obs = backend.step(action)
            print(
                f"{action.skill_name:7s} "
                f"obs_ready={int(obs.obs_ready)} frame={obs.frame} "
                f"requested={obs.requested_skill}:{obs.requested_pass_target} "
                f"robot=({obs.robot_x:.1f},{obs.robot_y:.1f},{obs.robot_theta:.2f}) "
                f"ball=({obs.ball_x:.1f},{obs.ball_y:.1f})"
            )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
