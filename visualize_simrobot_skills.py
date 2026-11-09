from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from action import SkillType, StructuredSkillAction
from config import SIMROBOT_BIN, SIMROBOT_SCENE_PRESETS
from pybh_backend import PyBHBackend


BALL_POS = np.array([0.0, 0.0], dtype=np.float32)
ROBOT_FAR_POS = np.array([-900.0, 0.0], dtype=np.float32)
ROBOT_NEAR_POS = np.array([-180.0, 0.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize all currently available RL skills in visible SimRobot.")
    parser.add_argument("--simrobot-profile", choices=("2d", "3d"), default="2d")
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--simrobot-bin", type=Path, default=SIMROBOT_BIN)
    parser.add_argument("--launch", action="store_true", help="Launch SimRobot automatically.")
    parser.add_argument("--timeout-ms", type=int, default=1500)
    parser.add_argument("--frames-per-step", type=int, default=1)
    parser.add_argument("--steps-per-skill", type=int, default=18)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--cycles", type=int, default=1)
    return parser.parse_args()


def reset_pose_for_skill(skill: SkillType) -> tuple[np.ndarray, float]:
    if skill in {SkillType.shoot, SkillType.pass_, SkillType.dribble}:
        return ROBOT_NEAR_POS.copy(), 0.0
    return ROBOT_FAR_POS.copy(), 0.0


def build_action(skill: SkillType) -> StructuredSkillAction:
    if skill is SkillType.walk:
        return StructuredSkillAction(skill_type=skill, target_x=1200.0, target_y=0.0, target_theta=0.0)
    if skill is SkillType.shoot:
        return StructuredSkillAction(skill_type=skill)
    if skill is SkillType.pass_:
        return StructuredSkillAction(skill_type=skill, pass_target=2)
    if skill is SkillType.dribble:
        return StructuredSkillAction(skill_type=skill, target_theta=0.0)
    if skill is SkillType.block:
        return StructuredSkillAction(skill_type=skill, target_x=700.0, target_y=200.0, target_theta=0.0)
    if skill is SkillType.mark:
        return StructuredSkillAction(skill_type=skill, target_x=1000.0, target_y=-350.0, target_theta=0.0)
    if skill is SkillType.observe:
        return StructuredSkillAction(skill_type=skill, target_x=1200.0, target_y=450.0, target_theta=0.0)
    return StructuredSkillAction(skill_type=skill)


def describe_action(action: StructuredSkillAction) -> str:
    if action.skill is SkillType.walk:
        return f"walkToField({action.target_x:.0f}, {action.target_y:.0f}, {action.target_theta:.2f})"
    if action.skill is SkillType.pass_:
        return f"pass(target={action.pass_target})"
    if action.skill in {SkillType.block, SkillType.mark, SkillType.observe}:
        return f"{action.skill_name}({action.target_x:.0f}, {action.target_y:.0f}, {action.target_theta:.2f})"
    if action.skill is SkillType.dribble:
        return f"dribble(theta={action.target_theta:.2f})"
    return action.skill_name


def run_skill(
    backend: PyBHBackend,
    skill: SkillType,
    *,
    steps_per_skill: int,
    hz: float,
    hold_seconds: float,
) -> None:
    robot_pos, robot_theta = reset_pose_for_skill(skill)
    action = build_action(skill)
    reset_obs = backend.reset(ball_pos=BALL_POS.copy(), robot_pos=robot_pos, robot_theta=robot_theta)
    print(
        "reset skill={skill} robot=({rx:.1f},{ry:.1f},{rt:.2f}) ball=({bx:.1f},{by:.1f})".format(
            skill=skill.public_name,
            rx=reset_obs.robot_x,
            ry=reset_obs.robot_y,
            rt=reset_obs.robot_theta,
            bx=reset_obs.ball_x,
            by=reset_obs.ball_y,
        )
    )
    period = 1.0 / max(hz, 0.1)
    print(f"run   skill={skill.public_name} action={describe_action(action)}")
    for step in range(steps_per_skill):
        obs = backend.step(action)
        print(
            "step={step:02d} skill={skill} frame={frame} requested={requested}:{pass_target} "
            "robot=({rx:.1f},{ry:.1f},{rt:.2f}) ball=({bx:.1f},{by:.1f})".format(
                step=step + 1,
                skill=skill.public_name,
                frame=obs.frame,
                requested=obs.requested_skill,
                pass_target=obs.requested_pass_target,
                rx=obs.robot_x,
                ry=obs.robot_y,
                rt=obs.robot_theta,
                bx=obs.ball_x,
                by=obs.ball_y,
            )
        )
        time.sleep(period)
    if hold_seconds > 0.0:
        time.sleep(hold_seconds)


def main() -> None:
    args = parse_args()
    if args.scene is None:
        args.scene = SIMROBOT_SCENE_PRESETS[args.simrobot_profile].scene_path
    backend = PyBHBackend(
        mode="simrobot_visible",
        frames_per_step=args.frames_per_step,
        obs_timeout_ms=args.timeout_ms,
        simrobot_profile=args.simrobot_profile,
        simrobot_scene_path=args.scene,
        simrobot_launch=args.launch,
        simrobot_bin=args.simrobot_bin,
    )
    print("Visible SimRobot skill visualizer")
    print(f"profile={args.simrobot_profile}")
    print(f"scene={args.scene}")
    print(f"cycles={args.cycles} steps_per_skill={args.steps_per_skill} hz={args.hz:.1f}")
    print("If SimRobot is open, live motion should start once the scene finishes loading.")
    backend.start()
    try:
        for cycle in range(max(1, args.cycles)):
            print(f"cycle={cycle + 1}")
            for skill in SkillType:
                run_skill(
                    backend,
                    skill,
                    steps_per_skill=max(1, args.steps_per_skill),
                    hz=args.hz,
                    hold_seconds=max(0.0, args.hold_seconds),
                )
        backend.reset(ball_pos=BALL_POS.copy(), robot_pos=ROBOT_FAR_POS.copy(), robot_theta=-math.pi / 2.0)
        backend.step(StructuredSkillAction(skill_type=SkillType.stand))
        print("done final_action=stand")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
