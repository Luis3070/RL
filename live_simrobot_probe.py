from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from config import BHUMAN_DIR, PLAYER_NUMBER, SIMROBOT_SCENE_PRESETS
from action import SkillType, StructuredSkillAction
from pybh_backend import adapt_action_to_backend

os.chdir(BHUMAN_DIR)
import pybh.controller as pybh


SIMROBOT_BIN = BHUMAN_DIR / "Build/Linux/SimRobot/Develop/SimRobot"
LIVE_SHOOT_DISTANCE_MM = 260.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive a visible SimRobot scene from Python through RLSharedState."
    )
    parser.add_argument("--player-number", type=int, default=PLAYER_NUMBER)
    parser.add_argument("--simrobot-profile", choices=("2d", "3d"), default="2d")
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--simrobot-bin", type=Path, default=SIMROBOT_BIN)
    parser.add_argument("--launch", action="store_true", help="Launch SimRobot with the selected scene.")
    parser.add_argument(
        "--policy",
        choices=("heuristic", "fixed"),
        default="heuristic",
        help="Action source for the visible robot.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Duration of the probe. Omit for continuous mode until Ctrl+C.",
    )
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum number of loop iterations. Omit for continuous mode.",
    )
    parser.add_argument("--timeout-ms", type=int, default=1500)
    return parser.parse_args()


def maybe_launch_simrobot(simrobot_bin: Path, scene: Path) -> subprocess.Popen[str] | None:
    if not simrobot_bin.exists():
        raise FileNotFoundError(f"SimRobot binary not found: {simrobot_bin}")
    if not scene.exists():
        raise FileNotFoundError(f"Scene file not found: {scene}")
    return subprocess.Popen([str(simrobot_bin), str(scene)], cwd=BHUMAN_DIR)


def send_phase_action(player_number: int, phase: int, obs: dict | None) -> str:
    if phase % 4 == 0:
        action = StructuredSkillAction(skill_type=SkillType.walk, target_x=1200.0, target_y=0.0, target_theta=0.0)
    elif phase % 4 == 1:
        action = StructuredSkillAction(
            skill_type=SkillType.walk, target_x=1200.0, target_y=700.0, target_theta=math.pi / 4.0
        )
    elif phase % 4 == 2:
        pybh.rl_set_action(player_number, "dribble", 0.0, 0.0, 0.0)
        return "dribble(0)"
    else:
        pybh.rl_set_action(player_number, "stand", 0.0, 0.0, 0.0)
        return "stand"

    adapted_action = adapt_action_to_backend(action, obs, mode="python")
    pybh.rl_set_action(
        player_number,
        "walkTo",
        float(adapted_action.target_x),
        float(adapted_action.target_y),
        float(adapted_action.target_theta),
    )
    return f"walkToRel({adapted_action.target_x:.0f}, {adapted_action.target_y:.0f}, {adapted_action.target_theta:.2f})"


def send_heuristic_action(player_number: int, obs: dict) -> str:
    robot_pos = np.array([float(obs.get("robot_x", 0.0)), float(obs.get("robot_y", 0.0))], dtype=np.float32)
    ball_pos = np.array([float(obs.get("ball_x", 0.0)), float(obs.get("ball_y", 0.0))], dtype=np.float32)
    delta = ball_pos - robot_pos
    distance = float(np.linalg.norm(delta))
    if distance <= LIVE_SHOOT_DISTANCE_MM:
        pybh.rl_set_action(player_number, "shoot", 0.0, 0.0, 0.0)
        return "shoot"

    global_action = StructuredSkillAction(
        skill_type=SkillType.walk,
        target_x=float(ball_pos[0]),
        target_y=float(ball_pos[1]),
        target_theta=float(math.atan2(float(delta[1]), float(delta[0]))),
    )
    adapted_action = adapt_action_to_backend(global_action, obs, mode="python")
    pybh.rl_set_action(
        player_number,
        "walkTo",
        float(adapted_action.target_x),
        float(adapted_action.target_y),
        float(adapted_action.target_theta),
    )
    return (
        f"walkToRel({adapted_action.target_x:.0f}, "
        f"{adapted_action.target_y:.0f}, {adapted_action.target_theta:.2f})"
    )


def main() -> None:
    args = parse_args()
    if args.scene is None:
        args.scene = SIMROBOT_SCENE_PRESETS[args.simrobot_profile].scene_path
    simrobot_process = maybe_launch_simrobot(args.simrobot_bin, args.scene) if args.launch else None

    print("Visible SimRobot probe")
    print(f"scene={args.scene}")
    if args.launch:
        print(f"launched={args.simrobot_bin}")
    print(f"policy={args.policy}")
    print("If the SimRobot window is open, start the simulation with F5 before expecting observations.")
    print("Press Ctrl+C to stop.")

    period = 1.0 / max(args.hz, 0.1)
    phase_steps = max(1, int(round(args.hz * 2.0)))
    if args.steps is not None:
        total_steps = max(1, args.steps)
    elif args.seconds is not None:
        total_steps = max(1, int(round(args.seconds * args.hz)))
    else:
        total_steps = None

    try:
        step = 0
        last_obs: dict | None = None
        while total_steps is None or step < total_steps:
            if args.policy == "heuristic" and last_obs is not None and last_obs.get("obs_ready", False):
                action_name = send_heuristic_action(args.player_number, last_obs)
            else:
                action_name = send_phase_action(args.player_number, step // phase_steps, last_obs)
            obs = pybh.rl_get_obs(args.player_number, args.timeout_ms)
            last_obs = obs
            if not obs.get("obs_ready", False):
                print(f"step={step + 1} action={action_name} obs_ready=0")
            else:
                print(
                    "step={step} action={action} frame={frame} "
                    "robot=({rx:.1f},{ry:.1f},{rt:.2f}) ball=({bx:.1f},{by:.1f})".format(
                        step=step + 1,
                        action=action_name,
                        frame=int(obs.get("frame", 0)),
                        rx=float(obs.get("robot_x", 0.0)),
                        ry=float(obs.get("robot_y", 0.0)),
                        rt=float(obs.get("robot_theta", 0.0)),
                        bx=float(obs.get("ball_x", 0.0)),
                        by=float(obs.get("ball_y", 0.0)),
                    )
                )
            step += 1
            time.sleep(period)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if simrobot_process is not None and simrobot_process.poll() is None:
            simrobot_process.terminate()


if __name__ == "__main__":
    main()
