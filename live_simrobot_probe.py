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

os.chdir(BHUMAN_DIR)
import pybh.controller as pybh


SIMROBOT_BIN = BHUMAN_DIR / "Build/Linux/SimRobot/Develop/SimRobot"
LIVE_SHOOT_DISTANCE_MM = 260.0
LIVE_SHOOT_HOLD_STEPS = 20
LIVE_RESET_BALL_POS = np.array([0.0, 0.0], dtype=np.float32)
LIVE_RESET_ROBOT_POS = np.array([-2438.0, 1875.0], dtype=np.float32)
LIVE_RESET_ROBOT_THETA = -math.pi / 2.0
LIVE_RESET_TOLERANCE_MM = 150.0
LIVE_BALL_SEEN_TIMEOUT_MS = 1000.0


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
    parser.add_argument("--shoot-distance", type=float, default=LIVE_SHOOT_DISTANCE_MM)
    parser.add_argument("--shoot-hold-steps", type=int, default=LIVE_SHOOT_HOLD_STEPS)
    return parser.parse_args()


def maybe_launch_simrobot(simrobot_bin: Path, scene: Path) -> subprocess.Popen[str] | None:
    if not simrobot_bin.exists():
        raise FileNotFoundError(f"SimRobot binary not found: {simrobot_bin}")
    if not scene.exists():
        raise FileNotFoundError(f"Scene file not found: {scene}")
    return subprocess.Popen([str(simrobot_bin), str(scene)], cwd=BHUMAN_DIR, env=os.environ.copy())


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

    pybh.rl_set_action(
        player_number,
        "walkTo",
        float(action.target_x),
        float(action.target_y),
        float(action.target_theta),
    )
    return f"walkToField({action.target_x:.0f}, {action.target_y:.0f}, {action.target_theta:.2f})"


def ball_distance_from_obs(obs: dict) -> float:
    robot_pos = np.array([float(obs.get("robot_x", 0.0)), float(obs.get("robot_y", 0.0))], dtype=np.float32)
    ball_pos = np.array([float(obs.get("ball_x", 0.0)), float(obs.get("ball_y", 0.0))], dtype=np.float32)
    return float(np.linalg.norm(ball_pos - robot_pos))


def ball_observation_reliable(obs: dict) -> bool:
    seen_percentage = float(obs.get("ball_seen_percentage", 0.0))
    time_since_seen = float(obs.get("time_since_ball_seen", 1e9))
    return seen_percentage > 0.0 and time_since_seen <= LIVE_BALL_SEEN_TIMEOUT_MS


def send_search_action(player_number: int, obs: dict) -> str:
    robot_pos = np.array([float(obs.get("robot_x", 0.0)), float(obs.get("robot_y", 0.0))], dtype=np.float32)
    delta = LIVE_RESET_BALL_POS - robot_pos
    target_theta = float(math.atan2(float(delta[1]), float(delta[0]))) if np.linalg.norm(delta) > 1.0 else 0.0
    pybh.rl_set_action(
        player_number,
        "walkTo",
        float(LIVE_RESET_BALL_POS[0]),
        float(LIVE_RESET_BALL_POS[1]),
        target_theta,
    )
    return f"searchBallAt({LIVE_RESET_BALL_POS[0]:.0f}, {LIVE_RESET_BALL_POS[1]:.0f}, {target_theta:.2f})"


def send_heuristic_action(player_number: int, obs: dict, shoot_distance: float, force_shoot: bool = False) -> str:
    if not ball_observation_reliable(obs):
        return send_search_action(player_number, obs)

    robot_pos = np.array([float(obs.get("robot_x", 0.0)), float(obs.get("robot_y", 0.0))], dtype=np.float32)
    ball_pos = np.array([float(obs.get("ball_x", 0.0)), float(obs.get("ball_y", 0.0))], dtype=np.float32)
    delta = ball_pos - robot_pos
    distance = float(np.linalg.norm(delta))
    if force_shoot or distance <= shoot_distance:
        pybh.rl_set_action(player_number, "shoot", 0.0, 0.0, 0.0)
        return f"shoot(dist={distance:.0f})"

    global_action = StructuredSkillAction(
        skill_type=SkillType.walk,
        target_x=float(ball_pos[0]),
        target_y=float(ball_pos[1]),
        target_theta=float(math.atan2(float(delta[1]), float(delta[0]))),
    )
    pybh.rl_set_action(
        player_number,
        "walkTo",
        float(global_action.target_x),
        float(global_action.target_y),
        float(global_action.target_theta),
    )
    return (
        f"walkToField({global_action.target_x:.0f}, "
        f"{global_action.target_y:.0f}, {global_action.target_theta:.2f})"
    )


def wait_for_reset_observation(player_number: int, timeout_ms: int, max_wait_s: float = 30.0) -> dict | None:
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        obs = pybh.rl_get_obs(player_number, timeout_ms)
        if not obs.get("obs_ready", False):
            continue
        robot_pos = np.array([float(obs.get("robot_x", 0.0)), float(obs.get("robot_y", 0.0))], dtype=np.float32)
        if float(np.linalg.norm(robot_pos - LIVE_RESET_ROBOT_POS)) <= LIVE_RESET_TOLERANCE_MM:
            return obs
    return None


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

    pybh.rl_reset_world(
        args.player_number,
        float(LIVE_RESET_BALL_POS[0]),
        float(LIVE_RESET_BALL_POS[1]),
        float(LIVE_RESET_ROBOT_POS[0]),
        float(LIVE_RESET_ROBOT_POS[1]),
        float(LIVE_RESET_ROBOT_THETA),
    )
    pybh.rl_set_action(args.player_number, "stand", 0.0, 0.0, 0.0)
    initial_obs = wait_for_reset_observation(args.player_number, args.timeout_ms) if args.launch else None
    if initial_obs is not None:
        print(
            "reset frame={frame} robot=({rx:.1f},{ry:.1f},{rt:.2f}) ball=({bx:.1f},{by:.1f})".format(
                frame=int(initial_obs.get("frame", 0)),
                rx=float(initial_obs.get("robot_x", 0.0)),
                ry=float(initial_obs.get("robot_y", 0.0)),
                rt=float(initial_obs.get("robot_theta", 0.0)),
                bx=float(initial_obs.get("ball_x", 0.0)),
                by=float(initial_obs.get("ball_y", 0.0)),
            )
        )
    elif args.launch:
        print("reset obs not confirmed yet; continuing with phase policy until observations settle")

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
        last_obs: dict | None = initial_obs
        shoot_hold = 0
        while total_steps is None or step < total_steps:
            if args.policy == "heuristic" and last_obs is not None and last_obs.get("obs_ready", False):
                if ball_observation_reliable(last_obs) and ball_distance_from_obs(last_obs) <= args.shoot_distance:
                    shoot_hold = max(shoot_hold, max(1, args.shoot_hold_steps))
                elif not ball_observation_reliable(last_obs):
                    shoot_hold = 0
                force_shoot = shoot_hold > 0
                action_name = send_heuristic_action(args.player_number, last_obs, args.shoot_distance, force_shoot)
                if shoot_hold > 0:
                    shoot_hold -= 1
            else:
                action_name = send_phase_action(args.player_number, step // phase_steps, last_obs)
            obs = pybh.rl_get_obs(args.player_number, args.timeout_ms)
            if not obs.get("obs_ready", False):
                stale_frame = int(last_obs.get("frame", 0)) if last_obs is not None else 0
                print(f"step={step + 1} action={action_name} obs_ready=0 stale_frame={stale_frame}")
            else:
                last_obs = obs
                print(
                    "step={step} action={action} frame={frame} "
                    "robot=({rx:.1f},{ry:.1f},{rt:.2f}) ball=({bx:.1f},{by:.1f},seen={seen:.1f},since={since:.0f},reliable={reliable}) "
                    "provider=({pmotion},calls={pcalls},target=({ptx:.1f},{pty:.1f},{ptt:.2f})) "
                    "skillctl=(skill={sbskill},motion={sbmotion},calls={sbcalls},target=({sbx:.1f},{sby:.1f},{sbt:.2f})) "
                    "engine=(in={mein},eff={meeff},phase={mephase},sit={mesit},gyro={megyro},fall={mefall},imu=({meix:.2f},{meiy:.2f})) "
                    "motion=({motion},{phase}) speed=({sx:.1f},{sy:.1f},{sr:.2f}) "
                    "legs=({lhp:.2f},{lkp:.2f},{rhp:.2f},{rkp:.2f})".format(
                        step=step + 1,
                        action=action_name,
                        frame=int(obs.get("frame", 0)),
                        rx=float(obs.get("robot_x", 0.0)),
                        ry=float(obs.get("robot_y", 0.0)),
                        rt=float(obs.get("robot_theta", 0.0)),
                        bx=float(obs.get("ball_x", 0.0)),
                        by=float(obs.get("ball_y", 0.0)),
                        seen=float(obs.get("ball_seen_percentage", 0.0)),
                        since=float(obs.get("time_since_ball_seen", 0.0)),
                        reliable=int(ball_observation_reliable(obs)),
                        pmotion=int(obs.get("provider_motion_request", -1)),
                        pcalls=int(obs.get("provider_call_count", -1)),
                        ptx=float(obs.get("provider_target_x", 0.0)),
                        pty=float(obs.get("provider_target_y", 0.0)),
                        ptt=float(obs.get("provider_target_theta", 0.0)),
                        sbskill=int(obs.get("skill_behavior_skill_request", -1)),
                        sbmotion=int(obs.get("skill_behavior_motion_request", -1)),
                        sbcalls=int(obs.get("skill_behavior_call_count", -1)),
                        sbx=float(obs.get("skill_behavior_walk_target_x", 0.0)),
                        sby=float(obs.get("skill_behavior_walk_target_y", 0.0)),
                        sbt=float(obs.get("skill_behavior_walk_target_theta", 0.0)),
                        mein=int(obs.get("motion_engine_input_request", -1)),
                        meeff=int(obs.get("motion_engine_effective_request", -1)),
                        mephase=int(obs.get("motion_engine_phase", -1)),
                        mesit=int(bool(obs.get("motion_engine_force_sit_down", False))),
                        megyro=int(bool(obs.get("motion_engine_gyro_offset_finished", False))),
                        mefall=int(obs.get("motion_engine_fall_state", -1)),
                        meix=float(obs.get("motion_engine_inertial_angle_x", 0.0)),
                        meiy=float(obs.get("motion_engine_inertial_angle_y", 0.0)),
                        motion=int(obs.get("motion_request", -1)),
                        phase=int(obs.get("motion_phase", -1)),
                        sx=float(obs.get("motion_speed_x", 0.0)),
                        sy=float(obs.get("motion_speed_y", 0.0)),
                        sr=float(obs.get("motion_speed_rot", 0.0)),
                        lhp=float(obs.get("joint_l_hip_pitch", 0.0)),
                        lkp=float(obs.get("joint_l_knee_pitch", 0.0)),
                        rhp=float(obs.get("joint_r_hip_pitch", 0.0)),
                        rkp=float(obs.get("joint_r_knee_pitch", 0.0)),
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
