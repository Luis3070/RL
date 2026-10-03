from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rlfootball.core.config import BHUMAN_DIR, SIMROBOT_BIN


RL_OVERRIDE_TEAM = 24
DEFAULT_ACTIVE_PLAYERS = (2, 3)
DEFAULT_SCENE = BHUMAN_DIR / "Config/Scenes/RLvsBH3v3_3D.ros2"
DEFAULT_SHOOT_DISTANCE_MM = 320.0
DEFAULT_LOOP_HZ = 8.0
DEFAULT_TIMEOUT_MS = 1200
DEFAULT_MAX_WAIT_S = 30.0


sys.path.insert(0, str(BHUMAN_DIR / "Make/Python"))
os.chdir(BHUMAN_DIR)
import pybh.controller as pybh


@dataclass(frozen=True)
class Action:
    skill: str
    target_x: float = 0.0
    target_y: float = 0.0
    target_theta: float = 0.0
    pass_target: int = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a mixed 3v3 scene: team 24 with BH goalkeeper + two RL field players versus three B-Human opponents."
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--simrobot-bin", type=Path, default=SIMROBOT_BIN)
    parser.add_argument("--launch", action="store_true", help="Launch SimRobot with the mixed scene.")
    parser.add_argument("--seconds", type=float, default=60.0, help="Run duration. Use <= 0 for endless mode.")
    parser.add_argument("--hz", type=float, default=DEFAULT_LOOP_HZ)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--shoot-distance", type=float, default=DEFAULT_SHOOT_DISTANCE_MM)
    parser.add_argument(
        "--active-players",
        type=str,
        default=",".join(str(player) for player in DEFAULT_ACTIVE_PLAYERS),
        help="Comma-separated RL jersey numbers on team 24. Default: 2,3",
    )
    return parser.parse_args()


def parse_player_numbers(raw: str) -> tuple[int, ...]:
    players: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        player = int(token)
        if player < 1 or player > 20:
            raise ValueError(f"player numbers must be within 1..20, got {player}")
        players.append(player)
    if not players:
        raise ValueError("active player list cannot be empty")
    return tuple(dict.fromkeys(players))


def maybe_launch_simrobot(simrobot_bin: Path, scene: Path, active_players: tuple[int, ...]) -> subprocess.Popen[str] | None:
    if not simrobot_bin.exists():
        raise FileNotFoundError(f"SimRobot binary not found: {simrobot_bin}")
    if not scene.exists():
        raise FileNotFoundError(f"Scene file not found: {scene}")

    env = os.environ.copy()
    env["PYBH_RL_OVERRIDE_TEAM"] = str(RL_OVERRIDE_TEAM)
    env["PYBH_RL_ACTIVE_PLAYERS"] = ",".join(str(player) for player in active_players)
    return subprocess.Popen(
        [str(simrobot_bin), str(scene)],
        cwd=BHUMAN_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def fetch_obs(player_number: int, timeout_ms: int) -> dict:
    return dict(pybh.rl_get_obs(player_number, timeout_ms))


def wait_for_initial_observations(player_numbers: tuple[int, ...], timeout_ms: int, max_wait_s: float) -> dict[int, dict]:
    deadline = time.monotonic() + max_wait_s
    ready: dict[int, dict] = {}
    while time.monotonic() < deadline:
        ready = {}
        for player in player_numbers:
            obs = fetch_obs(player, timeout_ms)
            if obs.get("obs_ready", False):
                ready[player] = obs
        if len(ready) == len(player_numbers):
            return ready
    missing = [str(player) for player in player_numbers if player not in ready]
    raise TimeoutError(f"Timed out waiting for initial observations for players: {', '.join(missing)}")


def ball_reliable(obs: dict) -> bool:
    return float(obs.get("ball_seen_percentage", 0.0)) > 0.0 and float(obs.get("time_since_ball_seen", 1e9)) <= 1000.0


def ball_distance(obs: dict) -> float:
    return math.hypot(
        float(obs.get("ball_x", 0.0)) - float(obs.get("robot_x", 0.0)),
        float(obs.get("ball_y", 0.0)) - float(obs.get("robot_y", 0.0)),
    )


def face_target(robot_x: float, robot_y: float, target_x: float, target_y: float) -> float:
    return math.atan2(target_y - robot_y, target_x - robot_x)


def choose_attacker(obs_by_player: dict[int, dict]) -> int:
    reliable_players = [player for player, obs in obs_by_player.items() if ball_reliable(obs)]
    if not reliable_players:
        return min(obs_by_player)
    return min(reliable_players, key=lambda player: ball_distance(obs_by_player[player]))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def plan_actions(obs_by_player: dict[int, dict], shoot_distance: float) -> dict[int, Action]:
    attacker = choose_attacker(obs_by_player)
    players = tuple(sorted(obs_by_player))
    support_player = next((player for player in players if player != attacker), attacker)

    actions: dict[int, Action] = {}
    for player, obs in obs_by_player.items():
        robot_x = float(obs.get("robot_x", 0.0))
        robot_y = float(obs.get("robot_y", 0.0))
        if not ball_reliable(obs):
            search_x = 0.0
            search_y = -450.0 if player == attacker else 900.0
            actions[player] = Action(
                skill="walkTo",
                target_x=search_x,
                target_y=search_y,
                target_theta=face_target(robot_x, robot_y, 0.0, 0.0),
            )
            continue

        ball_x = float(obs.get("ball_x", 0.0))
        ball_y = float(obs.get("ball_y", 0.0))
        target_theta = face_target(robot_x, robot_y, ball_x, ball_y)
        distance = ball_distance(obs)
        if player == attacker:
            if distance <= shoot_distance:
                actions[player] = Action(skill="shoot")
            else:
                actions[player] = Action(skill="walkTo", target_x=ball_x, target_y=ball_y, target_theta=target_theta)
        else:
            y_offset = 800.0 if support_player == player else -800.0
            support_x = clamp(ball_x - 650.0, -2500.0, 1500.0)
            support_y = clamp(ball_y + y_offset, -2200.0, 2200.0)
            actions[player] = Action(
                skill="walkTo",
                target_x=support_x,
                target_y=support_y,
                target_theta=face_target(robot_x, robot_y, ball_x, ball_y),
            )
    return actions


def send_action(player_number: int, action: Action) -> None:
    pybh.rl_set_action(
        player_number,
        action.skill,
        float(action.target_x),
        float(action.target_y),
        float(action.target_theta),
        int(action.pass_target),
    )


def format_action(action: Action) -> str:
    if action.skill == "walkTo":
        return f"walkTo({action.target_x:.0f},{action.target_y:.0f},{action.target_theta:.2f})"
    return action.skill


def print_zweikampf_events(step: int, obs_by_player: dict[int, dict], last_active: dict[int, bool]) -> None:
    for player, obs in sorted(obs_by_player.items()):
        active = bool(obs.get("zweikampf_active", False))
        if active and not last_active.get(player, False):
            print(
                "[zweikampf]",
                f"step={step}",
                f"player={player}",
                f"calls={int(obs.get('zweikampf_call_count', 0))}",
                f"ball_rel=({float(obs.get('ball_rel_x', 0.0)):.0f},{float(obs.get('ball_rel_y', 0.0)):.0f})",
                f"nearest_opp={float(obs.get('nearest_opponent_dist', 0.0)):.0f}",
                flush=True,
            )
        last_active[player] = active


def main() -> None:
    args = parse_args()
    active_players = parse_player_numbers(args.active_players)
    simrobot_process = maybe_launch_simrobot(args.simrobot_bin, args.scene, active_players) if args.launch else None

    print("Mixed RL vs B-Human 3v3", flush=True)
    print(f"scene={args.scene}", flush=True)
    print(f"rl_override_team={RL_OVERRIDE_TEAM}", flush=True)
    print(f"rl_active_players={','.join(str(player) for player in active_players)}", flush=True)
    if args.launch:
        print(f"launched={args.simrobot_bin}", flush=True)
    else:
        print(
            "attach_mode=1 (the existing SimRobot process must already have "
            "PYBH_RL_OVERRIDE_TEAM and PYBH_RL_ACTIVE_PLAYERS set)",
            flush=True,
        )

    for player in active_players:
        send_action(player, Action(skill="stand"))

    obs_by_player = wait_for_initial_observations(active_players, args.timeout_ms, DEFAULT_MAX_WAIT_S)
    print("observations_ready=1", flush=True)

    last_active = {player: bool(obs.get("zweikampf_active", False)) for player, obs in obs_by_player.items()}
    deadline = None if args.seconds <= 0 else time.monotonic() + args.seconds
    period_s = 1.0 / max(0.1, float(args.hz))
    step = 0
    try:
        while deadline is None or time.monotonic() < deadline:
            step += 1
            actions = plan_actions(obs_by_player, args.shoot_distance)
            for player, action in actions.items():
                send_action(player, action)

            time.sleep(period_s)
            obs_by_player = {player: fetch_obs(player, args.timeout_ms) for player in active_players}
            print_zweikampf_events(step, obs_by_player, last_active)

            line = [f"step={step}"]
            for player in sorted(active_players):
                obs = obs_by_player[player]
                action = actions[player]
                line.append(
                    " ".join(
                        [
                            f"p{player}",
                            f"act={format_action(action)}",
                            f"dist={ball_distance(obs):.0f}",
                            f"zk={int(bool(obs.get('zweikampf_active', False)))}/{int(obs.get('zweikampf_call_count', 0))}",
                            f"robot=({float(obs.get('robot_x', 0.0)):.0f},{float(obs.get('robot_y', 0.0)):.0f})",
                            f"ball=({float(obs.get('ball_x', 0.0)):.0f},{float(obs.get('ball_y', 0.0)):.0f})",
                            f"opp={float(obs.get('nearest_opponent_dist', 0.0)):.0f}",
                        ]
                    )
                )
            print(" | ".join(line), flush=True)
    finally:
        for player in active_players:
            send_action(player, Action(skill="stand"))
        if simrobot_process is not None:
            simrobot_process.terminate()
            try:
                simrobot_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                simrobot_process.kill()


if __name__ == "__main__":
    main()
