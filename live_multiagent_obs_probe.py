from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

LAUNCH_CWD = Path(os.environ.get("PWD", os.getcwd())).resolve()

from action import SkillType, StructuredSkillAction
from config import ENV, SIMROBOT_BIN
from environment import RealSimRobotEnv, TEAMMATE_STALE_THRESHOLD_MS
from observation import ObsIndex
from pybh_backend import PyBHBackend, WorldPlayerSpec, pybh

RL_OVERRIDE_TEAM_ENV = "PYBH_RL_OVERRIDE_TEAM"
RL_ACTIVE_PLAYERS_ENV = "PYBH_RL_ACTIVE_PLAYERS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end live probe for the 41-dim multi-agent observation wire."
    )
    parser.add_argument("--player-number", type=int, default=1)
    parser.add_argument("--simrobot-profile", choices=("2d", "3d"), default="2d")
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--launch-simrobot", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--launch-rl-override-team", type=int, default=None)
    parser.add_argument("--launch-rl-active-players", type=str, default=None)
    parser.add_argument("--simrobot-bin", type=Path, default=SIMROBOT_BIN)
    parser.add_argument("--drive-teammate-number", type=int, default=None)
    parser.add_argument("--drive-teammate-skill", choices=("stand", "walkTo", "shoot"), default="shoot")
    parser.add_argument("--self-skill", choices=("stand", "observe", "walk", "shoot"), default="stand")
    parser.add_argument("--ball-x", type=float, default=0.0)
    parser.add_argument("--ball-y", type=float, default=0.0)
    parser.add_argument("--robot-x", type=float, default=-1500.0)
    parser.add_argument("--robot-y", type=float, default=0.0)
    parser.add_argument("--robot-theta", type=float, default=0.0)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--request-timeout-ms", type=int, default=30000)
    parser.add_argument("--reset-obs-timeout-ms", type=int, default=30000)
    parser.add_argument("--status-log-interval-s", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sleep-s", type=float, default=0.05)
    parser.add_argument("--startup-wait-s", type=float, default=0.0)
    parser.add_argument("--post-reset-wait-s", type=float, default=0.0)
    parser.add_argument("--post-run-wait-s", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.scene is not None and not args.scene.is_absolute():
        args.scene = (LAUNCH_CWD / args.scene).resolve()
    if args.simrobot_bin is not None and not args.simrobot_bin.is_absolute():
        args.simrobot_bin = (LAUNCH_CWD / args.simrobot_bin).resolve()
    if args.output_json is not None and not args.output_json.is_absolute():
        args.output_json = (LAUNCH_CWD / args.output_json).resolve()
    return args


def _stand_action() -> StructuredSkillAction:
    return StructuredSkillAction(skill_type=SkillType.stand)


def _self_action(skill: str, *, ball_x: float, ball_y: float) -> StructuredSkillAction:
    if skill == "observe":
        return StructuredSkillAction(skill_type=SkillType.observe)
    if skill == "walk":
        return StructuredSkillAction(skill_type=SkillType.walk, target_x=ball_x, target_y=ball_y, target_theta=0.0)
    if skill == "shoot":
        return StructuredSkillAction(skill_type=SkillType.shoot)
    return _stand_action()


def _fmt_pair(values: tuple[float, float]) -> str:
    return f"({values[0]:+.3f}, {values[1]:+.3f})"


def _block_b(encoded) -> list[tuple[float, float]]:
    return [
        (
            float(encoded[ObsIndex.nearest_teammate_x_rel]),
            float(encoded[ObsIndex.nearest_teammate_y_rel]),
        ),
        (
            float(encoded[ObsIndex.second_teammate_x_rel]),
            float(encoded[ObsIndex.second_teammate_y_rel]),
        ),
        (
            float(encoded[ObsIndex.third_teammate_x_rel]),
            float(encoded[ObsIndex.third_teammate_y_rel]),
        ),
    ]


def _block_d(encoded) -> list[float]:
    return [
        float(encoded[ObsIndex.nearest_teammate_engaging_ball]),
        float(encoded[ObsIndex.second_teammate_engaging_ball]),
        float(encoded[ObsIndex.third_teammate_engaging_ball]),
    ]


def _has_nonzero_slot(slot: tuple[float, float], *, eps: float = 1e-4) -> bool:
    return abs(slot[0]) > eps or abs(slot[1]) > eps


def _fresh_teammate_count(ages_ms: list[float]) -> int:
    return sum(1 for age in ages_ms if age <= TEAMMATE_STALE_THRESHOLD_MS)


def _teammate_reset_specs(player_number: int, *, drive_teammate_number: int | None) -> tuple[WorldPlayerSpec, ...]:
    layout = {
        1: (
            WorldPlayerSpec(number=2, x=-250.0, y=150.0, theta=0.0),
            WorldPlayerSpec(number=3, x=-900.0, y=-900.0, theta=0.0),
        ),
        2: (
            WorldPlayerSpec(number=1, x=-1500.0, y=0.0, theta=0.0),
            WorldPlayerSpec(number=3, x=-900.0, y=-900.0, theta=0.0),
        ),
        3: (
            WorldPlayerSpec(number=1, x=-1500.0, y=0.0, theta=0.0),
            WorldPlayerSpec(number=2, x=-250.0, y=150.0, theta=0.0),
        ),
    }
    base = list(layout.get(player_number, ()))
    if drive_teammate_number is not None:
        for idx, teammate in enumerate(base):
            if int(teammate.number) == int(drive_teammate_number):
                base[idx] = WorldPlayerSpec(number=teammate.number, x=-220.0, y=0.0, theta=0.0)
                break
    return tuple(base)


def _drive_teammate(number: int, skill: str) -> None:
    if skill == "walkTo":
        pybh.rl_set_action(int(number), "walkTo", 0.0, 0.0, 0.0, -1)
    elif skill == "shoot":
        pybh.rl_set_action(int(number), "shoot", 0.0, 0.0, 0.0, -1)
    else:
        pybh.rl_set_action(int(number), "stand", 0.0, 0.0, 0.0, -1)


def _parse_active_players(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return ()
    players: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        players.append(int(token))
    return tuple(dict.fromkeys(players))


def _read_initial_obs(
    backend: PyBHBackend,
    env: RealSimRobotEnv,
    *,
    self_skill: str,
    ball_x: float,
    ball_y: float,
) -> tuple[object, object]:
    obs = backend.step(_self_action(self_skill, ball_x=ball_x, ball_y=ball_y))
    encoded = env._encode_obs(obs)
    return obs, encoded


def main() -> int:
    args = parse_args()
    if args.scene is not None and "RLvsBH3v3" in args.scene.name and not args.no_reset:
        raise ValueError("RLvsBH3v3_3D is a mixed match scene; run this probe with --no-reset to respect the documented scene flow")
    previous_env = {
        RL_OVERRIDE_TEAM_ENV: os.environ.get(RL_OVERRIDE_TEAM_ENV),
        RL_ACTIVE_PLAYERS_ENV: os.environ.get(RL_ACTIVE_PLAYERS_ENV),
    }
    if args.launch_rl_override_team is not None:
        os.environ[RL_OVERRIDE_TEAM_ENV] = str(int(args.launch_rl_override_team))
    if args.launch_rl_active_players:
        os.environ[RL_ACTIVE_PLAYERS_ENV] = str(args.launch_rl_active_players)
    passive_rl_players = tuple(
        player
        for player in _parse_active_players(args.launch_rl_active_players)
        if player != args.player_number and player != args.drive_teammate_number
    )

    backend = PyBHBackend(
        mode="simrobot_visible",
        player_number=args.player_number,
        simrobot_profile=args.simrobot_profile,
        simrobot_scene_path=args.scene,
        obs_timeout_ms=args.timeout_ms,
        visible_request_timeout_ms=args.request_timeout_ms,
        visible_reset_obs_timeout_ms=args.reset_obs_timeout_ms,
        visible_status_log_interval_s=args.status_log_interval_s,
        simrobot_launch=args.launch_simrobot,
        simrobot_bin=args.simrobot_bin,
    )
    env = RealSimRobotEnv(
        env_cfg=replace(
            ENV,
            backend=replace(
                ENV.backend,
                mode="simrobot_visible",
                player_number=args.player_number,
                simrobot_profile=args.simrobot_profile,
                simrobot_scene_path=args.scene,
                simrobot_launch=False,
                simrobot_bin=args.simrobot_bin,
            ),
        )
    )

    samples: list[dict[str, object]] = []
    best = None
    best_score = -1

    try:
        backend.start()
        if args.startup_wait_s > 0.0:
            print(
                f"[probe] waiting {args.startup_wait_s:.1f}s before attach/reset so the visible scene can finish loading",
                flush=True,
            )
            time.sleep(args.startup_wait_s)

        if args.no_reset:
            initial_obs, _initial_encoded = _read_initial_obs(
                backend,
                env,
                self_skill=args.self_skill,
                ball_x=args.ball_x,
                ball_y=args.ball_y,
            )
            print(
                f"[probe] attached without reset profile={args.simrobot_profile} player={args.player_number} "
                f"robot=({initial_obs.robot_x:.1f},{initial_obs.robot_y:.1f},{initial_obs.robot_theta:.2f}) "
                f"ball=({initial_obs.ball_x:.1f},{initial_obs.ball_y:.1f})",
                flush=True,
            )
            if args.post_reset_wait_s > 0.0:
                print(
                    f"[probe] holding {args.post_reset_wait_s:.1f}s after attach for visual inspection",
                    flush=True,
                )
                time.sleep(args.post_reset_wait_s)
        else:
            reset_obs = backend.reset(
                ball_pos=[args.ball_x, args.ball_y],
                robot_pos=[args.robot_x, args.robot_y],
                robot_theta=args.robot_theta,
                teammates=_teammate_reset_specs(
                    args.player_number,
                    drive_teammate_number=args.drive_teammate_number,
                ),
                opponents=(
                    WorldPlayerSpec(number=1, x=1600.0, y=700.0, theta=math.pi),
                    WorldPlayerSpec(number=2, x=1600.0, y=-700.0, theta=math.pi),
                ),
            )
            print(
                f"[probe] reset ok profile={args.simrobot_profile} player={args.player_number} "
                f"robot=({reset_obs.robot_x:.1f},{reset_obs.robot_y:.1f},{reset_obs.robot_theta:.2f}) "
                f"ball=({reset_obs.ball_x:.1f},{reset_obs.ball_y:.1f})",
                flush=True,
            )
            if args.post_reset_wait_s > 0.0:
                print(
                    f"[probe] holding {args.post_reset_wait_s:.1f}s after reset for visual inspection",
                    flush=True,
                )
                time.sleep(args.post_reset_wait_s)

        for step_idx in range(args.steps):
            for passive_player in passive_rl_players:
                _drive_teammate(passive_player, "stand")
            if args.drive_teammate_number is not None:
                _drive_teammate(args.drive_teammate_number, args.drive_teammate_skill)
            obs = backend.step(_self_action(args.self_skill, ball_x=args.ball_x, ball_y=args.ball_y))
            encoded = env._encode_obs(obs)
            block_b = _block_b(encoded)
            block_d = _block_d(encoded)
            team_ball_valid = bool(obs.global_team_ball_valid)
            team_ball_block_nonzero = any(
                abs(float(encoded[idx])) > 1e-4
                for idx in (
                    ObsIndex.global_team_ball_x_rel,
                    ObsIndex.global_team_ball_y_rel,
                    ObsIndex.time_since_team_ball_update,
                )
            )
            goal_block = (
                float(encoded[ObsIndex.opponent_goal_x_rel]),
                float(encoded[ObsIndex.opponent_goal_y_rel]),
            )
            consensus_active = float(encoded[ObsIndex.team_ball_agrees_with_mine]) > 0.5
            own_ball_field_x = float(obs.robot_x + obs.ball_rel_x)
            own_ball_field_y = float(obs.robot_y + obs.ball_rel_y)
            team_ball_distance_mm = math.hypot(
                own_ball_field_x - float(obs.global_team_ball_x_mm),
                own_ball_field_y - float(obs.global_team_ball_y_mm),
            )
            slot_nonzero_count = sum(1 for slot in block_b if _has_nonzero_slot(slot))
            activity_nonzero_count = sum(1 for value in obs.teammate_activities if int(value) != 0)
            score = (
                100 * slot_nonzero_count
                + 50 * int(team_ball_valid)
                + 10 * activity_nonzero_count
                + 5 * int(any(value > 0.5 for value in block_d))
                + 20 * int(consensus_active)
            )
            sample = {
                "step": step_idx + 1,
                "raw_teammate_count": len(obs.teammate_poses),
                "raw_teammate_ages_ms": [float(v) for v in obs.teammate_ages_ms],
                "fresh_teammate_count": _fresh_teammate_count([float(v) for v in obs.teammate_ages_ms]),
                "raw_teammate_activities": [int(v) for v in obs.teammate_activities],
                "raw_global_team_ball_valid": team_ball_valid,
                "raw_global_team_ball_age_ms": float(obs.global_team_ball_age_ms),
                "raw_ball_seen_percentage": float(obs.ball_seen_percentage),
                "raw_time_since_ball_seen_ms": float(obs.time_since_ball_seen),
                "raw_team_ball_distance_mm": float(team_ball_distance_mm),
                "raw_goal_opponent": [float(obs.goal_opponent_x_mm), float(obs.goal_opponent_y_mm)],
                "encoded_block_b": [[float(x), float(y)] for x, y in block_b],
                "encoded_block_c": [
                    float(encoded[ObsIndex.global_team_ball_x_rel]),
                    float(encoded[ObsIndex.global_team_ball_y_rel]),
                    float(encoded[ObsIndex.time_since_team_ball_update]),
                ],
                "encoded_block_d": block_d,
                "encoded_block_e": float(encoded[ObsIndex.team_ball_agrees_with_mine]),
                "encoded_block_f": [goal_block[0], goal_block[1]],
            }
            samples.append(sample)
            if score > best_score:
                best_score = score
                best = sample

            print(
                f"[probe step={step_idx+1:02d}] "
                f"teammates={sample['fresh_teammate_count']}/{len(obs.teammate_poses)} "
                f"ages_ms={sample['raw_teammate_ages_ms']} "
                f"activities={list(obs.teammate_activities)} "
                f"team_ball_valid={int(team_ball_valid)} "
                f"seen={sample['raw_ball_seen_percentage']:.1f}%/{sample['raw_time_since_ball_seen_ms']:.0f}ms "
                f"dist_tb={sample['raw_team_ball_distance_mm']:.0f} "
                f"B={', '.join(_fmt_pair(slot) for slot in block_b)} "
                f"D={block_d} "
                f"E={float(encoded[ObsIndex.team_ball_agrees_with_mine]):.0f} "
                f"F={_fmt_pair(goal_block)}",
                flush=True,
            )
            if args.sleep_s > 0.0:
                time.sleep(args.sleep_s)

        assert best is not None
        best_b = [tuple(slot) for slot in best["encoded_block_b"]]
        best_d = [float(v) for v in best["encoded_block_d"]]
        best_c = [float(v) for v in best["encoded_block_c"]]
        best_f = [float(v) for v in best["encoded_block_f"]]
        ever_two_raw_teammates = any(int(sample["raw_teammate_count"]) >= 2 for sample in samples)
        ever_two_nonzero_slots = any(
            sum(1 for slot in sample["encoded_block_b"] if _has_nonzero_slot(tuple(slot))) >= 2
            for sample in samples
        )
        ever_goal_nonzero = any(
            abs(float(sample["encoded_block_f"][0])) > 1e-4 or abs(float(sample["encoded_block_f"][1])) > 1e-4
            for sample in samples
        )
        ever_team_ball_valid = any(bool(sample["raw_global_team_ball_valid"]) for sample in samples)
        ever_team_ball_block_nonzero = any(
            any(abs(float(v)) > 1e-4 for v in sample["encoded_block_c"])
            for sample in samples
        )
        ever_engagement_active = any(any(float(v) > 0.5 for v in sample["encoded_block_d"]) for sample in samples)
        ever_consensus_active = any(float(sample["encoded_block_e"]) > 0.5 for sample in samples)

        failures: list[str] = []
        warnings: list[str] = []

        if not ever_two_raw_teammates:
            failures.append("raw teammate count never reached 2")
        if not ever_two_nonzero_slots:
            failures.append("encoded teammate slots 26..31 never showed two non-zero teammates")
        if not ever_goal_nonzero:
            failures.append("goal block 39..40 stayed zero")
        if len(best["raw_teammate_activities"]) != int(best["raw_teammate_count"]):
            failures.append("teammate_activity length mismatched teammate count")
        if not all(int(v) in {0, 1, 2, 3} for v in best["raw_teammate_activities"]):
            failures.append("teammate_activity exported values outside expected enum proxy range")
        if ever_team_ball_valid:
            if not ever_team_ball_block_nonzero:
                failures.append("team-ball block 32..34 stayed zero despite valid raw team ball")
        else:
            warnings.append("global_team_ball_valid never became true during the probe")
        if not ever_engagement_active:
            warnings.append("engagement block 35..37 never became active during the probe")
        if not ever_consensus_active:
            warnings.append("consensus block 38 never became active during the probe")

        result = {
            "profile": args.simrobot_profile,
            "steps": args.steps,
            "best_sample": best,
            "evidence": {
                "ever_two_raw_teammates": ever_two_raw_teammates,
                "ever_two_nonzero_slots": ever_two_nonzero_slots,
                "ever_team_ball_valid": ever_team_ball_valid,
                "ever_team_ball_block_nonzero": ever_team_ball_block_nonzero,
                "ever_engagement_active": ever_engagement_active,
                "ever_consensus_active": ever_consensus_active,
                "ever_goal_nonzero": ever_goal_nonzero,
            },
            "failures": failures,
            "warnings": warnings,
            "status": "PASS" if not failures else "FAIL",
        }

        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

        print("", flush=True)
        print(f"[probe result] status={result['status']}", flush=True)
        for item in failures:
            print(f"  FAIL {item}", flush=True)
        for item in warnings:
            print(f"  WARN {item}", flush=True)
        print(
            "  best raw: "
            f"teammates={best['fresh_teammate_count']}/{best['raw_teammate_count']} "
            f"ages_ms={best['raw_teammate_ages_ms']} "
            f"activities={best['raw_teammate_activities']} "
            f"team_ball_valid={int(bool(best['raw_global_team_ball_valid']))} "
            f"seen={best['raw_ball_seen_percentage']:.1f}%/{best['raw_time_since_ball_seen_ms']:.0f}ms "
            f"dist_tb={best['raw_team_ball_distance_mm']:.0f} "
            f"goal={best['raw_goal_opponent']}",
            flush=True,
        )
        print(
            "  best encoded: "
            f"B={best['encoded_block_b']} "
            f"C={best['encoded_block_c']} "
            f"D={best['encoded_block_d']} "
            f"E={best['encoded_block_e']} "
            f"F={best['encoded_block_f']}",
            flush=True,
        )
        if args.post_run_wait_s > 0.0:
            print(
                f"[probe] holding {args.post_run_wait_s:.1f}s before shutdown so the scene stays visible",
                flush=True,
            )
            time.sleep(args.post_run_wait_s)
        return 0 if not failures else 1
    finally:
        env.close()
        backend.close()
        for env_name, value in previous_env.items():
            if value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = value


if __name__ == "__main__":
    raise SystemExit(main())
