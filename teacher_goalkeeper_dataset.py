from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from goalkeeper_action import GK_SKILL_ORDER, GKSkillType, GKStructuredAction
from goalkeeper_teacher import (
    GK_OBS_SIZE,
    GOALIE_LINE_X_MM,
    goalkeeper_skill_mask_np,
    goalkeeper_teacher_action,
    encode_goalkeeper_obs,
    gk_action_to_current_bridge,
)
from pybh_backend import BackendObservation, PyBHBackend, WorldPlayerSpec
from config import ROOT_DIR, SIMROBOT_BIN


OUTPUT_ROOT = ROOT_DIR / "runs" / "teacher_goalkeeper_dataset"


def _obs(
    *,
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    ball_seen_ms: float = 0.0,
    pass_options: float = 0.0,
    team_ball_valid: bool = True,
    ground_contact: bool = True,
    fall_state: int = 0,
) -> BackendObservation:
    c = math.cos(-robot_theta)
    s = math.sin(-robot_theta)
    dx = ball_x - robot_x
    dy = ball_y - robot_y
    rel_x = dx * c - dy * s
    rel_y = dx * s + dy * c
    return BackendObservation(
        ball_x=float(ball_x),
        ball_y=float(ball_y),
        robot_x=float(robot_x),
        robot_y=float(robot_y),
        robot_theta=float(robot_theta),
        simrobot_robot_x=float(robot_x),
        simrobot_robot_y=float(robot_y),
        simrobot_robot_theta=float(robot_theta),
        frame=1,
        obs_ready=True,
        estimated_ball_x=float(ball_x),
        estimated_ball_y=float(ball_y),
        ball_rel_x=float(rel_x),
        ball_rel_y=float(rel_y),
        ball_end_rel_x=float(rel_x + 0.75 * ball_vx),
        ball_end_rel_y=float(rel_y + 0.75 * ball_vy),
        ball_vel_x=float(ball_vx),
        ball_vel_y=float(ball_vy),
        time_since_ball_seen=float(ball_seen_ms),
        natural_time_since_ball_seen=float(ball_seen_ms),
        ball_seen_percentage=100.0 if ball_seen_ms < 500.0 else 0.0,
        ball_consistent_with_game_state=True,
        pass_options_count=float(pass_options),
        global_team_ball_x_mm=float(ball_x),
        global_team_ball_y_mm=float(ball_y),
        global_team_ball_age_ms=float(ball_seen_ms),
        global_team_ball_valid=bool(team_ball_valid),
        goal_opponent_x_mm=4500.0,
        goal_opponent_y_mm=0.0,
        motion_engine_ground_contact=bool(ground_contact),
        motion_engine_fall_state=int(fall_state),
    )


def _synthetic_rows(
    episodes_per_scenario: int,
    steps_per_episode: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int], list[int], list[float], list[int], list[int], tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    stages: list[int] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    weights: list[float] = []
    label_consistent: list[int] = []
    scenario_ids: list[int] = []
    scenarios = (
        "central",
        "lateral_left",
        "lateral_right",
        "low_left",
        "low_right",
        "jump_left",
        "jump_right",
        "left",
        "right",
        "near_left_post",
        "near_right_post",
        "moving_away",
        "outside_goal_corridor",
        "unstable_body",
        "controlled_pass",
        "controlled_clear",
        "controlled_dribble",
        "stale_search",
        "stale_no_team_ball",
        "misposition_recover",
        "fast_blast_reaction",
        "close_range_reaction",
        "one_v_one_breakaway",
        "rebound_second_shot",
        "deflection_midflight",
        "screened_late_shot",
        "chip_lob_proxy",
        "dribble_in_left",
        "dribble_in_right",
    )
    ep = 0
    scenario_to_id = {name: idx for idx, name in enumerate(scenarios)}
    for scenario in scenarios:
        for _ in range(episodes_per_scenario):
            robot_x = GOALIE_LINE_X_MM
            robot_y = float(rng.normal(0.0, 120.0))
            if scenario == "near_left_post":
                robot_y = 650.0
            elif scenario == "near_right_post":
                robot_y = -650.0
            dribble_goal_y = 0.0
            dribble_round = False
            dribble_cut_x = -3500.0
            dribble_speed_lo, dribble_speed_hi = 380.0, 650.0
            if scenario in {"dribble_in_left", "dribble_in_right"}:
                # Cover the full dribble threat distribution, not just slow-central:
                # fast carries, off-center (wide) lanes and round-the-keeper late cuts.
                side = 1.0 if scenario == "dribble_in_left" else -1.0
                dribble_mode = str(rng.choice(["slow", "fast", "offcenter", "round"]))
                ball_x = -2700.0 + float(rng.normal(0.0, 90.0))
                ball_y = float(rng.uniform(-250.0, 250.0))
                dribble_goal_y = side * float(rng.uniform(340.0, 760.0))
                if dribble_mode == "fast":
                    dribble_speed_lo, dribble_speed_hi = 1000.0, 1500.0
                elif dribble_mode == "offcenter":
                    dribble_speed_lo, dribble_speed_hi = 430.0, 700.0
                    ball_y = side * float(rng.uniform(550.0, 950.0))
                elif dribble_mode == "round":
                    dribble_speed_lo, dribble_speed_hi = 620.0, 980.0
                    dribble_round = True
                    ball_x = -2500.0 + float(rng.normal(0.0, 90.0))
            for step in range(1, steps_per_episode + 1):
                if scenario in {"dribble_in_left", "dribble_in_right"}:
                    speed = float(rng.uniform(dribble_speed_lo, dribble_speed_hi))
                    aim_y = dribble_goal_y
                    if dribble_round and ball_x > dribble_cut_x:
                        aim_y = 0.0  # carry straight at the keeper, then cut late
                    dx_goal = -4500.0 - ball_x
                    dy_goal = aim_y - ball_y
                    norm = max(1e-6, math.hypot(dx_goal, dy_goal))
                    ball_vx = speed * dx_goal / norm
                    ball_vy = speed * dy_goal / norm
                    pass_options = 0.0
                    seen = 0.0
                    team_ball_valid = True
                    fall_state = 0
                    ground_contact = True
                elif scenario in {"controlled_pass", "controlled_clear", "controlled_dribble"}:
                    ball_x = GOALIE_LINE_X_MM + 220.0 + float(rng.normal(0.0, 30.0))
                    if scenario == "controlled_dribble":
                        robot_x = -3650.0
                        robot_y = 650.0 if robot_y >= 0.0 else -650.0
                        ball_x = robot_x + 180.0 + float(rng.normal(0.0, 30.0))
                        ball_y = (900.0 if robot_y >= 0.0 else -900.0) + float(rng.normal(0.0, 80.0))
                    else:
                        ball_y = robot_y + float(rng.normal(0.0, 80.0))
                    ball_vx, ball_vy = -30.0, 0.0
                    pass_options = 1.0 if scenario == "controlled_pass" else 0.0
                    seen = 0.0
                    team_ball_valid = True
                    fall_state = 0
                    ground_contact = True
                elif scenario in {"stale_search", "stale_no_team_ball"}:
                    ball_x = -2500.0 + float(rng.normal(0.0, 200.0))
                    ball_y = float(rng.normal(0.0, 700.0))
                    ball_vx, ball_vy = 0.0, 0.0
                    pass_options = 0.0
                    seen = 1600.0
                    team_ball_valid = scenario == "stale_search"
                    fall_state = 0
                    ground_contact = True
                elif scenario == "moving_away":
                    ball_x = -3600.0 + float(rng.normal(0.0, 120.0))
                    ball_y = float(rng.normal(0.0, 300.0))
                    ball_vx = 450.0 + float(rng.uniform(0.0, 450.0))
                    ball_vy = float(rng.normal(0.0, 80.0))
                    pass_options = 0.0
                    seen = 0.0
                    team_ball_valid = True
                    fall_state = 0
                    ground_contact = True
                elif scenario == "outside_goal_corridor":
                    ball_x = -3600.0 + float(rng.normal(0.0, 120.0))
                    ball_y = float(rng.choice([-1.0, 1.0]) * rng.uniform(1100.0, 1700.0))
                    ball_vx = -850.0 - float(rng.uniform(0.0, 500.0))
                    ball_vy = float(rng.normal(0.0, 80.0))
                    pass_options = 0.0
                    seen = 0.0
                    team_ball_valid = True
                    fall_state = 0
                    ground_contact = True
                elif scenario == "unstable_body":
                    target_y = float(rng.choice([-1.0, 1.0]) * rng.uniform(180.0, 550.0))
                    ball_x = -3300.0 + float(rng.normal(0.0, 80.0))
                    ball_y = target_y + float(rng.normal(0.0, 30.0))
                    ball_vx = -850.0 - float(rng.uniform(0.0, 500.0))
                    ball_vy = float(rng.normal(0.0, 60.0))
                    pass_options = 0.0
                    seen = 0.0
                    team_ball_valid = True
                    fall_state = 3
                    ground_contact = False
                elif scenario == "misposition_recover":
                    robot_y = float(rng.choice([-1.0, 1.0]) * rng.uniform(470.0, 760.0))
                    ball_x = -2300.0 + float(rng.normal(0.0, 160.0))
                    ball_y = -robot_y + float(rng.normal(0.0, 160.0))
                    ball_vx, ball_vy = 0.0, 0.0
                    pass_options = 0.0
                    seen = 0.0
                    team_ball_valid = True
                    fall_state = 0
                    ground_contact = True
                elif scenario in {
                    "fast_blast_reaction",
                    "close_range_reaction",
                    "one_v_one_breakaway",
                    "rebound_second_shot",
                    "deflection_midflight",
                    "screened_late_shot",
                    "chip_lob_proxy",
                }:
                    target_y = {
                        "fast_blast_reaction": float(rng.uniform(-520.0, 520.0)),
                        "close_range_reaction": float(rng.uniform(-360.0, 360.0)),
                        "one_v_one_breakaway": float(rng.uniform(-260.0, 260.0)),
                        "rebound_second_shot": float(rng.choice([-1.0, 1.0]) * rng.uniform(120.0, 520.0)),
                        "deflection_midflight": float(rng.choice([-1.0, 1.0]) * rng.uniform(220.0, 620.0)),
                        "screened_late_shot": float(rng.uniform(-420.0, 420.0)),
                        "chip_lob_proxy": float(rng.choice([-1.0, 1.0]) * rng.uniform(420.0, 690.0)),
                    }[scenario]
                    if scenario == "fast_blast_reaction":
                        t_cross = float(rng.uniform(0.03, 0.42))
                        speed = float(rng.uniform(1550.0, 2400.0))
                    elif scenario == "close_range_reaction":
                        t_cross = float(rng.uniform(0.04, 0.35))
                        speed = float(rng.uniform(950.0, 1700.0))
                    elif scenario == "one_v_one_breakaway":
                        t_cross = float(rng.uniform(0.25, 0.85))
                        speed = float(rng.uniform(700.0, 1250.0))
                    elif scenario == "screened_late_shot":
                        t_cross = float(rng.uniform(0.02, 0.28))
                        speed = float(rng.uniform(1100.0, 2100.0))
                    else:
                        t_cross = float(rng.uniform(0.10, 0.70))
                        speed = float(rng.uniform(800.0, 1700.0))
                    ball_vx = -speed
                    ball_vy = float(rng.normal(0.0, 120.0))
                    ball_x = robot_x + speed * t_cross
                    ball_y = robot_y + target_y - ball_vy * t_cross + float(rng.normal(0.0, 12.0))
                    pass_options = 0.0
                    seen = 0.0 if scenario != "screened_late_shot" else float(rng.uniform(0.0, 90.0))
                    team_ball_valid = True
                    fall_state = 0
                    ground_contact = True
                else:
                    ball_x = -2250.0 + 8.0 * step
                    target_y = {
                        "central": 0.0,
                        "lateral_left": 95.0,
                        "lateral_right": -95.0,
                        "low_left": 165.0,
                        "low_right": -165.0,
                        "jump_left": 500.0,
                        "jump_right": -500.0,
                        "left": 260.0,
                        "right": -260.0,
                        "near_left_post": 580.0,
                        "near_right_post": -580.0,
                    }[scenario]
                    ball_vx = -1050.0 - float(rng.uniform(0.0, 350.0))
                    ball_vy = float(rng.normal(0.0, 35.0))
                    rel_x = ball_x - robot_x
                    t_cross = max(0.15, -rel_x / ball_vx)
                    # Place the ball so B-Human-style y-axis prediction crosses
                    # the keeper line inside the named skill band.
                    ball_y = robot_y + target_y - ball_vy * t_cross + float(rng.normal(0.0, 10.0))
                    pass_options = 0.0
                    seen = 0.0
                    team_ball_valid = True
                    fall_state = 0
                    ground_contact = True
                obs = _obs(
                    robot_x=robot_x,
                    robot_y=robot_y,
                    robot_theta=0.0,
                    ball_x=ball_x,
                    ball_y=ball_y,
                    ball_vx=ball_vx,
                    ball_vy=ball_vy,
                    ball_seen_ms=seen,
                    pass_options=pass_options,
                    team_ball_valid=team_ball_valid,
                    ground_contact=ground_contact,
                    fall_state=fall_state,
                )
                action, cog = goalkeeper_teacher_action(obs)
                if scenario in {"dribble_in_left", "dribble_in_right"}:
                    rel_dist = math.hypot(ball_x - robot_x, ball_y - robot_y)
                    if rel_dist <= 360.0 and goalkeeper_skill_mask_np(encode_goalkeeper_obs(obs, cog))[int(GKSkillType.clear)] > 0.0:
                        action = GKStructuredAction(GKSkillType.clear)
                    else:
                        # Track a smother point on the dribble path. The point is
                        # ahead of the goal line so the keeper walks out and across,
                        # then uses clear once the carried ball is actually in reach.
                        engage_x = float(np.clip(ball_x - 420.0, -4200.0, -3300.0))
                        if ball_vx < -1e-6:
                            t_engage = max(0.0, (engage_x - ball_x) / ball_vx)
                            engage_y = ball_y + ball_vy * t_engage
                        else:
                            engage_y = ball_y
                        engage_y = float(np.clip(engage_y, -720.0, 720.0))
                        action = GKStructuredAction(
                            GKSkillType.walk_goalie_pose,
                            target_x=engage_x,
                            target_y=engage_y,
                            target_theta=math.atan2(ball_y - robot_y, ball_x - robot_x),
                        )
                encoded = encode_goalkeeper_obs(obs, cog)
                allowed = goalkeeper_skill_mask_np(encoded)[int(action.skill_type)] > 0.0
                observations.append(encoded)
                actions.append(action.as_numpy())
                stages.append(90)
                episode_ids.append(ep)
                step_ids.append(step)
                sample_weight = 10.0 if scenario in {"dribble_in_left", "dribble_in_right"} else 1.0
                weights.append(sample_weight if allowed else 0.0)
                label_consistent.append(1 if allowed else 0)
                scenario_ids.append(scenario_to_id[scenario])
                if scenario in {"dribble_in_left", "dribble_in_right"}:
                    if action.skill is GKSkillType.walk_goalie_pose:
                        step_dx = float(np.clip(action.target_x - robot_x, -220.0 * 0.10, 220.0 * 0.10))
                        step_dy = float(np.clip(action.target_y - robot_y, -260.0 * 0.10, 260.0 * 0.10))
                        robot_x += step_dx
                        robot_y += step_dy
                    ball_x += ball_vx * 0.10
                    ball_y += ball_vy * 0.10
            ep += 1
    return observations, actions, stages, episode_ids, step_ids, weights, label_consistent, scenario_ids, scenarios


def _simrobot_rows(args: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int], list[int], list[float], list[int], list[int], tuple[str, ...]]:
    backend = PyBHBackend(
        mode=args.backend,
        player_number=args.player_number,
        frames_per_step=args.frames_per_step,
        obs_timeout_ms=1200,
        simrobot_profile=args.simrobot_profile,
        simrobot_launch=args.launch_simrobot,
        simrobot_bin=SIMROBOT_BIN,
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    stages: list[int] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    weights: list[float] = []
    label_consistent: list[int] = []
    scenario_ids: list[int] = []
    scenarios = ("simrobot_teacher_rollout",)
    rng = np.random.default_rng(args.seed)
    backend.start()
    try:
        for ep in range(args.episodes):
            side = 1.0 if ep % 2 == 0 else -1.0
            ball = np.asarray([-1900.0 + rng.normal(0.0, 150.0), side * rng.uniform(0.0, 650.0)], dtype=np.float32)
            robot = np.asarray([GOALIE_LINE_X_MM, side * rng.uniform(-120.0, 120.0)], dtype=np.float32)
            obs = backend.reset(
                ball_pos=ball,
                robot_pos=robot,
                robot_theta=0.0,
                teammates=(WorldPlayerSpec(2, -2400.0, -900.0 * side, 0.0),),
                opponents=(WorldPlayerSpec(21, -900.0, 250.0 * side, math.pi),),
            )
            for step in range(1, args.steps + 1):
                action, cog = goalkeeper_teacher_action(obs)
                encoded = encode_goalkeeper_obs(obs, cog)
                allowed = goalkeeper_skill_mask_np(encoded)[int(action.skill_type)] > 0.0
                observations.append(encoded)
                actions.append(action.as_numpy())
                stages.append(90)
                episode_ids.append(ep)
                step_ids.append(step)
                weights.append(1.0 if allowed else 0.0)
                label_consistent.append(1 if allowed else 0)
                scenario_ids.append(0)
                obs = backend.step(gk_action_to_current_bridge(action, obs))
    finally:
        backend.close()
    return observations, actions, stages, episode_ids, step_ids, weights, label_consistent, scenario_ids, scenarios


def write_dataset(args: argparse.Namespace) -> Path:
    if args.synthetic:
        rows = _synthetic_rows(args.episodes, args.steps, args.seed)
        source = "synthetic"
    else:
        rows = _simrobot_rows(args)
        source = "simrobot"
    observations, actions, stages, episode_ids, step_ids, weights, label_consistent, scenario_ids, scenario_names = rows
    if not observations:
        raise RuntimeError("goalkeeper dataset has zero rows")
    out = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"dataset_gk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "samples.npz",
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        stages=np.asarray(stages, dtype=np.int16),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        step_ids=np.asarray(step_ids, dtype=np.int16),
        weights=np.asarray(weights, dtype=np.float32),
        label_consistent=np.asarray(label_consistent, dtype=np.int8),
        scenario_ids=np.asarray(scenario_ids, dtype=np.int16),
        scenario_names=np.asarray(scenario_names),
        skill_order=np.asarray(GK_SKILL_ORDER),
    )
    skill_ids = np.clip(np.rint(np.asarray(actions)[:, 0]).astype(np.int64), 0, len(GK_SKILL_ORDER) - 1)
    positive = np.asarray(weights) > 0.0
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "obs_size": GK_OBS_SIZE,
        "total_rows": len(observations),
        "positive_samples": int(np.count_nonzero(positive)),
        "skill_order": list(GK_SKILL_ORDER),
        "positive_skill_counts": {
            GK_SKILL_ORDER[i]: int(np.count_nonzero(positive & (skill_ids == i)))
            for i in range(len(GK_SKILL_ORDER))
            if np.count_nonzero(positive & (skill_ids == i))
        },
        "scenario_positive_skill_counts": {
            str(scenario_names[scenario_id]): {
                GK_SKILL_ORDER[i]: int(np.count_nonzero(positive & (np.asarray(scenario_ids) == scenario_id) & (skill_ids == i)))
                for i in range(len(GK_SKILL_ORDER))
                if np.count_nonzero(positive & (np.asarray(scenario_ids) == scenario_id) & (skill_ids == i))
            }
            for scenario_id in sorted(set(int(x) for x in scenario_ids))
        },
        "bridge_execution_note": (
            "factorized intercept labels are produced for the native GK C++ bridge: "
            "gk_center/gk_lat/gk_low_l/gk_low_r/gk_jump_l/gk_jump_r"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return out / "samples.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", default=os.getenv("RL_GK_TEACHER_SYNTHETIC", "1") == "1")
    parser.add_argument("--episodes", type=int, default=int(os.getenv("RL_GK_TEACHER_EPISODES", "24")))
    parser.add_argument("--steps", type=int, default=int(os.getenv("RL_GK_TEACHER_STEPS", "80")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("RL_GK_TEACHER_SEED", "42042")))
    parser.add_argument("--output-dir", default=os.getenv("RL_GK_TEACHER_OUTPUT_DIR", ""))
    parser.add_argument("--backend", choices=("simrobot", "simrobot_visible"), default=os.getenv("RL_GK_BACKEND", "simrobot_visible"))
    parser.add_argument("--simrobot-profile", default=os.getenv("RL_GK_SIMROBOT_PROFILE", "3d_visible"))
    parser.add_argument("--launch-simrobot", action="store_true", default=os.getenv("RL_GK_LAUNCH_SIMROBOT", "0") == "1")
    parser.add_argument("--frames-per-step", type=int, default=int(os.getenv("RL_GK_FRAMES_PER_STEP", "3")))
    parser.add_argument("--player-number", type=int, default=int(os.getenv("RL_GK_PLAYER_NUMBER", "1")))
    args = parser.parse_args()
    write_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
