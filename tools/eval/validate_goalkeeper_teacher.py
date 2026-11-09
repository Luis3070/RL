from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rlfootball.core.goalkeeper_action import GK_SKILL_ORDER, GKSkillType
from rlfootball.teachers.goalkeeper_teacher import (
    GOALIE_LINE_X_MM,
    BALL_CATCH_MAX_WALK_DISTANCE_MM,
    build_gk_cognition,
    encode_goalkeeper_obs,
    gk_action_to_current_bridge,
    gk_action_to_native_bridge_name,
    goalkeeper_skill_mask_np,
    goalkeeper_teacher_action,
)
from rlfootball.core.pybh_backend import BackendObservation, PyBHBackend
from rlfootball.core.config import SIMROBOT_BIN


def _obs(
    *,
    robot_x: float = GOALIE_LINE_X_MM,
    robot_y: float = 0.0,
    robot_theta: float = 0.0,
    ball_x: float = -1800.0,
    ball_y: float = 0.0,
    ball_vx: float = -1000.0,
    ball_vy: float = 0.0,
    seen_ms: float = 0.0,
    team_ball_valid: bool = True,
    pass_options: float = 0.0,
    opponent_front: float = 5000.0,
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
        ball_end_rel_x=float(rel_x + ball_vx),
        ball_end_rel_y=float(rel_y + ball_vy),
        ball_vel_x=float(ball_vx),
        ball_vel_y=float(ball_vy),
        time_since_ball_seen=float(seen_ms),
        time_since_ball_disappeared=float(seen_ms),
        natural_time_since_ball_seen=float(seen_ms),
        ball_seen_percentage=100.0 if seen_ms <= 500.0 else 0.0,
        ball_consistent_with_game_state=True,
        pass_options_count=float(pass_options),
        nearest_opponent_dist=float(opponent_front),
        nearest_opponent_front_dist=float(opponent_front),
        global_team_ball_x_mm=float(ball_x),
        global_team_ball_y_mm=float(ball_y),
        global_team_ball_age_ms=float(seen_ms),
        global_team_ball_valid=bool(team_ball_valid),
        goal_opponent_x_mm=4500.0,
        goal_opponent_y_mm=0.0,
        motion_engine_ground_contact=True,
        motion_engine_fall_state=0,
    )


def _case_obs() -> list[tuple[str, BackendObservation, set[str]]]:
    return [
        (
            "central_fast_shot",
            _obs(ball_x=-3800.0, ball_y=0.0, ball_vx=-1000.0, ball_vy=0.0),
            {"intercept_center", "intercept_lateral"},
        ),
        (
            "left_low_shot",
            _obs(ball_x=-3800.0, ball_y=180.0, ball_vx=-1000.0, ball_vy=0.0),
            {"intercept_low_left", "intercept_jump_left"},
        ),
        (
            "right_low_shot",
            _obs(ball_x=-3800.0, ball_y=-180.0, ball_vx=-1000.0, ball_vy=0.0),
            {"intercept_low_right", "intercept_jump_right"},
        ),
        (
            "left_wide_jump",
            _obs(ball_x=-3800.0, ball_y=520.0, ball_vx=-1000.0, ball_vy=0.0),
            {"intercept_jump_left"},
        ),
        (
            "right_wide_jump",
            _obs(ball_x=-3800.0, ball_y=-520.0, ball_vx=-1000.0, ball_vy=0.0),
            {"intercept_jump_right"},
        ),
        (
            "ball_moving_away",
            _obs(ball_x=-3300.0, ball_y=0.0, ball_vx=600.0, ball_vy=0.0),
            {"walk_goalie_pose", "stand"},
        ),
        (
            "controlled_outlet",
            _obs(ball_x=GOALIE_LINE_X_MM + 180.0, ball_y=0.0, ball_vx=0.0, ball_vy=0.0, pass_options=1.0),
            {"pass"},
        ),
        (
            "controlled_clear_no_outlet",
            _obs(ball_x=GOALIE_LINE_X_MM + 180.0, ball_y=0.0, ball_vx=0.0, ball_vy=0.0, pass_options=0.0),
            {"clear"},
        ),
        (
            "controlled_dribble_wide",
            _obs(robot_x=-3650.0, robot_y=650.0, ball_x=-3470.0, ball_y=900.0, ball_vx=0.0, ball_vy=0.0, pass_options=0.0),
            {"dribble_out"},
        ),
        (
            "stale_team_ball",
            _obs(ball_x=-2600.0, ball_y=700.0, ball_vx=0.0, ball_vy=0.0, seen_ms=1800.0, team_ball_valid=True),
            {"observe"},
        ),
        (
            "misposition_recover",
            _obs(robot_y=-650.0, ball_x=-2200.0, ball_y=650.0, ball_vx=0.0, ball_vy=0.0),
            {"walk_goalie_pose"},
        ),
    ]


def validate_teacher() -> dict[str, object]:
    rows = []
    failures = []
    for name, obs, expected in _case_obs():
        action, cog = goalkeeper_teacher_action(obs)
        encoded = encode_goalkeeper_obs(obs, cog)
        mask = goalkeeper_skill_mask_np(encoded)
        allowed = bool(mask[int(action.skill_type)] > 0.0)
        row = {
            "case": name,
            "skill": action.skill_name,
            "expected_any": sorted(expected),
            "allowed": allowed,
            "rolling": bool(cog.rolling_towards_own_goal),
            "tti_s": None if not math.isfinite(cog.time_to_y_axis_s) else round(cog.time_to_y_axis_s, 3),
            "intersection_y": round(cog.intersection_y, 1),
            "goalie_error": [round(cog.goalie_error_x, 1), round(cog.goalie_error_y, 1)],
            "bridge_skill": gk_action_to_current_bridge(action, obs).skill_name,
            "native_bridge_skill": gk_action_to_native_bridge_name(action),
        }
        rows.append(row)
        if action.skill_name not in expected:
            failures.append(f"{name}: got {action.skill_name}, expected one of {sorted(expected)}")
        if not allowed:
            failures.append(f"{name}: emitted masked skill {action.skill_name}")
    return {"rows": rows, "failures": failures}


def validate_diversity(samples: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    counts: Counter[str] = Counter()
    masked = 0
    for _ in range(samples):
        kind = rng.choice(["center", "left", "right", "wide_left", "wide_right", "stale", "controlled", "away", "mispos"])
        if kind == "center":
            obs = _obs(ball_x=-3800.0 + rng.normal(0, 120), ball_y=rng.normal(0, 35), ball_vx=-rng.uniform(700, 1500))
        elif kind == "left":
            obs = _obs(ball_x=-3800.0, ball_y=rng.uniform(90, 260), ball_vx=-rng.uniform(700, 1500))
        elif kind == "right":
            obs = _obs(ball_x=-3800.0, ball_y=-rng.uniform(90, 260), ball_vx=-rng.uniform(700, 1500))
        elif kind == "wide_left":
            obs = _obs(ball_x=-3800.0, ball_y=rng.uniform(320, 620), ball_vx=-rng.uniform(700, 1500))
        elif kind == "wide_right":
            obs = _obs(ball_x=-3800.0, ball_y=-rng.uniform(320, 620), ball_vx=-rng.uniform(700, 1500))
        elif kind == "stale":
            obs = _obs(ball_x=-2500.0, ball_y=rng.uniform(-900, 900), ball_vx=0.0, seen_ms=1800.0)
        elif kind == "controlled":
            if rng.random() < 0.35:
                side = rng.choice([-1.0, 1.0])
                obs = _obs(
                    robot_x=-3650.0,
                    robot_y=side * rng.uniform(450, 750),
                    ball_x=-3470.0,
                    ball_y=side * rng.uniform(750, 980),
                    ball_vx=0.0,
                    pass_options=0.0,
                )
            else:
                obs = _obs(ball_x=GOALIE_LINE_X_MM + rng.uniform(120, 280), ball_y=rng.uniform(-120, 120), ball_vx=0.0, pass_options=rng.choice([0.0, 1.0]))
        elif kind == "away":
            obs = _obs(ball_x=-3900.0, ball_y=rng.uniform(-400, 400), ball_vx=rng.uniform(300, 900))
        else:
            obs = _obs(robot_y=rng.choice([-1, 1]) * rng.uniform(400, 750), ball_x=-2300.0, ball_y=rng.uniform(-800, 800), ball_vx=0.0)
        action, cog = goalkeeper_teacher_action(obs)
        encoded = encode_goalkeeper_obs(obs, cog)
        if goalkeeper_skill_mask_np(encoded)[int(action.skill_type)] <= 0.0:
            masked += 1
        counts[action.skill_name] += 1
    required = {
        "intercept_jump_left",
        "intercept_jump_right",
        "intercept_low_left",
        "intercept_low_right",
        "observe",
        "pass",
        "clear",
        "dribble_out",
        "walk_goalie_pose",
    }
    failures = []
    missing = sorted(required - set(counts))
    if missing:
        failures.append(f"missing required skill diversity: {missing}")
    if masked:
        failures.append(f"{masked} emitted labels were masked")
    return {
        "samples": samples,
        "skill_counts": dict(sorted(counts.items())),
        "failures": failures,
    }


def simrobot_smoke(args: argparse.Namespace) -> dict[str, object]:
    backend = PyBHBackend(
        mode=args.backend,
        player_number=args.player_number,
        frames_per_step=args.frames_per_step,
        obs_timeout_ms=1500,
        simrobot_profile=args.simrobot_profile,
        simrobot_launch=args.launch_simrobot,
        simrobot_bin=SIMROBOT_BIN,
    )
    rows = []
    backend.start()
    try:
        obs = backend.reset(ball_pos=np.asarray([-3800.0, 450.0], dtype=np.float32), robot_pos=np.asarray([GOALIE_LINE_X_MM, 0.0], dtype=np.float32), robot_theta=0.0)
        for step in range(1, args.sim_steps + 1):
            action, cog = goalkeeper_teacher_action(obs)
            bridge = gk_action_to_current_bridge(action, obs)
            rows.append({
                "step": step,
                "gk_skill": action.skill_name,
                "bridge_skill": bridge.skill_name,
                "obs_ready": bool(obs.obs_ready),
                "ball": [round(float(obs.ball_x), 1), round(float(obs.ball_y), 1)],
                "robot": [round(float(obs.robot_x), 1), round(float(obs.robot_y), 1)],
                "tti_s": None if not math.isfinite(cog.time_to_y_axis_s) else round(cog.time_to_y_axis_s, 3),
            })
            obs = backend.step(bridge)
    finally:
        backend.close()
    return {"rows": rows, "failures": []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=8808)
    parser.add_argument("--simrobot-smoke", action="store_true")
    parser.add_argument("--backend", choices=("simrobot", "simrobot_visible"), default="simrobot_visible")
    parser.add_argument("--simrobot-profile", default="3d_visible")
    parser.add_argument("--launch-simrobot", action="store_true")
    parser.add_argument("--frames-per-step", type=int, default=3)
    parser.add_argument("--sim-steps", type=int, default=20)
    parser.add_argument("--player-number", type=int, default=1)
    args = parser.parse_args()
    result = {
        "teacher_cases": validate_teacher(),
        "diversity": validate_diversity(args.samples, args.seed),
    }
    if args.simrobot_smoke:
        result["simrobot_smoke"] = simrobot_smoke(args)
    failures = []
    failures.extend(result["teacher_cases"]["failures"])
    failures.extend(result["diversity"]["failures"])
    failures.extend(result.get("simrobot_smoke", {}).get("failures", []))
    result["passed"] = not failures
    result["failures"] = failures
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
