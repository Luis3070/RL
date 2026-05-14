from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from config import BHUMAN_DIR, FIELD_X_HALF, FIELD_Y_HALF, GOAL_X, MAX_EPISODE_STEPS
from pybh_backend import WorldPlayerSpec, WorldSpec, build_world_spec
from reward import RewardWeights

ACTIVE_SKILLS = ("walk", "dribble", "shoot")
MASKED_SKILLS = ("pass", "mark", "block", "observe")


@dataclass(frozen=True)
class StageSpec:
    stage_id: int
    name: str
    description: str
    ball_radius_max_mm: float | None
    full_field: bool
    obstacle_mode: str
    active_skill_names: tuple[str, ...]
    reward_weights: RewardWeights
    max_episode_steps: int = MAX_EPISODE_STEPS
    dynamic_update_interval_steps: int = 0


def _scene_path(stage_id: int, simrobot_profile: str) -> Path:
    if simrobot_profile == "2d":
        if stage_id >= 2:
            return BHUMAN_DIR / "Config/Scenes/RLFull.ros2d"
        return BHUMAN_DIR / "Config/Scenes/RL1v0.ros2d"
    if stage_id >= 2:
        return BHUMAN_DIR / "Config/Scenes/RLFull_3D.ros2"
    return BHUMAN_DIR / "Config/Scenes/RL1v0_3D.ros2"


_STAGE_SPECS = {
    0: StageSpec(
        stage_id=0,
        name="stage_0",
        description="Approach y takeover limpio sin obstáculos a distancia corta.",
        ball_radius_max_mm=800.0,
        full_field=False,
        obstacle_mode="none",
        active_skill_names=ACTIVE_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=1.2,
            align_ball=0.8,
            enter_takeover=2.5,
            hold_takeover=0.4,
            good_walk=0.5,
            bad_walk=-0.8,
            good_dribble=0.8,
            bad_dribble=-1.2,
            good_shoot=1.2,
            bad_shoot=-1.8,
            skill_switch=-0.15,
            thrash=-0.8,
            goal=4.0,
            timeout=-2.0,
            ball_progress_post_takeover=0.3,
        ),
    ),
    1: StageSpec(
        stage_id=1,
        name="stage_1",
        description="Misma tarea que Stage 0 con mayor exigencia temporal.",
        ball_radius_max_mm=1600.0,
        full_field=False,
        obstacle_mode="none",
        active_skill_names=ACTIVE_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=1.0,
            align_ball=0.7,
            enter_takeover=2.2,
            hold_takeover=0.4,
            good_walk=0.4,
            bad_walk=-0.8,
            good_dribble=0.9,
            bad_dribble=-1.2,
            good_shoot=1.3,
            bad_shoot=-2.0,
            skill_switch=-0.2,
            thrash=-1.0,
            goal=4.0,
            timeout=-2.5,
            ball_progress_post_takeover=0.5,
        ),
    ),
    2: StageSpec(
        stage_id=2,
        name="stage_2",
        description="Takeover limpio con clearance y obstáculos estáticos.",
        ball_radius_max_mm=3200.0,
        full_field=False,
        obstacle_mode="static",
        active_skill_names=ACTIVE_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=0.9,
            align_ball=0.6,
            enter_takeover=2.0,
            hold_takeover=0.5,
            good_walk=0.3,
            bad_walk=-0.9,
            good_dribble=1.1,
            bad_dribble=-1.3,
            good_shoot=1.4,
            bad_shoot=-2.0,
            collision=-3.0,
            clearance_gain=1.0,
            skill_switch=-0.2,
            thrash=-1.0,
            goal=4.5,
            timeout=-2.5,
            ball_progress_post_takeover=0.7,
        ),
    ),
    3: StageSpec(
        stage_id=3,
        name="stage_3",
        description="Robustez temporal con obstáculos dinámicos scripted.",
        ball_radius_max_mm=None,
        full_field=True,
        obstacle_mode="dynamic",
        active_skill_names=ACTIVE_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=0.7,
            align_ball=0.5,
            enter_takeover=1.8,
            hold_takeover=0.6,
            good_walk=0.2,
            bad_walk=-1.0,
            good_dribble=1.4,
            bad_dribble=-1.4,
            good_shoot=1.5,
            bad_shoot=-2.2,
            collision=-3.2,
            clearance_gain=1.0,
            corridor_keep=1.1,
            ball_progress_post_takeover=1.0,
            skill_switch=-0.25,
            thrash=-1.2,
            goal=5.0,
            timeout=-2.5,
        ),
        dynamic_update_interval_steps=4,
    ),
}


def get_stage_spec(stage_id: int) -> StageSpec:
    if stage_id not in _STAGE_SPECS:
        valid = ", ".join(str(k) for k in sorted(_STAGE_SPECS))
        raise ValueError(f"Unsupported stage_id {stage_id!r}. Expected one of: {valid}")
    return _STAGE_SPECS[stage_id]


def get_stage_scene_path(stage_id: int, simrobot_profile: str) -> Path:
    return _scene_path(stage_id, simrobot_profile)


def stage_skill_mask(stage_id: int) -> Dict[str, int]:
    stage = get_stage_spec(stage_id)
    return {
        "walk": 1 if "walk" in stage.active_skill_names else 0,
        "dribble": 1 if "dribble" in stage.active_skill_names else 0,
        "shoot": 1 if "shoot" in stage.active_skill_names else 0,
        "pass": 0,
        "mark": 0,
        "block": 0,
        "observe": 0,
    }


def _sample_pose_ball_pair(
    rng: np.random.Generator,
    *,
    radius_max_mm: float,
    robot_xy_bounds: tuple[float, float, float, float],
    line_angle_range_deg: float,
    heading_noise_deg: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_min, x_max, y_min, y_max = robot_xy_bounds
    for _ in range(128):
        robot_x = float(rng.uniform(x_min, x_max))
        robot_y = float(rng.uniform(y_min, y_max))
        radius = float(rng.uniform(max(150.0, radius_max_mm * 0.25), radius_max_mm))
        line_angle = float(rng.uniform(-math.radians(line_angle_range_deg), math.radians(line_angle_range_deg)))
        ball_x = robot_x + radius * math.cos(line_angle)
        ball_y = robot_y + radius * math.sin(line_angle)
        if abs(ball_x) > FIELD_X_HALF - 200.0 or abs(ball_y) > FIELD_Y_HALF - 200.0:
            continue
        robot_theta = line_angle + float(rng.uniform(-math.radians(heading_noise_deg), math.radians(heading_noise_deg)))
        return (
            np.array([ball_x, ball_y], dtype=np.float32),
            np.array([robot_x, robot_y], dtype=np.float32),
            robot_theta,
        )
    raise RuntimeError("Could not sample a valid stage reset within field bounds")


def _sample_full_field_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    for _ in range(128):
        robot_x = float(rng.uniform(-FIELD_X_HALF * 0.65, FIELD_X_HALF * 0.15))
        robot_y = float(rng.uniform(-FIELD_Y_HALF * 0.75, FIELD_Y_HALF * 0.75))
        ball_x = float(rng.uniform(-FIELD_X_HALF * 0.25, FIELD_X_HALF * 0.55))
        ball_y = float(rng.uniform(-FIELD_Y_HALF * 0.75, FIELD_Y_HALF * 0.75))
        delta_x = ball_x - robot_x
        delta_y = ball_y - robot_y
        distance = math.hypot(delta_x, delta_y)
        if distance < 600.0 or distance > 3500.0:
            continue
        robot_theta = math.atan2(delta_y, delta_x) + float(rng.uniform(-math.radians(55.0), math.radians(55.0)))
        return (
            np.array([ball_x, ball_y], dtype=np.float32),
            np.array([robot_x, robot_y], dtype=np.float32),
            robot_theta,
        )
    raise RuntimeError("Could not sample a valid full-field stage reset")


def _clip_field(x: float, y: float) -> tuple[float, float]:
    return (
        float(np.clip(x, -FIELD_X_HALF + 150.0, FIELD_X_HALF - 150.0)),
        float(np.clip(y, -FIELD_Y_HALF + 150.0, FIELD_Y_HALF - 150.0)),
    )


def _support_players_far() -> tuple[WorldPlayerSpec, ...]:
    return (
        WorldPlayerSpec(number=2, x=-FIELD_X_HALF + 500.0, y=-FIELD_Y_HALF + 500.0),
        WorldPlayerSpec(number=3, x=-FIELD_X_HALF + 500.0, y=FIELD_Y_HALF - 500.0),
    )


def _static_obstacles(robot_pos: np.ndarray, ball_pos: np.ndarray) -> tuple[WorldPlayerSpec, ...]:
    direction = ball_pos - robot_pos
    distance = float(np.linalg.norm(direction))
    if distance < 1.0:
        direction = np.array([1.0, 0.0], dtype=np.float32)
        distance = 1.0
    direction = direction / distance
    lateral = np.array([-direction[1], direction[0]], dtype=np.float32)
    midpoint = robot_pos + direction * min(distance * 0.55, 1800.0)
    opp1_x, opp1_y = _clip_field(float(midpoint[0] + lateral[0] * 260.0), float(midpoint[1] + lateral[1] * 260.0))
    opp2_x, opp2_y = _clip_field(float(midpoint[0] - lateral[0] * 260.0), float(midpoint[1] - lateral[1] * 260.0))
    return (
        WorldPlayerSpec(number=1, x=opp1_x, y=opp1_y, theta=math.atan2(-direction[1], -direction[0])),
        WorldPlayerSpec(number=2, x=opp2_x, y=opp2_y, theta=math.atan2(-direction[1], -direction[0])),
    )


def _dynamic_obstacles(robot_pos: np.ndarray, ball_pos: np.ndarray, episode_step: int) -> tuple[WorldPlayerSpec, ...]:
    direction = ball_pos - robot_pos
    distance = float(np.linalg.norm(direction))
    if distance < 1.0:
        direction = np.array([1.0, 0.0], dtype=np.float32)
        distance = 1.0
    direction = direction / distance
    lateral = np.array([-direction[1], direction[0]], dtype=np.float32)
    phase = float(episode_step) * 0.45
    midpoint = robot_pos + direction * min(distance * 0.58, 2200.0)
    lane_shift = 320.0 * math.sin(phase)
    near_ball_shift = 260.0 * math.cos(phase)
    opp1_x, opp1_y = _clip_field(float(midpoint[0] + lateral[0] * lane_shift), float(midpoint[1] + lateral[1] * lane_shift))
    opp2_anchor = ball_pos + direction * -220.0
    opp2_x, opp2_y = _clip_field(
        float(opp2_anchor[0] - lateral[0] * near_ball_shift),
        float(opp2_anchor[1] - lateral[1] * near_ball_shift),
    )
    theta = math.atan2(float(direction[1]), float(direction[0]))
    return (
        WorldPlayerSpec(number=1, x=opp1_x, y=opp1_y, theta=theta + math.pi),
        WorldPlayerSpec(number=2, x=opp2_x, y=opp2_y, theta=theta + math.pi),
    )


def sample_stage_world_spec(stage_id: int, rng: np.random.Generator) -> WorldSpec:
    stage = get_stage_spec(stage_id)
    if stage.stage_id == 0:
        ball_pos, robot_pos, robot_theta = _sample_pose_ball_pair(
            rng,
            radius_max_mm=stage.ball_radius_max_mm or 800.0,
            robot_xy_bounds=(-1600.0, -400.0, -700.0, 700.0),
            line_angle_range_deg=25.0,
            heading_noise_deg=20.0,
        )
        return build_world_spec(ball_pos=ball_pos, robot_pos=robot_pos, robot_theta=robot_theta)
    if stage.stage_id == 1:
        ball_pos, robot_pos, robot_theta = _sample_pose_ball_pair(
            rng,
            radius_max_mm=stage.ball_radius_max_mm or 1600.0,
            robot_xy_bounds=(-2200.0, -500.0, -900.0, 900.0),
            line_angle_range_deg=40.0,
            heading_noise_deg=30.0,
        )
        return build_world_spec(ball_pos=ball_pos, robot_pos=robot_pos, robot_theta=robot_theta)
    if stage.stage_id == 2:
        ball_pos, robot_pos, robot_theta = _sample_pose_ball_pair(
            rng,
            radius_max_mm=stage.ball_radius_max_mm or 3200.0,
            robot_xy_bounds=(-2800.0, -600.0, -1200.0, 1200.0),
            line_angle_range_deg=55.0,
            heading_noise_deg=40.0,
        )
        return build_world_spec(
            ball_pos=ball_pos,
            robot_pos=robot_pos,
            robot_theta=robot_theta,
            teammates=_support_players_far(),
            opponents=_static_obstacles(robot_pos, ball_pos),
        )
    if stage.stage_id == 3:
        ball_pos, robot_pos, robot_theta = _sample_full_field_pair(rng)
        return build_world_spec(
            ball_pos=ball_pos,
            robot_pos=robot_pos,
            robot_theta=robot_theta,
            teammates=_support_players_far(),
            opponents=_dynamic_obstacles(robot_pos, ball_pos, episode_step=0),
        )
    raise ValueError(f"Unsupported stage_id {stage_id!r}")


def get_dynamic_world_update(stage_id: int, episode_step: int, robot_x: float, robot_y: float, ball_x: float, ball_y: float) -> Dict[str, object] | None:
    stage = get_stage_spec(stage_id)
    if stage.obstacle_mode != "dynamic":
        return None
    if stage.dynamic_update_interval_steps <= 0 or episode_step % stage.dynamic_update_interval_steps != 0:
        return None
    opponents = _dynamic_obstacles(
        np.array([robot_x, robot_y], dtype=np.float32),
        np.array([ball_x, ball_y], dtype=np.float32),
        episode_step=episode_step,
    )
    return {"opponents": opponents}


def promote_stage(stage_id: int, metrics: Mapping[str, float], history: Mapping[int, Mapping[str, float]]) -> bool:
    if stage_id == 0:
        return (
            metrics.get("clean_walk_rate", 0.0) >= 0.95
            and metrics.get("clean_takeover_rate", 0.0) >= 0.95
            and metrics.get("clean_finish_rate", 0.0) >= 0.95
            and metrics.get("thrash_rate", 1.0) <= 0.05
        )
    if stage_id == 1:
        stage0 = history.get(0, {})
        return (
            metrics.get("clean_walk_rate", 0.0) >= 0.90
            and metrics.get("clean_takeover_rate", 0.0) >= 0.90
            and metrics.get("clean_finish_rate", 0.0) >= 0.90
            and metrics.get("mean_time_to_takeover", float("inf")) <= stage0.get("mean_time_to_takeover", float("inf"))
            and metrics.get("thrash_rate", 1.0) <= 0.08
        )
    if stage_id == 2:
        return (
            metrics.get("clean_takeover_rate", 0.0) >= 0.85
            and metrics.get("clean_finish_rate", 0.0) >= 0.85
            and metrics.get("collision_rate", 1.0) <= 0.05
            and metrics.get("obstacle_clearance_success", 0.0) >= 0.85
            and metrics.get("thrash_rate", 1.0) <= 0.10
        )
    if stage_id == 3:
        stage2 = history.get(2, {})
        return (
            metrics.get("clean_episode_rate", 0.0) >= 0.80
            and metrics.get("collision_rate", 1.0) <= 0.08
            and metrics.get("dynamic_obstacle_success", 0.0) > 0.50
            and metrics.get("stability_score", 0.0) >= stage2.get("stability_score", 0.0)
        )
    return False
