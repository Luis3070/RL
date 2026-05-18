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
WALK_ONLY_SKILLS = ("walk",)
TAKEOVER_SKILLS = ("walk", "dribble")
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
    if simrobot_profile == "3d_visible":
        if stage_id >= 2:
            return BHUMAN_DIR / "Config/Scenes/RLFull_3D.ros2"
        return BHUMAN_DIR / "Config/Scenes/RL1v0_3D_visible.ros2"
    if stage_id >= 2:
        return BHUMAN_DIR / "Config/Scenes/RLFull_3D.ros2"
    return BHUMAN_DIR / "Config/Scenes/RL1v0_3D.ros2"


_STAGE_SPECS = {
    -1: StageSpec(
        stage_id=-1,
        name="stage_0a_walk_approach",
        description="Pre-stage: caminar hacia la pelota sin cambio de skill.",
        ball_radius_max_mm=1200.0,
        full_field=False,
        obstacle_mode="none",
        active_skill_names=WALK_ONLY_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=1.2,
            align_ball=0.8,
            approach_ball=0.12,
            good_walk=0.6,
            projection_penalty=-0.05,
            skill_switch=-0.02,
            thrash=-0.5,
            timeout=-1.5,
            preferred_skill_match=0.08,
            keep_ball_front=0.12,
            bad_ball_angle=-0.10,
            rescued_ball=-0.40,
            stale_ball=-0.20,
            collapsed_ball=-1.20,
            corrected_ball=-0.30,
            too_close_walk=-0.35,
            lost_trusted_ball=-0.25,
        ),
    ),
    0: StageSpec(
        stage_id=0,
        name="stage_0",
        description="Approach y takeover limpio walk→dribble sin shoot.",
        ball_radius_max_mm=800.0,
        full_field=False,
        obstacle_mode="none",
        active_skill_names=TAKEOVER_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=1.2,
            align_ball=0.8,
            enter_takeover=2.5,
            hold_takeover=0.8,
            approach_ball=0.10,
            good_walk=0.45,
            bad_walk=-0.35,
            good_dribble=1.0,
            bad_dribble=-0.65,
            projection_penalty=-0.10,
            skill_switch=-0.05,
            thrash=-1.0,
            goal=4.0,
            timeout=-2.0,
            ball_progress_post_takeover=0.6,
            preferred_skill_match=0.05,
            keep_ball_front=0.15,
            bad_ball_angle=-0.18,
            rescued_ball=-0.75,
            stale_ball=-0.35,
            collapsed_ball=-2.0,
            corrected_ball=-0.60,
            too_close_walk=-0.60,
            lost_trusted_ball=-0.50,
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
            approach_ball=0.05,
            good_walk=0.4,
            bad_walk=-0.8,
            good_dribble=0.9,
            bad_dribble=-1.2,
            good_shoot=1.3,
            bad_shoot=-2.0,
            projection_penalty=-0.15,
            skill_switch=-0.2,
            thrash=-1.0,
            goal=4.0,
            timeout=-2.5,
            ball_progress_post_takeover=0.5,
            rescued_ball=-0.55,
            stale_ball=-0.25,
            collapsed_ball=-1.50,
            corrected_ball=-0.45,
            too_close_walk=-0.45,
            lost_trusted_ball=-0.35,
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
            approach_ball=0.04,
            good_walk=0.3,
            bad_walk=-0.9,
            good_dribble=1.1,
            bad_dribble=-1.3,
            good_shoot=1.4,
            bad_shoot=-2.0,
            projection_penalty=-0.20,
            collision=-3.0,
            clearance_gain=1.0,
            skill_switch=-0.2,
            thrash=-1.0,
            goal=4.5,
            timeout=-2.5,
            ball_progress_post_takeover=0.7,
            rescued_ball=-0.45,
            stale_ball=-0.20,
            collapsed_ball=-1.25,
            corrected_ball=-0.35,
            too_close_walk=-0.35,
            lost_trusted_ball=-0.25,
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
            approach_ball=0.03,
            good_walk=0.2,
            bad_walk=-1.0,
            good_dribble=1.4,
            bad_dribble=-1.4,
            good_shoot=1.5,
            bad_shoot=-2.2,
            projection_penalty=-0.20,
            collision=-3.2,
            clearance_gain=1.0,
            corridor_keep=1.1,
            ball_progress_post_takeover=1.0,
            skill_switch=-0.25,
            thrash=-1.2,
            goal=5.0,
            timeout=-2.5,
            rescued_ball=-0.40,
            stale_ball=-0.20,
            collapsed_ball=-1.20,
            corrected_ball=-0.30,
            too_close_walk=-0.30,
            lost_trusted_ball=-0.25,
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
    if stage.stage_id == -1:
        ball_pos, robot_pos, robot_theta = _sample_pose_ball_pair(
            rng,
            radius_max_mm=stage.ball_radius_max_mm or 1200.0,
            robot_xy_bounds=(-1700.0, -450.0, -650.0, 650.0),
            line_angle_range_deg=12.0,
            heading_noise_deg=8.0,
        )
        return build_world_spec(ball_pos=ball_pos, robot_pos=robot_pos, robot_theta=robot_theta)
    if stage.stage_id == 0:
        ball_pos, robot_pos, robot_theta = _sample_pose_ball_pair(
            rng,
            radius_max_mm=stage.ball_radius_max_mm or 800.0,
            robot_xy_bounds=(-1500.0, -500.0, -550.0, 550.0),
            line_angle_range_deg=18.0,
            heading_noise_deg=12.0,
        )
        return build_world_spec(ball_pos=ball_pos, robot_pos=robot_pos, robot_theta=robot_theta)
    if stage.stage_id == 1:
        ball_pos, robot_pos, robot_theta = _sample_pose_ball_pair(
            rng,
            radius_max_mm=stage.ball_radius_max_mm or 1600.0,
            robot_xy_bounds=(-2100.0, -550.0, -800.0, 800.0),
            line_angle_range_deg=28.0,
            heading_noise_deg=18.0,
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


def _metric(metrics: Mapping[str, float], name: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _wire_is_healthy(metrics: Mapping[str, float]) -> bool:
    return str(metrics.get("wire_audit_status", "PASS")) != "FAIL"


def _ball_semantics_ok(
    metrics: Mapping[str, float],
    *,
    trusted_min: float,
    rescued_max: float,
    confidence_min: float,
) -> bool:
    return (
        _metric(metrics, "ball_visible_rate") >= 0.90
        and _metric(metrics, "ball_trusted_rate") >= trusted_min
        and _metric(metrics, "ball_rescued_rate") <= rescued_max
        and _metric(metrics, "ball_confidence_score_mean") >= confidence_min
    )


def _behavior_is_stable(metrics: Mapping[str, float], *, thrash_max: float, collision_max: float) -> bool:
    return (
        _wire_is_healthy(metrics)
        and _metric(metrics, "thrash_rate", 1.0) <= thrash_max
        and _metric(metrics, "collision_rate", 1.0) <= collision_max
    )


def promote_stage(stage_id: int, metrics: Mapping[str, float], history: Mapping[int, Mapping[str, float]]) -> bool:
    if stage_id == -1:
        return (
            _ball_semantics_ok(metrics, trusted_min=0.55, rescued_max=0.40, confidence_min=0.48)
            and _behavior_is_stable(metrics, thrash_max=0.05, collision_max=0.10)
            and _metric(metrics, "episode_reward_mean", -999.0) >= 0.10
            and _metric(metrics, "takeover_success_rate") >= 0.80
            and _metric(metrics, "mean_time_to_takeover", float("inf")) <= 70.0
            and _metric(metrics, "fov_perception_failure_rate", 0.0) <= 0.20
        )
    if stage_id == 0:
        return (
            _ball_semantics_ok(metrics, trusted_min=0.65, rescued_max=0.30, confidence_min=0.55)
            and _behavior_is_stable(metrics, thrash_max=0.08, collision_max=0.10)
            and _metric(metrics, "episode_reward_mean", -999.0) >= -0.15
            and _metric(metrics, "stability_score") >= 0.45
            and _metric(metrics, "takeover_success_rate") >= 0.80
            and _metric(metrics, "mean_time_to_takeover", float("inf")) <= 35.0
            and _metric(metrics, "dribble_compat_rate") >= 0.20
            and _metric(metrics, "bad_walk_rate", 1.0) <= 0.45
            and _metric(metrics, "fov_perception_failure_rate", 0.0) <= 0.10
            and str(metrics.get("wire_audit_status", "PASS")) == "PASS"
        )
    if stage_id == 1:
        stage0 = history.get(0, {})
        return (
            _ball_semantics_ok(metrics, trusted_min=0.58, rescued_max=0.38, confidence_min=0.48)
            and _behavior_is_stable(metrics, thrash_max=0.10, collision_max=0.10)
            and _metric(metrics, "episode_reward_mean", -999.0) >= _metric(stage0, "episode_reward_mean", -0.10) - 0.05
            and (
                _metric(metrics, "walk_compat_rate") >= 0.08
                or _metric(metrics, "takeover_success_rate") >= 0.35
                or _metric(metrics, "mean_time_to_takeover", float("inf")) <= 35.0
            )
            and (
                _metric(metrics, "takeover_success_rate") >= 0.10
                or _metric(metrics, "mean_time_to_takeover", float("inf")) <= _metric(stage0, "mean_time_to_takeover", float("inf"))
            )
        )
    if stage_id == 2:
        return (
            _ball_semantics_ok(metrics, trusted_min=0.55, rescued_max=0.45, confidence_min=0.45)
            and _behavior_is_stable(metrics, thrash_max=0.12, collision_max=0.12)
            and _metric(metrics, "episode_reward_mean", -999.0) >= -0.25
            and (
                _metric(metrics, "obstacle_clearance_success") >= 0.25
                or _metric(metrics, "clearance_positive_steps") > 0.0
                or _metric(metrics, "stability_score") >= 0.42
            )
        )
    if stage_id == 3:
        stage2 = history.get(2, {})
        return (
            _ball_semantics_ok(metrics, trusted_min=0.55, rescued_max=0.45, confidence_min=0.45)
            and _behavior_is_stable(metrics, thrash_max=0.12, collision_max=0.12)
            and _metric(metrics, "dynamic_obstacle_success") >= 0.20
            and _metric(metrics, "stability_score") >= _metric(stage2, "stability_score", 0.0) - 0.05
        )
    return False
