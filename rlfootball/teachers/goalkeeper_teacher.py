from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from rlfootball.core.action import SkillType, StructuredSkillAction
from rlfootball.core.config import ACTION
from rlfootball.core.goalkeeper_action import GK_SKILL_ORDER, GKSkillType, GKStructuredAction
from rlfootball.core.pybh_backend import BackendObservation


GK_OBS_SIZE = 64
OWN_GOAL_X_MM = float(os.getenv("RL_GK_OWN_GOAL_X_MM", "-4500.0"))
OWN_GOAL_HALF_Y_MM = float(os.getenv("RL_GK_OWN_GOAL_HALF_Y_MM", "800.0"))
GOALIE_LINE_X_MM = float(os.getenv("RL_GK_GOALIE_LINE_X_MM", "-4300.0"))
GOALIE_LINE_Y_LIMIT_MM = float(os.getenv("RL_GK_GOALIE_LINE_Y_LIMIT_MM", "700.0"))
PENALTY_X_LIMIT_MM = float(os.getenv("RL_GK_PENALTY_X_LIMIT_MM", "-3900.0"))
PENALTY_HALF_Y_MM = float(os.getenv("RL_GK_PENALTY_HALF_Y_MM", "1100.0"))

STAND_RADIUS_MM = float(os.getenv("RL_GK_STAND_RADIUS_MM", "60.0"))
WALK_RADIUS_MM = float(os.getenv("RL_GK_WALK_RADIUS_MM", "125.0"))
GENUFLECT_RADIUS_MM = float(os.getenv("RL_GK_GENUFLECT_RADIUS_MM", "200.0"))
JUMP_RADIUS_MM = float(os.getenv("RL_GK_JUMP_RADIUS_MM", "600.0"))
BALL_CATCH_MAX_WALK_DISTANCE_MM = float(os.getenv("RL_GK_BALL_CATCH_MAX_WALK_DISTANCE_MM", "1200.0"))
INTERCEPT_TIME_MIN_S = float(os.getenv("RL_GK_INTERCEPT_TIME_MIN_S", "0.02"))
INTERCEPT_TIME_MAX_S = float(os.getenv("RL_GK_INTERCEPT_TIME_MAX_S", "3.00"))
FRESH_BALL_MAX_MS = float(os.getenv("RL_GK_FRESH_BALL_MAX_MS", "150.0"))
STALE_BALL_MS = float(os.getenv("RL_GK_STALE_BALL_MS", "1000.0"))
CONTROL_RADIUS_MM = float(os.getenv("RL_GK_CONTROL_RADIUS_MM", "450.0"))
KEEPER_JUMPING_ON = os.getenv("RL_GK_KEEPER_JUMPING_ON", "1").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GKCognition:
    goalie_x: float
    goalie_y: float
    goalie_theta: float
    goalie_error_x: float
    goalie_error_y: float
    goalie_error_theta: float
    rolling_towards_own_goal: bool
    time_to_y_axis_s: float
    intersection_y: float
    near_left_post: bool
    near_right_post: bool
    inside_penalty_area: bool
    ball_controlled: bool
    fresh_ball: bool
    stale_ball: bool


def _wrap_angle(theta: float) -> float:
    return float((theta + math.pi) % (2.0 * math.pi) - math.pi)


def _field_to_robot(obs: BackendObservation, x: float, y: float) -> tuple[float, float]:
    dx = float(x) - float(obs.robot_x)
    dy = float(y) - float(obs.robot_y)
    c = math.cos(-float(obs.robot_theta))
    s = math.sin(-float(obs.robot_theta))
    return dx * c - dy * s, dx * s + dy * c


def _goalie_pose(obs: BackendObservation) -> tuple[float, float, float]:
    bx = float(obs.ball_x)
    by = float(obs.ball_y)
    dx = max(1.0, bx - OWN_GOAL_X_MM)
    # Simple BOB-line approximation: shift on goal line opposite the incoming
    # shot angle, clipped between posts. B-Human's GoaliePoseProvider remains the
    # production reference; this is a deployable fallback until exported directly.
    y = float(np.clip(by * (abs(GOALIE_LINE_X_MM - OWN_GOAL_X_MM) / dx), -GOALIE_LINE_Y_LIMIT_MM, GOALIE_LINE_Y_LIMIT_MM))
    theta = math.atan2(by - y, bx - GOALIE_LINE_X_MM)
    return GOALIE_LINE_X_MM, y, theta


def _time_to_own_y_axis(obs: BackendObservation) -> tuple[float, float]:
    bx = float(obs.ball_rel_x)
    by = float(obs.ball_rel_y)
    vx = float(obs.ball_vel_x)
    vy = float(obs.ball_vel_y)
    if vx >= -1e-3 or bx < 0.0:
        return float("inf"), 0.0
    t = -bx / vx
    if t < 0.0:
        return float("inf"), 0.0
    return t, by + vy * t


def build_gk_cognition(obs: BackendObservation) -> GKCognition:
    gx, gy, gt = _goalie_pose(obs)
    rx, ry = _field_to_robot(obs, gx, gy)
    t_y, intersect_y = _time_to_own_y_axis(obs)
    ball_speed = math.hypot(float(obs.ball_vel_x), float(obs.ball_vel_y))
    rolling = (
        ball_speed > 80.0
        and float(obs.ball_vel_x) < 0.0
        and abs(intersect_y) <= OWN_GOAL_HALF_Y_MM + 250.0
        and math.isfinite(t_y)
    )
    inside_penalty = float(obs.robot_x) < PENALTY_X_LIMIT_MM and abs(float(obs.robot_y)) < PENALTY_HALF_Y_MM
    return GKCognition(
        goalie_x=gx,
        goalie_y=gy,
        goalie_theta=gt,
        goalie_error_x=rx,
        goalie_error_y=ry,
        goalie_error_theta=_wrap_angle(gt - float(obs.robot_theta)),
        rolling_towards_own_goal=rolling,
        time_to_y_axis_s=t_y,
        intersection_y=float(intersect_y),
        near_left_post=math.hypot(float(obs.robot_x) - OWN_GOAL_X_MM, float(obs.robot_y) - OWN_GOAL_HALF_Y_MM) < 450.0,
        near_right_post=math.hypot(float(obs.robot_x) - OWN_GOAL_X_MM, float(obs.robot_y) + OWN_GOAL_HALF_Y_MM) < 450.0,
        inside_penalty_area=inside_penalty,
        ball_controlled=math.hypot(float(obs.ball_rel_x), float(obs.ball_rel_y)) < CONTROL_RADIUS_MM,
        fresh_ball=float(obs.natural_time_since_ball_seen) <= FRESH_BALL_MAX_MS,
        stale_ball=float(obs.natural_time_since_ball_seen) > STALE_BALL_MS,
    )


def encode_goalkeeper_obs(obs: BackendObservation, cognition: GKCognition | None = None) -> np.ndarray:
    c = cognition or build_gk_cognition(obs)

    def dist(value: float) -> float:
        return float(np.clip(value / ACTION.x_range, -1.0, 1.0))

    def time_ms(value: float) -> float:
        return float(np.clip(value / 5000.0, 0.0, 1.0))

    tti = 1.0 if not math.isfinite(c.time_to_y_axis_s) else float(np.clip(c.time_to_y_axis_s / 4.0, 0.0, 1.0))
    features = [
        dist(float(obs.robot_x)),
        float(np.clip(float(obs.robot_y) / ACTION.y_range, -1.0, 1.0)),
        float(obs.robot_theta / math.pi),
        dist(float(obs.ball_rel_x)),
        float(np.clip(float(obs.ball_rel_y) / ACTION.y_range, -1.0, 1.0)),
        dist(float(obs.ball_end_rel_x)),
        float(np.clip(float(obs.ball_end_rel_y) / ACTION.y_range, -1.0, 1.0)),
        float(np.clip(float(obs.ball_vel_x) / 2500.0, -1.0, 1.0)),
        float(np.clip(float(obs.ball_vel_y) / 2500.0, -1.0, 1.0)),
        time_ms(float(obs.time_since_ball_seen)),
        time_ms(float(obs.natural_time_since_ball_seen)),
        float(np.clip(float(obs.ball_seen_percentage) / 100.0, 0.0, 1.0)),
        1.0 if obs.ball_consistent_with_game_state else 0.0,
        dist(c.goalie_error_x),
        float(np.clip(c.goalie_error_y / ACTION.y_range, -1.0, 1.0)),
        float(c.goalie_error_theta / math.pi),
        dist(c.goalie_x),
        float(np.clip(c.goalie_y / ACTION.y_range, -1.0, 1.0)),
        float(c.goalie_theta / math.pi),
        1.0 if c.rolling_towards_own_goal else 0.0,
        tti,
        float(np.clip(c.intersection_y / BALL_CATCH_MAX_WALK_DISTANCE_MM, -1.0, 1.0)),
        1.0 if c.near_left_post else 0.0,
        1.0 if c.near_right_post else 0.0,
        1.0 if c.inside_penalty_area else 0.0,
        1.0 if c.ball_controlled else 0.0,
        float(np.clip(float(obs.pass_options_count) / 4.0, 0.0, 1.0)),
        1.0 if c.fresh_ball else 0.0,
        1.0 if c.stale_ball else 0.0,
        dist(float(obs.global_team_ball_x_mm) - float(obs.robot_x)),
        float(np.clip((float(obs.global_team_ball_y_mm) - float(obs.robot_y)) / ACTION.y_range, -1.0, 1.0)),
        time_ms(float(obs.global_team_ball_age_ms)),
        1.0 if obs.global_team_ball_valid else 0.0,
        float(np.clip(float(obs.nearest_opponent_dist) / ACTION.x_range, 0.0, 1.0)),
        float(np.clip(float(obs.nearest_opponent_front_dist) / ACTION.x_range, 0.0, 1.0)),
        float(np.clip(float(obs.nearest_teammate_dist) / ACTION.x_range, 0.0, 1.0)),
        1.0 if obs.motion_engine_ground_contact else 0.0,
        float(np.clip(float(obs.motion_engine_fall_state) / 5.0, 0.0, 1.0)),
    ]
    if len(features) > GK_OBS_SIZE:
        raise AssertionError("GK feature vector grew past GK_OBS_SIZE")
    features.extend([0.0] * (GK_OBS_SIZE - len(features)))
    return np.asarray(features, dtype=np.float32)


def goalkeeper_skill_mask_np(encoded_obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(encoded_obs, dtype=np.float32)
    mask = np.zeros(len(GK_SKILL_ORDER), dtype=np.float32)
    mask[GKSkillType.stand] = 1.0
    mask[GKSkillType.walk_goalie_pose] = 1.0
    mask[GKSkillType.observe] = 1.0 if obs[28] >= 0.5 or obs[32] >= 0.5 else 0.0
    rolling = obs[19] >= 0.5
    # obs[20] stores time-to-intercept normalized by 4 seconds. Keep the mask
    # thresholds in the same unit so late close-range shots are not rejected.
    t_min = INTERCEPT_TIME_MIN_S / 4.0
    t_max = INTERCEPT_TIME_MAX_S / 4.0
    t_ok = t_min <= obs[20] <= t_max
    fresh = obs[27] >= 0.5
    y = float(obs[21]) * BALL_CATCH_MAX_WALK_DISTANCE_MM
    inside_penalty = obs[24] >= 0.5
    near_left = obs[22] >= 0.5
    near_right = obs[23] >= 0.5
    stable = obs[37] <= 0.4
    intercept_base = rolling and t_ok and fresh and stable
    if intercept_base and abs(y) <= STAND_RADIUS_MM:
        mask[GKSkillType.intercept_center] = 1.0
    if intercept_base and abs(y) <= WALK_RADIUS_MM:
        mask[GKSkillType.intercept_lateral] = 1.0
    if intercept_base and 0.0 < y <= GENUFLECT_RADIUS_MM:
        mask[GKSkillType.intercept_low_left] = 1.0
    if intercept_base and -GENUFLECT_RADIUS_MM <= y < 0.0:
        mask[GKSkillType.intercept_low_right] = 1.0
    if KEEPER_JUMPING_ON and intercept_base and inside_penalty and 0.0 < y <= JUMP_RADIUS_MM and not near_left:
        mask[GKSkillType.intercept_jump_left] = 1.0
    if KEEPER_JUMPING_ON and intercept_base and inside_penalty and -JUMP_RADIUS_MM <= y < 0.0 and not near_right:
        mask[GKSkillType.intercept_jump_right] = 1.0
    controlled = obs[25] >= 0.5
    if controlled:
        mask[GKSkillType.clear] = 1.0
        mask[GKSkillType.pass_] = 1.0
        mask[GKSkillType.dribble_out] = 1.0
    return mask


def goalkeeper_skill_mask_torch(observations: torch.Tensor) -> torch.Tensor:
    masks = [goalkeeper_skill_mask_np(row.detach().cpu().numpy()) for row in observations.reshape(-1, observations.shape[-1])]
    return torch.as_tensor(np.asarray(masks, dtype=np.float32), dtype=observations.dtype, device=observations.device)


def goalkeeper_teacher_action(obs: BackendObservation) -> tuple[GKStructuredAction, GKCognition]:
    c = build_gk_cognition(obs)
    encoded = encode_goalkeeper_obs(obs, c)
    mask = goalkeeper_skill_mask_np(encoded)
    for skill in (
        GKSkillType.intercept_center,
        GKSkillType.intercept_lateral,
        GKSkillType.intercept_low_left,
        GKSkillType.intercept_low_right,
        GKSkillType.intercept_jump_left,
        GKSkillType.intercept_jump_right,
    ):
        if mask[int(skill)] > 0.0:
            return GKStructuredAction(skill, target_y=c.intersection_y), c

    if c.ball_controlled:
        if mask[int(GKSkillType.pass_)] > 0.0 and float(obs.pass_options_count) > 0.0:
            return GKStructuredAction(GKSkillType.pass_, pass_target=2), c
        if float(obs.ball_x) < PENALTY_X_LIMIT_MM or abs(float(obs.ball_y)) < 350.0:
            return GKStructuredAction(GKSkillType.clear), c
        side = 1.0 if float(obs.robot_y) >= 0.0 else -1.0
        target_x = min(-1800.0, float(obs.robot_x) + 900.0)
        target_y = side * 1400.0
        return GKStructuredAction(
            GKSkillType.dribble_out,
            target_x=target_x,
            target_y=target_y,
            target_theta=math.atan2(target_y - float(obs.robot_y), target_x - float(obs.robot_x)),
        ), c

    if c.stale_ball and bool(obs.global_team_ball_valid):
        return GKStructuredAction(GKSkillType.observe, target_x=float(obs.global_team_ball_x_mm), target_y=float(obs.global_team_ball_y_mm)), c

    if math.hypot(c.goalie_error_x, c.goalie_error_y) > 180.0 or abs(c.goalie_error_theta) > 0.20:
        return GKStructuredAction(GKSkillType.walk_goalie_pose, target_x=c.goalie_x, target_y=c.goalie_y, target_theta=c.goalie_theta), c

    return GKStructuredAction(GKSkillType.stand), c


def gk_action_to_current_bridge(action: GKStructuredAction, obs: BackendObservation | None = None) -> StructuredSkillAction:
    if action.skill is GKSkillType.walk_goalie_pose:
        return StructuredSkillAction(SkillType.walk, action.target_x, action.target_y, action.target_theta)
    if action.skill is GKSkillType.observe:
        return StructuredSkillAction(SkillType.observe, action.target_x, action.target_y, 0.0)
    if action.skill in {GKSkillType.intercept_lateral}:
        # Legacy fallback only. Production uses gk_action_to_native_bridge_name()
        # so B-Human executes exact keeper intercept/dive skills.
        target_x = float(obs.robot_x) if obs is not None else 0.0
        target_y = float(obs.robot_y) + float(action.target_y) if obs is not None else 0.0
        return StructuredSkillAction(SkillType.walk, target_x, target_y, 0.0)
    if action.skill.name.startswith("intercept_"):
        return StructuredSkillAction(SkillType.stand)
    if action.skill is GKSkillType.clear:
        return StructuredSkillAction(SkillType.shoot)
    if action.skill is GKSkillType.pass_:
        return StructuredSkillAction(SkillType.pass_, pass_target=action.pass_target)
    if action.skill is GKSkillType.dribble_out:
        return StructuredSkillAction(SkillType.dribble, action.target_x, action.target_y, action.target_theta)
    return StructuredSkillAction(SkillType.stand)


def gk_action_to_native_bridge_name(action: GKStructuredAction) -> str:
    if action.skill is GKSkillType.intercept_center:
        return "gk_center"
    if action.skill is GKSkillType.intercept_lateral:
        return "gk_lat"
    if action.skill is GKSkillType.intercept_low_left:
        return "gk_low_l"
    if action.skill is GKSkillType.intercept_low_right:
        return "gk_low_r"
    if action.skill is GKSkillType.intercept_jump_left:
        return "gk_jump_l"
    if action.skill is GKSkillType.intercept_jump_right:
        return "gk_jump_r"
    return gk_action_to_current_bridge(action).skill_name
