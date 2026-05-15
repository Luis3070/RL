from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

from config import GOAL_X, GOAL_Y_HALF
from observation import ObsIndex
from pybh_backend import BackendObservation

EPS = 1e-6
ACTIVE_SKILL_NAMES = ("walk", "dribble", "shoot")


@dataclass(frozen=True)
class RewardWeights:
    progress_ball: float = 0.0
    align_ball: float = 0.0
    enter_takeover: float = 0.0
    hold_takeover: float = 0.0
    good_walk: float = 0.0
    bad_walk: float = 0.0
    good_dribble: float = 0.0
    bad_dribble: float = 0.0
    good_shoot: float = 0.0
    bad_shoot: float = 0.0
    projection_penalty: float = 0.0
    skill_switch: float = 0.0
    thrash: float = 0.0
    goal: float = 0.0
    timeout: float = 0.0
    collision: float = 0.0
    clearance_gain: float = 0.0
    corridor_keep: float = 0.0
    ball_progress_post_takeover: float = 0.0
    first_to_ball: float = 0.0
    bad_duel: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "progress_ball": self.progress_ball,
            "align_ball": self.align_ball,
            "enter_takeover": self.enter_takeover,
            "hold_takeover": self.hold_takeover,
            "good_walk": self.good_walk,
            "bad_walk": self.bad_walk,
            "good_dribble": self.good_dribble,
            "bad_dribble": self.bad_dribble,
            "good_shoot": self.good_shoot,
            "bad_shoot": self.bad_shoot,
            "projection_penalty": self.projection_penalty,
            "skill_switch": self.skill_switch,
            "thrash": self.thrash,
            "goal": self.goal,
            "timeout": self.timeout,
            "collision": self.collision,
            "clearance_gain": self.clearance_gain,
            "corridor_keep": self.corridor_keep,
            "ball_progress_post_takeover": self.ball_progress_post_takeover,
            "first_to_ball": self.first_to_ball,
            "bad_duel": self.bad_duel,
        }


@dataclass(frozen=True)
class TakeoverFeatures:
    d_ball_mm: float
    a_ball_rad: float
    t_seen_ms: float
    seen_pct: float
    score_now: bool
    shot_open: float
    opp_front_mm: float
    front_obstacle_mm: float
    front_obstacle_count: int


@dataclass(frozen=True)
class StepContext:
    stage: int
    episode_step: int
    max_episode_steps: int
    prev_obs: BackendObservation | None
    obs: BackendObservation
    skill_name: str
    policy_skill_name: str
    projection_applied: bool
    prev_skill_name: str | None
    skill_history: tuple[str, ...] = ()
    goal_scored: bool = False
    timeout: bool = False
    collision: bool = False
    corridor_open: bool = False
    first_to_ball: bool = False
    bad_duel: bool = False


@dataclass(frozen=True)
class StepEvaluation:
    features: TakeoverFeatures
    prev_features: TakeoverFeatures
    takeover_envelope: bool
    prev_takeover_envelope: bool
    walk_zone: bool
    dribble_zone: bool
    shoot_zone: bool
    ball_confident: bool
    preferred_skill_name: str
    heuristic_bias: Dict[str, float]
    terms: Dict[str, float]
    weighted_terms: Dict[str, float]
    total_reward: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "takeover_envelope": self.takeover_envelope,
            "prev_takeover_envelope": self.prev_takeover_envelope,
            "walk_zone": self.walk_zone,
            "dribble_zone": self.dribble_zone,
            "shoot_zone": self.shoot_zone,
            "ball_confident": self.ball_confident,
            "preferred_skill_name": self.preferred_skill_name,
            "heuristic_bias": dict(self.heuristic_bias),
            "reward_terms": dict(self.terms),
            "reward_weighted_terms": dict(self.weighted_terms),
            "reward_total": self.total_reward,
        }


def clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def goal_scored(ball_x: float, ball_y: float, *, goal_x: float = GOAL_X, goal_y_half: float = GOAL_Y_HALF) -> bool:
    return ball_x >= goal_x and abs(ball_y) <= goal_y_half


def ball_distance(obs: np.ndarray) -> float:
    dx = float(obs[ObsIndex.ball_rel_x])
    dy = float(obs[ObsIndex.ball_rel_y])
    return math.hypot(dx, dy)


def ball_distance_mm(obs: BackendObservation) -> float:
    return math.hypot(float(obs.ball_rel_x), float(obs.ball_rel_y))


def ball_angle_rad(obs: BackendObservation) -> float:
    return math.atan2(float(obs.ball_rel_y), float(obs.ball_rel_x))


def goal_ball_distance_mm(obs: BackendObservation) -> float:
    return math.hypot(GOAL_X - float(obs.ball_x), float(obs.ball_y))


def features_from_obs(obs: BackendObservation) -> TakeoverFeatures:
    return TakeoverFeatures(
        d_ball_mm=ball_distance_mm(obs),
        a_ball_rad=ball_angle_rad(obs),
        t_seen_ms=float(obs.time_since_ball_seen),
        seen_pct=clip(float(obs.ball_seen_percentage) / 100.0, 0.0, 1.0),
        score_now=bool(obs.can_score_now),
        shot_open=clip(float(obs.shot_opening_with_obstacles), 0.0, 1.0),
        opp_front_mm=float(obs.nearest_opponent_front_dist),
        front_obstacle_mm=float(obs.nearest_front_obstacle_dist),
        front_obstacle_count=int(obs.obstacle_front_count),
    )


def is_takeover_envelope(features: TakeoverFeatures) -> bool:
    return (
        features.d_ball_mm <= 450.0
        and abs(features.a_ball_rad) <= math.radians(35.0)
        and features.t_seen_ms <= 400.0
        and features.seen_pct >= 0.4
    )


def is_walk_zone(features: TakeoverFeatures) -> bool:
    return (
        not is_takeover_envelope(features)
        and features.t_seen_ms <= 1000.0
        and features.seen_pct >= 0.2
    )


def is_shoot_zone(features: TakeoverFeatures) -> bool:
    return (
        features.d_ball_mm <= 320.0
        and abs(features.a_ball_rad) <= math.radians(20.0)
        and features.t_seen_ms <= 300.0
        and features.seen_pct >= 0.5
        and (features.shot_open >= 0.7 or features.score_now)
    )


def is_dribble_zone(features: TakeoverFeatures) -> bool:
    return is_takeover_envelope(features) and not is_shoot_zone(features)


def preferred_skill(features: TakeoverFeatures) -> str:
    if is_shoot_zone(features):
        return "shoot"
    if is_dribble_zone(features):
        return "dribble"
    return "walk"


def pattern_like(skill_history: Sequence[str], patterns: Sequence[Sequence[str]]) -> bool:
    history = tuple(skill_history)
    for pattern in patterns:
        if len(history) >= len(pattern) and history[-len(pattern) :] == tuple(pattern):
            return True
    return False


def is_local_thrash(skill_history: Sequence[str]) -> bool:
    return pattern_like(
        skill_history,
        (
            ("walk", "shoot", "walk"),
            ("walk", "dribble", "walk"),
            ("shoot", "walk", "shoot"),
            ("dribble", "walk", "dribble"),
        ),
    )


def heuristic_bias(step_ctx: StepContext) -> Dict[str, float]:
    features = features_from_obs(step_ctx.obs)
    skill_bias = {name: 0.0 for name in ACTIVE_SKILL_NAMES}

    if not is_takeover_envelope(features):
        skill_bias["walk"] += 1.0
    if abs(features.a_ball_rad) > math.radians(20.0):
        skill_bias["walk"] += 0.5

    if is_shoot_zone(features):
        skill_bias["shoot"] += 1.2
    else:
        skill_bias["shoot"] -= 0.8

    if is_dribble_zone(features):
        skill_bias["dribble"] += 1.0

    if features.opp_front_mm < 900.0 and is_dribble_zone(features):
        skill_bias["dribble"] += 0.5

    if features.d_ball_mm > 450.0:
        skill_bias["dribble"] -= 0.7
    if features.d_ball_mm > 320.0:
        skill_bias["shoot"] -= 1.0

    if step_ctx.prev_skill_name == "shoot" and not is_shoot_zone(features):
        skill_bias["shoot"] -= 0.5
    if step_ctx.prev_skill_name == "dribble" and features.d_ball_mm > 500.0:
        skill_bias["dribble"] -= 0.4

    return skill_bias


def build_fallback_action_targets(skill_name: str, obs: BackendObservation) -> Dict[str, float]:
    ball_heading = ball_angle_rad(obs)
    goal_heading = math.atan2(-float(obs.ball_y), GOAL_X - float(obs.ball_x))
    if skill_name == "walk":
        return {
            "target_x": float(obs.ball_x),
            "target_y": float(obs.ball_y),
            "target_theta": ball_heading,
        }
    if skill_name == "dribble":
        return {
            "target_x": float(obs.ball_x),
            "target_y": float(obs.ball_y),
            "target_theta": goal_heading,
        }
    return {
        "target_x": 0.0,
        "target_y": 0.0,
        "target_theta": goal_heading,
    }


def is_ball_confident(step_ctx: StepContext) -> bool:
    obs = step_ctx.obs
    return (
        not bool(obs.synthetic_ball_observation)
        and not bool(obs.ball_observation_stale)
        and not bool(obs.ball_observation_collapsed)
        and bool(obs.ball_export_fresh)
    )


def evaluate_step(step_ctx: StepContext, weights: RewardWeights) -> StepEvaluation:
    prev_obs = step_ctx.prev_obs or step_ctx.obs
    prev_features = features_from_obs(prev_obs)
    cur_features = features_from_obs(step_ctx.obs)

    takeover_envelope = is_takeover_envelope(cur_features)
    prev_takeover_envelope = is_takeover_envelope(prev_features)
    walk_zone = is_walk_zone(cur_features)
    shoot_zone = is_shoot_zone(cur_features)
    dribble_zone = is_dribble_zone(cur_features)
    ball_confident = is_ball_confident(step_ctx)

    progress_ball = clip((prev_features.d_ball_mm - cur_features.d_ball_mm) / 250.0, -1.0, 1.0)
    align_ball = clip((abs(prev_features.a_ball_rad) - abs(cur_features.a_ball_rad)) / math.radians(15.0), -1.0, 1.0)
    enter_takeover = 1.0 if ball_confident and takeover_envelope and not prev_takeover_envelope else 0.0
    hold_takeover = 1.0 if ball_confident and takeover_envelope else 0.0

    good_walk = (
        1.0
        if (
            step_ctx.skill_name == "walk"
            and not step_ctx.projection_applied
            and walk_zone
            and (progress_ball > 0.0 or align_ball > 0.0)
        )
        else 0.0
    )
    bad_walk = (
        1.0
        if step_ctx.skill_name == "walk" and not step_ctx.projection_applied and (shoot_zone or dribble_zone)
        else 0.0
    )

    good_dribble = (
        1.0
        if step_ctx.skill_name == "dribble" and not step_ctx.projection_applied and ball_confident and dribble_zone
        else 0.0
    )
    bad_dribble = 1.0 if step_ctx.skill_name == "dribble" and not dribble_zone else 0.0

    good_shoot = 1.0 if step_ctx.skill_name == "shoot" and ball_confident and shoot_zone else 0.0
    bad_shoot = 1.0 if step_ctx.skill_name == "shoot" and not shoot_zone else 0.0
    projection_penalty = 1.0 if step_ctx.projection_applied else 0.0

    skill_switch = 1.0 if step_ctx.prev_skill_name is not None and step_ctx.skill_name != step_ctx.prev_skill_name else 0.0
    thrash = 1.0 if is_local_thrash(step_ctx.skill_history) else 0.0
    goal_reward = 1.0 if step_ctx.goal_scored else 0.0
    timeout_penalty = 1.0 if step_ctx.timeout else 0.0
    collision_penalty = 1.0 if step_ctx.collision else 0.0

    clearance_gain = 0.0
    if step_ctx.stage >= 2 and prev_features.front_obstacle_mm > 0.0 and cur_features.front_obstacle_mm > 0.0:
        clearance_gain = clip((cur_features.front_obstacle_mm - prev_features.front_obstacle_mm) / 250.0, -1.0, 1.0)

    corridor_keep = 1.0 if step_ctx.stage >= 3 and step_ctx.corridor_open else 0.0

    ball_progress_post_takeover = 0.0
    if ball_confident and takeover_envelope:
        prev_goal_dist = goal_ball_distance_mm(prev_obs)
        cur_goal_dist = goal_ball_distance_mm(step_ctx.obs)
        ball_progress_post_takeover = clip((prev_goal_dist - cur_goal_dist) / 300.0, -1.0, 1.0)

    first_to_ball_reward = 1.0 if step_ctx.stage >= 4 and step_ctx.first_to_ball else 0.0
    bad_duel_penalty = 1.0 if step_ctx.stage >= 4 and step_ctx.bad_duel else 0.0

    terms = {
        "progress_ball": progress_ball,
        "align_ball": align_ball,
        "enter_takeover": enter_takeover,
        "hold_takeover": hold_takeover,
        "good_walk": good_walk,
        "bad_walk": bad_walk,
        "good_dribble": good_dribble,
        "bad_dribble": bad_dribble,
        "good_shoot": good_shoot,
        "bad_shoot": bad_shoot,
        "projection_penalty": projection_penalty,
        "skill_switch": skill_switch,
        "thrash": thrash,
        "goal": goal_reward,
        "timeout": timeout_penalty,
        "collision": collision_penalty,
        "clearance_gain": clearance_gain,
        "corridor_keep": corridor_keep,
        "ball_progress_post_takeover": ball_progress_post_takeover,
        "first_to_ball": first_to_ball_reward,
        "bad_duel": bad_duel_penalty,
    }
    weight_map = weights.as_dict()
    weighted_terms = {name: float(weight_map.get(name, 0.0) * value) for name, value in terms.items()}
    total_reward = float(sum(weighted_terms.values()))

    return StepEvaluation(
        features=cur_features,
        prev_features=prev_features,
        takeover_envelope=takeover_envelope,
        prev_takeover_envelope=prev_takeover_envelope,
        walk_zone=walk_zone,
        dribble_zone=dribble_zone,
        shoot_zone=shoot_zone,
        ball_confident=ball_confident,
        preferred_skill_name=preferred_skill(cur_features),
        heuristic_bias=heuristic_bias(step_ctx),
        terms=terms,
        weighted_terms=weighted_terms,
        total_reward=total_reward,
    )


def compute_progress_reward(
    *,
    previous_ball_distance: float,
    current_ball_distance: float,
    goal_scored: bool,
    progress_scale: float,
    goal_bonus: float,
    step_penalty: float,
) -> StepEvaluation:
    del current_ball_distance, goal_scored, progress_scale, goal_bonus, step_penalty
    raise RuntimeError("compute_progress_reward is deprecated. Use evaluate_step(...) on raw BackendObservation data")
