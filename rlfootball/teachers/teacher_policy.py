from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np

from rlfootball.core.action import SKILL_ORDER, SkillType, StructuredSkillAction
from rlfootball.core.config import ACTION
from rlfootball.curriculum.curriculum import get_stage_spec
from rlfootball.core.gate import GateDecision, SkillGate
from rlfootball.core.pybh_backend import BackendObservation
from rlfootball.core.reward import (
    BALL_TRUSTED_MAX_AGE_MS,
    build_fallback_action_targets,
    features_from_obs,
)

BALL_SEARCH_MAX_AGE_MS = 1000.0
OBSTACLE_FRONT_CLOSE_MM = 900.0
OBSTACLE_LATERAL_SHIFT_MM = 420.0
TEACHER_ROBUST_TAKEOVER_MM = 560.0
TEACHER_ROBUST_TAKEOVER_ANGLE_RAD = math.radians(45.0)
# Natural-perception "ball-blind" threshold: beyond this the robot has not seen the
# ball on the honest (natural) channel within the gate's engagement grace, so it
# should actively re-acquire rather than blind-walk (walkTo's head does not reliably
# keep a ground ball in view). Aligned with GateConfig.dribble_grace_enter_age_ms so
# the teacher hands a fresh-enough sighting to the gate before it tries to engage.
NATURAL_BLIND_MAX_AGE_MS = 1500.0
BALL_SCAN_RANGE_MM = 2500.0  # NAO camera ball-detect range


def obs_from_raw_mapping(raw_obs: Mapping[str, object]) -> BackendObservation:
    return BackendObservation(**dict(raw_obs))


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _front_obstacle_distance_from_policy_obs(obs: BackendObservation) -> float:
    candidates = [
        float(obs.nearest_teammate_front_dist),
        float(obs.nearest_opponent_front_dist),
        float(obs.nearest_uncertain_front_dist),
    ]
    positive = [value for value in candidates if value > 0.0]
    return min(positive) if positive else 0.0


def _lateral_escape_sign(obs: BackendObservation) -> float:
    if abs(float(obs.ball_rel_y)) > 60.0:
        return -1.0 if float(obs.ball_rel_y) > 0.0 else 1.0
    if abs(float(obs.robot_y)) > 60.0:
        return -1.0 if float(obs.robot_y) > 0.0 else 1.0
    return 1.0


def _action_from_anchor(skill_name: str, obs: BackendObservation, *, stage_id: int) -> StructuredSkillAction:
    targets = dict(build_fallback_action_targets(skill_name, obs))
    front_obstacle_mm = _front_obstacle_distance_from_policy_obs(obs)
    if stage_id >= 2 and skill_name in {"walk", "dribble"} and 0.0 < front_obstacle_mm < OBSTACLE_FRONT_CLOSE_MM:
        sign = _lateral_escape_sign(obs)
        severity = float(np.clip((OBSTACLE_FRONT_CLOSE_MM - front_obstacle_mm) / OBSTACLE_FRONT_CLOSE_MM, 0.0, 1.0))
        targets["target_y"] = float(
            np.clip(
                float(targets["target_y"]) + sign * OBSTACLE_LATERAL_SHIFT_MM * (0.35 + 0.65 * severity),
                -ACTION.y_range,
                ACTION.y_range,
            )
        )
        if skill_name == "walk":
            targets["target_theta"] = math.atan2(
                float(obs.ball_y) - float(obs.robot_y),
                float(obs.ball_x) - float(obs.robot_x),
            )
    return StructuredSkillAction(
        skill_type=SkillType.from_public_name(skill_name),
        target_x=float(targets["target_x"]),
        target_y=float(targets["target_y"]),
        target_theta=float(targets["target_theta"]),
    )


def stage_aware_teacher_action(
    obs: BackendObservation,
    stage_id: int,
    prev_skill_name: str | None = None,
    *,
    gate_decision: GateDecision | None = None,
) -> StructuredSkillAction:
    stage = get_stage_spec(stage_id)
    active = set(stage.active_skill_names)
    features = features_from_obs(obs)
    nat_age_ms = float(getattr(obs, "natural_time_since_ball_seen", 0.0))

    # Thin cold-start fallback: when ball-blind on the NATURAL channel beyond the
    # engagement grace, actively scan with `observe` to acquire a real sighting (if
    # the ball is within camera range) instead of blind-walking; otherwise close
    # distance so it comes into view. `observe` degrades to walk when it is not an
    # active skill for the stage, so this is inert until observe is enabled.
    if nat_age_ms > NATURAL_BLIND_MAX_AGE_MS:
        if "observe" in active and features.d_ball_mm < BALL_SCAN_RANGE_MM:
            return StructuredSkillAction(skill_type=SkillType.observe)
        if "walk" in active:
            return _action_from_anchor("walk", obs, stage_id=stage_id)
        return StructuredSkillAction(skill_type=SkillType.stand)

    if features.t_seen_ms > BALL_SEARCH_MAX_AGE_MS:
        if "walk" in active:
            return _action_from_anchor("walk", obs, stage_id=stage_id)
        return StructuredSkillAction(skill_type=SkillType.stand)

    # Skill selection flows through the unified SkillGate — the same forbid-invalid
    # gate the RL policy masks on and the env bakes into the observation. When the
    # teacher runs inside the env (dataset/eval), the caller passes the env's own
    # decision so the recorded obs's armed flags exactly permit the recorded label;
    # otherwise we fall back to a single-step decision (no hysteresis latch). The
    # drive-commit latch is now the gate's dribble enter/exit hysteresis.
    if gate_decision is None:
        gate_decision = SkillGate().step(obs)
    shoot_ready = gate_decision.shoot_armed
    takeover = gate_decision.dribble_armed

    # stage>=3 robust takeover: commit on geometry alone across brief perception dips.
    if (
        stage_id >= 3
        and features.d_ball_mm <= TEACHER_ROBUST_TAKEOVER_MM
        and abs(features.a_ball_rad) <= TEACHER_ROBUST_TAKEOVER_ANGLE_RAD
        and features.t_seen_ms <= BALL_TRUSTED_MAX_AGE_MS
    ):
        takeover = True

    if stage_id < 0:
        skill_name = "walk"
    elif shoot_ready and "shoot" in active:
        skill_name = "shoot"
    elif takeover and "dribble" in active:
        skill_name = "dribble"
    else:
        skill_name = "walk"

    if skill_name not in active:
        skill_name = "walk" if "walk" in active else "stand"

    if skill_name == "stand":
        return StructuredSkillAction(skill_type=SkillType.stand)
    if skill_name == "shoot":
        return StructuredSkillAction(skill_type=SkillType.shoot)
    return _action_from_anchor(skill_name, obs, stage_id=stage_id)


def absolute_to_residual_action(action: StructuredSkillAction, obs: BackendObservation) -> StructuredSkillAction:
    if action.skill_name not in {"walk", "dribble"}:
        return StructuredSkillAction(
            skill_type=action.skill_type,
            target_x=0.0,
            target_y=0.0,
            target_theta=0.0,
            pass_target=action.pass_target,
        )
    anchors = build_fallback_action_targets(action.skill_name, obs)
    residual_x = 0.0
    if abs(float(ACTION.residual_x_range)) > 1e-6:
        residual_x = float(
            np.clip((float(action.target_x) - float(anchors["target_x"])) / ACTION.residual_x_range, -1.0, 1.0)
        )
    return StructuredSkillAction(
        skill_type=action.skill_type,
        target_x=residual_x,
        target_y=float(np.clip((float(action.target_y) - float(anchors["target_y"])) / ACTION.residual_y_range, -1.0, 1.0)),
        target_theta=float(
            np.clip(_wrap_angle(float(action.target_theta) - float(anchors["target_theta"])) / ACTION.residual_theta_range, -1.0, 1.0)
        ),
        pass_target=action.pass_target,
    )


def teacher_policy_payload(
    obs: BackendObservation,
    stage_id: int,
    prev_skill_name: str | None = None,
    *,
    gate_decision: GateDecision | None = None,
) -> tuple[dict[str, float | int], StructuredSkillAction, StructuredSkillAction]:
    absolute_action = stage_aware_teacher_action(obs, stage_id, prev_skill_name, gate_decision=gate_decision)
    residual_action = absolute_to_residual_action(absolute_action, obs)
    payload = residual_action.as_dict()
    payload["policy_skill_raw"] = float(residual_action.skill_type)
    for skill_name in SKILL_ORDER:
        payload[f"policy_skill_prob_{skill_name}"] = 1.0 if skill_name == residual_action.skill_name else 0.0
    return payload, absolute_action, residual_action


def raw_obs_dict(obs: BackendObservation) -> dict[str, object]:
    return asdict(obs)
