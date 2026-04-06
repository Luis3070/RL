from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional

import numpy as np

from config import CONTRACT, FIELD


@dataclass(frozen=True)
class LaunchTemplateSpec:
    name: str
    family: str
    anchor_backoff_mm: float
    ball_rel_x_mm: float
    ball_rel_y_mm: float
    heading_bias_deg: float
    teammate_depth_mm: float = 700.0
    teammate_spread_mm: float = 650.0
    opponent_depth_mm: float = 1500.0
    opponent_spread_mm: float = 1700.0


@dataclass(frozen=True)
class ResolvedLaunchTemplate:
    name: str
    family: str
    agent_pos: np.ndarray
    ball_pos: np.ndarray
    agent_theta: float
    teammate_offsets: List[np.ndarray]
    opponent_offsets: List[np.ndarray]


def _goal_center() -> np.ndarray:
    return np.array([FIELD.x_opponent_ground, 0.0], dtype=np.float32)


def _stage_anchor(stage_id: int) -> np.ndarray:
    x_span = float(FIELD.x_opponent_ground - FIELD.x_own_ground)
    if stage_id == 0:
        x = float(FIELD.x_own_ground + 0.42 * x_span)
    elif stage_id == 1:
        x = float(FIELD.x_own_ground + 0.56 * x_span)
    elif stage_id == 2:
        x = float(FIELD.x_own_ground + 0.68 * x_span)
    else:
        x = float(FIELD.x_own_ground + 0.74 * x_span)
    return np.array([x, 0.0], dtype=np.float32)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _rotate_local(offset: np.ndarray, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            offset[0] * c - offset[1] * s,
            offset[0] * s + offset[1] * c,
        ],
        dtype=np.float32,
    )


def _translate_inside_field(agent_pos: np.ndarray,
                            ball_pos: np.ndarray,
                            margin_x: float = 260.0,
                            margin_y: float = 240.0) -> tuple[np.ndarray, np.ndarray]:
    x_low = float(FIELD.x_own_ground + margin_x)
    x_high = float(FIELD.x_opponent_ground - margin_x)
    y_low = float(FIELD.y_right_sideline + margin_y)
    y_high = float(FIELD.y_left_sideline - margin_y)

    shift = np.zeros(2, dtype=np.float32)
    min_x = min(float(agent_pos[0]), float(ball_pos[0]))
    max_x = max(float(agent_pos[0]), float(ball_pos[0]))
    min_y = min(float(agent_pos[1]), float(ball_pos[1]))
    max_y = max(float(agent_pos[1]), float(ball_pos[1]))

    if min_x < x_low:
        shift[0] += x_low - min_x
    elif max_x > x_high:
        shift[0] -= max_x - x_high

    if min_y < y_low:
        shift[1] += y_low - min_y
    elif max_y > y_high:
        shift[1] -= max_y - y_high

    return (agent_pos + shift).astype(np.float32), (ball_pos + shift).astype(np.float32)


def _build_stage_specs(stage_id: int) -> List[LaunchTemplateSpec]:
    handoff = CONTRACT.handoff
    x_span = float(handoff.ball_rel_x_max - handoff.ball_rel_x_min)
    y_max = float(handoff.ball_rel_y_abs_max)

    center_x = float(handoff.ball_rel_x_min + 0.38 * x_span)
    deep_x = float(handoff.ball_rel_x_min + 0.62 * x_span)
    shield_x = float(handoff.ball_rel_x_min + 0.48 * x_span)
    inner_y = 0.22 * y_max
    outer_y = 0.42 * y_max

    depth_scale = 1.0 if stage_id <= 1 else 1.12
    spread_scale = 1.0 if stage_id <= 1 else 1.08

    return [
        LaunchTemplateSpec(
            name="corridor_center",
            family="deterministic_handoff",
            anchor_backoff_mm=90.0,
            ball_rel_x_mm=center_x,
            ball_rel_y_mm=0.0,
            heading_bias_deg=0.0,
            teammate_depth_mm=720.0 * depth_scale,
            teammate_spread_mm=620.0 * spread_scale,
        ),
        LaunchTemplateSpec(
            name="corridor_left",
            family="deterministic_handoff",
            anchor_backoff_mm=120.0,
            ball_rel_x_mm=shield_x,
            ball_rel_y_mm=inner_y,
            heading_bias_deg=6.0,
            teammate_depth_mm=760.0 * depth_scale,
            teammate_spread_mm=700.0 * spread_scale,
        ),
        LaunchTemplateSpec(
            name="corridor_right",
            family="deterministic_handoff",
            anchor_backoff_mm=120.0,
            ball_rel_x_mm=shield_x,
            ball_rel_y_mm=-inner_y,
            heading_bias_deg=-6.0,
            teammate_depth_mm=760.0 * depth_scale,
            teammate_spread_mm=700.0 * spread_scale,
        ),
        LaunchTemplateSpec(
            name="carry_left",
            family="deterministic_handoff",
            anchor_backoff_mm=155.0,
            ball_rel_x_mm=deep_x,
            ball_rel_y_mm=outer_y,
            heading_bias_deg=12.0,
            teammate_depth_mm=860.0 * depth_scale,
            teammate_spread_mm=780.0 * spread_scale,
        ),
        LaunchTemplateSpec(
            name="carry_right",
            family="deterministic_handoff",
            anchor_backoff_mm=155.0,
            ball_rel_x_mm=deep_x,
            ball_rel_y_mm=-outer_y,
            heading_bias_deg=-12.0,
            teammate_depth_mm=860.0 * depth_scale,
            teammate_spread_mm=780.0 * spread_scale,
        ),
        LaunchTemplateSpec(
            name="deep_control",
            family="deterministic_handoff",
            anchor_backoff_mm=185.0,
            ball_rel_x_mm=deep_x,
            ball_rel_y_mm=0.0,
            heading_bias_deg=0.0,
            teammate_depth_mm=940.0 * depth_scale,
            teammate_spread_mm=640.0 * spread_scale,
        ),
    ]


def get_stage_template_names(stage_id: int) -> List[str]:
    names = [spec.name for spec in _build_stage_specs(stage_id)]
    if stage_id == 0:
        # Priorizamos las únicas familias que ya mostraron takeover real.
        return [
            "carry_left",
            "carry_right",
        ]
    if stage_id == 1:
        return [
            "carry_left",
            "carry_right",
            "deep_control",
            "corridor_left",
            "corridor_right",
            "corridor_center",
        ]
    return names


def resolve_launch_template(stage_id: int,
                            template_name: str,
                            approach_target: Optional[np.ndarray] = None,
                            kick_target: Optional[np.ndarray] = None) -> ResolvedLaunchTemplate:
    spec_map = {spec.name: spec for spec in _build_stage_specs(stage_id)}
    if template_name not in spec_map:
        known = ", ".join(sorted(spec_map))
        raise KeyError(f"Unknown launch template '{template_name}' for stage {stage_id}. Known: {known}")

    spec = spec_map[template_name]
    anchor = (
        np.asarray(approach_target, dtype=np.float32)
        if approach_target is not None
        else _stage_anchor(stage_id)
    )
    target_hint = (
        np.asarray(kick_target, dtype=np.float32)
        if kick_target is not None
        else _goal_center()
    )

    forward = _normalize(target_hint - anchor)
    ball_pos = (anchor - forward * float(spec.anchor_backoff_mm)).astype(np.float32)

    theta = float(
        math.atan2(float(target_hint[1] - ball_pos[1]), float(target_hint[0] - ball_pos[0]))
        + math.radians(spec.heading_bias_deg)
    )
    heading = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    lateral = np.array([-heading[1], heading[0]], dtype=np.float32)

    agent_pos = (
        ball_pos
        - heading * float(spec.ball_rel_x_mm)
        - lateral * float(spec.ball_rel_y_mm)
    ).astype(np.float32)
    agent_pos, ball_pos = _translate_inside_field(agent_pos, ball_pos)

    teammate_offsets = [
        np.array([-spec.teammate_depth_mm, 0.0], dtype=np.float32),
        np.array([-spec.teammate_depth_mm * 0.95, spec.teammate_spread_mm], dtype=np.float32),
        np.array([-spec.teammate_depth_mm * 0.95, -spec.teammate_spread_mm], dtype=np.float32),
    ]
    opponent_offsets = [
        np.array([spec.opponent_depth_mm, spec.opponent_spread_mm], dtype=np.float32),
        np.array([spec.opponent_depth_mm, -spec.opponent_spread_mm], dtype=np.float32),
        np.array([spec.opponent_depth_mm * 1.2, 0.0], dtype=np.float32),
        np.array([spec.opponent_depth_mm * 1.45, spec.opponent_spread_mm * 0.55], dtype=np.float32),
    ]

    return ResolvedLaunchTemplate(
        name=spec.name,
        family=spec.family,
        agent_pos=agent_pos,
        ball_pos=ball_pos,
        agent_theta=theta,
        teammate_offsets=teammate_offsets,
        opponent_offsets=opponent_offsets,
    )
