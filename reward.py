"""
reward.py — Reward functions alineadas con contrato BHuman.

Filosofía:
  - Reward centrada en CALIDAD DE HANDOFF y RESULTADO DE TAKEOVER
  - NO premia "tocar el balón" directamente
  - NO usa física local inventada como fuente de verdad
  - Approach reward guía al agente a posición de handoff
  - Takeover reward premia que BHuman ejecute exitosamente
  - Support reward premia posicionamiento cooperativo

Estructura:
  R = w_approach * progress_to_takeover_pose
    + w_takeover * (handoff_ready_bonus + bhuman_takeover_success)
    + w_support * spacing_and_lanes
    + sparse (goal ±10)
    - collisions - out_of_bounds - oscillation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import FIELD, AGENT, CONTRACT


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentRewardInfo:
    """Per-agent state needed for reward computation."""
    pos:               np.ndarray
    theta:             float
    is_upright:        bool
    role:              str
    prev_ball_dist:    float = 0.0
    prev_theta:        float = 0.0
    ball_in_fov:       bool  = False
    prev_omega:        float = 0.0
    is_active:         bool  = False


@dataclass
class TeamRewardInfo:
    """Team state for shared reward computation."""
    agents:            List[AgentRewardInfo]
    ball_pos:          np.ndarray
    ball_vel:          np.ndarray
    prev_ball_pos:     np.ndarray
    prev_ball_speed:   float
    own_score:         int
    opp_score:         int
    goal_scored:       bool
    goal_conceded:     bool
    opp_positions:     List[np.ndarray]
    ball_out_of_bounds: bool = False
    # BHuman contract fields
    handoff_ready:     bool = False
    takeover_active:   bool = False
    takeover_success:  Optional[bool] = None   # None = no event, True/False = result
    takeover_kick:     bool = False
    macro_duration_frames: int = 0
    # Backend provenance: "real", "contract_validation", or "python_stub".
    # When not "real", takeover_success is contractual validation only —
    # NOT real BHuman execution evidence. Reward must be gated accordingly.
    sidecar_backend:   str  = "unknown"


@dataclass
class RewardComponents:
    """Breakdown of reward for logging and diagnostics."""
    sparse:              float = 0.0   # ±10 per goal
    approach_ball:       float = 0.0   # delta approach to ball
    handoff_alignment:   float = 0.0   # continuous shaping toward handoff envelope
    handoff_ready_bonus: float = 0.0   # bonus for being in handoff envelope
    takeover_success:    float = 0.0   # bonus for successful BHuman execution
    takeover_failure:    float = 0.0   # penalty for failed takeover
    support_spacing:     float = 0.0   # cooperative spacing quality
    anti_blocking:       float = 0.0   # penalty for blocking teammate lines
    out_of_bounds:       float = 0.0   # penalty for leaving field
    fall_penalty:        float = 0.0   # penalty for falling
    oscillation_penalty: float = 0.0   # penalty for direction changes
    ball_out_penalty:    float = 0.0   # penalty for ball out of bounds
    fov_bonus:           float = 0.0   # orientation toward ball
    total:               float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SHAPING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _sparse_reward(goal_scored: bool, goal_conceded: bool) -> float:
    """Terminal reward. +10 own goal, -10 conceded."""
    return 10.0 * goal_scored - 10.0 * goal_conceded


def _approach_ball_reward(agents: List[AgentRewardInfo],
                          ball_pos: np.ndarray,
                          ball_speed: float = 0.0) -> float:
    """
    Delta reward for approaching the ball.

    Only the nearest upright agent contributes.
    Clamped to >=0 when ball moves faster than agent can walk.
    """
    upright = [a for a in agents if a.is_upright]
    if not upright:
        return 0.0

    curr_dists = [float(np.linalg.norm(a.pos - ball_pos)) for a in upright]
    nearest_idx = int(np.argmin(curr_dists))
    nearest = upright[nearest_idx]

    curr_dist = curr_dists[nearest_idx]
    prev_dist = nearest.prev_ball_dist

    delta = (prev_dist - curr_dist) / FIELD.diagonal
    raw = 130.0 * delta

    if ball_speed > AGENT.max_forward_speed and raw < 0.0:
        return 0.0

    return raw


def _handoff_ready_reward(handoff_ready: bool) -> float:
    """
    Bonus for being in the handoff envelope.

    This is the central shaped reward: RL should learn to position
    itself so that BHuman can take over cleanly.
    """
    if handoff_ready:
        return 2.0
    return 0.0


def _handoff_alignment_reward(agents: List[AgentRewardInfo],
                              ball_pos: np.ndarray) -> float:
    """
    Continuous shaping toward the real handoff envelope.

    Stage 0 is not just "get closer to the ball". The active agent must place
    the ball in the forward/lateral window and face it correctly so the real
    BHuman takeover can trigger. This reward uses the same geometry as the
    handoff contract, but stays dense outside the exact envelope.
    """
    active = next((a for a in agents if a.is_active and a.is_upright), None)
    if active is None:
        upright = [a for a in agents if a.is_upright]
        if not upright:
            return 0.0
        active = min(upright, key=lambda a: float(np.linalg.norm(a.pos - ball_pos)))

    dx = float(ball_pos[0] - active.pos[0])
    dy = float(ball_pos[1] - active.pos[1])

    cos_t = math.cos(-active.theta)
    sin_t = math.sin(-active.theta)
    rel_x = dx * cos_t - dy * sin_t
    rel_y = dx * sin_t + dy * cos_t

    angle_to_ball = math.degrees(math.atan2(dy, dx) - active.theta)
    while angle_to_ball > 180.0:
        angle_to_ball -= 360.0
    while angle_to_ball < -180.0:
        angle_to_ball += 360.0
    angle_abs = abs(angle_to_ball)

    handoff = CONTRACT.handoff
    x_low = float(handoff.ball_rel_x_min)
    x_high = float(handoff.ball_rel_x_max)
    x_span = max(1.0, x_high - x_low)

    if rel_x < x_low:
        x_score = max(0.0, 1.0 - (x_low - rel_x) / x_span)
    elif rel_x > x_high:
        x_score = max(0.0, 1.0 - (rel_x - x_high) / x_span)
    else:
        x_center = 0.5 * (x_low + x_high)
        x_half = max(1.0, 0.5 * x_span)
        x_score = max(0.0, 1.0 - abs(rel_x - x_center) / x_half)

    y_limit = max(1.0, float(handoff.ball_rel_y_abs_max))
    y_score = max(0.0, 1.0 - abs(rel_y) / y_limit)

    angle_limit = max(1.0, float(handoff.angle_to_ball_abs_max))
    angle_score = max(0.0, 1.0 - angle_abs / angle_limit)

    combined = 0.40 * x_score + 0.35 * y_score + 0.25 * angle_score
    return 0.6 * combined


def _takeover_result_reward(takeover_success: Optional[bool],
                            takeover_kick: bool,
                            sidecar_backend: str = "unknown") -> float:
    """
    Reward based on BHuman takeover outcome.

    When sidecar_backend == "real":
      +5.0 for successful BHuman execution (goal, pass, dribble advance)
      -1.0 for failed takeover (ball lost, timeout, etc.)
      +1.0 for kick executed during takeover (immediate feedback)

    When sidecar_backend != "real" (contract_validation, python_stub, etc.):
      Takeover reward is zeroed out. The takeover_success signal comes from
      contractual validation, NOT from real BHuman execution. Treating it
      as real reward would contaminate the training signal.
      Only handoff_ready_bonus survives (structural positioning quality).
    """
    if sidecar_backend != "real":
        return 0.0
    reward = 0.0
    if takeover_success is True:
        reward += 5.0
    elif takeover_success is False:
        reward -= 1.0
    if takeover_kick:
        reward += 1.0
    return reward


def _support_spacing_reward(agents: List[AgentRewardInfo]) -> float:
    """
    Reward for team spacing quality.

    Upright agents should maintain reasonable distances from each other.
    Too close = blocking. Too far = no support. Ideal: 800-2000mm.
    """
    upright = [a for a in agents if a.is_upright]
    if len(upright) < 2:
        return 0.0

    positions = np.array([a.pos for a in upright])
    dists = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dists.append(float(np.linalg.norm(positions[i] - positions[j])))

    if not dists:
        return 0.0

    mean_dist = float(np.mean(dists))
    # Ideal spacing: normalize to [0, 0.05]
    return 0.05 * min(mean_dist / FIELD.diagonal, 1.0)


def _anti_blocking_penalty(agents: List[AgentRewardInfo],
                           ball_pos: np.ndarray) -> float:
    """
    Penalty for non-active agents blocking the active agent's path to ball.

    If a support agent is between the active agent and the ball,
    it blocks the pass/dribble line.
    """
    active = [a for a in agents if a.is_active and a.is_upright]
    support = [a for a in agents if not a.is_active and a.is_upright]

    if not active or not support:
        return 0.0

    active_agent = active[0]
    # Direction from active to ball
    to_ball = ball_pos - active_agent.pos
    dist_to_ball = float(np.linalg.norm(to_ball))
    if dist_to_ball < 1e-6:
        return 0.0
    to_ball_norm = to_ball / dist_to_ball

    penalty = 0.0
    for s in support:
        to_support = s.pos - active_agent.pos
        # Project support position onto active→ball line
        proj = float(np.dot(to_support, to_ball_norm))
        if 0 < proj < dist_to_ball:
            # Perpendicular distance from line
            perp = float(np.linalg.norm(to_support - proj * to_ball_norm))
            if perp < AGENT.min_robot_distance * 2:  # 400mm
                penalty -= 0.1

    return penalty


def _out_of_bounds_penalty(agents: List[AgentRewardInfo]) -> float:
    penalty = 0.0
    for a in agents:
        x, y = a.pos
        if abs(x) > FIELD.x_opponent_ground or abs(y) > FIELD.y_left_sideline:
            penalty -= 0.5
    return penalty


def _fall_penalty(agents: List[AgentRewardInfo]) -> float:
    return sum(-0.1 for a in agents if not a.is_upright)


def _oscillation_penalty(agents: List[AgentRewardInfo]) -> float:
    """Penalty for aggressive angular oscillation."""
    total = 0.0
    for a in agents:
        if not a.is_upright:
            continue
        curr_delta = a.theta - a.prev_theta
        curr_delta = math.atan2(math.sin(curr_delta), math.cos(curr_delta))
        prev_delta = a.prev_omega * 0.1  # dt=0.1s
        prev_delta = math.atan2(math.sin(prev_delta), math.cos(prev_delta))
        jerk = abs(curr_delta - prev_delta)
        threshold = 0.8
        if jerk > threshold:
            total -= 0.2 * (jerk - threshold)
    return total


def _fov_bonus(agents: List[AgentRewardInfo]) -> float:
    """Small bonus for ball being in FOV (orientation quality)."""
    for a in agents:
        if a.is_upright and a.ball_in_fov:
            return 0.03
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REWARD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(info: TeamRewardInfo, shaping_weight: float,
                   approach_weight: float = 1.0,
                   takeover_weight: float = 0.0,
                   support_weight: float = 0.0) -> RewardComponents:
    """
    Compute team reward for one simulation step.

    Args:
        info:            Complete team state.
        shaping_weight:  Shaping weight [0.0, 1.0]. 0 = sparse only.
        approach_weight: Weight for approach-to-ball signal.
        takeover_weight: Weight for handoff/takeover signals.
        support_weight:  Weight for cooperative spacing.

    Returns:
        RewardComponents with breakdown and total.
    """
    components = RewardComponents()
    components.sparse = _sparse_reward(info.goal_scored, info.goal_conceded)

    if shaping_weight > 0.0:
        ball_speed = float(np.linalg.norm(info.ball_vel))

        # Approach to ball (for active agent)
        components.approach_ball = (
            _approach_ball_reward(info.agents, info.ball_pos, ball_speed)
            * approach_weight
        )

        components.handoff_alignment = (
            _handoff_alignment_reward(info.agents, info.ball_pos)
            * max(approach_weight, takeover_weight)
        )

        # Handoff ready bonus
        components.handoff_ready_bonus = (
            _handoff_ready_reward(info.handoff_ready)
            * takeover_weight
        )

        # Takeover result (gated by backend provenance)
        takeover_r = _takeover_result_reward(
            info.takeover_success, info.takeover_kick,
            sidecar_backend=info.sidecar_backend,
        )
        if takeover_r >= 0:
            components.takeover_success = takeover_r * takeover_weight
        else:
            components.takeover_failure = takeover_r * takeover_weight

        # Support spacing
        components.support_spacing = (
            _support_spacing_reward(info.agents) * support_weight
        )

        # Anti-blocking
        components.anti_blocking = (
            _anti_blocking_penalty(info.agents, info.ball_pos) * support_weight
        )

        # Standard penalties
        components.out_of_bounds = _out_of_bounds_penalty(info.agents)
        components.fall_penalty = _fall_penalty(info.agents)
        components.oscillation_penalty = _oscillation_penalty(info.agents)
        components.fov_bonus = _fov_bonus(info.agents)

        # Ball out of bounds
        if info.ball_out_of_bounds:
            components.ball_out_penalty = -3.0

    shaped = (
        components.approach_ball
        + components.handoff_alignment
        + components.handoff_ready_bonus
        + components.takeover_success
        + components.takeover_failure
        + components.support_spacing
        + components.anti_blocking
        + components.out_of_bounds
        + components.fall_penalty
        + components.oscillation_penalty
        + components.fov_bonus
        + components.ball_out_penalty
    )
    components.total = components.sparse + shaping_weight * shaped
    return components


def compute_per_agent_reward(
    team_components: RewardComponents,
    agent_info: AgentRewardInfo,
    ball_pos: np.ndarray,
    shaping_weight: float,
) -> float:
    """
    Adjust team reward with individual bonus.
    90% team (cooperative) + 10% individual (approach incentive).
    """
    team_reward = team_components.total

    if shaping_weight > 0.0 and agent_info.is_upright:
        curr_dist = float(np.linalg.norm(agent_info.pos - ball_pos))
        delta = (agent_info.prev_ball_dist - curr_dist) / FIELD.diagonal
        individual_bonus = 0.1 * shaping_weight * 130.0 * delta
    else:
        individual_bonus = 0.0

    return 0.9 * team_reward + individual_bonus


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRIC UTILITIES (for obs space)
# ─────────────────────────────────────────────────────────────────────────────

def goal_opening_angle(pos: np.ndarray) -> float:
    """Opening angle to opponent goal from pos (rad)."""
    goal_left = np.array([FIELD.x_opponent_ground, FIELD.y_left_goal])
    goal_right = np.array([FIELD.x_opponent_ground, FIELD.y_right_goal])
    angle_left = math.atan2(goal_left[1] - pos[1], goal_left[0] - pos[0])
    angle_right = math.atan2(goal_right[1] - pos[1], goal_right[0] - pos[0])
    return float(angle_left - angle_right)


def goal_relative_angle(pos: np.ndarray, theta: float) -> float:
    """Angle to goal center relative to agent heading (rad)."""
    goal_center = np.array([FIELD.x_opponent_ground, 0.0])
    dx = goal_center[0] - pos[0]
    dy = goal_center[1] - pos[1]
    absolute_angle = math.atan2(dy, dx)
    return _normalize_angle(absolute_angle - theta)


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
