"""
environment.py — NaoSoccerEnv: entorno de simulación 2D para MAPPO NAO Soccer.

Alineado con contrato BHuman 4v4 Scaled:
  - Campo 4v4 Scaled (2813 x 1875 mm)
  - Tácticas t121_4v4_scaled / t13_4v4_scaled
  - Handoff/takeover flow: RL posiciona → BHuman ejecuta
  - Macrostate tracking para reward/metrics
  - NO usa física local como fuente de verdad de kick/dribble

Flujo de step():
  1. rl_control_step: agentes RL se mueven según acciones
  2. handoff_check: evalúa handoff_ready para agente activo
  3. bhuman_takeover_rollout: simula ejecución BHuman si handoff_ready
  4. post_takeover_update: registra resultado de takeover
  5. Oponentes, física del balón, gol check, reward
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from config import (
    FIELD, AGENT, BALL, OBSTACLE, FOV, SIM,
    OBS, ACTION, GLOBAL_STATE, NORM,
    CURRICULUM, CurriculumStageConfig, ROLES, TACTIC_NAMES,
    ACTION_SMOOTHING, CONTRACT, EXECUTION_MODE,
)
from bhuman_cfg_parser import (
    get_home_poses, MACRO_TYPES, MACRO_END_REASONS, TAKEOVER_SKILLS,
)

TAKEOVER_MACRO_TYPES = frozenset(mt for mt in MACRO_TYPES if mt != "none")
TAKEOVER_SUCCESS_METRIC_TYPES = (
    "playBall_dribble",
    "playBall_duel",
    "playBall_shoot",
    "playBall_pass",
)
TAKEOVER_END_REASONS = (
    "timeout",
    "skill_request_cleared",
    "active_agent_changed",
    "out",
    "goal",
    "game_state_changed",
    "other",
)


def _new_takeover_reason_counts() -> Dict[str, int]:
    return {reason: 0 for reason in TAKEOVER_END_REASONS}


def _new_takeover_type_counts() -> Dict[str, int]:
    return {macro_type: 0 for macro_type in TAKEOVER_SUCCESS_METRIC_TYPES}


def _effective_sidecar_macro_type(skill_request: Any,
                                  macro_type: Any,
                                  motion_request: Any) -> str:
    """
    Resolve the operational play-ball macro from BHuman outputs.

    The bridge can transiently keep a low-level dribble / walkToBallAndKick
    motion while the high-level skill has already committed to shoot / pass.
    For takeover semantics we preserve the higher-level BHuman intent first,
    then fall back to the reported play-ball macro, and only then to the
    motion request.
    """
    skill = str(skill_request or "none")
    macro = str(macro_type or "none")
    motion = str(motion_request or "none")

    if macro == "playBall_duel":
        return "playBall_duel"
    if skill == "shoot":
        return "playBall_shoot"
    if skill == "pass":
        return "playBall_pass"
    if skill == "dribble":
        return "playBall_dribble"
    if macro in TAKEOVER_MACRO_TYPES:
        return macro
    if motion == "dribble":
        return "playBall_dribble"
    return "none"


# ─────────────────────────────────────────────────────────────────────────────
# ESTADOS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BallState:
    pos: np.ndarray   # [x, y] mm
    vel: np.ndarray   # [vx, vy] mm/s
    last_touched_by: Optional[str] = None
    seen_age_ms: float = 0.0  # ms since last seen


@dataclass
class AgentState:
    agent_id:       str
    pos:            np.ndarray  # [x, y] mm
    theta:          float       # rad
    vel:            np.ndarray  # [vx, vy] mm/s
    omega:          float       # rad/s
    role:           str         # "defenderL", "defenderR", "forwardL", "forwardR"
    is_upright:     bool = True
    recovery_steps: int  = 0
    battery:        float = 1.0
    walk_target:    np.ndarray = field(default_factory=lambda: np.zeros(2))
    speed_scalar:   float = 0.0
    is_active:      bool = False   # True if this is the active (closest to ball) agent


@dataclass
class OpponentState:
    opp_id:   str
    pos:      np.ndarray
    theta:    float
    vel:      np.ndarray
    is_upright: bool = True
    home_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class StaticObstacle:
    obstacle_id: str
    pos: np.ndarray
    radius: float


@dataclass
class GameState:
    own_score:    int   = 0
    opp_score:    int   = 0
    step_count:   int   = 0
    is_playing:   bool  = True
    time_elapsed: float = 0.0  # seconds


@dataclass
class MacroState:
    """Tracks BHuman macro-action execution."""
    macro_active: bool = False
    macro_type: str = "none"
    macro_owner: Optional[str] = None     # agent_id
    macro_start_frame: int = 0
    macro_duration_frames: int = 0
    macro_end_reason: str = "none"
    macro_success: Optional[bool] = None
    # Takeover tracking
    handoff_ready: bool = False
    takeover_active: bool = False
    takeover_grace_counter: int = 0
    event_recorded: bool = False
    takeover_start_ball_pos: Optional[np.ndarray] = None
    takeover_last_ball_pos: Optional[np.ndarray] = None
    takeover_control_frames: int = 0
    takeover_progress_frames: int = 0
    takeover_ball_progress_mm: float = 0.0
    takeover_ball_progress_along_target_mm: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS — Ball (sim-only, NOT source of truth for kick)
# ─────────────────────────────────────────────────────────────────────────────

class BallPhysics:
    """Ball physics for training simulation. BHuman handles real kicks."""

    def step(self, ball: BallState, dt: float) -> Tuple[BallState, bool]:
        new_pos = ball.pos + ball.vel * dt
        new_vel = ball.vel * BALL.friction

        new_pos, new_vel, ball_out = self._handle_boundaries(new_pos, new_vel)

        if np.linalg.norm(new_vel) < BALL.min_speed:
            new_vel = np.zeros(2)

        if not np.all(np.isfinite(new_pos)) or not np.all(np.isfinite(new_vel)):
            new_pos = np.zeros(2, dtype=np.float32)
            new_vel = np.zeros(2, dtype=np.float32)

        return BallState(
            pos=new_pos, vel=new_vel,
            last_touched_by=ball.last_touched_by,
            seen_age_ms=ball.seen_age_ms,
        ), ball_out

    def _handle_boundaries(self, pos: np.ndarray,
                           vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        x, y = pos
        ball_out = False

        if y > FIELD.y_left_sideline or y < FIELD.y_right_sideline:
            ball_out = True
        in_goal_y = FIELD.y_right_goal < y < FIELD.y_left_goal
        if x > FIELD.x_opponent_ground and not in_goal_y:
            ball_out = True
        elif x < FIELD.x_own_ground and not in_goal_y:
            ball_out = True

        if ball_out:
            x = float(np.clip(x, FIELD.x_own_ground + 50.0, FIELD.x_opponent_ground - 50.0))
            y = float(np.clip(y, FIELD.y_right_sideline + 50.0, FIELD.y_left_sideline - 50.0))
            return np.array([x, y]), np.zeros(2), True

        return np.array([x, y]), vel, False

    def check_goal(self, ball: BallState) -> Optional[str]:
        x, y = ball.pos
        in_goal_y = FIELD.y_right_goal < y < FIELD.y_left_goal
        if x > FIELD.x_opponent_ground and in_goal_y:
            return "own"
        if x < FIELD.x_own_ground and in_goal_y:
            return "opp"
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MOVEMENT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class NaoMovementModel:
    ROTATION_THRESHOLD = 0.5   # rad (~28.6°)

    def __init__(self):
        self.rotation_threshold = float(self.ROTATION_THRESHOLD)

    def configure(self, stage_cfg) -> None:
        self.rotation_threshold = float(
            getattr(stage_cfg, "rotation_threshold_rad", self.ROTATION_THRESHOLD)
        )

    def step(self, agent: AgentState, target_pos: np.ndarray,
             target_theta: float, dt: float) -> AgentState:
        if not agent.is_upright:
            return self._handle_fallen(agent, dt)

        new_pos, new_theta, new_vel, new_omega = self._compute_motion(
            agent.pos, agent.theta, target_pos, target_theta, dt
        )

        speed = float(np.linalg.norm(new_vel))
        is_upright, recovery = self._check_fall(speed, agent.is_upright, agent.recovery_steps)

        walk_target = target_pos.copy()
        speed_scalar = min(float(np.linalg.norm(new_vel)), AGENT.max_forward_speed)

        new_state = AgentState(
            agent_id=agent.agent_id, pos=new_pos, theta=new_theta,
            vel=new_vel, omega=new_omega, role=agent.role,
            is_upright=is_upright, recovery_steps=recovery,
            battery=max(0.0, agent.battery - 0.00001),
            walk_target=walk_target, speed_scalar=speed_scalar,
            is_active=agent.is_active,
        )

        if not math.isfinite(new_state.theta):
            new_state.theta = agent.theta
        if not np.all(np.isfinite(new_state.vel)):
            new_state.vel = np.zeros(2)
        if not math.isfinite(new_state.omega):
            new_state.omega = 0.0

        return new_state

    def _compute_motion(self, pos, theta, target_pos, target_theta, dt):
        dx = target_pos[0] - pos[0]
        dy = target_pos[1] - pos[1]
        dist = math.hypot(dx, dy)

        if dist < 10.0:
            dtheta = _normalize_angle(target_theta - theta)
            omega = np.clip(dtheta / dt, -AGENT.max_rotation_speed, AGENT.max_rotation_speed)
            new_theta = theta + omega * dt
            return pos.copy(), new_theta, np.zeros(2), omega

        angle_to_target = math.atan2(dy, dx)
        dtheta = _normalize_angle(angle_to_target - theta)

        if abs(dtheta) > self.rotation_threshold:
            omega = np.clip(dtheta / dt, -AGENT.max_rotation_speed, AGENT.max_rotation_speed)
            new_theta = _normalize_angle(theta + omega * dt)
            return pos.copy(), new_theta, np.zeros(2), omega

        omega = np.clip(dtheta / dt, -AGENT.max_rotation_speed, AGENT.max_rotation_speed)
        new_theta = _normalize_angle(theta + omega * dt)

        speed = min(dist / dt, AGENT.max_forward_speed)
        vx_robot = speed * math.cos(dtheta)
        vy_robot = speed * math.sin(dtheta)
        vx_robot = np.clip(vx_robot, -AGENT.max_backward_speed, AGENT.max_forward_speed)
        vy_robot = np.clip(vy_robot, -AGENT.max_sideways_speed, AGENT.max_sideways_speed)

        cos_t, sin_t = math.cos(theta), math.sin(theta)
        vx_field = vx_robot * cos_t - vy_robot * sin_t
        vy_field = vx_robot * sin_t + vy_robot * cos_t
        vel_field = np.array([vx_field, vy_field])
        new_pos = pos + vel_field * dt

        if not np.all(np.isfinite(new_pos)):
            new_pos = pos.copy()
            vel_field = np.zeros(2)

        new_pos[0] = np.clip(new_pos[0], FIELD.x_own_ground - 500.0, FIELD.x_opponent_ground + 500.0)
        new_pos[1] = np.clip(new_pos[1], FIELD.y_right_sideline - 500.0, FIELD.y_left_sideline + 500.0)

        return new_pos, new_theta, vel_field, omega

    def _check_fall(self, speed, was_upright, recovery):
        if not was_upright:
            new_recovery = recovery - 1
            return (new_recovery <= 0, max(0, new_recovery))
        if speed > AGENT.fall_speed_thresh and random.random() < AGENT.fall_probability:
            return False, AGENT.recovery_steps
        return True, 0

    def _handle_fallen(self, agent, dt):
        new_recovery = agent.recovery_steps - 1
        return AgentState(
            agent_id=agent.agent_id, pos=agent.pos.copy(), theta=agent.theta,
            vel=np.zeros(2), omega=0.0, role=agent.role,
            is_upright=(new_recovery <= 0), recovery_steps=max(0, new_recovery),
            battery=agent.battery, walk_target=agent.walk_target.copy(),
            speed_scalar=0.0, is_active=agent.is_active,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FOV FILTER
# ─────────────────────────────────────────────────────────────────────────────

class FOVFilter:
    def __init__(self, use_fov: bool = True):
        self.use_fov = use_fov
        self._memory: Dict[str, Dict[str, int]] = {}

    def reset(self, agent_ids):
        self._memory = {aid: {} for aid in agent_ids}

    def get_visible_opponents(self, agent_id, agent_pos, agent_theta, opponents):
        if not self.use_fov:
            return [opp for opp in opponents if opp.is_upright]

        memory = self._memory.setdefault(agent_id, {})
        visible = []
        seen_ids = set()

        for opp in opponents:
            if not opp.is_upright:
                continue
            if self._is_in_fov(agent_pos, agent_theta, opp.pos):
                if random.random() < FOV.detection_prob:
                    visible.append(opp)
                    seen_ids.add(opp.opp_id)
                    memory[opp.opp_id] = FOV.memory_steps

        for opp in opponents:
            if opp.opp_id in memory and opp.opp_id not in seen_ids:
                visible.append(opp)

        expired = [oid for oid, steps in memory.items() if steps <= 0]
        for oid in expired:
            del memory[oid]
        for oid in list(memory.keys()):
            if oid not in expired:
                memory[oid] -= 1

        return visible

    def _is_in_fov(self, agent_pos, agent_theta, target_pos):
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        dist = math.hypot(dx, dy)
        if dist > FOV.max_distance or dist < 1e-6:
            return False
        angle_to = math.atan2(dy, dx)
        angle_diff = abs(_normalize_angle(angle_to - agent_theta))
        return angle_diff < FOV.half_angle


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPTED OPPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class ScriptedOpponent:
    def __init__(self, movement_model):
        self._movement = movement_model

    def decide_and_move(self, opp, ball_pos, is_nearest_to_ball, dt):
        if not opp.is_upright:
            tmp = AgentState(
                agent_id=opp.opp_id, pos=opp.pos, theta=opp.theta,
                vel=opp.vel, omega=0.0, role="forward",
                is_upright=False, recovery_steps=10,
            )
            updated = self._movement._handle_fallen(tmp, dt)
            return OpponentState(
                opp_id=opp.opp_id, pos=updated.pos, theta=updated.theta,
                vel=updated.vel, is_upright=updated.is_upright, home_pos=opp.home_pos,
            )

        if is_nearest_to_ball:
            target_pos = ball_pos
            target_theta = math.atan2(ball_pos[1] - opp.pos[1], ball_pos[0] - opp.pos[0])
        else:
            target_pos = opp.home_pos
            target_theta = math.atan2(ball_pos[1] - opp.pos[1], ball_pos[0] - opp.pos[0])

        tmp = AgentState(
            agent_id=opp.opp_id, pos=opp.pos, theta=opp.theta,
            vel=opp.vel, omega=0.0, role="forward", is_upright=True,
        )
        updated = self._movement.step(tmp, target_pos, target_theta, dt)

        return OpponentState(
            opp_id=opp.opp_id, pos=updated.pos, theta=updated.theta,
            vel=updated.vel, is_upright=updated.is_upright, home_pos=opp.home_pos,
        )


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION BUILDER — 60 floats per agent
# ─────────────────────────────────────────────────────────────────────────────

class ObservationBuilder:
    def build(self, agent, all_agents, ball, visible_opponents,
              game, current_tactic):
        obs = np.zeros(OBS.total, dtype=np.float32)
        idx = 0
        idx = self._write_self(obs, idx, agent)
        idx = self._write_ball(obs, idx, ball, agent)
        idx = self._write_teammates(obs, idx, agent, all_agents)
        idx = self._write_opponents(obs, idx, visible_opponents)
        idx = self._write_game(obs, idx, game)
        idx = self._write_tactic(obs, idx, current_tactic)
        assert idx == OBS.total, f"ObsBuilder: idx={idx} != {OBS.total}"
        return obs

    def _write_self(self, obs, idx, agent):
        obs[idx]     = agent.pos[0] / NORM.pos_x
        obs[idx + 1] = agent.pos[1] / NORM.pos_y
        obs[idx + 2] = agent.theta  / NORM.theta
        obs[idx + 3] = agent.vel[0] / NORM.vel_xy
        obs[idx + 4] = agent.vel[1] / NORM.vel_xy
        obs[idx + 5] = agent.omega  / NORM.vel_theta
        obs[idx + 6] = 1.0 if agent.is_upright else 0.0
        obs[idx + 7] = agent.battery
        # Role one-hot (4 roles: defenderL, defenderR, forwardL, forwardR)
        role_idx = ROLES.index(agent.role) if agent.role in ROLES else 0
        obs[idx + 8 + role_idx] = 1.0
        # Active agent flag
        obs[idx + 12] = 1.0 if agent.is_active else 0.0
        # Goal info
        pos = agent.pos
        obs[idx + 13] = _goal_opening_angle(pos) / NORM.theta
        obs[idx + 14] = _goal_relative_angle(pos, agent.theta) / NORM.theta
        return idx + OBS.self_total  # +15

    def _write_ball(self, obs, idx, ball, agent):
        obs[idx]     = ball.pos[0] / NORM.pos_x
        obs[idx + 1] = ball.pos[1] / NORM.pos_y
        obs[idx + 2] = ball.vel[0] / NORM.vel_xy
        obs[idx + 3] = ball.vel[1] / NORM.vel_xy
        obs[idx + 4] = min(ball.seen_age_ms / NORM.ball_age, 1.0)
        obs[idx + 5] = 0.0   # using_team_ball
        return idx + OBS.ball_size  # +6

    def _write_teammates(self, obs, idx, agent, all_agents):
        teammates = [a for a in all_agents if a.agent_id != agent.agent_id]
        for slot in range(OBS.tm_slots):
            base = idx + slot * OBS.tm_features
            if slot < len(teammates):
                tm = teammates[slot]
                obs[base]     = tm.pos[0] / NORM.pos_x
                obs[base + 1] = tm.pos[1] / NORM.pos_y
                dx = tm.walk_target[0] - agent.pos[0]
                dy = tm.walk_target[1] - agent.pos[1]
                obs[base + 2] = dx / NORM.pos_x
                obs[base + 3] = dy / NORM.pos_y
                obs[base + 4] = tm.speed_scalar / NORM.speed
                obs[base + 5] = 1.0 if tm.is_active else 0.0
                obs[base + 6] = 1.0  # exists
        return idx + OBS.tm_total  # +21

    def _write_opponents(self, obs, idx, visible_opponents):
        for slot in range(OBS.opp_slots):
            base = idx + slot * OBS.opp_features
            if slot < len(visible_opponents):
                opp = visible_opponents[slot]
                obs[base]     = opp.pos[0] / NORM.pos_x
                obs[base + 1] = opp.pos[1] / NORM.pos_y
                obs[base + 2] = 1.0  # exists
        return idx + OBS.opp_total  # +12

    def _write_game(self, obs, idx, game):
        obs[idx]     = 1.0 if game.is_playing else 0.0
        obs[idx + 1] = game.own_score / NORM.score
        obs[idx + 2] = game.opp_score / NORM.score
        time_remaining = max(0.0, 600.0 - game.time_elapsed)
        obs[idx + 3] = time_remaining / NORM.time
        return idx + OBS.game_size  # +4

    def _write_tactic(self, obs, idx, tactic):
        if tactic in TACTIC_NAMES:
            tactic_idx = TACTIC_NAMES.index(tactic)
            obs[idx + tactic_idx] = 1.0
        return idx + OBS.tactic_size  # +2


def build_global_state(agents, ball, opponents, game, static_obstacles=None):
    state = np.zeros(GLOBAL_STATE.total, dtype=np.float32)
    idx = 0
    for i in range(4):
        if i < len(agents):
            a = agents[i]
            state[idx]     = a.pos[0] / NORM.pos_x
            state[idx + 1] = a.pos[1] / NORM.pos_y
            state[idx + 2] = a.theta  / NORM.theta
        idx += 3
    for i in range(4):
        if i < len(agents):
            state[idx] = 1.0 if agents[i].is_upright else 0.0
        idx += 1
    state[idx]     = ball.pos[0] / NORM.pos_x
    state[idx + 1] = ball.pos[1] / NORM.pos_y
    state[idx + 2] = ball.vel[0] / NORM.vel_xy
    state[idx + 3] = ball.vel[1] / NORM.vel_xy
    idx += 4
    obstacle_list = static_obstacles or []
    opponent_like = list(opponents[:4])
    remaining = max(0, 4 - len(opponent_like))
    opponent_like.extend(obstacle_list[:remaining])
    for i in range(4):
        if i < len(opponent_like):
            state[idx]     = opponent_like[i].pos[0] / NORM.pos_x
            state[idx + 1] = opponent_like[i].pos[1] / NORM.pos_y
        idx += 2
    state[idx]     = game.own_score / NORM.score
    state[idx + 1] = game.opp_score / NORM.score
    state[idx + 2] = max(0.0, 600.0 - game.time_elapsed) / NORM.time
    idx += 3
    assert idx == GLOBAL_STATE.total
    return state


# ─────────────────────────────────────────────────────────────────────────────
# HANDOFF / TAKEOVER LOGIC
# ─────────────────────────────────────────────────────────────────────────────

class HandoffTakeoverManager:
    """
    Manages the handoff_ready / takeover_active lifecycle.

    Flow:
      1. Each frame, check if any RL agent satisfies handoff_ready conditions
      2. If handoff_ready and agent is closest to ball → activate takeover
      3. During takeover, BHuman "executes" (simulated as macro-action)
      4. Track macro duration, success/failure, end reason
    """

    def __init__(self, contract: "AllConfigs"):
        self._handoff_cfg = contract.handoff
        self._kick_info = contract.kick_info
        self._takeover_timeout_frames = contract.handoff.takeover_timeout_frames
        self.macro = MacroState()

    def reset(self):
        self.macro = MacroState()

    def set_takeover_timeout_frames(self, frames: int) -> None:
        self._takeover_timeout_frames = max(1, int(frames))

    def force_end(self, reason: str, success: bool) -> None:
        self._end_macro(reason, success)

    def compute_handoff_ready(self, agent: AgentState,
                              ball: BallState,
                              game: GameState) -> bool:
        """
        Evaluate handoff_ready for a specific agent.
        Conditions from BHUMANPARSER_GUIDE 2.2.
        """
        if not game.is_playing:
            return False
        if not agent.is_upright:
            return False

        # Ball relative to agent
        dx = ball.pos[0] - agent.pos[0]
        dy = ball.pos[1] - agent.pos[1]
        dist = math.hypot(dx, dy)

        # Ball must be in front and reachable
        # Rotate to agent frame
        cos_t = math.cos(-agent.theta)
        sin_t = math.sin(-agent.theta)
        rel_x = dx * cos_t - dy * sin_t
        rel_y = dx * sin_t + dy * cos_t

        if rel_x < self._handoff_cfg.ball_rel_x_min:
            return False
        if rel_x > self._handoff_cfg.ball_rel_x_max:
            return False
        if abs(rel_y) > self._handoff_cfg.ball_rel_y_abs_max:
            return False

        # Angle to ball
        angle_to_ball = math.atan2(dy, dx)
        angle_diff = abs(_normalize_angle(angle_to_ball - agent.theta))
        if math.degrees(angle_diff) > self._handoff_cfg.angle_to_ball_abs_max:
            return False

        # Ball visibility (seen_age)
        if ball.seen_age_ms > self._handoff_cfg.ball_seen_age_max:
            return False

        return True

    def update(self, agents: Dict[str, AgentState],
               ball: BallState, game: GameState,
               active_agent_id: Optional[str],
               goal_scored: bool, goal_conceded: bool,
               ball_out: bool,
               sidecar_macro_type: str = "none") -> MacroState:
        """
        Update macro state for this frame.
        Returns updated MacroState.
        """
        # If macro is active, check termination
        if self.macro.macro_active:
            self.macro.macro_duration_frames += 1

            # Timeout
            if self.macro.macro_duration_frames >= self._takeover_timeout_frames:
                self._end_macro("timeout", False)
                return self.macro

            # Goal scored during macro → success
            if goal_scored:
                self._end_macro("goal", True)
                return self.macro

            # Goal conceded → failure
            if goal_conceded:
                self._end_macro("goal", False)
                return self.macro

            # Ball out → failure
            if ball_out:
                self._end_macro("out", False)
                return self.macro

            # Game state change
            if not game.is_playing:
                self._end_macro("game_state_changed", False)
                return self.macro

            # Active agent changed
            if self.macro.macro_owner != active_agent_id:
                self._end_macro("active_agent_changed", False)
                return self.macro

            # Check if owner is still in handoff envelope (soft exit)
            if self.macro.macro_owner in agents:
                owner = agents[self.macro.macro_owner]
                still_ready = self.compute_handoff_ready(owner, ball, game)
                if not still_ready:
                    self.macro.takeover_grace_counter += 1
                    if self.macro.takeover_grace_counter >= self._handoff_cfg.takeover_grace_frames:
                        self._end_macro("skill_request_cleared", False)
                        return self.macro
                else:
                    self.macro.takeover_grace_counter = 0

            return self.macro

        # If macro is not active, check for new handoff
        if active_agent_id and active_agent_id in agents:
            agent = agents[active_agent_id]
            ready = self.compute_handoff_ready(agent, ball, game)
            self.macro.handoff_ready = ready

            if ready and sidecar_macro_type in TAKEOVER_MACRO_TYPES:
                self._start_macro(active_agent_id, game.step_count,
                                  macro_type=sidecar_macro_type)

        return self.macro

    def _start_macro(self, agent_id: str, frame: int,
                     macro_type: str = "none"):
        self.macro.macro_active = True
        self.macro.takeover_active = True
        self.macro.handoff_ready = True
        self.macro.macro_owner = agent_id
        self.macro.macro_start_frame = frame
        self.macro.macro_duration_frames = 0
        self.macro.macro_type = macro_type  # real type from sidecar; "none" if stub
        self.macro.macro_end_reason = "none"
        self.macro.macro_success = None
        self.macro.takeover_grace_counter = 0
        self.macro.event_recorded = False
        self.macro.takeover_start_ball_pos = None
        self.macro.takeover_last_ball_pos = None
        self.macro.takeover_control_frames = 0
        self.macro.takeover_progress_frames = 0
        self.macro.takeover_ball_progress_mm = 0.0
        self.macro.takeover_ball_progress_along_target_mm = 0.0

    def _end_macro(self, reason: str, success: bool):
        self.macro.macro_active = False
        self.macro.takeover_active = False
        self.macro.handoff_ready = False
        self.macro.macro_end_reason = reason
        self.macro.macro_success = success
        self.macro.event_recorded = False


# ─────────────────────────────────────────────────────────────────────────────
# BHUMAN TAKEOVER SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class BHumanTakeoverSimulator:
    """
    Simulates BHuman skill execution during takeover.

    During takeover, the active agent is controlled by BHuman (simulated).
    The agent moves toward the ball and executes a kick/pass/dribble.
    This is NOT the source of truth — it approximates what BHuman would do
    so the RL agent can learn the interface quality.
    """

    def __init__(self, movement: NaoMovementModel, contract: "AllConfigs"):
        self._movement = movement
        self._kick_info = contract.kick_info
        self._smash_or_pass = contract.smash_or_pass

    def simulate_step(self, agent: AgentState, ball: BallState,
                      dt: float) -> Tuple[AgentState, BallState, bool]:
        """
        Simulate one step of BHuman execution.

        Returns: (updated_agent, updated_ball, kick_executed)
        """
        if not agent.is_upright:
            return agent, ball, False

        # BHuman moves toward ball
        dx = ball.pos[0] - agent.pos[0]
        dy = ball.pos[1] - agent.pos[1]
        dist = math.hypot(dx, dy)

        target_theta = math.atan2(dy, dx)
        updated_agent = self._movement.step(agent, ball.pos, target_theta, dt)

        # Check if close enough to kick
        new_dist = float(np.linalg.norm(updated_agent.pos - ball.pos))
        kick_executed = False

        if new_dist < AGENT.ball_interaction_range * 0.7:  # 350mm
            # Angle check
            angle_to_ball = math.atan2(
                ball.pos[1] - updated_agent.pos[1],
                ball.pos[0] - updated_agent.pos[0],
            )
            angle_diff = abs(_normalize_angle(angle_to_ball - updated_agent.theta))
            if angle_diff < math.radians(45):
                # Execute kick toward opponent goal
                kick_dir = np.array([
                    math.cos(updated_agent.theta),
                    math.sin(updated_agent.theta),
                ])
                kick_speed = self._kick_info.ball_velocity_max  # 1800 mm/s
                ball = BallState(
                    pos=updated_agent.pos + kick_dir * (AGENT.min_robot_distance + 50),
                    vel=kick_dir * kick_speed,
                    last_touched_by=updated_agent.agent_id,
                )
                kick_executed = True

        return updated_agent, ball, kick_executed


class RealSidecarPhysicsExecutor:
    """
    Applies physical effects from the real sidecar output without inventing
    a separate decision policy.

    Decision semantics come from the real bridge. Python only rolls those
    semantics forward inside the 2D training world.
    """

    def __init__(self, movement: NaoMovementModel, contract: "AllConfigs"):
        self._movement = movement
        self._kick_info = contract.kick_info
        self._kick_entries = {
            entry.kick_type: entry for entry in contract.kick_info.kicks
        }
        self._last_kick_timestamps: Dict[str, int] = {}
        self._pending_kick_hold_frames: Dict[str, int] = {}
        self._pending_kick_hold_limit = 6
        self._pending_kick_ready_frames: Dict[str, int] = {}
        self._pending_kick_ready_limit = 2
        self._failed_kick_cooldown_frames: Dict[str, int] = {}
        self._failed_kick_cooldown_limit = 6
        self._post_kick_release_frames: Dict[str, int] = {}
        self._post_kick_release_limit = 5
        self._release_pose_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._pre_kick_pose_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._smoothed_motion_targets: Dict[str, np.ndarray] = {}
        self._smoothed_motion_thetas: Dict[str, float] = {}
        self._motion_target_alpha = 0.38
        self._motion_theta_alpha = 0.34
        self._motion_target_max_shift = 320.0
        self._motion_theta_max_shift = 0.28
        self._motion_target_snap_distance = 130.0
        self._motion_theta_snap_error = 0.14

    def reset(self) -> None:
        self._last_kick_timestamps.clear()
        self._pending_kick_hold_frames.clear()
        self._pending_kick_ready_frames.clear()
        self._failed_kick_cooldown_frames.clear()
        self._post_kick_release_frames.clear()
        self._release_pose_cache.clear()
        self._pre_kick_pose_cache.clear()
        self._smoothed_motion_targets.clear()
        self._smoothed_motion_thetas.clear()

    def apply(self,
              agent: AgentState,
              ball: BallState,
              output: Any,
              agents: Dict[str, AgentState],
              dt: float) -> Tuple[AgentState, BallState, bool, bool, bool, int]:
        if not agent.is_upright:
            return agent, ball, False, False, False, 0

        post_kick_release = max(
            0,
            self._post_kick_release_frames.get(agent.agent_id, 0) - 1,
        )
        self._post_kick_release_frames[agent.agent_id] = post_kick_release
        failed_kick_cooldown = max(
            0,
            self._failed_kick_cooldown_frames.get(agent.agent_id, 0) - 1,
        )
        self._failed_kick_cooldown_frames[agent.agent_id] = failed_kick_cooldown
        pending_kick_hold = False
        pending_kick_hold_frames = 0
        if post_kick_release > 0:
            updated_agent = self._apply_post_kick_release_motion(
                agent,
                ball,
                output,
                agents,
                dt,
            )
            self._pending_kick_hold_frames[agent.agent_id] = 0
            self._pending_kick_ready_frames[agent.agent_id] = 0
            self._failed_kick_cooldown_frames[agent.agent_id] = max(
                self._failed_kick_cooldown_frames.get(agent.agent_id, 0),
                1,
            )
            return updated_agent, ball, False, False, False, 0

        kick_intent_active = (
            post_kick_release <= 0
            and failed_kick_cooldown <= 0
            and self._is_kick_intent_active(output)
        )

        if kick_intent_active:
            updated_agent, pending_kick_hold = self._stabilize_pre_kick_pose(
                agent,
                ball,
                output,
                dt,
            )
            if pending_kick_hold:
                pending_kick_hold_frames = self._pending_kick_hold_frames.get(agent.agent_id, 0) + 1
                self._pending_kick_hold_frames[agent.agent_id] = pending_kick_hold_frames
                self._pending_kick_ready_frames[agent.agent_id] = 0
            else:
                self._pending_kick_hold_frames[agent.agent_id] = 0
                self._pending_kick_ready_frames[agent.agent_id] = (
                    self._pending_kick_ready_frames.get(agent.agent_id, 0) + 1
                )
        else:
            updated_agent = self._apply_agent_motion(agent, ball, output, dt)
            self._pending_kick_hold_frames[agent.agent_id] = 0
            self._pending_kick_ready_frames[agent.agent_id] = 0
            self._failed_kick_cooldown_frames[agent.agent_id] = 0
            self._release_pose_cache.pop(agent.agent_id, None)
            self._pre_kick_pose_cache.pop(agent.agent_id, None)

        if self._has_new_kick(agent.agent_id, output):
            self._pending_kick_hold_frames[agent.agent_id] = 0
            self._pending_kick_ready_frames[agent.agent_id] = 0
            self._failed_kick_cooldown_frames[agent.agent_id] = 0
            self._post_kick_release_frames[agent.agent_id] = self._post_kick_release_limit
            self._seed_release_pose(updated_agent, ball, output, agents)
            self._pre_kick_pose_cache.pop(agent.agent_id, None)
            self._clear_motion_smoothing(agent.agent_id)
            return updated_agent, self._apply_kick(updated_agent, ball, output, agents), True, False, False, 0

        if kick_intent_active and self._is_ready_for_guided_kick(updated_agent, ball, output, agents):
            touch_ball, touch_progress = self._apply_guided_pre_kick_touch(
                updated_agent,
                ball,
                output,
                agents,
            )
            self._pending_kick_hold_frames[agent.agent_id] = 0
            self._pending_kick_ready_frames[agent.agent_id] = 0
            self._pre_kick_pose_cache.pop(agent.agent_id, None)
            if touch_progress:
                self._failed_kick_cooldown_frames[agent.agent_id] = 0
                self._post_kick_release_frames[agent.agent_id] = self._post_kick_release_limit
                self._seed_release_pose(updated_agent, ball, output, agents)
                self._clear_motion_smoothing(agent.agent_id)
                return updated_agent, touch_ball, False, True, False, 0

        if kick_intent_active and (
            (pending_kick_hold and pending_kick_hold_frames >= self._pending_kick_hold_limit)
            or self._pending_kick_ready_frames.get(agent.agent_id, 0) >= self._pending_kick_ready_limit
        ):
            touch_ball, touch_progress = self._apply_guided_pre_kick_touch(
                updated_agent,
                ball,
                output,
                agents,
            )
            self._pending_kick_hold_frames[agent.agent_id] = 0
            self._pending_kick_ready_frames[agent.agent_id] = 0
            self._pre_kick_pose_cache.pop(agent.agent_id, None)
            if touch_progress:
                self._failed_kick_cooldown_frames[agent.agent_id] = 0
                self._post_kick_release_frames[agent.agent_id] = self._post_kick_release_limit
                self._seed_release_pose(updated_agent, ball, output, agents)
                self._clear_motion_smoothing(agent.agent_id)
                return updated_agent, touch_ball, False, True, False, pending_kick_hold_frames
            self._failed_kick_cooldown_frames[agent.agent_id] = self._failed_kick_cooldown_limit
            self._release_pose_cache.pop(agent.agent_id, None)
            self._clear_motion_smoothing(agent.agent_id)
            return updated_agent, ball, False, False, False, pending_kick_hold_frames

        if output.motion_request == "dribble" and post_kick_release <= 0:
            dribble_ball, dribble_progress = self._apply_dribble(updated_agent, ball, output)
            self._pending_kick_hold_frames[agent.agent_id] = 0
            self._pending_kick_ready_frames[agent.agent_id] = 0
            self._failed_kick_cooldown_frames[agent.agent_id] = 0
            self._release_pose_cache.pop(agent.agent_id, None)
            self._pre_kick_pose_cache.pop(agent.agent_id, None)
            return updated_agent, dribble_ball, False, dribble_progress, False, 0

        return updated_agent, ball, False, False, pending_kick_hold, pending_kick_hold_frames

    def _apply_agent_motion(self,
                            agent: AgentState,
                            ball: BallState,
                            output: Any,
                            dt: float) -> AgentState:
        raw_target_pos, raw_target_theta = self._resolve_motion_target(agent, ball, output, dt)
        target_pos, target_theta = self._smooth_motion_target(
            agent,
            raw_target_pos,
            raw_target_theta,
            dt,
        )
        return self._movement.step(agent, target_pos, target_theta, dt)

    def _clear_motion_smoothing(self, agent_id: str) -> None:
        self._smoothed_motion_targets.pop(agent_id, None)
        self._smoothed_motion_thetas.pop(agent_id, None)

    def _seed_release_pose(self,
                           agent: AgentState,
                           ball: BallState,
                           output: Any,
                           agents: Dict[str, AgentState]) -> None:
        kick_dir = self._resolve_kick_direction(agent, ball, output, agents)
        theta = math.atan2(float(kick_dir[1]), float(kick_dir[0]))
        follow_distance = 90.0
        follow_pos = self._clip_field_point(
            agent.pos + kick_dir * follow_distance,
            margin=60.0,
        )
        self._release_pose_cache[agent.agent_id] = (
            follow_pos.astype(np.float32),
            _normalize_angle(theta),
        )

    def _apply_post_kick_release_motion(self,
                                        agent: AgentState,
                                        ball: BallState,
                                        output: Any,
                                        agents: Dict[str, AgentState],
                                        dt: float) -> AgentState:
        release_pose = self._release_pose_cache.get(agent.agent_id)
        if release_pose is None:
            self._seed_release_pose(agent, ball, output, agents)
            release_pose = self._release_pose_cache.get(agent.agent_id)
        if release_pose is None:
            return agent

        target_pos, target_theta = release_pose
        target_pos, target_theta = self._smooth_motion_target(
            agent,
            np.asarray(target_pos, dtype=np.float32),
            float(target_theta),
            dt,
        )
        updated = self._movement.step(agent, target_pos, target_theta, dt)
        self._release_pose_cache[agent.agent_id] = (
            target_pos.astype(np.float32),
            float(target_theta),
        )
        return updated

    def _smooth_motion_target(self,
                              agent: AgentState,
                              target_pos: np.ndarray,
                              target_theta: float,
                              dt: float) -> Tuple[np.ndarray, float]:
        agent_id = agent.agent_id
        raw_target = np.asarray(target_pos, dtype=np.float32)
        raw_theta = _normalize_angle(float(target_theta))
        prev_target = self._smoothed_motion_targets.get(agent_id)
        prev_theta = self._smoothed_motion_thetas.get(agent_id)

        if prev_target is None or prev_theta is None:
            self._smoothed_motion_targets[agent_id] = raw_target.copy()
            self._smoothed_motion_thetas[agent_id] = raw_theta
            return raw_target, raw_theta

        target_delta = raw_target - prev_target
        target_delta_norm = float(np.linalg.norm(target_delta))
        max_shift = max(
            self._motion_target_max_shift,
            AGENT.max_forward_speed * max(dt, 1e-3) * 18.0,
        )
        bounded_target = raw_target.copy()
        if target_delta_norm > max_shift and target_delta_norm > 1e-6:
            bounded_target = (
                prev_target + (target_delta / target_delta_norm) * max_shift
            ).astype(np.float32)

        dist_from_agent = float(np.linalg.norm(raw_target - agent.pos))
        if dist_from_agent <= self._motion_target_snap_distance:
            smooth_target = raw_target
        else:
            smooth_target = (
                prev_target + self._motion_target_alpha * (bounded_target - prev_target)
            ).astype(np.float32)

        theta_error = _normalize_angle(raw_theta - float(prev_theta))
        bounded_theta_error = float(np.clip(
            theta_error,
            -self._motion_theta_max_shift,
            self._motion_theta_max_shift,
        ))
        if abs(theta_error) <= self._motion_theta_snap_error:
            smooth_theta = raw_theta
        else:
            smooth_theta = _normalize_angle(
                float(prev_theta) + self._motion_theta_alpha * bounded_theta_error
            )

        self._smoothed_motion_targets[agent_id] = smooth_target.copy()
        self._smoothed_motion_thetas[agent_id] = smooth_theta
        return smooth_target, smooth_theta

    def _is_kick_intent_active(self, output: Any) -> bool:
        skill_request = str(getattr(output, "skill_request", "none") or "none")
        motion_request = str(getattr(output, "motion_request", "none") or "none")
        macro_type = _effective_sidecar_macro_type(
            skill_request,
            getattr(output, "macro_type", "none"),
            motion_request,
        )
        if macro_type not in {"playBall_shoot", "playBall_pass"}:
            return False
        return motion_request in {"walkToBallAndKick", "stand", "dribble"}

    def _is_pending_kick_intent(self, output: Any) -> bool:
        skill_request = str(getattr(output, "skill_request", "none") or "none")
        motion_request = str(getattr(output, "motion_request", "none") or "none")
        return (
            skill_request in {"shoot", "pass"}
            and motion_request == "walkToBallAndKick"
        )

    def _stabilize_pre_kick_pose(self,
                                 agent: AgentState,
                                 ball: BallState,
                                 output: Any,
                                 dt: float) -> Tuple[AgentState, bool]:
        desired_pos, desired_theta = self._resolve_playball_body_target(agent, ball, output)
        cached_pose = self._pre_kick_pose_cache.get(agent.agent_id)
        if cached_pose is not None:
            cached_pos, cached_theta = cached_pose
            desired_pos = (
                0.7 * np.asarray(cached_pos, dtype=np.float32)
                + 0.3 * desired_pos.astype(np.float32)
            ).astype(np.float32)
            desired_theta = _normalize_angle(
                float(cached_theta) + 0.3 * _normalize_angle(desired_theta - float(cached_theta))
            )
        self._pre_kick_pose_cache[agent.agent_id] = (desired_pos.copy(), float(desired_theta))

        desired_delta = desired_pos - agent.pos
        desired_dist = float(np.linalg.norm(desired_delta))
        ball_dist = float(np.linalg.norm(agent.pos - ball.pos))
        min_clearance = AGENT.radius + BALL.radius + 8.0
        theta_error = _normalize_angle(desired_theta - agent.theta)

        if ball_dist >= min_clearance and desired_dist <= 25.0 and abs(theta_error) <= 0.15:
            return agent, False

        max_translation_step = max(70.0, AGENT.max_forward_speed * max(dt, 1e-3) * 8.0)
        if desired_dist > max_translation_step and desired_dist > 1e-6:
            corrected_pos = (
                agent.pos + (desired_delta / desired_dist) * max_translation_step
            ).astype(np.float32)
        else:
            corrected_pos = desired_pos.astype(np.float32)

        corrected_ball_dist = float(np.linalg.norm(corrected_pos - ball.pos))
        if corrected_ball_dist < min_clearance:
            safe_dir = corrected_pos - ball.pos
            safe_norm = float(np.linalg.norm(safe_dir))
            if safe_norm <= 1e-6:
                safe_dir = np.array([math.cos(desired_theta), math.sin(desired_theta)], dtype=np.float32)
                safe_norm = 1.0
            corrected_pos = (ball.pos + (safe_dir / safe_norm) * min_clearance).astype(np.float32)

        max_theta_step = max(0.16, AGENT.max_rotation_speed * max(dt, 1e-3) * 2.0)
        corrected_theta = _normalize_angle(
            agent.theta + float(np.clip(theta_error, -max_theta_step, max_theta_step))
        )
        corrected_pos, corrected_theta = self._smooth_motion_target(
            agent,
            corrected_pos,
            corrected_theta,
            dt,
        )
        corrected_vel = (corrected_pos - agent.pos) / max(dt, 1e-3)
        corrected_speed = float(np.linalg.norm(corrected_vel))
        return AgentState(
            agent_id=agent.agent_id,
            pos=corrected_pos,
            theta=corrected_theta,
            vel=corrected_vel,
            omega=float(np.clip(theta_error / max(dt, 1e-3), -AGENT.max_rotation_speed, AGENT.max_rotation_speed)),
            role=agent.role,
            is_upright=agent.is_upright,
            recovery_steps=agent.recovery_steps,
            battery=agent.battery,
            walk_target=desired_pos.copy(),
            speed_scalar=min(corrected_speed, AGENT.max_forward_speed),
            is_active=agent.is_active,
        ), True

    def _apply_guided_pre_kick_touch(self,
                                     agent: AgentState,
                                     ball: BallState,
                                     output: Any,
                                     agents: Dict[str, AgentState]) -> Tuple[BallState, bool]:
        distance = float(np.linalg.norm(agent.pos - ball.pos))
        contact_window = AGENT.radius + BALL.radius + 45.0
        if distance > contact_window:
            return ball, False

        # When BHuman is already in shoot/pass intent and we have reached the
        # execution pose, release the ball with a kick-like signature instead
        # of a tiny "nudge". That keeps the 2D executor aligned with the
        # semantic intent of walkToBallAndKick.
        kick_ball = self._apply_kick(agent, ball, output, agents)
        kick_speed = float(np.linalg.norm(kick_ball.vel))
        min_proxy_speed = max(
            420.0,
            min(float(self._kick_info.ball_velocity_max) * 0.32, self._kick_speed(output) * 0.55),
        )
        if kick_speed < min_proxy_speed:
            direction = kick_ball.vel / max(kick_speed, 1e-6)
            kick_ball = BallState(
                pos=kick_ball.pos.copy(),
                vel=direction * min_proxy_speed,
                last_touched_by=kick_ball.last_touched_by,
                seen_age_ms=kick_ball.seen_age_ms,
            )
        return kick_ball, True

    def _is_ready_for_guided_kick(self,
                                  agent: AgentState,
                                  ball: BallState,
                                  output: Any,
                                  agents: Dict[str, AgentState]) -> bool:
        motion_request = str(getattr(output, "motion_request", "none") or "none")
        if motion_request not in {"stand", "walkToBallAndKick"}:
            return False

        distance = float(np.linalg.norm(agent.pos - ball.pos))
        contact_window = AGENT.radius + BALL.radius + 45.0
        if distance > contact_window:
            return False

        desired_pos, desired_theta = self._resolve_playball_body_target(agent, ball, output)
        desired_dist = float(np.linalg.norm(desired_pos - agent.pos))
        desired_theta_error = abs(_normalize_angle(desired_theta - agent.theta))
        if desired_dist <= 42.0 and desired_theta_error <= math.radians(24.0):
            return True

        kick_dir = self._resolve_kick_direction(agent, ball, output, agents)
        kick_theta = math.atan2(float(kick_dir[1]), float(kick_dir[0]))
        theta_error = abs(_normalize_angle(kick_theta - agent.theta))
        if theta_error > math.radians(42.0):
            return False

        angle_to_ball = math.atan2(
            float(ball.pos[1] - agent.pos[1]),
            float(ball.pos[0] - agent.pos[0]),
        )
        ball_theta_error = abs(_normalize_angle(angle_to_ball - agent.theta))
        return ball_theta_error <= math.radians(58.0)

    def _target_array(self, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.size < 2:
                return None
            target = value[:2].astype(np.float32, copy=True)
            if not np.all(np.isfinite(target)):
                return None
            return target
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            target = np.array([value[0], value[1]], dtype=np.float32)
            if not np.all(np.isfinite(target)):
                return None
            return target
        return None

    def _sanitize_angle(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            angle = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(angle):
            return None
        if abs(angle) > math.pi + 0.25:
            return None
        return angle

    def _sanitize_kick_length(self, value: Any) -> float:
        try:
            kick_length = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(kick_length):
            return 0.0
        max_reasonable = max(
            float(getattr(self._kick_info, "forward_fast_max", 0.0) or 0.0),
            float(FIELD.length),
        ) * 1.1
        if kick_length > max_reasonable:
            return 0.0
        return float(np.clip(kick_length, 0.0, max_reasonable))

    def _clip_field_point(self,
                          target: np.ndarray,
                          *,
                          margin: float = 80.0) -> np.ndarray:
        clipped = np.array(target, dtype=np.float32, copy=True)
        clipped[0] = float(np.clip(
            clipped[0],
            FIELD.x_own_ground + margin,
            FIELD.x_opponent_ground - margin,
        ))
        clipped[1] = float(np.clip(
            clipped[1],
            FIELD.y_right_sideline + margin,
            FIELD.y_left_sideline - margin,
        ))
        return clipped

    def _sanitize_macro_ball_target(self, raw_target: np.ndarray) -> np.ndarray:
        # Keep BHuman's operational intent intact. Only reject impossible or
        # out-of-bounds targets; do not reproject them into a handcrafted cone.
        return self._clip_field_point(raw_target).astype(np.float32)

    def _max_support_offset(self,
                            contact_point: np.ndarray,
                            direction: np.ndarray,
                            *,
                            margin: float) -> float:
        max_offset = float("inf")
        x_min = FIELD.x_own_ground + margin
        x_max = FIELD.x_opponent_ground - margin
        y_min = FIELD.y_right_sideline + margin
        y_max = FIELD.y_left_sideline - margin

        for point_value, axis_dir, lower, upper in (
            (float(contact_point[0]), float(direction[0]), x_min, x_max),
            (float(contact_point[1]), float(direction[1]), y_min, y_max),
        ):
            if axis_dir > 1e-6:
                max_offset = min(max_offset, (point_value - lower) / axis_dir)
            elif axis_dir < -1e-6:
                max_offset = min(max_offset, (upper - point_value) / (-axis_dir))

        if not math.isfinite(max_offset):
            return float("inf")
        return max(0.0, float(max_offset))

    def _side_support_pose(self,
                           agent: AgentState,
                           contact_point: np.ndarray,
                           direction: np.ndarray,
                           *,
                           margin: float,
                           preferred_offset: float) -> np.ndarray:
        tangent = np.array([-direction[1], direction[0]], dtype=np.float32)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-6:
            tangent = np.array([
                -math.sin(agent.theta),
                math.cos(agent.theta),
            ], dtype=np.float32)
            tangent_norm = max(1e-6, float(np.linalg.norm(tangent)))
        tangent /= tangent_norm

        agent_delta = agent.pos - contact_point
        tangent_sign = 1.0 if float(np.dot(agent_delta, tangent)) >= 0.0 else -1.0
        lateral_offset = min(preferred_offset, 140.0)

        side_pose = contact_point + tangent * (tangent_sign * lateral_offset)
        side_pose = self._clip_field_point(side_pose, margin=margin)

        # Keep a small backward bias so the robot still approaches from behind
        # the ball instead of cutting directly across it at the goal line.
        backward_cap = self._max_support_offset(contact_point, direction, margin=margin)
        backward_offset = min(max(18.0, preferred_offset * 0.18), backward_cap)
        if backward_offset > 1e-6:
            side_pose = self._clip_field_point(
                side_pose - direction * backward_offset,
                margin=margin,
            )
        return side_pose.astype(np.float32)

    def _bridge_field_target(self,
                             value: Any,
                             ball: Optional[BallState] = None,
                             *,
                             allow_origin: bool = False) -> Optional[np.ndarray]:
        target = self._target_array(value)
        if target is None:
            return None
        if (
            not allow_origin
            and ball is not None
            and float(np.linalg.norm(target)) <= 1.0
            and float(np.linalg.norm(ball.pos)) > 250.0
        ):
            return None
        return self._clip_field_point(target)

    def _sanitize_sidecar_target(self,
                                 value: Any,
                                 ball: BallState,
                                 *,
                                 allow_origin: bool = False) -> Optional[np.ndarray]:
        target = self._target_array(value)
        if target is None:
            return None
        if allow_origin:
            return target

        # The bridge occasionally publishes (0, 0) as a placeholder for
        # optional targets. When the ball is far from midfield, treating that
        # as a real target makes the robot drift diagonally toward nowhere.
        if (
            float(np.linalg.norm(target)) <= 1.0
            and float(np.linalg.norm(ball.pos)) > 250.0
        ):
            return None
        return target

    def _operational_robot_pose(self,
                                agent: AgentState,
                                output: Optional[Any] = None) -> Tuple[np.ndarray, float]:
        if output is not None:
            pose = getattr(output, "robot_pose_field", None)
            if isinstance(pose, (list, tuple)) and len(pose) >= 3:
                try:
                    px = float(pose[0])
                    py = float(pose[1])
                    ptheta = float(pose[2])
                except (TypeError, ValueError):
                    pass
                else:
                    if math.isfinite(px) and math.isfinite(py) and math.isfinite(ptheta):
                        return np.array([px, py], dtype=np.float32), _normalize_angle(ptheta)
        return agent.pos.copy(), float(agent.theta)

    def _operational_ball_position(self,
                                   ball: BallState,
                                   output: Optional[Any] = None,
                                   *,
                                   prefer_intercepted: bool = False) -> np.ndarray:
        if output is not None:
            attrs = (
                ("field_ball_intercepted_position", "field_ball_position")
                if prefer_intercepted
                else ("field_ball_position", "field_ball_intercepted_position")
            )
            for attr in attrs:
                target = self._target_array(getattr(output, attr, None))
                if target is not None:
                    return target.astype(np.float32)
        return ball.pos.copy()

    def _robot_relative_target_to_field(self,
                                        agent: AgentState,
                                        value: Any,
                                        output: Optional[Any] = None) -> Optional[np.ndarray]:
        target = self._target_array(value)
        if target is None:
            return None
        if float(np.linalg.norm(target)) <= 1.0:
            return None
        origin, theta = self._operational_robot_pose(agent, output)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx = float(target[0]) * cos_t - float(target[1]) * sin_t
        dy = float(target[0]) * sin_t + float(target[1]) * cos_t
        return np.array([origin[0] + dx, origin[1] + dy], dtype=np.float32)

    def _motion_walk_target_to_field(self,
                                     agent: AgentState,
                                     value: Any,
                                     output: Optional[Any] = None) -> Optional[Tuple[np.ndarray, float]]:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        pos = self._robot_relative_target_to_field(agent, value[:2], output)
        if pos is None:
            return None
        _, base_theta = self._operational_robot_pose(agent, output)
        theta = _normalize_angle(base_theta + float(value[2]))
        return pos, theta

    def _resolve_playball_ball_target(self,
                                      agent: AgentState,
                                      ball: BallState,
                                      output: Any) -> Optional[np.ndarray]:
        skill_request = str(getattr(output, "skill_request", "none") or "none")
        motion_request = str(getattr(output, "motion_request", "none") or "none")
        macro_type = _effective_sidecar_macro_type(
            skill_request,
            getattr(output, "macro_type", "none"),
            motion_request,
        )
        operational_ball = self._operational_ball_position(
            ball,
            output,
            prefer_intercepted=True,
        )
        _, operational_theta = self._operational_robot_pose(agent, output)

        bridge_ball_target = self._bridge_field_target(
            getattr(output, "motion_ball_target_field", None),
            ball,
        )
        if bridge_ball_target is not None:
            return self._sanitize_macro_ball_target(bridge_ball_target)

        shoot_target_field = self._robot_relative_target_to_field(agent, output.behavior_shooting_to, output)
        if shoot_target_field is not None:
            return self._sanitize_macro_ball_target(shoot_target_field)

        skill_target = self._sanitize_sidecar_target(output.skill_target, ball)
        skill_target_dir = self._sanitize_angle(getattr(output, "skill_target_direction", None))
        motion_target_dir = self._sanitize_angle(getattr(output, "motion_target_direction", None))
        kick_length = self._sanitize_kick_length(getattr(output, "kick_length", 0.0))

        if skill_request in {"walk", "block", "mark", "observe"} and skill_target is not None:
            return self._clip_field_point(skill_target)

        if macro_type in {"playBall_shoot", "playBall_pass"}:
            fallback_dir = self._fallback_playball_direction(agent, ball, output)
            fallback_length = kick_length
            if skill_request == "pass" or macro_type == "playBall_pass":
                fallback_length = max(
                    700.0,
                    min(1800.0, fallback_length if fallback_length > 0.0 else 1100.0),
                )
            else:
                fallback_length = max(
                    950.0,
                    min(FIELD.length * 0.45, fallback_length if fallback_length > 0.0 else 1450.0),
                )
            target = (
                operational_ball + fallback_dir * fallback_length
            ).astype(np.float32)
            return self._sanitize_macro_ball_target(target)

        if macro_type == "playBall_dribble" and motion_request in {"walkToBallAndKick", "dribble"} and motion_target_dir is not None:
            target_theta = _normalize_angle(operational_theta + float(motion_target_dir))
            target_length = kick_length
            if motion_request == "dribble":
                target_length = max(450.0, min(700.0, target_length if target_length > 0.0 else 550.0))
            else:
                target_length = max(850.0, min(FIELD.length * 0.42, target_length if target_length > 0.0 else 1250.0))
            target = (
                operational_ball
                + np.array(
                    [math.cos(target_theta), math.sin(target_theta)],
                    dtype=np.float32,
                ) * target_length
            ).astype(np.float32)
            return self._sanitize_macro_ball_target(target)

        if skill_request == "dribble" and skill_target_dir is not None:
            target_theta = float(skill_target_dir)
            target = (
                ball.pos
                + np.array(
                    [math.cos(target_theta), math.sin(target_theta)],
                    dtype=np.float32,
                ) * 600.0
            ).astype(np.float32)
            return self._sanitize_macro_ball_target(target)

        return None

    def _fallback_playball_direction(self,
                                     agent: AgentState,
                                     ball: BallState,
                                     output: Any) -> np.ndarray:
        operational_ball = self._operational_ball_position(
            ball,
            output,
            prefer_intercepted=False,
        )
        _, operational_theta = self._operational_robot_pose(agent, output)

        skill_request = str(getattr(output, "skill_request", "none") or "none")
        motion_request = str(getattr(output, "motion_request", "none") or "none")
        macro_type = _effective_sidecar_macro_type(
            skill_request,
            getattr(output, "macro_type", "none"),
            motion_request,
        )
        motion_target_dir = self._sanitize_angle(getattr(output, "motion_target_direction", None))
        if motion_target_dir is not None:
            if macro_type == "playBall_dribble":
                theta = _normalize_angle(operational_theta + float(motion_target_dir))
                return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)

        skill_target_dir = self._sanitize_angle(getattr(output, "skill_target_direction", None))
        if skill_target_dir is not None:
            if macro_type in {"playBall_shoot", "playBall_pass"} or skill_request == "dribble":
                return np.array([math.cos(skill_target_dir), math.sin(skill_target_dir)], dtype=np.float32)

        for target in (
            self._robot_relative_target_to_field(agent, output.behavior_shooting_to, output),
            self._robot_relative_target_to_field(agent, output.behavior_walking_to, output),
            self._bridge_field_target(getattr(output, "motion_ball_target_field", None), ball),
        ):
            if target is None:
                continue
            delta = target - operational_ball
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return (delta / norm).astype(np.float32)

        return np.array(
            [math.cos(operational_theta), math.sin(operational_theta)],
            dtype=np.float32,
        )

    def _resolve_playball_body_target(self,
                                      agent: AgentState,
                                      ball: BallState,
                                      output: Any) -> Tuple[np.ndarray, float]:
        operational_ball = self._operational_ball_position(
            ball,
            output,
            prefer_intercepted=False,
        )
        bridge_contact_point = self._bridge_field_target(
            getattr(output, "motion_contact_point_field", None),
            ball,
        )
        if bridge_contact_point is not None:
            bridge_gap = float(np.linalg.norm(bridge_contact_point - operational_ball))
            if bridge_gap > 260.0:
                bridge_contact_point = None
        ball_target = self._resolve_playball_ball_target(agent, ball, output)
        contact_point = bridge_contact_point if bridge_contact_point is not None else operational_ball
        if ball_target is None:
            direction = self._fallback_playball_direction(agent, ball, output)
        else:
            direction = ball_target - contact_point
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-6:
                direction = self._fallback_playball_direction(agent, ball, output)
            else:
                direction = direction / norm
        contact_offset = AGENT.radius + BALL.radius + 35.0
        margin = 80.0
        max_offset = self._max_support_offset(contact_point, direction, margin=margin)
        support_offset = min(contact_offset, max_offset)

        if support_offset >= contact_offset * 0.72:
            body_target = contact_point - direction * support_offset
            body_target = self._clip_field_point(body_target, margin=margin)
        else:
            body_target = self._side_support_pose(
                agent,
                contact_point,
                direction,
                margin=margin,
                preferred_offset=contact_offset,
            )
        target_theta = math.atan2(direction[1], direction[0])
        return body_target.astype(np.float32), target_theta

    def _resolve_reference_target(self,
                                  agent: AgentState,
                                  ball: BallState,
                                  output: Any) -> Optional[np.ndarray]:
        skill_request = str(getattr(output, "skill_request", "none") or "none")
        motion_request = str(getattr(output, "motion_request", "none") or "none")
        macro_type = _effective_sidecar_macro_type(
            skill_request,
            getattr(output, "macro_type", "none"),
            motion_request,
        )
        in_playball = macro_type.startswith("playBall_")
        shoot_target = self._robot_relative_target_to_field(agent, output.behavior_shooting_to, output)
        skill_target = self._sanitize_sidecar_target(output.skill_target, ball)
        walk_target = self._robot_relative_target_to_field(agent, output.behavior_walking_to, output)
        motion_walk_target = self._motion_walk_target_to_field(agent, output.motion_walk_target, output)

        # During real takeover/playBall the strategic walk target can still be
        # populated by BHuman, but it should not override the local execution
        # target. Use the ball-direction semantics that BHuman actually emits.
        if in_playball or skill_request in {"shoot", "pass", "dribble"} or motion_request in {"walkToBallAndKick", "dribble"}:
            target_pos, _ = self._resolve_playball_body_target(agent, ball, output)
            return target_pos

        if skill_target is not None:
            return skill_target
        if walk_target is not None:
            return walk_target
        if motion_walk_target is not None:
            return motion_walk_target[0]
        if shoot_target is not None:
            return shoot_target
        if motion_request in {"walkToBallAndKick", "dribble"}:
            return ball.pos.copy()
        return None

    def _resolve_motion_target(self,
                               agent: AgentState,
                               ball: BallState,
                               output: Any,
                               dt: float) -> Tuple[np.ndarray, float]:
        skill_request = str(getattr(output, "skill_request", "none") or "none")
        motion_request = str(getattr(output, "motion_request", "none") or "none")
        macro_type = _effective_sidecar_macro_type(
            skill_request,
            getattr(output, "macro_type", "none"),
            motion_request,
        )
        in_playball = macro_type.startswith("playBall_")
        target_pos = self._resolve_reference_target(agent, ball, output)
        if target_pos is None:
            vx, vy, _ = output.motion_speed
            if abs(vx) > 1e-6 or abs(vy) > 1e-6:
                cos_t = math.cos(agent.theta)
                sin_t = math.sin(agent.theta)
                vx_field = vx * cos_t - vy * sin_t
                vy_field = vx * sin_t + vy * cos_t
                target_pos = agent.pos + np.array([vx_field, vy_field], dtype=np.float32) * max(dt, 0.2)
            else:
                target_pos = agent.pos.copy()

        if in_playball or skill_request in {"shoot", "pass", "dribble"} or motion_request in {"walkToBallAndKick", "dribble"}:
            target_pos, target_theta = self._resolve_playball_body_target(agent, ball, output)
        elif output.motion_walk_target is not None:
            motion_walk_target = self._motion_walk_target_to_field(agent, output.motion_walk_target, output)
            if motion_walk_target is not None:
                target_theta = motion_walk_target[1]
            else:
                target_theta = agent.theta
        else:
            motion_target_dir = self._sanitize_angle(output.motion_target_direction)
            skill_target_dir = self._sanitize_angle(output.skill_target_direction)
            _, operational_theta = self._operational_robot_pose(agent, output)
            if motion_target_dir is not None:
                target_theta = _normalize_angle(operational_theta + motion_target_dir)
            elif skill_target_dir is not None:
                target_theta = float(skill_target_dir)
            elif output.behavior_shooting_to is not None:
                shoot_to = self._robot_relative_target_to_field(agent, output.behavior_shooting_to, output)
                if shoot_to is None:
                    shoot_to = agent.pos.copy()
                target_theta = math.atan2(shoot_to[1] - agent.pos[1], shoot_to[0] - agent.pos[0])
            else:
                delta = target_pos - agent.pos
                if float(np.linalg.norm(delta)) > 1e-6:
                    target_theta = math.atan2(delta[1], delta[0])
                else:
                    target_theta = agent.theta

        return target_pos, target_theta

    def _has_new_kick(self, agent_id: str, output: Any) -> bool:
        if output.last_kick_type in (None, "", "none", "unavailable"):
            return False
        timestamp = int(output.last_kick_timestamp or 0)
        if timestamp <= 0:
            return False
        last_timestamp = self._last_kick_timestamps.get(agent_id, -1)
        if timestamp <= last_timestamp:
            return False
        self._last_kick_timestamps[agent_id] = timestamp
        return True

    def _apply_kick(self,
                    agent: AgentState,
                    ball: BallState,
                    output: Any,
                    agents: Dict[str, AgentState]) -> BallState:
        contact_offset = AGENT.min_robot_distance + BALL.radius + 25.0
        direction = self._resolve_kick_direction(agent, ball, output, agents)
        direction = self._constrain_contact_direction(agent, direction, contact_offset)
        speed = self._kick_speed(output)
        return BallState(
            pos=agent.pos + direction * contact_offset,
            vel=direction * speed,
            last_touched_by=agent.agent_id,
            seen_age_ms=0.0,
        )

    def _apply_dribble(self,
                       agent: AgentState,
                       ball: BallState,
                       output: Any) -> Tuple[BallState, bool]:
        distance = float(np.linalg.norm(agent.pos - ball.pos))
        dribble_capture_distance = AGENT.radius + BALL.radius + 35.0
        if distance > dribble_capture_distance:
            return ball, False

        contact_offset = AGENT.radius + BALL.radius + 10.0
        direction = self._dribble_direction(agent, ball, output)
        direction = self._constrain_contact_direction(agent, direction, contact_offset)
        dribble_speed = max(
            150.0,
            min(self._kick_info.ball_velocity_max * 0.45,
                max(agent.speed_scalar, output.behavior_speed, 180.0)),
        )
        next_pos = agent.pos + direction * contact_offset
        dribble_ball = BallState(
            pos=next_pos,
            vel=direction * dribble_speed,
            last_touched_by=agent.agent_id,
            seen_age_ms=0.0,
        )
        progress = float(np.dot(next_pos - ball.pos, direction)) > 25.0
        return dribble_ball, progress

    def _dribble_direction(self,
                           agent: AgentState,
                           ball: BallState,
                           output: Any) -> np.ndarray:
        forward = np.array(
            [math.cos(agent.theta), math.sin(agent.theta)],
            dtype=np.float32,
        )
        target_dir = self._direction_from_agent(agent, ball, output)
        dot = float(np.dot(forward, target_dir))

        if dot >= 0.85:
            return forward
        if dot <= 0.0:
            return forward

        blended = 0.78 * forward + 0.22 * target_dir
        norm = float(np.linalg.norm(blended))
        if norm > 1e-6:
            return blended / norm
        return forward

    def _resolve_kick_direction(self,
                                agent: AgentState,
                                ball: BallState,
                                output: Any,
                                agents: Dict[str, AgentState]) -> np.ndarray:
        if output.pass_target_player_number is not None:
            target_agent = self._agent_from_player_number(output.pass_target_player_number, agents)
            if target_agent is not None:
                delta = target_agent.pos - agent.pos
                norm = float(np.linalg.norm(delta))
                if norm > 1e-6:
                    return delta / norm

        ball_target = self._resolve_playball_ball_target(agent, ball, output)
        if ball_target is not None:
            delta = ball_target - ball.pos
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return delta / norm

        skill_target = self._sanitize_sidecar_target(output.skill_target, ball)
        if skill_target is not None:
            target = skill_target
            delta = target - agent.pos
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return delta / norm

        return self._direction_from_agent(agent, ball=ball, output=output)

    def _constrain_contact_direction(self,
                                     agent: AgentState,
                                     direction: np.ndarray,
                                     contact_offset: float) -> np.ndarray:
        candidate = agent.pos + direction * contact_offset
        margin = BALL.radius + 50.0
        safe_x = float(np.clip(
            candidate[0],
            FIELD.x_own_ground + margin,
            FIELD.x_opponent_ground - margin,
        ))
        safe_y = float(np.clip(
            candidate[1],
            FIELD.y_right_sideline + margin,
            FIELD.y_left_sideline - margin,
        ))
        safe_delta = np.array([safe_x, safe_y], dtype=np.float32) - agent.pos
        norm = float(np.linalg.norm(safe_delta))
        if norm > 1e-6:
            return safe_delta / norm
        return direction

    def _direction_from_agent(self,
                              agent: AgentState,
                              ball: Optional[BallState],
                              output: Any) -> np.ndarray:
        reference = self._resolve_reference_target(
            agent,
            ball if ball is not None else BallState(
                pos=agent.pos.copy(),
                vel=np.zeros(2, dtype=np.float32),
            ),
            output,
        )
        if reference is not None:
            delta = reference - agent.pos
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return delta / norm

        ball_target = self._resolve_playball_ball_target(agent, ball if ball is not None else BallState(
            pos=agent.pos.copy(),
            vel=np.zeros(2, dtype=np.float32),
        ), output)
        if ball is not None and ball_target is not None:
            delta = ball_target - ball.pos
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return delta / norm

        if ball is not None:
            delta = ball.pos - agent.pos
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return delta / norm

        motion_target_dir = self._sanitize_angle(output.motion_target_direction)
        skill_target_dir = self._sanitize_angle(output.skill_target_direction)
        if motion_target_dir is not None:
            _, operational_theta = self._operational_robot_pose(agent, output)
            theta = _normalize_angle(operational_theta + motion_target_dir)
        elif skill_target_dir is not None:
            theta = float(skill_target_dir)
        else:
            theta = agent.theta
        return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)

    def _agent_from_player_number(self,
                                  player_number: Any,
                                  agents: Dict[str, AgentState]) -> Optional[AgentState]:
        try:
            index = int(player_number) - 1
        except (TypeError, ValueError):
            return None

        ordered_ids = sorted(
            agents.keys(),
            key=lambda aid: int(aid.rsplit("_", 1)[-1]) if "_" in aid else 10_000,
        )
        if 0 <= index < len(ordered_ids):
            return agents.get(ordered_ids[index])
        return None

    def _kick_speed(self, output: Any) -> float:
        kick_key = output.last_kick_type
        if kick_key in (None, "", "none", "unavailable"):
            kick_key = output.kick_type

        entry = self._kick_entries.get(str(kick_key))
        kick_length = self._sanitize_kick_length(output.kick_length)
        if entry is not None:
            span = max(1.0, entry.range_max - entry.range_min)
            alpha = float(np.clip((kick_length - entry.range_min) / span, 0.0, 1.0))
            return float(entry.ball_velocity_min + alpha * (entry.ball_velocity_max - entry.ball_velocity_min))

        if kick_length <= 0.0:
            return float(self._kick_info.ball_velocity_max)

        ratio = float(np.clip(kick_length / max(1.0, self._kick_info.forward_fast_max), 0.35, 1.0))
        return ratio * float(self._kick_info.ball_velocity_max)

    def post_kick_release_frames(self, agent_id: Optional[str]) -> int:
        if not agent_id:
            return 0
        return int(self._post_kick_release_frames.get(agent_id, 0))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class NaoSoccerEnv(ParallelEnv):
    """
    Entorno 2D para MAPPO NAO Soccer, alineado con contrato BHuman 4v4 Scaled.

    Handoff/takeover flow:
      - RL agents output positioning actions (NOT kick/dribble)
      - When handoff_ready, BHuman takes over for skill execution
      - Reward based on quality of handoff state and BHuman outcome
    """

    metadata = {"render_modes": ["human"], "name": "nao_soccer_v0"}

    def __init__(self, stage_cfg: CurriculumStageConfig,
                 use_fov: bool = True,
                 seed: Optional[int] = None):
        super().__init__()

        self._stage = stage_cfg
        self._use_fov = use_fov
        self._rng = np.random.default_rng(seed)
        random.seed(seed)

        # Components
        self._ball_physics = BallPhysics()
        self._movement     = NaoMovementModel()
        self._movement.configure(self._stage)
        self._fov_filter   = FOVFilter(use_fov=use_fov)
        self._scripted_ai  = ScriptedOpponent(self._movement)
        self._obs_builder  = ObservationBuilder()
        self._handoff_mgr  = HandoffTakeoverManager(CONTRACT)
        self._handoff_mgr.set_takeover_timeout_frames(self._stage.takeover_timeout_steps)

        # Execution mode
        self._execution_mode = EXECUTION_MODE
        self._sidecar = None

        # Takeover simulator: ONLY for python_stub_smoke_only mode.
        # In bhuman_sidecar mode, the takeover simulator is NOT used.
        self._takeover_sim = None
        self._sidecar_physics = None
        if EXECUTION_MODE == "python_stub_smoke_only":
            self._takeover_sim = BHumanTakeoverSimulator(self._movement, CONTRACT)

        # Sidecar: BHuman decision chain
        if EXECUTION_MODE == "bhuman_sidecar":
            from bhuman_sidecar import BHumanDecisionSidecar
            self._sidecar = BHumanDecisionSidecar(CONTRACT,
                                                   execution_mode=EXECUTION_MODE)
            self._sidecar_physics = RealSidecarPhysicsExecutor(self._movement, CONTRACT)

        # PettingZoo
        self.possible_agents = [f"agent_{i}" for i in range(SIM.num_agents)]
        self.observation_spaces = {
            aid: spaces.Box(low=-np.inf, high=np.inf,
                            shape=(OBS.total,), dtype=np.float32)
            for aid in self.possible_agents
        }
        self.action_spaces = {
            aid: spaces.Box(low=-1.0, high=1.0,
                            shape=(ACTION.size,), dtype=np.float32)
            for aid in self.possible_agents
        }

        # State (initialized in reset)
        self._agents_state: Dict[str, AgentState] = {}
        self._opponents_state: List[OpponentState] = []
        self._static_obstacles: List[StaticObstacle] = []
        self._ball: BallState = BallState(np.zeros(2), np.zeros(2))
        self._game: GameState = GameState()
        self._prev_ball_pos: np.ndarray = np.zeros(2)
        self._prev_ball_speed: float = 0.0
        self._prev_ball_dists: Dict[str, float] = {}
        self._current_tactic: str = "t121_4v4_scaled"
        self._active_agent_id: Optional[str] = None
        self._ball_collision_skip_agent_id: Optional[str] = None

        # Smoothing and tracking
        self._prev_actions: Dict[str, np.ndarray] = {}
        self._ema_actions: Dict[str, np.ndarray] = {}
        self._prev_thetas: Dict[str, float] = {}
        self._prev_targets: Dict[str, np.ndarray] = {}
        self._curr_targets: Dict[str, np.ndarray] = {}
        self._prev_omegas: Dict[str, float] = {}
        self._prev_face_thetas: Dict[str, float] = {}

        # Episode tracking
        self._ball_out_this_step = False
        self._takeover_events: List[Dict] = []  # for metrics
        self._episode_takeover_count = 0
        self._episode_takeover_success = 0
        self._episode_takeover_count_by_type = _new_takeover_type_counts()
        self._episode_takeover_success_by_type = _new_takeover_type_counts()
        self._episode_handoff_achieved = False  # True if handoff_ready was ever True
        self._episode_handoff_ready_frames = 0
        self._episode_tkready_achieved = False
        self._episode_tkready_frames = 0
        self._episode_playball_entry_count = 0
        self._episode_playball_achieved = False
        self._episode_ready_but_walk_frames = 0
        self._episode_ready_touch_but_walk_frames = 0
        self._episode_takeover_reason_counts: Dict[str, int] = _new_takeover_reason_counts()
        self._last_macro_end_reason: str = "none"
        self._prev_playball_frame = False
        self._prev_handoff_ready = False
        self._tkready_streak = 0

        self.agents: List[str] = []

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        n_rl = self._stage.num_rl_agents
        self.agents = self.possible_agents[:n_rl]

        self._game = GameState()
        self._agents_state = self._spawn_agents(self.agents)
        self._opponents_state = self._spawn_opponents(self._stage.num_opponents)
        self._ball = self._spawn_ball()
        self._prev_ball_pos = self._ball.pos.copy()
        self._prev_ball_speed = 0.0
        self._current_tactic = "t121_4v4_scaled"

        # Select active agent (closest to ball)
        self._select_active_agent()
        self._static_obstacles = self._spawn_static_obstacles()

        # Init distances
        self._prev_ball_dists = {
            aid: float(np.linalg.norm(self._agents_state[aid].pos - self._ball.pos))
            for aid in self.agents
        }

        self._fov_filter.reset(self.agents)
        self._handoff_mgr.reset()
        self._handoff_mgr.set_takeover_timeout_frames(self._stage.takeover_timeout_steps)
        if self._sidecar is not None:
            # Reset the real bridge / sidecar state at episode boundaries.
            self._sidecar.reset()
        if self._sidecar_physics is not None:
            self._sidecar_physics.reset()

        # Init smoothing
        self._prev_actions = {aid: np.zeros(ACTION.size, dtype=np.float32) for aid in self.agents}
        self._ema_actions = {aid: np.zeros(ACTION.size, dtype=np.float32) for aid in self.agents}
        self._prev_thetas = {aid: self._agents_state[aid].theta for aid in self.agents}
        self._prev_targets = {aid: self._agents_state[aid].pos.copy() for aid in self.agents}
        self._curr_targets = {}
        self._prev_omegas = {aid: 0.0 for aid in self.agents}
        self._prev_face_thetas = {aid: self._agents_state[aid].theta for aid in self.agents}

        # Episode metrics
        self._ball_out_this_step = False
        self._takeover_events = []
        self._episode_takeover_count = 0
        self._episode_takeover_success = 0
        self._episode_takeover_count_by_type = _new_takeover_type_counts()
        self._episode_takeover_success_by_type = _new_takeover_type_counts()
        self._episode_handoff_achieved = False
        self._episode_handoff_ready_frames = 0
        self._episode_tkready_achieved = False
        self._episode_tkready_frames = 0
        self._episode_playball_entry_count = 0
        self._episode_playball_achieved = False
        self._episode_ready_but_walk_frames = 0
        self._episode_ready_touch_but_walk_frames = 0
        self._episode_takeover_reason_counts = _new_takeover_reason_counts()
        self._last_macro_end_reason = "none"
        self._prev_playball_frame = False
        self._prev_handoff_ready = False
        self._tkready_streak = 0
        self._ball_collision_skip_agent_id = None

        observations = self._build_observations()
        infos = {aid: {} for aid in self.agents}
        return observations, infos

    def step(self, actions):
        assert self.agents, "step() called without active agents. Call reset() first."

        self._ball_out_this_step = False

        # Save previous state for reward
        self._prev_ball_pos = self._ball.pos.copy()
        self._prev_ball_speed = float(np.linalg.norm(self._ball.vel))
        prev_ball_dists = {
            aid: float(np.linalg.norm(self._agents_state[aid].pos - self._ball.pos))
            for aid in self.agents
        }

        # ═══════════════════════════════════════════════════════════════
        # 1. RL CONTROL STEP — agents move according to actions
        # ═══════════════════════════════════════════════════════════════
        current_targets = {}
        for aid in self.agents:
            if aid not in actions:
                continue

            raw_action = actions[aid]
            if not np.all(np.isfinite(raw_action)):
                raw_action = self._prev_actions.get(aid, np.zeros(ACTION.size, dtype=np.float32))

            # EMA smoothing
            if ACTION_SMOOTHING.enabled:
                stage_alpha = getattr(self._stage, "ema_alpha", ACTION_SMOOTHING.alpha)
                prev = self._prev_actions.get(aid, np.zeros(ACTION.size, dtype=np.float32))
                action_to_use = stage_alpha * raw_action + (1.0 - stage_alpha) * prev
            else:
                action_to_use = raw_action

            self._prev_actions[aid] = action_to_use.copy()

            agent_state = self._agents_state[aid]
            target_pos, target_theta = self._decode_action(action_to_use, agent_state, aid)

            # Local horizon
            local_horizon = getattr(self._stage, "local_horizon", 0.0)
            if local_horizon > 0.0:
                agent_pos = agent_state.pos
                dist_to_target = float(np.linalg.norm(target_pos - agent_pos))
                if dist_to_target > local_horizon:
                    direction = (target_pos - agent_pos) / max(dist_to_target, 1e-8)
                    target_pos = agent_pos + local_horizon * direction

            current_targets[aid] = target_pos.copy()

            self._agents_state[aid] = self._movement.step(
                agent_state, target_pos, target_theta, SIM.dt
            )

        self._resolve_agent_obstacle_collisions()

        # ═══════════════════════════════════════════════════════════════
        # 2. SELECT ACTIVE AGENT + SIDECAR DECISION
        # ═══════════════════════════════════════════════════════════════
        sidecar_output = None
        if self._sidecar is not None:
            sidecar_output = self._sidecar.decide(
                agents=self._agents_state,
                ball=self._ball,
                opponents=self._opponents_state,
                game=self._game,
            )
            # Sidecar decides active_agent — use it
            if sidecar_output.active_agent in self._agents_state:
                self._active_agent_id = sidecar_output.active_agent
                for aid, agent in self._agents_state.items():
                    agent.is_active = (aid == self._active_agent_id)
            elif sidecar_output.active_agent is not None:
                # Ignore padded / non-trainable teammates as env owners.
                self._select_active_agent()
            elif sidecar_output.backend == "real" and self._active_agent_id in self._agents_state:
                for aid, agent in self._agents_state.items():
                    agent.is_active = (aid == self._active_agent_id)
            else:
                self._select_active_agent()
        else:
            self._select_active_agent()

        # ═══════════════════════════════════════════════════════════════
        # 3. HANDOFF CHECK + BHUMAN TAKEOVER
        # ═══════════════════════════════════════════════════════════════
        takeover_kick = False
        takeover_kick_proxy = False
        takeover_progress = False
        pending_kick_hold = False
        pending_kick_hold_frames = 0
        post_kick_release_frames = 0
        macro_resolved_pre_physics = False
        sidecar_macro_type = "none"
        playball_entry = False
        ready_but_walk = False
        ready_touch_but_walk = False
        if sidecar_output is not None:
            sidecar_macro_type = _effective_sidecar_macro_type(
                sidecar_output.skill_request,
                sidecar_output.macro_type,
                sidecar_output.motion_request,
            )

        if self._stage.takeover_required:
            macro = self._handoff_mgr.update(
                self._agents_state, self._ball, self._game,
                self._active_agent_id,
                goal_scored=False, goal_conceded=False,
                ball_out=False,
                sidecar_macro_type=sidecar_macro_type,
            )

            # Track episode-level handoff_ready
            if macro.handoff_ready:
                self._episode_handoff_achieved = True
                self._episode_handoff_ready_frames += 1

            sidecar_playball = sidecar_macro_type in TAKEOVER_MACRO_TYPES
            if sidecar_playball:
                self._episode_playball_achieved = True
            if sidecar_playball and not self._prev_playball_frame:
                self._episode_playball_entry_count += 1
                playball_entry = True
            self._prev_playball_frame = sidecar_playball

            tkready_frame = bool(
                macro.takeover_active
                and sidecar_output is not None
                and sidecar_playball
                and str(sidecar_output.skill_request or "none") != "walk"
                and bool(self._active_agent_id)
            )
            if tkready_frame:
                self._tkready_streak += 1
                self._episode_tkready_frames += 1
                if self._tkready_streak >= 2:
                    self._episode_tkready_achieved = True
            else:
                self._tkready_streak = 0

            ready_but_walk = (
                macro.handoff_ready
                and not sidecar_playball
                and sidecar_output is not None
                and sidecar_output.skill_request == "walk"
            )
            if ready_but_walk:
                self._episode_ready_but_walk_frames += 1
                if self._active_agent_id in self._agents_state:
                    active_agent = self._agents_state[self._active_agent_id]
                    touching_ball = (
                        float(np.linalg.norm(active_agent.pos - self._ball.pos))
                        <= AGENT.ball_interaction_range
                    )
                    if touching_ball:
                        ready_touch_but_walk = True
                        self._episode_ready_touch_but_walk_frames += 1

            # Takeover execution depends on mode:
            if macro.takeover_active and self._active_agent_id:
                if self._takeover_sim is not None:
                    # python_stub_smoke_only: use local physics simulator
                    agent = self._agents_state[self._active_agent_id]
                    agent, self._ball, takeover_kick = self._takeover_sim.simulate_step(
                        agent, self._ball, SIM.dt
                    )
                    self._agents_state[self._active_agent_id] = agent
                elif (
                    sidecar_output is not None
                    and sidecar_output.backend == "real"
                    and self._sidecar_physics is not None
                ):
                    agent = self._agents_state[self._active_agent_id]
                    agent, self._ball, takeover_kick, takeover_progress, pending_kick_hold, pending_kick_hold_frames = self._sidecar_physics.apply(
                        agent, self._ball, sidecar_output, self._agents_state, SIM.dt
                    )
                    self._agents_state[self._active_agent_id] = agent
                    post_kick_release_frames = self._sidecar_physics.post_kick_release_frames(
                        self._active_agent_id
                    )
                    takeover_kick_proxy = self._is_takeover_kick_proxy(
                        agent,
                        self._ball,
                        sidecar_output,
                        post_kick_release_frames=post_kick_release_frames,
                        pending_kick_hold=pending_kick_hold,
                    )
                    kick_like_macro = str(getattr(sidecar_output, "macro_type", "none") or "none") in {
                        "playBall_shoot",
                        "playBall_pass",
                    }
                    kick_like_skill = str(getattr(sidecar_output, "skill_request", "none") or "none") in {
                        "shoot",
                        "pass",
                    }
                    self._ball_collision_skip_agent_id = (
                        self._active_agent_id
                        if (
                            pending_kick_hold
                            or post_kick_release_frames > 0
                            or kick_like_macro
                            or kick_like_skill
                        )
                        else None
                    )
                    self._update_takeover_progress(
                        agent,
                        self._ball,
                        sidecar_output,
                        pending_kick_hold=pending_kick_hold,
                    )

                    if self._has_takeover_execution_evidence(
                        agent,
                        self._ball,
                        sidecar_output,
                        takeover_kick=takeover_kick,
                        takeover_kick_proxy=takeover_kick_proxy,
                        takeover_progress=takeover_progress,
                        pending_kick_hold=pending_kick_hold,
                    ):
                        self._handoff_mgr.force_end("completed", True)
                        macro_resolved_pre_physics = True
                else:
                    # contract_validation_mode keeps semantic evidence only.
                    pass

        self._resolve_agent_obstacle_collisions()

        # ═══════════════════════════════════════════════════════════════
        # 4. OPPONENT MOVEMENT
        # ═══════════════════════════════════════════════════════════════
        if self._opponents_state:
            nearest_opp_idx = self._nearest_to_ball_opponent()
            for i, opp in enumerate(self._opponents_state):
                self._opponents_state[i] = self._scripted_ai.decide_and_move(
                    opp, self._ball.pos,
                    is_nearest_to_ball=(i == nearest_opp_idx),
                    dt=SIM.dt,
                )
            # Opponent kicks if close to ball
            self._ball = self._handle_opponent_ball_interaction()

        # ═══════════════════════════════════════════════════════════════
        # 5. BALL PHYSICS
        # ═══════════════════════════════════════════════════════════════
        self._ball, ball_out = self._ball_physics.step(self._ball, SIM.dt)

        if ball_out:
            self._ball_out_this_step = True
            self._ball = BallState(
                pos=np.zeros(2, dtype=np.float32),
                vel=np.zeros(2, dtype=np.float32),
            )

        # Collision resolution
        self._ball = self._resolve_ball_collisions()
        self._ball_collision_skip_agent_id = None

        # ═══════════════════════════════════════════════════════════════
        # 6. GOAL CHECK
        # ═══════════════════════════════════════════════════════════════
        goal_scored = False
        goal_conceded = False
        goal_result = self._ball_physics.check_goal(self._ball)
        if goal_result == "own":
            self._game.own_score += 1
            goal_scored = True
            self._ball = self._spawn_ball(reason="goal_reset")
        elif goal_result == "opp":
            self._game.opp_score += 1
            goal_conceded = True
            self._ball = self._spawn_ball(reason="goal_reset")

        # ═══════════════════════════════════════════════════════════════
        # 7. POST-TAKEOVER UPDATE
        # ═══════════════════════════════════════════════════════════════
        if self._stage.takeover_required:
            if macro_resolved_pre_physics:
                macro = self._handoff_mgr.macro
            else:
                macro = self._handoff_mgr.update(
                    self._agents_state, self._ball, self._game,
                    self._active_agent_id,
                    goal_scored=goal_scored, goal_conceded=goal_conceded,
                    ball_out=self._ball_out_this_step,
                    sidecar_macro_type=sidecar_macro_type,
                )
            self._recover_cleared_macro_success(
                macro,
                takeover_kick=takeover_kick,
                takeover_kick_proxy=takeover_kick_proxy,
            )
            # Record takeover events
            if (
                macro.macro_end_reason != "none"
                and macro.macro_success is not None
                and not macro.event_recorded
            ):
                self._episode_takeover_count += 1
                if macro.macro_type in self._episode_takeover_count_by_type:
                    self._episode_takeover_count_by_type[macro.macro_type] += 1
                if macro.macro_success:
                    self._episode_takeover_success += 1
                    if macro.macro_type in self._episode_takeover_success_by_type:
                        self._episode_takeover_success_by_type[macro.macro_type] += 1
                self._takeover_events.append({
                    "type": macro.macro_type,
                    "duration": macro.macro_duration_frames,
                    "end_reason": macro.macro_end_reason,
                    "success": macro.macro_success,
                })
                reason_key = (
                    macro.macro_end_reason
                    if macro.macro_end_reason in self._episode_takeover_reason_counts
                    else "other"
                )
                self._episode_takeover_reason_counts[reason_key] += 1
                self._last_macro_end_reason = macro.macro_end_reason
                macro.event_recorded = True

        # ═══════════════════════════════════════════════════════════════
        # 8. UPDATE GAME STATE
        # ═══════════════════════════════════════════════════════════════
        self._game.step_count += 1
        self._game.time_elapsed = self._game.step_count * SIM.dt

        # ═══════════════════════════════════════════════════════════════
        # 9. COMPUTE REWARDS
        # ═══════════════════════════════════════════════════════════════
        rewards = self._compute_rewards(
            goal_scored=goal_scored,
            goal_conceded=goal_conceded,
            prev_ball_dists=prev_ball_dists,
            takeover_kick=(takeover_kick or takeover_kick_proxy),
            playball_entry=playball_entry,
            ready_but_walk=ready_but_walk,
            ready_touch_but_walk=ready_touch_but_walk,
        )

        # Update tracking
        self._prev_ball_dists = {
            aid: float(np.linalg.norm(self._agents_state[aid].pos - self._ball.pos))
            for aid in self.agents
        }
        self._prev_thetas = {aid: self._agents_state[aid].theta for aid in self.agents}
        self._prev_omegas = {aid: self._agents_state[aid].omega for aid in self.agents}
        self._curr_targets = current_targets.copy()
        self._prev_targets = current_targets.copy()
        self._prev_handoff_ready = self._handoff_mgr.macro.handoff_ready

        # ═══════════════════════════════════════════════════════════════
        # 10. EPISODE TERMINATION
        # ═══════════════════════════════════════════════════════════════
        live_agent_ids = list(self.agents)
        truncated = self._game.step_count >= self._stage.episode_steps
        terminated = False

        terminateds = {aid: terminated for aid in self.agents}
        truncateds = {aid: truncated for aid in self.agents}
        terminateds["__all__"] = terminated
        truncateds["__all__"] = truncated

        observations = self._build_observations(agent_ids=live_agent_ids)
        infos = self._build_infos(
            goal_scored,
            goal_conceded,
            sidecar_output=sidecar_output,
            agent_ids=live_agent_ids,
            playball_entry=playball_entry,
            ready_but_walk=ready_but_walk,
            ready_touch_but_walk=ready_touch_but_walk,
            pending_kick_hold=pending_kick_hold,
            pending_kick_hold_frames=pending_kick_hold_frames,
            takeover_kick=takeover_kick,
            takeover_kick_proxy=takeover_kick_proxy,
            post_kick_release_frames=post_kick_release_frames,
        )

        if truncated:
            self.agents = []

        return observations, rewards, terminateds, truncateds, infos

    # ── Active agent selection ─────────────────────────────────────────

    def _select_active_agent(self):
        """Select the closest upright agent to ball as active."""
        min_dist = float("inf")
        best = None
        for aid, agent in self._agents_state.items():
            if not agent.is_upright:
                continue
            dist = float(np.linalg.norm(agent.pos - self._ball.pos))
            if dist < min_dist:
                min_dist = dist
                best = aid

        self._active_agent_id = best
        for aid, agent in self._agents_state.items():
            agent.is_active = (aid == best)

    # ── Spawning ──────────────────────────────────────────────────────

    def _spawn_agents(self, agent_ids):
        # Use contract-based home poses
        home_poses = get_home_poses(self._current_tactic, CONTRACT)
        role_order = ["defenderL", "defenderR", "forwardL", "forwardR"]

        states = {}
        for i, aid in enumerate(agent_ids):
            role = role_order[i] if i < len(role_order) else role_order[0]
            home = home_poses.get(role, (0.0, 0.0))
            noise = self._rng.uniform(-150, 150, size=2)
            pos = np.array([home[0], home[1]], dtype=np.float32) + noise

            states[aid] = AgentState(
                agent_id=aid,
                pos=pos,
                theta=0.0,
                vel=np.zeros(2),
                omega=0.0,
                role=role,
                walk_target=pos.copy(),
            )
        return states

    def _spawn_opponents(self, num_opponents):
        opponents = []
        base_positions = [
            np.array([500.0, 400.0]),
            np.array([500.0, -400.0]),
            np.array([1500.0, 600.0]),
            np.array([1500.0, -600.0]),
        ]
        for i in range(min(num_opponents, len(base_positions))):
            home = base_positions[i]
            noise = self._rng.uniform(-200, 200, size=2)
            opponents.append(OpponentState(
                opp_id=f"opp_{i}",
                pos=home + noise,
                theta=math.pi,
                vel=np.zeros(2),
                home_pos=home.copy(),
            ))
        return opponents

    def _spawn_ball(self, reason: str = "reset"):
        if not getattr(self, "_agents_state", None):
            return BallState(pos=np.zeros(2, dtype=np.float32), vel=np.zeros(2, dtype=np.float32))

        ref_aid = self._active_agent_id if self._active_agent_id in self._agents_state else None
        if ref_aid is None:
            ref_aid = self.agents[0] if self.agents else next(iter(self._agents_state.keys()))
        ref_agent = self._agents_state[ref_aid]

        if reason == "goal_reset":
            min_goal_reset_dist = max(900.0, 0.28 * FIELD.diagonal)
            for _ in range(200):
                pos = np.array(
                    [
                        float(self._rng.uniform(-0.18 * FIELD.x_opponent_ground, 0.22 * FIELD.x_opponent_ground)),
                        float(self._rng.uniform(-0.42 * FIELD.y_left_sideline, 0.42 * FIELD.y_left_sideline)),
                    ],
                    dtype=np.float32,
                )
                pos[0] = float(np.clip(pos[0], FIELD.x_own_ground + 350.0, FIELD.x_opponent_ground - 350.0))
                pos[1] = float(np.clip(pos[1], FIELD.y_right_sideline + 350.0, FIELD.y_left_sideline - 350.0))
                if float(np.linalg.norm(pos - ref_agent.pos)) >= min_goal_reset_dist:
                    return BallState(pos=pos.astype(np.float32), vel=np.zeros(2, dtype=np.float32))

        max_dist = getattr(self._stage, "ball_max_dist", -1.0)
        if max_dist <= 0.0:
            max_dist = FIELD.diagonal * 0.5

        front_only = getattr(self._stage, "spawn_ball_in_front", False)
        half_angle_deg = getattr(self._stage, "spawn_front_angle_deg", 180.0)

        for _ in range(200):
            r = float(self._rng.uniform(150.0, max_dist))
            if front_only:
                rel = math.radians(float(self._rng.uniform(-half_angle_deg, half_angle_deg)))
                ang = ref_agent.theta + rel
            else:
                ang = float(self._rng.uniform(-math.pi, math.pi))

            pos = ref_agent.pos + r * np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
            pos[0] = np.clip(pos[0], FIELD.x_own_ground + 200.0, FIELD.x_opponent_ground - 200.0)
            pos[1] = np.clip(pos[1], FIELD.y_right_sideline + 200.0, FIELD.y_left_sideline - 200.0)

            if np.linalg.norm(pos - ref_agent.pos) <= max_dist + 1e-6:
                return BallState(pos=pos.astype(np.float32), vel=np.zeros(2, dtype=np.float32))

        return BallState(pos=np.zeros(2, dtype=np.float32), vel=np.zeros(2, dtype=np.float32))

    def _spawn_static_obstacles(self) -> List[StaticObstacle]:
        if self._stage.num_static_obstacles <= 0 or not self._agents_state:
            return []

        if self._active_agent_id and self._active_agent_id in self._agents_state:
            ref_agent = self._agents_state[self._active_agent_id]
        else:
            ref_agent = next(iter(self._agents_state.values()))

        diff = self._ball.pos - ref_agent.pos
        dist = float(np.linalg.norm(diff))
        if dist < 600.0:
            return []

        direction = diff / max(dist, 1e-6)
        lateral = np.array([-direction[1], direction[0]], dtype=np.float32)
        obstacles: List[StaticObstacle] = []
        fractions = np.linspace(0.42, 0.68, self._stage.num_static_obstacles)

        for idx, fraction in enumerate(fractions):
            base = ref_agent.pos + direction * dist * float(fraction)
            lateral_mag = float(self._rng.uniform(
                OBSTACLE.path_offset_min,
                OBSTACLE.path_offset_max,
            ))
            lateral_sign = 1.0 if idx % 2 == 0 else -1.0
            pos = base + lateral * lateral_sign * lateral_mag
            pos[0] = float(np.clip(
                pos[0],
                FIELD.x_own_ground + OBSTACLE.spawn_margin,
                FIELD.x_opponent_ground - OBSTACLE.spawn_margin,
            ))
            pos[1] = float(np.clip(
                pos[1],
                FIELD.y_right_sideline + OBSTACLE.spawn_margin,
                FIELD.y_left_sideline - OBSTACLE.spawn_margin,
            ))

            if (
                np.linalg.norm(pos - ref_agent.pos) < 550.0
                or np.linalg.norm(pos - self._ball.pos) < 450.0
            ):
                continue

            obstacles.append(StaticObstacle(
                obstacle_id=f"static_obs_{idx}",
                pos=pos.astype(np.float32),
                radius=OBSTACLE.radius,
            ))

        return obstacles

    def _resolve_agent_obstacle_collisions(self):
        if not self._static_obstacles:
            return

        for aid, agent in self._agents_state.items():
            corrected = agent
            for obstacle in self._static_obstacles:
                diff = corrected.pos - obstacle.pos
                dist = float(np.linalg.norm(diff))
                min_sep = AGENT.radius + obstacle.radius
                if dist < min_sep:
                    if dist <= 1e-6:
                        push_dir = np.array(
                            [math.cos(corrected.theta), math.sin(corrected.theta)],
                            dtype=np.float32,
                        )
                    else:
                        push_dir = diff / dist
                    new_pos = obstacle.pos + push_dir * (min_sep + 5.0)
                    new_pos[0] = float(np.clip(
                        new_pos[0],
                        FIELD.x_own_ground - 500.0,
                        FIELD.x_opponent_ground + 500.0,
                    ))
                    new_pos[1] = float(np.clip(
                        new_pos[1],
                        FIELD.y_right_sideline - 500.0,
                        FIELD.y_left_sideline + 500.0,
                    ))
                    corrected = AgentState(
                        agent_id=corrected.agent_id,
                        pos=new_pos.astype(np.float32),
                        theta=corrected.theta,
                        vel=np.zeros(2, dtype=np.float32),
                        omega=corrected.omega,
                        role=corrected.role,
                        is_upright=corrected.is_upright,
                        recovery_steps=corrected.recovery_steps,
                        battery=corrected.battery,
                        walk_target=corrected.walk_target.copy(),
                        speed_scalar=0.0,
                        is_active=corrected.is_active,
                    )
            self._agents_state[aid] = corrected

    def _guided_ball_target(self, agent: AgentState) -> np.ndarray:
        # Guide the agent to a body pose *behind the ball toward the opponent
        # goal*, not just behind the ball along the current chase line. This
        # makes the fallback behavior more offensive and reduces the incentive
        # to push the ball out of bounds to force a center reset.
        goal_center = np.array([FIELD.x_opponent_ground, 0.0], dtype=np.float32)
        goal_vec = goal_center - self._ball.pos
        # Pull slightly toward midfield near the sidelines so the preferred
        # approach lane stays in-bounds instead of hugging the line.
        sideline_bias = float(np.clip(-0.22 * self._ball.pos[1], -450.0, 450.0))
        goal_vec[1] += sideline_bias
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist > 1e-6:
            approach_dir = goal_vec / goal_dist
        else:
            ball_vec = self._ball.pos - agent.pos
            dist_to_ball = float(np.linalg.norm(ball_vec))
            if dist_to_ball > 1e-6:
                approach_dir = ball_vec / dist_to_ball
            else:
                approach_dir = np.array(
                    [math.cos(agent.theta), math.sin(agent.theta)],
                    dtype=np.float32,
                )

        desired_rel_x = float(getattr(
            self._stage,
            "handoff_target_rel_x_mm",
            0.5 * (
                float(self._handoff_mgr._handoff_cfg.ball_rel_x_min)
                + float(self._handoff_mgr._handoff_cfg.ball_rel_x_max)
            ),
        ))
        target = (
            self._ball.pos - approach_dir.astype(np.float32) * desired_rel_x
        ).astype(np.float32)
        target[0] = float(np.clip(
            target[0],
            FIELD.x_own_ground + 300.0,
            FIELD.x_opponent_ground - 300.0,
        ))
        target[1] = float(np.clip(
            target[1],
            FIELD.y_right_sideline + 300.0,
            FIELD.y_left_sideline - 300.0,
        ))
        return target

    # ── Action decoding ───────────────────────────────────────────────

    def _decode_action(self, action, agent, agent_id=""):
        max_disp = getattr(self._stage, 'local_horizon', ACTION.max_displacement_mm)
        if max_disp <= 0.0:
            max_disp = ACTION.max_displacement_mm

        dx_mm = float(action[0]) * max_disp
        dy_mm = float(action[1]) * max_disp
        target_pos = agent.pos + np.array([dx_mm, dy_mm], dtype=np.float32)

        target_pos[0] = float(np.clip(target_pos[0],
                              FIELD.x_own_ground + 300.0, FIELD.x_opponent_ground - 300.0))
        target_pos[1] = float(np.clip(target_pos[1],
                              FIELD.y_right_sideline + 300.0, FIELD.y_left_sideline - 300.0))

        target_alpha = float(getattr(self._stage, "target_pos_alpha", 1.0))
        if agent_id and 0.0 < target_alpha < 1.0:
            prev_target = self._prev_targets.get(agent_id, agent.pos.copy())
            target_pos = (
                target_alpha * target_pos
                + (1.0 - target_alpha) * np.asarray(prev_target, dtype=np.float32)
            ).astype(np.float32)
            target_pos[0] = float(np.clip(
                target_pos[0],
                FIELD.x_own_ground + 300.0,
                FIELD.x_opponent_ground - 300.0,
            ))
            target_pos[1] = float(np.clip(
                target_pos[1],
                FIELD.y_right_sideline + 300.0,
                FIELD.y_left_sideline - 300.0,
            ))

        if (
            self._stage.takeover_required
            and agent_id
            and agent_id == self._active_agent_id
        ):
            guided_target = self._guided_ball_target(agent)
            dist_to_ball = float(np.linalg.norm(self._ball.pos - agent.pos))
            chase_blend = float(getattr(self._stage, "ball_chase_blend", 0.0))
            pose_blend = float(getattr(self._stage, "handoff_pose_blend", 0.0))
            near_ball = dist_to_ball <= AGENT.ball_interaction_range * 1.15
            guidance_blend = pose_blend if near_ball else chase_blend
            if guidance_blend > 0.0:
                target_pos = (
                    guidance_blend * guided_target
                    + (1.0 - guidance_blend) * target_pos
                ).astype(np.float32)

            max_ball_radius = float(
                getattr(self._stage, "ball_target_max_radius_mm", 0.0)
            )
            if max_ball_radius > 0.0:
                target_offset = target_pos - self._ball.pos
                offset_norm = float(np.linalg.norm(target_offset))
                if offset_norm > max_ball_radius and offset_norm > 1e-6:
                    target_pos = (
                        self._ball.pos
                        + target_offset * (max_ball_radius / offset_norm)
                    ).astype(np.float32)

            target_pos[0] = float(np.clip(
                target_pos[0],
                FIELD.x_own_ground + 300.0,
                FIELD.x_opponent_ground - 300.0,
            ))
            target_pos[1] = float(np.clip(
                target_pos[1],
                FIELD.y_right_sideline + 300.0,
                FIELD.y_left_sideline - 300.0,
            ))

        if self._stage.theta_free:
            target_theta = float(action[2]) * ACTION.theta_range
        else:
            # Face-ball with hysteresis. Once the robot is in a valid handoff
            # pose, keep the heading stable instead of chasing tiny ball-angle
            # fluctuations frame to frame.
            bdx = self._ball.pos[0] - agent.pos[0]
            bdy = self._ball.pos[1] - agent.pos[1]
            dist_to_ball = math.hypot(bdx, bdy)
            raw_theta = math.atan2(bdy, bdx)
            prev_face = self._prev_face_thetas.get(agent_id, agent.theta)
            diff = math.atan2(math.sin(raw_theta - prev_face), math.cos(raw_theta - prev_face))
            in_handoff_pose = self._handoff_mgr.compute_handoff_ready(agent, self._ball, self._game)
            face_hold_dist = AGENT.ball_interaction_range * float(
                getattr(self._stage, "face_hold_dist_ratio", 0.70)
            )
            angle_deadband = math.radians(float(
                getattr(self._stage, "face_theta_deadband_deg", 6.0)
            ))
            face_alpha = float(getattr(self._stage, "face_theta_alpha", 0.18))

            if (
                dist_to_ball < face_hold_dist
                or in_handoff_pose
                or abs(diff) < angle_deadband
            ):
                target_theta = prev_face
            else:
                target_theta = _normalize_angle(prev_face + face_alpha * diff)

            if agent_id:
                self._prev_face_thetas[agent_id] = target_theta

        return target_pos, target_theta

    # ── Ball interactions ─────────────────────────────────────────────

    def _handle_opponent_ball_interaction(self):
        """Opponents kick ball if close enough."""
        ball = self._ball
        for opp in self._opponents_state:
            dist = float(np.linalg.norm(opp.pos - ball.pos))
            if dist < AGENT.ball_interaction_range * 0.7 and opp.is_upright:
                kick_theta = math.atan2(-opp.pos[1], FIELD.x_own_ground - opp.pos[0])
                kick_dir = np.array([math.cos(kick_theta), math.sin(kick_theta)])
                ball = BallState(
                    pos=opp.pos + kick_dir * (AGENT.min_robot_distance + 50),
                    vel=kick_dir * CONTRACT.kick_info.ball_velocity_max,
                    last_touched_by=None,
                )
                break
        return ball

    def _resolve_ball_collisions(self):
        ball = self._ball
        min_sep = AGENT.radius + BALL.radius  # 200 + 100 = 300mm

        for aid, agent in self._agents_state.items():
            if not agent.is_upright:
                continue
            if aid == self._ball_collision_skip_agent_id:
                continue
            diff = ball.pos - agent.pos
            dist = float(np.linalg.norm(diff))
            if dist < min_sep and dist > 1e-6:
                push_dir = diff / dist
                ball = BallState(
                    pos=ball.pos + push_dir * (min_sep - dist + 5.0),
                    vel=ball.vel.copy(),
                    last_touched_by=ball.last_touched_by,
                )
            elif dist <= 1e-6:
                push_dir = np.array([math.cos(agent.theta), math.sin(agent.theta)])
                ball = BallState(
                    pos=agent.pos + push_dir * (min_sep + 5.0),
                    vel=ball.vel.copy(),
                    last_touched_by=ball.last_touched_by,
                )

        for opp in self._opponents_state:
            if not opp.is_upright:
                continue
            diff = ball.pos - opp.pos
            dist = float(np.linalg.norm(diff))
            if dist < min_sep and dist > 1e-6:
                push_dir = diff / dist
                ball = BallState(
                    pos=ball.pos + push_dir * (min_sep - dist + 5.0),
                    vel=ball.vel.copy(),
                    last_touched_by=ball.last_touched_by,
                )

        for obstacle in self._static_obstacles:
            diff = ball.pos - obstacle.pos
            dist = float(np.linalg.norm(diff))
            min_sep = obstacle.radius + BALL.radius
            if dist < min_sep:
                if dist <= 1e-6:
                    push_dir = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    push_dir = diff / dist
                reflected = ball.vel - 2.0 * np.dot(ball.vel, push_dir) * push_dir
                ball = BallState(
                    pos=obstacle.pos + push_dir * (min_sep + 5.0),
                    vel=0.35 * reflected,
                    last_touched_by=ball.last_touched_by,
                    seen_age_ms=ball.seen_age_ms,
                )

        return ball

    def _nearest_to_ball_opponent(self):
        if not self._opponents_state:
            return -1
        dists = [np.linalg.norm(opp.pos - self._ball.pos) for opp in self._opponents_state]
        return int(np.argmin(dists))

    # ── Observations ──────────────────────────────────────────────────

    def _build_observations(self, agent_ids=None):
        observations = {}
        all_agents = list(self._agents_state.values())
        if agent_ids is None:
            agent_ids = self.agents

        for aid in agent_ids:
            agent = self._agents_state[aid]
            visible_opps = self._get_visible_obstacle_aware_entities(aid, agent)
            obs = self._obs_builder.build(
                agent, all_agents, self._ball,
                visible_opps, self._game, self._current_tactic
            )
            if not np.all(np.isfinite(obs)):
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            observations[aid] = obs
        return observations

    def _get_visible_obstacle_aware_entities(self, agent_id: str, agent: AgentState):
        visible_entities = self._fov_filter.get_visible_opponents(
            agent_id, agent.pos, agent.theta, self._opponents_state
        )

        for obstacle in self._static_obstacles:
            if not self._use_fov or self._fov_filter._is_in_fov(agent.pos, agent.theta, obstacle.pos):
                visible_entities.append(obstacle)

        def _entity_dist(entity):
            return float(np.linalg.norm(entity.pos - agent.pos))

        visible_entities.sort(key=_entity_dist)
        return visible_entities[:OBS.opp_slots]

    # ── Rewards ────────────────────────────────────────────────────────

    def _compute_rewards(self, goal_scored, goal_conceded,
                         prev_ball_dists, takeover_kick=False,
                         playball_entry=False,
                         ready_but_walk=False,
                         ready_touch_but_walk=False):
        from reward import (
            AgentRewardInfo, TeamRewardInfo,
            compute_reward, compute_per_agent_reward,
        )

        agents_list = [
            AgentRewardInfo(
                pos=self._agents_state[aid].pos,
                theta=self._agents_state[aid].theta,
                is_upright=self._agents_state[aid].is_upright,
                role=self._agents_state[aid].role,
                prev_ball_dist=prev_ball_dists.get(aid, 0.0),
                prev_theta=self._prev_thetas.get(aid, self._agents_state[aid].theta),
                ball_in_fov=self._is_ball_in_fov(aid),
                prev_omega=self._prev_omegas.get(aid, 0.0),
                is_active=self._agents_state[aid].is_active,
            )
            for aid in self.agents
        ]

        macro = self._handoff_mgr.macro

        team_info = TeamRewardInfo(
            agents=agents_list,
            ball_pos=self._ball.pos,
            ball_vel=self._ball.vel,
            prev_ball_pos=self._prev_ball_pos,
            prev_ball_speed=self._prev_ball_speed,
            own_score=self._game.own_score,
            opp_score=self._game.opp_score,
            goal_scored=goal_scored,
            goal_conceded=goal_conceded,
            opp_positions=[opp.pos for opp in self._opponents_state],
            ball_out_of_bounds=self._ball_out_this_step,
            # BHuman contract fields
            handoff_ready=macro.handoff_ready,
            prev_handoff_ready=self._prev_handoff_ready,
            takeover_active=macro.takeover_active,
            takeover_success=macro.macro_success if macro.macro_end_reason != "none" else None,
            takeover_kick=takeover_kick,
            playball_entry=playball_entry,
            ready_but_walk=ready_but_walk,
            ready_touch_but_walk=ready_touch_but_walk,
            macro_duration_frames=macro.macro_duration_frames,
            macro_end_reason=macro.macro_end_reason,
            # Backend provenance for reward gating
            sidecar_backend=(
                self._sidecar.backend_label if self._sidecar is not None
                else "python_stub"
            ),
        )

        team_components = compute_reward(
            team_info,
            self._stage.shaping_weight,
            approach_weight=self._stage.shaping_approach_w,
            takeover_weight=self._stage.shaping_takeover_w,
            support_weight=self._stage.shaping_support_w,
        )

        rewards = {}
        for i, aid in enumerate(self.agents):
            r = compute_per_agent_reward(
                team_components, agents_list[i],
                self._ball.pos, self._stage.shaping_weight,
                handoff_ready=macro.handoff_ready,
                takeover_active=macro.takeover_active,
            )
            if not math.isfinite(r):
                r = 0.0
            rewards[aid] = r
        return rewards

    def _is_takeover_kick_proxy(self,
                                agent: AgentState,
                                ball: BallState,
                                sidecar_output: Any,
                                *,
                                post_kick_release_frames: int,
                                pending_kick_hold: bool) -> bool:
        if sidecar_output is None:
            return False
        effective_macro_type = _effective_sidecar_macro_type(
            getattr(sidecar_output, "skill_request", "none"),
            getattr(sidecar_output, "macro_type", "none"),
            getattr(sidecar_output, "motion_request", "none"),
        )
        if effective_macro_type not in {"playBall_shoot", "playBall_pass"}:
            return False
        release_limit = 5
        if self._sidecar_physics is not None:
            release_limit = int(getattr(self._sidecar_physics, "_post_kick_release_limit", release_limit) or release_limit)
        # Proxy evidence should only be accepted right after a kick-like
        # release, not throughout the entire release tail.
        if pending_kick_hold or post_kick_release_frames < max(1, release_limit - 1):
            return False

        ball_speed = float(np.linalg.norm(ball.vel))
        expected_speed = float(CONTRACT.kick_info.ball_velocity_max)
        if self._sidecar_physics is not None:
            expected_speed = float(self._sidecar_physics._kick_speed(sidecar_output))
        min_speed = max(
            320.0,
            float(CONTRACT.kick_info.ball_velocity_max) * 0.18,
            expected_speed * 0.42,
        )
        if ball_speed < min_speed:
            return False

        release_distance = AGENT.radius + BALL.radius + 24.0
        if float(np.linalg.norm(ball.pos - agent.pos)) <= release_distance:
            return False

        away_dir = ball.pos - agent.pos
        away_norm = float(np.linalg.norm(away_dir))
        if away_norm <= 1e-6:
            return False
        if float(np.dot(ball.vel / max(ball_speed, 1e-6), away_dir / away_norm)) < 0.4:
            return False

        ball_target = None
        if self._sidecar_physics is not None:
            ball_target = self._sidecar_physics._resolve_playball_ball_target(
                agent, ball, sidecar_output
            )
        if ball_target is not None:
            delta = ball_target - ball.pos
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                target_dir = delta / norm
                ball_dir = ball.vel / max(ball_speed, 1e-6)
                if float(np.dot(ball_dir, target_dir)) < 0.62:
                    return False

        return True

    def _update_takeover_progress(self,
                                  agent: AgentState,
                                  ball: BallState,
                                  sidecar_output: Any,
                                  *,
                                  pending_kick_hold: bool) -> None:
        macro = self._handoff_mgr.macro
        if sidecar_output is None or not macro.macro_active:
            return

        ball_pos = ball.pos.astype(np.float32, copy=True)
        if macro.takeover_start_ball_pos is None:
            macro.takeover_start_ball_pos = ball_pos.copy()
        if macro.takeover_last_ball_pos is None:
            macro.takeover_last_ball_pos = ball_pos.copy()

        macro_type = str(getattr(sidecar_output, "macro_type", "none") or "none")
        skill_request = str(getattr(sidecar_output, "skill_request", "none") or "none")
        motion_request = str(getattr(sidecar_output, "motion_request", "none") or "none")
        effective_macro_type = _effective_sidecar_macro_type(
            skill_request,
            macro_type,
            motion_request,
        )

        control_distance = AGENT.radius + BALL.radius + 55.0
        ball_controlled = float(np.linalg.norm(agent.pos - ball.pos)) <= control_distance
        if ball_controlled and not pending_kick_hold:
            macro.takeover_control_frames += 1
        else:
            macro.takeover_control_frames = 0

        step_delta = ball_pos - macro.takeover_last_ball_pos
        step_progress_mm = float(np.linalg.norm(step_delta))
        total_delta = ball_pos - macro.takeover_start_ball_pos
        total_progress_mm = float(np.linalg.norm(total_delta))

        total_along_target_mm = 0.0
        target = None
        if self._sidecar_physics is not None:
            target = self._sidecar_physics._resolve_playball_ball_target(
                agent, ball, sidecar_output
            )
        if target is not None:
            target_delta = target - macro.takeover_start_ball_pos
            target_norm = float(np.linalg.norm(target_delta))
            if target_norm > 1e-6:
                target_dir = target_delta / target_norm
                total_along_target_mm = float(np.dot(total_delta, target_dir))
                step_along_target_mm = float(np.dot(step_delta, target_dir))
            else:
                step_along_target_mm = 0.0
        else:
            step_along_target_mm = 0.0

        macro.takeover_ball_progress_mm = max(macro.takeover_ball_progress_mm, total_progress_mm)
        macro.takeover_ball_progress_along_target_mm = max(
            macro.takeover_ball_progress_along_target_mm,
            total_along_target_mm,
        )

        is_progress_macro = (
            effective_macro_type in {"playBall_dribble", "playBall_duel"}
            or skill_request == "dribble"
            or motion_request == "dribble"
        )
        if is_progress_macro and ball_controlled and not pending_kick_hold:
            if step_progress_mm >= 8.0 and (target is None or step_along_target_mm >= 4.0):
                macro.takeover_progress_frames += 1

        macro.takeover_last_ball_pos = ball_pos

    def _has_takeover_execution_evidence(self,
                                         agent: AgentState,
                                         ball: BallState,
                                         sidecar_output: Any,
                                         *,
                                         takeover_kick: bool,
                                         takeover_kick_proxy: bool,
                                         takeover_progress: bool,
                                         pending_kick_hold: bool) -> bool:
        if sidecar_output is None:
            return False

        macro = self._handoff_mgr.macro
        skill_request = str(getattr(sidecar_output, "skill_request", "none") or "none")
        motion_request = str(getattr(sidecar_output, "motion_request", "none") or "none")
        macro_type = _effective_sidecar_macro_type(
            skill_request,
            getattr(sidecar_output, "macro_type", "none"),
            motion_request,
        )
        executed_phase = str(getattr(sidecar_output, "motion_executed_phase", "unavailable") or "unavailable")
        phase_compatible = executed_phase in {"walk", "kick", "stand"}

        if skill_request in {"shoot", "pass"} or macro_type in {"playBall_shoot", "playBall_pass"}:
            return phase_compatible and (takeover_kick or takeover_kick_proxy)

        control_distance = AGENT.radius + BALL.radius + 55.0
        ball_controlled = float(np.linalg.norm(agent.pos - ball.pos)) <= control_distance

        if skill_request == "dribble" or motion_request == "dribble" or macro_type == "playBall_dribble":
            strong_ball_progress = (
                macro.takeover_ball_progress_mm >= 800.0
                and macro.takeover_ball_progress_along_target_mm >= 600.0
            )
            controlled_progress = (
                macro.takeover_control_frames >= 1
                and macro.takeover_ball_progress_mm >= 180.0
                and macro.takeover_ball_progress_along_target_mm >= 120.0
            )
            return (
                phase_compatible
                and not pending_kick_hold
                and (strong_ball_progress or controlled_progress)
            )

        if skill_request == "block":
            return False

        if macro_type == "playBall_duel":
            strong_duel_progress = (
                macro.takeover_ball_progress_mm >= 500.0
                and macro.takeover_ball_progress_along_target_mm >= 250.0
            )
            return (
                phase_compatible
                and not pending_kick_hold
                and (
                    strong_duel_progress
                    or (
                        ball_controlled
                        and takeover_progress
                        and (
                            macro.takeover_progress_frames >= 1
                            or macro.takeover_ball_progress_mm >= 80.0
                        )
                    )
                )
            )

        return takeover_progress

    def _recover_cleared_macro_success(self,
                                       macro: MacroState,
                                       *,
                                       takeover_kick: bool,
                                       takeover_kick_proxy: bool) -> None:
        if macro.macro_end_reason != "skill_request_cleared" or macro.macro_success is not False:
            return

        if macro.macro_type in {"playBall_shoot", "playBall_pass"}:
            if takeover_kick or takeover_kick_proxy:
                macro.macro_end_reason = "completed"
                macro.macro_success = True
            return

        if macro.macro_type == "playBall_dribble":
            if (
                macro.takeover_ball_progress_mm >= 800.0
                and macro.takeover_ball_progress_along_target_mm >= 600.0
            ):
                macro.macro_end_reason = "completed"
                macro.macro_success = True
            return

        if macro.macro_type == "playBall_duel":
            if (
                macro.takeover_ball_progress_mm >= 500.0
                and macro.takeover_ball_progress_along_target_mm >= 250.0
            ):
                macro.macro_end_reason = "completed"
                macro.macro_success = True

    def _is_ball_in_fov(self, agent_id):
        if agent_id not in self._agents_state:
            return False
        agent = self._agents_state[agent_id]
        if not agent.is_upright:
            return False
        return self._fov_filter._is_in_fov(agent.pos, agent.theta, self._ball.pos)

    def _build_infos(self, goal_scored, goal_conceded, sidecar_output=None, agent_ids=None,
                     playball_entry=False, ready_but_walk=False,
                     ready_touch_but_walk=False, pending_kick_hold=False,
                     pending_kick_hold_frames=0, takeover_kick=False,
                     takeover_kick_proxy=False,
                     post_kick_release_frames=0):
        macro = self._handoff_mgr.macro
        if agent_ids is None:
            agent_ids = self.agents
        effective_macro_type = macro.macro_type
        if sidecar_output is not None and not macro.macro_active:
            effective_macro_type = sidecar_output.macro_type
        ready_frames = max(1, self._episode_handoff_ready_frames)
        takeover_total = max(1, self._episode_takeover_count)
        control_owner = "BHUMAN" if macro.takeover_active else ("HANDOFF" if macro.handoff_ready else "RL")
        base_info = {
            "own_score": self._game.own_score,
            "opp_score": self._game.opp_score,
            "step": self._game.step_count,
            "goal_scored": goal_scored,
            "goal_conceded": goal_conceded,
            "ball_out": self._ball_out_this_step,
            "ball_pos": self._ball.pos.tolist(),
            "ball_vel": self._ball.vel.tolist(),
            "ball_speed": float(np.linalg.norm(self._ball.vel)),
            "ball_heading_deg": (
                math.degrees(math.atan2(float(self._ball.vel[1]), float(self._ball.vel[0])))
                if float(np.linalg.norm(self._ball.vel)) > 1e-6 else None
            ),
            # BHuman contract info
            "control_owner": control_owner,
            "handoff_ready": macro.handoff_ready,
            "takeover_active": macro.takeover_active,
            "macro_active": macro.macro_active,
            "macro_type": effective_macro_type,
            "takeover_macro_type": macro.macro_type,
            "macro_duration_frames": macro.macro_duration_frames,
            "macro_end_reason": macro.macro_end_reason,
            "macro_success": macro.macro_success,
            "last_macro_end_reason": self._last_macro_end_reason,
            "playball_entry": playball_entry,
            "ready_but_walk": ready_but_walk,
            "ready_touch_but_walk": ready_touch_but_walk,
            "pending_kick_hold": bool(pending_kick_hold),
            "pending_kick_hold_frames": int(pending_kick_hold_frames),
            "takeover_kick": bool(takeover_kick),
            "takeover_kick_proxy": bool(takeover_kick_proxy),
            "post_kick_release_frames": int(post_kick_release_frames),
            "active_agent_id": self._active_agent_id,
            "takeover_control_frames": int(macro.takeover_control_frames),
            "takeover_progress_frames": int(macro.takeover_progress_frames),
            "takeover_ball_progress_mm": float(macro.takeover_ball_progress_mm),
            "takeover_ball_progress_along_target_mm": float(
                macro.takeover_ball_progress_along_target_mm
            ),
            "episode_takeover_count": self._episode_takeover_count,
            "episode_takeover_success": self._episode_takeover_success,
            "episode_takeover_count_dribble": self._episode_takeover_count_by_type["playBall_dribble"],
            "episode_takeover_count_duel": self._episode_takeover_count_by_type["playBall_duel"],
            "episode_takeover_count_shoot": self._episode_takeover_count_by_type["playBall_shoot"],
            "episode_takeover_count_pass": self._episode_takeover_count_by_type["playBall_pass"],
            "episode_takeover_success_dribble": self._episode_takeover_success_by_type["playBall_dribble"],
            "episode_takeover_success_duel": self._episode_takeover_success_by_type["playBall_duel"],
            "episode_takeover_success_shoot": self._episode_takeover_success_by_type["playBall_shoot"],
            "episode_takeover_success_pass": self._episode_takeover_success_by_type["playBall_pass"],
            "episode_handoff_achieved": self._episode_handoff_achieved,
            "episode_handoff_ready_frames": self._episode_handoff_ready_frames,
            "episode_tkready_achieved": self._episode_tkready_achieved,
            "episode_tkready_frames": self._episode_tkready_frames,
            "episode_playball_entry_count": self._episode_playball_entry_count,
            "episode_playball_achieved": self._episode_playball_achieved,
            "episode_ready_but_walk_frames": self._episode_ready_but_walk_frames,
            "episode_ready_touch_but_walk_frames": self._episode_ready_touch_but_walk_frames,
            "episode_ready_but_walk_rate": (
                self._episode_ready_but_walk_frames / ready_frames
                if self._episode_handoff_ready_frames > 0 else 0.0
            ),
            "episode_ready_touch_but_walk_rate": (
                self._episode_ready_touch_but_walk_frames / ready_frames
                if self._episode_handoff_ready_frames > 0 else 0.0
            ),
            "episode_takeover_timeout_count": self._episode_takeover_reason_counts["timeout"],
            "episode_takeover_skill_cleared_count": (
                self._episode_takeover_reason_counts["skill_request_cleared"]
            ),
            "episode_takeover_active_change_count": (
                self._episode_takeover_reason_counts["active_agent_changed"]
            ),
            "episode_takeover_other_end_count": (
                self._episode_takeover_reason_counts["out"]
                + self._episode_takeover_reason_counts["goal"]
                + self._episode_takeover_reason_counts["game_state_changed"]
                + self._episode_takeover_reason_counts["other"]
            ),
            "episode_takeover_timeout_rate": (
                self._episode_takeover_reason_counts["timeout"] / takeover_total
                if self._episode_takeover_count > 0 else 0.0
            ),
            "episode_takeover_skill_cleared_rate": (
                self._episode_takeover_reason_counts["skill_request_cleared"] / takeover_total
                if self._episode_takeover_count > 0 else 0.0
            ),
            "episode_takeover_active_change_rate": (
                self._episode_takeover_reason_counts["active_agent_changed"] / takeover_total
                if self._episode_takeover_count > 0 else 0.0
            ),
            "episode_takeover_other_end_rate": (
                (
                    self._episode_takeover_reason_counts["out"]
                    + self._episode_takeover_reason_counts["goal"]
                    + self._episode_takeover_reason_counts["game_state_changed"]
                    + self._episode_takeover_reason_counts["other"]
                ) / takeover_total
                if self._episode_takeover_count > 0 else 0.0
            ),
            "static_obstacle_count": len(self._static_obstacles),
            "static_obstacles": [
                {
                    "id": obstacle.obstacle_id,
                    "pos": obstacle.pos.tolist(),
                    "radius": obstacle.radius,
                }
                for obstacle in self._static_obstacles
            ],
            "global_state": build_global_state(
                list(self._agents_state.values()),
                self._ball, self._opponents_state, self._game,
                static_obstacles=self._static_obstacles,
            ),
            "execution_mode": self._execution_mode,
            "sidecar_active": self._sidecar is not None,
        }
        if self._active_agent_id in self._agents_state:
            active_agent = self._agents_state[self._active_agent_id]
            ball_delta = self._ball.pos - active_agent.pos
            ball_dist = float(np.linalg.norm(ball_delta))
            if ball_dist > 1e-6:
                ball_bearing_rad = _normalize_angle(
                    math.atan2(float(ball_delta[1]), float(ball_delta[0])) - float(active_agent.theta)
                )
                heading_alignment = float(np.dot(
                    ball_delta / ball_dist,
                    np.array([math.cos(active_agent.theta), math.sin(active_agent.theta)], dtype=np.float32),
                ))
            else:
                ball_bearing_rad = 0.0
                heading_alignment = 1.0
            base_info.update({
                "active_agent_pos": active_agent.pos.tolist(),
                "active_agent_theta": float(active_agent.theta),
                "active_agent_theta_deg": math.degrees(float(active_agent.theta)),
                "active_agent_vel": active_agent.vel.tolist(),
                "active_agent_speed": float(np.linalg.norm(active_agent.vel)),
                "active_agent_omega": float(active_agent.omega),
                "active_agent_walk_target": active_agent.walk_target.tolist(),
                "active_agent_ball_distance_mm": ball_dist,
                "active_agent_ball_bearing_rad": float(ball_bearing_rad),
                "active_agent_ball_bearing_deg": math.degrees(float(ball_bearing_rad)),
                "active_agent_ball_heading_alignment": heading_alignment,
            })
        # Sidecar-enriched fields
        if sidecar_output is not None:
            base_info["sidecar_macro_type"] = sidecar_output.macro_type
            base_info["sidecar_macro_type_effective"] = _effective_sidecar_macro_type(
                sidecar_output.skill_request,
                sidecar_output.macro_type,
                sidecar_output.motion_request,
            )
            base_info["sidecar_current_tactic"] = sidecar_output.current_tactic
            base_info["sidecar_skill_request"] = sidecar_output.skill_request
            base_info["sidecar_skill_target"] = sidecar_output.skill_target
            base_info["sidecar_skill_target_direction"] = sidecar_output.skill_target_direction
            base_info["sidecar_motion_request"] = sidecar_output.motion_request
            base_info["sidecar_motion_walk_target"] = sidecar_output.motion_walk_target
            base_info["sidecar_motion_target_direction"] = sidecar_output.motion_target_direction
            base_info["sidecar_motion_executed_phase"] = sidecar_output.motion_executed_phase
            base_info["sidecar_motion_ball_target_field"] = sidecar_output.motion_ball_target_field
            base_info["sidecar_motion_contact_point_field"] = sidecar_output.motion_contact_point_field
            base_info["sidecar_robot_pose_field"] = sidecar_output.robot_pose_field
            base_info["sidecar_field_ball_position"] = sidecar_output.field_ball_position
            base_info["sidecar_field_ball_intercepted_position"] = (
                sidecar_output.field_ball_intercepted_position
            )
            base_info["sidecar_behavior_status"] = sidecar_output.behavior_status
            base_info["sidecar_behavior_speed"] = sidecar_output.behavior_speed
            base_info["sidecar_behavior_walking_to"] = sidecar_output.behavior_walking_to
            base_info["sidecar_behavior_shooting_to"] = sidecar_output.behavior_shooting_to
            base_info["sidecar_strategy_status"] = sidecar_output.strategy_status
            base_info["sidecar_activation_graph"] = [
                n.to_dict() if hasattr(n, "to_dict") else str(n)
                for n in sidecar_output.activation_graph
            ]
            base_info["sidecar_kick_type"] = sidecar_output.kick_type
            base_info["sidecar_kick_length"] = sidecar_output.kick_length
            base_info["sidecar_kick_length_raw"] = getattr(
                sidecar_output, "kick_length_raw", sidecar_output.kick_length
            )
            base_info["sidecar_last_kick_type"] = sidecar_output.last_kick_type
            base_info["sidecar_last_kick_timestamp"] = sidecar_output.last_kick_timestamp
            base_info["sidecar_pass_target_player_number"] = sidecar_output.pass_target_player_number
            base_info["sidecar_pass_origin_player_number"] = sidecar_output.pass_origin_player_number
            base_info["sidecar_backend"] = sidecar_output.backend
            base_info["sidecar_bridge_mode"] = sidecar_output.bridge_mode
            if (
                self._active_agent_id in self._agents_state
                and self._sidecar_physics is not None
            ):
                active_agent = self._agents_state[self._active_agent_id]
                walk_target_field = self._sidecar_physics._robot_relative_target_to_field(
                    active_agent,
                    sidecar_output.behavior_walking_to,
                    sidecar_output,
                )
                if walk_target_field is not None:
                    base_info["sidecar_behavior_walking_to_field"] = walk_target_field.tolist()
                shoot_target_field = self._sidecar_physics._robot_relative_target_to_field(
                    active_agent,
                    sidecar_output.behavior_shooting_to,
                    sidecar_output,
                )
                if shoot_target_field is not None:
                    base_info["sidecar_behavior_shooting_to_field"] = shoot_target_field.tolist()
                motion_walk_target_field = self._sidecar_physics._motion_walk_target_to_field(
                    active_agent,
                    sidecar_output.motion_walk_target,
                    sidecar_output,
                )
                if motion_walk_target_field is not None:
                    base_info["sidecar_motion_walk_target_field"] = motion_walk_target_field[0].tolist()
                    base_info["sidecar_motion_walk_target_theta_field"] = float(motion_walk_target_field[1])
                playball_ball_target = self._sidecar_physics._resolve_playball_ball_target(
                    active_agent,
                    self._ball,
                    sidecar_output,
                )
                if playball_ball_target is not None:
                    base_info["sidecar_ball_target_field"] = playball_ball_target.tolist()
                effective_target = self._sidecar_physics._resolve_reference_target(
                    active_agent,
                    self._ball,
                    sidecar_output,
                )
                if effective_target is not None:
                    base_info["sidecar_effective_target"] = effective_target.tolist()
                effective_motion_target, effective_theta = self._sidecar_physics._resolve_motion_target(
                    active_agent,
                    self._ball,
                    sidecar_output,
                    SIM.dt,
                )
                base_info["sidecar_effective_motion_target"] = effective_motion_target.tolist()
                base_info["sidecar_effective_target_theta"] = float(effective_theta)
            base_info["sidecar_contract_validation_mode"] = sidecar_output.contract_validation_mode
            base_info["takeover_simulator_active"] = (self._takeover_sim is not None)
            base_info["bridge_frame_id"] = sidecar_output.bridge_frame_id
            base_info["bridge_player_number"] = sidecar_output.bridge_player_number
            base_info["bridge_metadata"] = dict(sidecar_output.bridge_metadata)
            for key, value in sidecar_output.bridge_metadata.items():
                base_info[f"bridge_{key}"] = value
        return {aid: base_info.copy() for aid in agent_ids}

    # ── Utilities ─────────────────────────────────────────────────────

    def set_stage(self, stage_cfg):
        self._stage = stage_cfg
        self._movement.configure(stage_cfg)
        self._fov_filter = FOVFilter(use_fov=self._use_fov)
        self._handoff_mgr.set_takeover_timeout_frames(self._stage.takeover_timeout_steps)

    def get_global_state(self):
        return build_global_state(
            list(self._agents_state.values()),
            self._ball, self._opponents_state, self._game,
            static_obstacles=self._static_obstacles,
        )

    def close(self):
        if self._sidecar is not None:
            self._sidecar.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _goal_opening_angle(pos: np.ndarray) -> float:
    goal_left = np.array([FIELD.x_opponent_ground, FIELD.y_left_goal])
    goal_right = np.array([FIELD.x_opponent_ground, FIELD.y_right_goal])
    angle_left = math.atan2(goal_left[1] - pos[1], goal_left[0] - pos[0])
    angle_right = math.atan2(goal_right[1] - pos[1], goal_right[0] - pos[0])
    return float(angle_left - angle_right)


def _goal_relative_angle(pos: np.ndarray, theta: float) -> float:
    goal_center = np.array([FIELD.x_opponent_ground, 0.0])
    dx = goal_center[0] - pos[0]
    dy = goal_center[1] - pos[1]
    absolute_angle = math.atan2(dy, dx)
    return _normalize_angle(absolute_angle - theta)
