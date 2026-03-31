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
    FIELD, AGENT, BALL, FOV, SIM,
    OBS, ACTION, GLOBAL_STATE, NORM,
    CURRICULUM, CurriculumStageConfig, ROLES, TACTIC_NAMES,
    ACTION_SMOOTHING, CONTRACT, EXECUTION_MODE,
)
from bhuman_cfg_parser import (
    get_home_poses, MACRO_TYPES, MACRO_END_REASONS, TAKEOVER_SKILLS,
)

TAKEOVER_MACRO_TYPES = frozenset(mt for mt in MACRO_TYPES if mt != "none")


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

        if abs(dtheta) > self.ROTATION_THRESHOLD:
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


def build_global_state(agents, ball, opponents, game):
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
    for i in range(4):
        if i < len(opponents):
            state[idx]     = opponents[i].pos[0] / NORM.pos_x
            state[idx + 1] = opponents[i].pos[1] / NORM.pos_y
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

    def reset(self) -> None:
        self._last_kick_timestamps.clear()

    def apply(self,
              agent: AgentState,
              ball: BallState,
              output: Any,
              agents: Dict[str, AgentState],
              dt: float) -> Tuple[AgentState, BallState, bool, bool]:
        if not agent.is_upright:
            return agent, ball, False, False

        updated_agent = self._apply_agent_motion(agent, ball, output, dt)

        if self._has_new_kick(agent.agent_id, output):
            return updated_agent, self._apply_kick(updated_agent, ball, output, agents), True, False

        if output.motion_request == "dribble":
            dribble_ball, dribble_progress = self._apply_dribble(updated_agent, ball, output)
            return updated_agent, dribble_ball, False, dribble_progress

        return updated_agent, ball, False, False

    def _apply_agent_motion(self,
                            agent: AgentState,
                            ball: BallState,
                            output: Any,
                            dt: float) -> AgentState:
        target_pos, target_theta = self._resolve_motion_target(agent, ball, output)
        return self._movement.step(agent, target_pos, target_theta, dt)

    def _resolve_motion_target(self,
                               agent: AgentState,
                               ball: BallState,
                               output: Any) -> Tuple[np.ndarray, float]:
        if output.behavior_walking_to is not None:
            target_pos = np.array(output.behavior_walking_to, dtype=np.float32)
        elif output.motion_walk_target is not None:
            target_pos = np.array(output.motion_walk_target[:2], dtype=np.float32)
        elif output.skill_target is not None:
            target_pos = np.array(output.skill_target, dtype=np.float32)
        elif output.motion_request in {"walkToBallAndKick", "dribble"}:
            target_pos = ball.pos.copy()
        else:
            vx, vy, _ = output.motion_speed
            if abs(vx) > 1e-6 or abs(vy) > 1e-6:
                cos_t = math.cos(agent.theta)
                sin_t = math.sin(agent.theta)
                vx_field = vx * cos_t - vy * sin_t
                vy_field = vx * sin_t + vy * cos_t
                target_pos = agent.pos + np.array([vx_field, vy_field], dtype=np.float32) * max(dt, 0.2)
            else:
                target_pos = agent.pos.copy()

        if output.motion_walk_target is not None:
            target_theta = float(output.motion_walk_target[2])
        elif output.motion_target_direction is not None:
            target_theta = float(output.motion_target_direction)
        elif output.skill_target_direction is not None:
            target_theta = float(output.skill_target_direction)
        elif output.behavior_shooting_to is not None:
            shoot_to = np.array(output.behavior_shooting_to, dtype=np.float32)
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
        direction = self._resolve_kick_direction(agent, output, agents)
        speed = self._kick_speed(output)
        contact_offset = AGENT.min_robot_distance + BALL.radius + 25.0
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
        if distance > AGENT.ball_interaction_range:
            return ball, False

        direction = self._direction_from_agent(agent, output)
        contact_offset = AGENT.radius + BALL.radius + 10.0
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
        progress = float(next_pos[0] - ball.pos[0]) > 40.0
        return dribble_ball, progress

    def _resolve_kick_direction(self,
                                agent: AgentState,
                                output: Any,
                                agents: Dict[str, AgentState]) -> np.ndarray:
        if output.pass_target_player_number is not None:
            target_agent = self._agent_from_player_number(output.pass_target_player_number, agents)
            if target_agent is not None:
                delta = target_agent.pos - agent.pos
                norm = float(np.linalg.norm(delta))
                if norm > 1e-6:
                    return delta / norm

        if output.behavior_shooting_to is not None:
            shoot_to = np.array(output.behavior_shooting_to, dtype=np.float32)
            delta = shoot_to - agent.pos
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return delta / norm

        if output.skill_target is not None:
            target = np.array(output.skill_target, dtype=np.float32)
            delta = target - agent.pos
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return delta / norm

        return self._direction_from_agent(agent, output)

    def _direction_from_agent(self, agent: AgentState, output: Any) -> np.ndarray:
        if output.motion_target_direction is not None:
            theta = float(output.motion_target_direction)
        elif output.skill_target_direction is not None:
            theta = float(output.skill_target_direction)
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
        kick_length = float(output.kick_length or 0.0)
        if entry is not None:
            span = max(1.0, entry.range_max - entry.range_min)
            alpha = float(np.clip((kick_length - entry.range_min) / span, 0.0, 1.0))
            return float(entry.ball_velocity_min + alpha * (entry.ball_velocity_max - entry.ball_velocity_min))

        if kick_length <= 0.0:
            return float(self._kick_info.ball_velocity_max)

        ratio = float(np.clip(kick_length / max(1.0, self._kick_info.forward_fast_max), 0.35, 1.0))
        return ratio * float(self._kick_info.ball_velocity_max)


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
        self._ball: BallState = BallState(np.zeros(2), np.zeros(2))
        self._game: GameState = GameState()
        self._prev_ball_pos: np.ndarray = np.zeros(2)
        self._prev_ball_speed: float = 0.0
        self._prev_ball_dists: Dict[str, float] = {}
        self._current_tactic: str = "t121_4v4_scaled"
        self._active_agent_id: Optional[str] = None

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
        self._episode_handoff_achieved = False  # True if handoff_ready was ever True
        self._prev_handoff_ready = False

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
        self._episode_handoff_achieved = False
        self._prev_handoff_ready = False

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
        takeover_progress = False
        macro_resolved_pre_physics = False
        sidecar_macro_type = "none"
        if sidecar_output is not None:
            sidecar_macro_type = sidecar_output.macro_type

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
                    agent, self._ball, takeover_kick, takeover_progress = self._sidecar_physics.apply(
                        agent, self._ball, sidecar_output, self._agents_state, SIM.dt
                    )
                    self._agents_state[self._active_agent_id] = agent

                    if takeover_kick and sidecar_output.skill_request in {"shoot", "pass"}:
                        self._handoff_mgr.force_end("completed", True)
                        macro_resolved_pre_physics = True
                    elif takeover_progress and sidecar_output.macro_type in {
                        "playBall_dribble", "playBall_duel"
                    }:
                        self._handoff_mgr.force_end("completed", True)
                        macro_resolved_pre_physics = True
                else:
                    # contract_validation_mode keeps semantic evidence only.
                    pass

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

        # ═══════════════════════════════════════════════════════════════
        # 6. GOAL CHECK
        # ═══════════════════════════════════════════════════════════════
        goal_scored = False
        goal_conceded = False
        goal_result = self._ball_physics.check_goal(self._ball)
        if goal_result == "own":
            self._game.own_score += 1
            goal_scored = True
            self._ball = self._spawn_ball()
        elif goal_result == "opp":
            self._game.opp_score += 1
            goal_conceded = True
            self._ball = self._spawn_ball()

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
            # Record takeover events
            if (
                macro.macro_end_reason != "none"
                and macro.macro_success is not None
                and not macro.event_recorded
            ):
                self._episode_takeover_count += 1
                if macro.macro_success:
                    self._episode_takeover_success += 1
                self._takeover_events.append({
                    "type": macro.macro_type,
                    "duration": macro.macro_duration_frames,
                    "end_reason": macro.macro_end_reason,
                    "success": macro.macro_success,
                })
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
            takeover_kick=takeover_kick,
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

    def _spawn_ball(self):
        if not getattr(self, "_agents_state", None):
            return BallState(pos=np.zeros(2, dtype=np.float32), vel=np.zeros(2, dtype=np.float32))

        ref_aid = self.agents[0] if self.agents else next(iter(self._agents_state.keys()))
        ref_agent = self._agents_state[ref_aid]

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
            angle_deadband = math.radians(6.0)

            if (
                dist_to_ball < AGENT.ball_interaction_range * 0.7
                or in_handoff_pose
                or abs(diff) < angle_deadband
            ):
                target_theta = prev_face
            else:
                face_alpha = 0.18
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
            visible_opps = self._fov_filter.get_visible_opponents(
                aid, agent.pos, agent.theta, self._opponents_state
            )
            obs = self._obs_builder.build(
                agent, all_agents, self._ball,
                visible_opps, self._game, self._current_tactic
            )
            if not np.all(np.isfinite(obs)):
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            observations[aid] = obs
        return observations

    # ── Rewards ────────────────────────────────────────────────────────

    def _compute_rewards(self, goal_scored, goal_conceded,
                         prev_ball_dists, takeover_kick=False):
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
            macro_duration_frames=macro.macro_duration_frames,
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
            )
            if not math.isfinite(r):
                r = 0.0
            rewards[aid] = r
        return rewards

    def _is_ball_in_fov(self, agent_id):
        if agent_id not in self._agents_state:
            return False
        agent = self._agents_state[agent_id]
        if not agent.is_upright:
            return False
        return self._fov_filter._is_in_fov(agent.pos, agent.theta, self._ball.pos)

    def _build_infos(self, goal_scored, goal_conceded, sidecar_output=None, agent_ids=None):
        macro = self._handoff_mgr.macro
        if agent_ids is None:
            agent_ids = self.agents
        effective_macro_type = macro.macro_type
        if sidecar_output is not None and not macro.macro_active:
            effective_macro_type = sidecar_output.macro_type
        base_info = {
            "own_score": self._game.own_score,
            "opp_score": self._game.opp_score,
            "step": self._game.step_count,
            "goal_scored": goal_scored,
            "goal_conceded": goal_conceded,
            "ball_out": self._ball_out_this_step,
            "ball_pos": self._ball.pos.tolist(),
            # BHuman contract info
            "handoff_ready": macro.handoff_ready,
            "takeover_active": macro.takeover_active,
            "macro_active": macro.macro_active,
            "macro_type": effective_macro_type,
            "takeover_macro_type": macro.macro_type,
            "macro_duration_frames": macro.macro_duration_frames,
            "macro_end_reason": macro.macro_end_reason,
            "macro_success": macro.macro_success,
            "active_agent_id": self._active_agent_id,
            "episode_takeover_count": self._episode_takeover_count,
            "episode_takeover_success": self._episode_takeover_success,
            "episode_handoff_achieved": self._episode_handoff_achieved,
            "global_state": build_global_state(
                list(self._agents_state.values()),
                self._ball, self._opponents_state, self._game
            ),
            "execution_mode": self._execution_mode,
            "sidecar_active": self._sidecar is not None,
        }
        # Sidecar-enriched fields
        if sidecar_output is not None:
            base_info["sidecar_macro_type"] = sidecar_output.macro_type
            base_info["sidecar_current_tactic"] = sidecar_output.current_tactic
            base_info["sidecar_skill_request"] = sidecar_output.skill_request
            base_info["sidecar_skill_target"] = sidecar_output.skill_target
            base_info["sidecar_skill_target_direction"] = sidecar_output.skill_target_direction
            base_info["sidecar_motion_request"] = sidecar_output.motion_request
            base_info["sidecar_motion_walk_target"] = sidecar_output.motion_walk_target
            base_info["sidecar_motion_target_direction"] = sidecar_output.motion_target_direction
            base_info["sidecar_motion_executed_phase"] = sidecar_output.motion_executed_phase
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
            base_info["sidecar_last_kick_type"] = sidecar_output.last_kick_type
            base_info["sidecar_last_kick_timestamp"] = sidecar_output.last_kick_timestamp
            base_info["sidecar_pass_target_player_number"] = sidecar_output.pass_target_player_number
            base_info["sidecar_pass_origin_player_number"] = sidecar_output.pass_origin_player_number
            base_info["sidecar_backend"] = sidecar_output.backend
            base_info["sidecar_bridge_mode"] = sidecar_output.bridge_mode
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
        self._fov_filter = FOVFilter(use_fov=self._use_fov)
        self._handoff_mgr.set_takeover_timeout_frames(self._stage.takeover_timeout_steps)

    def get_global_state(self):
        return build_global_state(
            list(self._agents_state.values()),
            self._ball, self._opponents_state, self._game
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
