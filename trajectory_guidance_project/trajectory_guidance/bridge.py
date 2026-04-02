"""
bridge.py — Adaptador entre el entorno real y el solver de trayectorias.

Cambios respecto a la versión anterior:
  FIX-A: Nuevo método get_live_world_state() que lee el estado ACTUAL del
         entorno sin resetearlo. Esto permite que evaluate_solver.replay_closed_loop
         re-planifique desde la posición real en cada paso.

El bridge es la capa de abstracción que impide que el solver toque
directamente los internals del entorno. Todo acceso pasa por aquí.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import numpy as np

from config import CURRICULUM, AGENT, FIELD
from wrapper import NaoSoccerEnvWrapper
from trajectory_guidance.settings import (
    ENV_BRIDGE_CFG,
    GumbelConfig,
    get_stage_objective,
)
from trajectory_guidance.heuristic import (
    SimplifiedWorldState,
    estimate_takeover_proxy,
)
from trajectory_guidance.launch_templates import ResolvedLaunchTemplate
from trajectory_guidance.solver import PlannedTrajectory


@dataclass
class WorldSnapshot:
    """Snapshot inmutable del mundo en el momento del reset."""
    stage_id: int
    agent_id: str
    agent_pos: np.ndarray
    ball_pos: np.ndarray
    agent_theta: float
    obs_local: np.ndarray
    state_global: np.ndarray
    reset_seed: Optional[int] = None
    start_mode: str = "reset_seed"
    template_name: Optional[str] = None
    template_family: Optional[str] = None

    def to_simplified_state(self) -> SimplifiedWorldState:
        handoff_ready, playball_active, ready_but_walk, takeover_success = (
            estimate_takeover_proxy(
                self.agent_pos,
                self.ball_pos,
                float(self.agent_theta),
                recontacts=0,
                objective=get_stage_objective(self.stage_id),
            )
        )
        return SimplifiedWorldState(
            agent_pos=self.agent_pos.copy(),
            ball_pos=self.ball_pos.copy(),
            heading=float(self.agent_theta),
            touched_ball=False,
            recontacts=0,
            goal_scored=False,
            handoff_ready=handoff_ready,
            playball_active=playball_active,
            ready_but_walk=ready_but_walk,
            takeover_success=takeover_success,
        )


class EnvBridge:
    def __init__(self,
                 controlled_agent_id: str = ENV_BRIDGE_CFG.controlled_agent_id,
                 use_fov: bool = True,
                 seed: int = ENV_BRIDGE_CFG.default_seed):
        self.env = NaoSoccerEnvWrapper(use_fov=use_fov, seed=seed)
        self.controlled_agent_id = controlled_agent_id
        self.episode_index = 0
        self._last_obs = None
        self._last_infos = None

    @property
    def raw_env(self):
        return self.env._env

    def reset_to_stage(self, stage_id: int) -> None:
        self.env.set_curriculum_stage(stage_id)

    def reset_episode(self, seed: Optional[int] = None):
        self.episode_index += 1
        obs, infos = self.env.reset(seed=seed)
        self._last_obs, self._last_infos = obs, infos
        return obs, infos

    def _refresh_bridge_state(self):
        obs = self.raw_env._build_observations()
        infos = self.raw_env._build_infos(
            goal_scored=False,
            goal_conceded=False,
            sidecar_output=None,
        )
        self.env._inject_global_state(infos)
        self._last_obs, self._last_infos = obs, infos
        return obs, infos

    def _apply_launch_template(self, launch_template: ResolvedLaunchTemplate) -> None:
        raw_env = self.raw_env
        def _clip_point(point: np.ndarray) -> np.ndarray:
            clipped = np.asarray(point, dtype=np.float32).copy()
            clipped[0] = float(np.clip(
                clipped[0],
                FIELD.x_own_ground + 220.0,
                FIELD.x_opponent_ground - 220.0,
            ))
            clipped[1] = float(np.clip(
                clipped[1],
                FIELD.y_right_sideline + 220.0,
                FIELD.y_left_sideline - 220.0,
            ))
            return clipped

        controlled_agent = raw_env._agents_state[self.controlled_agent_id]
        controlled_agent.pos = _clip_point(launch_template.agent_pos)
        controlled_agent.theta = float(launch_template.agent_theta)
        controlled_agent.vel = np.zeros(2, dtype=np.float32)
        controlled_agent.omega = 0.0
        controlled_agent.is_upright = True
        controlled_agent.recovery_steps = 0
        controlled_agent.walk_target = controlled_agent.pos.copy()
        controlled_agent.speed_scalar = 0.0

        teammate_ids = [aid for aid in raw_env.agents if aid != self.controlled_agent_id]
        for idx, teammate_id in enumerate(teammate_ids):
            teammate = raw_env._agents_state[teammate_id]
            local_offset = launch_template.teammate_offsets[
                min(idx, len(launch_template.teammate_offsets) - 1)
            ]
            world_offset = np.asarray(
                [
                    local_offset[0] * math.cos(launch_template.agent_theta)
                    - local_offset[1] * math.sin(launch_template.agent_theta),
                    local_offset[0] * math.sin(launch_template.agent_theta)
                    + local_offset[1] * math.cos(launch_template.agent_theta),
                ],
                dtype=np.float32,
            )
            teammate.pos = _clip_point(controlled_agent.pos + world_offset)
            teammate.theta = float(launch_template.agent_theta)
            teammate.vel = np.zeros(2, dtype=np.float32)
            teammate.omega = 0.0
            teammate.is_upright = True
            teammate.recovery_steps = 0
            teammate.walk_target = teammate.pos.copy()
            teammate.speed_scalar = 0.0

        for idx, opponent in enumerate(raw_env._opponents_state):
            local_offset = launch_template.opponent_offsets[
                min(idx, len(launch_template.opponent_offsets) - 1)
            ]
            world_offset = np.asarray(
                [
                    local_offset[0] * math.cos(launch_template.agent_theta)
                    - local_offset[1] * math.sin(launch_template.agent_theta),
                    local_offset[0] * math.sin(launch_template.agent_theta)
                    + local_offset[1] * math.cos(launch_template.agent_theta),
                ],
                dtype=np.float32,
            )
            opponent.pos = _clip_point(controlled_agent.pos + world_offset)
            opponent.theta = float(launch_template.agent_theta + math.pi)
            opponent.vel = np.zeros(2, dtype=np.float32)
            opponent.is_upright = True
            opponent.home_pos = opponent.pos.copy()

        raw_env._ball.pos = _clip_point(launch_template.ball_pos)
        raw_env._ball.vel = np.zeros(2, dtype=np.float32)
        raw_env._ball.last_touched_by = None
        raw_env._ball.seen_age_ms = 0.0

        raw_env._select_active_agent()
        raw_env._static_obstacles = raw_env._spawn_static_obstacles()
        raw_env._prev_ball_pos = raw_env._ball.pos.copy()
        raw_env._prev_ball_speed = 0.0
        raw_env._prev_ball_dists = {
            aid: float(np.linalg.norm(raw_env._agents_state[aid].pos - raw_env._ball.pos))
            for aid in raw_env.agents
        }
        raw_env._prev_actions = {
            aid: np.zeros(3, dtype=np.float32) for aid in raw_env.agents
        }
        raw_env._ema_actions = {
            aid: np.zeros(3, dtype=np.float32) for aid in raw_env.agents
        }
        raw_env._prev_thetas = {
            aid: float(raw_env._agents_state[aid].theta) for aid in raw_env.agents
        }
        raw_env._prev_targets = {
            aid: raw_env._agents_state[aid].pos.copy() for aid in raw_env.agents
        }
        raw_env._curr_targets = {}
        raw_env._prev_omegas = {aid: 0.0 for aid in raw_env.agents}
        raw_env._prev_face_thetas = {
            aid: float(raw_env._agents_state[aid].theta) for aid in raw_env.agents
        }
        raw_env._fov_filter.reset(raw_env.agents)

    def capture_world_snapshot(self,
                               stage_id: int,
                               seed: Optional[int] = None,
                               launch_template: Optional[ResolvedLaunchTemplate] = None
                               ) -> WorldSnapshot:
        """Resetea el entorno y captura un snapshot del estado inicial."""
        self.reset_to_stage(stage_id)
        obs, infos = self.reset_episode(seed=seed)
        start_mode = "reset_seed"
        template_name = None
        template_family = None
        if launch_template is not None:
            self._apply_launch_template(launch_template)
            obs, infos = self._refresh_bridge_state()
            start_mode = "launch_template"
            template_name = launch_template.name
            template_family = launch_template.family
        aid = self.controlled_agent_id
        agent = self.raw_env._agents_state[aid]
        ball = self.raw_env._ball
        return WorldSnapshot(
            stage_id=stage_id,
            agent_id=aid,
            agent_pos=np.asarray(agent.pos, dtype=np.float32),
            ball_pos=np.asarray(ball.pos, dtype=np.float32),
            agent_theta=float(agent.theta),
            obs_local=np.asarray(obs[aid], dtype=np.float32),
            state_global=np.asarray(infos[aid]["state"], dtype=np.float32),
            reset_seed=seed,
            start_mode=start_mode,
            template_name=template_name,
            template_family=template_family,
        )

    # ── FIX-A: Lectura del estado real sin reset ─────────────────────────

    def get_live_world_state(self) -> SimplifiedWorldState:
        """Lee el estado ACTUAL del entorno sin resetearlo.
        
        Este método es el corazón del fix closed-loop. En cada paso del
        replay, evaluate_solver llama aquí para obtener la posición REAL
        del agente y del balón, y se la pasa al solver para re-planificar.
        
        No toca el entorno — solo lee. El entorno se modifica solo via step().
        
        Returns:
            SimplifiedWorldState con datos reales del entorno actual.
        """
        aid = self.controlled_agent_id
        agent = self.raw_env._agents_state[aid]
        ball = self.raw_env._ball

        # Detectar si hay contacto activo
        touched = bool(getattr(self.raw_env, "_ball_touched_episode", False))
        
        # Contar recontactos acumulados en el episodio
        # (esto es aproximado; el evaluator lleva cuenta exacta)
        recontacts = 0
        
        # Detectar gol
        goal_scored = bool(self.raw_env._game.own_score > 0)

        agent_info = {}
        if self._last_infos is not None and aid in self._last_infos:
            agent_info = self._last_infos.get(aid, {}) or {}
        has_live_sidecar_data = (
            "sidecar_backend" in agent_info
            or "sidecar_macro_type" in agent_info
            or "bridge_frame_id" in agent_info
        )

        sidecar_macro_type = str(agent_info.get("sidecar_macro_type", "none"))
        handoff_ready = bool(agent_info.get("handoff_ready", False))
        playball_active = (
            bool(agent_info.get("takeover_active", False))
            or sidecar_macro_type.startswith("playBall_")
        )
        ready_but_walk = bool(
            handoff_ready
            and not playball_active
            and agent_info.get("sidecar_skill_request") == "walk"
        )
        takeover_success = bool(
            agent_info.get("macro_success") is True
            or int(agent_info.get("episode_takeover_success", 0)) > 0
        )

        if not agent_info or not has_live_sidecar_data:
            handoff_ready, playball_active, ready_but_walk, takeover_success = (
                estimate_takeover_proxy(
                    np.asarray(agent.pos, dtype=np.float32),
                    np.asarray(ball.pos, dtype=np.float32),
                    float(agent.theta),
                    recontacts=recontacts,
                    objective=get_stage_objective(self.env.curriculum_manager.stage_id),
                    goal_scored=goal_scored,
                    kicked=False,
                )
            )

        return SimplifiedWorldState(
            agent_pos=np.asarray(agent.pos, dtype=np.float32),
            ball_pos=np.asarray(ball.pos, dtype=np.float32),
            heading=float(agent.theta),
            touched_ball=touched,
            recontacts=recontacts,
            goal_scored=goal_scored,
            handoff_ready=handoff_ready,
            playball_active=playball_active,
            ready_but_walk=ready_but_walk,
            takeover_success=takeover_success,
        )

    def agent_position(self, agent_id: str) -> np.ndarray:
        return np.asarray(self.raw_env._agents_state[agent_id].pos,
                          dtype=np.float32)

    def ball_position(self) -> np.ndarray:
        return np.asarray(self.raw_env._ball.pos, dtype=np.float32)

    def make_action_dict(self, teacher_action: np.ndarray) -> Dict[str, np.ndarray]:
        """Construye dict de acciones: teacher para el controlado, zeros para el resto."""
        actions: Dict[str, np.ndarray] = {}
        for aid in self.env.agents:
            if aid == self.controlled_agent_id:
                actions[aid] = np.asarray(teacher_action, dtype=np.float32)
            else:
                actions[aid] = np.zeros(3, dtype=np.float32)
        return actions

    def step(self, actions: Dict[str, np.ndarray]):
        out = self.env.step(actions)
        self._last_obs, _, _, _, self._last_infos = out
        return out


# ─────────────────────────────────────────────────────────────────────────────
# MUESTREO DE PLANES (Gumbel — para diversidad en trayectorias candidatas)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampledPlan:
    selected: PlannedTrajectory
    candidates: List[PlannedTrajectory]
    scores: List[float]


class GumbelPlanSampler:
    def __init__(self, cfg: GumbelConfig):
        self.cfg = cfg

    def sample(self, plans: List[PlannedTrajectory],
               rng: np.random.Generator | None = None) -> SampledPlan:
        if not plans:
            raise ValueError("No plans available for sampling")
        rng = rng or np.random.default_rng()
        scores = np.asarray([p.quality for p in plans], dtype=np.float64)
        if not self.cfg.enabled or len(plans) == 1:
            idx = int(np.argmax(scores))
            return SampledPlan(plans[idx], plans, scores.tolist())
        logits = scores * self.cfg.score_scale
        gumbels = -np.log(-np.log(
            rng.uniform(1e-6, 1.0 - 1e-6, size=len(logits))))
        noisy = (logits + gumbels) / max(self.cfg.temperature, 1e-6)
        idx = int(np.argmax(noisy))
        return SampledPlan(plans[idx], plans, scores.tolist())
