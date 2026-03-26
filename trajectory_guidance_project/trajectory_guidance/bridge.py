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

from config import CURRICULUM, AGENT
from wrapper import NaoSoccerEnvWrapper
from trajectory_guidance.settings import ENV_BRIDGE_CFG, GumbelConfig
from trajectory_guidance.heuristic import SimplifiedWorldState
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

    def to_simplified_state(self) -> SimplifiedWorldState:
        return SimplifiedWorldState(
            agent_pos=self.agent_pos.copy(),
            ball_pos=self.ball_pos.copy(),
            heading=float(self.agent_theta),
            touched_ball=False,
            recontacts=0,
            goal_scored=False,
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

    def capture_world_snapshot(self, stage_id: int, seed: Optional[int] = None) -> WorldSnapshot:
        """Resetea el entorno y captura un snapshot del estado inicial."""
        self.reset_to_stage(stage_id)
        obs, infos = self.reset_episode(seed=seed)
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

        return SimplifiedWorldState(
            agent_pos=np.asarray(agent.pos, dtype=np.float32),
            ball_pos=np.asarray(ball.pos, dtype=np.float32),
            heading=float(agent.theta),
            touched_ball=touched,
            recontacts=recontacts,
            goal_scored=goal_scored,
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