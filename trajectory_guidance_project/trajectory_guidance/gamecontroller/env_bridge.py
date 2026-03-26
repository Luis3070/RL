"""
env_bridge.py — Bridge entre el gamecontroller y el environment real.

Conecta al NaoSoccerEnvWrapper sin duplicar lógica. Todo paso por el
action space real [-1,1]^3 para que el diagnóstico sea válido.
"""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    _here = Path(__file__).resolve().parent
    _ws = _here.parent.parent.parent          # RL/
    sys.path.insert(0, str(_ws))
    sys.path.insert(1, str(_here.parent.parent))  # trajectory_guidance_project/

import numpy as np
from typing import Dict, Optional, Tuple

from config import CURRICULUM, ACTION
from wrapper import NaoSoccerEnvWrapper
from visualizer import CoordTransform, EnvStateReader


class GameEnvBridge:
    """Adaptador fino entre el gamecontroller y NaoSoccerEnvWrapper."""

    def __init__(self, use_fov: bool = True, seed: int = 42):
        self.env = NaoSoccerEnvWrapper(use_fov=use_fov, seed=seed)
        self.reader = EnvStateReader(self.env._env)
        self.controlled_agent_id = "agent_0"
        self._current_stage = 0

    def reset(self, stage_id: int, seed: Optional[int] = None):
        self._current_stage = stage_id
        self.env.set_curriculum_stage(stage_id)
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray):
        """Ejecuta un paso: accion para agent_0, zeros para el resto."""
        actions: Dict[str, np.ndarray] = {}
        for aid in self.env.agents:
            if aid == self.controlled_agent_id:
                actions[aid] = np.asarray(action, dtype=np.float32)
            else:
                actions[aid] = np.zeros(3, dtype=np.float32)
        return self.env.step(actions)

    def get_agent_pos(self) -> np.ndarray:
        return np.asarray(
            self.env._env._agents_state[self.controlled_agent_id].pos,
            dtype=np.float32,
        )

    def get_ball_pos(self) -> np.ndarray:
        return np.asarray(self.env._env._ball.pos, dtype=np.float32)

    def get_max_displacement(self) -> float:
        """Devuelve el max_displacement que usa _decode_action para este stage."""
        stage = CURRICULUM[self._current_stage]
        md = getattr(stage, "local_horizon", ACTION.max_displacement_mm)
        return md if md > 0.0 else ACTION.max_displacement_mm

    @property
    def stage_id(self) -> int:
        return self._current_stage

    @property
    def has_agents(self) -> bool:
        return len(self.env.agents) > 0
