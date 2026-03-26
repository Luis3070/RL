"""
controller.py — Convierte inputs humanos (clicks, teclas) en acciones [-1,1]^3.

Mecánica de kick/pass:
  El environment NO tiene acciones explícitas de kick/pass. El kick se dispara
  automáticamente cuando:
    - El agente está a dist < ball_kick_range (350mm) del balón
    - El balón está en el hemisferio frontal (±90° de agent.theta)
    - throttle >= 280mm (walk_target lejos del agente)
    - angle_diff <= kick_half_rad (alineación frontal)

  Para generar un kick desde el controlador humano:
    1. Se calcula un target MUY lejano en la dirección deseada (a través del balón)
    2. Esto satura la acción a ±1 → max_displacement ~400mm → throttle > 280mm
    3. El environment detecta throttle alto + alineación → kick automático

  Para dribble: clicks normales cerca del agente → throttle bajo → dribble.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from config import FIELD


class HumanController:
    """Genera acciones normalizadas [-1,1]^3 a partir de inputs del usuario."""

    def __init__(self):
        self.target_mm: Optional[np.ndarray] = None
        self._mode = "move"   # "move" | "shoot" | "pass"

    @property
    def mode(self) -> str:
        return self._mode

    # ── Setters de target ──────────────────────────────────────────────────

    def set_move_target(self, x_mm: float, y_mm: float) -> None:
        """Click izquierdo → mover al agente a esta posición."""
        self.target_mm = np.array([x_mm, y_mm], dtype=np.float32)
        self._mode = "move"

    def set_shoot(self, ball_pos: np.ndarray) -> None:
        """Space → disparar al arco rival (x=4500, y=0).

        Genera target lejano a través del balón en dirección al arco rival.
        El throttle alto + alineación frontal activan el kick del environment.
        """
        goal_center = np.array(
            [FIELD.x_opponent_ground, 0.0], dtype=np.float32
        )
        direction = goal_center - ball_pos
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0], dtype=np.float32)

        self.target_mm = (ball_pos + direction * 2000.0).astype(np.float32)
        self._mode = "shoot"

    def set_pass(self, ball_pos: np.ndarray, cursor_mm: np.ndarray) -> None:
        """P → pasar en dirección del cursor.

        El cursor indica la dirección del pase. Se genera un target lejano
        a través del balón en esa dirección.
        """
        direction = cursor_mm - ball_pos
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0], dtype=np.float32)

        self.target_mm = (ball_pos + direction * 2000.0).astype(np.float32)
        self._mode = "pass"

    # ── Generación de acción ───────────────────────────────────────────────

    def compute_action(
        self, agent_pos: np.ndarray, max_displacement: float
    ) -> np.ndarray:
        """Devuelve acción normalizada [-1,1]^3 para el step del environment.

        action[0] = dx relativo / max_displacement  (clipped a [-1,1])
        action[1] = dy relativo / max_displacement  (clipped a [-1,1])
        action[2] = 0.0  (theta manejado por environment: face-ball automático)
        """
        if self.target_mm is None:
            return np.zeros(3, dtype=np.float32)

        displacement = self.target_mm - agent_pos
        action = np.zeros(3, dtype=np.float32)
        action[0] = float(np.clip(displacement[0] / max_displacement, -1.0, 1.0))
        action[1] = float(np.clip(displacement[1] / max_displacement, -1.0, 1.0))
        # action[2] = 0.0 → face-ball automático (stages 0-2), libre en stage 3
        return action
