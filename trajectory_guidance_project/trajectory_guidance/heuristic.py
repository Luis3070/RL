"""
heuristic.py — Funciones de costo para el solver A*.

Cambios respecto a la versión anterior:
  FIX-1: ball_out_of_bounds_penalty — penaliza balón fuera de la cancha.
  FIX-2: own_goal_penalty — costo catastrófico si el balón se acerca
         al arco propio. Permite dribble/pase hacia atrás pero impide autogoles.
  FIX-3: sustained_dribble_bonus — premio progresivo por recontactos.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from config import AGENT, CONTRACT, FIELD
from trajectory_guidance.settings import StageObjective


# ── Funciones geométricas ────────────────────────────────────────────────────

def opponent_goal_center() -> np.ndarray:
    return np.array([FIELD.x_opponent_ground, 0.0], dtype=np.float32)

def own_goal_center() -> np.ndarray:
    return np.array([FIELD.x_own_ground, 0.0], dtype=np.float32)

def nearest_goal_post(ball_pos: np.ndarray) -> np.ndarray:
    y = float(np.clip(ball_pos[1], FIELD.y_right_goal, FIELD.y_left_goal))
    return np.array([FIELD.x_opponent_ground, y], dtype=np.float32)

def behind_ball_point(ball_pos: np.ndarray, margin_mm: float = 220.0) -> np.ndarray:
    goal = nearest_goal_post(ball_pos)
    vec = goal - ball_pos
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return ball_pos.copy()
    return ball_pos - (vec / n) * margin_mm

def goal_alignment_score(agent_pos: np.ndarray, ball_pos: np.ndarray) -> float:
    goal = nearest_goal_post(ball_pos)
    ab = ball_pos - agent_pos
    bg = goal - ball_pos
    nab = np.linalg.norm(ab)
    nbg = np.linalg.norm(bg)
    if nab < 1e-6 or nbg < 1e-6:
        return 0.0
    c = float(np.dot(ab, bg) / (nab * nbg))
    return max(-1.0, min(1.0, c))

def signed_lateral_offset(agent_pos: np.ndarray, ball_pos: np.ndarray) -> float:
    goal = nearest_goal_post(ball_pos)
    bg = goal - ball_pos
    ba = agent_pos - ball_pos
    denom = float(np.linalg.norm(bg)) + 1e-6
    cross_z = float(bg[0] * ba[1] - bg[1] * ba[0])
    return cross_z / denom

def ball_goal_distance(ball_pos: np.ndarray) -> float:
    goal = nearest_goal_post(ball_pos)
    return float(np.linalg.norm(goal - ball_pos))

def angle_to_point(origin: np.ndarray, target: np.ndarray) -> float:
    d = target - origin
    return float(math.atan2(d[1], d[0]))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def proxy_handoff_ready(agent_pos: np.ndarray,
                        ball_pos: np.ndarray,
                        heading: float) -> bool:
    """Approximate the real handoff envelope using the same geometry contract."""
    dx = float(ball_pos[0] - agent_pos[0])
    dy = float(ball_pos[1] - agent_pos[1])

    cos_t = math.cos(-heading)
    sin_t = math.sin(-heading)
    rel_x = dx * cos_t - dy * sin_t
    rel_y = dx * sin_t + dy * cos_t

    handoff = CONTRACT.handoff
    if rel_x < float(handoff.ball_rel_x_min):
        return False
    if rel_x > float(handoff.ball_rel_x_max):
        return False
    if abs(rel_y) > float(handoff.ball_rel_y_abs_max):
        return False

    angle_to_ball = math.atan2(dy, dx)
    angle_diff = abs(_normalize_angle(angle_to_ball - heading))
    if math.degrees(angle_diff) > float(handoff.angle_to_ball_abs_max):
        return False
    return True


def estimate_takeover_proxy(agent_pos: np.ndarray,
                            ball_pos: np.ndarray,
                            heading: float,
                            recontacts: int,
                            objective: StageObjective,
                            goal_scored: bool = False,
                            kicked: bool = False) -> tuple[bool, bool, bool, bool]:
    """
    Planner-side proxy flags aligned with handoff/takeover semantics.

    They are search hints, not claims about the real backend. The final judge
    remains the real rollout and its validity gates.
    """
    handoff_ready = proxy_handoff_ready(agent_pos, ball_pos, heading)
    control_dist = max(
        float(getattr(AGENT, "ball_dribble_range", 340.0)),
        0.9 * float(getattr(AGENT, "ball_kick_range", 400.0)),
    )
    touching_ball = euclidean(agent_pos, ball_pos) <= control_dist
    playball_active = handoff_ready and touching_ball
    ready_but_walk = handoff_ready and not playball_active

    takeover_success = bool(goal_scored)
    if objective.success_mode == "touch_and_control":
        takeover_success = takeover_success or (playball_active and recontacts >= 1)
    elif objective.success_mode == "stable_dribble":
        takeover_success = takeover_success or (playball_active and recontacts >= 2)
    elif objective.success_mode == "dribble_align_kick":
        takeover_success = takeover_success or kicked or (playball_active and recontacts >= 2)
    elif objective.success_mode == "goal_completion":
        takeover_success = takeover_success or kicked

    return handoff_ready, playball_active, ready_but_walk, takeover_success


# ── Costos individuales ──────────────────────────────────────────────────────

def agent_ball_distance_cost(agent_pos: np.ndarray,
                             ball_pos: np.ndarray, weight: float) -> float:
    return weight * euclidean(agent_pos, ball_pos)

def ball_goal_distance_cost(ball_pos: np.ndarray, weight: float) -> float:
    return weight * ball_goal_distance(ball_pos)

def turn_penalty(prev_heading: float, next_heading: float,
                 weight: float) -> float:
    d = abs(_normalize_angle(next_heading - prev_heading))
    return weight * d

def control_bonus(agent_pos: np.ndarray, ball_pos: np.ndarray,
                  prefer_dribble: bool, weight: float) -> float:
    dist = euclidean(agent_pos, ball_pos)
    if dist < 240.0:
        return -weight * (1.5 if prefer_dribble else 1.0)
    if dist < 320.0:
        return -weight * 0.7
    return 0.0

def goal_alignment_bonus(agent_pos: np.ndarray, ball_pos: np.ndarray,
                         weight: float) -> float:
    score = goal_alignment_score(agent_pos, ball_pos)
    return -weight * max(score, 0.0)


# ── FIX-1: Balón fuera de cancha ─────────────────────────────────────────────

def ball_out_of_bounds_penalty(ball_pos: np.ndarray, weight: float) -> float:
    """Penaliza posiciones del balón fuera de las líneas del campo.
    Aplica repulsión continua a medida que se acerca a los bordes.
    """
    bx, by = float(ball_pos[0]), float(ball_pos[1])
    abs_y = abs(by)

    # 1. Castigo Categórico si sale de bounds
    if abs_y > FIELD.y_left_sideline or bx > FIELD.x_opponent_ground + 100.0 or bx < FIELD.x_own_ground:
        return 100_000.0

    # 2. Zona de Peligro Continua en los laterales (repulsión progresiva)
    danger_y_margin = 850.0 # Empieza a penalizar a 850mm de la banda
    danger_y = FIELD.y_left_sideline - danger_y_margin
    if abs_y > danger_y:
        penetration = (abs_y - danger_y) / danger_y_margin
        return weight * 400.0 * (penetration ** 2)

    # 3. Zona de Peligro Continua en los fondos (evitar atascos en esquinas)
    danger_x_margin = 600.0
    opponent_danger_x = FIELD.x_opponent_ground - danger_x_margin
    if bx > opponent_danger_x and abs_y > FIELD.y_left_goal + 200.0:
        penetration = (bx - opponent_danger_x) / danger_x_margin
        return weight * 400.0 * (penetration ** 2)

    return 0.0


# ── FIX-2: Autogol ──────────────────────────────────────────────────────────

def own_goal_penalty(ball_pos: np.ndarray, weight: float) -> float:
    """FIX-A: Costo unwalkable si el balón entra en el arco propio."""
    bx, by = float(ball_pos[0]), float(ball_pos[1])
    abs_y = abs(by)

    # Balón dentro del arco propio → unwalkable
    if bx <= FIELD.x_own_ground and abs_y <= FIELD.y_left_goal:
        return 100_000.0

    # ZONA DE PELIGRO: 1200mm enfrente del arco propio (4v4 bounds)
    safe_zone_start = FIELD.x_own_ground + 1200.0
    if bx < safe_zone_start and abs_y < 1500.0:
        depth = (safe_zone_start - bx) / 1200.0
        width_factor = max(0.0, 1.0 - abs_y / 1500.0)
        return weight * 600.0 * depth * width_factor

    return 0.0


# ── FIX-3: Dribble sostenido ────────────────────────────────────────────────

def sustained_dribble_bonus(recontacts: int, prefer_dribble: bool,
                            weight: float) -> float:
    """Bonus progresivo por cadena de recontactos, con cap para evitar infinitos dribbles."""
    if recontacts <= 0:
        return 0.0
    multiplier = 1.5 if prefer_dribble else 1.0
    # Cap a 10 recontactos para evitar que driblear supere el bono de gol
    eff_recontacts = min(float(recontacts), 10.0)
    return -weight * multiplier * math.sqrt(eff_recontacts)


# ── Estado simplificado ──────────────────────────────────────────────────────

@dataclass
class SimplifiedWorldState:
    agent_pos: np.ndarray
    ball_pos: np.ndarray
    heading: float
    touched_ball: bool = False
    recontacts: int = 0
    goal_scored: bool = False
    own_goal: bool = False
    kicked: bool = False
    handoff_ready: bool = False
    playball_active: bool = False
    ready_but_walk: bool = False
    takeover_success: bool = False


# ── Costo heurístico principal ───────────────────────────────────────────────

def heuristic_cost(state: SimplifiedWorldState,
                   objective: StageObjective,
                   next_heading: float | None = None) -> float:
    w = objective.weights
    cost = 0.0

    cost += agent_ball_distance_cost(
        state.agent_pos, state.ball_pos, w.get("agent_ball_dist", 1.0))
    # ELIMINADO FIX-C (Gate binario). La distancia siempre computa
    # orgánicamente para evitar discontinuidades y overfits matemáticos.
    cost += ball_goal_distance_cost(
        state.ball_pos, w.get("ball_goal_dist", 0.0))
    cost += control_bonus(
        state.agent_pos, state.ball_pos,
        objective.prefer_dribble, w.get("control_bonus", 0.0))
    cost += goal_alignment_bonus(
        state.agent_pos, state.ball_pos, w.get("goal_alignment", 0.0))
    if next_heading is not None:
        cost += turn_penalty(
            state.heading, next_heading, w.get("turn_penalty", 0.0))

    # FIX-1 + FIX-2
    cost += ball_out_of_bounds_penalty(
        state.ball_pos, w.get("ball_oob_penalty", 1.0))
    cost += own_goal_penalty(
        state.ball_pos, w.get("own_goal_penalty", 1.0))
    if state.own_goal:
        cost += 5000.0

    # FIX-3
    cost += sustained_dribble_bonus(
        state.recontacts, objective.prefer_dribble,
        w.get("dribble_bonus", 80.0))
    if state.handoff_ready:
        cost -= 90.0 if objective.require_handoff_ready else 35.0
    if state.playball_active:
        cost -= 120.0
    if state.ready_but_walk:
        cost += 85.0
    if state.takeover_success:
        cost -= 220.0 if objective.require_takeover_success else 110.0

    # Bonos por modo de éxito
    if objective.success_mode == "touch_and_control":
        if state.touched_ball:
            cost -= 200.0
        cost -= 120.0 * state.recontacts
    elif objective.success_mode == "stable_dribble":
        cost -= 150.0 * state.recontacts
    elif objective.success_mode == "dribble_align_kick":
        if state.touched_ball:
            cost -= 120.0
        # Bonus escalado por recontactos: incentiva dribble antes del kick
        cost -= 100.0 * state.recontacts
    elif objective.success_mode == "goal_completion":
        # Bono inmenso por anotar el gol
        if state.goal_scored:
            if state.recontacts >= 2:
                cost -= 1200.0
            else:
                cost -= 300.0
        # Bono fuerte incentivo por patear AL ARCO si todavía no entramos (evita evadir el pateo por siempre)
        elif state.kicked:
            cost -= 400.0

    return cost


# ── Score terminal ───────────────────────────────────────────────────────────

def terminal_score(state: SimplifiedWorldState,
                   objective: StageObjective) -> float:
    dist_ab = float(np.linalg.norm(state.agent_pos - state.ball_pos))
    score = 0.0

    if dist_ab < 260.0:
        score += 1.0

    score += 0.5 * math.sqrt(max(0, state.recontacts))
    if state.handoff_ready:
        score += 1.0
    if state.playball_active:
        score += 1.5
    if state.takeover_success:
        score += 4.0
    if state.ready_but_walk:
        score -= 1.5

    if state.goal_scored:
        score += 10.0

    if state.own_goal:
        score -= 20.0

    bx, by = float(state.ball_pos[0]), float(state.ball_pos[1])
    if (bx > FIELD.x_opponent_ground + 50 or bx < FIELD.x_own_ground - 50
            or by > FIELD.y_left_sideline + 50
            or by < FIELD.y_right_sideline - 50):
        score -= 3.0

    return score
