"""
solver.py — A* planner con soporte closed-loop (MPC) para trajectory guidance.

Cambios respecto a la versión anterior:
  FIX-A: Nuevo método solve_greedy_step() que re-planifica desde el estado real
         del entorno en cada paso. Elimina la divergencia open-loop que causaba
         que el agente fuera al centro en vez de al balón.
  FIX-B: quality normalizada por número de pasos. Trayectorias largas (Stage 1-3)
         ya no son penalizadas desproporcionadamente.

Arquitectura MPC (Model Predictive Control):
  En cada step del entorno:
    1. Leer (agent_pos, ball_pos, heading) reales desde el bridge
    2. solver.solve_greedy_step(state, objective) → waypoint inmediato
    3. waypoint_to_normalized_action(agent_pos, waypoint) → teacher_action
    4. env.step(teacher_action) → observar resultado
    5. Repetir
  
  El solver NUNCA asume que su modelo cinemático (step_mm) coincide con la
  física del entorno. Solo dice "la mejor dirección AHORA es esta". El entorno
  se encarga de la velocidad real, EMA, rotación, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Iterable, List, Optional, Tuple

import numpy as np

from config import AGENT, FIELD
from trajectory_guidance.settings import AStarConfig, StageObjective
from trajectory_guidance.heuristic import (
    SimplifiedWorldState,
    behind_ball_point,
    estimate_takeover_proxy,
    heuristic_cost,
    nearest_goal_post,
    terminal_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# PRIMITIVAS DE MOVIMIENTO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Primitive:
    name: str
    step_mm: float
    lateral_mm: float = 0.0
    dribble_gain: float = 0.0
    kick_gain: float = 0.0
    align_bias: float = 0.0
    # FIX: dribble_angle controla la DIRECCIÓN del dribble relativa al arco rival.
    #   0.0  = hacia el arco rival (forward dribble, default)
    #   π/3  = ~60° izquierda (lateral left)
    #  -π/3  = ~60° derecha (lateral right)
    #   π    = hacia atrás (retreat)
    dribble_angle: float = 0.0


def _rotate_vec2(v: np.ndarray, angle: float) -> np.ndarray:
    """Rota un vector 2D por el ángulo dado (radianes)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]],
                    dtype=np.float32)


def _is_ball_in_own_goal(ball_pos: np.ndarray) -> bool:
    """Detecta si el balón está dentro del arco propio."""
    return (float(ball_pos[0]) <= FIELD.x_own_ground
            and abs(float(ball_pos[1])) <= FIELD.y_left_goal)


def _is_ball_out_of_bounds(ball_pos: np.ndarray) -> bool:
    """Detecta si el balón está fuera de la cancha."""
    bx, by = float(ball_pos[0]), float(ball_pos[1])
    return (bx > FIELD.x_opponent_ground + 100
            or bx < FIELD.x_own_ground - 100
            or by > FIELD.y_left_sideline + 100
            or by < FIELD.y_right_sideline - 100)


def primitives_for_stage(stage_id: int) -> List[Primitive]:
    """Genera todas las primitivas disponibles para el stage.
    
    Approach primitives (usadas cuando lejos del balón):
      straight, left_arc, right_arc, recover_contact
    
    Dribble primitives (usadas cuando cerca del balón):
      dribble_safe:    forward, ganancia 0.65
      dribble_aggressive: forward, ganancia 0.90
      dribble_lateral_left:  60° izquierda, repositioning
      dribble_lateral_right: 60° derecha, repositioning
      dribble_retreat: 180° (atrás), ganancia baja, para control
    
    Kick (stage 2+):
      align_then_kick: alinearse y patear al arco
    """
    base = [
        Primitive("straight", step_mm=240.0),
        Primitive("left_arc", step_mm=220.0, lateral_mm=120.0, align_bias=-1.0),
        Primitive("right_arc", step_mm=220.0, lateral_mm=-120.0, align_bias=1.0),
        Primitive("recover_contact", step_mm=160.0),
    ]
    # Forward dribbles
    base += [
        Primitive("dribble_safe", step_mm=180.0, dribble_gain=0.65),
        Primitive("dribble_aggressive", step_mm=240.0, dribble_gain=0.90),
    ]
    # Directional dribbles (disponibles desde stage 1)
    if stage_id >= 1:
        base += [
            Primitive("dribble_lateral_left", step_mm=160.0,
                      dribble_gain=0.50, dribble_angle=math.pi / 3),
            Primitive("dribble_lateral_right", step_mm=160.0,
                      dribble_gain=0.50, dribble_angle=-math.pi / 3),
            Primitive("dribble_retreat", step_mm=140.0,
                      dribble_gain=0.35, dribble_angle=math.pi),
        ]
    if stage_id >= 2:
        base += [
            Primitive("align_then_kick", step_mm=160.0, kick_gain=1.0),
        ]
    return base


def primitive_modes(stage_id: int) -> List[str]:
    return [p.name for p in primitives_for_stage(stage_id)]


def apply_primitive(primitive: Primitive,
                    agent_pos: np.ndarray,
                    ball_pos: np.ndarray,
                    heading: float,
                    stage_id: int,
                    objective: StageObjective) -> tuple[np.ndarray, np.ndarray, float,
                                                        bool, int, bool, np.ndarray, bool, bool]:
    """Simula un paso del agente usando la primitiva dada.
    
    Returns:
        (next_agent, next_ball, next_heading, touched_ball, recontact_inc,
         goal_scored, waypoint_target, own_goal, kicked)
         
    Cambios:
      - dribble_angle: dirección configurable del dribble
      - own_goal detection: si el dribble/kick manda el balón al arco propio
      - OOB clamp: si el balón sale de la cancha, se clampea al borde
    """
    goal = nearest_goal_post(ball_pos)
    target_pos = getattr(objective, "kick_target", None)
    if target_pos is None:
        target_pos = goal.copy()

    to_ball = ball_pos - agent_pos
    d_ball = float(np.linalg.norm(to_ball))
    touched_ball = d_ball <= getattr(AGENT, "ball_dribble_range", 340.0)
    recontact_inc = 1 if touched_ball else 0
    goal_scored = False
    own_goal = False
    kicked = False

    if primitive.name == "align_then_kick":
        if d_ball > getattr(AGENT, "ball_kick_range", 400.0):
            # FIX-C: behind_ball_point hardcodea el arco. Hacemos uno manual para target_pos
            tb_dir = ball_pos - target_pos
            tbn = float(np.linalg.norm(tb_dir)) + 1e-6
            target = ball_pos + (tb_dir / tbn) * 350.0
            
            step_vec = target - agent_pos
            n = float(np.linalg.norm(step_vec)) + 1e-6
            next_agent = agent_pos + (step_vec / n) * min(primitive.step_mm, n)
            next_ball = ball_pos.copy()
        else:
            kick_dir = target_pos - ball_pos
            n = float(np.linalg.norm(kick_dir)) + 1e-6
            # FIX-D: Kick distance for A* tree
            kick_dist = 300.0 + 100.0 * stage_id
            next_ball = ball_pos + (kick_dir / n) * kick_dist
            
            # FIX-A: Para activar el kick en la física (throttle >= 280),
            # el waypoint del agente debe atravesar el balón (120mm + ~300mm = ~420mm)
            next_agent = ball_pos + (kick_dir / n) * 120.0
            target = next_agent.copy()
            
            # FIX-A: NO clampear el balón. Si el kick lo saca de la cancha,
            # la heurística (100K penalty) matará ese nodo del A*.
            goal_scored = bool(next_ball[0] >= goal[0] - 100.0
                               and abs(next_ball[1]) <= 900.0)
            touched_ball = True
            recontact_inc = 1
            kicked = True
    else:
        if d_ball > getattr(AGENT, "ball_dribble_range", 340.0):
            # APPROACH: ir hacia el balón (con offset lateral si aplica)
            target = ball_pos.copy()
            move_dir = to_ball / (d_ball + 1e-6)
            lateral = (np.array([-move_dir[1], move_dir[0]],
                                dtype=np.float32)
                       * primitive.lateral_mm)
            desired = (target + lateral) - agent_pos
            n = float(np.linalg.norm(desired)) + 1e-6
            next_agent = agent_pos + (desired / n) * min(primitive.step_mm, n)
            next_ball = ball_pos.copy()
        else:
            # DRIBBLE: empujar balón en la dirección configurada hacia el target
            goal_dir = target_pos - ball_pos
            n = float(np.linalg.norm(goal_dir)) + 1e-6
            base_dir = goal_dir / n

            # Aplicar rotación de dribble_angle
            if abs(primitive.dribble_angle) > 0.01:
                dribble_dir = _rotate_vec2(base_dir, primitive.dribble_angle)
            else:
                dribble_dir = base_dir

            dribble_step = primitive.step_mm * 0.75
            gain = max(primitive.dribble_gain, 0.25)
            next_agent = agent_pos + dribble_dir * dribble_step
            next_ball = ball_pos + dribble_dir * (dribble_step * gain)
            touched_ball = True
            recontact_inc = 1
        target = next_agent.copy()

    # ── FIX-2: Own goal check ────────────────────────────────────────────
    own_goal = _is_ball_in_own_goal(next_ball)

    # ── FIX-1: OOB safety — clamp ball within field ─────────────────────
    # No clampeamos el balón (el heurístico ya penaliza), pero marcamos
    # para que el solver lo sepa. El entorno real sí maneja el OOB.

    delta = target - agent_pos
    next_heading = (float(np.arctan2(delta[1], delta[0]))
                    if np.linalg.norm(delta) > 1e-6 else heading)
    return (next_agent.astype(np.float32), next_ball.astype(np.float32),
            next_heading, touched_ball, recontact_inc, goal_scored,
            target.astype(np.float32), own_goal, kicked)


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSIÓN WAYPOINT → ACCIÓN TEACHER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TeacherActionStep:
    waypoint: np.ndarray
    action: np.ndarray
    primitive_name: str = ""


def waypoint_to_normalized_action(agent_pos: np.ndarray,
                                  waypoint: np.ndarray,
                                  local_horizon: float,
                                  theta_action: float = 0.0) -> np.ndarray:
    """Convierte waypoint absoluto a acción normalizada [-1, 1].
    
    En modo closed-loop, el waypoint viene del solver y puede estar
    a step_mm de distancia (180-240mm). El local_horizon del stage
    (400-700mm) normaliza eso a un valor en [-1, 1] que el entorno
    decodifica como desplazamiento relativo.
    
    Ejemplo Stage 0: waypoint a 180mm, local_horizon=400mm
      → action = 180/400 = 0.45 → entorno mueve 0.45×400 = 180mm target
      → pero el entorno aplica EMA y velocidad real, así que se mueve ~25mm
      → en el siguiente step, solve_greedy_step recalcula desde la pos real
    """
    local_horizon = max(local_horizon, 1e-6)
    delta = (waypoint - agent_pos) / local_horizon
    return np.array([
        float(np.clip(delta[0], -1.0, 1.0)),
        float(np.clip(delta[1], -1.0, 1.0)),
        float(theta_action),
    ], dtype=np.float32)


def build_teacher_actions(agent_positions: Iterable[np.ndarray],
                          waypoints: Iterable[np.ndarray],
                          local_horizon: float) -> List[TeacherActionStep]:
    out: List[TeacherActionStep] = []
    for pos, wp in zip(agent_positions, waypoints):
        out.append(TeacherActionStep(
            waypoint=np.asarray(wp, dtype=np.float32),
            action=waypoint_to_normalized_action(
                np.asarray(pos, dtype=np.float32),
                np.asarray(wp, dtype=np.float32),
                local_horizon),
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NODOS Y TRAYECTORIAS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryNode:
    state: SimplifiedWorldState
    depth: int
    g_cost: float
    h_cost: float
    mode: str
    primitive_seq: List[str] = field(default_factory=list)
    waypoints: List[np.ndarray] = field(default_factory=list)
    parent_key: tuple | None = None

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def key(self, grid_mm: float = 100.0) -> Tuple[int, int, int, int, int, int]:
        ap = self.state.agent_pos
        bp = self.state.ball_pos
        return (
            int(round(ap[0] / grid_mm)),
            int(round(ap[1] / grid_mm)),
            int(round(bp[0] / grid_mm)),
            int(round(bp[1] / grid_mm)),
            int(self.state.recontacts),
            int(self.state.goal_scored),
        )


def trajectory_distance(a: TrajectoryNode, b: TrajectoryNode) -> float:
    end_agent = float(np.linalg.norm(a.state.agent_pos - b.state.agent_pos))
    end_ball = float(np.linalg.norm(a.state.ball_pos - b.state.ball_pos))
    return end_agent + end_ball


@dataclass
class PlannedTrajectory:
    mode: str
    primitive_seq: List[str]
    waypoints: List[np.ndarray]
    terminal_state: SimplifiedWorldState
    planner_cost: float
    terminal_score: float

    @property
    def quality(self) -> float:
        """FIX-B: Quality normalizada por número de pasos.
        
        Problema anterior: quality = terminal_score - 0.001 * planner_cost
          Stage 0 (500mm, ~3 pasos): cost≈600 → quality ≈ 2.0 - 0.6 = 1.4 ✓
          Stage 3 (3500mm, ~20 pasos): cost≈8000 → quality ≈ 2.0 - 8.0 = -6.0 ✗
        
        Fix: normalizar por horizonte. El costo PER-STEP es lo comparable.
          Stage 3: cost_per_step = 8000/20 = 400 → quality ≈ 2.0 - 0.4 = 1.6 ✓
        """
        n_steps = max(len(self.waypoints), 1)
        cost_per_step = self.planner_cost / n_steps
        return float(self.terminal_score - 0.001 * cost_per_step)

    @property
    def trajectory_id(self) -> str:
        prefix = "-".join(self.primitive_seq[:5]) if self.primitive_seq else self.mode
        return f"{self.mode}::{prefix}"


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO DE UN PASO GREEDY (para closed-loop)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GreedyStepResult:
    """Resultado de solve_greedy_step: un waypoint inmediato + metadata."""
    waypoint: np.ndarray
    primitive_name: str
    h_cost: float
    touched_ball: bool
    goal_scored: bool


# ─────────────────────────────────────────────────────────────────────────────
# A* SOLVER
# ─────────────────────────────────────────────────────────────────────────────

class AStarSolver:
    def __init__(self, cfg: AStarConfig):
        self.cfg = cfg

    # ── Método principal existente (para análisis de trayectorias completas) ──

    def solve_topk(self,
                   initial_state: SimplifiedWorldState,
                   objective: StageObjective,
                   modes: Optional[List[str]] = None,
                   top_k: Optional[int] = None) -> List[PlannedTrajectory]:
        modes = modes or objective.primitive_modes
        top_k = top_k or self.cfg.top_k
        candidates: List[PlannedTrajectory] = []
        for mode in modes:
            solved = self._solve_single_mode(initial_state, objective, mode)
            if solved is not None:
                candidates.append(solved)
        candidates.sort(key=lambda x: x.quality, reverse=True)
        return self._filter_diverse(candidates, top_k)

    # ── FIX-A: Nuevo método greedy para closed-loop (MPC) ────────────────

    def solve_greedy_step(self,
                          state: SimplifiedWorldState,
                          objective: StageObjective,
                          lookahead: int = 3,
                          forced_primitive: str | None = None,
                          forbidden_primitives: list[str] | None = None,
                          ) -> GreedyStepResult:
        """Planifica el MEJOR SIGUIENTE PASO desde el estado actual real.
        
        Arquitectura MPC: busca con profundidad 'lookahead' pasos hacia
        adelante, pero solo retorna el PRIMER waypoint del mejor camino.
        
        Args:
            state: Estado real del mundo (del bridge, NO del solver).
            objective: Objetivo del stage actual.
            lookahead: Profundidad de búsqueda (default=3).
            forced_primitive: Si se especifica, SOLO usa esta primitiva.
                Esto genera diversidad entre episodios: un episodio usa
                "left_arc" (trayectoria curva izquierda), otro "right_arc"
                (curva derecha), otro "straight" (directo), etc.
                EXCEPCIÓN: si el agente está en rango de dribble/kick,
                se desbloquean las primitivas de dribble/kick para que
                el agente no se quede pegado sin poder actuar.
            
        Returns:
            GreedyStepResult con el waypoint inmediato a seguir.
        """
        all_primitives = primitives_for_stage(objective.stage_id)

        if forced_primitive is not None:
            # Mode-locked: usar solo la primitiva forzada para approach.
            # Pero si estamos cerca del balón, desbloquear dribble/kick.
            dist_to_ball = float(np.linalg.norm(
                state.agent_pos - state.ball_pos))
            in_contact_zone = dist_to_ball <= getattr(AGENT, "ball_dribble_range", 340.0) * 1.3

            if in_contact_zone:
                # Cerca del balón: usar la forzada + dribble + kick
                dribble_kick = [p for p in all_primitives
                                if "dribble" in p.name or "kick" in p.name]
                forced = [p for p in all_primitives
                          if p.name == forced_primitive]
                primitives = list({p.name: p for p in forced + dribble_kick}.values())
            else:
                # Lejos del balón: solo la primitiva forzada (approach)
                primitives = [p for p in all_primitives
                              if p.name == forced_primitive]
        else:
            active_modes = objective.primitive_modes
            primitives = [p for p in all_primitives
                          if p.name in active_modes]

        if not primitives:
            primitives = all_primitives

        # Solución 4: Filtrar primitivas prohibidas
        if forbidden_primitives:
            primitives = [p for p in primitives
                          if p.name not in forbidden_primitives]
            if not primitives:
                primitives = all_primitives

        mode_label = forced_primitive or "greedy"
        root = TrajectoryNode(
            state=state,
            depth=0,
            g_cost=0.0,
            h_cost=heuristic_cost(state, objective),
            mode=mode_label,
            primitive_seq=[],
            waypoints=[],
        )

        frontier = [(root.f_cost, 0, root)]
        push_id = 1
        best_seen = {root.key(self.cfg.grid_mm): root.g_cost}
        best_leaf: Optional[TrajectoryNode] = None
        best_leaf_score = -math.inf
        expansions = 0

        # Límite de expansiones proporcional al lookahead
        max_exp = min(self.cfg.max_expansions, lookahead * len(primitives) * 50)

        while frontier and expansions < max_exp:
            _, _, current = heapq.heappop(frontier)
            expansions += 1

            # Evaluar como candidato si tiene al menos 1 paso
            if current.depth >= 1:
                score = terminal_score(current.state, objective)
                # Bonus por profundidad: preferir caminos que miran más lejos
                score += 0.05 * current.depth
                if score > best_leaf_score:
                    best_leaf = current
                    best_leaf_score = score

            # No expandir más allá del lookahead
            if current.depth >= lookahead:
                continue

            for primitive in primitives:
                (next_agent, next_ball, next_heading,
                 touched, recontact_inc, goal_scored,
                 waypoint, prim_own_goal, kicked) = apply_primitive(
                    primitive,
                    current.state.agent_pos,
                    current.state.ball_pos,
                    current.state.heading,
                    objective.stage_id,
                    objective,
                )
                next_recontacts = current.state.recontacts + recontact_inc
                handoff_ready, playball_active, ready_but_walk, takeover_success = (
                    estimate_takeover_proxy(
                        next_agent,
                        next_ball,
                        next_heading,
                        recontacts=next_recontacts,
                        objective=objective,
                        goal_scored=(current.state.goal_scored or goal_scored),
                        kicked=(current.state.kicked or kicked),
                    )
                )
                next_state = SimplifiedWorldState(
                    agent_pos=next_agent,
                    ball_pos=next_ball,
                    heading=next_heading,
                    touched_ball=current.state.touched_ball or touched,
                    recontacts=next_recontacts,
                    goal_scored=current.state.goal_scored or goal_scored,
                    own_goal=current.state.own_goal or prim_own_goal,
                    kicked=current.state.kicked or kicked,
                    handoff_ready=handoff_ready,
                    playball_active=playball_active,
                    ready_but_walk=ready_but_walk,
                    takeover_success=current.state.takeover_success or takeover_success,
                )
                step_cost = heuristic_cost(next_state, objective,
                                           next_heading=next_heading)
                g_cost = current.g_cost + max(step_cost, -300.0)
                child = TrajectoryNode(
                    state=next_state,
                    depth=current.depth + 1,
                    g_cost=g_cost,
                    h_cost=self.cfg.heuristic_weight * heuristic_cost(
                        next_state, objective),
                    mode=mode_label,
                    primitive_seq=current.primitive_seq + [primitive.name],
                    waypoints=current.waypoints + [waypoint],
                )
                k = child.key(self.cfg.grid_mm)
                if self.cfg.use_closed_set and k in best_seen and best_seen[k] <= child.g_cost:
                    continue
                best_seen[k] = child.g_cost
                heapq.heappush(frontier, (child.f_cost, push_id, child))
                push_id += 1

        # Fallback: si no encontramos nada, ir directo al balón
        if best_leaf is None or not best_leaf.waypoints:
            fallback_wp = state.ball_pos.copy()
            return GreedyStepResult(
                waypoint=fallback_wp,
                primitive_name="fallback_to_ball",
                h_cost=0.0,
                touched_ball=False,
                goal_scored=False,
            )

        # Retornar solo el PRIMER waypoint del mejor camino
        return GreedyStepResult(
            waypoint=best_leaf.waypoints[0],
            primitive_name=best_leaf.primitive_seq[0],
            h_cost=best_leaf.h_cost,
            touched_ball=best_leaf.state.touched_ball,
            goal_scored=best_leaf.state.goal_scored,
        )

    # ── A* completo (usado para análisis, NO para replay) ────────────────

    def _solve_single_mode(self,
                           initial_state: SimplifiedWorldState,
                           objective: StageObjective,
                           mode: str) -> Optional[PlannedTrajectory]:
        all_primitives = primitives_for_stage(objective.stage_id)
        if "dribble" in mode:
            primitives = [p for p in all_primitives if p.name == mode]
        else:
            primitives = [p for p in all_primitives
                          if p.name == mode or (mode in {"straight", "left_arc", "right_arc"}
                                                and p.name == mode)]
        if not primitives:
            primitives = all_primitives

        root = TrajectoryNode(
            state=initial_state, depth=0, g_cost=0.0,
            h_cost=heuristic_cost(initial_state, objective),
            mode=mode, primitive_seq=[], waypoints=[],
        )

        frontier = [(root.f_cost, 0, root)]
        push_id = 1
        best_seen = {root.key(self.cfg.grid_mm): root.g_cost}
        best_terminal: Optional[TrajectoryNode] = None
        best_terminal_score = -math.inf
        expansions = 0

        while frontier and expansions < self.cfg.max_expansions:
            _, _, current = heapq.heappop(frontier)
            expansions += 1

            is_terminal = self._is_terminal(current, objective)
            if is_terminal:
                score = terminal_score(current.state, objective)
                if score > best_terminal_score:
                    best_terminal = current
                    best_terminal_score = score

            if current.depth >= min(self.cfg.max_search_depth,
                                    objective.planning_horizon_steps):
                if current.depth > 0:
                    score = terminal_score(current.state, objective)
                    if score > best_terminal_score:
                        best_terminal = current
                        best_terminal_score = score
                continue

            for primitive in primitives:
                (next_agent, next_ball, next_heading,
                 touched, recontact_inc, goal_scored,
                 waypoint, prim_own_goal, kicked) = apply_primitive(
                    primitive,
                    current.state.agent_pos,
                    current.state.ball_pos,
                    current.state.heading,
                    objective.stage_id,
                    objective,
                )
                next_recontacts = current.state.recontacts + recontact_inc
                handoff_ready, playball_active, ready_but_walk, takeover_success = (
                    estimate_takeover_proxy(
                        next_agent,
                        next_ball,
                        next_heading,
                        recontacts=next_recontacts,
                        objective=objective,
                        goal_scored=(current.state.goal_scored or goal_scored),
                        kicked=(current.state.kicked or kicked),
                    )
                )
                next_state = SimplifiedWorldState(
                    agent_pos=next_agent,
                    ball_pos=next_ball,
                    heading=next_heading,
                    touched_ball=current.state.touched_ball or touched,
                    recontacts=next_recontacts,
                    goal_scored=current.state.goal_scored or goal_scored,
                    own_goal=current.state.own_goal or prim_own_goal,
                    kicked=current.state.kicked or kicked,
                    handoff_ready=handoff_ready,
                    playball_active=playball_active,
                    ready_but_walk=ready_but_walk,
                    takeover_success=current.state.takeover_success or takeover_success,
                )
                step_cost = heuristic_cost(next_state, objective,
                                           next_heading=next_heading)
                g_cost = current.g_cost + max(step_cost, -300.0)
                child = TrajectoryNode(
                    state=next_state,
                    depth=current.depth + 1,
                    g_cost=g_cost,
                    h_cost=self.cfg.heuristic_weight * heuristic_cost(
                        next_state, objective),
                    mode=mode,
                    primitive_seq=current.primitive_seq + [primitive.name],
                    waypoints=current.waypoints + [waypoint],
                    parent_key=current.key(self.cfg.grid_mm),
                )
                k = child.key(self.cfg.grid_mm)
                if self.cfg.use_closed_set and k in best_seen and best_seen[k] <= child.g_cost:
                    continue
                best_seen[k] = child.g_cost
                heapq.heappush(frontier, (child.f_cost, push_id, child))
                push_id += 1

        if best_terminal is None:
            return None

        return PlannedTrajectory(
            mode=mode,
            primitive_seq=best_terminal.primitive_seq,
            waypoints=best_terminal.waypoints,
            terminal_state=best_terminal.state,
            planner_cost=best_terminal.g_cost,
            terminal_score=best_terminal_score,
        )

    def _is_terminal(self, node: TrajectoryNode,
                     objective: StageObjective) -> bool:
        state = node.state
        if objective.require_handoff_ready and not state.handoff_ready:
            return False
        if objective.require_takeover_success and not state.takeover_success:
            return False
        if objective.success_mode == "goal_completion":
            return state.goal_scored
        elif objective.success_mode == "touch_and_control":
            return state.touched_ball
        elif objective.success_mode == "stable_dribble":
            return (state.touched_ball
                    and state.recontacts >= objective.target_recontacts)
        elif objective.success_mode == "dribble_align_kick":
            return state.touched_ball
        else:
            return state.touched_ball

    def _filter_diverse(self, candidates: List[PlannedTrajectory],
                        top_k: int) -> List[PlannedTrajectory]:
        if len(candidates) <= top_k:
            return candidates
        selected = []
        for candidate in candidates:
            if len(selected) >= top_k:
                break
            is_diverse = True
            for existing in selected:
                ep_dist = float(np.linalg.norm(
                    candidate.terminal_state.agent_pos
                    - existing.terminal_state.agent_pos))
                ball_dist = float(np.linalg.norm(
                    candidate.terminal_state.ball_pos
                    - existing.terminal_state.ball_pos))
                if (ep_dist < self.cfg.diversity_min_endpoint_dist_mm
                        and ball_dist < self.cfg.diversity_min_ball_dist_mm):
                    is_diverse = False
                    break
            if is_diverse:
                selected.append(candidate)
        return selected
