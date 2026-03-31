"""
evaluate_solver.py — Closed-loop MPC con approach a posición de lanzamiento.

REDISEÑO FUNDAMENTAL vs versión anterior:

  ANTES (dual-phase, fallaba):
    Si lejos → solver va al balón
    Si cerca → filtro reposiciona  
    Problema: 480mm de margen = 5 steps para reposicionar = insuficiente.
    El agente llegaba del lado equivocado y empujaba el balón fuera.

  AHORA (single-phase, launch position):
    SIEMPRE → ir a la posición de lanzamiento (detrás del balón).
    La posición de lanzamiento se calcula para que ball_dir (agent→ball)
    apunte en la dirección deseada cuando el agente llega.
    
    Ejemplo: quiero empujar el balón hacia x+ (arco rival).
    → launch_pos = ball_pos - [1,0] * 350mm
    → El agente va a un punto 350mm a la IZQUIERDA del balón (en x).
    → Cuando llega, ball_dir = (ball - launch) = [1,0] → balón va a x+.

  La dirección de empuje se calcula con constraints:
    - NO empujar hacia la sideline más cercana
    - NO empujar hacia el arco propio
    - dribble_bias_angle rota la dirección base para diversidad
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CURRICULUM, FIELD, AGENT
from trajectory_guidance.settings import (
    DATASET_CFG,
    StageObjective,
    get_stage_objective,
)
from trajectory_guidance.dataset import DemoSnapshot
from trajectory_guidance.bridge import EnvBridge, WorldSnapshot
from trajectory_guidance.heuristic import SimplifiedWorldState, nearest_goal_post
from trajectory_guidance.solver import (
    AStarSolver, PlannedTrajectory, GreedyStepResult,
    waypoint_to_normalized_action,
)


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE POSICIÓN DE LANZAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def is_out_of_bounds(ball_pos: np.ndarray) -> bool:
    """Detecta si el balón está fuera de la cancha (sidelines, fondos)."""
    bx, by = float(ball_pos[0]), float(ball_pos[1])
    return (bx > FIELD.x_opponent_ground + 100
            or bx < FIELD.x_own_ground - 100
            or by > FIELD.y_left_sideline + 100
            or by < FIELD.y_right_sideline - 100)


def _rotate_vec2(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]],
                    dtype=np.float32)


def compute_safe_push_direction(ball_pos: np.ndarray,
                                dribble_bias_angle: float = 0.0,
                                target_dir: Optional[np.ndarray] = None
                                ) -> np.ndarray:
    """Calcula la dirección de empuje SEGURA para el balón.
    
    Paso 1: Dirección base = hacia arco rival (o hacia target_dir si se provee).
    Paso 2: Rotar por dribble_bias_angle (diversidad).
    Paso 3: Aplicar constraints de seguridad:
      - Si empujaría el balón fuera por la sideline → redirigir
      - Si empujaría hacia el arco propio → redirigir
    
    Returns:
        Vector unitario 2D de la dirección segura de empuje.
    """
    bx, by = float(ball_pos[0]), float(ball_pos[1])
    
    # Paso 1: dirección base
    if target_dir is not None:
        base_dir = target_dir
    else:
        goal_center = np.array([FIELD.x_opponent_ground, 0.0], dtype=np.float32)
        base_dir = goal_center - ball_pos
        
    base_norm = float(np.linalg.norm(base_dir))
    if base_norm < 1e-6:
        base_dir = np.array([1.0, 0.0], dtype=np.float32)
    else:
        base_dir = base_dir / base_norm
    
    # Paso 2: aplicar sesgo angular
    push = _rotate_vec2(base_dir, dribble_bias_angle)
    
    # Paso 3: constraints de seguridad

    # ── Sidelines (izquierda y derecha) ──────────────────────────────
    # Se clampea push[y] de forma continua: cuando el espacio libre
    # entre el balón y la zona de keep-out (1000mm) se agota, push[y] → 0.
    # Esto reemplaza la fórmula anterior "push[1] -= correction*(push[1]+0.5)"
    # que tenía dos bugs:
    #   (a) El +0.5 restaba incluso cuando push[1]≈0, causando un flip
    #       de dirección que congelaba al agente en la sideline (Prob 1).
    #   (b) La zona de 600mm no capturaba ángulos de 90°+ desde el centro
    #       del campo, permitiendo que el balón llegara a la sideline (Prob 2).
    _KEEPOUT = 1000.0
    if push[1] > 0.0:
        remaining_left = FIELD.y_left_sideline - by - _KEEPOUT
        max_push_y = max(0.0, remaining_left / _KEEPOUT)
        if push[1] > max_push_y:
            push[1] = float(max_push_y)
            push = push / (float(np.linalg.norm(push)) + 1e-6)
    elif push[1] < 0.0:
        remaining_right = by - FIELD.y_right_sideline - _KEEPOUT
        max_push_y_neg = max(0.0, remaining_right / _KEEPOUT)
        if -push[1] > max_push_y_neg:
            push[1] = -float(max_push_y_neg)
            push = push / (float(np.linalg.norm(push)) + 1e-6)

    # ── Fondo rival (salvo que apunte al gol) ───────────────────────
    dist_to_opp_ground = FIELD.x_opponent_ground - bx
    if dist_to_opp_ground < 600.0 and push[0] > 0.0:
        if abs(by) > FIELD.y_left_goal + 200:
            # Fuera de zona de gol → no empujar más hacia fondo
            push[0] = min(push[0], 0.1)
            push = push / (float(np.linalg.norm(push)) + 1e-6)

    # ── Arco propio (own goal prevention) ───────────────────────────
    # La predicción anterior a 2000mm era insuficiente: un balón en
    # x=-2000 con push_x=-0.7 predice a x=-3400, que no alcanzaba el
    # umbral x_own_ground+800=-3700 (autogol ocurría igual). Se extiende
    # la predicción hasta el fondo propio * 1.3 para capturar trayectorias
    # backward desde el mediocampo. El redirect usa dirección diagonal
    # forward en vez del [-0.1,±0.9] anterior que causaba el mismo bucle
    # de sideline del Problema 2.
    if push[0] < -0.05:
        dist_to_own_ground = max(bx - FIELD.x_own_ground, 500.0)
        prediction_dist = dist_to_own_ground * 1.3
        predicted = ball_pos + push * prediction_dist
        pred_x = float(predicted[0])
        pred_y_abs = abs(float(predicted[1]))

        if pred_x < FIELD.x_own_ground + 800 and pred_y_abs < FIELD.y_left_goal + 500:
            # Redirigir diagonal forward para salir del peligro
            if by >= 0:
                push = np.array([0.6, -0.8], dtype=np.float32)
            else:
                push = np.array([0.6, 0.8], dtype=np.float32)
            push = push / float(np.linalg.norm(push))
    
    # ── Zona crítica: MUY cerca del arco propio → SIEMPRE forward ──
    if bx < -3500.0 and abs(by) < 1200.0:
        # Forzar dirección forward+lateral (escape zone)
        if abs(by) < 400:
            # Centro → ir diagonal para escapar
            push = np.array([0.8, 0.6 if by >= 0 else -0.6], dtype=np.float32)
        else:
            # Lateral → ir forward
            push = np.array([0.9, 0.1 if by >= 0 else -0.1], dtype=np.float32)
        push = push / float(np.linalg.norm(push))
    
    # Normalizar resultado final
    norm = float(np.linalg.norm(push))
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return push / norm


def compute_orbit_waypoint(agent_pos: np.ndarray,
                           ball_pos: np.ndarray,
                           push_dir: np.ndarray,
                           orbit_radius: float = 280.0) -> np.ndarray:
    """Waypoint orbital para stages de dribble (Stage 1).

    El movimiento orbital cambia ball_direction_from_agent a ~4.5°/step.
    La EMA (face_alpha=0.30) introduce un lag de ~11° respecto a theta.
    Con frontal_kick_angle_deg=10° (kick_half_rad=5°), 11° > 5° → dribble.

    El orbit_radius=280mm está dentro de ball_dribble_range (320mm),
    por lo que el dribble puede activarse directamente desde la órbita.

    Dirección de órbita: hacia "detrás del balón" (arco propio),
    manteniendo ball_dir apuntando hacia delante → dribble empuja forward.
    """
    diff = agent_pos - ball_pos
    current_dist = float(np.linalg.norm(diff))
    if current_dist < 1e-6:
        current_angle = math.atan2(float(-push_dir[1]), float(-push_dir[0]))
    else:
        current_angle = math.atan2(float(diff[1]), float(diff[0]))

    # Dirección de órbita: hacia el lado donde el agente ya está
    perp = np.array([-push_dir[1], push_dir[0]], dtype=np.float32)
    lateral = float(np.dot(diff, perp))
    orbit_step = math.pi / 4  # 45° adelante en la órbita cada step
    if lateral >= 0.0:
        next_angle = current_angle - orbit_step  # sentido horario
    else:
        next_angle = current_angle + orbit_step  # antihorario

    return (ball_pos + np.array([
        orbit_radius * math.cos(next_angle),
        orbit_radius * math.sin(next_angle),
    ], dtype=np.float32))


def compute_launch_position(agent_pos: np.ndarray,
                            ball_pos: np.ndarray,
                            dribble_bias_angle: float = 0.0,
                            offset_mm: float = 350.0,
                            close_offset_mm: float = 80.0,
                            target_dir: np.ndarray | None = None,
                            ) -> tuple[np.ndarray, str]:
    """
    target_dir: si se proporciona, usa esta dirección como base de push en lugar
                de goal_center. Permite que el approach pre-kick respete kick_target
                en vez de siempre apuntar al centro del arco.
    """
    """Calcula la posición desde la cual el agente debe llegar al balón.

    INVARIANTE CRÍTICO: el waypoint SIEMPRE está detrás del balón en push_dir.
    El agente nunca se dirige más allá del balón → theta nunca apunta hacia
    atrás → el kick siempre sale hacia el arco rival.

    Por qué es crítico:
      - El entorno congela agent.theta al entrar en ball_kick_range (350mm).
      - Si el agente sobrepasa el balón, se convierte en el objetivo más cercano
        al balón, theta congela apuntando HACIA ATRÁS, y el kick sale en dirección
        al arco propio → autogol / balón fuera por fondo propio.
      - Con contact_push_mm=+180 (versión anterior), el waypoint cruzaba al otro
        lado del balón a ~864mm de distancia → loop en 8 y kicks al revés.

    Bias angle:
      - Lejos (>1920mm): bias completo → curvas de approach diversas.
      - Cerca (≤320mm): bias=0 → agente llega casi directo desde atrás.

    Si el agente ya está DELANTE del balón (wrong side), se calcula un waypoint
    lateral para circular alrededor sin giro de 180° (que dispararía ROTATION_THRESHOLD).
    """
    dist = float(np.linalg.norm(agent_pos - ball_pos))

    # Fase
    if dist > AGENT.ball_dribble_range * 2.0:
        phase = "approach"
    elif dist > AGENT.ball_dribble_range:
        phase = "positioning"
    else:
        phase = "contact"

    # Se elimina la reducción gradual (espiral) de bias. El bias se mantiene constante
    # para generar un vector de acercamiento estable hacia el punto deseado.
    effective_bias = dribble_bias_angle
    
    # Suavizado de offset (distancia atrás) para que no rebote
    approach_dist = AGENT.ball_dribble_range * 6.0  # ≈1920mm
    t = float(np.clip((dist - AGENT.ball_dribble_range) / (approach_dist - AGENT.ball_dribble_range), 0.0, 1.0))

    if target_dir is not None:
        # Respetar la intencion del target (pase/tiro o postura lateral),
        # pero siempre pasando por el filtro de seguridad del campo.
        push_dir = compute_safe_push_direction(
            ball_pos, effective_bias, target_dir=target_dir)
    else:
        push_dir = compute_safe_push_direction(ball_pos, effective_bias)

    # Detección: ¿el agente está DELANTE del balón (wrong side)?
    # ahead_component > 0 → agente entre balón y arco rival → kick iría hacia atrás
    ahead_component = float(np.dot(agent_pos - ball_pos, push_dir))

    if ahead_component > 0.0:
        # Agente pasó el balón → swing lateral para circular sin giro de 180°.
        # Elegir el lado donde el agente ya está (minimiza rotación).
        perp = np.array([-push_dir[1], push_dir[0]], dtype=np.float32)
        lateral = float(np.dot(agent_pos - ball_pos, perp))
        swing_sign = 1.0 if lateral >= 0.0 else -1.0
        # Punto 400mm detrás del balón + 400mm al lateral → llega por detrás sin U-turn
        launch = (ball_pos
                  - push_dir * 400.0
                  + perp * swing_sign * 400.0).astype(np.float32)
    else:
        # Caso normal: waypoint SIEMPRE detrás del balón (offset siempre negativo).
        # Lejos (t=1.0): -350mm detrás del balón.
        # Cerca (t=0.0):  -80mm detrás del balón (contacto suave, sin sobrepasar).
        effective_offset = -(offset_mm * t + close_offset_mm * (1.0 - t))
        launch = (ball_pos + push_dir * effective_offset).astype(np.float32)

    # Clamp dentro del campo
    launch[0] = float(np.clip(launch[0],
                               FIELD.x_own_ground + 250,
                               FIELD.x_opponent_ground - 150))
    launch[1] = float(np.clip(launch[1],
                               FIELD.y_right_sideline + 250,
                               FIELD.y_left_sideline - 250))

    return launch, phase


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RolloutMetrics:
    stage_id: int
    intent: str = "control"
    steps: int = 0
    touched_ball: bool = False
    goal_scored: bool = False
    recontacts: int = 0
    mean_reward: float = 0.0
    own_score: int = 0
    opp_score: int = 0
    dribble_steps: int = 0
    valid_plan: bool = False
    trajectory_id: str = ""
    quality_score: float = 0.0
    events: List[Dict] = field(default_factory=list)
    primitives_used: List[str] = field(default_factory=list)
    handoff_ready_steps: int = 0
    kick_phase_activated: bool = False
    episode_handoff_achieved: bool = False
    episode_takeover_count: int = 0
    episode_takeover_success: int = 0
    all_real_backend: bool = True
    wrong_bridge_mode_seen: bool = False
    contract_validation_seen: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# EVALUADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class SolverEvaluator:
    def __init__(self, bridge: EnvBridge, solver: AStarSolver):
        self.bridge = bridge
        self.solver = solver

    def plan_from_current_reset(self, stage_id: int,
                                seed: int | None = None
                                ) -> Tuple[WorldSnapshot, List[PlannedTrajectory]]:
        world = self.bridge.capture_world_snapshot(stage_id=stage_id,
                                                    seed=seed)
        objective = get_stage_objective(stage_id)
        plans = self.solver.solve_topk(world.to_simplified_state(), objective)
        return world, plans

    def replay_closed_loop(self,
                           stage_id: int,
                           initial_snapshot: WorldSnapshot,
                           max_steps: int = 0,
                           lookahead: int = 3,
                           collect_snapshots: bool = True,
                           reference_plan: PlannedTrajectory | None = None,
                           forced_primitive: str | None = None,
                           intent_label: str | None = None,
                           approach_target: np.ndarray | None = None,
                           dribble_bias_angle: float = 0.0,
                           kick_target: np.ndarray | None = None,
                           kick_angle_deg: float = 20.0,
                           kick_phase_dist: float = 1800.0,
                           ) -> tuple[RolloutMetrics, List[DemoSnapshot]]:
        """Ejecuta un episodio con approach directo a launch position.

        Modos:
          kick_target=None  → dribble puro (frontal_kick_angle_deg=0, lateral offset)
          kick_target=pos   → dribble hacia pos, luego disparo/pase cuando
                              dist(ball, target) < kick_phase_dist.
                              Fase dribble: frontal_kick_angle_deg=0 (sin kicks accidentales).
                              Fase kick:    frontal_kick_angle_deg=kick_angle_deg,
                                            waypoint radial detrás del balón → EMA converge
                                            → angle_diff→0 → kick dispara hacia kick_target.
        """
        if max_steps <= 0:
            max_steps = (
                DATASET_CFG.max_replay_steps
                if DATASET_CFG.max_replay_steps > 0
                else CURRICULUM[stage_id].episode_steps
            )

        objective = get_stage_objective(stage_id)
        local_horizon = CURRICULUM[stage_id].local_horizon

        obs = self.bridge._last_obs
        infos = self.bridge._last_infos
        controlled_agent = self.bridge.controlled_agent_id

        # Trajectory ID
        angle_label = f"{math.degrees(dribble_bias_angle):.0f}deg"
        approach_label = forced_primitive or "free"
        traj_id = f"mpc_{approach_label}_{angle_label}"
        ref_quality = reference_plan.quality if reference_plan else 0.0

        metrics = RolloutMetrics(
            stage_id=stage_id,
            trajectory_id=traj_id,
            intent=intent_label or "control",
        )
        snapshots: List[DemoSnapshot] = []
        total_rewards = []
        touched_prev = False
        recontacts = 0
        # Signo lateral fijo para todo el episodio (evita flips de signo que
        # crean un approach radial momentáneo → angle_diff→0 → kick inesperado).
        # Derivado del dribble_bias_angle: diversidad entre episodios,
        # consistencia dentro del episodio.
        episode_lateral_sign = 1.0 if dribble_bias_angle >= 0.0 else -1.0
        # FIX-B: Histéresis en orbitación — evitar oscilación frame-a-frame
        # entre orbit y approach cuando ahead_component baila alrededor de 100.
        orbit_latch = 0  # ticks restantes de orbit forzado

        # Override dinámico de frontal_kick_angle_deg:
        # Siempre empezamos a 0 (fase dribble, sin kicks).
        # Si hay kick_target, se sube a kick_angle_deg al entrar en kick_phase.
        _stage_cfg = CURRICULUM[stage_id]
        _orig_kick_angle = getattr(_stage_cfg, "frontal_kick_angle_deg", 0.0)
        _stage_cfg.frontal_kick_angle_deg = 0.0
        kick_phase_active = False
        kick_mode_label = intent_label or "control"
        if intent_label is None:
            if kick_target is not None:
                is_goal_shot = (kick_target[0] > FIELD.x_opponent_ground - 600.0)
                kick_mode_label = "shoot_setup" if is_goal_shot else "pass_setup"
            else:
                kick_mode_label = "dribble"
        traj_id = traj_id.replace("mpc_", f"mpc_{kick_mode_label}_", 1)
        metrics.intent = kick_mode_label

        for step in range(max_steps):
            if controlled_agent not in self.bridge.env.agents:
                break

            # ── 1. Estado real ────────────────────────────────────────────
            agent_pos = self.bridge.agent_position(controlled_agent)
            ball_pos = self.bridge.ball_position()
            
            # SOLUCIÓN 3: Abortar si el balón salió y PyGame lo devolvió
            if (ball_pos[0] > FIELD.x_opponent_ground + 100 or ball_pos[0] < FIELD.x_own_ground - 100 or
                ball_pos[1] > FIELD.y_left_sideline + 100 or ball_pos[1] < FIELD.y_right_sideline - 100):
                metrics.valid_plan = False
                break

            # ── 2 y 3. Híbrido: Geométrico para approach, A* para el resto ──
            state = self.bridge.get_live_world_state()
            state.recontacts = recontacts
            state.touched_ball = metrics.touched_ball or state.touched_ball

            base_target = (
                kick_target if kick_target is not None
                else approach_target if approach_target is not None
                else nearest_goal_post(ball_pos)
            )
            target_dir = base_target - ball_pos

            # Calcular el launch_pos manual para evitar que A* quede en mínimos locales desde lejos
            launch_pos, launch_phase = compute_launch_position(
                agent_pos, ball_pos, dribble_bias_angle,
                offset_mm=350.0, close_offset_mm=120.0, target_dir=target_dir
            )
            dist_to_launch = float(np.linalg.norm(launch_pos - agent_pos))
            dist_ball = float(np.linalg.norm(ball_pos - agent_pos))

            if dist_to_launch > 150.0 and not state.touched_ball and dist_ball > 350.0:
                # Approach phase: usar el waypoint geométrico (suave y determinista)
                waypoint = launch_pos
                phase = launch_phase
            else:
                # PHASE DE CONTACTO/MANEJO: Dirección segura para guiar el A* (Solución 2)
                # Nunca establecer objective.kick_target usando rotación ignorante de bandas.
                # Obligar a que la dirección pase por compute_safe_push_direction.
                safe_dir = compute_safe_push_direction(
                    ball_pos, dribble_bias_angle, target_dir=target_dir)
                target_dist = max(900.0, float(np.linalg.norm(base_target - ball_pos)))
                objective.kick_target = ball_pos + safe_dir * target_dist

                # Solución: Suprimir kicks de acuerdo al currículum.
                # Evita que patee siempre que tiene balón bloqueándolo hasta cumplir su tarea de dribble.
                forbidden = []

                # Intentos de control/dribble nunca deben venderse como kicks.
                if kick_target is None:
                    forbidden.append("align_then_kick")

                # Para stages que requieren control primero
                if state.recontacts < objective.target_recontacts:
                    forbidden.append("align_then_kick")

                # Opcional (Kick_phase_dist fallback)
                dist_target = float(np.linalg.norm(
                    (kick_target if kick_target is not None else nearest_goal_post(ball_pos))
                    - ball_pos))

                # Sólo bloquear pateo lejano si es estrictamente al arco (evitar bloqueos a pases)
                if kick_target is None or kick_target[0] > FIELD.x_opponent_ground - 600.0:
                    # En lugar de 1800 fijo, ser más perdonador a tiros de media distancia si ya hizo los dribbles
                    if dist_target > (kick_phase_dist + 500.0):
                        if "align_then_kick" not in forbidden:
                            forbidden.append("align_then_kick")

                # Resolver con el A* Planner (Evaluando OOB, Own Goals, Dribbles, Kicks)
                result = self.solver.solve_greedy_step(
                    state, objective, forced_primitive=None,
                    forbidden_primitives=forbidden if forbidden else None,
                )
                waypoint = result.waypoint
                phase = result.primitive_name
            
            if "kick" in phase:
                kick_phase_active = True
                metrics.kick_phase_activated = True
                _stage_cfg.frontal_kick_angle_deg = max(kick_angle_deg, 25.0)

            prim_name = f"{phase}_{angle_label}"
            metrics.primitives_used.append(prim_name)

            # ── 4. Convertir a teacher_action ────────────────────────────
            teacher_action = waypoint_to_normalized_action(
                agent_pos, waypoint, local_horizon)

            # ── 5. Ejecutar en entorno real ──────────────────────────────
            actions = self.bridge.make_action_dict(teacher_action)
            (next_obs, rewards, terminateds,
             truncateds, next_infos) = self.bridge.step(actions)

            # ── 6. Métricas ──────────────────────────────────────────────
            reward = float(rewards.get(controlled_agent, 0.0))
            total_rewards.append(reward)
            episode_done = bool(terminateds.get("__all__", False)
                                or truncateds.get("__all__", False))

            touched = (dist_ball <= getattr(AGENT, "ball_dribble_range", 340.0))
            if touched and touched_prev:
                recontacts += 1
            touched_prev = touched

            agent_info = next_infos.get(controlled_agent, None) or {}
            backend_label = str(agent_info.get("sidecar_backend", "unknown"))
            bridge_mode = str(agent_info.get("sidecar_bridge_mode", ""))
            contract_validation_mode = bool(
                agent_info.get("sidecar_contract_validation_mode", False))

            if backend_label != "real":
                metrics.all_real_backend = False
            if bridge_mode and bridge_mode != DATASET_CFG.required_bridge_mode:
                metrics.wrong_bridge_mode_seen = True
            if contract_validation_mode:
                metrics.contract_validation_seen = True

            if bool(agent_info.get("handoff_ready", False)):
                metrics.handoff_ready_steps += 1
            metrics.episode_handoff_achieved = (
                metrics.episode_handoff_achieved
                or bool(agent_info.get("episode_handoff_achieved", False))
            )
            metrics.episode_takeover_count = max(
                metrics.episode_takeover_count,
                int(agent_info.get("episode_takeover_count", 0)),
            )
            metrics.episode_takeover_success = max(
                metrics.episode_takeover_success,
                int(agent_info.get("episode_takeover_success", 0)),
            )
            event = {
                "ball_touched": touched,
                "dribbling": touched,
                "goal_scored": bool(agent_info.get("goal_scored", False)),
                "own_score": int(agent_info.get("own_score", 0)),
                "opp_score": int(agent_info.get("opp_score", 0)),
                "primitive": prim_name,
                "phase": phase,
                # FIX-3: Metadata de intención en el event (para análisis offline)
                "kick_phase_active": kick_phase_active,
                "handoff_ready": bool(agent_info.get("handoff_ready", False)),
                "takeover_active": bool(agent_info.get("takeover_active", False)),
                "sidecar_backend": backend_label,
                "sidecar_bridge_mode": bridge_mode,
            }
            metrics.events.append(event)
            if event["dribbling"]:
                metrics.dribble_steps += 1

            # ── 7. Snapshot BC-ready ─────────────────────────────────────
            if collect_snapshots and controlled_agent in next_obs:
                state_global = agent_info.get("state", None)
                if state_global is None:
                    state_global = np.zeros(31, dtype=np.float32)
                snapshots.append(DemoSnapshot.from_transition(
                    stage_id=stage_id,
                    episode_id=self.bridge.episode_index,
                    t=step,
                    obs_local=np.asarray(obs[controlled_agent],
                                         dtype=np.float32),
                    state_global=np.asarray(state_global,
                                            dtype=np.float32),
                    teacher_action=np.asarray(teacher_action,
                                              dtype=np.float32),
                    next_obs_local=np.asarray(
                        next_obs[controlled_agent], dtype=np.float32),
                    done=episode_done,
                    event=event,
                    trajectory_id=traj_id,
                    quality_score=ref_quality,
                    reset_seed=initial_snapshot.reset_seed,
                    world_snapshot={
                        "stage_id": int(initial_snapshot.stage_id),
                        "agent_id": initial_snapshot.agent_id,
                        "agent_pos": np.asarray(initial_snapshot.agent_pos,
                                                 dtype=np.float32),
                        "ball_pos": np.asarray(initial_snapshot.ball_pos,
                                                dtype=np.float32),
                        "agent_theta": float(initial_snapshot.agent_theta),
                    },
                    initial_agent_pos=initial_snapshot.agent_pos,
                    initial_ball_pos=initial_snapshot.ball_pos,
                    # FIX-3: Campos de intención BC-ready
                    intent=kick_mode_label,
                    kick_phase_active=kick_phase_active,
                    kick_target=(np.asarray(kick_target, dtype=np.float32)
                                 if kick_target is not None else None),
                    approach_target=(np.asarray(approach_target, dtype=np.float32)
                                     if approach_target is not None else None),
                    handoff_ready=bool(agent_info.get("handoff_ready", False)),
                    takeover_active=bool(agent_info.get("takeover_active", False)),
                    episode_handoff_achieved=bool(
                        agent_info.get("episode_handoff_achieved", False)),
                    episode_takeover_count=int(
                        agent_info.get("episode_takeover_count", 0)),
                    episode_takeover_success=int(
                        agent_info.get("episode_takeover_success", 0)),
                    sidecar_backend=backend_label,
                    sidecar_bridge_mode=bridge_mode,
                    sidecar_contract_validation_mode=contract_validation_mode,
                ))

            metrics.steps += 1
            metrics.touched_ball = metrics.touched_ball or touched
            metrics.goal_scored = metrics.goal_scored or event["goal_scored"]
            obs = next_obs
            if episode_done:
                break

        # Restaurar frontal_kick_angle_deg original
        if hasattr(_stage_cfg, "frontal_kick_angle_deg"):
            _stage_cfg.frontal_kick_angle_deg = _orig_kick_angle

        metrics.recontacts = recontacts
        metrics.mean_reward = (float(np.mean(total_rewards))
                               if total_rewards else 0.0)
        metrics.own_score = int(self.bridge.raw_env._game.own_score)
        metrics.opp_score = int(self.bridge.raw_env._game.opp_score)
        metrics.valid_plan = self._check_validity(metrics, objective,
                                                  kick_target=kick_target)
        metrics.quality_score = ref_quality
        return metrics, snapshots

    def _check_validity(self, metrics: RolloutMetrics,
                        objective: StageObjective,
                        kick_target: np.ndarray | None = None) -> bool:
        if DATASET_CFG.require_real_backend:
            if not metrics.all_real_backend:
                return False
            if metrics.contract_validation_seen:
                return False
            if metrics.wrong_bridge_mode_seen:
                return False

        # Autogol → rechazar siempre independientemente del stage
        if metrics.own_score > 0:
            return False

        if not metrics.episode_handoff_achieved:
            return False

        if objective.stage_id == 0 and metrics.intent != "control":
            return False
        if objective.stage_id == 1 and metrics.intent != "dribble":
            return False

        if metrics.intent == "control":
            return (
                metrics.touched_ball
                and metrics.handoff_ready_steps >= 1
                and metrics.episode_takeover_success >= 1
                and metrics.dribble_steps >= 2
            )

        if metrics.intent == "dribble":
            return (
                metrics.touched_ball
                and metrics.handoff_ready_steps >= 1
                and metrics.episode_takeover_success >= 1
                and metrics.dribble_steps >= max(objective.target_recontacts, 3)
            )

        if metrics.intent == "pass_setup":
            return (
                metrics.touched_ball
                and metrics.kick_phase_activated
                and metrics.dribble_steps >= objective.target_recontacts
            )

        if metrics.intent == "shoot_setup":
            min_dribble = max(objective.target_recontacts, 3)
            return (
                metrics.touched_ball
                and metrics.kick_phase_activated
                and metrics.dribble_steps >= min_dribble
            )

        if objective.success_mode == "touch_and_control":
            return metrics.touched_ball and metrics.dribble_steps >= 3
        elif objective.success_mode == "stable_dribble":
            return metrics.touched_ball and metrics.dribble_steps >= 5
        elif objective.success_mode == "dribble_align_kick":
            return (metrics.goal_scored
                    or (metrics.touched_ball and metrics.dribble_steps >= 2))
        elif objective.success_mode == "goal_completion":
            if metrics.goal_scored:
                return True
            if kick_target is not None:
                is_goal_shot = float(kick_target[0]) > FIELD.x_opponent_ground - 600.0
                min_dribble = 3 if is_goal_shot else 2
            else:
                min_dribble = 1
            return metrics.touched_ball and metrics.dribble_steps >= min_dribble
        return False


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def plot_plan(agent_start: np.ndarray,
              ball_start: np.ndarray,
              waypoints: Iterable[np.ndarray],
              out_path: str | Path,
              title: str = "Planned trajectory") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wps = np.asarray(list(waypoints), dtype=np.float32)
    plt.figure(figsize=(10, 6))
    plt.axvline(FIELD.x_opponent_ground, linestyle="--", linewidth=1.0)
    plt.axhline(FIELD.y_left_sideline, linestyle=":", linewidth=1.0)
    plt.axhline(FIELD.y_right_sideline, linestyle=":", linewidth=1.0)
    plt.scatter([agent_start[0]], [agent_start[1]], marker="o",
                label="agent start")
    plt.scatter([ball_start[0]], [ball_start[1]], marker="x",
                label="ball start")
    if len(wps) > 0:
        plt.plot(wps[:, 0], wps[:, 1], marker=".", label="waypoints")
    plt.xlim(FIELD.x_own_ground - 200, FIELD.x_opponent_ground + 200)
    plt.ylim(FIELD.y_right_sideline - 200, FIELD.y_left_sideline + 200)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
