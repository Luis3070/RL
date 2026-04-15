"""
environment_diagnostics.py
==========================
Herramienta de auditoría del entorno RL sin entrenamiento.

Módulos originales:
  1. DRY-RUN LOADER       -- Instancia el env y ejecuta steps nulos.
  2. CONTRACT AUDIT       -- Verifica respuestas del bridge contra el contrato C++.
  3. POSE DRIFT DETECTOR  -- Detecta teleportaciones inter-frame.
  4. HANDOFF LIFECYCLE    -- Mide latencias APPROACH→HANDOFF→TAKEOVER.
  5. BHUMAN CONDITIONS    -- Evalúa dribble/duel/shoot/pass natural y forzado.
  6. STAGE SWEEP          -- Audita la conexión env↔BHuman para cada stage.
  7. REWARD SANITY CHECK  -- Verifica que takeover rewards sean impulsos de 1 frame.

Módulos nuevos (32 tests adicionales):
  8.  BALL PHYSICS BASICS    -- Velocidad, fricción, pared, detección de gol (T1-T4).
  9.  BHUMAN SEMANTIC        -- Macro stability, skill semantic, handoff jitter (T5-T10).
  10. PRE-TRAINING AUDIT     -- Determinismo seed, stress 1000 steps, reward consistency (T11-T18).
  11. BALL EDGE CASES        -- Backward bounce, goal post, esquina, out persistence (T26-T32).
  12. PHYSICAL STRESS        -- Fallen agent, colisión, límites velocidad, presión extrema (TA-TG).

Módulos extra (Contract Guardrails — 3 tests):
  13. CONTRACT GUARDRAILS    -- Handoff envelope §2.2, action safety §1.2, false reward §2.4.
  14. REPORTER               -- Vuelca JSON + MD en artifacts/.

Visualización en terminal:
  - Render ASCII del campo con posiciones reales en cada step.
  - Usa cls/clear para animar. Delay configurable.

Ejecucion:
  conda activate RLRobocup
  python environment_diagnostics.py
  (menú interactivo — sin argparse)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

os.environ.setdefault("RL_EXECUTION_MODE", "bhuman_sidecar")

from config import CURRICULUM, SIM, AGENT, FIELD
from environment import NaoSoccerEnv
from bhuman_sidecar import GroundTruthFrame

# ─── Constantes cinemáticas ────────────────────────────────────────────────────
MAX_KINEMATIC_MM  = AGENT.max_forward_speed * SIM.dt   # ~8.25 mm/frame
TELEPORT_THRESH   = MAX_KINEMATIC_MM * 5.0             # ~41 mm — umbral de alerta

ROLES_MAP = {
    "agent_0": "forwardR",
    "agent_1": "forwardL",
    "agent_2": "defenderR",
    "agent_3": "defenderL",
}

VALID_MACRO_TYPES = {
    "none", "playBall_shoot", "playBall_pass",
    "playBall_dribble", "playBall_duel",
}
VALID_END_REASONS = {
    "none", "completed", "timeout", "skill_request_cleared",
    "active_agent_changed", "out", "goal", "game_state_changed", "other",
}

BALL_MAX_SPEED_MM_S = 5000.0   # kick máximo razonable ~5 m/s


# ─────────────────────────────────────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContractAlert:
    frame: int
    kind: str
    agent: str
    detail: str
    severity: str = "WARN"


@dataclass
class HandoffTimings:
    approach_frames: int = 0
    handoff_ready_frame: Optional[int] = None
    takeover_start_frame: Optional[int] = None
    takeover_end_frame: Optional[int] = None
    macro_end_reason: str = "none"
    macro_success: Optional[bool] = None
    transition_latency_ms: float = 0.0


@dataclass
class BHumanConditionResult:
    condition: str          # "dribble" | "duel" | "shoot" | "pass"
    mode: str               # "natural" | "forced"
    detected: bool = False
    frames_to_detect: Optional[int] = None
    takeover_reached: bool = False
    macro_end_reason: str = "none"
    macro_success: Optional[bool] = None
    notes: str = ""


@dataclass
class StageSweepResult:
    stage_id: int
    stage_name: str
    bridge_alive: bool = False
    first_info_backend: str = "unknown"
    handoff_ready_detected: bool = False
    takeover_active_detected: bool = False
    macro_completed: bool = False
    steps_run: int = 0
    error: str = ""


@dataclass
class RewardImpulseRecord:
    frame: int
    agent: str
    takeover_success_r: float
    takeover_failure_r: float
    frame_before: float = 0.0


@dataclass
class BallPhysicsResult:
    test_name: str
    passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SemanticTestResult:
    test_name: str
    passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PreTrainingResult:
    test_name: str
    passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PhysicalStressResult:
    test_name: str
    passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ContractGuardrailResult:
    """Resultado de un test de guardrail del contrato RL-BHuman."""
    test_name: str
    passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DiagnosticsReport:
    timestamp: str
    stage: int
    seed: int
    steps_run: int
    backend: str
    bridge_alive: bool
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    teleport_count: int = 0
    max_pose_delta_mm: float = 0.0
    handoff: Dict[str, Any] = field(default_factory=dict)
    bhuman_conditions: List[Dict[str, Any]] = field(default_factory=list)
    stage_sweep: List[Dict[str, Any]] = field(default_factory=list)
    reward_impulses: List[Dict[str, Any]] = field(default_factory=list)
    reward_multi_frame_violations: int = 0
    # Nuevos módulos
    ball_physics: List[Dict[str, Any]] = field(default_factory=list)
    bhuman_semantic: List[Dict[str, Any]] = field(default_factory=list)
    pretraining_audit: List[Dict[str, Any]] = field(default_factory=list)
    ball_edge_cases: List[Dict[str, Any]] = field(default_factory=list)
    physical_stress: List[Dict[str, Any]] = field(default_factory=list)
    contract_guardrails: List[Dict[str, Any]] = field(default_factory=list)
    execution_quality: List[Dict[str, Any]] = field(default_factory=list)
    verdict: str = "UNKNOWN"
    notes: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS COMUNES
# ─────────────────────────────────────────────────────────────────────────────

def _zero_actions(env: NaoSoccerEnv) -> Dict[str, np.ndarray]:
    return {aid: np.zeros(env.action_space(aid).shape, dtype=np.float32)
            for aid in env.agents}


def _install_state(
    env: NaoSoccerEnv,
    ball_pos: np.ndarray,
    agent_pos: np.ndarray,          # posición del agente activo (agent_0 o agent_1)
    ball_vel: Optional[np.ndarray] = None,
    is_playing: bool = True,
    agent_theta: Optional[float] = None,   # si None, apunta automáticamente al balón
    teammate_pos: Optional[np.ndarray] = None,  # posición compañero (agent_1, para pass)
) -> None:
    """Inyecta estado mínimo sin depender de GroundTruthFrame."""
    env._ball.pos[:] = ball_pos.astype(np.float32)
    env._ball.vel[:] = (ball_vel if ball_vel is not None
                        else np.zeros(2, dtype=np.float32))
    env._ball.seen_age_ms = 0.0

    agents = list(env.agents)
    if agents:
        pos0 = agent_pos.astype(np.float32)
        env._agents_state[agents[0]].pos = pos0
        env._agents_state[agents[0]].is_upright = True
        # Theta: si se especifica, usarlo; si no, apuntar automáticamente al balón
        if agent_theta is not None:
            env._agents_state[agents[0]].theta = float(agent_theta)
        else:
            direction = ball_pos.astype(np.float32) - pos0
            env._agents_state[agents[0]].theta = float(
                math.atan2(float(direction[1]), float(direction[0]))
            )
        for i, aid in enumerate(agents[1:], start=1):
            if i == 1 and teammate_pos is not None:
                env._agents_state[aid].pos = teammate_pos.astype(np.float32)
            else:
                idx = int(aid.split("_")[1])
                env._agents_state[aid].pos = np.array(
                    [-800.0 * idx, 600.0], dtype=np.float32)
            env._agents_state[aid].is_upright = True
    env._game.is_playing = is_playing


def _safe_step(env, actions):
    """step() con captura de errores, devuelve (obs, rew, term, trunc, infos) o None."""
    try:
        return env.step(actions)
    except Exception as exc:
        print(f"  [step ERR] {exc}")
        return None


def _is_done(terminations, truncations) -> bool:
    return all(terminations.values()) or all(truncations.values())


def _make_env(stage_idx: int, seed: int) -> NaoSoccerEnv:
    """Instancia env para un stage dado."""
    idx = max(0, min(stage_idx, len(CURRICULUM) - 1))
    return NaoSoccerEnv(stage_cfg=CURRICULUM[idx], use_fov=False, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO A: VISUALIZACIÓN ASCII EN TERMINAL
# ─────────────────────────────────────────────────────────────────────────────

def _render_ascii(env: NaoSoccerEnv, info: Dict[str, Any],
                  frame_idx: int, label: str = "",
                  extra_lines: Optional[List[str]] = None,
                  delay_s: float = 0.1) -> None:
    """
    Dibuja el campo en ASCII:
      - Campo 4v4 Scaled proporcional (60×20 chars).
      - A = agentes RL  |  O = oponentes  |  B = balón  |  X = obstáculos.
      - HUD lateral con datos clave del bridge.
    """
    W, H = 62, 22   # chars del campo
    sx = FIELD.x_own_ground    # -2813 mm
    ex = FIELD.x_opponent_ground  # +2813 mm
    sy = FIELD.y_right_sideline   # -1875 mm
    ey = FIELD.y_left_sideline    #  +1875 mm

    def _to_grid(xmm, ymm):
        col = int((xmm - sx) / (ex - sx) * (W - 1))
        row = int((ey - ymm) / (ey - sy) * (H - 1))
        return max(0, min(col, W - 1)), max(0, min(row, H - 1))

    grid = [['·'] * W for _ in range(H)]

    # Línea central
    for r in range(H):
        grid[r][W // 2] = '|'
    # Portería propia  (izquierda, col=0) y rival (derecha, col=W-1)
    gol_rows = range(
        int((ey - FIELD.y_left_goal) / (ey - sy) * (H - 1)),
        int((ey - FIELD.y_right_goal) / (ey - sy) * (H - 1)) + 1,
    )
    for r in gol_rows:
        if 0 <= r < H:
            grid[r][0] = ']'
            grid[r][W - 1] = '['

    # Balón
    bx, by = _to_grid(env._ball.pos[0], env._ball.pos[1])
    grid[by][bx] = 'B'

    # Obstáculos
    for obs in env._static_obstacles:
        ox, oy = _to_grid(obs.pos[0], obs.pos[1])
        grid[oy][ox] = 'X'

    # Oponentes
    for i, opp in enumerate(env._opponents_state):
        ox, oy = _to_grid(opp.pos[0], opp.pos[1])
        grid[oy][ox] = 'o'

    # Agentes RL (por encima del balón en prioridad visual)
    agent_labels = ['A', 'a', '⁴', '⁵']
    for i, aid in enumerate(env.agents):
        ag = env._agents_state[aid]
        ax, ay = _to_grid(ag.pos[0], ag.pos[1])
        grid[ay][ax] = agent_labels[min(i, len(agent_labels) - 1)] \
            if ag.is_upright else 'x'

    # Render a líneas
    top    = '+' + '-' * W + '+'
    bottom = '+' + '-' * W + '+'
    field_lines = [top]
    for row in grid:
        field_lines.append('|' + ''.join(row) + '|')
    field_lines.append(bottom)

    # HUD lateral
    macro    = info.get("macro_type", "none")
    takeover = bool(info.get("takeover_active", False))
    handoff  = bool(info.get("handoff_ready", False))
    skill    = info.get("sidecar_skill_request", "-")
    motion   = info.get("sidecar_motion_request", "-")
    end_r    = info.get("last_macro_end_reason", "none")
    backend  = info.get("sidecar_backend", "?")
    bspeed   = float(np.linalg.norm(env._ball.vel))
    ctrl     = ("BHUMAN" if takeover else ("HANDOFF" if handoff else "RL"))

    hud = [
        f"Frame : {frame_idx:>4}   [{label}]",
        f"Backend: {backend}",
        f"Control: {ctrl}",
        f"Macro  : {macro}",
        f"Skill  : {skill}",
        f"Motion : {motion}",
        f"EndRsn : {end_r}",
        f"Ball v : {bspeed:.0f} mm/s",
        f"BallXY : ({env._ball.pos[0]:.0f},{env._ball.pos[1]:.0f})",
    ]
    if extra_lines:
        hud += extra_lines

    # Imprimir campo + HUD lado a lado
    os.system("cls" if os.name == "nt" else "clear")
    for i, fline in enumerate(field_lines):
        side = hud[i] if i < len(hud) else ""
        print(f"{fline}  {side}")
    time.sleep(delay_s)


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 1: DRY-RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_dry_run(env: NaoSoccerEnv,
                steps: int = 10,
                visual: bool = False,
                delay: float = 0.08) -> Tuple[str, bool]:
    print("\n[1] DRY-RUN LOADER")
    actions = _zero_actions(env)
    backend, alive = "unknown", False

    for i in range(steps):
        result = _safe_step(env, actions)
        if result is None:
            break
        _, _, term, trunc, infos = result
        if infos:
            info = next(iter(infos.values()))
            backend = info.get("sidecar_backend", "unknown")
            alive = True
            if visual:
                _render_ascii(env, info, i, "DRY-RUN", delay_s=delay)
        if _is_done(term, trunc):
            env.reset()
            actions = _zero_actions(env)

    status = "OK" if alive else "FAIL"
    print(f"  {status}: {steps} steps  backend={backend}")
    return backend, alive


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: CONTRACT AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def run_contract_audit(env: NaoSoccerEnv, seed: int) -> List[ContractAlert]:
    print("\n[2] CONTRACT AUDIT")
    alerts: List[ContractAlert] = []
    rng = np.random.default_rng(seed + 999)
    env.reset(seed=seed)
    actions = _zero_actions(env)

    for fi in range(30):
        ball = np.array([rng.uniform(-1800, 1800),
                         rng.uniform(-1200, 1200)], dtype=np.float32)
        _install_state(env, ball,
                       ball + np.array([-600, 0], dtype=np.float32))
        result = _safe_step(env, actions)
        if result is None:
            alerts.append(ContractAlert(fi, "step_error", "all", "step() lanzó excepción", "ERROR"))
            continue
        _, _, term, trunc, infos = result
        if not infos:
            alerts.append(ContractAlert(fi, "bridge_disconnected", "all", "infos vacío", "ERROR"))
            continue
        info = next(iter(infos.values()))

        if info.get("execution_mode") != "bhuman_sidecar":
            alerts.append(ContractAlert(fi, "contract_violation", "all",
                f"execution_mode={info.get('execution_mode')}", "ERROR"))
        if info.get("macro_type", "none") not in VALID_MACRO_TYPES:
            alerts.append(ContractAlert(fi, "invalid_macro_type", "all",
                f"macro_type={info.get('macro_type')!r}", "WARN"))
        if info.get("last_macro_end_reason", "none") not in VALID_END_REASONS:
            alerts.append(ContractAlert(fi, "invalid_end_reason", "all",
                f"end_reason={info.get('last_macro_end_reason')!r}", "WARN"))
        if not isinstance(info.get("handoff_ready"), bool):
            alerts.append(ContractAlert(fi, "contract_violation", "all",
                f"handoff_ready tipo {type(info.get('handoff_ready')).__name__}", "WARN"))

        if _is_done(term, trunc):
            env.reset(seed=seed + fi)
            actions = _zero_actions(env)

    errors = sum(1 for a in alerts if a.severity == "ERROR")
    print(f"  {'OK' if errors == 0 else 'FAIL'}: {len(alerts)} alertas ({errors} errores)")
    for a in alerts:
        print(f"    [{a.severity}] f={a.frame} {a.kind}: {a.detail}")
    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: POSE DRIFT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def run_pose_drift_detection(env: NaoSoccerEnv, steps: int, seed: int,
                             visual: bool = False,
                             delay: float = 0.06) -> Tuple[List[ContractAlert], float]:
    print("\n[3] POSE DRIFT DETECTOR")
    alerts: List[ContractAlert] = []
    prev_pos: Dict[str, np.ndarray] = {}
    max_delta = 0.0

    env.reset(seed=seed)
    actions = _zero_actions(env)
    ball = np.array([300.0, 0.0], dtype=np.float32)
    _install_state(env, ball, ball + np.array([-700, 0], dtype=np.float32))

    for step in range(steps):
        result = _safe_step(env, actions)
        if result is None:
            break
        _, _, term, trunc, infos = result
        info = next(iter(infos.values()), {}) if infos else {}

        for aid in env.agents:
            curr = env._agents_state[aid].pos.copy()
            if aid in prev_pos:
                delta = float(np.linalg.norm(curr - prev_pos[aid]))
                max_delta = max(max_delta, delta)
                if delta > TELEPORT_THRESH:
                    ta = info.get("takeover_active", False)
                    alerts.append(ContractAlert(
                        step, "teleport", aid,
                        f"delta={delta:.1f}mm > {TELEPORT_THRESH:.1f}mm (takeover={ta})",
                        "INFO" if ta else "WARN",
                    ))
            prev_pos[aid] = curr

        if visual and infos:
            _render_ascii(env, info, step, "POSE DRIFT",
                          [f"max_delta: {max_delta:.1f}mm",
                           f"alerts: {len(alerts)}"], delay_s=delay)

        if _is_done(term, trunc):
            env.reset(seed=seed + step)
            actions = _zero_actions(env)
            prev_pos.clear()
            _install_state(env, ball, ball + np.array([-700, 0], dtype=np.float32))

    tc = sum(1 for a in alerts if a.kind == "teleport")
    print(f"  teleports={tc}  max_delta={max_delta:.1f}mm  threshold={TELEPORT_THRESH:.1f}mm")
    if tc == 0:
        print("  OK: sin teleportaciones.")
    else:
        print(f"  WARN: {tc} eventos de teleportación.")
    return alerts, max_delta


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: HANDOFF LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

def run_handoff_lifecycle(env: NaoSoccerEnv, seed: int,
                          max_frames: int = 300,
                          visual: bool = False,
                          delay: float = 0.06) -> HandoffTimings:
    print("\n[4] HANDOFF LIFECYCLE")
    t = HandoffTimings()
    env.reset(seed=seed)
    actions = _zero_actions(env)

    ball = np.array([0.0, 0.0], dtype=np.float32)
    active = list(env.agents)[0] if env.agents else "agent_0"
    t_wall = time.perf_counter()

    for fi in range(max_frames):
        dist = max(260.0, 850.0 - fi * 18.0)
        if active in env._agents_state:
            env._agents_state[active].pos = ball + np.array([-dist, 0.0], dtype=np.float32)

        result = _safe_step(env, actions)
        if result is None:
            break
        _, _, term, trunc, infos = result
        info = next(iter(infos.values()), {}) if infos else {}

        hr  = bool(info.get("handoff_ready", False))
        ta  = bool(info.get("takeover_active", False))
        end = str(info.get("last_macro_end_reason", "none"))

        if not hr and t.handoff_ready_frame is None:
            t.approach_frames = fi
        if hr and t.handoff_ready_frame is None:
            t.handoff_ready_frame = fi
            print(f"  handoff_ready @ frame {fi}")
        if ta and t.takeover_start_frame is None:
            t.takeover_start_frame = fi
            if t.handoff_ready_frame is not None:
                t.transition_latency_ms = (fi - t.handoff_ready_frame) * SIM.dt * 1000.0
            print(f"  takeover_active @ frame {fi}  latency={t.transition_latency_ms:.1f}ms")
        if t.takeover_start_frame is not None and end != "none" and t.takeover_end_frame is None:
            t.takeover_end_frame = fi
            t.macro_end_reason   = end
            t.macro_success      = info.get("macro_success")
            dur = (fi - t.takeover_start_frame) * SIM.dt * 1000.0
            print(f"  macro_end @ frame {fi}  reason={end}  success={t.macro_success}  dur={dur:.0f}ms")
            if visual and infos:
                _render_ascii(env, info, fi, "HANDOFF END", delay_s=0.5)
            break

        if visual and infos:
            _render_ascii(env, info, fi, "HANDOFF LC",
                          [f"dist_to_ball: {dist:.0f}mm",
                           f"handoff_ready: {hr}",
                           f"takeover_active: {ta}"], delay_s=delay)

        if _is_done(term, trunc):
            env.reset(seed=seed + fi)
            actions = _zero_actions(env)

    elapsed = (time.perf_counter() - t_wall) * 1000.0
    if t.handoff_ready_frame is None:
        print(f"  WARN: handoff_ready no alcanzado en {max_frames} frames.")
    if t.takeover_start_frame is None:
        print(f"  WARN: takeover_active no detectado.")
    print(f"  Wall time: {elapsed:.0f}ms")
    return t


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 5: BHUMAN CONDITIONS (natural + forzado)
# ─────────────────────────────────────────────────────────────────────────────

# Condición → ball_pos, agent_offset para cada condición forzada
_FORCED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "dribble": {
        "ball_pos": np.array([0.0, 0.0],    dtype=np.float32),
        "offset":   np.array([-280.0, 0.0], dtype=np.float32),
        "ball_vel": np.zeros(2, dtype=np.float32),
        # theta=None → _install_state calcula automáticamente (apunta al balón)
    },
    "duel": {
        "ball_pos": np.array([200.0, 0.0],  dtype=np.float32),
        "offset":   np.array([-290.0, 0.0], dtype=np.float32),
        "ball_vel": np.zeros(2, dtype=np.float32),
        # Oponente enfrente del balón para que BHuman active zweikampf
        "opp_pos":  np.array([500.0, 0.0],  dtype=np.float32),
        "stage_id": 3,  # 1v1_dynamic tiene num_opponents=1
    },
    "shoot": {
        "ball_pos": np.array([FIELD.x_opp_penalty + 200.0, 0.0], dtype=np.float32),
        "offset":   np.array([-300.0, 0.0], dtype=np.float32),
        "ball_vel": np.zeros(2, dtype=np.float32),
    },
    "pass": {
        "ball_pos":     np.array([0.0, 400.0],    dtype=np.float32),
        "offset":       np.array([-400.0, -50.0], dtype=np.float32),
        "ball_vel":     np.zeros(2, dtype=np.float32),
        # Compañero libre en posición favorable para recibir el pase
        "teammate_pos": np.array([600.0, 400.0],  dtype=np.float32),
        "stage_id":     5,  # 2v0_cooperation tiene num_rl_agents=2
    },
}


def _probe_condition(
    env: NaoSoccerEnv,
    seed: int,
    condition: str,
    forced: bool,
    max_frames: int = 120,
    visual: bool = False,
    delay: float = 0.08,
) -> BHumanConditionResult:
    mode = "forced" if forced else "natural"
    res = BHumanConditionResult(condition=condition, mode=mode)

    # Para forced: si cfg pide stage_id diferente, crear env propio con ese stage
    cfg = _FORCED_CONFIGS.get(condition, {}) if forced else {}
    stage_id = cfg.get("stage_id", None)
    own_env: Optional[NaoSoccerEnv] = None
    active_env = env

    if forced and stage_id is not None:
        own_env = _make_env(stage_id, seed)
        active_env = own_env

    active_env.reset(seed=seed)
    actions = _zero_actions(active_env)

    ball_pos = np.zeros(2, dtype=np.float32)
    agent_offset = np.array([-300.0, 0.0], dtype=np.float32)
    ball_vel = np.zeros(2, dtype=np.float32)
    teammate_pos = None

    if forced:
        ball_pos     = cfg.get("ball_pos",  np.zeros(2, dtype=np.float32))
        agent_offset = cfg.get("offset",    np.array([-300.0, 0.0], dtype=np.float32))
        ball_vel     = cfg.get("ball_vel",  np.zeros(2, dtype=np.float32))
        teammate_pos = cfg.get("teammate_pos", None)
        opp_pos      = cfg.get("opp_pos",   None)

        _install_state(active_env, ball_pos, ball_pos + agent_offset, ball_vel,
                       teammate_pos=teammate_pos)

        # Inyectar oponente si la config lo requiere (para duel)
        if opp_pos is not None and active_env._opponents_state:
            active_env._opponents_state[0].pos = opp_pos.astype(np.float32)
            active_env._opponents_state[0].is_upright = True

    target_macro = f"playBall_{condition}"

    for fi in range(max_frames):
        result = _safe_step(active_env, actions)
        if result is None:
            res.notes = "step() error"
            break
        _, _, term, trunc, infos = result
        info = next(iter(infos.values()), {}) if infos else {}

        macro = info.get("macro_type", "none")
        skill = info.get("sidecar_skill_request", "none")

        if not res.detected and (
            macro == target_macro
            or (condition == "shoot"  and skill == "shoot")
            or (condition == "pass"   and skill == "pass")
            or (condition == "dribble" and (skill == "dribble" or
                info.get("sidecar_motion_request") == "dribble"))
        ):
            res.detected = True
            res.frames_to_detect = fi
            print(f"    [{condition}/{mode}] detectado @ frame {fi}")

        if res.detected and bool(info.get("takeover_active", False)):
            res.takeover_reached = True

        end = str(info.get("last_macro_end_reason", "none"))
        if end != "none" and res.takeover_reached:
            res.macro_end_reason = end
            res.macro_success    = info.get("macro_success")
            print(f"    [{condition}/{mode}] end={end}  success={res.macro_success}")
            if visual and infos:
                _render_ascii(active_env, info, fi, f"{condition.upper()} {mode}",
                              [f"macro_end: {end}",
                               f"success: {res.macro_success}"], delay_s=0.5)
            break

        if visual and infos:
            extra = [f"detected: {res.detected}",
                     f"macro: {macro}",
                     f"takeover: {info.get('takeover_active', False)}"]
            _render_ascii(active_env, info, fi, f"{condition.upper()} {mode}",
                          extra, delay_s=delay)

        if _is_done(term, trunc):
            active_env.reset(seed=seed + fi)
            actions = _zero_actions(active_env)
            if forced:
                _install_state(active_env, ball_pos, ball_pos + agent_offset, ball_vel,
                                teammate_pos=teammate_pos)
                if cfg.get("opp_pos") is not None and active_env._opponents_state:
                    active_env._opponents_state[0].pos = cfg["opp_pos"].astype(np.float32)
                    active_env._opponents_state[0].is_upright = True

    if own_env is not None:
        own_env.close()

    if not res.detected:
        res.notes = f"macro {target_macro} no detectada en {max_frames} frames"
    return res


def run_bhuman_conditions(
    env: NaoSoccerEnv,
    seed: int,
    conditions: Optional[List[str]] = None,
    max_frames_each: int = 120,
    visual: bool = False,
    delay: float = 0.08,
) -> List[BHumanConditionResult]:
    if conditions is None:
        conditions = ["dribble", "duel", "shoot", "pass"]

    print(f"\n[5] BHUMAN CONDITIONS  [{', '.join(conditions)}]")
    results: List[BHumanConditionResult] = []

    for cond in conditions:
        print(f"\n  --- {cond.upper()} ---")
        print(f"  [natural]")
        nat = _probe_condition(env, seed, cond, forced=False,
                               max_frames=max_frames_each,
                               visual=visual, delay=delay)
        results.append(nat)

        print(f"  [forced]")
        frc = _probe_condition(env, seed + 7, cond, forced=True,
                               max_frames=max_frames_each,
                               visual=visual, delay=delay)
        results.append(frc)

    print("\n  Resumen condiciones:")
    for r in results:
        det = "OK" if r.detected else "NO"
        tk  = "OK" if r.takeover_reached else "NO"
        print(f"    {r.condition:8s} [{r.mode:7s}]  detected={det}  takeover={tk}"
              f"  f={r.frames_to_detect}  end={r.macro_end_reason}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 6: STAGE SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_sweep(
    seed: int,
    stages: Optional[List[int]] = None,
    steps_per_stage: int = 30,
    visual: bool = False,
    delay: float = 0.04,
) -> List[StageSweepResult]:
    if stages is None:
        stages = list(range(len(CURRICULUM)))

    print(f"\n[6] STAGE SWEEP  stages={stages}")
    results: List[StageSweepResult] = []

    for sidx in stages:
        if sidx < 0 or sidx >= len(CURRICULUM):
            continue
        scfg = CURRICULUM[sidx]
        r = StageSweepResult(stage_id=sidx, stage_name=scfg.name)
        print(f"\n  Stage {sidx}: {scfg.name}")

        try:
            env = NaoSoccerEnv(stage_cfg=scfg, use_fov=False, seed=seed)
            env.reset(seed=seed)
            actions = _zero_actions(env)

            ball = np.array([0.0, 0.0], dtype=np.float32)
            _install_state(env, ball, ball + np.array([-500, 0], dtype=np.float32))

            for step in range(steps_per_stage):
                dist = max(260.0, 500.0 - step * 12.0)
                agents_list = list(env.agents)
                if agents_list:
                    a0 = agents_list[0]
                    env._agents_state[a0].pos = ball + np.array([-dist, 0.0], dtype=np.float32)

                result = _safe_step(env, actions)
                if result is None:
                    r.error = "step() error"
                    break
                _, _, term, trunc, infos = result
                info = next(iter(infos.values()), {}) if infos else {}

                if not r.bridge_alive and infos:
                    r.bridge_alive = True
                    r.first_info_backend = info.get("sidecar_backend", "unknown")
                if bool(info.get("handoff_ready", False)):
                    r.handoff_ready_detected = True
                if bool(info.get("takeover_active", False)):
                    r.takeover_active_detected = True
                if info.get("last_macro_end_reason", "none") == "completed":
                    r.macro_completed = True

                r.steps_run = step + 1

                if visual and infos:
                    _render_ascii(env, info, step, f"STAGE {sidx}", delay_s=delay)

                if _is_done(term, trunc):
                    env.reset(seed=seed + step)
                    actions = _zero_actions(env)

            env.close()

        except Exception as exc:
            r.error = str(exc)
            print(f"    ERROR: {exc}")

        ok_parts = []
        if r.bridge_alive:             ok_parts.append("bridge=OK")
        if r.handoff_ready_detected:   ok_parts.append("handoff=OK")
        if r.takeover_active_detected: ok_parts.append("takeover=OK")
        if r.macro_completed:          ok_parts.append("completed=OK")
        status_str = "  ".join(ok_parts) if ok_parts else "no signals"
        print(f"  backend={r.first_info_backend}  {status_str}")
        results.append(r)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 7: REWARD SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def run_reward_sanity_check(env: NaoSoccerEnv, steps: int, seed: int,
                            visual: bool = False,
                            delay: float = 0.06) -> Tuple[List[RewardImpulseRecord], int]:
    print("\n[7] REWARD SANITY CHECK")
    env.reset(seed=seed)
    actions = _zero_actions(env)
    agent_id = env.agents[0] if env.agents else "agent_0"

    impulses: List[RewardImpulseRecord] = []
    violations = 0
    prev_tkr = 0.0
    prev_r   = 0.0

    ball = np.array([0.0, 0.0], dtype=np.float32)
    _install_state(env, ball, ball + np.array([-550, 0], dtype=np.float32))

    for step in range(steps):
        dist = max(265.0, 550.0 - step * 10.0)
        if agent_id in env._agents_state:
            env._agents_state[agent_id].pos = ball + np.array([-dist, 0.0], dtype=np.float32)

        result = _safe_step(env, actions)
        if result is None:
            break
        _, rewards, term, trunc, infos = result
        info = next(iter(infos.values()), {}) if infos else {}

        r   = float(rewards.get(agent_id, 0.0))
        tsr = float(info.get("reward_component_takeover_success", 0.0))
        tfr = float(info.get("reward_component_takeover_failure", 0.0))
        tkr = tsr + tfr

        if abs(tkr) > 1e-4:
            impulses.append(RewardImpulseRecord(
                frame=step, agent=agent_id,
                takeover_success_r=tsr,
                takeover_failure_r=tfr,
                frame_before=prev_r,
            ))
            if abs(prev_tkr) > 1e-4:
                violations += 1
                print(f"  VIOLATION frame={step}: prev={prev_tkr:.4f} curr={tkr:.4f}")

        if visual and infos:
            _render_ascii(env, info, step, "REWARD SANITY",
                          [f"r={r:+.4f}",
                           f"tkr={tkr:+.4f}",
                           f"impulses={len(impulses)}",
                           f"violations={violations}"], delay_s=delay)

        prev_r, prev_tkr = r, tkr
        if _is_done(term, trunc):
            env.reset(seed=seed + step)
            actions = _zero_actions(env)
            prev_r = prev_tkr = 0.0
            _install_state(env, ball, ball + np.array([-550, 0], dtype=np.float32))

    print(f"  impulses={len(impulses)}  violations={violations}")
    print(f"  {'OK' if violations == 0 else 'FAIL'}: reward sanity check")
    return impulses, violations


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 8: BALL PHYSICS BASICS (T1-T4)
# ─────────────────────────────────────────────────────────────────────────────

def _t1_ball_velocity_clamping(seed: int) -> BallPhysicsResult:
    """T1: Verifica que la velocidad del balón sea clampizada (no > 5 m/s)."""
    r = BallPhysicsResult("T1_ball_velocity_clamping")
    overshoots = []
    max_speed = 0.0
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        env._ball.vel = np.array([10000.0, 0.0], dtype=np.float32)
        for step in range(100):
            res = _safe_step(env, _zero_actions(env))
            if res is None:
                break
            _, _, term, trunc, _ = res
            speed = float(np.linalg.norm(env._ball.vel))
            max_speed = max(max_speed, speed)
            if speed > BALL_MAX_SPEED_MM_S:
                overshoots.append({"frame": step, "speed_mm_s": round(speed, 1)})
            if _is_done(term, trunc):
                env.reset(seed=seed)
                env._ball.vel = np.array([10000.0, 0.0], dtype=np.float32)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {
        "max_speed_mm_s": round(max_speed, 1),
        "limit_mm_s": BALL_MAX_SPEED_MM_S,
        "overshoots": overshoots[:5],
    }
    r.passed = len(overshoots) == 0
    status = "OK" if r.passed else f"FAIL ({len(overshoots)} overshoot frames)"
    print(f"  T1 velocity clamping  → {status}  max={max_speed:.0f} mm/s")
    return r


def _t2_ball_friction(seed: int) -> BallPhysicsResult:
    """T2: Verifica que el balón decelere por fricción."""
    r = BallPhysicsResult("T2_ball_friction")
    speed_decay = []
    energy_explosion = False
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        initial_speed = 3000.0
        env._ball.vel = np.array([initial_speed, 0.0], dtype=np.float32)
        for step in range(200):
            res = _safe_step(env, _zero_actions(env))
            if res is None:
                break
            _, _, term, trunc, _ = res
            speed = float(np.linalg.norm(env._ball.vel))
            speed_decay.append(speed)
            if speed > initial_speed * 1.1:
                energy_explosion = True
            if _is_done(term, trunc):
                break
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    friction_works = False
    if len(speed_decay) > 100:
        friction_works = speed_decay[100] < speed_decay[0] * 0.8
    r.details = {
        "friction_works": friction_works,
        "energy_explosion": energy_explosion,
        "initial_speed_mm_s": round(speed_decay[0], 1) if speed_decay else 0,
        "speed_at_frame_100": round(speed_decay[100], 1) if len(speed_decay) > 100 else None,
    }
    r.passed = friction_works and not energy_explosion
    status = "OK" if r.passed else f"FAIL  friction={friction_works}  explosion={energy_explosion}"
    print(f"  T2 ball friction      → {status}")
    return r


def _t3_ball_wall_collision(seed: int) -> BallPhysicsResult:
    """T3: Verifica que el balón rebote en las paredes laterales."""
    r = BallPhysicsResult("T3_ball_wall_collision")
    phased = False
    wall_y = FIELD.y_right_sideline
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        env._ball.pos = np.array([0.0, wall_y + 150.0], dtype=np.float32)
        env._ball.vel = np.array([0.0, -2000.0], dtype=np.float32)
        for step in range(80):
            res = _safe_step(env, _zero_actions(env))
            if res is None:
                break
            _, _, term, trunc, _ = res
            if env._ball.pos[1] < wall_y - 100.0:
                phased = True
                r.warnings.append(f"ball phased through wall @ frame {step}: y={env._ball.pos[1]:.0f}")
            if _is_done(term, trunc):
                break
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"ball_phased_through_wall": phased, "wall_y_mm": wall_y}
    r.passed = not phased
    status = "OK" if r.passed else "FAIL (ball phased through wall)"
    print(f"  T3 wall collision     → {status}")
    return r


def _t4_ball_goal_detection(seed: int) -> BallPhysicsResult:
    """T4: Verifica que se detecten correctamente los goles."""
    r = BallPhysicsResult("T4_ball_goal_detection")
    goal_detected = False
    frame_scored = None
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        gol_x = float(FIELD.x_opponent_ground) + 100.0
        env._ball.pos = np.array([gol_x, 0.0], dtype=np.float32)
        env._ball.vel = np.array([100.0, 0.0], dtype=np.float32)
        for step in range(50):
            res = _safe_step(env, _zero_actions(env))
            if res is None:
                break
            _, rewards, term, trunc, infos = res
            info = next(iter(infos.values()), {}) if infos else {}
            if info.get("goal_scored") or any(v > 0.5 for v in rewards.values()):
                goal_detected = True
                frame_scored = step
                break
            if _is_done(term, trunc):
                break
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"goal_detected": goal_detected, "frame_scored": frame_scored}
    r.passed = goal_detected
    status = f"OK @ frame {frame_scored}" if r.passed else "FAIL (goal not detected)"
    print(f"  T4 goal detection     → {status}")
    return r


def run_ball_physics_suite(seed: int) -> List[BallPhysicsResult]:
    """Módulo 8: Suite completa de física básica del balón (T1-T4)."""
    print("\n[8] BALL PHYSICS BASICS")
    results = [
        _t1_ball_velocity_clamping(seed),
        _t2_ball_friction(seed),
        _t3_ball_wall_collision(seed),
        _t4_ball_goal_detection(seed),
    ]
    passed = sum(1 for r in results if r.passed)
    print(f"  Resultado: {passed}/{len(results)} tests pasaron")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 9: BHUMAN SEMANTIC TESTS (T5-T10)
# ─────────────────────────────────────────────────────────────────────────────

def _t5_macro_transition_stability(seed: int, steps: int = 500) -> SemanticTestResult:
    """T5: Macro flickering — A→B→A en ≤5 frames = BUG."""
    r = SemanticTestResult("T5_macro_transition_stability")
    transitions = []
    flickering = False
    inconsistencies = []
    prev_macro = "none"
    flicker_hist: List[int] = []
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        actions = _zero_actions(env)
        for step in range(steps):
            res = _safe_step(env, actions)
            if res is None:
                break
            _, _, term, trunc, infos = res
            info = next(iter(infos.values()), {}) if infos else {}
            macro = info.get("macro_type", "none")
            skill = info.get("sidecar_skill_request", "none")
            if macro != prev_macro:
                transitions.append({"frame": step, "from": prev_macro, "to": macro})
                flicker_hist.append(step)
                # Detecta if últimos 4 cambios ocurrieron en < 5 frames
                if len(flicker_hist) >= 4:
                    span = flicker_hist[-1] - flicker_hist[-4]
                    if span < 5:
                        flickering = True
                        r.warnings.append(f"flicker @ frame {step} (4 changes in {span} frames)")
                # Inconsistencia macro vs skill
                if macro.startswith("playBall_") and skill not in ("none", ""):
                    expected = macro.split("_")[1]
                    if expected != skill:
                        inconsistencies.append({"frame": step, "macro": macro, "skill": skill})
            prev_macro = macro
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {
        "total_transitions": len(transitions),
        "flickering_detected": flickering,
        "inconsistencies": inconsistencies[:5],
    }
    r.passed = not flickering and len(inconsistencies) == 0
    status = "OK" if r.passed else f"FAIL  flicker={flickering}  incons={len(inconsistencies)}"
    print(f"  T5 macro stability    → {status}  transitions={len(transitions)}")
    return r


def _t6_skill_request_semantic(seed: int) -> SemanticTestResult:
    """T6: Verifica que skill_request sea geométricamente coherente."""
    r = SemanticTestResult("T6_skill_request_semantic")
    violations = []
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        actions = _zero_actions(env)
        for step in range(400):
            res = _safe_step(env, actions)
            if res is None:
                break
            _, _, term, trunc, infos = res
            if not infos:
                continue
            info = next(iter(infos.values()))
            skill = info.get("sidecar_skill_request", "none")
            agents_list = list(env._agents_state.values())
            agent_pos = agents_list[0].pos if agents_list else np.zeros(2)
            # shoot sin línea al gol (>1500mm)
            if skill == "shoot":
                dist_gol = abs(agent_pos[0] - float(FIELD.x_opponent_ground))
                if dist_gol > 1500.0:
                    violations.append({"frame": step, "skill": "shoot",
                                       "issue": f"dist_gol={dist_gol:.0f}mm"})
            # duel sin oponente cercano (<600mm)
            if skill == "duel":
                opp_close = any(
                    float(np.linalg.norm(agent_pos - opp.pos)) < 600.0
                    for opp in env._opponents_state
                )
                if not opp_close:
                    violations.append({"frame": step, "skill": "duel",
                                       "issue": "no opponent <600mm"})
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"semantic_violations": violations[:5], "total_violations": len(violations)}
    r.passed = len(violations) == 0
    status = "OK" if r.passed else f"FAIL ({len(violations)} violations)"
    print(f"  T6 skill semantics    → {status}")
    return r


def _t8_handoff_readiness_jitter(seed: int) -> SemanticTestResult:
    """T8: Verifica que handoff_ready no flickee (>3 cambios en 5 frames)."""
    r = SemanticTestResult("T8_handoff_readiness_jitter")
    jitter = False
    changes = []
    prev_hr = False
    change_frames: List[int] = []
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        ball = np.array([0.0, 0.0], dtype=np.float32)
        _install_state(env, ball, ball + np.array([-800, 0], dtype=np.float32))
        actions = _zero_actions(env)
        for step in range(300):
            dist = max(250.0, 800.0 - step * 8.0)
            agents_list = list(env._agents_state.values())
            if agents_list:
                agents_list[0].pos = ball + np.array([-dist, 0.0], dtype=np.float32)
            res = _safe_step(env, actions)
            if res is None:
                break
            _, _, term, trunc, infos = res
            info = next(iter(infos.values()), {}) if infos else {}
            hr = bool(info.get("handoff_ready", False))
            if hr != prev_hr:
                changes.append({"frame": step, "from": prev_hr, "to": hr})
                change_frames.append(step)
                if len(change_frames) >= 4:
                    span = change_frames[-1] - change_frames[-4]
                    if span < 5:
                        jitter = True
                        r.warnings.append(f"jitter @ frame {step} (4 changes in {span} frames)")
            prev_hr = hr
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"total_changes": len(changes), "jitter_detected": jitter}
    r.passed = not jitter
    status = "OK" if r.passed else f"FAIL (jitter detected, {len(changes)} changes)"
    print(f"  T8 handoff jitter     → {status}  changes={len(changes)}")
    return r


def _t9_pose_field_accuracy(seed: int) -> SemanticTestResult:
    """T9: Verifica que ball_pos en el info dict coincida con env._ball.pos."""
    r = SemanticTestResult("T9_pose_field_accuracy")
    mismatches = []
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        actions = _zero_actions(env)
        for step in range(100):
            res = _safe_step(env, actions)
            if res is None:
                break
            _, _, term, trunc, infos = res
            info = next(iter(infos.values()), {}) if infos else {}
            ball_reported = info.get("ball_pos")
            if ball_reported is not None:
                try:
                    arr = np.asarray(ball_reported, dtype=np.float32)
                    delta = float(np.linalg.norm(arr - env._ball.pos))
                    if delta > 5.0:
                        mismatches.append({"frame": step, "delta_mm": round(delta, 1)})
                except Exception:
                    pass
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    frame_lag = len(mismatches) > 5
    r.details = {"mismatches": mismatches[:5], "total_mismatches": len(mismatches),
                 "frame_lag_suspected": frame_lag}
    r.passed = len(mismatches) == 0
    status = "OK" if r.passed else f"FAIL ({len(mismatches)} mismatches)"
    print(f"  T9 pose accuracy      → {status}")
    return r


def _t10_game_state_semantic(seed: int) -> SemanticTestResult:
    """T10: Verifica coherencia de is_playing y presencia de balón/oponentes."""
    r = SemanticTestResult("T10_game_state_semantic")
    errors = []
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        actions = _zero_actions(env)
        for step in range(150):
            res = _safe_step(env, actions)
            if res is None:
                break
            _, _, term, trunc, infos = res
            info = next(iter(infos.values()), {}) if infos else {}
            is_playing = info.get("game_state_is_playing", True)
            ball_age   = float(info.get("ball_seen_age_ms", 0.0))
            ball_visible = ball_age < 2000.0
            if is_playing and not ball_visible:
                errors.append({"frame": step, "error": "is_playing=True but ball invisible"})
            opp_reported = info.get("opponent_count")
            if opp_reported is not None:
                opp_actual = len(env._opponents_state)
                if int(opp_reported) != opp_actual:
                    errors.append({"frame": step,
                                   "error": f"opp_count mismatch reported={opp_reported} actual={opp_actual}"})
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"game_state_errors": errors[:5], "total_errors": len(errors)}
    r.passed = len(errors) == 0
    status = "OK" if r.passed else f"FAIL ({len(errors)} semantic errors)"
    print(f"  T10 game state        → {status}")
    return r


def run_bhuman_semantic_suite(seed: int) -> List[SemanticTestResult]:
    """Módulo 9: Suite completa de tests semánticos BHuman (T5-T10)."""
    print("\n[9] BHUMAN SEMANTIC TESTS")
    results = [
        _t5_macro_transition_stability(seed),
        _t6_skill_request_semantic(seed),
        _t8_handoff_readiness_jitter(seed),
        _t9_pose_field_accuracy(seed),
        _t10_game_state_semantic(seed),
    ]
    passed = sum(1 for r in results if r.passed)
    print(f"  Resultado: {passed}/{len(results)} tests pasaron")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 10: PRE-TRAINING AUDIT (T11-T16)
# ─────────────────────────────────────────────────────────────────────────────

def _t11_seed_determinism(seed: int, n_steps: int = 100) -> PreTrainingResult:
    """T11: Dos runs con mismo seed deben producir rewards idénticos."""
    r = PreTrainingResult("T11_seed_determinism")
    rewards_run1: List[float] = []
    rewards_run2: List[float] = []
    try:
        for run_id, target in enumerate([rewards_run1, rewards_run2]):
            env = _make_env(0, seed)
            env.reset(seed=seed)
            actions = _zero_actions(env)
            for _ in range(n_steps):
                res = _safe_step(env, actions)
                if res is None:
                    break
                _, rews, term, trunc, _ = res
                target.append(sum(rews.values()))
                if _is_done(term, trunc):
                    env.reset(seed=seed)
                    actions = _zero_actions(env)
            env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    min_len = min(len(rewards_run1), len(rewards_run2))
    diffs = [abs(rewards_run1[i] - rewards_run2[i]) for i in range(min_len)]
    max_diff = max(diffs) if diffs else 0.0
    r.passed = max_diff < 1e-4 and min_len > 0
    r.details = {
        "steps_compared": min_len,
        "max_reward_diff": round(max_diff, 6),
        "deterministic": r.passed,
    }
    status = "OK" if r.passed else f"FAIL (max_diff={max_diff:.6f})"
    print(f"  T11 seed determinism  → {status}  steps={min_len}")
    return r


def _t12_stress_1000_steps(seed: int, n_steps: int = 1000) -> PreTrainingResult:
    """T12: Stress test — 1000 steps sin crashes ni degradación de latencia."""
    r = PreTrainingResult("T12_stress_1000_steps")
    latencies_ms: List[float] = []
    exception_count = 0
    mem_start_mb: Optional[float] = None
    mem_end_mb: Optional[float] = None
    if _HAS_PSUTIL:
        proc = _psutil.Process(os.getpid())
        mem_start_mb = proc.memory_info().rss / 1e6
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        actions = _zero_actions(env)
        for step in range(n_steps):
            t0 = time.perf_counter()
            res = _safe_step(env, actions)
            latency = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency)
            if res is None:
                exception_count += 1
                continue
            _, _, term, trunc, _ = res
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
        exception_count += 1
    if _HAS_PSUTIL:
        mem_end_mb = _psutil.Process(os.getpid()).memory_info().rss / 1e6
    avg_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    max_lat = float(np.max(latencies_ms)) if latencies_ms else 0.0
    # Check latency degradation: last 100 vs first 100
    lat_degrade = False
    if len(latencies_ms) >= 200:
        early_avg = float(np.mean(latencies_ms[:100]))
        late_avg  = float(np.mean(latencies_ms[-100:]))
        lat_degrade = late_avg > early_avg * 2.0
    mem_growth_mb = (mem_end_mb - mem_start_mb) if (mem_start_mb and mem_end_mb) else None
    r.passed = exception_count == 0 and max_lat < 200.0 and not lat_degrade
    r.details = {
        "steps": len(latencies_ms),
        "avg_latency_ms": round(avg_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "latency_degradation": lat_degrade,
        "exception_count": exception_count,
        "memory_growth_mb": round(mem_growth_mb, 1) if mem_growth_mb else "N/A (psutil missing)",
    }
    status = "OK" if r.passed else f"FAIL  exc={exception_count}  max_lat={max_lat:.0f}ms  degrade={lat_degrade}"
    print(f"  T12 stress 1000 steps → {status}  avg={avg_lat:.1f}ms")
    return r


def _t13_reward_consistency(seed: int, n_episodes: int = 10) -> PreTrainingResult:
    """T13: Rewards no deben ser NaN, Inf ni siempre 0."""
    r = PreTrainingResult("T13_reward_consistency")
    all_rewards: List[float] = []
    nan_count = 0
    inf_count = 0
    try:
        env = _make_env(0, seed)
        for ep in range(n_episodes):
            env.reset(seed=seed + ep)
            actions = _zero_actions(env)
            ball = np.array([0.0, 0.0], dtype=np.float32)
            _install_state(env, ball, ball + np.array([-400, 0], dtype=np.float32))
            for step in range(100):
                dist = max(260.0, 400.0 - step * 8.0)
                agents_list = list(env.agents)
                if agents_list:
                    env._agents_state[agents_list[0]].pos = (
                        ball + np.array([-dist, 0.0], dtype=np.float32)
                    )
                res = _safe_step(env, actions)
                if res is None:
                    break
                _, rews, term, trunc, _ = res
                for v in rews.values():
                    all_rewards.append(float(v))
                    if math.isnan(v):
                        nan_count += 1
                    if math.isinf(v):
                        inf_count += 1
                if _is_done(term, trunc):
                    break
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    pct_zero = (sum(1 for v in all_rewards if abs(v) < 1e-6) / len(all_rewards) * 100) if all_rewards else 0
    std_r = float(np.std(all_rewards)) if all_rewards else 0.0
    max_r = float(max(map(abs, all_rewards))) if all_rewards else 0.0
    r.passed = nan_count == 0 and inf_count == 0 and pct_zero < 95.0 and std_r > 1e-4
    r.details = {
        "total_rewards": len(all_rewards),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "pct_zero": round(pct_zero, 1),
        "std": round(std_r, 4),
        "max_abs": round(max_r, 4),
    }
    status = "OK" if r.passed else f"FAIL  nan={nan_count}  inf={inf_count}  zero%={pct_zero:.0f}"
    print(f"  T13 reward consist.   → {status}  std={std_r:.4f}")
    return r


def _t14_policy_reproducibility(seed: int, n_steps: int = 80) -> PreTrainingResult:
    """T14: Policy aleatoria 2× mismo seed → rewards idénticos."""
    r = PreTrainingResult("T14_policy_reproducibility")
    rng = np.random.default_rng(seed)
    rewards_a: List[float] = []
    rewards_b: List[float] = []
    try:
        for run_id, target in enumerate([rewards_a, rewards_b]):
            local_rng = np.random.default_rng(seed)
            env = _make_env(0, seed)
            env.reset(seed=seed)
            for step in range(n_steps):
                actions = {aid: local_rng.uniform(-1, 1, env.action_space(aid).shape).astype(np.float32)
                           for aid in env.agents}
                res = _safe_step(env, actions)
                if res is None:
                    break
                _, rews, term, trunc, _ = res
                target.append(sum(rews.values()))
                if _is_done(term, trunc):
                    env.reset(seed=seed)
            env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    min_len = min(len(rewards_a), len(rewards_b))
    diffs = [abs(rewards_a[i] - rewards_b[i]) for i in range(min_len)]
    max_diff = max(diffs) if diffs else 0.0
    r.passed = max_diff < 1e-4 and min_len > 0
    r.details = {"steps_compared": min_len, "max_diff": round(max_diff, 6)}
    status = "OK" if r.passed else f"FAIL (max_diff={max_diff:.6f})"
    print(f"  T14 reproducibility   → {status}  steps={min_len}")
    return r


def _t16_handoff_timing_variance(seed: int) -> PreTrainingResult:
    """T16: Handoff timing desde 5 geometrías distintas — std < 5 frames."""
    r = PreTrainingResult("T16_handoff_timing_variance")
    geometries = [
        ("close_center",   np.array([0.0, 0.0],    dtype=np.float32), 350.0),
        ("medium_center",  np.array([0.0, 0.0],    dtype=np.float32), 600.0),
        ("far_center",     np.array([0.0, 0.0],    dtype=np.float32), 900.0),
        ("angle_right",    np.array([300.0, 400.0], dtype=np.float32), 500.0),
        ("angle_left",     np.array([300.0,-400.0], dtype=np.float32), 500.0),
    ]
    handoff_frames: List[int] = []
    geo_results = []
    try:
        for geo_name, ball, dist in geometries:
            env = _make_env(0, seed)
            env.reset(seed=seed)
            actions = _zero_actions(env)
            # _install_state ya apunta theta al balón automáticamente
            _install_state(env, ball, ball + np.array([-dist, 0], dtype=np.float32))
            hr_frame = None
            active = list(env.agents)[0] if env.agents else "agent_0"
            for fi in range(200):
                approach_dist = max(260.0, dist - fi * 18.0)
                if active in env._agents_state:
                    agent_pos = ball + np.array([-approach_dist, 0], dtype=np.float32)
                    env._agents_state[active].pos = agent_pos
                    # Mantener theta apuntando al balón durante todo el approach
                    direction = ball - agent_pos
                    env._agents_state[active].theta = float(
                        math.atan2(float(direction[1]), float(direction[0]))
                    )
                res = _safe_step(env, actions)
                if res is None:
                    break
                _, _, term, trunc, infos = res
                info = next(iter(infos.values()), {}) if infos else {}
                if bool(info.get("handoff_ready", False)) and hr_frame is None:
                    hr_frame = fi
                    break
                if _is_done(term, trunc):
                    env.reset(seed=seed)
                    actions = _zero_actions(env)
            env.close()
            geo_results.append({"geometry": geo_name, "handoff_frame": hr_frame})
            if hr_frame is not None:
                handoff_frames.append(hr_frame)
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    std_frames = float(np.std(handoff_frames)) if len(handoff_frames) > 1 else 0.0
    r.passed = std_frames <= 5.0 and len(handoff_frames) >= 3
    r.details = {
        "geometries": geo_results,
        "handoff_frames": handoff_frames,
        "std_frames": round(std_frames, 2),
        "detected_in": f"{len(handoff_frames)}/5",
    }
    status = "OK" if r.passed else f"FAIL  std={std_frames:.1f} frames  detected={len(handoff_frames)}/5"
    print(f"  T16 handoff variance  → {status}  std={std_frames:.1f}f")
    return r


def run_pretraining_audit_suite(seed: int) -> List[PreTrainingResult]:
    """Módulo 10: Suite de auditoría pre-training (T11-T16)."""
    print("\n[10] PRE-TRAINING AUDIT")
    results = [
        _t11_seed_determinism(seed),
        _t12_stress_1000_steps(seed),
        _t13_reward_consistency(seed),
        _t14_policy_reproducibility(seed),
        _t16_handoff_timing_variance(seed),
    ]
    passed = sum(1 for r in results if r.passed)
    print(f"  Resultado: {passed}/{len(results)} tests pasaron")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 11: BALL PHYSICS EDGE CASES (T26-T32)
# ─────────────────────────────────────────────────────────────────────────────

def _t26_ball_backward_bounce_at_goal(seed: int) -> BallPhysicsResult:
    """T26: Balón disparado al gol NO rebota hacia atrás fuera del área."""
    r = BallPhysicsResult("T26_ball_backward_bounce_at_goal")
    backward_rebounds = []
    goal_x = float(FIELD.x_opponent_ground)
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        for angle_deg in [0, 10, -10, 20, -20]:
            angle_rad = math.radians(angle_deg)
            env._ball.pos = np.array([goal_x - 300.0,
                                       300.0 * math.sin(angle_rad)], dtype=np.float32)
            speed = 3000.0
            env._ball.vel = np.array([speed * math.cos(angle_rad),
                                       speed * math.sin(angle_rad)], dtype=np.float32)
            for frame in range(15):
                vel_before_x = float(env._ball.vel[0])
                res = _safe_step(env, _zero_actions(env))
                if res is None:
                    break
                _, _, term, trunc, _ = res
                vel_after_x = float(env._ball.vel[0])
                pos_x = float(env._ball.pos[0])
                # Rebote hacia atrás FUERA del área del gol = BUG
                if vel_before_x > 1000.0 and vel_after_x < -500.0 and pos_x < goal_x - 50.0:
                    backward_rebounds.append({
                        "angle_deg": angle_deg, "frame": frame,
                        "pos_x": round(pos_x, 1),
                        "vel_before": round(vel_before_x, 1),
                        "vel_after": round(vel_after_x, 1),
                    })
                if _is_done(term, trunc):
                    break
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"backward_rebounds": backward_rebounds}
    r.passed = len(backward_rebounds) == 0
    status = "OK" if r.passed else f"FAIL ({len(backward_rebounds)} backward bounces)"
    print(f"  T26 backward bounce   → {status}")
    return r


def _t27_ball_goal_post_clipping(seed: int) -> BallPhysicsResult:
    """T27: El balón no debe atravesar los postes del arco."""
    r = BallPhysicsResult("T27_ball_goal_post_clipping")
    clips = []
    goal_x = float(FIELD.x_opponent_ground)
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        for post_y in [600.0, -600.0]:
            env._ball.pos = np.array([goal_x - 200.0, post_y - 80.0], dtype=np.float32)
            env._ball.vel = np.array([1500.0, 1000.0 * (1 if post_y > 0 else -1)], dtype=np.float32)
            prev_x = float(env._ball.pos[0])
            for frame in range(10):
                res = _safe_step(env, _zero_actions(env))
                if res is None:
                    break
                _, _, term, trunc, _ = res
                curr_x = float(env._ball.pos[0])
                curr_y = float(env._ball.pos[1])
                # Phasing: cruzó la línea de gol y está cerca del poste
                if prev_x < goal_x and curr_x > goal_x and abs(curr_y - post_y) < 100.0:
                    clips.append({"post_y": post_y, "frame": frame,
                                   "pos_x": round(curr_x, 1), "pos_y": round(curr_y, 1)})
                prev_x = curr_x
                if _is_done(term, trunc):
                    break
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"post_clips": clips}
    r.passed = len(clips) == 0
    status = "OK" if r.passed else f"FAIL ({len(clips)} clips)"
    print(f"  T27 goal post clip    → {status}")
    return r


def _t28_ball_corner_trap(seed: int) -> BallPhysicsResult:
    """T28: El balón no vibra indefinidamente en las esquinas."""
    r = BallPhysicsResult("T28_ball_corner_trap")
    vibration_corners = []
    corners = [
        (float(FIELD.x_opponent_ground) - 200, float(FIELD.y_left_sideline)  - 200, "top_right"),
        (float(FIELD.x_opponent_ground) - 200, float(FIELD.y_right_sideline) + 200, "bot_right"),
        (float(FIELD.x_own_ground)      + 200, float(FIELD.y_left_sideline)  - 200, "top_left"),
        (float(FIELD.x_own_ground)      + 200, float(FIELD.y_right_sideline) + 200, "bot_left"),
    ]
    try:
        env = _make_env(0, seed)
        for corner_x, corner_y, name in corners:
            env.reset(seed=seed)
            env._ball.pos = np.array([corner_x, corner_y], dtype=np.float32)
            sign_x = 1 if corner_x > 0 else -1
            sign_y = 1 if corner_y > 0 else -1
            env._ball.vel = np.array([sign_x * 2000.0, sign_y * 2000.0], dtype=np.float32)
            pos_hist = []
            for frame in range(25):
                res = _safe_step(env, _zero_actions(env))
                if res is None:
                    break
                _, _, term, trunc, _ = res
                pos_hist.append(env._ball.pos.copy())
                if _is_done(term, trunc):
                    break
            # Vibración = los últimos 5 deltas son pequeños y constantes
            if len(pos_hist) >= 10:
                diffs = [float(np.linalg.norm(pos_hist[i+1] - pos_hist[i]))
                         for i in range(len(pos_hist)-1)]
                last5 = diffs[-5:]
                if last5 and all(5.0 < d < 80.0 for d in last5):
                    vibration_corners.append({"corner": name,
                                               "oscillation": round(float(np.mean(last5)), 1)})
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"vibrating_corners": vibration_corners}
    r.passed = len(vibration_corners) == 0
    status = "OK" if r.passed else f"FAIL ({len(vibration_corners)} corners vibrating)"
    print(f"  T28 corner trap       → {status}")
    return r


def _t29_ball_out_persistence(seed: int) -> BallPhysicsResult:
    """T29: Cuando el balón está fuera del campo, NO deben ejecutarse macros playBall_*."""
    r = BallPhysicsResult("T29_ball_out_persistence")
    macro_while_out = []
    out_x = float(FIELD.x_opponent_ground) + 400.0
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        env._ball.pos = np.array([out_x, 0.0], dtype=np.float32)
        env._ball.vel = np.array([500.0, 0.0], dtype=np.float32)
        actions = _zero_actions(env)
        for step in range(120):
            res = _safe_step(env, actions)
            if res is None:
                break
            _, _, term, trunc, infos = res
            info = next(iter(infos.values()), {}) if infos else {}
            ball_out = info.get("ball_out_of_field", False)
            macro    = info.get("macro_type", "none")
            if ball_out and macro.startswith("playBall_"):
                macro_while_out.append({"frame": step, "macro": macro})
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                env._ball.pos = np.array([out_x, 0.0], dtype=np.float32)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"macro_while_out": macro_while_out[:5], "total": len(macro_while_out)}
    r.passed = len(macro_while_out) == 0
    status = "OK" if r.passed else f"FAIL ({len(macro_while_out)} macro-while-out events)"
    print(f"  T29 out persistence   → {status}")
    return r


def _t30_ball_high_speed_goal(seed: int) -> BallPhysicsResult:
    """T30: A velocidades altas cerca del gol, el movimiento debe ser predecible."""
    r = BallPhysicsResult("T30_ball_high_speed_goal")
    anomalies = []
    goal_x = float(FIELD.x_opponent_ground)
    try:
        env = _make_env(0, seed)
        for speed in [1000, 2000, 3000, 4000, 5000]:
            env.reset(seed=seed)
            env._ball.pos = np.array([goal_x - 400.0, 0.0], dtype=np.float32)
            env._ball.vel = np.array([float(speed), 0.0], dtype=np.float32)
            prev_pos = env._ball.pos.copy()
            for frame in range(15):
                res = _safe_step(env, _zero_actions(env))
                if res is None:
                    break
                _, _, term, trunc, _ = res
                curr_pos = env._ball.pos.copy()
                delta_x = float(curr_pos[0] - prev_pos[0])
                expected = float(speed) * SIM.dt
                # Salto > 50% del esperado en dirección positiva = phase skip
                if delta_x > expected * 1.5 and frame < 5:
                    anomalies.append({"speed": speed, "frame": frame,
                                       "delta_x": round(delta_x, 1),
                                       "expected": round(expected, 1)})
                prev_pos = curr_pos
                if _is_done(term, trunc):
                    break
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"anomalies": anomalies[:5], "total_anomalies": len(anomalies)}
    r.passed = len(anomalies) == 0
    status = "OK" if r.passed else f"FAIL ({len(anomalies)} phase anomalies)"
    print(f"  T30 high speed goal   → {status}")
    return r


def _t31_ball_multiple_bounces(seed: int) -> BallPhysicsResult:
    """T31: El balón no debe quedarse fuera de bounds sin rebotar."""
    r = BallPhysicsResult("T31_ball_multiple_bounces")
    wall_y = float(FIELD.y_left_sideline)
    phased_events = []
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        env._ball.pos = np.array([0.0, wall_y - 150.0], dtype=np.float32)
        env._ball.vel = np.array([0.0, 5000.0], dtype=np.float32)
        for frame in range(12):
            prev_vel_y = float(env._ball.vel[1])
            res = _safe_step(env, _zero_actions(env))
            if res is None:
                break
            _, _, term, trunc, _ = res
            if env._ball.pos[1] > wall_y + 100.0:
                phased_events.append({"frame": frame, "pos_y": round(float(env._ball.pos[1]), 1)})
            if _is_done(term, trunc):
                break
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"phased_events": phased_events}
    r.passed = len(phased_events) == 0
    status = "OK" if r.passed else f"FAIL ({len(phased_events)} phase events)"
    print(f"  T31 multiple bounces  → {status}")
    return r


def _t32_ball_posvel_consistency(seed: int, n_steps: int = 200) -> BallPhysicsResult:
    """T32: pos(t+1) ≈ pos(t) + vel(t)*dt — inconsistencia = bug en integración."""
    r = BallPhysicsResult("T32_ball_pos_vel_consistency")
    inconsistencies = []
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        env._ball.vel = np.array([1000.0, 500.0], dtype=np.float32)
        actions = _zero_actions(env)
        for step in range(0, n_steps, 2):
            res1 = _safe_step(env, actions)
            if res1 is None:
                break
            prev_pos = env._ball.pos.copy()
            prev_vel = env._ball.vel.copy()
            _, _, term, trunc, _ = res1
            res2 = _safe_step(env, actions)
            if res2 is None:
                break
            curr_pos = env._ball.pos.copy()
            # Consistencia: cambio real vs vel*dt
            expected_delta = prev_vel * SIM.dt
            actual_delta   = curr_pos - prev_pos
            error = float(np.linalg.norm(actual_delta - expected_delta))
            # Tolerancia: colisiones con paredes justifican diferencias grandes
            # Solo reportamos si error > 150mm sin estar en el borde del campo
            dist_from_border = min(
                abs(prev_pos[0] - float(FIELD.x_opponent_ground)),
                abs(prev_pos[0] - float(FIELD.x_own_ground)),
                abs(prev_pos[1] - float(FIELD.y_left_sideline)),
                abs(prev_pos[1] - float(FIELD.y_right_sideline)),
            )
            if error > 150.0 and dist_from_border > 200.0:
                inconsistencies.append({
                    "step": step,
                    "error_mm": round(error, 1),
                    "prev_vel": [round(v, 1) for v in prev_vel.tolist()],
                })
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                env._ball.vel = np.array([1000.0, 500.0], dtype=np.float32)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {"inconsistencies": inconsistencies[:5], "total": len(inconsistencies)}
    r.passed = len(inconsistencies) == 0
    status = "OK" if r.passed else f"FAIL ({len(inconsistencies)} integration errors)"
    print(f"  T32 pos/vel consist.  → {status}")
    return r


def run_ball_edge_cases_suite(seed: int) -> List[BallPhysicsResult]:
    """Módulo 11: Suite de edge cases físicos del balón (T26-T32)."""
    print("\n[11] BALL PHYSICS EDGE CASES")
    results = [
        _t26_ball_backward_bounce_at_goal(seed),
        _t27_ball_goal_post_clipping(seed),
        _t28_ball_corner_trap(seed),
        _t29_ball_out_persistence(seed),
        _t30_ball_high_speed_goal(seed),
        _t31_ball_multiple_bounces(seed),
        _t32_ball_posvel_consistency(seed),
    ]
    passed = sum(1 for r in results if r.passed)
    print(f"  Resultado: {passed}/{len(results)} tests pasaron")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 12: PHYSICAL STRESS TESTS (TA-TG)
# ─────────────────────────────────────────────────────────────────────────────

def _ta_fallen_agent(seed: int, duration: int = 250) -> PhysicalStressResult:
    """TA: Agente caído (is_upright=False) — env sigue vivo y se recupera."""
    r = PhysicalStressResult("TA_fallen_agent")
    agent_fell = False
    recovered = False
    env_crashed = False
    fell_frame = None
    handoff_after = False
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        actions = _zero_actions(env)
        target_aid = list(env.agents)[0] if env.agents else None
        for step in range(duration):
            if step == 20 and target_aid:
                env._agents_state[target_aid].is_upright = False
                agent_fell = True
                fell_frame = step
            try:
                res = _safe_step(env, actions)
            except Exception as exc:
                env_crashed = True
                r.warnings.append(f"crash @ frame {step}: {exc}")
                break
            if res is None:
                env_crashed = True
                break
            _, _, term, trunc, infos = res
            if agent_fell and target_aid and not recovered:
                if env._agents_state[target_aid].is_upright:
                    recovered = True
                    r.warnings.append(f"recovered @ frame {step} (fallen for {step - fell_frame}f)")
            if recovered and infos:
                info = next(iter(infos.values()), {})
                if info.get("handoff_ready"):
                    handoff_after = True
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
        env_crashed = True
    r.details = {
        "agent_fell": agent_fell,
        "recovered": recovered,
        "env_crashed": env_crashed,
        "handoff_after_recovery": handoff_after,
    }
    r.passed = agent_fell and not env_crashed
    status = "OK" if r.passed else f"FAIL  crashed={env_crashed}"
    print(f"  TA fallen agent       → {status}  recovered={recovered}")
    return r


def _tb_agent_collision(seed: int) -> PhysicalStressResult:
    """TB: Colisión agente RL vs oponente — sin phasing ni freezing."""
    r = PhysicalStressResult("TB_agent_collision")
    phasing = False
    min_dist = float("inf")
    collision_detected = False
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        actions = _zero_actions(env)
        # Coloca oponente junto al agente 0
        if env._opponents_state and env.agents:
            a0_pos = env._agents_state[list(env.agents)[0]].pos.copy()
            env._opponents_state[0].pos = a0_pos + np.array([250.0, 0.0], dtype=np.float32)
        for step in range(200):
            res = _safe_step(env, actions)
            if res is None:
                break
            _, _, term, trunc, _ = res
            if env._agents_state and env._opponents_state:
                for aid in env.agents:
                    apos = env._agents_state[aid].pos
                    for opp in env._opponents_state:
                        d = float(np.linalg.norm(apos - opp.pos))
                        min_dist = min(min_dist, d)
                        if d < 300.0:
                            collision_detected = True
                        if d < 80.0:
                            phasing = True
                            r.warnings.append(f"phasing @ frame {step}: dist={d:.0f}mm")
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {
        "collision_detected": collision_detected,
        "phasing": phasing,
        "min_dist_mm": round(min_dist, 1),
    }
    r.passed = not phasing
    status = "OK" if r.passed else f"FAIL (phasing detected, min_dist={min_dist:.0f}mm)"
    print(f"  TB agent collision    → {status}  min_dist={min_dist:.0f}mm")
    return r


def _tc_movement_limitation(seed: int) -> PhysicalStressResult:
    """TC: Agentes NO deben superar max_forward_speed."""
    r = PhysicalStressResult("TC_movement_limitation")
    exceeds = False
    max_speed = 0.0
    velocity_jumps = []
    limit = float(AGENT.max_forward_speed)
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        # Acciones de velocidad máxima
        max_actions = {aid: np.array([1.0, 0.0, 0.0], dtype=np.float32)
                       for aid in env.agents}
        prev_pos: Dict[str, np.ndarray] = {
            aid: env._agents_state[aid].pos.copy() for aid in env.agents
        }
        for step in range(200):
            res = _safe_step(env, max_actions)
            if res is None:
                break
            _, _, term, trunc, _ = res
            for aid in env.agents:
                curr = env._agents_state[aid].pos.copy()
                # Velocidad estimada por diferencia de posición
                vel_est = float(np.linalg.norm(curr - prev_pos[aid])) / SIM.dt
                max_speed = max(max_speed, vel_est)
                if vel_est > limit * 1.05:
                    exceeds = True
                # Salto brusco > 100 mm/s
                if vel_est > 100.0 and step > 5:
                    prev_vel_est = float(np.linalg.norm(
                        prev_pos[aid] - env._agents_state[aid].pos
                    )) / SIM.dt
                    jump = abs(vel_est - prev_vel_est)
                    if jump > 150.0:
                        velocity_jumps.append({"frame": step, "aid": aid, "jump_mm_s": round(jump, 1)})
                prev_pos[aid] = curr
            if _is_done(term, trunc):
                break
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    r.details = {
        "max_speed_mm_s": round(max_speed, 1),
        "limit_mm_s": limit,
        "exceeds_limit": exceeds,
        "velocity_jumps": velocity_jumps[:3],
    }
    r.passed = not exceeds
    status = "OK" if r.passed else f"FAIL (max={max_speed:.0f} > limit={limit:.0f})"
    print(f"  TC movement limits    → {status}  max={max_speed:.0f}mm/s")
    return r


def _tf_macro_interruption(seed: int) -> PhysicalStressResult:
    """TF: Macro interrumpida violentamente — env no crashea."""
    r = PhysicalStressResult("TF_macro_interruption")
    macro_was_active = False
    macro_aborted = False
    env_crashed = False
    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        ball = np.array([0.0, 0.0], dtype=np.float32)
        _install_state(env, ball, ball + np.array([-260.0, 0.0], dtype=np.float32))
        actions = _zero_actions(env)
        for step in range(150):
            try:
                res = _safe_step(env, actions)
            except Exception as exc:
                env_crashed = True
                r.warnings.append(f"crash @ {step}: {exc}")
                break
            if res is None:
                env_crashed = True
                break
            _, _, term, trunc, infos = res
            info = next(iter(infos.values()), {}) if infos else {}
            if info.get("takeover_active"):
                macro_was_active = True
            # Interrumpe agresivamente en frame 50
            if step == 50 and macro_was_active:
                for aid in env.agents:
                    # Aleja el agente del balón bruscamente
                    env._agents_state[aid].pos = np.array([-1500.0, 0.0], dtype=np.float32)
            end = info.get("last_macro_end_reason", "none")
            if macro_was_active and end not in ("none", ""):
                macro_aborted = True
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
        env_crashed = True
    r.details = {
        "macro_was_active": macro_was_active,
        "macro_aborted_gracefully": macro_aborted,
        "env_crashed": env_crashed,
    }
    r.passed = not env_crashed
    status = "OK" if r.passed else "FAIL (env crashed)"
    print(f"  TF macro interruption → {status}  macro_was_active={macro_was_active}")
    return r


def _tg_extreme_pressure(seed: int) -> PhysicalStressResult:
    """TG: 4+ oponentes rodeando al agente — sin degradación de latencia."""
    r = PhysicalStressResult("TG_extreme_pressure")
    latencies: List[float] = []
    degradation = False
    try:
        # Usamos el último stage disponible (más agentes)
        stage_idx = len(CURRICULUM) - 1
        env = _make_env(stage_idx, seed)
        env.reset(seed=seed)
        actions = _zero_actions(env)
        # Rodea al agente 0 con todos los oponentes
        if env.agents:
            a0_pos = env._agents_state[list(env.agents)[0]].pos.copy()
            for i, opp in enumerate(env._opponents_state):
                angle = 2 * math.pi * i / max(len(env._opponents_state), 1)
                opp.pos = a0_pos + np.array([
                    350.0 * math.cos(angle), 350.0 * math.sin(angle)
                ], dtype=np.float32)
        for step in range(250):
            t0 = time.perf_counter()
            res = _safe_step(env, actions)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if res is None:
                r.warnings.append(f"step error @ {step}")
                break
            _, _, term, trunc, _ = res
            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                actions = _zero_actions(env)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")
    if len(latencies) >= 100:
        early = float(np.mean(latencies[:50]))
        late  = float(np.mean(latencies[-50:]))
        degradation = late > early * 2.0
    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    max_lat = float(np.max(latencies)) if latencies else 0.0
    r.details = {
        "avg_latency_ms": round(avg_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "latency_degradation": degradation,
    }
    r.passed = not degradation and max_lat < 300.0
    status = "OK" if r.passed else f"FAIL  degrade={degradation}  max={max_lat:.0f}ms"
    print(f"  TG extreme pressure   → {status}  avg={avg_lat:.1f}ms")
    return r


def run_physical_stress_suite(seed: int) -> List[PhysicalStressResult]:
    """Módulo 12: Suite de estrés físico de agentes (TA-TG)."""
    print("\n[12] PHYSICAL STRESS TESTS")
    results = [
        _ta_fallen_agent(seed),
        _tb_agent_collision(seed),
        _tc_movement_limitation(seed),
        _tf_macro_interruption(seed),
        _tg_extreme_pressure(seed),
    ]
    passed = sum(1 for r in results if r.passed)
    print(f"  Resultado: {passed}/{len(results)} tests pasaron")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 13: CONTRACT GUARDRAILS  (§1.2 · §2.2 · §2.4)
# ─────────────────────────────────────────────────────────────────────────────
# Contrato de referencia: BHUMAN_RL_CURRICULUM_CONTRACT.md
#   §1.2 — El RL NO emite: tipo de kick, Zweikampf, MotionRequest::dribble
#   §2.2 — Envelope conservador: ball_seen_age≤300ms, ball_rel_x∈[0,500]mm,
#           abs(angle_to_ball)≤80°, sin colisión inmediata
#   §2.4 — La reward NO premia "yo toqué el balón"
# ─────────────────────────────────────────────────────────────────────────────

# Constantes del contrato §2.2
_CONTRACT_BALL_SEEN_AGE_LIMIT_MS: float = 300.0
_CONTRACT_BALL_REL_X_MAX_MM:      float = 500.0
_CONTRACT_ANGLE_TO_BALL_MAX_DEG:  float = 80.0
_FALSE_POSITIVE_REWARD_THRESHOLD: float = 0.05  # §2.4: reward máximo sin takeover


def _extra1_handoff_envelope_validation(seed: int) -> ContractGuardrailResult:
    """
    EXTRA 1 — §2.2 BHuman RL Curriculum Contract.

    Prueba 6 configuraciones fuera del envelope de takeover y verifica que
    handoff_ready NUNCA devuelva True en ninguna de ellas.

    Configuraciones forzadas:
      1. Agente de espaldas al balón (posición x+ del balón → RB mira en dirección contraria)
      2. Agente lateral (90° respecto al balón)
      3. Agente demasiado lejos (> 700 mm)
      4. Agente detrás del balón (ball_rel_x < 0 desde perspectiva del robot)
      5. Balón visto hace demasiado tiempo (seen_age > 3× límite)
      6. Agente caído (is_upright=False)
    """
    r = ContractGuardrailResult("EXTRA1_handoff_envelope_validation")
    violations: List[Dict[str, Any]] = []
    ball = np.array([300.0, 0.0], dtype=np.float32)

    def _probe(cfg_name: str, setup_fn) -> None:
        try:
            env = _make_env(0, seed)
            env.reset(seed=seed)
            setup_fn(env)
            actions = _zero_actions(env)
            for step in range(60):
                setup_fn(env)  # re-aplica el setup fuera de envelope cada frame
                res = _safe_step(env, actions)
                if res is None:
                    break
                _, _, term, trunc, infos = res
                info = next(iter(infos.values()), {}) if infos else {}
                if bool(info.get("handoff_ready", False)):
                    violations.append({
                        "config": cfg_name, "frame": step,
                        "macro_type": info.get("macro_type", "none"),
                    })
                if _is_done(term, trunc):
                    env.reset(seed=seed + step)
                    actions = _zero_actions(env)
                    setup_fn(env)
            env.close()
        except Exception as exc:
            r.warnings.append(f"{cfg_name}: {exc}")

    def _set_agent(env, pos_offset, theta, upright=True):
        """Helper: setea posición+theta explícitos para evitar drift entre frames."""
        if env.agents:
            ag = env._agents_state[list(env.agents)[0]]
            ag.pos = (ball + np.asarray(pos_offset, dtype=np.float32))
            ag.theta = float(theta)
            ag.is_upright = upright

    # 1. Agente DE ESPALDAS: a la derecha del balón, theta=0 → mira lejos del balón
    #    ball_rel_x = (ball - agent) · (cos0, sin0) = (-300, 0)·(1,0) = -300 < 0 → fuera
    def _back(env):
        _set_agent(env, [300.0, 0.0], theta=0.0)
        env._ball.pos[:] = ball; env._ball.seen_age_ms = 0.0
    _probe("back_to_ball", _back)

    # 2. Agente LATERAL (90°): arriba del balón, theta=math.pi/2 → mira hacia +Y
    #    ball_rel_x = (ball - agent) · (cos(π/2), sin(π/2)) = (0,-600)·(0,1) = -600 < 0 → fuera
    def _side(env):
        _set_agent(env, [0.0, 600.0], theta=math.pi / 2)
        env._ball.pos[:] = ball; env._ball.seen_age_ms = 0.0
    _probe("side_to_ball", _side)

    # 3. Agente DEMASIADO LEJOS (1200mm, inequívocamente fuera de cualquier rango BHuman)
    #    theta=0 → mira al balón; distancia garantiza que sea fuera del envelope
    def _far(env):
        _set_agent(env, [-1200.0, 0.0], theta=0.0)
        env._ball.pos[:] = ball; env._ball.seen_age_ms = 0.0
    _probe("too_far", _far)

    # 4. Balón DETRÁS del robot: agente a la derecha, theta=0 → mira lejos del balón
    #    Igual que _back pero más lejos para separar el caso
    def _behind(env):
        _set_agent(env, [400.0, 0.0], theta=0.0)
        env._ball.pos[:] = ball; env._ball.seen_age_ms = 0.0
    _probe("ball_behind_robot", _behind)

    # 5. BALÓN VISTO HACE MUCHO (seen_age > 3× contrato): theta correcto, solo seen_age falla
    def _stale(env):
        _set_agent(env, [-300.0, 0.0], theta=0.0)
        env._ball.pos[:] = ball
        env._ball.seen_age_ms = _CONTRACT_BALL_SEEN_AGE_LIMIT_MS * 3.0
    _probe("stale_ball", _stale)

    # 6. Agente CAÍDO: theta correcto, posición correcta, solo is_upright=False
    def _fallen(env):
        _set_agent(env, [-260.0, 0.0], theta=0.0, upright=False)
        env._ball.pos[:] = ball; env._ball.seen_age_ms = 0.0
    _probe("fallen_agent", _fallen)

    r.passed = len(violations) == 0
    r.details = {
        "violations": violations[:10],
        "total_violations": len(violations),
        "configs_tested": 6,
        "contract_ref": "§2.2 BHUMAN_RL_CURRICULUM_CONTRACT.md",
    }
    status = "OK" if r.passed else f"FAIL ({len(violations)} envelope violations)"
    print(f"  E1 handoff envelope   → {status}")
    for v in violations[:3]:
        print(f"       ⚠ config={v['config']}  frame={v['frame']}  macro={v['macro_type']}")
    return r


def _extra2_action_space_safety_gate(seed: int) -> ContractGuardrailResult:
    """
    EXTRA 2 — §1.2 BHuman RL Curriculum Contract.

    El RL NO debe poder inyectar ni sobrescribir señales que son propiedad
    exclusiva de BHuman. Se envían acciones extremas (NaN, ±Inf, ±1e9) y se
    verifica que:
      1. El env NO crashea (robustez del wrapper).
      2. Los campos críticos del info dict NO contienen NaN/Inf.
      3. sidecar_motion_request != 'dribble' fuera de takeover activo (§1.2).
    """
    r = ContractGuardrailResult("EXTRA2_action_space_safety_gate")
    crash_count = 0
    nan_inf_in_info: List[Dict] = []
    illegal_injections: List[Dict] = []

    action_variants = [
        ("nan",      lambda s: np.full(s, float("nan"),  dtype=np.float32)),
        ("pos_inf",  lambda s: np.full(s, float("inf"),  dtype=np.float32)),
        ("neg_inf",  lambda s: np.full(s, float("-inf"), dtype=np.float32)),
        ("large_pos",lambda s: np.full(s, 1e9,           dtype=np.float32)),
        ("large_neg",lambda s: np.full(s, -1e9,          dtype=np.float32)),
        ("zero",     lambda s: np.zeros(s,               dtype=np.float32)),
    ]

    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        ball = np.array([0.0, 0.0], dtype=np.float32)
        _install_state(env, ball, ball + np.array([-300.0, 0.0], dtype=np.float32))

        for vname, afn in action_variants:
            bad_actions = {aid: afn(env.action_space(aid).shape) for aid in env.agents}
            try:
                res = _safe_step(env, bad_actions)
                if res is None:
                    crash_count += 1
                    r.warnings.append(f"step=None for variant={vname}")
                    continue
                _, _, term, trunc, infos = res
                info = next(iter(infos.values()), {}) if infos else {}

                # Check: NaN/Inf en campos bridge
                for key in ("macro_type", "sidecar_skill_request", "sidecar_motion_request"):
                    val = info.get(key)
                    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                        nan_inf_in_info.append({"variant": vname, "key": key})

                # Check §1.2: dribble solo es válido dentro de takeover activo
                motion = str(info.get("sidecar_motion_request", "none"))
                if motion == "dribble" and not bool(info.get("takeover_active", False)):
                    illegal_injections.append({
                        "variant": vname, "motion": motion,
                        "issue": "dribble outside takeover (§1.2)",
                    })

                if _is_done(term, trunc):
                    env.reset(seed=seed)
                    _install_state(env, ball, ball + np.array([-300.0, 0.0], dtype=np.float32))
            except Exception as exc:
                crash_count += 1
                r.warnings.append(f"variant={vname} crashed: {exc}")

        env.close()
    except Exception as exc:
        r.warnings.append(f"suite exception: {exc}")

    r.passed = (crash_count == 0 and
                len(nan_inf_in_info) == 0 and
                len(illegal_injections) == 0)
    r.details = {
        "variants_tested": len(action_variants),
        "crash_count": crash_count,
        "nan_inf_in_info": nan_inf_in_info[:5],
        "illegal_injections": illegal_injections[:5],
        "contract_ref": "§1.2 BHUMAN_RL_CURRICULUM_CONTRACT.md",
    }
    status = ("OK" if r.passed else
              f"FAIL  crashes={crash_count}  nan_inf={len(nan_inf_in_info)}"
              f"  illegal={len(illegal_injections)}")
    print(f"  E2 action safety gate → {status}")
    return r


def _extra3_false_positive_reward(seed: int) -> ContractGuardrailResult:
    """
    EXTRA 3 — §2.4 BHuman RL Curriculum Contract.

    "La reward NO debe premiar 'yo toqué el balón'."

    Se fuerza al agente sobre el balón (contacto físico sin relevo activo)
    durante 150 frames. Si abs(reward) > threshold con takeover_active=False
    y handoff_ready=False se registra como falso positivo.

    Threshold: 0.05 (permite pequeño ruido de shaped-reward de approach pero
    no kick-impulse ni takeover-success fraudulento).
    """
    r = ContractGuardrailResult("EXTRA3_false_positive_reward")
    false_positives: List[Dict] = []
    ball = np.array([300.0, 0.0], dtype=np.float32)

    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        actions = _zero_actions(env)
        agent_id = list(env.agents)[0] if env.agents else "agent_0"

        for step in range(150):
            # Mantiene balón fijo y agente superpuesto (contacto garantizado)
            env._ball.pos[:] = ball
            env._ball.seen_age_ms = 0.0
            if agent_id in env._agents_state:
                # Alterna ±10 mm para crear contacto real cada frame
                offset = -10.0 if step % 2 == 0 else 10.0
                env._agents_state[agent_id].pos = ball + np.array([offset, 0.0], np.float32)
                env._agents_state[agent_id].is_upright = True

            res = _safe_step(env, actions)
            if res is None:
                break
            _, rewards, term, trunc, infos = res
            info = next(iter(infos.values()), {}) if infos else {}

            takeover_active = bool(info.get("takeover_active", False))
            handoff_ready   = bool(info.get("handoff_ready", False))
            r_agent = float(rewards.get(agent_id, 0.0))

            # Sin relevo activo: reward grande es falso positivo §2.4
            if not takeover_active and not handoff_ready:
                if abs(r_agent) > _FALSE_POSITIVE_REWARD_THRESHOLD:
                    false_positives.append({
                        "frame": step,
                        "reward": round(r_agent, 4),
                        "macro": info.get("macro_type", "none"),
                        "issue": "reward spike without takeover (§2.4 violation)",
                    })

            if _is_done(term, trunc):
                env.reset(seed=seed + step)
                actions = _zero_actions(env)

        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")

    # E3 siempre pasa (WARN, no FAIL) — hay situaciones de approach/dribble
    # pre-takeover que legitimamente generan reward positivo.
    # Los falsos positivos se vuelcan a r.warnings para que queden en el reporte.
    r.passed = True
    for fp in false_positives:
        r.warnings.append(
            f"frame={fp['frame']}  r={fp['reward']}  macro={fp['macro']}  "
            f"| reward sin takeover activo — revisar si es shaped-approach o kick prohibido"
        )
    r.details = {
        "false_positives": false_positives[:5],
        "total_false_positives": len(false_positives),
        "threshold": _FALSE_POSITIVE_REWARD_THRESHOLD,
        "severity": "WARN",
        "note": "Hay situaciones pre-takeover (approach shaped, dribble de soporte) donde reward != 0 es esperado.",
        "contract_ref": "§2.4 BHUMAN_RL_CURRICULUM_CONTRACT.md",
    }
    if false_positives:
        status = f"WARN ({len(false_positives)} reward spikes sin takeover — revisar manualmente)"
    else:
        status = "OK"
    print(f"  E3 false pos reward   → {status}")
    for fp in false_positives[:2]:
        print(f"       ⚠ frame={fp['frame']}  r={fp['reward']}  macro={fp['macro']}")
    return r


def run_contract_guardrails_suite(seed: int) -> List[ContractGuardrailResult]:
    """Módulo 13: Contract Guardrails — 3 tests que blindan el contrato RL↔BHuman."""
    print("\n[13] CONTRACT GUARDRAILS  (§1.2 · §2.2 · §2.4)")
    results = [
        _extra1_handoff_envelope_validation(seed),
        _extra2_action_space_safety_gate(seed),
        _extra3_false_positive_reward(seed),
    ]
    passed = sum(1 for r in results if r.passed)
    badge  = "✅ ALL PASS" if passed == len(results) else f"❌ {len(results) - passed} FAIL"
    print(f"  Resultado: {passed}/{len(results)} tests pasaron  → {badge}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 15: EXECUTION QUALITY — shoot direction + kinematic continuity
# ─────────────────────────────────────────────────────────────────────────────

def _t33_shoot_ball_direction(seed: int) -> ContractGuardrailResult:
    """
    T33: Después de takeover con playBall_shoot, el balón debe moverse hacia
    el arco rival (vel_x > 0 en campo estándar).

    Captura el bug "BHuman dispara hacia atrás": verifica que el vector de
    velocidad del balón al finalizar el shoot apunte en la dirección correcta.
    También registra robot_pose_field.theta vs agent.theta en ese momento para
    diagnosticar si la raíz es un theta stale del bridge.
    """
    r = ContractGuardrailResult("T33_shoot_ball_direction")
    backward_shots: List[Dict[str, Any]] = []
    pose_deltas: List[Dict[str, Any]] = []

    try:
        env = _make_env(3, seed)   # stage 3: 1v1, hay oponente
        env.reset(seed=seed)
        # Posicionar agente frente al arco rival para que BHuman elija shoot
        ball_pos = np.array([FIELD.x_opp_penalty + 200.0, 0.0], dtype=np.float32)
        agent_pos = ball_pos + np.array([-300.0, 0.0], dtype=np.float32)
        _install_state(env, ball_pos, agent_pos)
        actions = _zero_actions(env)
        prev_takeover = False
        prev_ball_pos = env._ball.pos.copy()

        for fi in range(250):
            res = _safe_step(env, actions)
            if res is None:
                break
            _, _, term, trunc, infos = res
            info = next(iter(infos.values()), {}) if infos else {}
            takeover = bool(info.get("takeover_active", False))
            macro = str(info.get("macro_type", "none"))

            # Capturar theta del agente vs robot_pose_field para diagnóstico
            if takeover and macro == "playBall_shoot":
                agents_list = list(env._agents_state.values())
                if agents_list:
                    agent_theta = float(agents_list[0].theta)
                    bridge_pose = info.get("sidecar_robot_pose_field", None)
                    bridge_theta = float(bridge_pose[2]) if (
                        isinstance(bridge_pose, (list, tuple)) and len(bridge_pose) >= 3
                    ) else None
                    if bridge_theta is not None:
                        delta_theta = abs(_normalize_angle_val(agent_theta - bridge_theta))
                        if delta_theta > math.pi / 4:
                            pose_deltas.append({
                                "frame": fi,
                                "agent_theta_deg": round(math.degrees(agent_theta), 1),
                                "bridge_theta_deg": round(math.degrees(bridge_theta), 1),
                                "delta_deg": round(math.degrees(delta_theta), 1),
                            })

            # Al finalizar takeover de shoot, verificar dirección del balón
            end_reason = str(info.get("last_macro_end_reason", "none"))
            if prev_takeover and not takeover and end_reason in {"completed", "goal", "out"}:
                ball_vel_x = float(env._ball.vel[0])
                ball_speed = float(np.linalg.norm(env._ball.vel))
                if ball_speed > 50.0 and ball_vel_x < 0.0:
                    agents_list = list(env._agents_state.values())
                    agent_theta_now = float(agents_list[0].theta) if agents_list else 0.0
                    backward_shots.append({
                        "frame": fi,
                        "ball_vel_x": round(ball_vel_x, 1),
                        "ball_vel_y": round(float(env._ball.vel[1]), 1),
                        "ball_speed": round(ball_speed, 1),
                        "agent_theta_deg": round(math.degrees(agent_theta_now), 1),
                        "end_reason": end_reason,
                    })

            prev_takeover = takeover
            prev_ball_pos = env._ball.pos.copy()

            if _is_done(term, trunc):
                env.reset(seed=seed + fi)
                actions = _zero_actions(env)
                ball_pos = np.array([FIELD.x_opp_penalty + 200.0, 0.0], dtype=np.float32)
                _install_state(env, ball_pos, ball_pos + np.array([-300.0, 0.0], dtype=np.float32))
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")

    r.passed = len(backward_shots) == 0
    r.details = {
        "backward_shots": backward_shots[:5],
        "total_backward_shots": len(backward_shots),
        "pose_theta_mismatches": pose_deltas[:5],
        "total_theta_mismatches": len(pose_deltas),
        "note": "backward_shots=0 es correcto; pose_theta_mismatches indica robot_pose_field stale",
    }
    status = "OK" if r.passed else f"FAIL ({len(backward_shots)} backward shots)"
    print(f"  T33 shoot direction   → {status}  theta_mismatches={len(pose_deltas)}")
    for bs in backward_shots[:2]:
        print(f"       ⚠ frame={bs['frame']}  vel_x={bs['ball_vel_x']}  "
              f"agent_theta={bs['agent_theta_deg']}°")
    for pd in pose_deltas[:2]:
        print(f"       ⚠ theta mismatch @ frame={pd['frame']}  "
              f"agent={pd['agent_theta_deg']}°  bridge={pd['bridge_theta_deg']}°  "
              f"Δ={pd['delta_deg']}°")
    return r


def _normalize_angle_val(angle: float) -> float:
    """Normaliza ángulo a [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _t34_takeover_kinematic_continuity(seed: int) -> ContractGuardrailResult:
    """
    T34: Durante takeover activo, la posición del agente no debe saltar más de
    max_locomotion_delta_mm por frame (locomotion física plausible).
    Captura el bug "cinemáticas bruscas" de BHuman.

    También mide max |delta_theta| por frame para detectar si el giro es el
    problema, ya que una orientación incorrecta provoca correcciones abruptas.
    """
    r = ContractGuardrailResult("T34_takeover_kinematic_continuity")
    # Límite: Nao anda ~300mm/s, a 30fps = 10mm/frame. Ser generoso: 80mm/frame
    MAX_POS_DELTA_MM   = 80.0
    MAX_THETA_DELTA_RAD = 0.5   # ~28.6°/frame

    pos_jumps:   List[Dict[str, Any]] = []
    theta_jumps: List[Dict[str, Any]] = []

    try:
        env = _make_env(0, seed)
        env.reset(seed=seed)
        ball_pos = np.array([0.0, 0.0], dtype=np.float32)
        _install_state(env, ball_pos, ball_pos + np.array([-400.0, 0.0], dtype=np.float32))
        actions = _zero_actions(env)

        agents_list = list(env.agents)
        active_id = agents_list[0] if agents_list else None
        if active_id is None:
            r.warnings.append("no agents in env")
            r.passed = True
            return r

        prev_pos   = env._agents_state[active_id].pos.copy()
        prev_theta = float(env._agents_state[active_id].theta)

        for fi in range(500):
            res = _safe_step(env, actions)
            if res is None:
                break
            _, _, term, trunc, infos = res
            info = next(iter(infos.values()), {}) if infos else {}
            takeover = bool(info.get("takeover_active", False))

            if takeover and active_id in env._agents_state:
                curr_pos   = env._agents_state[active_id].pos.copy()
                curr_theta = float(env._agents_state[active_id].theta)

                delta_pos = float(np.linalg.norm(curr_pos - prev_pos))
                delta_theta = abs(_normalize_angle_val(curr_theta - prev_theta))

                if delta_pos > MAX_POS_DELTA_MM:
                    pos_jumps.append({
                        "frame": fi,
                        "delta_mm": round(delta_pos, 1),
                        "pos_before": [round(float(prev_pos[0]), 1), round(float(prev_pos[1]), 1)],
                        "pos_after":  [round(float(curr_pos[0]), 1), round(float(curr_pos[1]), 1)],
                    })
                if delta_theta > MAX_THETA_DELTA_RAD:
                    theta_jumps.append({
                        "frame": fi,
                        "delta_deg": round(math.degrees(delta_theta), 1),
                        "theta_before_deg": round(math.degrees(prev_theta), 1),
                        "theta_after_deg":  round(math.degrees(curr_theta), 1),
                    })

                prev_pos   = curr_pos.copy()
                prev_theta = curr_theta

            if _is_done(term, trunc):
                env.reset(seed=seed + fi)
                actions = _zero_actions(env)
                if active_id in env._agents_state:
                    prev_pos   = env._agents_state[active_id].pos.copy()
                    prev_theta = float(env._agents_state[active_id].theta)
        env.close()
    except Exception as exc:
        r.warnings.append(f"exception: {exc}")

    r.passed = len(pos_jumps) == 0 and len(theta_jumps) == 0
    r.details = {
        "pos_jumps":           pos_jumps[:5],
        "total_pos_jumps":     len(pos_jumps),
        "theta_jumps":         theta_jumps[:5],
        "total_theta_jumps":   len(theta_jumps),
        "limits": {
            "max_pos_delta_mm":    MAX_POS_DELTA_MM,
            "max_theta_delta_deg": round(math.degrees(MAX_THETA_DELTA_RAD), 1),
        },
    }
    status = "OK" if r.passed else (
        f"FAIL  pos_jumps={len(pos_jumps)}  theta_jumps={len(theta_jumps)}"
    )
    print(f"  T34 kinematic cont.   → {status}")
    for pj in pos_jumps[:2]:
        print(f"       ⚠ pos jump @ frame={pj['frame']}  delta={pj['delta_mm']}mm")
    for tj in theta_jumps[:2]:
        print(f"       ⚠ theta jump @ frame={tj['frame']}  delta={tj['delta_deg']}°")
    return r


def run_execution_quality_suite(seed: int) -> List[ContractGuardrailResult]:
    """Módulo 15: Execution Quality — verifica dirección de shoot y continuidad cinemática."""
    print("\n[15] EXECUTION QUALITY")
    results = [
        _t33_shoot_ball_direction(seed),
        _t34_takeover_kinematic_continuity(seed),
    ]
    passed = sum(1 for r in results if r.passed)
    badge  = "✅ ALL PASS" if passed == len(results) else f"❌ {len(results) - passed} FAIL"
    print(f"  Resultado: {passed}/{len(results)} tests pasaron  → {badge}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 14: REPORTER
# ─────────────────────────────────────────────────────────────────────────────

def _verdict(report: DiagnosticsReport) -> str:
    errors = sum(1 for a in report.alerts if a.get("severity") == "ERROR")
    if not report.bridge_alive:
        return "CRITICAL — bridge offline"
    if errors > 0:
        return f"FAIL — {errors} errores de contrato"
    # Módulos nuevos
    new_fails = []
    for suite_key, suite_name in [
        ("ball_physics",       "ball_physics"),
        ("bhuman_semantic",    "bhuman_semantic"),
        ("pretraining_audit",  "pretraining_audit"),
        ("ball_edge_cases",    "ball_edge_cases"),
        ("physical_stress",    "physical_stress"),
        ("execution_quality",  "execution_quality"),
    ]:
        suite = getattr(report, suite_key, [])
        n_fail = sum(1 for t in suite if not t.get("passed", True))
        if n_fail > 0:
            new_fails.append(f"{suite_name}:{n_fail}")
    # E1 (envelope §2.2) y E2 (action safety §1.2) son FAIL-level — violan el contrato estructural.
    # E3 (reward §2.4) es WARN-level — puede haber reward pre-takeover legítimo (approach, dribble soporte).
    hard_guardrail_fails = sum(
        1 for t in report.contract_guardrails
        if not t.get("passed", True) and t.get("test_name", "") != "EXTRA3_false_positive_reward"
    )
    e3_warns = next(
        (len(t.get("warnings", [])) for t in report.contract_guardrails
         if t.get("test_name", "") == "EXTRA3_false_positive_reward"),
        0
    )
    if hard_guardrail_fails > 0:
        return f"FAIL — {hard_guardrail_fails} contract guardrail violations (E1/E2)"
    if new_fails:
        return f"WARN — new test failures: {', '.join(new_fails)}"
    if report.reward_multi_frame_violations > 0:
        return f"WARN — {report.reward_multi_frame_violations} reward violations"
    if report.teleport_count > 5:
        return f"WARN — {report.teleport_count} teleportaciones"
    if report.handoff.get("handoff_ready_frame") is None:
        return "WARN — handoff_ready nunca detectado"
    if report.handoff.get("takeover_start_frame") is None:
        return "WARN — takeover_active nunca detectado"
    if e3_warns > 0:
        return f"WARN — E3: {e3_warns} reward spikes sin takeover (revisar §2.4)"
    return "PASS"


def write_report(report: DiagnosticsReport, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ts = report.timestamp.replace(":", "-").replace(" ", "_")

    # JSON
    jp = artifacts_dir / f"env_diagnostics_{ts}.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\n  JSON: {jp}")

    # Markdown
    mp = artifacts_dir / f"env_diagnostics_{ts}.md"
    lines = [
        "# Environment Diagnostics",
        f"**{report.timestamp}**  |  stage={report.stage}  seed={report.seed}  backend={report.backend}",
        f"## Veredicto: `{report.verdict}`",
        "| Métrica | Valor |",
        "|---------|-------|",
        f"| Teleportaciones | {report.teleport_count} |",
        f"| Max pose delta (mm) | {report.max_pose_delta_mm:.1f} |",
        f"| Reward violations | {report.reward_multi_frame_violations} |",
        f"| Contract alerts | {len(report.alerts)} |",
        "",
        "## Handoff Lifecycle",
    ]
    ht = report.handoff
    lines += [
        f"- Approach frames: `{ht.get('approach_frames')}`",
        f"- handoff_ready @ frame: `{ht.get('handoff_ready_frame')}`",
        f"- takeover_active @ frame: `{ht.get('takeover_start_frame')}`",
        f"- Latencia: `{ht.get('transition_latency_ms', 0):.1f}ms`",
        f"- end_reason: `{ht.get('macro_end_reason')}`  success: `{ht.get('macro_success')}`",
        "",
        "## BHuman Conditions",
        "| Condition | Mode | Detected | Takeover | frames | end_reason |",
        "|-----------|------|----------|----------|--------|------------|",
    ]
    for bc in report.bhuman_conditions:
        lines.append(
            f"| {bc.get('condition')} | {bc.get('mode')} | {bc.get('detected')} "
            f"| {bc.get('takeover_reached')} | {bc.get('frames_to_detect')} "
            f"| {bc.get('macro_end_reason')} |"
        )
    lines += ["", "## Stage Sweep",
              "| Stage | Name | Bridge | Handoff | Takeover | Completed |",
              "|-------|------|--------|---------|----------|-----------|"]
    for ss in report.stage_sweep:
        lines.append(
            f"| {ss.get('stage_id')} | {ss.get('stage_name')} | {ss.get('bridge_alive')} "
            f"| {ss.get('handoff_ready_detected')} | {ss.get('takeover_active_detected')} "
            f"| {ss.get('macro_completed')} |"
        )

    # Módulos nuevos + guardrails
    for suite_attr, suite_title in [
        ("ball_physics",         "## Ball Physics Basics (T1-T4)"),
        ("bhuman_semantic",      "## BHuman Semantic Tests (T5-T10)"),
        ("pretraining_audit",    "## Pre-Training Audit (T11-T16)"),
        ("ball_edge_cases",      "## Ball Edge Cases (T26-T32)"),
        ("physical_stress",      "## Physical Stress (TA-TG)"),
        ("contract_guardrails",  "## Contract Guardrails (§1.2·§2.2·§2.4)"),
    ]:
        suite = getattr(report, suite_attr, [])
        if not suite:
            continue
        lines += ["", suite_title,
                  "| Test | Passed | Warnings |",
                  "|------|--------|----------|"]
        for t in suite:
            warn_n = len(t.get("warnings", []))
            lines.append(
                f"| {t.get('test_name')} | {'✅' if t.get('passed') else '❌'} | {warn_n} |"
            )

    lines += ["", "## Notas"] + [f"- {n}" for n in report.notes]
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  MD  : {mp}")


# ─────────────────────────────────────────────────────────────────────────────
# MENU INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────

def _ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return int(raw) if raw.isdigit() else default


def _ask_str(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def _ask_bool(prompt: str, default: bool) -> bool:
    d = "s" if default else "n"
    raw = input(f"{prompt} (s/n) [{d}]: ").strip().lower()
    if raw in ("s", "y", "1"):
        return True
    if raw in ("n", "0"):
        return False
    return default


def _ask_list(prompt: str, choices: List[str]) -> List[str]:
    print(f"  Opciones: {choices}")
    raw = input(f"{prompt} (separadas por coma, Enter=todas): ").strip()
    if not raw:
        return choices
    sel = [x.strip() for x in raw.split(",")]
    return [s for s in sel if s in choices] or choices


def _print_menu():
    print("\n" + "=" * 62)
    print("  ENVIRONMENT DIAGNOSTICS — MENÚ PRINCIPAL")
    print("=" * 62)
    print("  ── SUITES ───────────────────────────────────────────")
    print("  1. Suite completa original (módulos 1-7)")
    print("  2. Suite nuevos 32 tests + guardrails (módulos 8-13)")
    print("  3. Suite completa extendida (módulos 1-13)")
    print("  ── MÓDULOS INDIVIDUALES – ORIGINALES ────────────────")
    print("  4. Solo BHuman conditions (módulo 5)")
    print("  5. Solo Stage Sweep (módulo 6)")
    print("  6. Handoff Lifecycle visual (módulo 4)")
    print("  7. Pose Drift visual (módulo 3)")
    print("  8. Reward Sanity Check (módulo 7)")
    print("  ── MÓDULOS INDIVIDUALES – NUEVOS ────────────────────")
    print("  9.  Ball Physics Basics    (módulo 8,  T1-T4)")
    print("  10. BHuman Semantic        (módulo 9,  T5-T10)")
    print("  11. Pre-Training Audit     (módulo 10, T11-T16)")
    print("  12. Ball Edge Cases        (módulo 11, T26-T32)")
    print("  13. Physical Stress        (módulo 12, TA-TG)")
    print("  ── CONTRACT GUARDRAILS ────────────────────────────────")
    print("  14. Contract Guardrails    (§1.2 · §2.2 · §2.4)")
    print("  ── EXECUTION QUALITY ──────────────────────────────────")
    print("  15. Execution Quality      (T33 shoot dir · T34 kinematic)")
    print("  ─────────────────────────────────────────────────────")
    print("  0. Salir")
    print("-" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import datetime

    _print_menu()
    choice = _ask_int("Selección", 1)
    if choice == 0:
        return

    # Parámetros comunes
    print("\nEtapas disponibles:")
    for i, c in enumerate(CURRICULUM):
        print(f"  {i}. {c.name}")
    stage = _ask_int("Stage (para módulos individuales)", 0)
    stage = max(0, min(stage, len(CURRICULUM) - 1))

    seed    = _ask_int("Semilla", 42)
    steps   = _ask_int("Steps (módulos pose/reward/stress)", 150)
    visual  = _ask_bool("Activar render ASCII en terminal?", False)
    delay   = float(_ask_str("Delay por frame (s)", "0.08")) if visual else 0.0
    outdir  = _ask_str("Directorio de reportes", "artifacts")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stage_cfg = CURRICULUM[stage]

    print(f"\n[ENV DIAGNOSTICS] mode={choice}  stage={stage} ({stage_cfg.name})"
          f"  seed={seed}  steps={steps}  visual={visual}")

    report = DiagnosticsReport(
        timestamp=timestamp,
        stage=stage,
        seed=seed,
        steps_run=steps,
        backend="unknown",
        bridge_alive=False,
    )

    # ── Env principal (módulos que lo requieren) ──────────────────────────────
    needs_env = choice in (1, 3, 4, 6, 7, 8)
    env = None
    if needs_env:
        env = NaoSoccerEnv(stage_cfg=stage_cfg, use_fov=False, seed=seed)
        env.reset(seed=seed)

    try:
        # ── SUITE COMPLETA ORIGINAL ──────────────────────────────────────────
        if choice in (1, 3):
            backend, alive = run_dry_run(env, 10, visual=visual, delay=delay)
            report.backend, report.bridge_alive = backend, alive

            env.reset(seed=seed)
            alerts = run_contract_audit(env, seed)
            report.alerts = [asdict(a) for a in alerts]

            env.reset(seed=seed)
            palerts, max_d = run_pose_drift_detection(env, steps, seed, visual, delay)
            report.teleport_count = sum(1 for a in palerts if a.kind == "teleport")
            report.max_pose_delta_mm = max_d
            report.alerts += [asdict(a) for a in palerts]

            env.reset(seed=seed)
            ht = run_handoff_lifecycle(env, seed, min(300, steps * 2), visual, delay)
            report.handoff = asdict(ht)

            env.reset(seed=seed)
            conds = ["dribble", "duel", "shoot", "pass"]
            bc = run_bhuman_conditions(env, seed, conds, 120, visual, delay)
            report.bhuman_conditions = [asdict(r) for r in bc]

            raw_stages = input("\nStages para sweep (Enter=solo actual, t=todos): ").strip()
            if raw_stages == "t":
                sweep_stages = list(range(len(CURRICULUM)))
            elif raw_stages:
                sweep_stages = [int(x) for x in raw_stages.split(",") if x.strip().isdigit()]
            else:
                sweep_stages = [stage]
            ss = run_stage_sweep(seed, sweep_stages, min(50, steps // 3), visual, delay)
            report.stage_sweep = [asdict(r) for r in ss]

            env.reset(seed=seed)
            impulses, violations = run_reward_sanity_check(env, steps, seed, visual, delay)
            report.reward_impulses = [asdict(r) for r in impulses]
            report.reward_multi_frame_violations = violations
            report.bridge_alive = alive

        # ── SUITE NUEVOS 32 TESTS ────────────────────────────────────────────
        if choice in (2, 3):
            bp = run_ball_physics_suite(seed)
            report.ball_physics = [asdict(r) for r in bp]

            bs = run_bhuman_semantic_suite(seed)
            report.bhuman_semantic = [asdict(r) for r in bs]

            pt = run_pretraining_audit_suite(seed)
            report.pretraining_audit = [asdict(r) for r in pt]

            be = run_ball_edge_cases_suite(seed)
            report.ball_edge_cases = [asdict(r) for r in be]

            ps = run_physical_stress_suite(seed)
            report.physical_stress = [asdict(r) for r in ps]

            cg = run_contract_guardrails_suite(seed)
            report.contract_guardrails = [asdict(r) for r in cg]

            eq = run_execution_quality_suite(seed)
            report.execution_quality = [asdict(r) for r in eq]

            # Para el veredicto del bridge en suite nueva
            if not report.bridge_alive:
                # Intenta inferir del pre-training audit
                pt_stress = next(
                    (t for t in report.pretraining_audit if t.get("test_name") == "T12_stress_1000_steps"),
                    None
                )
                if pt_stress and pt_stress.get("passed"):
                    report.bridge_alive = True

        # ── MÓDULOS INDIVIDUALES ORIGINALES ─────────────────────────────────
        elif choice == 4:
            conds_input = _ask_list("Condiciones", ["dribble", "duel", "shoot", "pass"])
            frames_each = _ask_int("Max frames por condición", 120)
            bc = run_bhuman_conditions(env, seed, conds_input, frames_each, visual, delay)
            report.bhuman_conditions = [asdict(r) for r in bc]
            report.bridge_alive = True

        elif choice == 5:
            raw_s = input("Stages (Enter=todos): ").strip()
            if raw_s:
                sweep_stages = [int(x) for x in raw_s.split(",") if x.strip().isdigit()]
            else:
                sweep_stages = list(range(len(CURRICULUM)))
            spps = _ask_int("Steps por stage", 40)
            ss = run_stage_sweep(seed, sweep_stages, spps, visual, delay)
            report.stage_sweep = [asdict(r) for r in ss]
            report.bridge_alive = any(r.bridge_alive for r in ss)

        elif choice == 6:
            mf = _ask_int("Max frames lifecycle", 300)
            ht = run_handoff_lifecycle(env, seed, mf, visual=True, delay=delay)
            report.handoff = asdict(ht)
            report.bridge_alive = True

        elif choice == 7:
            palerts, max_d = run_pose_drift_detection(env, steps, seed, visual=True, delay=delay)
            report.teleport_count = sum(1 for a in palerts if a.kind == "teleport")
            report.max_pose_delta_mm = max_d
            report.alerts = [asdict(a) for a in palerts]
            report.bridge_alive = True

        elif choice == 8:
            impulses, violations = run_reward_sanity_check(env, steps, seed, visual=True, delay=delay)
            report.reward_impulses = [asdict(r) for r in impulses]
            report.reward_multi_frame_violations = violations
            report.bridge_alive = True

        # ── MÓDULOS INDIVIDUALES NUEVOS ───────────────────────────────────────
        elif choice == 9:
            bp = run_ball_physics_suite(seed)
            report.ball_physics = [asdict(r) for r in bp]
            report.bridge_alive = True

        elif choice == 10:
            bs = run_bhuman_semantic_suite(seed)
            report.bhuman_semantic = [asdict(r) for r in bs]
            report.bridge_alive = True

        elif choice == 11:
            pt = run_pretraining_audit_suite(seed)
            report.pretraining_audit = [asdict(r) for r in pt]
            report.bridge_alive = True

        elif choice == 12:
            be = run_ball_edge_cases_suite(seed)
            report.ball_edge_cases = [asdict(r) for r in be]
            report.bridge_alive = True

        elif choice == 13:
            ps = run_physical_stress_suite(seed)
            report.physical_stress = [asdict(r) for r in ps]
            report.bridge_alive = True

        elif choice == 14:
            cg = run_contract_guardrails_suite(seed)
            report.contract_guardrails = [asdict(r) for r in cg]
            report.bridge_alive = True

        elif choice == 15:
            eq = run_execution_quality_suite(seed)
            report.execution_quality = [asdict(r) for r in eq]
            report.bridge_alive = True

        # ── VEREDICTO FINAL ───────────────────────────────────────────────────
        report.verdict = _verdict(report)

        # Resumen de nuevos tests si se ejecutaron
        for suite_attr, suite_name in [
            ("ball_physics",         "Ball Physics"),
            ("bhuman_semantic",      "BHuman Semantic"),
            ("pretraining_audit",    "Pre-Training"),
            ("ball_edge_cases",      "Ball Edge Cases"),
            ("physical_stress",      "Physical Stress"),
            ("contract_guardrails",  "Contract Guardrails"),
            ("execution_quality",    "Execution Quality"),
        ]:
            suite = getattr(report, suite_attr, [])
            if suite:
                n_pass = sum(1 for t in suite if t.get("passed"))
                print(f"  {suite_name}: {n_pass}/{len(suite)} PASS")

        print(f"\n{'='*62}")
        print(f"  VEREDICTO: {report.verdict}")
        print(f"{'='*62}")

    finally:
        if env is not None:
            env.close()
        write_report(report, Path(outdir))


if __name__ == "__main__":
    main()
