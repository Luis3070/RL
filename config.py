"""
config.py — Fuente de verdad para MAPPO NAO Soccer, alineada con contrato BHuman 4v4 Scaled.

Consume bhuman_cfg_parser.AllConfigs como fuente de verdad de campo, tácticas y parámetros.
NO usa física local del balón como fuente de verdad (kick/dribble los ejecuta BHuman).
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import List, Optional
import math
import os

from bhuman_cfg_parser import (
    load_contract, AllConfigs,
    VALID_TACTIC_NAMES, VALID_STRATEGY_NAME,
)
from runtime_paths import resolve_cfg_root as _shared_resolve_cfg_root


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION MODE
# ─────────────────────────────────────────────────────────────────────────────
#
# EXECUTION_MODE is the REQUESTED mode, not necessarily the effective backend.
#
# "python_stub_smoke_only" — Python stub for smoke tests / debug only.
#                            BHumanTakeoverSimulator allowed.
#
# "bhuman_sidecar"         — Attempts real C++ backend. If unavailable,
#                            falls to "contract_validation" mode:
#                            uses PythonContractStub with explicit marking.
#                            BHumanTakeoverSimulator NOT used.
#                            Results are contractual validation, NOT real
#                            BHuman in loop. Do not treat takeover_success
#                            as real BHuman execution evidence.
#
# The EFFECTIVE backend is reported by the sidecar at runtime:
#   sidecar.backend_label -> "real" | "contract_validation" | "python_stub"
#   sidecar.is_real_backend -> True/False
#
# "bhuman_sidecar" first attempts the real backend. If the bridge is not
# available, it falls back explicitly to contract_validation mode.

EXECUTION_MODE: str = os.environ.get("RL_EXECUTION_MODE", "bhuman_sidecar")


# ─────────────────────────────────────────────────────────────────────────────
# CONTRATO BHUMAN — Cargado del parser
# cfg_root resolution: 1) .env override  2) repo-relative sibling path
# ─────────────────────────────────────────────────────────────────────────────

def resolve_cfg_root() -> str | None:
    """Resolve the shared BHuman CFG root for runtime and smoke tests."""
    return _shared_resolve_cfg_root()


# Keep backward compat alias
_resolve_cfg_root = resolve_cfg_root

CONTRACT: AllConfigs = load_contract(cfg_root=resolve_cfg_root())


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONES DEL CAMPO  (mm — 4v4 Scaled, derivado del contrato)
# Fuente: bhuman_cfg_parser.FieldConfig4v4 (auditoría 29/29 símbolos ✅)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FieldConfig:
    """Dimensiones del campo 4v4 Scaled en milímetros. Derivado del contrato."""
    x_opponent_ground: float = 2813.0    # xPosOpponentGroundLine (4v4 Scaled)
    x_own_ground:      float = -2813.0   # xPosOwnGroundLine
    y_left_sideline:   float = 1875.0    # yPosLeftSideline
    y_right_sideline:  float = -1875.0   # yPosRightSideline
    y_left_goal:       float = 800.0     # yPosLeftGoal
    y_right_goal:      float = -800.0    # yPosRightGoal
    goal_depth:        float = 600.0     # profundidad del arco
    center_circle_r:   float = 469.0     # centerCircleRadius (4v4 Scaled)
    # Zonas de penalti (4v4 Scaled)
    x_opp_penalty:     float = 1781.0    # xPosOpponentPenaltyArea
    x_own_penalty:     float = -1781.0   # xPosOwnPenaltyArea

    @property
    def length(self) -> float:
        return self.x_opponent_ground - self.x_own_ground   # 5626 mm

    @property
    def width(self) -> float:
        return self.y_left_sideline - self.y_right_sideline  # 3750 mm

    @property
    def goal_width(self) -> float:
        return self.y_left_goal - self.y_right_goal          # 1600 mm

    @property
    def diagonal(self) -> float:
        return math.hypot(self.length, self.width)           # ~6759 mm


FIELD = FieldConfig()


# ─────────────────────────────────────────────────────────────────────────────
# MODELO DE MOVIMIENTO DEL NAO
# Fuente: contrato WalkConfig (walkingEngine.cfg, valores conservadores)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentConfig:
    """Parámetros físicos del robot NAO V6."""
    max_forward_speed:  float = 250.0   # mm/s
    max_sideways_speed: float = 200.0   # mm/s
    max_backward_speed: float = 150.0   # mm/s
    max_rotation_speed: float = 2.09    # rad/s (120°/s)
    radius:             float = 200.0   # mm — radio de colisión agente
    # Rangos de interacción con el balón — usados SOLO para detección de
    # proximidad y handoff_ready envelope, NO como fuente de verdad de kick.
    # La ejecución real de kick/dribble es responsabilidad de BHuman.
    ball_interaction_range: float = 500.0   # mm — max range for handoff envelope
    min_robot_distance:    float = 200.0    # mm — distancia mínima entre robots
    fall_speed_thresh:  float = 280.0   # mm/s
    fall_probability:   float = 0.003
    recovery_steps:     int   = 91      # ~3s a dt=0.033 (3000ms / 33ms)


AGENT = AgentConfig()


# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS DEL BALÓN — Solo para simulación de entrenamiento
# NOTA: Estos parámetros NO son la fuente de verdad para kick/dribble.
# BHuman ejecuta kicks reales. Estos son para la simulación de entrenamiento
# donde necesitamos un modelo de balón para el reward de avance.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BallConfig:
    """Parámetros del balón para sim — NO fuente de verdad de kick."""
    radius:        float = 100.0    # mm
    friction:      float = 0.9867   # factor de fricción por paso (dt=0.033s)
    min_speed:     float = 10.0     # mm/s — por debajo se considera quieto


BALL = BallConfig()


# ─────────────────────────────────────────────────────────────────────────────
# CAMPO DE VISIÓN (FOV)
# Fuente: CameraIntrinsics.h#L24 — openingAngleWidth = 54.7°
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FOVConfig:
    half_angle:     float = 0.477   # rad — mitad de 54.7°
    max_distance:   float = 5000.0  # mm
    detection_prob: float = 0.90
    memory_steps:   int   = 30      # ~1000ms a dt=0.033 (1000ms / 33ms)


FOV = FOVConfig()


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimConfig:
    dt:          float = 0.033  # segundos por paso — alineado con Cognition cycle (33ms)
    num_agents:  int   = 4      # jugadores de campo (excluye portero)
    num_opp_max: int   = 4      # máximo de oponentes


SIM = SimConfig()


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION SPACE — Alineado con contrato BHuman
# Total: 60 floats por agente
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ObsConfig:
    """
    Espacio de observación por agente, alineado con contrato documental.

    Componentes:
      Self pose (x, y, θ)           = 3
      Self velocity (vx, vy, ω)     = 3
      Self state (upright, battery)  = 2
      Self role (one_hot 4 roles)    = 4  (defenderL, defenderR, forwardL, forwardR)
      Self active flag               = 1
      Goal info (opening, relative)  = 2
      Ball (x, y, vx, vy, age, team) = 6
      3 Teammates × 7               = 21
      4 Opponents × 3               = 12
      Game (playing, own, opp, time) = 4
      Tactic (one_hot 2: t121, t13)  = 2
    Total = 60
    """
    # Self (16)
    self_pose_size:   int = 3    # x, y, theta
    self_vel_size:    int = 3    # vx, vy, omega
    self_state_size:  int = 2    # is_upright, battery
    self_role_size:   int = 4    # one_hot: defenderL, defenderR, forwardL, forwardR
    self_active_size: int = 1    # is_active_agent flag
    self_goal_size:   int = 2    # goal_opening_angle, goal_relative_angle

    # Ball (6)
    ball_size:        int = 6    # x, y, vx, vy, age_ms_norm, using_team_ball

    # Teammates — 3 slots, 7 floats each (21)
    tm_slots:         int = 3
    tm_features:      int = 7    # x, y, walk_dx, walk_dy, speed, is_active, exists

    # Opponents — 4 slots, 3 floats each (12)
    opp_slots:        int = 4
    opp_features:     int = 3    # x, y, exists

    # Game state (4)
    game_size:        int = 4    # is_playing, own_score, opp_score, time_remaining

    # Tactic (2) — only 2 valid tactics in 4v4 Scaled
    tactic_size:      int = 2    # one_hot: t121_4v4_scaled, t13_4v4_scaled

    @property
    def self_total(self) -> int:
        return (self.self_pose_size + self.self_vel_size + self.self_state_size
                + self.self_role_size + self.self_active_size
                + self.self_goal_size)                         # 15

    @property
    def tm_total(self) -> int:
        return self.tm_slots * self.tm_features                # 21

    @property
    def opp_total(self) -> int:
        return self.opp_slots * self.opp_features              # 12

    @property
    def total(self) -> int:
        return (self.self_total + self.ball_size + self.tm_total
                + self.opp_total + self.game_size + self.tactic_size)  # 60


OBS = ObsConfig()
assert OBS.total == 60, f"OBS size mismatch: {OBS.total}"


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE (Critic centralizado — CTDE)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GlobalStateConfig:
    """Estado global para el critic centralizado (CTDE)."""
    # 4 agentes × (x, y, θ) = 12
    # 4 agentes × (is_upright) = 4
    # Ball (x, y, vx, vy) = 4
    # 4 oponentes × (x, y) = 8
    # Game (own_score, opp_score, time) = 3
    total: int = 31


GLOBAL_STATE = GlobalStateConfig()


# ─────────────────────────────────────────────────────────────────────────────
# ACTION SPACE — Positioning output (NOT ball control)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionConfig:
    """
    Espacio de acción por agente. RL aprende posicionamiento, NO kick/dribble.

    action[0] → target_x  relative displacement mm
    action[1] → target_y  relative displacement mm
    action[2] → target_theta heading rad (or face-ball auto)
    """
    size: int = 3
    x_range:             float = FIELD.x_opponent_ground   # 2813 mm
    y_range:             float = FIELD.y_left_sideline      # 1875 mm
    theta_range:         float = math.pi
    max_displacement_mm: float = 400.0


ACTION = ActionConfig()


# ─────────────────────────────────────────────────────────────────────────────
# TACTIC AND ROLE NAMES — 4v4 Scaled contract
# ─────────────────────────────────────────────────────────────────────────────

# Roles del equipo (4v4 Scaled, sin portero en RL)
ROLES = ["defenderL", "defenderR", "forwardL", "forwardR"]

# Solo las 2 tácticas válidas del contrato 4v4 Scaled
TACTIC_NAMES = list(VALID_TACTIC_NAMES)  # ["t121_4v4_scaled", "t13_4v4_scaled"]


# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM — 13 ETAPAS (BHUMAN_RL_CURRICULUM_CONTRACT.md)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurriculumStageConfig:
    """Configuración de una etapa del curriculum alineada con contrato BHuman."""
    stage_id:              int
    name:                  str
    num_rl_agents:         int
    num_opponents:         int
    theta_free:            bool
    episode_steps:         int
    shaping_weight:        float
    advance_metric:        str       # "takeover_rate", "win_rate", etc.
    advance_threshold:     float
    advance_window:        int

    # BHuman takeover
    takeover_required:     bool = False   # whether handoff/takeover flow is active
    takeover_timeout_steps: int = 90      # max frames for BHuman execution
    bhuman_mode:           str = "frozen" # "frozen" or "adaptive"

    # Obstacles
    num_static_obstacles:  int = 0
    num_dynamic_obstacles: int = 0

    # Spawn
    ball_max_dist:         float = -1.0
    spawn_ball_in_front:   bool = False
    spawn_front_angle_deg: float = 180.0

    # Smoothing
    ema_alpha:             float = 0.6
    rollout_steps:         int   = 2048
    local_horizon:         float = 0.0

    # Reward tuning
    shaping_approach_w:    float = 1.0
    shaping_takeover_w:    float = 0.0
    shaping_support_w:     float = 0.0

    @property
    def episode_seconds(self) -> float:
        return self.episode_steps * SIM.dt


CURRICULUM: List[CurriculumStageConfig] = [
    # ═══════════════════════════════════════════════════════════════════
    # STAGE 0: Ball ≤800mm, no obstacles — learn approach to takeover pose
    # Gate: ≥95% clean takeover
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=0,
        name="1v0_approach_800",
        num_rl_agents=1,
        num_opponents=0,
        theta_free=False,
        episode_steps=455,        # ~15s at dt=0.033
        shaping_weight=1.0,
        advance_metric="takeover_rate",
        advance_threshold=0.95,
        advance_window=50,
        takeover_required=True,
        ball_max_dist=800.0,
        spawn_ball_in_front=True,
        spawn_front_angle_deg=60.0,
        ema_alpha=0.55,
        rollout_steps=1024,
        local_horizon=400.0,
        shaping_approach_w=0.8,
        shaping_takeover_w=1.0,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 1: Ball ≤1600mm — extended approach
    # Gate: ≥90% takeover before timeout
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=1,
        name="1v0_approach_1600",
        num_rl_agents=1,
        num_opponents=0,
        theta_free=False,
        episode_steps=909,        # ~30s at dt=0.033
        shaping_weight=1.0,
        advance_metric="takeover_rate",
        advance_threshold=0.90,
        advance_window=80,
        takeover_required=True,
        ball_max_dist=1600.0,
        spawn_ball_in_front=True,
        spawn_front_angle_deg=90.0,
        ema_alpha=0.35,
        rollout_steps=1024,
        local_horizon=500.0,
        shaping_approach_w=0.8,
        shaping_takeover_w=0.7,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 2: Ball ≤3200mm, static obstacles
    # Gate: ≥85% clean takeover
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=2,
        name="1v0_obstacles_3200",
        num_rl_agents=1,
        num_opponents=0,
        theta_free=False,
        episode_steps=1212,       # ~40s at dt=0.033
        shaping_weight=1.0,
        advance_metric="takeover_rate",
        advance_threshold=0.85,
        advance_window=100,
        takeover_required=True,
        num_static_obstacles=2,
        ball_max_dist=3200.0,
        spawn_ball_in_front=False,
        ema_alpha=0.35,
        rollout_steps=1024,
        local_horizon=600.0,
        shaping_approach_w=0.7,
        shaping_takeover_w=0.8,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 3: Any position, dynamic obstacles
    # Gate: win rate >55% against scripted
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=3,
        name="1v1_dynamic",
        num_rl_agents=1,
        num_opponents=1,
        theta_free=False,
        episode_steps=1515,       # ~50s at dt=0.033
        shaping_weight=0.9,
        advance_metric="win_rate",
        advance_threshold=0.55,
        advance_window=150,
        takeover_required=True,
        num_dynamic_obstacles=1,
        ball_max_dist=-1.0,
        ema_alpha=0.35,
        rollout_steps=1024,
        local_horizon=700.0,
        shaping_approach_w=0.6,
        shaping_takeover_w=1.0,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 4: 1v1 self-play, BHuman frozen
    # Gate: win rate >55% + stable takeover
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=4,
        name="1v1_selfplay",
        num_rl_agents=1,
        num_opponents=1,
        theta_free=True,
        episode_steps=1818,       # ~60s at dt=0.033
        shaping_weight=0.8,
        advance_metric="win_rate",
        advance_threshold=0.55,
        advance_window=200,
        takeover_required=True,
        bhuman_mode="frozen",
        shaping_approach_w=0.5,
        shaping_takeover_w=1.0,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 5: 2v0 cooperation — learn support positioning
    # Gate: ≥85% useful support
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=5,
        name="2v0_cooperation",
        num_rl_agents=2,
        num_opponents=0,
        theta_free=False,
        episode_steps=1818,       # ~60s at dt=0.033
        shaping_weight=0.8,
        advance_metric="support_quality",
        advance_threshold=0.85,
        advance_window=200,
        takeover_required=True,
        shaping_approach_w=0.4,
        shaping_takeover_w=0.8,
        shaping_support_w=0.6,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 6: 2v2 self-play
    # Gate: win rate >52%
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=6,
        name="2v2_selfplay",
        num_rl_agents=2,
        num_opponents=2,
        theta_free=True,
        episode_steps=2727,       # ~90s at dt=0.033
        shaping_weight=0.6,
        advance_metric="win_rate",
        advance_threshold=0.52,
        advance_window=300,
        takeover_required=True,
        shaping_approach_w=0.3,
        shaping_takeover_w=1.0,
        shaping_support_w=0.8,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 7: 3v0 cooperation
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=7,
        name="3v0_cooperation",
        num_rl_agents=3,
        num_opponents=0,
        theta_free=False,
        episode_steps=2727,       # ~90s at dt=0.033
        shaping_weight=0.5,
        advance_metric="support_quality",
        advance_threshold=0.80,
        advance_window=200,
        takeover_required=True,
        shaping_approach_w=0.3,
        shaping_takeover_w=0.8,
        shaping_support_w=0.8,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 8: 3v2 scripted
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=8,
        name="3v2_scripted",
        num_rl_agents=3,
        num_opponents=2,
        theta_free=True,
        episode_steps=3636,       # ~120s at dt=0.033
        shaping_weight=0.4,
        advance_metric="win_rate",
        advance_threshold=0.52,
        advance_window=300,
        takeover_required=True,
        shaping_approach_w=0.2,
        shaping_takeover_w=1.0,
        shaping_support_w=1.0,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 9: 3v3 self-play
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=9,
        name="3v3_selfplay",
        num_rl_agents=3,
        num_opponents=3,
        theta_free=True,
        episode_steps=4545,       # ~150s at dt=0.033
        shaping_weight=0.3,
        advance_metric="win_rate",
        advance_threshold=0.52,
        advance_window=400,
        takeover_required=True,
        shaping_approach_w=0.2,
        shaping_takeover_w=1.0,
        shaping_support_w=1.0,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 10: 4v0 full team cooperation
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=10,
        name="4v0_full_team",
        num_rl_agents=4,
        num_opponents=0,
        theta_free=True,
        episode_steps=3636,       # ~120s at dt=0.033
        shaping_weight=0.3,
        advance_metric="support_quality",
        advance_threshold=0.75,
        advance_window=300,
        takeover_required=True,
        shaping_approach_w=0.2,
        shaping_takeover_w=0.8,
        shaping_support_w=1.0,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 11: 4v3 scripted
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=11,
        name="4v3_scripted",
        num_rl_agents=4,
        num_opponents=3,
        theta_free=True,
        episode_steps=6060,       # ~200s at dt=0.033
        shaping_weight=0.2,
        advance_metric="win_rate",
        advance_threshold=0.52,
        advance_window=400,
        takeover_required=True,
        shaping_approach_w=0.1,
        shaping_takeover_w=1.0,
        shaping_support_w=1.0,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 12: 4v4 full — final stage
    # Gate: >70% takeover success in 4v4
    # ═══════════════════════════════════════════════════════════════════
    CurriculumStageConfig(
        stage_id=12,
        name="4v4_full",
        num_rl_agents=4,
        num_opponents=4,
        theta_free=True,
        episode_steps=9090,       # ~300s at dt=0.033
        shaping_weight=0.1,
        advance_metric="takeover_success_rate",
        advance_threshold=0.70,
        advance_window=500,
        takeover_required=True,
        shaping_approach_w=0.1,
        shaping_takeover_w=1.0,
        shaping_support_w=1.0,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# RED NEURONAL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    hidden_sizes:       tuple = (256, 128)
    activation:         str   = "elu"
    log_std_init:       float = -0.8
    log_std_min:        float = -4.0
    log_std_max:        float = 0.0


NETWORK = NetworkConfig()


# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS PPO / MAPPO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    rollout_steps:    int   = 1024
    update_epochs:    int   = 4
    mini_batches:     int   = 4
    learning_rate:    float = 2e-4
    gamma:            float = 0.99
    lambda_gae:       float = 0.95
    clip_ratio:       float = 0.20
    value_coeff:      float = 0.50
    entropy_coeff:    float = 0.003
    max_grad_norm:    float = 0.50
    normalize_adv:    bool  = True
    kl_threshold:     float = 0.01
    reward_scale:     float = 0.05

    @property
    def batch_size(self) -> int:
        return self.rollout_steps // self.mini_batches


PPO_CFG = PPOConfig()


# ─────────────────────────────────────────────────────────────────────────────
# ACTION SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionSmoothingConfig:
    alpha:       float = 0.6
    enabled:     bool  = True


ACTION_SMOOTHING = ActionSmoothingConfig()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    total_steps:        int   = 2_000_000
    seed:               int   = 42
    device:             str   = "cuda"
    use_fov:            bool  = True
    cfg_root:           str   = ""
    selfplay_snapshot_every: int = 200


TRAINING = TrainingConfig()


@dataclass(frozen=True)
class QuickDebugTrainingProfile:
    """
    Lightweight profile for debugging the full RL <-> BHuman loop.

    Goal: make episodes, rollouts and curriculum advancement small enough to
    confirm that training is operational without waiting hours for a single
    stage transition under the Windows one-shot bridge transport.
    """
    enabled: bool = False
    total_steps: int = 8_000
    rollout_steps: int = 128
    update_epochs: int = 2
    mini_batches: int = 2
    stage0_episode_steps: int = 96
    stage1_episode_steps: int = 120
    stage2_episode_steps: int = 144
    stage3_episode_steps: int = 180
    stage0_window: int = 5
    stage1_window: int = 6
    stage2_window: int = 8
    stage3_window: int = 10
    stage0_threshold: float = 0.60
    stage1_threshold: float = 0.60
    stage2_threshold: float = 0.55
    stage3_threshold: float = 0.51
    save_every_eps: int = 25
    plot_every_eps: int = 10
    print_every_eps: int = 1


QUICK_DEBUG = QuickDebugTrainingProfile()


@dataclass(frozen=True)
class PracticalTrainingProfile:
    """
    Practical long-run profile for the first serious BHuman-in-loop trainings.

    It is intentionally smaller than the historical 50M-step target so we can:
    - verify persistent bridge stability
    - validate curriculum progression under real load
    - catch memory/tooling issues before a multi-day run
    """
    enabled: bool = True
    total_steps: int = 2_000_000
    save_every_eps: int = 100
    plot_every_eps: int = 25
    print_every_eps: int = 5


PRACTICAL_TRAINING = PracticalTrainingProfile()


def build_quick_debug_curriculum(
    stages: List["CurriculumStageConfig"],
    profile: QuickDebugTrainingProfile = QUICK_DEBUG,
) -> List["CurriculumStageConfig"]:
    """Return a shorter curriculum for smoke/debug training with BHuman in loop."""
    overrides = {
        0: dict(
            episode_steps=profile.stage0_episode_steps,
            advance_window=profile.stage0_window,
            advance_threshold=profile.stage0_threshold,
            rollout_steps=profile.rollout_steps,
        ),
        1: dict(
            episode_steps=profile.stage1_episode_steps,
            advance_window=profile.stage1_window,
            advance_threshold=profile.stage1_threshold,
            rollout_steps=profile.rollout_steps,
        ),
        2: dict(
            episode_steps=profile.stage2_episode_steps,
            advance_window=profile.stage2_window,
            advance_threshold=profile.stage2_threshold,
            rollout_steps=profile.rollout_steps,
        ),
        3: dict(
            episode_steps=profile.stage3_episode_steps,
            advance_window=profile.stage3_window,
            advance_threshold=profile.stage3_threshold,
            rollout_steps=profile.rollout_steps,
        ),
    }
    return [replace(stage, **overrides.get(stage.stage_id, {})) for stage in stages]


def build_quick_debug_ppo_cfg(
    base_cfg: PPOConfig,
    profile: QuickDebugTrainingProfile = QUICK_DEBUG,
) -> PPOConfig:
    """Smaller PPO buffers/updates for fast RLBridge diagnostics."""
    return replace(
        base_cfg,
        rollout_steps=profile.rollout_steps,
        update_epochs=profile.update_epochs,
        mini_batches=profile.mini_batches,
    )


def build_quick_debug_training_cfg(
    base_cfg: TrainingConfig,
    profile: QuickDebugTrainingProfile = QUICK_DEBUG,
) -> TrainingConfig:
    return replace(base_cfg, total_steps=profile.total_steps)


def build_quick_debug_logging_cfg(
    base_cfg: "LoggingConfig",
    profile: QuickDebugTrainingProfile = QUICK_DEBUG,
) -> "LoggingConfig":
    return replace(
        base_cfg,
        save_every_eps=profile.save_every_eps,
        plot_every_eps=profile.plot_every_eps,
        print_every_eps=profile.print_every_eps,
    )


def build_practical_training_cfg(
    base_cfg: TrainingConfig,
    profile: PracticalTrainingProfile = PRACTICAL_TRAINING,
) -> TrainingConfig:
    return replace(base_cfg, total_steps=profile.total_steps)


def build_practical_logging_cfg(
    base_cfg: "LoggingConfig",
    profile: PracticalTrainingProfile = PRACTICAL_TRAINING,
) -> "LoggingConfig":
    return replace(
        base_cfg,
        save_every_eps=profile.save_every_eps,
        plot_every_eps=profile.plot_every_eps,
        print_every_eps=profile.print_every_eps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoggingConfig:
    log_dir:        str = "runs"
    csv_file:       str = "metrics.csv"
    checkpoint_dir: str = "checkpoints"
    save_every_eps: int = 500
    plot_every_eps: int = 100
    print_every_eps: int = 10


LOG = LoggingConfig()


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

class ObsNormFactors:
    pos_x:      float = FIELD.x_opponent_ground   # 2813
    pos_y:      float = FIELD.y_left_sideline      # 1875
    theta:      float = math.pi
    vel_xy:     float = AGENT.max_forward_speed    # 250
    vel_theta:  float = AGENT.max_rotation_speed   # 2.09
    ball_age:   float = 10_000.0                   # ms
    speed:      float = AGENT.max_forward_speed    # 250
    score:      float = 10.0
    time:       float = 600.0                      # segundos de partido


NORM = ObsNormFactors()
