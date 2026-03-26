"""
settings.py — Configuración del subproyecto trajectory_guidance.

Cambios respecto a la versión anterior:
  FIX-A: Nuevo parámetro greedy_lookahead en AStarConfig.
  FIX-B: min_quality_score bajado a -5.0 (era 0.05). Con la quality
         normalizada por horizonte, los valores están en rango [-2, +12].
         El umbral de 0.05 rechazaba TODAS las trayectorias de stages 1-3.
         Con -5.0, solo rechazamos trayectorias verdaderamente degeneradas
         y dejamos que _check_validity haga el filtrado conductual.
  
  Nuevo: max_episode_steps por stage para closed-loop replay.
         Antes el replay duraba len(plan.waypoints) pasos (~70).
         Ahora dura hasta episode_steps del curriculum stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from config import CURRICULUM


@dataclass
class AStarConfig:
    grid_mm: float = 100.0
    max_search_depth: int = 70
    max_expansions: int = 5000
    top_k: int = 6
    diversity_min_endpoint_dist_mm: float = 250.0
    diversity_min_ball_dist_mm: float = 200.0
    heuristic_weight: float = 1.0
    goal_tolerance_mm: float = 220.0
    dribble_tolerance_mm: float = 260.0
    use_closed_set: bool = True
    # FIX-A: Profundidad de búsqueda para MPC greedy
    greedy_lookahead: int = 3


@dataclass
class GumbelConfig:
    temperature: float = 0.55
    score_scale: float = 1.0
    enabled: bool = True


@dataclass
class DatasetConfig:
    controlled_agent_id: str = "agent_0"
    max_steps_per_plan: int = 80
    save_every_episodes: int = 25
    # FIX-B: Umbral bajado drásticamente.
    # Con quality normalizada, valores válidos están en rango [-2, +12].
    # El umbral anterior de 0.05 rechazaba TODO para stages de distancia larga.
    # Ahora: -5.0 solo filtra trayectorias completamente degeneradas.
    # El filtrado real lo hace _check_validity (conductual, no numérico).
    min_quality_score: float = -5.0
    out_dir: str = "trajectory_demos"
    # Closed-loop: máximo de pasos por episodio replay.
    # 0 = usar episode_steps del curriculum stage.
    max_replay_steps: int = 0


@dataclass
class EvaluationConfig:
    evaluate_rollout: bool = True
    plot_examples: bool = False
    collect_snapshots: bool = True
    verbose: bool = True


@dataclass
class PrimitiveConfig:
    stage_modes: Dict[int, List[str]] = field(default_factory=lambda: {
        0: ["straight", "left_arc", "right_arc", "recover_contact",
            "dribble_safe"],
        1: ["straight", "left_arc", "right_arc", "dribble_safe",
            "dribble_aggressive"],
        2: ["straight", "left_arc", "right_arc", "dribble_safe",
            "align_then_kick"],
        3: ["straight", "left_arc", "right_arc", "dribble_aggressive",
            "align_then_kick"],
    })


@dataclass
class EnvBridgeConfig:
    controlled_agent_id: str = "agent_0"
    default_seed: int = 42
    deterministic_eval: bool = False
    stage_sync_on_reset: bool = True


@dataclass
class StageObjective:
    stage_id: int
    name: str
    objective: str
    success_mode: str
    target_recontacts: int = 1
    prefer_dribble: bool = False
    prefer_goal_alignment: bool = False
    prefer_goal_completion: bool = False
    planning_horizon_steps: int = 8
    primitive_modes: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    kick_target: Optional[np.ndarray] = None


STAGE_OBJECTIVES: Dict[int, StageObjective] = {
    0: StageObjective(
        stage_id=0,
        name="1v0_find_ball",
        objective="Llegar al balon y mantener control corto.",
        success_mode="touch_and_control",
        target_recontacts=2,
        prefer_dribble=True,
        planning_horizon_steps=100,
        primitive_modes=["straight", "left_arc", "right_arc",
                         "recover_contact", "dribble_safe"],
        weights={
            "agent_ball_dist": 1.0,
            "ball_goal_dist": 0.1,
            "turn_penalty": 0.3,
            "control_bonus": 1.2,
            "goal_alignment": 0.05,
            "ball_oob_penalty": 0.5,
            "own_goal_penalty": 1.0,
            "dribble_bonus": 80.0,
        },
    ),
    1: StageObjective(
        stage_id=1,
        name="1v0_r800",
        objective="Llegar y dribblar de forma estable con variedad.",
        success_mode="stable_dribble",
        target_recontacts=3,
        prefer_dribble=True,
        planning_horizon_steps=8,
        # Incluye dribbles direccionales para variabilidad post-contacto
        primitive_modes=["straight", "left_arc", "right_arc",
                         "dribble_safe", "dribble_aggressive",
                         "dribble_lateral_left", "dribble_lateral_right",
                         "dribble_retreat"],
        weights={
            "agent_ball_dist": 0.9,
            "ball_goal_dist": 0.2,
            "turn_penalty": 0.3,
            "control_bonus": 1.5,
            "goal_alignment": 0.15,
            "ball_oob_penalty": 1.0,
            "own_goal_penalty": 1.0,
            "dribble_bonus": 100.0,
        },
    ),
    2: StageObjective(
        stage_id=2,
        name="1v0_r1200",
        objective="Media distancia + dribble variado + alineacion + kick.",
        success_mode="dribble_align_kick",
        target_recontacts=2,
        prefer_dribble=True,
        prefer_goal_alignment=True,
        planning_horizon_steps=12,
        primitive_modes=["straight", "left_arc", "right_arc",
                         "dribble_safe", "dribble_aggressive",
                         "dribble_lateral_left", "dribble_lateral_right",
                         "dribble_retreat", "align_then_kick"],
        weights={
            "agent_ball_dist": 0.7,
            "ball_goal_dist": 0.8,
            "turn_penalty": 0.25,
            "control_bonus": 1.2,
            "goal_alignment": 0.8,
            # Penalización OOB más agresiva: el kick no puede mandar el
            # balón fuera; con 2.0 el costo de acercarse a la sideline
            # supera cualquier ganancia de alineación al arco.
            "ball_oob_penalty": 2.0,
            "own_goal_penalty": 1.5,
            # Mayor peso de dribble: más recontactos = menor costo A*,
            # desincentivando el kick inmediato sin dribble previo.
            "dribble_bonus": 120.0,
        },
    ),
    3: StageObjective(
        stage_id=3,
        name="1v0_r3500",
        objective="Campo abierto + navegacion + dribble variado + gol.",
        success_mode="goal_completion",
        target_recontacts=1,
        # prefer_dribble=True: activa el multiplicador 1.5× en
        # sustained_dribble_bonus, aumentando la recompensa por dribble
        # y reduciendo la brecha con el kick inmediato.
        prefer_dribble=True,
        prefer_goal_alignment=True,
        prefer_goal_completion=True,
        planning_horizon_steps=20,
        primitive_modes=["straight", "left_arc", "right_arc",
                         "dribble_safe", "dribble_aggressive",
                         "dribble_lateral_left", "dribble_lateral_right",
                         "dribble_retreat", "align_then_kick"],
        weights={
            "agent_ball_dist": 0.5,
            "ball_goal_dist": 1.2,
            "turn_penalty": 0.2,
            "control_bonus": 0.9,
            "goal_alignment": 1.0,
            # OOB más agresivo para stage 3 (campo abierto, más riesgo
            # de que el kick diagonal saque el balón por la sideline).
            "ball_oob_penalty": 2.5,
            "own_goal_penalty": 2.0,
            # Mayor dribble_bonus: junto con prefer_dribble=True y el
            # gate en heuristic_cost, A* debe planear 2+ dribbles antes
            # de intentar kick.
            "dribble_bonus": 100.0,
        },
    ),
}


def get_stage_cfg(stage_id: int):
    return CURRICULUM[stage_id]


def get_stage_objective(stage_id: int) -> StageObjective:
    if stage_id not in STAGE_OBJECTIVES:
        raise KeyError(
            f"No stage objective configured for stage {stage_id}")
    return STAGE_OBJECTIVES[stage_id]


ASTAR_CFG = AStarConfig()
GUMBEL_CFG = GumbelConfig()
DATASET_CFG = DatasetConfig()
EVAL_CFG = EvaluationConfig()
PRIMITIVE_CFG = PrimitiveConfig()
ENV_BRIDGE_CFG = EnvBridgeConfig()