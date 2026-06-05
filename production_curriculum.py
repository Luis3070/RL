"""Shared production scenarios for both behavioral evaluation and BC dataset generation.

These are the deployment-facing situations the agent must clear before transplant: finishing
from the 2.5-3.0 m scoring band (clean and under static-lane pressure), refusing premature
bad-angle shots, and keeping corridor/collision sanity under dynamic obstacle pressure.

Both `behavioral_evaluator.py` (which grades a checkpoint) and `teacher_production_dataset.py`
(which collects gate-driven teacher demos) consume the SAME world builders defined here, so the
BC dataset covers exactly the states the policy is graded on. The 2.5-3.0 m band is a hard field
constraint and matches the gate's shoot_goal_enter window (3000 mm in gate.py): the
agent starts in the band, drives the ball to ~2.5 m, then shoots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from config import GOAL_X
from curriculum import (
    _dynamic_obstacles,
    _static_obstacles,
    sample_stage_world_spec_with_options,
)
from pybh_backend import WorldPlayerSpec, WorldSpec, build_world_spec

# Hard field constraint: the scoring band the shoot skill must convert from. Matches the
# gate's shoot_goal_enter_mm (3000) so "ball in band" and "shoot may arm" line up.
PRODUCTION_BAND_MM: tuple[float, float] = (2500.0, 3000.0)


@dataclass(frozen=True)
class ProductionScenario:
    """One deployment situation, shared by the evaluator and the dataset collector.

    `obstacles` and `is_contrast` are metadata the dataset collector uses for filtering and
    per-scenario summary breakdowns; the evaluator only needs name/stage_id/world_fn/episodes.
    """

    name: str
    stage_id: int
    description: str
    world_fn: Callable[[np.random.Generator], WorldSpec]
    episodes: int
    obstacles: str = "none"  # none | static | dynamic | blocking
    is_contrast: bool = False  # bad-angle negatives: teacher should reposition, not shoot


def _clip_y(value: float) -> float:
    return float(np.clip(value, -2400.0, 2400.0))


def _support_players_far() -> tuple[WorldPlayerSpec, ...]:
    return (
        WorldPlayerSpec(number=2, x=-4000.0, y=-2500.0, theta=0.0),
        WorldPlayerSpec(number=3, x=-4000.0, y=2500.0, theta=0.0),
    )


def _drive_finish_world(
    rng: np.random.Generator,
    *,
    goal_distance_mm: tuple[float, float],
    approach_mm: tuple[float, float],
    lateral_mm: tuple[float, float],
    heading_noise_deg: float,
    stage_id: int,
    obstacles: str,
) -> WorldSpec:
    ball_to_goal = float(rng.uniform(*goal_distance_mm))
    ball_x = float(GOAL_X - ball_to_goal)
    ball_y = _clip_y(float(rng.uniform(-520.0, 520.0)))
    approach = float(rng.uniform(*approach_mm))
    lateral = float(rng.uniform(*lateral_mm))
    robot_x = ball_x - approach
    robot_y = _clip_y(ball_y + lateral)
    robot_theta = math.atan2(ball_y - robot_y, ball_x - robot_x) + float(
        rng.uniform(-math.radians(heading_noise_deg), math.radians(heading_noise_deg))
    )
    ball_pos = np.asarray([ball_x, ball_y], dtype=np.float32)
    robot_pos = np.asarray([robot_x, robot_y], dtype=np.float32)
    opponents: tuple[WorldPlayerSpec, ...] = ()
    teammates: tuple[WorldPlayerSpec, ...] = ()
    if stage_id >= 2:
        teammates = _support_players_far()
    if obstacles == "static":
        opponents = _static_obstacles(robot_pos, ball_pos)
    elif obstacles == "dynamic":
        opponents = _dynamic_obstacles(robot_pos, ball_pos, episode_step=0)
    elif obstacles == "blocking":
        lane_side = 1.0 if ball_y <= 0.0 else -1.0
        opponents = (
            WorldPlayerSpec(number=1, x=ball_x - 850.0, y=ball_y + 180.0 * lane_side, theta=math.pi),
            WorldPlayerSpec(number=2, x=ball_x - 450.0, y=ball_y - 260.0 * lane_side, theta=math.pi),
        )
    return build_world_spec(
        ball_pos=ball_pos,
        robot_pos=robot_pos,
        robot_theta=robot_theta,
        teammates=teammates,
        opponents=opponents,
    )


def _preshoot_boundary_world(rng: np.random.Generator) -> WorldSpec:
    ball_x = float(GOAL_X - rng.uniform(2500.0, 3000.0))
    ball_y = _clip_y(float(rng.uniform(-650.0, 650.0)))
    distance = float(rng.uniform(270.0, 380.0))
    bad_angle = float(rng.choice((-1.0, 1.0)) * rng.uniform(math.radians(24.0), math.radians(38.0)))
    goal_heading = math.atan2(-ball_y, GOAL_X - ball_x)
    robot_theta = goal_heading - bad_angle
    robot_x = ball_x - distance * math.cos(goal_heading)
    robot_y = _clip_y(ball_y - distance * math.sin(goal_heading))
    return build_world_spec(
        ball_pos=np.asarray([ball_x, ball_y], dtype=np.float32),
        robot_pos=np.asarray([robot_x, robot_y], dtype=np.float32),
        robot_theta=robot_theta,
        teammates=_support_players_far(),
        opponents=_dynamic_obstacles(
            np.asarray([robot_x, robot_y], dtype=np.float32),
            np.asarray([ball_x, ball_y], dtype=np.float32),
            episode_step=0,
        ),
    )


def _sample_stage4_world(rng: np.random.Generator) -> WorldSpec:
    return sample_stage_world_spec_with_options(4, rng, diverse_resets=True)


def production_topup_scenarios(episodes: int) -> list[ProductionScenario]:
    """Targeted dataset-only situations for repairing production fine-tune gaps.

    These are deliberately not part of `production_scenarios`: the evaluator keeps
    its original deployment contract, while the teacher collector can request this
    set to generate dense bridge states that the 20k agent lacked:
    walk/stand -> dribble -> shoot_armed -> shoot under clean/static/blocking/dynamic
    pressure. All are stage 4 so the raw dataset already matches production skills.
    """
    return [
        ProductionScenario(
            name="finish_near_arm_clean_topup",
            stage_id=4,
            description="Clean near-arm shoot states: ball already near the 2.5 m edge and on-foot.",
            world_fn=lambda rng: _drive_finish_world(
                rng,
                goal_distance_mm=(2250.0, 2625.0),
                approach_mm=(260.0, 380.0),
                lateral_mm=(-45.0, 45.0),
                heading_noise_deg=4.0,
                stage_id=4,
                obstacles="none",
            ),
            episodes=max(episodes, 6),
            obstacles="none",
        ),
        ProductionScenario(
            name="finish_bridge_clean_topup",
            stage_id=4,
            description="Clean 2.5-3.0 m bridge: start just outside takeover, dribble into shoot.",
            world_fn=lambda rng: _drive_finish_world(
                rng,
                goal_distance_mm=(2350.0, 2750.0),
                approach_mm=(310.0, 540.0),
                lateral_mm=(-70.0, 70.0),
                heading_noise_deg=6.0,
                stage_id=4,
                obstacles="none",
            ),
            episodes=max(episodes, 8),
            obstacles="none",
        ),
        ProductionScenario(
            name="finish_bridge_static_topup",
            stage_id=4,
            description="Static-pressure bridge in the 2.5-3.0 m band with an open-enough lane.",
            world_fn=lambda rng: _drive_finish_world(
                rng,
                goal_distance_mm=(2350.0, 2800.0),
                approach_mm=(320.0, 590.0),
                lateral_mm=(-95.0, 95.0),
                heading_noise_deg=8.0,
                stage_id=4,
                obstacles="static",
            ),
            episodes=max(episodes, 6),
            obstacles="static",
        ),
        ProductionScenario(
            name="blocked_corridor_finish_topup",
            stage_id=4,
            description="Blocked-corridor finishing bridge: repair lane, keep ball, then shoot.",
            world_fn=lambda rng: _drive_finish_world(
                rng,
                goal_distance_mm=(2250.0, 2800.0),
                approach_mm=(320.0, 650.0),
                lateral_mm=(-140.0, 140.0),
                heading_noise_deg=10.0,
                stage_id=4,
                obstacles="blocking",
            ),
            episodes=max(episodes, 6),
            obstacles="blocking",
        ),
        ProductionScenario(
            name="dynamic_finish_pressure_topup",
            stage_id=4,
            description="Near-finish dynamic pressure without full-field walk-only dilution.",
            world_fn=lambda rng: _drive_finish_world(
                rng,
                goal_distance_mm=(2250.0, 2850.0),
                approach_mm=(330.0, 700.0),
                lateral_mm=(-150.0, 150.0),
                heading_noise_deg=10.0,
                stage_id=4,
                obstacles="dynamic",
            ),
            episodes=max(episodes, 8),
            obstacles="dynamic",
        ),
    ]


def production_scenarios(episodes: int) -> list[ProductionScenario]:
    """Return the production situations, scaling episode counts off the requested `episodes`.

    The per-scenario minimums (preshoot/dynamic/blocked) match behavioral_evaluator's previous
    inline values so a checkpoint is graded on the same workload the dataset was built from.
    """
    return [
        ProductionScenario(
            name="finish_band_2p5_3m_clean",
            stage_id=1,
            description="Ball starts 2.5-3.0 m from goal; no opponents. Must walk, dribble, then shoot.",
            world_fn=lambda rng: _drive_finish_world(
                rng,
                goal_distance_mm=PRODUCTION_BAND_MM,
                approach_mm=(650.0, 1050.0),
                lateral_mm=(-160.0, 160.0),
                heading_noise_deg=10.0,
                stage_id=1,
                obstacles="none",
            ),
            episodes=episodes,
            obstacles="none",
        ),
        ProductionScenario(
            name="finish_band_2p5_3m_static_lane",
            stage_id=2,
            description="Same 2.5-3.0 m finish band with static lane pressure.",
            world_fn=lambda rng: _drive_finish_world(
                rng,
                goal_distance_mm=PRODUCTION_BAND_MM,
                approach_mm=(750.0, 1250.0),
                lateral_mm=(-220.0, 220.0),
                heading_noise_deg=16.0,
                stage_id=2,
                obstacles="static",
            ),
            episodes=episodes,
            obstacles="static",
        ),
        ProductionScenario(
            name="preshoot_bad_angle_boundary",
            stage_id=4,
            description="Ball is in the scoring band but robot angle is deliberately loose. Avoid premature bad shots.",
            world_fn=_preshoot_boundary_world,
            episodes=max(episodes, 6),
            obstacles="dynamic",
            is_contrast=True,
        ),
        ProductionScenario(
            name="dynamic_full_field_pressure",
            stage_id=4,
            description="Stage-4 full-field dynamic obstacle reset distribution.",
            world_fn=_sample_stage4_world,
            episodes=max(episodes, 8),
            obstacles="dynamic",
        ),
        ProductionScenario(
            name="blocked_corridor_repair",
            stage_id=4,
            description="Two opponents pinch the lane; policy should keep corridor/collision sanity.",
            world_fn=lambda rng: _drive_finish_world(
                rng,
                goal_distance_mm=(2200.0, 3000.0),
                approach_mm=(800.0, 1350.0),
                lateral_mm=(-260.0, 260.0),
                heading_noise_deg=18.0,
                stage_id=4,
                obstacles="blocking",
            ),
            episodes=max(episodes, 6),
            obstacles="blocking",
        ),
    ]
