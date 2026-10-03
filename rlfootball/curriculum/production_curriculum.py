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

from rlfootball.core.config import GOAL_X
from rlfootball.curriculum.curriculum import (
    _dynamic_obstacles,
    _static_obstacles,
    sample_stage_world_spec_with_options,
)
from rlfootball.core.pybh_backend import WorldPlayerSpec, WorldSpec, build_world_spec

# Hard field constraint: the scoring band the shoot skill must convert from. Matches the
# gate's shoot_goal_enter_mm (3000) so "ball in band" and "shoot may arm" line up.
PRODUCTION_BAND_MM: tuple[float, float] = (2500.0, 3000.0)
OWN_GOAL_X = -GOAL_X


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


# ---------------------------------------------------------------------------
# Multi-agent scenarios — §14.21 LOCKED weights for Monday 2026-06-15 demo
# ---------------------------------------------------------------------------

def _team_2v0_clean_world(rng: np.random.Generator) -> WorldSpec:
    """Stage 6 baseline: ball in production band, 1 BH GK + 2 RL field, no opponents.

    Striker spawns approaching the ball; the open-support spawns ahead-and-side so
    the open-lane teacher has a clean receive position to anchor on. GK is parked
    in own goal area (BHuman handles it).
    """
    ball_to_goal = float(rng.uniform(*PRODUCTION_BAND_MM))
    ball_x = float(GOAL_X - ball_to_goal)
    ball_y = _clip_y(float(rng.uniform(-450.0, 450.0)))
    striker_x = ball_x - float(rng.uniform(700.0, 1050.0))
    striker_y = _clip_y(ball_y + float(rng.uniform(-180.0, 180.0)))
    support_side = 1.0 if ball_y >= 0.0 else -1.0
    # Forward spawn (BHuman Tactic templates position the forward high from the
    # start). Spawning at x=800 made the support unreachable for the whole chain:
    # the striker outruns it and pass_dist exceeds MAX_PASS_DISTANCE_MM by
    # mid-drive (2026-06-10 probe: p90 pass_dist 5846mm, zero pass proposals).
    support = WorldPlayerSpec(
        number=2,
        x=float(rng.uniform(2200.0, 2600.0)),
        y=float(rng.uniform(1150.0, 1400.0)) * support_side,
        theta=math.atan2(-support_side, 1.0),
    )
    gk = WorldPlayerSpec(number=3, x=-4000.0, y=0.0, theta=0.0)
    robot_theta = math.atan2(ball_y - striker_y, ball_x - striker_x)
    return build_world_spec(
        ball_pos=np.asarray([ball_x, ball_y], dtype=np.float32),
        robot_pos=np.asarray([striker_x, striker_y], dtype=np.float32),
        robot_theta=robot_theta,
        teammates=(support, gk),
        opponents=(),
    )


def _team_2v0_static_blocker_world(rng: np.random.Generator) -> WorldSpec:
    """Stage 6 with one static opponent blocking the striker's direct lane.

    Goal of this scenario: force the smashOrPass decision to consider pass to the
    support. Static opponent placed near the ball-to-goal line within the front-cone
    proxy, so `nearest_opponent_front_dist` drops below OPPONENT_LANE_CLEARANCE_MM
    and pass_reachable_proxy excludes the direct shoot.
    """
    base = _team_2v0_clean_world(rng)
    support_side = 1.0 if base.ball_y >= 0.0 else -1.0
    support = WorldPlayerSpec(
        number=2,
        x=float(rng.uniform(2200.0, 2600.0)),
        y=float(rng.uniform(1150.0, 1400.0)) * support_side,
        theta=0.0,
    )
    gk = WorldPlayerSpec(number=3, x=-4000.0, y=0.0, theta=0.0)
    # Place one opponent ~700mm in front of the ball.
    opp_x = float(base.ball_x + 700.0)
    opp_y = _clip_y(float(base.ball_y + float(rng.uniform(-200.0, 200.0))))
    blocker = WorldPlayerSpec(number=1, x=opp_x, y=opp_y, theta=math.pi)
    return build_world_spec(
        ball_pos=np.asarray([base.ball_x, base.ball_y], dtype=np.float32),
        robot_pos=np.asarray([base.robot_x, base.robot_y], dtype=np.float32),
        robot_theta=base.robot_theta,
        teammates=(support, gk),
        opponents=(blocker,),
    )


def _team_3v0_clean_world(rng: np.random.Generator) -> WorldSpec:
    """Stage 7 baseline: 1 BH GK + 3 RL field, no opponents. All three roles emerge."""
    ball_to_goal = float(rng.uniform(*PRODUCTION_BAND_MM))
    ball_x = float(GOAL_X - ball_to_goal)
    ball_y = _clip_y(float(rng.uniform(-500.0, 500.0)))
    striker_x = ball_x - float(rng.uniform(800.0, 1100.0))
    striker_y = _clip_y(ball_y + float(rng.uniform(-180.0, 180.0)))
    side = 1.0 if ball_y >= 0.0 else -1.0
    # Forward spawns mirroring the Tactic-template slots (see stage-6 rationale):
    # open_support high on the ball side as the primary receive option, off_ball
    # mid-depth opposite side as the triangle vertex.
    open_support = WorldPlayerSpec(
        number=2,
        x=float(rng.uniform(2200.0, 2600.0)),
        y=float(rng.uniform(1150.0, 1400.0)) * side,
        theta=math.atan2(-side, 1.0),
    )
    off_ball_support = WorldPlayerSpec(
        number=3,
        x=float(rng.uniform(1300.0, 1700.0)),
        y=float(rng.uniform(-1400.0, -1150.0)) * side,
        theta=math.atan2(side, 1.0),
    )
    gk = WorldPlayerSpec(number=4, x=-4000.0, y=0.0, theta=0.0)
    return build_world_spec(
        ball_pos=np.asarray([ball_x, ball_y], dtype=np.float32),
        robot_pos=np.asarray([striker_x, striker_y], dtype=np.float32),
        robot_theta=math.atan2(ball_y - striker_y, ball_x - striker_x),
        teammates=(open_support, off_ball_support, gk),
        opponents=(),
    )


def _team_3v0_static_pinch_world(rng: np.random.Generator) -> WorldSpec:
    """Stage 7 with 2 static opponents pinching the central lane."""
    base = _team_3v0_clean_world(rng)
    pinch_a = WorldPlayerSpec(
        number=1, x=float(base.ball_x + 700.0),
        y=_clip_y(float(base.ball_y + 350.0)), theta=math.pi,
    )
    pinch_b = WorldPlayerSpec(
        number=2, x=float(base.ball_x + 700.0),
        y=_clip_y(float(base.ball_y - 350.0)), theta=math.pi,
    )
    return build_world_spec(
        ball_pos=np.asarray([base.ball_x, base.ball_y], dtype=np.float32),
        robot_pos=np.asarray([base.robot_x, base.robot_y], dtype=np.float32),
        robot_theta=base.robot_theta,
        teammates=base.teammates,
        opponents=(pinch_a, pinch_b),
    )


def _team_3v0_dynamic_world(rng: np.random.Generator) -> WorldSpec:
    """Stage 7 with 2-3 dynamic opponents using BH default behaviour."""
    base = _team_3v0_clean_world(rng)
    robot_pos_arr = np.asarray([base.robot_x, base.robot_y], dtype=np.float32)
    ball_pos_arr = np.asarray([base.ball_x, base.ball_y], dtype=np.float32)
    opps = _dynamic_obstacles(robot_pos_arr, ball_pos_arr, episode_step=0)
    return build_world_spec(
        ball_pos=ball_pos_arr,
        robot_pos=robot_pos_arr,
        robot_theta=base.robot_theta,
        teammates=base.teammates,
        opponents=opps,
    )


# ---------------------------------------------------------------------------
# v3 (doc-35) contrastive A/B worlds. Each pair is identical except one degree of
# freedom (a teammate's openness / lane coverage) so the teacher demonstrates BOTH
# branches on near-identical ego-states — the matched data the policy needs to learn
# the contingency. B-variants carry is_contrast=True (covered → reposition/dribble).
# ---------------------------------------------------------------------------

def _team_giveandgo_world(rng: np.random.Generator, *, outlet_present: bool) -> WorldSpec:
    """v3.1 TEAMMATE-varying pass contrast — opponent layout CONSTANT (none) in both.

    A (outlet_present=True):  an open up-field outlet teammate exists → teacher PASSES.
    B (outlet_present=False): that teammate is pulled DEEP/behind (no up-field outlet) →
       teacher DRIBBLES/shoots.
    The ONLY thing that differs is the outlet teammate's position, so the pass↔dribble
    flip is attributable to the teammate dims (obs[26:31]) — NOT to any opponent (the v3
    bug, where the contrast varied an opponent the policy reads via its own front cone).
    """
    ball_x = float(rng.uniform(800.0, 1600.0))
    ball_y = _clip_y(float(rng.uniform(-500.0, 500.0)))
    side = 1.0 if ball_y >= 0.0 else -1.0
    striker = (ball_x - float(rng.uniform(90.0, 200.0)), _clip_y(ball_y + float(rng.uniform(-120.0, 120.0))))
    deep = (float(rng.uniform(-600.0, 200.0)), _clip_y(-side * float(rng.uniform(900.0, 1400.0))))
    if outlet_present:
        # reachable up-field outlet (high goal-proximity, in pass range)
        outlet = (float(rng.uniform(2400.0, 3000.0)), _clip_y(side * float(rng.uniform(900.0, 1500.0))))
    else:
        # same teammate, pulled DEEP behind the ball → low goal-proximity, not an up-field
        # option → smash_or_pass finds no pass target → dribble. Body count identical.
        outlet = (float(rng.uniform(-1200.0, -400.0)), _clip_y(side * float(rng.uniform(200.0, 900.0))))
    gk = WorldPlayerSpec(number=4, x=-4000.0, y=0.0, theta=0.0)
    return build_world_spec(
        ball_pos=np.asarray([ball_x, ball_y], dtype=np.float32),
        robot_pos=np.asarray(striker, dtype=np.float32),
        robot_theta=math.atan2(ball_y - striker[1], ball_x - striker[0]),
        teammates=(
            WorldPlayerSpec(number=2, x=outlet[0], y=outlet[1], theta=math.atan2(ball_y - outlet[1], ball_x - outlet[0])),
            WorldPlayerSpec(number=3, x=deep[0], y=deep[1], theta=math.atan2(ball_y - deep[1], ball_x - deep[0])),
            gk,
        ),
        opponents=(),  # CONSTANT across A/B — the flip must come from the teammate, not an opponent
    )


def _team_support_world(rng: np.random.Generator, *, striker_live: bool) -> WorldSpec:
    """v3.1 TEAMMATE-varying support contrast — opponent layout CONSTANT (none) in both.

    A (striker_live=True):  the striker (teammate) is ON the ball, controlling → this
       support is a live outlet → teacher support HOLDS (stand).
    B (striker_live=False): the ball is loose and the striker is FAR (not controlling) →
       support REPOSITIONS toward play.
    Only the striker's (teammate's) position relative to the ball differs, so the support's
    hold↔reposition is attributable to the teammate dims, not to the support's own opponent.
    """
    ball_x = float(rng.uniform(900.0, 1700.0))
    ball_y = _clip_y(float(rng.uniform(-400.0, 400.0)))
    side = 1.0 if ball_y >= 0.0 else -1.0
    support = (float(rng.uniform(2000.0, 2600.0)), _clip_y(side * float(rng.uniform(1100.0, 1400.0))))
    deep = (float(rng.uniform(-600.0, 200.0)), _clip_y(-side * float(rng.uniform(900.0, 1300.0))))
    if striker_live:
        striker = (ball_x - float(rng.uniform(90.0, 180.0)), _clip_y(ball_y))            # on the ball
    else:
        striker = (ball_x - float(rng.uniform(2000.0, 2800.0)),                          # far → ball loose
                   _clip_y(ball_y - side * float(rng.uniform(400.0, 900.0))))
    gk = WorldPlayerSpec(number=4, x=-4000.0, y=0.0, theta=0.0)
    return build_world_spec(
        ball_pos=np.asarray([ball_x, ball_y], dtype=np.float32),
        robot_pos=np.asarray(striker, dtype=np.float32),
        robot_theta=math.atan2(ball_y - striker[1], ball_x - striker[0]),
        teammates=(
            WorldPlayerSpec(number=2, x=support[0], y=support[1], theta=math.atan2(ball_y - support[1], ball_x - support[0])),
            WorldPlayerSpec(number=3, x=deep[0], y=deep[1], theta=math.atan2(ball_y - deep[1], ball_x - deep[0])),
            gk,
        ),
        opponents=(),
    )


def _giveandgo_outlet_present_world(rng): return _team_giveandgo_world(rng, outlet_present=True)
def _giveandgo_outlet_absent_world(rng):  return _team_giveandgo_world(rng, outlet_present=False)
def _support_striker_live_world(rng):     return _team_support_world(rng, striker_live=True)
def _support_striker_far_world(rng):      return _team_support_world(rng, striker_live=False)


def _pass_forcing_world(rng: np.random.Generator) -> WorldSpec:
    """v4-S3 PASS-FORCING — pass is the winning move, dribble/shoot is low value.

    The striker has the ball FAR from goal (low own shot value) with a clear front, and an
    OPEN teammate is ADVANCED near the opponent goal (high pass value). smash_or_pass then
    chooses pass (own_value < advanced teammate value), and the completed-pass reward fires
    when the ball reaches the advanced teammate. Front kept clear so the pass-reachability
    proxy (striker front-cone) allows the pass; opponents (if any) sit behind, never in the
    striker→outlet lane. This is the data where PPO can actually learn that passing pays.
    """
    ball_x = float(rng.uniform(-400.0, 600.0))                 # far from opponent goal → low own shot value
    ball_y = _clip_y(float(rng.uniform(-700.0, 700.0)))
    side = 1.0 if ball_y >= 0.0 else -1.0
    striker = (ball_x - float(rng.uniform(90.0, 200.0)), _clip_y(ball_y + float(rng.uniform(-120.0, 120.0))))
    # advanced outlet (high goal-proximity) on the ball side, clear lane — kept at a
    # COMPLETABLE pass distance (~2.2-2.8m ahead) so the completed-pass reward can fire.
    outlet = (float(rng.uniform(2200.0, 2800.0)), _clip_y(side * float(rng.uniform(700.0, 1200.0))))
    deep = (float(rng.uniform(-1400.0, -600.0)), _clip_y(-side * float(rng.uniform(700.0, 1200.0))))
    gk = WorldPlayerSpec(number=4, x=-4000.0, y=0.0, theta=0.0)
    # optional opponent kept BEHIND the striker (own half), never in the striker→outlet lane,
    # so it doesn't trip the front-cone pass proxy.
    opp = WorldPlayerSpec(number=1, x=striker[0] - float(rng.uniform(700.0, 1200.0)),
                          y=_clip_y(ball_y + side * float(rng.uniform(200.0, 600.0))), theta=0.0)
    return build_world_spec(
        ball_pos=np.asarray([ball_x, ball_y], dtype=np.float32),
        robot_pos=np.asarray(striker, dtype=np.float32),
        robot_theta=math.atan2(outlet[1] - striker[1], outlet[0] - striker[0]),  # face the outlet
        teammates=(
            WorldPlayerSpec(number=2, x=outlet[0], y=outlet[1], theta=math.atan2(ball_y - outlet[1], ball_x - outlet[0])),
            WorldPlayerSpec(number=3, x=deep[0], y=deep[1], theta=math.atan2(ball_y - deep[1], ball_x - deep[0])),
            gk,
        ),
        opponents=(opp,),
    )


def _team_3v3_full_dynamic_world(rng: np.random.Generator) -> WorldSpec:
    """Monday demo distribution: 1 BH GK + 3 RL field vs full BH (1 GK + 3 field, dynamic).

    Opponent placement matches `_team_3v0_dynamic_world` for the field opponents plus an
    opponent-side GK at the opposite goal. PPO training spends ~3.0 weight here per
    §14.21.3 — non-negotiable; this is the only scenario with the exact perceptual
    conditions of the Monday demo.
    """
    base = _team_3v0_dynamic_world(rng)
    # Add opponent-side GK at opposite goal area.
    opp_gk = WorldPlayerSpec(number=3, x=float(GOAL_X) - 300.0, y=0.0, theta=math.pi)
    opps = tuple(base.opponents) + (opp_gk,)
    return build_world_spec(
        ball_pos=np.asarray([base.ball_x, base.ball_y], dtype=np.float32),
        robot_pos=np.asarray([base.robot_x, base.robot_y], dtype=np.float32),
        robot_theta=base.robot_theta,
        teammates=base.teammates,
        opponents=opps,
    )


def _defense_team_world(
    rng: np.random.Generator,
    *,
    ball_x_range: tuple[float, float],
    attacker_count: int,
    dynamic: bool,
    transition: bool = False,
) -> WorldSpec:
    ball_x = float(rng.uniform(*ball_x_range))
    ball_y = _clip_y(float(rng.uniform(-1200.0, 1200.0) if not transition else rng.uniform(-800.0, 800.0)))
    side = 1.0 if ball_y >= 0.0 else -1.0
    presser = (
        ball_x - float(rng.uniform(550.0, 850.0)),
        _clip_y(ball_y + side * float(rng.uniform(80.0, 180.0))),
    )
    defender = (
        float(np.clip(ball_x - 700.0, -3400.0, -2600.0)),
        _clip_y(ball_y - side * 700.0),
    )
    supporter = (
        float(rng.uniform(-1600.0, -1000.0)),
        _clip_y(-side * float(rng.uniform(700.0, 1300.0))),
    )
    gk = WorldPlayerSpec(number=4, x=-4300.0, y=0.0, theta=0.0)
    opponents: list[WorldPlayerSpec] = []
    if attacker_count >= 1:
        opponents.append(WorldPlayerSpec(number=1, x=ball_x + 400.0, y=ball_y, theta=math.pi))
    if attacker_count >= 2:
        opponents.append(WorldPlayerSpec(number=2, x=ball_x + 1100.0, y=_clip_y(-ball_y), theta=math.pi))
    if attacker_count >= 3:
        opponents.append(WorldPlayerSpec(number=3, x=float(GOAL_X) - 300.0, y=0.0, theta=math.pi))
    if dynamic and attacker_count < 3:
        dyn = _dynamic_obstacles(
            np.asarray(presser, dtype=np.float32),
            np.asarray([ball_x, ball_y], dtype=np.float32),
            episode_step=0,
        )
        used_numbers = {int(player.number) for player in opponents}
        next_number = 1
        for player in dyn:
            while next_number in used_numbers:
                next_number += 1
            if next_number > 3:
                break
            opponents.append(WorldPlayerSpec(
                number=next_number,
                x=float(player.x),
                y=float(player.y),
                theta=float(player.theta),
            ))
            used_numbers.add(next_number)
    return build_world_spec(
        ball_pos=np.asarray([ball_x, ball_y], dtype=np.float32),
        robot_pos=np.asarray(presser, dtype=np.float32),
        robot_theta=math.atan2(ball_y - presser[1], ball_x - presser[0]),
        teammates=(
            WorldPlayerSpec(number=2, x=defender[0], y=defender[1], theta=math.atan2(ball_y - defender[1], ball_x - defender[0])),
            WorldPlayerSpec(number=3, x=supporter[0], y=supporter[1], theta=math.atan2(ball_y - supporter[1], ball_x - supporter[0])),
            gk,
        ),
        opponents=tuple(opponents),
    )


def _defense_1v3_static_entry_world(rng: np.random.Generator) -> WorldSpec:
    return _defense_team_world(rng, ball_x_range=(-2200.0, -1200.0), attacker_count=1, dynamic=False)


def _defense_2v3_dynamic_world(rng: np.random.Generator) -> WorldSpec:
    return _defense_team_world(rng, ball_x_range=(-2500.0, -800.0), attacker_count=2, dynamic=True)


def _defense_3v3_full_world(rng: np.random.Generator) -> WorldSpec:
    return _defense_team_world(rng, ball_x_range=(-3000.0, -1000.0), attacker_count=3, dynamic=True)


def _transition_midfield_world(rng: np.random.Generator) -> WorldSpec:
    # Keep this just inside the defensive phase instead of straddling x=0. When
    # it flickers between attack/defense, support targets switch too often and
    # the sanitizer drops every episode for hold-too-short/collision.
    return _defense_team_world(rng, ball_x_range=(-900.0, -250.0), attacker_count=1, dynamic=True, transition=True)


def _defense_recovery_world(rng: np.random.Generator) -> WorldSpec:
    """Stage 8 recovery: our player wins the ball deep and must clear/pass out.

    The ball spawns ON one of our field players in the defensive third (it controls
    the ball immediately → D1 promotes it to striker), an attacker presses from
    behind (the duel it just won), and an up-field teammate is available as a
    clearance-pass outlet. Produces the recover→safe-pass / clearance / drive-out
    rows that the §Part2 D2/D3 teacher logic needs but the positional defense
    scenarios never generate (there our players never get the ball).
    """
    ball_x = float(rng.uniform(-3000.0, -2000.0))
    ball_y = _clip_y(float(rng.uniform(-800.0, 800.0)))
    side = 1.0 if ball_y >= 0.0 else -1.0
    # Recovering player spawns essentially on the ball (within ball-control radius).
    recoverer = (ball_x - float(rng.uniform(80.0, 220.0)), _clip_y(ball_y + float(rng.uniform(-120.0, 120.0))))
    # Up-field outlet teammate: ahead of the ball toward midfield, off to one side.
    outlet = (float(rng.uniform(-1200.0, 200.0)), _clip_y(side * float(rng.uniform(700.0, 1500.0))))
    # Deep cover teammate stays back.
    cover = (float(rng.uniform(-3600.0, -3200.0)), _clip_y(ball_y * 0.4))
    gk = WorldPlayerSpec(number=4, x=-4300.0, y=0.0, theta=0.0)
    # One pressing attacker just goal-side of the ball (recently dispossessed).
    presser_x = ball_x + float(rng.uniform(250.0, 600.0))
    opponents = (WorldPlayerSpec(number=1, x=presser_x, y=_clip_y(ball_y + float(rng.uniform(-250.0, 250.0))), theta=math.pi),)
    return build_world_spec(
        ball_pos=np.asarray([ball_x, ball_y], dtype=np.float32),
        robot_pos=np.asarray(recoverer, dtype=np.float32),
        robot_theta=math.atan2(ball_y - recoverer[1], ball_x - recoverer[0]),
        teammates=(
            WorldPlayerSpec(number=2, x=outlet[0], y=outlet[1], theta=math.atan2(ball_y - outlet[1], ball_x - outlet[0])),
            WorldPlayerSpec(number=3, x=cover[0], y=cover[1], theta=math.atan2(ball_y - cover[1], ball_x - cover[0])),
            gk,
        ),
        opponents=opponents,
    )


def _defense_mark_open_world(rng: np.random.Generator) -> WorldSpec:
    """Stage 8 mark validation: an attacker isolated ~1.1m in front of the off-ball
    defender in OPEN space.

    Unlike defense_3v3_full (deep ball + attackers all crammed into our third, no room
    for a clean mark target), here the ball and the bulk of play sit up-field while a
    lone attacker lurks deep on our defender. That makes the mark target (a) inside the
    1.5m mark gate band, (b) directly in the off-ball defender's gaze cone (it faces the
    ball, the target sits between it and the ball), and (c) a big, isolated blob for the
    vision NN (close range, nothing else nearby) — so mark can fire from REAL perception,
    which is what the Stage-3 no-sim2real-gap check needs. Not used in the locked training
    pool; validation/collection only.
    """
    ball_x = float(rng.uniform(-1200.0, -400.0))
    ball_y = _clip_y(float(rng.uniform(-700.0, 700.0)))
    side = 1.0 if ball_y >= 0.0 else -1.0
    # On-ball presser (becomes striker) contests the ball up-field.
    presser = (ball_x - float(rng.uniform(150.0, 350.0)), _clip_y(ball_y + side * float(rng.uniform(60.0, 160.0))))
    # Off-ball defender deep, with space ahead of it (toward the ball/up-field).
    defender = (float(rng.uniform(-3300.0, -2900.0)), _clip_y(float(rng.uniform(-900.0, 900.0))))
    # Mark target: lone attacker in front of the defender along its defender->ball
    # sightline, ~1.1m away BUT capped to stay >=800mm short of the ball so it never
    # stacks onto the play (the failure mode that broke the cramped defense_3v3_full
    # attempt). Floored at 700mm so it stays clear of the defender's own body.
    ddx, ddy = ball_x - defender[0], ball_y - defender[1]
    dn = math.hypot(ddx, ddy) or 1.0
    mark_dist = float(np.clip(min(1100.0, dn - 800.0), 700.0, 1100.0))
    mark_x = float(np.clip(defender[0] + (ddx / dn) * mark_dist, -4300.0, 4300.0))
    mark_y = _clip_y(defender[1] + (ddy / dn) * mark_dist)
    # Open-support teammate up-field on the far side.
    supporter = (float(rng.uniform(-1400.0, -200.0)), _clip_y(-side * float(rng.uniform(700.0, 1300.0))))
    gk = WorldPlayerSpec(number=4, x=-4300.0, y=0.0, theta=0.0)
    opponents = (
        WorldPlayerSpec(number=1, x=mark_x, y=mark_y,
                        theta=math.atan2(defender[1] - mark_y, defender[0] - mark_x)),
        WorldPlayerSpec(number=2, x=float(np.clip(ball_x + 350.0, -4300.0, 4300.0)), y=ball_y, theta=math.pi),
    )
    return build_world_spec(
        ball_pos=np.asarray([ball_x, ball_y], dtype=np.float32),
        robot_pos=np.asarray(presser, dtype=np.float32),
        robot_theta=math.atan2(ball_y - presser[1], ball_x - presser[0]),
        teammates=(
            WorldPlayerSpec(number=2, x=defender[0], y=defender[1],
                            theta=math.atan2(ball_y - defender[1], ball_x - defender[0])),
            WorldPlayerSpec(number=3, x=supporter[0], y=supporter[1],
                            theta=math.atan2(ball_y - supporter[1], ball_x - supporter[0])),
            gk,
        ),
        opponents=opponents,
    )


def team_defense_scenarios(episodes: int) -> list[ProductionScenario]:
    e = max(1, int(episodes))
    return [
        ProductionScenario(
            name="defense_1v3_static_entry", stage_id=8,
            description="Stage 8 defensive entry: one static attacker, presser + deep defender + support.",
            world_fn=_defense_1v3_static_entry_world, episodes=e * 1, obstacles="static",
        ),
        ProductionScenario(
            name="defense_2v3_dynamic", stage_id=8,
            description="Stage 8 defensive pressure with two dynamic attackers.",
            world_fn=_defense_2v3_dynamic_world, episodes=e * 2, obstacles="dynamic",
        ),
        ProductionScenario(
            name="defense_3v3_full", stage_id=8,
            description="Stage 8 full defensive 3v3 distribution.",
            world_fn=_defense_3v3_full_world, episodes=e * 3, obstacles="dynamic",
        ),
        ProductionScenario(
            name="transition_midfield", stage_id=8,
            description="Stage 8 midfield transition with sticky attack/defense phase.",
            world_fn=_transition_midfield_world, episodes=max(1, int(e * 1.5)), obstacles="dynamic",
        ),
        ProductionScenario(
            name="defense_recovery_clear", stage_id=8,
            description="Stage 8 recovery: our player wins the ball deep and clears/passes out (D2/D3).",
            world_fn=_defense_recovery_world, episodes=e * 2, obstacles="dynamic",
        ),
        ProductionScenario(
            name="defense_mark_open", stage_id=8,
            description="Stage 8 mark validation: lone attacker isolated ~1.1m in front of the "
                        "off-ball defender in open space (clean mark target for gate + vision NN).",
            world_fn=_defense_mark_open_world, episodes=e * 2, obstacles="static",
        ),
    ]


def team_production_scenarios(episodes: int) -> list[ProductionScenario]:
    """LOCKED scenario set + weights for Monday 2026-06-15 demo (§14.21.2 and §14.21.3).

    Weights are encoded by repeating scenarios — the dataset collector iterates the list
    and runs `scenario.episodes` per entry. Repeating means more dataset rows from that
    distribution. This matches the existing `production_scenarios` shape so no
    teacher_production_dataset.py rewrite is required for stage 6+ iteration.

    Stage 6 (warmup 2v0): 3.0 clean + 1.0 static_blocker.
    Stage 7 (adversarial 3v3): 0.5 clean + 1.5 static_pinch + 2.0 dynamic + 3.0 full_dynamic.

    Weighting is integer-scaled by multiplying base `episodes`. For the canonical sprint
    config (episodes=4), this gives:
      - 2v0_clean × 12, 2v0_static_blocker × 4
      - 3v0_clean × 2, 3v0_static_pinch × 6, 3v0_dynamic × 8, 3v3_full_dynamic × 12
    """
    e = max(1, int(episodes))
    return [
        ProductionScenario(
            name="team_2v0_clean_band", stage_id=6,
            description="Stage 6 baseline: 1 GK + 2 RL field, ball in 2.5-3.0m band, no opponents.",
            world_fn=_team_2v0_clean_world, episodes=e * 3, obstacles="none",
        ),
        ProductionScenario(
            name="team_2v0_static_blocker", stage_id=6,
            description="Stage 6 with one static opponent in striker's direct lane — forces pass consideration.",
            world_fn=_team_2v0_static_blocker_world, episodes=e * 1, obstacles="static",
        ),
        ProductionScenario(
            name="team_3v0_clean", stage_id=7,
            description="Stage 7 baseline: 1 GK + 3 RL field, no opponents, single-agent retention reference.",
            world_fn=_team_3v0_clean_world, episodes=max(e // 2, 1), obstacles="none",
        ),
        ProductionScenario(
            name="team_3v0_static_pinch", stage_id=7,
            description="Stage 7 with 2 static opponents pinching the central lane.",
            world_fn=_team_3v0_static_pinch_world, episodes=int(e * 1.5), obstacles="static",
        ),
        ProductionScenario(
            name="team_3v0_dynamic", stage_id=7,
            description="Stage 7 with 2-3 dynamic opponents using BH default behaviour.",
            world_fn=_team_3v0_dynamic_world, episodes=e * 2, obstacles="dynamic",
        ),
        ProductionScenario(
            name="team_3v3_full_dynamic", stage_id=7,
            description=(
                "Monday demo: 1 BH GK + 3 RL field vs full BH (1 GK + 3 field, dynamic). "
                "Highest weight — the only scenario with the exact perceptual conditions of the demo."
            ),
            world_fn=_team_3v3_full_dynamic_world, episodes=e * 3, obstacles="dynamic",
        ),
        # v3 (doc-35) contrastive pairs — matched A/B for teammate-contingent learning.
        # v3.1 TEAMMATE-varying contrast (opponent layout constant) — the flip must come
        # from the teammate dims, not an opponent the policy reads via its own front cone.
        ProductionScenario(
            name="giveandgo_outlet_present", stage_id=7,
            description="v3.1 A: open up-field outlet teammate present → striker PASSES.",
            world_fn=_giveandgo_outlet_present_world, episodes=e * 2, obstacles="none",
        ),
        ProductionScenario(
            name="giveandgo_outlet_absent", stage_id=7,
            description="v3.1 B: outlet teammate pulled deep (no up-field option) → striker DRIBBLES.",
            world_fn=_giveandgo_outlet_absent_world, episodes=e * 2, obstacles="none", is_contrast=True,
        ),
        ProductionScenario(
            name="support_striker_live", stage_id=7,
            description="v3.1 A: striker on the ball → support is a live outlet → HOLDS.",
            world_fn=_support_striker_live_world, episodes=e * 2, obstacles="none",
        ),
        ProductionScenario(
            name="support_striker_far", stage_id=7,
            description="v3.1 B: ball loose, striker far → support REPOSITIONS.",
            world_fn=_support_striker_far_world, episodes=e * 2, obstacles="none", is_contrast=True,
        ),
        ProductionScenario(
            name="pass_forcing", stage_id=7,
            description="v4-S3: striker far from goal + advanced open outlet → pass is the winning move.",
            world_fn=_pass_forcing_world, episodes=e * 3, obstacles="static",
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
