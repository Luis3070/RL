from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from config import BHUMAN_DIR, DT, FIELD_X_HALF, FIELD_Y_HALF, GOAL_X, MAX_EPISODE_STEPS
from pybh_backend import WorldPlayerSpec, WorldSpec, build_world_spec
from reward import RewardWeights

ACTIVE_SKILLS = ("stand", "walk", "dribble", "shoot")
WALK_ONLY_SKILLS = ("stand", "walk")
TAKEOVER_SKILLS = ("stand", "walk", "dribble")
MASKED_SKILLS = ("pass", "mark", "block", "observe")

# Multi-agent skill set: striker may emit pass when smashOrPass fires AND pass_armed.
# mark/block/observe stay masked off this sprint (need opponents in obs and a
# defensive teacher branch). See docs/notes/10-graph-coordination-post-production-roadmap.md §14.4 and §14.26 for the deferred work.
ACTIVE_SKILLS_MA = ("stand", "walk", "dribble", "shoot", "pass")
ACTIVE_SKILLS_MA_DEFENDER = ("stand", "walk", "dribble", "shoot", "pass", "block")


@dataclass(frozen=True)
class StageSpec:
    stage_id: int
    name: str
    description: str
    ball_radius_min_mm: float | None
    ball_radius_max_mm: float | None
    full_field: bool
    obstacle_mode: str
    active_skill_names: tuple[str, ...]
    reward_weights: RewardWeights
    max_episode_steps: int = MAX_EPISODE_STEPS
    dynamic_update_interval_steps: int = 0


@dataclass
class DynamicWorldState:
    stage_id: int
    opponent_positions: np.ndarray
    opponent_velocities: np.ndarray
    ball_pos: np.ndarray
    ball_vel: np.ndarray


def _scene_path(stage_id: int, simrobot_profile: str) -> Path:
    if simrobot_profile == "3d_visible" and stage_id == 6:
        return BHUMAN_DIR / "Config/Scenes/RLMAStage6_3D.ros2"
    if simrobot_profile == "3d_visible" and stage_id == 7:
        return BHUMAN_DIR / "Config/Scenes/RLMAStage7_3D.ros2"
    if simrobot_profile == "3d_visible" and stage_id == 8:
        return BHUMAN_DIR / "Config/Scenes/RLMAStage7_3D.ros2"
    if simrobot_profile == "2d":
        if stage_id >= 2:
            return BHUMAN_DIR / "Config/Scenes/RLFull.ros2d"
        return BHUMAN_DIR / "Config/Scenes/RL1v0.ros2d"
    if simrobot_profile == "3d_visible":
        if stage_id >= 2:
            return BHUMAN_DIR / "Config/Scenes/RLFull_3D.ros2"
        return BHUMAN_DIR / "Config/Scenes/RL1v0_3D_visible.ros2"
    if stage_id >= 2:
        return BHUMAN_DIR / "Config/Scenes/RLFull_3D.ros2"
    return BHUMAN_DIR / "Config/Scenes/RL1v0_3D.ros2"


_STAGE_SPECS = {
    -1: StageSpec(
        stage_id=-1,
        name="stage_0a_walk_approach",
        description="Pre-stage: caminar hacia la pelota sin cambio de skill.",
        ball_radius_min_mm=300.0,
        ball_radius_max_mm=1200.0,
        full_field=False,
        obstacle_mode="none",
        active_skill_names=WALK_ONLY_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=1.2,
            align_ball=0.8,
            approach_ball=0.12,
            good_walk=0.6,
            idle_stand=-0.20,
            walk_persistence=0.10,
            projection_penalty=-0.05,
            skill_switch=-0.02,
            thrash=-0.5,
            timeout=-1.5,
            preferred_skill_match=0.08,
            keep_ball_front=0.12,
            bad_ball_angle=-0.10,
            rescued_ball=-0.40,
            stale_ball=-0.20,
            collapsed_ball=-1.20,
            corrected_ball=-0.30,
            too_close_walk=-0.35,
            lost_trusted_ball=-0.25,
            missed_dribble_handoff=-0.20,
            dribble_skill_intent=0.20,
            shoot_skill_intent=0.0,
        ),
    ),
    0: StageSpec(
        stage_id=0,
        name="stage_0_clean_handoff",
        description="Handoff limpio walk→dribble desde distancia real de aproximación, sin shoot.",
        ball_radius_min_mm=700.0,
        ball_radius_max_mm=1600.0,
        full_field=False,
        obstacle_mode="none",
        active_skill_names=TAKEOVER_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=1.35,
            align_ball=0.9,
            enter_takeover=2.4,
            hold_takeover=0.90,
            approach_ball=0.10,
            good_walk=0.30,
            bad_walk=-0.65,
            idle_stand=-0.35,
            walk_persistence=0.05,
            good_dribble=1.30,
            bad_dribble=-0.85,
            projection_penalty=-0.10,
            skill_switch=-0.05,
            thrash=-0.3,
            goal=2.0,
            timeout=-2.0,
            ball_progress_post_takeover=0.80,
            preferred_skill_match=0.10,
            dribble_handoff=1.60,
            missed_dribble_handoff=-0.45,
            dribble_skill_intent=0.45,
            shoot_skill_intent=0.0,
            keep_ball_front=0.18,
            bad_ball_angle=-0.18,
            rescued_ball=-0.25,
            stale_ball=-0.15,
            collapsed_ball=-2.0,
            corrected_ball=-0.20,
            too_close_walk=-0.60,
            lost_trusted_ball=-0.20,
        ),
    ),
    1: StageSpec(
        stage_id=1,
        name="stage_1_finish_after_handoff",
        description="Cadena walk→dribble→shoot sin obstáculos.",
        ball_radius_min_mm=900.0,
        ball_radius_max_mm=2400.0,
        full_field=False,
        obstacle_mode="none",
        active_skill_names=ACTIVE_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=1.1,
            align_ball=0.75,
            enter_takeover=2.2,
            # Smooth the stage-0 -> stage-1 reward cliff: the old jump (good_dribble
            # 1.3->0.9, bad_dribble -0.85->-1.6, good_walk 0.30->0.45) punished the
            # promoted handoff policy into walk. Keep dribble/holding closer to stage 0.
            hold_takeover=0.6,
            approach_ball=0.08,
            good_walk=0.35,
            bad_walk=-0.8,
            idle_stand=-0.60,
            walk_persistence=0.20,
            good_dribble=1.1,
            bad_dribble=-1.0,
            # Make shoot learnable: bad_shoot -2.2 made any off-zone shoot
            # catastrophic so the agent never explored it (shoot_compat stayed 0 and
            # the stage 1->2 shoot gate was unreachable). Soften the deterrent and
            # raise the good_shoot payoff + shoot intent pull.
            good_shoot=1.5,
            bad_shoot=-0.9,
            projection_penalty=-0.15,
            skill_switch=-0.3,
            thrash=-0.4,
            goal=4.0,
            timeout=-2.5,
            # Drive-to-goal is the production objective but was paid ~1/46th of in-place
            # dribbling (good_dribble 1.1), so the agent dribbled in place and only scored
            # when the reset pre-placed the ball at goal (normal-reset goal_rate ~0.09).
            # Boost the goalward-progress reward to DOMINATE in-place dribbling so the
            # natural walk->dribble->shoot chain emerges. Paired with the hold_takeover
            # progress-gate (reward.py) and the 0.75->1.0 progress clip.
            ball_progress_post_takeover=1.5,
            rescued_ball=-0.30,
            stale_ball=-0.15,
            collapsed_ball=-1.50,
            corrected_ball=-0.25,
            too_close_walk=-0.45,
            lost_trusted_ball=-0.20,
            dribble_handoff=1.1,
            missed_dribble_handoff=-0.25,
            dribble_skill_intent=0.20,
            shoot_skill_intent=0.7,
            missed_shoot=-0.3,
        ),
    ),
    2: StageSpec(
        stage_id=2,
        name="stage_2",
        description="Takeover limpio con clearance y obstáculos estáticos.",
        ball_radius_min_mm=1200.0,
        ball_radius_max_mm=3200.0,
        full_field=False,
        obstacle_mode="static",
        active_skill_names=ACTIVE_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=0.9,
            align_ball=0.6,
            enter_takeover=2.0,
            hold_takeover=0.5,
            approach_ball=0.04,
            good_walk=0.3,
            bad_walk=-0.9,
            idle_stand=-0.45,
            walk_persistence=0.16,
            good_dribble=1.1,
            # Smooth the stage1->2 cliff (was -1.6, vs stage1 -1.0): a harsh jump at
            # promotion re-degenerates the dribble the same way the 0->1 cliff did.
            bad_dribble=-1.2,
            good_shoot=1.4,
            bad_shoot=-2.0,
            projection_penalty=-0.20,
            collision=-3.0,
            clearance_gain=1.0,
            skill_switch=-0.2,
            # Smooth the thrash cliff (was -1.0, vs stage1 -0.4) so the promoted policy
            # is not destabilized at the transition while it is still settling.
            thrash=-0.6,
            goal=4.5,
            timeout=-2.5,
            # Keep drive-to-goal dominant across the transition (see stage 1).
            ball_progress_post_takeover=1.6,
            rescued_ball=-0.45,
            stale_ball=-0.20,
            collapsed_ball=-1.25,
            corrected_ball=-0.35,
            too_close_walk=-0.35,
            lost_trusted_ball=-0.25,
            dribble_handoff=0.50,
            missed_dribble_handoff=-0.20,
            dribble_skill_intent=0.20,
            shoot_skill_intent=0.45,
            missed_shoot=-0.3,
        ),
    ),
    3: StageSpec(
        stage_id=3,
        name="stage_3",
        description="Robustez temporal con obstáculos dinámicos scripted.",
        ball_radius_min_mm=600.0,
        ball_radius_max_mm=None,
        full_field=True,
        obstacle_mode="dynamic",
        active_skill_names=ACTIVE_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=0.7,
            align_ball=0.5,
            enter_takeover=1.8,
            hold_takeover=0.6,
            approach_ball=0.03,
            good_walk=0.2,
            bad_walk=-1.0,
            idle_stand=-0.35,
            walk_persistence=0.12,
            good_dribble=1.4,
            # Smooth the stage2->3 cliff (was -1.7), keeping adjacent stages close.
            bad_dribble=-1.3,
            good_shoot=1.5,
            bad_shoot=-2.0,
            projection_penalty=-0.20,
            collision=-3.2,
            clearance_gain=1.0,
            corridor_keep=1.1,
            # Keep drive-to-goal dominant across the transition (see stage 1).
            ball_progress_post_takeover=1.7,
            skill_switch=-0.25,
            # Smooth the thrash cliff (was -1.2) to match the stage-2 smoothing.
            thrash=-0.8,
            goal=5.0,
            timeout=-2.5,
            rescued_ball=-0.40,
            stale_ball=-0.20,
            collapsed_ball=-1.20,
            corrected_ball=-0.30,
            too_close_walk=-0.30,
            lost_trusted_ball=-0.25,
            dribble_handoff=0.50,
            missed_dribble_handoff=-0.20,
            dribble_skill_intent=0.20,
            shoot_skill_intent=0.50,
            missed_shoot=-0.3,
        ),
        dynamic_update_interval_steps=4,
    ),
    4: StageSpec(
        stage_id=4,
        name="stage_4_production",
        description="Validación productiva: finish real, obstáculos dinámicos y sin colapso de skill.",
        ball_radius_min_mm=600.0,
        ball_radius_max_mm=None,
        full_field=True,
        obstacle_mode="dynamic",
        active_skill_names=ACTIVE_SKILLS,
        reward_weights=RewardWeights(
            progress_ball=0.6,
            align_ball=0.45,
            enter_takeover=1.6,
            hold_takeover=0.5,
            approach_ball=0.02,
            good_walk=0.05,
            bad_walk=-1.0,
            idle_stand=-0.35,
            walk_persistence=0.02,
            good_dribble=1.2,
            bad_dribble=-1.8,
            good_shoot=2.0,
            bad_shoot=-2.0,
            projection_penalty=-0.25,
            collision=-3.0,
            clearance_gain=0.8,
            corridor_keep=1.0,
            ball_progress_post_takeover=1.8,
            skill_switch=-0.25,
            thrash=-1.4,
            goal=7.0,
            timeout=-3.0,
            rescued_ball=-0.40,
            stale_ball=-0.20,
            collapsed_ball=-1.20,
            corrected_ball=-0.30,
            too_close_walk=-0.30,
            lost_trusted_ball=-0.25,
            dribble_handoff=0.45,
            missed_dribble_handoff=-0.80,
            dribble_skill_intent=0.35,
            shoot_skill_intent=0.70,
            missed_shoot=-0.3,
        ),
        dynamic_update_interval_steps=4,
    ),
}


# Multi-agent stages 6 (warmup 2v0) and 7 (adversarial 3v3 vs full BH) — §14.21 LOCKED.
# Reward weights inherit stage-4's production weights; team-specific terms (pass_attempt,
# pass_completed, support_open_hold, team_goal_bonus, role_thrash, teammate_separation)
# are applied via the reward shaper at runtime when role_id is present in the info packet
# rather than baked into RewardWeights — keeps the dataclass backward-compatible.
_STAGE_4_REWARDS = _STAGE_SPECS[4].reward_weights
_STAGE_SPECS[6] = StageSpec(
    stage_id=6,
    name="multiagent_warmup_2v0",
    description="1 BH GK + 2 RL field. Warmup: role one-hot + pass head; minimum opponent exposure.",
    ball_radius_min_mm=600.0,
    ball_radius_max_mm=None,
    full_field=False,
    obstacle_mode="static",
    active_skill_names=ACTIVE_SKILLS_MA,
    reward_weights=_STAGE_4_REWARDS,
    max_episode_steps=400,
    dynamic_update_interval_steps=0,
)
_STAGE_SPECS[7] = StageSpec(
    stage_id=7,
    name="multiagent_adversarial_3v3",
    description="1 BH GK + 3 RL field vs full BH. Monday demo configuration.",
    ball_radius_min_mm=600.0,
    ball_radius_max_mm=None,
    full_field=True,
    obstacle_mode="dynamic",
    active_skill_names=ACTIVE_SKILLS_MA,
    reward_weights=_STAGE_4_REWARDS,
    max_episode_steps=400,
    dynamic_update_interval_steps=4,
)
_STAGE_SPECS[8] = StageSpec(
    stage_id=8,
    name="multiagent_defender_3v3",
    description="1 BH GK + 3 RL field vs defensive stage-8 worlds; block enabled for defensive off-ball role.",
    ball_radius_min_mm=600.0,
    ball_radius_max_mm=None,
    full_field=True,
    obstacle_mode="dynamic",
    active_skill_names=ACTIVE_SKILLS_MA_DEFENDER,
    reward_weights=_STAGE_4_REWARDS,
    max_episode_steps=600,
    dynamic_update_interval_steps=4,
)


def get_stage_spec(stage_id: int) -> StageSpec:
    if stage_id not in _STAGE_SPECS:
        valid = ", ".join(str(k) for k in sorted(_STAGE_SPECS))
        raise ValueError(f"Unsupported stage_id {stage_id!r}. Expected one of: {valid}")
    return _STAGE_SPECS[stage_id]


def get_stage_scene_path(stage_id: int, simrobot_profile: str) -> Path:
    return _scene_path(stage_id, simrobot_profile)


def stage_skill_mask(stage_id: int) -> Dict[str, int]:
    stage = get_stage_spec(stage_id)
    return {
        "stand": 1 if "stand" in stage.active_skill_names else 0,
        "walk": 1 if "walk" in stage.active_skill_names else 0,
        "dribble": 1 if "dribble" in stage.active_skill_names else 0,
        "shoot": 1 if "shoot" in stage.active_skill_names else 0,
        "pass": 1 if "pass" in stage.active_skill_names else 0,
        "block": 1 if "block" in stage.active_skill_names else 0,
        "mark": 1 if "mark" in stage.active_skill_names else 0,
        "observe": 1 if "observe" in stage.active_skill_names else 0,
    }


def _sample_pose_ball_pair(
    rng: np.random.Generator,
    *,
    radius_min_mm: float,
    radius_max_mm: float,
    robot_xy_bounds: tuple[float, float, float, float],
    line_angle_range_deg: float,
    heading_noise_deg: float,
    ball_jitter_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_min, x_max, y_min, y_max = robot_xy_bounds
    radius_min = min(float(radius_min_mm), float(radius_max_mm))
    for _ in range(128):
        robot_x = float(rng.uniform(x_min, x_max))
        robot_y = float(rng.uniform(y_min, y_max))
        radius = float(rng.uniform(radius_min, radius_max_mm))
        line_angle = float(rng.uniform(-math.radians(line_angle_range_deg), math.radians(line_angle_range_deg)))
        jitter_x = float(rng.uniform(-ball_jitter_mm, ball_jitter_mm)) if ball_jitter_mm > 0.0 else 0.0
        jitter_y = float(rng.uniform(-ball_jitter_mm, ball_jitter_mm)) if ball_jitter_mm > 0.0 else 0.0
        ball_x = robot_x + radius * math.cos(line_angle) + jitter_x
        ball_y = robot_y + radius * math.sin(line_angle) + jitter_y
        if abs(ball_x) > FIELD_X_HALF - 200.0 or abs(ball_y) > FIELD_Y_HALF - 200.0:
            continue
        # For the widened stage-0/1 reset regime, vary whether the robot starts
        # aligned to the ball approach line or partly to the goal-ball line.
        goal_heading = math.atan2(-ball_y, GOAL_X - ball_x)
        heading_center = (
            line_angle
            if heading_noise_deg <= 18.0
            else float(rng.uniform(min(line_angle, goal_heading), max(line_angle, goal_heading)))
        )
        robot_theta = heading_center + float(rng.uniform(-math.radians(heading_noise_deg), math.radians(heading_noise_deg)))
        return (
            np.array([ball_x, ball_y], dtype=np.float32),
            np.array([robot_x, robot_y], dtype=np.float32),
            robot_theta,
        )
    raise RuntimeError("Could not sample a valid stage reset within field bounds")


def _sample_full_field_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    for _ in range(128):
        robot_x = float(rng.uniform(-FIELD_X_HALF * 0.65, FIELD_X_HALF * 0.15))
        robot_y = float(rng.uniform(-FIELD_Y_HALF * 0.75, FIELD_Y_HALF * 0.75))
        ball_x = float(rng.uniform(-FIELD_X_HALF * 0.25, FIELD_X_HALF * 0.55))
        ball_y = float(rng.uniform(-FIELD_Y_HALF * 0.75, FIELD_Y_HALF * 0.75))
        delta_x = ball_x - robot_x
        delta_y = ball_y - robot_y
        distance = math.hypot(delta_x, delta_y)
        if distance < 600.0 or distance > 3500.0:
            continue
        robot_theta = math.atan2(delta_y, delta_x) + float(rng.uniform(-math.radians(55.0), math.radians(55.0)))
        return (
            np.array([ball_x, ball_y], dtype=np.float32),
            np.array([robot_x, robot_y], dtype=np.float32),
            robot_theta,
        )
    raise RuntimeError("Could not sample a valid full-field stage reset")


def _clip_field(x: float, y: float) -> tuple[float, float]:
    return (
        float(np.clip(x, -FIELD_X_HALF + 150.0, FIELD_X_HALF - 150.0)),
        float(np.clip(y, -FIELD_Y_HALF + 150.0, FIELD_Y_HALF - 150.0)),
    )


def _support_players_far() -> tuple[WorldPlayerSpec, ...]:
    return (
        WorldPlayerSpec(number=2, x=-FIELD_X_HALF + 500.0, y=-FIELD_Y_HALF + 500.0),
        WorldPlayerSpec(number=3, x=-FIELD_X_HALF + 500.0, y=FIELD_Y_HALF - 500.0),
    )


def _static_obstacles(robot_pos: np.ndarray, ball_pos: np.ndarray) -> tuple[WorldPlayerSpec, ...]:
    direction = ball_pos - robot_pos
    distance = float(np.linalg.norm(direction))
    if distance < 1.0:
        direction = np.array([1.0, 0.0], dtype=np.float32)
        distance = 1.0
    direction = direction / distance
    lateral = np.array([-direction[1], direction[0]], dtype=np.float32)
    midpoint = robot_pos + direction * min(distance * 0.55, 1800.0)
    opp1_x, opp1_y = _clip_field(float(midpoint[0] + lateral[0] * 260.0), float(midpoint[1] + lateral[1] * 260.0))
    opp2_x, opp2_y = _clip_field(float(midpoint[0] - lateral[0] * 260.0), float(midpoint[1] - lateral[1] * 260.0))
    return (
        WorldPlayerSpec(number=1, x=opp1_x, y=opp1_y, theta=math.atan2(-direction[1], -direction[0])),
        WorldPlayerSpec(number=2, x=opp2_x, y=opp2_y, theta=math.atan2(-direction[1], -direction[0])),
    )


def _dynamic_obstacles(robot_pos: np.ndarray, ball_pos: np.ndarray, episode_step: int) -> tuple[WorldPlayerSpec, ...]:
    direction = ball_pos - robot_pos
    distance = float(np.linalg.norm(direction))
    if distance < 1.0:
        direction = np.array([1.0, 0.0], dtype=np.float32)
        distance = 1.0
    direction = direction / distance
    lateral = np.array([-direction[1], direction[0]], dtype=np.float32)
    phase = float(episode_step) * 0.45
    midpoint = robot_pos + direction * min(distance * 0.58, 2200.0)
    lane_shift = 320.0 * math.sin(phase)
    near_ball_shift = 260.0 * math.cos(phase)
    opp1_x, opp1_y = _clip_field(float(midpoint[0] + lateral[0] * lane_shift), float(midpoint[1] + lateral[1] * lane_shift))
    opp2_anchor = ball_pos + direction * -220.0
    opp2_x, opp2_y = _clip_field(
        float(opp2_anchor[0] - lateral[0] * near_ball_shift),
        float(opp2_anchor[1] - lateral[1] * near_ball_shift),
    )
    theta = math.atan2(float(direction[1]), float(direction[0]))
    return (
        WorldPlayerSpec(number=1, x=opp1_x, y=opp1_y, theta=theta + math.pi),
        WorldPlayerSpec(number=2, x=opp2_x, y=opp2_y, theta=theta + math.pi),
    )


def init_dynamic_world_state(stage_id: int, world_spec: WorldSpec, rng: np.random.Generator) -> DynamicWorldState | None:
    stage = get_stage_spec(stage_id)
    if stage.obstacle_mode != "dynamic":
        return None
    if world_spec.opponents:
        opponent_positions = np.asarray([[player.x, player.y] for player in world_spec.opponents], dtype=np.float32)
    else:
        opponent_positions = np.asarray(
            [
                [world_spec.ball_x - 700.0, world_spec.ball_y + 450.0],
                [world_spec.ball_x - 550.0, world_spec.ball_y - 450.0],
            ],
            dtype=np.float32,
        )
    opponent_velocities = rng.uniform(-40.0, 40.0, size=opponent_positions.shape).astype(np.float32)
    ball_vel = rng.uniform(-20.0, 20.0, size=(2,)).astype(np.float32)
    return DynamicWorldState(
        stage_id=stage_id,
        opponent_positions=opponent_positions,
        opponent_velocities=opponent_velocities,
        ball_pos=np.asarray([world_spec.ball_x, world_spec.ball_y], dtype=np.float32),
        ball_vel=ball_vel,
    )


def sample_stage_world_spec(stage_id: int, rng: np.random.Generator) -> WorldSpec:
    # Legacy entry point: preserves the original (narrow) reset distribution for
    # offline callers (teacher_dataset.py, heuristic_agent_experiment.py) so BC
    # dataset generation stays reproducible. The RL training env opts into the
    # widened WS2 regime explicitly via sample_stage_world_spec_with_options(
    # diverse_resets=cfg.diverse_stage_resets).
    return sample_stage_world_spec_with_options(stage_id, rng, diverse_resets=False)


def sample_finish_world_spec(
    stage_id: int,
    rng: np.random.Generator,
    *,
    contrast: bool = False,
) -> WorldSpec:
    """Near-goal finish reset used to make shoot opportunities reachable.

    Normal stage-1/2/3 resets often put the ball several meters from goal, so
    valid shoot states are rare. Finish interleaves start near an open shot;
    contrast variants keep the same region but make the approach angle slightly
    invalid so the teacher can show "do not shoot yet" examples too.
    """
    ball_x = float(rng.uniform(GOAL_X - 950.0, GOAL_X - 520.0))
    ball_y = float(rng.uniform(-360.0, 360.0))
    distance = float(rng.uniform(230.0, 320.0))
    lateral_offset = float(rng.uniform(-55.0, 55.0))
    if contrast:
        lateral_offset = float(rng.choice((-1.0, 1.0)) * rng.uniform(260.0, 420.0))
    robot_x = ball_x - distance
    robot_y = ball_y + lateral_offset
    if contrast:
        # Face mostly toward the goal line, not the ball. The lateral offset
        # then appears as a real bad-angle ball observation for contrastive BC.
        robot_theta = float(rng.uniform(math.radians(-5.0), math.radians(5.0)))
    else:
        robot_theta = math.atan2(ball_y - robot_y, ball_x - robot_x)
    ball_pos = np.asarray(
        [
            np.clip(ball_x, -FIELD_X_HALF + 500.0, FIELD_X_HALF - 300.0),
            np.clip(ball_y, -FIELD_Y_HALF + 500.0, FIELD_Y_HALF - 500.0),
        ],
        dtype=np.float32,
    )
    robot_pos = np.asarray(
        [
            np.clip(robot_x, -FIELD_X_HALF + 500.0, FIELD_X_HALF - 600.0),
            np.clip(robot_y, -FIELD_Y_HALF + 500.0, FIELD_Y_HALF - 500.0),
        ],
        dtype=np.float32,
    )
    lateral = 520.0 if ball_y <= 0.0 else -520.0
    opponents: tuple[WorldPlayerSpec, ...] = ()
    if stage_id >= 2:
        opponents = (
            WorldPlayerSpec(number=1, x=ball_x - 900.0, y=ball_y + lateral, theta=0.0),
            WorldPlayerSpec(number=2, x=ball_x - 250.0, y=ball_y - lateral, theta=0.0),
        )
    if stage_id >= 3:
        opponents = (
            WorldPlayerSpec(number=1, x=ball_x - 1000.0, y=ball_y + lateral, theta=0.0),
            WorldPlayerSpec(number=2, x=ball_x - 750.0, y=ball_y - lateral, theta=0.0),
        )
    return build_world_spec(ball_pos=ball_pos, robot_pos=robot_pos, robot_theta=robot_theta, opponents=opponents)


def sample_finish_sequence_world_spec(stage_id: int, rng: np.random.Generator) -> WorldSpec:
    """Near-goal sequence reset that requires walk->dribble before shooting.

    The ball is still close enough to goal that finishing is reachable, but the
    robot starts outside the takeover envelope. These episodes teach the whole
    approach/handoff/finish chain instead of only the final shoot frame.
    """
    ball_x = float(rng.uniform(GOAL_X - 1050.0, GOAL_X - 620.0))
    ball_y = float(rng.uniform(-380.0, 380.0))
    distance = float(rng.uniform(520.0, 760.0))
    lateral_offset = float(rng.uniform(-140.0, 140.0))
    robot_x = ball_x - distance
    robot_y = ball_y + lateral_offset
    robot_theta = math.atan2(ball_y - robot_y, ball_x - robot_x) + float(
        rng.uniform(math.radians(-8.0), math.radians(8.0))
    )
    ball_pos = np.asarray(
        [
            np.clip(ball_x, -FIELD_X_HALF + 500.0, FIELD_X_HALF - 300.0),
            np.clip(ball_y, -FIELD_Y_HALF + 500.0, FIELD_Y_HALF - 500.0),
        ],
        dtype=np.float32,
    )
    robot_pos = np.asarray(
        [
            np.clip(robot_x, -FIELD_X_HALF + 500.0, FIELD_X_HALF - 600.0),
            np.clip(robot_y, -FIELD_Y_HALF + 500.0, FIELD_Y_HALF - 500.0),
        ],
        dtype=np.float32,
    )
    lateral = 520.0 if ball_y <= 0.0 else -520.0
    opponents: tuple[WorldPlayerSpec, ...] = ()
    if stage_id >= 2:
        opponents = (
            WorldPlayerSpec(number=1, x=ball_x - 900.0, y=ball_y + lateral, theta=0.0),
            WorldPlayerSpec(number=2, x=ball_x - 250.0, y=ball_y - lateral, theta=0.0),
        )
    if stage_id >= 3:
        opponents = (
            WorldPlayerSpec(number=1, x=ball_x - 1000.0, y=ball_y + lateral, theta=0.0),
            WorldPlayerSpec(number=2, x=ball_x - 750.0, y=ball_y - lateral, theta=0.0),
        )
    return build_world_spec(ball_pos=ball_pos, robot_pos=robot_pos, robot_theta=robot_theta, opponents=opponents)


def sample_preshoot_dribble_world_spec(stage_id: int, rng: np.random.Generator) -> WorldSpec:
    """Near-goal takeover reset that is deliberately not yet a shot.

    These states fill the narrow bridge between approach and shooting: the ball
    is close and trusted enough for dribble, but outside shoot-zone by either
    distance or angle. This is the BC antidote to dribble being confused with
    shoot at the edge of the finish window.
    """
    ball_x = float(rng.uniform(GOAL_X - 980.0, GOAL_X - 560.0))
    ball_y = float(rng.uniform(-360.0, 360.0))
    use_angle_boundary = bool(rng.random() < 0.5)
    if use_angle_boundary:
        distance = float(rng.uniform(270.0, 350.0))
        relative_angle = float(rng.choice((-1.0, 1.0)) * rng.uniform(math.radians(27.0), math.radians(33.0)))
    else:
        distance = float(rng.uniform(395.0, 445.0))
        relative_angle = float(rng.choice((-1.0, 1.0)) * rng.uniform(math.radians(4.0), math.radians(18.0)))
    ball_heading = float(rng.uniform(math.radians(-3.0), math.radians(3.0)))
    robot_theta = ball_heading - relative_angle
    robot_x = ball_x - distance * math.cos(ball_heading)
    robot_y = ball_y - distance * math.sin(ball_heading)
    ball_pos = np.asarray(
        [
            np.clip(ball_x, -FIELD_X_HALF + 500.0, FIELD_X_HALF - 300.0),
            np.clip(ball_y, -FIELD_Y_HALF + 500.0, FIELD_Y_HALF - 500.0),
        ],
        dtype=np.float32,
    )
    robot_pos = np.asarray(
        [
            np.clip(robot_x, -FIELD_X_HALF + 500.0, FIELD_X_HALF - 600.0),
            np.clip(robot_y, -FIELD_Y_HALF + 500.0, FIELD_Y_HALF - 500.0),
        ],
        dtype=np.float32,
    )
    lateral = 520.0 if ball_y <= 0.0 else -520.0
    opponents: tuple[WorldPlayerSpec, ...] = ()
    if stage_id >= 2:
        opponents = (
            WorldPlayerSpec(number=1, x=ball_x - 900.0, y=ball_y + lateral, theta=0.0),
            WorldPlayerSpec(number=2, x=ball_x - 250.0, y=ball_y - lateral, theta=0.0),
        )
    if stage_id >= 3:
        opponents = (
            WorldPlayerSpec(number=1, x=ball_x - 1000.0, y=ball_y + lateral, theta=0.0),
            WorldPlayerSpec(number=2, x=ball_x - 750.0, y=ball_y - lateral, theta=0.0),
        )
    return build_world_spec(ball_pos=ball_pos, robot_pos=robot_pos, robot_theta=robot_theta, opponents=opponents)


# --- Drive curriculum (stage 1/2) -------------------------------------------------
# The natural walk->dribble->drive->shoot chain only completes when the ball starts
# close enough to goal that the episode budget allows driving it in. The legacy stage-1/2
# resets placed the ball by ROBOT-to-ball distance, leaving it 3-6 m from goal where the
# chain truncates (the teacher drives the ball ~10-30mm/dribble-step but runs out of steps,
# so neither the teacher demos nor the agent ever complete it). These resets instead control
# BALL-to-GOAL distance, near-biased so most episodes are completable; widen via
# RL_DRIVE_GOAL_MAX_MM as competence rises (start moderate ~1.0-1.8m, grow toward ~5.5m).
DRIVE_GOAL_MIN_MM = float(os.getenv("RL_DRIVE_GOAL_MIN_MM", "1000.0"))
DRIVE_GOAL_MAX_MM = float(os.getenv("RL_DRIVE_GOAL_MAX_MM", "1800.0"))
DRIVE_NEAR_BIAS = float(os.getenv("RL_DRIVE_NEAR_BIAS", "1.6"))  # >1 => bias toward shorter (completable)


def _sample_drive_curriculum_pair(
    rng: np.random.Generator, *, diverse_resets: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    goal_dist = DRIVE_GOAL_MIN_MM + (DRIVE_GOAL_MAX_MM - DRIVE_GOAL_MIN_MM) * float(rng.random() ** DRIVE_NEAR_BIAS)
    ball_x = GOAL_X - goal_dist
    ball_y = float(rng.uniform(-700.0, 700.0)) if diverse_resets else float(rng.uniform(-300.0, 300.0))
    # Robot a real walk behind the ball, FACING it: a clean approach establishes natural
    # ball detection from step 1 (kills the stochastic trust dropout the teacher sweep
    # showed) and provides the walk leg of the chain.
    approach = float(rng.uniform(600.0, 1500.0)) if diverse_resets else float(rng.uniform(650.0, 1050.0))
    approach_angle = (
        float(rng.uniform(math.radians(-35.0), math.radians(35.0)))
        if diverse_resets
        else float(rng.uniform(math.radians(-15.0), math.radians(15.0)))
    )
    robot_x = ball_x - approach * math.cos(approach_angle)
    robot_y = ball_y - approach * math.sin(approach_angle)
    heading_noise = math.radians(20.0 if diverse_resets else 8.0)
    robot_theta = math.atan2(ball_y - robot_y, ball_x - robot_x) + float(rng.uniform(-heading_noise, heading_noise))
    ball_pos = np.asarray(
        [
            np.clip(ball_x, -FIELD_X_HALF + 500.0, FIELD_X_HALF - 300.0),
            np.clip(ball_y, -FIELD_Y_HALF + 500.0, FIELD_Y_HALF - 500.0),
        ],
        dtype=np.float32,
    )
    robot_pos = np.asarray(
        [
            np.clip(robot_x, -FIELD_X_HALF + 500.0, FIELD_X_HALF - 600.0),
            np.clip(robot_y, -FIELD_Y_HALF + 500.0, FIELD_Y_HALF - 500.0),
        ],
        dtype=np.float32,
    )
    return ball_pos, robot_pos, robot_theta


def sample_stage_world_spec_with_options(
    stage_id: int,
    rng: np.random.Generator,
    *,
    diverse_resets: bool = True,
) -> WorldSpec:
    stage = get_stage_spec(stage_id)
    if stage.stage_id == -1:
        ball_pos, robot_pos, robot_theta = _sample_pose_ball_pair(
            rng,
            radius_min_mm=stage.ball_radius_min_mm or max(150.0, (stage.ball_radius_max_mm or 1200.0) * 0.25),
            radius_max_mm=stage.ball_radius_max_mm or 1200.0,
            robot_xy_bounds=(-1700.0, -450.0, -650.0, 650.0),
            line_angle_range_deg=12.0,
            heading_noise_deg=8.0,
        )
        return build_world_spec(ball_pos=ball_pos, robot_pos=robot_pos, robot_theta=robot_theta)
    if stage.stage_id == 0:
        ball_pos, robot_pos, robot_theta = _sample_pose_ball_pair(
            rng,
            radius_min_mm=stage.ball_radius_min_mm or 700.0,
            radius_max_mm=stage.ball_radius_max_mm or 1600.0,
            robot_xy_bounds=(-1500.0, -500.0, -550.0, 550.0),
            line_angle_range_deg=42.0 if diverse_resets else 18.0,
            heading_noise_deg=35.0 if diverse_resets else 12.0,
            ball_jitter_mm=80.0 if diverse_resets else 0.0,
        )
        return build_world_spec(ball_pos=ball_pos, robot_pos=robot_pos, robot_theta=robot_theta)
    if stage.stage_id == 1:
        # Drive curriculum: ball at a moderate (near-biased) BALL-to-goal distance with the
        # robot a walk behind it facing it, so the full walk->dribble->drive->shoot chain is
        # completable within the budget. Replaces the legacy robot-to-ball radius reset that
        # left the ball 3-6 m from goal (chain truncated -> agent learned dribble-in-place).
        ball_pos, robot_pos, robot_theta = _sample_drive_curriculum_pair(rng, diverse_resets=diverse_resets)
        return build_world_spec(ball_pos=ball_pos, robot_pos=robot_pos, robot_theta=robot_theta)
    if stage.stage_id == 2:
        # Same drive curriculum, plus static obstacles between robot and ball/goal.
        ball_pos, robot_pos, robot_theta = _sample_drive_curriculum_pair(rng, diverse_resets=diverse_resets)
        return build_world_spec(
            ball_pos=ball_pos,
            robot_pos=robot_pos,
            robot_theta=robot_theta,
            teammates=_support_players_far(),
            opponents=_static_obstacles(robot_pos, ball_pos),
        )
    if stage.stage_id in (3, 4):
        ball_pos, robot_pos, robot_theta = _sample_full_field_pair(rng)
        return build_world_spec(
            ball_pos=ball_pos,
            robot_pos=robot_pos,
            robot_theta=robot_theta,
            teammates=_support_players_far(),
            opponents=_dynamic_obstacles(robot_pos, ball_pos, episode_step=0),
        )
    raise ValueError(f"Unsupported stage_id {stage_id!r}")


def _limit_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-6:
        return vector
    return vector * float(max_norm / norm)


def _natural_dynamic_world_update(
    state: DynamicWorldState,
    *,
    stage_id: int,
    episode_step: int,
    robot_x: float,
    robot_y: float,
    ball_x: float,
    ball_y: float,
    interval_steps: int,
) -> Dict[str, object]:
    dt_s = max(DT, DT * max(1, interval_steps))
    robot_pos = np.asarray([robot_x, robot_y], dtype=np.float32)
    observed_ball_pos = np.asarray([ball_x, ball_y], dtype=np.float32)
    state.ball_pos = 0.85 * state.ball_pos + 0.15 * observed_ball_pos

    direction = state.ball_pos - robot_pos
    distance = float(np.linalg.norm(direction))
    if distance < 1.0:
        direction = np.asarray([1.0, 0.0], dtype=np.float32)
        distance = 1.0
    direction = direction / distance
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float32)

    base_targets = np.asarray(
        [
            robot_pos + direction * min(distance * 0.55, 1900.0) + lateral * 420.0,
            state.ball_pos - direction * 520.0 - lateral * 430.0,
        ],
        dtype=np.float32,
    )
    phase = float(state.stage_id * 0.73 + interval_steps * 0.11)
    sway = np.asarray(
        [
            lateral * (170.0 * math.sin(phase + episode_step * 0.035)),
            lateral * (140.0 * math.cos(phase + episode_step * 0.029)),
        ],
        dtype=np.float32,
    )
    targets = base_targets + sway
    if stage_id == 8:
        max_opp_speed_mm_s = float(os.getenv("RL_MA_STAGE8_OPP_SPEED_MM_S", "180.0"))
    else:
        max_opp_speed_mm_s = 420.0 if stage_id >= 4 else 330.0
    max_opp_step = max_opp_speed_mm_s * dt_s
    for index in range(min(len(state.opponent_positions), len(targets))):
        delta = _limit_vector(targets[index] - state.opponent_positions[index], max_opp_step)
        state.opponent_velocities[index] = delta / dt_s
        state.opponent_positions[index] = state.opponent_positions[index] + delta
        state.opponent_positions[index, 0], state.opponent_positions[index, 1] = _clip_field(
            float(state.opponent_positions[index, 0]),
            float(state.opponent_positions[index, 1]),
        )

    ball_target = observed_ball_pos + direction * (35.0 * math.sin(episode_step * 0.021)) + lateral * (
        55.0 * math.sin(episode_step * 0.017)
    )
    if stage_id == 8:
        max_ball_speed_mm_s = float(os.getenv("RL_MA_STAGE8_BALL_SPEED_MM_S", "60.0"))
    else:
        max_ball_speed_mm_s = 120.0 if stage_id >= 4 else 80.0
    ball_delta = _limit_vector(ball_target - state.ball_pos, max_ball_speed_mm_s * dt_s)
    state.ball_vel = ball_delta / dt_s
    state.ball_pos = state.ball_pos + ball_delta
    state.ball_pos[0], state.ball_pos[1] = _clip_field(float(state.ball_pos[0]), float(state.ball_pos[1]))

    opponents = []
    for index, pos in enumerate(state.opponent_positions[:2], start=1):
        to_robot = robot_pos - pos
        theta = math.atan2(float(to_robot[1]), float(to_robot[0]))
        opponents.append(WorldPlayerSpec(number=index, x=float(pos[0]), y=float(pos[1]), theta=theta))
    return {"opponents": tuple(opponents), "ball_pos": state.ball_pos.copy()}


def get_dynamic_world_update(
    stage_id: int,
    episode_step: int,
    robot_x: float,
    robot_y: float,
    ball_x: float,
    ball_y: float,
    dynamic_state: DynamicWorldState | None = None,
) -> Dict[str, object] | None:
    stage = get_stage_spec(stage_id)
    if stage_id == 8 and os.getenv("RL_MA_STAGE8_DYNAMIC_WORLD_UPDATES", "1") in {"0", "false", "False", "no", "NO"}:
        return None
    if stage.obstacle_mode != "dynamic":
        return None
    if stage.dynamic_update_interval_steps <= 0 or episode_step % stage.dynamic_update_interval_steps != 0:
        return None
    if dynamic_state is not None:
        return _natural_dynamic_world_update(
            dynamic_state,
            stage_id=stage_id,
            episode_step=episode_step,
            robot_x=robot_x,
            robot_y=robot_y,
            ball_x=ball_x,
            ball_y=ball_y,
            interval_steps=stage.dynamic_update_interval_steps,
        )
    opponents = _dynamic_obstacles(
        np.array([robot_x, robot_y], dtype=np.float32),
        np.array([ball_x, ball_y], dtype=np.float32),
        episode_step=episode_step,
    )
    return {"opponents": opponents}


def _metric(metrics: Mapping[str, float], name: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _policy_incompatible_takeover_rate(metrics: Mapping[str, float], default: float = 1.0) -> float:
    return _metric(
        metrics,
        "policy_incompatible_takeover_rate",
        _metric(metrics, "incompatible_takeover_rate", default),
    )


def _wire_is_healthy(metrics: Mapping[str, float]) -> bool:
    return str(metrics.get("wire_audit_status", "PASS")) != "FAIL"


def _ball_semantics_ok(
    metrics: Mapping[str, float],
    *,
    trusted_min: float,
    rescued_max: float,
    confidence_min: float,
) -> bool:
    return (
        _metric(metrics, "ball_visible_rate") >= 0.90
        and _metric(metrics, "ball_trusted_rate") >= trusted_min
        and _metric(metrics, "ball_rescued_rate") <= rescued_max
        and _metric(metrics, "ball_confidence_score_mean") >= confidence_min
    )


def _behavior_is_stable(metrics: Mapping[str, float], *, thrash_max: float, collision_max: float) -> bool:
    return (
        _wire_is_healthy(metrics)
        and _metric(metrics, "thrash_rate", 1.0) <= thrash_max
        and _metric(metrics, "collision_rate", 1.0) <= collision_max
        and _metric(metrics, "pose_jump_rate", 0.0) <= 0.03
    )


def _skill_not_collapsed(
    metrics: Mapping[str, float],
    *,
    dribble_prob_max: float,
    dribble_rate_max: float,
) -> bool:
    return (
        _metric(metrics, "policy_skill_prob_dribble_mean", 0.0) <= dribble_prob_max
        and _metric(metrics, "policy_dribble_rate", 0.0) <= dribble_rate_max
    )


def _shoot_finish_ok(metrics: Mapping[str, float], *, min_rate: float) -> bool:
    if "shoot_finish_rate" in metrics:
        return _metric(metrics, "shoot_finish_rate", 0.0) >= min_rate
    # Backward-compatible fallback for older telemetry before shoot_finish existed.
    return _metric(metrics, "shoot_compat_rate", 0.0) >= min_rate


def promote_stage(stage_id: int, metrics: Mapping[str, float], history: Mapping[int, Mapping[str, float]]) -> bool:
    if stage_id == -1:
        return (
            _ball_semantics_ok(metrics, trusted_min=0.55, rescued_max=0.40, confidence_min=0.48)
            and _behavior_is_stable(metrics, thrash_max=0.05, collision_max=0.10)
            and _metric(metrics, "episode_reward_mean", -999.0) >= 0.10
            and _metric(metrics, "takeover_success_rate") >= 0.80
            and _metric(metrics, "mean_time_to_takeover", float("inf")) <= 70.0
            and _metric(metrics, "fov_perception_failure_rate", 0.0) <= 0.20
        )
    if stage_id == 0:
        return (
            _ball_semantics_ok(metrics, trusted_min=0.65, rescued_max=0.30, confidence_min=0.55)
            and _behavior_is_stable(metrics, thrash_max=0.08, collision_max=0.10)
            and _metric(metrics, "episode_reward_mean", -999.0) >= -0.25
            and _metric(metrics, "stability_score") >= 0.50
            and _metric(metrics, "takeover_success_rate") >= 0.75
            and _metric(metrics, "mean_time_to_takeover", float("inf")) <= 95.0
            # Aligned with the de-brittled clean_takeover gate (environment.py allows
            # incompat <= 0.15): an agent already producing "clean" takeovers by that
            # definition should be promotable. Was 0.12, which blocked agents hovering
            # at ~0.13 despite clean_takeover_rate of 0.75.
            and _policy_incompatible_takeover_rate(metrics) <= 0.15
            and _metric(metrics, "dribble_compat_rate") >= 0.55
            and _metric(metrics, "bad_walk_rate", 1.0) <= 0.45
            and _metric(metrics, "handoff_override_rate", 1.0) <= 0.08
            and _metric(metrics, "target_repair_rate", 1.0) <= 0.20
            and (
                _metric(metrics, "policy_walk_rate") >= 0.02
                or _metric(metrics, "policy_skill_prob_walk_mean") >= 0.02
            )
            and _skill_not_collapsed(metrics, dribble_prob_max=0.98, dribble_rate_max=0.985)
            and _metric(metrics, "fov_perception_failure_rate", 0.0) <= 0.10
            and str(metrics.get("wire_audit_status", "PASS")) == "PASS"
        )
    if stage_id == 1:
        return (
            _ball_semantics_ok(metrics, trusted_min=0.58, rescued_max=0.38, confidence_min=0.48)
            and _behavior_is_stable(metrics, thrash_max=0.10, collision_max=0.10)
            and _metric(metrics, "episode_reward_mean", -999.0) >= -0.20
            and (
                _metric(metrics, "clean_takeover_rate") >= 0.70
                or (
                    _metric(metrics, "takeover_success_rate") >= 0.80
                    and _metric(metrics, "incompatible_takeover_rate", 1.0) <= 0.03
                    and _policy_incompatible_takeover_rate(metrics) <= 0.03
                )
            )
            and _metric(metrics, "clean_finish_rate") >= 0.60
            and _shoot_finish_ok(metrics, min_rate=0.10)
            and _metric(metrics, "takeover_success_rate") >= 0.80
            and _metric(metrics, "dribble_compat_rate") >= 0.65
            and _metric(metrics, "bad_dribble_rate", 1.0) <= 0.20
            and _metric(metrics, "policy_stand_rate", 1.0) <= 0.15
            and _metric(metrics, "target_repair_rate", 1.0) <= 0.15
            and _metric(metrics, "policy_target_error_mean", 9999.0) <= 520.0
            and (
                _metric(metrics, "shoot_compat_rate") >= 0.10
                or _metric(metrics, "policy_shoot_rate") > 0.02
                or _metric(metrics, "policy_skill_prob_shoot_mean") > 0.02
            )
            and _skill_not_collapsed(metrics, dribble_prob_max=0.95, dribble_rate_max=0.96)
        )
    if stage_id == 2:
        return (
            _ball_semantics_ok(metrics, trusted_min=0.55, rescued_max=0.45, confidence_min=0.45)
            and _behavior_is_stable(metrics, thrash_max=0.12, collision_max=0.12)
            and _metric(metrics, "episode_reward_mean", -999.0) >= -0.25
            and (
                _metric(metrics, "clean_takeover_rate") >= 0.50
                or (
                    _metric(metrics, "takeover_success_rate") >= 0.75
                    and _metric(metrics, "incompatible_takeover_rate", 1.0) <= 0.05
                    and _policy_incompatible_takeover_rate(metrics) <= 0.05
                )
            )
            and _metric(metrics, "clean_finish_rate") >= 0.40
            and _shoot_finish_ok(metrics, min_rate=0.05)
            and _metric(metrics, "target_repair_rate", 1.0) <= 0.10
            and _metric(metrics, "policy_target_error_mean", 9999.0) <= 480.0
            and _skill_not_collapsed(metrics, dribble_prob_max=0.92, dribble_rate_max=0.94)
            and (
                _metric(metrics, "obstacle_clearance_success") >= 0.25
                or _metric(metrics, "clearance_positive_steps") > 0.0
                or _metric(metrics, "stability_score") >= 0.42
            )
        )
    if stage_id == 3:
        stage2 = history.get(2, {})
        return (
            _ball_semantics_ok(metrics, trusted_min=0.55, rescued_max=0.45, confidence_min=0.45)
            and _behavior_is_stable(metrics, thrash_max=0.12, collision_max=0.12)
            and (
                _metric(metrics, "clean_takeover_rate") >= 0.50
                or (
                    _metric(metrics, "takeover_success_rate") >= 0.75
                    and _metric(metrics, "incompatible_takeover_rate", 1.0) <= 0.05
                    and _policy_incompatible_takeover_rate(metrics) <= 0.05
                )
            )
            and _metric(metrics, "clean_finish_rate") >= 0.40
            and _shoot_finish_ok(metrics, min_rate=0.05)
            and _metric(metrics, "dynamic_obstacle_success") >= 0.20
            and _metric(metrics, "target_repair_rate", 1.0) <= 0.08
            and _metric(metrics, "policy_target_error_mean", 9999.0) <= 450.0
            and _metric(metrics, "stability_score") >= _metric(stage2, "stability_score", 0.0) - 0.05
            and _skill_not_collapsed(metrics, dribble_prob_max=0.92, dribble_rate_max=0.94)
        )
    if stage_id == 4:
        return (
            _ball_semantics_ok(metrics, trusted_min=0.55, rescued_max=0.45, confidence_min=0.45)
            and _behavior_is_stable(metrics, thrash_max=0.08, collision_max=0.08)
            and _metric(metrics, "clean_episode_rate") >= 0.50
            and _metric(metrics, "clean_finish_rate") >= 0.70
            and (
                _metric(metrics, "goal_rate") >= 0.10
                or _metric(metrics, "ball_progress_post_takeover_mean") >= 0.02
            )
            and _metric(metrics, "handoff_override_rate", 1.0) <= 0.01
            and _metric(metrics, "skill_projection_rate", 1.0) <= 0.01
            and _policy_incompatible_takeover_rate(metrics) <= 0.03
            and _metric(metrics, "policy_shoot_rate") >= 0.02
            and _skill_not_collapsed(metrics, dribble_prob_max=0.88, dribble_rate_max=0.90)
        )
    return False
