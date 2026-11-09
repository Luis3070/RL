from __future__ import annotations

import math
from dataclasses import dataclass

from action import SkillType
from multiagent_teacher import (
    BALL_DIST_MAX_PENALTY_MM,
    BALL_DIST_MIN_PENALTY_MM,
    DECISION_PENALTY_MAX,
    DECISION_PENALTY_MIN,
    TeamRoleHistory,
    TeammateSlot,
    _map_to_range,
    assign_roles,
    compute_open_lane_target,
    smash_or_pass,
    stage_aware_team_teacher_action,
    teammate_slots_from_obs,
    team_teacher_payload,
)
from pybh_backend import BackendObservation


def _approx(actual: float, expected: float, tol: float = 1e-6) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol)


@dataclass(frozen=True)
class _GateLike:
    shoot_armed: bool = False
    dribble_armed: bool = False
    shoot_arm_progress: float = 0.0
    allowed_skills: tuple[str, ...] = ("stand", "walk", "dribble", "shoot")
    pass_armed: bool = False


def _obs(**overrides) -> BackendObservation:
    base = dict(
        ball_x=350.0,
        ball_y=0.0,
        robot_x=0.0,
        robot_y=0.0,
        robot_theta=0.0,
        simrobot_robot_x=0.0,
        simrobot_robot_y=0.0,
        simrobot_robot_theta=0.0,
        frame=1,
        obs_ready=True,
        ball_rel_x=350.0,
        ball_rel_y=0.0,
        ball_end_rel_x=350.0,
        ball_end_rel_y=0.0,
        ball_vel_x=0.0,
        ball_vel_y=0.0,
        time_since_ball_seen=0.0,
        time_since_ball_disappeared=0.0,
        ball_seen_percentage=100.0,
        ball_consistent_with_game_state=True,
        natural_time_since_ball_seen=0.0,
        can_score_now=False,
        shot_quality_no_obstacles=0.0,
        shot_opening_with_obstacles=0.0,
        pass_options_count=0.0,
        nearest_teammate_dist=0.0,
        nearest_opponent_dist=0.0,
        nearest_uncertain_obstacle_dist=0.0,
        nearest_teammate_front_dist=0.0,
        nearest_opponent_front_dist=5000.0,
        nearest_uncertain_front_dist=0.0,
        teammate_poses=(),
        teammate_ages_ms=(),
        teammate_activities=(),
        global_team_ball_x_mm=350.0,
        global_team_ball_y_mm=0.0,
        global_team_ball_age_ms=0.0,
        global_team_ball_valid=True,
        goal_opponent_x_mm=4500.0,
        goal_opponent_y_mm=0.0,
    )
    base.update(overrides)
    return BackendObservation(**base)


def test_smash_or_pass_returns_bhuman_agent_id_and_skips_bad_slots():
    striker = _obs(shot_opening_with_obstacles=0.1, can_score_now=False)
    slots = [
        TeammateSlot(slot_idx=0, agent_id=-1, x_field=3000.0, y_field=0.0, is_gk=False),
        TeammateSlot(slot_idx=1, agent_id=1, x_field=-2500.0, y_field=0.0, is_gk=True),
        TeammateSlot(slot_idx=2, agent_id=4, x_field=2500.0, y_field=400.0, is_gk=False),
    ]

    skill_name, pass_target = smash_or_pass(striker, slots, None, 0.0)

    assert skill_name == "pass"
    assert pass_target == 4


def test_decision_penalty_is_larger_when_ball_is_close():
    close_penalty = _map_to_range(
        BALL_DIST_MIN_PENALTY_MM,
        BALL_DIST_MIN_PENALTY_MM,
        BALL_DIST_MAX_PENALTY_MM,
        DECISION_PENALTY_MAX,
        DECISION_PENALTY_MIN,
    )
    far_penalty = _map_to_range(
        BALL_DIST_MAX_PENALTY_MM,
        BALL_DIST_MIN_PENALTY_MM,
        BALL_DIST_MAX_PENALTY_MM,
        DECISION_PENALTY_MAX,
        DECISION_PENALTY_MIN,
    )

    assert _approx(close_penalty, DECISION_PENALTY_MAX)
    assert _approx(far_penalty, DECISION_PENALTY_MIN)
    assert close_penalty > far_penalty


def test_teacher_striker_pass_override_requires_pass_armed_and_uses_agent_id():
    obs = _obs(
        ball_x=300.0,
        ball_rel_x=300.0,
        shot_opening_with_obstacles=0.1,
        can_score_now=False,
    )
    teammate = TeammateSlot(
        slot_idx=0,
        agent_id=3,
        x_field=2500.0,
        y_field=0.0,
        is_gk=False,
    )

    no_pass_gate = _GateLike(shoot_armed=True, pass_armed=False)
    no_pass = stage_aware_team_teacher_action(
        obs,
        4,
        gate_decision=no_pass_gate,
        role="striker",
        coordination_target=(obs.ball_x, obs.ball_y),
        teammate_slots=[teammate],
        last_pass_target=None,
        last_pass_target_age_ms=0.0,
    )

    pass_gate = _GateLike(shoot_armed=True, pass_armed=True)
    with_pass = stage_aware_team_teacher_action(
        obs,
        4,
        gate_decision=pass_gate,
        role="striker",
        coordination_target=(obs.ball_x, obs.ball_y),
        teammate_slots=[teammate],
        last_pass_target=None,
        last_pass_target_age_ms=0.0,
    )

    assert no_pass.skill_name == "shoot"
    assert with_pass.skill_type == SkillType.pass_
    assert with_pass.pass_target == 3


def test_compute_open_lane_target_returns_field_coordinates_not_skill_name():
    target = compute_open_lane_target(
        _obs(robot_x=100.0, robot_y=200.0),
        ball_field_xy=(0.0, 500.0),
        striker_field_xy=(0.0, 0.0),
    )

    assert isinstance(target, tuple)
    assert len(target) == 2
    assert all(isinstance(value, float) for value in target)


def test_teammate_slots_from_obs_uses_raw_fresh_distance_sorted_poses():
    obs = _obs(
        teammate_poses=(
            (1000.0, 0.0, 0.0),
            (200.0, 0.0, 0.0),
            (500.0, 100.0, 0.0),
            (-2600.0, 0.0, 0.0),
        ),
        teammate_ages_ms=(500.0, 1500.0, 100.0, 50.0),
    )

    slots = teammate_slots_from_obs(obs, {0: 3, 1: 4, 2: 1})

    assert [(slot.slot_idx, slot.agent_id) for slot in slots] == [(0, 3), (1, 4), (2, 1)]
    assert [(slot.x_field, slot.y_field) for slot in slots] == [
        (500.0, 100.0),
        (1000.0, 0.0),
        (-2600.0, 0.0),
    ]
    assert [slot.is_gk for slot in slots] == [False, False, True]


def test_assign_roles_all_stale_uses_milliseconds_without_extra_scaling():
    history = TeamRoleHistory()
    agent_obs = {
        2: _obs(robot_x=0.0, time_since_ball_seen=999.0),
        3: _obs(robot_x=10.0, time_since_ball_seen=1200.0),
    }

    roles = assign_roles(agent_obs, (0.0, 0.0, False), history)

    assert roles[2] == "striker"
    assert roles[3] == "open_support"


def test_team_teacher_payload_residualizes_support_walk_on_coordination_target():
    obs = _obs(robot_x=0.0, robot_y=0.0)

    payload, absolute, residual = team_teacher_payload(
        obs,
        4,
        None,
        gate_decision=_GateLike(),
        role="open_support",
        coordination_target=(1200.0, 800.0),
        teammate_slots=[],
        last_pass_target=None,
        last_pass_target_age_ms=0.0,
    )

    assert absolute.skill_name == "walk"
    assert residual.skill_name == "walk"
    assert _approx(residual.target_y, 0.0)
    assert math.isclose(residual.target_theta, 0.0, abs_tol=1e-6)
    assert payload["role_id"] == 1.0
    assert payload["policy_skill_prob_walk"] == 1.0
