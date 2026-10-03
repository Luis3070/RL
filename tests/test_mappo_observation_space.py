from __future__ import annotations

import math

import numpy as np

from rlfootball.core.config import ACTION
from rlfootball.env.environment import (
    TEAMMATE_ACTIVITY_GOING_TO_BALL,
    TEAMMATE_ACTIVITY_IDLE,
    TEAMMATE_ACTIVITY_KICKING,
    RealSimRobotEnv,
)
from rlfootball.core.observation import OBS_SIZE, ObsIndex
from rlfootball.core.pybh_backend import BackendObservation


def _make_obs(**overrides) -> BackendObservation:
    base = dict(
        ball_x=400.0,
        ball_y=0.0,
        robot_x=0.0,
        robot_y=0.0,
        robot_theta=0.0,
        simrobot_robot_x=0.0,
        simrobot_robot_y=0.0,
        simrobot_robot_theta=0.0,
        frame=1,
        obs_ready=True,
        ball_rel_x=400.0,
        ball_rel_y=0.0,
        ball_end_rel_x=400.0,
        ball_end_rel_y=0.0,
        ball_vel_x=0.0,
        ball_vel_y=0.0,
        time_since_ball_seen=0.0,
        time_since_ball_disappeared=0.0,
        ball_seen_percentage=100.0,
        ball_consistent_with_game_state=True,
        can_score_now=False,
        shot_quality_no_obstacles=0.0,
        shot_opening_with_obstacles=0.0,
        pass_options_count=0.0,
        nearest_teammate_dist=0.0,
        nearest_opponent_dist=0.0,
        nearest_uncertain_obstacle_dist=0.0,
        nearest_teammate_front_dist=0.0,
        nearest_opponent_front_dist=0.0,
        nearest_uncertain_front_dist=0.0,
        teammate_poses=(),
        teammate_ages_ms=(),
        teammate_activities=(),
        global_team_ball_x_mm=400.0,
        global_team_ball_y_mm=0.0,
        global_team_ball_age_ms=48.0,
        global_team_ball_valid=True,
        goal_opponent_x_mm=4500.0,
        goal_opponent_y_mm=0.0,
    )
    base.update(overrides)
    return BackendObservation(**base)


def test_mappo_obs_and_state_spaces_are_47_dimensional():
    env = RealSimRobotEnv()
    try:
        assert env.observation_space("player_1").shape == (OBS_SIZE,)
        assert env.state_space.shape == (OBS_SIZE,)
        assert env.state_spaces["player_1"].shape == (OBS_SIZE,)
        assert env.get_global_state().shape == (OBS_SIZE,)
        assert np.count_nonzero(env.get_global_state()) == 0
    finally:
        env.close()


def test_mappo_state_returns_copy_of_last_obs():
    env = RealSimRobotEnv()
    try:
        env._last_obs = np.arange(OBS_SIZE, dtype=np.float32)
        state = env.state()
        assert np.array_equal(state, env._last_obs)
        state[0] = -999.0
        assert env._last_obs[0] == 0.0
    finally:
        env.close()


def test_mappo_stale_zero_and_slot_alignment_follow_spec():
    env = RealSimRobotEnv()
    try:
        obs = _make_obs(
            teammate_poses=(
                (1000.0, 0.0, 0.0),
                (200.0, 0.0, 0.0),
                (500.0, 100.0, 0.0),
            ),
            teammate_ages_ms=(500.0, 1500.0, 100.0),
            teammate_activities=(
                TEAMMATE_ACTIVITY_KICKING,
                TEAMMATE_ACTIVITY_KICKING,
                TEAMMATE_ACTIVITY_IDLE,
            ),
            global_team_ball_x_mm=1000.0,
            global_team_ball_y_mm=0.0,
        )
        encoded = env._encode_obs(obs)
    finally:
        env.close()

    assert math.isclose(encoded[ObsIndex.nearest_teammate_x_rel], 500.0 / ACTION.x_range, abs_tol=1e-6)
    assert math.isclose(encoded[ObsIndex.nearest_teammate_y_rel], 100.0 / ACTION.y_range, abs_tol=1e-6)
    assert math.isclose(encoded[ObsIndex.second_teammate_x_rel], 1000.0 / ACTION.x_range, abs_tol=1e-6)
    assert math.isclose(encoded[ObsIndex.second_teammate_y_rel], 0.0, abs_tol=1e-6)
    assert encoded[ObsIndex.third_teammate_x_rel] == 0.0
    assert encoded[ObsIndex.third_teammate_y_rel] == 0.0
    assert encoded[ObsIndex.nearest_teammate_engaging_ball] == 0.0
    assert encoded[ObsIndex.second_teammate_engaging_ball] == 1.0
    assert encoded[ObsIndex.third_teammate_engaging_ball] == 0.0


def test_mappo_consensus_bit_uses_team_ball_distance_threshold():
    env = RealSimRobotEnv()
    try:
        close_obs = _make_obs(global_team_ball_x_mm=1100.0, global_team_ball_y_mm=0.0)
        far_obs = _make_obs(global_team_ball_x_mm=1300.0, global_team_ball_y_mm=0.0)
        invalid_obs = _make_obs(global_team_ball_x_mm=1100.0, global_team_ball_y_mm=0.0, global_team_ball_valid=False)
        close_encoded = env._encode_obs(close_obs)
        far_encoded = env._encode_obs(far_obs)
        invalid_encoded = env._encode_obs(invalid_obs)
    finally:
        env.close()

    assert close_encoded[ObsIndex.team_ball_agrees_with_mine] == 1.0
    assert far_encoded[ObsIndex.team_ball_agrees_with_mine] == 0.0
    assert invalid_encoded[ObsIndex.team_ball_agrees_with_mine] == 0.0


def test_mappo_consensus_bit_does_not_depend_on_team_ball_age():
    env = RealSimRobotEnv()
    try:
        fresh_obs = _make_obs(global_team_ball_x_mm=1100.0, global_team_ball_y_mm=0.0, global_team_ball_age_ms=0.0)
        stale_obs = _make_obs(global_team_ball_x_mm=1100.0, global_team_ball_y_mm=0.0, global_team_ball_age_ms=5000.0)
        fresh_encoded = env._encode_obs(fresh_obs)
        stale_encoded = env._encode_obs(stale_obs)
    finally:
        env.close()

    assert fresh_encoded[ObsIndex.team_ball_agrees_with_mine] == 1.0
    assert stale_encoded[ObsIndex.team_ball_agrees_with_mine] == 1.0


def test_mappo_going_to_ball_only_engages_when_near_team_ball():
    env = RealSimRobotEnv()
    try:
        near_obs = _make_obs(
            teammate_poses=((950.0, 0.0, 0.0),),
            teammate_ages_ms=(100.0,),
            teammate_activities=(TEAMMATE_ACTIVITY_GOING_TO_BALL,),
            global_team_ball_x_mm=1000.0,
            global_team_ball_y_mm=0.0,
        )
        far_obs = _make_obs(
            teammate_poses=((50.0, 0.0, 0.0),),
            teammate_ages_ms=(100.0,),
            teammate_activities=(TEAMMATE_ACTIVITY_GOING_TO_BALL,),
            global_team_ball_x_mm=1000.0,
            global_team_ball_y_mm=0.0,
        )
        near_encoded = env._encode_obs(near_obs)
        far_encoded = env._encode_obs(far_obs)
    finally:
        env.close()

    assert near_encoded[ObsIndex.nearest_teammate_engaging_ball] == 1.0
    assert far_encoded[ObsIndex.nearest_teammate_engaging_ball] == 0.0
