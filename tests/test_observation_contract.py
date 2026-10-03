from __future__ import annotations

import math

from rlfootball.core.config import ACTION
from rlfootball.env.environment import (
    TEAMMATE_ACTIVITY_GOING_TO_BALL,
    TEAMMATE_ACTIVITY_KICKING,
    RealSimRobotEnv,
)
from rlfootball.core.observation import OBS_SIZE, ObsIndex
from rlfootball.core.pybh_backend import BackendObservation
from rlfootball.core.reward import ball_distance


def _close(value: float, expected: float) -> bool:
    return math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_observation_contract_size_and_reward_indices():
    env = RealSimRobotEnv()
    try:
        obs = BackendObservation(
            ball_x=1000.0,
            ball_y=200.0,
            robot_x=100.0,
            robot_y=50.0,
            robot_theta=0.0,
            simrobot_robot_x=100.0,
            simrobot_robot_y=50.0,
            simrobot_robot_theta=0.0,
            frame=123,
            obs_ready=True,
            ball_rel_x=900.0,
            ball_rel_y=150.0,
            ball_end_rel_x=1100.0,
            ball_end_rel_y=100.0,
            ball_vel_x=300.0,
            ball_vel_y=-150.0,
            time_since_ball_seen=250.0,
            time_since_ball_disappeared=500.0,
            ball_seen_percentage=80.0,
            ball_consistent_with_game_state=True,
            can_score_now=True,
            shot_quality_no_obstacles=0.7,
            shot_opening_with_obstacles=0.4,
            pass_options_count=2.0,
            nearest_teammate_dist=1200.0,
            nearest_opponent_dist=900.0,
            nearest_uncertain_obstacle_dist=3000.0,
            nearest_teammate_front_dist=1300.0,
            nearest_opponent_front_dist=1000.0,
            nearest_uncertain_front_dist=3000.0,
            teammate_poses=((300.0, 100.0, 0.2), (100.0, 600.0, -0.5), (900.0, 50.0, 1.2)),
            teammate_ages_ms=(50.0, 250.0, 1500.0),
            teammate_activities=(
                TEAMMATE_ACTIVITY_GOING_TO_BALL,
                TEAMMATE_ACTIVITY_KICKING,
                TEAMMATE_ACTIVITY_GOING_TO_BALL,
            ),
            global_team_ball_x_mm=320.0,
            global_team_ball_y_mm=120.0,
            global_team_ball_age_ms=300.0,
            global_team_ball_valid=True,
            goal_opponent_x_mm=4500.0,
            goal_opponent_y_mm=0.0,
        )
        encoded = env._encode_obs(obs)
    finally:
        env.close()

    assert encoded.shape == (OBS_SIZE,)
    assert encoded[ObsIndex.ball_consistent_with_game_state] == 1.0
    assert encoded[ObsIndex.can_score_now] == 1.0
    assert _close(encoded[ObsIndex.nearest_teammate_x_rel], 200.0 / ACTION.x_range)
    assert _close(encoded[ObsIndex.nearest_teammate_y_rel], 50.0 / ACTION.y_range)
    assert _close(encoded[ObsIndex.second_teammate_x_rel], 0.0)
    assert _close(encoded[ObsIndex.second_teammate_y_rel], 550.0 / ACTION.y_range)
    assert _close(encoded[ObsIndex.third_teammate_x_rel], 0.0)
    assert _close(encoded[ObsIndex.third_teammate_y_rel], 0.0)
    assert encoded[ObsIndex.nearest_teammate_engaging_ball] == 1.0
    assert encoded[ObsIndex.second_teammate_engaging_ball] == 1.0
    assert encoded[ObsIndex.third_teammate_engaging_ball] == 0.0
    assert _close(encoded[ObsIndex.global_team_ball_x_rel], 220.0 / ACTION.x_range)
    assert _close(encoded[ObsIndex.global_team_ball_y_rel], 70.0 / ACTION.y_range)
    assert _close(encoded[ObsIndex.time_since_team_ball_update], 300.0 / 5000.0)
    assert encoded[ObsIndex.team_ball_agrees_with_mine] == 1.0
    assert _close(encoded[ObsIndex.opponent_goal_x_rel], 4400.0 / ACTION.x_range)
    assert _close(encoded[ObsIndex.opponent_goal_y_rel], -50.0 / ACTION.y_range)
    assert _close(ball_distance(encoded), math.hypot(encoded[ObsIndex.ball_rel_x], encoded[ObsIndex.ball_rel_y]))
