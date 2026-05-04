from __future__ import annotations

import math

from environment import RealSimRobotEnv
from observation import OBS_SIZE, ObsIndex
from pybh_backend import BackendObservation
from reward import ball_distance


def test_observation_contract_size_and_reward_indices():
    env = RealSimRobotEnv()
    try:
        obs = BackendObservation(
            ball_x=1000.0,
            ball_y=200.0,
            robot_x=100.0,
            robot_y=50.0,
            robot_theta=0.25,
            frame=123,
            obs_ready=True,
            sim_enabled=True,
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
        )
        encoded = env._encode_obs(obs)
    finally:
        env.close()

    assert encoded.shape == (OBS_SIZE,)
    assert encoded[ObsIndex.ball_consistent_with_game_state] == 1.0
    assert encoded[ObsIndex.can_score_now] == 1.0
    assert math.isclose(ball_distance(encoded), math.hypot(encoded[ObsIndex.ball_rel_x], encoded[ObsIndex.ball_rel_y]))
