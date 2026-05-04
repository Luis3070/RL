from __future__ import annotations

from enum import IntEnum


class ObsIndex(IntEnum):
    robot_x = 0
    robot_y = 1
    robot_theta = 2
    ball_rel_x = 3
    ball_rel_y = 4
    ball_end_rel_x = 5
    ball_end_rel_y = 6
    ball_vel_x = 7
    ball_vel_y = 8
    time_since_ball_seen = 9
    time_since_ball_disappeared = 10
    ball_seen_percentage = 11
    ball_consistent_with_game_state = 12
    can_score_now = 13
    shot_quality_no_obstacles = 14
    shot_opening_with_obstacles = 15
    pass_options_count = 16
    nearest_teammate_dist = 17
    nearest_opponent_dist = 18
    nearest_uncertain_obstacle_dist = 19
    nearest_teammate_front_dist = 20
    nearest_opponent_front_dist = 21
    nearest_uncertain_front_dist = 22


OBS_FEATURE_NAMES = tuple(index.name for index in ObsIndex)
OBS_SIZE = len(OBS_FEATURE_NAMES)
