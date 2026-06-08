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
    # Skill-gate state baked into the obs (Design A). The per-step forbid-invalid
    # mask is derived from these columns, so it is a pure function of the stored
    # observation and MAPPO's update-time log-prob recompute reproduces it exactly.
    shoot_armed = 23
    dribble_armed = 24
    shoot_arm_progress = 25
    nearest_teammate_x_rel = 26
    nearest_teammate_y_rel = 27
    second_teammate_x_rel = 28
    second_teammate_y_rel = 29
    third_teammate_x_rel = 30
    third_teammate_y_rel = 31
    global_team_ball_x_rel = 32
    global_team_ball_y_rel = 33
    time_since_team_ball_update = 34
    nearest_teammate_engaging_ball = 35
    second_teammate_engaging_ball = 36
    third_teammate_engaging_ball = 37
    team_ball_agrees_with_mine = 38
    opponent_goal_x_rel = 39
    opponent_goal_y_rel = 40


OBS_FEATURE_NAMES = tuple(index.name for index in ObsIndex)
OBS_SIZE = len(OBS_FEATURE_NAMES)