from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RewardBreakdown:
    progress: float
    goal: float
    step_penalty: float

    @property
    def total(self) -> float:
        return self.progress + self.goal - self.step_penalty


def compute_progress_reward(
    *,
    previous_ball_distance: float,
    current_ball_distance: float,
    goal_scored: bool,
    progress_scale: float,
    goal_bonus: float,
    step_penalty: float,
) -> RewardBreakdown:
    progress = max(0.0, previous_ball_distance - current_ball_distance) * progress_scale
    goal = goal_bonus if goal_scored else 0.0
    return RewardBreakdown(progress=progress, goal=goal, step_penalty=step_penalty)


def goal_scored(ball_x: float, ball_y: float, *, goal_x: float, goal_y_half: float) -> bool:
    return ball_x >= goal_x and abs(ball_y) <= goal_y_half


def ball_distance(obs: np.ndarray) -> float:
    dx = float(obs[0])
    dy = float(obs[1])
    return math.hypot(dx, dy)
