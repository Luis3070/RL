from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from action import SKILL_ORDER
from observation import ObsIndex


ACTIVE_DEFENDER_SKILLS = ("stand", "walk", "pass", "block")
PASS_FRONT_CLEAR_THRESHOLDS = {
    "teammate": 1200.0,
    "opponent": 850.0,
    "uncertain": 1000.0,
}

FIELD_X_HALF = 4500.0
FIELD_Y_HALF = 3000.0
OWN_GOAL_X = -4500.0


def defender_skill_mask_from_obs_np(obs: np.ndarray) -> np.ndarray:
    observations = np.asarray(obs, dtype=np.float32)
    if observations.ndim == 1:
        observations = observations.reshape(1, -1)
    mask = np.zeros((observations.shape[0], len(SKILL_ORDER)), dtype=np.float32)
    for name in ("stand", "walk"):
        mask[:, SKILL_ORDER.index(name)] = 1.0

    pass_armed = observations[:, int(ObsIndex.shoot_armed)] >= 0.5
    engage_armed = observations[:, int(ObsIndex.dribble_armed)] >= 0.5
    traffic_clear = _traffic_clear_np(observations)

    pass_idx = SKILL_ORDER.index("pass")
    block_idx = SKILL_ORDER.index("block")
    mask[pass_armed & traffic_clear, :] = 0.0
    mask[pass_armed & traffic_clear, pass_idx] = 1.0
    mask[engage_armed & ~(pass_armed & traffic_clear), block_idx] = 1.0
    return mask


def defender_skill_mask_from_obs_torch(obs: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros((obs.shape[0], len(SKILL_ORDER)), dtype=obs.dtype, device=obs.device)
    mask[:, SKILL_ORDER.index("stand")] = 1.0
    mask[:, SKILL_ORDER.index("walk")] = 1.0

    pass_armed = obs[:, int(ObsIndex.shoot_armed)] >= 0.5
    engage_armed = obs[:, int(ObsIndex.dribble_armed)] >= 0.5
    traffic_clear = _traffic_clear_torch(obs)
    pass_commit = pass_armed & traffic_clear

    pass_idx = SKILL_ORDER.index("pass")
    block_idx = SKILL_ORDER.index("block")
    mask = torch.where(pass_commit.unsqueeze(1), torch.zeros_like(mask), mask)
    mask[:, pass_idx] = pass_commit.to(mask.dtype)
    mask[:, block_idx] = torch.where(
        engage_armed & ~pass_commit,
        torch.ones_like(mask[:, block_idx]),
        mask[:, block_idx],
    )
    return mask


def masked_defender_logits(skill_logits: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    mask = defender_skill_mask_from_obs_torch(obs).to(dtype=skill_logits.dtype, device=skill_logits.device)
    return skill_logits.masked_fill(mask <= 0.0, -1.0e9)


class DefenderActor(torch.nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None):
        super().__init__()
        self.log_std_parameter = torch.nn.Parameter(torch.full((4,), -0.5))
        self.register_buffer("param_low", torch.full((4,), -1.0))
        self.register_buffer("param_high", torch.full((4,), 1.0))
        active_mask = torch.zeros(len(SKILL_ORDER), dtype=torch.float32)
        for skill in ACTIVE_DEFENDER_SKILLS:
            active_mask[SKILL_ORDER.index(skill)] = 1.0
        self.register_buffer("active_skill_mask", active_mask)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(26, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
        )
        self.skill_head = torch.nn.Linear(128, len(SKILL_ORDER))
        self.param_head = torch.nn.Linear(128, 4)
        if state_dict:
            self.load_state_dict(state_dict, strict=False)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        skill_logits = masked_defender_logits(self.skill_head(h), obs)
        param_mean = torch.tanh(self.param_head(h))
        return skill_logits, param_mean


@dataclass(frozen=True)
class DefenderTeacherAction:
    skill: str
    target_x: float = 0.0
    target_y: float = 0.0
    target_theta: float = 0.0
    pass_target: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                float(SKILL_ORDER.index(self.skill)),
                float(self.target_x),
                float(self.target_y),
                float(self.target_theta),
                float(self.pass_target),
            ],
            dtype=np.float32,
        )


def teacher_action(obs: np.ndarray) -> DefenderTeacherAction:
    row = np.asarray(obs, dtype=np.float32).reshape(-1)
    pass_armed = row[int(ObsIndex.shoot_armed)] >= 0.5
    engage_armed = row[int(ObsIndex.dribble_armed)] >= 0.5
    traffic_clear = bool(_traffic_clear_np(row.reshape(1, -1))[0])
    if pass_armed and traffic_clear:
        return DefenderTeacherAction("pass", pass_target=1.0)
    if engage_armed:
        return DefenderTeacherAction("block", target_y=_signed_block_offset(row))
    if row[int(ObsIndex.time_since_ball_seen)] > 0.45 or row[int(ObsIndex.ball_seen_percentage)] < 0.12:
        return DefenderTeacherAction("stand")
    return DefenderTeacherAction("walk", target_y=_defensive_walk_y(row), target_theta=_face_ball_theta(row))


def _traffic_clear_np(obs: np.ndarray) -> np.ndarray:
    teammate = obs[:, int(ObsIndex.nearest_teammate_front_dist)] * FIELD_X_HALF
    opponent = obs[:, int(ObsIndex.nearest_opponent_front_dist)] * FIELD_X_HALF
    uncertain = obs[:, int(ObsIndex.nearest_uncertain_front_dist)] * FIELD_X_HALF
    return (
        (teammate >= PASS_FRONT_CLEAR_THRESHOLDS["teammate"])
        & (opponent >= PASS_FRONT_CLEAR_THRESHOLDS["opponent"])
        & (uncertain >= PASS_FRONT_CLEAR_THRESHOLDS["uncertain"])
    )


def _traffic_clear_torch(obs: torch.Tensor) -> torch.Tensor:
    teammate = obs[:, int(ObsIndex.nearest_teammate_front_dist)] * FIELD_X_HALF
    opponent = obs[:, int(ObsIndex.nearest_opponent_front_dist)] * FIELD_X_HALF
    uncertain = obs[:, int(ObsIndex.nearest_uncertain_front_dist)] * FIELD_X_HALF
    return (
        (teammate >= PASS_FRONT_CLEAR_THRESHOLDS["teammate"])
        & (opponent >= PASS_FRONT_CLEAR_THRESHOLDS["opponent"])
        & (uncertain >= PASS_FRONT_CLEAR_THRESHOLDS["uncertain"])
    )


def _signed_block_offset(obs: np.ndarray) -> float:
    ball_y = float(obs[int(ObsIndex.ball_rel_y)] * FIELD_Y_HALF)
    return float(np.clip(ball_y / 900.0, -1.0, 1.0))


def _defensive_walk_y(obs: np.ndarray) -> float:
    ball_y = float(obs[int(ObsIndex.ball_rel_y)] * FIELD_Y_HALF)
    return float(np.clip(ball_y / 1500.0, -1.0, 1.0))


def _face_ball_theta(obs: np.ndarray) -> float:
    x = float(obs[int(ObsIndex.ball_rel_x)] * FIELD_X_HALF)
    y = float(obs[int(ObsIndex.ball_rel_y)] * FIELD_Y_HALF)
    return float(np.clip(math.atan2(y, max(x, 1.0)) / math.pi, -1.0, 1.0))
