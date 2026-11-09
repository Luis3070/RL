from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np


class GKSkillType(IntEnum):
    stand = 0
    walk_goalie_pose = 1
    observe = 2
    intercept_center = 3
    intercept_lateral = 4
    intercept_low_left = 5
    intercept_low_right = 6
    intercept_jump_left = 7
    intercept_jump_right = 8
    clear = 9
    pass_ = 10
    dribble_out = 11

    @property
    def public_name(self) -> str:
        return "pass" if self is GKSkillType.pass_ else self.name

    @classmethod
    def from_public_name(cls, name: str) -> "GKSkillType":
        return cls["pass_" if name == "pass" else name]


GK_SKILL_ORDER = tuple(skill.public_name for skill in GKSkillType)

# Action vector is [skill_id, target_x, target_y, target_theta, pass_target].
# GKStructuredAction stores targets in field millimeters/radians; as_numpy()
# normalizes those values for the policy target tensor.
GK_ACTION_PARAM_MASKS = {
    "stand": (0.0, 0.0, 0.0, 0.0),
    "walk_goalie_pose": (1.0, 1.0, 1.0, 0.0),
    "observe": (1.0, 1.0, 0.0, 0.0),
    "intercept_center": (0.0, 1.0, 0.0, 0.0),
    "intercept_lateral": (0.0, 1.0, 0.0, 0.0),
    "intercept_low_left": (0.0, 1.0, 0.0, 0.0),
    "intercept_low_right": (0.0, 1.0, 0.0, 0.0),
    "intercept_jump_left": (0.0, 1.0, 0.0, 0.0),
    "intercept_jump_right": (0.0, 1.0, 0.0, 0.0),
    "clear": (0.0, 0.0, 0.0, 0.0),
    # pass_target is currently fixed by the GK bridge/runtime policy; the tanh
    # param head cannot represent raw teammate numbers, so do not optimize it as
    # a continuous residual until GK multiagent passing owns a normalized target.
    "pass": (0.0, 0.0, 0.0, 0.0),
    "dribble_out": (0.0, 1.0, 1.0, 0.0),
}


@dataclass(frozen=True)
class GKStructuredAction:
    skill_type: int
    target_x: float = 0.0
    target_y: float = 0.0
    target_theta: float = 0.0
    pass_target: int = 0

    @property
    def skill(self) -> GKSkillType:
        return GKSkillType(int(self.skill_type))

    @property
    def skill_name(self) -> str:
        return self.skill.public_name

    def as_numpy(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.skill_type),
                float(np.clip(self.target_x / 4500.0, -1.0, 1.0)),
                float(np.clip(self.target_y / 3000.0, -1.0, 1.0)),
                float(np.clip(self.target_theta / np.pi, -1.0, 1.0)),
                float(self.pass_target),
            ],
            dtype=np.float32,
        )


def gk_skill_mask(active_skills: Iterable[str] = GK_SKILL_ORDER) -> np.ndarray:
    active = set(active_skills)
    return np.asarray([1.0 if name in active else 0.0 for name in GK_SKILL_ORDER], dtype=np.float32)
