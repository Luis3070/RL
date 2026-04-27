from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, Mapping

import numpy as np


class SkillType(IntEnum):
    stand = 0
    walk = 1
    shoot = 2
    pass_ = 3
    dribble = 4
    block = 5
    mark = 6
    observe = 7

    @property
    def public_name(self) -> str:
        return "pass" if self is SkillType.pass_ else self.name

    @classmethod
    def from_public_name(cls, name: str) -> "SkillType":
        normalized = "pass_" if name == "pass" else name
        return cls[normalized]


SKILL_ORDER = tuple(skill.public_name for skill in SkillType)
DEFAULT_TRAIN_SKILLS = ("stand", "walk", "shoot", "pass", "dribble")


@dataclass(frozen=True)
class StructuredSkillAction:
    skill_type: int
    target_x: float = 0.0
    target_y: float = 0.0
    target_theta: float = 0.0
    pass_target: int = 0

    @property
    def skill(self) -> SkillType:
        return SkillType(int(self.skill_type))

    @property
    def skill_name(self) -> str:
        return self.skill.public_name

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "skill_type": int(self.skill_type),
            "target_x": float(self.target_x),
            "target_y": float(self.target_y),
            "target_theta": float(self.target_theta),
            "pass_target": int(self.pass_target),
        }


def clamp_action(
    action: StructuredSkillAction,
    *,
    x_range: float,
    y_range: float,
    max_pass_target: int,
) -> StructuredSkillAction:
    return StructuredSkillAction(
        skill_type=max(0, min(int(action.skill_type), len(SKILL_ORDER) - 1)),
        target_x=float(np.clip(action.target_x, -x_range, x_range)),
        target_y=float(np.clip(action.target_y, -y_range, y_range)),
        target_theta=float(np.clip(action.target_theta, -math.pi, math.pi)),
        pass_target=max(0, min(int(action.pass_target), max_pass_target)),
    )


def action_from_mapping(payload: Mapping[str, float | int]) -> StructuredSkillAction:
    return StructuredSkillAction(
        skill_type=int(payload["skill_type"]),
        target_x=float(payload.get("target_x", 0.0)),
        target_y=float(payload.get("target_y", 0.0)),
        target_theta=float(payload.get("target_theta", 0.0)),
        pass_target=int(payload.get("pass_target", 0)),
    )


def action_from_numpy(values: np.ndarray) -> StructuredSkillAction:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size != 5:
        raise ValueError(f"Expected flat action with 5 values, got shape {flat.shape}")
    return StructuredSkillAction(
        skill_type=int(round(float(flat[0]))),
        target_x=float(flat[1]),
        target_y=float(flat[2]),
        target_theta=float(flat[3]),
        pass_target=int(round(float(flat[4]))),
    )


def action_to_numpy(action: StructuredSkillAction) -> np.ndarray:
    return np.asarray(
        [
            float(action.skill_type),
            float(action.target_x),
            float(action.target_y),
            float(action.target_theta),
            float(action.pass_target),
        ],
        dtype=np.float32,
    )


def skill_mask(active_skills: Iterable[str]) -> np.ndarray:
    allowed = set(active_skills)
    return np.asarray([1.0 if name in allowed else 0.0 for name in SKILL_ORDER], dtype=np.float32)

