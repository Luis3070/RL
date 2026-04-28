from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from action import DEFAULT_TRAIN_SKILLS, SKILL_ORDER


ROOT_DIR = Path(__file__).resolve().parent
BHUMAN_DIR = (ROOT_DIR.parent / "SabanaHerons2026").resolve()
SIMROBOT_SCENE_PATH = BHUMAN_DIR / "Config/Scenes/RL1v0.ros2d"
SIMROBOT_ROBOT_NAME = "robot1"

TEAM_NUMBER = 24
PLAYER_NUMBER = 1
LOCATION = "field_e"
SCENARIO = "4v4_Scaled"

FIELD_X_HALF = 4500.0
FIELD_Y_HALF = 3000.0
GOAL_X = 4500.0
GOAL_Y_HALF = 750.0
MM_PER_M = 1000.0
DT = 0.033
MAX_EPISODE_STEPS = 400


@dataclass(frozen=True)
class ActionConfig:
    skill_names: tuple[str, ...] = SKILL_ORDER
    train_skill_names: tuple[str, ...] = DEFAULT_TRAIN_SKILLS
    x_range: float = FIELD_X_HALF
    y_range: float = FIELD_Y_HALF
    theta_range: float = math.pi
    max_pass_target: int = 4
    flat_size: int = 5


ACTION = ActionConfig()


@dataclass(frozen=True)
class BackendConfig:
    mode: str = "simrobot"
    frames_per_step: int = 1
    obs_timeout_ms: int = 1000


@dataclass(frozen=True)
class EnvConfig:
    seed: int = 42
    max_episode_steps: int = MAX_EPISODE_STEPS
    backend: BackendConfig = field(default_factory=BackendConfig)
    active_skill_names: tuple[str, ...] = ACTION.train_skill_names
    reward_goal_bonus: float = 5.0
    reward_progress_scale: float = 0.003
    reward_step_penalty: float = 0.01


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    smoke_steps: int = 32
    trainer_timesteps: int = 64
    rollout_steps: int = 16
    learning_epochs: int = 2
    mini_batches: int = 2
    hidden_size: int = 128
    device: str = "cpu"


ENV = EnvConfig()
TRAIN = TrainConfig()
