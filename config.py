from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from action import DEFAULT_TRAIN_SKILLS, SKILL_ORDER


ROOT_DIR = Path(__file__).resolve().parent
BHUMAN_DIR = (ROOT_DIR.parent / "SabanaHerons2026").resolve()
SIMROBOT_BIN = BHUMAN_DIR / "Build/Linux/SimRobot/Develop/SimRobot"

TEAM_NUMBER = 24
PLAYER_NUMBER = 1
LOCATION = "field_e"
SCENARIO = "4v4_RL2D"

FIELD_X_HALF = 4500.0
FIELD_Y_HALF = 3000.0
GOAL_X = 4500.0
GOAL_Y_HALF = 750.0
MM_PER_M = 1000.0
DT = 0.033
MAX_EPISODE_STEPS = 400


@dataclass(frozen=True)
class SimRobotSceneConfig:
    profile: str
    scene_path: Path
    robot_name: str = "robot1"
    position_scale: float = -1.0 / MM_PER_M
    theta_offset: float = -math.pi


SIMROBOT_SCENE_PRESETS = {
    "2d": SimRobotSceneConfig(
        profile="2d",
        scene_path=BHUMAN_DIR / "Config/Scenes/RL1v0.ros2d",
        robot_name="robot1",
        position_scale=-1.0 / MM_PER_M,
        theta_offset=-math.pi,
    ),
    "3d": SimRobotSceneConfig(
        profile="3d",
        scene_path=BHUMAN_DIR / "Config/Scenes/RL1v0_3D.ros2",
        robot_name="robot1",
        position_scale=-1.0 / MM_PER_M,
        theta_offset=-(math.pi / 2.0),
    ),
}

SIMROBOT_VISIBLE_SCENE_PRESETS = {
    "2d": SimRobotSceneConfig(
        profile="2d",
        scene_path=BHUMAN_DIR / "Config/Scenes/RLFull.ros2d",
        robot_name="robot1",
        position_scale=-1.0 / MM_PER_M,
        theta_offset=-math.pi,
    ),
    "3d": SimRobotSceneConfig(
        profile="3d",
        scene_path=BHUMAN_DIR / "Config/Scenes/RLFull_3D.ros2",
        robot_name="robot1",
        position_scale=-1.0 / MM_PER_M,
        theta_offset=-(math.pi / 2.0),
    ),
}


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
    obs_mode: str = "corrected"
    frames_per_step: int = 3
    obs_timeout_ms: int = 1000
    simrobot_profile: str = "3d"
    simrobot_scene_path: Path | None = None
    simrobot_robot_name: str | None = None
    simrobot_theta_offset: float | None = None
    simrobot_launch: bool = False
    embedded_qt_platform: str | None = None
    embedded_3d_preflight: bool = True
    simrobot_bin: Path = SIMROBOT_BIN


def resolve_simrobot_scene_config(backend: BackendConfig) -> SimRobotSceneConfig:
    scene_presets = SIMROBOT_VISIBLE_SCENE_PRESETS if backend.mode == "simrobot_visible" else SIMROBOT_SCENE_PRESETS

    if backend.simrobot_profile not in scene_presets:
        valid_profiles = ", ".join(sorted(scene_presets))
        raise ValueError(f"Unsupported SimRobot profile {backend.simrobot_profile!r}. Expected one of: {valid_profiles}")

    scene_cfg = scene_presets[backend.simrobot_profile]
    if backend.simrobot_scene_path is not None:
        scene_cfg = replace(scene_cfg, scene_path=Path(backend.simrobot_scene_path).resolve())
    if backend.simrobot_robot_name is not None:
        scene_cfg = replace(scene_cfg, robot_name=backend.simrobot_robot_name)
    if backend.simrobot_theta_offset is not None:
        scene_cfg = replace(scene_cfg, theta_offset=float(backend.simrobot_theta_offset))
    return scene_cfg


@dataclass(frozen=True)
class EnvConfig:
    seed: int = 42
    max_episode_steps: int = MAX_EPISODE_STEPS
    stage_id: int = 0
    backend: BackendConfig = field(default_factory=BackendConfig)
    active_skill_names: tuple[str, ...] = ("walk", "dribble", "shoot")
    reward_goal_bonus: float = 5.0
    reward_progress_scale: float = 0.003
    reward_step_penalty: float = 0.01


@dataclass(frozen=True)
class WireAuditConfig:
    min_expected_obs_mode_rate: float = 0.99
    max_synthetic_pose_rate: float = 0.25
    max_synthetic_ball_rate: float = 0.40
    max_pose_alignment_warning_rate: float = 0.10
    max_ball_stale_rate: float = 0.10
    max_ball_collapsed_rate: float = 0.10
    max_estimated_mode_corrected_pose_rate: float = 0.01
    max_estimated_mode_corrected_ball_rate: float = 0.01
    max_collision_rate_warning: float = 0.25


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    smoke_steps: int = 32
    trainer_timesteps: int = 20000
    rollout_steps: int = 256
    learning_epochs: int = 4
    mini_batches: int = 4
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    entropy_loss_scale: float = 0.001
    value_loss_scale: float = 1.0
    ratio_clip: float = 0.2
    grad_norm_clip: float = 0.5
    hidden_size: int = 128
    device: str = "cuda:0"
    checkpoint_interval: int = 5000
    eval_interval: int = 5000
    eval_episodes: int = 4
    initial_stage_id: int = 0
    max_stage_id: int = 3
    auto_promote: bool = True
    experiment_name: str = "pybh_mappo_takeover_ppo"
    wire_audit: WireAuditConfig = field(default_factory=WireAuditConfig)


ENV = EnvConfig()
TRAIN = TrainConfig()
