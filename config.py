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
    # ci 30: camera rendered at 30 fps sim-time. Safe baseline — ci 15 caused
    # obs_timeout_ms=1000 failures (67ms delay > timeout window).
    # Only valid for simrobot_visible (separate subprocess with real GL context).
    "3d_visible": SimRobotSceneConfig(
        profile="3d",
        scene_path=BHUMAN_DIR / "Config/Scenes/RL1v0_3D_visible.ros2",
        robot_name="robot1",
        position_scale=-1.0 / MM_PER_M,
        theta_offset=-(math.pi / 2.0),
    ),
    # H2 experiment: ci 60 halves the camera frame period (33ms → 17ms),
    # cutting worst-case obs wait from ~61ms to ~34ms → target ~29 steps/sec.
    # Risk: 2× perception CPU load. Profile with RL_PROFILE_STEP=1 to confirm.
    "3d_visible_ci60": SimRobotSceneConfig(
        profile="3d",
        scene_path=BHUMAN_DIR / "Config/Scenes/RL1v0_3D_visible_ci60.ros2",
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
    residual_x_range: float = 650.0
    residual_y_range: float = 450.0
    residual_theta_range: float = 0.75
    max_pass_target: int = 4
    flat_size: int = 5


ACTION = ActionConfig()


@dataclass(frozen=True)
class BackendConfig:
    mode: str = "simrobot"
    obs_mode: str = "corrected"
    player_number: int = PLAYER_NUMBER
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
    parallel_player_numbers: tuple[int, ...] = (PLAYER_NUMBER,)
    active_skill_names: tuple[str, ...] = ("walk", "dribble", "shoot")
    enable_handoff_override: bool = False
    diverse_stage_resets: bool = True
    stage0_finish_interleave_probability: float = 0.0
    post_takeover_progress_enabled: bool = True
    post_takeover_progress_window_steps: int = 120
    reward_goal_bonus: float = 5.0
    reward_progress_scale: float = 0.003
    reward_step_penalty: float = 0.01


@dataclass(frozen=True)
class WireAuditConfig:
    min_expected_obs_mode_rate: float = 0.99
    # In 3D headless (calculateImage=False) the C++ bridge always exports SimRobot
    # ground-truth pose because visual self-localization is unavailable → pose is
    # structurally synthetic; 1.0 is the correct ceiling for this profile.
    max_synthetic_pose_rate: float = 1.0
    # Ball is synthetic during the warmup window (Python) and whenever staleBall=True
    # (C++).  In 3D headless both phases produce correct position values — the
    # synthetic flag is a delivery mechanism, not inaccurate data.
    max_synthetic_ball_rate: float = 0.70
    max_pose_alignment_warning_rate: float = 0.10
    # Without a camera seenPercentage=0 always; ball is stale after ~1 s in every
    # episode.  C++ synthetic injection provides the correct position when stale.
    max_ball_stale_rate: float = 0.90
    max_ball_collapsed_rate: float = 0.10
    max_estimated_mode_corrected_pose_rate: float = 0.01
    max_estimated_mode_corrected_ball_rate: float = 0.01
    # Calibrated against the current N=6 visible parallel pipeline:
    # healthy evals sit roughly at trusted≈0.59-0.75, rescued≈0.15-0.35,
    # confidence≈0.53-0.58.  Short random-policy smoke runs can be much worse,
    # so FAIL is reserved for clearly pathological ball semantics.
    min_visible_ball_visible_rate: float = 0.90
    max_visible_ball_rescued_rate: float = 0.75
    max_visible_misleading_seen_synthetic_rate: float = 0.75
    min_visible_ball_trusted_rate_warning: float = 0.55
    max_visible_ball_rescued_rate_warning: float = 0.40
    max_visible_misleading_seen_synthetic_rate_warning: float = 0.40
    min_visible_ball_confidence_score_mean_warning: float = 0.50
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
    entropy_loss_scale: float = 0.03
    param_log_prob_weight: float = 0.10
    value_loss_scale: float = 0.5
    ratio_clip: float = 0.2
    grad_norm_clip: float = 0.5
    hidden_size: int = 128
    device: str = "cuda:0"
    checkpoint_interval: int = 5000
    eval_interval: int = 5000
    eval_episodes: int = 8
    bc_competence_beta_max: float = 0.08
    bc_competence_beta_min: float = 0.005
    bc_competence_half_life_evals: float = 3.0
    bc_param_loss_scale: float = 0.35
    initial_stage_id: int = -1
    max_stage_id: int = 4
    auto_promote: bool = True
    experiment_name: str = "pybh_mappo_takeover_ppo"
    wire_audit: WireAuditConfig = field(default_factory=WireAuditConfig)


ENV = EnvConfig()
TRAIN = TrainConfig()
