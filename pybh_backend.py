from __future__ import annotations

import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

from action import SkillType, StructuredSkillAction, clamp_action
from config import (
    ACTION,
    BackendConfig,
    BHUMAN_DIR,
    PLAYER_NUMBER,
    TEAM_NUMBER,
    resolve_simrobot_scene_config,
)

os.chdir(BHUMAN_DIR)
import pybh.controller as pybh

MAX_WORLD_PLAYERS_PER_TEAM = 20
WORLD_STATUS_POLL_INTERVAL_S = 0.01
DEFAULT_VISIBLE_LOG_INTERVAL_S = 1.0
SETTINGS_CFG_PATH = BHUMAN_DIR / "Config/settings.cfg"

# Steps post-reset during which synthetic ball injection is allowed.
# After this window the world_spec ball position diverges from the actual ball,
# so using it would corrupt the observation with stale data.
SYNTHETIC_OBS_WARMUP_STEPS = 8


def _normalize_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def _read_bridge_settings(settings_path: Path = SETTINGS_CFG_PATH) -> dict[str, int] | None:
    if not settings_path.exists():
        return None

    text = settings_path.read_text(encoding="utf-8", errors="ignore")
    result: dict[str, int] = {}
    for key in ("teamNumber", "playerNumber", "magicNumber"):
        match = re.search(rf"{key}\s*=\s*(-?\d+)\s*;", text)
        if match:
            result[key] = int(match.group(1))
    return result


def _rewrite_settings_values(text: str, replacements: Mapping[str, int]) -> str:
    updated = text
    for key, value in replacements.items():
        pattern = rf"({key}\s*=\s*)(-?\d+)(\s*;)"
        updated, count = re.subn(pattern, rf"\g<1>{int(value)}\g<3>", updated, count=1)
        if count == 0:
            raise RuntimeError(f"Could not find {key} in {SETTINGS_CFG_PATH}")
    return updated


def _field_to_robot(robot_x: float, robot_y: float, robot_theta: float, target_x: float, target_y: float) -> tuple[float, float]:
    dx = target_x - robot_x
    dy = target_y - robot_y
    c = math.cos(robot_theta)
    s = math.sin(robot_theta)
    return c * dx + s * dy, -s * dx + c * dy


def adapt_action_to_backend(
    action: StructuredSkillAction,
    obs: "BackendObservation | Mapping[str, float | int] | None",
    *,
    mode: str,
) -> StructuredSkillAction:
    del obs, mode
    # All supported backends now forward field-global walk targets directly to
    # B-Human through SkillRequest::walkTo.
    return action


@dataclass(frozen=True)
class WorldPlayerSpec:
    number: int
    x: float
    y: float
    theta: float = 0.0
    upright: bool = True

    def as_tuple(self) -> tuple[int, float, float, float, bool]:
        return (int(self.number), float(self.x), float(self.y), float(self.theta), bool(self.upright))


@dataclass(frozen=True)
class WorldSpec:
    ball_x: float
    ball_y: float
    robot_x: float
    robot_y: float
    robot_theta: float
    teammates: tuple[WorldPlayerSpec, ...] = ()
    opponents: tuple[WorldPlayerSpec, ...] = ()


def _coerce_world_players(players: Iterable[WorldPlayerSpec | Mapping[str, float | int | bool] | tuple | list]) -> tuple[WorldPlayerSpec, ...]:
    specs: list[WorldPlayerSpec] = []
    for player in players:
        if isinstance(player, WorldPlayerSpec):
            specs.append(player)
            continue
        if isinstance(player, Mapping):
            specs.append(
                WorldPlayerSpec(
                    number=int(player["number"]),
                    x=float(player["x"]),
                    y=float(player["y"]),
                    theta=float(player.get("theta", 0.0)),
                    upright=bool(player.get("upright", True)),
                )
            )
            continue
        values = tuple(player)
        if len(values) == 4:
            values = (*values, True)
        if len(values) != 5:
            raise ValueError(f"Expected world player spec with 4 or 5 values, got {values!r}")
        specs.append(
            WorldPlayerSpec(
                number=int(values[0]),
                x=float(values[1]),
                y=float(values[2]),
                theta=float(values[3]),
                upright=bool(values[4]),
            )
        )
    return tuple(specs)


def build_world_spec(
    *,
    ball_pos: np.ndarray,
    robot_pos: np.ndarray,
    robot_theta: float,
    teammates: Iterable[WorldPlayerSpec | Mapping[str, float | int | bool] | tuple | list] = (),
    opponents: Iterable[WorldPlayerSpec | Mapping[str, float | int | bool] | tuple | list] = (),
) -> WorldSpec:
    return WorldSpec(
        ball_x=float(ball_pos[0]),
        ball_y=float(ball_pos[1]),
        robot_x=float(robot_pos[0]),
        robot_y=float(robot_pos[1]),
        robot_theta=float(robot_theta),
        teammates=_coerce_world_players(teammates),
        opponents=_coerce_world_players(opponents),
    )


def _validate_world_players(
    players: tuple[WorldPlayerSpec, ...],
    *,
    label: str,
    allow_non_upright: bool,
) -> None:
    if len(players) > MAX_WORLD_PLAYERS_PER_TEAM:
        raise ValueError(
            f"{label} supports at most {MAX_WORLD_PLAYERS_PER_TEAM} players, got {len(players)}"
        )

    seen_numbers: set[int] = set()
    for player in players:
        if not 1 <= int(player.number) <= MAX_WORLD_PLAYERS_PER_TEAM:
            raise ValueError(
                f"{label} player numbers must be within 1..{MAX_WORLD_PLAYERS_PER_TEAM}, got {player.number}"
            )
        if int(player.number) in seen_numbers:
            raise ValueError(f"{label} player number {player.number} is duplicated")
        seen_numbers.add(int(player.number))
        if not allow_non_upright and not bool(player.upright):
            raise ValueError(f"{label} player {player.number} requested upright=False in an unsupported backend")


def validate_world_spec(world_spec: WorldSpec, *, mode: str) -> None:
    if mode not in {"simrobot", "simrobot_visible"}:
        raise ValueError(f"Unsupported backend mode {mode!r}. Expected one of: simrobot, simrobot_visible")
    if any(player.number == PLAYER_NUMBER for player in world_spec.teammates):
        raise ValueError(
            f"teammates must not include the controlled robot number {PLAYER_NUMBER}; use robot_pose for the RL robot"
        )
    _validate_world_players(world_spec.teammates, label="teammates", allow_non_upright=False)
    _validate_world_players(world_spec.opponents, label="opponents", allow_non_upright=False)


def simrobot_body_name_for_player(number: int, *, opponent: bool) -> str:
    absolute_number = int(number) + (MAX_WORLD_PLAYERS_PER_TEAM if opponent else 0)
    return f"robot{absolute_number}"


@dataclass
class BackendObservation:
    ball_x: float
    ball_y: float
    robot_x: float
    robot_y: float
    robot_theta: float
    simrobot_robot_x: float
    simrobot_robot_y: float
    simrobot_robot_theta: float
    frame: int
    obs_ready: bool
    requested_skill: str = ""
    requested_pass_target: int = -1
    ball_rel_x: float = 0.0
    ball_rel_y: float = 0.0
    ball_end_rel_x: float = 0.0
    ball_end_rel_y: float = 0.0
    ball_vel_x: float = 0.0
    ball_vel_y: float = 0.0
    time_since_ball_seen: float = 0.0
    time_since_ball_disappeared: float = 0.0
    ball_seen_percentage: float = 0.0
    ball_consistent_with_game_state: bool = False
    can_score_now: bool = False
    shot_quality_no_obstacles: float = 0.0
    shot_opening_with_obstacles: float = 0.0
    pass_options_count: float = 0.0
    nearest_teammate_dist: float = 0.0
    nearest_opponent_dist: float = 0.0
    nearest_uncertain_obstacle_dist: float = 0.0
    nearest_teammate_front_dist: float = 0.0
    nearest_opponent_front_dist: float = 0.0
    nearest_uncertain_front_dist: float = 0.0
    obstacle_count: int = 0
    obstacle_teammate_count: int = 0
    obstacle_opponent_count: int = 0
    obstacle_uncertain_count: int = 0
    obstacle_front_count: int = 0
    nearest_obstacle_dist: float = 0.0
    nearest_obstacle_angle: float = 0.0
    nearest_front_obstacle_dist: float = 0.0
    nearest_opponent_angle: float = 0.0
    obstacle_arm_contact: bool = False
    obstacle_foot_contact: bool = False
    zweikampf_active: bool = False
    zweikampf_call_count: int = 0
    motion_obstacle_path_count: int = 0
    walk_to_pose_path_count: int = 0
    libwalk_active_obstacle_count: int = 0
    libwalk_disable_obstacle_avoidance: bool = False
    motion_engine_ground_contact: bool = False
    motion_engine_fall_state: int = -1
    motion_speed_x: float = 0.0
    motion_speed_y: float = 0.0
    motion_speed_rot: float = 0.0
    synthetic_pose_observation: bool = False
    synthetic_ball_observation: bool = False

    @classmethod
    def from_payload(cls, payload: dict) -> "BackendObservation":
        robot_x = float(payload["robot_x"])
        robot_y = float(payload["robot_y"])
        robot_theta = float(payload["robot_theta"])
        simrobot_robot_x = float(payload.get("simrobot_robot_x", robot_x))
        simrobot_robot_y = float(payload.get("simrobot_robot_y", robot_y))
        simrobot_robot_theta = float(payload.get("simrobot_robot_theta", robot_theta))
        ball_x = float(payload["ball_x"])
        ball_y = float(payload["ball_y"])
        fallback_rel_x, fallback_rel_y = _field_to_robot(robot_x, robot_y, robot_theta, ball_x, ball_y)
        return cls(
            ball_x=ball_x,
            ball_y=ball_y,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_theta=robot_theta,
            simrobot_robot_x=simrobot_robot_x,
            simrobot_robot_y=simrobot_robot_y,
            simrobot_robot_theta=simrobot_robot_theta,
            frame=int(payload.get("frame", 0)),
            obs_ready=bool(payload.get("obs_ready", False)),
            requested_skill=str(payload.get("requested_skill", "")),
            requested_pass_target=int(payload.get("requested_pass_target", -1)),
            ball_rel_x=float(payload.get("ball_rel_x", fallback_rel_x)),
            ball_rel_y=float(payload.get("ball_rel_y", fallback_rel_y)),
            ball_end_rel_x=float(payload.get("ball_end_rel_x", fallback_rel_x)),
            ball_end_rel_y=float(payload.get("ball_end_rel_y", fallback_rel_y)),
            ball_vel_x=float(payload.get("ball_vel_x", 0.0)),
            ball_vel_y=float(payload.get("ball_vel_y", 0.0)),
            time_since_ball_seen=float(payload.get("time_since_ball_seen", 0.0)),
            time_since_ball_disappeared=float(payload.get("time_since_ball_disappeared", 0.0)),
            ball_seen_percentage=float(payload.get("ball_seen_percentage", 0.0)),
            ball_consistent_with_game_state=bool(payload.get("ball_consistent_with_game_state", False)),
            can_score_now=bool(payload.get("can_score_now", False)),
            shot_quality_no_obstacles=float(payload.get("shot_quality_no_obstacles", 0.0)),
            shot_opening_with_obstacles=float(payload.get("shot_opening_with_obstacles", 0.0)),
            pass_options_count=float(payload.get("pass_options_count", 0.0)),
            nearest_teammate_dist=float(payload.get("nearest_teammate_dist", 0.0)),
            nearest_opponent_dist=float(payload.get("nearest_opponent_dist", 0.0)),
            nearest_uncertain_obstacle_dist=float(payload.get("nearest_uncertain_obstacle_dist", 0.0)),
            nearest_teammate_front_dist=float(payload.get("nearest_teammate_front_dist", 0.0)),
            nearest_opponent_front_dist=float(payload.get("nearest_opponent_front_dist", 0.0)),
            nearest_uncertain_front_dist=float(payload.get("nearest_uncertain_front_dist", 0.0)),
            obstacle_count=int(payload.get("obstacle_count", 0)),
            obstacle_teammate_count=int(payload.get("obstacle_teammate_count", 0)),
            obstacle_opponent_count=int(payload.get("obstacle_opponent_count", 0)),
            obstacle_uncertain_count=int(payload.get("obstacle_uncertain_count", 0)),
            obstacle_front_count=int(payload.get("obstacle_front_count", 0)),
            nearest_obstacle_dist=float(payload.get("nearest_obstacle_dist", 0.0)),
            nearest_obstacle_angle=float(payload.get("nearest_obstacle_angle", 0.0)),
            nearest_front_obstacle_dist=float(payload.get("nearest_front_obstacle_dist", 0.0)),
            nearest_opponent_angle=float(payload.get("nearest_opponent_angle", 0.0)),
            obstacle_arm_contact=bool(payload.get("obstacle_arm_contact", False)),
            obstacle_foot_contact=bool(payload.get("obstacle_foot_contact", False)),
            zweikampf_active=bool(payload.get("zweikampf_active", False)),
            zweikampf_call_count=int(payload.get("zweikampf_call_count", 0)),
            motion_obstacle_path_count=int(payload.get("motion_obstacle_path_count", 0)),
            walk_to_pose_path_count=int(payload.get("walk_to_pose_path_count", 0)),
            libwalk_active_obstacle_count=int(payload.get("libwalk_active_obstacle_count", 0)),
            libwalk_disable_obstacle_avoidance=bool(payload.get("libwalk_disable_obstacle_avoidance", False)),
            motion_engine_ground_contact=bool(payload.get("motion_engine_ground_contact", False)),
            motion_engine_fall_state=int(payload.get("motion_engine_fall_state", -1)),
            motion_speed_x=float(payload.get("motion_speed_x", 0.0)),
            motion_speed_y=float(payload.get("motion_speed_y", 0.0)),
            motion_speed_rot=float(payload.get("motion_speed_rot", 0.0)),
        )


@dataclass(frozen=True)
class WorldUpdateStatus:
    result_request_id: int
    result_ok: bool
    result_error: str
    reset_request_id: int
    reset_applied_id: int
    reset_pending: bool
    dynamic_request_id: int
    dynamic_applied_id: int
    dynamic_pending: bool

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldUpdateStatus":
        return cls(
            result_request_id=int(payload.get("result_request_id", 0)),
            result_ok=bool(payload.get("result_ok", False)),
            result_error=str(payload.get("result_error", "")),
            reset_request_id=int(payload.get("reset_request_id", 0)),
            reset_applied_id=int(payload.get("reset_applied_id", 0)),
            reset_pending=bool(payload.get("reset_pending", False)),
            dynamic_request_id=int(payload.get("dynamic_request_id", 0)),
            dynamic_applied_id=int(payload.get("dynamic_applied_id", 0)),
            dynamic_pending=bool(payload.get("dynamic_pending", False)),
        )

    def summary(self) -> str:
        return (
            f"result_request_id={self.result_request_id} "
            f"result_ok={int(self.result_ok)} "
            f"reset=({self.reset_applied_id}/{self.reset_request_id},pending={int(self.reset_pending)}) "
            f"dynamic=({self.dynamic_applied_id}/{self.dynamic_request_id},pending={int(self.dynamic_pending)})"
        )


class PyBHBackend:
    def __init__(
        self,
        *,
        mode: str = "simrobot",
        frames_per_step: int = 1,
        obs_timeout_ms: int = 1000,
        visible_request_timeout_ms: int | None = None,
        visible_reset_obs_timeout_ms: int | None = None,
        visible_status_log_interval_s: float = 0.0,
        simrobot_profile: str = "2d",
        simrobot_scene_path: Path | None = None,
        simrobot_robot_name: str | None = None,
        simrobot_theta_offset: float | None = None,
        simrobot_launch: bool = False,
        simrobot_bin: Path | None = None,
    ):
        if mode not in {"simrobot", "simrobot_visible"}:
            raise ValueError(f"Unsupported backend mode {mode!r}. Expected one of: simrobot, simrobot_visible")
        self.mode = mode
        self.frames_per_step = max(1, int(frames_per_step))
        self.obs_timeout_ms = int(obs_timeout_ms)
        self.visible_request_timeout_ms = (
            self.obs_timeout_ms if visible_request_timeout_ms is None else max(1, int(visible_request_timeout_ms))
        )
        self.visible_reset_obs_timeout_ms = (
            self.obs_timeout_ms
            if visible_reset_obs_timeout_ms is None
            else max(1, int(visible_reset_obs_timeout_ms))
        )
        self.visible_status_log_interval_s = max(0.0, float(visible_status_log_interval_s))
        self.simrobot_scene = resolve_simrobot_scene_config(
            BackendConfig(
                mode=mode,
                simrobot_profile=simrobot_profile,
                simrobot_scene_path=simrobot_scene_path,
                simrobot_robot_name=simrobot_robot_name,
                simrobot_theta_offset=simrobot_theta_offset,
            )
        )
        self._controller = None
        self._started = False
        self._last_obs: Optional[BackendObservation] = None
        self._world_spec: Optional[WorldSpec] = None
        self._simrobot_launch = bool(simrobot_launch)
        self._simrobot_bin = Path(simrobot_bin).resolve() if simrobot_bin is not None else None
        self._simrobot_process: subprocess.Popen[str] | None = None
        self._settings_backup_text: str | None = None
        self._qt_qpa_backup: str | None = None
        self._steps_since_reset: int = 0

    def _uses_embedded_simrobot(self) -> bool:
        return self.mode == "simrobot"

    def _uses_visible_simrobot(self) -> bool:
        return self.mode == "simrobot_visible"

    def _prepare_embedded_qt_platform(self) -> None:
        if self.simrobot_scene.profile != "3d":
            return
        if os.environ.get("QT_QPA_PLATFORM"):
            return
        if not os.environ.get("DISPLAY"):
            return
        self._qt_qpa_backup = None
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    def _restore_embedded_qt_platform(self) -> None:
        if self._qt_qpa_backup is not None:
            os.environ["QT_QPA_PLATFORM"] = self._qt_qpa_backup
            self._qt_qpa_backup = None
        elif os.environ.get("QT_QPA_PLATFORM") == "xcb":
            os.environ.pop("QT_QPA_PLATFORM", None)

    def _apply_bridge_settings_override(self) -> None:
        if self._settings_backup_text is not None:
            return
        if not SETTINGS_CFG_PATH.exists():
            raise FileNotFoundError(f"SimRobot settings file not found: {SETTINGS_CFG_PATH}")

        original_text = SETTINGS_CFG_PATH.read_text(encoding="utf-8")
        original_settings = _read_bridge_settings(SETTINGS_CFG_PATH) or {}
        if (
            original_settings.get("teamNumber") == TEAM_NUMBER
            and original_settings.get("playerNumber") == PLAYER_NUMBER
        ):
            return

        updated_text = _rewrite_settings_values(
            original_text,
            {
                "teamNumber": TEAM_NUMBER,
                "playerNumber": PLAYER_NUMBER,
                "magicNumber": TEAM_NUMBER,
            },
        )
        SETTINGS_CFG_PATH.write_text(updated_text, encoding="utf-8")
        self._settings_backup_text = original_text

    def _restore_bridge_settings_override(self) -> None:
        if self._settings_backup_text is None:
            return
        SETTINGS_CFG_PATH.write_text(self._settings_backup_text, encoding="utf-8")
        self._settings_backup_text = None

    def _validate_bridge_settings(self) -> None:
        settings = _read_bridge_settings()
        if not settings:
            return

        actual_team = settings.get("teamNumber")
        actual_player = settings.get("playerNumber")
        problems: list[str] = []
        if actual_team is not None and actual_team != TEAM_NUMBER:
            problems.append(f"teamNumber={actual_team}")
        if actual_player is not None and actual_player != PLAYER_NUMBER:
            problems.append(f"playerNumber={actual_player}")

        if problems:
            raise RuntimeError(
                "SimRobot bridge settings are misaligned with the RL shared-state bridge. "
                f"Expected teamNumber={TEAM_NUMBER}, playerNumber={PLAYER_NUMBER} in {SETTINGS_CFG_PATH}, "
                f"but found {', '.join(problems)}. "
                "This leaves RLSharedState disconnected because LocalConsole listens on a different slot."
            )

    def start(self) -> None:
        if self._started:
            return
        if self._uses_embedded_simrobot():
            self._qt_qpa_backup = os.environ.get("QT_QPA_PLATFORM")
            self._prepare_embedded_qt_platform()
            self._apply_bridge_settings_override()
            if not pybh.simrobot_load_scene(str(self.simrobot_scene.scene_path)):
                self._restore_bridge_settings_override()
                self._restore_embedded_qt_platform()
                raise RuntimeError(f"simrobot_load_scene failed: {pybh.simrobot_last_error()}")
            self._validate_bridge_settings()
        elif self._uses_visible_simrobot():
            if self._simrobot_launch:
                self._apply_bridge_settings_override()
                if self._simrobot_bin is None or not self._simrobot_bin.exists():
                    self._restore_bridge_settings_override()
                    raise FileNotFoundError(f"SimRobot binary not found: {self._simrobot_bin}")
                if not self.simrobot_scene.scene_path.exists():
                    self._restore_bridge_settings_override()
                    raise FileNotFoundError(f"SimRobot scene not found: {self.simrobot_scene.scene_path}")
                self._simrobot_process = subprocess.Popen(
                    [str(self._simrobot_bin), str(self.simrobot_scene.scene_path)],
                    cwd=BHUMAN_DIR,
                    env=os.environ.copy(),
                )
            try:
                self._validate_bridge_settings()
            except Exception:
                if self._simrobot_process is not None and self._simrobot_process.poll() is None:
                    self._simrobot_process.terminate()
                    self._simrobot_process.wait(timeout=5)
                self._simrobot_process = None
                self._restore_bridge_settings_override()
                raise
        else:
            raise RuntimeError(f"Unsupported backend mode {self.mode!r}")
        self._started = True

    def close(self) -> None:
        if not self._started:
            return
        if self._uses_embedded_simrobot():
            pybh.simrobot_unload_scene()
            self._restore_embedded_qt_platform()
            self._restore_bridge_settings_override()
        elif self._uses_visible_simrobot():
            if self._simrobot_process is not None and self._simrobot_process.poll() is None:
                self._simrobot_process.terminate()
                self._simrobot_process.wait(timeout=5)
            self._simrobot_process = None
            self._restore_bridge_settings_override()
        self._started = False
        self._last_obs = None
        self._world_spec = None
        self._steps_since_reset = 0

    def reset(
        self,
        *,
        ball_pos: np.ndarray | None = None,
        robot_pos: np.ndarray | None = None,
        robot_theta: float | None = None,
        teammates: Iterable[WorldPlayerSpec | Mapping[str, float | int | bool] | tuple | list] = (),
        opponents: Iterable[WorldPlayerSpec | Mapping[str, float | int | bool] | tuple | list] = (),
        world_spec: WorldSpec | None = None,
    ) -> BackendObservation:
        self.start()
        if world_spec is None:
            if ball_pos is None or robot_pos is None or robot_theta is None:
                raise ValueError("reset() requires either world_spec or ball_pos/robot_pos/robot_theta")
            world_spec = build_world_spec(
                ball_pos=np.asarray(ball_pos, dtype=np.float32),
                robot_pos=np.asarray(robot_pos, dtype=np.float32),
                robot_theta=float(robot_theta),
                teammates=teammates,
                opponents=opponents,
            )
        validate_world_spec(world_spec, mode=self.mode)
        self._world_spec = world_spec

        self._steps_since_reset = 0
        if self._uses_embedded_simrobot():
            self._simrobot_reset(world_spec)
            self._set_action(StructuredSkillAction(skill_type=SkillType.stand))
            obs = self._wait_for_embedded_reset_obs(
                np.array([world_spec.robot_x, world_spec.robot_y], dtype=np.float32)
            )
            self._last_obs = obs
            return obs
        elif self._uses_visible_simrobot():
            request_id = self._visible_simrobot_reset(world_spec)
            self._wait_for_visible_world_request(request_id=request_id)
            self._set_action(StructuredSkillAction(skill_type=SkillType.stand))
            obs = self._wait_for_visible_reset_obs(
                np.array([world_spec.robot_x, world_spec.robot_y], dtype=np.float32)
            )
            self._last_obs = obs
            return obs
        else:
            raise RuntimeError(f"Unsupported backend mode {self.mode!r}")
        obs = self.read_obs()
        self._last_obs = obs
        return obs

    def set_dynamic_world(
        self,
        *,
        ball_pos: np.ndarray | None = None,
        robot_pos: np.ndarray | None = None,
        robot_theta: float | None = None,
        teammates: Iterable[WorldPlayerSpec | Mapping[str, float | int | bool] | tuple | list] | None = None,
        opponents: Iterable[WorldPlayerSpec | Mapping[str, float | int | bool] | tuple | list] | None = None,
    ) -> BackendObservation:
        if (robot_pos is None) != (robot_theta is None):
            raise ValueError("robot_pos and robot_theta must be provided together")
        teammate_specs = None if teammates is None else _coerce_world_players(teammates)
        opponent_specs = None if opponents is None else _coerce_world_players(opponents)
        if teammate_specs is not None:
            _validate_world_players(teammate_specs, label="teammates", allow_non_upright=False)
            if any(player.number == PLAYER_NUMBER for player in teammate_specs):
                raise ValueError(
                    f"teammates must not include the controlled robot number {PLAYER_NUMBER}; use robot_pose for the RL robot"
                )
        if opponent_specs is not None:
            _validate_world_players(opponent_specs, label="opponents", allow_non_upright=False)

        self.start()
        if self._uses_embedded_simrobot():
            self._apply_embedded_dynamic_world(
                ball_pos=ball_pos,
                robot_pos=robot_pos,
                robot_theta=robot_theta,
                teammates=teammate_specs,
                opponents=opponent_specs,
            )
            obs = self.read_obs()
            self._last_obs = obs
            return obs

        assert self._uses_visible_simrobot()
        request_id = int(
            pybh.rl_set_dynamic_world(
                PLAYER_NUMBER,
                ball=None if ball_pos is None else (float(ball_pos[0]), float(ball_pos[1])),
                own_pose=None
                if robot_pos is None or robot_theta is None
                else (float(robot_pos[0]), float(robot_pos[1]), float(robot_theta)),
                teammates=None if teammate_specs is None else self._serialize_players(teammate_specs),
                opponents=None if opponent_specs is None else self._serialize_players(opponent_specs),
            )
        )
        self._wait_for_visible_world_request(request_id=request_id)
        obs = self.read_obs()
        self._last_obs = obs
        return obs

    def step(self, action: StructuredSkillAction) -> BackendObservation:
        action = clamp_action(
            action,
            x_range=ACTION.x_range,
            y_range=ACTION.y_range,
            max_pass_target=ACTION.max_pass_target,
        )
        adapted_action = adapt_action_to_backend(action, self._last_obs, mode=self.mode)
        self._set_action(adapted_action)
        for _ in range(self.frames_per_step):
            if self._uses_embedded_simrobot():
                pybh.simrobot_step(1)
            elif self._uses_visible_simrobot():
                break
        obs = self.read_obs()
        self._last_obs = obs
        self._steps_since_reset += 1
        return obs

    def read_obs(self) -> BackendObservation:
        if self._uses_embedded_simrobot():
            # In embedded mode, waiting a full timeout without advancing SimRobot
            # can stall observation delivery. Poll briefly and keep stepping.
            poll_timeout_ms = max(1, min(20, self.obs_timeout_ms // 20 if self.obs_timeout_ms > 0 else 1))
            max_poll_steps = max(10, self.frames_per_step * 20)
            last_obs: BackendObservation | None = None
            for _ in range(max_poll_steps):
                obs = self._stabilize_obs(
                    BackendObservation.from_payload(pybh.rl_get_obs(PLAYER_NUMBER, poll_timeout_ms))
                )
                last_obs = obs
                if obs.obs_ready:
                    self._last_obs = obs
                    return obs
                pybh.simrobot_step(1)

            if last_obs is not None and self._last_obs is not None and last_obs.frame == self._last_obs.frame:
                return self._last_obs
            raise RuntimeError(
                "pybh.rl_get_obs timed out in backend=simrobot after progressive polling; "
                f"max_poll_steps={max_poll_steps} poll_timeout_ms={poll_timeout_ms}"
            )

        obs = self._stabilize_obs(BackendObservation.from_payload(pybh.rl_get_obs(PLAYER_NUMBER, self.obs_timeout_ms)))
        if obs.obs_ready:
            self._last_obs = obs
            return obs
        raise RuntimeError(f"pybh.rl_get_obs timed out in backend={self.mode}")

    def _set_action(self, action: StructuredSkillAction) -> None:
        skill_name = action.skill_name
        pybh_name = "walkTo" if skill_name == "walk" else skill_name
        kwargs = dict(
            player_number=PLAYER_NUMBER,
            skill=pybh_name,
            target_x=float(action.target_x),
            target_y=float(action.target_y),
            target_theta=float(action.target_theta),
        )
        if action.skill is SkillType.pass_:
            try:
                pybh.rl_set_action(pass_target=int(action.pass_target), **kwargs)
                return
            except TypeError:
                pass
        pybh.rl_set_action(**kwargs)

    def _serialize_players(self, players: tuple[WorldPlayerSpec, ...]) -> list[tuple[int, float, float, float, bool]]:
        return [player.as_tuple() for player in players]

    def _read_world_status(self) -> WorldUpdateStatus:
        return WorldUpdateStatus.from_payload(pybh.rl_get_world_status(PLAYER_NUMBER))

    def _maybe_log_visible_wait(self, message: str, *, next_log_at: float, now: float) -> float:
        if self.visible_status_log_interval_s <= 0.0 or now < next_log_at:
            return next_log_at
        print(message, flush=True)
        return now + self.visible_status_log_interval_s

    def _simrobot_reset(self, world_spec: WorldSpec) -> None:
        scale = self.simrobot_scene.position_scale
        scene_ball_x = float(world_spec.ball_x * scale)
        scene_ball_y = float(world_spec.ball_y * scale)
        scene_robot_x = float(world_spec.robot_x * scale)
        scene_robot_y = float(world_spec.robot_y * scale)
        scene_robot_theta = float(_normalize_angle(world_spec.robot_theta + self.simrobot_scene.theta_offset))
        if not pybh.simrobot_set_ball_state(scene_ball_x, scene_ball_y, 0.0, 0.0):
            raise RuntimeError("simrobot_set_ball_state failed")
        if not pybh.simrobot_set_robot_pose(
            self.simrobot_scene.robot_name, scene_robot_x, scene_robot_y, scene_robot_theta
        ):
            raise RuntimeError(f"simrobot_set_robot_pose failed for {self.simrobot_scene.robot_name}")
        for player in world_spec.teammates:
            body_name = simrobot_body_name_for_player(player.number, opponent=False)
            scene_x = float(player.x * scale)
            scene_y = float(player.y * scale)
            scene_theta = float(_normalize_angle(player.theta + self.simrobot_scene.theta_offset))
            if not pybh.simrobot_set_robot_pose(body_name, scene_x, scene_y, scene_theta):
                raise RuntimeError(f"simrobot_set_robot_pose failed for {body_name}")
        for player in world_spec.opponents:
            body_name = simrobot_body_name_for_player(player.number, opponent=True)
            scene_x = float(player.x * scale)
            scene_y = float(player.y * scale)
            scene_theta = float(_normalize_angle(player.theta + self.simrobot_scene.theta_offset))
            if not pybh.simrobot_set_robot_pose(body_name, scene_x, scene_y, scene_theta):
                raise RuntimeError(f"simrobot_set_robot_pose failed for {body_name}")

    def _apply_embedded_dynamic_world(
        self,
        *,
        ball_pos: np.ndarray | None,
        robot_pos: np.ndarray | None,
        robot_theta: float | None,
        teammates: tuple[WorldPlayerSpec, ...] | None,
        opponents: tuple[WorldPlayerSpec, ...] | None,
    ) -> None:
        scale = self.simrobot_scene.position_scale
        if ball_pos is not None:
            if not pybh.simrobot_set_ball_state(float(ball_pos[0] * scale), float(ball_pos[1] * scale), 0.0, 0.0):
                raise RuntimeError("simrobot_set_ball_state failed during dynamic world update")
        if robot_pos is not None:
            if robot_theta is None:
                raise ValueError("robot_theta is required when robot_pos is provided")
            scene_robot_theta = float(_normalize_angle(float(robot_theta) + self.simrobot_scene.theta_offset))
            if not pybh.simrobot_set_robot_pose(
                self.simrobot_scene.robot_name,
                float(robot_pos[0] * scale),
                float(robot_pos[1] * scale),
                scene_robot_theta,
            ):
                raise RuntimeError(f"simrobot_set_robot_pose failed for {self.simrobot_scene.robot_name}")
        if teammates is not None:
            for player in teammates:
                if not pybh.simrobot_set_robot_pose(
                    simrobot_body_name_for_player(player.number, opponent=False),
                    float(player.x * scale),
                    float(player.y * scale),
                    float(_normalize_angle(player.theta + self.simrobot_scene.theta_offset)),
                ):
                    raise RuntimeError(f"simrobot_set_robot_pose failed for teammate robot {player.number}")
        if opponents is not None:
            for player in opponents:
                if not pybh.simrobot_set_robot_pose(
                    simrobot_body_name_for_player(player.number, opponent=True),
                    float(player.x * scale),
                    float(player.y * scale),
                    float(_normalize_angle(player.theta + self.simrobot_scene.theta_offset)),
                ):
                    raise RuntimeError(f"simrobot_set_robot_pose failed for opponent robot {player.number}")
        pybh.simrobot_step(1)

    def _wait_for_embedded_reset_obs(self, robot_pos: np.ndarray, *, max_steps: int = 80) -> BackendObservation:
        expected_x = float(robot_pos[0])
        expected_y = float(robot_pos[1])
        last_obs: BackendObservation | None = None
        for _ in range(max_steps):
            pybh.simrobot_step(1)
            obs = self._stabilize_obs(
                BackendObservation.from_payload(pybh.rl_get_obs(PLAYER_NUMBER, max(1, self.obs_timeout_ms // 10)))
            )
            last_obs = obs
            if not obs.obs_ready:
                continue
            if math.hypot(obs.simrobot_robot_x - expected_x, obs.simrobot_robot_y - expected_y) <= 150.0:
                return obs
        if last_obs is not None and last_obs.obs_ready:
            return last_obs
        raise RuntimeError("pybh.rl_get_obs timed out while waiting for embedded SimRobot reset")

    def _wait_for_visible_world_request(self, *, request_id: int, timeout_ms: int | None = None) -> None:
        effective_timeout_ms = self.visible_request_timeout_ms if timeout_ms is None else max(1, int(timeout_ms))
        deadline = time.monotonic() + (effective_timeout_ms / 1000.0)
        started_at = time.monotonic()
        next_log_at = started_at + min(DEFAULT_VISIBLE_LOG_INTERVAL_S, self.visible_status_log_interval_s or 1e9)
        while time.monotonic() < deadline:
            now = time.monotonic()
            status = self._read_world_status()
            next_log_at = self._maybe_log_visible_wait(
                (
                    f"[visible wait request {request_id}] "
                    f"+{now - started_at:.1f}s {status.summary()}"
                ),
                next_log_at=next_log_at,
                now=now,
            )
            if status.result_request_id >= request_id:
                if status.result_ok:
                    return
                raise RuntimeError(status.result_error or f"Visible SimRobot world request {request_id} failed")
            time.sleep(WORLD_STATUS_POLL_INTERVAL_S)
        raise RuntimeError(f"Timed out waiting for visible SimRobot world request {request_id}")

    def _wait_for_visible_reset_obs(self, robot_pos: np.ndarray, *, timeout_ms: int | None = None) -> BackendObservation:
        effective_timeout_ms = self.visible_reset_obs_timeout_ms if timeout_ms is None else max(1, int(timeout_ms))
        deadline = time.monotonic() + (effective_timeout_ms / 1000.0)
        last_obs: BackendObservation | None = None
        expected_x = float(robot_pos[0])
        expected_y = float(robot_pos[1])
        per_try_timeout_ms = max(1, min(100, effective_timeout_ms // 20))
        started_at = time.monotonic()
        next_log_at = started_at + min(DEFAULT_VISIBLE_LOG_INTERVAL_S, self.visible_status_log_interval_s or 1e9)
        while time.monotonic() < deadline:
            obs = self._stabilize_obs(BackendObservation.from_payload(pybh.rl_get_obs(PLAYER_NUMBER, per_try_timeout_ms)))
            last_obs = obs
            now = time.monotonic()
            next_log_at = self._maybe_log_visible_wait(
                (
                    "[visible wait reset-obs] "
                    f"+{now - started_at:.1f}s frame={obs.frame} "
                    f"obs_ready={int(obs.obs_ready)} "
                    f"robot_est=({obs.robot_x:.1f},{obs.robot_y:.1f},{obs.robot_theta:.2f}) "
                    f"robot_sim=({obs.simrobot_robot_x:.1f},{obs.simrobot_robot_y:.1f},{obs.simrobot_robot_theta:.2f})"
                ),
                next_log_at=next_log_at,
                now=now,
            )
            if not obs.obs_ready:
                continue
            if math.hypot(obs.simrobot_robot_x - expected_x, obs.simrobot_robot_y - expected_y) <= 150.0:
                return obs
        if last_obs is not None and last_obs.obs_ready:
            distance_mm = math.hypot(last_obs.simrobot_robot_x - expected_x, last_obs.simrobot_robot_y - expected_y)
            raise RuntimeError(
                "Visible SimRobot reset obs did not converge to requested pose within timeout: "
                f"expected=({expected_x:.1f},{expected_y:.1f}) "
                f"got_sim=({last_obs.simrobot_robot_x:.1f},{last_obs.simrobot_robot_y:.1f},{last_obs.simrobot_robot_theta:.2f}) "
                f"got_est=({last_obs.robot_x:.1f},{last_obs.robot_y:.1f},{last_obs.robot_theta:.2f}) "
                f"distance_mm={distance_mm:.1f} timeout_ms={effective_timeout_ms}"
            )
        raise RuntimeError("pybh.rl_get_obs timed out while waiting for visible SimRobot reset")

    def _visible_simrobot_reset(self, world_spec: WorldSpec) -> int:
        return int(
            pybh.rl_reset_world_extended(
            PLAYER_NUMBER,
            float(world_spec.ball_x),
            float(world_spec.ball_y),
            float(world_spec.robot_x),
            float(world_spec.robot_y),
            float(world_spec.robot_theta),
            teammates=self._serialize_players(world_spec.teammates),
            opponents=self._serialize_players(world_spec.opponents),
            )
        )

    def _stabilize_obs(self, obs: BackendObservation) -> BackendObservation:
        if not obs.obs_ready:
            return obs

        updated = obs
        synthetic_pose = False
        synthetic_ball = False

        pose_error = math.hypot(obs.robot_x - obs.simrobot_robot_x, obs.robot_y - obs.simrobot_robot_y)
        if self._uses_embedded_simrobot() and pose_error > 250.0:
            field_theta = _normalize_angle(float(obs.simrobot_robot_theta) - float(self.simrobot_scene.theta_offset))
            updated = replace(
                updated,
                robot_x=float(obs.simrobot_robot_x),
                robot_y=float(obs.simrobot_robot_y),
                robot_theta=field_theta,
            )
            synthetic_pose = True

        if self._world_spec is not None:
            ball_stale = (
                float(updated.ball_seen_percentage) <= 0.0
                and float(updated.time_since_ball_seen) >= 1000.0
                and self._steps_since_reset < SYNTHETIC_OBS_WARMUP_STEPS
            )
            collapsed_ball = math.hypot(updated.ball_x - updated.robot_x, updated.ball_y - updated.robot_y) <= 1.0
            if ball_stale or collapsed_ball:
                truth_ball_x = float(self._world_spec.ball_x)
                truth_ball_y = float(self._world_spec.ball_y)
                rel_x, rel_y = _field_to_robot(
                    float(updated.robot_x),
                    float(updated.robot_y),
                    float(updated.robot_theta),
                    truth_ball_x,
                    truth_ball_y,
                )
                updated = replace(
                    updated,
                    ball_x=truth_ball_x,
                    ball_y=truth_ball_y,
                    ball_rel_x=float(rel_x),
                    ball_rel_y=float(rel_y),
                    ball_end_rel_x=float(rel_x),
                    ball_end_rel_y=float(rel_y),
                    ball_vel_x=0.0,
                    ball_vel_y=0.0,
                    time_since_ball_seen=0.0,
                    time_since_ball_disappeared=0.0,
                    ball_seen_percentage=100.0,
                    ball_consistent_with_game_state=True,
                )
                synthetic_ball = True
            elif synthetic_pose:
                rel_x, rel_y = _field_to_robot(
                    float(updated.robot_x),
                    float(updated.robot_y),
                    float(updated.robot_theta),
                    float(updated.ball_x),
                    float(updated.ball_y),
                )
                updated = replace(
                    updated,
                    ball_rel_x=float(rel_x),
                    ball_rel_y=float(rel_y),
                    ball_end_rel_x=float(rel_x),
                    ball_end_rel_y=float(rel_y),
                )

        if synthetic_pose or synthetic_ball:
            updated = replace(
                updated,
                synthetic_pose_observation=synthetic_pose,
                synthetic_ball_observation=synthetic_ball,
            )
        return updated
