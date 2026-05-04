from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from action import SkillType, StructuredSkillAction, clamp_action
from config import (
    ACTION,
    BackendConfig,
    BHUMAN_DIR,
    DT,
    PLAYER_NUMBER,
    SCENARIO,
    TEAM_NUMBER,
    LOCATION,
    resolve_simrobot_scene_config,
)

os.chdir(BHUMAN_DIR)
import pybh.controller as pybh


def _normalize_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


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
    # B-Human SkillRequest::walkTo consumes field-global targets. Only the
    # legacy Python world-state backend needs robot-relative walk commands.
    if mode in {"cpp", "simrobot", "simrobot_visible"} or obs is None or action.skill is not SkillType.walk:
        return action

    if isinstance(obs, BackendObservation):
        robot_x = obs.robot_x
        robot_y = obs.robot_y
        robot_theta = obs.robot_theta
    else:
        robot_x = float(obs.get("robot_x", 0.0))
        robot_y = float(obs.get("robot_y", 0.0))
        robot_theta = float(obs.get("robot_theta", 0.0))

    local_x, local_y = _field_to_robot(robot_x, robot_y, robot_theta, action.target_x, action.target_y)
    local_theta = _normalize_angle(float(action.target_theta) - robot_theta)
    return StructuredSkillAction(
        skill_type=int(action.skill_type),
        target_x=float(local_x),
        target_y=float(local_y),
        target_theta=float(local_theta),
        pass_target=int(action.pass_target),
    )


@dataclass
class BackendObservation:
    ball_x: float
    ball_y: float
    robot_x: float
    robot_y: float
    robot_theta: float
    frame: int
    obs_ready: bool
    sim_enabled: bool
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

    @classmethod
    def from_payload(cls, payload: dict) -> "BackendObservation":
        robot_x = float(payload["robot_x"])
        robot_y = float(payload["robot_y"])
        robot_theta = float(payload["robot_theta"])
        ball_x = float(payload["ball_x"])
        ball_y = float(payload["ball_y"])
        fallback_rel_x, fallback_rel_y = _field_to_robot(robot_x, robot_y, robot_theta, ball_x, ball_y)
        return cls(
            ball_x=ball_x,
            ball_y=ball_y,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_theta=robot_theta,
            frame=int(payload.get("frame", 0)),
            obs_ready=bool(payload.get("obs_ready", False)),
            sim_enabled=bool(payload.get("sim_enabled", False)),
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
        )


class PyBHBackend:
    def __init__(
        self,
        *,
        mode: str = "simrobot",
        frames_per_step: int = 1,
        obs_timeout_ms: int = 1000,
        simrobot_profile: str = "2d",
        simrobot_scene_path: Path | None = None,
        simrobot_robot_name: str | None = None,
        simrobot_theta_offset: float | None = None,
        simrobot_launch: bool = False,
        simrobot_bin: Path | None = None,
    ):
        self.mode = mode
        self.frames_per_step = max(1, int(frames_per_step))
        self.obs_timeout_ms = int(obs_timeout_ms)
        self.simrobot_scene = resolve_simrobot_scene_config(
            BackendConfig(
                simrobot_profile=simrobot_profile,
                simrobot_scene_path=simrobot_scene_path,
                simrobot_robot_name=simrobot_robot_name,
                simrobot_theta_offset=simrobot_theta_offset,
            )
        )
        self._controller = None
        self._started = False
        self._last_obs: Optional[BackendObservation] = None
        self._simrobot_launch = bool(simrobot_launch)
        self._simrobot_bin = Path(simrobot_bin).resolve() if simrobot_bin is not None else None
        self._simrobot_process: subprocess.Popen[str] | None = None

    def _uses_embedded_simrobot(self) -> bool:
        return self.mode == "simrobot"

    def _uses_visible_simrobot(self) -> bool:
        return self.mode == "simrobot_visible"

    def start(self) -> None:
        if self._started:
            return
        if self._uses_embedded_simrobot():
            pybh.rl_sim_set_enabled(PLAYER_NUMBER, False)
            if not pybh.simrobot_load_scene(str(self.simrobot_scene.scene_path)):
                raise RuntimeError(f"simrobot_load_scene failed: {pybh.simrobot_last_error()}")
        elif self._uses_visible_simrobot():
            pybh.rl_sim_set_enabled(PLAYER_NUMBER, False)
            if self._simrobot_launch:
                if self._simrobot_bin is None or not self._simrobot_bin.exists():
                    raise FileNotFoundError(f"SimRobot binary not found: {self._simrobot_bin}")
                if not self.simrobot_scene.scene_path.exists():
                    raise FileNotFoundError(f"SimRobot scene not found: {self.simrobot_scene.scene_path}")
                self._simrobot_process = subprocess.Popen(
                    [str(self._simrobot_bin), str(self.simrobot_scene.scene_path)],
                    cwd=BHUMAN_DIR,
                    env=os.environ.copy(),
                )
        else:
            self._controller = pybh.Controller()
            self._controller.add_player(
                name=f"Heron{PLAYER_NUMBER}",
                team_number=TEAM_NUMBER,
                field_player_color=pybh.TeamColor.black,
                goalkeeper_color=pybh.TeamColor.black,
                player_number=PLAYER_NUMBER,
                location=LOCATION,
                scenario=SCENARIO,
            )
            self._controller.start()
            pybh.rl_sim_set_enabled(PLAYER_NUMBER, self.mode == "cpp")
        self._started = True

    def close(self) -> None:
        if not self._started:
            return
        if self._uses_embedded_simrobot():
            pybh.simrobot_unload_scene()
        elif self._uses_visible_simrobot():
            if self._simrobot_process is not None and self._simrobot_process.poll() is None:
                self._simrobot_process.terminate()
                self._simrobot_process.wait(timeout=5)
            self._simrobot_process = None
        elif self._controller is not None:
            self._controller.stop()
            self._controller = None
        self._started = False
        self._last_obs = None

    def reset(self, *, ball_pos: np.ndarray, robot_pos: np.ndarray, robot_theta: float) -> BackendObservation:
        self.start()
        if self._uses_embedded_simrobot():
            self._simrobot_reset(ball_pos, robot_pos, robot_theta)
            self._set_action(StructuredSkillAction(skill_type=SkillType.stand))
            obs = self._wait_for_embedded_reset_obs(robot_pos)
            self._last_obs = obs
            return obs
        elif self._uses_visible_simrobot():
            self._visible_simrobot_reset(ball_pos, robot_pos, robot_theta)
            self._set_action(StructuredSkillAction(skill_type=SkillType.stand))
        elif self.mode == "cpp":
            pybh.rl_sim_reset(
                PLAYER_NUMBER,
                float(ball_pos[0]),
                float(ball_pos[1]),
                float(robot_pos[0]),
                float(robot_pos[1]),
                float(robot_theta),
            )
            self._step_controller()
        else:
            pybh.rl_sim_set_enabled(PLAYER_NUMBER, False)
            self._set_world_state(ball_pos, robot_pos, robot_theta)
            self._step_controller()
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
        if self.mode == "python":
            obs = self.read_obs()
            self._set_world_state(
                np.array([obs.ball_x, obs.ball_y], dtype=np.float32),
                np.array([obs.robot_x, obs.robot_y], dtype=np.float32),
                float(obs.robot_theta),
            )
        adapted_action = adapt_action_to_backend(action, self._last_obs, mode=self.mode)
        self._set_action(adapted_action)
        for _ in range(self.frames_per_step):
            if self._uses_embedded_simrobot():
                pybh.simrobot_step(1)
            elif self._uses_visible_simrobot():
                break
            else:
                self._step_controller()
        obs = self.read_obs()
        self._last_obs = obs
        return obs

    def read_obs(self) -> BackendObservation:
        attempts = 5 if self.mode in {"cpp", "simrobot"} else 1
        for attempt in range(attempts):
            obs = BackendObservation.from_payload(pybh.rl_get_obs(PLAYER_NUMBER, self.obs_timeout_ms))
            if obs.obs_ready:
                self._last_obs = obs
                return obs
            if self._uses_embedded_simrobot():
                pybh.simrobot_step(1)
            elif self.mode == "cpp" and attempt + 1 < attempts:
                self._step_controller()
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

    def _step_controller(self) -> None:
        if self._controller is None:
            raise RuntimeError("Controller backend is not running")
        self._controller.update()

    def _set_world_state(self, ball_pos: np.ndarray, robot_pos: np.ndarray, robot_theta: float) -> None:
        if self._controller is None:
            return
        pybh.set_world_state(
            self._controller,
            0,
            float(ball_pos[0]),
            float(ball_pos[1]),
            float(robot_pos[0]),
            float(robot_pos[1]),
            float(robot_theta),
        )

    def _simrobot_reset(self, ball_pos: np.ndarray, robot_pos: np.ndarray, robot_theta: float) -> None:
        scale = self.simrobot_scene.position_scale
        scene_ball_x = float(ball_pos[0] * scale)
        scene_ball_y = float(ball_pos[1] * scale)
        scene_robot_x = float(robot_pos[0] * scale)
        scene_robot_y = float(robot_pos[1] * scale)
        scene_robot_theta = float(_normalize_angle(robot_theta + self.simrobot_scene.theta_offset))
        if not pybh.simrobot_set_ball_state(scene_ball_x, scene_ball_y, 0.0, 0.0):
            raise RuntimeError("simrobot_set_ball_state failed")
        if not pybh.simrobot_set_robot_pose(
            self.simrobot_scene.robot_name, scene_robot_x, scene_robot_y, scene_robot_theta
        ):
            raise RuntimeError(f"simrobot_set_robot_pose failed for {self.simrobot_scene.robot_name}")

    def _wait_for_embedded_reset_obs(self, robot_pos: np.ndarray, *, max_steps: int = 80) -> BackendObservation:
        expected_x = float(robot_pos[0])
        expected_y = float(robot_pos[1])
        last_obs: BackendObservation | None = None
        for _ in range(max_steps):
            pybh.simrobot_step(1)
            obs = BackendObservation.from_payload(pybh.rl_get_obs(PLAYER_NUMBER, max(1, self.obs_timeout_ms // 10)))
            last_obs = obs
            if not obs.obs_ready:
                continue
            if math.hypot(obs.robot_x - expected_x, obs.robot_y - expected_y) <= 150.0:
                return obs
        if last_obs is not None and last_obs.obs_ready:
            return last_obs
        raise RuntimeError("pybh.rl_get_obs timed out while waiting for embedded SimRobot reset")

    def _visible_simrobot_reset(self, ball_pos: np.ndarray, robot_pos: np.ndarray, robot_theta: float) -> None:
        pybh.rl_reset_world(
            PLAYER_NUMBER,
            float(ball_pos[0]),
            float(ball_pos[1]),
            float(robot_pos[0]),
            float(robot_pos[1]),
            float(robot_theta),
        )
