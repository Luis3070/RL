from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from action import SkillType, StructuredSkillAction, clamp_action
from config import (
    ACTION,
    BHUMAN_DIR,
    DT,
    MM_PER_M,
    PLAYER_NUMBER,
    SCENARIO,
    SIMROBOT_ROBOT_NAME,
    SIMROBOT_SCENE_PATH,
    TEAM_NUMBER,
    LOCATION,
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
    # The C++ RLSim2D path consumes field-global walk targets directly.
    if mode == "cpp" or obs is None or action.skill is not SkillType.walk:
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

    @classmethod
    def from_payload(cls, payload: dict) -> "BackendObservation":
        return cls(
            ball_x=float(payload["ball_x"]),
            ball_y=float(payload["ball_y"]),
            robot_x=float(payload["robot_x"]),
            robot_y=float(payload["robot_y"]),
            robot_theta=float(payload["robot_theta"]),
            frame=int(payload.get("frame", 0)),
            obs_ready=bool(payload.get("obs_ready", False)),
            sim_enabled=bool(payload.get("sim_enabled", False)),
            requested_skill=str(payload.get("requested_skill", "")),
            requested_pass_target=int(payload.get("requested_pass_target", -1)),
        )


class PyBHBackend:
    def __init__(self, *, mode: str = "simrobot", frames_per_step: int = 1, obs_timeout_ms: int = 1000):
        self.mode = mode
        self.frames_per_step = max(1, int(frames_per_step))
        self.obs_timeout_ms = int(obs_timeout_ms)
        self._controller = None
        self._started = False
        self._last_obs: Optional[BackendObservation] = None

    def start(self) -> None:
        if self._started:
            return
        if self.mode == "simrobot":
            if not pybh.simrobot_load_scene(str(SIMROBOT_SCENE_PATH)):
                raise RuntimeError(f"simrobot_load_scene failed: {pybh.simrobot_last_error()}")
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
        if self.mode == "simrobot":
            pybh.simrobot_unload_scene()
        elif self._controller is not None:
            self._controller.stop()
            self._controller = None
        self._started = False
        self._last_obs = None

    def reset(self, *, ball_pos: np.ndarray, robot_pos: np.ndarray, robot_theta: float) -> BackendObservation:
        self.start()
        if self.mode == "simrobot":
            self._simrobot_reset(ball_pos, robot_pos, robot_theta)
            self._set_action(StructuredSkillAction(skill_type=SkillType.stand))
            pybh.simrobot_step(1)
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
            if self.mode == "simrobot":
                pybh.simrobot_step(1)
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
            if self.mode == "simrobot":
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

    @staticmethod
    def _simrobot_reset(ball_pos: np.ndarray, robot_pos: np.ndarray, robot_theta: float) -> None:
        scene_ball_x = float(-ball_pos[0] / MM_PER_M)
        scene_ball_y = float(-ball_pos[1] / MM_PER_M)
        scene_robot_x = float(-robot_pos[0] / MM_PER_M)
        scene_robot_y = float(-robot_pos[1] / MM_PER_M)
        scene_robot_theta = float(math.atan2(math.sin(robot_theta - math.pi), math.cos(robot_theta - math.pi)))
        if not pybh.simrobot_set_ball_state(scene_ball_x, scene_ball_y, 0.0, 0.0):
            raise RuntimeError("simrobot_set_ball_state failed")
        if not pybh.simrobot_set_robot_pose(SIMROBOT_ROBOT_NAME, scene_robot_x, scene_robot_y, scene_robot_theta):
            raise RuntimeError(f"simrobot_set_robot_pose failed for {SIMROBOT_ROBOT_NAME}")
