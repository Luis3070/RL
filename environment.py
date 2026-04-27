from __future__ import annotations

import math
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from action import StructuredSkillAction, action_from_mapping, action_from_numpy, skill_mask
from config import ACTION, ENV, FIELD_X_HALF, FIELD_Y_HALF, GOAL_X, GOAL_Y_HALF
from pybh_backend import BackendObservation, PyBHBackend
from reward import ball_distance, compute_progress_reward, goal_scored


class RealSimRobotEnv(ParallelEnv):
    metadata = {"name": "real-simrobot-skillrequest-v1"}

    def __init__(self, env_cfg=ENV):
        super().__init__()
        self.cfg = env_cfg
        self.possible_agents = ["player_1"]
        self.agents = list(self.possible_agents)
        self.backend = PyBHBackend(
            mode=self.cfg.backend.mode,
            frames_per_step=self.cfg.backend.frames_per_step,
            obs_timeout_ms=self.cfg.backend.obs_timeout_ms,
        )
        self.observation_spaces = {
            "player_1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        }
        self.action_spaces = {
            "player_1": gym.spaces.Dict(
                {
                    "skill_type": gym.spaces.Discrete(len(ACTION.skill_names)),
                    "target_x": gym.spaces.Box(low=-ACTION.x_range, high=ACTION.x_range, shape=(), dtype=np.float32),
                    "target_y": gym.spaces.Box(low=-ACTION.y_range, high=ACTION.y_range, shape=(), dtype=np.float32),
                    "target_theta": gym.spaces.Box(low=-math.pi, high=math.pi, shape=(), dtype=np.float32),
                    "pass_target": gym.spaces.Discrete(ACTION.max_pass_target + 1),
                }
            )
        }
        self._rng = np.random.default_rng(self.cfg.seed)
        self._last_obs: Optional[np.ndarray] = None
        self._episode_step = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._episode_step = 0
        ball_pos = np.array(
            [
                self._rng.uniform(-500.0, FIELD_X_HALF * 0.3),
                self._rng.uniform(-FIELD_Y_HALF * 0.2, FIELD_Y_HALF * 0.2),
            ],
            dtype=np.float32,
        )
        robot_pos = ball_pos + np.array([-900.0, self._rng.uniform(-200.0, 200.0)], dtype=np.float32)
        robot_theta = float(math.atan2(ball_pos[1] - robot_pos[1], ball_pos[0] - robot_pos[0]))
        obs = self.backend.reset(ball_pos=ball_pos, robot_pos=robot_pos, robot_theta=robot_theta)
        encoded = self._encode_obs(obs)
        self._last_obs = encoded
        info = self._build_info(obs)
        return {"player_1": encoded}, {"player_1": info}

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("step() called on terminated environment")
        action_payload = actions["player_1"]
        if isinstance(action_payload, dict):
            action = action_from_mapping(action_payload)
        else:
            action = action_from_numpy(np.asarray(action_payload, dtype=np.float32))
        obs = self.backend.step(action)
        encoded = self._encode_obs(obs)
        prev_distance = ball_distance(self._last_obs) if self._last_obs is not None else ball_distance(encoded)
        curr_distance = ball_distance(encoded)
        scored = goal_scored(obs.ball_x, obs.ball_y, goal_x=GOAL_X, goal_y_half=GOAL_Y_HALF)
        reward = compute_progress_reward(
            previous_ball_distance=prev_distance,
            current_ball_distance=curr_distance,
            goal_scored=scored,
            progress_scale=self.cfg.reward_progress_scale,
            goal_bonus=self.cfg.reward_goal_bonus,
            step_penalty=self.cfg.reward_step_penalty,
        )
        self._episode_step += 1
        done = scored or self._episode_step >= self.cfg.max_episode_steps
        self._last_obs = encoded
        info = self._build_info(obs)
        rewards = {"player_1": reward.total}
        terminations = {"player_1": done, "__all__": done}
        truncations = {"player_1": False, "__all__": False}
        if done:
            self.agents = []
        return {"player_1": encoded}, rewards, terminations, truncations, {"player_1": info}

    def close(self):
        self.backend.close()

    def get_global_state(self) -> np.ndarray:
        return self._last_obs.copy() if self._last_obs is not None else np.zeros(9, dtype=np.float32)

    def _encode_obs(self, obs: BackendObservation) -> np.ndarray:
        dx = obs.ball_x - obs.robot_x
        dy = obs.ball_y - obs.robot_y
        return np.asarray(
            [
                dx / ACTION.x_range,
                dy / ACTION.y_range,
                obs.robot_theta / math.pi,
                obs.ball_x / ACTION.x_range,
                obs.ball_y / ACTION.y_range,
                obs.robot_x / ACTION.x_range,
                obs.robot_y / ACTION.y_range,
                float(obs.frame) / 10000.0,
                1.0 if obs.sim_enabled else 0.0,
            ],
            dtype=np.float32,
        )

    def _build_info(self, obs: BackendObservation) -> Dict[str, object]:
        return {
            "state": self.get_global_state(),
            "backend_mode": self.cfg.backend.mode,
            "active_skill_mask": skill_mask(self.cfg.active_skill_names),
            "bridge_obs_ready": obs.obs_ready,
            "bridge_frame": obs.frame,
            "requested_skill": obs.requested_skill,
            "requested_pass_target": obs.requested_pass_target,
            "ball_x": obs.ball_x,
            "ball_y": obs.ball_y,
            "robot_x": obs.robot_x,
            "robot_y": obs.robot_y,
            "robot_theta": obs.robot_theta,
        }
