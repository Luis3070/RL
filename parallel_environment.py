from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from config import EnvConfig
from environment import RealSimRobotEnv


class ParallelSlotSimRobotEnv(ParallelEnv):
    """PettingZoo ParallelEnv backed by one SimRobot slot per agent."""

    metadata = {"name": "parallel-slot-simrobot-v1"}

    def __init__(self, env_cfg: EnvConfig):
        super().__init__()
        self.cfg = env_cfg
        self.player_numbers = tuple(int(player) for player in env_cfg.parallel_player_numbers)
        if not self.player_numbers:
            raise ValueError("parallel_player_numbers must contain at least one player")

        self.possible_agents = [f"player_{player}" for player in self.player_numbers]
        self.agents = list(self.possible_agents)
        self._envs: dict[str, RealSimRobotEnv] = {}
        for agent, player_number in zip(self.possible_agents, self.player_numbers, strict=True):
            backend_cfg = replace(env_cfg.backend, player_number=player_number, simrobot_launch=False)
            slot_cfg = replace(env_cfg, backend=backend_cfg, parallel_player_numbers=(player_number,))
            self._envs[agent] = RealSimRobotEnv(env_cfg=slot_cfg)

        first_env = next(iter(self._envs.values()))
        self.observation_spaces = {
            agent: first_env.observation_spaces["player_1"] for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: first_env.action_spaces["player_1"] for agent in self.possible_agents
        }
        self.state_spaces = {
            agent: first_env.state_spaces["player_1"] for agent in self.possible_agents
        }
        self.state_space = first_env.state_space

    @property
    def num_envs(self) -> int:
        return 1

    def observation_space(self, agent: str) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        observations = {}
        infos = {}
        base_seed = None if seed is None else int(seed)
        for idx, (agent, env) in enumerate(self._envs.items()):
            slot_seed = None if base_seed is None else base_seed + idx
            try:
                obs, info = env.reset(seed=slot_seed, options=options)
            except Exception as exc:
                raise RuntimeError(f"parallel reset failed for {agent}") from exc
            observations[agent] = obs["player_1"]
            infos[agent] = dict(info["player_1"])
            infos[agent]["parallel_agent_id"] = agent
        return observations, infos

    def step(self, actions: Mapping[str, object]):
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        done_any = False

        for agent in list(self.agents):
            env = self._envs[agent]
            obs, reward, terminated, truncated, info = env.step({"player_1": actions[agent]})
            observations[agent] = obs["player_1"]
            rewards[agent] = reward["player_1"]
            terminations[agent] = bool(terminated["player_1"])
            truncations[agent] = bool(truncated["player_1"])
            infos[agent] = dict(info["player_1"])
            infos[agent]["parallel_agent_id"] = agent
            done_any = done_any or terminations[agent] or truncations[agent]

        if done_any:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        states = [env.state() for env in self._envs.values() if env._last_obs is not None]
        if not states:
            return np.zeros(self.state_space.shape, dtype=np.float32)
        return np.mean(np.stack(states, axis=0), axis=0).astype(np.float32)

    def close(self):
        for env in self._envs.values():
            env.close()
