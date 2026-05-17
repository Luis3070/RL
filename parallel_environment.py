from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Mapping

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from config import EnvConfig
from environment import RealSimRobotEnv


class ParallelSlotSimRobotEnv(ParallelEnv):
    """PettingZoo ParallelEnv backed by one SimRobot slot per agent.

    Steps and resets are dispatched to all slots concurrently via a thread
    pool.  Each slot's rl_get_obs() blocks on its own shared-memory region
    and releases the GIL, so the 6 waits overlap instead of serialising.
    Observed gain: ~6x throughput on 6-slot visible training (174ms→~33ms
    per RL step).
    """

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

        n = len(self.player_numbers)
        self._executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix="slot_step")

    @property
    def num_envs(self) -> int:
        return 1

    def observation_space(self, agent: str) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        base_seed = None if seed is None else int(seed)

        def _reset_one(args: tuple) -> tuple[str, dict, dict]:
            idx, agent, env = args
            slot_seed = None if base_seed is None else base_seed + idx
            obs, info = env.reset(seed=slot_seed, options=options)
            return agent, obs["player_1"], dict(info["player_1"])

        futures = [
            self._executor.submit(_reset_one, (idx, agent, env))
            for idx, (agent, env) in enumerate(self._envs.items())
        ]
        observations: dict[str, object] = {}
        infos: dict[str, dict] = {}
        for future in as_completed(futures):
            agent, obs, info = future.result()
            observations[agent] = obs
            infos[agent] = info
            infos[agent]["parallel_agent_id"] = agent
        return observations, infos

    def step(self, actions: Mapping[str, object]):
        def _step_one(agent: str) -> tuple[str, tuple]:
            env = self._envs[agent]
            result = env.step({"player_1": actions[agent]})
            return agent, result

        futures = {
            self._executor.submit(_step_one, agent): agent
            for agent in list(self.agents)
        }
        observations: dict[str, object] = {}
        rewards: dict[str, object] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}
        done_any = False

        for future in as_completed(futures):
            agent, (obs, reward, terminated, truncated, info) = future.result()
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
        self._executor.shutdown(wait=False)
        for env in self._envs.values():
            env.close()
