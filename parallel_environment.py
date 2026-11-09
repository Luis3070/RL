from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Mapping

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from config import EnvConfig
from environment import RealSimRobotEnv

# H1 — Phase synchronization
# Each visible SimRobot slot has a fixed camera frame clock (~30ms period) that
# is independent of when Python dispatches actions. Without staggering, slots
# dispatched simultaneously arrive at different points in their frame cycle,
# giving wait times that range from T_frame (best) to 2*T_frame (worst).
# With RL_PHASE_SYNC=1 each slot is dispatched just before its next predicted
# frame so all waits converge toward T_frame, cutting worst-case latency.
_PHASE_SYNC = os.getenv("RL_PHASE_SYNC", "0") == "1"
_PHASE_T_FRAME_S = float(os.getenv("RL_PHASE_T_FRAME_MS", "30")) / 1000.0
_PHASE_T_LEAD_S = float(os.getenv("RL_PHASE_T_LEAD_MS", "3")) / 1000.0
_PARALLEL_RESET_WORKERS = max(1, int(os.getenv("RL_PARALLEL_RESET_WORKERS", "1")))
# No EMA — _PHASE_T_FRAME_S is the camera frame period fixed by ci configuration.
# ci 30 → 33ms, ci 60 → 17ms. Do NOT estimate it from step-to-step return
# intervals (those measure step duration, not camera period).

# H2 — ci=60 (ABANDONED): tested with RL1v0_3D_visible_ci60.ros2/.con.
# Speed gain confirmed (~37 steps/sec), but BHuman BallPerceptor fails at 60fps:
# natural_time_since_ball_seen=26-270s, synthetic_ball_observation=100%, fine_scale=0.
# Ball perception is calibrated for ci=30. Do not use ci=60 for training.

# H3 — Second training process: set PYBH_RL_SHM_NAME to a unique value in each
# shell before launching train.py. Each process gets its own SHM region and its
# own SimRobot instances, so they never conflict. Both train independent policies
# at ~27 steps/sec each — useful for parallel hyperparameter search.
# Example:
#   PYBH_RL_SHM_NAME=/pybh_rl_A RL_RUN_NAME=run_A python train.py &
#   PYBH_RL_SHM_NAME=/pybh_rl_B RL_RUN_NAME=run_B python train.py &


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
        # Phase sync state — wall-clock time when each slot last delivered obs
        self._last_return_time: dict[str, float] = {}

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

        observations: dict[str, object] = {}
        infos: dict[str, dict] = {}
        reset_items = list(enumerate(self._envs.items()))
        if _PARALLEL_RESET_WORKERS <= 1:
            for idx, (agent, env) in reset_items:
                agent, obs, info = _reset_one((idx, agent, env))
                observations[agent] = obs
                infos[agent] = info
                infos[agent]["parallel_agent_id"] = agent
        else:
            with ThreadPoolExecutor(max_workers=_PARALLEL_RESET_WORKERS, thread_name_prefix="slot_reset") as reset_executor:
                futures = [
                    reset_executor.submit(_reset_one, (idx, agent, env))
                    for idx, (agent, env) in reset_items
                ]
                for future in as_completed(futures):
                    agent, obs, info = future.result()
                    observations[agent] = obs
                    infos[agent] = info
                    infos[agent]["parallel_agent_id"] = agent
        return observations, infos

    def step(self, actions: Mapping[str, object]):
        def _step_one(agent: str) -> tuple[str, tuple, float]:
            if _PHASE_SYNC and agent in self._last_return_time:
                # Predict when the slot's next camera frame fires:
                #   last_return ≈ time the frame fired → next at last_return + T_frame
                # Dispatch just before that so the action arrives in the same cycle,
                # reducing worst-case wait from 2*T_frame to ~T_frame.
                t_target = self._last_return_time[agent] + _PHASE_T_FRAME_S - _PHASE_T_LEAD_S
                sleep_s = t_target - time.perf_counter()
                if sleep_s > 0.001:
                    time.sleep(sleep_s)

            env = self._envs[agent]
            result = env.step({"player_1": actions[agent]})
            return agent, result, time.perf_counter()

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
            agent, (obs, reward, terminated, truncated, info), t_ret = future.result()
            observations[agent] = obs["player_1"]
            rewards[agent] = reward["player_1"]
            terminations[agent] = bool(terminated["player_1"])
            truncations[agent] = bool(truncated["player_1"])
            infos[agent] = dict(info["player_1"])
            infos[agent]["parallel_agent_id"] = agent
            done_any = done_any or terminations[agent] or truncations[agent]

            if _PHASE_SYNC:
                self._last_return_time[agent] = t_ret

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
