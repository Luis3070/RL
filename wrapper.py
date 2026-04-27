from __future__ import annotations

from typing import Dict

import numpy as np

from action import StructuredSkillAction, action_from_numpy, action_to_numpy
from environment import RealSimRobotEnv


class RealSimRobotEnvWrapper:
    def __init__(self, env: RealSimRobotEnv):
        self._env = env
        self.possible_agents = env.possible_agents
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.shared_observation_spaces = {
            agent: env.observation_spaces[agent] for agent in env.possible_agents
        }

    @property
    def agents(self):
        return self._env.agents

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, actions: Dict[str, np.ndarray | StructuredSkillAction]):
        converted = {}
        for agent, action in actions.items():
            if isinstance(action, StructuredSkillAction):
                converted[agent] = action.as_dict()
            else:
                converted[agent] = action_from_numpy(np.asarray(action, dtype=np.float32)).as_dict()
        return self._env.step(converted)

    def close(self):
        self._env.close()

    def collect_episode(self, policy_fn, seed=None):
        obs, infos = self.reset(seed=seed)
        rewards = {agent: 0.0 for agent in self.possible_agents}
        steps = 0
        while self.agents:
            action_dict = policy_fn(obs, infos)
            obs, step_rewards, terminateds, truncateds, infos = self.step(action_dict)
            steps += 1
            for agent, value in step_rewards.items():
                rewards[agent] += value
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break
        return {
            "steps": steps,
            "mean_reward": float(np.mean(list(rewards.values()))),
            "total_rewards": rewards,
        }
