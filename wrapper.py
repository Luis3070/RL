from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np

from action import StructuredSkillAction, action_from_numpy
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

    def collect_episode(self, policy_fn, seed=None, capture_steps: bool = False):
        obs, infos = self.reset(seed=seed)
        rewards = {agent: 0.0 for agent in self.possible_agents}
        steps = 0
        reward_terms_sum = defaultdict(float)
        weighted_terms_sum = defaultdict(float)
        final_info = None
        step_records = []
        while self.agents:
            action_dict = policy_fn(obs, infos)
            obs, step_rewards, terminateds, truncateds, infos = self.step(action_dict)
            steps += 1
            final_info = infos.get(self.possible_agents[0], final_info)
            for agent, value in step_rewards.items():
                rewards[agent] += value
            info = infos.get(self.possible_agents[0], {})
            for key, value in info.get("reward_terms", {}).items():
                reward_terms_sum[key] += float(value)
            for key, value in info.get("reward_weighted_terms", {}).items():
                weighted_terms_sum[key] += float(value)
            if capture_steps:
                raw_obs = info.get("raw_obs", {})
                step_records.append(
                    {
                        "step": int(steps),
                        "reward": float(step_rewards.get(self.possible_agents[0], 0.0)),
                        "policy_skill_name": info.get("policy_skill_name"),
                        "executed_skill_name": info.get("executed_skill_name"),
                        "preferred_skill_name": info.get("preferred_skill_name"),
                        "projection_applied": int(bool(info.get("projection_applied", False))),
                        "takeover_envelope": int(bool(info.get("takeover_envelope", False))),
                        "walk_zone": int(bool(info.get("walk_zone", False))),
                        "dribble_zone": int(bool(info.get("dribble_zone", False))),
                        "shoot_zone": int(bool(info.get("shoot_zone", False))),
                        "bridge_frame": int(info.get("bridge_frame", 0)),
                        "ball_rel_x": float(raw_obs.get("ball_rel_x", 0.0)),
                        "ball_rel_y": float(raw_obs.get("ball_rel_y", 0.0)),
                        "ball_x": float(raw_obs.get("ball_x", 0.0)),
                        "ball_y": float(raw_obs.get("ball_y", 0.0)),
                        "robot_x": float(raw_obs.get("robot_x", 0.0)),
                        "robot_y": float(raw_obs.get("robot_y", 0.0)),
                        "time_since_ball_seen": float(raw_obs.get("time_since_ball_seen", 0.0)),
                        "ball_seen_percentage": float(raw_obs.get("ball_seen_percentage", 0.0)),
                        "shot_opening_with_obstacles": float(raw_obs.get("shot_opening_with_obstacles", 0.0)),
                        "nearest_opponent_front_dist": float(raw_obs.get("nearest_opponent_front_dist", 0.0)),
                        "nearest_front_obstacle_dist": float(raw_obs.get("nearest_front_obstacle_dist", 0.0)),
                    }
                )
            if not self.agents or any(terminateds.values()) or any(truncateds.values()):
                break
        episode_metrics = {}
        if final_info is not None:
            episode_metrics = dict(final_info.get("episode_metrics", {}))
        return {
            "steps": steps,
            "mean_reward": float(np.mean(list(rewards.values()))),
            "total_rewards": rewards,
            "reward_terms_sum": dict(reward_terms_sum),
            "reward_weighted_terms_sum": dict(weighted_terms_sum),
            "episode_metrics": episode_metrics,
            "final_info": final_info,
            "step_records": step_records,
        }
