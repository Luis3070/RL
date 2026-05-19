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

    def step(self, actions: Dict[str, np.ndarray | StructuredSkillAction | dict]):
        converted = {}
        for agent, action in actions.items():
            if isinstance(action, dict):
                converted[agent] = action
            elif isinstance(action, StructuredSkillAction):
                converted[agent] = action.as_dict()
            else:
                raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
                payload = action_from_numpy(raw_action).as_dict()
                payload["policy_skill_raw"] = float(raw_action[0]) if raw_action.size > 0 else 0.0
                converted[agent] = payload
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
                        "configured_obs_mode": info.get("configured_obs_mode"),
                        "reward": float(step_rewards.get(self.possible_agents[0], 0.0)),
                        "policy_skill_name": info.get("policy_skill_name"),
                        "executed_skill_name": info.get("executed_skill_name"),
                        "preferred_skill_name": info.get("preferred_skill_name"),
                        "policy_skill_raw": float(info.get("policy_skill_raw", 0.0)),
                        "policy_skill_round": int(info.get("policy_skill_round", 0)),
                        "policy_skill_index": int(info.get("policy_skill_index", 0)),
                        "policy_compact_skill_index": int(info.get("policy_compact_skill_index", 0)),
                        "policy_skill_dribble_margin": float(info.get("policy_skill_dribble_margin", 0.0)),
                        "policy_skill_entropy": float(info.get("policy_skill_entropy", 0.0)),
                        "policy_param_std_mean": float(info.get("policy_param_std_mean", 0.0)),
                        "policy_skill_prob_stand": float(info.get("policy_skill_prob_stand", 0.0)),
                        "policy_skill_prob_walk": float(info.get("policy_skill_prob_walk", 0.0)),
                        "policy_skill_prob_dribble": float(info.get("policy_skill_prob_dribble", 0.0)),
                        "policy_skill_prob_shoot": float(info.get("policy_skill_prob_shoot", 0.0)),
                        "policy_skill_prob_pass": float(info.get("policy_skill_prob_pass", 0.0)),
                        "policy_skill_prob_block": float(info.get("policy_skill_prob_block", 0.0)),
                        "policy_skill_prob_mark": float(info.get("policy_skill_prob_mark", 0.0)),
                        "policy_skill_prob_observe": float(info.get("policy_skill_prob_observe", 0.0)),
                        "handoff_override_applied": int(bool(info.get("handoff_override_applied", False))),
                        "target_repair_applied": int(bool(info.get("target_repair_applied", False))),
                        "target_repair_reason": info.get("target_repair_reason"),
                        "policy_target_x": float(info.get("policy_target_x", 0.0)),
                        "policy_target_y": float(info.get("policy_target_y", 0.0)),
                        "policy_target_theta": float(info.get("policy_target_theta", 0.0)),
                        "executed_target_x": float(info.get("executed_target_x", 0.0)),
                        "executed_target_y": float(info.get("executed_target_y", 0.0)),
                        "executed_target_theta": float(info.get("executed_target_theta", 0.0)),
                        "projection_applied": int(bool(info.get("projection_applied", False))),
                        "synthetic_pose_observation": int(bool(info.get("synthetic_pose_observation", False))),
                        "synthetic_ball_observation": int(bool(info.get("synthetic_ball_observation", False))),
                        "synthetic_pose_reason": info.get("synthetic_pose_reason"),
                        "synthetic_ball_reason": info.get("synthetic_ball_reason"),
                        "ball_regime": info.get("ball_regime"),
                        "ball_confidence_score": float(info.get("ball_confidence_score", 0.0)),
                        "ball_visible_mask": int(bool(info.get("ball_visible_mask", False))),
                        "ball_trusted_mask": int(bool(info.get("ball_trusted_mask", False))),
                        "ball_semi_trusted_mask": int(bool(info.get("ball_semi_trusted_mask", False))),
                        "ball_degraded_mask": int(bool(info.get("ball_degraded_mask", False))),
                        "ball_rescued_mask": int(bool(info.get("ball_rescued_mask", False))),
                        "ball_lost_mask": int(bool(info.get("ball_lost_mask", False))),
                        "obs_export_mode": int(info.get("obs_export_mode", 0)),
                        "corrected_exported_robot_pose": int(bool(info.get("corrected_exported_robot_pose", False))),
                        "corrected_exported_ball": int(bool(info.get("corrected_exported_ball", False))),
                        "ball_export_source": int(info.get("ball_export_source", 0)),
                        "ball_export_fresh": int(bool(info.get("ball_export_fresh", False))),
                        "natural_time_since_ball_seen": float(info.get("natural_time_since_ball_seen", 0.0)),
                        "natural_ball_seen_percentage": float(info.get("natural_ball_seen_percentage", 0.0)),
                        "natural_ball_consistent_with_game_state": int(bool(info.get("natural_ball_consistent_with_game_state", False))),
                        "pose_alignment_error_mm": float(info.get("pose_alignment_error_mm", 0.0)),
                        "pose_alignment_warning": int(bool(info.get("pose_alignment_warning", False))),
                        "ball_observation_stale": int(bool(info.get("ball_observation_stale", False))),
                        "ball_observation_collapsed": int(bool(info.get("ball_observation_collapsed", False))),
                        "estimated_robot_x": float(info.get("estimated_robot_x", 0.0)),
                        "estimated_robot_y": float(info.get("estimated_robot_y", 0.0)),
                        "estimated_ball_x": float(info.get("estimated_ball_x", 0.0)),
                        "estimated_ball_y": float(info.get("estimated_ball_y", 0.0)),
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
                        "robot_theta": float(raw_obs.get("robot_theta", 0.0)),
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
            "ball_semantics": {
                "ball_trusted_rate": float(episode_metrics.get("ball_trusted_rate", 0.0)),
                "ball_rescued_rate": float(episode_metrics.get("ball_rescued_rate", 0.0)),
                "ball_confidence_score_mean": float(episode_metrics.get("ball_confidence_score_mean", 0.0)),
                "first_trusted_ball_after_reset_ms": float(episode_metrics.get("first_trusted_ball_after_reset_ms", 0.0)),
            },
            "final_info": final_info,
            "step_records": step_records,
        }
