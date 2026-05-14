from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from action import SkillType, StructuredSkillAction, action_from_mapping, action_from_numpy, skill_mask
from config import ACTION, ENV
from curriculum import (
    get_dynamic_world_update,
    get_stage_spec,
    sample_stage_world_spec,
    stage_skill_mask,
)
from observation import OBS_FEATURE_NAMES, OBS_SIZE
from pybh_backend import BackendObservation, PyBHBackend
from reward import (
    StepContext,
    StepEvaluation,
    build_fallback_action_targets,
    evaluate_step,
    features_from_obs,
    goal_scored,
    preferred_skill,
)

BALL_VEL_NORM = 3000.0
TIME_NORM_MS = 5000.0
PASS_COUNT_NORM = 5.0
UPRIGHT_MAX_FALL_STATE = 2


class RealSimRobotEnv(ParallelEnv):
    metadata = {"name": "real-simrobot-skillrequest-v2"}

    def __init__(self, env_cfg=ENV):
        super().__init__()
        self.cfg = env_cfg
        self.stage_spec = get_stage_spec(self.cfg.stage_id)
        self.possible_agents = ["player_1"]
        self.agents = list(self.possible_agents)
        self.backend = PyBHBackend(
            mode=self.cfg.backend.mode,
            frames_per_step=self.cfg.backend.frames_per_step,
            obs_timeout_ms=self.cfg.backend.obs_timeout_ms,
            simrobot_profile=self.cfg.backend.simrobot_profile,
            simrobot_scene_path=self.cfg.backend.simrobot_scene_path,
            simrobot_robot_name=self.cfg.backend.simrobot_robot_name,
            simrobot_theta_offset=self.cfg.backend.simrobot_theta_offset,
            simrobot_launch=self.cfg.backend.simrobot_launch,
            simrobot_bin=self.cfg.backend.simrobot_bin,
        )
        self.observation_spaces = {
            "player_1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        }
        self.action_spaces = {
            "player_1": gym.spaces.Box(
                low=np.asarray([0.0, -ACTION.x_range, -ACTION.y_range, -math.pi, 0.0], dtype=np.float32),
                high=np.asarray(
                    [
                        float(len(ACTION.skill_names) - 1),
                        ACTION.x_range,
                        ACTION.y_range,
                        math.pi,
                        float(ACTION.max_pass_target),
                    ],
                    dtype=np.float32,
                ),
                shape=(5,),
                dtype=np.float32,
            )
        }
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.state_spaces = {"player_1": self.state_space}
        self._rng = np.random.default_rng(self.cfg.seed)
        self._last_obs: Optional[np.ndarray] = None
        self._last_backend_obs: Optional[BackendObservation] = None
        self._last_evaluation: Optional[StepEvaluation] = None
        self._episode_step = 0
        self._skill_history: list[str] = []
        self._current_world_spec = None
        self._episode_tracker: Dict[str, object] = {}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._episode_step = 0
        self._skill_history = []
        self._last_evaluation = None
        self._current_world_spec = sample_stage_world_spec(self.stage_spec.stage_id, self._rng)
        obs = self.backend.reset(world_spec=self._current_world_spec)
        encoded = self._encode_obs(obs)
        self._last_obs = encoded
        self._last_backend_obs = obs
        self._reset_episode_tracker()
        info = self._build_info(
            obs=obs,
            policy_action=None,
            executed_action=None,
            evaluation=None,
            projection_applied=False,
            dynamic_update_applied=False,
            done=False,
        )
        return {"player_1": encoded}, {"player_1": info}

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("step() called on terminated environment")
        action_payload = actions["player_1"]
        if isinstance(action_payload, dict):
            policy_action = action_from_mapping(action_payload)
        else:
            policy_action = action_from_numpy(np.asarray(action_payload, dtype=np.float32))

        if self._last_backend_obs is None:
            raise RuntimeError("step() called before reset()")

        executed_action, projection_applied = self.project_action(policy_action, obs=self._last_backend_obs)
        obs = self.backend.step(executed_action)
        self._episode_step += 1

        dynamic_update_applied = False
        world_update = get_dynamic_world_update(
            self.stage_spec.stage_id,
            self._episode_step,
            robot_x=obs.robot_x,
            robot_y=obs.robot_y,
            ball_x=obs.ball_x,
            ball_y=obs.ball_y,
        )
        if world_update:
            obs = self.backend.set_dynamic_world(**world_update)
            dynamic_update_applied = True

        scored = goal_scored(obs.ball_x, obs.ball_y)
        timeout = (not scored) and self._episode_step >= self.cfg.max_episode_steps
        collision = self._derive_collision(obs)
        corridor_open = self._derive_corridor_open(obs)
        prev_skill_name = self._skill_history[-1] if self._skill_history else None
        skill_history = tuple((*self._skill_history, executed_action.skill_name))

        step_ctx = StepContext(
            stage=self.stage_spec.stage_id,
            episode_step=self._episode_step,
            max_episode_steps=self.cfg.max_episode_steps,
            prev_obs=self._last_backend_obs,
            obs=obs,
            skill_name=executed_action.skill_name,
            policy_skill_name=policy_action.skill_name,
            prev_skill_name=prev_skill_name,
            skill_history=skill_history,
            goal_scored=scored,
            timeout=timeout,
            collision=collision,
            corridor_open=corridor_open,
            first_to_ball=False,
            bad_duel=False,
        )
        evaluation = evaluate_step(step_ctx, self.stage_spec.reward_weights)
        encoded = self._encode_obs(obs)
        self._update_episode_tracker(step_ctx, evaluation, projection_applied, dynamic_update_applied)

        self._last_obs = encoded
        self._last_backend_obs = obs
        self._last_evaluation = evaluation
        self._skill_history.append(executed_action.skill_name)

        done = scored or timeout
        info = self._build_info(
            obs=obs,
            policy_action=policy_action,
            executed_action=executed_action,
            evaluation=evaluation,
            projection_applied=projection_applied,
            dynamic_update_applied=dynamic_update_applied,
            done=done,
        )

        rewards = {"player_1": float(evaluation.total_reward)}
        terminations = {"player_1": done}
        truncations = {"player_1": False}
        if done:
            info["episode_metrics"] = self._finalize_episode_metrics()
            self.agents = []
        return {"player_1": encoded}, rewards, terminations, truncations, {"player_1": info}

    def project_action(self, action: StructuredSkillAction, *, obs: BackendObservation) -> tuple[StructuredSkillAction, bool]:
        if action.skill_name in self.stage_spec.active_skill_names:
            return action, False

        skill_name = preferred_skill(features_from_obs(obs))
        fallback_targets = build_fallback_action_targets(skill_name, obs)
        projected = StructuredSkillAction(
            skill_type=SkillType.from_public_name(skill_name),
            target_x=float(fallback_targets["target_x"]),
            target_y=float(fallback_targets["target_y"]),
            target_theta=float(fallback_targets["target_theta"]),
            pass_target=int(action.pass_target),
        )
        return projected, True

    def close(self):
        self.backend.close()

    def get_global_state(self) -> np.ndarray:
        return self._last_obs.copy() if self._last_obs is not None else np.zeros(OBS_SIZE, dtype=np.float32)

    def state(self) -> np.ndarray:
        return self.get_global_state()

    def _encode_obs(self, obs: BackendObservation) -> np.ndarray:
        def norm_time(value: float) -> float:
            return float(np.clip(value / TIME_NORM_MS, 0.0, 1.0))

        def norm_dist(value: float) -> float:
            return float(np.clip(value / ACTION.x_range, 0.0, 1.0))

        return np.asarray(
            [
                obs.robot_x / ACTION.x_range,
                obs.robot_y / ACTION.y_range,
                obs.robot_theta / math.pi,
                obs.ball_rel_x / ACTION.x_range,
                obs.ball_rel_y / ACTION.y_range,
                obs.ball_end_rel_x / ACTION.x_range,
                obs.ball_end_rel_y / ACTION.y_range,
                obs.ball_vel_x / BALL_VEL_NORM,
                obs.ball_vel_y / BALL_VEL_NORM,
                norm_time(obs.time_since_ball_seen),
                norm_time(obs.time_since_ball_disappeared),
                float(np.clip(obs.ball_seen_percentage / 100.0, 0.0, 1.0)),
                1.0 if obs.ball_consistent_with_game_state else 0.0,
                1.0 if obs.can_score_now else 0.0,
                float(np.clip(obs.shot_quality_no_obstacles, 0.0, 1.0)),
                float(np.clip(obs.shot_opening_with_obstacles, 0.0, 1.0)),
                float(np.clip(obs.pass_options_count / PASS_COUNT_NORM, 0.0, 1.0)),
                norm_dist(obs.nearest_teammate_dist),
                norm_dist(obs.nearest_opponent_dist),
                norm_dist(obs.nearest_uncertain_obstacle_dist),
                norm_dist(obs.nearest_teammate_front_dist),
                norm_dist(obs.nearest_opponent_front_dist),
                norm_dist(obs.nearest_uncertain_front_dist),
            ],
            dtype=np.float32,
        )

    def _reset_episode_tracker(self) -> None:
        self._episode_tracker = {
            "reward_total": 0.0,
            "reward_terms_sum": {},
            "reward_weighted_terms_sum": {},
            "skill_counts": {name: 0 for name in ACTION.skill_names},
            "policy_skill_counts": {name: 0 for name in ACTION.skill_names},
            "skill_projection_count": 0,
            "walk_steps": 0,
            "dribble_steps": 0,
            "shoot_steps": 0,
            "good_walk_steps": 0,
            "good_dribble_steps": 0,
            "good_shoot_steps": 0,
            "bad_walk_steps": 0,
            "bad_dribble_steps": 0,
            "bad_shoot_steps": 0,
            "walk_zone_steps": 0,
            "dribble_zone_steps": 0,
            "shoot_zone_steps": 0,
            "takeover_entered": False,
            "takeover_first_step": None,
            "takeover_hold_current": 0,
            "takeover_hold_max": 0,
            "incompatible_takeover_requests": 0,
            "good_finish_steps": 0,
            "goal": False,
            "collision_count": 0,
            "thrash_count": 0,
            "clearance_positive_steps": 0,
            "obstacle_present_steps": 0,
            "corridor_keep_steps": 0,
            "dynamic_updates": 0,
            "post_takeover_progress_sum": 0.0,
            "synthetic_pose_steps": 0,
            "synthetic_ball_steps": 0,
        }

    def _update_episode_tracker(
        self,
        step_ctx: StepContext,
        evaluation: StepEvaluation,
        projection_applied: bool,
        dynamic_update_applied: bool,
    ) -> None:
        tracker = self._episode_tracker
        tracker["reward_total"] += evaluation.total_reward
        for container_name, source in (
            ("reward_terms_sum", evaluation.terms),
            ("reward_weighted_terms_sum", evaluation.weighted_terms),
        ):
            container = tracker[container_name]
            for key, value in source.items():
                container[key] = float(container.get(key, 0.0) + value)

        tracker["skill_counts"][step_ctx.skill_name] += 1
        tracker["policy_skill_counts"][step_ctx.policy_skill_name] += 1
        if projection_applied:
            tracker["skill_projection_count"] += 1
        if dynamic_update_applied:
            tracker["dynamic_updates"] += 1
        tracker["synthetic_pose_steps"] += int(bool(step_ctx.obs.synthetic_pose_observation))
        tracker["synthetic_ball_steps"] += int(bool(step_ctx.obs.synthetic_ball_observation))

        if step_ctx.skill_name == "walk":
            tracker["walk_steps"] += 1
        elif step_ctx.skill_name == "dribble":
            tracker["dribble_steps"] += 1
        elif step_ctx.skill_name == "shoot":
            tracker["shoot_steps"] += 1

        if evaluation.walk_zone:
            tracker["walk_zone_steps"] += 1
        if evaluation.dribble_zone:
            tracker["dribble_zone_steps"] += 1
        if evaluation.shoot_zone:
            tracker["shoot_zone_steps"] += 1
        if evaluation.features.front_obstacle_count > 0 or evaluation.features.front_obstacle_mm > 0.0:
            tracker["obstacle_present_steps"] += 1

        tracker["good_walk_steps"] += int(evaluation.terms["good_walk"] > 0.0)
        tracker["good_dribble_steps"] += int(evaluation.terms["good_dribble"] > 0.0)
        tracker["good_shoot_steps"] += int(evaluation.terms["good_shoot"] > 0.0)
        tracker["bad_walk_steps"] += int(evaluation.terms["bad_walk"] > 0.0)
        tracker["bad_dribble_steps"] += int(evaluation.terms["bad_dribble"] > 0.0)
        tracker["bad_shoot_steps"] += int(evaluation.terms["bad_shoot"] > 0.0)

        if evaluation.takeover_envelope:
            tracker["takeover_hold_current"] += 1
            tracker["takeover_hold_max"] = max(tracker["takeover_hold_max"], tracker["takeover_hold_current"])
            if not tracker["takeover_entered"]:
                tracker["takeover_entered"] = True
                tracker["takeover_first_step"] = step_ctx.episode_step
        else:
            tracker["takeover_hold_current"] = 0

        if evaluation.takeover_envelope and step_ctx.skill_name not in {"dribble", "shoot"}:
            tracker["incompatible_takeover_requests"] += 1

        if evaluation.terms["good_shoot"] > 0.0 or evaluation.terms["good_dribble"] > 0.0:
            tracker["good_finish_steps"] += 1
        if evaluation.terms["collision"] > 0.0:
            tracker["collision_count"] += 1
        if evaluation.terms["thrash"] > 0.0:
            tracker["thrash_count"] += 1
        if evaluation.terms["clearance_gain"] > 0.0:
            tracker["clearance_positive_steps"] += 1
        if evaluation.terms["corridor_keep"] > 0.0:
            tracker["corridor_keep_steps"] += 1

        tracker["post_takeover_progress_sum"] += evaluation.terms["ball_progress_post_takeover"]
        tracker["goal"] = tracker["goal"] or step_ctx.goal_scored

    def _derive_collision(self, obs: BackendObservation) -> bool:
        fall_event = int(obs.motion_engine_fall_state) > UPRIGHT_MAX_FALL_STATE
        contact_event = bool(obs.obstacle_arm_contact or obs.obstacle_foot_contact)
        return bool(fall_event or contact_event)

    def _derive_corridor_open(self, obs: BackendObservation) -> bool:
        if int(obs.obstacle_front_count) <= 0:
            return True
        if float(obs.nearest_front_obstacle_dist) >= 700.0:
            return True
        return float(obs.shot_opening_with_obstacles) >= 0.35

    def _running_metrics_snapshot(self) -> Dict[str, float]:
        tracker = self._episode_tracker
        steps = max(1, self._episode_step)
        walk_steps = max(1, tracker["walk_steps"])
        dribble_steps = max(1, tracker["dribble_steps"])
        shoot_steps = max(1, tracker["shoot_steps"])
        return {
            "reward_total": float(tracker["reward_total"]),
            "thrash_rate": float(tracker["thrash_count"] / steps),
            "collision_step_rate": float(tracker["collision_count"] / steps),
            "walk_compat_rate": float(tracker["good_walk_steps"] / walk_steps),
            "dribble_compat_rate": float(tracker["good_dribble_steps"] / dribble_steps),
            "shoot_compat_rate": float(tracker["good_shoot_steps"] / shoot_steps),
            "takeover_entered": float(bool(tracker["takeover_entered"])),
            "takeover_hold_max": float(tracker["takeover_hold_max"]),
            "synthetic_pose_rate": float(tracker["synthetic_pose_steps"] / steps),
            "synthetic_ball_rate": float(tracker["synthetic_ball_steps"] / steps),
        }

    def _finalize_episode_metrics(self) -> Dict[str, float]:
        tracker = self._episode_tracker
        steps = max(1, self._episode_step)
        walk_steps = tracker["walk_steps"]
        dribble_steps = tracker["dribble_steps"]
        shoot_steps = tracker["shoot_steps"]

        walk_compat_rate = float(tracker["good_walk_steps"] / max(1, walk_steps))
        dribble_compat_rate = float(tracker["good_dribble_steps"] / max(1, dribble_steps))
        shoot_compat_rate = float(tracker["good_shoot_steps"] / max(1, shoot_steps))
        bad_walk_rate = float(tracker["bad_walk_steps"] / max(1, walk_steps))
        bad_dribble_rate = float(tracker["bad_dribble_steps"] / max(1, dribble_steps))
        bad_shoot_rate = float(tracker["bad_shoot_steps"] / max(1, shoot_steps))
        thrash_rate = float(tracker["thrash_count"] / steps)
        collision_episode = float(tracker["collision_count"] > 0)
        mean_time_to_takeover = float(tracker["takeover_first_step"] or self.cfg.max_episode_steps)
        takeover_success = bool(tracker["takeover_entered"] and tracker["takeover_hold_max"] >= 2)

        clean_walk = bool(walk_steps > 0 and walk_compat_rate >= 0.7 and tracker["thrash_count"] == 0)
        clean_takeover = bool(takeover_success and tracker["incompatible_takeover_requests"] == 0)
        clean_finish = bool(
            tracker["good_finish_steps"] > 0
            and (tracker["goal"] or tracker["post_takeover_progress_sum"] > 0.2)
            and tracker["thrash_count"] <= 1
        )
        clean_episode = bool(clean_takeover and clean_finish and tracker["collision_count"] == 0)

        obstacle_clearance_success = 0.0
        if self.stage_spec.stage_id >= 2 and tracker["obstacle_present_steps"] > 0:
            obstacle_clearance_success = float(clean_takeover and tracker["collision_count"] == 0 and tracker["clearance_positive_steps"] > 0)

        corridor_keep_rate = 0.0
        if self.stage_spec.stage_id >= 3 and tracker["obstacle_present_steps"] > 0:
            corridor_keep_rate = float(tracker["corridor_keep_steps"] / tracker["obstacle_present_steps"])

        dynamic_obstacle_success = 0.0
        if self.stage_spec.stage_id >= 3 and tracker["dynamic_updates"] > 0:
            dynamic_obstacle_success = float(clean_takeover and tracker["collision_count"] == 0 and corridor_keep_rate >= 0.5)

        stability_score = float(
            np.clip(
                0.4 * (1.0 - thrash_rate)
                + 0.2 * walk_compat_rate
                + 0.2 * dribble_compat_rate
                + 0.2 * shoot_compat_rate,
                0.0,
                1.0,
            )
        )

        return {
            "stage_id": float(self.stage_spec.stage_id),
            "steps": float(self._episode_step),
            "episode_reward_mean": float(tracker["reward_total"] / steps),
            "episode_reward_total": float(tracker["reward_total"]),
            "clean_walk": float(clean_walk),
            "clean_takeover": float(clean_takeover),
            "clean_finish": float(clean_finish),
            "clean_episode": float(clean_episode),
            "collision_episode": collision_episode,
            "takeover_success": float(takeover_success),
            "goal_rate": float(tracker["goal"]),
            "thrash_rate": thrash_rate,
            "mean_time_to_takeover": mean_time_to_takeover,
            "walk_compat_rate": walk_compat_rate,
            "dribble_compat_rate": dribble_compat_rate,
            "shoot_compat_rate": shoot_compat_rate,
            "bad_walk_rate": bad_walk_rate,
            "bad_dribble_rate": bad_dribble_rate,
            "bad_shoot_rate": bad_shoot_rate,
            "obstacle_clearance_success": obstacle_clearance_success,
            "dynamic_obstacle_success": dynamic_obstacle_success,
            "corridor_keep_rate": corridor_keep_rate,
            "ball_progress_post_takeover_mean": float(tracker["post_takeover_progress_sum"] / steps),
            "stability_score": stability_score,
            "skill_projection_rate": float(tracker["skill_projection_count"] / steps),
            "synthetic_pose_rate": float(tracker["synthetic_pose_steps"] / steps),
            "synthetic_ball_rate": float(tracker["synthetic_ball_steps"] / steps),
            "reward_terms_sum": dict(tracker["reward_terms_sum"]),
            "reward_weighted_terms_sum": dict(tracker["reward_weighted_terms_sum"]),
            "skill_counts": dict(tracker["skill_counts"]),
            "policy_skill_counts": dict(tracker["policy_skill_counts"]),
        }

    def _build_info(
        self,
        *,
        obs: BackendObservation,
        policy_action: StructuredSkillAction | None,
        executed_action: StructuredSkillAction | None,
        evaluation: StepEvaluation | None,
        projection_applied: bool,
        dynamic_update_applied: bool,
        done: bool,
    ) -> Dict[str, object]:
        raw_obs = asdict(obs)
        info: Dict[str, object] = {
            "state": self.get_global_state(),
            "obs_feature_names": OBS_FEATURE_NAMES,
            "backend_mode": self.cfg.backend.mode,
            "stage_id": self.stage_spec.stage_id,
            "stage_name": self.stage_spec.name,
            "active_skill_mask": skill_mask(self.stage_spec.active_skill_names),
            "stage_skill_mask": stage_skill_mask(self.stage_spec.stage_id),
            "bridge_obs_ready": obs.obs_ready,
            "bridge_frame": obs.frame,
            "synthetic_pose_observation": bool(obs.synthetic_pose_observation),
            "synthetic_ball_observation": bool(obs.synthetic_ball_observation),
            "raw_obs": raw_obs,
            "policy_skill_name": None if policy_action is None else policy_action.skill_name,
            "executed_skill_name": None if executed_action is None else executed_action.skill_name,
            "projection_applied": bool(projection_applied),
            "dynamic_update_applied": bool(dynamic_update_applied),
            "skill_history": tuple(self._skill_history),
            "running_metrics": self._running_metrics_snapshot(),
            # Compatibility shims for existing tooling
            "requested_skill": obs.requested_skill,
            "requested_pass_target": obs.requested_pass_target,
            "ball_x": obs.ball_x,
            "ball_y": obs.ball_y,
            "robot_x": obs.robot_x,
            "robot_y": obs.robot_y,
            "robot_theta": obs.robot_theta,
            "ball_rel_x": obs.ball_rel_x,
            "ball_rel_y": obs.ball_rel_y,
            "time_since_ball_seen": obs.time_since_ball_seen,
            "ball_seen_percentage": obs.ball_seen_percentage,
            "ball_consistent_with_game_state": obs.ball_consistent_with_game_state,
            "shot_quality_no_obstacles": obs.shot_quality_no_obstacles,
            "shot_opening_with_obstacles": obs.shot_opening_with_obstacles,
            "pass_options_count": obs.pass_options_count,
        }
        if evaluation is not None:
            info.update(evaluation.as_dict())
        if done:
            info["episode_done"] = True
        return info
