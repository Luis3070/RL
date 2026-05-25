from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, Mapping, Optional

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from action import (
    SKILL_ORDER,
    SkillType,
    StructuredSkillAction,
    action_from_mapping,
    action_from_numpy,
    action_param_mask,
    skill_mask,
)
from config import ACTION, ENV
from curriculum import (
    get_dynamic_world_update,
    get_stage_spec,
    init_dynamic_world_state,
    sample_finish_sequence_world_spec,
    sample_finish_world_spec,
    sample_preshoot_dribble_world_spec,
    sample_stage_world_spec_with_options,
    stage_skill_mask,
)
from observation import OBS_FEATURE_NAMES, OBS_SIZE
from pybh_backend import BackendObservation, PyBHBackend, WorldSpec
from reward import (
    TAKEOVER_GRACE_STEPS,
    StepContext,
    StepEvaluation,
    build_fallback_action_targets,
    evaluate_step,
    features_from_obs,
    goal_scored,
    is_geometric_takeover,
)

BALL_VEL_NORM = 3000.0
TIME_NORM_MS = 5000.0
PASS_COUNT_NORM = 5.0
UPRIGHT_MAX_FALL_STATE = 2
BALL_VISIBLE_SEEN_PCT_MIN = 5.0
BALL_VISIBLE_MAX_AGE_MS = 1200.0
# BALL_TRUSTED_SEEN_PCT_MIN: removed — with ci 30 the seenPercentage window (60
# frames) contains only 2 image frames, capping the achievable value at 3.3%.
# The 20% threshold could never be met regardless of detection quality.
# Trust is now based on natural_time_since_ball_seen (Option B).
BALL_TRUSTED_MAX_AGE_MS = 600.0
BALL_SEMI_TRUSTED_MAX_AGE_MS = 1000.0

_STEP_PROFILE = os.getenv("RL_PROFILE_STEP", "0") == "1"
_PROFILE_INTERVAL = int(os.getenv("RL_PROFILE_INTERVAL", "256"))
POSE_JUMP_WARNING_MM = 350.0


def _ball_observation_masks(obs: BackendObservation) -> Dict[str, object]:
    seen_pct = float(obs.ball_seen_percentage)
    time_since_seen = float(obs.time_since_ball_seen)
    rescued = bool(obs.synthetic_ball_observation or obs.corrected_exported_ball)
    visible = bool(seen_pct > BALL_VISIBLE_SEEN_PCT_MIN or time_since_seen <= BALL_VISIBLE_MAX_AGE_MS)
    trusted = bool(
        (not rescued)
        and float(obs.natural_time_since_ball_seen) <= BALL_TRUSTED_MAX_AGE_MS
    )
    semi_trusted = bool(
        (not rescued)
        and seen_pct > 0.0
        and time_since_seen <= BALL_SEMI_TRUSTED_MAX_AGE_MS
    )
    lost = bool((not rescued) and seen_pct <= 0.0 and time_since_seen > BALL_SEMI_TRUSTED_MAX_AGE_MS)
    degraded = bool((not rescued) and visible and (not trusted))
    confidence = 0.0
    if visible:
        seen_term = float(np.clip(seen_pct / 100.0, 0.0, 1.0))
        age_term = float(np.clip(1.0 - (time_since_seen / BALL_VISIBLE_MAX_AGE_MS), 0.0, 1.0))
        confidence = 0.5 * seen_term + 0.5 * age_term
        if rescued:
            confidence *= 0.25

    if rescued:
        regime = "rescued"
    elif trusted:
        regime = "trusted"
    elif semi_trusted:
        regime = "semi_trusted"
    elif degraded:
        regime = "degraded"
    else:
        regime = "lost"

    return {
        "ball_visible_mask": visible,
        "ball_trusted_mask": trusted,
        "ball_semi_trusted_mask": semi_trusted,
        "ball_degraded_mask": degraded,
        "ball_rescued_mask": rescued,
        "ball_lost_mask": lost,
        "ball_confidence_score": float(confidence),
        "ball_regime": regime,
    }


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
            obs_mode=self.cfg.backend.obs_mode,
            player_number=self.cfg.backend.player_number,
            frames_per_step=self.cfg.backend.frames_per_step,
            obs_timeout_ms=self.cfg.backend.obs_timeout_ms,
            simrobot_profile=self.cfg.backend.simrobot_profile,
            simrobot_scene_path=self.cfg.backend.simrobot_scene_path,
            simrobot_robot_name=self.cfg.backend.simrobot_robot_name,
            simrobot_theta_offset=self.cfg.backend.simrobot_theta_offset,
            simrobot_launch=self.cfg.backend.simrobot_launch,
            embedded_qt_platform=self.cfg.backend.embedded_qt_platform,
            embedded_3d_preflight=self.cfg.backend.embedded_3d_preflight,
            simrobot_bin=self.cfg.backend.simrobot_bin,
        )
        self.observation_spaces = {
            "player_1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        }
        self.action_spaces = {
            "player_1": gym.spaces.Box(
                low=np.asarray([0.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32),
                high=np.asarray(
                    [
                        float(len(SKILL_ORDER) - 1),
                        1.0,
                        1.0,
                        1.0,
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
        self._trusted_takeover_grace = 0
        self._post_takeover_finishing_steps = 0
        self._episode_stage_id = self.stage_spec.stage_id
        self._episode_stage_spec = self.stage_spec
        self._current_world_spec = None
        self._dynamic_world_state = None
        self._episode_tracker: Dict[str, object] = {}
        if _STEP_PROFILE:
            self._prof_acc: Dict[str, float] = defaultdict(float)
            self._prof_count = 0

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
        self._trusted_takeover_grace = 0
        self._post_takeover_finishing_steps = 0
        self._last_evaluation = None
        options = options or {}
        self._episode_stage_id = self._sample_episode_stage_id(options)
        self._episode_stage_spec = get_stage_spec(self._episode_stage_id)
        self._current_world_spec = options.get("world_spec") or self._sample_episode_world_spec()
        self._dynamic_world_state = init_dynamic_world_state(self._episode_stage_id, self._current_world_spec, self._rng)
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

    def _sample_episode_stage_id(self, options: Mapping[str, object]) -> int:
        if "stage_id" in options:
            return int(options["stage_id"])
        probability = float(np.clip(self.cfg.stage0_finish_interleave_probability, 0.0, 1.0))
        if self.stage_spec.stage_id == 0 and probability > 0.0 and float(self._rng.random()) < probability:
            return 1
        return self.stage_spec.stage_id

    def _sample_episode_world_spec(self) -> WorldSpec:
        finish_probability = {
            1: float(getattr(self.cfg, "stage1_finish_interleave_probability", 0.0)),
            2: float(getattr(self.cfg, "stage2_finish_interleave_probability", 0.0)),
            3: float(getattr(self.cfg, "stage3_finish_interleave_probability", 0.0)),
        }.get(self._episode_stage_id, 0.0)
        if finish_probability > 0.0 and float(self._rng.random()) < float(np.clip(finish_probability, 0.0, 1.0)):
            draw = float(self._rng.random())
            if draw < 0.60:
                return sample_finish_sequence_world_spec(self._episode_stage_id, self._rng)
            if draw < 0.85:
                return sample_preshoot_dribble_world_spec(self._episode_stage_id, self._rng)
            return sample_finish_world_spec(self._episode_stage_id, self._rng)
        return sample_stage_world_spec_with_options(
            self._episode_stage_id,
            self._rng,
            diverse_resets=bool(self.cfg.diverse_stage_resets),
        )

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("step() called on terminated environment")
        action_payload = actions["player_1"]
        policy_skill_raw = 0.0
        policy_skill_probs = {name: 0.0 for name in SKILL_ORDER}
        policy_skill_entropy = 0.0
        policy_param_std_mean = 0.0
        policy_residual_x = 0.0
        policy_residual_y = 0.0
        policy_residual_theta = 0.0
        if isinstance(action_payload, dict):
            policy_skill_raw = float(action_payload.get("policy_skill_raw", action_payload.get("skill_type", 0.0)))
            policy_skill_entropy = float(action_payload.get("policy_skill_entropy", 0.0))
            policy_param_std_mean = float(action_payload.get("policy_param_std_mean", 0.0))
            for name in SKILL_ORDER:
                policy_skill_probs[name] = float(action_payload.get(f"policy_skill_prob_{name}", 0.0))
            policy_action = action_from_mapping(action_payload)
        else:
            raw_action = np.asarray(action_payload, dtype=np.float32).reshape(-1)
            policy_skill_raw = float(raw_action[0]) if raw_action.size > 0 else 0.0
            policy_action = action_from_numpy(raw_action)

        policy_skill_round = max(0, min(int(policy_action.skill_type), len(SKILL_ORDER) - 1))
        policy_compact_skill_index = policy_skill_round
        raw_policy_action = StructuredSkillAction(
            skill_type=policy_skill_round,
            target_x=policy_action.target_x,
            target_y=policy_action.target_y,
            target_theta=policy_action.target_theta,
            pass_target=policy_action.pass_target,
        )

        if self._last_backend_obs is None:
            raise RuntimeError("step() called before reset()")

        policy_residual_x = float(np.clip(raw_policy_action.target_x, -1.0, 1.0))
        policy_residual_y = float(np.clip(raw_policy_action.target_y, -1.0, 1.0))
        policy_residual_theta = float(np.clip(raw_policy_action.target_theta, -1.0, 1.0))
        policy_action = self._decode_residual_policy_action(raw_policy_action, obs=self._last_backend_obs)

        handoff_override_applied = False
        target_repair_applied = False
        target_repair_reason = ""
        action_for_execution = policy_action
        if (
            self.cfg.enable_handoff_override
            and policy_action.skill_name == "walk"
            and self._should_force_dribble_handoff(self._last_backend_obs)
        ):
            handoff_override_applied = True
            fallback_targets = build_fallback_action_targets("dribble", self._last_backend_obs)
            action_for_execution = StructuredSkillAction(
                skill_type=SkillType.from_public_name("dribble"),
                target_x=fallback_targets["target_x"],
                target_y=fallback_targets["target_y"],
                target_theta=fallback_targets["target_theta"],
                pass_target=policy_action.pass_target,
            )
        else:
            action_for_execution, target_repair_applied, target_repair_reason = self._repair_execution_targets(
                policy_action,
                obs=self._last_backend_obs,
            )
        policy_target_error_mm, policy_theta_error_rad = self._policy_anchor_errors(
            policy_action,
            obs=self._last_backend_obs,
        )

        executed_action, projection_applied = self.project_action(action_for_execution, obs=self._last_backend_obs)
        if _STEP_PROFILE:
            _t0 = time.perf_counter()
        obs = self.backend.step(executed_action)
        if _STEP_PROFILE:
            _t1 = time.perf_counter()
        self._episode_step += 1

        dynamic_update_applied = False
        world_update = get_dynamic_world_update(
            self._episode_stage_id,
            self._episode_step,
            robot_x=obs.robot_x,
            robot_y=obs.robot_y,
            ball_x=obs.ball_x,
            ball_y=obs.ball_y,
            dynamic_state=self._dynamic_world_state,
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

        # Takeover commitment latch: once a stable trusted takeover has occurred,
        # keep treating the robot as committed for a short grace window as long as
        # the ball stays geometrically in front, so a brief perception-trust
        # dropout does not flip the dribble reward sign (see reward.evaluate_step).
        takeover_committed = bool(
            self._trusted_takeover_grace > 0
            and is_geometric_takeover(features_from_obs(obs))
        )
        finishing_active = bool(
            self.cfg.post_takeover_progress_enabled
            and self._post_takeover_finishing_steps > 0
        )

        step_ctx = StepContext(
            stage=self._episode_stage_id,
            episode_step=self._episode_step,
            max_episode_steps=self.cfg.max_episode_steps,
            prev_obs=self._last_backend_obs,
            obs=obs,
            skill_name=executed_action.skill_name,
            policy_skill_name=policy_action.skill_name,
            projection_applied=projection_applied,
            prev_skill_name=prev_skill_name,
            skill_history=skill_history,
            goal_scored=scored,
            timeout=timeout,
            collision=collision,
            corridor_open=corridor_open,
            first_to_ball=False,
            bad_duel=False,
            policy_skill_raw=policy_skill_raw,
            policy_skill_prob_walk=float(policy_skill_probs.get("walk", 0.0)),
            policy_skill_prob_dribble=float(policy_skill_probs.get("dribble", 0.0)),
            policy_skill_prob_shoot=float(policy_skill_probs.get("shoot", 0.0)),
            handoff_override_applied=handoff_override_applied,
            takeover_committed=takeover_committed,
            post_takeover_finishing_active=finishing_active,
        )
        evaluation = evaluate_step(step_ctx, self._episode_stage_spec.reward_weights)
        if evaluation.stable_trusted_takeover:
            self._trusted_takeover_grace = TAKEOVER_GRACE_STEPS
            self._post_takeover_finishing_steps = max(
                self._post_takeover_finishing_steps,
                int(self.cfg.post_takeover_progress_window_steps),
            )
        else:
            self._trusted_takeover_grace = max(0, self._trusted_takeover_grace - 1)
            self._post_takeover_finishing_steps = max(0, self._post_takeover_finishing_steps - 1)
        if _STEP_PROFILE:
            _t2 = time.perf_counter()
        encoded = self._encode_obs(obs)
        if _STEP_PROFILE:
            _t3 = time.perf_counter()
        self._update_episode_tracker(
            step_ctx,
            evaluation,
            projection_applied,
            dynamic_update_applied,
            target_repair_applied=target_repair_applied,
            target_repair_reason=target_repair_reason,
            policy_target_error_mm=policy_target_error_mm,
            policy_theta_error_rad=policy_theta_error_rad,
            policy_skill_entropy=policy_skill_entropy,
            policy_param_std_mean=policy_param_std_mean,
        )

        self._last_obs = encoded
        self._last_backend_obs = obs
        self._last_evaluation = evaluation
        self._skill_history.append(executed_action.skill_name)

        prestage_takeover_done = (
            self._episode_stage_id < 0
            and evaluation.takeover_envelope
            and evaluation.prev_takeover_envelope
        )
        done = scored or timeout or prestage_takeover_done
        info = self._build_info(
            obs=obs,
            policy_action=policy_action,
            executed_action=executed_action,
            evaluation=evaluation,
            projection_applied=projection_applied,
            dynamic_update_applied=dynamic_update_applied,
            policy_skill_raw=policy_skill_raw,
            policy_skill_round=policy_skill_round,
            policy_compact_skill_index=policy_compact_skill_index,
            policy_skill_probs=policy_skill_probs,
            policy_skill_entropy=policy_skill_entropy,
            policy_param_std_mean=policy_param_std_mean,
            policy_residual_x=policy_residual_x,
            policy_residual_y=policy_residual_y,
            policy_residual_theta=policy_residual_theta,
            policy_target_error_mm=policy_target_error_mm,
            policy_theta_error_rad=policy_theta_error_rad,
            handoff_override_applied=handoff_override_applied,
            target_repair_applied=target_repair_applied,
            target_repair_reason=target_repair_reason,
            done=done,
        )

        if _STEP_PROFILE:
            _t4 = time.perf_counter()
            self._prof_acc["backend_ms"] += (_t1 - _t0) * 1e3
            self._prof_acc["reward_ms"] += (_t2 - _t1) * 1e3
            self._prof_acc["encode_ms"] += (_t3 - _t2) * 1e3
            self._prof_acc["other_ms"] += (_t4 - _t3) * 1e3
            self._prof_count += 1
            if self._prof_count >= _PROFILE_INTERVAL:
                n = self._prof_count
                total = sum(self._prof_acc.values())
                player = getattr(self.cfg.backend, "player_number", "?")
                print(
                    f"[step_prof p={player} n={n}] "
                    f"backend={self._prof_acc['backend_ms']/n:.1f}ms "
                    f"reward={self._prof_acc['reward_ms']/n:.2f}ms "
                    f"encode={self._prof_acc['encode_ms']/n:.2f}ms "
                    f"other={self._prof_acc['other_ms']/n:.2f}ms "
                    f"step_total={total/n:.1f}ms "
                    f"step_rate={1e3*n/total:.1f}/s",
                    flush=True,
                )
                self._prof_acc = defaultdict(float)
                self._prof_count = 0

        rewards = {"player_1": float(evaluation.total_reward)}
        terminations = {"player_1": done}
        truncations = {"player_1": False}
        if done:
            info["episode_metrics"] = self._finalize_episode_metrics()
            self.agents = []
        return {"player_1": encoded}, rewards, terminations, truncations, {"player_1": info}

    def _policy_anchor_errors(self, policy_action: StructuredSkillAction, *, obs: BackendObservation) -> tuple[float, float]:
        if policy_action.skill_name not in {"walk", "dribble"}:
            return 0.0, 0.0
        anchors = build_fallback_action_targets(policy_action.skill_name, obs)
        target_error_mm = math.hypot(
            float(policy_action.target_x) - float(anchors["target_x"]),
            float(policy_action.target_y) - float(anchors["target_y"]),
        )
        theta_error_rad = abs(self._wrap_angle(float(policy_action.target_theta) - float(anchors["target_theta"])))
        return float(target_error_mm), float(theta_error_rad)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    def _decode_residual_policy_action(
        self,
        raw_action: StructuredSkillAction,
        *,
        obs: BackendObservation,
    ) -> StructuredSkillAction:
        residual_x = float(np.clip(raw_action.target_x, -1.0, 1.0))
        residual_y = float(np.clip(raw_action.target_y, -1.0, 1.0))
        residual_theta = float(np.clip(raw_action.target_theta, -1.0, 1.0))
        param_mask = action_param_mask(raw_action.skill_name)
        residual_x *= float(param_mask[0])
        residual_y *= float(param_mask[1])
        residual_theta *= float(param_mask[2])
        anchors = build_fallback_action_targets(raw_action.skill_name, obs)
        target_x = float(anchors["target_x"] + residual_x * ACTION.residual_x_range)
        target_y = float(anchors["target_y"] + residual_y * ACTION.residual_y_range)
        target_theta = self._wrap_angle(float(anchors["target_theta"] + residual_theta * ACTION.residual_theta_range))
        return StructuredSkillAction(
            skill_type=raw_action.skill_type,
            target_x=float(np.clip(target_x, -ACTION.x_range, ACTION.x_range)),
            target_y=float(np.clip(target_y, -ACTION.y_range, ACTION.y_range)),
            target_theta=target_theta,
            pass_target=int(raw_action.pass_target) if float(param_mask[3]) > 0.0 else 0,
        )

    def project_action(self, action: StructuredSkillAction, *, obs: BackendObservation) -> tuple[StructuredSkillAction, bool]:
        if action.skill_name in self._episode_stage_spec.active_skill_names:
            return action, False

        # Always project to walk (not preferred_skill) so telemetry reflects what
        # the policy actually learned. Using preferred_skill here caused dribble to
        # appear as "executed" on policy errors, masking that the policy never emitted it.
        fallback_targets = build_fallback_action_targets("walk", obs)
        projected = StructuredSkillAction(
            skill_type=SkillType.from_public_name("walk"),
            target_x=float(fallback_targets["target_x"]),
            target_y=float(fallback_targets["target_y"]),
            target_theta=float(fallback_targets["target_theta"]),
            pass_target=int(action.pass_target),
        )
        return projected, True

    def _repair_execution_targets(
        self,
        policy_action: StructuredSkillAction,
        *,
        obs: BackendObservation,
    ) -> tuple[StructuredSkillAction, bool, str]:
        if policy_action.skill_name not in {"walk", "dribble"}:
            return policy_action, False, ""

        fallback_targets = build_fallback_action_targets(policy_action.skill_name, obs)
        target_error_mm = math.hypot(
            float(policy_action.target_x) - float(fallback_targets["target_x"]),
            float(policy_action.target_y) - float(fallback_targets["target_y"]),
        )
        theta_error = abs(self._wrap_angle(float(policy_action.target_theta) - float(fallback_targets["target_theta"])))
        residual_radius_limit = math.hypot(ACTION.residual_x_range, ACTION.residual_y_range) * 1.05
        if target_error_mm <= residual_radius_limit and theta_error <= ACTION.residual_theta_range * 1.05:
            return policy_action, False, ""
        repaired = StructuredSkillAction(
            skill_type=policy_action.skill_type,
            target_x=float(fallback_targets["target_x"]),
            target_y=float(fallback_targets["target_y"]),
            target_theta=float(fallback_targets["target_theta"]),
            pass_target=policy_action.pass_target,
        )
        return repaired, True, f"{policy_action.skill_name}_heuristic_anchor"

    def _should_force_dribble_handoff(self, obs: BackendObservation) -> bool:
        if self._episode_stage_id != 0 or "dribble" not in self._episode_stage_spec.active_skill_names:
            return False
        if self._last_evaluation is None:
            return False
        stable_takeover = bool(self._last_evaluation.takeover_envelope and self._last_evaluation.prev_takeover_envelope)
        if not stable_takeover:
            return False
        rescued = bool(obs.synthetic_ball_observation or obs.corrected_exported_ball)
        trusted = bool((not rescued) and float(obs.natural_time_since_ball_seen) <= BALL_TRUSTED_MAX_AGE_MS)
        clean_export = bool(
            obs.ball_export_fresh
            and not obs.ball_observation_stale
            and not obs.ball_observation_collapsed
        )
        return bool(trusted and clean_export)

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
            "policy_skill_prob_sums": {name: 0.0 for name in ACTION.skill_names},
            "policy_skill_entropy_sum": 0.0,
            "policy_param_std_sum": 0.0,
            "skill_projection_count": 0,
            "handoff_override_count": 0,
            "target_repair_count": 0,
            "target_repair_reason_counts": defaultdict(int),
            "policy_target_error_sum": 0.0,
            "policy_theta_error_sum": 0.0,
            "policy_target_quality_count": 0,
            "walk_steps": 0,
            "dribble_steps": 0,
            "shoot_steps": 0,
            "shoot_distance_ready_steps": 0,
            "shoot_angle_ready_steps": 0,
            "shoot_fresh_ready_steps": 0,
            "shoot_open_ready_steps": 0,
            "shoot_geometry_ready_steps": 0,
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
            "stable_takeover_steps": 0,
            "incompatible_takeover_requests": 0,
            "walk_in_stable_takeover_steps": 0,
            "stand_in_stable_takeover_steps": 0,
            "policy_incompatible_takeover_requests": 0,
            "policy_walk_in_stable_takeover_steps": 0,
            "policy_stand_in_stable_takeover_steps": 0,
            "good_finish_steps": 0,
            "shoot_finish_steps": 0,
            "goal": False,
            "collision_count": 0,
            "thrash_count": 0,
            "clearance_positive_steps": 0,
            "obstacle_present_steps": 0,
            "corridor_keep_steps": 0,
            "dynamic_updates": 0,
            "post_takeover_progress_sum": 0.0,
            "post_takeover_finishing_steps": 0,
            "synthetic_pose_steps": 0,
            "synthetic_ball_steps": 0,
            "obs_mode_estimated_steps": 0,
            "obs_mode_corrected_steps": 0,
            "pose_alignment_warning_steps": 0,
            "pose_jump_steps": 0,
            "pose_jump_max_mm": 0.0,
            "ball_stale_steps": 0,
            "ball_collapsed_steps": 0,
            "corrected_exported_robot_pose_steps": 0,
            "corrected_exported_ball_steps": 0,
            "ball_export_fresh_steps": 0,
            "ball_export_source_estimated_steps": 0,
            "ball_export_source_reset_steps": 0,
            "ball_export_source_dynamic_steps": 0,
            "natural_ball_seen_steps": 0,
            "natural_ball_consistent_steps": 0,
            "ball_visible_steps": 0,
            "ball_trusted_steps": 0,
            "fov_steps": 0,           # steps ball is inside NAO camera FOV (by GT pos)
            "fov_not_seen_steps": 0,  # of those, BHuman natural_ball_seen_pct == 0
            "ball_semi_trusted_steps": 0,
            "ball_degraded_steps": 0,
            "ball_rescued_steps": 0,
            "ball_lost_steps": 0,
            "misleading_seen_synthetic_steps": 0,
            "ball_confidence_score_sum": 0.0,
            "first_ball_seen_after_reset_ms": None,
            "first_fresh_export_after_reset_ms": None,
            "first_non_synthetic_ball_after_reset_ms": None,
            "first_trusted_ball_after_reset_ms": None,
            "first_natural_ball_seen_after_reset_ms": None,
        }

    def _update_episode_tracker(
        self,
        step_ctx: StepContext,
        evaluation: StepEvaluation,
        projection_applied: bool,
        dynamic_update_applied: bool,
        *,
        target_repair_applied: bool,
        target_repair_reason: str,
        policy_target_error_mm: float,
        policy_theta_error_rad: float,
        policy_skill_entropy: float,
        policy_param_std_mean: float,
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
        tracker["policy_skill_prob_sums"]["walk"] += float(step_ctx.policy_skill_prob_walk)
        tracker["policy_skill_prob_sums"]["dribble"] += float(step_ctx.policy_skill_prob_dribble)
        tracker["policy_skill_prob_sums"]["shoot"] += float(step_ctx.policy_skill_prob_shoot)
        tracker["policy_skill_entropy_sum"] += float(policy_skill_entropy)
        tracker["policy_param_std_sum"] += float(policy_param_std_mean)
        if projection_applied:
            tracker["skill_projection_count"] += 1
        if step_ctx.handoff_override_applied:
            tracker["handoff_override_count"] += 1
        if target_repair_applied:
            tracker["target_repair_count"] += 1
            if target_repair_reason:
                tracker["target_repair_reason_counts"][target_repair_reason] += 1
        if step_ctx.policy_skill_name in {"walk", "dribble"}:
            tracker["policy_target_error_sum"] += float(policy_target_error_mm)
            tracker["policy_theta_error_sum"] += float(policy_theta_error_rad)
            tracker["policy_target_quality_count"] += 1
        if dynamic_update_applied:
            tracker["dynamic_updates"] += 1
        tracker["post_takeover_finishing_steps"] += int(bool(step_ctx.post_takeover_finishing_active))
        tracker["synthetic_pose_steps"] += int(bool(step_ctx.obs.synthetic_pose_observation))
        tracker["synthetic_ball_steps"] += int(bool(step_ctx.obs.synthetic_ball_observation))
        tracker["obs_mode_estimated_steps"] += int(int(step_ctx.obs.obs_export_mode) == 0)
        tracker["obs_mode_corrected_steps"] += int(int(step_ctx.obs.obs_export_mode) == 1)
        tracker["pose_alignment_warning_steps"] += int(bool(step_ctx.obs.pose_alignment_warning))
        if step_ctx.prev_obs is not None:
            pose_jump_mm = math.hypot(
                float(step_ctx.obs.robot_x) - float(step_ctx.prev_obs.robot_x),
                float(step_ctx.obs.robot_y) - float(step_ctx.prev_obs.robot_y),
            )
            tracker["pose_jump_max_mm"] = max(float(tracker["pose_jump_max_mm"]), float(pose_jump_mm))
            tracker["pose_jump_steps"] += int(pose_jump_mm > POSE_JUMP_WARNING_MM)
        tracker["ball_stale_steps"] += int(bool(step_ctx.obs.ball_observation_stale))
        tracker["ball_collapsed_steps"] += int(bool(step_ctx.obs.ball_observation_collapsed))
        tracker["corrected_exported_robot_pose_steps"] += int(bool(step_ctx.obs.corrected_exported_robot_pose))
        tracker["corrected_exported_ball_steps"] += int(bool(step_ctx.obs.corrected_exported_ball))
        tracker["ball_export_fresh_steps"] += int(bool(step_ctx.obs.ball_export_fresh))
        tracker["natural_ball_seen_steps"] += int(
            bool(step_ctx.obs.natural_ball_seen_percentage > 0.0 or step_ctx.obs.natural_time_since_ball_seen <= 1000.0)
        )
        tracker["natural_ball_consistent_steps"] += int(bool(step_ctx.obs.natural_ball_consistent_with_game_state))
        masks = _ball_observation_masks(step_ctx.obs)
        tracker["ball_visible_steps"] += int(bool(masks["ball_visible_mask"]))
        tracker["ball_trusted_steps"] += int(bool(masks["ball_trusted_mask"]))
        # FOV diagnostic: did BHuman detect the ball when it was inside the camera FOV?
        # ball_rel_x/y are in robot frame from the corrected (GT) obs — reliable even
        # when BHuman's own model is stale. NAO camera: ~±30° horiz, <2500mm range.
        _bx = float(step_ctx.obs.ball_rel_x)
        _by = float(step_ctx.obs.ball_rel_y)
        if (_bx > 100.0
                and math.hypot(_bx, _by) < 2500.0
                and abs(math.atan2(_by, _bx)) < math.radians(30)):
            tracker["fov_steps"] += 1
            if float(step_ctx.obs.natural_ball_seen_percentage) == 0.0:
                tracker["fov_not_seen_steps"] += 1
        tracker["ball_semi_trusted_steps"] += int(bool(masks["ball_semi_trusted_mask"]))
        tracker["ball_degraded_steps"] += int(bool(masks["ball_degraded_mask"]))
        tracker["ball_rescued_steps"] += int(bool(masks["ball_rescued_mask"]))
        tracker["ball_lost_steps"] += int(bool(masks["ball_lost_mask"]))
        tracker["misleading_seen_synthetic_steps"] += int(
            bool(float(step_ctx.obs.ball_seen_percentage) > 0.0 and bool(step_ctx.obs.synthetic_ball_observation))
        )
        tracker["ball_confidence_score_sum"] += float(masks["ball_confidence_score"])
        if int(step_ctx.obs.ball_export_source) == 0:
            tracker["ball_export_source_estimated_steps"] += 1
        elif int(step_ctx.obs.ball_export_source) == 1:
            tracker["ball_export_source_reset_steps"] += 1
        elif int(step_ctx.obs.ball_export_source) == 2:
            tracker["ball_export_source_dynamic_steps"] += 1

        reset_frame = int(step_ctx.obs.reset_applied_frame)
        delay_since_reset_ms = int(step_ctx.obs.frame - reset_frame) if reset_frame > 0 and step_ctx.obs.frame >= reset_frame else None
        if delay_since_reset_ms is not None:
            if tracker["first_ball_seen_after_reset_ms"] is None and (
                float(step_ctx.obs.ball_seen_percentage) > 0.0 or float(step_ctx.obs.time_since_ball_seen) <= 1000.0
            ):
                tracker["first_ball_seen_after_reset_ms"] = float(delay_since_reset_ms)
            if tracker["first_fresh_export_after_reset_ms"] is None and bool(step_ctx.obs.ball_export_fresh):
                tracker["first_fresh_export_after_reset_ms"] = float(delay_since_reset_ms)
            if tracker["first_non_synthetic_ball_after_reset_ms"] is None and not bool(step_ctx.obs.synthetic_ball_observation):
                tracker["first_non_synthetic_ball_after_reset_ms"] = float(delay_since_reset_ms)
            if tracker["first_trusted_ball_after_reset_ms"] is None and bool(masks["ball_trusted_mask"]):
                tracker["first_trusted_ball_after_reset_ms"] = float(delay_since_reset_ms)
            if tracker["first_natural_ball_seen_after_reset_ms"] is None and (
                float(step_ctx.obs.natural_ball_seen_percentage) > 0.0
                or float(step_ctx.obs.natural_time_since_ball_seen) <= 1000.0
            ):
                tracker["first_natural_ball_seen_after_reset_ms"] = float(delay_since_reset_ms)

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
        shoot_distance_ready = evaluation.features.d_ball_mm <= 380.0
        shoot_angle_ready = abs(evaluation.features.a_ball_rad) <= math.radians(25.0)
        shoot_fresh_ready = evaluation.features.t_seen_ms <= BALL_TRUSTED_MAX_AGE_MS
        shoot_open_ready = evaluation.features.shot_open >= 0.6 or evaluation.features.score_now
        tracker["shoot_distance_ready_steps"] += int(shoot_distance_ready)
        tracker["shoot_angle_ready_steps"] += int(shoot_angle_ready)
        tracker["shoot_fresh_ready_steps"] += int(shoot_fresh_ready)
        tracker["shoot_open_ready_steps"] += int(shoot_open_ready)
        tracker["shoot_geometry_ready_steps"] += int(shoot_distance_ready and shoot_angle_ready and shoot_fresh_ready)
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

        stable_takeover_envelope = evaluation.takeover_envelope and evaluation.prev_takeover_envelope
        if stable_takeover_envelope:
            tracker["stable_takeover_steps"] += 1
        if stable_takeover_envelope and step_ctx.skill_name not in {"dribble", "shoot"}:
            tracker["incompatible_takeover_requests"] += 1
            if step_ctx.skill_name == "walk":
                tracker["walk_in_stable_takeover_steps"] += 1
            elif step_ctx.skill_name == "stand":
                tracker["stand_in_stable_takeover_steps"] += 1
        if stable_takeover_envelope and step_ctx.policy_skill_name not in {"dribble", "shoot"}:
            tracker["policy_incompatible_takeover_requests"] += 1
            if step_ctx.policy_skill_name == "walk":
                tracker["policy_walk_in_stable_takeover_steps"] += 1
            elif step_ctx.policy_skill_name == "stand":
                tracker["policy_stand_in_stable_takeover_steps"] += 1

        if evaluation.terms["good_shoot"] > 0.0 or evaluation.terms["good_dribble"] > 0.0:
            tracker["good_finish_steps"] += 1
        if evaluation.terms["good_shoot"] > 0.0:
            tracker["shoot_finish_steps"] += 1
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
            "stable_takeover_steps": float(tracker["stable_takeover_steps"]),
            "incompatible_takeover_steps": float(tracker["incompatible_takeover_requests"]),
            "incompatible_takeover_rate": float(
                tracker["incompatible_takeover_requests"] / max(1, tracker["stable_takeover_steps"])
            ),
            "walk_in_stable_takeover_rate": float(
                tracker["walk_in_stable_takeover_steps"] / max(1, tracker["stable_takeover_steps"])
            ),
            "stand_in_stable_takeover_rate": float(
                tracker["stand_in_stable_takeover_steps"] / max(1, tracker["stable_takeover_steps"])
            ),
            "policy_incompatible_takeover_steps": float(tracker["policy_incompatible_takeover_requests"]),
            "policy_incompatible_takeover_rate": float(
                tracker["policy_incompatible_takeover_requests"] / max(1, tracker["stable_takeover_steps"])
            ),
            "policy_walk_in_stable_takeover_rate": float(
                tracker["policy_walk_in_stable_takeover_steps"] / max(1, tracker["stable_takeover_steps"])
            ),
            "policy_stand_in_stable_takeover_rate": float(
                tracker["policy_stand_in_stable_takeover_steps"] / max(1, tracker["stable_takeover_steps"])
            ),
            "target_repair_rate": float(tracker["target_repair_count"] / steps),
            "policy_target_error_mean": float(
                tracker["policy_target_error_sum"] / max(1, tracker["policy_target_quality_count"])
            ),
            "policy_theta_error_mean": float(
                tracker["policy_theta_error_sum"] / max(1, tracker["policy_target_quality_count"])
            ),
            "synthetic_pose_rate": float(tracker["synthetic_pose_steps"] / steps),
            "synthetic_ball_rate": float(tracker["synthetic_ball_steps"] / steps),
            "obs_mode_estimated_rate": float(tracker["obs_mode_estimated_steps"] / steps),
            "obs_mode_corrected_rate": float(tracker["obs_mode_corrected_steps"] / steps),
            "pose_alignment_warning_rate": float(tracker["pose_alignment_warning_steps"] / steps),
            "pose_jump_rate": float(tracker["pose_jump_steps"] / steps),
            "pose_jump_max_mm": float(tracker["pose_jump_max_mm"]),
            "ball_stale_rate": float(tracker["ball_stale_steps"] / steps),
            "ball_collapsed_rate": float(tracker["ball_collapsed_steps"] / steps),
            "corrected_exported_robot_pose_rate": float(tracker["corrected_exported_robot_pose_steps"] / steps),
            "corrected_exported_ball_rate": float(tracker["corrected_exported_ball_steps"] / steps),
            "ball_export_fresh_rate": float(tracker["ball_export_fresh_steps"] / steps),
            "ball_export_source_estimated_rate": float(tracker["ball_export_source_estimated_steps"] / steps),
            "ball_export_source_reset_rate": float(tracker["ball_export_source_reset_steps"] / steps),
            "ball_export_source_dynamic_rate": float(tracker["ball_export_source_dynamic_steps"] / steps),
            "natural_ball_seen_rate": float(tracker["natural_ball_seen_steps"] / steps),
            "natural_ball_consistent_rate": float(tracker["natural_ball_consistent_steps"] / steps),
            "ball_visible_rate": float(tracker["ball_visible_steps"] / steps),
            "ball_trusted_rate": float(tracker["ball_trusted_steps"] / steps),
            "ball_semi_trusted_rate": float(tracker["ball_semi_trusted_steps"] / steps),
            "ball_degraded_rate": float(tracker["ball_degraded_steps"] / steps),
            "ball_rescued_rate": float(tracker["ball_rescued_steps"] / steps),
            "ball_lost_rate": float(tracker["ball_lost_steps"] / steps),
            "misleading_seen_synthetic_rate": float(tracker["misleading_seen_synthetic_steps"] / steps),
            "fov_perception_failure_rate": float(
                tracker["fov_not_seen_steps"] / max(1, tracker["fov_steps"])
            ),
            "ball_confidence_score_mean": float(tracker["ball_confidence_score_sum"] / steps),
            "first_ball_seen_after_reset_ms": float(tracker["first_ball_seen_after_reset_ms"] or 0.0),
            "first_fresh_export_after_reset_ms": float(tracker["first_fresh_export_after_reset_ms"] or 0.0),
            "first_non_synthetic_ball_after_reset_ms": float(tracker["first_non_synthetic_ball_after_reset_ms"] or 0.0),
            "first_trusted_ball_after_reset_ms": float(tracker["first_trusted_ball_after_reset_ms"] or 0.0),
            "first_natural_ball_seen_after_reset_ms": float(tracker["first_natural_ball_seen_after_reset_ms"] or 0.0),
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
        stable_takeover_steps = max(1, tracker["stable_takeover_steps"])
        incompatible_takeover_rate = float(tracker["incompatible_takeover_requests"] / stable_takeover_steps)
        walk_in_stable_takeover_rate = float(tracker["walk_in_stable_takeover_steps"] / stable_takeover_steps)
        stand_in_stable_takeover_rate = float(tracker["stand_in_stable_takeover_steps"] / stable_takeover_steps)
        policy_incompatible_takeover_rate = float(
            tracker["policy_incompatible_takeover_requests"] / stable_takeover_steps
        )
        policy_walk_in_stable_takeover_rate = float(
            tracker["policy_walk_in_stable_takeover_steps"] / stable_takeover_steps
        )
        policy_stand_in_stable_takeover_rate = float(
            tracker["policy_stand_in_stable_takeover_steps"] / stable_takeover_steps
        )

        clean_walk = bool(walk_steps > 0 and walk_compat_rate >= 0.7 and tracker["thrash_count"] == 0)
        if self._episode_stage_id < 0:
            clean_takeover = bool(takeover_success and tracker["thrash_count"] == 0 and tracker["collision_count"] == 0)
        else:
            # Tolerate a small incompatibility fraction instead of requiring
            # exactly zero: one walk/stand step in the stable envelope used to
            # zero the whole episode, masking otherwise-clean handoffs and
            # leaving no gradient signal for partial success.
            clean_takeover = bool(
                takeover_success
                and incompatible_takeover_rate <= 0.15
                and policy_incompatible_takeover_rate <= 0.15
            )
        # thrash_rate-based gate (was thrash_count <= 1, which is unreachable at a
        # normal thrash_rate of ~0.07 ≈ 17 thrash steps/episode and made clean_finish
        # structurally 0 even when the dribble genuinely advanced the ball past the
        # 0.2 progress bar). Mirrors the de-brittled clean_takeover and the stage-1
        # _behavior_is_stable threshold (thrash_max=0.10).
        clean_finish = bool(
            tracker["good_finish_steps"] > 0
            and (tracker["goal"] or tracker["post_takeover_progress_sum"] > 0.2)
            and thrash_rate <= 0.10
        )
        # Shoot-aware finish for stage 1+: clean_finish can be satisfied by
        # dribble/progress only, which is useful but overstates shooting sanity.
        shoot_finish = bool(
            tracker["shoot_finish_steps"] > 0
            and (tracker["goal"] or tracker["post_takeover_progress_sum"] > 0.2)
            and thrash_rate <= 0.10
        )
        clean_episode = bool(clean_takeover and clean_finish and tracker["collision_count"] == 0)

        obstacle_clearance_success = 0.0
        if self._episode_stage_id >= 2 and tracker["obstacle_present_steps"] > 0:
            obstacle_clearance_success = float(clean_takeover and tracker["collision_count"] == 0 and tracker["clearance_positive_steps"] > 0)

        corridor_keep_rate = 0.0
        if self._episode_stage_id >= 3 and tracker["obstacle_present_steps"] > 0:
            corridor_keep_rate = float(tracker["corridor_keep_steps"] / tracker["obstacle_present_steps"])

        dynamic_obstacle_success = 0.0
        if self._episode_stage_id >= 3 and tracker["dynamic_updates"] > 0:
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
            "stage_id": float(self._episode_stage_id),
            "steps": float(self._episode_step),
            "episode_reward_mean": float(tracker["reward_total"] / steps),
            "episode_reward_total": float(tracker["reward_total"]),
            "clean_walk": float(clean_walk),
            "clean_takeover": float(clean_takeover),
            "clean_finish": float(clean_finish),
            "shoot_finish": float(shoot_finish),
            "clean_episode": float(clean_episode),
            "collision_episode": collision_episode,
            "takeover_success": float(takeover_success),
            # Disambiguates policy_incompatible_takeover_rate==0: with the max(1,..)
            # denominator, "no takeover at all" reads identically to "clean takeover".
            # Condition incompat metrics on this to tell failure (never reached the
            # ball) apart from success (dribbled cleanly).
            "takeover_occurred": float(tracker["stable_takeover_steps"] > 0),
            "goal_rate": float(tracker["goal"]),
            "thrash_rate": thrash_rate,
            "mean_time_to_takeover": mean_time_to_takeover,
            "stable_takeover_steps": float(tracker["stable_takeover_steps"]),
            "incompatible_takeover_steps": float(tracker["incompatible_takeover_requests"]),
            "incompatible_takeover_rate": incompatible_takeover_rate,
            "walk_in_stable_takeover_rate": walk_in_stable_takeover_rate,
            "stand_in_stable_takeover_rate": stand_in_stable_takeover_rate,
            "policy_incompatible_takeover_steps": float(tracker["policy_incompatible_takeover_requests"]),
            "policy_incompatible_takeover_rate": policy_incompatible_takeover_rate,
            "policy_walk_in_stable_takeover_rate": policy_walk_in_stable_takeover_rate,
            "policy_stand_in_stable_takeover_rate": policy_stand_in_stable_takeover_rate,
            "walk_compat_rate": walk_compat_rate,
            "dribble_compat_rate": dribble_compat_rate,
            "shoot_compat_rate": shoot_compat_rate,
            "shoot_finish_steps": float(tracker["shoot_finish_steps"]),
            "shoot_zone_rate": float(tracker["shoot_zone_steps"] / steps),
            "shoot_distance_ready_rate": float(tracker["shoot_distance_ready_steps"] / steps),
            "shoot_angle_ready_rate": float(tracker["shoot_angle_ready_steps"] / steps),
            "shoot_fresh_ready_rate": float(tracker["shoot_fresh_ready_steps"] / steps),
            "shoot_open_ready_rate": float(tracker["shoot_open_ready_steps"] / steps),
            "shoot_geometry_ready_rate": float(tracker["shoot_geometry_ready_steps"] / steps),
            "bad_walk_rate": bad_walk_rate,
            "bad_dribble_rate": bad_dribble_rate,
            "bad_shoot_rate": bad_shoot_rate,
            "policy_stand_rate": float(tracker["policy_skill_counts"].get("stand", 0) / steps),
            "policy_walk_rate": float(tracker["policy_skill_counts"].get("walk", 0) / steps),
            "policy_dribble_rate": float(tracker["policy_skill_counts"].get("dribble", 0) / steps),
            "policy_shoot_rate": float(tracker["policy_skill_counts"].get("shoot", 0) / steps),
            "executed_stand_rate": float(tracker["skill_counts"].get("stand", 0) / steps),
            "executed_walk_rate": float(tracker["skill_counts"].get("walk", 0) / steps),
            "executed_dribble_rate": float(tracker["skill_counts"].get("dribble", 0) / steps),
            "executed_shoot_rate": float(tracker["skill_counts"].get("shoot", 0) / steps),
            "policy_skill_prob_walk_mean": float(tracker["policy_skill_prob_sums"].get("walk", 0.0) / steps),
            "policy_skill_prob_dribble_mean": float(tracker["policy_skill_prob_sums"].get("dribble", 0.0) / steps),
            "policy_skill_prob_shoot_mean": float(tracker["policy_skill_prob_sums"].get("shoot", 0.0) / steps),
            "policy_skill_entropy": float(tracker["policy_skill_entropy_sum"] / steps),
            "policy_param_std_mean": float(tracker["policy_param_std_sum"] / steps),
            "obstacle_clearance_success": obstacle_clearance_success,
            "dynamic_obstacle_success": dynamic_obstacle_success,
            "corridor_keep_rate": corridor_keep_rate,
            "ball_progress_post_takeover_mean": float(tracker["post_takeover_progress_sum"] / steps),
            "post_takeover_finishing_rate": float(tracker["post_takeover_finishing_steps"] / steps),
            "stability_score": stability_score,
            "skill_projection_rate": float(tracker["skill_projection_count"] / steps),
            "handoff_override_rate": float(tracker["handoff_override_count"] / steps),
            "target_repair_rate": float(tracker["target_repair_count"] / steps),
            "policy_target_error_mean": float(
                tracker["policy_target_error_sum"] / max(1, tracker["policy_target_quality_count"])
            ),
            "policy_theta_error_mean": float(
                tracker["policy_theta_error_sum"] / max(1, tracker["policy_target_quality_count"])
            ),
            "synthetic_pose_rate": float(tracker["synthetic_pose_steps"] / steps),
            "synthetic_ball_rate": float(tracker["synthetic_ball_steps"] / steps),
            "obs_mode_estimated_rate": float(tracker["obs_mode_estimated_steps"] / steps),
            "obs_mode_corrected_rate": float(tracker["obs_mode_corrected_steps"] / steps),
            "pose_alignment_warning_rate": float(tracker["pose_alignment_warning_steps"] / steps),
            "pose_jump_rate": float(tracker["pose_jump_steps"] / steps),
            "pose_jump_max_mm": float(tracker["pose_jump_max_mm"]),
            "ball_stale_rate": float(tracker["ball_stale_steps"] / steps),
            "ball_collapsed_rate": float(tracker["ball_collapsed_steps"] / steps),
            "corrected_exported_robot_pose_rate": float(tracker["corrected_exported_robot_pose_steps"] / steps),
            "corrected_exported_ball_rate": float(tracker["corrected_exported_ball_steps"] / steps),
            "ball_export_fresh_rate": float(tracker["ball_export_fresh_steps"] / steps),
            "ball_export_source_estimated_rate": float(tracker["ball_export_source_estimated_steps"] / steps),
            "ball_export_source_reset_rate": float(tracker["ball_export_source_reset_steps"] / steps),
            "ball_export_source_dynamic_rate": float(tracker["ball_export_source_dynamic_steps"] / steps),
            "natural_ball_seen_rate": float(tracker["natural_ball_seen_steps"] / steps),
            "natural_ball_consistent_rate": float(tracker["natural_ball_consistent_steps"] / steps),
            "ball_visible_rate": float(tracker["ball_visible_steps"] / steps),
            "ball_trusted_rate": float(tracker["ball_trusted_steps"] / steps),
            "ball_semi_trusted_rate": float(tracker["ball_semi_trusted_steps"] / steps),
            "ball_degraded_rate": float(tracker["ball_degraded_steps"] / steps),
            "ball_rescued_rate": float(tracker["ball_rescued_steps"] / steps),
            "ball_lost_rate": float(tracker["ball_lost_steps"] / steps),
            "misleading_seen_synthetic_rate": float(tracker["misleading_seen_synthetic_steps"] / steps),
            "fov_perception_failure_rate": float(
                tracker["fov_not_seen_steps"] / max(1, tracker["fov_steps"])
            ),
            "ball_confidence_score_mean": float(tracker["ball_confidence_score_sum"] / steps),
            "first_ball_seen_after_reset_ms": float(tracker["first_ball_seen_after_reset_ms"] or 0.0),
            "first_fresh_export_after_reset_ms": float(tracker["first_fresh_export_after_reset_ms"] or 0.0),
            "first_non_synthetic_ball_after_reset_ms": float(tracker["first_non_synthetic_ball_after_reset_ms"] or 0.0),
            "first_trusted_ball_after_reset_ms": float(tracker["first_trusted_ball_after_reset_ms"] or 0.0),
            "first_natural_ball_seen_after_reset_ms": float(tracker["first_natural_ball_seen_after_reset_ms"] or 0.0),
            "reward_terms_sum": dict(tracker["reward_terms_sum"]),
            "reward_weighted_terms_sum": dict(tracker["reward_weighted_terms_sum"]),
            "skill_counts": dict(tracker["skill_counts"]),
            "policy_skill_counts": dict(tracker["policy_skill_counts"]),
            "target_repair_reason_counts": dict(tracker["target_repair_reason_counts"]),
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
        policy_skill_raw: float = 0.0,
        policy_skill_round: int = 0,
        policy_compact_skill_index: int = 0,
        policy_skill_probs: Mapping[str, float] | None = None,
        policy_skill_entropy: float = 0.0,
        policy_param_std_mean: float = 0.0,
        policy_residual_x: float = 0.0,
        policy_residual_y: float = 0.0,
        policy_residual_theta: float = 0.0,
        policy_target_error_mm: float = 0.0,
        policy_theta_error_rad: float = 0.0,
        handoff_override_applied: bool = False,
        target_repair_applied: bool = False,
        target_repair_reason: str = "",
        done: bool = False,
    ) -> Dict[str, object]:
        raw_obs = asdict(obs)
        ball_masks = _ball_observation_masks(obs)
        policy_skill_probs = policy_skill_probs or {name: 0.0 for name in SKILL_ORDER}
        policy_dribble_margin = (
            float(policy_skill_probs.get("dribble", 0.0) - policy_skill_probs.get("walk", 0.0))
            if sum(float(value) for value in policy_skill_probs.values()) > 0.0
            else (1.0 if policy_action is not None and policy_action.skill_name == "dribble" else -1.0 if policy_action is not None and policy_action.skill_name == "walk" else 0.0)
        )
        info: Dict[str, object] = {
            "state": self.get_global_state(),
            "obs_feature_names": OBS_FEATURE_NAMES,
            "backend_mode": self.cfg.backend.mode,
            "configured_obs_mode": self.cfg.backend.obs_mode,
            "stage_id": self._episode_stage_id,
            "stage_name": self._episode_stage_spec.name,
            "active_skill_mask": skill_mask(self._episode_stage_spec.active_skill_names),
            "stage_skill_mask": stage_skill_mask(self._episode_stage_id),
            "bridge_obs_ready": obs.obs_ready,
            "bridge_frame": obs.frame,
            "obs_export_mode": int(obs.obs_export_mode),
            "corrected_exported_robot_pose": bool(obs.corrected_exported_robot_pose),
            "corrected_exported_ball": bool(obs.corrected_exported_ball),
            "ball_export_source": int(obs.ball_export_source),
            "ball_export_fresh": bool(obs.ball_export_fresh),
            "natural_time_since_ball_seen": float(obs.natural_time_since_ball_seen),
            "natural_ball_seen_percentage": float(obs.natural_ball_seen_percentage),
            "natural_ball_consistent_with_game_state": bool(obs.natural_ball_consistent_with_game_state),
            "reset_applied_frame": int(obs.reset_applied_frame),
            "dynamic_applied_frame": int(obs.dynamic_applied_frame),
            "reset_applied_age_ms": int(obs.frame - obs.reset_applied_frame) if obs.reset_applied_frame > 0 and obs.frame >= obs.reset_applied_frame else -1,
            "dynamic_applied_age_ms": int(obs.frame - obs.dynamic_applied_frame) if obs.dynamic_applied_frame > 0 and obs.frame >= obs.dynamic_applied_frame else -1,
            "synthetic_pose_observation": bool(obs.synthetic_pose_observation),
            "synthetic_ball_observation": bool(obs.synthetic_ball_observation),
            "synthetic_pose_reason": obs.synthetic_pose_reason,
            "synthetic_ball_reason": obs.synthetic_ball_reason,
            "pose_alignment_error_mm": float(obs.pose_alignment_error_mm),
            "pose_alignment_warning": bool(obs.pose_alignment_warning),
            "ball_observation_stale": bool(obs.ball_observation_stale),
            "ball_observation_collapsed": bool(obs.ball_observation_collapsed),
            "estimated_robot_x": float(obs.estimated_robot_x),
            "estimated_robot_y": float(obs.estimated_robot_y),
            "estimated_robot_theta": float(obs.estimated_robot_theta),
            "estimated_ball_x": float(obs.estimated_ball_x),
            "estimated_ball_y": float(obs.estimated_ball_y),
            "raw_obs": raw_obs,
            "policy_skill_name": None if policy_action is None else policy_action.skill_name,
            "executed_skill_name": None if executed_action is None else executed_action.skill_name,
            "policy_skill_raw": float(policy_skill_raw),
            "policy_skill_round": int(policy_skill_round),
            "policy_skill_index": int(policy_skill_round),
            "policy_compact_skill_index": int(policy_compact_skill_index),
            "policy_skill_dribble_margin": float(policy_dribble_margin),
            "policy_skill_entropy": float(policy_skill_entropy),
            "policy_param_std_mean": float(policy_param_std_mean),
            "policy_residual_x": float(policy_residual_x),
            "policy_residual_y": float(policy_residual_y),
            "policy_residual_theta": float(policy_residual_theta),
            "policy_target_error_mm": float(policy_target_error_mm),
            "policy_theta_error_rad": float(policy_theta_error_rad),
            "handoff_override_enabled": bool(self.cfg.enable_handoff_override),
            "handoff_override_applied": bool(handoff_override_applied),
            "target_repair_applied": bool(target_repair_applied),
            "target_repair_reason": str(target_repair_reason),
            "policy_target_x": 0.0 if policy_action is None else float(policy_action.target_x),
            "policy_target_y": 0.0 if policy_action is None else float(policy_action.target_y),
            "policy_target_theta": 0.0 if policy_action is None else float(policy_action.target_theta),
            "executed_target_x": 0.0 if executed_action is None else float(executed_action.target_x),
            "executed_target_y": 0.0 if executed_action is None else float(executed_action.target_y),
            "executed_target_theta": 0.0 if executed_action is None else float(executed_action.target_theta),
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
            **ball_masks,
        }
        for skill_name in SKILL_ORDER:
            info[f"policy_skill_prob_{skill_name}"] = float(policy_skill_probs.get(skill_name, 0.0))
        if evaluation is not None:
            info.update(evaluation.as_dict())
        if done:
            info["episode_done"] = True
        return info
