"""Production BC dataset collection over the deployment-facing situations.

The legacy `teacher_dataset.py` collects along the training curriculum (drive resets +
near-goal shoot top-ups). Behavioral evaluation showed the resulting checkpoint is
walk-dominant exactly where it must finish: in the 2.5-3.0 m band it almost never armed
shoot, and on `dribble_armed` states it drifted to walk. The old dataset simply lacked
production coverage (no 2.5-3.0 m finishes, no bad-angle contrast, thin obstacle pressure).

This collector reuses the SAME scenarios the evaluator grades on (production_curriculum), and
drives the gate-wired teacher through them, so every BC sample is gate-consistent (the recorded
obs's armed flags exactly permit the recorded label — Design A) and lands in the states the
policy is judged on:

  * finish_band_2p5_3m_clean / _static_lane   -> walk -> dribble (drive to <=2.5 m) -> shoot
  * preshoot_bad_angle_boundary               -> in-band but mis-aligned; teacher must NOT shoot
                                                 (gate keeps shoot disarmed) and dribbles to fix it
  * dynamic_full_field_pressure / blocked_corridor_repair -> dynamic/blocking obstacle chains

Output mirrors the old BC format exactly (samples.npz with observations[N,26], actions[N,5],
stages, episode_ids, step_ids, weights, skill_order; steps.csv; episodes.csv; summary.json), so
train_bc.py / train.py BC-replay consume it unchanged.

Run (interactive conda env; `conda run` corrupts SimRobot):

  source ~/anaconda3/etc/profile.d/conda.sh && conda activate RLNAO \
    && cd $RL_REPO_DIR && PYTHONPATH=$RL_REPO_DIR \
       python teacher_production_dataset.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

from rlfootball.core.action import SKILL_ORDER
from rlfootball.core.config import ENV, ROOT_DIR
from rlfootball.curriculum.curriculum import get_stage_scene_path, get_stage_spec
from rlfootball.env.environment import RealSimRobotEnv
from rlfootball.curriculum.production_curriculum import (
    PRODUCTION_BAND_MM,
    ProductionScenario,
    production_scenarios,
    production_topup_scenarios,
)
from rlfootball.core.reward import goal_ball_distance_mm
from rlfootball.datasets.teacher_dataset import _append_csv_row, _ball_distance, _distance_bin
from rlfootball.teachers.teacher_policy import obs_from_raw_mapping, teacher_policy_payload

# --- Collection config (env-overridable for reproducible overnight runs) ---------------
BACKEND_MODE = os.getenv("RL_PROD_TEACHER_BACKEND_MODE", "simrobot_visible")
SIMROBOT_PROFILE = os.getenv("RL_PROD_TEACHER_SIMROBOT_PROFILE", "3d_visible")
SIMROBOT_LAUNCH = os.getenv("RL_PROD_TEACHER_SIMROBOT_LAUNCH", "1") == "1"
# The env is built ONCE at the full-field stage (scene with obstacle slots); each scenario
# re-specs its own stage via reset options={"stage_id": ...} exactly like behavioral_evaluator.
# All production stages (1/2/4) share active skills (stand/walk/dribble/shoot), so the fixed
# obs/action spaces are valid for every scenario.
ENV_STAGE_ID = int(os.getenv("RL_PROD_TEACHER_ENV_STAGE_ID", "4"))
EPISODES_PER_SCENARIO = int(os.getenv("RL_PROD_TEACHER_EPISODES", "4"))
MAX_STEPS_PER_EPISODE = int(os.getenv("RL_PROD_TEACHER_MAX_STEPS", "600"))
MAX_ATTEMPT_MULTIPLIER = int(os.getenv("RL_PROD_TEACHER_MAX_ATTEMPT_MULTIPLIER", "3"))
SEED_BASE = int(os.getenv("RL_PROD_TEACHER_SEED_BASE", "770000"))
OUTPUT_ROOT = Path(os.getenv("RL_PROD_TEACHER_OUTPUT_ROOT", str(ROOT_DIR / "runs" / "teacher_production_dataset")))
MIN_SAMPLE_WEIGHT_CONFIDENCE = 0.35
SCENARIO_SET = os.getenv("RL_PROD_TEACHER_SCENARIO_SET", "eval").strip().lower()
SCENARIO_FILTER = tuple(
    name.strip() for name in os.getenv("RL_PROD_TEACHER_SCENARIOS", "").split(",") if name.strip()
)

# Per-episode success filter (see _episode_kept). Default clean_finish to match the stage gate:
# a clean walk->dribble->shoot chain is success; conversion is a kick refinement. Pure-walk
# stalls are always dropped; bad-angle contrast episodes are kept on engagement (the value is
# the "reposition instead of shoot" labels, not a goal).
SUCCESS_FILTER_ENABLED = os.getenv("RL_PROD_TEACHER_SUCCESS_FILTER", "1") == "1"
SUCCESS_FILTER_METRIC = os.getenv("RL_PROD_TEACHER_SUCCESS_METRIC", "clean_finish")

BAND_MIN_MM, BAND_MAX_MM = PRODUCTION_BAND_MM


def _make_env() -> RealSimRobotEnv:
    stage = get_stage_spec(ENV_STAGE_ID)
    backend_cfg = replace(
        ENV.backend,
        mode=BACKEND_MODE,
        simrobot_profile=SIMROBOT_PROFILE,
        simrobot_scene_path=get_stage_scene_path(ENV_STAGE_ID, SIMROBOT_PROFILE),
        simrobot_launch=SIMROBOT_LAUNCH,
    )
    env_cfg = replace(
        ENV,
        stage_id=ENV_STAGE_ID,
        max_episode_steps=MAX_STEPS_PER_EPISODE,
        backend=backend_cfg,
        active_skill_names=stage.active_skill_names,
        enable_handoff_override=False,
    )
    return RealSimRobotEnv(env_cfg=env_cfg)


def _sample_quality_weight(info: Mapping[str, object]) -> float:
    """Same down-weighting policy as teacher_dataset: drop poisoned frames (warned pose,
    collapsed ball, synthetic pose under visible), down-weight low-confidence / synthetic."""
    if bool(info.get("pose_alignment_warning", False)):
        return 0.0
    if bool(info.get("ball_observation_collapsed", False)):
        return 0.0
    if BACKEND_MODE == "simrobot_visible" and bool(info.get("synthetic_pose_observation", False)):
        return 0.0
    confidence = float(info.get("ball_confidence_score", 0.0))
    if confidence < MIN_SAMPLE_WEIGHT_CONFIDENCE:
        return 0.25
    if bool(info.get("synthetic_ball_observation", False)) or bool(info.get("corrected_exported_ball", False)):
        return 0.5
    if bool(info.get("synthetic_pose_observation", False)):
        return 0.75
    return 1.0


def _gate_bucket(gate) -> str:
    if gate is None:
        return "no_gate"
    if bool(getattr(gate, "shoot_armed", False)):
        return "shoot_armed"
    if bool(getattr(gate, "dribble_armed", False)):
        return "dribble_armed"
    return "none_armed"


def _gate_label_consistent(bucket: str, skill_name: str) -> bool:
    if bucket == "shoot_armed":
        return skill_name == "shoot"
    if bucket == "dribble_armed":
        return skill_name == "dribble"
    if bucket == "none_armed":
        return skill_name == "walk"
    return False


def _episode_kept(
    scenario: ProductionScenario,
    episode_metrics: Mapping[str, float],
    episode_skill_counts: Counter[str],
) -> tuple[bool, str]:
    """Keep production chains, drop pure-walk stalls; keep bad-angle contrast on engagement."""
    if not SUCCESS_FILTER_ENABLED:
        return True, "filter_off"
    dribble = int(episode_skill_counts.get("dribble", 0))
    shoot = int(episode_skill_counts.get("shoot", 0))
    if dribble == 0 and shoot == 0:
        # Never started the chain: no takeover, no shot -> a walk stall, not a demo.
        return False, "pure_walk_stall"
    if scenario.is_contrast:
        # The value of the bad-angle scenario is the "do not shoot yet, reposition" labels;
        # the teacher engaged (dribbled/repositioned), so keep it without requiring a goal.
        return True, "contrast_engaged"
    if shoot == 0:
        # Production BC must not reinforce no-shot "finishes" that only satisfy
        # the progress-based clean_finish metric.
        return False, "no_shoot"
    goal = float(episode_metrics.get("goal_rate", episode_metrics.get("goal", 0.0))) > 0.0
    clean_finish = float(episode_metrics.get("clean_finish", 0.0)) > 0.0
    if SUCCESS_FILTER_METRIC == "clean_finish":
        return clean_finish, "clean_finish" if clean_finish else "no_clean_finish"
    if SUCCESS_FILTER_METRIC == "either":
        return (goal or clean_finish), "either" if (goal or clean_finish) else "no_goal_no_clean_finish"
    return goal, "goal" if goal else "no_goal"


def _collect_scenario(
    env: RealSimRobotEnv,
    scenario: ProductionScenario,
    output_dir: Path,
    seed_base: int,
    episode_offset: int,
) -> dict[str, object]:
    steps_csv = output_dir / "steps.csv"
    episodes_csv = output_dir / "episodes.csv"
    stage_id = scenario.stage_id
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    stages: list[int] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    weights: list[float] = []
    label_consistent: list[int] = []  # parallel to weights; 1 if gate bucket permits the label
    skill_counts: Counter[str] = Counter()
    distance_bins: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    gate_bucket_counts: Counter[str] = Counter()
    # cross-tab: how the teacher labelled within each gate state (validates gate<->label wiring)
    gate_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    drop_reasons: Counter[str] = Counter()
    band_steps = 0  # committed samples with ball-to-goal in [2.5, 3.0] m
    band_steps_attempted = 0  # all stepped frames in band (before success filter)

    rng = np.random.default_rng(seed_base)
    episodes = scenario.episodes
    max_attempts = max(episodes, episodes * max(1, MAX_ATTEMPT_MULTIPLIER))
    kept_episodes = 0
    dropped_episodes = 0
    attempted_episodes = 0
    started = time.perf_counter()
    agent_id = "player_1"

    for attempt in range(max_attempts):
        if kept_episodes >= episodes:
            break
        attempted_episodes += 1
        seed = seed_base + attempt
        world_spec = scenario.world_fn(rng)
        obs_dict, infos = env.reset(seed=seed, options={"stage_id": stage_id, "world_spec": world_spec})
        prev_skill_name: str | None = None
        total_reward = 0.0
        episode_skill_counts: Counter[str] = Counter()
        final_info: Mapping[str, object] = {}
        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        ep_step_ids: list[int] = []
        ep_weights: list[float] = []
        ep_label_consistent: list[int] = []
        ep_distance_bins: Counter[str] = Counter()
        ep_quality: Counter[str] = Counter()
        ep_gate_buckets: Counter[str] = Counter()
        ep_gate_labels: dict[str, Counter[str]] = defaultdict(Counter)
        ep_band_steps = 0
        ep_step_rows: list[Mapping[str, object]] = []

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            info = infos.get(agent_id, {})
            raw_obs = info.get("raw_obs", {})
            if not isinstance(raw_obs, Mapping):
                raise RuntimeError("production teacher dataset requires raw_obs in env info")
            backend_obs = obs_from_raw_mapping(raw_obs)
            gate = env.last_gate_decision  # the decision baked into current_obs; teacher acts on it
            payload, absolute_action, residual_action = teacher_policy_payload(
                backend_obs,
                stage_id,
                prev_skill_name,
                gate_decision=gate,
            )
            current_obs = np.asarray(obs_dict[agent_id], dtype=np.float32).reshape(-1)
            current_action = np.asarray(
                [
                    float(residual_action.skill_type),
                    float(residual_action.target_x),
                    float(residual_action.target_y),
                    float(residual_action.target_theta),
                    float(residual_action.pass_target),
                ],
                dtype=np.float32,
            )
            goal_dist = goal_ball_distance_mm(backend_obs)
            in_band = BAND_MIN_MM <= goal_dist <= BAND_MAX_MM

            obs_dict, rewards, terminateds, truncateds, infos = env.step({agent_id: payload})
            step_info = infos.get(agent_id, {})
            reward = float(rewards.get(agent_id, 0.0))
            total_reward += reward
            final_info = step_info
            weight = _sample_quality_weight(step_info)
            ball_distance = _ball_distance(raw_obs)
            distance_bin = _distance_bin(ball_distance)
            skill_name = residual_action.skill_name
            bucket = _gate_bucket(gate)
            gate_label_consistent = _gate_label_consistent(bucket, skill_name)
            if not gate_label_consistent:
                # Production BC must be pure Design A: labels are a function of
                # the baked gate bits. Keep the diagnostic step row, but make
                # the sample invisible to BC/BC-replay.
                weight = 0.0

            ep_obs.append(current_obs)
            ep_actions.append(current_action)
            ep_step_ids.append(step)
            ep_weights.append(weight)
            ep_label_consistent.append(int(gate_label_consistent))
            episode_skill_counts[skill_name] += 1
            ep_distance_bins[distance_bin] += 1
            ep_quality["accepted_weighted"] += int(weight > 0.0)
            ep_quality["full_weight"] += int(weight >= 1.0)
            ep_quality["low_weight"] += int(0.0 < weight < 1.0)
            ep_quality["rejected"] += int(weight <= 0.0)
            ep_quality["gate_label_inconsistent"] += int(not gate_label_consistent)
            ep_gate_buckets[bucket] += 1
            ep_gate_labels[bucket][skill_name] += 1
            band_steps_attempted += int(in_band)
            ep_band_steps += int(in_band)

            ep_step_rows.append(
                {
                    "scenario": scenario.name,
                    "stage_id": stage_id,
                    "seed": seed,
                    "is_contrast": int(scenario.is_contrast),
                    "obstacles": scenario.obstacles,
                    "step": step,
                    "reward": reward,
                    "sample_weight": weight,
                    "teacher_skill": skill_name,
                    "label_skill_type": int(residual_action.skill_type),
                    "label_residual_x": float(residual_action.target_x),
                    "label_residual_y": float(residual_action.target_y),
                    "label_residual_theta": float(residual_action.target_theta),
                    "teacher_abs_target_x": float(absolute_action.target_x),
                    "teacher_abs_target_y": float(absolute_action.target_y),
                    "teacher_abs_target_theta": float(absolute_action.target_theta),
                    "executed_skill_name": step_info.get("executed_skill_name"),
                    "preferred_skill_name": step_info.get("preferred_skill_name"),
                    "gate_bucket": bucket,
                    "gate_shoot_armed": int(bool(getattr(gate, "shoot_armed", False))) if gate else 0,
                    "gate_dribble_armed": int(bool(getattr(gate, "dribble_armed", False))) if gate else 0,
                    "gate_shoot_arm_progress": float(getattr(gate, "shoot_arm_progress", 0.0)) if gate else 0.0,
                    "gate_label_consistent": int(gate_label_consistent),
                    "ball_distance_mm": ball_distance,
                    "distance_bin": distance_bin,
                    "goal_distance_mm": goal_dist,
                    "in_band_2p5_3m": int(in_band),
                    "takeover_envelope": int(bool(step_info.get("takeover_envelope", False))),
                    "shoot_zone": int(bool(step_info.get("shoot_zone", False))),
                    "ball_confidence_score": float(step_info.get("ball_confidence_score", 0.0)),
                    "ball_trusted_mask": int(bool(step_info.get("ball_trusted_mask", False))),
                    "ball_rescued_mask": int(bool(step_info.get("ball_rescued_mask", False))),
                    "can_score_now": int(bool(raw_obs.get("can_score_now", False))),
                    "shot_quality_no_obstacles": float(raw_obs.get("shot_quality_no_obstacles", 0.0)),
                    "shot_opening_with_obstacles": float(raw_obs.get("shot_opening_with_obstacles", 0.0)),
                    "nearest_opponent_front_dist": float(raw_obs.get("nearest_opponent_front_dist", 0.0)),
                    "nearest_uncertain_front_dist": float(raw_obs.get("nearest_uncertain_front_dist", 0.0)),
                    "synthetic_pose_observation": int(bool(step_info.get("synthetic_pose_observation", False))),
                    "pose_alignment_warning": int(bool(step_info.get("pose_alignment_warning", False))),
                    "ball_observation_collapsed": int(bool(step_info.get("ball_observation_collapsed", False))),
                },
            )
            prev_skill_name = skill_name
            if not env.agents or any(terminateds.values()) or any(truncateds.values()):
                break

        episode_metrics = dict(final_info.get("episode_metrics", {})) if final_info else {}
        kept, reason = _episode_kept(scenario, episode_metrics, episode_skill_counts)
        global_episode_id = episode_offset + kept_episodes
        _append_csv_row(
            episodes_csv,
            {
                "scenario": scenario.name,
                "stage_id": stage_id,
                "is_contrast": int(scenario.is_contrast),
                "episode": global_episode_id if kept else -1,
                "attempt": attempt,
                "seed": seed,
                "kept": int(kept),
                "drop_reason": "" if kept else reason,
                "steps": sum(episode_skill_counts.values()),
                "total_reward": total_reward,
                "goal": float(episode_metrics.get("goal_rate", episode_metrics.get("goal", 0.0))),
                "clean_takeover": float(episode_metrics.get("clean_takeover", 0.0)),
                "clean_finish": float(episode_metrics.get("clean_finish", 0.0)),
                "shoot_finish": float(episode_metrics.get("shoot_finish", 0.0)),
                "collision": float(episode_metrics.get("collision_episode", episode_metrics.get("collision_rate", 0.0))),
                "band_steps": ep_band_steps,
                "walk_steps": episode_skill_counts.get("walk", 0),
                "dribble_steps": episode_skill_counts.get("dribble", 0),
                "shoot_steps": episode_skill_counts.get("shoot", 0),
                "stand_steps": episode_skill_counts.get("stand", 0),
                "gate_shoot_armed_steps": ep_gate_buckets.get("shoot_armed", 0),
                "gate_dribble_armed_steps": ep_gate_buckets.get("dribble_armed", 0),
                "gate_none_armed_steps": ep_gate_buckets.get("none_armed", 0),
            },
        )
        if kept:
            for row in ep_step_rows:
                _append_csv_row(steps_csv, row)
            observations.extend(ep_obs)
            actions.extend(ep_actions)
            stages.extend([stage_id] * len(ep_obs))
            episode_ids.extend([global_episode_id] * len(ep_obs))
            step_ids.extend(ep_step_ids)
            weights.extend(ep_weights)
            label_consistent.extend(ep_label_consistent)
            skill_counts.update(episode_skill_counts)
            distance_bins.update(ep_distance_bins)
            quality_counts.update(ep_quality)
            gate_bucket_counts.update(ep_gate_buckets)
            for bkt, counter in ep_gate_labels.items():
                gate_label_counts[bkt].update(counter)
            band_steps += ep_band_steps
            kept_episodes += 1
        else:
            dropped_episodes += 1
            drop_reasons[reason] += 1
        print(
            f"[prod-teacher] {scenario.name} attempt={attempted_episodes}/{max_attempts} "
            f"{'KEPT' if kept else 'DROP'}({reason}) "
            f"goal={float(episode_metrics.get('goal_rate', episode_metrics.get('goal', 0.0))):.0f} "
            f"clean_finish={float(episode_metrics.get('clean_finish', 0.0)):.0f} "
            f"kept={kept_episodes}/{episodes} dropped={dropped_episodes} "
            f"steps={sum(episode_skill_counts.values())} band={ep_band_steps} "
            f"skills={dict(episode_skill_counts)} gate={dict(ep_gate_buckets)}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    return {
        "scenario": scenario.name,
        "stage_id": stage_id,
        "is_contrast": scenario.is_contrast,
        "obstacles": scenario.obstacles,
        "requested_episodes": episodes,
        "kept_episodes": kept_episodes,
        "dropped_episodes": dropped_episodes,
        "attempted_episodes": attempted_episodes,
        "drop_reasons": dict(drop_reasons),
        "samples": len(stages),
        "band_steps": band_steps,
        "band_steps_attempted": band_steps_attempted,
        "elapsed_s": elapsed,
        "skill_counts": dict(skill_counts),
        "distance_bins": dict(distance_bins),
        "quality_counts": dict(quality_counts),
        "gate_bucket_counts": dict(gate_bucket_counts),
        "gate_label_counts": {bkt: dict(counter) for bkt, counter in gate_label_counts.items()},
        # popped by main() before serialization
        "observations": observations,
        "actions": actions,
        "stages": stages,
        "episode_ids": episode_ids,
        "step_ids": step_ids,
        "weights": weights,
        "label_consistent": label_consistent,
    }


def _selected_scenarios(episodes: int) -> list[ProductionScenario]:
    if SCENARIO_SET == "eval":
        scenarios = production_scenarios(episodes)
    elif SCENARIO_SET == "topup":
        scenarios = production_topup_scenarios(episodes)
    elif SCENARIO_SET == "all":
        scenarios = [*production_scenarios(episodes), *production_topup_scenarios(episodes)]
    else:
        raise ValueError("RL_PROD_TEACHER_SCENARIO_SET must be one of: eval, topup, all")
    if SCENARIO_FILTER:
        wanted = set(SCENARIO_FILTER)
        scenarios = [scenario for scenario in scenarios if scenario.name in wanted]
        missing = sorted(wanted.difference({scenario.name for scenario in scenarios}))
        if missing:
            raise ValueError(f"Unknown RL_PROD_TEACHER_SCENARIOS entries: {missing}")
    return scenarios


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"dataset_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _selected_scenarios(EPISODES_PER_SCENARIO)
    summary: dict[str, object] = {
        "backend_mode": BACKEND_MODE,
        "simrobot_profile": SIMROBOT_PROFILE,
        "simrobot_launch": SIMROBOT_LAUNCH,
        "env_stage_id": ENV_STAGE_ID,
        "episodes_per_scenario": EPISODES_PER_SCENARIO,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "max_attempt_multiplier": MAX_ATTEMPT_MULTIPLIER,
        "seed_base": SEED_BASE,
        "success_filter_enabled": SUCCESS_FILTER_ENABLED,
        "success_filter_metric": SUCCESS_FILTER_METRIC,
        "production_band_mm": list(PRODUCTION_BAND_MM),
        "scenario_set": SCENARIO_SET,
        "scenario_filter": list(SCENARIO_FILTER),
        "scenario_names": [s.name for s in scenarios],
        "scenarios": [],
    }

    all_observations: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_stages: list[int] = []
    all_episode_ids: list[int] = []
    all_step_ids: list[int] = []
    all_weights: list[float] = []
    all_label_consistent: list[int] = []
    global_skill_counts: Counter[str] = Counter()
    global_distance_bins: Counter[str] = Counter()
    global_quality_counts: Counter[str] = Counter()
    global_gate_buckets: Counter[str] = Counter()
    global_gate_labels: dict[str, Counter[str]] = defaultdict(Counter)
    global_band_steps = 0
    episode_offset = 0

    env = _make_env()
    try:
        for index, scenario in enumerate(scenarios):
            print(f"\n[prod-teacher] scenario={scenario.name} stage={scenario.stage_id} "
                  f"episodes={scenario.episodes} obstacles={scenario.obstacles} "
                  f"contrast={scenario.is_contrast}", flush=True)
            result = _collect_scenario(
                env,
                scenario,
                output_dir,
                seed_base=SEED_BASE + index * 10_000,
                episode_offset=episode_offset,
            )
            all_observations.extend(result.pop("observations"))
            all_actions.extend(result.pop("actions"))
            all_stages.extend(result.pop("stages"))
            all_episode_ids.extend(result.pop("episode_ids"))
            all_step_ids.extend(result.pop("step_ids"))
            all_weights.extend(result.pop("weights"))
            all_label_consistent.extend(result.pop("label_consistent"))
            episode_offset += int(result["kept_episodes"])
            global_skill_counts.update(result["skill_counts"])
            global_distance_bins.update(result["distance_bins"])
            global_quality_counts.update(result["quality_counts"])
            global_gate_buckets.update(result["gate_bucket_counts"])
            for bkt, counter in result["gate_label_counts"].items():
                global_gate_labels[bkt].update(counter)
            global_band_steps += int(result["band_steps"])
            summary["scenarios"].append(result)
    finally:
        env.close()

    observations_arr = (
        np.stack(all_observations).astype(np.float32) if all_observations else np.zeros((0, 0), dtype=np.float32)
    )
    actions_arr = np.stack(all_actions).astype(np.float32) if all_actions else np.zeros((0, 5), dtype=np.float32)
    weights_arr = np.asarray(all_weights, dtype=np.float32)
    np.savez_compressed(
        output_dir / "samples.npz",
        observations=observations_arr,
        actions=actions_arr,
        stages=np.asarray(all_stages, dtype=np.int16),
        episode_ids=np.asarray(all_episode_ids, dtype=np.int32),
        step_ids=np.asarray(all_step_ids, dtype=np.int16),
        weights=weights_arr,
        skill_order=np.asarray(SKILL_ORDER),
    )
    total = int(actions_arr.shape[0])
    accepted = int(np.count_nonzero(weights_arr > 0.0))
    # Independent Design-A audit: cross-check the saved weights against per-sample gate-label
    # consistency (not just trusting the in-loop zeroing). positive_weight_inconsistent MUST be 0
    # — if it isn't, the dataset would teach BC a label the baked gate forbids (the exact bug
    # that produced the first aborted run). This lets each dataset self-certify the guarantee.
    consistent_arr = np.asarray(all_label_consistent, dtype=bool)
    inconsistent_samples = int((~consistent_arr).sum()) if consistent_arr.size else 0
    positive_weight_inconsistent = (
        int(np.count_nonzero((weights_arr > 0.0) & ~consistent_arr)) if consistent_arr.size else 0
    )
    summary.update(
        {
            "output_dir": str(output_dir),
            "total_samples": total,
            "accepted_samples": accepted,
            "accepted_rate": float(accepted / max(1, total)),
            "full_weight_samples": int(np.count_nonzero(weights_arr >= 1.0)),
            "observation_shape": list(observations_arr.shape),
            "action_shape": list(actions_arr.shape),
            "global_skill_counts": dict(global_skill_counts),
            "global_distance_bins": dict(global_distance_bins),
            "global_quality_counts": dict(global_quality_counts),
            "global_gate_bucket_counts": dict(global_gate_buckets),
            "global_gate_label_counts": {bkt: dict(counter) for bkt, counter in global_gate_labels.items()},
            "band_2p5_3m_samples": global_band_steps,
            "band_2p5_3m_sample_rate": float(global_band_steps / max(1, total)),
            "gate_label_inconsistent_samples": inconsistent_samples,
            "positive_weight_gate_label_inconsistent_samples": positive_weight_inconsistent,
            "positive_weight_gate_label_consistent": bool(positive_weight_inconsistent == 0),
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
