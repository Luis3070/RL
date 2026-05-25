from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from action import SKILL_ORDER
from config import ENV, ROOT_DIR
from curriculum import (
    get_stage_scene_path,
    get_stage_spec,
    sample_finish_sequence_world_spec,
    sample_finish_world_spec,
    sample_preshoot_dribble_world_spec,
    sample_stage_world_spec,
)
from environment import RealSimRobotEnv
from pybh_backend import BackendObservation, WorldSpec
from teacher_policy import obs_from_raw_mapping, teacher_policy_payload

# Edit these constants directly for production collection.
BACKEND_MODE = os.getenv("RL_TEACHER_BACKEND_MODE", "simrobot_visible")
SIMROBOT_PROFILE = os.getenv("RL_TEACHER_SIMROBOT_PROFILE", "3d_visible")
SIMROBOT_LAUNCH = os.getenv("RL_TEACHER_SIMROBOT_LAUNCH", "1") == "1"
STAGE_TRAJECTORIES = {
    -1: 1,
    0: 2,
    1: 2,
    2: 1,
    3: 1,
}
# 700 (was 360): match the longer training budget so the teacher can DRIVE the ball to
# goal and finish within a collection episode (the full chain, not a truncated approach).
MAX_STEPS_PER_EPISODE = int(os.getenv("RL_TEACHER_MAX_STEPS", "700"))
SEED_BASE = 93000
OUTPUT_ROOT = ROOT_DIR / "runs" / "teacher_bc_dataset"
MIN_SAMPLE_WEIGHT_CONFIDENCE = 0.35
SEED_STRATEGY = os.getenv("RL_TEACHER_SEED_STRATEGY", "stratified")
SEED_SCAN_LIMIT = int(os.getenv("RL_TEACHER_SEED_SCAN_LIMIT", "3000"))
MAX_EPISODE_ATTEMPT_MULTIPLIER = int(os.getenv("RL_TEACHER_MAX_ATTEMPT_MULTIPLIER", "4"))
# Reduced near-goal shoot-scenario multipliers (were 8/6/4/4 = 22 near-goal episodes per
# unit, which dominated collection and starved natural walk/dribble -> dribble came out at
# only 593 samples). The drive curriculum now produces the full walk->dribble->drive->shoot
# chain in the *natural* "curriculum" episodes, so these near-goal scenarios only need to
# top up shoot/contrast coverage, not dominate.
SHOOT_SEQUENCE_EPISODES_PER_STAGE = int(os.getenv("RL_TEACHER_SHOOT_SEQUENCE_EPISODES_PER_STAGE", "2"))
PRESHOOT_DRIBBLE_EPISODES_PER_STAGE = int(os.getenv("RL_TEACHER_PRESHOOT_DRIBBLE_EPISODES_PER_STAGE", "2"))
SHOOT_FINISH_EPISODES_PER_STAGE = int(os.getenv("RL_TEACHER_SHOOT_FINISH_EPISODES_PER_STAGE", "1"))
SHOOT_CONTRAST_EPISODES_PER_STAGE = int(os.getenv("RL_TEACHER_SHOOT_CONTRAST_EPISODES_PER_STAGE", "1"))
SHOOT_CONTRAST_MAX_STEPS = int(os.getenv("RL_TEACHER_SHOOT_CONTRAST_MAX_STEPS", "90"))
SHOOT_CONTRAST_STOP_ON_SHOOT = os.getenv("RL_TEACHER_SHOOT_CONTRAST_STOP_ON_SHOOT", "1") == "1"

STAGE_DISTANCE_BANDS_MM = {
    -1: ((300.0, 650.0), (650.0, 950.0), (950.0, 1200.0)),
    0: ((700.0, 1000.0), (1000.0, 1300.0), (1300.0, 1600.0)),
    # Stage 1/2 now use the drive curriculum: ROBOT-to-ball distance is the approach leg
    # (~650-1050mm, sample_stage_world_spec diverse=False), so the seed-stratification bands
    # match that range (the old 900-3200mm bands never filled -> full 3000-seed scan + no
    # stratification). Ball-to-GOAL drive diversity comes from the reset rng.
    1: ((650.0, 800.0), (800.0, 920.0), (920.0, 1050.0)),
    2: ((650.0, 800.0), (800.0, 920.0), (920.0, 1050.0)),
    3: ((600.0, 1200.0), (1200.0, 2200.0), (2200.0, 3500.0)),
    4: ((600.0, 1200.0), (1200.0, 2200.0), (2200.0, 3500.0)),
}

# Rebalanced: prioritize DRIBBLE (the drive — the behavior the agent lost) and pull back the
# shoot inflation (shoot now arrives naturally at the end of each completable drive chain, so
# it no longer needs to be force-collected via near-goal scenarios).
DEFAULT_STAGE_MIN_SKILL_COUNTS = {
    0: {"walk": 20, "dribble": 20},
    1: {"walk": 140, "dribble": 220, "shoot": 90},
    2: {"walk": 120, "dribble": 190, "shoot": 80},
    3: {"walk": 120, "dribble": 190, "shoot": 80},
}


def _stage_trajectories_from_env() -> dict[int, int]:
    spec = os.getenv("RL_TEACHER_STAGE_TRAJECTORIES")
    if not spec:
        return dict(STAGE_TRAJECTORIES)
    result: dict[int, int] = {}
    for chunk in spec.split(","):
        if not chunk.strip():
            continue
        stage_text, episodes_text = chunk.split(":", maxsplit=1)
        result[int(stage_text.strip())] = int(episodes_text.strip())
    return result


def _world_distance_for_seed(stage_id: int, seed: int) -> float:
    world_spec = sample_stage_world_spec(stage_id, np.random.default_rng(seed))
    return float(math.hypot(world_spec.ball_x - world_spec.robot_x, world_spec.ball_y - world_spec.robot_y))


def _select_stage_seeds(stage_id: int, seed_offset: int, episodes: int) -> list[int]:
    if SEED_STRATEGY != "stratified" or episodes <= 0:
        return [seed_offset + idx for idx in range(episodes)]
    bands = STAGE_DISTANCE_BANDS_MM.get(stage_id)
    if not bands:
        return [seed_offset + idx for idx in range(episodes)]
    buckets: list[list[int]] = [[] for _ in bands]
    for delta in range(SEED_SCAN_LIMIT):
        seed = seed_offset + delta
        distance = _world_distance_for_seed(stage_id, seed)
        for band_index, (lower, upper) in enumerate(bands):
            if lower <= distance <= upper:
                buckets[band_index].append(seed)
                break
        minimum_total = max(episodes * 3, episodes + len(bands))
        has_required_near_band = bool(buckets[0]) if stage_id >= 1 else True
        has_all_bands = all(bool(bucket) for bucket in buckets)
        if sum(len(bucket) for bucket in buckets) >= minimum_total and has_required_near_band and has_all_bands:
            break

    selected: list[int] = []
    if stage_id >= 1:
        # For finish stages, prioritize the valid near band first so the dataset
        # contains handoff/finish labels, then add farther reset diversity.
        band_order = tuple(range(len(bands)))
    else:
        band_order = tuple(range(len(bands)))
    while len(selected) < episodes and any(buckets):
        for band_index in band_order:
            if buckets[band_index]:
                selected.append(buckets[band_index].pop(0))
                if len(selected) >= episodes:
                    break
        else:
            break
    if len(selected) < episodes:
        used = set(selected)
        for idx in range(SEED_SCAN_LIMIT):
            seed = seed_offset + idx
            if seed not in used:
                selected.append(seed)
                if len(selected) >= episodes:
                    break
    return selected


def _stage_skill_targets(stage_id: int, episodes: int) -> dict[str, int]:
    scale = max(1, int(episodes))
    return {
        skill_name: int(count * scale)
        for skill_name, count in DEFAULT_STAGE_MIN_SKILL_COUNTS.get(stage_id, {}).items()
    }


def _skill_targets_satisfied(skill_counts: Counter[str], targets: Mapping[str, int]) -> bool:
    return all(int(skill_counts.get(skill_name, 0)) >= int(target) for skill_name, target in targets.items())


def _append_csv_row(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _make_env(stage_id: int) -> RealSimRobotEnv:
    stage = get_stage_spec(stage_id)
    backend_cfg = replace(
        ENV.backend,
        mode=BACKEND_MODE,
        simrobot_profile=SIMROBOT_PROFILE,
        simrobot_scene_path=get_stage_scene_path(stage_id, SIMROBOT_PROFILE),
        simrobot_launch=SIMROBOT_LAUNCH,
    )
    env_cfg = replace(
        ENV,
        stage_id=stage_id,
        max_episode_steps=MAX_STEPS_PER_EPISODE,
        backend=backend_cfg,
        active_skill_names=stage.active_skill_names,
        enable_handoff_override=False,
    )
    return RealSimRobotEnv(env_cfg=env_cfg)


def _ball_distance(raw_obs: Mapping[str, object]) -> float:
    dx = float(raw_obs.get("ball_x", 0.0)) - float(raw_obs.get("robot_x", 0.0))
    dy = float(raw_obs.get("ball_y", 0.0)) - float(raw_obs.get("robot_y", 0.0))
    return float(math.hypot(dx, dy))


def _sample_quality_weight(info: Mapping[str, object]) -> float:
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


def _distance_bin(distance_mm: float) -> str:
    if distance_mm < 450.0:
        return "takeover"
    if distance_mm < 900.0:
        return "near"
    if distance_mm < 1600.0:
        return "mid"
    if distance_mm < 2600.0:
        return "far"
    return "full"


def _collect_stage(stage_id: int, episodes: int, output_dir: Path, seed_offset: int) -> dict[str, object]:
    env = _make_env(stage_id)
    stage = get_stage_spec(stage_id)
    steps_csv = output_dir / "steps.csv"
    episodes_csv = output_dir / "episodes.csv"
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    stages: list[int] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    weights: list[float] = []
    skill_counts: Counter[str] = Counter()
    distance_bins: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    started = time.perf_counter()
    try:
        skill_targets = _stage_skill_targets(stage_id, episodes)
        max_attempts = max(episodes, episodes * max(1, MAX_EPISODE_ATTEMPT_MULTIPLIER))
        selected_seeds = _select_stage_seeds(stage_id, seed_offset, max_attempts)
        episode_plan: list[tuple[str, int, WorldSpec | None]] = []
        if "shoot" in skill_targets:
            sequence_episodes = max(1, SHOOT_SEQUENCE_EPISODES_PER_STAGE * max(1, episodes))
            bridge_episodes = max(1, PRESHOOT_DRIBBLE_EPISODES_PER_STAGE * max(1, episodes))
            finish_episodes = max(1, SHOOT_FINISH_EPISODES_PER_STAGE * max(1, episodes))
            contrast_episodes = max(0, SHOOT_CONTRAST_EPISODES_PER_STAGE * max(1, episodes))
            for sequence_index in range(sequence_episodes):
                sequence_seed = seed_offset + 80_000 + sequence_index
                episode_plan.append(
                    (
                        "shoot_sequence",
                        sequence_seed,
                        sample_finish_sequence_world_spec(stage_id, np.random.default_rng(sequence_seed)),
                    )
                )
            for bridge_index in range(bridge_episodes):
                bridge_seed = seed_offset + 90_000 + bridge_index
                episode_plan.append(
                    (
                        "pre_shoot_dribble",
                        bridge_seed,
                        sample_preshoot_dribble_world_spec(stage_id, np.random.default_rng(bridge_seed)),
                    )
                )
            for shoot_index in range(finish_episodes):
                shoot_seed = seed_offset + 100_000 + shoot_index
                episode_plan.append(
                    (
                        "shoot_finish",
                        shoot_seed,
                        sample_finish_world_spec(stage_id, np.random.default_rng(shoot_seed), contrast=False),
                    )
                )
            for contrast_index in range(contrast_episodes):
                contrast_seed = seed_offset + 120_000 + contrast_index
                episode_plan.append(
                    (
                        "shoot_contrast",
                        contrast_seed,
                        sample_finish_world_spec(stage_id, np.random.default_rng(contrast_seed), contrast=True),
                    )
                )
        episode_plan.extend(("curriculum", seed, None) for seed in selected_seeds)
        seed_distances = {}
        for scenario_type, seed, world_spec in episode_plan:
            if world_spec is None:
                seed_distances[seed] = _world_distance_for_seed(stage_id, seed)
            else:
                seed_distances[seed] = float(
                    math.hypot(world_spec.ball_x - world_spec.robot_x, world_spec.ball_y - world_spec.robot_y)
                )
        print(
            f"[teacher-dataset] stage={stage_id} selected_seeds="
            f"{[(scenario, seed, round(seed_distances[seed])) for scenario, seed, _ in episode_plan]}",
            flush=True,
        )
        kept_episodes = 0
        kept_curriculum_episodes = 0
        attempted_episodes = 0
        for episode_idx, (scenario_type, seed, world_spec) in enumerate(episode_plan):
            if kept_curriculum_episodes >= episodes and _skill_targets_satisfied(skill_counts, skill_targets):
                break
            attempted_episodes += 1
            reset_options = {"world_spec": world_spec} if world_spec is not None else None
            obs_dict, infos = env.reset(seed=seed, options=reset_options)
            agent_id = "player_1"
            prev_skill_name: str | None = None
            total_reward = 0.0
            episode_skill_counts: Counter[str] = Counter()
            final_info: Mapping[str, object] = {}
            for step in range(1, MAX_STEPS_PER_EPISODE + 1):
                info = infos.get(agent_id, {})
                raw_obs = info.get("raw_obs", {})
                if not isinstance(raw_obs, Mapping):
                    raise RuntimeError("teacher dataset collection requires raw_obs in env info")
                backend_obs = obs_from_raw_mapping(raw_obs)
                payload, absolute_action, residual_action = teacher_policy_payload(
                    backend_obs,
                    stage_id,
                    prev_skill_name,
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
                obs_dict, rewards, terminateds, truncateds, infos = env.step({agent_id: payload})
                step_info = infos.get(agent_id, {})
                reward = float(rewards.get(agent_id, 0.0))
                total_reward += reward
                final_info = step_info
                weight = _sample_quality_weight(step_info)
                ball_distance = _ball_distance(raw_obs)
                distance_bin = _distance_bin(ball_distance)
                skill_name = residual_action.skill_name
                if scenario_type == "shoot_contrast" and SHOOT_CONTRAST_STOP_ON_SHOOT and skill_name == "shoot":
                    # Contrast episodes are for the decision boundary before a shot is valid.
                    # Once the teacher has recovered into a valid shot, finish episodes already
                    # provide cleaner positive labels, so stop instead of diluting negatives.
                    break

                observations.append(current_obs)
                actions.append(current_action)
                stages.append(stage_id)
                episode_ids.append(episode_idx)
                step_ids.append(step)
                weights.append(weight)
                skill_counts[skill_name] += 1
                episode_skill_counts[skill_name] += 1
                distance_bins[distance_bin] += 1
                quality_counts["accepted_weighted"] += int(weight > 0.0)
                quality_counts["full_weight"] += int(weight >= 1.0)
                quality_counts["low_weight"] += int(0.0 < weight < 1.0)
                quality_counts["rejected"] += int(weight <= 0.0)

                _append_csv_row(
                    steps_csv,
                    {
                        "stage_id": stage_id,
                        "episode": episode_idx,
                        "seed": seed,
                        "scenario_type": scenario_type,
                        "initial_seed_distance_mm": float(seed_distances.get(seed, 0.0)),
                        "step": step,
                        "reward": reward,
                        "sample_weight": weight,
                        "teacher_skill": skill_name,
                        "teacher_abs_target_x": float(absolute_action.target_x),
                        "teacher_abs_target_y": float(absolute_action.target_y),
                        "teacher_abs_target_theta": float(absolute_action.target_theta),
                        "label_skill_type": int(residual_action.skill_type),
                        "label_residual_x": float(residual_action.target_x),
                        "label_residual_y": float(residual_action.target_y),
                        "label_residual_theta": float(residual_action.target_theta),
                        "executed_skill_name": step_info.get("executed_skill_name"),
                        "preferred_skill_name": step_info.get("preferred_skill_name"),
                        "takeover_envelope": int(bool(step_info.get("takeover_envelope", False))),
                        "walk_zone": int(bool(step_info.get("walk_zone", False))),
                        "dribble_zone": int(bool(step_info.get("dribble_zone", False))),
                        "shoot_zone": int(bool(step_info.get("shoot_zone", False))),
                        "ball_distance_mm": ball_distance,
                        "distance_bin": distance_bin,
                        "ball_confidence_score": float(step_info.get("ball_confidence_score", 0.0)),
                        "ball_trusted_mask": int(bool(step_info.get("ball_trusted_mask", False))),
                        "ball_rescued_mask": int(bool(step_info.get("ball_rescued_mask", False))),
                        "can_score_now": int(bool(raw_obs.get("can_score_now", False))),
                        "shot_quality_no_obstacles": float(raw_obs.get("shot_quality_no_obstacles", 0.0)),
                        "shot_opening_with_obstacles": float(raw_obs.get("shot_opening_with_obstacles", 0.0)),
                        "synthetic_ball_observation": int(bool(step_info.get("synthetic_ball_observation", False))),
                        "synthetic_pose_observation": int(bool(step_info.get("synthetic_pose_observation", False))),
                        "corrected_exported_ball": int(bool(step_info.get("corrected_exported_ball", False))),
                        "pose_alignment_warning": int(bool(step_info.get("pose_alignment_warning", False))),
                        "ball_observation_collapsed": int(bool(step_info.get("ball_observation_collapsed", False))),
                        "nearest_teammate_front_dist": float(raw_obs.get("nearest_teammate_front_dist", 0.0)),
                        "nearest_opponent_front_dist": float(raw_obs.get("nearest_opponent_front_dist", 0.0)),
                        "nearest_uncertain_front_dist": float(raw_obs.get("nearest_uncertain_front_dist", 0.0)),
                    },
                )
                prev_skill_name = skill_name
                if scenario_type == "shoot_contrast" and step >= SHOOT_CONTRAST_MAX_STEPS:
                    break
                if not env.agents or any(terminateds.values()) or any(truncateds.values()):
                    break

            episode_metrics = dict(final_info.get("episode_metrics", {})) if final_info else {}
            _append_csv_row(
                episodes_csv,
                {
                    "stage_id": stage_id,
                    "episode": episode_idx,
                    "seed": seed,
                    "scenario_type": scenario_type,
                    "initial_seed_distance_mm": float(seed_distances.get(seed, 0.0)),
                    "steps": sum(episode_skill_counts.values()),
                    "total_reward": total_reward,
                    "goal": float(episode_metrics.get("goal", 0.0)),
                    "clean_takeover": float(episode_metrics.get("clean_takeover", 0.0)),
                    "clean_finish": float(episode_metrics.get("clean_finish", 0.0)),
                    "ball_trusted_rate": float(episode_metrics.get("ball_trusted_rate", 0.0)),
                    "ball_rescued_rate": float(episode_metrics.get("ball_rescued_rate", 0.0)),
                    "pose_alignment_warning_rate": float(episode_metrics.get("pose_alignment_warning_rate", 0.0)),
                    "walk_steps": episode_skill_counts.get("walk", 0),
                    "dribble_steps": episode_skill_counts.get("dribble", 0),
                    "shoot_steps": episode_skill_counts.get("shoot", 0),
                    "stand_steps": episode_skill_counts.get("stand", 0),
                },
            )
            print(
                f"[teacher-dataset] stage={stage_id} attempt={attempted_episodes}/{max_attempts} "
                f"kept={kept_episodes + 1} curriculum_kept={kept_curriculum_episodes + int(scenario_type == 'curriculum')}/{episodes} "
                f"steps={sum(episode_skill_counts.values())} "
                f"skills={dict(episode_skill_counts)} total_skills={dict(skill_counts)}",
                flush=True,
            )
            kept_episodes += 1
            kept_curriculum_episodes += int(scenario_type == "curriculum")
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    return {
        "stage_id": stage_id,
        "episodes": episodes,
        "kept_episodes": kept_episodes,
        "kept_curriculum_episodes": kept_curriculum_episodes,
        "attempted_episodes": attempted_episodes,
        "skill_targets": dict(skill_targets),
        "skill_targets_satisfied": _skill_targets_satisfied(skill_counts, skill_targets),
        "selected_seed_distances": {str(seed): float(distance) for seed, distance in seed_distances.items()},
        "samples": len(stages),
        "elapsed_s": elapsed,
        "skill_counts": dict(skill_counts),
        "distance_bins": dict(distance_bins),
        "quality_counts": dict(quality_counts),
        "active_skills": stage.active_skill_names,
        "observations": observations,
        "actions": actions,
        "stages": stages,
        "episode_ids": episode_ids,
        "step_ids": step_ids,
        "weights": weights,
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"dataset_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_observations: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_stages: list[int] = []
    all_episode_ids: list[int] = []
    all_step_ids: list[int] = []
    all_weights: list[float] = []
    summary = {
        "backend_mode": BACKEND_MODE,
        "simrobot_profile": SIMROBOT_PROFILE,
        "simrobot_launch": SIMROBOT_LAUNCH,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "seed_strategy": SEED_STRATEGY,
        "seed_scan_limit": SEED_SCAN_LIMIT,
        "shoot_sequence_episodes_per_stage": SHOOT_SEQUENCE_EPISODES_PER_STAGE,
        "preshoot_dribble_episodes_per_stage": PRESHOOT_DRIBBLE_EPISODES_PER_STAGE,
        "shoot_finish_episodes_per_stage": SHOOT_FINISH_EPISODES_PER_STAGE,
        "shoot_contrast_episodes_per_stage": SHOOT_CONTRAST_EPISODES_PER_STAGE,
        "shoot_contrast_max_steps": SHOOT_CONTRAST_MAX_STEPS,
        "shoot_contrast_stop_on_shoot": SHOOT_CONTRAST_STOP_ON_SHOOT,
        "stage_trajectories": _stage_trajectories_from_env(),
        "stages": [],
    }
    global_skill_counts: Counter[str] = Counter()
    global_distance_bins: Counter[str] = Counter()
    global_quality_counts: Counter[str] = Counter()
    episode_offset = 0
    for stage_id, episodes in summary["stage_trajectories"].items():
        stage_result = _collect_stage(stage_id, episodes, output_dir, SEED_BASE + stage_id * 1000)
        all_observations.extend(stage_result.pop("observations"))
        all_actions.extend(stage_result.pop("actions"))
        stage_ids = list(stage_result.pop("stages"))
        all_stages.extend(stage_ids)
        all_episode_ids.extend([episode_offset + int(value) for value in stage_result.pop("episode_ids")])
        all_step_ids.extend(stage_result.pop("step_ids"))
        all_weights.extend(stage_result.pop("weights"))
        episode_offset += int(stage_result.get("attempted_episodes", episodes))
        global_skill_counts.update(stage_result["skill_counts"])
        global_distance_bins.update(stage_result["distance_bins"])
        global_quality_counts.update(stage_result["quality_counts"])
        summary["stages"].append(stage_result)

    observations_arr = np.stack(all_observations).astype(np.float32) if all_observations else np.zeros((0, 0), dtype=np.float32)
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
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
