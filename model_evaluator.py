from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

import torch

import train
from action import SKILL_ORDER
from config import ENV, TRAIN
from train import (
    RUNS_DIR,
    _aggregate_eval_metrics,
    _append_csv_row,
    _tensorize_eval_obs,
    _tensorize_eval_state,
    build_env,
    build_mappo,
    make_env_config,
)

LIVE_POSE_JUMP_WARNING_MM = 350.0


def _latest_checkpoint(experiment_name: str) -> Path:
    checkpoint_dir = RUNS_DIR / experiment_name
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_stage*_step*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints[-1]


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def _load_checkpoint_metadata(checkpoint_path: Path, device: str) -> Mapping[str, object]:
    return torch.load(checkpoint_path, map_location=device, weights_only=True)


def _policy_payloads(agent, eval_env, obs, *, device: str, timestep: int, timesteps: int) -> dict[str, dict]:
    obs_tensor = _tensorize_eval_obs(obs, device=device)
    state_value = _tensorize_eval_state(eval_env._env.state(), device=device)
    states = {agent_id: state_value for agent_id in eval_env.possible_agents}
    with torch.no_grad():
        actions, outputs = agent.act(
            observations=obs_tensor,
            states=states,
            timestep=timestep,
            timesteps=timesteps,
        )

    payloads: dict[str, dict] = {}
    for agent_id in eval_env.possible_agents:
        action_tensor = torch.nan_to_num(
            actions[agent_id],
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).detach().cpu().reshape(-1).numpy()
        payload = {
            "skill_type": int(round(float(action_tensor[0]))),
            "policy_skill_raw": float(action_tensor[0]),
            "target_x": float(action_tensor[1]),
            "target_y": float(action_tensor[2]),
            "target_theta": float(action_tensor[3]),
            "pass_target": int(round(float(action_tensor[4]))),
        }
        agent_outputs = outputs.get(agent_id, {})
        skill_probs = agent_outputs.get("skill_probs")
        if torch.is_tensor(skill_probs) and skill_probs.numel() >= len(SKILL_ORDER):
            flat_probs = skill_probs.detach().cpu().reshape(-1, len(SKILL_ORDER))[0].numpy()
            for skill_index, skill_name in enumerate(SKILL_ORDER):
                payload[f"policy_skill_prob_{skill_name}"] = float(flat_probs[skill_index])
        skill_entropy = agent_outputs.get("skill_entropy")
        if torch.is_tensor(skill_entropy) and skill_entropy.numel():
            payload["policy_skill_entropy"] = float(skill_entropy.detach().cpu().reshape(-1)[0].item())
        log_std = agent_outputs.get("log_std")
        if torch.is_tensor(log_std) and log_std.numel():
            payload["policy_param_std_mean"] = float(log_std.detach().cpu().exp().mean().item())
        payloads[agent_id] = payload
    return payloads


def _format_probs(info: Mapping[str, object]) -> str:
    return (
        f"p(stand)={float(info.get('policy_skill_prob_stand', 0.0)):.3f} "
        f"p(walk)={float(info.get('policy_skill_prob_walk', 0.0)):.3f} "
        f"p(dribble)={float(info.get('policy_skill_prob_dribble', 0.0)):.3f} "
        f"p(shoot)={float(info.get('policy_skill_prob_shoot', 0.0)):.3f}"
    )


def _ball_distance_from_info(info: Mapping[str, object]) -> float:
    raw_obs = info.get("raw_obs", {})
    if not isinstance(raw_obs, Mapping):
        return 0.0
    robot_x = float(raw_obs.get("robot_x", 0.0))
    robot_y = float(raw_obs.get("robot_y", 0.0))
    ball_x = float(raw_obs.get("ball_x", 0.0))
    ball_y = float(raw_obs.get("ball_y", 0.0))
    dx = ball_x - robot_x
    dy = ball_y - robot_y
    return float((dx * dx + dy * dy) ** 0.5)


def _live_rollout(
    agent,
    eval_env,
    *,
    device: str,
    seed: int,
    max_steps: int,
    sleep_s: float,
    csv_path: Path,
) -> Mapping[str, object]:
    previous_mode = getattr(agent, "training", None)
    agent.enable_training_mode(False)
    obs, infos = eval_env.reset(seed=seed)
    total_reward = 0.0
    final_info: Mapping[str, object] = {}
    previous_robot_xy: tuple[float, float] | None = None
    live_counts = {
        "steps": 0,
        "target_repair": 0,
        "handoff_override": 0,
        "pose_jump": 0,
        "corrected_exported_robot_pose": 0,
    }
    live_sums = {
        "policy_target_error_mm": 0.0,
        "policy_theta_error_rad": 0.0,
        "pose_jump_max_mm": 0.0,
    }
    live_policy_skill_counts = {name: 0 for name in SKILL_ORDER}
    live_executed_skill_counts = {name: 0 for name in SKILL_ORDER}
    print("[live] reset complete. Watch SimRobot now.", flush=True)
    try:
        for step in range(1, max_steps + 1):
            payloads = _policy_payloads(agent, eval_env, obs, device=device, timestep=step, timesteps=max_steps)
            obs, rewards, _terminated, _truncated, infos = eval_env.step(payloads)
            agent_id = eval_env.possible_agents[0]
            reward = float(rewards.get(agent_id, 0.0))
            total_reward += reward
            info = infos.get(agent_id, {})
            raw_obs = info.get("raw_obs", {})
            if not isinstance(raw_obs, Mapping):
                raw_obs = {}
            final_info = info
            robot_x = float(raw_obs.get("robot_x", 0.0))
            robot_y = float(raw_obs.get("robot_y", 0.0))
            robot_theta = float(raw_obs.get("robot_theta", 0.0))
            ball_x = float(raw_obs.get("ball_x", 0.0))
            ball_y = float(raw_obs.get("ball_y", 0.0))
            pose_jump_mm = 0.0
            if previous_robot_xy is not None:
                dx = robot_x - previous_robot_xy[0]
                dy = robot_y - previous_robot_xy[1]
                pose_jump_mm = float((dx * dx + dy * dy) ** 0.5)
            previous_robot_xy = (robot_x, robot_y)
            live_counts["steps"] += 1
            live_counts["target_repair"] += int(bool(info.get("target_repair_applied", False)))
            live_counts["handoff_override"] += int(bool(info.get("handoff_override_applied", False)))
            live_counts["pose_jump"] += int(pose_jump_mm > LIVE_POSE_JUMP_WARNING_MM)
            live_counts["corrected_exported_robot_pose"] += int(bool(info.get("corrected_exported_robot_pose", False)))
            live_sums["policy_target_error_mm"] += float(info.get("policy_target_error_mm", 0.0))
            live_sums["policy_theta_error_rad"] += float(info.get("policy_theta_error_rad", 0.0))
            live_sums["pose_jump_max_mm"] = max(live_sums["pose_jump_max_mm"], pose_jump_mm)
            policy_skill_name = str(info.get("policy_skill_name") or "")
            executed_skill_name = str(info.get("executed_skill_name") or "")
            if policy_skill_name in live_policy_skill_counts:
                live_policy_skill_counts[policy_skill_name] += 1
            if executed_skill_name in live_executed_skill_counts:
                live_executed_skill_counts[executed_skill_name] += 1
            row = {
                "step": step,
                "reward": reward,
                "total_reward": total_reward,
                "policy_skill_name": info.get("policy_skill_name"),
                "executed_skill_name": info.get("executed_skill_name"),
                "preferred_skill_name": info.get("preferred_skill_name"),
                "takeover_envelope": int(bool(info.get("takeover_envelope", False))),
                "walk_zone": int(bool(info.get("walk_zone", False))),
                "dribble_zone": int(bool(info.get("dribble_zone", False))),
                "shoot_zone": int(bool(info.get("shoot_zone", False))),
                "handoff_override_applied": int(bool(info.get("handoff_override_applied", False))),
                "target_repair_applied": int(bool(info.get("target_repair_applied", False))),
                "target_repair_reason": info.get("target_repair_reason"),
                "requested_skill": info.get("requested_skill"),
                "requested_pass_target": int(info.get("requested_pass_target", 0)),
                "policy_residual_x": float(info.get("policy_residual_x", 0.0)),
                "policy_residual_y": float(info.get("policy_residual_y", 0.0)),
                "policy_residual_theta": float(info.get("policy_residual_theta", 0.0)),
                "policy_target_error_mm": float(info.get("policy_target_error_mm", 0.0)),
                "policy_theta_error_rad": float(info.get("policy_theta_error_rad", 0.0)),
                "target_x": float(info.get("policy_target_x", 0.0)),
                "target_y": float(info.get("policy_target_y", 0.0)),
                "target_theta": float(info.get("policy_target_theta", 0.0)),
                "executed_target_x": float(info.get("executed_target_x", 0.0)),
                "executed_target_y": float(info.get("executed_target_y", 0.0)),
                "executed_target_theta": float(info.get("executed_target_theta", 0.0)),
                "policy_skill_prob_stand": float(info.get("policy_skill_prob_stand", 0.0)),
                "policy_skill_prob_walk": float(info.get("policy_skill_prob_walk", 0.0)),
                "policy_skill_prob_dribble": float(info.get("policy_skill_prob_dribble", 0.0)),
                "policy_skill_prob_shoot": float(info.get("policy_skill_prob_shoot", 0.0)),
                "obs_export_mode": int(info.get("obs_export_mode", 0)),
                "corrected_exported_robot_pose": int(bool(info.get("corrected_exported_robot_pose", False))),
                "corrected_exported_ball": int(bool(info.get("corrected_exported_ball", False))),
                "ball_export_source": int(info.get("ball_export_source", 0)),
                "ball_export_fresh": int(bool(info.get("ball_export_fresh", False))),
                "pose_alignment_error_mm": float(info.get("pose_alignment_error_mm", 0.0)),
                "pose_alignment_warning": int(bool(info.get("pose_alignment_warning", False))),
                "synthetic_pose_observation": int(bool(info.get("synthetic_pose_observation", False))),
                "synthetic_ball_observation": int(bool(info.get("synthetic_ball_observation", False))),
                "ball_observation_stale": int(bool(info.get("ball_observation_stale", False))),
                "ball_observation_collapsed": int(bool(info.get("ball_observation_collapsed", False))),
                "natural_time_since_ball_seen": float(info.get("natural_time_since_ball_seen", 0.0)),
                "natural_ball_seen_percentage": float(info.get("natural_ball_seen_percentage", 0.0)),
                "ball_trusted": int(bool(info.get("ball_trusted_mask", False))),
                "ball_visible": int(bool(info.get("ball_visible_mask", False))),
                "ball_rescued": int(bool(info.get("ball_rescued_mask", False))),
                "ball_confidence_score": float(info.get("ball_confidence_score", 0.0)),
                "ball_distance_mm": _ball_distance_from_info(info),
                "ball_rel_x": float(raw_obs.get("ball_rel_x", 0.0)),
                "ball_rel_y": float(raw_obs.get("ball_rel_y", 0.0)),
                "robot_x": robot_x,
                "robot_y": robot_y,
                "robot_theta": robot_theta,
                "pose_jump_mm": pose_jump_mm,
                "estimated_robot_x": float(info.get("estimated_robot_x", 0.0)),
                "estimated_robot_y": float(info.get("estimated_robot_y", 0.0)),
                "estimated_robot_theta": float(info.get("estimated_robot_theta", 0.0)),
                "ball_x": ball_x,
                "ball_y": ball_y,
                "estimated_ball_x": float(info.get("estimated_ball_x", 0.0)),
                "estimated_ball_y": float(info.get("estimated_ball_y", 0.0)),
            }
            _append_csv_row(csv_path, row)
            print(
                "[live] "
                f"step={step:03d} reward={reward:+.3f} "
                f"policy={info.get('policy_skill_name')} exec={info.get('executed_skill_name')} "
                f"takeover={int(bool(info.get('takeover_envelope', False)))} "
                f"zones=w{int(bool(info.get('walk_zone', False)))}"
                f"/d{int(bool(info.get('dribble_zone', False)))}"
                f"/s{int(bool(info.get('shoot_zone', False)))} "
                f"ball_dist={_ball_distance_from_info(info):.0f}mm "
                f"repair={int(bool(info.get('target_repair_applied', False)))} "
                f"override={int(bool(info.get('handoff_override_applied', False)))} "
                f"pose_warn={int(bool(info.get('pose_alignment_warning', False)))} "
                + _format_probs(info),
                flush=True,
            )
            if not eval_env.agents:
                break
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    finally:
        if previous_mode is not None:
            agent.enable_training_mode(bool(previous_mode))
    steps = max(1, int(live_counts["steps"]))
    metrics = {
        "live_steps": float(live_counts["steps"]),
        "live_target_repair_rate": float(live_counts["target_repair"] / steps),
        "live_handoff_override_rate": float(live_counts["handoff_override"] / steps),
        "live_policy_target_error_mean": float(live_sums["policy_target_error_mm"] / steps),
        "live_policy_theta_error_mean": float(live_sums["policy_theta_error_rad"] / steps),
        "live_pose_jump_rate": float(live_counts["pose_jump"] / steps),
        "live_pose_jump_max_mm": float(live_sums["pose_jump_max_mm"]),
        "live_corrected_exported_robot_pose_rate": float(live_counts["corrected_exported_robot_pose"] / steps),
    }
    for skill_name in ("stand", "walk", "dribble", "shoot"):
        metrics[f"live_policy_{skill_name}_rate"] = float(live_policy_skill_counts.get(skill_name, 0) / steps)
        metrics[f"live_executed_{skill_name}_rate"] = float(live_executed_skill_counts.get(skill_name, 0) / steps)
    return {"total_reward": total_reward, "final_info": dict(final_info), "metrics": metrics}


def _collect_metric_episodes(
    agent,
    eval_env,
    *,
    device: str,
    episodes: int,
    seed: int,
    csv_path: Path,
) -> dict[str, float]:
    previous_mode = getattr(agent, "training", None)
    agent.enable_training_mode(False)

    def policy_fn(obs, infos):
        del infos
        return _policy_payloads(agent, eval_env, obs, device=device, timestep=0, timesteps=1)

    try:
        episode_results = [
            eval_env.collect_episode(policy_fn, seed=seed + episode_idx, capture_steps=True)
            for episode_idx in range(episodes)
        ]
    finally:
        if previous_mode is not None:
            agent.enable_training_mode(bool(previous_mode))

    for episode_index, result in enumerate(episode_results):
        row = {
            "episode_index": int(episode_index),
            "steps": int(result.get("steps", 0)),
            "mean_reward": float(result.get("mean_reward", 0.0)),
        }
        for key, value in result.get("episode_metrics", {}).items():
            row[f"episode_metrics_{key}"] = value
        _append_csv_row(csv_path, row)
    return _aggregate_eval_metrics(episode_results)


def _print_health(summary: Mapping[str, float]) -> None:
    print("\n[health] skill usage", flush=True)
    for key in (
        "policy_stand_rate",
        "policy_walk_rate",
        "policy_dribble_rate",
        "policy_shoot_rate",
        "executed_stand_rate",
        "executed_walk_rate",
        "executed_dribble_rate",
        "executed_shoot_rate",
    ):
        print(f"  {key}={float(summary.get(key, 0.0)):.3f}", flush=True)
    print("[health] probabilities", flush=True)
    for key in (
        "policy_skill_prob_walk_mean",
        "policy_skill_prob_dribble_mean",
        "policy_skill_prob_shoot_mean",
        "policy_skill_entropy",
        "policy_param_std_mean",
    ):
        print(f"  {key}={float(summary.get(key, 0.0)):.3f}", flush=True)
    print("[health] behavior", flush=True)
    for key in (
        "takeover_success_rate",
        "clean_takeover_rate",
        "clean_finish_rate",
        "incompatible_takeover_rate",
        "walk_in_stable_takeover_rate",
        "stand_in_stable_takeover_rate",
        "policy_incompatible_takeover_rate",
        "policy_walk_in_stable_takeover_rate",
        "policy_stand_in_stable_takeover_rate",
        "dribble_compat_rate",
        "shoot_compat_rate",
        "bad_dribble_rate",
        "bad_walk_rate",
        "handoff_override_rate",
        "target_repair_rate",
        "policy_target_error_mean",
        "policy_theta_error_mean",
        "pose_jump_rate",
        "pose_jump_max_mm",
        "stability_score",
        "ball_trusted_rate",
    ):
        print(f"  {key}={float(summary.get(key, 0.0)):.3f}", flush=True)
    if "live_steps" in summary:
        print("[health] live rollout", flush=True)
        for key in (
            "live_target_repair_rate",
            "live_handoff_override_rate",
            "live_policy_target_error_mean",
            "live_policy_theta_error_mean",
            "live_pose_jump_rate",
            "live_pose_jump_max_mm",
            "live_policy_walk_rate",
            "live_policy_dribble_rate",
            "live_policy_shoot_rate",
            "live_corrected_exported_robot_pose_rate",
        ):
            print(f"  {key}={float(summary.get(key, 0.0)):.3f}", flush=True)


def main() -> None:
    # Edit these values directly from the IDE. No argparse on purpose.
    CHECKPOINT_PATH: Path | None = None
    # Exampl
    # CHECKPOINT_PATH = Path("runs/pybh_mappo_takeover_ppo/checkpoint_stage0_step50000.pt")

    EXPERIMENT_NAME = "pybh_mappo_takeover_ppo"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    STAGE_ID: int | None = None  # None means use checkpoint metadata.
    SEED = 42

    LIVE_STEPS = 220
    LIVE_SLEEP_SECONDS = 0.06
    METRIC_EPISODES = 2

    BACKEND = "simrobot_visible"
    OBS_MODE = "corrected"
    SIMROBOT_PROFILE = "3d_visible"
    SIMROBOT_LAUNCH = True
    MAX_EPISODE_STEPS = 220

    checkpoint_path = _latest_checkpoint(EXPERIMENT_NAME) if CHECKPOINT_PATH is None else _resolve_repo_path(CHECKPOINT_PATH)
    checkpoint = _load_checkpoint_metadata(checkpoint_path, DEVICE)
    stage_id = int(checkpoint.get("stage_id", 0) if STAGE_ID is None else STAGE_ID)

    session_dir = (
        Path(__file__).resolve().parent
        / "runs"
        / "model_evaluator"
        / datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    live_steps_csv = session_dir / "live_steps.csv"
    metric_episodes_csv = session_dir / "metric_episodes.csv"
    summary_csv = session_dir / "summary.csv"

    backend_cfg = replace(
        ENV.backend,
        mode=BACKEND,
        obs_mode=OBS_MODE,
        simrobot_profile=SIMROBOT_PROFILE,
        simrobot_launch=SIMROBOT_LAUNCH,
    )
    effective_env = replace(
        ENV,
        max_episode_steps=MAX_EPISODE_STEPS,
        parallel_player_numbers=(1,),
        backend=backend_cfg,
    )
    effective_train = replace(TRAIN, experiment_name=EXPERIMENT_NAME)

    original_train = train.TRAIN
    original_env = train.ENV
    train.TRAIN = effective_train
    train.ENV = effective_env
    env = None
    try:
        env_cfg = make_env_config(stage_id, backend_cfg)
        env_cfg = replace(env_cfg, max_episode_steps=MAX_EPISODE_STEPS, parallel_player_numbers=(1,))
        env, wrapped_env, eval_env = build_env(env_cfg)
        agent, models = build_mappo(wrapped_env, DEVICE, effective_train, env_cfg.active_skill_names)
        models["policy"].load_state_dict(checkpoint["policy_state_dict"])
        if "value_state_dict" in checkpoint:
            models["value"].load_state_dict(checkpoint["value_state_dict"])

        print("[model-evaluator]", flush=True)
        print(f"  checkpoint={checkpoint_path}", flush=True)
        print(f"  checkpoint_stage={checkpoint.get('stage_id')} eval_stage={stage_id}", flush=True)
        print(f"  device={DEVICE} backend={BACKEND} profile={SIMROBOT_PROFILE}", flush=True)
        print(f"  active_skills={env_cfg.active_skill_names}", flush=True)
        print(f"  telemetry={session_dir}", flush=True)
        print("  If a SimRobot window opens, press F5/play if it is paused.", flush=True)

        live_result = _live_rollout(
            agent,
            eval_env,
            device=DEVICE,
            seed=SEED,
            max_steps=LIVE_STEPS,
            sleep_s=LIVE_SLEEP_SECONDS,
            csv_path=live_steps_csv,
        )
        summary = _collect_metric_episodes(
            agent,
            eval_env,
            device=DEVICE,
            episodes=METRIC_EPISODES,
            seed=SEED + 1000,
            csv_path=metric_episodes_csv,
        )
        summary["checkpoint_stage_id"] = float(checkpoint.get("stage_id", stage_id))
        summary["eval_stage_id"] = float(stage_id)
        summary["live_total_reward"] = float(live_result["total_reward"])
        live_metrics = live_result.get("metrics", {})
        if isinstance(live_metrics, Mapping):
            summary.update(live_metrics)
        summary["checkpoint_path"] = str(checkpoint_path)
        _append_csv_row(summary_csv, summary)
        _print_health(summary)
        print(f"\n[done] CSV written to {session_dir}", flush=True)
    finally:
        if env is not None:
            env.close()
        train.TRAIN = original_train
        train.ENV = original_env


if __name__ == "__main__":
    main()
