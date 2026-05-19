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
    print("[live] reset complete. Watch SimRobot now.", flush=True)
    try:
        for step in range(1, max_steps + 1):
            payloads = _policy_payloads(agent, eval_env, obs, device=device, timestep=step, timesteps=max_steps)
            obs, rewards, _terminated, _truncated, infos = eval_env.step(payloads)
            agent_id = eval_env.possible_agents[0]
            reward = float(rewards.get(agent_id, 0.0))
            total_reward += reward
            info = infos.get(agent_id, {})
            final_info = info
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
                "target_x": float(info.get("policy_target_x", 0.0)),
                "target_y": float(info.get("policy_target_y", 0.0)),
                "target_theta": float(info.get("policy_target_theta", 0.0)),
                "policy_skill_prob_stand": float(info.get("policy_skill_prob_stand", 0.0)),
                "policy_skill_prob_walk": float(info.get("policy_skill_prob_walk", 0.0)),
                "policy_skill_prob_dribble": float(info.get("policy_skill_prob_dribble", 0.0)),
                "policy_skill_prob_shoot": float(info.get("policy_skill_prob_shoot", 0.0)),
                "ball_trusted": int(bool(info.get("ball_trusted_mask", False))),
                "ball_visible": int(bool(info.get("ball_visible_mask", False))),
                "ball_confidence_score": float(info.get("ball_confidence_score", 0.0)),
                "robot_x": float(info.get("raw_obs", {}).get("robot_x", 0.0)),
                "robot_y": float(info.get("raw_obs", {}).get("robot_y", 0.0)),
                "ball_x": float(info.get("raw_obs", {}).get("ball_x", 0.0)),
                "ball_y": float(info.get("raw_obs", {}).get("ball_y", 0.0)),
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
                f"override={int(bool(info.get('handoff_override_applied', False)))} "
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
    return {"total_reward": total_reward, "final_info": dict(final_info)}


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
        "dribble_compat_rate",
        "shoot_compat_rate",
        "bad_dribble_rate",
        "bad_walk_rate",
        "handoff_override_rate",
        "stability_score",
        "ball_trusted_rate",
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
