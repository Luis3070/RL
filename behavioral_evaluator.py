from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

import train
from config import ENV, TRAIN
from curriculum import get_stage_scene_path
from model_evaluator import _load_checkpoint_metadata, _policy_payloads
from production_curriculum import production_scenarios
from pybh_backend import WorldSpec
from train import _aggregate_eval_metrics, build_env, build_mappo, make_env_config


ROOT = Path(__file__).resolve().parent
#DEFAULT_CHECKPOINT = ROOT / "runs/stage4_candidate/checkpoints/stage4_candidate_checkpoint_stage3_step55000.pt"
DEFAULT_CHECKPOINT = ROOT / "runs/production_readiness_v2/checkpoint_stage4_step30000.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/behavioral_evaluator"


@dataclass(frozen=True)
class Situation:
    name: str
    stage_id: int
    description: str
    world_fn: Callable[[np.random.Generator], WorldSpec]
    episodes: int
    pass_fn: Callable[[Mapping[str, float]], tuple[bool, str]]


def _append_csv_row(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def _pass_finish(summary: Mapping[str, float]) -> tuple[bool, str]:
    checks = {
        "clean_takeover_rate>=0.50": float(summary.get("clean_takeover_rate", 0.0)) >= 0.50,
        "clean_finish_rate>=0.60": float(summary.get("clean_finish_rate", 0.0)) >= 0.60,
        "shoot_finish_rate>=0.25": float(summary.get("shoot_finish_rate", 0.0)) >= 0.25,
        "goal_or_progress": (
            float(summary.get("goal_rate", 0.0)) >= 0.10
            or float(summary.get("ball_progress_post_takeover_mean", 0.0)) >= 0.015
        ),
        "collision_rate<=0.12": float(summary.get("collision_rate", 1.0)) <= 0.12,
        "policy_shoot_rate>=0.02": float(summary.get("policy_shoot_rate", 0.0)) >= 0.02,
    }
    return all(checks.values()), "; ".join(k for k, passed in checks.items() if not passed)


def _pass_obstacles(summary: Mapping[str, float]) -> tuple[bool, str]:
    checks = {
        "clean_episode_rate>=0.50": float(summary.get("clean_episode_rate", 0.0)) >= 0.50,
        "dynamic_obstacle_success>=0.20": float(summary.get("dynamic_obstacle_success", 0.0)) >= 0.20,
        "collision_rate<=0.12": float(summary.get("collision_rate", 1.0)) <= 0.12,
        "corridor_keep_rate>=0.50": float(summary.get("corridor_keep_rate", 0.0)) >= 0.50,
        "target_repair_rate<=0.08": float(summary.get("target_repair_rate", 1.0)) <= 0.08,
    }
    return all(checks.values()), "; ".join(k for k, passed in checks.items() if not passed)


def _pass_static_obstacles(summary: Mapping[str, float]) -> tuple[bool, str]:
    checks = {
        "clean_episode_rate>=0.50": float(summary.get("clean_episode_rate", 0.0)) >= 0.50,
        "obstacle_clearance_success>=0.25": float(summary.get("obstacle_clearance_success", 0.0)) >= 0.25,
        "collision_rate<=0.12": float(summary.get("collision_rate", 1.0)) <= 0.12,
        "target_repair_rate<=0.10": float(summary.get("target_repair_rate", 1.0)) <= 0.10,
        "shoot_finish_rate>=0.20": float(summary.get("shoot_finish_rate", 0.0)) >= 0.20,
    }
    return all(checks.values()), "; ".join(k for k, passed in checks.items() if not passed)


def _pass_preshoot(summary: Mapping[str, float]) -> tuple[bool, str]:
    checks = {
        "clean_takeover_rate>=0.50": float(summary.get("clean_takeover_rate", 0.0)) >= 0.50,
        "bad_shoot_rate<=0.25": float(summary.get("bad_shoot_rate", 1.0)) <= 0.25,
        "policy_shoot_rate<=0.35": float(summary.get("policy_shoot_rate", 1.0)) <= 0.35,
        "collision_rate<=0.12": float(summary.get("collision_rate", 1.0)) <= 0.12,
    }
    return all(checks.values()), "; ".join(k for k, passed in checks.items() if not passed)


# Pass criteria are evaluation-only; the world builders + episode counts are shared with the
# BC dataset collector via production_curriculum.production_scenarios (keyed by scenario name).
_PASS_FNS: dict[str, Callable[[Mapping[str, float]], tuple[bool, str]]] = {
    "finish_band_2p5_3m_clean": _pass_finish,
    "finish_band_2p5_3m_static_lane": _pass_static_obstacles,
    "preshoot_bad_angle_boundary": _pass_preshoot,
    "dynamic_full_field_pressure": _pass_obstacles,
    "blocked_corridor_repair": _pass_obstacles,
}


def _situations(episodes: int) -> list[Situation]:
    situations: list[Situation] = []
    for scenario in production_scenarios(episodes):
        situations.append(
            Situation(
                name=scenario.name,
                stage_id=scenario.stage_id,
                description=scenario.description,
                world_fn=scenario.world_fn,
                episodes=scenario.episodes,
                pass_fn=_PASS_FNS[scenario.name],
            )
        )
    return situations


def _collect_episode_with_world(
    eval_env,
    policy_fn,
    *,
    seed: int,
    stage_id: int,
    world_spec: WorldSpec,
    capture_steps: bool,
) -> dict:
    obs, infos = eval_env.reset(seed=seed, options={"stage_id": stage_id, "world_spec": world_spec})
    rewards = {agent: 0.0 for agent in eval_env.possible_agents}
    steps = 0
    final_info: Mapping[str, object] | None = None
    step_records: list[dict[str, object]] = []
    reward_terms_sum: dict[str, float] = {}
    reward_weighted_terms_sum: dict[str, float] = {}
    while eval_env.agents:
        actions = policy_fn(obs, infos)
        obs, step_rewards, terminateds, truncateds, infos = eval_env.step(actions)
        steps += 1
        info = infos.get(eval_env.possible_agents[0], {})
        final_info = info
        for agent, value in step_rewards.items():
            rewards[agent] = float(rewards.get(agent, 0.0) + float(value))
        for key, value in info.get("reward_terms", {}).items():
            reward_terms_sum[key] = float(reward_terms_sum.get(key, 0.0) + float(value))
        for key, value in info.get("reward_weighted_terms", {}).items():
            reward_weighted_terms_sum[key] = float(reward_weighted_terms_sum.get(key, 0.0) + float(value))
        if capture_steps:
            raw_obs = info.get("raw_obs", {})
            if not isinstance(raw_obs, Mapping):
                raw_obs = {}
            step_records.append(
                {
                    "step": steps,
                    "reward": float(step_rewards.get(eval_env.possible_agents[0], 0.0)),
                    "policy_skill_name": info.get("policy_skill_name"),
                    "executed_skill_name": info.get("executed_skill_name"),
                    "preferred_skill_name": info.get("preferred_skill_name"),
                    "policy_shoot_prob": float(info.get("policy_skill_prob_shoot", 0.0)),
                    "policy_dribble_prob": float(info.get("policy_skill_prob_dribble", 0.0)),
                    "policy_walk_prob": float(info.get("policy_skill_prob_walk", 0.0)),
                    "takeover_envelope": int(bool(info.get("takeover_envelope", False))),
                    "shoot_zone": int(bool(info.get("shoot_zone", False))),
                    "target_repair_applied": int(bool(info.get("target_repair_applied", False))),
                    "policy_target_error_mm": float(info.get("policy_target_error_mm", 0.0)),
                    "ball_trusted": int(bool(info.get("ball_trusted_mask", False))),
                    "ball_rescued": int(bool(info.get("ball_rescued_mask", False))),
                    "ball_confidence_score": float(info.get("ball_confidence_score", 0.0)),
                    "robot_x": float(raw_obs.get("robot_x", 0.0)),
                    "robot_y": float(raw_obs.get("robot_y", 0.0)),
                    "ball_x": float(raw_obs.get("ball_x", 0.0)),
                    "ball_y": float(raw_obs.get("ball_y", 0.0)),
                    "shot_opening_with_obstacles": float(raw_obs.get("shot_opening_with_obstacles", 0.0)),
                    "nearest_front_obstacle_dist": float(raw_obs.get("nearest_front_obstacle_dist", 0.0)),
                }
            )
        if not eval_env.agents or any(terminateds.values()) or any(truncateds.values()):
            break
    episode_metrics = dict(final_info.get("episode_metrics", {})) if final_info is not None else {}
    return {
        "steps": steps,
        "mean_reward": float(np.mean(list(rewards.values()))) if rewards else 0.0,
        "total_rewards": rewards,
        "episode_metrics": episode_metrics,
        "reward_terms_sum": reward_terms_sum,
        "reward_weighted_terms_sum": reward_weighted_terms_sum,
        "final_info": final_info or {},
        "step_records": step_records,
    }

def _load_state_dict_obs_padded(
    model: torch.nn.Module, saved: Mapping[str, torch.Tensor], *, label: str
) -> None:
    """Load a checkpoint whose first Linear layer may have fewer input features than
    the current model (the observation grew). Old input columns are copied verbatim;
    new columns are zeroed so day-0 behaviour matches the warm-start exactly and PPO
    later wires the new obs features in organically. Shape-matched params load as usual.
    """
    model_sd = model.state_dict()
    adapted: Dict[str, torch.Tensor] = {}
    padded: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key, tensor in saved.items():
        if key not in model_sd:
            continue
        target = model_sd[key]
        if tuple(tensor.shape) == tuple(target.shape):
            adapted[key] = tensor
        elif (
            tensor.dim() == 2
            and target.dim() == 2
            and tensor.shape[0] == target.shape[0]
            and tensor.shape[1] < target.shape[1]
        ):
            grown = torch.zeros_like(target)
            grown[:, : tensor.shape[1]] = tensor.to(device=target.device, dtype=target.dtype)
            adapted[key] = grown
            padded.append((key, tuple(tensor.shape), tuple(target.shape)))
        # else: shape we cannot adapt -> keep fresh init (surfaced as missing below)
    result = model.load_state_dict(adapted, strict=False)
    if padded:
        print(f"[warm-start] {label}: zero-padded grown obs columns -> {padded}")
    if result.missing_keys:
        print(f"[warm-start] {label}: kept fresh init for {list(result.missing_keys)}")

def _load_agent(checkpoint_path: Path, device: str, eval_stage_id: int, max_episode_steps: int):
    checkpoint = _load_checkpoint_metadata(checkpoint_path, device)
    backend_cfg = replace(
        ENV.backend,
        mode="simrobot_visible",
        obs_mode="corrected",
        simrobot_profile="3d_visible",
        simrobot_launch=True,
        simrobot_scene_path=get_stage_scene_path(eval_stage_id, "3d_visible"),
    )
    effective_env = replace(
        ENV,
        stage_id=eval_stage_id,
        max_episode_steps=max_episode_steps,
        parallel_player_numbers=(1,),
        backend=backend_cfg,
    )
    effective_train = replace(TRAIN, experiment_name="behavioral_evaluator")
    original_train = train.TRAIN
    original_env = train.ENV
    train.TRAIN = effective_train
    train.ENV = effective_env
    env_cfg = make_env_config(eval_stage_id, backend_cfg)
    env_cfg = replace(env_cfg, max_episode_steps=max_episode_steps, parallel_player_numbers=(1,))
    env, wrapped_env, eval_env = build_env(env_cfg)
    agent, models = build_mappo(wrapped_env, device, effective_train, env_cfg.active_skill_names)
    _load_state_dict_obs_padded(models["policy"], checkpoint["policy_state_dict"], label="policy")
    if "value_state_dict" in checkpoint:
        _load_state_dict_obs_padded(models["value"], checkpoint["value_state_dict"], label="value")
    return checkpoint, env, eval_env, agent, original_train, original_env


def run(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    session_dir = output_root.resolve() / datetime.now().strftime("behavior_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = session_dir / "summary.csv"
    episodes_csv = session_dir / "episodes.csv"
    steps_csv = session_dir / "steps.csv"
    (session_dir / "run_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "output_dir": str(session_dir),
                "episodes": int(args.episodes),
                "seed": int(args.seed),
                "device": device,
                "max_episode_steps": int(args.max_episode_steps),
                "capture_steps": bool(args.capture_steps),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    checkpoint, env, eval_env, agent, original_train, original_env = _load_agent(
        checkpoint_path,
        device,
        4,
        int(args.max_episode_steps),
    )
    rng = np.random.default_rng(int(args.seed))

    def policy_fn(obs, infos):
        del infos
        return _policy_payloads(agent, eval_env, obs, device=device, timestep=0, timesteps=1)

    previous_mode = getattr(agent, "training", None)
    agent.enable_training_mode(False)
    overall_pass = True
    try:
        print("[behavioral-evaluator]", flush=True)
        print(f"  checkpoint={checkpoint_path}", flush=True)
        print(f"  checkpoint_stage={checkpoint.get('stage_id')} eval_stage=4", flush=True)
        print(f"  device={device}", flush=True)
        print(f"  telemetry={session_dir}", flush=True)
        for situation in _situations(int(args.episodes)):
            results = []
            print(f"\n[situation] {situation.name}: {situation.description}", flush=True)
            for episode_index in range(situation.episodes):
                world = situation.world_fn(rng)
                result = _collect_episode_with_world(
                    eval_env,
                    policy_fn,
                    seed=int(args.seed) + 1000 * len(results) + episode_index,
                    stage_id=situation.stage_id,
                    world_spec=world,
                    capture_steps=bool(args.capture_steps),
                )
                results.append(result)
                metrics = result.get("episode_metrics", {})
                row = {
                    "situation": situation.name,
                    "episode_index": episode_index,
                    "steps": result.get("steps", 0),
                    "mean_reward": result.get("mean_reward", 0.0),
                    **{f"episode_metrics_{k}": v for k, v in metrics.items()},
                }
                _append_csv_row(episodes_csv, row)
                for step in result.get("step_records", []):
                    _append_csv_row(steps_csv, {"situation": situation.name, "episode_index": episode_index, **step})
            summary = _aggregate_eval_metrics(results)
            passed, failures = situation.pass_fn(summary)
            overall_pass = overall_pass and passed
            row = {
                "situation": situation.name,
                "stage_id": situation.stage_id,
                "episodes": situation.episodes,
                "passed": int(passed),
                "failures": failures,
                "description": situation.description,
                **summary,
            }
            _append_csv_row(summary_csv, row)
            print(
                "[summary] "
                f"{situation.name} pass={int(passed)} "
                f"reward={float(summary.get('episode_reward_mean', 0.0)):.3f} "
                f"clean_ep={float(summary.get('clean_episode_rate', 0.0)):.3f} "
                f"finish={float(summary.get('clean_finish_rate', 0.0)):.3f} "
                f"shoot_finish={float(summary.get('shoot_finish_rate', 0.0)):.3f} "
                f"goal={float(summary.get('goal_rate', 0.0)):.3f} "
                f"collision={float(summary.get('collision_rate', 0.0)):.3f} "
                f"dynamic={float(summary.get('dynamic_obstacle_success', 0.0)):.3f} "
                f"policy_shoot={float(summary.get('policy_shoot_rate', 0.0)):.3f} "
                f"failures={failures or '-'}",
                flush=True,
            )
    finally:
        if previous_mode is not None:
            agent.enable_training_mode(bool(previous_mode))
        env.close()
        train.TRAIN = original_train
        train.ENV = original_env
    print(f"\n[done] overall_pass={int(overall_pass)} summary={summary_csv}", flush=True)
    return 0 if overall_pass else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on hand-picked soccer behavior situations.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--capture-steps", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(run(args))
