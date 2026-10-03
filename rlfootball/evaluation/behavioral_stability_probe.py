from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from rlfootball.evaluation.behavioral_evaluator import (
    DEFAULT_CHECKPOINT,
    ROOT,
    _append_csv_row,
    _collect_episode_with_world,
    _load_agent,
    _situations,
)
from rlfootball.evaluation.behavioral_stability import aggregate_rep_rows, parse_scenario_list, parse_seed_list
from rlfootball.evaluation.model_evaluator import _policy_payloads
from rlfootball.training.train import _aggregate_eval_metrics

DEFAULT_OUTPUT_ROOT = ROOT / "runs/behavioral_stability"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run behavioral_evaluator scenarios across a fixed stable seed set and report pass-rate stability."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--pass-rate-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--capture-steps", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    session_dir = output_root.resolve() / datetime.now().strftime("stability_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)

    seed_list = parse_seed_list(args.seed_list)
    scenario_filter = set(parse_scenario_list(args.scenarios))
    situations = [s for s in _situations(int(args.episodes)) if not scenario_filter or s.name in scenario_filter]
    if not situations:
        raise ValueError("No situations selected")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    per_rep_csv = session_dir / "per_rep_summary.csv"
    per_episode_csv = session_dir / "per_episode.csv"
    stability_csv = session_dir / "stability_summary.csv"
    steps_csv = session_dir / "steps.csv"
    (session_dir / "run_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "output_dir": str(session_dir),
                "episodes": int(args.episodes),
                "seed_list": list(seed_list),
                "scenarios": [s.name for s in situations],
                "pass_rate_threshold": float(args.pass_rate_threshold),
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

    def policy_fn(obs, infos):
        del infos
        return _policy_payloads(agent, eval_env, obs, device=device, timestep=0, timesteps=1)

    previous_mode = getattr(agent, "training", None)
    agent.enable_training_mode(False)
    overall_stable = True
    try:
        print("[behavioral-stability]", flush=True)
        print(f"  checkpoint={checkpoint_path}", flush=True)
        print(f"  checkpoint_stage={checkpoint.get('stage_id')} eval_stage=4", flush=True)
        print(f"  stable_seeds={seed_list}", flush=True)
        print(f"  scenarios={[s.name for s in situations]}", flush=True)
        print(f"  telemetry={session_dir}", flush=True)

        all_rep_rows: list[dict[str, object]] = []
        for rep_index, stable_seed in enumerate(seed_list):
            print(f"\n[rep {rep_index + 1}/{len(seed_list)}] stable_seed={stable_seed}", flush=True)
            for situation_index, situation in enumerate(situations):
                results = []
                for episode_index in range(situation.episodes):
                    world_seed = int(stable_seed) * 100_000 + situation_index * 1_000 + episode_index
                    reset_seed = int(stable_seed) * 100_000 + 50_000 + situation_index * 1_000 + episode_index
                    world_rng = np.random.default_rng(world_seed)
                    world = situation.world_fn(world_rng)
                    result = _collect_episode_with_world(
                        eval_env,
                        policy_fn,
                        seed=reset_seed,
                        stage_id=situation.stage_id,
                        world_spec=world,
                        capture_steps=bool(args.capture_steps),
                    )
                    results.append(result)
                    episode_row = {
                        "stable_seed": stable_seed,
                        "rep_index": rep_index,
                        "situation": situation.name,
                        "stage_id": situation.stage_id,
                        "episode_index": episode_index,
                        "world_seed": world_seed,
                        "reset_seed": reset_seed,
                        "steps": result.get("steps", 0),
                        "mean_reward": result.get("mean_reward", 0.0),
                        "world_json": json.dumps(asdict(world), sort_keys=True),
                        **{f"episode_metrics_{k}": v for k, v in result.get("episode_metrics", {}).items()},
                    }
                    _append_csv_row(per_episode_csv, episode_row)
                    for step in result.get("step_records", []):
                        _append_csv_row(
                            steps_csv,
                            {
                                "stable_seed": stable_seed,
                                "rep_index": rep_index,
                                "situation": situation.name,
                                "episode_index": episode_index,
                                **step,
                            },
                        )

                summary = _aggregate_eval_metrics(results)
                passed, failures = situation.pass_fn(summary)
                rep_row = {
                    "stable_seed": stable_seed,
                    "rep_index": rep_index,
                    "situation": situation.name,
                    "stage_id": situation.stage_id,
                    "episodes": situation.episodes,
                    "passed": int(passed),
                    "failures": failures,
                    "description": situation.description,
                    **summary,
                }
                all_rep_rows.append(rep_row)
                _append_csv_row(per_rep_csv, rep_row)
                print(
                    "[rep-summary] "
                    f"{situation.name} pass={int(passed)} "
                    f"clean_ep={float(summary.get('clean_episode_rate', 0.0)):.3f} "
                    f"finish={float(summary.get('clean_finish_rate', 0.0)):.3f} "
                    f"shoot_finish={float(summary.get('shoot_finish_rate', 0.0)):.3f} "
                    f"collision={float(summary.get('collision_rate', 0.0)):.3f} "
                    f"dynamic={float(summary.get('dynamic_obstacle_success', 0.0)):.3f} "
                    f"corridor={float(summary.get('corridor_keep_rate', 0.0)):.3f} "
                    f"policy_shoot={float(summary.get('policy_shoot_rate', 0.0)):.3f} "
                    f"failures={failures or '-'}",
                    flush=True,
                )

        print("\n[stability-summary]", flush=True)
        for situation in situations:
            rows = [row for row in all_rep_rows if row["situation"] == situation.name]
            aggregate = aggregate_rep_rows(
                rows,
                pass_rate_threshold=float(args.pass_rate_threshold),
            )
            stable_passed = bool(aggregate["stable_passed"])
            overall_stable = overall_stable and stable_passed
            row = {
                "situation": situation.name,
                "stage_id": situation.stage_id,
                "description": situation.description,
                **aggregate,
            }
            _append_csv_row(stability_csv, row)
            print(
                f"  {situation.name}: stable_pass={int(stable_passed)} "
                f"pass_rate={float(aggregate['pass_rate']):.3f} "
                f"collision_mean={float(aggregate['collision_rate_mean']):.3f} "
                f"collision_max={float(aggregate['collision_rate_max']):.3f} "
                f"top_failures={aggregate['top_failures'] or '-'}",
                flush=True,
            )
    finally:
        if previous_mode is not None:
            agent.enable_training_mode(bool(previous_mode))
        env.close()
        from rlfootball.training import train

        train.TRAIN = original_train
        train.ENV = original_env

    print(f"\n[done] overall_stable={int(overall_stable)} summary={stability_csv}", flush=True)
    return 0 if overall_stable else 2


if __name__ == "__main__":
    raise SystemExit(run(_parse_args()))
