from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from rlfootball.training import train


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET = ROOT / "runs/fine_tuning_production_dataset/dataset_20260527_220501_production_merged/samples.npz"
SOURCE_SUMMARY = SOURCE_DATASET.with_name("summary.json")
STAGE4_REPLAY_DIR = ROOT / "runs/fine_tuning_production_dataset/production_replay_stage4"
STAGE4_REPLAY_DATASET = STAGE4_REPLAY_DIR / "samples.npz"
DEFAULT_CHECKPOINT_CANDIDATES = (
    ROOT / "runs/production_finetune_v1/checkpoint_stage4_step35000.pt",
    ROOT / "runs/production_finetune_v1/checkpoint_stage4_step25000.pt",
    ROOT / "runs/production_finetune_v1/checkpoint_stage4_step20000.pt",
    ROOT / "runs/stage4_candidate/checkpoints/stage4_candidate_checkpoint_stage3_step55000.pt",
)
DEFAULT_SCENARIO_WEIGHTS = (
    # dynamic_full_field_pressure dropped to weight 0: behavioral eval converted 0/8
    # at v1-step-35000 and the full-field reset is the most expensive episode in the
    # set. The near-finish dynamic variant (dynamic_finish_pressure_topup) keeps the
    # obstacle-avoidance pressure without the dilution.
    # Re-focus after evaluator mismatch: topups looked clean in-loop, but the
    # canonical behavioral gate still failed on clean/static/blocked seeds because
    # the policy over-walked in stable takeover. Weight the canonical scenarios
    # higher and keep topups as support instead of the dominant distribution.
    "finish_band_2p5_3m_clean:8,"
    "finish_band_2p5_3m_static_lane:6,"
    "preshoot_bad_angle_boundary:3,"
    "dynamic_full_field_pressure:0,"
    "blocked_corridor_repair:5,"
    "finish_near_arm_clean_topup:2,"
    "finish_bridge_clean_topup:3,"
    "finish_bridge_static_topup:3,"
    "blocked_corridor_finish_topup:3,"
    "dynamic_finish_pressure_topup:2"
)


def _env_int(name: str, fallback: int) -> int:
    value = os.getenv(name)
    return fallback if value is None else int(value)


def _env_float(name: str, fallback: float) -> float:
    value = os.getenv(name)
    return fallback if value is None else float(value)


def _env_bool(name: str, fallback: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_players(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _default_checkpoint() -> Path:
    for path in DEFAULT_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_CHECKPOINT_CANDIDATES[0]


def _load_source_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def prepare_stage4_replay_dataset(source_dataset: Path, output_dataset: Path) -> Path:
    """Normalize production BC labels to stage 4 so train.py replays all scenarios.

    BCReplayRegularizer samples by current stage. A stage-4 fine-tune would otherwise
    ignore the stage-1 clean-band and stage-2 static-lane rows, exactly the behavior we
    most need to repair. Active skills are identical across stages 1/2/4, and the gate
    bits still own the forbid-invalid mask, so normalizing `stages` to 4 is the right
    production-replay view of this dataset.
    """
    source_dataset = _resolve_repo_path(source_dataset)
    output_dataset = _resolve_repo_path(output_dataset)
    if not source_dataset.exists():
        raise FileNotFoundError(source_dataset)

    data = np.load(source_dataset, allow_pickle=True)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    weights = data["weights"].astype(np.float32)
    if observations.ndim != 2 or observations.shape[1] < 26:
        raise ValueError(f"Expected observations [N,M] with M>=26, got {observations.shape}")
    if actions.ndim != 2 or actions.shape[1] != 5:
        raise ValueError(f"Expected actions [N,5], got {actions.shape}")
    if len(weights) != len(observations):
        raise ValueError("weights length does not match observations")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all() or not np.isfinite(weights).all():
        raise ValueError("Production replay dataset contains non-finite values")

    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dataset,
        observations=observations,
        actions=actions,
        stages=np.full((len(observations),), 4, dtype=np.int16),
        episode_ids=data["episode_ids"].astype(np.int32),
        step_ids=data["step_ids"].astype(np.int16),
        weights=weights,
        skill_order=data["skill_order"],
    )

    summary = _load_source_summary(source_dataset.with_name("summary.json"))
    replay_summary = {
        "source_dataset": str(source_dataset),
        "output_dataset": str(output_dataset),
        "reason": "Stage-normalized production replay view for stage-4 fine-tuning.",
        "observation_shape": list(observations.shape),
        "action_shape": list(actions.shape),
        "stage_values": [4],
        "positive_samples": int(np.count_nonzero(weights > 0.0)),
        "total_samples": int(len(weights)),
        "accepted_rate": float(np.count_nonzero(weights > 0.0) / max(1, len(weights))),
        "source_summary": {
            key: summary.get(key)
            for key in (
                "sanitized",
                "positive_weight_gate_label_consistent",
                "band_2p5_3m_samples",
                "global_skill_counts",
                "sanitization",
            )
            if key in summary
        },
    }
    (output_dataset.parent / "summary.json").write_text(
        json.dumps(replay_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[production-replay] prepared={output_dataset} "
        f"positive={replay_summary['positive_samples']}/{replay_summary['total_samples']}",
        flush=True,
    )
    return output_dataset


def _build_run_cfg(args: argparse.Namespace, replay_dataset: Path) -> train.ScriptRunConfig:
    experiment_name = os.getenv("RL_PROD_EXPERIMENT_NAME", "production_readiness_v2")
    checkpoint = _resolve_repo_path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return train.ScriptRunConfig(
        mode="train",
        backend="simrobot_visible",
        obs_mode="corrected",
        simrobot_profile="3d_visible",
        simrobot_launch=False,
        embedded_3d_preflight=False,
        learner_device=os.getenv("RL_LEARNER_DEVICE", "cuda:0"),
        stage_id=4,
        max_stage_id=4,
        train_steps=_env_int("RL_PROD_TRAIN_STEPS", 80_000),
        # Production finish chains can spend >500 steps before a valid shoot; 400
        # caused eval episodes to time out before goals even after clean_finish.
        max_episode_steps=_env_int("RL_PROD_MAX_EPISODE_STEPS", 1000),
        rollout_steps=_env_int("RL_PROD_ROLLOUT_STEPS", 256),
        learning_epochs=_env_int("RL_PROD_LEARNING_EPOCHS", 4),
        mini_batches=_env_int("RL_PROD_MINI_BATCHES", 4),
        eval_interval=_env_int("RL_PROD_EVAL_INTERVAL", 5_000),
        checkpoint_interval=_env_int("RL_PROD_CHECKPOINT_INTERVAL", 5_000),
        # 24 visible-SimRobot eval episodes was the single largest wallclock sink
        # between checkpoints. 8 (= train.py default) keeps signal but cuts eval ~3x.
        eval_episodes=_env_int("RL_PROD_EVAL_EPISODES", 8),
        save_telemetry=True,
        save_plots=True,
        capture_eval_steps=_env_bool("RL_PROD_CAPTURE_EVAL_STEPS", False),
        pack_parallel_eval=_env_bool("RL_PROD_PACK_PARALLEL_EVAL", True),
        abort_on_wire_fail=False,
        bc_warm_start=True,
        bc_checkpoint_path=str(checkpoint),
        bc_replay=True,
        bc_replay_dataset_path=str(replay_dataset),
        bc_replay_steps=_env_int("RL_PROD_BC_REPLAY_STEPS", train.IDE_RUN.bc_replay_steps),
        bc_replay_weight=_env_float("RL_PROD_BC_REPLAY_WEIGHT", 0.10),
        bc_replay_batch_size=_env_int("RL_PROD_BC_REPLAY_BATCH_SIZE", train.IDE_RUN.bc_replay_batch_size),
        # 4 extra distillation passes per PPO update with batch=128 was redundant once
        # the BC competence saturated; 2 keeps the shoot-row signal without burning compute.
        bc_replay_updates=_env_int("RL_PROD_BC_REPLAY_UPDATES", 2),
        bc_replay_learning_rate=_env_float("RL_PROD_BC_REPLAY_LR", train.IDE_RUN.bc_replay_learning_rate),
        # Stronger BC pull and slower decay: v1 hit bc_beta=0.011 by step 5k because
        # competence ramped fast, effectively turning BC off before the agent ever
        # learned to commit to shoot. Peak higher, decay slower, keep the shoot signal alive.
        bc_beta_max=_env_float("RL_PROD_BC_BETA_MAX", 0.15),
        bc_beta_min=_env_float("RL_PROD_BC_BETA_MIN", train.IDE_RUN.bc_beta_min),
        bc_competence_half_life_evals=_env_float("RL_PROD_BC_HALF_LIFE_EVALS", 10.0),
        bc_skill_loss_scale=_env_float("RL_PROD_BC_SKILL_LOSS_SCALE", 0.40),
        bc_param_loss_scale=_env_float("RL_PROD_BC_PARAM_LOSS_SCALE", 0.50),
        diverse_stage_resets=True,
        stage0_finish_interleave_probability=0.0,
        stage1_finish_interleave_probability=0.0,
        stage2_finish_interleave_probability=0.0,
        stage3_finish_interleave_probability=0.0,
        post_takeover_progress_enabled=True,
        post_takeover_progress_window_steps=120,
        telemetry_tag=os.getenv("RL_PROD_TELEMETRY_TAG", "production_finetune"),
        experiment_name=experiment_name,
        # 4 visible-SimRobot slots is the sweet spot on 32 logical CPUs / 24 render
        # threads. 6 slots saturate Qt+OpenGL and made p=6 timeout-prone in v2.
        parallel_player_numbers=_parse_players(os.getenv("RL_PROD_PLAYERS", "1,2,3,4")),
        visible_parallel_display_base=_env_int("RL_PROD_VISIBLE_DISPLAY_BASE", 430),
    )


def run(args: argparse.Namespace) -> None:
    replay_dataset = prepare_stage4_replay_dataset(Path(args.dataset), Path(args.output_dataset))
    if args.prepare_only:
        return

    # Stage fixed at 4: this is a production repair fine-tune, not a promotion climb.
    train.TRAIN = replace(train.TRAIN, auto_promote=False)
    os.environ.setdefault("RL_PRODUCTION_SCENARIO_RESETS", "1")
    os.environ.setdefault("RL_PRODUCTION_SCENARIO_SET", os.getenv("RL_PROD_SCENARIO_SET", "all"))
    os.environ.setdefault("RL_PRODUCTION_SCENARIO_WEIGHTS", os.getenv("RL_PROD_SCENARIO_WEIGHTS", DEFAULT_SCENARIO_WEIGHTS))
    # Healthy frames return in <50 ms; the 1500ms only deepened stalls without preventing them.
    os.environ.setdefault("RL_VISIBLE_STEP_OBS_TIMEOUT_MS", os.getenv("RL_PROD_VISIBLE_STEP_OBS_TIMEOUT_MS", "1000"))
    # Roll back the inflated reset budgets: v2's slot-6 wedge proved the longer budget
    # didn't prevent timeouts, it just multiplied wallclock per failure. Fall back to the
    # original 4s/5/3 limits; can be overridden per-run via RL_VISIBLE_* env vars.
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS", "4000")
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_MAX_REPLACES", "5")
    os.environ.setdefault("RL_VISIBLE_RESET_MAX_ATTEMPTS", "3")
    if os.getenv("RL_PROD_VISIBLE_OBS_TIMEOUT_MS") is not None:
        backend_cfg = replace(
            train.ENV.backend,
            obs_timeout_ms=_env_int("RL_PROD_VISIBLE_OBS_TIMEOUT_MS", train.ENV.backend.obs_timeout_ms),
        )
        train.ENV = replace(train.ENV, backend=backend_cfg)
    run_cfg = _build_run_cfg(args, replay_dataset)
    print(
        "[production-train] "
        f"checkpoint={Path(args.checkpoint).expanduser().resolve()} "
        f"dataset={replay_dataset} "
        f"experiment={run_cfg.experiment_name} "
        f"steps={run_cfg.train_steps} players={run_cfg.parallel_player_numbers} "
        f"production_resets={os.getenv('RL_PRODUCTION_SCENARIO_SET')} "
        f"eval_episodes={run_cfg.eval_episodes} max_episode_steps={run_cfg.max_episode_steps} "
        f"capture_eval_steps={int(run_cfg.capture_eval_steps)} "
        f"bc[beta={run_cfg.bc_beta_min:.3f}..{run_cfg.bc_beta_max:.3f},half_life={run_cfg.bc_competence_half_life_evals:g},"
        f"replay_w={run_cfg.bc_replay_weight:.3f},updates={run_cfg.bc_replay_updates},"
        f"skill={run_cfg.bc_skill_loss_scale:.2f},param={run_cfg.bc_param_loss_scale:.2f}] "
        f"reset_budgets[settle={os.getenv('RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS')}ms,"
        f"replaces={os.getenv('RL_VISIBLE_BALL_SETTLE_MAX_REPLACES')},"
        f"attempts={os.getenv('RL_VISIBLE_RESET_MAX_ATTEMPTS')},"
        f"step_obs={os.getenv('RL_VISIBLE_STEP_OBS_TIMEOUT_MS')}ms]",
        flush=True,
    )
    train.main(run_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-4 production fine-tune entrypoint.")
    parser.add_argument("--dataset", default=str(SOURCE_DATASET))
    parser.add_argument("--output-dataset", default=str(STAGE4_REPLAY_DATASET))
    parser.add_argument("--checkpoint", default=str(_default_checkpoint()))
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
