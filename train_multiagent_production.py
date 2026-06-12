"""Multi-agent production fine-tune entrypoint — sibling of `train_production.py`.

Mirrors train_production.py's shape (argparse, defaults, _build_run_cfg, run)
but configures the §14.19 multi-agent retunings and delegates to
train_multiagent.main (not train.main). train.py and train_production.py stay
untouched in case the single-agent PPO needs fine-tuning or improvements.

§14.19 retunings applied here by default:
  * bc_beta_max = 0.08             (softer than production's 0.15)
  * bc_beta_min = 0.02
  * bc_competence_half_life_evals = 25  (longer than production's 10)
  * bc_replay_weight = 0.10
  * bc_replay_updates = 2
  * bc_skill_loss_scale = 0.40
  * bc_param_loss_scale = 0.50
  * eval_episodes = 8
  * max_episode_steps = 400
  * parallel_player_numbers = (1, 2, 3) for stage 6, (1, 2, 3, 4) for stage 7
  * KL anchor = `runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt`
  * Role-stratified BC sampling + min-across-roles competence via train_multiagent
    monkey-patches (no source change to train.py)
  * Dataset = latest under `runs/teacher_multiagent_dataset/` (or RL_MA_BC_DATASET)

Stage 6 (2v0 + BH GK) is the warmup; promote to stage 7 (3v3 vs full BH) per
§14.21 LOCKED scenario weights for the Monday 2026-06-15 demo.

Run:
  source /home/luis/anaconda3/etc/profile.d/conda.sh && conda activate RLNAO \\
    && cd /home/luis/GitHub/RL && PYTHONPATH=/home/luis/GitHub/RL \\
       python train_multiagent_production.py --stage-id 6

  # Then, after stage 6 promotes, run stage 7:
  python train_multiagent_production.py --stage-id 7 \\
    --checkpoint runs/<stage6-best>/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import train
import train_multiagent


ROOT = Path(__file__).resolve().parent

# §11.B.5.3 KL anchor — the 4/5 stabilized single-agent striker. This is the
# canonical warm-start; do NOT swap for an untested checkpoint.
DEFAULT_KL_ANCHOR_CANDIDATES = (
    ROOT / "runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt",
    ROOT / "runs/production_stability_repair_v1/checkpoint_stage4_step7500.pt",
    ROOT / "runs/production_stability_repair_v1/checkpoint_stage4_step10000.pt",
    ROOT / "runs/production_stability_repair_v1/checkpoint_stage4_step2500.pt",
)

# Multi-agent BC dataset produced by teacher_multiagent_dataset.py.
# Picks the most recent dataset under runs/teacher_multiagent_dataset/ unless
# overridden via --dataset or RL_MA_BC_DATASET.
DEFAULT_DATASET_ROOT = ROOT / "runs/teacher_multiagent_dataset"

# §14.21 LOCKED scenario weights for Monday 2026-06-15 demo. Stage 6 totals 4;
# stage 7 totals ~7 (0.5+1.5+2+3, rounded to integer weighting per env var format).
# Note: `team_3v3_full_dynamic:3` is non-negotiable — this is the only scenario
# with the exact perceptual conditions of the Monday demo.
DEFAULT_STAGE6_SCENARIO_WEIGHTS = (
    "team_2v0_clean_band:3,"
    "team_2v0_static_blocker:1"
)
DEFAULT_STAGE7_SCENARIO_WEIGHTS = (
    "team_3v0_clean:1,"
    "team_3v0_static_pinch:2,"
    "team_3v0_dynamic:2,"
    "team_3v3_full_dynamic:3"
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


def _default_kl_anchor() -> Path:
    for path in DEFAULT_KL_ANCHOR_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_KL_ANCHOR_CANDIDATES[0]


def _default_dataset() -> Path:
    """Latest dataset under DEFAULT_DATASET_ROOT/dataset_stage{6,7}_<ts>/samples.npz.

    Picks the most recently created directory; warns if both stage 6 and stage 7
    datasets exist — the trainer expects ONE combined dataset for the current
    stage, so use merge_production_datasets.py beforehand to combine them.
    """
    if not DEFAULT_DATASET_ROOT.exists():
        return DEFAULT_DATASET_ROOT / "MISSING_DATASET" / "samples.npz"
    candidates = sorted(
        (p for p in DEFAULT_DATASET_ROOT.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        samples = candidate / "samples.npz"
        if samples.exists():
            return samples
    return DEFAULT_DATASET_ROOT / "EMPTY" / "samples.npz"


def _default_players_for_stage(stage_id: int) -> str:
    """Parallel field-player count per stage (RL field players + 1 BH GK).

    Stage 6: 2 RL field (+ player 3 = BH GK).
    Stage 7: 3 RL field (+ player 4 = BH GK).
    parallel_player_numbers controls how many RealSimRobotEnv slots run in
    parallel — keeping at the RL-field count so the scheduler doesn't oversubscribe.
    """
    if stage_id == 6:
        return "1,2"
    if stage_id == 7:
        return "1,2,3"
    raise ValueError(f"Multi-agent production fine-tune supports stage 6 or 7, got {stage_id}")


def _default_scenario_weights_for_stage(stage_id: int) -> str:
    if stage_id == 6:
        return DEFAULT_STAGE6_SCENARIO_WEIGHTS
    if stage_id == 7:
        return DEFAULT_STAGE7_SCENARIO_WEIGHTS
    raise ValueError(f"No scenario weights for stage {stage_id}")


def _load_dataset_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_run_cfg(args: argparse.Namespace) -> train.ScriptRunConfig:
    experiment_name = os.getenv("RL_MA_PROD_EXPERIMENT_NAME", "multiagent_production_v1")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(
            f"Multi-agent BC dataset not found at {dataset}. "
            "Run teacher_multiagent_dataset.py first (RL_MA_TEACHER_STAGE=6 then =7)."
        )

    summary = _load_dataset_summary(dataset.with_name("summary.json"))
    if summary.get("obs_size", 47) != 47:
        raise ValueError(
            f"Dataset {dataset} has obs_size={summary.get('obs_size')}, expected 47."
        )
    if not bool(summary.get("positive_weight_role_mask_consistent", False)):
        print(
            f"[ma-prod] WARNING: dataset summary missing positive_weight_role_mask_consistent flag",
            flush=True,
        )

    stage_id = int(args.stage_id)
    players_str = os.getenv("RL_MA_PROD_PLAYERS", _default_players_for_stage(stage_id))
    return train.ScriptRunConfig(
        mode="train",
        backend="simrobot_visible",
        obs_mode="corrected",
        simrobot_profile="3d_visible",
        simrobot_launch=False,
        embedded_3d_preflight=False,
        learner_device=os.getenv("RL_LEARNER_DEVICE", "cuda:0" if _has_cuda() else "cpu"),
        stage_id=stage_id,
        max_stage_id=stage_id,                              # pin: no promotion within a fine-tune run
        train_steps=_env_int("RL_MA_PROD_TRAIN_STEPS", 20_000),
        # §14.6 Fri: stage-6 ≤ 5k, stage-7 10-15k. Total budget kept at 20k as a ceiling.
        max_episode_steps=_env_int("RL_MA_PROD_MAX_EPISODE_STEPS", 400),
        rollout_steps=_env_int("RL_MA_PROD_ROLLOUT_STEPS", 256),
        learning_epochs=_env_int("RL_MA_PROD_LEARNING_EPOCHS", 4),
        mini_batches=_env_int("RL_MA_PROD_MINI_BATCHES", 4),
        eval_interval=_env_int("RL_MA_PROD_EVAL_INTERVAL", 5_000),
        checkpoint_interval=_env_int("RL_MA_PROD_CHECKPOINT_INTERVAL", 2_500),
        eval_episodes=_env_int("RL_MA_PROD_EVAL_EPISODES", 8),
        save_telemetry=True,
        save_plots=True,
        capture_eval_steps=_env_bool("RL_MA_PROD_CAPTURE_EVAL_STEPS", False),
        pack_parallel_eval=_env_bool("RL_MA_PROD_PACK_PARALLEL_EVAL", True),
        abort_on_wire_fail=False,
        bc_warm_start=True,
        bc_checkpoint_path=str(checkpoint),
        bc_replay=True,
        bc_replay_dataset_path=str(dataset),
        bc_replay_steps=_env_int("RL_MA_PROD_BC_REPLAY_STEPS", train.IDE_RUN.bc_replay_steps),
        # §14.19 retunings — DEFAULTS LOCKED.
        bc_replay_weight=_env_float("RL_MA_PROD_BC_REPLAY_WEIGHT", 0.10),
        bc_replay_batch_size=_env_int("RL_MA_PROD_BC_REPLAY_BATCH_SIZE", train.IDE_RUN.bc_replay_batch_size),
        bc_replay_updates=_env_int("RL_MA_PROD_BC_REPLAY_UPDATES", 2),
        bc_replay_learning_rate=_env_float("RL_MA_PROD_BC_REPLAY_LR", train.IDE_RUN.bc_replay_learning_rate),
        # Softer BC pull than single-agent production (was 0.15). Supports need
        # learning room while striker BC keeps single-agent finishing alive.
        bc_beta_max=_env_float("RL_MA_PROD_BC_BETA_MAX", 0.08),
        bc_beta_min=_env_float("RL_MA_PROD_BC_BETA_MIN", 0.02),
        # Longer half-life than single-agent (was 10). Min-across-roles competence
        # in train_multiagent.MultiAgentBCCompetenceGate ensures β stays alive
        # until every role clears the bar.
        bc_competence_half_life_evals=_env_float("RL_MA_PROD_BC_HALF_LIFE_EVALS", 25.0),
        bc_skill_loss_scale=_env_float("RL_MA_PROD_BC_SKILL_LOSS_SCALE", 0.40),
        bc_param_loss_scale=_env_float("RL_MA_PROD_BC_PARAM_LOSS_SCALE", 0.50),
        diverse_stage_resets=True,
        # Stage 6/7 do not use the early-stage interleave knobs; single-agent
        # retention comes from the KL anchor (production checkpoint), not from
        # scenario interleaving.
        stage0_finish_interleave_probability=0.0,
        stage1_finish_interleave_probability=0.0,
        stage2_finish_interleave_probability=0.0,
        stage3_finish_interleave_probability=0.0,
        post_takeover_progress_enabled=True,
        post_takeover_progress_window_steps=120,
        telemetry_tag=os.getenv("RL_MA_PROD_TELEMETRY_TAG", f"multiagent_stage{stage_id}_finetune"),
        experiment_name=experiment_name,
        parallel_player_numbers=_parse_players(players_str),
        visible_parallel_display_base=_env_int("RL_MA_PROD_VISIBLE_DISPLAY_BASE", 430),
    )


def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def run(args: argparse.Namespace) -> None:
    # Production scenario plumbing — use the team scenario set defined in
    # production_curriculum.team_production_scenarios. The env reads
    # RL_PRODUCTION_SCENARIO_SET to pick the iterator; "team" must be wired into
    # the env's scenario factory (small env-side TODO if not yet done).
    os.environ.setdefault("RL_PRODUCTION_SCENARIO_RESETS", "1")
    os.environ.setdefault(
        "RL_PRODUCTION_SCENARIO_SET",
        os.getenv("RL_MA_PROD_SCENARIO_SET", "team"),
    )
    os.environ.setdefault(
        "RL_PRODUCTION_SCENARIO_WEIGHTS",
        os.getenv(
            "RL_MA_PROD_SCENARIO_WEIGHTS",
            _default_scenario_weights_for_stage(int(args.stage_id)),
        ),
    )
    # Visible SimRobot reset budgets — same defaults as single-agent production.
    os.environ.setdefault("RL_VISIBLE_STEP_OBS_TIMEOUT_MS", "1000")
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS", "4000")
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_MAX_REPLACES", "5")
    os.environ.setdefault("RL_VISIBLE_RESET_MAX_ATTEMPTS", "3")

    # Tell the env that pass and the support roles are live for this run.
    # (active_skill_names comes from curriculum.get_stage_spec(stage_id).)
    if os.getenv("RL_MA_PROD_VISIBLE_OBS_TIMEOUT_MS") is not None:
        backend_cfg = replace(
            train.ENV.backend,
            obs_timeout_ms=_env_int("RL_MA_PROD_VISIBLE_OBS_TIMEOUT_MS", train.ENV.backend.obs_timeout_ms),
        )
        train.ENV = replace(train.ENV, backend=backend_cfg)

    # Stage pinned: this is a multi-agent fine-tune, not a curriculum climb.
    train.TRAIN = replace(train.TRAIN, auto_promote=False)

    run_cfg = _build_run_cfg(args)
    print(
        "[ma-production-train] "
        f"stage={run_cfg.stage_id} "
        f"checkpoint={Path(args.checkpoint).expanduser().resolve()} "
        f"dataset={Path(args.dataset).expanduser().resolve()} "
        f"experiment={run_cfg.experiment_name} "
        f"steps={run_cfg.train_steps} players={run_cfg.parallel_player_numbers} "
        f"scenario_set={os.getenv('RL_PRODUCTION_SCENARIO_SET')} "
        f"eval_episodes={run_cfg.eval_episodes} max_episode_steps={run_cfg.max_episode_steps} "
        f"bc[beta={run_cfg.bc_beta_min:.3f}..{run_cfg.bc_beta_max:.3f},half_life={run_cfg.bc_competence_half_life_evals:g},"
        f"replay_w={run_cfg.bc_replay_weight:.3f},updates={run_cfg.bc_replay_updates},"
        f"skill={run_cfg.bc_skill_loss_scale:.2f},param={run_cfg.bc_param_loss_scale:.2f}]",
        flush=True,
    )
    train_multiagent.main(run_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent production fine-tune (stage 6 or 7) — sibling of train_production.py."
    )
    parser.add_argument("--stage-id", type=int, choices=(6, 7), default=int(os.getenv("RL_MA_PROD_STAGE", "6")))
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("RL_MA_PROD_CHECKPOINT", str(_default_kl_anchor())),
        help="KL anchor checkpoint. Defaults to runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt.",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("RL_MA_BC_DATASET", str(_default_dataset())),
        help="Multi-agent BC dataset samples.npz. Defaults to the most recent under runs/teacher_multiagent_dataset/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
