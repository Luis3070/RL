from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from rlfootball.curriculum import curriculum
from rlfootball.training import train_production


ROOT = Path(__file__).resolve().parents[2]
FOCUSED_DATASET = ROOT / "runs" / "teacher_production_dataset" / "dataset_20260609_222021" / "samples.npz"
FOCUSED_REPLAY_DATASET = (
    ROOT / "runs" / "fine_tuning_production_dataset" / "preshoot_blocked_gate_repair_replay_stage4" / "samples.npz"
)
DEFAULT_CHECKPOINT_CANDIDATES = (
    ROOT / "runs" / "production_preshoot_blocked_stability_repair_v1" / "checkpoint_stage4_step4000.pt",
    ROOT / "runs" / "production_preshoot_blocked_stability_repair_v1" / "checkpoint_stage4_step6000.pt",
    ROOT / "runs" / "production_preshoot_repair_v1" / "checkpoint_stage4_step12000.pt",
)

DEFAULT_SCENARIO_WEIGHTS = (
    "preshoot_bad_angle_boundary:12,"
    "blocked_corridor_repair:10,"
    "blocked_corridor_finish_topup:8,"
    "finish_band_2p5_3m_static_lane:6,"
    "finish_bridge_static_topup:5,"
    "dynamic_full_field_pressure:2,"
    "dynamic_finish_pressure_topup:2,"
    "finish_band_2p5_3m_clean:1"
)


def _default_checkpoint() -> Path:
    for path in DEFAULT_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_CHECKPOINT_CANDIDATES[0]


def _default_dataset() -> Path:
    return FOCUSED_DATASET if FOCUSED_DATASET.exists() else train_production.SOURCE_DATASET


def _env_float(name: str, fallback: float) -> float:
    raw = os.getenv(name)
    return fallback if raw in (None, "") else float(raw)


def _apply_stage4_reward_overrides() -> None:
    stage4 = curriculum.get_stage_spec(4)
    reward = stage4.reward_weights
    collision = _env_float("RL_GATE_REPAIR_STAGE4_COLLISION", reward.collision)
    corridor_keep = _env_float("RL_GATE_REPAIR_STAGE4_CORRIDOR_KEEP", reward.corridor_keep)
    bad_ball_angle = _env_float("RL_GATE_REPAIR_STAGE4_BAD_BALL_ANGLE", reward.bad_ball_angle)
    if (
        collision == reward.collision
        and corridor_keep == reward.corridor_keep
        and bad_ball_angle == reward.bad_ball_angle
    ):
        return
    curriculum._STAGE_SPECS[4] = replace(  # type: ignore[attr-defined]
        stage4,
        reward_weights=replace(
            reward,
            collision=collision,
            corridor_keep=corridor_keep,
            bad_ball_angle=bad_ball_angle,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-4 PPO fine-tune with gate-biased heuristics: make dribble easier to arm/hold "
            "while making shoot harder to arm under bad-angle / blocked-pressure situations."
        )
    )
    parser.add_argument("--dataset", default=str(_default_dataset()))
    parser.add_argument("--output-dataset", default=str(FOCUSED_REPLAY_DATASET))
    parser.add_argument("--checkpoint", default=str(_default_checkpoint()))
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def _apply_gate_defaults() -> None:
    # Dribble earlier and hold longer so the policy leaves the walk/stand fallback sooner.
    os.environ.setdefault("RL_GATE_DRIBBLE_ENTER_BALL_MM", "500")
    os.environ.setdefault("RL_GATE_DRIBBLE_EXIT_BALL_MM", "650")
    os.environ.setdefault("RL_GATE_DRIBBLE_GRACE_ENTER_AGE_MS", "1800")
    os.environ.setdefault("RL_GATE_DRIBBLE_GRACE_HOLD_AGE_MS", "3200")
    # Keep shoot stricter so preshoot does not turn into premature low-quality kicks.
    os.environ.setdefault("RL_GATE_SHOOT_OPEN_ENTER", "0.68")
    os.environ.setdefault("RL_GATE_SHOOT_OPEN_EXIT", "0.52")
    os.environ.setdefault("RL_GATE_SHOOT_ALIGN_ENTER_DEG", "24")
    os.environ.setdefault("RL_GATE_SHOOT_ALIGN_EXIT_DEG", "36")
    os.environ.setdefault("RL_GATE_SHOOT_ARM_FRAMES", "4")


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("RL_PROD_EXPERIMENT_NAME", "production_preshoot_blocked_gate_repair_v1")
    os.environ.setdefault("RL_PROD_TRAIN_STEPS", "6000")
    os.environ.setdefault("RL_PROD_EVAL_INTERVAL", "2000")
    os.environ.setdefault("RL_PROD_CHECKPOINT_INTERVAL", "2000")
    os.environ.setdefault("RL_PROD_EVAL_EPISODES", "8")
    os.environ.setdefault("RL_PROD_MAX_EPISODE_STEPS", "1000")
    os.environ.setdefault("RL_PROD_SCENARIO_SET", "all")
    os.environ.setdefault("RL_PROD_SCENARIO_WEIGHTS", DEFAULT_SCENARIO_WEIGHTS)
    os.environ.setdefault("RL_PROD_PLAYERS", "1,2")
    os.environ.setdefault("RL_PROD_BC_REPLAY_WEIGHT", "0.18")
    os.environ.setdefault("RL_PROD_BC_REPLAY_UPDATES", "4")
    os.environ.setdefault("RL_PROD_BC_BETA_MAX", "0.22")
    os.environ.setdefault("RL_PROD_BC_HALF_LIFE_EVALS", "10")
    os.environ.setdefault("RL_PROD_BC_SKILL_LOSS_SCALE", "0.55")
    os.environ.setdefault("RL_PROD_BC_PARAM_LOSS_SCALE", "0.65")
    os.environ.setdefault("RL_PROD_VISIBLE_STEP_OBS_TIMEOUT_MS", "1000")
    os.environ.setdefault("RL_VISIBLE_RESET_MAX_ATTEMPTS", "5")
    os.environ.setdefault("RL_VISIBLE_RESET_POSE_TOL_MM", "250")
    os.environ.setdefault("RL_VISIBLE_RESET_UPRIGHT_STREAK", "1")
    os.environ.setdefault("RL_VISIBLE_RESET_MAX_FALL_STATE", "1")
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS", "4000")
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_MAX_REPLACES", "5")
    os.environ.setdefault("RL_GATE_REPAIR_STAGE4_COLLISION", "-4.2")
    os.environ.setdefault("RL_GATE_REPAIR_STAGE4_CORRIDOR_KEEP", "1.15")
    os.environ.setdefault("RL_GATE_REPAIR_STAGE4_BAD_BALL_ANGLE", "-0.24")
    _apply_gate_defaults()
    _apply_stage4_reward_overrides()
    dataset_path = Path(args.dataset).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    print(
        "[preshoot-blocked-gate-train] "
        f"dataset={dataset_path} "
        f"checkpoint={checkpoint_path} "
        f"shoot_open_enter={os.environ['RL_GATE_SHOOT_OPEN_ENTER']} "
        f"shoot_align_enter_deg={os.environ['RL_GATE_SHOOT_ALIGN_ENTER_DEG']} "
        f"dribble_enter_ball_mm={os.environ['RL_GATE_DRIBBLE_ENTER_BALL_MM']} "
        f"dribble_grace_hold_age_ms={os.environ['RL_GATE_DRIBBLE_GRACE_HOLD_AGE_MS']}",
        flush=True,
    )
    train_production.run(args)


if __name__ == "__main__":
    run(parse_args())
