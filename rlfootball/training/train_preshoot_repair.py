from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from rlfootball.curriculum import curriculum
from rlfootball.training import train_production


ROOT = Path(__file__).resolve().parents[2]
PRESHOOT_REPLAY_DATASET = (
    ROOT / "runs/fine_tuning_production_dataset/preshoot_replay_stage4/samples.npz"
)
DEFAULT_CHECKPOINT_CANDIDATES = (
    ROOT / "runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt",
    ROOT / "runs/production_readiness_v2/checkpoint_stage4_step10000.pt",
    ROOT / "runs/production_finetune_v1/checkpoint_stage4_step35000.pt",
    ROOT / "runs/production_finetune_v1/checkpoint_stage4_step25000.pt",
)

# Keep the canonical evaluator scenarios in the loop, but bias the run toward the known
# unresolved PPO gap: bad-angle pre-shoot behavior under dynamic / blocked pressure.
DEFAULT_PRESHOOT_SCENARIO_WEIGHTS = (
    "preshoot_bad_angle_boundary:8,"
    "blocked_corridor_repair:6,"
    "dynamic_full_field_pressure:4,"
    "blocked_corridor_finish_topup:6,"
    "dynamic_finish_pressure_topup:5,"
    "finish_band_2p5_3m_static_lane:3,"
    "finish_bridge_static_topup:3,"
    "finish_band_2p5_3m_clean:2,"
    "finish_bridge_clean_topup:2,"
    "finish_near_arm_clean_topup:1"
)


def _default_checkpoint() -> Path:
    for path in DEFAULT_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_CHECKPOINT_CANDIDATES[0]


def _env_float(name: str, fallback: float) -> float:
    raw = os.getenv(name)
    return fallback if raw in (None, "") else float(raw)


def _apply_stage4_reward_overrides() -> None:
    stage4 = curriculum.get_stage_spec(4)
    reward = stage4.reward_weights
    collision = _env_float("RL_PRESHOOT_STAGE4_COLLISION", reward.collision)
    corridor_keep = _env_float("RL_PRESHOOT_STAGE4_CORRIDOR_KEEP", reward.corridor_keep)
    thrash = _env_float("RL_PRESHOOT_STAGE4_THRASH", reward.thrash)
    if (
        collision == reward.collision
        and corridor_keep == reward.corridor_keep
        and thrash == reward.thrash
    ):
        return
    curriculum._STAGE_SPECS[4] = replace(  # type: ignore[attr-defined]
        stage4,
        reward_weights=replace(
            reward,
            collision=collision,
            corridor_keep=corridor_keep,
            thrash=thrash,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-4 PPO repair fine-tune focused on preshoot_bad_angle_boundary and pressure finishes."
    )
    parser.add_argument("--dataset", default=str(train_production.SOURCE_DATASET))
    parser.add_argument("--output-dataset", default=str(PRESHOOT_REPLAY_DATASET))
    parser.add_argument("--checkpoint", default=str(_default_checkpoint()))
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("RL_PROD_EXPERIMENT_NAME", "production_preshoot_repair_v1")
    os.environ.setdefault("RL_PROD_TRAIN_STEPS", "12000")
    os.environ.setdefault("RL_PROD_EVAL_INTERVAL", "2000")
    os.environ.setdefault("RL_PROD_CHECKPOINT_INTERVAL", "2000")
    os.environ.setdefault("RL_PROD_EVAL_EPISODES", "8")
    os.environ.setdefault("RL_PROD_MAX_EPISODE_STEPS", "1000")
    os.environ.setdefault("RL_PROD_SCENARIO_SET", "all")
    os.environ.setdefault("RL_PROD_SCENARIO_WEIGHTS", DEFAULT_PRESHOOT_SCENARIO_WEIGHTS)
    os.environ.setdefault("RL_PROD_PLAYERS", "1,2")
    os.environ.setdefault("RL_PROD_BC_REPLAY_WEIGHT", "0.12")
    os.environ.setdefault("RL_PROD_BC_REPLAY_UPDATES", "3")
    os.environ.setdefault("RL_PROD_BC_BETA_MAX", "0.18")
    os.environ.setdefault("RL_PROD_BC_HALF_LIFE_EVALS", "12")
    os.environ.setdefault("RL_PROD_BC_SKILL_LOSS_SCALE", "0.45")
    os.environ.setdefault("RL_PROD_BC_PARAM_LOSS_SCALE", "0.55")
    os.environ.setdefault("RL_PROD_VISIBLE_STEP_OBS_TIMEOUT_MS", "1000")
    os.environ.setdefault("RL_VISIBLE_RESET_MAX_ATTEMPTS", "5")
    os.environ.setdefault("RL_VISIBLE_RESET_POSE_TOL_MM", "250")
    os.environ.setdefault("RL_VISIBLE_RESET_UPRIGHT_STREAK", "1")
    os.environ.setdefault("RL_VISIBLE_RESET_MAX_FALL_STATE", "1")
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS", "4000")
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_MAX_REPLACES", "5")
    os.environ.setdefault("RL_PRESHOOT_STAGE4_COLLISION", "-3.4")
    _apply_stage4_reward_overrides()
    train_production.run(args)


if __name__ == "__main__":
    run(parse_args())
