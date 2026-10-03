from __future__ import annotations

import argparse
import os
from pathlib import Path

from rlfootball.training import train_production


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_CANDIDATES = (
    ROOT / "runs/production_readiness_v2/checkpoint_stage4_step10000.pt",
    ROOT / "runs/production_finetune_v1/checkpoint_stage4_step35000.pt",
    ROOT / "runs/production_finetune_v1/checkpoint_stage4_step25000.pt",
    ROOT / "runs/production_finetune_v1/checkpoint_stage4_step20000.pt",
)

DEFAULT_STABILITY_SCENARIO_WEIGHTS = (
    "dynamic_full_field_pressure:6,"
    "blocked_corridor_repair:8,"
    "blocked_corridor_finish_topup:6,"
    "dynamic_finish_pressure_topup:6,"
    "finish_band_2p5_3m_static_lane:3,"
    "finish_band_2p5_3m_clean:2,"
    "finish_bridge_static_topup:3,"
    "finish_bridge_clean_topup:2,"
    "finish_near_arm_clean_topup:1,"
    "preshoot_bad_angle_boundary:1"
)


def _default_checkpoint() -> Path:
    for path in DEFAULT_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_CHECKPOINT_CANDIDATES[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-4 stability repair fine-tune entrypoint.")
    parser.add_argument("--dataset", default=str(train_production.SOURCE_DATASET))
    parser.add_argument("--output-dataset", default=str(train_production.STAGE4_REPLAY_DATASET))
    parser.add_argument("--checkpoint", default=str(_default_checkpoint()))
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("RL_PROD_EXPERIMENT_NAME", "production_stability_repair_v1")
    os.environ.setdefault("RL_PROD_TRAIN_STEPS", "20000")
    os.environ.setdefault("RL_PROD_EVAL_INTERVAL", "2500")
    os.environ.setdefault("RL_PROD_CHECKPOINT_INTERVAL", "2500")
    os.environ.setdefault("RL_PROD_EVAL_EPISODES", "8")
    os.environ.setdefault("RL_PROD_MAX_EPISODE_STEPS", "1000")
    os.environ.setdefault("RL_PROD_SCENARIO_SET", "eval_plus_topup")
    os.environ.setdefault("RL_PROD_SCENARIO_WEIGHTS", DEFAULT_STABILITY_SCENARIO_WEIGHTS)
    os.environ.setdefault("RL_PROD_VISIBLE_STEP_OBS_TIMEOUT_MS", "1000")
    os.environ.setdefault("RL_VISIBLE_RESET_MAX_ATTEMPTS", "3")
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS", "4000")
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_MAX_REPLACES", "5")
    os.environ.setdefault("RL_VISIBLE_RESET_UPRIGHT_STREAK", "3")
    os.environ.setdefault("RL_VISIBLE_RESET_MAX_FALL_STATE", "1")
    train_production.run(args)


if __name__ == "__main__":
    run(parse_args())
