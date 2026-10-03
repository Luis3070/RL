"""Goalkeeper RL training entrypoint — single-agent, stage 9.

The goalkeeper robot:
  * obs layout: identical 47-dim vector, forced is_off_ball_support=1 (obs[46])
  * action mask: stand + walk always; block when dribble_armed (RL_GATE_GK_MODE=1
    widens the arming radius to 1200 mm so fast-approaching balls trigger it)
  * reward: evaluate_step_gk() from reward.py (danger cover, goal-line pos,
    block success, out-of-area penalty, clearance, goal conceded)
  * walk anchor: (own_goal_x+400, ball_y*0.55) — cover point, NOT ball approach
  * warm-start: defender step2500 checkpoint (same 47-dim architecture)
  * stage: 9 (pinned; no auto-promote)

Architecture identical to the defender/multi-agent policy — no OBS_SIZE change,
no SKILL_ORDER change, no new weights. The is_off_ball_support bit routes the
existing mask logic to allow block-when-armed and suppresses shoot/pass/observe.

Run (single slot — requires SimRobot with RLMAStage7_3D.ros2):
    source .venv_env.sh
    RL_GATE_GK_MODE=1 python train_goalkeeper.py

Or use tools/run_gkv1_finetune.sh for the full chunk launcher.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import replace
from pathlib import Path

# Must be set before gate / environment import so GateConfig.from_env() picks it up.
os.environ.setdefault("RL_GATE_GK_MODE", "1")

import numpy as np

import train
from config import GOAL_Y_HALF

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Warm-start + dataset defaults
# ---------------------------------------------------------------------------

DEFAULT_GK_WARM_START_CANDIDATES = (
    ROOT / "runs/defender_47_v2/checkpoint_stage8_step2500.pt",
    ROOT / "runs/defender_47_v2/checkpoint_stage8_step5000.pt",
    ROOT / "runs/defender_47_v2/checkpoint_stage8_step7500.pt",
    ROOT / "runs/defender_47_v2/checkpoint_stage8_step10000.pt",
)
DEFAULT_GK_DATASET_ROOT = ROOT / "runs/teacher_goalkeeper_dataset"


def _default_warm_start() -> Path:
    for p in DEFAULT_GK_WARM_START_CANDIDATES:
        if p.exists():
            return p
    return DEFAULT_GK_WARM_START_CANDIDATES[0]


def _default_dataset() -> Path:
    if not DEFAULT_GK_DATASET_ROOT.exists():
        return DEFAULT_GK_DATASET_ROOT / "MISSING_DATASET" / "samples.npz"
    candidates = sorted(
        (p for p in DEFAULT_GK_DATASET_ROOT.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        s = c / "samples.npz"
        if s.exists():
            return s
    return DEFAULT_GK_DATASET_ROOT / "EMPTY" / "samples.npz"


# ---------------------------------------------------------------------------
# GK env patches (monkey-patches installed/restored around train.main)
# ---------------------------------------------------------------------------

_OWN_GOAL_X: float = -4500.0
_GK_COVER_OFFSET_X: float = 400.0


def _make_gk_team_hook():
    """Return a team_hook for a single GK slot.

    Sets is_off_ball_support=True (role bit obs[46]) and installs the GK walk
    anchor (cover point: own_goal_x+400, ball_y*0.55) via env._team_coordination_target.
    Also forces env._team_role = "off_ball_support" so the env's walk-anchor
    decoder uses build_open_lane_anchor(obs, coordination_target) instead of
    ball-approach anchor.
    """
    from gate import GateDecision
    from multiagent_teacher import build_team_gate_decision

    def hook(env, obs, base_decision: GateDecision):
        ball_y = float(getattr(obs, "ball_y", 0.0))
        cover_x = _OWN_GOAL_X + _GK_COVER_OFFSET_X
        cover_y = float(np.clip(ball_y * 0.55, -GOAL_Y_HALF, GOAL_Y_HALF))
        # Drive the env's walk anchor decoder to GK cover logic.
        env._team_role = "off_ball_support"
        env._team_coordination_target = (cover_x, cover_y)
        decision = build_team_gate_decision(base_decision, role="goalkeeper")
        return obs, decision

    return hook


def _install_gk_patches() -> tuple:
    """Monkey-patch environment module to use evaluate_step_gk and GK team hook."""
    import environment as env_mod
    from reward import evaluate_step_gk

    # Swap reward function: environment.py calls `evaluate_step(step_ctx, weights)`
    # at module scope; patching the module attribute redirects all calls in that file.
    orig_evaluate_step = env_mod.evaluate_step
    env_mod.evaluate_step = evaluate_step_gk  # type: ignore[attr-defined]

    # Patch ParallelSlotSimRobotEnv.__init__ to inject the GK team hook into each slot.
    from train import ParallelSlotSimRobotEnv

    orig_psse_init = ParallelSlotSimRobotEnv.__init__

    def patched_init(self, env_cfg):
        orig_psse_init(self, env_cfg)
        hook = _make_gk_team_hook()
        for agent in self._envs:
            self._envs[agent]._team_hook = hook

    ParallelSlotSimRobotEnv.__init__ = patched_init  # type: ignore[method-assign]

    return orig_evaluate_step, orig_psse_init, ParallelSlotSimRobotEnv


def _restore_gk_patches(orig_evaluate_step, orig_psse_init, psse_cls) -> None:
    import environment as env_mod
    env_mod.evaluate_step = orig_evaluate_step  # type: ignore[attr-defined]
    psse_cls.__init__ = orig_psse_init  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------

def _env_int(name: str, fallback: int) -> int:
    v = os.getenv(name)
    return fallback if v is None else int(v)


def _env_float(name: str, fallback: float) -> float:
    v = os.getenv(name)
    return fallback if v is None else float(v)


def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _build_run_cfg(args: argparse.Namespace) -> train.ScriptRunConfig:
    experiment_name = os.getenv("RL_GK_EXPERIMENT_NAME", "goalkeeper_gkv1")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"GK warm-start checkpoint not found: {checkpoint}")
    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(
            f"GK BC dataset not found at {dataset}. "
            "Run teacher_goalkeeper_dataset.py first."
        )

    return train.ScriptRunConfig(
        mode="train",
        backend="simrobot_visible",
        obs_mode="corrected",
        simrobot_profile="3d_visible",
        simrobot_launch=True,
        embedded_3d_preflight=False,
        learner_device=os.getenv("RL_LEARNER_DEVICE", "cuda:0" if _has_cuda() else "cpu"),
        stage_id=9,
        max_stage_id=9,          # pinned: no auto-promote
        train_steps=_env_int("RL_GK_TRAIN_STEPS", 15_000),
        max_episode_steps=_env_int("RL_GK_MAX_EPISODE_STEPS", 600),
        rollout_steps=_env_int("RL_GK_ROLLOUT_STEPS", 256),
        learning_epochs=_env_int("RL_GK_LEARNING_EPOCHS", 4),
        mini_batches=_env_int("RL_GK_MINI_BATCHES", 4),
        eval_interval=_env_int("RL_GK_EVAL_INTERVAL", 2_500),
        checkpoint_interval=_env_int("RL_GK_CHECKPOINT_INTERVAL", 2_500),
        eval_episodes=_env_int("RL_GK_EVAL_EPISODES", 6),
        save_telemetry=True,
        save_plots=True,
        capture_eval_steps=False,
        pack_parallel_eval=True,
        abort_on_wire_fail=False,
        # BC warm-start from defender checkpoint (same 47-dim architecture).
        bc_warm_start=True,
        bc_checkpoint_path=str(checkpoint),
        bc_replay=True,
        bc_replay_dataset_path=str(dataset),
        bc_replay_steps=_env_int("RL_GK_BC_REPLAY_STEPS", train.IDE_RUN.bc_replay_steps),
        # Moderate BC pull — GK needs learning room to discover block timing.
        bc_replay_weight=_env_float("RL_GK_BC_REPLAY_WEIGHT", 0.12),
        bc_replay_batch_size=_env_int("RL_GK_BC_REPLAY_BATCH_SIZE", train.IDE_RUN.bc_replay_batch_size),
        bc_replay_updates=_env_int("RL_GK_BC_REPLAY_UPDATES", 2),
        bc_replay_learning_rate=_env_float("RL_GK_BC_REPLAY_LR", train.IDE_RUN.bc_replay_learning_rate),
        # Stronger beta than multi-agent (GK teacher is clean; block is rare so BC
        # must stay alive until the policy learns when to dive).
        bc_beta_max=_env_float("RL_GK_BC_BETA_MAX", 0.12),
        bc_beta_min=_env_float("RL_GK_BC_BETA_MIN", 0.02),
        bc_competence_half_life_evals=_env_float("RL_GK_BC_HALF_LIFE_EVALS", 20.0),
        bc_skill_loss_scale=_env_float("RL_GK_BC_SKILL_LOSS_SCALE", 0.50),
        bc_param_loss_scale=_env_float("RL_GK_BC_PARAM_LOSS_SCALE", 0.50),
        diverse_stage_resets=True,
        stage0_finish_interleave_probability=0.0,
        stage1_finish_interleave_probability=0.0,
        stage2_finish_interleave_probability=0.0,
        stage3_finish_interleave_probability=0.0,
        post_takeover_progress_enabled=True,
        post_takeover_progress_window_steps=120,
        telemetry_tag=os.getenv("RL_GK_TELEMETRY_TAG", "goalkeeper_gkv1_stage9"),
        experiment_name=experiment_name,
        parallel_player_numbers=(1,),   # single GK slot
        visible_parallel_display_base=_env_int("RL_GK_VISIBLE_DISPLAY_BASE", 440),
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    os.environ["RL_GATE_GK_MODE"] = "1"
    os.environ.setdefault("RL_VISIBLE_STEP_OBS_TIMEOUT_MS", "2500")
    os.environ.setdefault("RL_VISIBLE_RESET_MAX_ATTEMPTS", "12")
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS", "4000")
    os.environ.setdefault("RL_VISIBLE_BALL_SETTLE_MAX_REPLACES", "5")
    # Scene must be the multi-robot scene (GK needs a 2-player scene with at least
    # the own-goal area; RLMAStage7_3D.ros2 has the correct layout).
    os.environ.setdefault("RL_SCENE_NAME", "RLMAStage7_3D")
    # Stage pin: no production scenario iteration; stage 9 uses curriculum resets.
    os.environ.setdefault("RL_PRODUCTION_SCENARIO_RESETS", "0")

    train.TRAIN = replace(train.TRAIN, auto_promote=False)

    run_cfg = _build_run_cfg(args)
    print(
        "[gk-train] "
        f"stage=9 experiment={run_cfg.experiment_name} "
        f"checkpoint={args.checkpoint} "
        f"dataset={args.dataset} "
        f"steps={run_cfg.train_steps} "
        f"gk_mode=ON gate_block_enter=1200mm "
        f"bc[beta={run_cfg.bc_beta_min:.3f}..{run_cfg.bc_beta_max:.3f},"
        f"half_life={run_cfg.bc_competence_half_life_evals:g},"
        f"replay_w={run_cfg.bc_replay_weight:.3f}]",
        flush=True,
    )
    patches = _install_gk_patches()
    try:
        train.main(run_cfg)
    finally:
        _restore_gk_patches(*patches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Goalkeeper RL training — stage 9, single GK slot, block→InterceptBall via RL_GATE_GK_MODE=1."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("RL_GK_CHECKPOINT", str(_default_warm_start())),
        help="Warm-start checkpoint (default: defender_47_v2/checkpoint_stage8_step2500.pt).",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("RL_GK_BC_DATASET", str(_default_dataset())),
        help="GK BC dataset samples.npz (generate with teacher_goalkeeper_dataset.py).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
