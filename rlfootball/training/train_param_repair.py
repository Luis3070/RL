from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rlfootball.core.action import SKILL_ORDER
from rlfootball.curriculum.curriculum import get_stage_spec
from rlfootball.core.observation import ObsIndex
from tools.export.export_ppo_policy import ActorExport
from rlfootball.training.train import gate_skill_mask_from_obs
from rlfootball.training.train_stand_repair import SkillRepairActor, _make_synthetic_no_finish_walk_states


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "runs/production_stand_repair_v1/checkpoint_stage4_stand_repair.pt"
DEFAULT_DATASET = ROOT / "runs/fine_tuning_production_dataset/production_replay_stage4/samples.npz"
DEFAULT_OUTPUT = ROOT / "runs/production_param_repair_v1/checkpoint_stage4_param_repair.pt"

FIELD_X_HALF = 4500.0
FIELD_Y_HALF = 3000.0
GOAL_X = 4500.0
APPROACH_OFFSET_X = -210.0
RESIDUAL_Y_RANGE = 450.0
RESIDUAL_THETA_RANGE = 0.75
WALK_START_THRESHOLD = 150.0
WALK_ROTATION_START_THRESHOLD = math.radians(5.0)


def _wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _stage4_mask(device: torch.device) -> torch.Tensor:
    mask = np.asarray([1.0 if name in get_stage_spec(4).active_skill_names else 0.0 for name in SKILL_ORDER], dtype=np.float32)
    return torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)


def _masked_logits(logits: torch.Tensor, obs: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill((gate_skill_mask_from_obs(obs) * active_mask) <= 0.0, -1.0e9)


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    obs = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    weights = data["weights"].astype(np.float32)
    if obs.ndim != 2 or obs.shape[1] != 26:
        raise ValueError(f"Expected observations [N,26], got {obs.shape}")
    positive = weights > 0.0
    return obs[positive], actions[positive], weights[positive]


def _make_near_dead_walk_states(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obs = np.zeros((n, 26), dtype=np.float32)

    d_ball = rng.uniform(430.0, 720.0, size=n)
    a_ball = np.deg2rad(rng.uniform(-70.0, 70.0, size=n))
    ball_rel_x = d_ball * np.cos(a_ball)
    ball_rel_y = d_ball * np.sin(a_ball)
    robot_x = rng.uniform(-2500.0, 2500.0, size=n)
    robot_y = rng.uniform(-1800.0, 1800.0, size=n)
    robot_theta = rng.uniform(-math.pi, math.pi, size=n)

    obs[:, ObsIndex.robot_x] = np.clip(robot_x / FIELD_X_HALF, -1.0, 1.0)
    obs[:, ObsIndex.robot_y] = np.clip(robot_y / FIELD_Y_HALF, -1.0, 1.0)
    obs[:, ObsIndex.robot_theta] = robot_theta / math.pi
    obs[:, ObsIndex.ball_rel_x] = np.clip(ball_rel_x / FIELD_X_HALF, -1.0, 1.0)
    obs[:, ObsIndex.ball_rel_y] = np.clip(ball_rel_y / FIELD_Y_HALF, -1.0, 1.0)
    obs[:, ObsIndex.ball_end_rel_x] = obs[:, ObsIndex.ball_rel_x]
    obs[:, ObsIndex.ball_end_rel_y] = obs[:, ObsIndex.ball_rel_y]
    obs[:, ObsIndex.time_since_ball_seen] = np.clip(rng.uniform(0.0, 800.0, size=n) / 5000.0, 0.0, 1.0)
    obs[:, ObsIndex.time_since_ball_disappeared] = obs[:, ObsIndex.time_since_ball_seen]
    obs[:, ObsIndex.ball_seen_percentage] = rng.uniform(0.0, 0.08, size=n)
    obs[:, ObsIndex.ball_consistent_with_game_state] = 1.0
    obs[:, ObsIndex.shot_quality_no_obstacles] = rng.uniform(0.01, 0.55, size=n)
    obs[:, ObsIndex.shot_opening_with_obstacles] = rng.uniform(0.01, 0.55, size=n)
    obs[:, ObsIndex.pass_options_count] = rng.uniform(0.0, 0.8, size=n)
    obs[:, ObsIndex.nearest_teammate_dist] = rng.uniform(0.2, 1.0, size=n)
    obs[:, ObsIndex.nearest_opponent_dist] = rng.uniform(0.2, 1.0, size=n)
    obs[:, ObsIndex.nearest_uncertain_obstacle_dist] = rng.uniform(0.2, 1.0, size=n)
    obs[:, ObsIndex.nearest_teammate_front_dist] = rng.uniform(0.2, 1.0, size=n)
    obs[:, ObsIndex.nearest_opponent_front_dist] = rng.uniform(0.2, 1.0, size=n)
    obs[:, ObsIndex.nearest_uncertain_front_dist] = rng.uniform(0.2, 1.0, size=n)
    return obs


def _raw_geometry(obs: np.ndarray) -> dict[str, np.ndarray]:
    robot_x = obs[:, ObsIndex.robot_x].astype(np.float64) * FIELD_X_HALF
    robot_y = obs[:, ObsIndex.robot_y].astype(np.float64) * FIELD_Y_HALF
    robot_theta = obs[:, ObsIndex.robot_theta].astype(np.float64) * math.pi
    ball_rel_x = obs[:, ObsIndex.ball_rel_x].astype(np.float64) * FIELD_X_HALF
    ball_rel_y = obs[:, ObsIndex.ball_rel_y].astype(np.float64) * FIELD_Y_HALF
    c = np.cos(robot_theta)
    s = np.sin(robot_theta)
    ball_x = robot_x + c * ball_rel_x - s * ball_rel_y
    ball_y = robot_y + s * ball_rel_x + c * ball_rel_y
    return {
        "robot_x": robot_x,
        "robot_y": robot_y,
        "robot_theta": robot_theta,
        "ball_rel_x": ball_rel_x,
        "ball_rel_y": ball_rel_y,
        "ball_x": np.clip(ball_x, -FIELD_X_HALF, FIELD_X_HALF),
        "ball_y": np.clip(ball_y, -FIELD_Y_HALF, FIELD_Y_HALF),
    }


def _decode_walk_relative(obs: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = _raw_geometry(obs)
    residual_y = np.clip(params[:, 1].astype(np.float64), -1.0, 1.0)
    residual_theta = np.clip(params[:, 2].astype(np.float64), -1.0, 1.0)

    anchor_x = np.clip(g["ball_x"] + APPROACH_OFFSET_X, -FIELD_X_HALF, FIELD_X_HALF)
    anchor_y = np.clip(g["ball_y"], -FIELD_Y_HALF, FIELD_Y_HALF)
    anchor_theta = np.arctan2(g["ball_y"] - g["robot_y"], g["ball_x"] - g["robot_x"])
    target_x = anchor_x
    target_y = np.clip(anchor_y + residual_y * RESIDUAL_Y_RANGE, -FIELD_Y_HALF, FIELD_Y_HALF)
    target_theta = _wrap_angle_np(anchor_theta + residual_theta * RESIDUAL_THETA_RANGE)

    dx = target_x - g["robot_x"]
    dy = target_y - g["robot_y"]
    c = np.cos(g["robot_theta"])
    s = np.sin(g["robot_theta"])
    target_rel_x = c * dx + s * dy
    target_rel_y = -s * dx + c * dy
    target_rel_theta = _wrap_angle_np(target_theta - g["robot_theta"])
    return target_rel_x, target_rel_y, target_rel_theta


def _walk_dead_mask(obs: np.ndarray, params: np.ndarray) -> np.ndarray:
    target_rel_x, target_rel_y, target_rel_theta = _decode_walk_relative(obs, params)
    return (
        (np.abs(target_rel_x) <= WALK_START_THRESHOLD)
        & (np.abs(target_rel_y) <= WALK_START_THRESHOLD)
        & (np.abs(target_rel_theta) <= WALK_ROTATION_START_THRESHOLD)
    )


def _policy_outputs(actor: torch.nn.Module, obs_np: np.ndarray, *, device: torch.device, active_mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actor.eval()
    with torch.no_grad():
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        logits, params = actor(obs)
        masked = _masked_logits(logits, obs, active_mask)
        pred = masked.argmax(dim=-1)
    actor.train()
    return pred.cpu().numpy(), logits.cpu().numpy(), params.cpu().numpy()


def _param_metrics(actor: torch.nn.Module, obs_np: np.ndarray, *, device: torch.device, active_mask: torch.Tensor) -> dict[str, float]:
    walk = SKILL_ORDER.index("walk")
    pred, _, params = _policy_outputs(actor, obs_np, device=device, active_mask=active_mask)
    walk_mask = pred == walk
    if not np.any(walk_mask):
        return {"n": float(len(obs_np)), "walk_n": 0.0, "walk_dead_rate": 0.0}
    walk_params = params[walk_mask]
    dead = _walk_dead_mask(obs_np[walk_mask], walk_params)
    rel_x, rel_y, rel_theta = _decode_walk_relative(obs_np[walk_mask], walk_params)
    return {
        "n": float(len(obs_np)),
        "walk_n": float(np.sum(walk_mask)),
        "walk_rate": float(np.mean(walk_mask)),
        "walk_dead_rate": float(np.mean(dead)),
        "walk_param_y_mean": float(np.mean(walk_params[:, 1])),
        "walk_param_y_p01": float(np.quantile(walk_params[:, 1], 0.01)),
        "walk_param_y_p50": float(np.quantile(walk_params[:, 1], 0.50)),
        "walk_param_y_p99": float(np.quantile(walk_params[:, 1], 0.99)),
        "walk_param_theta_mean": float(np.mean(walk_params[:, 2])),
        "walk_param_theta_p01": float(np.quantile(walk_params[:, 2], 0.01)),
        "walk_param_theta_p50": float(np.quantile(walk_params[:, 2], 0.50)),
        "walk_param_theta_p99": float(np.quantile(walk_params[:, 2], 0.99)),
        "target_rel_abs_x_p10": float(np.quantile(np.abs(rel_x), 0.10)),
        "target_rel_abs_y_p10": float(np.quantile(np.abs(rel_y), 0.10)),
        "target_rel_abs_theta_p10_deg": float(np.degrees(np.quantile(np.abs(rel_theta), 0.10))),
    }


def _sample_indices(rng: np.random.Generator, n: int, batch_size: int) -> np.ndarray:
    return rng.integers(0, n, size=batch_size)


def repair(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_arg = Path(args.output).expanduser()
    output_path = (ROOT / output_arg).resolve() if not output_arg.is_absolute() else output_arg.resolve()
    device = torch.device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    original_state = checkpoint["policy_state_dict"]
    actor = SkillRepairActor(original_state).to(device)
    reference = ActorExport({key: value.detach().cpu() for key, value in original_state.items()}).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    for name, parameter in actor.named_parameters():
        parameter.requires_grad_(name.startswith("param_head"))

    active_mask = _stage4_mask(device)
    walk = SKILL_ORDER.index("walk")
    dribble = SKILL_ORDER.index("dribble")
    rng = np.random.default_rng(args.seed)

    obs_pos, actions_pos, weights_pos = _load_dataset(dataset_path)
    skill_targets = np.clip(np.rint(actions_pos[:, 0]).astype(np.int64), 0, len(SKILL_ORDER) - 1)
    param_bc_mask = (skill_targets == walk) | (skill_targets == dribble)
    obs_bc = obs_pos[param_bc_mask]
    actions_bc = actions_pos[param_bc_mask]
    weights_bc = weights_pos[param_bc_mask]

    synthetic_no_finish = _make_synthetic_no_finish_walk_states(args.synthetic_samples, args.seed + 17)
    synthetic_near_dead = _make_near_dead_walk_states(args.synthetic_samples, args.seed + 23)
    synthetic_params = np.zeros((args.synthetic_samples, 4), dtype=np.float32)

    optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)

    before = {
        "dataset_bc": _param_metrics(actor, obs_bc, device=device, active_mask=active_mask),
        "synthetic_no_finish": _param_metrics(actor, synthetic_no_finish, device=device, active_mask=active_mask),
        "synthetic_near_dead": _param_metrics(actor, synthetic_near_dead, device=device, active_mask=active_mask),
    }

    for step in range(1, args.steps + 1):
        bc_idx = _sample_indices(rng, len(obs_bc), args.batch_size)
        syn_idx = _sample_indices(rng, len(synthetic_near_dead), args.batch_size)

        obs_batch = torch.as_tensor(obs_bc[bc_idx], dtype=torch.float32, device=device)
        param_target = torch.as_tensor(actions_bc[bc_idx, 1:5], dtype=torch.float32, device=device)
        weight_batch = torch.as_tensor(weights_bc[bc_idx], dtype=torch.float32, device=device)
        syn_batch = torch.as_tensor(synthetic_near_dead[syn_idx], dtype=torch.float32, device=device)
        syn_target = torch.as_tensor(synthetic_params[syn_idx], dtype=torch.float32, device=device)

        _, params = actor(obs_batch)
        bc_loss_raw = F.smooth_l1_loss(params, param_target, reduction="none").mean(dim=-1)
        bc_loss = (bc_loss_raw * weight_batch / torch.clamp(weight_batch.mean(), min=1e-6)).mean()

        _, syn_params = actor(syn_batch)
        syn_loss = F.smooth_l1_loss(syn_params[:, 1:3], syn_target[:, 1:3])

        with torch.no_grad():
            _, ref_params = reference(obs_batch)
        keep_loss = F.smooth_l1_loss(params, ref_params)

        loss = bc_loss + args.synthetic_weight * syn_loss + args.keep_weight * keep_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps:
            print(
                f"[param-repair] step={step} loss={loss.item():.5f} "
                f"bc={bc_loss.item():.5f} syn={syn_loss.item():.5f} keep={keep_loss.item():.5f}",
                flush=True,
            )

    repaired_state = dict(original_state)
    repaired_state["param_head.weight"] = actor.param_head.weight.detach().cpu()
    repaired_state["param_head.bias"] = actor.param_head.bias.detach().cpu()
    repaired = dict(checkpoint)
    repaired["policy_state_dict"] = repaired_state
    repaired["param_repair"] = {
        "source_checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "obs_size": 26,
        "steps": int(args.steps),
        "synthetic_samples": int(args.synthetic_samples),
        "method": "param-head BC repair for walk/dribble residuals plus near-dead walk states",
    }

    after = {
        "dataset_bc": _param_metrics(actor, obs_bc, device=device, active_mask=active_mask),
        "synthetic_no_finish": _param_metrics(actor, synthetic_no_finish, device=device, active_mask=active_mask),
        "synthetic_near_dead": _param_metrics(actor, synthetic_near_dead, device=device, active_mask=active_mask),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(repaired, output_path)
    summary = {
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "before": before,
        "after": after,
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def export_compat(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = ActorExport(checkpoint["policy_state_dict"]).eval()
    dummy = torch.zeros(1, 26, dtype=torch.float32)
    torch.onnx.export(
        actor,
        dummy,
        str(output_path),
        input_names=["obs"],
        output_names=["skill_logits", "param_mean"],
        opset_version=11,
        dynamic_axes=None,
        dynamo=False,
    )
    manifest = {
        "format": "onnx",
        "obs_size_input": 26,
        "obs_size_env": 26,
        "obs_size_model_trained": 26,
        "skill_order": list(SKILL_ORDER),
        "active_skills": ["stand", "walk", "dribble", "shoot"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "normalization_contract": "ppo_obs_v26_hsl2026",
        "deterministic": True,
        "compatibility": "SabanaHerons CompiledNN static batch opset11",
        "repair": "stand_repair + param_repair",
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline obs26 repair for PPO walk/dribble continuous residuals.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--synthetic-samples", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--synthetic-weight", type=float, default=2.0)
    parser.add_argument("--keep-weight", type=float, default=0.02)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--export-compat", action="store_true")
    parser.add_argument("--onnx-output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    repair(parsed)
    if parsed.export_compat:
        out = Path(parsed.onnx_output) if parsed.onnx_output else Path(parsed.output).with_name("ppo_striker_hsl2026_param_repair_compat.onnx")
        if not out.is_absolute():
            out = ROOT / out
        export_compat(Path(parsed.output).resolve(), out.resolve())
