from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rlfootball.core.action import SKILL_ORDER, skill_mask
from rlfootball.curriculum.curriculum import get_stage_spec
from rlfootball.core.observation import ObsIndex
from tools.export.export_ppo_policy import ActorExport
from rlfootball.training.train import gate_skill_mask_from_obs


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt"
DEFAULT_DATASET = ROOT / "runs/fine_tuning_production_dataset/production_replay_stage4/samples.npz"
DEFAULT_OUTPUT = ROOT / "runs/production_stand_repair_v1/checkpoint_stage4_stand_repair.pt"

FIELD_X_HALF = 4500.0
FIELD_Y_HALF = 3000.0
TIME_NORM_MS = 5000.0


class SkillRepairActor(torch.nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(26, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
        )
        self.skill_head = torch.nn.Linear(128, len(SKILL_ORDER))
        self.param_head = torch.nn.Linear(128, 4)
        self.load_state_dict(
            {
                "trunk.0.weight": state_dict["trunk.0.weight"],
                "trunk.0.bias": state_dict["trunk.0.bias"],
                "trunk.2.weight": state_dict["trunk.2.weight"],
                "trunk.2.bias": state_dict["trunk.2.bias"],
                "skill_head.weight": state_dict["skill_head.weight"],
                "skill_head.bias": state_dict["skill_head.bias"],
                "param_head.weight": state_dict["param_head.weight"],
                "param_head.bias": state_dict["param_head.bias"],
            }
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.skill_head(features), torch.tanh(self.param_head(features))


def _stage4_mask(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        skill_mask(get_stage_spec(4).active_skill_names),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def _masked_logits(logits: torch.Tensor, obs: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill((gate_skill_mask_from_obs(obs) * active_mask) <= 0.0, -1.0e9)


def _load_positive_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    obs = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    weights = data["weights"].astype(np.float32)
    if obs.ndim != 2 or obs.shape[1] != 26:
        raise ValueError(f"Expected observations [N,26], got {obs.shape}")
    positive = weights > 0.0
    return obs[positive], actions[positive], weights[positive]


def _make_synthetic_no_finish_walk_states(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obs = np.zeros((n, 26), dtype=np.float32)

    d_ball = rng.uniform(500.0, 1300.0, size=n)
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
    obs[:, ObsIndex.ball_vel_x] = rng.uniform(-0.05, 0.05, size=n)
    obs[:, ObsIndex.ball_vel_y] = rng.uniform(-0.05, 0.05, size=n)
    obs[:, ObsIndex.time_since_ball_seen] = np.clip(rng.uniform(0.0, 1000.0, size=n) / TIME_NORM_MS, 0.0, 1.0)
    obs[:, ObsIndex.time_since_ball_disappeared] = obs[:, ObsIndex.time_since_ball_seen]
    obs[:, ObsIndex.ball_seen_percentage] = rng.uniform(0.0, 0.08, size=n)
    obs[:, ObsIndex.ball_consistent_with_game_state] = 1.0
    obs[:, ObsIndex.can_score_now] = 0.0
    obs[:, ObsIndex.shot_quality_no_obstacles] = rng.uniform(0.01, 0.55, size=n)
    obs[:, ObsIndex.shot_opening_with_obstacles] = rng.uniform(0.01, 0.55, size=n)
    obs[:, ObsIndex.pass_options_count] = rng.uniform(0.0, 0.8, size=n)
    obs[:, ObsIndex.nearest_teammate_dist] = rng.uniform(0.2, 1.0, size=n)
    obs[:, ObsIndex.nearest_opponent_dist] = rng.uniform(0.2, 1.0, size=n)
    obs[:, ObsIndex.nearest_uncertain_obstacle_dist] = rng.uniform(0.2, 1.0, size=n)
    obs[:, ObsIndex.nearest_teammate_front_dist] = rng.uniform(0.2, 1.0, size=n)
    obs[:, ObsIndex.nearest_opponent_front_dist] = rng.uniform(0.2, 1.0, size=n)
    obs[:, ObsIndex.nearest_uncertain_front_dist] = rng.uniform(0.2, 1.0, size=n)
    obs[:, ObsIndex.shoot_armed] = 0.0
    obs[:, ObsIndex.dribble_armed] = 0.0
    obs[:, ObsIndex.shoot_arm_progress] = 0.0
    return obs


def _sample_indices(rng: np.random.Generator, n: int, batch_size: int) -> np.ndarray:
    return rng.integers(0, n, size=batch_size)


def _policy_metrics(
    actor: torch.nn.Module,
    obs_np: np.ndarray,
    *,
    device: torch.device,
    active_mask: torch.Tensor,
) -> dict[str, float]:
    actor.eval()
    stand = SKILL_ORDER.index("stand")
    walk = SKILL_ORDER.index("walk")
    with torch.no_grad():
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        logits, _ = actor(obs)
        masked = _masked_logits(logits, obs, active_mask)
        pred = masked.argmax(dim=-1)
        margin = logits[:, walk] - logits[:, stand]
        probs = torch.softmax(masked, dim=-1)
    actor.train()
    return {
        "n": float(obs.shape[0]),
        "stand_rate": float((pred == stand).float().mean().cpu().item()),
        "walk_rate": float((pred == walk).float().mean().cpu().item()),
        "p_stand_mean": float(probs[:, stand].mean().cpu().item()),
        "p_walk_mean": float(probs[:, walk].mean().cpu().item()),
        "walk_minus_stand_margin_p01": float(torch.quantile(margin, 0.01).cpu().item()),
        "walk_minus_stand_margin_p10": float(torch.quantile(margin, 0.10).cpu().item()),
        "walk_minus_stand_margin_mean": float(margin.mean().cpu().item()),
    }


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
        parameter.requires_grad_(name.startswith("skill_head"))

    active_mask = _stage4_mask(device)
    stand = SKILL_ORDER.index("stand")
    walk = SKILL_ORDER.index("walk")
    rng = np.random.default_rng(args.seed)

    obs_pos, actions_pos, weights_pos = _load_positive_dataset(dataset_path)
    target_pos = np.clip(np.rint(actions_pos[:, 0]).astype(np.int64), 0, len(SKILL_ORDER) - 1)
    no_finish_pos = (obs_pos[:, ObsIndex.shoot_armed] <= 0.5) & (obs_pos[:, ObsIndex.dribble_armed] <= 0.5)
    synthetic = _make_synthetic_no_finish_walk_states(args.synthetic_samples, args.seed + 17)
    synthetic_targets = np.full((len(synthetic),), walk, dtype=np.int64)

    optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)

    before = {
        "dataset_no_finish_positive": _policy_metrics(actor, obs_pos[no_finish_pos], device=device, active_mask=active_mask),
        "synthetic_no_finish": _policy_metrics(actor, synthetic, device=device, active_mask=active_mask),
    }

    for step in range(1, args.steps + 1):
        pos_idx = _sample_indices(rng, len(obs_pos), args.batch_size)
        syn_idx = _sample_indices(rng, len(synthetic), args.batch_size)

        obs_batch = torch.as_tensor(obs_pos[pos_idx], dtype=torch.float32, device=device)
        target_batch = torch.as_tensor(target_pos[pos_idx], dtype=torch.long, device=device)
        weight_batch = torch.as_tensor(weights_pos[pos_idx], dtype=torch.float32, device=device)
        syn_batch = torch.as_tensor(synthetic[syn_idx], dtype=torch.float32, device=device)
        syn_target = torch.as_tensor(synthetic_targets[syn_idx], dtype=torch.long, device=device)

        logits, _ = actor(obs_batch)
        masked = _masked_logits(logits, obs_batch, active_mask)
        ce = F.cross_entropy(masked, target_batch, reduction="none")
        ce_loss = (ce * weight_batch / torch.clamp(weight_batch.mean(), min=1e-6)).mean()

        syn_logits, _ = actor(syn_batch)
        syn_masked = _masked_logits(syn_logits, syn_batch, active_mask)
        syn_ce = F.cross_entropy(syn_masked, syn_target)
        margin_loss = F.relu(args.margin - (syn_logits[:, walk] - syn_logits[:, stand])).mean()

        with torch.no_grad():
            ref_logits, _ = reference(obs_batch)
            ref_masked = _masked_logits(ref_logits, obs_batch, active_mask)
            ref_probs = F.softmax(ref_masked, dim=-1)
        log_probs = F.log_softmax(masked, dim=-1)
        kl_keep = F.kl_div(log_probs, ref_probs, reduction="batchmean")

        loss = ce_loss + args.synthetic_weight * syn_ce + args.margin_weight * margin_loss + args.kl_weight * kl_keep
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps:
            print(
                f"[stand-repair] step={step} loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                f"syn_ce={syn_ce.item():.4f} margin={margin_loss.item():.4f} kl={kl_keep.item():.4f}",
                flush=True,
            )

    repaired_state = dict(original_state)
    repaired_state["skill_head.weight"] = actor.skill_head.weight.detach().cpu()
    repaired_state["skill_head.bias"] = actor.skill_head.bias.detach().cpu()
    repaired = dict(checkpoint)
    repaired["policy_state_dict"] = repaired_state
    repaired["stand_repair"] = {
        "source_checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "obs_size": 26,
        "steps": int(args.steps),
        "margin": float(args.margin),
        "synthetic_samples": int(args.synthetic_samples),
        "method": "skill-head-only CE+walk-margin repair for no-finish stand states",
    }

    after = {
        "dataset_no_finish_positive": _policy_metrics(actor, obs_pos[no_finish_pos], device=device, active_mask=active_mask),
        "synthetic_no_finish": _policy_metrics(actor, synthetic, device=device, active_mask=active_mask),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline obs26 repair for PPO stand-vs-walk no-finish states.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--synthetic-samples", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--synthetic-weight", type=float, default=0.8)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    repair(parse_args())
