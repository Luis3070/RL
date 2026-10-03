from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from rlfootball.core.config import ROOT_DIR, TRAIN
from rlfootball.core.goalkeeper_action import GK_ACTION_PARAM_MASKS, GK_SKILL_ORDER
from rlfootball.teachers.goalkeeper_teacher import goalkeeper_skill_mask_torch
from rlfootball.training.train_goalkeeper_bc import GoalkeeperPolicy


RUNS_DIR = ROOT_DIR / "runs"


@dataclass(frozen=True)
class GKPPOConfig:
    dataset: Path
    bc_checkpoint: Path
    output_root: Path = RUNS_DIR / "goalkeeper_ppo"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 260619
    total_steps: int = 120_000
    rollout_size: int = 4096
    minibatch_size: int = 512
    update_epochs: int = 4
    learning_rate: float = 8e-5
    weight_decay: float = 1e-5
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_ratio: float = 0.15
    entropy_coef: float = 0.012
    value_coef: float = 0.50
    param_coef: float = 0.20
    max_grad_norm: float = 0.75
    checkpoint_interval: int = 30_000
    eval_samples: int = 8192


class GoalkeeperPPOActor(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        self.skill_head = nn.Linear(hidden_size, len(GK_SKILL_ORDER))
        self.param_head = nn.Linear(hidden_size, 4)
        self.log_std = nn.Parameter(torch.full((4,), -1.15))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.skill_head(h)
        params = torch.tanh(self.param_head(h))
        logits, params = _apply_slow_corner_dribble_head(obs, logits, params)
        return logits, params, self.log_std.clamp(-3.0, 0.5)


def _apply_slow_corner_dribble_head(
    obs: torch.Tensor,
    logits: torch.Tensor,
    params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The scripted head is coupled to the old 021627 weights; closed-loop-trained
    # candidates are head-free. RL_GK_DISABLE_HEAD=1 turns it off everywhere.
    import os as _os
    if _os.getenv("RL_GK_DISABLE_HEAD") == "1":
        return logits, params
    speed = torch.sqrt(torch.clamp(obs[:, 7] * obs[:, 7] + obs[:, 8] * obs[:, 8], min=0.0))
    high_corner = torch.abs(obs[:, 21]) >= 0.22
    central_carry = torch.abs(obs[:, 4]) <= 0.12
    slow = speed <= 0.30
    rolling = obs[:, 19] >= 0.5
    fresh = obs[:, 27] >= 0.5
    stable = obs[:, 37] <= 0.4
    in_front = obs[:, 3] > 0.0
    controlled = obs[:, 25] >= 0.5
    slow_corner = slow & high_corner & central_carry & rolling & fresh & stable & in_front
    walk = slow_corner & (~controlled)
    clear = slow_corner & controlled
    if bool(torch.any(walk | clear)):
        logits = logits.clone()
        params = params.clone()
        logits[:, 1] = torch.where(walk, logits[:, 1] + 80.0, logits[:, 1])
        logits[:, 9] = torch.where(clear, logits[:, 9] + 80.0, logits[:, 9])
        params[:, 0] = torch.where(walk, torch.full_like(params[:, 0], -3700.0 / 4500.0), params[:, 0])
        params[:, 1] = torch.where(walk, torch.clamp(obs[:, 21] * (1200.0 / 3000.0), -1.0, 1.0), params[:, 1])
        params[:, 2] = torch.where(walk, torch.zeros_like(params[:, 2]), params[:, 2])
    return logits, params


class GoalkeeperValue(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def _latest_dataset() -> Path:
    candidates = sorted(
        (ROOT_DIR / "runs" / "teacher_goalkeeper_dataset").glob("dataset_gk_*/samples.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No GK teacher dataset found")
    return candidates[0]


def _latest_bc_checkpoint() -> Path:
    candidates = sorted(
        (ROOT_DIR / "runs" / "goalkeeper_bc").glob("bc_gk_*/checkpoint_gk_bc_step*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No GK BC checkpoint found")
    return candidates[0]


def _resolve(path: str | None, fallback) -> Path:
    if path:
        p = Path(path).expanduser()
        return (ROOT_DIR / p).resolve() if not p.is_absolute() else p.resolve()
    return fallback()


def _param_masks(skill_ids: torch.Tensor) -> torch.Tensor:
    masks = torch.as_tensor(
        [GK_ACTION_PARAM_MASKS[name] for name in GK_SKILL_ORDER],
        dtype=torch.float32,
        device=skill_ids.device,
    )
    return masks[skill_ids]


def _load_bc(actor: GoalkeeperPPOActor, checkpoint: Path, device: torch.device) -> dict[str, object]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["policy_state_dict"]
    missing, unexpected = actor.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if key != "log_std"]
    if unexpected:
        raise RuntimeError(f"Unexpected BC keys for PPO actor: {unexpected}")
    return {"checkpoint": str(checkpoint), "missing": list(missing)}


def _masked_logits(skill_logits: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    mask = goalkeeper_skill_mask_torch(obs).to(dtype=skill_logits.dtype, device=skill_logits.device)
    return skill_logits.masked_fill(mask <= 0.0, -1.0e9)


def _act(
    actor: GoalkeeperPPOActor,
    value: GoalkeeperValue,
    obs: torch.Tensor,
    rng: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    del rng
    skill_logits, param_mean, log_std = actor(obs)
    masked = _masked_logits(skill_logits, obs)
    skill_dist = Categorical(logits=masked)
    skill = skill_dist.sample()
    param_std = log_std.exp().expand_as(param_mean)
    param_dist = Normal(param_mean, param_std)
    raw_params = param_dist.rsample()
    params = torch.clamp(raw_params, -1.0, 1.0)
    p_mask = _param_masks(skill)
    param_log_prob = (param_dist.log_prob(raw_params) * p_mask).sum(dim=-1)
    entropy = skill_dist.entropy() + (param_dist.entropy() * p_mask).sum(dim=-1)
    return {
        "skill": skill,
        "params": params * p_mask,
        "raw_params": raw_params,
        "log_prob": skill_dist.log_prob(skill) + param_log_prob,
        "entropy": entropy,
        "value": value(obs),
        "skill_logits": skill_logits,
    }


def _evaluate_actions(
    actor: GoalkeeperPPOActor,
    value: GoalkeeperValue,
    obs: torch.Tensor,
    skill: torch.Tensor,
    raw_params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    skill_logits, param_mean, log_std = actor(obs)
    masked = _masked_logits(skill_logits, obs)
    skill_dist = Categorical(logits=masked)
    param_std = log_std.exp().expand_as(param_mean)
    param_dist = Normal(param_mean, param_std)
    p_mask = _param_masks(skill)
    log_prob = skill_dist.log_prob(skill) + (param_dist.log_prob(raw_params) * p_mask).sum(dim=-1)
    entropy = skill_dist.entropy() + (param_dist.entropy() * p_mask).sum(dim=-1)
    return log_prob, entropy, value(obs)


def _reward(
    chosen_skill: torch.Tensor,
    chosen_params: torch.Tensor,
    teacher_actions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_skill = torch.clamp(teacher_actions[:, 0].round().long(), 0, len(GK_SKILL_ORDER) - 1)
    teacher_params = torch.clamp(teacher_actions[:, 1:], -1.0, 1.0)
    exact = (chosen_skill == teacher_skill).to(torch.float32)
    teacher_names = np.asarray(GK_SKILL_ORDER)

    chosen_np = chosen_skill.detach().cpu().numpy()
    teacher_np = teacher_skill.detach().cpu().numpy()
    both_intercept = torch.as_tensor(
        [
            teacher_names[int(c)].startswith("intercept_") and teacher_names[int(t)].startswith("intercept_")
            for c, t in zip(chosen_np, teacher_np)
        ],
        dtype=torch.float32,
        device=chosen_skill.device,
    )
    side_ok = torch.sign(chosen_params[:, 1]) == torch.sign(teacher_params[:, 1])
    side_bonus = (both_intercept * side_ok.to(torch.float32)) * 0.20

    p_mask = _param_masks(teacher_skill)
    param_l1 = (torch.abs(chosen_params - teacher_params) * p_mask).sum(dim=-1) / torch.clamp(p_mask.sum(dim=-1), min=1.0)
    active = (p_mask.sum(dim=-1) > 0.0).to(torch.float32)
    reward = exact * 1.00 + side_bonus - active * 0.20 * param_l1 - (1.0 - exact) * 0.35
    return reward, {
        "exact": float(exact.mean().detach().cpu()),
        "intercept_side_bonus": float(side_bonus.mean().detach().cpu()),
        "param_l1": float((param_l1 * active).sum().detach().cpu() / torch.clamp(active.sum(), min=1.0).detach().cpu()),
        "reward": float(reward.mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate(actor: GoalkeeperPPOActor, observations: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor, samples: int, rng: np.random.Generator) -> dict[str, object]:
    actor.eval()
    valid = torch.nonzero(weights > 0.0, as_tuple=False).view(-1).detach().cpu().numpy()
    idx_np = rng.choice(valid, size=min(samples, len(valid)), replace=False)
    idx = torch.as_tensor(idx_np, dtype=torch.long, device=observations.device)
    obs = observations[idx]
    target = torch.clamp(actions[idx, 0].round().long(), 0, len(GK_SKILL_ORDER) - 1)
    logits, params, _ = actor(obs)
    pred = torch.argmax(_masked_logits(logits, obs), dim=-1)
    p_mask = _param_masks(target)
    param_l1 = (torch.abs(params - actions[idx, 1:]) * p_mask).sum(dim=-1) / torch.clamp(p_mask.sum(dim=-1), min=1.0)
    active = p_mask.sum(dim=-1) > 0.0
    by_skill = {}
    for skill_id, name in enumerate(GK_SKILL_ORDER):
        m = target == skill_id
        if bool(m.any()):
            by_skill[name] = {
                "samples": int(m.sum().item()),
                "acc": float((pred[m] == target[m]).to(torch.float32).mean().item()),
            }
    actor.train()
    return {
        "skill_acc": float((pred == target).to(torch.float32).mean().item()),
        "param_l1": float(param_l1[active].mean().item()) if bool(active.any()) else 0.0,
        "by_skill": by_skill,
    }


def train(cfg: GKPPOConfig) -> Path:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    data = np.load(cfg.dataset, allow_pickle=True)
    observations_np = data["observations"].astype(np.float32)
    actions_np = data["actions"].astype(np.float32)
    weights_np = data["weights"].astype(np.float32)
    valid_idx = np.flatnonzero(weights_np > 0.0)
    if valid_idx.size == 0:
        raise RuntimeError("GK PPO dataset has zero positive rows")
    sample_probs = weights_np[valid_idx].astype(np.float64)
    sample_probs /= max(float(sample_probs.sum()), 1e-12)

    observations = torch.as_tensor(observations_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)

    actor = GoalkeeperPPOActor(observations_np.shape[1], TRAIN.hidden_size).to(device)
    value = GoalkeeperValue(observations_np.shape[1], TRAIN.hidden_size).to(device)
    load_report = _load_bc(actor, cfg.bc_checkpoint, device)
    opt = torch.optim.AdamW(
        list(actor.parameters()) + list(value.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    out = cfg.output_root / f"ppo_gk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    train_rows: list[dict[str, float]] = []
    eval_rows: list[dict[str, object]] = []
    global_step = 0
    print(
        f"[gk-ppo] dataset={cfg.dataset} bc={cfg.bc_checkpoint} valid={len(valid_idx)} "
        f"steps={cfg.total_steps} rollout={cfg.rollout_size} device={device} output={out}",
        flush=True,
    )

    while global_step < cfg.total_steps:
        n = min(cfg.rollout_size, cfg.total_steps - global_step)
        idx_np = rng.choice(valid_idx, size=n, replace=True, p=sample_probs)
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        obs_b = observations[idx]
        teacher_b = actions[idx]
        with torch.no_grad():
            sampled = _act(actor, value, obs_b)
            rewards, reward_metrics = _reward(sampled["skill"], sampled["params"], teacher_b)
            returns = rewards
            advantages = returns - sampled["value"]
            advantages = (advantages - advantages.mean()) / torch.clamp(advantages.std(unbiased=False), min=1e-6)

        order = np.arange(n)
        epoch_metrics = []
        for _ in range(cfg.update_epochs):
            rng.shuffle(order)
            for start in range(0, n, cfg.minibatch_size):
                mb_np = order[start:start + cfg.minibatch_size]
                mb = torch.as_tensor(mb_np, dtype=torch.long, device=device)
                new_log_prob, entropy, new_value = _evaluate_actions(
                    actor,
                    value,
                    obs_b[mb],
                    sampled["skill"][mb],
                    sampled["raw_params"][mb],
                )
                ratio = torch.exp(new_log_prob - sampled["log_prob"][mb])
                adv = advantages[mb]
                policy_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv,
                ).mean()
                value_loss = F.mse_loss(new_value, returns[mb])
                teacher_skill = torch.clamp(teacher_b[mb, 0].round().long(), 0, len(GK_SKILL_ORDER) - 1)
                logits, param_mean, _ = actor(obs_b[mb])
                ce = F.cross_entropy(_masked_logits(logits, obs_b[mb]), teacher_skill)
                p_mask = _param_masks(teacher_skill)
                param_loss = ((F.smooth_l1_loss(param_mean, teacher_b[mb, 1:], reduction="none") * p_mask).sum(dim=-1) / torch.clamp(p_mask.sum(dim=-1), min=1.0)).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean() + cfg.param_coef * (ce + param_loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(value.parameters()), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    approx_kl = (sampled["log_prob"][mb] - new_log_prob).mean()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_ratio).to(torch.float32).mean()
                    epoch_metrics.append({
                        "loss": float(loss.detach().cpu()),
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "entropy": float(entropy.mean().detach().cpu()),
                        "approx_kl": float(approx_kl.detach().cpu()),
                        "clipfrac": float(clipfrac.detach().cpu()),
                    })

        global_step += n
        row = {
            "step": int(global_step),
            **reward_metrics,
            "loss": float(np.mean([m["loss"] for m in epoch_metrics])),
            "policy_loss": float(np.mean([m["policy_loss"] for m in epoch_metrics])),
            "value_loss": float(np.mean([m["value_loss"] for m in epoch_metrics])),
            "entropy": float(np.mean([m["entropy"] for m in epoch_metrics])),
            "approx_kl": float(np.mean([m["approx_kl"] for m in epoch_metrics])),
            "clipfrac": float(np.mean([m["clipfrac"] for m in epoch_metrics])),
        }
        train_rows.append(row)
        if global_step == n or global_step % cfg.checkpoint_interval < n or global_step >= cfg.total_steps:
            val = evaluate(actor, observations, actions, weights, cfg.eval_samples, rng)
            eval_rows.append({"step": int(global_step), **val})
            print(
                f"[gk-ppo] step={global_step}/{cfg.total_steps} reward={row['reward']:.3f} "
                f"exact={row['exact']:.3f} val_acc={val['skill_acc']:.3f} "
                f"param_l1={val['param_l1']:.4f} kl={row['approx_kl']:.5f}",
                flush=True,
            )

    final_eval = evaluate(actor, observations, actions, weights, cfg.eval_samples, rng)
    ckpt = out / f"checkpoint_gk_ppo_step{global_step}.pt"
    torch.save(
        {
            "timestep": int(global_step),
            "stage_id": 91,
            "policy_state_dict": actor.state_dict(),
            "value_state_dict": value.state_dict(),
            "eval_summary": final_eval,
            "ppo_metadata": {
                "dataset_path": str(cfg.dataset),
                "bc_checkpoint_path": str(cfg.bc_checkpoint),
                "skill_order": list(GK_SKILL_ORDER),
                "obs_size": int(observations_np.shape[1]),
                "config": {**asdict(cfg), "dataset": str(cfg.dataset), "bc_checkpoint": str(cfg.bc_checkpoint), "output_root": str(cfg.output_root)},
                "bc_load": load_report,
                "training_mode": "contextual_masked_ppo_teacher_corpus",
            },
        },
        ckpt,
    )
    report = {
        "checkpoint_path": str(ckpt),
        "final_eval": final_eval,
        "train_metrics": train_rows,
        "eval_metrics": eval_rows,
    }
    (out / "ppo_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint_path": str(ckpt), "final_eval": final_eval}, indent=2, sort_keys=True), flush=True)
    return ckpt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--bc-checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--rollout-size", type=int, default=4096)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default=str(RUNS_DIR / "goalkeeper_ppo"))
    args = parser.parse_args()
    cfg = GKPPOConfig(
        dataset=_resolve(args.dataset, _latest_dataset),
        bc_checkpoint=_resolve(args.bc_checkpoint, _latest_bc_checkpoint),
        total_steps=int(args.steps),
        rollout_size=int(args.rollout_size),
        minibatch_size=int(args.minibatch_size),
        update_epochs=int(args.update_epochs),
        device=str(args.device),
        output_root=Path(args.output_root).expanduser().resolve(),
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
