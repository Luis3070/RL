from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlfootball.core.config import ROOT_DIR, TRAIN
from rlfootball.core.goalkeeper_action import GK_ACTION_PARAM_MASKS, GK_SKILL_ORDER
from rlfootball.teachers.goalkeeper_teacher import goalkeeper_skill_mask_torch


RUNS_DIR = ROOT_DIR / "runs"


@dataclass(frozen=True)
class GKBCConfig:
    dataset: Path
    init_checkpoint: Path | None = None
    output_root: Path = RUNS_DIR / "goalkeeper_bc"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    epochs: int = 36
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.12
    seed: int = 260618
    skill_loss_weight: float = 1.0
    param_loss_weight: float = 0.35


class GoalkeeperPolicy(nn.Module):
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
        with torch.no_grad():
            nn.init.orthogonal_(self.skill_head.weight, gain=0.01)
            nn.init.zeros_(self.skill_head.bias)
            nn.init.orthogonal_(self.param_head.weight, gain=0.01)
            nn.init.zeros_(self.param_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.skill_head(h), torch.tanh(self.param_head(h))


def _latest_dataset() -> Path:
    candidates = sorted((ROOT_DIR / "runs" / "teacher_goalkeeper_dataset").glob("dataset_*/samples.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No GK dataset found. Run teacher_goalkeeper_dataset.py first.")
    return candidates[0]


def _resolve_dataset(path: str | None) -> Path:
    if path:
        p = Path(path).expanduser()
        return (ROOT_DIR / p).resolve() if not p.is_absolute() else p.resolve()
    return _latest_dataset()


def _param_masks(skill_ids: torch.Tensor) -> torch.Tensor:
    masks = torch.as_tensor(
        [GK_ACTION_PARAM_MASKS[name] for name in GK_SKILL_ORDER],
        dtype=torch.float32,
        device=skill_ids.device,
    )
    return masks[skill_ids]


def _safe_logits(masked_logits: torch.Tensor, raw_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_ok = torch.gather(torch.isfinite(masked_logits), 1, targets.view(-1, 1)).squeeze(1)
    if bool(target_ok.all()):
        return masked_logits
    repaired = masked_logits.clone()
    bad = ~target_ok
    repaired[bad] = raw_logits[bad]
    return repaired


def _split(n: int, weights: np.ndarray, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    valid = np.flatnonzero(weights > 0.0)
    shuffled = rng.permutation(valid)
    val_n = max(1, int(round(len(shuffled) * frac))) if len(shuffled) > 1 else 0
    return shuffled[val_n:], shuffled[:val_n]


def _batch_indices(train_idx: np.ndarray, weights: np.ndarray, batch_size: int, rng: np.random.Generator):
    probs = weights[train_idx].astype(np.float64)
    probs = probs / max(float(probs.sum()), 1e-12)
    steps = max(1, math.ceil(len(train_idx) / batch_size))
    for _ in range(steps):
        yield rng.choice(train_idx, size=batch_size, replace=True, p=probs)


def _loss(
    policy: GoalkeeperPolicy,
    obs: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
    *,
    param_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    skill_targets = torch.clamp(actions[:, 0].round().long(), min=0, max=len(GK_SKILL_ORDER) - 1)
    param_targets = torch.clamp(actions[:, 1:], -1.0, 1.0)
    skill_logits, param_mean = policy(obs)
    mask = goalkeeper_skill_mask_torch(obs).to(dtype=skill_logits.dtype)
    masked = skill_logits.masked_fill(mask <= 0.0, -1.0e9)
    ce = F.cross_entropy(_safe_logits(masked, skill_logits, skill_targets), skill_targets, reduction="none")
    p_mask = _param_masks(skill_targets)
    p_raw = F.smooth_l1_loss(param_mean, param_targets, reduction="none") * p_mask
    p_den = torch.clamp(p_mask.sum(dim=-1), min=1.0)
    p_loss_each = p_raw.sum(dim=-1) / p_den
    active = (p_mask.sum(dim=-1) > 0.0).to(weights.dtype)
    w = weights / torch.clamp(weights.mean(), min=1e-6)
    skill_loss = (ce * w).mean()
    param_loss = (p_loss_each * active * w).sum() / torch.clamp((active * w).sum(), min=1.0)
    total = skill_loss + float(param_loss_weight) * param_loss
    with torch.no_grad():
        pred = torch.argmax(masked, dim=-1)
        acc = ((pred == skill_targets).to(torch.float32) * weights).sum() / torch.clamp(weights.sum(), min=1e-6)
    return total, {
        "loss": float(total.detach().cpu()),
        "skill_loss": float(skill_loss.detach().cpu()),
        "param_loss": float(param_loss.detach().cpu()),
        "skill_acc": float(acc.detach().cpu()),
    }


@torch.no_grad()
def _validate(
    policy: GoalkeeperPolicy,
    obs: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
    *,
    param_loss_weight: float,
) -> dict[str, object]:
    policy.eval()
    loss, metrics = _loss(policy, obs, actions, weights, param_loss_weight=param_loss_weight)
    targets = torch.clamp(actions[:, 0].round().long(), min=0, max=len(GK_SKILL_ORDER) - 1)
    logits, _ = policy(obs)
    pred = torch.argmax(logits.masked_fill(goalkeeper_skill_mask_torch(obs) <= 0.0, -1e9), dim=-1)
    by_skill = {}
    for idx, name in enumerate(GK_SKILL_ORDER):
        m = targets == idx
        if bool(m.any()):
            by_skill[name] = {
                "samples": int(m.sum().item()),
                "acc": float((pred[m] == targets[m]).to(torch.float32).mean().item()),
            }
    policy.train()
    return {**metrics, "by_skill": by_skill, "loss_tensor": float(loss.detach().cpu())}


def train(cfg: GKBCConfig) -> Path:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = np.load(cfg.dataset, allow_pickle=True)
    observations_np = data["observations"].astype(np.float32)
    actions_np = data["actions"].astype(np.float32)
    weights_np = data["weights"].astype(np.float32)
    train_idx, val_idx = _split(len(observations_np), weights_np, cfg.validation_fraction, rng)
    device = torch.device(cfg.device)
    policy = GoalkeeperPolicy(observations_np.shape[1], TRAIN.hidden_size).to(device)
    init_report = None
    if cfg.init_checkpoint is not None:
        ckpt = torch.load(cfg.init_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
        unexpected = [key for key in unexpected if key != "log_std"]
        if unexpected:
            raise RuntimeError(f"Unexpected init checkpoint keys for BC policy: {unexpected}")
        init_report = {"checkpoint": str(cfg.init_checkpoint), "missing": list(missing)}
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    observations = torch.as_tensor(observations_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)

    out = cfg.output_root / f"bc_gk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    step = 0
    print(f"[gk-bc] dataset={cfg.dataset} train={len(train_idx)} val={len(val_idx)} output={out}", flush=True)
    if init_report is not None:
        print(f"[gk-bc] init={init_report['checkpoint']} missing={init_report['missing']}", flush=True)
    for epoch in range(1, cfg.epochs + 1):
        epoch_rows = []
        for batch in _batch_indices(train_idx, weights_np, cfg.batch_size, rng):
            step += 1
            idx = torch.as_tensor(batch, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss, metrics = _loss(
                policy,
                observations[idx],
                actions[idx],
                weights[idx],
                param_loss_weight=cfg.param_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            epoch_rows.append(metrics)
        row = {
            "epoch": epoch,
            "step": step,
            "loss": float(np.mean([r["loss"] for r in epoch_rows])),
            "skill_acc": float(np.mean([r["skill_acc"] for r in epoch_rows])),
            "param_loss": float(np.mean([r["param_loss"] for r in epoch_rows])),
        }
        rows.append(row)
        if epoch == 1 or epoch == cfg.epochs or epoch % 6 == 0:
            print(f"[gk-bc] epoch={epoch}/{cfg.epochs} loss={row['loss']:.4f} skill_acc={row['skill_acc']:.3f}", flush=True)

    val = _validate(
        policy,
        observations[torch.as_tensor(val_idx, dtype=torch.long, device=device)],
        actions[torch.as_tensor(val_idx, dtype=torch.long, device=device)],
        weights[torch.as_tensor(val_idx, dtype=torch.long, device=device)],
        param_loss_weight=cfg.param_loss_weight,
    )
    ckpt = out / f"checkpoint_gk_bc_step{step}.pt"
    torch.save(
        {
            "timestep": int(step),
            "stage_id": 90,
            "policy_state_dict": policy.state_dict(),
            "eval_summary": val,
            "bc_metadata": {
                "dataset_path": str(cfg.dataset),
                "skill_order": list(GK_SKILL_ORDER),
                "obs_size": int(observations_np.shape[1]),
                "config": {
                    **asdict(cfg),
                    "dataset": str(cfg.dataset),
                    "init_checkpoint": None if cfg.init_checkpoint is None else str(cfg.init_checkpoint),
                    "output_root": str(cfg.output_root),
                },
                "init_load": init_report,
            },
        },
        ckpt,
    )
    report = {
        "checkpoint_path": str(ckpt),
        "dataset_path": str(cfg.dataset),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "validation": val,
        "metrics": rows,
    }
    (out / "bc_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint_path": str(ckpt), "validation": val}, indent=2, sort_keys=True), flush=True)
    return ckpt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default=str(RUNS_DIR / "goalkeeper_bc"))
    args = parser.parse_args()
    cfg = GKBCConfig(
        dataset=_resolve_dataset(args.dataset),
        init_checkpoint=None if not args.init_checkpoint else Path(args.init_checkpoint).expanduser().resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_root=Path(args.output_root).expanduser().resolve(),
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
