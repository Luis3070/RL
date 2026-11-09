from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from action import ACTION_PARAM_MASKS, SKILL_ORDER
from defender_policy import ACTIVE_DEFENDER_SKILLS, DefenderActor, masked_defender_logits
from generate_defender_dataset import generate_samples


def _param_mask(skill_ids: torch.Tensor) -> torch.Tensor:
    masks = torch.as_tensor([ACTION_PARAM_MASKS[name] for name in SKILL_ORDER], dtype=torch.float32, device=skill_ids.device)
    return masks[skill_ids]


def _load_or_generate_dataset(path: str, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path:
        data = np.load(Path(path), allow_pickle=True)
        return data["observations"].astype(np.float32), data["actions"].astype(np.float32), data["weights"].astype(np.float32)
    return generate_samples(samples, seed)


def _load_policy(path: str, device: torch.device) -> dict[str, torch.Tensor] | None:
    if not path:
        return None
    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=device, weights_only=False)
    if "policy_state_dict" not in checkpoint:
        raise KeyError(f"{path} does not contain policy_state_dict")
    return checkpoint["policy_state_dict"]


def _fresh_value_state(seed: int) -> dict[str, torch.Tensor]:
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(26, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 1),
    )
    state = {f"net.{key}": value for key, value in net.state_dict().items()}
    torch.random.set_rng_state(generator_state)
    return state


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    obs_np, actions_np, weights_np = _load_or_generate_dataset(args.dataset, args.samples, args.seed)

    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
    skill_targets = torch.clamp(actions[:, 0].round().long(), 0, len(SKILL_ORDER) - 1)
    param_targets = torch.clamp(actions[:, 1:], -1.0, 1.0)

    model = DefenderActor(_load_policy(args.init_checkpoint, device)).to(device)
    reference_model = None
    if args.reference_checkpoint:
        reference_model = DefenderActor(_load_policy(args.reference_checkpoint, device)).to(device)
        reference_model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = Path(args.output_dir) if args.output_dir else Path("runs/defender_bc") / f"defender_bc_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    n = obs.shape[0]
    last_metrics = {}
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        batch_obs = obs[idx]
        batch_skill = skill_targets[idx]
        batch_params = param_targets[idx]
        batch_weights = weights[idx]

        raw_logits, param_mean = model(batch_obs)
        logits = masked_defender_logits(raw_logits, batch_obs)
        ce = F.cross_entropy(logits, batch_skill, reduction="none")
        masks = _param_mask(batch_skill)
        param_loss = (F.smooth_l1_loss(param_mean, batch_params, reduction="none") * masks).sum(dim=1) / torch.clamp(masks.sum(dim=1), min=1.0)
        norm_w = batch_weights / torch.clamp(batch_weights.mean(), min=1e-6)
        loss = (ce * norm_w).mean() + args.param_weight * (param_loss * norm_w).mean()
        distill_loss = torch.zeros((), dtype=loss.dtype, device=device)
        if reference_model is not None and args.distill_weight > 0.0:
            with torch.no_grad():
                ref_logits, ref_params = reference_model(batch_obs)
            distill_logits = F.mse_loss(logits.clamp(-50.0, 50.0), ref_logits.clamp(-50.0, 50.0))
            distill_params = F.mse_loss(param_mean, ref_params)
            distill_loss = distill_logits + args.param_distill_weight * distill_params
            loss = loss + args.distill_weight * distill_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_interval == 0 or step == 1:
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                acc = (pred == batch_skill).float().mean().item()
            last_metrics = {
                "loss": float(loss.item()),
                "skill_acc": float(acc),
                "ce_loss": float((ce * norm_w).mean().item()),
                "param_loss": float((param_loss * norm_w).mean().item()),
                "distill_loss": float(distill_loss.item()),
            }
            print(
                f"step={step} loss={last_metrics['loss']:.5f} acc={acc:.3f} "
                f"distill={last_metrics['distill_loss']:.5f}"
            )

    checkpoint_path = run_dir / f"checkpoint_defender_bc_step{args.steps}.pt"
    payload = {
        "policy_state_dict": model.state_dict(),
        "value_state_dict": _fresh_value_state(args.seed),
        "optimizer_state_dict": opt.state_dict(),
        "timestep": int(args.steps),
        "stage_id": 4,
        "eval_summary": {
            "source": "synthetic_defender_bc",
            "final_metrics": last_metrics,
            "init_checkpoint": str(Path(args.init_checkpoint).expanduser().resolve()) if args.init_checkpoint else "",
            "reference_checkpoint": str(Path(args.reference_checkpoint).expanduser().resolve()) if args.reference_checkpoint else "",
            "distill_weight": float(args.distill_weight),
        },
        "meta": {
            "role": "defender",
            "obs_size": 26,
            "skill_order": list(SKILL_ORDER),
            "active_skills": list(ACTIVE_DEFENDER_SKILLS),
            "normalization_contract": "ppo_obs_v26_hsl2026",
            "steps": args.steps,
            "seed": args.seed,
            "train_args": vars(args),
        },
    }
    torch.save(payload, checkpoint_path)
    manifest = {
        **payload["meta"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    (run_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synthetic BC obs26 defender skill selector.")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--param-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=260611)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--reference-checkpoint", default="")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--param-distill-weight", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(train(parse_args()))
