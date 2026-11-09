from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from defender_policy import DefenderActor
from generate_defender_dataset import generate_samples


def _load_actor(path: Path, device: torch.device) -> DefenderActor:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "policy_state_dict" not in checkpoint:
        raise KeyError(f"{path} does not contain policy_state_dict")
    return DefenderActor(checkpoint["policy_state_dict"]).to(device).eval()


def compare(reference: Path, candidate: Path, *, samples: int, seed: int, device_name: str) -> dict[str, float]:
    device = torch.device(device_name)
    obs_np, _, _ = generate_samples(samples, seed)
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    ref = _load_actor(reference, device)
    cand = _load_actor(candidate, device)
    with torch.no_grad():
        ref_logits, ref_params = ref(obs)
        cand_logits, cand_params = cand(obs)
        ref_pred = torch.argmax(ref_logits, dim=1)
        cand_pred = torch.argmax(cand_logits, dim=1)
        logits_a = ref_logits.clamp(-50.0, 50.0)
        logits_b = cand_logits.clamp(-50.0, 50.0)
        param_abs = (ref_params - cand_params).abs()
    return {
        "samples": float(samples),
        "skill_argmax_match": float((ref_pred == cand_pred).float().mean().item()),
        "logit_mse_clamped": float(F.mse_loss(logits_b, logits_a).item()),
        "logit_max_abs_clamped": float((logits_a - logits_b).abs().max().item()),
        "param_mse": float(F.mse_loss(cand_params, ref_params).item()),
        "param_mean_abs": float(param_abs.mean().item()),
        "param_max_abs": float(param_abs.max().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two obs26 defender checkpoints on synthetic observations.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=260612)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = compare(
        Path(args.reference).expanduser().resolve(),
        Path(args.candidate).expanduser().resolve(),
        samples=args.samples,
        seed=args.seed,
        device_name=args.device,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
