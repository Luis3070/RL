from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from goalkeeper_action import GK_ACTION_PARAM_MASKS, GK_SKILL_ORDER
from goalkeeper_teacher import goalkeeper_skill_mask_np
from train_goalkeeper_ppo import GoalkeeperPPOActor, _masked_logits
from config import TRAIN


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return (_REPO / p).resolve() if not p.is_absolute() else p.resolve()


def _load_actor(checkpoint: Path, obs_size: int, device: torch.device) -> GoalkeeperPPOActor:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    actor = GoalkeeperPPOActor(obs_size, TRAIN.hidden_size).to(device)
    actor.load_state_dict(ckpt["policy_state_dict"], strict=False)
    actor.eval()
    return actor


def _scenario_names(data) -> tuple[str, ...]:
    if "scenario_names" not in data.files:
        return ("unknown",)
    return tuple(str(item) for item in data["scenario_names"])


def _metrics_for(
    actor: GoalkeeperPPOActor,
    observations: np.ndarray,
    actions: np.ndarray,
    weights: np.ndarray,
    *,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    noise_std: float = 0.0,
    seed: int = 0,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    pred_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    masked_violations = 0
    finite_failures = 0
    max_abs_logit = 0.0
    entropy_values: list[float] = []
    param_l1_values: list[float] = []
    active_param_count = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        obs_np = observations[batch_idx].copy()
        if noise_std > 0.0:
            noise = rng.normal(0.0, noise_std, size=obs_np.shape).astype(np.float32)
            # Do not perturb binary gate/safety columns hard enough to invert labels.
            obs_np = np.clip(obs_np + noise, -1.25, 1.25).astype(np.float32)
            obs_np[:, 12] = observations[batch_idx, 12]
            obs_np[:, 19:29] = observations[batch_idx, 19:29]
            obs_np[:, 32] = observations[batch_idx, 32]
            obs_np[:, 36:38] = observations[batch_idx, 36:38]
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        target = np.clip(np.rint(actions[batch_idx, 0]).astype(np.int64), 0, len(GK_SKILL_ORDER) - 1)
        with torch.no_grad():
            logits, params, _ = actor(obs)
            masked = _masked_logits(logits, obs)
            probs = torch.softmax(masked, dim=-1)
            pred = torch.argmax(masked, dim=-1).detach().cpu().numpy()
            entropy_values.extend((-(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)).detach().cpu().numpy().tolist())
            finite_failures += int((~torch.isfinite(logits)).sum().item())
            max_abs_logit = max(max_abs_logit, float(torch.nan_to_num(logits).abs().max().detach().cpu().item()))
            params_np = params.detach().cpu().numpy()
        for row, skill_id in zip(obs_np, pred):
            mask = goalkeeper_skill_mask_np(row)
            if mask[int(skill_id)] <= 0.0:
                masked_violations += 1
        p_active = np.zeros(len(batch_idx), dtype=bool)
        p_l1 = np.zeros(len(batch_idx), dtype=np.float32)
        for i, skill_id in enumerate(target):
            name = GK_SKILL_ORDER[int(skill_id)]
            mask = np.asarray(GK_ACTION_PARAM_MASKS[name], dtype=np.float32)
            if float(mask.sum()) > 0.0:
                p_active[i] = True
                p_l1[i] = float((np.abs(params_np[i] - actions[batch_idx[i], 1:]) * mask).sum() / max(float(mask.sum()), 1.0))
        if p_active.any():
            param_l1_values.extend(p_l1[p_active].tolist())
            active_param_count += int(p_active.sum())
        pred_chunks.append(pred)
        target_chunks.append(target)
    pred_all = np.concatenate(pred_chunks)
    target_all = np.concatenate(target_chunks)
    skill_counts = Counter(GK_SKILL_ORDER[int(i)] for i in pred_all)
    exact = pred_all == target_all
    safe_idle = np.asarray(
        [
            GK_SKILL_ORDER[int(pred)] in {"stand", "walk_goalie_pose"}
            and GK_SKILL_ORDER[int(target)] in {"stand", "walk_goalie_pose"}
            for pred, target in zip(pred_all, target_all)
        ],
        dtype=bool,
    )
    semantic_ok = exact | safe_idle
    by_skill = {}
    for skill_id, name in enumerate(GK_SKILL_ORDER):
        m = target_all == skill_id
        if np.any(m):
            by_skill[name] = {
                "samples": int(np.count_nonzero(m)),
                "acc": float(np.mean(pred_all[m] == target_all[m])),
                "pred_counts": dict(sorted(Counter(GK_SKILL_ORDER[int(i)] for i in pred_all[m]).items())),
            }
    return {
        "samples": int(len(indices)),
        "skill_acc": float(np.mean(exact)) if len(exact) else 0.0,
        "semantic_skill_acc": float(np.mean(semantic_ok)) if len(semantic_ok) else 0.0,
        "safe_idle_substitutions": int(np.count_nonzero(safe_idle & ~exact)),
        "param_l1": float(np.mean(param_l1_values)) if param_l1_values else 0.0,
        "active_param_samples": int(active_param_count),
        "masked_violations": int(masked_violations),
        "finite_failures": int(finite_failures),
        "max_abs_logit": float(max_abs_logit),
        "entropy_mean": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "predicted_skill_counts": dict(sorted(skill_counts.items())),
        "by_skill": by_skill,
    }


def validate(
    checkpoint: Path,
    dataset: Path,
    *,
    device_name: str,
    samples: int,
    batch_size: int,
    seed: int,
    noise_std: float,
    min_clean_acc: float,
    min_noisy_acc: float,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    data = np.load(dataset, allow_pickle=True)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    weights = data["weights"].astype(np.float32)
    scenario_ids = data["scenario_ids"].astype(np.int64) if "scenario_ids" in data.files else np.zeros(len(actions), dtype=np.int64)
    names = _scenario_names(data)
    valid = np.flatnonzero(weights > 0.0)
    chosen = rng.choice(valid, size=min(samples, len(valid)), replace=False)
    device = torch.device(device_name)
    actor = _load_actor(checkpoint, observations.shape[1], device)
    clean = _metrics_for(actor, observations, actions, weights, indices=chosen, device=device, batch_size=batch_size, seed=seed)
    noisy = _metrics_for(actor, observations, actions, weights, indices=chosen, device=device, batch_size=batch_size, noise_std=noise_std, seed=seed + 1)

    scenario_metrics = {}
    for scenario_id, scenario_name in enumerate(names):
        idx = chosen[scenario_ids[chosen] == scenario_id]
        if idx.size:
            scenario_metrics[scenario_name] = _metrics_for(
                actor,
                observations,
                actions,
                weights,
                indices=idx,
                device=device,
                batch_size=batch_size,
                seed=seed + 17 + scenario_id,
            )

    required_predicted = set(GK_SKILL_ORDER)
    predicted = set(clean["predicted_skill_counts"])
    failures = []
    if clean["skill_acc"] < min_clean_acc:
        failures.append(f"clean skill_acc {clean['skill_acc']:.4f} < {min_clean_acc:.4f}")
    if noisy["skill_acc"] < min_noisy_acc:
        failures.append(f"noisy skill_acc {noisy['skill_acc']:.4f} < {min_noisy_acc:.4f}")
    if clean["masked_violations"] or noisy["masked_violations"]:
        failures.append(f"masked violations clean={clean['masked_violations']} noisy={noisy['masked_violations']}")
    if clean["finite_failures"] or noisy["finite_failures"]:
        failures.append(f"non-finite logits clean={clean['finite_failures']} noisy={noisy['finite_failures']}")
    missing_pred = sorted(required_predicted - predicted)
    if missing_pred:
        failures.append(f"missing predicted skills in clean sample: {missing_pred}")
    bad_scenarios = {
        name: metrics["skill_acc"]
        for name, metrics in scenario_metrics.items()
        if metrics["samples"] >= 20
        and metrics["skill_acc"] < min_noisy_acc
        and metrics["semantic_skill_acc"] < min_noisy_acc
    }
    if bad_scenarios:
        failures.append(f"scenario acc below {min_noisy_acc:.4f}: {bad_scenarios}")

    return {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "device": str(device),
        "samples": int(len(chosen)),
        "noise_std": float(noise_std),
        "clean": clean,
        "noisy": noisy,
        "scenarios": scenario_metrics,
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=61942)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--min-clean-acc", type=float, default=0.985)
    parser.add_argument("--min-noisy-acc", type=float, default=0.970)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = validate(
        _resolve(args.checkpoint),
        _resolve(args.dataset),
        device_name=args.device,
        samples=args.samples,
        batch_size=args.batch_size,
        seed=args.seed,
        noise_std=args.noise_std,
        min_clean_acc=args.min_clean_acc,
        min_noisy_acc=args.min_noisy_acc,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output:
        out = _resolve(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
