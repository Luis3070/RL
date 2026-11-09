from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from rlfootball.core.action import ACTION_PARAM_MASKS, SKILL_ORDER, skill_mask
from rlfootball.core.config import ACTION, ROOT_DIR, TRAIN
from rlfootball.curriculum.curriculum import get_stage_spec
from rlfootball.training.train import (
    PolicyNetwork,
    RUNS_DIR,
    ValueNetwork,
    _load_state_dict_obs_padded,
    bc_ce_safe_logits,
    gate_skill_mask_from_obs,
)


# Edit these constants directly for IDE-driven experiments. Env vars only exist
# to make overnight runs reproducible from shell scripts.
DATASET_PATH = os.getenv("RL_BC_DATASET", "")
# Optional warm-start: initialize the BC policy/value from an existing checkpoint
# (e.g. the 720k base) before fine-tuning on the dataset, instead of training from
# scratch. The first Linear layer is zero-padded if the checkpoint has fewer obs
# columns (23 -> 26), so day-0 behaviour matches the base and BC re-anchors it on the
# clean drive->shoot data while preserving the base's learned features.
INIT_CHECKPOINT = os.getenv("RL_BC_INIT_CHECKPOINT", "")
OUTPUT_ROOT = RUNS_DIR / "bc_warm_start"
DEVICE = os.getenv("RL_BC_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.getenv("RL_BC_BATCH_SIZE", "256"))
LEARNING_RATE = float(os.getenv("RL_BC_LEARNING_RATE", "0.0003"))
WEIGHT_DECAY = float(os.getenv("RL_BC_WEIGHT_DECAY", "0.00001"))
MAX_GRAD_NORM = float(os.getenv("RL_BC_MAX_GRAD_NORM", "1.0"))
VALIDATION_FRACTION = float(os.getenv("RL_BC_VALIDATION_FRACTION", "0.15"))
SEED = int(os.getenv("RL_BC_SEED", "24052026"))
SKILL_LOSS_WEIGHT = float(os.getenv("RL_BC_SKILL_LOSS_WEIGHT", "1.0"))
PARAM_LOSS_WEIGHT = float(os.getenv("RL_BC_PARAM_LOSS_WEIGHT", "0.35"))
MIN_SKILL_WEIGHT = float(os.getenv("RL_BC_MIN_SKILL_WEIGHT", "0.35"))
LOG_INTERVAL = int(os.getenv("RL_BC_LOG_INTERVAL", "20"))


@dataclass(frozen=True)
class BCPhase:
    name: str
    stages: tuple[int, ...]
    epochs: int


PHASES = (
    BCPhase("basics_stage_minus1_0", (-1, 0), int(os.getenv("RL_BC_PHASE1_EPOCHS", "12"))),
    BCPhase("handoff_finish_stage0_1", (0, 1), int(os.getenv("RL_BC_PHASE2_EPOCHS", "14"))),
    BCPhase("obstacles_stage1_2_3", (1, 2, 3), int(os.getenv("RL_BC_PHASE3_EPOCHS", "16"))),
    BCPhase("balanced_replay_all", (-1, 0, 1, 2, 3), int(os.getenv("RL_BC_PHASE4_EPOCHS", "12"))),
)


def _latest_dataset_path() -> Path:
    root = ROOT_DIR / "runs" / "teacher_bc_dataset"
    candidates = sorted(root.glob("dataset_*/samples.npz"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No teacher dataset found under {root}. Set RL_BC_DATASET=/path/to/samples.npz or run teacher_dataset.py."
        )
    return candidates[0]


def _resolve_dataset_path() -> Path:
    return Path(DATASET_PATH).expanduser().resolve() if DATASET_PATH else _latest_dataset_path().resolve()


def _make_spaces(observation_size: int) -> tuple[gym.spaces.Box, gym.spaces.Box, gym.spaces.Box]:
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.asarray([0.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32),
        high=np.asarray([float(len(SKILL_ORDER) - 1), 1.0, 1.0, 1.0, float(ACTION.max_pass_target)], dtype=np.float32),
        shape=(5,),
        dtype=np.float32,
    )
    state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32)
    return observation_space, action_space, state_space


def _param_mask_for_skills(skill_ids: torch.Tensor) -> torch.Tensor:
    masks = torch.as_tensor(
        [ACTION_PARAM_MASKS.get(name, (0.0, 0.0, 0.0, 0.0)) for name in SKILL_ORDER],
        dtype=torch.float32,
        device=skill_ids.device,
    )
    return masks[skill_ids]


def _stage_masks(stages: torch.Tensor) -> torch.Tensor:
    unique_stages = sorted(int(value) for value in stages.detach().cpu().unique().tolist())
    masks = {
        stage_id: torch.as_tensor(
            skill_mask(get_stage_spec(stage_id).active_skill_names),
            dtype=torch.float32,
            device=stages.device,
        )
        for stage_id in unique_stages
    }
    return torch.stack([masks[int(stage_id.item())] for stage_id in stages], dim=0)


def _masked_logits(
    skill_logits: torch.Tensor, stages: torch.Tensor, observations: torch.Tensor | None = None
) -> torch.Tensor:
    masks = _stage_masks(stages).to(dtype=skill_logits.dtype)
    if observations is not None:
        # Same forbid-invalid gate the RL rollout applies, so BC and RL share mask
        # semantics and the dataset labels are scored over the identical allowed set.
        masks = masks * gate_skill_mask_from_obs(observations).to(dtype=masks.dtype)
    return skill_logits.masked_fill(masks <= 0.0, -1.0e9)


def _skill_class_weights(skill_ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    counts = torch.zeros(len(SKILL_ORDER), dtype=torch.float32, device=skill_ids.device)
    counts.scatter_add_(0, skill_ids, weights)
    nonzero = counts[counts > 0.0]
    if nonzero.numel() == 0:
        return torch.ones(len(SKILL_ORDER), dtype=torch.float32, device=skill_ids.device)
    median = torch.median(nonzero)
    class_weights = torch.ones_like(counts)
    class_weights[counts > 0.0] = torch.sqrt(median / torch.clamp(counts[counts > 0.0], min=1.0))
    return torch.clamp(class_weights, min=MIN_SKILL_WEIGHT, max=4.0)


def _split_indices(stages: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    valid_indices = np.flatnonzero(weights > 0.0)
    for stage_id in sorted(int(value) for value in np.unique(stages[valid_indices])):
        stage_indices = valid_indices[stages[valid_indices] == stage_id]
        shuffled = rng.permutation(stage_indices)
        val_count = max(1, int(round(len(shuffled) * VALIDATION_FRACTION))) if len(shuffled) > 1 else 0
        val_parts.append(shuffled[:val_count])
        train_parts.append(shuffled[val_count:])
    return np.concatenate(train_parts), np.concatenate(val_parts)


def _phase_batches(
    train_indices: np.ndarray,
    stages_np: np.ndarray,
    weights_np: np.ndarray,
    phase: BCPhase,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    stage_to_indices = {
        stage_id: train_indices[stages_np[train_indices] == stage_id]
        for stage_id in phase.stages
        if np.any(stages_np[train_indices] == stage_id)
    }
    if not stage_to_indices:
        raise ValueError(f"Phase {phase.name} has no samples for stages {phase.stages}")
    steps = max(1, math.ceil(sum(len(values) for values in stage_to_indices.values()) / BATCH_SIZE))
    active_stages = tuple(stage_to_indices)
    per_stage = max(1, BATCH_SIZE // len(active_stages))
    for _ in range(steps):
        chunks = []
        for stage_id in active_stages:
            indices = stage_to_indices[stage_id]
            probabilities = weights_np[indices].astype(np.float64)
            probabilities = probabilities / max(float(probabilities.sum()), 1e-12)
            chunks.append(rng.choice(indices, size=per_stage, replace=True, p=probabilities))
        batch = np.concatenate(chunks)
        if len(batch) < BATCH_SIZE:
            extra = rng.choice(batch, size=BATCH_SIZE - len(batch), replace=True)
            batch = np.concatenate([batch, extra])
        rng.shuffle(batch)
        yield batch[:BATCH_SIZE]


def _loss_for_batch(
    policy: PolicyNetwork,
    observations: torch.Tensor,
    actions: torch.Tensor,
    stages: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    skill_targets = torch.clamp(actions[:, 0].round().long(), min=0, max=len(SKILL_ORDER) - 1)
    param_targets = torch.clamp(actions[:, 1:], min=-1.0, max=1.0)
    skill_logits, outputs = policy.compute({"observations": observations}, role="policy")
    masked_skill_logits = _masked_logits(skill_logits, stages, observations)
    param_mean = outputs["param_mean"]

    ce = F.cross_entropy(
        bc_ce_safe_logits(masked_skill_logits, skill_logits, skill_targets),
        skill_targets,
        weight=class_weights,
        reduction="none",
    )
    param_mask = _param_mask_for_skills(skill_targets)
    param_raw = F.smooth_l1_loss(param_mean, param_targets, reduction="none") * param_mask
    param_denominator = torch.clamp(param_mask.sum(dim=-1), min=1.0)
    param_loss_per_sample = param_raw.sum(dim=-1) / param_denominator
    param_active = (param_mask.sum(dim=-1) > 0.0).to(dtype=sample_weights.dtype)

    normalized_weights = sample_weights / torch.clamp(sample_weights.mean(), min=1e-6)
    skill_loss = (ce * normalized_weights).mean()
    param_weighted = param_loss_per_sample * normalized_weights * param_active
    param_loss = param_weighted.sum() / torch.clamp((normalized_weights * param_active).sum(), min=1.0)
    total_loss = SKILL_LOSS_WEIGHT * skill_loss + PARAM_LOSS_WEIGHT * param_loss

    with torch.no_grad():
        predictions = torch.argmax(masked_skill_logits, dim=-1)
        accuracy = (predictions == skill_targets).to(dtype=torch.float32)
        weighted_accuracy = (accuracy * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1e-6)
    return total_loss, {
        "loss": float(total_loss.detach().cpu().item()),
        "skill_loss": float(skill_loss.detach().cpu().item()),
        "param_loss": float(param_loss.detach().cpu().item()),
        "skill_acc": float(weighted_accuracy.detach().cpu().item()),
    }


@torch.no_grad()
def _validate(
    policy: PolicyNetwork,
    observations: torch.Tensor,
    actions: torch.Tensor,
    stages: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, object]:
    policy.eval()
    class_weights = _skill_class_weights(torch.clamp(actions[:, 0].round().long(), min=0, max=len(SKILL_ORDER) - 1), weights)
    _, metrics = _loss_for_batch(policy, observations, actions, stages, weights, class_weights)
    skill_targets = torch.clamp(actions[:, 0].round().long(), min=0, max=len(SKILL_ORDER) - 1)
    skill_logits, outputs = policy.compute({"observations": observations}, role="policy")
    predictions = torch.argmax(_masked_logits(skill_logits, stages, observations), dim=-1)
    param_mask = _param_mask_for_skills(skill_targets)
    param_abs_error = torch.abs(outputs["param_mean"] - torch.clamp(actions[:, 1:], min=-1.0, max=1.0)) * param_mask

    by_stage = {}
    for stage_id in sorted(int(value) for value in stages.detach().cpu().unique().tolist()):
        mask = stages == stage_id
        stage_weights = weights[mask]
        stage_correct = (predictions[mask] == skill_targets[mask]).to(dtype=torch.float32)
        by_stage[str(stage_id)] = {
            "samples": int(mask.sum().item()),
            "skill_acc": float((stage_correct * stage_weights).sum().div(torch.clamp(stage_weights.sum(), min=1e-6)).item()),
        }

    by_skill = {}
    for skill_index, skill_name in enumerate(SKILL_ORDER):
        mask = skill_targets == skill_index
        if not bool(mask.any()):
            continue
        active_params = torch.clamp(param_mask[mask].sum(), min=1.0)
        by_skill[skill_name] = {
            "samples": int(mask.sum().item()),
            "acc": float((predictions[mask] == skill_targets[mask]).to(dtype=torch.float32).mean().item()),
            "param_mae": float(param_abs_error[mask].sum().div(active_params).item()),
        }
    policy.train()
    return {**metrics, "by_stage": by_stage, "by_skill": by_skill}


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dataset_path = _resolve_dataset_path()
    data = np.load(dataset_path, allow_pickle=True)
    observations_np = data["observations"].astype(np.float32)
    actions_np = data["actions"].astype(np.float32)
    stages_np = data["stages"].astype(np.int64)
    weights_np = data["weights"].astype(np.float32)
    train_indices, val_indices = _split_indices(stages_np, weights_np, rng)

    observation_space, action_space, state_space = _make_spaces(observations_np.shape[1])
    policy = PolicyNetwork(
        observation_space,
        action_space,
        DEVICE,
        TRAIN.hidden_size,
        param_log_prob_weight=TRAIN.param_log_prob_weight,
    )
    policy.set_active_skill_names(SKILL_ORDER)
    value = ValueNetwork(observation_space, state_space, action_space, DEVICE, TRAIN.hidden_size)

    if INIT_CHECKPOINT:
        init_path = Path(INIT_CHECKPOINT).expanduser().resolve()
        init_ckpt = torch.load(init_path, map_location=DEVICE, weights_only=False)
        if "policy_state_dict" not in init_ckpt:
            raise ValueError(f"RL_BC_INIT_CHECKPOINT {init_path} has no policy_state_dict")
        _load_state_dict_obs_padded(policy, init_ckpt["policy_state_dict"], label="bc-init policy")
        if "value_state_dict" in init_ckpt:
            _load_state_dict_obs_padded(value, init_ckpt["value_state_dict"], label="bc-init value")
        print(f"[bc] warm-started from {init_path} (obs zero-padded if grown)", flush=True)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    observations = torch.as_tensor(observations_np, dtype=torch.float32, device=DEVICE)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=DEVICE)
    stages = torch.as_tensor(stages_np, dtype=torch.long, device=DEVICE)
    sample_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=DEVICE)
    train_skill_ids = torch.clamp(actions[train_indices, 0].round().long(), min=0, max=len(SKILL_ORDER) - 1)
    class_weights = _skill_class_weights(train_skill_ids, sample_weights[train_indices])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"bc_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    global_step = 0
    print(
        f"[bc] dataset={dataset_path} train={len(train_indices)} val={len(val_indices)} "
        f"device={DEVICE} output={output_dir}",
        flush=True,
    )

    for phase in PHASES:
        for epoch in range(1, phase.epochs + 1):
            epoch_metrics = []
            for batch_indices in _phase_batches(train_indices, stages_np, weights_np, phase, rng):
                global_step += 1
                idx = torch.as_tensor(batch_indices, dtype=torch.long, device=DEVICE)
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = _loss_for_batch(
                    policy,
                    observations[idx],
                    actions[idx],
                    stages[idx],
                    sample_weights[idx],
                    class_weights,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                epoch_metrics.append(metrics)
            row = {
                "step": global_step,
                "phase": phase.name,
                "epoch": epoch,
                "loss": float(np.mean([m["loss"] for m in epoch_metrics])),
                "skill_loss": float(np.mean([m["skill_loss"] for m in epoch_metrics])),
                "param_loss": float(np.mean([m["param_loss"] for m in epoch_metrics])),
                "skill_acc": float(np.mean([m["skill_acc"] for m in epoch_metrics])),
            }
            metrics_rows.append(row)
            if epoch == phase.epochs or epoch % max(1, LOG_INTERVAL) == 0:
                print(
                    f"[bc] phase={phase.name} epoch={epoch}/{phase.epochs} "
                    f"loss={row['loss']:.4f} skill_acc={row['skill_acc']:.3f} param={row['param_loss']:.4f}",
                    flush=True,
                )

    val_idx = torch.as_tensor(val_indices, dtype=torch.long, device=DEVICE)
    validation = _validate(policy, observations[val_idx], actions[val_idx], stages[val_idx], sample_weights[val_idx])
    checkpoint_path = output_dir / f"checkpoint_bc_step{global_step}.pt"
    torch.save(
        {
            "timestep": int(global_step),
            "stage_id": -1,
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value.state_dict(),
            "eval_summary": validation,
            "bc_metadata": {
                "dataset_path": str(dataset_path),
                "phases": [asdict(phase) for phase in PHASES],
                "train_samples": int(len(train_indices)),
                "val_samples": int(len(val_indices)),
                "skill_order": list(SKILL_ORDER),
            },
        },
        checkpoint_path,
    )
    report = {
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "device": DEVICE,
        "global_step": int(global_step),
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "validation": validation,
        "phases": [asdict(phase) for phase in PHASES],
        "metrics": metrics_rows,
    }
    (output_dir / "bc_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("checkpoint_path", "train_samples", "val_samples", "validation")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
