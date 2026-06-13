from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from action import SKILL_ORDER

REQUIRED_KEYS = (
    "observations",
    "actions",
    "stages",
    "episode_ids",
    "step_ids",
    "weights",
    "role_ids",
    "coordination_target_xy",
    "label_consistent",
)


def _load_summary(dataset_dir: Path) -> dict[str, object]:
    path = dataset_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def subsample(
    source_dir: Path,
    output_dir: Path,
    *,
    target_rows: int | None,
    defense_rows: int | None,
    attack_fraction_of_defense: float,
    seed: int,
) -> Path:
    samples_path = source_dir / "samples.npz"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)
    data = np.load(samples_path, allow_pickle=True)
    arrays = {key: np.asarray(data[key]) for key in REQUIRED_KEYS}
    rows = int(arrays["observations"].shape[0])
    if target_rows is None:
        if defense_rows is None:
            raise ValueError("Pass --target-rows or --defense-rows")
        target_rows = max(1, int(round(float(defense_rows) * float(attack_fraction_of_defense))))

    episode_ids = arrays["episode_ids"].astype(np.int64)
    unique_eps = np.asarray(sorted(set(episode_ids.tolist())), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)
    selected_eps: list[int] = []
    selected_rows = 0
    for episode_id in unique_eps.tolist():
        count = int(np.count_nonzero(episode_ids == episode_id))
        if selected_rows >= target_rows and selected_eps:
            break
        selected_eps.append(int(episode_id))
        selected_rows += count

    keep = np.isin(episode_ids, np.asarray(selected_eps, dtype=np.int64))
    output_dir.mkdir(parents=True, exist_ok=True)
    remapped_episode_ids = arrays["episode_ids"][keep].astype(np.int64)
    episode_map = {old: new for new, old in enumerate(sorted(set(remapped_episode_ids.tolist())))}
    remapped_episode_ids = np.asarray([episode_map[int(value)] for value in remapped_episode_ids], dtype=np.int32)

    np.savez_compressed(
        output_dir / "samples.npz",
        observations=arrays["observations"][keep].astype(np.float32),
        actions=arrays["actions"][keep].astype(np.float32),
        stages=arrays["stages"][keep].astype(np.int16),
        episode_ids=remapped_episode_ids,
        step_ids=arrays["step_ids"][keep].astype(np.int16),
        weights=arrays["weights"][keep].astype(np.float32),
        role_ids=arrays["role_ids"][keep].astype(np.int8),
        coordination_target_xy=arrays["coordination_target_xy"][keep].astype(np.float32),
        label_consistent=arrays["label_consistent"][keep].astype(np.int8),
        skill_order=np.asarray(SKILL_ORDER),
    )
    summary_in = _load_summary(source_dir)
    summary = {
        "source_dataset": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": int(seed),
        "target_rows": int(target_rows),
        "source_rows": rows,
        "total_rows": int(np.count_nonzero(keep)),
        "source_episodes": int(len(unique_eps)),
        "total_episodes": int(len(selected_eps)),
        "selected_episode_ids": selected_eps,
        "obs_size": int(arrays["observations"].shape[1]),
        "positive_weight_gate_label_consistent": bool(summary_in.get("positive_weight_gate_label_consistent", True)),
        "positive_weight_role_mask_consistent": bool(summary_in.get("positive_weight_role_mask_consistent", True)),
        "filters": dict(summary_in.get("filters", {})),
        "gate_config": summary_in.get("gate_config"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return output_dir / "samples.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subsample a multi-agent attack dataset by whole episodes.")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-rows", type=int)
    parser.add_argument("--defense-rows", type=int)
    parser.add_argument("--attack-fraction-of-defense", type=float, default=0.43)
    parser.add_argument("--seed", type=int, default=260613)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    subsample(
        args.source_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        target_rows=args.target_rows,
        defense_rows=args.defense_rows,
        attack_fraction_of_defense=args.attack_fraction_of_defense,
        seed=args.seed,
    )
