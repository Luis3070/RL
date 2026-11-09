from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from rlfootball.core.goalkeeper_action import GK_SKILL_ORDER


REQUIRED = ("observations", "actions", "stages", "episode_ids", "step_ids", "weights", "label_consistent")


def _dataset_dir(path: str) -> Path:
    p = Path(path).expanduser()
    if p.name == "samples.npz":
        return p.parent.resolve()
    return p.resolve()


def _load(path: Path) -> dict[str, np.ndarray]:
    samples = path / "samples.npz"
    data = np.load(samples, allow_pickle=True)
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise KeyError(f"{samples} missing keys {missing}")
    arrays = {key: np.asarray(data[key]) for key in REQUIRED}
    rows = arrays["observations"].shape[0]
    arrays["scenario_ids"] = np.asarray(data["scenario_ids"]) if "scenario_ids" in data else np.zeros(rows, dtype=np.int16)
    arrays["scenario_names"] = np.asarray(data["scenario_names"]) if "scenario_names" in data else np.asarray(["unknown"])
    if arrays["actions"].shape != (rows, 5):
        raise ValueError(f"{samples}: actions shape {arrays['actions'].shape}, expected ({rows}, 5)")
    for key in (*REQUIRED, "scenario_ids"):
        value = arrays[key]
        if value.shape[0] != rows:
            raise ValueError(f"{samples}: {key} length mismatch")
    return arrays


def merge(inputs: list[Path], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    loaded = [_load(path) for path in inputs]
    observations = np.concatenate([d["observations"].astype(np.float32) for d in loaded], axis=0)
    actions = np.concatenate([d["actions"].astype(np.float32) for d in loaded], axis=0)
    stages = np.concatenate([d["stages"].astype(np.int16) for d in loaded], axis=0)
    step_ids = np.concatenate([d["step_ids"].astype(np.int16) for d in loaded], axis=0)
    weights = np.concatenate([d["weights"].astype(np.float32) for d in loaded], axis=0)
    label_consistent = np.concatenate([d["label_consistent"].astype(np.int8) for d in loaded], axis=0)
    remapped_eps = []
    remapped_scenarios = []
    scenario_names: list[str] = []
    scenario_offsets: dict[tuple[int, str], int] = {}
    offset = 0
    for dataset_idx, d in enumerate(loaded):
        eps = d["episode_ids"].astype(np.int32) + offset
        remapped_eps.append(eps)
        if eps.size:
            offset = int(eps.max()) + 1
        names = [str(item) for item in d["scenario_names"]]
        local_to_global = {}
        for local_id, name in enumerate(names):
            key = (dataset_idx, name)
            if key not in scenario_offsets:
                scenario_offsets[key] = len(scenario_names)
                scenario_names.append(name)
            local_to_global[local_id] = scenario_offsets[key]
        remapped_scenarios.append(
            np.asarray([local_to_global.get(int(item), 0) for item in d["scenario_ids"]], dtype=np.int16)
        )
    episode_ids = np.concatenate(remapped_eps, axis=0)
    scenario_ids = np.concatenate(remapped_scenarios, axis=0)
    np.savez_compressed(
        output / "samples.npz",
        observations=observations,
        actions=actions,
        stages=stages,
        episode_ids=episode_ids,
        step_ids=step_ids,
        weights=weights,
        label_consistent=label_consistent,
        scenario_ids=scenario_ids,
        scenario_names=np.asarray(scenario_names),
        skill_order=np.asarray(GK_SKILL_ORDER),
    )
    positive = weights > 0.0
    ids = np.clip(np.rint(actions[:, 0]).astype(np.int64), 0, len(GK_SKILL_ORDER) - 1)
    counts = Counter(GK_SKILL_ORDER[i] for i in ids[positive])
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": [str(path) for path in inputs],
        "obs_size": int(observations.shape[1]),
        "total_rows": int(observations.shape[0]),
        "positive_samples": int(np.count_nonzero(positive)),
        "positive_skill_counts": dict(sorted(counts.items())),
        "skill_order": list(GK_SKILL_ORDER),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return output / "samples.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    merge([_dataset_dir(item) for item in args.datasets], Path(args.output_dir).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
