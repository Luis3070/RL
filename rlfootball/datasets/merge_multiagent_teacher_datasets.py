from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from rlfootball.core.action import SKILL_ORDER


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "teacher_multiagent_dataset"
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


def _load_npz(dataset_dir: Path) -> dict[str, np.ndarray]:
    path = dataset_dir / "samples.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")

    arrays = {key: np.asarray(data[key]) for key in REQUIRED_KEYS}
    if arrays["observations"].ndim != 2 or arrays["observations"].shape[1] != 47:
        raise ValueError(f"{path}: expected observations [N,47], got {arrays['observations'].shape}")
    if arrays["actions"].ndim != 2 or arrays["actions"].shape[1] != 5:
        raise ValueError(f"{path}: expected actions [N,5], got {arrays['actions'].shape}")
    if arrays["coordination_target_xy"].ndim != 2 or arrays["coordination_target_xy"].shape[1] != 2:
        raise ValueError(
            f"{path}: expected coordination_target_xy [N,2], got {arrays['coordination_target_xy'].shape}"
        )

    rows = arrays["observations"].shape[0]
    for key, value in arrays.items():
        if value.shape[0] != rows:
            raise ValueError(f"{path}: {key} length {value.shape[0]} != observations length {rows}")
    for key in ("observations", "actions", "weights", "coordination_target_xy"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{path}: non-finite values in {key}")
    return arrays


def _load_summary(dataset_dir: Path) -> dict[str, object]:
    path = dataset_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _concat_csv(name: str, dataset_dirs: list[Path], output_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        return

    frames = []
    episode_offset = 0
    for dataset_index, dataset_dir in enumerate(dataset_dirs):
        path = dataset_dir / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "dataset_source", dataset_dir.name)
        frame.insert(1, "dataset_index", dataset_index)
        if name == "episodes.csv" and "episode" in frame.columns:
            kept = frame["episode"].fillna(-1).astype(int) >= 0
            frame.loc[kept, "episode"] = frame.loc[kept, "episode"].astype(int) + episode_offset
            if kept.any():
                episode_offset = int(frame.loc[kept, "episode"].max()) + 1
        frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(output_dir / name, index=False)


def _merge_scenarios(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[int, str], dict[str, object]] = {}
    for summary in summaries:
        stage_id = int(summary.get("stage_id", -1))
        for scenario in summary.get("scenarios", []):
            if not isinstance(scenario, dict):
                continue
            name = str(scenario.get("name", "unknown"))
            key = (stage_id, name)
            target = merged.setdefault(
                key,
                {
                    "stage_id": stage_id,
                    "name": name,
                    "kept_episodes": 0,
                    "rows": 0,
                    "drop_reasons": {},
                },
            )
            target["kept_episodes"] = int(target["kept_episodes"]) + int(scenario.get("kept_episodes", 0))
            role_step_counts = scenario.get("role_step_counts", {})
            inferred_rows = sum(int(v) for v in dict(role_step_counts).values()) if role_step_counts else 0
            target["rows"] = int(target["rows"]) + int(scenario.get("rows", inferred_rows))
            drops = Counter({str(k): int(v) for k, v in dict(target["drop_reasons"]).items()})
            drops.update({str(k): int(v) for k, v in dict(scenario.get("drop_reasons", {})).items()})
            target["drop_reasons"] = dict(drops)
    return [merged[key] for key in sorted(merged)]


def merge(dataset_dirs: list[Path], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = [_load_npz(path) for path in dataset_dirs]
    summaries = [_load_summary(path) for path in dataset_dirs]

    observations = np.concatenate([item["observations"].astype(np.float32) for item in loaded], axis=0)
    actions = np.concatenate([item["actions"].astype(np.float32) for item in loaded], axis=0)
    stages = np.concatenate([item["stages"].astype(np.int16) for item in loaded], axis=0)
    step_ids = np.concatenate([item["step_ids"].astype(np.int16) for item in loaded], axis=0)
    weights = np.concatenate([item["weights"].astype(np.float32) for item in loaded], axis=0)
    role_ids = np.concatenate([item["role_ids"].astype(np.int8) for item in loaded], axis=0)
    coords = np.concatenate([item["coordination_target_xy"].astype(np.float32) for item in loaded], axis=0)
    label_consistent = np.concatenate([item["label_consistent"].astype(np.int8) for item in loaded], axis=0)

    episode_chunks: list[np.ndarray] = []
    episode_offset = 0
    for item in loaded:
        ids = item["episode_ids"].astype(np.int32)
        remapped = ids + episode_offset
        episode_chunks.append(remapped)
        if remapped.size:
            episode_offset = int(remapped.max()) + 1
    episode_ids = np.concatenate(episode_chunks, axis=0).astype(np.int32)

    np.savez_compressed(
        output_dir / "samples.npz",
        observations=observations,
        actions=actions,
        stages=stages,
        episode_ids=episode_ids,
        step_ids=step_ids,
        weights=weights,
        role_ids=role_ids,
        coordination_target_xy=coords,
        label_consistent=label_consistent,
        skill_order=np.asarray(SKILL_ORDER),
    )
    _concat_csv("steps.csv", dataset_dirs, output_dir)
    _concat_csv("episodes.csv", dataset_dirs, output_dir)

    positive = weights > 0.0
    skill_ids = np.clip(np.rint(actions[:, 0]).astype(np.int64), 0, len(SKILL_ORDER) - 1)
    skill_counts = {
        SKILL_ORDER[index]: int(np.count_nonzero(positive & (skill_ids == index)))
        for index in range(len(SKILL_ORDER))
    }
    stage_values = sorted({int(value) for value in stages.tolist()})
    source_filters = [dict(summary.get("filters", {})) for summary in summaries if summary.get("filters")]
    filters = source_filters[-1] if source_filters else {}
    for key in ("open_support_hold_radius_mm", "off_ball_hold_radius_mm"):
        values = [float(item[key]) for item in source_filters if key in item]
        if values:
            filters[key] = max(values)

    # Gate-threshold consistency: the Design-A gate bits baked into obs and the
    # labels were generated under each source's gate_config. Mixing datasets
    # generated under different thresholds silently breaks label consistency.
    source_gate_configs = [summary.get("gate_config") for summary in summaries if summary.get("gate_config")]
    if source_gate_configs:
        first = source_gate_configs[0]
        for other in source_gate_configs[1:]:
            if other != first:
                raise ValueError(
                    "Refusing to merge: source datasets were generated under different "
                    f"gate_config values.\nfirst={first}\nother={other}"
                )
    gate_config = source_gate_configs[0] if source_gate_configs else None

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dirs": [str(path.resolve()) for path in dataset_dirs],
        "output_dir": str(output_dir.resolve()),
        "stage_id": stage_values[0] if len(stage_values) == 1 else -1,
        "stages": stage_values,
        "obs_size": 47,
        "total_rows": int(observations.shape[0]),
        "total_episodes": int(len(set(episode_ids.tolist()))),
        "positive_samples": int(np.count_nonzero(positive)),
        "full_weight_samples": int(np.count_nonzero(weights >= 1.0)),
        "positive_skill_counts": {key: value for key, value in skill_counts.items() if value},
        "positive_weight_gate_label_consistent": all(
            bool(summary.get("positive_weight_gate_label_consistent", False)) for summary in summaries
        ),
        "positive_weight_role_mask_consistent": all(
            bool(summary.get("positive_weight_role_mask_consistent", False)) for summary in summaries
        ),
        "filters": filters,
        "gate_config": gate_config,
        "scenarios": _merge_scenarios(summaries),
        "source_summaries": summaries,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    (output_dir / "summary.json").write_text(text + "\n", encoding="utf-8")
    for dataset_dir in dataset_dirs:
        manifest = dataset_dir / "MANIFEST.md"
        if manifest.exists():
            shutil.copy2(manifest, output_dir / f"MANIFEST_{dataset_dir.name}.md")
    print(text, flush=True)
    return output_dir / "samples.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge 47-dim multi-agent teacher dataset shards.")
    parser.add_argument("dataset_dirs", nargs="+", help="Dataset directories containing samples.npz.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT / f"dataset_multiagent_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dirs = [Path(path).expanduser().resolve() for path in args.dataset_dirs]
    merge(dataset_dirs, Path(args.output_dir).expanduser().resolve())


if __name__ == "__main__":
    main()
