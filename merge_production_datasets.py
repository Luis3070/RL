from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from action import SKILL_ORDER


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT / "runs/fine_tuning_production_dataset/dataset_20260527_083253_sanitized"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/fine_tuning_production_dataset"


def _load_npz(dataset_dir: Path) -> dict[str, np.ndarray]:
    path = dataset_dir / "samples.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    required = ("observations", "actions", "stages", "episode_ids", "step_ids", "weights")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    arrays = {key: data[key] for key in required}
    if arrays["observations"].ndim != 2 or arrays["observations"].shape[1] != 26:
        raise ValueError(f"{path}: expected observations [N,26], got {arrays['observations'].shape}")
    if arrays["actions"].ndim != 2 or arrays["actions"].shape[1] != 5:
        raise ValueError(f"{path}: expected actions [N,5], got {arrays['actions'].shape}")
    n = arrays["observations"].shape[0]
    for key, value in arrays.items():
        if value.shape[0] != n:
            raise ValueError(f"{path}: {key} length {value.shape[0]} != observations length {n}")
    for key in ("observations", "actions", "weights"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{path}: non-finite values in {key}")
    return arrays


def _concat_csv(name: str, dataset_dirs: list[Path], output_dir: Path) -> None:
    import pandas as pd

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


def merge(dataset_dirs: list[Path], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = [_load_npz(path) for path in dataset_dirs]

    observations = np.concatenate([item["observations"].astype(np.float32) for item in loaded], axis=0)
    actions = np.concatenate([item["actions"].astype(np.float32) for item in loaded], axis=0)
    stages = np.concatenate([item["stages"].astype(np.int16) for item in loaded], axis=0)
    step_ids = np.concatenate([item["step_ids"].astype(np.int16) for item in loaded], axis=0)
    weights = np.concatenate([item["weights"].astype(np.float32) for item in loaded], axis=0)

    episode_chunks: list[np.ndarray] = []
    episode_offset = 0
    for item in loaded:
        ids = item["episode_ids"].astype(np.int32)
        remapped = ids + episode_offset
        episode_chunks.append(remapped)
        episode_offset = int(remapped.max()) + 1 if remapped.size else episode_offset
    episode_ids = np.concatenate(episode_chunks, axis=0).astype(np.int32)

    np.savez_compressed(
        output_dir / "samples.npz",
        observations=observations,
        actions=actions,
        stages=stages,
        episode_ids=episode_ids,
        step_ids=step_ids,
        weights=weights,
        skill_order=np.asarray(SKILL_ORDER),
    )
    _concat_csv("steps.csv", dataset_dirs, output_dir)
    _concat_csv("episodes.csv", dataset_dirs, output_dir)

    summaries = []
    for dataset_dir in dataset_dirs:
        path = dataset_dir / "summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    skill_ids = np.clip(np.rint(actions[:, 0]).astype(np.int64), 0, len(SKILL_ORDER) - 1)
    positive = weights > 0.0
    skill_counts = {
        SKILL_ORDER[index]: int(np.count_nonzero(positive & (skill_ids == index))) for index in range(len(SKILL_ORDER))
    }
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dirs": [str(path.resolve()) for path in dataset_dirs],
        "output_dir": str(output_dir.resolve()),
        "total_samples": int(len(weights)),
        "positive_samples": int(np.count_nonzero(positive)),
        "full_weight_samples": int(np.count_nonzero(weights >= 1.0)),
        "observation_shape": list(observations.shape),
        "action_shape": list(actions.shape),
        "positive_skill_counts": {key: value for key, value in skill_counts.items() if value},
        "source_summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    for dataset_dir in dataset_dirs:
        manifest = dataset_dir / "MANIFEST.md"
        if manifest.exists():
            shutil.copy2(manifest, output_dir / f"MANIFEST_{dataset_dir.name}.md")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return output_dir / "samples.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge production BC datasets into one samples.npz.")
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="Existing canonical/sanitized dataset directory.")
    parser.add_argument("--add", action="append", required=True, help="Additional dataset directory. Repeatable.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT / f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}_production_merged"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dirs = [Path(args.base).expanduser().resolve(), *[Path(path).expanduser().resolve() for path in args.add]]
    merge(dataset_dirs, Path(args.output_dir).expanduser().resolve())


if __name__ == "__main__":
    main()
