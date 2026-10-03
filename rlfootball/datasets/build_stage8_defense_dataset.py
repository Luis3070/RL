from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from rlfootball.core.action import SKILL_ORDER

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


def _artifact_pair(dataset_dir: Path, *, include_partials: bool) -> tuple[Path, Path] | None:
    final = (dataset_dir / "samples.npz", dataset_dir / "summary.json")
    partial = (dataset_dir / "samples.partial.npz", dataset_dir / "summary.partial.json")
    if final[0].exists() and final[1].exists():
        return final
    if include_partials and partial[0].exists() and partial[1].exists():
        return partial
    return None


def _load_source(dataset_dir: Path, *, include_partials: bool) -> dict[str, object] | None:
    pair = _artifact_pair(dataset_dir, include_partials=include_partials)
    if pair is None:
        return None
    samples_path, summary_path = pair
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("stage_id", -1)) != 8:
        return None
    if int(summary.get("obs_size", 0)) != 47:
        return None
    if not bool(summary.get("positive_weight_gate_label_consistent", False)):
        return None
    if not bool(summary.get("positive_weight_role_mask_consistent", False)):
        return None

    data = np.load(samples_path, allow_pickle=True)
    arrays = {key: np.asarray(data[key]) for key in REQUIRED_KEYS}
    rows = int(arrays["observations"].shape[0])
    if rows <= 0 or arrays["observations"].ndim != 2 or arrays["observations"].shape[1] != 47:
        return None
    if arrays["actions"].ndim != 2 or arrays["actions"].shape[1] != 5:
        return None
    for key, value in arrays.items():
        if value.shape[0] != rows:
            return None
    if not np.isfinite(arrays["observations"]).all() or not np.isfinite(arrays["actions"]).all():
        return None
    return {
        "dataset_dir": dataset_dir,
        "samples_path": samples_path,
        "summary_path": summary_path,
        "summary": summary,
        "arrays": arrays,
    }


def _scenario_by_row(arrays: dict[str, np.ndarray], summary: dict[str, object]) -> np.ndarray:
    episode_ids = arrays["episode_ids"].astype(np.int64)
    scenario_names = np.full(episode_ids.shape[0], "unknown", dtype=object)
    start = 0
    for scenario in summary.get("scenarios", []):
        name = str(scenario.get("name", "unknown"))
        kept = int(scenario.get("kept_episodes", 0))
        if kept <= 0:
            continue
        end = start + kept
        scenario_names[(episode_ids >= start) & (episode_ids < end)] = name
        start = end
    return scenario_names


def _concat_sources(sources: list[dict[str, object]]) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in REQUIRED_KEYS}
    source_rows: list[dict[str, object]] = []
    episode_offset = 0
    for source_index, source in enumerate(sources):
        arrays = source["arrays"]  # type: ignore[assignment]
        assert isinstance(arrays, dict)
        rows = int(arrays["observations"].shape[0])
        for key in REQUIRED_KEYS:
            value = np.asarray(arrays[key])
            if key == "episode_ids":
                value = value.astype(np.int64) + episode_offset
            chunks[key].append(value)
        episode_ids = np.asarray(arrays["episode_ids"], dtype=np.int64)
        if episode_ids.size:
            episode_offset += int(episode_ids.max()) + 1
        source_rows.append({
            "dataset_dir": str(source["dataset_dir"]),
            "samples": str(source["samples_path"]),
            "summary": str(source["summary_path"]),
            "rows": rows,
            "partial": bool(source["summary"].get("partial", False)),  # type: ignore[index]
        })
    merged = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    return merged, source_rows


def _oversample_indices(
    arrays: dict[str, np.ndarray],
    scenario_names: np.ndarray,
    *,
    min_transition_rows: int,
    min_block_frac: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.arange(arrays["observations"].shape[0], dtype=np.int64)
    extras: list[np.ndarray] = []

    if min_transition_rows > 0:
        transition = np.flatnonzero(scenario_names == "transition_midfield")
        if transition.size and transition.size < min_transition_rows:
            extras.append(rng.choice(transition, size=min_transition_rows - transition.size, replace=True))

    block_id = SKILL_ORDER.index("block")
    skill_ids = np.rint(arrays["actions"][:, 0]).astype(np.int64)
    block = np.flatnonzero(skill_ids == block_id)
    if block.size and min_block_frac > 0.0:
        current_total = base.size + sum(extra.size for extra in extras)
        current_block = int(block.size + sum(np.count_nonzero(skill_ids[extra] == block_id) for extra in extras))
        target_block = int(np.ceil(min_block_frac * current_total))
        if current_block < target_block:
            # Solve approximately for extra block rows because adding rows changes denominator.
            needed = int(np.ceil((min_block_frac * current_total - current_block) / max(1e-6, 1.0 - min_block_frac)))
            extras.append(rng.choice(block, size=max(0, needed), replace=True))

    if not extras:
        return base
    return np.concatenate([base, *extras], axis=0)


def build(
    root: Path,
    output_dir: Path,
    *,
    include_partials: bool,
    min_rows: int,
    min_transition_rows: int,
    min_block_frac: float,
    seed: int,
) -> Path:
    candidates = sorted(root.glob("dataset_stage8_*"))
    sources = [
        source for path in candidates
        if (source := _load_source(path, include_partials=include_partials)) is not None
    ]
    sources = [source for source in sources if int(source["summary"].get("total_rows", 0)) >= min_rows]  # type: ignore[index]
    if not sources:
        raise RuntimeError(f"No useful stage8 datasets found under {root}")

    arrays, source_rows = _concat_sources(sources)
    scenario_chunks = []
    for source in sources:
        scenario_chunks.append(_scenario_by_row(source["arrays"], source["summary"]))  # type: ignore[arg-type]
    scenario_names = np.concatenate(scenario_chunks, axis=0)

    indices = _oversample_indices(
        arrays,
        scenario_names,
        min_transition_rows=min_transition_rows,
        min_block_frac=min_block_frac,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    balanced = {key: value[indices] for key, value in arrays.items()}
    balanced_scenarios = scenario_names[indices]
    np.savez_compressed(
        output_dir / "samples.npz",
        **balanced,
        skill_order=np.asarray(SKILL_ORDER),
    )

    skill_ids = np.rint(balanced["actions"][:, 0]).astype(np.int64)
    role_ids = balanced["role_ids"].astype(np.int64)
    skill_counts = Counter(SKILL_ORDER[int(np.clip(skill, 0, len(SKILL_ORDER) - 1))] for skill in skill_ids)
    role_counts = Counter({0: "striker", 1: "open_support", 2: "off_ball_support"}.get(int(role), "unknown") for role in role_ids)
    scenario_counts = Counter(str(name) for name in balanced_scenarios.tolist())
    scenario_summaries = [
        {
            "name": name,
            "stage_id": 8,
            "kept_episodes": 1,
            "rows": int(count),
            "drop_reasons": {},
            "role_step_counts": {},
        }
        for name, count in sorted(scenario_counts.items())
        if name != "unknown" and int(count) > 0
    ]
    gate_configs = [source["summary"].get("gate_config") for source in sources if source["summary"].get("gate_config")]  # type: ignore[index]
    gate_config = gate_configs[0] if gate_configs else None
    if gate_configs and any(config != gate_config for config in gate_configs[1:]):
        raise RuntimeError("Useful datasets have incompatible gate_config values")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage_id": 8,
        "obs_size": 47,
        "synthetic_from_existing_rows": True,
        "synthetic_method": "bootstrap_oversample_transition_and_block",
        "source_root": str(root.resolve()),
        "source_count": len(sources),
        "source_rows": source_rows,
        "total_rows_raw": int(arrays["observations"].shape[0]),
        "total_rows": int(balanced["observations"].shape[0]),
        "total_episodes": int(len(set(balanced["episode_ids"].astype(np.int64).tolist()))),
        "scenario_row_counts": dict(scenario_counts),
        "scenarios": scenario_summaries,
        "skill_counts": dict(skill_counts),
        "role_counts": dict(role_counts),
        "positive_weight_gate_label_consistent": True,
        "positive_weight_role_mask_consistent": True,
        "gate_config": gate_config,
        "filters": {
            "min_transition_rows": int(min_transition_rows),
            "min_block_frac": float(min_block_frac),
            "include_partials": bool(include_partials),
            "min_source_rows": int(min_rows),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copy2(__file__, output_dir / "build_stage8_defense_dataset.py")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return output_dir / "samples.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a useful stage-8 defender dataset from complete/partial runs.")
    parser.add_argument("--root", type=Path, default=Path("runs/teacher_multiagent_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/teacher_multiagent_dataset/dataset_defense_stage8_bootstrap_v1"))
    parser.add_argument("--include-partials", action="store_true", default=True)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-transition-rows", type=int, default=4000)
    parser.add_argument("--min-block-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    build(
        args.root,
        args.output_dir,
        include_partials=args.include_partials,
        min_rows=args.min_rows,
        min_transition_rows=args.min_transition_rows,
        min_block_frac=args.min_block_frac,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
