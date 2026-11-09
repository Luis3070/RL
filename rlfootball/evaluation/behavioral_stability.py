from __future__ import annotations

import json
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

DEFAULT_STABLE_SEEDS: tuple[int, ...] = (11, 23, 37, 51, 67, 83)
DEFAULT_KEY_METRICS: tuple[str, ...] = (
    "clean_episode_rate",
    "clean_finish_rate",
    "shoot_finish_rate",
    "collision_rate",
    "dynamic_obstacle_success",
    "corridor_keep_rate",
    "policy_shoot_rate",
    "natural_ball_seen_rate",
    "ball_trusted_rate",
    "synthetic_ball_rate",
)


def parse_seed_list(raw: str | None) -> tuple[int, ...]:
    if raw is None or not str(raw).strip():
        return DEFAULT_STABLE_SEEDS
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("Seed list cannot be empty")
    return values


def parse_scenario_list(raw: str | None) -> tuple[str, ...]:
    if raw is None or not str(raw).strip():
        return ()
    values = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("Scenario list cannot be empty")
    return values


def split_failures(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(";") if part.strip())


def aggregate_rep_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    pass_rate_threshold: float,
    metric_names: Iterable[str] = DEFAULT_KEY_METRICS,
) -> dict[str, object]:
    if not rows:
        raise ValueError("aggregate_rep_rows requires at least one row")

    passed = np.asarray([float(row.get("passed", 0.0)) for row in rows], dtype=np.float64)
    failure_counts: Counter[str] = Counter()
    for row in rows:
        failure_counts.update(split_failures(row.get("failures", "")))

    aggregate: dict[str, object] = {
        "reps": int(len(rows)),
        "pass_count": int(passed.sum()),
        "pass_rate": float(np.mean(passed)),
        "stable_passed": int(float(np.mean(passed)) >= float(pass_rate_threshold)),
        "pass_rate_threshold": float(pass_rate_threshold),
        "failure_counts_json": json.dumps(dict(sorted(failure_counts.items())), sort_keys=True),
        "top_failures": ",".join(name for name, _count in failure_counts.most_common(3)),
    }

    for metric_name in metric_names:
        values = [float(row.get(metric_name, 0.0)) for row in rows]
        series = np.asarray(values, dtype=np.float64)
        aggregate[f"{metric_name}_mean"] = float(np.mean(series))
        aggregate[f"{metric_name}_std"] = float(np.std(series))
        aggregate[f"{metric_name}_min"] = float(np.min(series))
        aggregate[f"{metric_name}_max"] = float(np.max(series))

    return aggregate
