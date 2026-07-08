from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from goalkeeper_action import GK_SKILL_ORDER
from goalkeeper_teacher import BALL_CATCH_MAX_WALK_DISTANCE_MM, goalkeeper_skill_mask_np


REQUIRED_SKILLS = {
    "stand",
    "walk_goalie_pose",
    "observe",
    "intercept_center",
    "intercept_lateral",
    "intercept_low_left",
    "intercept_low_right",
    "intercept_jump_left",
    "intercept_jump_right",
    "clear",
    "pass",
    "dribble_out",
}

REQUIRED_SCENARIOS = {
    "central",
    "lateral_left",
    "lateral_right",
    "low_left",
    "low_right",
    "jump_left",
    "jump_right",
    "moving_away",
    "outside_goal_corridor",
    "unstable_body",
    "controlled_pass",
    "controlled_clear",
    "controlled_dribble",
    "stale_search",
    "stale_no_team_ball",
    "misposition_recover",
    "fast_blast_reaction",
    "close_range_reaction",
    "one_v_one_breakaway",
    "rebound_second_shot",
    "deflection_midflight",
    "screened_late_shot",
    "chip_lob_proxy",
    "dribble_in_left",
    "dribble_in_right",
}


def verify_dataset(path: Path, *, min_skill_count: int, min_rows: int) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    weights = data["weights"].astype(np.float32)
    label_consistent = data["label_consistent"].astype(np.int8)
    scenario_ids = data["scenario_ids"].astype(np.int64) if "scenario_ids" in data.files else np.zeros(len(actions), dtype=np.int64)
    scenario_names = tuple(str(x) for x in data["scenario_names"]) if "scenario_names" in data.files else ("unknown",)

    failures: list[str] = []
    if observations.ndim != 2:
        failures.append(f"observations must be 2D, got shape={observations.shape}")
    if actions.shape != (len(observations), 5):
        failures.append(f"actions must have shape (N,5), got {actions.shape}")
    if len(observations) < min_rows:
        failures.append(f"dataset too small: rows={len(observations)} min_rows={min_rows}")
    if not np.isfinite(observations).all():
        failures.append("observations contain NaN/Inf")
    if not np.isfinite(actions).all():
        failures.append("actions contain NaN/Inf")

    skill_ids = np.clip(np.rint(actions[:, 0]).astype(np.int64), 0, len(GK_SKILL_ORDER) - 1)
    positive = weights > 0.0
    skill_counts = Counter(GK_SKILL_ORDER[int(i)] for i in skill_ids[positive])
    missing_skills = sorted(REQUIRED_SKILLS - set(skill_counts))
    if missing_skills:
        failures.append(f"missing required skills: {missing_skills}")
    low_count = {name: int(skill_counts[name]) for name in REQUIRED_SKILLS if 0 < skill_counts[name] < min_skill_count}
    if low_count:
        failures.append(f"skills below min count {min_skill_count}: {low_count}")

    scenario_counts = Counter(scenario_names[int(i)] for i in scenario_ids[positive] if 0 <= int(i) < len(scenario_names))
    missing_scenarios = sorted(REQUIRED_SCENARIOS - set(scenario_counts))
    if missing_scenarios and set(scenario_names) != {"simrobot_teacher_rollout"}:
        failures.append(f"missing required scenarios: {missing_scenarios}")

    masked_rows = []
    for idx in np.flatnonzero(positive):
        mask = goalkeeper_skill_mask_np(observations[idx])
        if mask[skill_ids[idx]] <= 0.0:
            masked_rows.append(int(idx))
            if len(masked_rows) >= 10:
                break
    if masked_rows:
        failures.append(f"positive labels masked out at rows: {masked_rows}")
    inconsistent = int(np.count_nonzero((label_consistent <= 0) & positive))
    if inconsistent:
        failures.append(f"{inconsistent} positive rows are marked label_consistent=0")

    intercept = np.isin([GK_SKILL_ORDER[int(i)] for i in skill_ids], [name for name in GK_SKILL_ORDER if name.startswith("intercept_")])
    intercept_y = actions[:, 2][positive & intercept]
    if len(intercept_y) and float(np.max(np.abs(intercept_y))) > 1.0001:
        failures.append("normalized intercept target_y is outside [-1, 1]")
    if len(intercept_y) and float(np.max(np.abs(intercept_y))) < (60.0 / BALL_CATCH_MAX_WALK_DISTANCE_MM):
        failures.append("intercept target_y range collapsed; check double-normalization")

    return {
        "dataset": str(path),
        "rows": int(len(observations)),
        "positive_rows": int(np.count_nonzero(positive)),
        "skill_counts": dict(sorted(skill_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-skill-count", type=int, default=20)
    parser.add_argument("--min-rows", type=int, default=1000)
    args = parser.parse_args()
    result = verify_dataset(args.dataset.expanduser().resolve(), min_skill_count=args.min_skill_count, min_rows=args.min_rows)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
