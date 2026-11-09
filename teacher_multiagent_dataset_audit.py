from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from action import SKILL_ORDER
from config import ACTION
from observation import ObsIndex

ROLE_NAMES = {0: "striker", 1: "open_support", 2: "off_ball_support"}


def _latest_dataset(root: Path) -> Path:
    candidates = sorted(root.glob("dataset_stage*/summary.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No summary.json found under {root}")
    return candidates[-1].parent


def _role_mask_allows(role_id: int, skill_name: str, obs: np.ndarray) -> bool:
    shoot_armed = float(obs[int(ObsIndex.shoot_armed)]) >= 0.5
    dribble_armed = float(obs[int(ObsIndex.dribble_armed)]) >= 0.5
    pass_armed = float(obs[int(ObsIndex.pass_armed)]) >= 0.5
    observe_armed = float(obs[int(ObsIndex.observe_armed)]) >= 0.5
    role = ROLE_NAMES.get(int(role_id), "unknown")
    if role == "striker":
        if skill_name == "shoot":
            return shoot_armed
        if skill_name == "dribble":
            return dribble_armed
        if skill_name == "pass":
            return pass_armed
        return skill_name in {"stand", "walk"}
    if role == "open_support":
        if skill_name == "observe":
            return observe_armed
        return skill_name in {"stand", "walk"}
    if role == "off_ball_support":
        if skill_name == "dribble":
            return dribble_armed
        if skill_name == "block":
            return dribble_armed
        if skill_name == "observe":
            return observe_armed
        return skill_name in {"stand", "walk"}
    return False


def audit_dataset(
    dataset_dir: Path,
    *,
    min_rows: int,
    min_episodes: int,
    min_striker_dribble_armed: int,
    min_striker_shoot_armed: int,
    min_striker_dribble_frac: float = 0.15,
    min_striker_shoot_frac: float = 0.03,
    min_striker_pass_armed: int = 1,
) -> tuple[dict[str, object], int]:
    summary_path = dataset_dir / "summary.json"
    samples_path = dataset_dir / "samples.npz"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    summary = json.loads(summary_path.read_text())
    samples = np.load(samples_path, allow_pickle=True)
    observations = np.asarray(samples["observations"], dtype=np.float32)
    actions = np.asarray(samples["actions"], dtype=np.float32)
    role_ids = np.asarray(samples["role_ids"], dtype=np.int64)
    episode_ids = np.asarray(samples["episode_ids"], dtype=np.int64)
    stages = np.asarray(samples["stages"], dtype=np.int64)
    coords = np.asarray(samples["coordination_target_xy"], dtype=np.float32)
    weights = np.asarray(samples["weights"], dtype=np.float32)
    label_consistent = np.asarray(samples["label_consistent"], dtype=np.int64)

    rows = int(observations.shape[0]) if observations.ndim == 2 else 0
    obs_size = int(observations.shape[1]) if observations.ndim == 2 and rows > 0 else int(summary.get("obs_size", 0))
    skill_ids = actions[:, 0].round().astype(np.int64) if rows else np.asarray([], dtype=np.int64)
    skill_names = [SKILL_ORDER[int(np.clip(skill_id, 0, len(SKILL_ORDER) - 1))] for skill_id in skill_ids]

    role_counts = Counter(ROLE_NAMES.get(int(r), f"unknown_{int(r)}") for r in role_ids.tolist())
    skill_counts = Counter(skill_names)
    role_skill_counts = Counter(
        f"{ROLE_NAMES.get(int(role_ids[i]), f'unknown_{int(role_ids[i])}')}/{skill_names[i]}"
        for i in range(rows)
    )
    striker_mask = role_ids == 0
    striker_skill_counts = Counter(skill_names[i] for i in range(rows) if bool(striker_mask[i]))
    scenario_rows = {
        str(s.get("name")): int(s.get("kept_episodes", 0)) for s in summary.get("scenarios", [])
    }
    all_scenarios_kept = bool(scenario_rows) and all(count > 0 for count in scenario_rows.values())
    drop_reasons = Counter()
    for scenario in summary.get("scenarios", []):
        drop_reasons.update({str(k): int(v) for k, v in dict(scenario.get("drop_reasons", {})).items()})

    role_mask_bad = 0
    positive_weight_bad = 0
    if rows:
        for i in range(rows):
            allowed = _role_mask_allows(int(role_ids[i]), skill_names[i], observations[i])
            if not allowed:
                role_mask_bad += 1
                if float(weights[i]) > 0.0:
                    positive_weight_bad += 1

    robot_x = observations[:, int(ObsIndex.robot_x)] * ACTION.x_range if rows else np.asarray([])
    robot_y = observations[:, int(ObsIndex.robot_y)] * ACTION.y_range if rows else np.asarray([])
    coord_dist = np.hypot(robot_x - coords[:, 0], robot_y - coords[:, 1]) if rows else np.asarray([])
    striker_dribble_armed = int(np.sum(striker_mask & (observations[:, int(ObsIndex.dribble_armed)] >= 0.5))) if rows else 0
    striker_shoot_armed = int(np.sum(striker_mask & (observations[:, int(ObsIndex.shoot_armed)] >= 0.5))) if rows else 0
    striker_pass_armed = int(np.sum(striker_mask & (observations[:, int(ObsIndex.pass_armed)] >= 0.5))) if rows else 0

    filters = dict(summary.get("filters", {}))
    open_radius = float(filters.get("open_support_hold_radius_mm", 500.0))
    off_ball_radius = float(filters.get("off_ball_hold_radius_mm", 600.0))
    min_open_hold_steps = int(filters.get("min_open_hold_steps", 20))
    min_off_ball_hold_steps = int(filters.get("min_off_ball_hold_steps", 30))
    support_mask = role_ids == 1
    off_ball_mask = role_ids == 2
    open_hold_steps = int(np.sum((coord_dist < open_radius) & support_mask)) if rows else 0
    off_ball_hold_steps = int(np.sum((coord_dist < off_ball_radius) & off_ball_mask)) if rows else 0
    open_dist_mean = float(np.mean(coord_dist[support_mask])) if np.any(support_mask) else None
    open_dist_p10 = float(np.percentile(coord_dist[support_mask], 10)) if np.any(support_mask) else None
    open_dist_min = float(np.min(coord_dist[support_mask])) if np.any(support_mask) else None

    unique_coord_bins = 0
    if rows:
        quantized = np.round(coords / 250.0).astype(np.int64)
        unique_coord_bins = int(len({tuple(x) for x in quantized.tolist()}))

    # Semantic liveness checks (added 2026-06-10 after the sane_v1 audit passed a
    # dataset whose policy-visible team obs was entirely dead — see memory
    # multiagent-dataset-deadness-20260610). Count-based gate minimums alone
    # cannot catch these.
    striker_rows = int(np.sum(striker_mask))
    striker_skill_ids = skill_ids[striker_mask] if rows else np.asarray([], dtype=np.int64)
    dribble_id = SKILL_ORDER.index("dribble")
    shoot_id = SKILL_ORDER.index("shoot")
    stand_id = SKILL_ORDER.index("stand")
    block_id = SKILL_ORDER.index("block")
    striker_dribble_frac = float(np.mean(striker_skill_ids == dribble_id)) if striker_rows else 0.0
    striker_shoot_frac = float(np.mean(striker_skill_ids == shoot_id)) if striker_rows else 0.0

    teammate_block = observations[:, int(ObsIndex.nearest_teammate_x_rel):int(ObsIndex.third_teammate_y_rel) + 1] if rows else np.zeros((0, 6))
    teammate_block_live = bool(rows and np.any(np.abs(teammate_block) > 1e-6))
    team_ball_block = observations[:, [int(ObsIndex.global_team_ball_x_rel), int(ObsIndex.global_team_ball_y_rel)]] if rows else np.zeros((0, 2))
    goal_block = observations[:, [int(ObsIndex.opponent_goal_x_rel), int(ObsIndex.opponent_goal_y_rel)]] if rows else np.zeros((0, 2))
    team_ball_distinct_from_goal = bool(rows and np.any(np.abs(team_ball_block - goal_block) > 1e-6))
    goal_vector_live = bool(rows and np.any(np.abs(goal_block) > 1e-6))

    per_stage_striker = {}
    for stage in sorted(set(stages.tolist())):
        stage_striker = striker_mask & (stages == stage)
        n = int(np.sum(stage_striker))
        sk = skill_ids[stage_striker]
        per_stage_striker[int(stage)] = {
            "rows": n,
            "dribble_frac": float(np.mean(sk == dribble_id)) if n else 0.0,
            "shoot_frac": float(np.mean(sk == shoot_id)) if n else 0.0,
            "shoot_armed_rows": int(np.sum(stage_striker & (observations[:, int(ObsIndex.shoot_armed)] >= 0.5))),
            "pass_armed_rows": int(np.sum(stage_striker & (observations[:, int(ObsIndex.pass_armed)] >= 0.5))),
        }

    stage_id = int(summary.get("stage_id", -1))
    team_ball_valid = observations[:, int(ObsIndex.time_since_team_ball_update)] < 0.5 if rows else np.asarray([], dtype=bool)
    # global_team_ball_x_rel is robot-relative, not absolute field x. For stage 8
    # the scenario set itself is defensive, so use the defender role rows as the
    # defense-phase audit surface instead of inferring phase from a relative obs.
    team_ball_x = observations[:, int(ObsIndex.global_team_ball_x_rel)] * ACTION.x_range if rows else np.asarray([])
    defense_mask = off_ball_mask if int(summary.get("stage_id", -1)) == 8 else (team_ball_x < -300.0) if rows else np.asarray([], dtype=bool)
    defender_defense_mask = off_ball_mask & defense_mask
    defender_valid_team_ball = defender_defense_mask & team_ball_valid
    defender_defense_rows = int(np.sum(defender_defense_mask)) if rows else 0
    defender_block_rows = int(np.sum(defender_defense_mask & (skill_ids == block_id))) if rows else 0
    defender_stand_valid_rows = int(np.sum(defender_valid_team_ball & (skill_ids == stand_id))) if rows else 0
    defender_valid_rows = int(np.sum(defender_valid_team_ball)) if rows else 0
    defender_block_frac = float(defender_block_rows / max(1, defender_defense_rows))
    defender_stand_valid_frac = float(defender_stand_valid_rows / max(1, defender_valid_rows))

    checks = {
        "rows_min": rows >= min_rows,
        "episodes_min": int(summary.get("total_episodes", 0)) >= min_episodes,
        "obs_size_47": obs_size == 47,
        "summary_gate_consistent": bool(summary.get("positive_weight_gate_label_consistent", False)),
        "summary_role_mask_consistent": bool(summary.get("positive_weight_role_mask_consistent", False)),
        "computed_role_mask_consistent": positive_weight_bad == 0,
        "label_consistent_all": bool(rows and np.all(label_consistent == 1)),
        "has_striker_rows": role_counts.get("striker", 0) > 0,
        "has_open_support_rows": role_counts.get("open_support", 0) > 0,
        "striker_dribble_armed_min": striker_dribble_armed >= min_striker_dribble_armed,
        "striker_shoot_armed_min": striker_shoot_armed >= min_striker_shoot_armed,
        "striker_dribble_frac_min": striker_dribble_frac >= min_striker_dribble_frac,
        "striker_shoot_frac_min": striker_shoot_frac >= min_striker_shoot_frac,
        "striker_pass_armed_min": striker_pass_armed >= min_striker_pass_armed,
        "teammate_block_live": teammate_block_live,
        "team_ball_distinct_from_goal": team_ball_distinct_from_goal,
        "goal_vector_live": goal_vector_live,
        "coord_diverse": unique_coord_bins >= 2,
        "stage_present": len(set(stages.tolist())) > 0 if rows else False,
        "open_support_hold_present": (not np.any(support_mask)) or open_hold_steps >= min_open_hold_steps,
        "off_ball_hold_present": (not np.any(off_ball_mask)) or off_ball_hold_steps >= min_off_ball_hold_steps,
        "all_scenarios_kept": all_scenarios_kept,
    }
    if stage_id == 8:
        checks.update({
            "defense_phase_rows_present": defender_defense_rows > 0,
            "defender_block_frac_positive": defender_block_frac > 0.0,
            "defender_stand_valid_team_ball_below_0p5": defender_stand_valid_frac < 0.5,
        })
    passed = all(checks.values())

    report = {
        "dataset_dir": str(dataset_dir),
        "summary": {
            "stage_id": int(summary.get("stage_id", -1)),
            "total_rows": rows,
            "total_episodes": int(summary.get("total_episodes", 0)),
            "obs_size": obs_size,
            "scenario_kept_episodes": scenario_rows,
            "drop_reasons": dict(drop_reasons),
        },
        "counts": {
            "role_counts": dict(role_counts),
            "skill_counts": dict(skill_counts),
            "striker_skill_counts": dict(striker_skill_counts),
            "role_skill_counts": dict(role_skill_counts),
            "unique_coord_bins_250mm": unique_coord_bins,
        },
        "striker_gate_quality": {
            "dribble_armed_rows": striker_dribble_armed,
            "shoot_armed_rows": striker_shoot_armed,
            "pass_armed_rows": striker_pass_armed,
            "min_dribble_armed_rows": int(min_striker_dribble_armed),
            "min_shoot_armed_rows": int(min_striker_shoot_armed),
            "dribble_frac": striker_dribble_frac,
            "shoot_frac": striker_shoot_frac,
            "min_dribble_frac": float(min_striker_dribble_frac),
            "min_shoot_frac": float(min_striker_shoot_frac),
            "min_pass_armed_rows": int(min_striker_pass_armed),
            "per_stage_striker": per_stage_striker,
        },
        "obs_liveness": {
            "teammate_block_live": teammate_block_live,
            "team_ball_distinct_from_goal": team_ball_distinct_from_goal,
            "goal_vector_live": goal_vector_live,
        },
        "support_quality": {
            "open_hold_radius_mm": open_radius,
            "off_ball_hold_radius_mm": off_ball_radius,
            "open_hold_steps": open_hold_steps,
            "off_ball_hold_steps": off_ball_hold_steps,
            "open_dist_mean_mm": open_dist_mean,
            "open_dist_p10_mm": open_dist_p10,
            "open_dist_min_mm": open_dist_min,
        },
        "defender_quality": {
            "defender_defense_rows": defender_defense_rows,
            "defender_block_rows": defender_block_rows,
            "defender_block_frac": defender_block_frac,
            "defender_valid_team_ball_rows": defender_valid_rows,
            "defender_stand_valid_team_ball_rows": defender_stand_valid_rows,
            "defender_stand_valid_team_ball_frac": defender_stand_valid_frac,
        },
        "consistency": {
            "role_mask_bad_rows": role_mask_bad,
            "positive_weight_role_mask_bad_rows": positive_weight_bad,
        },
        "checks": checks,
        "passed": passed,
    }
    return report, 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit multi-agent teacher dataset artifacts.")
    parser.add_argument("dataset_dir", nargs="?", type=Path)
    parser.add_argument("--root", type=Path, default=Path("runs/teacher_multiagent_dataset"))
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--min-striker-dribble-armed", type=int, default=1)
    parser.add_argument("--min-striker-shoot-armed", type=int, default=1)
    parser.add_argument("--min-striker-dribble-frac", type=float, default=0.15)
    parser.add_argument("--min-striker-shoot-frac", type=float, default=0.03)
    parser.add_argument("--min-striker-pass-armed", type=int, default=1)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or _latest_dataset(args.root)
    report, code = audit_dataset(
        dataset_dir,
        min_rows=args.min_rows,
        min_episodes=args.min_episodes,
        min_striker_dribble_armed=args.min_striker_dribble_armed,
        min_striker_shoot_armed=args.min_striker_shoot_armed,
        min_striker_dribble_frac=args.min_striker_dribble_frac,
        min_striker_shoot_frac=args.min_striker_shoot_frac,
        min_striker_pass_armed=args.min_striker_pass_armed,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.write:
        (dataset_dir / "audit.json").write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
