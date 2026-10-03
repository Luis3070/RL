from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from action import SKILL_ORDER
from defender_policy import FIELD_X_HALF, FIELD_Y_HALF, teacher_action
from observation import OBS_SIZE, ObsIndex


def generate_samples(count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = np.zeros((count, OBS_SIZE), dtype=np.float32)

    scenario = rng.choice(
        ("normal", "front_threat", "clearance", "stale", "lost", "box_danger"),
        size=count,
        p=(0.30, 0.22, 0.18, 0.12, 0.08, 0.10),
    )
    robot_x = rng.uniform(-3600.0, -800.0, size=count)
    robot_y = rng.uniform(-1800.0, 1800.0, size=count)
    robot_theta = rng.uniform(-np.pi, np.pi, size=count)
    ball_rel_x = rng.uniform(250.0, 3200.0, size=count)
    ball_rel_y = rng.uniform(-1900.0, 1900.0, size=count)

    danger = np.isin(scenario, ("front_threat", "box_danger"))
    clearance = scenario == "clearance"
    stale = np.isin(scenario, ("stale", "lost"))
    ball_rel_x[danger] = rng.uniform(250.0, 1500.0, size=int(danger.sum()))
    ball_rel_y[danger] = rng.uniform(-900.0, 900.0, size=int(danger.sum()))
    ball_rel_x[clearance] = rng.uniform(120.0, 600.0, size=int(clearance.sum()))
    ball_rel_y[clearance] = rng.uniform(-500.0, 500.0, size=int(clearance.sum()))

    obs[:, ObsIndex.robot_x] = np.clip(robot_x / FIELD_X_HALF, -1.0, 1.0)
    obs[:, ObsIndex.robot_y] = np.clip(robot_y / FIELD_Y_HALF, -1.0, 1.0)
    obs[:, ObsIndex.robot_theta] = robot_theta / np.pi
    obs[:, ObsIndex.ball_rel_x] = np.clip(ball_rel_x / FIELD_X_HALF, -1.0, 1.0)
    obs[:, ObsIndex.ball_rel_y] = np.clip(ball_rel_y / FIELD_Y_HALF, -1.0, 1.0)
    obs[:, ObsIndex.ball_end_rel_x] = obs[:, ObsIndex.ball_rel_x]
    obs[:, ObsIndex.ball_end_rel_y] = obs[:, ObsIndex.ball_rel_y]
    obs[:, ObsIndex.ball_vel_x] = rng.uniform(-0.25, 0.25, size=count)
    obs[:, ObsIndex.ball_vel_y] = rng.uniform(-0.25, 0.25, size=count)
    obs[:, ObsIndex.time_since_ball_seen] = rng.uniform(0.0, 0.18, size=count)
    obs[:, ObsIndex.time_since_ball_disappeared] = obs[:, ObsIndex.time_since_ball_seen]
    obs[:, ObsIndex.ball_seen_percentage] = rng.uniform(0.55, 1.0, size=count)
    obs[:, ObsIndex.ball_consistent_with_game_state] = 1.0
    obs[:, ObsIndex.can_score_now] = 0.0
    obs[:, ObsIndex.shot_quality_no_obstacles] = rng.uniform(0.0, 0.35, size=count)
    obs[:, ObsIndex.shot_opening_with_obstacles] = rng.uniform(0.0, 0.35, size=count)
    obs[:, ObsIndex.pass_options_count] = rng.uniform(0.0, 1.0, size=count)
    obs[:, ObsIndex.nearest_teammate_dist] = rng.uniform(0.20, 1.0, size=count)
    obs[:, ObsIndex.nearest_opponent_dist] = rng.uniform(0.12, 1.0, size=count)
    obs[:, ObsIndex.nearest_uncertain_obstacle_dist] = rng.uniform(0.20, 1.0, size=count)
    obs[:, ObsIndex.nearest_teammate_front_dist] = rng.uniform(0.20, 1.0, size=count)
    obs[:, ObsIndex.nearest_opponent_front_dist] = rng.uniform(0.12, 1.0, size=count)
    obs[:, ObsIndex.nearest_uncertain_front_dist] = rng.uniform(0.20, 1.0, size=count)

    obs[stale, ObsIndex.time_since_ball_seen] = rng.uniform(0.50, 1.0, size=int(stale.sum()))
    obs[stale, ObsIndex.time_since_ball_disappeared] = obs[stale, ObsIndex.time_since_ball_seen]
    obs[stale, ObsIndex.ball_seen_percentage] = rng.uniform(0.0, 0.10, size=int(stale.sum()))

    obs[danger, ObsIndex.dribble_armed] = 1.0
    obs[clearance, ObsIndex.shoot_armed] = 1.0
    obs[clearance, ObsIndex.shoot_arm_progress] = rng.uniform(0.55, 1.0, size=int(clearance.sum()))
    obs[clearance, ObsIndex.nearest_teammate_front_dist] = rng.uniform(0.35, 1.0, size=int(clearance.sum()))
    obs[clearance, ObsIndex.nearest_opponent_front_dist] = rng.uniform(0.25, 1.0, size=int(clearance.sum()))
    obs[clearance, ObsIndex.nearest_uncertain_front_dist] = rng.uniform(0.30, 1.0, size=int(clearance.sum()))

    actions = np.stack([teacher_action(row).as_array() for row in obs], axis=0).astype(np.float32)
    weights = np.ones(count, dtype=np.float32)
    weights[np.isin(actions[:, 0].round().astype(int), [SKILL_ORDER.index("pass"), SKILL_ORDER.index("block")])] = 1.8
    return obs, actions, weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic obs26 defender BC dataset.")
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=260611)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    root = Path(args.output_dir) if args.output_dir else Path("runs/defender_bc") / f"dataset_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=True)
    obs, actions, weights = generate_samples(args.samples, args.seed)
    np.savez_compressed(root / "samples.npz", observations=obs, actions=actions, weights=weights, skill_order=np.asarray(SKILL_ORDER))
    skill_ids = actions[:, 0].round().astype(int)
    summary = {
        "samples": int(args.samples),
        "obs_size": int(obs.shape[1]),
        "skill_counts": {SKILL_ORDER[i]: int(np.count_nonzero(skill_ids == i)) for i in range(len(SKILL_ORDER))},
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(root / "samples.npz")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
