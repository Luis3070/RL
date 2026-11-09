from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from action import SkillType, StructuredSkillAction
from config import BHUMAN_DIR, DT, ROOT_DIR, SIMROBOT_BIN
from pybh_backend import PyBHBackend, WorldSpec, build_world_spec

RUNS_DIR = ROOT_DIR / "runs" / "probe_protocol_experiment"

LIVE_SHOOT_DISTANCE_MM = 260.0
LIVE_BALL_SEEN_TIMEOUT_MS = 1000.0
LIVE_RESET_BALL_POS = np.array([0.0, 0.0], dtype=np.float32)
LIVE_RESET_ROBOT_POS = np.array([-2438.0, 1875.0], dtype=np.float32)
LIVE_RESET_ROBOT_THETA = -math.pi / 2.0


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    backend_mode: str
    policy_mode: str
    simrobot_launch: bool = False
    frames_per_step: int = 3
    hz: float = 10.0
    steps: int = 24
    obs_mode: str = "corrected"
    obs_timeout_ms: int = 1500


CASES = (
    ExperimentCase(name="embedded_passive", backend_mode="simrobot", policy_mode="passive", frames_per_step=3),
    ExperimentCase(name="embedded_search", backend_mode="simrobot", policy_mode="search", frames_per_step=3),
    ExperimentCase(
        name="visible_search",
        backend_mode="simrobot_visible",
        policy_mode="search",
        simrobot_launch=True,
        frames_per_step=1,
        hz=10.0,
        obs_timeout_ms=2000,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare probe-aligned reset/stepping protocols.")
    parser.add_argument("--case", choices=[case.name for case in CASES], default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def fixed_probe_world_spec() -> WorldSpec:
    return build_world_spec(
        ball_pos=LIVE_RESET_BALL_POS,
        robot_pos=LIVE_RESET_ROBOT_POS,
        robot_theta=LIVE_RESET_ROBOT_THETA,
    )


def ball_observation_reliable_exported(obs) -> bool:
    return float(obs.ball_seen_percentage) > 0.0 and float(obs.time_since_ball_seen) <= LIVE_BALL_SEEN_TIMEOUT_MS


def ball_observation_reliable_natural(obs) -> bool:
    return float(obs.natural_ball_seen_percentage) > 0.0 and float(obs.natural_time_since_ball_seen) <= LIVE_BALL_SEEN_TIMEOUT_MS


def search_action(obs) -> StructuredSkillAction:
    robot_pos = np.array([float(obs.robot_x), float(obs.robot_y)], dtype=np.float32)
    delta = LIVE_RESET_BALL_POS - robot_pos
    target_theta = float(math.atan2(float(delta[1]), float(delta[0]))) if np.linalg.norm(delta) > 1.0 else 0.0
    return StructuredSkillAction(
        skill_type=SkillType.walk,
        target_x=float(LIVE_RESET_BALL_POS[0]),
        target_y=float(LIVE_RESET_BALL_POS[1]),
        target_theta=target_theta,
    )


def heuristic_probe_action(obs) -> StructuredSkillAction:
    if not ball_observation_reliable_exported(obs):
        return search_action(obs)

    robot_pos = np.array([float(obs.robot_x), float(obs.robot_y)], dtype=np.float32)
    ball_pos = np.array([float(obs.ball_x), float(obs.ball_y)], dtype=np.float32)
    delta = ball_pos - robot_pos
    distance = float(np.linalg.norm(delta))
    if distance <= LIVE_SHOOT_DISTANCE_MM:
        return StructuredSkillAction(skill_type=SkillType.shoot)

    target_theta = float(math.atan2(float(delta[1]), float(delta[0]))) if distance > 1.0 else 0.0
    return StructuredSkillAction(
        skill_type=SkillType.walk,
        target_x=float(ball_pos[0]),
        target_y=float(ball_pos[1]),
        target_theta=target_theta,
    )


def passive_action(obs) -> StructuredSkillAction:
    del obs
    return StructuredSkillAction(skill_type=SkillType.stand)


def row_from_obs(case: ExperimentCase, step: int, obs, action: StructuredSkillAction) -> dict[str, object]:
    return {
        "case": case.name,
        "backend_mode": case.backend_mode,
        "policy_mode": case.policy_mode,
        "step": step,
        "frame": int(obs.frame),
        "reset_applied_frame": int(obs.reset_applied_frame),
        "reset_applied_age_ms": int(obs.frame - obs.reset_applied_frame) if obs.reset_applied_frame and obs.frame >= obs.reset_applied_frame else -1,
        "skill_name": action.skill_name,
        "robot_x": float(obs.robot_x),
        "robot_y": float(obs.robot_y),
        "robot_theta": float(obs.robot_theta),
        "ball_x": float(obs.ball_x),
        "ball_y": float(obs.ball_y),
        "export_ball_seen_percentage": float(obs.ball_seen_percentage),
        "export_time_since_ball_seen": float(obs.time_since_ball_seen),
        "natural_ball_seen_percentage": float(obs.natural_ball_seen_percentage),
        "natural_time_since_ball_seen": float(obs.natural_time_since_ball_seen),
        "natural_ball_consistent": int(bool(obs.natural_ball_consistent_with_game_state)),
        "export_ball_consistent": int(bool(obs.ball_consistent_with_game_state)),
        "export_reliable": int(ball_observation_reliable_exported(obs)),
        "natural_reliable": int(ball_observation_reliable_natural(obs)),
        "ball_export_source": int(obs.ball_export_source),
        "ball_export_fresh": int(bool(obs.ball_export_fresh)),
        "corrected_exported_ball": int(bool(obs.corrected_exported_ball)),
        "synthetic_ball_observation": int(bool(obs.synthetic_ball_observation)),
        "ball_observation_stale": int(bool(obs.ball_observation_stale)),
    }


def summarize_case(rows: list[dict[str, object]], *, case: ExperimentCase, error: str | None = None) -> dict[str, object]:
    if not rows:
        return {"case": case.name, "backend_mode": case.backend_mode, "policy_mode": case.policy_mode, "error": error or "no_rows"}

    def first_step(pred: Callable[[dict[str, object]], bool]) -> int:
        for row in rows:
            if pred(row):
                return int(row["reset_applied_age_ms"])
        return -1

    total = len(rows)
    return {
        "case": case.name,
        "backend_mode": case.backend_mode,
        "policy_mode": case.policy_mode,
        "steps": total,
        "export_reliable_rate": sum(int(row["export_reliable"]) for row in rows) / total,
        "natural_reliable_rate": sum(int(row["natural_reliable"]) for row in rows) / total,
        "synthetic_ball_rate": sum(int(row["synthetic_ball_observation"]) for row in rows) / total,
        "ball_stale_rate": sum(int(row["ball_observation_stale"]) for row in rows) / total,
        "ball_export_fresh_rate": sum(int(row["ball_export_fresh"]) for row in rows) / total,
        "ball_export_source_reset_rate": sum(int(row["ball_export_source"] == 1) for row in rows) / total,
        "ball_export_source_estimated_rate": sum(int(row["ball_export_source"] == 0) for row in rows) / total,
        "first_export_reliable_after_reset_ms": first_step(lambda row: bool(row["export_reliable"])),
        "first_natural_reliable_after_reset_ms": first_step(lambda row: bool(row["natural_reliable"])),
        "first_non_synthetic_after_reset_ms": first_step(lambda row: not bool(row["synthetic_ball_observation"])),
        "error": error or "",
    }


def run_case(case: ExperimentCase, output_dir: Path) -> dict[str, object]:
    world_spec = fixed_probe_world_spec()
    steps_csv = output_dir / f"{case.name}_steps.csv"
    rows: list[dict[str, object]] = []
    error: str | None = None
    backend = PyBHBackend(
        mode=case.backend_mode,
        obs_mode=case.obs_mode,
        frames_per_step=case.frames_per_step,
        obs_timeout_ms=case.obs_timeout_ms,
        simrobot_profile="3d",
        simrobot_launch=case.simrobot_launch,
        embedded_3d_preflight=True,
        simrobot_bin=SIMROBOT_BIN,
    )
    try:
        obs = backend.reset(world_spec=world_spec)
        action_fn = passive_action if case.policy_mode == "passive" else heuristic_probe_action
        period = 1.0 / max(case.hz, 0.1)
        for step in range(1, case.steps + 1):
            action = action_fn(obs)
            obs = backend.step(action)
            row = row_from_obs(case, step, obs, action)
            rows.append(row)
            if case.backend_mode == "simrobot_visible":
                time.sleep(period)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()

    if rows:
        with steps_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return summarize_case(rows, case=case, error=error)


def main() -> None:
    args = parse_args()
    if args.case is not None:
        output_dir = (args.output_dir or RUNS_DIR / "manual").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        case = next(case for case in CASES if case.name == args.case)
        summary = run_case(case, output_dir)
        summary_path = output_dir / f"{case.name}_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        print(summary, flush=True)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or (RUNS_DIR / f"probe_protocol_{timestamp}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for case in CASES:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--case", case.name, "--output-dir", str(output_dir)]
        completed = subprocess.run(
            cmd,
            cwd=BHUMAN_DIR.parent / "RL",
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
        if completed.returncode != 0:
            summaries.append(
                {
                    "case": case.name,
                    "backend_mode": case.backend_mode,
                    "policy_mode": case.policy_mode,
                    "steps": 0,
                    "export_reliable_rate": 0.0,
                    "natural_reliable_rate": 0.0,
                    "synthetic_ball_rate": 0.0,
                    "ball_stale_rate": 0.0,
                    "ball_export_fresh_rate": 0.0,
                    "ball_export_source_reset_rate": 0.0,
                    "ball_export_source_estimated_rate": 0.0,
                    "first_export_reliable_after_reset_ms": -1,
                    "first_natural_reliable_after_reset_ms": -1,
                    "first_non_synthetic_after_reset_ms": -1,
                    "error": f"subprocess_exit_{completed.returncode}",
                }
            )
            continue
        summary_path = output_dir / f"{case.name}_summary.csv"
        with summary_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        summaries.append(dict(rows[0]))

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"[probe_protocol_experiment] output_dir={output_dir}", flush=True)
    for summary in summaries:
        print(summary, flush=True)


if __name__ == "__main__":
    main()
