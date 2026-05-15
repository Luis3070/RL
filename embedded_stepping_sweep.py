from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from action import SkillType, StructuredSkillAction
from config import ROOT_DIR
from pybh_backend import PyBHBackend, build_world_spec

RUNS_DIR = ROOT_DIR / "runs" / "embedded_stepping_sweep"

LIVE_BALL_SEEN_TIMEOUT_MS = 1000.0
LIVE_RESET_BALL_POS = np.array([0.0, 0.0], dtype=np.float32)
LIVE_RESET_ROBOT_POS = np.array([-2438.0, 1875.0], dtype=np.float32)
LIVE_RESET_ROBOT_THETA = -math.pi / 2.0


@dataclass(frozen=True)
class StepCase:
    frames_per_step: int
    policy_mode: str  # stand | search

    @property
    def name(self) -> str:
        return f"embedded_{self.policy_mode}_fps{self.frames_per_step}"


CASES = tuple(
    StepCase(frames_per_step=fps, policy_mode=mode)
    for fps in (1, 3, 6, 12)
    for mode in ("stand", "search")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep embedded frames_per_step with probe geometry.")
    parser.add_argument("--case", choices=[case.name for case in CASES], default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def world_spec():
    return build_world_spec(
        ball_pos=LIVE_RESET_BALL_POS,
        robot_pos=LIVE_RESET_ROBOT_POS,
        robot_theta=LIVE_RESET_ROBOT_THETA,
    )


def natural_reliable(obs) -> bool:
    return float(obs.natural_ball_seen_percentage) > 0.0 and float(obs.natural_time_since_ball_seen) <= LIVE_BALL_SEEN_TIMEOUT_MS


def export_reliable(obs) -> bool:
    return float(obs.ball_seen_percentage) > 0.0 and float(obs.time_since_ball_seen) <= LIVE_BALL_SEEN_TIMEOUT_MS


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


def stand_action(obs) -> StructuredSkillAction:
    del obs
    return StructuredSkillAction(skill_type=SkillType.stand)


def row_from_obs(case: StepCase, step: int, obs, action: StructuredSkillAction) -> dict[str, object]:
    return {
        "case": case.name,
        "frames_per_step": case.frames_per_step,
        "policy_mode": case.policy_mode,
        "step": step,
        "frame": int(obs.frame),
        "reset_applied_age_ms": int(obs.frame - obs.reset_applied_frame) if obs.reset_applied_frame and obs.frame >= obs.reset_applied_frame else -1,
        "skill_name": action.skill_name,
        "export_reliable": int(export_reliable(obs)),
        "natural_reliable": int(natural_reliable(obs)),
        "export_ball_seen_percentage": float(obs.ball_seen_percentage),
        "export_time_since_ball_seen": float(obs.time_since_ball_seen),
        "natural_ball_seen_percentage": float(obs.natural_ball_seen_percentage),
        "natural_time_since_ball_seen": float(obs.natural_time_since_ball_seen),
        "ball_export_source": int(obs.ball_export_source),
        "ball_export_fresh": int(bool(obs.ball_export_fresh)),
        "synthetic_ball_observation": int(bool(obs.synthetic_ball_observation)),
        "ball_observation_stale": int(bool(obs.ball_observation_stale)),
    }


def summarize(case: StepCase, rows: list[dict[str, object]], error: str = "") -> dict[str, object]:
    if not rows:
        return {
            "case": case.name,
            "frames_per_step": case.frames_per_step,
            "policy_mode": case.policy_mode,
            "steps": 0,
            "final_reset_age_ms": -1,
            "export_reliable_rate": 0.0,
            "natural_reliable_rate": 0.0,
            "synthetic_ball_rate": 0.0,
            "ball_stale_rate": 0.0,
            "first_export_reliable_after_reset_ms": -1,
            "first_natural_reliable_after_reset_ms": -1,
            "error": error or "no_rows",
        }
    def first_age(key: str) -> int:
        for row in rows:
            if int(row[key]):
                return int(row["reset_applied_age_ms"])
        return -1
    total = len(rows)
    return {
        "case": case.name,
        "frames_per_step": case.frames_per_step,
        "policy_mode": case.policy_mode,
        "steps": total,
        "final_reset_age_ms": int(rows[-1]["reset_applied_age_ms"]),
        "export_reliable_rate": sum(int(r["export_reliable"]) for r in rows) / total,
        "natural_reliable_rate": sum(int(r["natural_reliable"]) for r in rows) / total,
        "synthetic_ball_rate": sum(int(r["synthetic_ball_observation"]) for r in rows) / total,
        "ball_stale_rate": sum(int(r["ball_observation_stale"]) for r in rows) / total,
        "first_export_reliable_after_reset_ms": first_age("export_reliable"),
        "first_natural_reliable_after_reset_ms": first_age("natural_reliable"),
        "error": error,
    }


def run_case(case: StepCase, output_dir: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    error = ""
    backend = PyBHBackend(
        mode="simrobot",
        obs_mode="corrected",
        frames_per_step=case.frames_per_step,
        obs_timeout_ms=1500,
        simrobot_profile="3d",
        embedded_3d_preflight=True,
    )
    try:
        obs = backend.reset(world_spec=world_spec())
        action_fn = stand_action if case.policy_mode == "stand" else search_action
        for step in range(1, 25):
            action = action_fn(obs)
            obs = backend.step(action)
            rows.append(row_from_obs(case, step, obs, action))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()
    steps_csv = output_dir / f"{case.name}_steps.csv"
    if rows:
        with steps_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(case, rows, error)
    with (output_dir / f"{case.name}_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.case is not None:
        out = (args.output_dir or (RUNS_DIR / "manual")).resolve()
        out.mkdir(parents=True, exist_ok=True)
        case = next(case for case in CASES if case.name == args.case)
        print(run_case(case, out), flush=True)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output_dir or (RUNS_DIR / f"stepping_{timestamp}")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for case in CASES:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", case.name, "--output-dir", str(out)],
            cwd=ROOT_DIR,
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
                    "frames_per_step": case.frames_per_step,
                    "policy_mode": case.policy_mode,
                    "steps": 0,
                    "final_reset_age_ms": -1,
                    "export_reliable_rate": 0.0,
                    "natural_reliable_rate": 0.0,
                    "synthetic_ball_rate": 0.0,
                    "ball_stale_rate": 0.0,
                    "first_export_reliable_after_reset_ms": -1,
                    "first_natural_reliable_after_reset_ms": -1,
                    "error": f"subprocess_exit_{completed.returncode}",
                }
            )
            continue
        with (out / f"{case.name}_summary.csv").open(encoding="utf-8") as f:
            summaries.extend(dict(row) for row in csv.DictReader(f))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[embedded_stepping_sweep] output_dir={out}", flush=True)
    for summary in summaries:
        print(summary, flush=True)


if __name__ == "__main__":
    main()
