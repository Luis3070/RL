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
from pybh_backend import PyBHBackend, WorldSpec, build_world_spec

RUNS_DIR = ROOT_DIR / "runs" / "embedded_warmup_sweep"

LIVE_BALL_SEEN_TIMEOUT_MS = 1000.0
LIVE_RESET_BALL_POS = np.array([0.0, 0.0], dtype=np.float32)
LIVE_RESET_ROBOT_POS = np.array([-2438.0, 1875.0], dtype=np.float32)
LIVE_RESET_ROBOT_THETA = -math.pi / 2.0


@dataclass(frozen=True)
class WarmupCase:
    warmup_ms: int
    policy_mode: str  # stand | search

    @property
    def name(self) -> str:
        return f"embedded_{self.policy_mode}_{self.warmup_ms}ms"


CASES = tuple(
    WarmupCase(warmup_ms=warmup_ms, policy_mode=policy_mode)
    for warmup_ms in (500, 1000, 1500, 2000)
    for policy_mode in ("stand", "search")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep embedded post-reset warmup windows.")
    parser.add_argument("--case", choices=[case.name for case in CASES], default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def fixed_probe_world_spec() -> WorldSpec:
    return build_world_spec(
        ball_pos=LIVE_RESET_BALL_POS,
        robot_pos=LIVE_RESET_ROBOT_POS,
        robot_theta=LIVE_RESET_ROBOT_THETA,
    )


def natural_reliable(obs) -> bool:
    return (
        float(obs.natural_ball_seen_percentage) > 0.0
        and float(obs.natural_time_since_ball_seen) <= LIVE_BALL_SEEN_TIMEOUT_MS
    )


def export_reliable(obs) -> bool:
    return (
        float(obs.ball_seen_percentage) > 0.0
        and float(obs.time_since_ball_seen) <= LIVE_BALL_SEEN_TIMEOUT_MS
    )


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


def row_from_obs(case: WarmupCase, step: int, obs, action: StructuredSkillAction) -> dict[str, object]:
    reset_age = int(obs.frame - obs.reset_applied_frame) if obs.reset_applied_frame and obs.frame >= obs.reset_applied_frame else -1
    return {
        "case": case.name,
        "warmup_ms": case.warmup_ms,
        "policy_mode": case.policy_mode,
        "step": step,
        "frame": int(obs.frame),
        "reset_applied_frame": int(obs.reset_applied_frame),
        "reset_applied_age_ms": reset_age,
        "skill_name": action.skill_name,
        "export_ball_seen_percentage": float(obs.ball_seen_percentage),
        "export_time_since_ball_seen": float(obs.time_since_ball_seen),
        "natural_ball_seen_percentage": float(obs.natural_ball_seen_percentage),
        "natural_time_since_ball_seen": float(obs.natural_time_since_ball_seen),
        "natural_ball_consistent": int(bool(obs.natural_ball_consistent_with_game_state)),
        "export_reliable": int(export_reliable(obs)),
        "natural_reliable": int(natural_reliable(obs)),
        "ball_export_source": int(obs.ball_export_source),
        "ball_export_fresh": int(bool(obs.ball_export_fresh)),
        "corrected_exported_ball": int(bool(obs.corrected_exported_ball)),
        "synthetic_ball_observation": int(bool(obs.synthetic_ball_observation)),
        "ball_observation_stale": int(bool(obs.ball_observation_stale)),
    }


def summarize_rows(case: WarmupCase, rows: list[dict[str, object]], error: str = "") -> dict[str, object]:
    if not rows:
        return {
            "case": case.name,
            "warmup_ms": case.warmup_ms,
            "policy_mode": case.policy_mode,
            "steps": 0,
            "export_reliable_rate": 0.0,
            "natural_reliable_rate": 0.0,
            "synthetic_ball_rate": 0.0,
            "ball_stale_rate": 0.0,
            "ball_export_fresh_rate": 0.0,
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
        "warmup_ms": case.warmup_ms,
        "policy_mode": case.policy_mode,
        "steps": total,
        "final_reset_age_ms": int(rows[-1]["reset_applied_age_ms"]),
        "export_reliable_rate": sum(int(row["export_reliable"]) for row in rows) / total,
        "natural_reliable_rate": sum(int(row["natural_reliable"]) for row in rows) / total,
        "synthetic_ball_rate": sum(int(row["synthetic_ball_observation"]) for row in rows) / total,
        "ball_stale_rate": sum(int(row["ball_observation_stale"]) for row in rows) / total,
        "ball_export_fresh_rate": sum(int(row["ball_export_fresh"]) for row in rows) / total,
        "first_export_reliable_after_reset_ms": first_age("export_reliable"),
        "first_natural_reliable_after_reset_ms": first_age("natural_reliable"),
        "error": error,
    }


def run_case(case: WarmupCase, output_dir: Path) -> dict[str, object]:
    world_spec = fixed_probe_world_spec()
    rows: list[dict[str, object]] = []
    error = ""
    backend = PyBHBackend(
        mode="simrobot",
        obs_mode="corrected",
        frames_per_step=3,
        obs_timeout_ms=1500,
        simrobot_profile="3d",
        embedded_3d_preflight=True,
    )
    try:
        obs = backend.reset(world_spec=world_spec)
        action_fn = stand_action if case.policy_mode == "stand" else search_action
        step = 0
        while True:
            step += 1
            action = action_fn(obs)
            obs = backend.step(action)
            row = row_from_obs(case, step, obs, action)
            rows.append(row)
            if int(row["reset_applied_age_ms"]) >= case.warmup_ms:
                break
            if step >= 160:
                error = "step_cap_reached"
                break
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

    summary = summarize_rows(case, rows, error=error)
    summary_csv = output_dir / f"{case.name}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.case is not None:
        output_dir = (args.output_dir or (RUNS_DIR / "manual")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        case = next(case for case in CASES if case.name == args.case)
        print(run_case(case, output_dir), flush=True)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or (RUNS_DIR / f"warmup_{timestamp}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for case in CASES:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", case.name, "--output-dir", str(output_dir)],
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
                    "warmup_ms": case.warmup_ms,
                    "policy_mode": case.policy_mode,
                    "steps": 0,
                    "final_reset_age_ms": -1,
                    "export_reliable_rate": 0.0,
                    "natural_reliable_rate": 0.0,
                    "synthetic_ball_rate": 0.0,
                    "ball_stale_rate": 0.0,
                    "ball_export_fresh_rate": 0.0,
                    "first_export_reliable_after_reset_ms": -1,
                    "first_natural_reliable_after_reset_ms": -1,
                    "error": f"subprocess_exit_{completed.returncode}",
                }
            )
            continue
        with (output_dir / f"{case.name}_summary.csv").open(encoding="utf-8") as f:
            summaries.extend(dict(row) for row in csv.DictReader(f))

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[embedded_warmup_sweep] output_dir={output_dir}", flush=True)
    for summary in summaries:
        print(summary, flush=True)


if __name__ == "__main__":
    main()
