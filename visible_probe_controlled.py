from __future__ import annotations

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from action import SkillType, StructuredSkillAction
from config import ROOT_DIR, SIMROBOT_BIN
from pybh_backend import PyBHBackend, build_world_spec

RUNS_DIR = ROOT_DIR / "runs" / "visible_probe_controlled"

LIVE_SHOOT_DISTANCE_MM = 260.0
LIVE_BALL_SEEN_TIMEOUT_MS = 1000.0
LIVE_RESET_BALL_POS = np.array([0.0, 0.0], dtype=np.float32)
LIVE_RESET_ROBOT_POS = np.array([-2438.0, 1875.0], dtype=np.float32)
LIVE_RESET_ROBOT_THETA = -math.pi / 2.0


def exported_reliable(obs) -> bool:
    return float(obs.ball_seen_percentage) > 0.0 and float(obs.time_since_ball_seen) <= LIVE_BALL_SEEN_TIMEOUT_MS


def natural_reliable(obs) -> bool:
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


def heuristic_action(obs) -> StructuredSkillAction:
    if not exported_reliable(obs):
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


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RUNS_DIR / f"visible_probe_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_csv = output_dir / "steps.csv"
    summary_csv = output_dir / "summary.csv"

    backend = PyBHBackend(
        mode="simrobot_visible",
        obs_mode="corrected",
        frames_per_step=1,
        obs_timeout_ms=30000,
        visible_request_timeout_ms=30000,
        visible_reset_obs_timeout_ms=30000,
        visible_status_log_interval_s=1.0,
        simrobot_profile="3d",
        simrobot_launch=True,
        simrobot_bin=SIMROBOT_BIN,
    )
    world_spec = build_world_spec(
        ball_pos=LIVE_RESET_BALL_POS,
        robot_pos=LIVE_RESET_ROBOT_POS,
        robot_theta=LIVE_RESET_ROBOT_THETA,
    )

    rows: list[dict[str, object]] = []
    try:
        backend.start()
        print("Abre/trae SimRobot al frente, presiona F5 y luego vuelve aqui.", flush=True)
        input("Cuando la simulacion visible ya este corriendo, presiona Enter para enviar el reset...")
        obs = backend.reset(world_spec=world_spec)
        period = 0.1
        for step in range(1, 25):
            action = heuristic_action(obs)
            obs = backend.step(action)
            reset_age = int(obs.frame - obs.reset_applied_frame) if obs.reset_applied_frame and obs.frame >= obs.reset_applied_frame else -1
            row = {
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
                "export_reliable": int(exported_reliable(obs)),
                "natural_reliable": int(natural_reliable(obs)),
                "ball_export_source": int(obs.ball_export_source),
                "ball_export_fresh": int(bool(obs.ball_export_fresh)),
                "synthetic_ball_observation": int(bool(obs.synthetic_ball_observation)),
                "ball_observation_stale": int(bool(obs.ball_observation_stale)),
            }
            rows.append(row)
            print(row, flush=True)
            time.sleep(period)
    finally:
        backend.close()

    if rows:
        with steps_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "steps": len(rows),
            "export_reliable_rate": sum(int(r["export_reliable"]) for r in rows) / len(rows),
            "natural_reliable_rate": sum(int(r["natural_reliable"]) for r in rows) / len(rows),
            "synthetic_ball_rate": sum(int(r["synthetic_ball_observation"]) for r in rows) / len(rows),
            "ball_stale_rate": sum(int(r["ball_observation_stale"]) for r in rows) / len(rows),
            "ball_export_fresh_rate": sum(int(r["ball_export_fresh"]) for r in rows) / len(rows),
            "first_export_reliable_after_reset_ms": next((int(r["reset_applied_age_ms"]) for r in rows if int(r["export_reliable"])), -1),
            "first_natural_reliable_after_reset_ms": next((int(r["reset_applied_age_ms"]) for r in rows if int(r["natural_reliable"])), -1),
        }
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        print(f"[visible_probe_controlled] output_dir={output_dir}", flush=True)
        print(summary, flush=True)


if __name__ == "__main__":
    main()
