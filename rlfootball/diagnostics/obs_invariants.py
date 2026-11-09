from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from rlfootball.core.action import StructuredSkillAction
from rlfootball.core.config import ACTION, ENV, ROOT_DIR
from rlfootball.env.environment import (
    BALL_TRUSTED_MAX_AGE_MS,
    BALL_VISIBLE_MAX_AGE_MS,
    BALL_VISIBLE_SEEN_PCT_MIN,
    OBS_SIZE,
    RealSimRobotEnv,
)
from rlfootball.core.observation import ObsIndex
from rlfootball.core.pybh_backend import SYNTHETIC_BALL_WARMUP_STEPS
from rlfootball.core.visible_parallel_runtime import VisibleParallelRuntime

RUNS_DIR = ROOT_DIR / "runs" / "obs_invariants"
SHM_NAME = "/pybh_rl_shared_state_v3"
DISPLAY_BASE = 440
STEPS = 96
WARMUP_STEPS = SYNTHETIC_BALL_WARMUP_STEPS


def _write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _choose_action(step_idx: int, raw_obs: dict[str, object]) -> StructuredSkillAction:
    ball_x = float(raw_obs.get("ball_x", 0.0))
    ball_y = float(raw_obs.get("ball_y", 0.0))
    if step_idx % 3 == 0:
        return StructuredSkillAction(skill_type=1, target_x=ball_x, target_y=ball_y, target_theta=0.0, pass_target=0)
    if step_idx % 3 == 1:
        return StructuredSkillAction(skill_type=4, target_x=min(ACTION.x_range, ball_x + 800.0), target_y=ball_y, target_theta=0.0, pass_target=0)
    return StructuredSkillAction(skill_type=2, target_x=ACTION.x_range, target_y=0.0, target_theta=0.0, pass_target=0)


def _reconstruct_ball_field(robot_x: float, robot_y: float, robot_theta: float, ball_rel_x: float, ball_rel_y: float) -> tuple[float, float]:
    c = math.cos(robot_theta)
    s = math.sin(robot_theta)
    return (
        robot_x + c * ball_rel_x - s * ball_rel_y,
        robot_y + s * ball_rel_x + c * ball_rel_y,
    )


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RUNS_DIR / f"obs_invariants_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_cfg = replace(
        ENV.backend,
        mode="simrobot_visible",
        obs_mode="corrected",
        simrobot_profile="3d_visible",
        frames_per_step=3,
        simrobot_launch=False,
        player_number=1,
    )
    env_cfg = replace(ENV, backend=backend_cfg, stage_id=0)

    runtime = VisibleParallelRuntime(
        player_numbers=(1,),
        shm_name=SHM_NAME,
        display_base=DISPLAY_BASE,
    )
    runtime.start()

    env = RealSimRobotEnv(env_cfg=env_cfg)
    obs_rows: list[dict[str, object]] = []
    violations: list[str] = []
    warning_count = 0
    failure_count = 0
    violation_counts: dict[str, int] = {}
    scored_steps = 0

    def record_violation(kind: str, *, step: int, detail: str, severity: str) -> None:
        nonlocal warning_count, failure_count
        message = f"step {step}: {kind}: {detail}"
        violations.append(message)
        violation_counts[kind] = violation_counts.get(kind, 0) + 1
        if severity == "fail":
            failure_count += 1
        else:
            warning_count += 1

    try:
        obs_dict, infos = env.reset(seed=ENV.seed)
        prev_frame = int(infos["player_1"].get("bridge_frame", 0))

        for step_idx in range(STEPS):
            base_info = infos["player_1"]
            action = _choose_action(step_idx, dict(base_info["raw_obs"]))
            obs_dict, rewards, terminations, truncations, infos = env.step({"player_1": action.as_dict()})
            obs = np.asarray(obs_dict["player_1"], dtype=np.float32).reshape(-1)
            info = infos["player_1"]
            raw = dict(info["raw_obs"])
            frame = int(info.get("bridge_frame", 0))

            if step_idx >= WARMUP_STEPS:
                scored_steps += 1
                if obs.shape != (OBS_SIZE,):
                    record_violation("obs_shape", step=step_idx + 1, detail=f"got {obs.shape}, expected {(OBS_SIZE,)}", severity="fail")
                if not np.all(np.isfinite(obs)):
                    record_violation("obs_non_finite", step=step_idx + 1, detail="encoded obs contains NaN/Inf", severity="fail")
                if frame <= prev_frame:
                    record_violation("frame_non_monotonic", step=step_idx + 1, detail=f"prev={prev_frame} current={frame}", severity="fail")

                if not (-1.2 <= float(obs[ObsIndex.robot_x]) <= 1.2):
                    record_violation("robot_x_norm_range", step=step_idx + 1, detail=f"value={float(obs[ObsIndex.robot_x]):.3f}", severity="warn")
                if not (-1.2 <= float(obs[ObsIndex.robot_y]) <= 1.2):
                    record_violation("robot_y_norm_range", step=step_idx + 1, detail=f"value={float(obs[ObsIndex.robot_y]):.3f}", severity="warn")
                if not (-1.2 <= float(obs[ObsIndex.robot_theta]) <= 1.2):
                    record_violation("robot_theta_norm_range", step=step_idx + 1, detail=f"value={float(obs[ObsIndex.robot_theta]):.3f}", severity="warn")

                for idx_name in (
                    ObsIndex.time_since_ball_seen,
                    ObsIndex.time_since_ball_disappeared,
                    ObsIndex.ball_seen_percentage,
                    ObsIndex.ball_consistent_with_game_state,
                    ObsIndex.can_score_now,
                    ObsIndex.shot_quality_no_obstacles,
                    ObsIndex.shot_opening_with_obstacles,
                    ObsIndex.pass_options_count,
                    ObsIndex.nearest_teammate_dist,
                    ObsIndex.nearest_opponent_dist,
                    ObsIndex.nearest_uncertain_obstacle_dist,
                    ObsIndex.nearest_teammate_front_dist,
                    ObsIndex.nearest_opponent_front_dist,
                    ObsIndex.nearest_uncertain_front_dist,
                ):
                    value = float(obs[idx_name])
                    if not (0.0 <= value <= 1.0):
                        record_violation("encoded_unit_interval", step=step_idx + 1, detail=f"{idx_name.name}={value:.3f}", severity="fail")

                ball_x_recon, ball_y_recon = _reconstruct_ball_field(
                    float(raw.get("robot_x", 0.0)),
                    float(raw.get("robot_y", 0.0)),
                    float(raw.get("robot_theta", 0.0)),
                    float(raw.get("ball_rel_x", 0.0)),
                    float(raw.get("ball_rel_y", 0.0)),
                )
                rel_error = math.hypot(ball_x_recon - float(raw.get("ball_x", 0.0)), ball_y_recon - float(raw.get("ball_y", 0.0)))
                if rel_error > 75.0:
                    record_violation("ball_relative_geometry", step=step_idx + 1, detail=f"error_mm={rel_error:.1f}", severity="fail")

                corrected_ball = bool(info.get("corrected_exported_ball", False))
                ball_export_fresh = bool(info.get("ball_export_fresh", False))
                ball_export_source = int(info.get("ball_export_source", 0))
                if corrected_ball == ball_export_fresh:
                    record_violation(
                        "ball_export_fresh_consistency",
                        step=step_idx + 1,
                        detail=f"corrected_ball={int(corrected_ball)} ball_export_fresh={int(ball_export_fresh)}",
                        severity="fail",
                    )
                if corrected_ball and ball_export_source not in (1, 2):
                    record_violation("ball_export_source_corrected", step=step_idx + 1, detail=f"source={ball_export_source}", severity="fail")
                if (not corrected_ball) and ball_export_source != 0:
                    record_violation("ball_export_source_estimated", step=step_idx + 1, detail=f"source={ball_export_source}", severity="fail")

                seen_pct = float(raw.get("ball_seen_percentage", 0.0))
                time_since_seen = float(raw.get("time_since_ball_seen", 0.0))
                visible_mask_expected = bool(seen_pct > BALL_VISIBLE_SEEN_PCT_MIN or time_since_seen <= BALL_VISIBLE_MAX_AGE_MS)
                natural_time_since_seen = float(info.get("natural_time_since_ball_seen", float("inf")))
                trusted_mask_expected = bool(
                    (not bool(info.get("ball_rescued_mask", False)))
                    and natural_time_since_seen <= BALL_TRUSTED_MAX_AGE_MS
                )
                if bool(info.get("ball_visible_mask", False)) != visible_mask_expected:
                    record_violation(
                        "ball_visible_mask_consistency",
                        step=step_idx + 1,
                        detail=f"reported={int(bool(info.get('ball_visible_mask', False)))} expected={int(visible_mask_expected)}",
                        severity="fail",
                    )
                if bool(info.get("ball_trusted_mask", False)) != trusted_mask_expected:
                    record_violation(
                        "ball_trusted_mask_consistency",
                        step=step_idx + 1,
                        detail=f"reported={int(bool(info.get('ball_trusted_mask', False)))} expected={int(trusted_mask_expected)}",
                        severity="fail",
                    )

                if bool(terminations["player_1"]) or bool(truncations["player_1"]):
                    record_violation("early_episode_end", step=step_idx + 1, detail="episode terminated during invariant sweep", severity="warn")

                obs_rows.append(
                    {
                        "step": step_idx + 1,
                        "bridge_frame": frame,
                        "policy_skill_name": info.get("policy_skill_name"),
                        "executed_skill_name": info.get("executed_skill_name"),
                        "projection_applied": int(bool(info.get("projection_applied", False))),
                        "reward": float(rewards["player_1"]),
                        "ball_rel_reconstruction_error_mm": float(rel_error),
                        "ball_visible_mask": int(bool(info.get("ball_visible_mask", False))),
                        "ball_trusted_mask": int(bool(info.get("ball_trusted_mask", False))),
                        "ball_rescued_mask": int(bool(info.get("ball_rescued_mask", False))),
                        "corrected_exported_ball": int(corrected_ball),
                        "ball_export_fresh": int(ball_export_fresh),
                        "ball_export_source": ball_export_source,
                        "time_since_ball_seen": time_since_seen,
                        "time_since_ball_disappeared": float(raw.get("time_since_ball_disappeared", 0.0)),
                        "ball_seen_percentage": seen_pct,
                        "robot_x_norm": float(obs[ObsIndex.robot_x]),
                        "robot_y_norm": float(obs[ObsIndex.robot_y]),
                        "robot_theta_norm": float(obs[ObsIndex.robot_theta]),
                    }
                )

            prev_frame = frame
            if not env.agents:
                break
    finally:
        env.close()
        runtime.close()

    summary = {
        "timestamp": timestamp,
        "backend_mode": backend_cfg.mode,
        "simrobot_profile": backend_cfg.simrobot_profile,
        "steps_requested": STEPS,
        "warmup_steps_skipped": WARMUP_STEPS,
        "scored_steps": scored_steps,
        "result": "FAIL" if failure_count else "WARN" if warning_count else "PASS",
        "failure_count": failure_count,
        "warning_count": warning_count,
        "violation_counts": violation_counts,
        "violations_preview": violations[:20],
        "output_dir": str(output_dir),
    }

    _write_csv(
        output_dir / "obs_invariants_steps.csv",
        obs_rows,
        fieldnames=list(obs_rows[0].keys()) if obs_rows else ["step"],
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"[obs_invariants] output_dir={output_dir}", flush=True)
    print(f"[obs_invariants] result={summary['result']} failures={failure_count} warnings={warning_count}", flush=True)


if __name__ == "__main__":
    main()
