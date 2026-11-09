from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from rlfootball.core.action import SKILL_ORDER, StructuredSkillAction, clamp_action
from rlfootball.core.config import ACTION, ENV, ROOT_DIR
from rlfootball.env.environment import RealSimRobotEnv
from rlfootball.core.pybh_backend import SYNTHETIC_BALL_WARMUP_STEPS
from rlfootball.core.visible_parallel_runtime import VisibleParallelRuntime

RUNS_DIR = ROOT_DIR / "runs" / "action_sweep"
SHM_NAME = "/pybh_rl_shared_state_v3"
DISPLAY_BASE = 430


def _write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _robot_to_field(robot_x: float, robot_y: float, robot_theta: float, rel_x: float, rel_y: float) -> tuple[float, float]:
    c = math.cos(robot_theta)
    s = math.sin(robot_theta)
    return (
        robot_x + c * rel_x - s * rel_y,
        robot_y + s * rel_x + c * rel_y,
    )


def _make_action(skill_name: str, *, x: float = 0.0, y: float = 0.0, theta: float = 0.0, pass_target: int = 0) -> StructuredSkillAction:
    return StructuredSkillAction(
        skill_type=SKILL_ORDER.index(skill_name),
        target_x=float(x),
        target_y=float(y),
        target_theta=float(theta),
        pass_target=int(pass_target),
    )


def _build_cases() -> list[tuple[str, StructuredSkillAction]]:
    return [
        ("stand_zero", _make_action("stand")),
        ("walk_center", _make_action("walk", x=1200.0, y=0.0, theta=0.0)),
        ("walk_clamp_hi", _make_action("walk", x=ACTION.x_range * 1.40, y=ACTION.y_range * 1.30, theta=5.0)),
        ("walk_clamp_lo", _make_action("walk", x=-ACTION.x_range * 1.35, y=-ACTION.y_range * 1.25, theta=-5.0)),
        ("shoot_forward", _make_action("shoot", x=3500.0, y=0.0, theta=0.0)),
        ("dribble_forward", _make_action("dribble", x=2500.0, y=250.0, theta=0.2)),
        ("pass_target_max", _make_action("pass", x=2000.0, y=0.0, theta=0.0, pass_target=ACTION.max_pass_target)),
        ("pass_target_oob", _make_action("pass", x=2000.0, y=0.0, theta=0.0, pass_target=ACTION.max_pass_target + 3)),
        ("block_masked", _make_action("block", x=500.0, y=0.0, theta=0.0)),
        ("mark_masked", _make_action("mark", x=500.0, y=-500.0, theta=0.0)),
        ("observe_masked", _make_action("observe", x=0.0, y=0.0, theta=math.pi / 2.0)),
        ("invalid_skill_hi", StructuredSkillAction(skill_type=999, target_x=1000.0, target_y=0.0, target_theta=0.0, pass_target=99)),
    ]


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RUNS_DIR / f"action_sweep_{timestamp}"
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
    cases = _build_cases()
    effect_steps = 3

    step_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    executed_counts: dict[str, dict[str, int]] = {}
    masked_cases: list[str] = []
    projected_cases: list[str] = []
    clamp_changed_cases: list[str] = []
    preflight_rejected_cases: list[str] = []

    try:
        for case_index, (case_name, raw_action) in enumerate(cases):
            obs, infos = env.reset(seed=ENV.seed)
            del obs
            base_info = infos["player_1"]
            base_raw = dict(base_info["raw_obs"])
            clamped_action = clamp_action(
                raw_action,
                x_range=ACTION.x_range,
                y_range=ACTION.y_range,
                max_pass_target=ACTION.max_pass_target,
            )
            projected_preview, preview_projection = env.project_action(clamped_action, obs=env._last_backend_obs)
            if preview_projection:
                projected_cases.append(case_name)
            if projected_preview.skill_name != clamped_action.skill_name:
                masked_cases.append(case_name)
            if clamped_action.as_dict() != raw_action.as_dict():
                clamp_changed_cases.append(case_name)
            if not (0 <= int(raw_action.skill_type) < len(SKILL_ORDER)):
                preflight_rejected_cases.append(case_name)
                case_rows.append(
                    {
                        "case_name": case_name,
                        "requested_skill_name": f"invalid_{int(raw_action.skill_type)}",
                        "clamped_skill_name": clamped_action.skill_name,
                        "projected_skill_name": projected_preview.skill_name,
                        "effect_steps_run": 0,
                        "projection_preview": int(bool(preview_projection)),
                        "projection_applied_rate": 0.0,
                        "executed_skill_counts": json.dumps({}, sort_keys=True),
                        "raw_skill_type": int(raw_action.skill_type),
                        "clamped_skill_type": int(clamped_action.skill_type),
                        "raw_target_x": float(raw_action.target_x),
                        "raw_target_y": float(raw_action.target_y),
                        "raw_target_theta": float(raw_action.target_theta),
                        "raw_pass_target": int(raw_action.pass_target),
                        "clamped_target_x": float(clamped_action.target_x),
                        "clamped_target_y": float(clamped_action.target_y),
                        "clamped_target_theta": float(clamped_action.target_theta),
                        "clamped_pass_target": int(clamped_action.pass_target),
                        "projected_target_x": float(projected_preview.target_x),
                        "projected_target_y": float(projected_preview.target_y),
                        "projected_target_theta": float(projected_preview.target_theta),
                        "projected_target_field_x_from_reset": 0.0,
                        "projected_target_field_y_from_reset": 0.0,
                        "robot_delta_mm": 0.0,
                        "ball_delta_mm": 0.0,
                        "reward_sum": 0.0,
                        "terminated": 0,
                        "truncated": 0,
                    }
                )
                continue

            last_info = base_info
            reward_sum = 0.0
            termination_seen = False
            truncation_seen = False
            executed_counts[case_name] = {}

            for step_idx in range(effect_steps):
                obs, rewards, terminations, truncations, infos = env.step({"player_1": raw_action.as_dict()})
                del obs
                last_info = infos["player_1"]
                reward_sum += float(rewards["player_1"])
                termination_seen = termination_seen or bool(terminations["player_1"])
                truncation_seen = truncation_seen or bool(truncations["player_1"])
                executed_skill = str(last_info.get("executed_skill_name") or "unknown")
                executed_counts[case_name][executed_skill] = executed_counts[case_name].get(executed_skill, 0) + 1
                raw_obs = dict(last_info["raw_obs"])
                step_rows.append(
                    {
                        "case_name": case_name,
                        "step_index": step_idx + 1,
                        "requested_skill_name": SKILL_ORDER[max(0, min(int(raw_action.skill_type), len(SKILL_ORDER) - 1))]
                        if 0 <= int(raw_action.skill_type) < len(SKILL_ORDER)
                        else f"invalid_{int(raw_action.skill_type)}",
                        "clamped_skill_name": clamped_action.skill_name,
                        "projected_skill_name": projected_preview.skill_name,
                        "policy_skill_name": last_info.get("policy_skill_name"),
                        "executed_skill_name": executed_skill,
                        "projection_applied": int(bool(last_info.get("projection_applied", False))),
                        "raw_skill_type": int(raw_action.skill_type),
                        "clamped_skill_type": int(clamped_action.skill_type),
                        "raw_target_x": float(raw_action.target_x),
                        "raw_target_y": float(raw_action.target_y),
                        "raw_target_theta": float(raw_action.target_theta),
                        "raw_pass_target": int(raw_action.pass_target),
                        "clamped_target_x": float(clamped_action.target_x),
                        "clamped_target_y": float(clamped_action.target_y),
                        "clamped_target_theta": float(clamped_action.target_theta),
                        "clamped_pass_target": int(clamped_action.pass_target),
                        "bridge_frame": int(last_info.get("bridge_frame", 0)),
                        "reward": float(rewards["player_1"]),
                        "robot_x": float(raw_obs.get("robot_x", 0.0)),
                        "robot_y": float(raw_obs.get("robot_y", 0.0)),
                        "robot_theta": float(raw_obs.get("robot_theta", 0.0)),
                        "ball_x": float(raw_obs.get("ball_x", 0.0)),
                        "ball_y": float(raw_obs.get("ball_y", 0.0)),
                        "synthetic_pose_observation": int(bool(last_info.get("synthetic_pose_observation", False))),
                        "synthetic_ball_observation": int(bool(last_info.get("synthetic_ball_observation", False))),
                    }
                )
                if termination_seen or truncation_seen:
                    break

            final_raw = dict(last_info["raw_obs"])
            robot_delta_mm = math.hypot(
                float(final_raw.get("robot_x", 0.0)) - float(base_raw.get("robot_x", 0.0)),
                float(final_raw.get("robot_y", 0.0)) - float(base_raw.get("robot_y", 0.0)),
            )
            ball_delta_mm = math.hypot(
                float(final_raw.get("ball_x", 0.0)) - float(base_raw.get("ball_x", 0.0)),
                float(final_raw.get("ball_y", 0.0)) - float(base_raw.get("ball_y", 0.0)),
            )
            target_field_x, target_field_y = _robot_to_field(
                float(base_raw.get("robot_x", 0.0)),
                float(base_raw.get("robot_y", 0.0)),
                float(base_raw.get("robot_theta", 0.0)),
                float(projected_preview.target_x),
                float(projected_preview.target_y),
            )
            case_rows.append(
                {
                    "case_name": case_name,
                    "requested_skill_name": SKILL_ORDER[max(0, min(int(raw_action.skill_type), len(SKILL_ORDER) - 1))]
                    if 0 <= int(raw_action.skill_type) < len(SKILL_ORDER)
                    else f"invalid_{int(raw_action.skill_type)}",
                    "clamped_skill_name": clamped_action.skill_name,
                    "projected_skill_name": projected_preview.skill_name,
                    "effect_steps_run": sum(executed_counts[case_name].values()),
                    "projection_preview": int(bool(preview_projection)),
                    "projection_applied_rate": float(
                        sum(1 for row in step_rows if row["case_name"] == case_name and row["projection_applied"]) /
                        max(1, sum(1 for row in step_rows if row["case_name"] == case_name))
                    ),
                    "executed_skill_counts": json.dumps(executed_counts[case_name], sort_keys=True),
                    "raw_skill_type": int(raw_action.skill_type),
                    "clamped_skill_type": int(clamped_action.skill_type),
                    "raw_target_x": float(raw_action.target_x),
                    "raw_target_y": float(raw_action.target_y),
                    "raw_target_theta": float(raw_action.target_theta),
                    "raw_pass_target": int(raw_action.pass_target),
                    "clamped_target_x": float(clamped_action.target_x),
                    "clamped_target_y": float(clamped_action.target_y),
                    "clamped_target_theta": float(clamped_action.target_theta),
                    "clamped_pass_target": int(clamped_action.pass_target),
                    "projected_target_x": float(projected_preview.target_x),
                    "projected_target_y": float(projected_preview.target_y),
                    "projected_target_theta": float(projected_preview.target_theta),
                    "projected_target_field_x_from_reset": float(target_field_x),
                    "projected_target_field_y_from_reset": float(target_field_y),
                    "robot_delta_mm": float(robot_delta_mm),
                    "ball_delta_mm": float(ball_delta_mm),
                    "reward_sum": float(reward_sum),
                    "terminated": int(termination_seen),
                    "truncated": int(truncation_seen),
                }
            )
    finally:
        env.close()
        runtime.close()

    summary = {
        "timestamp": timestamp,
        "backend_mode": backend_cfg.mode,
        "simrobot_profile": backend_cfg.simrobot_profile,
        "stage_id": env_cfg.stage_id,
        "cases": len(cases),
        "effect_steps": effect_steps,
        "warmup_steps_reference": SYNTHETIC_BALL_WARMUP_STEPS,
        "masked_cases": masked_cases,
        "projected_cases": projected_cases,
        "clamp_changed_cases": clamp_changed_cases,
        "preflight_rejected_cases": preflight_rejected_cases,
        "mean_projection_rate": float(
            sum(float(row["projection_applied_rate"]) for row in case_rows) / max(1, len(case_rows))
        ),
        "cases_with_zero_robot_effect": [
            row["case_name"] for row in case_rows if float(row["robot_delta_mm"]) < 5.0
        ],
        "output_dir": str(output_dir),
    }

    _write_csv(
        output_dir / "action_sweep_cases.csv",
        case_rows,
        fieldnames=list(case_rows[0].keys()) if case_rows else ["case_name"],
    )
    _write_csv(
        output_dir / "action_sweep_steps.csv",
        step_rows,
        fieldnames=list(step_rows[0].keys()) if step_rows else ["case_name", "step_index"],
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"[action_sweep] output_dir={output_dir}", flush=True)
    print(f"[action_sweep] mean_projection_rate={summary['mean_projection_rate']:.3f}", flush=True)
    print(f"[action_sweep] projected_cases={projected_cases}", flush=True)


if __name__ == "__main__":
    main()
