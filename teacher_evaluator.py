from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import ENV, ROOT_DIR
from curriculum import get_stage_scene_path, get_stage_spec
from environment import RealSimRobotEnv
from teacher_policy import obs_from_raw_mapping, teacher_policy_payload

STAGE_ID = int(os.getenv("RL_TEACHER_EVAL_STAGE_ID", "1"))
MAX_STEPS = int(os.getenv("RL_TEACHER_EVAL_MAX_STEPS", "900"))
SEED = int(os.getenv("RL_TEACHER_EVAL_SEED", "42000"))
BACKEND_MODE = os.getenv("RL_TEACHER_BACKEND_MODE", "simrobot_visible")
SIMROBOT_PROFILE = os.getenv("RL_TEACHER_SIMROBOT_PROFILE", "3d_visible")
SIMROBOT_LAUNCH = os.getenv("RL_TEACHER_SIMROBOT_LAUNCH", "1") == "1"
SLEEP_S = float(os.getenv("RL_TEACHER_EVAL_SLEEP_S", "0.0"))


def _append_csv_row(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _make_env() -> RealSimRobotEnv:
    stage = get_stage_spec(STAGE_ID)
    backend_cfg = replace(
        ENV.backend,
        mode=BACKEND_MODE,
        simrobot_profile=SIMROBOT_PROFILE,
        simrobot_scene_path=get_stage_scene_path(STAGE_ID, SIMROBOT_PROFILE),
        simrobot_launch=SIMROBOT_LAUNCH,
    )
    return RealSimRobotEnv(
        env_cfg=replace(
            ENV,
            stage_id=STAGE_ID,
            max_episode_steps=MAX_STEPS,
            backend=backend_cfg,
            active_skill_names=stage.active_skill_names,
            enable_handoff_override=False,
        )
    )


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT_DIR / "runs" / "teacher_evaluator" / f"eval_{timestamp}"
    steps_csv = output_dir / "teacher_live_steps.csv"
    stage = get_stage_spec(STAGE_ID)
    print(
        "[teacher-evaluator]\n"
        f"  stage={STAGE_ID} active_skills={stage.active_skill_names}\n"
        f"  backend={BACKEND_MODE} profile={SIMROBOT_PROFILE} launch={SIMROBOT_LAUNCH}\n"
        f"  telemetry={output_dir}",
        flush=True,
    )
    env = _make_env()
    try:
        obs_dict, infos = env.reset(seed=SEED)
        prev_skill_name: str | None = None
        agent_id = "player_1"
        total_reward = 0.0
        for step in range(1, MAX_STEPS + 1):
            info = infos.get(agent_id, {})
            raw_obs = info.get("raw_obs", {})
            if not isinstance(raw_obs, Mapping):
                raise RuntimeError("teacher evaluator requires raw_obs in env info")
            backend_obs = obs_from_raw_mapping(raw_obs)
            payload, absolute_action, residual_action = teacher_policy_payload(backend_obs, STAGE_ID, prev_skill_name)
            obs_dict, rewards, terminateds, truncateds, infos = env.step({agent_id: payload})
            step_info = infos.get(agent_id, {})
            reward = float(rewards.get(agent_id, 0.0))
            total_reward += reward
            row = {
                "step": step,
                "reward": reward,
                "total_reward": total_reward,
                "teacher_skill": residual_action.skill_name,
                "executed_skill_name": step_info.get("executed_skill_name"),
                "preferred_skill_name": step_info.get("preferred_skill_name"),
                "teacher_abs_target_x": float(absolute_action.target_x),
                "teacher_abs_target_y": float(absolute_action.target_y),
                "teacher_abs_target_theta": float(absolute_action.target_theta),
                "label_residual_x": float(residual_action.target_x),
                "label_residual_y": float(residual_action.target_y),
                "label_residual_theta": float(residual_action.target_theta),
                "takeover_envelope": int(bool(step_info.get("takeover_envelope", False))),
                "walk_zone": int(bool(step_info.get("walk_zone", False))),
                "dribble_zone": int(bool(step_info.get("dribble_zone", False))),
                "shoot_zone": int(bool(step_info.get("shoot_zone", False))),
                "ball_trusted_mask": int(bool(step_info.get("ball_trusted_mask", False))),
                "ball_confidence_score": float(step_info.get("ball_confidence_score", 0.0)),
                "ball_rel_x": float(raw_obs.get("ball_rel_x", 0.0)),
                "ball_rel_y": float(raw_obs.get("ball_rel_y", 0.0)),
                "time_since_ball_seen": float(raw_obs.get("time_since_ball_seen", 0.0)),
                "nearest_teammate_front_dist": float(raw_obs.get("nearest_teammate_front_dist", 0.0)),
                "nearest_opponent_front_dist": float(raw_obs.get("nearest_opponent_front_dist", 0.0)),
                "nearest_uncertain_front_dist": float(raw_obs.get("nearest_uncertain_front_dist", 0.0)),
            }
            _append_csv_row(steps_csv, row)
            print(
                "[teacher-live] "
                f"step={step:03d} reward={reward:+.3f} "
                f"teacher={residual_action.skill_name} exec={step_info.get('executed_skill_name')} "
                f"zones=w{row['walk_zone']}/d{row['dribble_zone']}/s{row['shoot_zone']} "
                f"trusted={row['ball_trusted_mask']} conf={row['ball_confidence_score']:.2f}",
                flush=True,
            )
            prev_skill_name = residual_action.skill_name
            if not env.agents or any(terminateds.values()) or any(truncateds.values()):
                break
            if SLEEP_S > 0.0:
                time.sleep(SLEEP_S)
    finally:
        env.close()


if __name__ == "__main__":
    main()
