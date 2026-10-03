"""Seed-stable verification that the gate-wired teacher completes the full
walk -> dribble -> shoot chain on the stage-1 drive curriculum.

Sequencing gate (see [[skill-gate-dynamic-masking]]): the teacher must reliably
finish the chain BEFORE we regenerate the 26-dim BC dataset from it. An earlier
ad-hoc mid-distance run reported "1/4 goals", but that may have been seed noise
(same lesson as the kick probe). Here we run a fixed, named seed set in one
backend process and read the *distribution* of chain completion.

Per seed we report: goal (chain completed), steps-to-goal, initial & closest
ball-to-goal distance, and the executed-skill histogram (did it actually walk,
then dribble, then shoot?).

Run (interactive conda env; `conda run` corrupts SimRobot's env):

  source /home/luis/anaconda3/etc/profile.d/conda.sh && conda activate RLNAO \
    && cd /home/luis/GitHub/RL && PYTHONPATH=/home/luis/GitHub/RL \
       python teacher_chain_verify.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

from rlfootball.core.config import ENV, GOAL_X, ROOT_DIR
from rlfootball.curriculum.curriculum import get_stage_scene_path, get_stage_spec
from rlfootball.env.environment import RealSimRobotEnv
from rlfootball.core.reward import goal_ball_distance_mm, goal_scored
from rlfootball.teachers.teacher_policy import obs_from_raw_mapping, teacher_policy_payload

STAGE_ID = int(os.getenv("TCV_STAGE_ID", "1"))
MAX_STEPS = int(os.getenv("TCV_MAX_STEPS", "800"))
STABLE_SEEDS = tuple(int(x) for x in os.getenv("TCV_STABLE_SEEDS", "11,23,37,51,67,83").split(","))
BACKEND_MODE = os.getenv("TCV_BACKEND_MODE", "simrobot_visible")
SIMROBOT_PROFILE = os.getenv("TCV_SIMROBOT_PROFILE", "3d_visible")
TRACE = os.getenv("TCV_TRACE", "0") == "1"


def _make_env() -> RealSimRobotEnv:
    stage = get_stage_spec(STAGE_ID)
    backend_cfg = replace(
        ENV.backend,
        mode=BACKEND_MODE,
        simrobot_profile=SIMROBOT_PROFILE,
        simrobot_scene_path=get_stage_scene_path(STAGE_ID, SIMROBOT_PROFILE),
        simrobot_launch=True,
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


def _ball_to_robot_mm(obs) -> float:
    return math.hypot(float(obs.ball_x) - float(obs.simrobot_robot_x),
                      float(obs.ball_y) - float(obs.simrobot_robot_y))


def run_seed(env: RealSimRobotEnv, seed: int, out_dir: Path) -> dict[str, object]:
    obs_dict, infos = env.reset(seed=seed)
    agent_id = "player_1"
    prev_skill = None
    skills: Counter[str] = Counter()
    init_dist = float("nan")
    min_dist = float("inf")
    goal = False
    steps = 0
    trace: list[dict[str, object]] = []

    for step in range(1, MAX_STEPS + 1):
        info = infos.get(agent_id, {})
        raw_obs = info.get("raw_obs", {})
        if not isinstance(raw_obs, Mapping):
            raise RuntimeError("teacher chain verify requires raw_obs in env info")
        backend_obs = obs_from_raw_mapping(raw_obs)
        if step == 1:
            init_dist = goal_ball_distance_mm(backend_obs)
        dist = goal_ball_distance_mm(backend_obs)
        min_dist = min(min_dist, dist)

        # gate decision the teacher acts on this step (computed by env on this obs)
        gate = env.last_gate_decision
        payload, _absolute, residual = teacher_policy_payload(
            backend_obs, STAGE_ID, prev_skill, gate_decision=gate
        )
        if TRACE:
            trace.append({
                "step": step,
                "teacher_skill": residual.skill_name,
                "ball_to_robot_mm": round(_ball_to_robot_mm(backend_obs), 1),
                "ball_to_goal_mm": round(dist, 1),
                "ball_x": round(float(backend_obs.ball_x), 1),
                "ball_y": round(float(backend_obs.ball_y), 1),
                "robot_x": round(float(backend_obs.simrobot_robot_x), 1),
                "robot_y": round(float(backend_obs.simrobot_robot_y), 1),
                "robot_theta": round(float(backend_obs.simrobot_robot_theta), 3),
                "t_seen_ms": round(float(getattr(backend_obs, "time_since_ball_seen", 0.0)), 0),
                "nat_t_seen_ms": round(float(getattr(backend_obs, "natural_time_since_ball_seen", 0.0)), 0),
                "shoot_armed": int(getattr(gate, "shoot_armed", False)) if gate else -1,
                "dribble_armed": int(getattr(gate, "dribble_armed", False)) if gate else -1,
                "shoot_arm_progress": round(float(getattr(gate, "shoot_arm_progress", 0.0)), 2) if gate else -1,
            })
        obs_dict, rewards, terminateds, truncateds, infos = env.step({agent_id: payload})
        steps = step
        step_info = infos.get(agent_id, {})
        executed = step_info.get("executed_skill_name") or residual.skill_name
        skills[str(executed)] += 1
        prev_skill = residual.skill_name
        if TRACE and trace:
            trace[-1]["executed_skill"] = str(executed)

        step_raw = step_info.get("raw_obs", {})
        if isinstance(step_raw, Mapping) and "ball_x" in step_raw:
            if goal_scored(float(step_raw["ball_x"]), float(step_raw["ball_y"])):
                goal = True
        if not env.agents or any(terminateds.values()) or any(truncateds.values()):
            # recompute min/goal on the terminal obs
            if isinstance(step_raw, Mapping) and "ball_x" in step_raw:
                term_obs = obs_from_raw_mapping(step_raw)
                min_dist = min(min_dist, goal_ball_distance_mm(term_obs))
            break

    if TRACE and trace:
        trace_csv = out_dir / f"trace_seed_{seed}.csv"
        with trace_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            w.writeheader()
            w.writerows(trace)

    return {
        "seed": seed,
        "goal": int(goal),
        "steps": steps,
        "init_goal_dist_mm": round(init_dist, 1),
        "min_goal_dist_mm": round(min_dist, 1),
        "walk": skills.get("walk", 0),
        "dribble": skills.get("dribble", 0),
        "shoot": skills.get("shoot", 0),
        "stand": skills.get("stand", 0),
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT_DIR / "runs" / "teacher_chain_verify" / f"verify_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "seeds.csv"

    print(
        "[teacher-chain-verify]\n"
        f"  stage={STAGE_ID} max_steps={MAX_STEPS} goal_x={GOAL_X}\n"
        f"  stable_seeds={STABLE_SEEDS}\n"
        f"  backend={BACKEND_MODE} profile={SIMROBOT_PROFILE}\n"
        f"  out={out_dir}",
        flush=True,
    )

    env = _make_env()
    rows: list[dict[str, object]] = []
    try:
        for seed in STABLE_SEEDS:
            row = run_seed(env, seed, out_dir)
            rows.append(row)
            print(
                f"[seed {seed:4d}] goal={row['goal']} steps={row['steps']:3d} "
                f"init={row['init_goal_dist_mm']:7.0f}mm closest={row['min_goal_dist_mm']:7.0f}mm "
                f"skills w/d/s/st={row['walk']}/{row['dribble']}/{row['shoot']}/{row['stand']}",
                flush=True,
            )
    finally:
        env.close()

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    goals = sum(int(r["goal"]) for r in rows)
    print("\n=== TEACHER CHAIN COMPLETION (stable seeds) ===", flush=True)
    print(f"  goal_rate = {goals}/{n} = {100.0*goals/n:.0f}%", flush=True)
    print(f"  wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
