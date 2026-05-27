"""Observe re-acquisition probe: does `observe` re-acquire the ball that `walkTo` loses?

Diagnosed deadlock (teacher-chain traces): while the robot drives the ball with the
raw `walkTo` skill, BHuman's head does NOT keep a ground ball in the camera, so the
HONEST natural channel goes stale (natural_time_since_ball_seen >> 600ms trust
window) -> the gate cannot arm dribble/shoot -> the robot is stuck in walk and never
sees the ball. see->engage->see deadlock.

`observe` maps to BHuman's ObservePoint skill (RLSkillProvider.cpp:87 ->
SkillBehaviorControl.cpp:424 -> ObservePoint.cpp): it WALKS toward a field point while
the HEAD looks AT that point (far state: LookAtPoint) or scans LookLeftAndRight +-40deg
(mid state, 300-1200mm). If we point observe at the ball's (last-known) field position,
the head should look where the ball is -> re-acquire it on the natural channel.

This probe tests that mechanism head-to-head on IDENTICAL fixed geometry:
  arm "walk"    -> step walkTo (anchor recomputed each frame) for N frames
  arm "observe" -> step observe(target = ball field pos) for N frames
For each arm we watch the HONEST natural channel (natural_time_since_ball_seen,
natural_ball_seen_percentage). Hypothesis: observe keeps natural vision much fresher.

Run:
  source /home/luis/anaconda3/etc/profile.d/conda.sh && conda activate RLNAO \
    && cd /home/luis/GitHub/RL && PYTHONPATH=/home/luis/GitHub/RL \
       python observe_reacquire_probe.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from action import SkillType, StructuredSkillAction
from config import GOAL_X, SIMROBOT_BIN
from curriculum import get_stage_scene_path
from pybh_backend import PyBHBackend, build_world_spec
from reward import build_fallback_action_targets

N_TRIALS = int(os.getenv("OBS_N_TRIALS", "3"))        # resets per arm (avg over SimRobot non-determinism)
N_FRAMES = int(os.getenv("OBS_N_FRAMES", "40"))        # frames per arm-trial
WARMUP_SKIP = int(os.getenv("OBS_WARMUP_SKIP", "14"))  # ignore synthetic-ball warmup frames in stats
BALL_GOAL_DIST_MM = float(os.getenv("OBS_BALL_GOAL_DIST_MM", "2000"))
ROBOT_BEHIND_MM = float(os.getenv("OBS_ROBOT_BEHIND_MM", "350"))
FRESH_AGE_MS = float(os.getenv("OBS_FRESH_AGE_MS", "600"))  # gate trusted_max_age_ms


def _summary(name: str, ages: list[float], seen: list[float], dists: list[float]) -> str:
    if not ages:
        return f"  [{name:8s}] no post-warmup frames"
    fresh_frac = sum(1 for a in ages if a <= FRESH_AGE_MS) / len(ages)
    return (
        f"  [{name:8s}] natural-fresh(<= {FRESH_AGE_MS:.0f}ms)={fresh_frac*100:5.1f}%  "
        f"min_age={min(ages):7.0f}ms  median_age={float(np.median(ages)):7.0f}ms  "
        f"mean_seen%={float(np.mean(seen)):5.1f}  ball_dist={float(np.mean(dists)):6.0f}mm"
    )


def _run_arm(backend: PyBHBackend, spec, spec_ball, arm: str) -> tuple[list[float], list[float], list[float]]:
    backend.reset(world_spec=spec)
    ages: list[float] = []
    seen: list[float] = []
    dists: list[float] = []
    for f in range(N_FRAMES):
        if arm == "observe":
            action = StructuredSkillAction(
                skill_type=SkillType.observe,
                target_x=float(spec_ball[0]),
                target_y=float(spec_ball[1]),
            )
        else:  # walk toward ball (anchor recomputed from latest obs)
            last = backend._last_obs
            if last is None:
                action = StructuredSkillAction(skill_type=SkillType.walk)
            else:
                t = build_fallback_action_targets("walk", last)
                action = StructuredSkillAction(
                    skill_type=SkillType.walk,
                    target_x=float(t["target_x"]),
                    target_y=float(t["target_y"]),
                    target_theta=float(t["target_theta"]),
                )
        obs = backend.step(action)
        if f >= WARMUP_SKIP:
            ages.append(float(obs.natural_time_since_ball_seen))
            seen.append(float(obs.natural_ball_seen_percentage))
            dists.append(math.hypot(
                float(obs.simrobot_robot_x) - float(spec_ball[0]),
                float(obs.simrobot_robot_y) - float(spec_ball[1]),
            ))
    return ages, seen, dists


def main() -> None:
    spec_ball = np.array([GOAL_X - BALL_GOAL_DIST_MM, 0.0], dtype=np.float32)
    spec_robot = np.array([GOAL_X - BALL_GOAL_DIST_MM - ROBOT_BEHIND_MM, 0.0], dtype=np.float32)
    spec = build_world_spec(ball_pos=spec_ball, robot_pos=spec_robot, robot_theta=0.0)
    print(
        f"[observe-probe] ball=({spec_ball[0]:.0f},{spec_ball[1]:.0f}) "
        f"robot=({spec_robot[0]:.0f},{spec_robot[1]:.0f}) facing+x  "
        f"trials={N_TRIALS} frames={N_FRAMES} (skip first {WARMUP_SKIP})\n"
        f"  hypothesis: observe keeps natural ball vision fresher than walkTo",
        flush=True,
    )

    backend = PyBHBackend(
        mode="simrobot_visible", obs_mode="corrected", frames_per_step=1, obs_timeout_ms=2000,
        simrobot_profile="3d_visible", simrobot_scene_path=get_stage_scene_path(1, "3d_visible"),
        simrobot_launch=True, simrobot_bin=SIMROBOT_BIN,
    )
    agg: dict[str, tuple[list[float], list[float], list[float]]] = {
        "walk": ([], [], []), "observe": ([], [], []),
    }
    try:
        for trial in range(N_TRIALS):
            for arm in ("walk", "observe"):
                ages, seen, dists = _run_arm(backend, spec, spec_ball, arm)
                fresh = (sum(1 for a in ages if a <= FRESH_AGE_MS) / len(ages) * 100) if ages else 0.0
                print(f"[trial {trial} {arm:8s}] natural-fresh={fresh:5.1f}%  "
                      f"min_age={min(ages) if ages else -1:7.0f}ms  "
                      f"mean_seen%={float(np.mean(seen)) if seen else 0:5.1f}", flush=True)
                agg[arm][0].extend(ages)
                agg[arm][1].extend(seen)
                agg[arm][2].extend(dists)
    finally:
        backend.close()

    print("\n[observe-probe] AGGREGATE over all trials:", flush=True)
    for arm in ("walk", "observe"):
        print(_summary(arm, *agg[arm]), flush=True)


if __name__ == "__main__":
    main()
