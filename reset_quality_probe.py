"""Reset-quality probe: is the ball placed at spec & STATIONARY after a visible reset?

Diagnoses the self-rolling-ball confound found in the teacher-chain traces (ball
drifts into the goal on its own while the robot is 1.5m away). The visible reset
verifies only the ROBOT pose; the ball is teleported position-only via
rl_reset_world_extended, which may not zero its velocity -> residual velocity from
the previous episode's kick rolls the ball.

For each of N consecutive resets (one backend process, fixed geometry) we step
`stand` and watch the GROUND-TRUTH ball: when it diverges from spec (end of the
synthetic-ball warmup that masks the obs) we learn its true placement, and the
per-step displacement tells us if it is rolling. For half the resets we then
re-place the ball via set_dynamic_world and keep watching to test whether
re-placement STOPS it (i.e. whether a Python-side re-teleport zeroes velocity).

Run:
  source /home/luis/anaconda3/etc/profile.d/conda.sh && conda activate RLNAO \
    && cd /home/luis/GitHub/RL && PYTHONPATH=/home/luis/GitHub/RL \
       python reset_quality_probe.py
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

N_RESETS = int(os.getenv("RQ_N_RESETS", "8"))
STAND_FRAMES = int(os.getenv("RQ_STAND_FRAMES", "18"))
REPLACE_AT = int(os.getenv("RQ_REPLACE_AT", "9"))   # on odd resets, re-place ball here
BALL_GOAL_DIST_MM = float(os.getenv("RQ_BALL_GOAL_DIST_MM", "2000"))
ROBOT_BEHIND_MM = float(os.getenv("RQ_ROBOT_BEHIND_MM", "350"))
# Test the residual-velocity hypothesis: KICK the ball before resetting so the next
# reset inherits a *moving* ball. If reset doesn't zero velocity, that reset's ball
# keeps rolling -> reproduces the teacher-trace self-roll.
KICK_AFTER = os.getenv("RQ_KICK_AFTER", "0") == "1"
KICK_FRAMES = int(os.getenv("RQ_KICK_FRAMES", "25"))

_STAND = StructuredSkillAction(skill_type=SkillType.stand)
_SHOOT = StructuredSkillAction(skill_type=SkillType.shoot)


def main() -> None:
    spec_ball = np.array([GOAL_X - BALL_GOAL_DIST_MM, 0.0], dtype=np.float32)
    spec_robot = np.array([GOAL_X - BALL_GOAL_DIST_MM - ROBOT_BEHIND_MM, 0.0], dtype=np.float32)
    spec = build_world_spec(ball_pos=spec_ball, robot_pos=spec_robot, robot_theta=0.0)
    print(f"[reset-quality] spec ball=({spec_ball[0]:.0f},{spec_ball[1]:.0f}) robot=({spec_robot[0]:.0f},{spec_robot[1]:.0f})\n"
          f"  N_RESETS={N_RESETS} STAND_FRAMES={STAND_FRAMES} replace_at(odd resets)={REPLACE_AT}", flush=True)

    backend = PyBHBackend(
        mode="simrobot_visible", obs_mode="corrected", frames_per_step=1, obs_timeout_ms=2000,
        simrobot_profile="3d_visible", simrobot_scene_path=get_stage_scene_path(1, "3d_visible"),
        simrobot_launch=True, simrobot_bin=SIMROBOT_BIN,
    )
    try:
        for r in range(N_RESETS):
            do_replace = (r % 2 == 1)
            backend.reset(world_spec=spec)
            prev = None
            max_disp = 0.0
            max_disp_after_replace = 0.0
            final_drift = 0.0
            diverged_at = -1
            replaced = False
            for f in range(STAND_FRAMES):
                if do_replace and f == REPLACE_AT:
                    backend.set_dynamic_world(ball_pos=spec_ball)  # re-teleport ball to spec
                    replaced = True
                obs = backend.step(_STAND)
                bx, by = float(obs.ball_x), float(obs.ball_y)
                drift = math.hypot(bx - float(spec_ball[0]), by - float(spec_ball[1]))
                disp = 0.0 if prev is None else math.hypot(bx - prev[0], by - prev[1])
                prev = (bx, by)
                if diverged_at < 0 and drift > 60.0:
                    diverged_at = f
                max_disp = max(max_disp, disp)
                if replaced and f > REPLACE_AT:
                    max_disp_after_replace = max(max_disp_after_replace, disp)
                final_drift = drift
            tag = "REPLACE" if do_replace else "plain  "
            extra = f" disp_after_replace_max={max_disp_after_replace:6.1f}mm/f" if do_replace else ""
            print(f"[reset {r} {tag}] diverged_at_frame={diverged_at:2d} "
                  f"max_disp={max_disp:6.1f}mm/f final_drift={final_drift:7.1f}mm{extra}", flush=True)
            if KICK_AFTER:
                # Kick the ball so the NEXT reset inherits a moving ball.
                kbx0 = float(backend.step(_SHOOT).ball_x)
                for _ in range(KICK_FRAMES - 1):
                    obs = backend.step(_SHOOT)
                kbx1 = float(obs.ball_x)
                print(f"          (kicked before next reset: ball_x {kbx0:.0f}->{kbx1:.0f})", flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
