"""Seed-stable kick-lethality probe.

Answers a single question rigorously: *can the B-Human "shoot" skill reliably
drive the ball into the goal from 2.5-3.0 m, or does it cap out ~1 m?*

Why this exists: an earlier ad-hoc probe reported wildly erratic kicks
(ball speeds 0 -> 1973, one freak goal at 2.5 m, zeros elsewhere). That is the
fingerprint of a *small-sample / seed-noise* measurement, not a clean read on
kick capability. Here we control the geometry exactly (fixed ball/robot poses
per distance) so the only thing varying within a distance is SimRobot's own
run-to-run jitter, and we sweep a *fixed, named seed set x N reps* so the output
is a reproducible distribution rather than a single freak.

We measure ground-truth ball position (`ball_x/ball_y`, NOT the perceived
`estimated_*`) so the carry/goal numbers are not themselves seed-contaminated.

Decision rule the output is built to support:
  * If *clean-strike* carry (carry | connected) reaches ~2.5-3.0 m -> the kick
    CAN do it; the earlier zeros were jitter -> keep the 2.5/2.9 m shoot band
    and invest in approach/alignment reliability.
  * If even clean strikes cap ~1.0-1.1 m -> genuine power ceiling -> a B-Human
    kick-type change is required (no RL-side power param exists).

Run (must use the interactive conda env; `conda run` corrupts SimRobot's env):

  source /home/luis/anaconda3/etc/profile.d/conda.sh && conda activate RLNAO \
    && cd /home/luis/GitHub/RL && python kick_lethality_probe.py
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from action import SkillType, StructuredSkillAction
from config import GOAL_X, GOAL_Y_HALF, ROOT_DIR, SIMROBOT_BIN
from curriculum import get_stage_scene_path
from pybh_backend import PyBHBackend, build_world_spec

# --- experiment configuration (env-overridable) -----------------------------
DISTANCES_MM = tuple(
    float(x) for x in os.getenv("KICK_DISTANCES_MM", "1000,1500,2000,2500,3000").split(",")
)
# A fixed, NAMED seed set. Stable => the experiment is reproducible run-to-run;
# each seed is one independent rep at every distance, so we get a distribution.
STABLE_SEEDS = tuple(
    int(x) for x in os.getenv("KICK_STABLE_SEEDS", "11,23,37,51,67,83,101,127").split(",")
)
ROBOT_BEHIND_MM = float(os.getenv("KICK_ROBOT_BEHIND_MM", "220"))
SETTLE_FRAMES = int(os.getenv("KICK_SETTLE_FRAMES", "8"))
TRACK_FRAMES = int(os.getenv("KICK_TRACK_FRAMES", "140"))
CONNECT_CARRY_MM = float(os.getenv("KICK_CONNECT_CARRY_MM", "250"))
STAGE_ID = int(os.getenv("KICK_STAGE_ID", "1"))
SIMROBOT_PROFILE = os.getenv("KICK_SIMROBOT_PROFILE", "3d_visible")
BACKEND_MODE = os.getenv("KICK_BACKEND_MODE", "simrobot_visible")
HZ = float(os.getenv("KICK_HZ", "10.0"))

_SHOOT = StructuredSkillAction(skill_type=SkillType.shoot)
_STAND = StructuredSkillAction(skill_type=SkillType.stand)


@dataclass
class RepResult:
    distance_mm: float
    seed: int
    init_ball_x: float
    init_ball_y: float
    carry_mm: float          # max forward (+x) travel of the ball
    final_ball_x: float
    final_ball_y: float
    peak_speed_mm_s: float    # peak ball velocity magnitude reported by sim
    crossed_goal_line: bool   # ball_x reached GOAL_X
    goal: bool                # crossed within the goal mouth (|y| <= GOAL_Y_HALF)
    goal_cross_y: float       # ball_y at the moment it crossed GOAL_X (nan if never)
    connected: bool           # carry > CONNECT_CARRY_MM (foot cleanly struck ball)
    kick_frame: int           # frame at which ball first started moving (-1 if never)


def _ball_speed(obs, dt: float) -> float:
    vx, vy = float(obs.ball_vel_x), float(obs.ball_vel_y)
    return math.hypot(vx, vy)


def run_rep(backend: PyBHBackend, distance_mm: float, seed: int) -> RepResult:
    ball = np.array([GOAL_X - distance_mm, 0.0], dtype=np.float32)
    robot = np.array([GOAL_X - distance_mm - ROBOT_BEHIND_MM, 0.0], dtype=np.float32)
    # theta = atan2(ball - robot) = 0 -> robot faces +x straight at the goal.
    spec = build_world_spec(ball_pos=ball, robot_pos=robot, robot_theta=0.0)

    obs = backend.reset(world_spec=spec)
    init_bx, init_by = float(obs.ball_x), float(obs.ball_y)

    for _ in range(SETTLE_FRAMES):
        obs = backend.step(_STAND)

    init_bx, init_by = float(obs.ball_x), float(obs.ball_y)
    max_x = init_bx
    peak_speed = 0.0
    kick_frame = -1
    crossed = False
    goal_cross_y = float("nan")
    prev_x = init_bx
    still_after_move = 0
    period = 1.0 / max(HZ, 0.1)
    dt = period

    for f in range(TRACK_FRAMES):
        obs = backend.step(_SHOOT)
        bx, by = float(obs.ball_x), float(obs.ball_y)
        spd = _ball_speed(obs, dt)
        peak_speed = max(peak_speed, spd)
        if kick_frame < 0 and (bx - init_bx) > 50.0:
            kick_frame = f
        if bx > max_x:
            max_x = bx
        # goal-line crossing within the mouth
        if not crossed and prev_x < GOAL_X <= bx:
            crossed = True
            goal_cross_y = by
        # stop early once the ball is past the goal line, or has clearly stopped
        # after having been struck
        if crossed:
            break
        if kick_frame >= 0:
            if abs(bx - prev_x) < 3.0:
                still_after_move += 1
            else:
                still_after_move = 0
            if still_after_move >= 6:
                break
        prev_x = bx
        if BACKEND_MODE == "simrobot_visible":
            time.sleep(period)

    final_bx, final_by = float(obs.ball_x), float(obs.ball_y)
    carry = max_x - init_bx
    goal = crossed and abs(goal_cross_y) <= GOAL_Y_HALF
    connected = carry > CONNECT_CARRY_MM
    return RepResult(
        distance_mm=distance_mm,
        seed=seed,
        init_ball_x=init_bx,
        init_ball_y=init_by,
        carry_mm=carry,
        final_ball_x=final_bx,
        final_ball_y=final_by,
        peak_speed_mm_s=peak_speed,
        crossed_goal_line=crossed,
        goal=goal,
        goal_cross_y=goal_cross_y,
        connected=connected,
        kick_frame=kick_frame,
    )


def summarize(results: list[RepResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for d in DISTANCES_MM:
        reps = [r for r in results if r.distance_mm == d]
        if not reps:
            continue
        n = len(reps)
        connected = [r for r in reps if r.connected]
        carries = [r.carry_mm for r in reps]
        clean_carries = [r.carry_mm for r in connected]
        rows.append(
            {
                "distance_mm": d,
                "n": n,
                "goal_rate": sum(r.goal for r in reps) / n,
                "connect_rate": len(connected) / n,
                "carry_mean": statistics.mean(carries),
                "carry_std": statistics.pstdev(carries) if n > 1 else 0.0,
                "carry_min": min(carries),
                "carry_max": max(carries),
                "clean_carry_mean": statistics.mean(clean_carries) if clean_carries else 0.0,
                "clean_carry_max": max(clean_carries) if clean_carries else 0.0,
                "peak_speed_max": max(r.peak_speed_mm_s for r in reps),
            }
        )
    return rows


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT_DIR / "runs" / "kick_lethality_probe" / f"probe_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reps_csv = out_dir / "reps.csv"
    summary_csv = out_dir / "summary.csv"

    print(
        "[kick-lethality]\n"
        f"  distances_mm={DISTANCES_MM}\n"
        f"  stable_seeds={STABLE_SEEDS} (reps/distance={len(STABLE_SEEDS)})\n"
        f"  robot_behind_mm={ROBOT_BEHIND_MM} goal_x={GOAL_X} goal_half={GOAL_Y_HALF}\n"
        f"  backend={BACKEND_MODE} profile={SIMROBOT_PROFILE}\n"
        f"  out={out_dir}",
        flush=True,
    )

    backend = PyBHBackend(
        mode=BACKEND_MODE,
        obs_mode="corrected",
        frames_per_step=1,
        obs_timeout_ms=2000,
        simrobot_profile=SIMROBOT_PROFILE,
        simrobot_scene_path=get_stage_scene_path(STAGE_ID, SIMROBOT_PROFILE),
        simrobot_launch=True,
        simrobot_bin=SIMROBOT_BIN,
    )

    results: list[RepResult] = []
    try:
        for d in DISTANCES_MM:
            for seed in STABLE_SEEDS:
                # Seed every RNG we control so the run is reproducible. (SimRobot
                # physics jitter is what we are characterizing across reps.)
                np.random.seed(seed)
                r = run_rep(backend, d, seed)
                results.append(r)
                print(
                    f"[rep] d={d:5.0f}mm seed={seed:4d} "
                    f"carry={r.carry_mm:7.0f}mm peak={r.peak_speed_mm_s:6.0f}mm/s "
                    f"connected={int(r.connected)} crossed={int(r.crossed_goal_line)} "
                    f"goal={int(r.goal)} (cross_y={r.goal_cross_y:7.1f}) kick_f={r.kick_frame}",
                    flush=True,
                )
    finally:
        backend.close()

    # per-rep CSV
    with reps_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(vars(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(vars(r))

    rows = summarize(results)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== KICK LETHALITY SUMMARY (per ball-to-goal distance) ===", flush=True)
    print(
        f"{'dist':>6} {'n':>3} {'goal%':>6} {'conn%':>6} "
        f"{'carry mean±sd [min,max]':>30} {'clean_carry(mean/max)':>22} {'pk_spd':>7}",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['distance_mm']:6.0f} {row['n']:3d} "
            f"{row['goal_rate']*100:5.0f}% {row['connect_rate']*100:5.0f}% "
            f"{row['carry_mean']:7.0f}±{row['carry_std']:5.0f} "
            f"[{row['carry_min']:5.0f},{row['carry_max']:5.0f}] "
            f"{row['clean_carry_mean']:7.0f}/{row['clean_carry_max']:7.0f} "
            f"{row['peak_speed_max']:7.0f}",
            flush=True,
        )
    print(f"\n[kick-lethality] wrote {reps_csv}\n[kick-lethality] wrote {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
