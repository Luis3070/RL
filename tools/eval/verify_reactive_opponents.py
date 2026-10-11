"""Verify training-scene opponents play REAL B-Human football when not teleported.

The whole teacher corpus is compromised by non-reactive opponents: `_static_obstacles`
never moves and `_dynamic_obstacles` is a closed-form sin/cos wobble the env teleports
onto the field every `dynamic_update_interval_steps`. But the training scene
(RLMAStage7_3D) spawns opponents as real Nao bodies on team 70 under `gc playing`, so
they should run ordinary B-Human behaviour whenever the env stops overriding them
(RL_MA_STAGE8_DYNAMIC_WORLD_UPDATES=0).

This launches the training scene, touches nothing, and watches opponent positions via
the GlobalOpponentsModel export. Reactive opponents converge on the ball; scripted ones
hold station or orbit. Run BEFORE regenerating the corpus -- if this fails, the
regeneration is pointless.

Usage:  python tools/eval/verify_reactive_opponents.py
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("PYBH_RL_OVERRIDE_TEAM", "24")
os.environ.setdefault("PYBH_RL_ACTIVE_PLAYERS", "2,3,4")

from rlfootball.core.config import BHUMAN_DIR, SIMROBOT_BIN  # noqa: E402
from rlfootball.datasets import teacher_multiagent_dataset as tmd  # noqa: E402  (chdir + pybh load)
from rlfootball.core.pybh_backend import BackendObservation  # noqa: E402
import pybh.controller as pybh  # noqa: E402

SCENE = BHUMAN_DIR / "Config/Scenes/RLMAStage7_3D.ros2"
OBSERVE_S = 70.0


def main() -> int:
    proc = subprocess.Popen(
        [str(SIMROBOT_BIN), str(SCENE)], cwd=BHUMAN_DIR,
        env=os.environ.copy(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 40.0
        while time.monotonic() < deadline:
            o = BackendObservation.from_payload(dict(pybh.rl_get_obs(2, 1200)))
            if o.obs_ready:
                break
            time.sleep(0.5)
        else:
            print("FAIL: obs never ready")
            return 1

        samples: list[tuple[float, float, float]] = []  # (t, nearest_opp_to_ball, total_opp_travel)
        prev_pts: list[tuple[float, float]] = []
        travel = 0.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < OBSERVE_S:
            o = BackendObservation.from_payload(dict(pybh.rl_get_obs(2, 1200)))
            pts = list(o.opponent_positions)
            if pts:
                d = min(math.hypot(px - float(o.ball_x), py - float(o.ball_y)) for px, py in pts)
                if prev_pts and len(prev_pts) == len(pts):
                    travel += sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pts, prev_pts))
                prev_pts = pts
                samples.append((time.monotonic() - t0, d, travel))
            time.sleep(0.25)

        if not samples:
            print("FAIL: opponent positions never populated")
            return 1

        first = [s[1] for s in samples[: max(1, len(samples) // 4)]]
        last = [s[1] for s in samples[-max(1, len(samples) // 4):]]
        mean_first = sum(first) / len(first)
        mean_last = sum(last) / len(last)
        min_ever = min(s[1] for s in samples)
        print(f"samples={len(samples)}  opponent cumulative travel={travel:.0f}mm")
        print(f"nearest-opponent-to-ball: first quarter {mean_first:.0f}mm -> last quarter {mean_last:.0f}mm "
              f"(min ever {min_ever:.0f}mm)")
        # Reactive B-Human closes on the ball and racks up real travel; a station-holding
        # or orbiting script does neither.
        if travel > 3000.0 and min_ever < 600.0:
            print("PASS: opponents are REACTIVE (they travel and reach the ball)")
            return 0
        print("FAIL/INCONCLUSIVE: opponents look scripted or idle -- corpus regeneration "
              "against them would reproduce the existing non-reactive-opponent problem")
        return 2
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
