"""Prove the GameController score export actually INCREMENTS on a real goal.

A 0-0 smoke match only proves the field exists and reads zero -- indistinguishable
from a dead wire. This teleports the ball into the opponent goal via rl_reset_world
and watches own_score, so a passing run is positive evidence the counter is live.

Usage (SimRobot must NOT already be running):
  python tools/eval/verify_gc_score_wire.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

os.environ.setdefault("PYBH_RL_OVERRIDE_TEAM", "24")
os.environ.setdefault("PYBH_RL_ACTIVE_PLAYERS", "2,3,4")

from rlfootball.core.config import BHUMAN_DIR, SIMROBOT_BIN  # noqa: E402
from rlfootball.datasets import teacher_multiagent_dataset as tmd  # noqa: E402  (chdirs into BHUMAN_DIR, loads pybh)
from rlfootball.core.pybh_backend import BackendObservation  # noqa: E402
import pybh.controller as pybh  # noqa: E402

SCENE = BHUMAN_DIR / "Config/Scenes/RLvsBH4v4_3D.ros2"
PLAYERS = (2, 3, 4)
GOAL_X = 4500.0


def obs_for(player: int) -> BackendObservation:
    return BackendObservation.from_payload(dict(pybh.rl_get_obs(player, 1200)))


def scores() -> tuple[int, int]:
    best = (0, 0)
    for p in PLAYERS:
        o = obs_for(p)
        best = (max(best[0], int(o.own_score)), max(best[1], int(o.opponent_score)))
    return best


def main() -> int:
    proc = subprocess.Popen(
        [str(SIMROBOT_BIN), str(SCENE)], cwd=BHUMAN_DIR,
        env=os.environ.copy(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 40.0
        while time.monotonic() < deadline:
            if all(obs_for(p).obs_ready for p in PLAYERS):
                break
            time.sleep(0.5)
        else:
            print("FAIL: observations never became ready")
            return 1

        before = scores()
        print(f"score before teleport: own={before[0]} opponent={before[1]}", flush=True)

        # Put the ball inside the opponent goal mouth, striker just behind it.
        pybh.rl_reset_world(2, GOAL_X + 250.0, 0.0, GOAL_X - 600.0, 0.0, 0.0)
        print("teleported ball to (4750, 0) -- inside opponent goal", flush=True)

        observed_max = before
        t_end = time.monotonic() + 25.0
        while time.monotonic() < t_end:
            cur = scores()
            observed_max = (max(observed_max[0], cur[0]), max(observed_max[1], cur[1]))
            if cur[0] > before[0]:
                print(f"PASS: own_score incremented {before[0]} -> {cur[0]} "
                      f"(GameController score wire is LIVE)", flush=True)
                return 0
            time.sleep(0.5)

        print(f"INCONCLUSIVE: own_score never moved (max seen own={observed_max[0]} "
              f"opponent={observed_max[1]}). Either the autoreferee did not award a goal "
              f"for a teleported ball, or the wire is dead.", flush=True)
        return 2
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
