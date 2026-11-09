from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rlfootball.core.config import PLAYER_NUMBER, SIMROBOT_BIN
from rlfootball.core.pybh_backend import PyBHBackend

import pybh.controller as pybh


SKILL_REQUEST_NAMES = {
    0: "none",
    1: "stand",
    2: "walk",
    3: "shoot",
    4: "pass",
    5: "dribble",
    6: "block",
    7: "mark",
    8: "observe",
    9: "interceptBall",
    10: "keeperDive",
}

MOTION_REQUEST_NAMES = {
    0: "playDead",
    1: "stand",
    2: "walkAtAbsoluteSpeed",
    3: "walkAtRelativeSpeed",
    4: "walkToPose",
    5: "walkToBallAndKick",
    6: "dribble",
    7: "dive",
    8: "special",
    9: "replayWalk",
    10: "calibration",
}

DIVE_REQUEST_NAMES = {
    0: "prepare",
    1: "jumpLeft",
    2: "jumpRight",
    3: "squatArmsBackLeft",
    4: "squatArmsBackRight",
    5: "squatWideArmsBackLeft",
    6: "squatWideArmsBackRight",
}


ACTION_CASES = (
    {"gk_action": "stand", "wire_skill": "stand", "expected_skill_request": "stand"},
    {"gk_action": "walk_goalie_pose", "wire_skill": "walkTo", "expected_skill_request": "walk", "target": (-4300.0, 0.0, 0.0)},
    {"gk_action": "observe", "wire_skill": "observe", "expected_skill_request": "observe", "target": (-2600.0, 700.0, 0.0)},
    {"gk_action": "intercept_center", "wire_skill": "gk_center", "expected_skill_request": "interceptBall"},
    {"gk_action": "intercept_lateral", "wire_skill": "gk_lat", "expected_skill_request": "interceptBall"},
    {"gk_action": "intercept_low_left", "wire_skill": "gk_low_l", "expected_skill_request": "keeperDive"},
    {"gk_action": "intercept_low_right", "wire_skill": "gk_low_r", "expected_skill_request": "keeperDive"},
    {"gk_action": "intercept_jump_left", "wire_skill": "gk_jump_l", "expected_skill_request": "keeperDive"},
    {"gk_action": "intercept_jump_right", "wire_skill": "gk_jump_r", "expected_skill_request": "keeperDive"},
    {"gk_action": "clear", "wire_skill": "shoot", "expected_skill_request": "shoot"},
    {"gk_action": "pass", "wire_skill": "pass", "expected_skill_request": "pass", "pass_target": 2},
    {"gk_action": "dribble_out", "wire_skill": "dribble", "expected_skill_request": "dribble", "target": (0.0, 0.0, 0.0)},
)


def _name(table: dict[int, str], value: Any) -> str | int:
    try:
        return table.get(int(value), int(value))
    except (TypeError, ValueError):
        return -1


def _read_fresh(player_number: int, min_frame: int, timeout_reads: int) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    for _ in range(max(1, timeout_reads)):
        payload = dict(pybh.rl_get_obs(player_number, 1000))
        if payload.get("obs_ready"):
            last = payload
            if int(payload.get("frame", 0)) > min_frame:
                return payload
    if last is None:
        raise RuntimeError("No obs received from pybh.rl_get_obs")
    return last


def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "requested_skill": str(payload.get("requested_skill", "")),
        "skill_behavior_skill_request": _name(SKILL_REQUEST_NAMES, payload.get("skill_behavior_skill_request", -1)),
        "provider_motion_request": _name(MOTION_REQUEST_NAMES, payload.get("provider_motion_request", -1)),
        "skill_behavior_motion_request": _name(MOTION_REQUEST_NAMES, payload.get("skill_behavior_motion_request", -1)),
        "motion_engine_input_request": _name(MOTION_REQUEST_NAMES, payload.get("motion_engine_input_request", -1)),
        "motion_engine_effective_request": _name(MOTION_REQUEST_NAMES, payload.get("motion_engine_effective_request", -1)),
        "skill_behavior_should_intercept_ball": bool(payload.get("skill_behavior_should_intercept_ball", False)),
        "motion_engine_should_intercept_ball": bool(payload.get("motion_engine_should_intercept_ball", False)),
        "skill_behavior_dive_request": _name(DIVE_REQUEST_NAMES, payload.get("skill_behavior_dive_request", -1)),
        "motion_engine_dive_request": _name(DIVE_REQUEST_NAMES, payload.get("motion_engine_dive_request", -1)),
        "motion_engine_phase": int(payload.get("motion_engine_phase", -1)),
        "frame": int(payload.get("frame", 0)),
        "obs_ready": bool(payload.get("obs_ready", False)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe all GK action-space wire names against B-Human SkillBehavior telemetry.")
    parser.add_argument("--backend", choices=("simrobot_visible",), default="simrobot_visible")
    parser.add_argument("--profile", default="3d_visible")
    parser.add_argument("--launch-simrobot", action="store_true")
    parser.add_argument("--player-number", type=int, default=PLAYER_NUMBER)
    parser.add_argument("--reads-per-action", type=int, default=6)
    args = parser.parse_args()

    backend = PyBHBackend(
        mode=args.backend,
        simrobot_profile=args.profile,
        simrobot_launch=args.launch_simrobot,
        simrobot_bin=SIMROBOT_BIN,
        player_number=args.player_number,
        obs_timeout_ms=2500,
    )
    rows = []
    failures = []
    backend.start()
    try:
        initial = backend.read_obs()
        min_frame = int(initial.frame)
        set_action = pybh.rl_set_visible_action
        for case in ACTION_CASES:
            tx, ty, tt = case.get("target", (0.0, 0.0, 0.0))
            set_action(
                args.player_number,
                str(case["wire_skill"]),
                float(tx),
                float(ty),
                float(tt),
                int(case.get("pass_target", -1)),
            )
            payload = _read_fresh(args.player_number, min_frame, args.reads_per_action)
            min_frame = max(min_frame, int(payload.get("frame", min_frame)))
            row = {
                "gk_action": case["gk_action"],
                "wire_skill": case["wire_skill"],
                "expected_skill_request": case["expected_skill_request"],
                **_summarize(payload),
            }
            row["request_ok"] = row["requested_skill"] == case["wire_skill"]
            row["skill_request_ok"] = row["skill_behavior_skill_request"] == case["expected_skill_request"]
            if not row["request_ok"] or not row["skill_request_ok"]:
                failures.append(row)
            rows.append(row)
    finally:
        backend.close()

    result = {
        "passed": not failures,
        "count": len(rows),
        "failures": failures,
        "rows": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
