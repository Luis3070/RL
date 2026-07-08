from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from action import SkillType, StructuredSkillAction
from config import PLAYER_NUMBER, SIMROBOT_BIN
from goalkeeper_action import GKSkillType, GKStructuredAction
from goalkeeper_teacher import GOALIE_LINE_X_MM, gk_action_to_current_bridge
from pybh_backend import PyBHBackend

import pybh.controller as pybh


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

DIVE_REQUEST_NAMES = {
    0: "prepare",
    1: "jumpLeft",
    2: "jumpRight",
    3: "squatArmsBackLeft",
    4: "squatArmsBackRight",
    5: "squatWideArmsBackLeft",
    6: "squatWideArmsBackRight",
}

GK_NATIVE_BRIDGE_NAMES = {
    "intercept_center": "gk_center",
    "intercept_lateral": "gk_lat",
    "intercept_low_left": "gk_low_l",
    "intercept_low_right": "gk_low_r",
    "intercept_jump_left": "gk_jump_l",
    "intercept_jump_right": "gk_jump_r",
}

INTERCEPT_CASES = (
    GKStructuredAction(GKSkillType.intercept_center, target_y=0.0),
    GKStructuredAction(GKSkillType.intercept_lateral, target_y=0.06),
    GKStructuredAction(GKSkillType.intercept_low_left, target_y=0.12),
    GKStructuredAction(GKSkillType.intercept_low_right, target_y=-0.12),
    GKStructuredAction(GKSkillType.intercept_jump_left, target_y=0.43),
    GKStructuredAction(GKSkillType.intercept_jump_right, target_y=-0.43),
)


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    skill_req = int(payload.get("skill_behavior_skill_request", -1))
    provider_motion = int(payload.get("provider_motion_request", -1))
    skill_motion = int(payload.get("skill_behavior_motion_request", -1))
    engine_input = int(payload.get("motion_engine_input_request", -1))
    engine_effective = int(payload.get("motion_engine_effective_request", -1))
    skill_dive = int(payload.get("skill_behavior_dive_request", -1))
    engine_dive = int(payload.get("motion_engine_dive_request", -1))
    return {
        "requested_skill": str(payload.get("requested_skill", "")),
        "skill_behavior_skill_request": SKILL_REQUEST_NAMES.get(skill_req, skill_req),
        "provider_motion_request": MOTION_REQUEST_NAMES.get(provider_motion, provider_motion),
        "skill_behavior_motion_request": MOTION_REQUEST_NAMES.get(skill_motion, skill_motion),
        "motion_engine_input_request": MOTION_REQUEST_NAMES.get(engine_input, engine_input),
        "motion_engine_effective_request": MOTION_REQUEST_NAMES.get(engine_effective, engine_effective),
        "skill_behavior_should_intercept_ball": bool(payload.get("skill_behavior_should_intercept_ball", False)),
        "motion_engine_should_intercept_ball": bool(payload.get("motion_engine_should_intercept_ball", False)),
        "skill_behavior_dive_request": DIVE_REQUEST_NAMES.get(skill_dive, skill_dive),
        "motion_engine_dive_request": DIVE_REQUEST_NAMES.get(engine_dive, engine_dive),
        "motion_engine_phase": int(payload.get("motion_engine_phase", -1)),
        "fall_state": int(payload.get("motion_engine_fall_state", -1)),
        "frame": int(payload.get("frame", 0)),
    }


def _run_structured(backend: PyBHBackend, action: StructuredSkillAction, frames: int) -> dict[str, Any]:
    obs = backend.step(action)
    for _ in range(max(0, frames - 1)):
        obs = backend.step(action)
    payload = pybh.rl_get_obs(backend.player_number, 1)
    row = _payload_summary(payload)
    row["obs_ready"] = bool(obs.obs_ready)
    return row


def _run_raw_pybh(backend: PyBHBackend, skill_name: str, frames: int) -> dict[str, Any]:
    set_action = pybh.rl_set_visible_action if backend.mode == "simrobot_visible" else pybh.rl_set_action
    set_action(backend.player_number, skill_name, 0.0, 0.0, 0.0, -1)
    min_frame = backend._last_obs.frame if backend._last_obs is not None else -1
    obs = None
    for _ in range(max(1, frames)):
        if backend.mode == "simrobot":
            pybh.simrobot_step(1)
        else:
            pass
        payload = pybh.rl_get_obs(backend.player_number, 1000)
        if bool(payload.get("obs_ready", False)) and int(payload.get("frame", 0)) > min_frame:
            obs = payload
            break
    if obs is None:
        payload = pybh.rl_get_obs(backend.player_number, 1000)
    else:
        payload = obs
    row = _payload_summary(payload)
    row["obs_ready"] = bool(payload.get("obs_ready", False))
    if backend._last_obs is not None:
        backend._last_obs.frame = max(backend._last_obs.frame, int(payload.get("frame", backend._last_obs.frame)))
    return row


def _truth_label(row: dict[str, Any], *, expected_native: bool) -> str:
    motion = row["skill_behavior_motion_request"]
    engine = row["motion_engine_input_request"]
    skill = row["skill_behavior_skill_request"]
    if motion == "dive" and engine == "dive":
        return "native_dive_executed"
    if motion == "walkToPose" and engine in {"walkToPose", "walkAtRelativeSpeed", "walkAtAbsoluteSpeed"}:
        return "walking_intercept_or_fallback"
    if skill == "stand" and motion == "stand":
        return "stand_fallback_not_intercept"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify whether GK intercept requests are truthfully executed by B-Human.")
    parser.add_argument("--backend", choices=("simrobot", "simrobot_visible"), default="simrobot")
    parser.add_argument("--profile", default="2d")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--player-number", type=int, default=PLAYER_NUMBER)
    parser.add_argument("--launch-simrobot", action="store_true")
    parser.add_argument("--no-reset", action="store_true", help="Skip world reset and only verify raw native bridge telemetry on the live scene.")
    parser.add_argument("--require-native", action="store_true", help="Return non-zero unless at least one intercept reaches native dive execution.")
    args = parser.parse_args()

    backend = PyBHBackend(
        mode=args.backend,
        simrobot_profile=args.profile,
        frames_per_step=max(1, args.frames),
        obs_timeout_ms=2500,
        player_number=args.player_number,
        simrobot_launch=args.launch_simrobot,
        simrobot_bin=SIMROBOT_BIN,
    )
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    backend.start()
    try:
        if args.no_reset:
            backend.read_obs()
        else:
            backend.reset(
                ball_pos=np.asarray([-3800.0, 420.0], dtype=np.float32),
                robot_pos=np.asarray([GOALIE_LINE_X_MM, 0.0], dtype=np.float32),
                robot_theta=0.0,
            )
            for gk_action in INTERCEPT_CASES:
                bridge_action = gk_action_to_current_bridge(gk_action, backend._last_obs)
                row = _run_structured(backend, bridge_action, args.frames)
                row.update(
                    {
                        "path": "python_gk_bridge",
                        "gk_requested": gk_action.skill_name,
                        "bridge_sent": bridge_action.skill_name,
                    }
                )
                row["truth"] = _truth_label(row, expected_native=False)
                rows.append(row)

        for gk_action in INTERCEPT_CASES:
            raw_name = GK_NATIVE_BRIDGE_NAMES[gk_action.skill_name]
            row = _run_raw_pybh(backend, raw_name, args.frames)
            row.update(
                {
                    "path": "raw_pybh_name",
                    "gk_requested": raw_name,
                    "bridge_sent": raw_name,
                }
            )
            row["truth"] = _truth_label(row, expected_native=True)
            raw_rows.append(row)
    finally:
        backend.close()

    all_rows = rows + raw_rows
    native_request_count = sum(1 for row in all_rows if row.get("skill_behavior_skill_request") == "interceptBall")
    native_dive_count = sum(1 for row in all_rows if row["truth"] == "native_dive_executed")
    stand_fallback_count = sum(1 for row in all_rows if row["truth"] == "stand_fallback_not_intercept")
    walking_count = sum(1 for row in all_rows if row["truth"] == "walking_intercept_or_fallback")
    result = {
        "passed": (native_request_count > 0) if args.require_native else True,
        "native_intercept_exposed": native_request_count > 0,
        "native_request_count": native_request_count,
        "native_dive_count": native_dive_count,
        "stand_fallback_count": stand_fallback_count,
        "walking_count": walking_count,
        "rows": all_rows,
        "verdict": (
            "native GK intercept reaches MotionRequest::dive"
            if native_dive_count > 0
            else (
                "native GK intercept SkillRequest is exposed; no dive executed in this scene"
                if native_request_count > 0
                else "current RL bridge does not expose native GK intercept; intercept labels fall back to stand/walk"
            )
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
