"""
Capture real BHuman pass telemetry and export a reusable snapshot.

What it does:
  1. Probes the real sidecar backend with pass-biased 4v4 snapshots.
  2. Writes every real-backend attempt to a CSV for auditability.
  3. When a true playBall_pass is observed, exports that exact snapshot as JSON.

This script exists to replace hardcoded pass fixtures with a snapshot that
comes from real BHuman bridge telemetry under the current baseline.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from bhuman_cfg_parser import load_contract
from bhuman_sidecar import BHumanDecisionSidecar, GroundTruthFrame
from config import FIELD, resolve_cfg_root


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
PASS_TELEMETRY_CSV = ARTIFACTS_DIR / "real_pass_telemetry.csv"
PASS_SNAPSHOT_JSON = ARTIFACTS_DIR / "real_pass_snapshot.json"
PASS_REPLAY_CSV = ARTIFACTS_DIR / "real_pass_snapshot_replay.csv"


def _to_list_xy(value: np.ndarray | Iterable[float]) -> list[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(arr[0]), float(arr[1])]


def _serialize_frame(frame: GroundTruthFrame) -> dict:
    return {
        "ball_pos": _to_list_xy(frame.ball_pos),
        "ball_vel": _to_list_xy(frame.ball_vel),
        "ball_seen_age_ms": float(frame.ball_seen_age_ms),
        "own_poses": {aid: _to_list_xy(pos) for aid, pos in frame.own_poses.items()},
        "own_thetas": {aid: float(theta) for aid, theta in frame.own_thetas.items()},
        "own_upright": {aid: bool(flag) for aid, flag in frame.own_upright.items()},
        "own_roles": dict(frame.own_roles),
        "opp_poses": [_to_list_xy(pos) for pos in frame.opp_poses],
        "opp_upright": [bool(flag) for flag in frame.opp_upright],
        "is_playing": bool(frame.is_playing),
        "own_score": int(frame.own_score),
        "opp_score": int(frame.opp_score),
    }


def _frame_from_json(data: dict) -> GroundTruthFrame:
    return GroundTruthFrame(
        ball_pos=np.array(data["ball_pos"], dtype=np.float32),
        ball_vel=np.array(data.get("ball_vel", [0.0, 0.0]), dtype=np.float32),
        ball_seen_age_ms=float(data.get("ball_seen_age_ms", 0.0)),
        own_poses={
            aid: np.array(pos, dtype=np.float32)
            for aid, pos in data["own_poses"].items()
        },
        own_thetas={aid: float(theta) for aid, theta in data["own_thetas"].items()},
        own_upright={aid: bool(flag) for aid, flag in data["own_upright"].items()},
        own_roles=dict(data["own_roles"]),
        opp_poses=[np.array(pos, dtype=np.float32) for pos in data["opp_poses"]],
        opp_upright=[bool(flag) for flag in data["opp_upright"]],
        is_playing=bool(data.get("is_playing", True)),
        own_score=int(data.get("own_score", 0)),
        opp_score=int(data.get("opp_score", 0)),
    )


def _load_snapshot_frame() -> GroundTruthFrame:
    payload = json.loads(PASS_SNAPSHOT_JSON.read_text(encoding="utf-8"))
    return _frame_from_json(payload["frame"])


def _write_snapshot_replay_csv(max_steps: int = 20) -> None:
    from config import CURRICULUM
    from environment import NaoSoccerEnv
from smoke_legacy.smoke_test_real_env_semantic_variants import _install_frame
from smoke_legacy.smoke_test_real_takeover_playball import _zero_actions

    frame = _load_snapshot_frame()
    contract = load_contract(cfg_root=resolve_cfg_root())
    sidecar = BHumanDecisionSidecar(contract, execution_mode="bhuman_sidecar")
    rows: list[dict[str, object]] = []
    try:
        for step in range(max_steps):
            direct = sidecar.decide_from_frame(frame)
            rows.append({
                "source": "direct_sidecar",
                "step": step,
                "sidecar_backend": direct.backend,
                "sidecar_macro_type": direct.macro_type,
                "sidecar_skill_request": direct.skill_request,
                "sidecar_motion_request": direct.motion_request,
                "sidecar_active_agent": direct.active_agent or "",
                "sidecar_pass_target_player_number": direct.pass_target_player_number,
                "sidecar_last_kick_timestamp": direct.last_kick_timestamp,
                "takeover_active": "",
                "takeover_kick": "",
                "takeover_kick_proxy": "",
                "macro_end_reason": "",
                "macro_success": "",
                "episode_takeover_success": "",
            })
    finally:
        sidecar.shutdown()

    env = NaoSoccerEnv(stage_cfg=CURRICULUM[12], use_fov=False, seed=21)
    try:
        env.reset(seed=21)
        _install_frame(env, frame)
        for step in range(max_steps):
            _, _, _, _, infos = env.step(_zero_actions(env))
            info = next(iter(infos.values()))
            rows.append({
                "source": "env_step",
                "step": step,
                "sidecar_backend": info.get("sidecar_backend", ""),
                "sidecar_macro_type": info.get("sidecar_macro_type", ""),
                "sidecar_skill_request": info.get("sidecar_skill_request", ""),
                "sidecar_motion_request": info.get("sidecar_motion_request", ""),
                "sidecar_active_agent": info.get("active_agent_id", ""),
                "sidecar_pass_target_player_number": info.get("sidecar_pass_target_player_number", ""),
                "sidecar_last_kick_timestamp": info.get("sidecar_last_kick_timestamp", ""),
                "takeover_active": int(bool(info.get("takeover_active", False))),
                "takeover_kick": int(bool(info.get("takeover_kick", False))),
                "takeover_kick_proxy": int(bool(info.get("takeover_kick_proxy", False))),
                "macro_end_reason": info.get("macro_end_reason", ""),
                "macro_success": info.get("macro_success", ""),
                "episode_takeover_success": info.get("episode_takeover_success", ""),
            })
    finally:
        env.close()

    with PASS_REPLAY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_pass_candidate(rng: np.random.Generator) -> GroundTruthFrame:
    # Keep the ball in midfield / upper own half so the receiver lane can beat
    # the direct shot under real BHuman semantics.
    ball_x = float(rng.uniform(-1400.0, -250.0))
    ball_y = float(rng.uniform(-450.0, 450.0))
    ball_pos = np.array([ball_x, ball_y], dtype=np.float32)

    carrier_idx = int(rng.integers(0, 4))
    receiver_idx = int((carrier_idx + rng.integers(1, 4)) % 4)
    lateral_sign = 1.0 if rng.random() < 0.5 else -1.0

    carrier = ball_pos + np.array(
        [float(rng.uniform(-130.0, -60.0)), float(rng.uniform(-85.0, 85.0))],
        dtype=np.float32,
    )
    carrier_theta = math.atan2(
        lateral_sign * float(rng.uniform(420.0, 760.0)),
        float(rng.uniform(2100.0, 2800.0)),
    )

    # Pass receiver ahead at a favorable angle window (~10-22 deg).
    pass_dx = float(rng.uniform(2200.0, 2900.0))
    pass_dy = lateral_sign * float(rng.uniform(420.0, 780.0))
    receiver = ball_pos + np.array([pass_dx, pass_dy], dtype=np.float32)

    support_a = ball_pos + np.array(
        [float(rng.uniform(-900.0, -350.0)), lateral_sign * float(rng.uniform(800.0, 1300.0))],
        dtype=np.float32,
    )
    support_b = ball_pos + np.array(
        [float(rng.uniform(-1500.0, -700.0)), -lateral_sign * float(rng.uniform(650.0, 1200.0))],
        dtype=np.float32,
    )

    agent_ids = ["agent_0", "agent_1", "agent_2", "agent_3"]
    own_poses: dict[str, np.ndarray] = {}
    own_thetas: dict[str, float] = {}
    fallback_positions = [support_a, support_b]
    fallback_ptr = 0
    for idx, aid in enumerate(agent_ids):
        if idx == carrier_idx:
            own_poses[aid] = carrier
            own_thetas[aid] = float(carrier_theta)
        elif idx == receiver_idx:
            own_poses[aid] = receiver
            own_thetas[aid] = 0.0
        else:
            own_poses[aid] = fallback_positions[fallback_ptr]
            own_thetas[aid] = 0.0
            fallback_ptr += 1
    own_upright = {aid: True for aid in own_poses}
    own_roles = {
        "agent_0": "forwardR",
        "agent_1": "forwardL",
        "agent_2": "defenderR",
        "agent_3": "defenderL",
    }

    # Place opponents to make the direct shot lane less attractive while leaving
    # the receiver lane comparatively open.
    blocking_wall = [
        ball_pos + np.array([float(rng.uniform(250.0, 520.0)), float(rng.uniform(-180.0, 180.0))], dtype=np.float32),
        ball_pos + np.array([float(rng.uniform(620.0, 980.0)), float(rng.uniform(-260.0, 260.0))], dtype=np.float32),
    ]
    receiver_lane_blockers = [
        ball_pos + np.array([float(rng.uniform(900.0, 1300.0)), -pass_dy * 0.35], dtype=np.float32),
        np.array([float(rng.uniform(1400.0, 2200.0)), float(rng.uniform(-1200.0, 1200.0))], dtype=np.float32),
    ]
    opp_poses = blocking_wall + receiver_lane_blockers
    opp_upright = [True] * 4

    # Clamp to field bounds.
    for pos in list(own_poses.values()) + opp_poses:
        pos[0] = float(np.clip(pos[0], FIELD.x_own_ground + 120.0, FIELD.x_opponent_ground - 120.0))
        pos[1] = float(np.clip(pos[1], FIELD.y_right_sideline + 120.0, FIELD.y_left_sideline - 120.0))

    return GroundTruthFrame(
        ball_pos=ball_pos,
        ball_vel=np.zeros(2, dtype=np.float32),
        ball_seen_age_ms=0.0,
        own_poses=own_poses,
        own_thetas=own_thetas,
        own_upright=own_upright,
        own_roles=own_roles,
        opp_poses=opp_poses,
        opp_upright=opp_upright,
        is_playing=True,
        own_score=0,
        opp_score=0,
    )


def _iter_pass_candidates(seed: int, attempts: int) -> Iterable[GroundTruthFrame]:
    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        yield _build_pass_candidate(rng)


def main() -> None:
    cfg_root = resolve_cfg_root()
    if cfg_root is None:
        raise SystemExit("Real CFG root could not be resolved")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    contract = load_contract(cfg_root=cfg_root)
    sidecar = BHumanDecisionSidecar(contract, execution_mode="bhuman_sidecar")

    rows: list[dict[str, object]] = []
    winning_frame: GroundTruthFrame | None = None
    winning_output = None

    try:
        for idx, frame in enumerate(_iter_pass_candidates(seed=20260408, attempts=400), start=1):
            output = sidecar.decide_from_frame(frame)
            row = {
                "attempt": idx,
                "backend": output.backend,
                "bridge_mode": output.bridge_mode,
                "macro_type": output.macro_type,
                "skill_request": output.skill_request,
                "active_agent": output.active_agent,
                "pass_target_player_number": output.pass_target_player_number,
                "kick_type": output.kick_type,
                "kick_length": output.kick_length,
                "last_kick_timestamp": output.last_kick_timestamp,
                "motion_request": output.motion_request,
                "behavior_status": output.behavior_status,
                "current_tactic": output.current_tactic,
                "bridge_frame_id": output.bridge_frame_id,
                "ball_x": float(frame.ball_pos[0]),
                "ball_y": float(frame.ball_pos[1]),
                "ball_vx": float(frame.ball_vel[0]),
                "ball_vy": float(frame.ball_vel[1]),
                "agent_0_x": float(frame.own_poses["agent_0"][0]),
                "agent_0_y": float(frame.own_poses["agent_0"][1]),
                "agent_0_theta": float(frame.own_thetas["agent_0"]),
                "agent_0_role": frame.own_roles["agent_0"],
                "agent_1_x": float(frame.own_poses["agent_1"][0]),
                "agent_1_y": float(frame.own_poses["agent_1"][1]),
                "agent_1_theta": float(frame.own_thetas["agent_1"]),
                "agent_1_role": frame.own_roles["agent_1"],
                "agent_2_x": float(frame.own_poses["agent_2"][0]),
                "agent_2_y": float(frame.own_poses["agent_2"][1]),
                "agent_2_theta": float(frame.own_thetas["agent_2"]),
                "agent_2_role": frame.own_roles["agent_2"],
                "agent_3_x": float(frame.own_poses["agent_3"][0]),
                "agent_3_y": float(frame.own_poses["agent_3"][1]),
                "agent_3_theta": float(frame.own_thetas["agent_3"]),
                "agent_3_role": frame.own_roles["agent_3"],
                "opp0_x": float(frame.opp_poses[0][0]),
                "opp0_y": float(frame.opp_poses[0][1]),
                "opp1_x": float(frame.opp_poses[1][0]),
                "opp1_y": float(frame.opp_poses[1][1]),
                "opp2_x": float(frame.opp_poses[2][0]),
                "opp2_y": float(frame.opp_poses[2][1]),
                "opp3_x": float(frame.opp_poses[3][0]),
                "opp3_y": float(frame.opp_poses[3][1]),
            }
            rows.append(row)

            if output.backend == "real" and output.macro_type == "playBall_pass" and output.skill_request == "pass":
                winning_frame = frame
                winning_output = output
                break
    finally:
        sidecar.shutdown()

    with PASS_TELEMETRY_CSV.open("w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if winning_frame is None or winning_output is None:
        print(f"[capture_real_pass_telemetry] No real playBall_pass found after {len(rows)} attempts")
        print(f"[capture_real_pass_telemetry] Telemetry CSV written to: {PASS_TELEMETRY_CSV}")
        raise SystemExit(1)

    payload = {
        "provenance": {
            "source": "real_bhuman_bridge_search",
            "description": "Snapshot discovered by querying the real persistent_2d_cognition backend until it emitted playBall_pass.",
            "attempt": len(rows),
            "backend": winning_output.backend,
            "bridge_mode": winning_output.bridge_mode,
            "macro_type": winning_output.macro_type,
            "skill_request": winning_output.skill_request,
            "active_agent": winning_output.active_agent,
            "pass_target_player_number": winning_output.pass_target_player_number,
            "current_tactic": winning_output.current_tactic,
        },
        "frame": _serialize_frame(winning_frame),
    }
    PASS_SNAPSHOT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_snapshot_replay_csv()

    print(f"[capture_real_pass_telemetry] PASS snapshot found on attempt {len(rows)}")
    print(f"[capture_real_pass_telemetry] Telemetry CSV: {PASS_TELEMETRY_CSV}")
    print(f"[capture_real_pass_telemetry] Snapshot JSON: {PASS_SNAPSHOT_JSON}")
    print(f"[capture_real_pass_telemetry] Snapshot replay CSV: {PASS_REPLAY_CSV}")


if __name__ == "__main__":
    main()
