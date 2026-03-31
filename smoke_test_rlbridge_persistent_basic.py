"""
smoke_test_rlbridge_persistent_basic.py -- direct persistent-transport smoke.

Goal:
  Validate that RLBridge can stay alive across multiple frames and a reset
  without falling back to one-shot transport.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bridge.bridge_protocol import BHumanBridge, BridgeConfig


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def _build_teams():
    own_team = [
        {"number": 1, "x": 0.0, "y": 0.0, "theta": 0.0, "upright": True},
        {"number": 2, "x": -800.0, "y": 500.0, "theta": 0.0, "upright": True},
        {"number": 3, "x": -1200.0, "y": -500.0, "theta": 0.0, "upright": True},
        {"number": 4, "x": -1800.0, "y": 0.0, "theta": 0.0, "upright": True},
    ]
    opp_team = [
        {"number": 1, "x": 1200.0, "y": 0.0, "theta": np.pi, "upright": True},
        {"number": 2, "x": 1500.0, "y": 700.0, "theta": np.pi, "upright": True},
        {"number": 3, "x": 1500.0, "y": -700.0, "theta": np.pi, "upright": True},
        {"number": 4, "x": 2200.0, "y": 0.0, "theta": np.pi, "upright": True},
    ]
    return own_team, opp_team


def main() -> None:
    print("[TEST] RLBridge persistent transport basic smoke")

    cfg = BridgeConfig(
        preferred_transport="pipe_process",
        allow_oneshot_fallback=False,
    )
    _assert(Path(cfg.bridge_executable).exists(), f"bridge executable exists at {cfg.bridge_executable}")
    _assert(cfg.bridge_mode == "persistent_2d_cognition",
            f"bridge_mode = persistent_2d_cognition (got {cfg.bridge_mode})")

    bridge = BHumanBridge(cfg)
    own_team, opp_team = _build_teams()
    ball_pos = np.array([200.0, 0.0], dtype=np.float32)
    ball_vel = np.zeros(2, dtype=np.float32)

    try:
        print("\n[TEST] Startup")
        bridge.start()
        _assert(bridge.is_running, "bridge started")
        _assert(bridge._transport_mode == "pipe_process",
                f"transport_mode = pipe_process (got {bridge._transport_mode})")

        print("\n[TEST] Three consecutive frames")
        frame_ids = []
        for idx in range(3):
            result = bridge.step(
                ball_pos=ball_pos,
                ball_vel=ball_vel,
                own_team=own_team,
                opp_team=opp_team,
                game_state="playing",
                own_score=0,
                opp_score=0,
            )
            _assert(result.get("type") == "decision",
                    f"decision response on frame {idx + 1} (got {result.get('type')})")
            _assert(result.get("debug_transport_mode") == "pipe_process",
                    f"debug_transport_mode = pipe_process (got {result.get('debug_transport_mode')})")
            _assert(result.get("debug_bridge_mode") == "persistent_2d_cognition",
                    f"debug_bridge_mode = persistent_2d_cognition "
                    f"(got {result.get('debug_bridge_mode')})")
            frame_id = result.get("frame_id")
            frame_ids.append(frame_id)
            print(f"  INFO: frame={idx + 1}, bridge_frame_id={frame_id}, skill={result.get('skill_request')}, motion={result.get('motion_request')}")

        _assert(frame_ids == [1, 2, 3], f"frame ids increment correctly (got {frame_ids})")

        print("\n[TEST] Reset + one more frame")
        bridge.reset()
        result_after_reset = bridge.step(
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            own_team=own_team,
            opp_team=opp_team,
            game_state="playing",
            own_score=0,
            opp_score=0,
        )
        _assert(result_after_reset.get("frame_id") == 1,
                f"frame id resets to 1 after reset (got {result_after_reset.get('frame_id')})")
        _assert(result_after_reset.get("debug_transport_mode") == "pipe_process",
                "transport stayed persistent after reset")
        _assert(result_after_reset.get("debug_bridge_mode") == "persistent_2d_cognition",
                "bridge_mode stayed persistent_2d_cognition after reset")

        print("\nRESULT: PASS")
        print("  RLBridge persistent pipe transport is operational across frames and reset.")
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nRESULT: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        raise SystemExit(1)
