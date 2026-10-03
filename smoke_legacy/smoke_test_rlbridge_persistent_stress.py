"""
LEGACY REDIRECT

Archivo legado. La version viva esta en:
`smoke_legacy.smoke_test_rlbridge_persistent_stress`.
Usar la superficie oficial: `smoke_diagnostic_extended.py` o `smoke_suite_option1.py`.



smoke_test_rlbridge_persistent_stress.py -- stress test for persistent transport.

Validates that pipe_process survives:
  - 100 consecutive frames
  - 5 reset cycles
  - repeated decision traffic
  - no crash or fallback to one-shot transport
"""

from __future__ import annotations

import time
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
    print("[TEST] RLBridge persistent transport STRESS test")

    cfg = BridgeConfig(
        preferred_transport="pipe_process",
        allow_oneshot_fallback=False,
    )
    _assert(Path(cfg.bridge_executable).exists(), f"bridge executable exists at {cfg.bridge_executable}")

    bridge = BHumanBridge(cfg)
    own_team, opp_team = _build_teams()
    ball_pos = np.array([200.0, 0.0], dtype=np.float32)
    ball_vel = np.zeros(2, dtype=np.float32)

    frames_per_episode = 100
    num_episodes = 5

    try:
        bridge.start()
        _assert(bridge.is_running, "bridge started")
        _assert(
            bridge._transport_mode == "pipe_process",
            f"transport = pipe_process (got {bridge._transport_mode})",
        )

        all_latencies_ms: list[float] = []

        for episode in range(num_episodes):
            print(f"\n[TEST] Episode {episode + 1}/{num_episodes}")
            if episode > 0:
                bridge.reset()
                _assert(bridge.is_running, "bridge alive after reset")

            episode_latencies_ms: list[float] = []
            for frame_idx in range(frames_per_episode):
                t0 = time.perf_counter()
                result = bridge.step(
                    ball_pos=ball_pos,
                    ball_vel=ball_vel,
                    own_team=own_team,
                    opp_team=opp_team,
                    game_state="playing",
                    own_score=0,
                    opp_score=0,
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0
                episode_latencies_ms.append(dt_ms)

                _assert(
                    result.get("type") == "decision",
                    f"ep{episode + 1} frame{frame_idx + 1}: got decision",
                )
                _assert(
                    result.get("debug_transport_mode") == "pipe_process",
                    f"ep{episode + 1} frame{frame_idx + 1}: still pipe_process",
                )

                expected_frame_id = frame_idx + 1
                _assert(
                    result.get("frame_id") == expected_frame_id,
                    f"ep{episode + 1}: frame_id={expected_frame_id} (got {result.get('frame_id')})",
                )

            mean_ms = sum(episode_latencies_ms) / len(episode_latencies_ms)
            p99_index = min(len(episode_latencies_ms) - 1, int(0.99 * len(episode_latencies_ms)))
            p99_ms = sorted(episode_latencies_ms)[p99_index]
            all_latencies_ms.extend(episode_latencies_ms)
            print(
                f"  INFO: ep{episode + 1}: mean={mean_ms:.2f}ms, "
                f"p99={p99_ms:.2f}ms, frames={len(episode_latencies_ms)}"
            )

        total_mean_ms = sum(all_latencies_ms) / len(all_latencies_ms)
        total_p99_index = min(len(all_latencies_ms) - 1, int(0.99 * len(all_latencies_ms)))
        total_p99_ms = sorted(all_latencies_ms)[total_p99_index]
        _assert(total_mean_ms < 50.0, f"overall mean latency < 50ms (got {total_mean_ms:.2f}ms)")
        _assert(bridge.is_running, "bridge still alive at end of stress test")

        print("\nRESULT: PASS")
        print(f"  {len(all_latencies_ms)} frames across {num_episodes} episodes.")
        print(f"  Mean latency: {total_mean_ms:.2f}ms")
        print(f"  P99 latency: {total_p99_ms:.2f}ms")
        print("  Pipe process transport survived stress test.")
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nRESULT: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        raise SystemExit(1)
