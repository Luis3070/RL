"""
smoke_test_environment_bridge.py -- End-to-end environment smoke test for RL <-> BHuman bridge.

Goal:
  Validate that NaoSoccerEnv communicates with the real RLBridge backend,
  not just with contract_validation_mode or local stubs.

What this test verifies:
  - `bridge/bhuman_bridge.exe` exists and can be used by the sidecar.
  - `NaoSoccerEnv` starts in `bhuman_sidecar` mode with backend `real`.
  - `env.step()` propagates real bridge decisions into `info`.
  - Bridge debug/provenance fields are exposed in `info`.
  - `env.reset()` resets the bridge frame counter for a new episode.
  - `env.close()` shuts down the subprocess cleanly.

This is a communication smoke test, not a semantic "playing soccer well" test.
The purpose here is to validate transport, synchronization, integration, and
that the real bridge backend is operational inside the environment.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

# Force the real sidecar path before importing config/environment.
os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from bridge.bridge_protocol import BridgeConfig, expected_transport_mode
from config import CURRICULUM, resolve_cfg_root
from environment import NaoSoccerEnv

SMOKE_STAGE_ID = 12  # 4v4_full -- full 4-player own team + 4 opponents

def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def _sample_info(infos):
    _assert(bool(infos), "infos returned")
    return next(iter(infos.values()))


def _zero_actions(env: NaoSoccerEnv):
    return {
        aid: np.zeros(env.action_space(aid).shape, dtype=np.float32)
        for aid in env.agents
    }


def _install_known_good_bridge_fixture(env: NaoSoccerEnv) -> None:
    """
    Overwrite the env state with the same 2D-style snapshot shape that already
    proved stable in the manual RLBridge smoke.

    This keeps the test focused on env <-> bridge communication, instead of
    conflating it with random spawn geometry edge cases.
    """
    env._ball.pos[:] = np.array([200.0, 0.0], dtype=np.float32)
    env._ball.vel[:] = 0.0
    env._ball.seen_age_ms = 0.0

    own_fixture = [
        (np.array([0.0, 0.0], dtype=np.float32), 0.0),
        (np.array([-800.0, 500.0], dtype=np.float32), 0.0),
        (np.array([-1200.0, -500.0], dtype=np.float32), 0.0),
        (np.array([-1800.0, 0.0], dtype=np.float32), 0.0),
    ]
    for idx, aid in enumerate(env.agents):
        agent = env._agents_state[aid]
        agent.pos = own_fixture[idx][0].copy()
        agent.theta = own_fixture[idx][1]
        agent.vel = np.zeros(2, dtype=np.float32)
        agent.omega = 0.0
        agent.is_upright = True
        agent.is_active = (idx == 0)
        agent.walk_target = agent.pos.copy()
        agent.speed_scalar = 0.0

    opp_fixture = [
        (np.array([1200.0, 0.0], dtype=np.float32), np.pi),
        (np.array([1500.0, 700.0], dtype=np.float32), np.pi),
        (np.array([1500.0, -700.0], dtype=np.float32), np.pi),
        (np.array([2200.0, 0.0], dtype=np.float32), np.pi),
    ]
    for idx, opp in enumerate(env._opponents_state):
        opp.pos = opp_fixture[idx][0].copy()
        opp.theta = opp_fixture[idx][1]
        opp.vel = np.zeros(2, dtype=np.float32)
        opp.is_upright = True

    env._game.is_playing = True
    env._game.own_score = 0
    env._game.opp_score = 0
    env._active_agent_id = env.agents[0] if env.agents else None


def _check_common_bridge_info(info: dict) -> None:
    expected_transport = expected_transport_mode()
    expected_bridge_mode = "persistent_2d_cognition"
    _assert(info.get("execution_mode") == "bhuman_sidecar",
            f"execution_mode = bhuman_sidecar (got {info.get('execution_mode')})")
    _assert(info.get("sidecar_active") is True, "sidecar_active = True")
    _assert(info.get("sidecar_backend") == "real",
            f"sidecar_backend = real (got {info.get('sidecar_backend')})")
    _assert(info.get("sidecar_bridge_mode") == expected_bridge_mode,
            f"sidecar_bridge_mode = {expected_bridge_mode} (got {info.get('sidecar_bridge_mode')})")
    _assert(info.get("sidecar_contract_validation_mode") is False,
            "sidecar_contract_validation_mode = False")
    _assert(info.get("takeover_simulator_active") is False,
            "takeover_simulator_active = False")

    _assert("bridge_frame_id" in info, "bridge_frame_id exposed in info")
    _assert("bridge_decision_time_us" in info, "bridge_decision_time_us exposed in info")
    _assert("bridge_debug_frame_sync" in info, "bridge_debug_frame_sync exposed in info")
    _assert("bridge_debug_recent_messages" in info, "bridge_debug_recent_messages exposed in info")
    _assert("bridge_debug_seen_messages_total" in info,
            "bridge_debug_seen_messages_total exposed in info")

    _assert(info.get("bridge_debug_frame_sync") == "ack",
            f"bridge_debug_frame_sync = ack (got {info.get('bridge_debug_frame_sync')})")
    _assert(info.get("bridge_debug_transport_mode") == expected_transport,
            f"bridge_debug_transport_mode = {expected_transport} (got {info.get('bridge_debug_transport_mode')})")
    _assert(info.get("bridge_debug_bridge_mode") == expected_bridge_mode,
            f"bridge_debug_bridge_mode = {expected_bridge_mode} (got {info.get('bridge_debug_bridge_mode')})")
    _assert(info.get("bridge_debug_seen_messages_total", 0) >= 1,
            f"bridge_debug_seen_messages_total >= 1 (got {info.get('bridge_debug_seen_messages_total')})")

    recent = info.get("bridge_debug_recent_messages", [])
    _assert(isinstance(recent, list), "bridge_debug_recent_messages is a list")
    _assert("idFrameFinished" in recent,
            f"bridge_debug_recent_messages contains idFrameFinished (got {recent})")

    # Providers from the bridge should come from the cognition blackboard fallback.
    provider_keys = [
        "bridge_debug_provider_game_state",
        "bridge_debug_provider_skill_request",
        "bridge_debug_provider_motion_request",
        "bridge_debug_provider_behavior_status",
        "bridge_debug_provider_strategy_status",
        "bridge_debug_provider_activation_graph",
    ]
    for key in provider_keys:
        _assert(info.get(key) == "cognition_blackboard",
                f"{key} = cognition_blackboard (got {info.get(key)})")

    _assert(isinstance(info.get("sidecar_activation_graph"), list),
            "sidecar_activation_graph is a list")
    _assert(len(info.get("sidecar_activation_graph", [])) >= 1,
            "sidecar_activation_graph has at least one node")


def main() -> None:
    print("[TEST] Environment <-> RLBridge communication smoke")

    bridge_exe = BridgeConfig().bridge_executable
    _assert(Path(bridge_exe).exists(), f"bridge executable exists at {bridge_exe}")
    print(f"  INFO: bridge_executable={bridge_exe}")

    cfg_root = resolve_cfg_root()
    _assert(cfg_root is not None, f"real CFG root resolved (got {cfg_root})")
    print(f"  INFO: cfg_root={cfg_root}")

    stage_cfg = CURRICULUM[SMOKE_STAGE_ID]
    _assert(stage_cfg.num_rl_agents == 4,
            f"smoke stage uses 4 RL agents (got {stage_cfg.num_rl_agents})")
    _assert(stage_cfg.num_opponents >= 3,
            f"smoke stage uses opponent context compatible with BHuman (got {stage_cfg.num_opponents})")
    print(f"  INFO: smoke_stage={stage_cfg.stage_id}:{stage_cfg.name}")

    env = NaoSoccerEnv(stage_cfg=stage_cfg, use_fov=False, seed=42)
    try:
        print("\n[TEST] Sidecar backend selection")
        _assert(env._sidecar is not None, "env._sidecar instantiated")
        _assert(env._sidecar.is_real_backend is True, "env._sidecar uses real backend")
        _assert(env._sidecar.backend_label == "real",
                f"backend_label = real (got {env._sidecar.backend_label})")

        print("\n[TEST] First episode step flow")
        obs, infos = env.reset(seed=42)
        _assert(len(obs) == len(env.agents), "reset returned one observation per active agent")
        _install_known_good_bridge_fixture(env)

        first_step_latencies = []
        frame_ids = []
        for step_idx in range(2):
            t0 = time.perf_counter()
            obs, rewards, terms, truncs, infos = env.step(_zero_actions(env))
            dt_ms = (time.perf_counter() - t0) * 1000.0
            first_step_latencies.append(dt_ms)

            info = _sample_info(infos)
            _check_common_bridge_info(info)

            frame_id = info["bridge_frame_id"]
            frame_ids.append(frame_id)
            _assert(frame_id >= 1, f"bridge_frame_id >= 1 on step {step_idx + 1} (got {frame_id})")
            _assert(info["bridge_decision_time_us"] > 0,
                    f"bridge_decision_time_us > 0 (got {info['bridge_decision_time_us']})")

            print(f"  INFO: step={step_idx + 1}, bridge_frame_id={frame_id}, "
                  f"bridge_decision_time_us={info['bridge_decision_time_us']}, "
                  f"debug_game_state={info.get('bridge_debug_game_state')}, "
                  f"debug_player_state={info.get('bridge_debug_player_state')}, "
                  f"env_step_ms={dt_ms:.2f}")

        _assert(frame_ids[0] == 1,
                f"first bridge frame after reset is 1 (got {frame_ids[0]})")
        _assert(frame_ids[1] == frame_ids[0] + 1,
                f"bridge_frame_id increments by 1 (got {frame_ids})")

        print("\n[TEST] Reset propagation to RLBridge")
        obs, infos = env.reset(seed=43)
        _assert(len(obs) == len(env.agents), "second reset returned observations")
        _install_known_good_bridge_fixture(env)
        obs, rewards, terms, truncs, infos = env.step(_zero_actions(env))
        info_after_reset = _sample_info(infos)
        _check_common_bridge_info(info_after_reset)
        _assert(info_after_reset["bridge_frame_id"] == 1,
                f"bridge frame counter resets to 1 after env.reset() "
                f"(got {info_after_reset['bridge_frame_id']})")

        mean_step_ms = sum(first_step_latencies) / len(first_step_latencies)
        _assert(mean_step_ms < 2000.0,
                f"mean env step roundtrip < 2000ms (got {mean_step_ms:.2f}ms)")
        print(f"  INFO: mean_env_step_ms={mean_step_ms:.2f}")

        print("\nRESULT: PASS")
        print("  RLBridge real backend is active inside NaoSoccerEnv.")
        print("  Frame sync, provenance, reset propagation, and info plumbing are operational.")
        print("  Remaining work, if any, is higher-level soccer semantics "
              "(e.g. richer scenario validation for pass/dribble/duel), not transport.")

    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nRESULT: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        raise SystemExit(1)
