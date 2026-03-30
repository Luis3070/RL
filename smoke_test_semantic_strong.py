"""
smoke_test_semantic_strong.py -- Strong semantic validation for BHuman-in-loop.

Purpose:
  Validate higher-level soccer semantics after transport/integration is already
  known-good. This test is stricter than the environment bridge smoke:

  1. Reuses deterministic sidecar semantic fixtures:
     - shoot
     - pass
     - dribble
     - duel
     - activation graph structure
     - kick metadata discipline

  2. Verifies semantic provenance inside the real environment loop:
     - backend is real
     - no contract_validation fallback
     - no takeover simulator
     - activation graph stays structured
     - macro_type / skill_request remain contract-consistent

This is the recommended gate before long BHuman-in-loop training runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from bridge.bridge_protocol import BridgeConfig, expected_transport_mode
from bhuman_sidecar import VALID_SKILL_REQUESTS
from config import CURRICULUM, resolve_cfg_root
from environment import NaoSoccerEnv
from smoke_test_sidecar import (
    test_activation_graph_structured,
    test_active_agent_output,
    test_is_playing_gating,
    test_kick_metadata,
    test_scenario_dribble,
    test_scenario_duel,
    test_scenario_pass,
    test_scenario_shoot,
)

VALID_MACRO_TYPES = {
    "none",
    "playBall_shoot",
    "playBall_pass",
    "playBall_dribble",
    "playBall_duel",
}


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def _zero_actions(env: NaoSoccerEnv):
    return {
        aid: np.zeros(env.action_space(aid).shape, dtype=np.float32)
        for aid in env.agents
    }


def _install_semantic_fixture(env: NaoSoccerEnv) -> None:
    """
    Stable fixture that keeps the environment in a meaningful playing snapshot.

    This is not trying to force a specific final BHuman decision. It tries to
    guarantee that the data coming back through the bridge is semantically
    structured and operational.
    """
    env._ball.pos[:] = np.array([350.0, 50.0], dtype=np.float32)
    env._ball.vel[:] = np.zeros(2, dtype=np.float32)
    env._ball.seen_age_ms = 0.0

    own_fixture = [
        (np.array([0.0, 0.0], dtype=np.float32), 0.0),
        (np.array([-900.0, 600.0], dtype=np.float32), 0.0),
        (np.array([-1000.0, -550.0], dtype=np.float32), 0.0),
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
        (np.array([1200.0, 100.0], dtype=np.float32), np.pi),
        (np.array([1500.0, 850.0], dtype=np.float32), np.pi),
        (np.array([1500.0, -850.0], dtype=np.float32), np.pi),
        (np.array([2250.0, -50.0], dtype=np.float32), np.pi),
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


def test_real_environment_semantics() -> None:
    print("\n[TEST] Real environment semantic provenance")

    bridge_exe = BridgeConfig().bridge_executable
    _assert(Path(bridge_exe).exists(), f"bridge executable exists at {bridge_exe}")

    cfg_root = resolve_cfg_root()
    _assert(cfg_root is not None, f"real CFG root resolved (got {cfg_root})")

    env = NaoSoccerEnv(stage_cfg=CURRICULUM[12], use_fov=False, seed=42)
    try:
        _assert(env._sidecar is not None, "env._sidecar instantiated")
        _assert(env._sidecar.backend_label == "real",
                f"backend_label = real (got {env._sidecar.backend_label})")
        _assert(env._sidecar.is_real_backend is True, "sidecar is real backend")

        obs, infos = env.reset(seed=42)
        _assert(len(obs) == len(env.agents), "reset returned one observation per active agent")
        _install_semantic_fixture(env)

        obs, rewards, terms, truncs, infos = env.step(_zero_actions(env))
        _assert(bool(infos), "infos returned after first semantic step")
        info = next(iter(infos.values()))

        _assert(info.get("execution_mode") == "bhuman_sidecar",
                f"execution_mode = bhuman_sidecar (got {info.get('execution_mode')})")
        _assert(info.get("sidecar_active") is True, "sidecar_active = True")
        _assert(info.get("sidecar_backend") == "real",
                f"sidecar_backend = real (got {info.get('sidecar_backend')})")
        _assert(info.get("sidecar_contract_validation_mode") is False,
                "sidecar_contract_validation_mode = False")
        _assert(info.get("takeover_simulator_active") is False,
                "takeover_simulator_active = False")
        expected_transport = expected_transport_mode()
        _assert(info.get("bridge_debug_transport_mode") == expected_transport,
                f"bridge_debug_transport_mode = {expected_transport} (got {info.get('bridge_debug_transport_mode')})")
        _assert(info.get("bridge_debug_game_state") == "playing",
                f"bridge_debug_game_state = playing (got {info.get('bridge_debug_game_state')})")
        _assert(info.get("bridge_debug_requested_state") == "playing",
                f"bridge_debug_requested_state = playing (got {info.get('bridge_debug_requested_state')})")

        macro_type = info.get("macro_type", "none")
        skill_request = info.get("sidecar_skill_request", "none")
        activation_graph = info.get("sidecar_activation_graph", [])

        _assert(macro_type in VALID_MACRO_TYPES,
                f"macro_type is valid contract value (got {macro_type})")
        _assert(skill_request in VALID_SKILL_REQUESTS,
                f"sidecar_skill_request is valid BHuman enum (got {skill_request})")
        _assert(skill_request != "duel",
                f"sidecar_skill_request is never 'duel' (got {skill_request})")
        _assert(info.get("active_agent_id") is not None,
                f"active_agent_id present (got {info.get('active_agent_id')})")
        _assert(isinstance(activation_graph, list), "activation_graph is list")
        _assert(len(activation_graph) >= 1, "activation_graph has at least one node")
        _assert(isinstance(activation_graph[0], dict),
                f"activation_graph entries are dicts (got {type(activation_graph[0]).__name__})")
        _assert("option" in activation_graph[0], "activation_graph node has option")
        _assert("depth" in activation_graph[0], "activation_graph node has depth")
        _assert("state" in activation_graph[0], "activation_graph node has state")

        if macro_type == "playBall_shoot":
            _assert(skill_request == "shoot",
                    f"macro shoot implies skill_request shoot (got {skill_request})")
        elif macro_type == "playBall_pass":
            _assert(skill_request == "pass",
                    f"macro pass implies skill_request pass (got {skill_request})")
        elif macro_type == "playBall_dribble":
            _assert(skill_request == "dribble",
                    f"macro dribble implies skill_request dribble (got {skill_request})")
        elif macro_type == "playBall_duel":
            has_zweikampf = any(node.get("state") == "zweikampf" for node in activation_graph)
            _assert(has_zweikampf,
                    "macro duel implies activation_graph contains zweikampf state")

        print(f"  INFO: macro_type={macro_type}")
        print(f"  INFO: sidecar_skill_request={skill_request}")
        print(f"  INFO: active_agent_id={info.get('active_agent_id')}")
        print(f"  INFO: bridge_frame_id={info.get('bridge_frame_id')}")

    finally:
        env.close()


def main() -> None:
    print("[TEST] Strong semantic validation for BHuman-in-loop")

    deterministic_tests = [
        test_active_agent_output,
        test_activation_graph_structured,
        test_scenario_shoot,
        test_scenario_pass,
        test_scenario_dribble,
        test_scenario_duel,
        test_is_playing_gating,
        test_kick_metadata,
    ]

    for fn in deterministic_tests:
        fn()

    test_real_environment_semantics()

    print("\nRESULT: PASS")
    print("  Deterministic semantic fixtures and real environment provenance are consistent.")
    print("  This is a strong semantic gate before long BHuman-in-loop training.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nRESULT: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        raise SystemExit(1)
