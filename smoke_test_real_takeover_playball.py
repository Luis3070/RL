"""
smoke_test_real_takeover_playball.py -- Diagnostic gate for real playBall semantics.

Purpose:
  1. Verify that the real RLBridge can emit a true playBall_* macro in 4v4.
  2. Verify that reduced-agent curriculum stages still preserve 4-player
     sidecar context for BHuman reasoning.
  3. Verify that the env does NOT mark takeover_active when the real bridge
     is still returning walk/stand/none instead of a playBall macro.

This test complements:
  - smoke_test_environment_bridge.py   (transport / info plumbing)
  - smoke_test_semantic_strong.py      (contract semantics + provenance)

This one focuses on the alignment between:
  local handoff_ready
  vs
  real BHuman playBall intent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from bhuman_cfg_parser import load_contract
from bhuman_sidecar import (
    BHumanDecisionSidecar,
    GroundTruthFrame,
)
from bridge.bridge_protocol import BridgeConfig
from config import CURRICULUM, resolve_cfg_root
from environment import NaoSoccerEnv


REAL_PLAYBALL_MACROS = {
    "playBall_shoot",
    "playBall_pass",
    "playBall_dribble",
    "playBall_duel",
}
EXPECTED_TACTICS = {"t121_4v4_scaled", "t13_4v4_scaled"}


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


def _make_real_playball_frame() -> GroundTruthFrame:
    """
    Stable 4v4 context where the real bridge already proved capable of
    emitting a real playBall_* semantic (typically shoot).
    """
    return GroundTruthFrame(
        ball_pos=np.array([2350.0, 0.0], dtype=np.float32),
        ball_vel=np.zeros(2, dtype=np.float32),
        ball_seen_age_ms=0.0,
        own_poses={
            "agent_0": np.array([2250.0, 0.0], dtype=np.float32),
            "agent_1": np.array([1500.0, 700.0], dtype=np.float32),
            "agent_2": np.array([1300.0, -700.0], dtype=np.float32),
            "agent_3": np.array([-1800.0, 0.0], dtype=np.float32),
        },
        own_thetas={
            "agent_0": 0.0,
            "agent_1": 0.0,
            "agent_2": 0.0,
            "agent_3": 0.0,
        },
        own_upright={
            "agent_0": True,
            "agent_1": True,
            "agent_2": True,
            "agent_3": True,
        },
        own_roles={
            "agent_0": "forwardR",
            "agent_1": "forwardL",
            "agent_2": "defenderR",
            "agent_3": "defenderL",
        },
        opp_poses=[
            np.array([1600.0, 100.0], dtype=np.float32),
            np.array([1700.0, 900.0], dtype=np.float32),
            np.array([1700.0, -900.0], dtype=np.float32),
            np.array([-2200.0, 0.0], dtype=np.float32),
        ],
        opp_upright=[True, True, True, True],
        is_playing=True,
        own_score=0,
        opp_score=0,
    )


def _make_real_pass_frame() -> GroundTruthFrame:
    """
    Stable 4v4 context where the real bridge has produced a true pass
    decision in repeated probes.
    """
    return GroundTruthFrame(
        ball_pos=np.array([819.2516200456632, 403.05814402973715], dtype=np.float32),
        ball_vel=np.zeros(2, dtype=np.float32),
        ball_seen_age_ms=0.0,
        own_poses={
            "agent_0": np.array([650.2150286884579, 548.3966855946207], dtype=np.float32),
            "agent_1": np.array([2159.2993529482387, 766.3581725306999], dtype=np.float32),
            "agent_2": np.array([-577.7218925692251, -414.64062769712814], dtype=np.float32),
            "agent_3": np.array([-1367.9467972223988, -30.63538556879189], dtype=np.float32),
        },
        own_thetas={
            "agent_0": 0.0,
            "agent_1": 0.0,
            "agent_2": 0.0,
            "agent_3": 0.0,
        },
        own_upright={
            "agent_0": True,
            "agent_1": True,
            "agent_2": True,
            "agent_3": True,
        },
        own_roles={
            "agent_0": "forwardR",
            "agent_1": "forwardL",
            "agent_2": "defenderR",
            "agent_3": "defenderL",
        },
        opp_poses=[
            np.array([905.1018446197895, -277.1437845362635], dtype=np.float32),
            np.array([903.0372319183127, -10.980739174441624], dtype=np.float32),
            np.array([1807.7812441535934, -905.4640521614305], dtype=np.float32),
            np.array([1075.3657741984584, 1109.0058713291578], dtype=np.float32),
        ],
        opp_upright=[True, True, True, True],
        is_playing=True,
        own_score=0,
        opp_score=0,
    )


def _install_walk_only_handoff_fixture(env: NaoSoccerEnv) -> None:
    """
    Fixture where local handoff_ready is True, but the real bridge still
    responds with walk/stand/none. The env must NOT treat that as a real
    takeover anymore.
    """
    env._ball.pos[:] = np.array([200.0, 0.0], dtype=np.float32)
    env._ball.vel[:] = 0.0
    env._ball.seen_age_ms = 0.0

    own_fixture = [
        (np.array([0.0, 0.0], dtype=np.float32), 0.0),
        (np.array([-1000.0, 1000.0], dtype=np.float32), 0.0),
        (np.array([-1000.0, -1000.0], dtype=np.float32), 0.0),
        (np.array([-1800.0, 0.0], dtype=np.float32), 0.0),
    ]
    for idx, aid in enumerate(env.agents):
        agent = env._agents_state[aid]
        agent.pos = own_fixture[idx][0].copy()
        agent.theta = own_fixture[idx][1]
        agent.vel = np.zeros(2, dtype=np.float32)
        agent.omega = 0.0
        agent.is_upright = True

    opp_fixture = [
        (np.array([2300.0, 1200.0], dtype=np.float32), np.pi),
        (np.array([2300.0, -1200.0], dtype=np.float32), np.pi),
        (np.array([2500.0, 0.0], dtype=np.float32), np.pi),
        (np.array([1500.0, 0.0], dtype=np.float32), np.pi),
    ]
    for idx, opp in enumerate(env._opponents_state):
        opp.pos = opp_fixture[idx][0].copy()
        opp.theta = opp_fixture[idx][1]
        opp.vel = np.zeros(2, dtype=np.float32)
        opp.is_upright = True

    env._game.is_playing = True
    env._game.own_score = 0
    env._game.opp_score = 0


def test_real_bridge_can_emit_playball() -> None:
    print("\n[TEST] Real bridge emits playBall_* in full 4v4 context")

    bridge_exe = BridgeConfig().bridge_executable
    _assert(Path(bridge_exe).exists(), f"bridge executable exists at {bridge_exe}")

    cfg_root = resolve_cfg_root()
    _assert(cfg_root is not None, f"real CFG root resolved (got {cfg_root})")

    contract = load_contract(cfg_root=cfg_root)
    sidecar = BHumanDecisionSidecar(contract, execution_mode="bhuman_sidecar")
    try:
        output = sidecar.decide_from_frame(_make_real_playball_frame())
        _assert(output.backend == "real",
                f"backend = real (got {output.backend})")
        _assert(output.macro_type in REAL_PLAYBALL_MACROS,
                f"macro_type is real playBall_* (got {output.macro_type})")
        _assert(output.current_tactic in EXPECTED_TACTICS,
                f"current_tactic is normalized 4v4 tactic (got {output.current_tactic})")
        _assert(output.skill_request not in {"none", "walk", "stand"},
                f"skill_request is an execution skill (got {output.skill_request})")
        _assert(output.active_agent is not None,
                f"active_agent resolved (got {output.active_agent})")
        print(f"  INFO: current_tactic={output.current_tactic}")
        print(f"  INFO: active_agent={output.active_agent}")
    finally:
        sidecar.shutdown()


def test_real_bridge_can_emit_pass() -> None:
    print("\n[TEST] Real bridge emits playBall_pass in full 4v4 context")

    contract = load_contract(cfg_root=resolve_cfg_root())
    sidecar = BHumanDecisionSidecar(contract, execution_mode="bhuman_sidecar")
    try:
        output = None
        for probe_idx in range(3):
            output = sidecar.decide_from_frame(_make_real_pass_frame())
            if output.macro_type == "playBall_pass":
                break
        _assert(output is not None, "pass probe returned an output")
        _assert(output.backend == "real",
                f"backend = real (got {output.backend})")
        _assert(output.current_tactic in EXPECTED_TACTICS,
                f"current_tactic is normalized 4v4 tactic (got {output.current_tactic})")
        _assert(output.macro_type == "playBall_pass",
                f"macro_type = playBall_pass (got {output.macro_type})")
        _assert(output.skill_request == "pass",
                f"skill_request = pass (got {output.skill_request})")
        _assert(output.active_agent is not None,
                f"active_agent resolved (got {output.active_agent})")
        print(f"  INFO: pass_active_agent={output.active_agent}")
    finally:
        sidecar.shutdown()


def test_curriculum_stage0_keeps_4_player_sidecar_context() -> None:
    print("\n[TEST] Stage 0 preserves 4-player sidecar context")

    env = NaoSoccerEnv(stage_cfg=CURRICULUM[0], use_fov=False, seed=7)
    try:
        obs, infos = env.reset(seed=7)
        _assert(len(env.agents) == 1,
                f"stage 0 still trains 1 RL agent (got {len(env.agents)})")
        frame = env._sidecar._build_frame(
            env._agents_state,
            env._ball,
            env._opponents_state,
            env._game,
        )
        _assert(len(frame.own_poses) == 4,
                f"sidecar frame keeps 4 own players (got {len(frame.own_poses)})")
        for aid in ("agent_0", "agent_1", "agent_2", "agent_3"):
            _assert(aid in frame.own_poses, f"{aid} present in sidecar frame")
    finally:
        env.close()


def test_handoff_does_not_trigger_false_takeover() -> None:
    print("\n[TEST] Handoff-ready does not trigger false takeover on walk/stand/none")

    env = NaoSoccerEnv(stage_cfg=CURRICULUM[12], use_fov=False, seed=42)
    try:
        obs, infos = env.reset(seed=42)
        _install_walk_only_handoff_fixture(env)

        for step_idx in range(3):
            obs, rewards, terms, truncs, infos = env.step(_zero_actions(env))
            sample = next(iter(infos.values()))

            _assert(sample.get("handoff_ready") is True,
                    f"handoff_ready = True on probe step {step_idx + 1}")
            _assert(sample.get("sidecar_macro_type") == "none",
                    f"sidecar_macro_type = none on probe step {step_idx + 1} "
                    f"(got {sample.get('sidecar_macro_type')})")
            _assert(sample.get("sidecar_skill_request") in {"walk", "stand", "none"},
                    f"sidecar_skill_request stays non-playBall on probe step {step_idx + 1} "
                    f"(got {sample.get('sidecar_skill_request')})")
            _assert(sample.get("takeover_active") is False,
                    f"takeover_active remains False on probe step {step_idx + 1}")
            _assert(sample.get("macro_type") == "none",
                    f"macro_type remains none on probe step {step_idx + 1} "
                    f"(got {sample.get('macro_type')})")

    finally:
        env.close()


def main() -> None:
    print("[TEST] Real takeover / playBall alignment smoke")
    test_real_bridge_can_emit_playball()
    test_real_bridge_can_emit_pass()
    test_curriculum_stage0_keeps_4_player_sidecar_context()
    test_handoff_does_not_trigger_false_takeover()
    print("\nRESULT: PASS")
    print("  Real playBall semantics exist in 4v4 context.")
    print("  Real playBall_pass semantics are reachable from a stable 4v4 fixture.")
    print("  Reduced-agent curriculum keeps 4-player sidecar context.")
    print("  The env no longer marks false takeovers when bridge intent is walk/stand/none.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nRESULT: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        raise SystemExit(1)
