"""
smoke_test_sidecar.py -- Hardened integration smoke tests for BHuman Decision Sidecar.

Separated into:
  - SECTION A: Sidecar core tests (work with stub or real backend)
  - SECTION B: Contract validation tests (require real CFG loading)
  - SECTION C: Integration tests (require full env, detect stub/simulator leaks)

Per BHUMAN_IMPLEMENTATION_CORRECTION_GUIDE.md and BHUMAN_IN_LOOP_SIDECAR_GUIDE.md:
  - Tests MUST load real CFGs (repo-relative), not defaults
  - skill_request must never be "duel"
  - playBall_duel must come from activation_graph zweikampf state
  - last_kick_type must be "unavailable" without real execution evidence
  - Tests must detect BHumanTakeoverSimulator usage in normal mode
  - Pass scenario must not accept dribble

Ejecutar: python smoke_test_sidecar.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# Force sidecar mode for these tests
os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from config import resolve_cfg_root
from bhuman_cfg_parser import load_contract, AllConfigs
from bhuman_sidecar import (
    BHumanDecisionSidecar, PythonContractStub,
    GroundTruthFrame, SidecarOutput, ActivationNode,
    VALID_SKILL_REQUESTS, VALID_MOTION_REQUESTS,
)


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def _make_contract() -> AllConfigs:
    """Load contract using shared resolver — prefers real CFGs."""
    cfg_root = resolve_cfg_root()
    return load_contract(cfg_root=cfg_root)


def _make_deterministic_sidecar(contract: AllConfigs) -> PythonContractStub:
    """Deterministic semantic fixtures should validate the contract stub."""
    sidecar = PythonContractStub(contract)
    sidecar.reset()
    return sidecar


def _get_contract_source() -> str:
    """Return 'cfg_files' if real CFGs found, 'defaults' otherwise."""
    cfg_root = resolve_cfg_root()
    return "cfg_files" if cfg_root is not None else "defaults"


def _make_basic_frame(**overrides) -> GroundTruthFrame:
    defaults = dict(
        ball_pos=np.array([500.0, 0.0]),
        ball_vel=np.zeros(2),
        ball_seen_age_ms=0.0,
        own_poses={
            "agent_0": np.array([-500.0, 0.0]),
            "agent_1": np.array([-1000.0, 500.0]),
            "agent_2": np.array([-1000.0, -500.0]),
            "agent_3": np.array([-1500.0, 0.0]),
        },
        own_thetas={"agent_0": 0.0, "agent_1": 0.0, "agent_2": 0.0, "agent_3": 0.0},
        own_upright={"agent_0": True, "agent_1": True, "agent_2": True, "agent_3": True},
        own_roles={"agent_0": "forwardR", "agent_1": "forwardL",
                   "agent_2": "defenderR", "agent_3": "defenderL"},
        opp_poses=[],
        opp_upright=[],
        is_playing=True,
        own_score=0,
        opp_score=0,
    )
    defaults.update(overrides)
    return GroundTruthFrame(**defaults)


# =========================================================================
# SECTION A: SIDECAR CORE TESTS
# =========================================================================

def test_sidecar_startup():
    print("\n[TEST] Sidecar startup")
    contract = _make_contract()
    sidecar = BHumanDecisionSidecar(contract, execution_mode="bhuman_sidecar")
    _assert(sidecar is not None, "sidecar instantiated")
    _assert(hasattr(sidecar, "decide"), "has decide method")
    _assert(hasattr(sidecar, "decide_from_frame"), "has decide_from_frame method")
    _assert(hasattr(sidecar, "backend_label"), "has backend_label property")
    # Backend label must be explicit
    _assert(sidecar.backend_label in ("real", "contract_validation", "python_stub"),
            f"backend_label is explicit (got {sidecar.backend_label})")
    print(f"  INFO: backend_label={sidecar.backend_label}")
    sidecar.shutdown()


def test_cfg_loading_real():
    """CFG loading MUST use real files, not defaults."""
    print("\n[TEST] CFG loading (real files required)")
    source = _get_contract_source()
    print(f"  INFO: contract_source={source}")
    _assert(source == "cfg_files",
            f"real CFGs resolved from sibling repo (got {source})")
    contract = _make_contract()
    _assert(isinstance(contract, AllConfigs), "AllConfigs loaded")
    _assert(contract.field.x_opponent_ground == 2813.0, "4v4 Scaled field")
    _assert(contract.smash_or_pass.shoot_threshold == 0.8, "smashOrPass params loaded")
    _assert(contract.kick_info.ball_velocity_max == 1800.0, "kickInfo loaded")
    _assert(len(contract.strategy.tactics) == 2, "strategy FSM has 2 tactics")


def test_frame_input():
    print("\n[TEST] GroundTruthWorldState frame input")
    frame = _make_basic_frame()
    _assert(frame.ball_pos is not None, "ball_pos present")
    _assert(len(frame.own_poses) == 4, "4 own agents")
    _assert(frame.is_playing is True, "is_playing = True")


def test_active_agent_output():
    print("\n[TEST] active_agent output")
    contract = _make_contract()
    sidecar = _make_deterministic_sidecar(contract)
    frame = _make_basic_frame()
    output = sidecar.decide(frame, backend_label="contract_validation")

    _assert(isinstance(output, SidecarOutput), "output is SidecarOutput")
    _assert(output.active_agent is not None, "active_agent is not None")
    _assert(output.active_agent in frame.own_poses, "active_agent is valid agent")
    _assert(output.active_agent == "agent_0",
            f"active_agent = agent_0 (closest, got {output.active_agent})")


def test_skill_request_output():
    """skill_request must be a real BHuman enum — never 'duel'."""
    print("\n[TEST] SkillRequest output (real enum, no duel)")
    contract = _make_contract()
    sidecar = _make_deterministic_sidecar(contract)
    frame = _make_basic_frame()
    output = sidecar.decide(frame, backend_label="contract_validation")

    _assert(output.skill_request in VALID_SKILL_REQUESTS,
            f"skill_request is real BHuman enum (got {output.skill_request})")
    _assert(output.skill_request != "duel",
            f"skill_request is NEVER 'duel' (got {output.skill_request})")
    _assert(output.skill_request != "none",
            f"skill_request is not 'none' when playing (got {output.skill_request})")


def test_motion_request_output():
    """motion_request must use real BHuman MotionRequest.Motion enum."""
    print("\n[TEST] MotionRequest output (real enum)")
    contract = _make_contract()
    sidecar = _make_deterministic_sidecar(contract)
    frame = _make_basic_frame()
    output = sidecar.decide(frame, backend_label="contract_validation")

    _assert(output.motion_request in VALID_MOTION_REQUESTS,
            f"motion_request is real BHuman enum (got {output.motion_request})")
    _assert(isinstance(output.motion_speed, tuple), "motion_speed is tuple")
    _assert(len(output.motion_speed) == 3, "motion_speed has 3 components")


def test_activation_graph_structured():
    """activation_graph must be List[ActivationNode], not List[str]."""
    print("\n[TEST] ActivationGraph output (structured nodes)")
    contract = _make_contract()
    sidecar = _make_deterministic_sidecar(contract)
    frame = _make_basic_frame()
    output = sidecar.decide(frame, backend_label="contract_validation")

    _assert(isinstance(output.activation_graph, list), "activation_graph is list")
    _assert(len(output.activation_graph) >= 2, "activation_graph has entries")
    for node in output.activation_graph:
        _assert(isinstance(node, ActivationNode),
                f"node is ActivationNode (got {type(node).__name__})")
        _assert(hasattr(node, "option"), "node has option field")
        _assert(hasattr(node, "depth"), "node has depth field")
        _assert(hasattr(node, "state"), "node has state field")

    has_strategy = any(n.option == "Strategy" for n in output.activation_graph)
    _assert(has_strategy, "activation_graph includes Strategy node")


# =========================================================================
# SECTION B: SCENARIO TESTS (deterministic fixtures)
# =========================================================================

def test_scenario_shoot():
    print("\n[TEST] Scenario: shoot")
    contract = _make_contract()
    sidecar = _make_deterministic_sidecar(contract)

    frame = _make_basic_frame(
        ball_pos=np.array([2500.0, 0.0]),
        own_poses={
            "agent_0": np.array([2400.0, 0.0]),
            "agent_1": np.array([0.0, 500.0]),
            "agent_2": np.array([0.0, -500.0]),
            "agent_3": np.array([-1000.0, 0.0]),
        },
    )
    output = sidecar.decide(frame, backend_label="contract_validation")
    _assert(output.macro_type == "playBall_shoot",
            f"macro_type = playBall_shoot (got {output.macro_type})")
    _assert(output.skill_request == "shoot",
            f"skill_request = shoot (got {output.skill_request})")
    _assert(output.skill_request != "duel", "skill_request is not duel")
    _assert(output.kick_type != "none", f"kick_type present (got {output.kick_type})")
    _assert(output.kick_length > 0, f"kick_length > 0 (got {output.kick_length})")


def test_scenario_pass():
    """Pass scenario must produce 'pass', not accept 'dribble' as fallback."""
    print("\n[TEST] Scenario: pass (strict — no dribble accepted)")
    contract = _make_contract()
    sidecar = _make_deterministic_sidecar(contract)

    # Agent far from goal, teammate well-positioned ahead and offset.
    # Rating must exceed pass.rating_threshold (0.7).
    # forward_progress = (tm_x - agent_x) / field_length
    # dist_rating = 1 - |dist - 2500| / max_pass_distance
    # rating = 0.5 * forward_progress + 0.5 * dist_rating
    # Agent facing ~0 rad (straight ahead). Teammate at 45-ish deg offset.
    # angle_to_tm = atan2(1200, 1600) = ~36.9 deg, agent_theta = 0 deg
    # offset = 36.9 deg -> within [7, 25]? No, 36.9 > 25.
    # Need offset in [7, 25]. Agent facing 0 rad, tm at offset ~15 deg:
    # atan2(700, 2600) = ~15.1 deg. offset = 15.1 -> in [7,25] OK
    # dist = sqrt(2600^2 + 700^2) = 2692, fwd = 2600/5626 = 0.462
    # dist_rating = 1 - |2692-2500|/4500 = 0.957
    # rating = 0.5*0.462 + 0.5*0.957 = 0.710 > 0.7 OK
    frame = _make_basic_frame(
        ball_pos=np.array([-500.0, 0.0]),
        own_poses={
            "agent_0": np.array([-600.0, 0.0]),
            "agent_1": np.array([2000.0, 700.0]),    # ahead, good angle offset
            "agent_2": np.array([-1000.0, -800.0]),
            "agent_3": np.array([-1500.0, 0.0]),
        },
        own_thetas={"agent_0": 0.0, "agent_1": 0.0, "agent_2": 0.0, "agent_3": 0.0},
    )
    output = sidecar.decide(frame, backend_label="contract_validation")
    _assert(output.macro_type == "playBall_pass",
            f"macro_type = playBall_pass (got {output.macro_type})")
    _assert(output.skill_request == "pass",
            f"skill_request = pass (got {output.skill_request})")
    _assert(output.skill_request != "duel", "skill_request is not duel")


def test_scenario_dribble():
    print("\n[TEST] Scenario: dribble")
    contract = _make_contract()
    sidecar = _make_deterministic_sidecar(contract)

    frame = _make_basic_frame(
        ball_pos=np.array([-1000.0, 0.0]),
        own_poses={
            "agent_0": np.array([-1100.0, 0.0]),
            "agent_1": np.array([-2000.0, 1500.0]),
            "agent_2": np.array([-2000.0, -1500.0]),
            "agent_3": np.array([-2500.0, 0.0]),
        },
    )
    output = sidecar.decide(frame, backend_label="contract_validation")
    _assert(output.macro_type == "playBall_dribble",
            f"macro_type = playBall_dribble (got {output.macro_type})")
    _assert(output.skill_request == "dribble",
            f"skill_request = dribble (got {output.skill_request})")
    _assert(output.skill_request != "duel", "skill_request is not duel")


def test_scenario_duel():
    """Duel: macro_type from activation_graph, NOT from skill_request."""
    print("\n[TEST] Scenario: duel (macro_type + activation_graph, skill != duel)")
    contract = _make_contract()
    sidecar = _make_deterministic_sidecar(contract)

    frame = _make_basic_frame(
        ball_pos=np.array([0.0, 0.0]),
        own_poses={
            "agent_0": np.array([-100.0, 0.0]),
            "agent_1": np.array([-1000.0, 500.0]),
            "agent_2": np.array([-1000.0, -500.0]),
            "agent_3": np.array([-1500.0, 0.0]),
        },
        opp_poses=[np.array([100.0, 0.0])],
        opp_upright=[True],
    )
    output = sidecar.decide(frame, backend_label="contract_validation")

    # macro_type must be playBall_duel
    _assert(output.macro_type == "playBall_duel",
            f"macro_type = playBall_duel (got {output.macro_type})")

    # skill_request must NOT be "duel" — duel is a sub-state, not a skill
    _assert(output.skill_request != "duel",
            f"skill_request is NOT 'duel' (got {output.skill_request})")
    _assert(output.skill_request in VALID_SKILL_REQUESTS,
            f"skill_request is valid BHuman enum (got {output.skill_request})")

    # activation_graph must contain a Zweikampf node with state="zweikampf"
    zweikampf_nodes = [
        n for n in output.activation_graph
        if isinstance(n, ActivationNode) and n.state == "zweikampf"
    ]
    _assert(len(zweikampf_nodes) > 0,
            "activation_graph contains Zweikampf node with state='zweikampf'")


def test_is_playing_gating():
    print("\n[TEST] is_playing gating")
    contract = _make_contract()
    sidecar = _make_deterministic_sidecar(contract)

    frame = _make_basic_frame(is_playing=False)
    output = sidecar.decide(frame, backend_label="contract_validation")

    _assert(output.active_agent is None, "active_agent = None when not playing")
    _assert(output.macro_type == "none", "macro_type = none when not playing")
    _assert(output.behavior_status == "not_playing", "behavior_status = not_playing")
    _assert(output.skill_request == "none", "skill_request = none when not playing")


def test_kick_metadata():
    """last_kick_type must be 'unavailable' — no real execution evidence."""
    print("\n[TEST] kickType, kickLength, lastKickType (evidence-based)")
    contract = _make_contract()
    sidecar = _make_deterministic_sidecar(contract)

    frame = _make_basic_frame(
        ball_pos=np.array([2500.0, 0.0]),
        own_poses={"agent_0": np.array([2400.0, 0.0]),
                   "agent_1": np.array([0.0, 500.0]),
                   "agent_2": np.array([0.0, -500.0]),
                   "agent_3": np.array([-1000.0, 0.0])},
    )
    output = sidecar.decide(frame, backend_label="contract_validation")

    _assert(output.kick_type != "none", f"kick_type is set (got {output.kick_type})")
    _assert(output.kick_length > 0, f"kick_length > 0 (got {output.kick_length})")

    # last_kick_type: without real MotionInfo, must be "unavailable"
    _assert(output.last_kick_type == "unavailable",
            f"last_kick_type = 'unavailable' without execution evidence "
            f"(got {output.last_kick_type})")

    # Even after multiple calls, still unavailable (no real execution)
    output2 = sidecar.decide(frame, backend_label="contract_validation")
    _assert(output2.last_kick_type == "unavailable",
            f"last_kick_type stays 'unavailable' after 2nd call "
            f"(got {output2.last_kick_type})")


# =========================================================================
# SECTION C: INTEGRATION TESTS (env + anti-regression)
# =========================================================================

def test_stub_not_used_in_normal_mode():
    """
    Verify that in bhuman_sidecar mode:
      - sidecar exists and is not None
      - backend is 'real' or 'contract_validation' (never 'python_stub')
      - BHumanTakeoverSimulator is NOT active
    """
    print("\n[TEST] Stub NOT used in normal mode + no TakeoverSimulator")
    from config import EXECUTION_MODE
    _assert(EXECUTION_MODE == "bhuman_sidecar",
            f"EXECUTION_MODE = bhuman_sidecar (got {EXECUTION_MODE})")

    contract = _make_contract()
    sidecar = BHumanDecisionSidecar(contract, execution_mode="bhuman_sidecar")
    _assert(sidecar is not None, "sidecar is real, not None")
    _assert(sidecar.backend_label in ("real", "contract_validation"),
            f"backend is real or contract_validation (got {sidecar.backend_label})")
    _assert(sidecar.backend_label != "python_stub",
            "backend is NOT python_stub in sidecar mode")
    _assert(sidecar.uses_takeover_simulator is False,
            "sidecar does NOT use takeover simulator")

    # Verify environment does not instantiate BHumanTakeoverSimulator
    from config import CURRICULUM
    from environment import NaoSoccerEnv
    env = NaoSoccerEnv(stage_cfg=CURRICULUM[0], use_fov=False, seed=42)
    _assert(env._sidecar is not None,
            "env._sidecar is active in bhuman_sidecar mode")
    _assert(env._execution_mode == "bhuman_sidecar",
            f"env._execution_mode = bhuman_sidecar (got {env._execution_mode})")
    _assert(env._takeover_sim is None,
            "env._takeover_sim is None in bhuman_sidecar mode "
            "(BHumanTakeoverSimulator not used)")
    env.close()
    sidecar.shutdown()


def test_temporal_budget():
    print("\n[TEST] Temporal budget per frame")
    contract = _make_contract()
    sidecar = BHumanDecisionSidecar(contract, execution_mode="bhuman_sidecar")

    frame = _make_basic_frame()
    for _ in range(5):
        sidecar.decide_from_frame(frame)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        sidecar.decide_from_frame(frame)
        dt_ms = (time.perf_counter() - t0) * 1000
        times.append(dt_ms)

    mean_ms = sum(times) / len(times)
    max_ms = max(times)
    # Real 4-player BHuman-in-loop runs a persistent bridge per player. The
    # guard here is for steady-state viability on this host, not stub-level
    # microbenchmarks.
    _assert(mean_ms < 75.0,
            f"mean decision time < 75ms (got {mean_ms:.3f}ms)")
    _assert(max_ms < 150.0,
            f"max decision time < 150ms (got {max_ms:.3f}ms)")
    print(f"  INFO: mean={mean_ms:.3f}ms, max={max_ms:.3f}ms, p99={sorted(times)[98]:.3f}ms")
    sidecar.shutdown()


def test_env_step_with_sidecar():
    print("\n[TEST] End-to-end env step with sidecar")
    from config import CURRICULUM
    from environment import NaoSoccerEnv

    env = NaoSoccerEnv(stage_cfg=CURRICULUM[0], use_fov=False, seed=42)
    obs, infos = env.reset()

    _assert(len(obs) >= 1, "observations returned")
    _assert(len(infos) >= 1, "infos returned")

    # Do one step
    actions = {aid: env.action_space(aid).sample() for aid in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)

    sample_info = next(iter(infos.values()))

    _assert("active_agent_id" in sample_info, "active_agent_id in info")
    _assert("macro_type" in sample_info, "macro_type in info")
    _assert("execution_mode" in sample_info, "execution_mode in info")
    _assert("sidecar_active" in sample_info, "sidecar_active in info")
    _assert(sample_info["sidecar_active"] is True, "sidecar is active")
    _assert(sample_info["execution_mode"] == "bhuman_sidecar",
            f"execution_mode = bhuman_sidecar (got {sample_info['execution_mode']})")

    # Sidecar fields
    _assert("sidecar_skill_request" in sample_info, "sidecar_skill_request in info")
    _assert("sidecar_motion_request" in sample_info, "sidecar_motion_request in info")
    _assert("sidecar_activation_graph" in sample_info, "sidecar_activation_graph in info")
    _assert("sidecar_kick_type" in sample_info, "sidecar_kick_type in info")
    _assert("sidecar_kick_length" in sample_info, "sidecar_kick_length in info")
    _assert("sidecar_last_kick_type" in sample_info, "sidecar_last_kick_type in info")

    # New fields: backend provenance and takeover simulator status
    _assert("sidecar_backend" in sample_info, "sidecar_backend in info")
    _assert("sidecar_bridge_mode" in sample_info, "sidecar_bridge_mode in info")
    _assert("sidecar_contract_validation_mode" in sample_info,
            "sidecar_contract_validation_mode in info")
    _assert("takeover_simulator_active" in sample_info,
            "takeover_simulator_active in info")
    _assert(sample_info["takeover_simulator_active"] is False,
            "takeover_simulator_active is False in sidecar mode")
    _assert(sample_info["sidecar_bridge_mode"] == "persistent_2d_cognition",
            f"sidecar_bridge_mode = persistent_2d_cognition "
            f"(got {sample_info['sidecar_bridge_mode']})")

    # skill_request must never be "duel"
    _assert(sample_info["sidecar_skill_request"] != "duel",
            f"sidecar_skill_request is not 'duel' (got {sample_info['sidecar_skill_request']})")

    # last_kick_type must be "unavailable"
    _assert(sample_info["sidecar_last_kick_type"] == "unavailable",
            f"sidecar_last_kick_type = 'unavailable' (got {sample_info['sidecar_last_kick_type']})")

    # activation_graph entries must be dicts (structured), not plain strings
    ag = sample_info["sidecar_activation_graph"]
    if ag:
        _assert(isinstance(ag[0], dict),
                f"activation_graph entries are dicts (got {type(ag[0]).__name__})")

    print(f"  INFO: macro_type={sample_info['macro_type']}")
    print(f"  INFO: sidecar_skill_request={sample_info['sidecar_skill_request']}")
    print(f"  INFO: sidecar_backend={sample_info['sidecar_backend']}")
    print(f"  INFO: sidecar_bridge_mode={sample_info['sidecar_bridge_mode']}")
    print(f"  INFO: active_agent_id={sample_info['active_agent_id']}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    tests = [
        # Section A: Core
        test_sidecar_startup,
        test_cfg_loading_real,
        test_frame_input,
        test_active_agent_output,
        test_skill_request_output,
        test_motion_request_output,
        test_activation_graph_structured,
        # Section B: Scenarios
        test_scenario_shoot,
        test_scenario_pass,
        test_scenario_dribble,
        test_scenario_duel,
        test_is_playing_gating,
        test_kick_metadata,
        # Section C: Integration
        test_stub_not_used_in_normal_mode,
        test_temporal_budget,
        test_env_step_with_sidecar,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except (AssertionError, AssertionError, Exception) as e:
            failed += 1
            print(f"  FAILED: {e}")

    print(f"\n{'='*60}")
    print(f"SIDECAR SMOKE TESTS: {passed} passed, {failed} failed, {passed + failed} total")
    source = _get_contract_source()
    print(f"Contract source: {source}")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
