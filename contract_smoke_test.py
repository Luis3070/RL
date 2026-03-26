"""
contract_smoke_test.py — Smoke tests contractuales mínimos.

Verifica que el parser y contract loader producen estructuras
consistentes con la documentación de BHuman 4v4 Scaled.

Fuente de verdad:
  - BHUMANPARSER_GUIDE.md
  - BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md
  - PROJECT_AUDIT_FOR_RL.md
  - BHUMAN_RL_CURRICULUM_CONTRACT.md

Ejecutar: python contract_smoke_test.py
"""

from __future__ import annotations

import math
import sys

from bhuman_cfg_parser import (
    AllConfigs,
    BHumanCfgParser,
    FIELD_SYMBOLS_4V4_SCALED,
    MACRO_END_REASONS,
    MACRO_TYPES,
    TAKEOVER_SKILLS,
    VALID_STRATEGY_NAME,
    VALID_TACTIC_NAMES,
    load_contract,
    get_home_poses,
)


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def test_parser_basic():
    """Test parser can parse basic CFG syntax."""
    print("\n[TEST] Parser basic syntax")
    parser = BHumanCfgParser()

    # Simple key = value
    result = parser.parse_string("speed = 250;")
    _assert(result["speed"] == 250.0, "parse int value")

    # Float
    result = parser.parse_string("ratio = 0.8;")
    _assert(abs(result["ratio"] - 0.8) < 1e-6, "parse float value")

    # Boolean
    result = parser.parse_string("enabled = true; disabled = false;")
    _assert(result["enabled"] is True, "parse true")
    _assert(result["disabled"] is False, "parse false")

    # String
    result = parser.parse_string('name = "hello";')
    _assert(result["name"] == "hello", "parse string")

    # Array
    result = parser.parse_string("items = [1, 2, 3];")
    _assert(result["items"] == [1.0, 2.0, 3.0], "parse array")

    # Nested struct
    result = parser.parse_string("outer = { inner = 42; };")
    _assert(result["outer"]["inner"] == 42.0, "parse nested struct")

    # Expression with symbol
    result = parser.parse_string("pos = xPosOwnGoalArea - 94;")
    expected = FIELD_SYMBOLS_4V4_SCALED["xPosOwnGoalArea"] - 94.0
    _assert(abs(result["pos"] - expected) < 1e-6,
            f"parse expression with symbol: {result['pos']} == {expected}")

    # Negative number
    result = parser.parse_string("val = -1500;")
    _assert(result["val"] == -1500.0, "parse negative number")

    # Multiplication
    result = parser.parse_string("val = centerCircleRadius * 1.5;")
    expected = FIELD_SYMBOLS_4V4_SCALED["centerCircleRadius"] * 1.5
    _assert(abs(result["val"] - expected) < 1e-6,
            f"parse multiplication: {result['val']} == {expected}")


def test_parser_comments():
    """Test parser handles all comment types."""
    print("\n[TEST] Parser comments")
    parser = BHumanCfgParser()

    result = parser.parse_string("""
    // line comment
    a = 1;
    /* block
       comment */
    b = 2;
    # hash comment
    c = 3;
    """)
    _assert(result["a"] == 1.0, "value after // comment")
    _assert(result["b"] == 2.0, "value after /* */ comment")
    _assert(result["c"] == 3.0, "value after # comment")


def test_contract_loader():
    """Test contract loader produces valid AllConfigs."""
    print("\n[TEST] Contract loader")
    configs = load_contract()

    _assert(isinstance(configs, AllConfigs), "AllConfigs instance")
    _assert(configs.field is not None, "field config exists")
    _assert(configs.strategy is not None, "strategy config exists")
    _assert(configs.tactics is not None, "tactics dict exists")
    _assert(configs.smash_or_pass is not None, "smash_or_pass config exists")
    _assert(configs.kick_at_goal is not None, "kick_at_goal config exists")
    _assert(configs.pass_to_teammate is not None, "pass_to_teammate config exists")
    _assert(configs.zweikampf is not None, "zweikampf config exists")
    _assert(configs.walk is not None, "walk config exists")
    _assert(configs.kick_info is not None, "kick_info config exists")
    _assert(configs.handoff is not None, "handoff config exists")
    _assert(configs.timing is not None, "timing config exists")
    _assert(configs.skill_behavior_control is not None, "skill_behavior_control config exists")
    _assert(configs.strategy_behavior_control is not None, "strategy_behavior_control config exists")
    _assert(configs.behavior is not None, "behavior config exists")
    _assert(hasattr(configs, "loaded_from"), "loaded_from field exists")


def test_field_4v4_scaled():
    """Test field dimensions are 4v4 Scaled, NOT legacy 5v5."""
    print("\n[TEST] Field 4v4 Scaled (NOT 5v5)")
    configs = load_contract()
    f = configs.field

    # 4v4 Scaled field: 5626mm x 3750mm
    _assert(abs(f.x_opponent_ground - 2813.0) < 1e-6,
            f"x_opponent_ground = 2813 (got {f.x_opponent_ground})")
    _assert(abs(f.x_own_ground - (-2813.0)) < 1e-6,
            f"x_own_ground = -2813 (got {f.x_own_ground})")
    _assert(abs(f.y_left_sideline - 1875.0) < 1e-6,
            f"y_left_sideline = 1875 (got {f.y_left_sideline})")
    _assert(abs(f.y_right_sideline - (-1875.0)) < 1e-6,
            f"y_right_sideline = -1875 (got {f.y_right_sideline})")

    # NOT legacy 5v5 (4500 x 3000)
    _assert(f.x_opponent_ground != 4500.0, "NOT legacy 5v5 x=4500")
    _assert(f.y_left_sideline != 3000.0, "NOT legacy 5v5 y=3000")

    # Derived
    _assert(abs(f.length - 5626.0) < 1e-6, f"field length = 5626 (got {f.length})")
    _assert(abs(f.width - 3750.0) < 1e-6, f"field width = 3750 (got {f.width})")
    _assert(abs(f.goal_width - 1600.0) < 1e-6, f"goal width = 1600 (got {f.goal_width})")
    _assert(abs(f.center_circle_r - 469.0) < 1e-6,
            f"center circle = 469 (got {f.center_circle_r})")


def test_tactics_correct():
    """Test only valid 4v4 tactics exist."""
    print("\n[TEST] Tactics correct (t121, t13 only)")
    configs = load_contract()

    _assert(len(configs.tactics) == 2, f"exactly 2 tactics (got {len(configs.tactics)})")
    _assert("t121_4v4_scaled" in configs.tactics, "t121_4v4_scaled exists")
    _assert("t13_4v4_scaled" in configs.tactics, "t13_4v4_scaled exists")

    # No legacy tactics
    legacy = ["t22", "t112", "t4", "t121", "t211", "t31", "t13", "t_Tortuga"]
    for name in legacy:
        _assert(name not in configs.tactics, f"no legacy tactic '{name}'")

    # t121 positions
    t121 = configs.tactics["t121_4v4_scaled"]
    _assert(len(t121.positions) == 5, f"t121 has 5 positions (got {len(t121.positions)})")
    roles = {rp.role_type for rp in t121.positions}
    _assert("goalkeeper" in roles, "t121 has goalkeeper")
    _assert("defenderL" in roles, "t121 has defenderL")
    _assert("forwardR" in roles, "t121 has forwardR")

    # Goalkeeper position
    gk = [rp for rp in t121.positions if rp.role_type == "goalkeeper"][0]
    _assert(abs(gk.pose.x - (-2532.0)) < 1e-6,
            f"GK x = -2532 (got {gk.pose.x})")

    # t13 positions
    t13 = configs.tactics["t13_4v4_scaled"]
    _assert(len(t13.positions) == 4, f"t13 has 4 positions (got {len(t13.positions)})")


def test_strategy_fsm():
    """Test strategy FSM is s4v4_scaled with correct transitions."""
    print("\n[TEST] Strategy FSM")
    configs = load_contract()
    s = configs.strategy

    _assert(s.name == "s4v4_scaled", f"strategy = s4v4_scaled (got {s.name})")
    _assert(len(s.tactics) == 2, f"2 tactic states in FSM (got {len(s.tactics)})")

    # First tactic state
    ts0 = s.tactics[0]
    _assert(ts0.tactic == "t121_4v4_scaled", f"initial tactic = t121 (got {ts0.tactic})")
    _assert(len(ts0.transitions) >= 1, "t121 has at least 1 transition")
    _assert(ts0.transitions[0].to == "t13_4v4_scaled",
            f"t121 transitions to t13 (got {ts0.transitions[0].to})")


def test_handoff_ready_exists():
    """Test handoff_ready envelope is defined."""
    print("\n[TEST] handoff_ready config exists")
    configs = load_contract()
    h = configs.handoff

    _assert(h.ball_seen_age_max == 300.0, f"ball_seen_age_max = 300ms (got {h.ball_seen_age_max})")
    _assert(h.ball_rel_x_min == 0.0, f"ball_rel_x_min = 0 (got {h.ball_rel_x_min})")
    _assert(h.ball_rel_x_max == 500.0, f"ball_rel_x_max = 500 (got {h.ball_rel_x_max})")
    _assert(h.ball_rel_y_abs_max == 300.0, f"ball_rel_y_abs_max = 300 (got {h.ball_rel_y_abs_max})")
    _assert(h.angle_to_ball_abs_max == 80.0,
            f"angle_to_ball_abs_max = 80deg (got {h.angle_to_ball_abs_max})")
    _assert(h.takeover_timeout_frames == 90,
            f"takeover_timeout = 90 frames (got {h.takeover_timeout_frames})")
    _assert(h.takeover_grace_frames == 2,
            f"takeover_grace = 2 frames (got {h.takeover_grace_frames})")


def test_takeover_active_exists():
    """Test takeover_active skill set is defined."""
    print("\n[TEST] takeover_active skills exist")
    _assert("shoot" in TAKEOVER_SKILLS, "shoot in takeover skills")
    _assert("pass" in TAKEOVER_SKILLS, "pass in takeover skills")
    _assert("dribble" in TAKEOVER_SKILLS, "dribble in takeover skills")
    _assert("duel" in TAKEOVER_SKILLS, "duel in takeover skills")


def test_bridge_schema_macrostates():
    """Test macro-action types and end reasons match documentation."""
    print("\n[TEST] Bridge schema macrostates")
    _assert("none" in MACRO_TYPES, "none macro type")
    _assert("playBall_shoot" in MACRO_TYPES, "playBall_shoot macro type")
    _assert("playBall_pass" in MACRO_TYPES, "playBall_pass macro type")
    _assert("playBall_dribble" in MACRO_TYPES, "playBall_dribble macro type")
    _assert("playBall_duel" in MACRO_TYPES, "playBall_duel macro type")

    _assert("completed" in MACRO_END_REASONS, "completed end reason")
    _assert("timeout" in MACRO_END_REASONS, "timeout end reason")
    _assert("ball_lost" in MACRO_END_REASONS, "ball_lost end reason")
    _assert("role_changed" in MACRO_END_REASONS, "role_changed end reason")
    _assert("game_state_changed" in MACRO_END_REASONS, "game_state_changed end reason")


def test_no_local_physics_as_truth():
    """Test that contract does NOT use local ball physics as source of truth."""
    print("\n[TEST] No local physics as source of truth")
    configs = load_contract()

    # Kick info exists and references BHuman executor, not env physics
    _assert(configs.kick_info.walk_forward_min == 400.0,
            "kick range min from BHuman (400mm)")
    _assert(configs.kick_info.forward_fast_max == 5400.0,
            "kick range max from BHuman (5400mm)")
    _assert(configs.kick_info.ball_velocity_max == 1800.0,
            "ball velocity max from BHuman (1800mm/s)")


def test_temporal_budget():
    """Test temporal budget constraints."""
    print("\n[TEST] Temporal budget")
    configs = load_contract()
    t = configs.timing

    _assert(t.cognition_cycle_ms == 33.0, f"Cognition cycle = 33ms (got {t.cognition_cycle_ms})")
    _assert(t.rl_max_inference_ms == 3.0, f"RL max inference = 3ms (got {t.rl_max_inference_ms})")
    _assert(t.bridge_max_latency_ms == 2.0,
            f"Bridge max latency = 2ms (got {t.bridge_max_latency_ms})")


def test_physical_params():
    """Test critical physical parameters."""
    print("\n[TEST] Physical parameters")
    configs = load_contract()
    w = configs.walk

    _assert(w.max_forward == 250.0, f"max_forward = 250 mm/s (got {w.max_forward})")
    _assert(w.max_sideways == 200.0, f"max_sideways = 200 mm/s (got {w.max_sideways})")
    _assert(w.max_backwards == 150.0, f"max_backwards = 150 mm/s (got {w.max_backwards})")
    _assert(w.max_rotation == 120.0, f"max_rotation = 120 deg/s (got {w.max_rotation})")
    _assert(w.obstacle_avoidance_distance == 480.0,
            f"obstacle avoidance = 480mm (got {w.obstacle_avoidance_distance})")
    _assert(w.field_border_expansion == 300.0,
            f"field_border_expansion = 300mm (got {w.field_border_expansion})")


def test_decision_params():
    """Test decision point parameters (Fase 2 learnable)."""
    print("\n[TEST] Decision parameters")
    configs = load_contract()
    sp = configs.smash_or_pass
    kg = configs.kick_at_goal
    pt = configs.pass_to_teammate

    _assert(sp.shoot_threshold == 0.8, f"shootThreshold = 0.8 (got {sp.shoot_threshold})")
    _assert(sp.min_pass_distance == 1500.0,
            f"minPassDistance = 1500 (got {sp.min_pass_distance})")
    _assert(sp.max_pass_distance == 4500.0,
            f"maxPassDistance = 4500 (got {sp.max_pass_distance})")
    _assert(sp.min_penalty == 0.1, f"minPenalty = 0.1 (got {sp.min_penalty})")
    _assert(sp.max_penalty == 0.8, f"maxPenalty = 0.8 (got {sp.max_penalty})")

    _assert(kg.min_opening_angle == 6.0,
            f"minOpeningAngle = 6deg (got {kg.min_opening_angle})")
    _assert(kg.hysteresis_number == 300,
            f"hysteresis = 300ms (got {kg.hysteresis_number})")

    _assert(pt.min_angle_offset == 7.0, f"minAngleOffset = 7deg (got {pt.min_angle_offset})")
    _assert(pt.max_angle_offset == 25.0, f"maxAngleOffset = 25deg (got {pt.max_angle_offset})")
    _assert(pt.rating_threshold == 0.7, f"ratingThreshold = 0.7 (got {pt.rating_threshold})")


def test_home_poses_lookup():
    """Test home pose lookup excludes goalkeeper."""
    print("\n[TEST] Home poses lookup")
    configs = load_contract()
    poses = get_home_poses("t121_4v4_scaled", configs)

    _assert("goalkeeper" not in poses, "goalkeeper excluded from RL poses")
    _assert("defenderL" in poses, "defenderL in poses")
    _assert("forwardR" in poses, "forwardR in poses")
    _assert(len(poses) == 4, f"4 field player poses (got {len(poses)})")

    # Check specific position
    dl = poses["defenderL"]
    _assert(abs(dl[0] - (-1468.0)) < 1e-6,
            f"defenderL x = -1468 (got {dl[0]})")


def test_field_symbols_completeness():
    """Test all 29 FIELD_SYMBOLS are present."""
    print("\n[TEST] Field symbols completeness")
    _assert(len(FIELD_SYMBOLS_4V4_SCALED) >= 29,
            f"at least 29 symbols (got {len(FIELD_SYMBOLS_4V4_SCALED)})")
    required = [
        "xPosOpponentGroundLine", "xPosOwnGroundLine",
        "yPosLeftSideline", "yPosRightSideline",
        "yPosLeftGoal", "yPosRightGoal",
        "centerCircleRadius", "xPosOpponentPenaltyArea",
        "xPosOwnPenaltyArea", "xPosOwnGoalArea",
        "goalPostRadius", "fieldLinesWidth",
    ]
    for sym in required:
        _assert(sym in FIELD_SYMBOLS_4V4_SCALED, f"symbol '{sym}' present")


def test_zweikampf_subsystems():
    """Test Zweikampf has all 6 subsystems."""
    print("\n[TEST] Zweikampf subsystems")
    configs = load_contract()
    z = configs.zweikampf

    _assert(z.search is not None, "search subsystem exists")
    _assert(z.ratings is not None, "ratings subsystem exists")
    _assert(z.timings is not None, "timings subsystem exists")
    _assert(z.obstacle_handling is not None, "obstacle_handling subsystem exists")
    _assert(z.steal_ball is not None, "steal_ball subsystem exists")
    _assert(z.last_kick_hysterese is not None, "last_kick_hysterese subsystem exists")

    _assert(z.obstacle_handling.obstacle_blocking_radius == 200.0,
            f"blocking_radius = 200mm (got {z.obstacle_handling.obstacle_blocking_radius})")
    _assert(z.field_border_safe_distance == 625.0,
            f"field_border_safe = 625mm (got {z.field_border_safe_distance})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
def test_priority_groups():
    """Test that tactics have priority groups (H2)."""
    print("\n[TEST] Priority groups in tactics")
    configs = load_contract()
    t121 = configs.tactics["t121_4v4_scaled"]
    _assert(len(t121.priority_groups) >= 3,
            f"t121 has >=3 priority groups (got {len(t121.priority_groups)})")
    t13 = configs.tactics["t13_4v4_scaled"]
    _assert(len(t13.priority_groups) >= 2,
            f"t13 has >=2 priority groups (got {len(t13.priority_groups)})")


def test_strategy_transitions_real_values():
    """Test strategy transitions use real CFG values, not placeholder defaults."""
    print("\n[TEST] Strategy transition values match real CFG")
    configs = load_contract()
    t121_state = configs.strategy.tactics[0]
    _assert(t121_state.tactic == "t121_4v4_scaled", "first tactic is t121")
    tr = t121_state.transitions[0]
    _assert(tr.conditions[0].ball_x_threshold == 500.0,
            f"ball_x_threshold = 500 (got {tr.conditions[0].ball_x_threshold})")
    _assert(tr.conditions[0].time_ball_ahead_ge == 280,
            f"time_ball_ahead_ge = 280 (got {tr.conditions[0].time_ball_ahead_ge})")
    _assert(tr.conditions[0].num_field_players_ge == 3,
            f"num_field_players_ge = 3 (got {tr.conditions[0].num_field_players_ge})")

    t13_state = configs.strategy.tactics[1]
    _assert(len(t13_state.transitions) >= 2,
            f"t13 has >=2 transitions (got {len(t13_state.transitions)})")
    _assert(t13_state.transitions[0].conditions[0].num_field_players_le == 2,
            "t13 first transition: num_field_players_le = 2")


def test_set_plays():
    """Test strategy set plays are populated."""
    print("\n[TEST] Strategy set plays")
    configs = load_contract()
    s = configs.strategy
    _assert(len(s.own_kickoffs) >= 1, f"own_kickoffs has entries (got {len(s.own_kickoffs)})")
    _assert(s.own_kickoffs[0].type == "attackKickOff_scaled",
            f"own kickoff = attackKickOff_scaled (got {s.own_kickoffs[0].type})")
    _assert(len(s.opponent_kickoffs) >= 1, "opponent_kickoffs has entries")
    _assert(len(s.own_penalty_kicks) >= 1, "own_penalty_kicks has entries")
    _assert(len(s.own_free_kicks) >= 1, "own_free_kicks has entries")


def test_skill_behavior_control():
    """Test skill behavior control gating (H2)."""
    print("\n[TEST] Skill behavior control gating")
    configs = load_contract()
    sbc = configs.skill_behavior_control
    _assert("HandleGameState" in sbc.options, "HandleGameState in options")
    _assert("HandlePenaltyKick" in sbc.options, "HandlePenaltyKick in options")
    _assert(len(sbc.playing_options) >= 1,
            f"playing_options non-empty (got {len(sbc.playing_options)})")
    _assert("HandleGoalkeeperCatchBall" in sbc.playing_options,
            "HandleGoalkeeperCatchBall in playing_options")


def test_real_cfg_loading():
    """Test loading from real CFG files (H1/H3)."""
    print("\n[TEST] Real CFG loading")
    import os
    cfg_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "SabanaHerons2026-ScaledDimensions",
        "SabanaHerons2026-ScaledDimensions", "Config"
    )
    if not os.path.isdir(cfg_root):
        print("  SKIP: Real CFG root not found (run from project root)")
        return

    configs = load_contract(cfg_root=cfg_root)
    _assert(configs.loaded_from == "cfg_files",
            f"loaded_from = cfg_files (got {configs.loaded_from})")
    _assert(configs.field.x_opponent_ground == 2813.0,
            "field parsed from real CFG")
    _assert(len(configs.kick_info.kicks) >= 10,
            f"kickInfo parsed >= 10 kicks (got {len(configs.kick_info.kicks)})")
    _assert(configs.walk.obstacle_avoidance_distance == 480.0,
            f"walk parsed from real CFG (got {configs.walk.obstacle_avoidance_distance})")
    _assert("HandleGameState" in configs.skill_behavior_control.options,
            "skill_bc parsed from real CFG")
    _assert(configs.strategy_behavior_control.strategy == "s4v4_scaled",
            "strategy_bc parsed from real CFG")
    _assert(configs.behavior.jump_radius == 600.0,
            "behavior parsed from real CFG")

    # Verify it's NOT just defaults
    t121 = configs.tactics["t121_4v4_scaled"]
    _assert(len(t121.priority_groups) >= 3,
            "priority groups parsed from real CFG")

    # Verify fail-hard on bad path
    try:
        load_contract(cfg_root="/nonexistent")
        _assert(False, "should have raised FileNotFoundError")
    except FileNotFoundError:
        _assert(True, "fail-hard on missing cfg_root")


def test_sim_dt_cognition_aligned():
    """Test SIM.dt is aligned with Cognition cycle (H7)."""
    print("\n[TEST] SIM.dt aligned with Cognition")
    from config import SIM, CONTRACT
    _assert(abs(SIM.dt - 0.033) < 0.001,
            f"SIM.dt ~ 0.033s (got {SIM.dt})")
    _assert(abs(CONTRACT.timing.cognition_cycle_ms - 33.0) < 1.0,
            f"Cognition cycle = 33ms (got {CONTRACT.timing.cognition_cycle_ms})")
    # dt in seconds should match cognition_cycle_ms / 1000
    _assert(abs(SIM.dt - CONTRACT.timing.cognition_cycle_ms / 1000.0) < 0.001,
            "SIM.dt consistent with cognition_cycle_ms")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_parser_basic,
        test_parser_comments,
        test_contract_loader,
        test_field_4v4_scaled,
        test_tactics_correct,
        test_strategy_fsm,
        test_handoff_ready_exists,
        test_takeover_active_exists,
        test_bridge_schema_macrostates,
        test_no_local_physics_as_truth,
        test_temporal_budget,
        test_physical_params,
        test_decision_params,
        test_home_poses_lookup,
        test_field_symbols_completeness,
        test_zweikampf_subsystems,
        test_priority_groups,
        test_strategy_transitions_real_values,
        test_set_plays,
        test_skill_behavior_control,
        test_real_cfg_loading,
        test_sim_dt_cognition_aligned,
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
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
