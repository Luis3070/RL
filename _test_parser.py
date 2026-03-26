"""Test parser with defaults and real CFGs."""
import sys
from bhuman_cfg_parser import load_contract

# Test 1: Load defaults
c = load_contract()
print("=== DEFAULTS ===")
print(f"loaded_from: {c.loaded_from}")
print(f"field: {c.field.x_opponent_ground}x{c.field.y_left_sideline}")
print(f"strategy: {c.strategy.name}, {len(c.strategy.tactics)} tactic states")
print(f"tactics: {list(c.tactics.keys())}")
print(f"t121 priority_groups: {len(c.tactics['t121_4v4_scaled'].priority_groups)}")
tr0 = c.strategy.tactics[0].transitions[0]
print(f"strategy t121->t13: ball_x={tr0.conditions[0].ball_x_threshold}, ahead={tr0.conditions[0].time_ball_ahead_ge}")
print(f"set plays: own_kickoffs={len(c.strategy.own_kickoffs)}")
print(f"skill_bc: {len(c.skill_behavior_control.options)} options")
print(f"strategy_bc: {c.strategy_behavior_control.strategy}")
print(f"behavior: jump_radius={c.behavior.jump_radius}")
print(f"kick_at_goal kicks: {c.kick_at_goal.allowed_kicks}")
assert c.loaded_from == "defaults"
print("DEFAULTS OK")

# Test 2: Load from real CFGs
cfg_root = "../SabanaHerons2026-ScaledDimensions/SabanaHerons2026-ScaledDimensions/Config"
c2 = load_contract(cfg_root=cfg_root)
print()
print("=== REAL CFGs ===")
print(f"loaded_from: {c2.loaded_from}")
assert c2.loaded_from == "cfg_files"
print(f"field: {c2.field.x_opponent_ground}x{c2.field.y_left_sideline}")
assert c2.field.x_opponent_ground == 2813.0
t121 = c2.tactics["t121_4v4_scaled"]
print(f"t121 positions: {len(t121.positions)}")
for p in t121.positions:
    print(f"  {p.role_type}: ({p.pose.x}, {p.pose.y})")
assert len(t121.positions) == 5
assert len(t121.priority_groups) > 0
print(f"t121 priority_groups: {len(t121.priority_groups)}")
tr = c2.strategy.tactics[0].transitions[0]
print(f"t121->t13: ball_x={tr.conditions[0].ball_x_threshold}, ahead={tr.conditions[0].time_ball_ahead_ge}")
assert tr.conditions[0].ball_x_threshold == 500.0
assert tr.conditions[0].time_ball_ahead_ge == 280
print(f"set plays: {[sp.type for sp in c2.strategy.own_kickoffs]}")
assert len(c2.strategy.own_kickoffs) > 0
print(f"kick_info: {len(c2.kick_info.kicks)} kicks, max_vel={c2.kick_info.ball_velocity_max}")
assert len(c2.kick_info.kicks) > 10
print(f"walk: obstacle={c2.walk.obstacle_avoidance_distance}, border={c2.walk.field_border_expansion}")
assert c2.walk.obstacle_avoidance_distance == 480.0
print(f"skill_bc: {c2.skill_behavior_control.options}")
assert "HandleGameState" in c2.skill_behavior_control.options
print(f"strategy_bc: {c2.strategy_behavior_control.strategy}")
assert c2.strategy_behavior_control.strategy == "s4v4_scaled"
print(f"behavior: jump_radius={c2.behavior.jump_radius}")
assert c2.behavior.jump_radius == 600.0
print("REAL CFGs OK")

# Test 3: Fail hard on missing cfg_root
try:
    load_contract(cfg_root="/nonexistent/path")
    print("ERROR: should have raised FileNotFoundError")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"\nFail-hard test OK: {type(e).__name__}")

print("\nALL PARSER TESTS PASSED")
