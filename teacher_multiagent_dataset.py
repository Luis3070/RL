"""Multi-agent BC dataset collection over the team production scenarios.

Pattern A multi-backend orchestrator (per .docs/10 §14.21 + §14 sketch):
  * Spawn N PyBHBackend instances (one per RL field player; GK stays BHuman C++).
  * Each backend reads its own player's BackendObservation from the shared-memory
    bridge; only the primary backend launches SimRobot and triggers world resets.
  * Per step: collect per-agent obs → assign_roles → EMA coordination targets →
    smash_or_pass for striker → team_teacher_payload per agent → record + execute.

Output mirrors the single-agent format (samples.npz, steps.csv, episodes.csv,
summary.json) plus two new arrays per §11.B.4.4:
  * role_ids[N] (int8)            0=striker / 1=open_support / 2=off_ball_support
  * coordination_target_xy[N, 2]  the (EMA-smoothed) target the teacher anchored on

Runs (interactive conda env; `conda run` corrupts SimRobot):

  source /home/luis/anaconda3/etc/profile.d/conda.sh && conda activate RLNAO \\
    && cd /home/luis/GitHub/RL && PYTHONPATH=/home/luis/GitHub/RL \\
       RL_MA_TEACHER_STAGE=6 python teacher_multiagent_dataset.py
  # Then re-run with RL_MA_TEACHER_STAGE=7 for the stage-7 scenarios.

Stage 6 (2v0 + BH GK) and Stage 7 (3v0 / 3v3 + BH GK) are run separately so the
backend pool size (N RL field players) is fixed per invocation. The dataset is
written into a single timestamped run directory by default; per-stage subdirs
can be combined later with `merge_production_datasets.py`.

Wed 06-10 status: skeleton wired against multiagent_teacher.team_teacher_payload
and gate.GateDecision.obs_features_team. Integration-test loop runs Wed AM.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_EARLY_STAGE_ID = int(os.getenv("RL_MA_TEACHER_STAGE", "6"))
_EARLY_RL_FIELD_AGENTS_BY_STAGE = {6: 2, 7: 3, 8: 3}
if _EARLY_STAGE_ID in _EARLY_RL_FIELD_AGENTS_BY_STAGE:
    _early_active = tuple(range(1, _EARLY_RL_FIELD_AGENTS_BY_STAGE[_EARLY_STAGE_ID] + 1))
    os.environ.setdefault("PYBH_RL_OVERRIDE_TEAM", "24")
    os.environ.setdefault("PYBH_RL_ACTIVE_PLAYERS", ",".join(str(p) for p in _early_active))

from action import SKILL_ORDER, SkillType, StructuredSkillAction
from config import ENV, ROOT_DIR, SIMROBOT_BIN
from curriculum import get_stage_scene_path, get_stage_spec
from environment import RealSimRobotEnv
from gate import GateConfig, GateDecision, SkillGate
from multiagent_teacher import (
    EMA_ALPHA_TRIANGLE,
    EMA_ALPHA_OPEN_LANE,
    CONTINGENT_HOLD,
    TeamCoordinationState,
    advance_pass_target_age,
    assign_roles,
    build_agent_id_map,
    build_team_gate_decision,
    defender_block_armed,
    defender_clear_armed,
    defender_cover_target,
    defense_support_target,
    compute_open_lane_target,
    compute_triangle_target,
    ema_step,
    record_pass_target,
    smash_or_pass,
    team_teacher_payload,
    teammate_slots_from_obs,
    update_team_phase,
)
from production_curriculum import ProductionScenario, team_defense_scenarios, team_production_scenarios
from pybh_backend import BackendObservation, PyBHBackend, WorldPlayerSpec
from visible_parallel_runtime import VisibleParallelRuntime

# ---------------------------------------------------------------------------
# Run config — env-overridable for reproducible overnight runs.
# ---------------------------------------------------------------------------

STAGE_ID = int(os.getenv("RL_MA_TEACHER_STAGE", "6"))                # 6, 7, or 8
BACKEND_MODE = os.getenv("RL_MA_TEACHER_BACKEND_MODE", "simrobot_visible")
SIMROBOT_PROFILE = os.getenv("RL_MA_TEACHER_SIMROBOT_PROFILE", "3d_visible")
SIMROBOT_LAUNCH = os.getenv("RL_MA_TEACHER_SIMROBOT_LAUNCH", "1") == "1"
_DEFAULT_VISIBLE_PARALLEL_SLOTS = "0" if STAGE_ID == 8 else "1"
VISIBLE_PARALLEL_SLOTS = os.getenv("RL_MA_TEACHER_VISIBLE_PARALLEL_SLOTS", _DEFAULT_VISIBLE_PARALLEL_SLOTS) == "1"
# Per-slot parallel collection renders each robot in its own 1v0 scene with the other
# players added as dummy bodies (_ensure_stage_obstacle_bodies). The multi-robot
# RLMAStage*_3D scene that get_stage_scene_path returns is the WRONG scene for this
# mode: its pre-placed robots are not produced by the per-slot dummy path and — verified
# 2026-06-17 — the robot detector yields ZERO obstacle detections under it (defender
# perceives 0 opponents, all roles obstacle_count=0). The dedicated 1v0 scene the dummy
# path was built for detects opponents in ~100% of frames. So parallel slots must use
# RL1v0_3D_visible regardless of stage; only single-scene (Pattern A) uses the stage scene.
_PARALLEL_SLOT_SCENE_NAME = "RL1v0_3D_visible"
_DEFAULT_VISIBLE_SCENE_NAME = (
    _PARALLEL_SLOT_SCENE_NAME if VISIBLE_PARALLEL_SLOTS
    else get_stage_scene_path(_EARLY_STAGE_ID, "3d_visible").stem
)
VISIBLE_SCENE_NAME = os.getenv("RL_MA_TEACHER_VISIBLE_SCENE_NAME", _DEFAULT_VISIBLE_SCENE_NAME)
VISIBLE_DISPLAY_BASE = int(os.getenv("RL_MA_TEACHER_DISPLAY_BASE", "180"))
EPISODES_PER_SCENARIO = int(os.getenv("RL_MA_TEACHER_EPISODES", "4"))
MAX_STEPS_PER_EPISODE = int(os.getenv("RL_MA_TEACHER_MAX_STEPS", "400"))
MAX_ATTEMPT_MULTIPLIER = int(os.getenv("RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER", "2"))
STEP_RETRIES = int(os.getenv("RL_MA_TEACHER_STEP_RETRIES", "3"))
SEED_BASE = int(os.getenv("RL_MA_TEACHER_SEED_BASE", "880000"))
SCENARIO_FILTER = tuple(
    item.strip() for item in os.getenv("RL_MA_TEACHER_SCENARIOS", "").split(",") if item.strip()
)
OUTPUT_ROOT = Path(os.getenv(
    "RL_MA_TEACHER_OUTPUT_ROOT",
    str(ROOT_DIR / "runs" / "teacher_multiagent_dataset"),
))
STEP_DT_MS = float(os.getenv("RL_MA_TEACHER_STEP_DT_MS", "33.0"))
DRY_RUN = os.getenv("RL_MA_TEACHER_DRY_RUN", "0") == "1"

# Opponent-perception diagnostic (option 2): per-step raw ObstacleModel counts per role,
# aggregated and dumped at exit. Distinguishes "vision sees nothing" (obstacle_count=0)
# from "robots seen but opponents misclassified" (obstacle_count>0, opponent_count=0).
OBSTACLE_DEBUG = os.getenv("RL_MA_TEACHER_OBSTACLE_DEBUG", "0") == "1"
OBSTACLE_DEBUG_PATH = os.getenv("RL_MA_TEACHER_OBSTACLE_DEBUG_PATH", "/tmp/obstacle_debug.json")
_OBSTACLE_DEBUG_ROWS: list[dict] = []

# ---------------------------------------------------------------------------
# MULTI-AGENT MEASUREMENT HARNESS (eval mode). When RL_MA_EVAL_POLICY=<ckpt> is set,
# the collector drives ALL agents with that policy (gate-aware argmax + env's role-aware
# residual->absolute converter) instead of the teacher, keeps EVERY episode (no sanitizer),
# and logs per-step cooperation records. Reuses the STABLE shared-sim (Pattern-A) orchestration
# — the same path that collected 34k rows — so it sidesteps the unusable parallel-slots eval.
# ---------------------------------------------------------------------------
_EVAL_POLICY_PATH = os.getenv("RL_MA_EVAL_POLICY", "").strip()
_EVAL_POLICY = None
_EVAL_ROWS: list[dict] = []
_MATCH_SUMMARY_ENABLED = os.getenv("RL_MA_MATCH_SUMMARY", "0") == "1"
# Inference-side upgrade overrides (deploy as a thin wrapper next to the ONNX; flag-gated
# so we can A/B them in the harness). #3 block-priority: when both block & mark are gate-
# allowed and the BALL is the nearer in-front threat, intercept the lane (block) instead of
# tracking the man (mark). #4 collision-avoid: when a close opponent sits in the walk/dribble
# heading, deflect the target laterally away (reduces pushing fouls / falls).
_EVAL_BLOCK_PRIORITY = os.getenv("RL_EVAL_BLOCK_PRIORITY", "0") == "1"
_EVAL_COLLISION_AVOID = os.getenv("RL_EVAL_COLLISION_AVOID", "0") == "1"
_EVAL_COLLISION_AVOID_MM = float(os.getenv("RL_EVAL_COLLISION_AVOID_MM", "500.0"))


def _record_match_row(
    *,
    scenario_name: str,
    attempt: int,
    step: int,
    aid: int,
    role: str,
    skill: str,
    team_ball: tuple[float, float, bool],
    obs,
) -> None:
    if not (_EVAL_POLICY is not None or _MATCH_SUMMARY_ENABLED):
        return
    bx = float(team_ball[0])
    by = float(team_ball[1])
    rx = float(obs.robot_x)
    ry = float(obs.robot_y)
    _EVAL_ROWS.append({
        "scn": scenario_name,
        "att": int(attempt),
        "step": int(step),
        "aid": int(aid),
        "role": role,
        "skill": skill,
        "bx": bx,
        "by": by,
        "rx": rx,
        "ry": ry,
        "ball_dist": float(math.hypot(rx - bx, ry - by)),
        "opp_dist": float(obs.nearest_opponent_dist),
        "opp_seen": int(float(obs.nearest_opponent_dist) < 4500.0),
        "ball_seen": float(obs.ball_seen_percentage),
    })


def _load_eval_policy(path: str):
    import torch, torch.nn.functional as F
    sd = torch.load(path, map_location="cpu")["policy_state_dict"]

    class _P(torch.nn.Module):
        def __init__(s):
            super().__init__()
            for k in ("trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
                      "skill_head.weight", "skill_head.bias", "param_head.weight", "param_head.bias"):
                s.register_buffer(k.replace(".", "_"), sd[k])

        def forward(s, o):
            h = F.elu(F.linear(o, s.trunk_0_weight, s.trunk_0_bias))
            h = F.elu(F.linear(h, s.trunk_2_weight, s.trunk_2_bias))
            return (F.linear(h, s.skill_head_weight, s.skill_head_bias),
                    torch.tanh(F.linear(h, s.param_head_weight, s.param_head_bias)))
    return _P().eval()


def _dump_obstacle_debug() -> None:
    if not (OBSTACLE_DEBUG and _OBSTACLE_DEBUG_ROWS):
        return
    import json
    agg: dict = {}
    for role in sorted({r["role"] for r in _OBSTACLE_DEBUG_ROWS}):
        rows = [r for r in _OBSTACLE_DEBUG_ROWS if r["role"] == role]
        n = len(rows)
        agg[role] = {
            "n": n,
            "mean_obstacle_count": round(sum(r["oc"] for r in rows) / n, 3),
            "mean_teammate_count": round(sum(r["otc"] for r in rows) / n, 3),
            "mean_opponent_count": round(sum(r["ooc"] for r in rows) / n, 3),
            "mean_uncertain_count": round(sum(r["ouc"] for r in rows) / n, 3),
            "frac_any_obstacle_seen": round(sum(1 for r in rows if r["oc"] > 0) / n, 3),
            "frac_opponent_seen": round(sum(1 for r in rows if r["ooc"] > 0) / n, 3),
            "frac_nearest_obstacle_lt2000mm": round(sum(1 for r in rows if 0 < r["nod"] < 2000) / n, 3),
            "opp_angle_abs_mean_rad": round(sum(abs(r.get("noppa", 0.0)) for r in rows) / n, 4),
            "opp_angle_nonzero_frac": round(sum(1 for r in rows if abs(r.get("noppa", 0.0)) > 1e-4) / n, 3),
            "opp_front_seen_frac": round(sum(1 for r in rows if 0 < r.get("noppfd", 9e9) < 1500) / n, 3),
        }
    with open(OBSTACLE_DEBUG_PATH, "w") as fh:
        json.dump(agg, fh, indent=2)
    print(f"[obstacle-debug] wrote {OBSTACLE_DEBUG_PATH}: {agg}", flush=True)


if OBSTACLE_DEBUG:
    import atexit
    atexit.register(_dump_obstacle_debug)

# §14.18 sanitizer filters
MIN_KEEP_STEPS_OPEN_SUPPORT = int(os.getenv("RL_MA_TEACHER_MIN_OPEN_HOLD_STEPS", "20"))
MIN_KEEP_STEPS_OFF_BALL_SUPPORT = int(os.getenv("RL_MA_TEACHER_MIN_TRIANGLE_HOLD_STEPS", "30"))
OPEN_SUPPORT_HOLD_RADIUS_MM = float(os.getenv("RL_MA_TEACHER_OPEN_HOLD_RADIUS_MM", "600.0"))
OFF_BALL_HOLD_RADIUS_MM = float(os.getenv("RL_MA_TEACHER_OFF_BALL_HOLD_RADIUS_MM", "600.0"))
MAX_ROLE_SWAPS_PER_EPISODE = int(os.getenv("RL_MA_TEACHER_MAX_ROLE_SWAPS", "10"))
PASS_INTERCEPTED_DROP = os.getenv("RL_MA_TEACHER_PASS_INTERCEPTED_DROP", "1") == "1"
# Teammate-collision drop: agents live in separate sims, so a one-frame crossing of
# virtual positions is not a physical collision. Only sustained overlap counts.
COLLISION_RADIUS_MM = float(os.getenv("RL_MA_TEACHER_COLLISION_RADIUS_MM", "350.0"))
COLLISION_SUSTAIN_STEPS = int(os.getenv("RL_MA_TEACHER_COLLISION_SUSTAIN_STEPS", "8"))
# Max distance between the requested episode ball position and any agent's sim ball
# right after reset; beyond this the visible-slot ball settle failed and the episode
# would be degenerate (teacher chases a misplaced ball).
BALL_RESET_TOLERANCE_MM = float(os.getenv("RL_MA_TEACHER_BALL_RESET_TOLERANCE_MM", "800.0"))
# Team-comm cadence emulation: on real NAOs BHuman sends team packets at
# minSendInterval=1000ms, so teammate poses and the team ball arrive stale and
# step-wise, never per-frame fresh. The injected team state refreshes at this
# interval and holds (with honest growing ages) in between, so the policy trains
# on the signal texture it will see in production. 600ms default keeps slots
# mostly inside the 1000ms freshness filters while still exercising staleness.
TEAM_COMM_INTERVAL_MS = float(os.getenv("RL_MA_TEACHER_TEAM_COMM_INTERVAL_MS", "600.0"))
# TeammatesBallModel-style inactivity invalidation for the injected team ball.
TEAM_BALL_VALID_MAX_MS = float(os.getenv("RL_MA_TEACHER_TEAM_BALL_VALID_MAX_MS", "2000.0"))

# Stage → number of RL field players (GK is always BH, not counted).
_RL_FIELD_AGENTS_BY_STAGE = {6: 2, 7: 3, 8: 3}


# ---------------------------------------------------------------------------
# Dry smoke — no SimRobot launch, validates mechanical train-readiness.
# ---------------------------------------------------------------------------


def _synthetic_team_observation(
    player_number: int,
    *,
    robot_x: float,
    robot_y: float,
    team_positions: Mapping[int, tuple[float, float]],
    ball_xy: tuple[float, float],
) -> BackendObservation:
    teammate_poses = tuple(
        (float(x), float(y), 0.0)
        for aid, (x, y) in team_positions.items()
        if int(aid) != int(player_number)
    )
    teammate_ages = tuple(100.0 for _ in teammate_poses)
    ball_x, ball_y = ball_xy
    return BackendObservation(
        ball_x=float(ball_x),
        ball_y=float(ball_y),
        robot_x=float(robot_x),
        robot_y=float(robot_y),
        robot_theta=0.0,
        simrobot_robot_x=float(robot_x),
        simrobot_robot_y=float(robot_y),
        simrobot_robot_theta=0.0,
        frame=1,
        obs_ready=True,
        ball_rel_x=float(ball_x - robot_x),
        ball_rel_y=float(ball_y - robot_y),
        ball_end_rel_x=float(ball_x - robot_x),
        ball_end_rel_y=float(ball_y - robot_y),
        ball_vel_x=0.0,
        ball_vel_y=0.0,
        time_since_ball_seen=0.0,
        time_since_ball_disappeared=0.0,
        ball_seen_percentage=100.0,
        ball_consistent_with_game_state=True,
        natural_time_since_ball_seen=0.0,
        can_score_now=False,
        shot_quality_no_obstacles=0.2,
        shot_opening_with_obstacles=0.1,
        pass_options_count=1.0,
        nearest_teammate_dist=900.0,
        nearest_opponent_dist=5000.0,
        nearest_uncertain_obstacle_dist=0.0,
        nearest_teammate_front_dist=1200.0,
        nearest_opponent_front_dist=5000.0,
        nearest_uncertain_front_dist=0.0,
        teammate_poses=teammate_poses,
        teammate_ages_ms=teammate_ages,
        teammate_activities=tuple(0 for _ in teammate_poses),
        global_team_ball_x_mm=float(ball_x),
        global_team_ball_y_mm=float(ball_y),
        global_team_ball_age_ms=0.0,
        global_team_ball_valid=True,
        goal_opponent_x_mm=4500.0,
        goal_opponent_y_mm=0.0,
    )


def _dry_run_smoke() -> int:
    """Mechanical readiness check that does not start SimRobot.

    Covers the pieces that must be sane before launching the expensive collector:
    stage scenarios, role assignment, agent-id slots, EMA targets, 47-dim encoding,
    role mask contract, and team_teacher_payload shape.
    """
    import torch
    from train import gate_skill_mask_from_obs

    if STAGE_ID not in _RL_FIELD_AGENTS_BY_STAGE:
        raise ValueError(f"RL_MA_TEACHER_STAGE={STAGE_ID} not supported for dry-run")

    n_rl_field = _RL_FIELD_AGENTS_BY_STAGE[STAGE_ID]
    active = _active_player_numbers(n_rl_field)
    scenario_pool = team_defense_scenarios(1) if STAGE_ID == 8 else team_production_scenarios(1)
    scenarios = [s for s in scenario_pool if s.stage_id == STAGE_ID]
    if SCENARIO_FILTER:
        wanted = set(SCENARIO_FILTER)
        scenarios = [s for s in scenarios if s.name in wanted]
        missing = sorted(wanted - {s.name for s in scenarios})
        if missing:
            raise ValueError(f"RL_MA_TEACHER_SCENARIOS unknown for stage {STAGE_ID}: {missing}")
    if not scenarios:
        raise ValueError(f"No scenarios selected for stage {STAGE_ID}")
    rng = np.random.default_rng(SEED_BASE)
    for scenario in scenarios:
        world = scenario.world_fn(rng)
        expected_teammates = n_rl_field  # RL peers excluding player 1 plus BH GK.
        if len(world.teammates) != expected_teammates:
            raise AssertionError(
                f"{scenario.name}: expected {expected_teammates} teammates, got {len(world.teammates)}"
            )

    ball_xy = (1800.0, 0.0)
    team_positions: dict[int, tuple[float, float]] = {
        1: (1200.0, 0.0),
        2: (1900.0, 1000.0),
    }
    if n_rl_field >= 3:
        team_positions[3] = (900.0, -1000.0)

    obs_by_agent = {
        aid: _synthetic_team_observation(
            aid,
            robot_x=xy[0],
            robot_y=xy[1],
            team_positions=team_positions,
            ball_xy=ball_xy,
        )
        for aid, xy in team_positions.items()
    }
    team_state = TeamCoordinationState()
    phase = update_team_phase(team_state, (ball_xy[0], ball_xy[1], True))
    roles = assign_roles(obs_by_agent, (ball_xy[0], ball_xy[1], True), team_state.roles, phase=phase)
    if "striker" not in roles.values() or "open_support" not in roles.values():
        raise AssertionError(f"dry-run roles missing striker/open_support: {roles}")
    if n_rl_field >= 3 and "off_ball_support" not in roles.values():
        raise AssertionError(f"dry-run roles missing off_ball_support: {roles}")

    striker_id = next(aid for aid, role in roles.items() if role == "striker")
    open_id = next(aid for aid, role in roles.items() if role == "open_support")
    raw_open = (
        defense_support_target(
            obs_by_agent[open_id],
            ball_xy,
            team_positions[striker_id],
            team_positions[off_id],
        )
        if phase == "defense" and off_id is not None
        else compute_open_lane_target(obs_by_agent[open_id], ball_xy, team_positions[striker_id])
    )
    team_state.open_lane_ema = ema_step(team_state.open_lane_ema, raw_open, EMA_ALPHA_OPEN_LANE)
    off_id = next((aid for aid, role in roles.items() if role == "off_ball_support"), None)
    if off_id is not None:
        if phase == "defense":
            raw_triangle, side = defender_cover_target(obs_by_agent[off_id], ball_xy)
            team_state.defender_side_by_agent[off_id] = side
        else:
            raw_triangle = compute_triangle_target(
                obs_by_agent[off_id],
                ball_xy,
                team_positions[striker_id],
                team_positions[open_id],
            )
        team_state.triangle_ema = ema_step(team_state.triangle_ema, raw_triangle, EMA_ALPHA_TRIANGLE)

    encoder_env = _make_encoder_env()
    rows = 0
    try:
        for aid in active:
            role = roles[aid]
            obs = obs_by_agent[aid]
            agent_id_map = build_agent_id_map(
                aid,
                team_positions,
                {agent_id: 0.0 for agent_id in team_positions},
            )
            teammate_slots = teammate_slots_from_obs(obs, agent_id_map)
            base_decision = GateDecision(
                shoot_armed=True,
                dribble_armed=True,
                shoot_arm_progress=1.0,
                allowed_skills=tuple(SKILL_ORDER),
            )
            pass_armed = False
            if role == "striker":
                proposal, _target = smash_or_pass(obs, teammate_slots, None, 0.0)
                pass_armed = proposal == "pass"
            team_decision = build_team_gate_decision(
                base_decision,
                role=role,
                pass_armed=pass_armed,
            )
            coord = (
                team_state.open_lane_ema if role == "open_support"
                else team_state.triangle_ema if role == "off_ball_support" and team_state.triangle_ema is not None
                else ball_xy
            )
            encoded = _encode_for_agent(encoder_env, obs, team_decision)
            if encoded.shape != (47,):
                raise AssertionError(f"encoded obs shape for player {aid}: {encoded.shape}")
            mask = gate_skill_mask_from_obs(torch.as_tensor(encoded, dtype=torch.float32).reshape(1, -1))
            if role == "open_support":
                for skill_name in ("shoot", "dribble", "pass"):
                    if float(mask[0, SKILL_ORDER.index(skill_name)].item()) != 0.0:
                        raise AssertionError(f"open_support mask allowed {skill_name}")
            payload, absolute, residual = team_teacher_payload(
                obs,
                STAGE_ID,
                None,
                gate_decision=team_decision,
                role=role,
                coordination_target=coord,
                teammate_slots=teammate_slots,
                last_pass_target=None,
                last_pass_target_age_ms=0.0,
                phase=phase,
                team_ball_valid=True,
                face_ball_xy=ball_xy,
            )
            if "role_id" not in payload or residual.skill_name not in SKILL_ORDER:
                raise AssertionError(f"bad payload for player {aid}: {payload}")
            rows += 1
    finally:
        encoder_env.close()

    print(
        "[ma-teacher dry-run] PASS "
        f"stage={STAGE_ID} scenarios={len(scenarios)} active={active} roles={roles} rows={rows}"
    )
    print(
        "[ma-teacher dry-run] PASS obs_size=47 role_mask_contract=true "
        "team_teacher_payload=true ready_for_simrobot_collection=true"
    )
    return 0


# ---------------------------------------------------------------------------
# Backend pool — Pattern A (multi-backend, one per RL field player).
# ---------------------------------------------------------------------------


def _use_visible_parallel_slots() -> bool:
    return (
        BACKEND_MODE == "simrobot_visible"
        and SIMROBOT_LAUNCH
        and VISIBLE_PARALLEL_SLOTS
        and SIMROBOT_PROFILE.startswith("3d_visible")
    )


def _make_backend(player_number: int, *, is_primary: bool, parallel_slots: bool = False) -> PyBHBackend:
    """One PyBHBackend per RL field player. Only the primary launches SimRobot;
    secondaries attach to the running shared-memory bridge."""
    return PyBHBackend(
        mode=BACKEND_MODE,
        player_number=player_number,
        simrobot_profile=SIMROBOT_PROFILE,
        simrobot_scene_path=None if parallel_slots else get_stage_scene_path(STAGE_ID, SIMROBOT_PROFILE),
        simrobot_launch=(is_primary and SIMROBOT_LAUNCH and not parallel_slots),
        simrobot_bin=SIMROBOT_BIN,
    )


def _make_encoder_env() -> RealSimRobotEnv:
    """One RealSimRobotEnv used ONLY as an obs encoder. Never stepped; the env's
    `_gate_decision` is mutated per agent per step before calling `_encode_obs`
    so the resulting 47-dim vector carries the agent's team-augmented gate bits."""
    stage = get_stage_spec(STAGE_ID)
    backend_cfg = replace(
        ENV.backend,
        mode=BACKEND_MODE,
        simrobot_profile=SIMROBOT_PROFILE,
        simrobot_scene_path=get_stage_scene_path(STAGE_ID, SIMROBOT_PROFILE),
        simrobot_launch=False,                                          # encoder never launches
    )
    env_cfg = replace(
        ENV,
        stage_id=STAGE_ID,
        max_episode_steps=MAX_STEPS_PER_EPISODE,
        backend=backend_cfg,
        active_skill_names=stage.active_skill_names,
        enable_handoff_override=False,
    )
    return RealSimRobotEnv(env_cfg=env_cfg)


# ---------------------------------------------------------------------------
# Per-episode multi-backend orchestration.
# ---------------------------------------------------------------------------


def _world_player_pose(world_spec, player_number: int) -> tuple[float, float, float]:
    if int(player_number) == 1:
        return float(world_spec.robot_x), float(world_spec.robot_y), float(world_spec.robot_theta)
    for spec in world_spec.teammates:
        if int(spec.number) == int(player_number):
            return float(spec.x), float(spec.y), float(spec.theta)
    raise KeyError(f"player {player_number} not present in WorldSpec")


def _world_spec_to_reset_args(world_spec, *, controlled_player_number: int = 1) -> dict:
    """Translate a WorldSpec from production_curriculum into PyBHBackend.reset kwargs.

    For Pattern A, controlled_player_number is 1 and this is the legacy single-scene
    reset. For visible parallel slots, each slot has a different controlled player,
    so that player's pose becomes `robot_pos` and all other same-team bodies are
    passed as teammates/dummies in that slot.
    """
    robot_x, robot_y, robot_theta = _world_player_pose(world_spec, controlled_player_number)
    teammates = []
    if int(controlled_player_number) != 1:
        teammates.append(WorldPlayerSpec(
            number=1,
            x=float(world_spec.robot_x),
            y=float(world_spec.robot_y),
            theta=float(world_spec.robot_theta),
        ))
    for spec in world_spec.teammates:
        if int(spec.number) == int(controlled_player_number):
            continue
        teammates.append(WorldPlayerSpec(
            number=int(spec.number), x=float(spec.x), y=float(spec.y), theta=float(spec.theta),
        ))
    opponents = []
    for spec in world_spec.opponents:
        opponents.append(WorldPlayerSpec(
            number=int(spec.number), x=float(spec.x), y=float(spec.y), theta=float(spec.theta),
        ))
    return dict(
        ball_pos=[float(world_spec.ball_x), float(world_spec.ball_y)],
        robot_pos=[float(robot_x), float(robot_y)],
        robot_theta=float(robot_theta),
        teammates=tuple(teammates),
        opponents=tuple(opponents),
    )


def _active_player_numbers(n_rl_field: int) -> tuple[int, ...]:
    """Players 1..N are RL field. Player N+1 is the BH GK (never gets a backend).

    Stage 6: RL = (1, 2), GK = 3.
    Stage 7: RL = (1, 2, 3), GK = 4.
    """
    return tuple(range(1, n_rl_field + 1))


def _setup_pybh_active_players(active: tuple[int, ...]) -> None:
    """Set the env var BHuman reads to decide which players the RL bridge controls.

    Players NOT in this list are driven by BHuman's own behaviour tree (the GK).
    """
    os.environ["PYBH_RL_OVERRIDE_TEAM"] = "24"
    os.environ["PYBH_RL_ACTIVE_PLAYERS"] = ",".join(str(p) for p in active)


def _ensure_visible_shm_name() -> str | None:
    """Use a fresh RLSharedState segment for visible collection runs.

    The default /pybh_rl_shared_state_v3 can survive crashes with valid
    magic/version and stale request counters.  A unique name per run avoids
    request-id jumps such as 1070137xxx in both single-SimRobot and parallel
    visible modes.
    """
    if BACKEND_MODE != "simrobot_visible" or not SIMROBOT_LAUNCH:
        return os.environ.get("PYBH_RL_SHM_NAME")
    shm_name = os.environ.get("PYBH_RL_SHM_NAME")
    if not shm_name:
        shm_name = f"/pybh_rl_ma_teacher_stage{STAGE_ID}_{os.getpid()}"
        os.environ["PYBH_RL_SHM_NAME"] = shm_name
    return shm_name


def _write_dataset_artifacts(
    output_dir: Path,
    *,
    active: tuple[int, ...],
    parallel_slots: bool,
    all_obs: list[np.ndarray],
    all_actions: list[np.ndarray],
    all_stages: list[int],
    all_episode_ids: list[int],
    all_step_ids: list[int],
    all_weights: list[float],
    all_role_ids: list[int],
    all_coords: list[tuple[float, float]],
    all_label_consistent: list[int],
    scenario_summaries: list[dict[str, object]],
    total_episodes: int,
    partial: bool,
) -> None:
    samples_path = output_dir / ("samples.partial.npz" if partial else "samples.npz")
    np.savez_compressed(
        samples_path,
        observations=np.asarray(all_obs, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.float32),
        stages=np.asarray(all_stages, dtype=np.int64),
        episode_ids=np.asarray(all_episode_ids, dtype=np.int64),
        step_ids=np.asarray(all_step_ids, dtype=np.int64),
        weights=np.asarray(all_weights, dtype=np.float32),
        role_ids=np.asarray(all_role_ids, dtype=np.int8),
        coordination_target_xy=np.asarray(all_coords, dtype=np.float32),
        label_consistent=np.asarray(all_label_consistent, dtype=np.int8),
        skill_order=np.asarray(list(SKILL_ORDER)),
    )

    summary = {
        "stage_id": STAGE_ID,
        "active_rl_players": list(active),
        "visible_parallel_slots": bool(parallel_slots),
        "visible_scene_name": VISIBLE_SCENE_NAME if parallel_slots else None,
        "pybh_rl_shm_name": os.environ.get("PYBH_RL_SHM_NAME"),
        "episodes_per_scenario": EPISODES_PER_SCENARIO,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "obs_size": int(np.asarray(all_obs[0]).shape[-1]) if all_obs else 0,
        "total_rows": int(len(all_obs)),
        "total_episodes": int(total_episodes),
        "scenarios": scenario_summaries,
        "partial": bool(partial),
        "positive_weight_gate_label_consistent": True,
        "positive_weight_role_mask_consistent": True,
        "filters": {
            "min_open_hold_steps": MIN_KEEP_STEPS_OPEN_SUPPORT,
            "min_off_ball_hold_steps": MIN_KEEP_STEPS_OFF_BALL_SUPPORT,
            "open_support_hold_radius_mm": OPEN_SUPPORT_HOLD_RADIUS_MM,
            "off_ball_hold_radius_mm": OFF_BALL_HOLD_RADIUS_MM,
            "max_role_swaps_per_episode": MAX_ROLE_SWAPS_PER_EPISODE,
            "pass_intercepted_drop": PASS_INTERCEPTED_DROP,
            "collision_radius_mm": COLLISION_RADIUS_MM,
            "collision_sustain_steps": COLLISION_SUSTAIN_STEPS,
            "team_state_injection": True,
        },
        "gate_config": asdict(GateConfig.from_env()),
    }
    summary_path = output_dir / ("summary.partial.json" if partial else "summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[ma-teacher] wrote {samples_path} ({len(all_obs)} rows, partial={int(partial)})")
    print(f"[ma-teacher] wrote {summary_path}")


def _coordinated_reset(
    backends: dict[int, PyBHBackend],
    primary: int,
    world_spec,
) -> dict[int, BackendObservation]:
    """Primary backend places all robots + ball. Secondaries step once with `stand`
    to acquire their own first observation from the shared-memory bridge.
    Returns {player_number: initial BackendObservation}.
    """
    if _use_visible_parallel_slots():
        return {
            player_number: backend.reset(**_world_spec_to_reset_args(
                world_spec, controlled_player_number=player_number,
            ))
            for player_number, backend in backends.items()
        }

    primary_obs = backends[primary].reset(**_world_spec_to_reset_args(
        world_spec, controlled_player_number=primary,
    ))
    obs_by_agent: dict[int, BackendObservation] = {primary: primary_obs}
    for player_number, backend in backends.items():
        if player_number == primary:
            continue
        obs_by_agent[player_number] = backend.step(
            StructuredSkillAction(skill_type=SkillType.stand),
        )
    return obs_by_agent


def _step_backends(
    backends: dict[int, PyBHBackend],
    actions_by_agent: dict[int, StructuredSkillAction],
) -> dict[int, BackendObservation]:
    """Apply per-agent action via that agent's backend, return new obs per agent."""
    new_obs: dict[int, BackendObservation] = {}
    for player_number, backend in backends.items():
        action = actions_by_agent.get(player_number)
        if action is None:
            action = StructuredSkillAction(skill_type=SkillType.stand)
        last_exc: Exception | None = None
        for attempt in range(max(1, STEP_RETRIES)):
            try:
                new_obs[player_number] = backend.step(action)
                break
            except RuntimeError as exc:
                last_exc = exc
                message = str(exc)
                retryable = (
                    "stale visible frames" in message
                    or "rl_get_obs timed out" in message
                    or "visible frame rewind" in message
                )
                if not retryable or attempt >= max(1, STEP_RETRIES) - 1:
                    raise
                print(
                    f"[ma-teacher] step retry player={player_number} "
                    f"attempt={attempt + 1}/{STEP_RETRIES}: {exc}",
                    flush=True,
                )
                time.sleep(0.05 * (attempt + 1))
        if player_number not in new_obs and last_exc is not None:
            raise last_exc
    return new_obs


def _team_ball_from_obs(obs_by_agent: dict[int, BackendObservation]) -> tuple[float, float, bool]:
    """Use the primary's (player 1) reported global team ball — every agent's
    BackendObservation carries the same TeammatesBallModel-derived fields."""
    primary_obs = next(iter(obs_by_agent.values()))
    return (
        float(primary_obs.global_team_ball_x_mm),
        float(primary_obs.global_team_ball_y_mm),
        bool(primary_obs.global_team_ball_valid),
    )


# Activity codes mirror RLTeammateActivity (SkillBehaviorControl.cpp:encodeTeammateActivity)
# and environment.py:_teammate_activity_is_engaging.
_ACTIVITY_IDLE = 0
_ACTIVITY_GOING_TO_BALL = 1
_ACTIVITY_DRIBBLING = 2
_ACTIVITY_KICKING = 3


def _activity_for(skill_name: str | None, role: str | None) -> int:
    if skill_name in ("shoot", "pass"):
        return _ACTIVITY_KICKING
    if skill_name == "dribble":
        return _ACTIVITY_DRIBBLING
    if skill_name == "walk" and role == "striker":
        return _ACTIVITY_GOING_TO_BALL
    return _ACTIVITY_IDLE


def _ball_authority(obs_by_agent: dict[int, BackendObservation]) -> int:
    """The agent whose own-sim ball is the team's authoritative ball.

    Fixed per episode: each slot sim has its own ball, but the scenario intends
    exactly one to be played — the one nearest its agent at reset. Keeping the
    authority fixed keeps the team ball on the sim where the ball actually moves
    and keeps role election anchored to a single coherent ball.
    """
    # Exported field-frame ball — perception-independent, valid right after reset
    # (supports spawn facing away from the ball, so their ball_rel is unseeded).
    return min(
        sorted(obs_by_agent),
        key=lambda aid: math.hypot(
            float(obs_by_agent[aid].ball_x) - float(obs_by_agent[aid].robot_x),
            float(obs_by_agent[aid].ball_y) - float(obs_by_agent[aid].robot_y),
        ),
    )


def _refresh_comm_snapshot(
    obs_by_agent: dict[int, BackendObservation],
    *,
    ball_authority: int,
    prev_skill_name: dict[int, str | None],
    committed_roles: dict[int, str],
    step: int,
) -> dict:
    """Capture the team-comm packet content at this step (BHuman send event).

    The team ball comes from the authority's PERCEPTION ball (estimated_ball,
    field frame) and its natural age — not the GT-corrected export — so the
    policy obs carries the noise and dropouts the TeammatesBallModel shows on
    real NAOs."""
    auth = obs_by_agent[ball_authority]
    return {
        "step": step,
        "poses": {
            tid: (float(o.robot_x), float(o.robot_y), float(o.robot_theta))
            for tid, o in obs_by_agent.items()
        },
        "activities": {
            tid: _activity_for(prev_skill_name.get(tid), committed_roles.get(tid))
            for tid in obs_by_agent
        },
        "ball": (float(auth.estimated_ball_x), float(auth.estimated_ball_y)),
        "ball_age_ms": float(auth.natural_time_since_ball_seen),
    }


def _inject_team_state(
    obs_by_agent: dict[int, BackendObservation],
    *,
    snapshot: dict,
    step: int,
    static_teammates: tuple[tuple[int, float, float, float], ...],
) -> dict[int, BackendObservation]:
    """Overwrite per-sim team wire fields with comm-cadence-realistic team state.

    Each agent runs alone in its own RL1v0 slot, so the wire's team fields
    (teammate poses via team comm, TeammatesBallModel) are empty/invalid there.
    The orchestrator is the only process that sees every sim, so it acts as the
    team comm layer — but at BHuman's real packet cadence: values come from the
    last comm snapshot and age between refreshes, never per-frame fresh. Without
    this injection obs[26:38] is dead and smash_or_pass never has a pass
    candidate; without the cadence the policy trains on an ideal team signal it
    will never see in production.
    """
    elapsed_ms = max(0.0, (step - int(snapshot["step"])) * STEP_DT_MS)
    ball_x, ball_y = snapshot["ball"]
    ball_age = float(snapshot["ball_age_ms"]) + elapsed_ms
    ball_valid = ball_age <= TEAM_BALL_VALID_MAX_MS

    injected: dict[int, BackendObservation] = {}
    for aid, obs in obs_by_agent.items():
        poses: list[tuple[float, float, float]] = []
        ages: list[float] = []
        acts: list[int] = []
        for tid in sorted(obs_by_agent):
            if tid == aid:
                continue
            poses.append(snapshot["poses"][tid])
            ages.append(elapsed_ms)
            acts.append(int(snapshot["activities"][tid]))
        for _num, sx, sy, stheta in static_teammates:
            poses.append((sx, sy, stheta))
            ages.append(elapsed_ms)
            acts.append(_ACTIVITY_IDLE)
        injected[aid] = replace(
            obs,
            teammate_poses=tuple(poses),
            teammate_ages_ms=tuple(ages),
            teammate_activities=tuple(acts),
            global_team_ball_x_mm=ball_x,
            global_team_ball_y_mm=ball_y,
            global_team_ball_age_ms=ball_age,
            global_team_ball_valid=ball_valid,
        )
    return injected


# ---------------------------------------------------------------------------
# Per-step encoding helpers
# ---------------------------------------------------------------------------


def _encode_for_agent(
    encoder_env: RealSimRobotEnv,
    obs: BackendObservation,
    team_decision,
) -> np.ndarray:
    """Encode the 47-dim obs for one agent using the encoder env's `_encode_obs`.

    We mutate the encoder's `_gate_decision` so `obs_features_team()` returns the
    correct team-augmented bits at indices 41-46.
    """
    encoder_env._gate_decision = team_decision
    return np.asarray(encoder_env._encode_obs(obs), dtype=np.float32)


def _payload_to_action(payload: Mapping[str, float]) -> StructuredSkillAction:
    """Convert the dataset payload back into a StructuredSkillAction for the backend.

    The payload carries the RESIDUAL action — the backend's anchor + residual converter
    is the symmetric inverse used by env.step at training time. For dataset rollout we
    let the backend execute the ABSOLUTE action directly (returned by team_teacher_payload
    as the second tuple element). This helper exists for the policy-action path; for the
    teacher rollout, prefer the absolute action handed back by team_teacher_payload.
    """
    return StructuredSkillAction(
        skill_type=int(round(float(payload.get("policy_skill_raw", payload.get("skill_type", 0))))),
        target_x=float(payload.get("target_x", 0.0)),
        target_y=float(payload.get("target_y", 0.0)),
        target_theta=float(payload.get("target_theta", 0.0)),
        pass_target=int(round(float(payload.get("pass_target", 0)))),
    )


_GATE_MASK_FN = None


def _team_label_allowed(skill_name: str, encoded) -> bool:
    """Single-source Design-A label filter: keep a BC label iff the REAL train-time
    gate (train.gate_skill_mask_from_obs) allows that skill at the baked 47-dim obs.

    This replaces the former hand-mirror, which drifted from the gate — it never
    allowed the defender's `mark` and only allowed `block` via dribble_armed, so the
    corrected teacher's block/mark labels would have been silently dropped here (the
    gate-mark-leak class of bug). Calling the gate directly makes drift impossible:
    block is armed from the encoded obs[3] (ball in front) and mark from obs[21]
    (opponent in front), exactly as train-time MAPPO recomputes it.
    """
    global _GATE_MASK_FN
    if _GATE_MASK_FN is None:
        # Lazy import: importing train at module load triggers its chdir side-effect.
        from train import gate_skill_mask_from_obs as _fn
        _GATE_MASK_FN = _fn
    import torch
    mask = _GATE_MASK_FN(torch.as_tensor(encoded, dtype=torch.float32).reshape(1, -1))
    return float(mask[0, SKILL_ORDER.index(skill_name)].item()) != 0.0


# ---------------------------------------------------------------------------
# Per-scenario episode loop
# ---------------------------------------------------------------------------


def _episode_kept_v2(
    scenario: ProductionScenario,
    metrics: Mapping[str, float],
    role_step_counts: Mapping[str, int],
    role_hold_steps: Mapping[str, int],
    role_swap_count: int,
    pass_intercepted: bool,
    teammate_collision: bool,
    agent_out_of_bounds: bool = False,
) -> tuple[bool, str]:
    """§14.18 sanitizer + §11.B.4.1 per-role success filter.

    Drop rules (any one drops the episode):
      - teammate-teammate collision detected
      - role-thrash > MAX_ROLE_SWAPS_PER_EPISODE
      - pass-intercepted within 1.5s if PASS_INTERCEPTED_DROP set
      - per-role hold-steps thresholds: open ≥ MIN_KEEP_STEPS_OPEN_SUPPORT,
        off-ball ≥ MIN_KEEP_STEPS_OFF_BALL_SUPPORT (only when that role was active)
    """
    if agent_out_of_bounds:
        return False, "agent_out_of_bounds"
    if teammate_collision:
        return False, "teammate_collision"
    if role_swap_count > MAX_ROLE_SWAPS_PER_EPISODE:
        return False, f"role_thrash({role_swap_count})"
    if PASS_INTERCEPTED_DROP and pass_intercepted:
        return False, "pass_intercepted"
    # v3 (doc-35): the geometric hold-step minimum is a v2 premise — it assumes a
    # support that doesn't sit near its slot long enough is a bad episode. The
    # contingent-hold teacher deliberately REPOSITIONS supports unless they're a live
    # pass option, so it drives hold-steps low ON PURPOSE (and short give-and-go /
    # contrast scenarios end before a deep support can accrue 30 steps). Applying the
    # filter under CONTINGENT_HOLD dropped 100% of contrastive episodes and ~80% of
    # 3v3. Skip it in the contingent regime — the contingent teacher + the audit
    # (contrast balance, skill distribution) are the quality gate now. v2 behaviour
    # (filter active, env-tunable) is preserved when CONTINGENT_HOLD is off.
    if not CONTINGENT_HOLD:
        open_steps = int(role_step_counts.get("open_support", 0))
        off_steps = int(role_step_counts.get("off_ball_support", 0))
        if open_steps > 0:
            if int(role_hold_steps.get("open_support", 0)) < MIN_KEEP_STEPS_OPEN_SUPPORT:
                return False, "open_support_hold_too_short"
        if off_steps > 0:
            if int(role_hold_steps.get("off_ball_support", 0)) < MIN_KEEP_STEPS_OFF_BALL_SUPPORT:
                return False, "off_ball_hold_too_short"
    return True, "kept"


def _collect_team_scenario(
    backends: dict[int, PyBHBackend],
    encoder_env: RealSimRobotEnv,
    scenario: ProductionScenario,
    output_dir: Path,
    seed_base: int,
    episode_offset: int,
) -> dict[str, object]:
    """Collect one ProductionScenario across `scenario.episodes` kept episodes."""
    stage_id = scenario.stage_id
    primary = next(iter(backends))  # lowest player_number is primary
    agent_ids = tuple(sorted(backends.keys()))

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    stages: list[int] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    weights: list[float] = []
    role_ids: list[int] = []
    coord_xys: list[tuple[float, float]] = []
    label_consistent: list[int] = []

    role_step_counts: Counter[str] = Counter()
    drop_reasons: Counter[str] = Counter()
    collision_contexts: list[dict] = []
    skipped_masked_label_rows = 0
    rng = np.random.default_rng(seed_base)
    kept_episodes = 0
    dropped_episodes = 0
    attempted_episodes = 0
    max_attempts = max(scenario.episodes, scenario.episodes * MAX_ATTEMPT_MULTIPLIER)
    started = time.perf_counter()

    # One SkillGate per backend (gate state is per-agent across steps).
    gates: dict[int, SkillGate] = {aid: SkillGate(GateConfig.from_env()) for aid in agent_ids}

    for attempt in range(max_attempts):
        if kept_episodes >= scenario.episodes:
            break
        attempted_episodes += 1
        seed = seed_base + attempt
        rng_ep = np.random.default_rng(seed)
        world_spec = scenario.world_fn(rng_ep)
        # Let the obs encoder see this episode's GT opponents so that, when
        # RL_TRAIN_GT_OPPONENT_OBS=1, _encode_obs fills obs[18]/obs[21] from ground
        # truth. Without this the encoder env (never reset with a world_spec) has no
        # opponents and the off-ball defender's block/mark labels get gate-dropped even
        # under GT — the long-standing multi-agent dataset deadness. No effect with the
        # flag off. Defense scenarios used for GT collection are static, so spawn==current.
        encoder_env._current_world_spec = world_spec

        # Reset all backends to this episode's world. Primary places robots.
        for gate in gates.values():
            gate.reset()
        team_state = TeamCoordinationState()
        obs_by_agent = _coordinated_reset(backends, primary, world_spec)

        # Non-RL same-team bodies (BH GK) from the WorldSpec — injected as static
        # teammates so slot order matches build_agent_id_map's view.
        static_teammates = tuple(
            (int(spec.number), float(spec.x), float(spec.y), float(spec.theta))
            for spec in world_spec.teammates
            if int(spec.number) not in agent_ids
        )
        ball_authority = _ball_authority(obs_by_agent)

        # Reset sanity: the visible-slot ball settle occasionally fails and the sim
        # ball lands metres from the requested spot (drift lines in pybh logs). The
        # teacher then chases a misplaced ball for the whole episode — a degenerate
        # walk-only episode that can survive the filters and poison the dataset.
        # Discard the attempt immediately instead of wasting the step budget.
        # Uses the exported field-frame ball (o.ball_x/ball_y) — valid for every
        # agent right after reset. ball_rel is perception-seeded and is garbage for
        # supports that spawn facing away from the ball.
        ball_drift_bad = False
        reset_ball_obs = obs_by_agent.items() if _use_visible_parallel_slots() else ((primary, obs_by_agent[primary]),)
        for aid, o in reset_ball_obs:
            drift = math.hypot(
                float(o.ball_x) - float(world_spec.ball_x),
                float(o.ball_y) - float(world_spec.ball_y),
            )
            if drift > BALL_RESET_TOLERANCE_MM:
                ball_drift_bad = True
                break
        if ball_drift_bad:
            dropped_episodes += 1
            drop_reasons["reset_ball_drift"] += 1
            continue

        # Per-episode buffers
        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        ep_step_ids: list[int] = []
        ep_weights: list[float] = []
        ep_role_ids: list[int] = []
        ep_coord_xys: list[tuple[float, float]] = []
        ep_label_consistent: list[int] = []
        ep_role_step_counts: Counter[str] = Counter()
        ep_role_hold_steps: Counter[str] = Counter()
        prev_skill_name: dict[int, str | None] = {aid: None for aid in agent_ids}
        prev_role: dict[int, str | None] = {aid: None for aid in agent_ids}
        ep_role_swap_count = 0
        ep_pass_intercepted = False
        ep_teammate_collision = False
        ep_collision_streak = 0
        # Per-ROBOT side latch (not per role): if the two supports swap roles
        # mid-episode, a role-keyed latch sends each to the other's side and they
        # cross through the corridor (2026-06-10 probe: support-vs-support
        # collisions on the same flank). Spawn side is the BHuman-faithful sticky
        # formation side.
        spawn_side: dict[int, float] = {
            aid: (1.0 if float(o.robot_y) >= 0.0 else -1.0)
            for aid, o in obs_by_agent.items()
        }
        ep_oob_streak = 0
        ep_agent_out_of_bounds = False
        ep_backend_error = False
        comm_snapshot: dict | None = None

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            # 0. Cross-sim team comm injection at BHuman packet cadence — must
            # precede role assignment, teammate slots, and encoding so they all
            # see the same (cadence-realistic) team state.
            if (
                comm_snapshot is None
                or (step - int(comm_snapshot["step"])) * STEP_DT_MS >= TEAM_COMM_INTERVAL_MS
            ):
                comm_snapshot = _refresh_comm_snapshot(
                    obs_by_agent,
                    ball_authority=ball_authority,
                    prev_skill_name=prev_skill_name,
                    committed_roles=dict(team_state.roles.current_roles),
                    step=step,
                )
            obs_by_agent = _inject_team_state(
                obs_by_agent,
                snapshot=comm_snapshot,
                step=step,
                static_teammates=static_teammates,
            )

            # 1. Team-aware state
            agent_positions = {
                aid: (float(o.robot_x), float(o.robot_y)) for aid, o in obs_by_agent.items()
            }
            agent_pose_ages = {aid: 0.0 for aid in obs_by_agent}     # self-pose is fresh
            for num, sx, sy, _stheta in static_teammates:
                agent_positions[num] = (sx, sy)
                agent_pose_ages[num] = 0.0
            team_ball = _team_ball_from_obs(obs_by_agent)
            phase = update_team_phase(team_state, team_ball)
            roles = assign_roles(obs_by_agent, team_ball, team_state.roles, phase=phase)

            # 2. Coordination targets (EMA-smoothed)
            striker_id = next((aid for aid, r in roles.items() if r == "striker"), None)
            open_id = next((aid for aid, r in roles.items() if r == "open_support"), None)
            off_id = next((aid for aid, r in roles.items() if r == "off_ball_support"), None)

            if off_id is not None and striker_id is not None and phase == "defense":
                raw_tri, side = defender_cover_target(
                    obs_by_agent[off_id],
                    (team_ball[0], team_ball[1]),
                    current_side=team_state.defender_side_by_agent.get(off_id),
                )
                team_state.defender_side_by_agent[off_id] = side
                team_state.triangle_ema = (
                    raw_tri if team_state.triangle_ema is None
                    else (
                        EMA_ALPHA_TRIANGLE * raw_tri[0] + (1 - EMA_ALPHA_TRIANGLE) * team_state.triangle_ema[0],
                        EMA_ALPHA_TRIANGLE * raw_tri[1] + (1 - EMA_ALPHA_TRIANGLE) * team_state.triangle_ema[1],
                    )
                )
            if open_id is not None and striker_id is not None:
                raw_open = (
                    defense_support_target(
                        obs_by_agent[open_id],
                        (team_ball[0], team_ball[1]),
                        agent_positions[striker_id],
                        agent_positions[off_id],
                        forced_side=spawn_side.get(open_id),
                    )
                    if phase == "defense" and off_id is not None
                    else compute_open_lane_target(
                        obs_by_agent[open_id],
                        (team_ball[0], team_ball[1]),
                        agent_positions[striker_id],
                        forced_side=spawn_side.get(open_id),
                    )
                )
                team_state.open_lane_ema = (
                    raw_open if team_state.open_lane_ema is None
                    else (
                        EMA_ALPHA_OPEN_LANE * raw_open[0] + (1 - EMA_ALPHA_OPEN_LANE) * team_state.open_lane_ema[0],
                        EMA_ALPHA_OPEN_LANE * raw_open[1] + (1 - EMA_ALPHA_OPEN_LANE) * team_state.open_lane_ema[1],
                    )
                )
            if off_id is not None and striker_id is not None and open_id is not None and phase != "defense":
                raw_tri = compute_triangle_target(
                    obs_by_agent[off_id],
                    (team_ball[0], team_ball[1]),
                    agent_positions[striker_id],
                    agent_positions[open_id],
                    forced_side=spawn_side.get(off_id),
                )
                team_state.triangle_ema = (
                    raw_tri if team_state.triangle_ema is None
                    else (
                        EMA_ALPHA_TRIANGLE * raw_tri[0] + (1 - EMA_ALPHA_TRIANGLE) * team_state.triangle_ema[0],
                        EMA_ALPHA_TRIANGLE * raw_tri[1] + (1 - EMA_ALPHA_TRIANGLE) * team_state.triangle_ema[1],
                    )
                )

            # 3. Per-agent payload
            advance_pass_target_age(team_state, STEP_DT_MS)
            actions_for_step: dict[int, StructuredSkillAction] = {}

            for aid in agent_ids:
                role = roles.get(aid, "off_ball_support")
                obs = obs_by_agent[aid]

                if OBSTACLE_DEBUG:
                    _OBSTACLE_DEBUG_ROWS.append({
                        "role": role,
                        "oc": int(obs.obstacle_count),
                        "otc": int(obs.obstacle_teammate_count),
                        "ooc": int(obs.obstacle_opponent_count),
                        "ouc": int(obs.obstacle_uncertain_count),
                        "nod": float(obs.nearest_obstacle_dist),
                        "noppd": float(obs.nearest_opponent_dist),
                        "noppa": float(obs.nearest_opponent_angle),
                        "noppfd": float(obs.nearest_opponent_front_dist),
                    })

                # Role swap tracking
                if prev_role[aid] is not None and prev_role[aid] != role:
                    ep_role_swap_count += 1
                prev_role[aid] = role

                # Per-agent agent_id_map (Block-B-aligned)
                agent_id_map = build_agent_id_map(aid, agent_positions, agent_pose_ages)
                teammate_slots = teammate_slots_from_obs(obs, agent_id_map)

                # Base gate (shoot/dribble armed); team augmentation comes next
                base_decision = gates[aid].step(obs)

                # Coordination target for this role
                coord = (
                    team_state.open_lane_ema if role == "open_support" and team_state.open_lane_ema is not None
                    else team_state.triangle_ema if role == "off_ball_support" and team_state.triangle_ema is not None
                    else (team_ball[0], team_ball[1])
                )
                if coord is None:
                    coord = agent_positions[aid]  # safe fallback

                if role == "off_ball_support" and phase == "defense" and defender_block_armed(
                    obs,
                    team_ball_valid=bool(team_ball[2]),
                    face_ball_xy=(team_ball[0], team_ball[1]),
                    coordination_target=coord,
                ):
                    base_decision = replace(
                        base_decision,
                        dribble_armed=True,
                        shoot_arm_progress=max(float(base_decision.shoot_arm_progress), 1.0),
                    )

                # Compute pass_armed for the striker only. PlayBall.cpp runs
                # smashOrPass whenever the robot is playing the ball, not only when
                # a shot is armed — gating on shoot_armed alone starves pass: at
                # mid-advance shot_open is below the arming threshold (xG is low
                # mid-field) which is exactly where a forward support beats the
                # striker's own rating. Ball control = shoot OR dribble armed.
                pass_armed = False
                if role == "striker" and (base_decision.shoot_armed or base_decision.dribble_armed):
                    skill_proposal, _agent_target = smash_or_pass(
                        obs, teammate_slots,
                        team_state.last_pass_target_by_striker.get(aid),
                        team_state.last_pass_target_age_ms_by_striker.get(aid, 0.0),
                    )
                    pass_armed = skill_proposal == "pass"

                # D4: arm shoot for a deep clearance (no safe outlet). Mirrors the
                # block baking above — bake shoot_armed=True so the baked obs, the
                # train-time mask, and the clearance-shoot label all agree (Design-A).
                # Runtime reachability (live gate.py + C++ encoder) tracked in §Part2 D4.
                if (
                    role == "striker"
                    and not pass_armed
                    and (base_decision.shoot_armed or base_decision.dribble_armed)
                    and defender_clear_armed(obs, teammate_slots)
                ):
                    base_decision = replace(
                        base_decision,
                        shoot_armed=True,
                        shoot_arm_progress=max(float(base_decision.shoot_arm_progress), 1.0),
                    )

                team_decision = build_team_gate_decision(
                    base_decision, role=role, pass_armed=pass_armed,
                )

                # Encode 47-dim obs with team-augmented gate bits
                encoded = _encode_for_agent(encoder_env, obs, team_decision)

                # ---- EVAL MODE: drive this agent with the policy, not the teacher ----
                if _EVAL_POLICY is not None:
                    import torch
                    enc_t = torch.as_tensor(encoded, dtype=torch.float32).reshape(1, -1)
                    with torch.no_grad():
                        sl, pmn = _EVAL_POLICY(enc_t)
                    gmask = _GATE_MASK_FN(enc_t) if callable(_GATE_MASK_FN) else None
                    if gmask is None:
                        from train import gate_skill_mask_from_obs as _g
                        gmask = _g(enc_t)
                    sl = sl + (gmask - 1.0) * 1e9
                    sk = int(sl.argmax(1).item())
                    # #3 BLOCK-PRIORITY: if mark chosen but block is also armed and the ball
                    # is the nearer in-front threat (enc obs[3] ball_rel_x < obs[21] opp_front,
                    # both normalized), intercept the lane instead of tracking the man.
                    if _EVAL_BLOCK_PRIORITY and SKILL_ORDER[sk] == "mark":
                        _bi = SKILL_ORDER.index("block")
                        if float(gmask[0, _bi].item()) > 0.5:
                            _ballf = float(enc_t[0, 3].item()); _oppf = float(enc_t[0, 21].item())
                            if 0.0 < _ballf < _oppf:
                                sk = _bi
                    pr = pmn[0].numpy()
                    resid = StructuredSkillAction(
                        skill_type=sk, target_x=float(pr[0]), target_y=float(pr[1]),
                        target_theta=float(pr[2]),
                        pass_target=int(np.clip(round((float(pr[3]) + 1.0) * 0.5 * 2.0), 0, 2)),
                    )
                    encoder_env._team_role = role
                    encoder_env._team_coordination_target = coord
                    absolute = encoder_env._decode_residual_policy_action(resid, obs=obs)
                    # #4 COLLISION-AVOID: deflect walk/dribble target away from a close opponent.
                    if _EVAL_COLLISION_AVOID and resid.skill_name in ("walk", "dribble"):
                        _dopp = float(obs.nearest_opponent_dist)
                        if 0.0 < _dopp < _EVAL_COLLISION_AVOID_MM:
                            _ang = float(obs.robot_theta) + float(obs.nearest_opponent_angle)
                            _ox = float(obs.robot_x) + _dopp * math.cos(_ang)
                            _oy = float(obs.robot_y) + _dopp * math.sin(_ang)
                            _ax = float(obs.robot_x) - _ox; _ay = float(obs.robot_y) - _oy
                            _n = math.hypot(_ax, _ay) or 1.0
                            _push = (_EVAL_COLLISION_AVOID_MM - _dopp)
                            absolute = replace(
                                absolute,
                                target_x=float(np.clip(absolute.target_x + _ax / _n * _push, -4500.0, 4500.0)),
                                target_y=float(np.clip(absolute.target_y + _ay / _n * _push, -3000.0, 3000.0)),
                            )
                    actions_for_step[aid] = absolute
                    prev_skill_name[aid] = resid.skill_name
                    ep_role_step_counts[role] += 1
                    _record_match_row(
                        scenario_name=scenario.name,
                        attempt=attempt,
                        step=step,
                        aid=aid,
                        role=role,
                        skill=SKILL_ORDER[sk],
                        team_ball=team_ball,
                        obs=obs,
                    )
                    continue
                # ---- end eval mode ----

                # Teacher payload (absolute + residual)
                payload, absolute, residual = team_teacher_payload(
                    obs, stage_id, prev_skill_name[aid],
                    gate_decision=team_decision,
                    role=role,
                    coordination_target=coord,
                    teammate_slots=teammate_slots,
                    last_pass_target=team_state.last_pass_target_by_striker.get(aid),
                    last_pass_target_age_ms=team_state.last_pass_target_age_ms_by_striker.get(aid, 0.0),
                    phase=phase,
                    team_ball_valid=bool(team_ball[2]),
                    face_ball_xy=(team_ball[0], team_ball[1]),
                )

                # If striker committed a pass, register it for hysteresis
                if residual.skill_name == "pass":
                    record_pass_target(team_state, aid, int(residual.pass_target))

                ep_role_step_counts[role] += 1
                _record_match_row(
                    scenario_name=scenario.name,
                    attempt=attempt,
                    step=step,
                    aid=aid,
                    role=role,
                    skill=residual.skill_name,
                    team_ball=team_ball,
                    obs=obs,
                )

                # Hold-step accounting (close to target = "holding")
                dist_to_target = math.hypot(
                    float(obs.robot_x) - float(coord[0]),
                    float(obs.robot_y) - float(coord[1]),
                )
                if role == "open_support" and dist_to_target < OPEN_SUPPORT_HOLD_RADIUS_MM:
                    ep_role_hold_steps["open_support"] += 1
                elif role == "off_ball_support" and dist_to_target < OFF_BALL_HOLD_RADIUS_MM:
                    ep_role_hold_steps["off_ball_support"] += 1

                # Record only labels that the train-time mask can actually select.
                # The teacher may execute a short continuation action for rollout
                # stability; those frames must not become positive BC labels.
                if _team_label_allowed(residual.skill_name, encoded):
                    action_vec = np.asarray([
                        float(residual.skill_type),
                        float(residual.target_x),
                        float(residual.target_y),
                        float(residual.target_theta),
                        float(residual.pass_target),
                    ], dtype=np.float32)

                    ep_obs.append(encoded)
                    ep_actions.append(action_vec)
                    ep_step_ids.append(step)
                    ep_weights.append(1.0)                              # base weight; sanitizer may adjust
                    ep_role_ids.append({"striker": 0, "open_support": 1, "off_ball_support": 2}[role])
                    ep_coord_xys.append((float(coord[0]), float(coord[1])))
                    ep_label_consistent.append(1)                       # Design A positive label
                else:
                    skipped_masked_label_rows += 1

                actions_for_step[aid] = absolute
                prev_skill_name[aid] = residual.skill_name

            # 4. Execute on each backend, collect next obs. A visible SimRobot can
            # stop producing frames after a WalkingEngine assertion/fall. Treat
            # that as a bad episode, not as a fatal dataset-run error.
            try:
                obs_by_agent = _step_backends(backends, actions_for_step)
            except RuntimeError as exc:
                ep_backend_error = True
                drop_reasons["backend_step_error"] += 1
                print(
                    f"[ma-teacher] drop episode scenario={scenario.name} "
                    f"attempt={attempt} step={step}: backend_step_error: {exc}",
                    flush=True,
                )
                break

            # Teammate-collision proxy: agents live in separate sims, so positions can
            # legitimately cross for a frame with no physics involved. Only flag when
            # two agents stay within COLLISION_RADIUS_MM for COLLISION_SUSTAIN_STEPS
            # consecutive steps — otherwise the filter culls exactly the goal-driving
            # episodes (striker crossing the support slot) and poisons the dataset.
            new_positions = {aid: (float(o.robot_x), float(o.robot_y)) for aid, o in obs_by_agent.items()}
            step_colliding = False
            colliding_pair = None
            for a, b in [(a, b) for a in agent_ids for b in agent_ids if a < b]:
                if math.hypot(
                    new_positions[a][0] - new_positions[b][0],
                    new_positions[a][1] - new_positions[b][1],
                ) < COLLISION_RADIUS_MM:
                    step_colliding = True
                    colliding_pair = (a, b)
                    break
            # Fallen/parked guard: a robot that falls in the visible slot gets
            # teleported to the sim parking spot (~±3200, ±3012) outside the
            # field; episodes containing parked robots produce garbage labels and
            # phantom "collisions" between parked bodies.
            oob = any(
                abs(p[0]) > 4600.0 or abs(p[1]) > 3050.0 for p in new_positions.values()
            )
            ep_oob_streak = ep_oob_streak + 1 if oob else 0
            if ep_oob_streak >= 5:
                ep_agent_out_of_bounds = True

            ep_collision_streak = ep_collision_streak + 1 if step_colliding else 0
            if ep_collision_streak >= COLLISION_SUSTAIN_STEPS and not ep_teammate_collision:
                ep_teammate_collision = True
                if len(collision_contexts) < 8 and colliding_pair is not None:
                    a, b = colliding_pair
                    collision_contexts.append({
                        "attempt": attempt,
                        "step": step,
                        "pair_roles": [roles.get(a, "?"), roles.get(b, "?")],
                        "pair_pos": [
                            [round(new_positions[a][0]), round(new_positions[a][1])],
                            [round(new_positions[b][0]), round(new_positions[b][1])],
                        ],
                    })

        if ep_backend_error:
            dropped_episodes += 1
            continue

        # Episode complete — apply v2 sanitizer (EVAL mode keeps every episode; the
        # behavior under collisions/outcomes IS the measurement).
        metrics: dict[str, float] = {}  # TODO Wed AM: extract from primary backend per-episode tracker
        if _EVAL_POLICY is not None or _MATCH_SUMMARY_ENABLED:
            kept, reason = True, "match_summary_keep"
        else:
            kept, reason = _episode_kept_v2(
                scenario, metrics,
                role_step_counts=ep_role_step_counts,
                role_hold_steps=ep_role_hold_steps,
                role_swap_count=ep_role_swap_count,
                pass_intercepted=ep_pass_intercepted,
                teammate_collision=ep_teammate_collision,
                agent_out_of_bounds=ep_agent_out_of_bounds,
            )
        if not kept:
            dropped_episodes += 1
            drop_reasons[reason] += 1
            continue

        # Promote episode rows to global buffers
        ep_id = episode_offset + kept_episodes
        observations.extend(ep_obs)
        actions.extend(ep_actions)
        stages.extend([stage_id] * len(ep_obs))
        episode_ids.extend([ep_id] * len(ep_obs))
        step_ids.extend(ep_step_ids)
        weights.extend(ep_weights)
        role_ids.extend(ep_role_ids)
        coord_xys.extend(ep_coord_xys)
        label_consistent.extend(ep_label_consistent)
        for role, count in ep_role_step_counts.items():
            role_step_counts[role] += count
        kept_episodes += 1

    elapsed = time.perf_counter() - started

    return {
        "name": scenario.name,
        "stage_id": stage_id,
        "kept_episodes": kept_episodes,
        "attempted_episodes": attempted_episodes,
        "dropped_episodes": dropped_episodes,
        "drop_reasons": dict(drop_reasons),
        "collision_contexts": collision_contexts,
        "elapsed_s": elapsed,
        "role_step_counts": dict(role_step_counts),
        "skipped_masked_label_rows": int(skipped_masked_label_rows),
        "observations": observations,
        "actions": actions,
        "stages": stages,
        "episode_ids": episode_ids,
        "step_ids": step_ids,
        "weights": weights,
        "role_ids": role_ids,
        "coordination_target_xy": coord_xys,
        "label_consistent": label_consistent,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    if DRY_RUN:
        return _dry_run_smoke()

    global _EVAL_POLICY
    if _EVAL_POLICY_PATH:
        _EVAL_POLICY = _load_eval_policy(_EVAL_POLICY_PATH)
        # prime the lazy gate fn used in the per-agent eval branch
        _team_label_allowed("walk", np.zeros(47, dtype=np.float32))
        print(f"[ma-eval] MULTI-AGENT MEASUREMENT MODE — policy={_EVAL_POLICY_PATH} (no dataset sanitizer)", flush=True)
    elif _MATCH_SUMMARY_ENABLED:
        print("[ma-eval] MULTI-AGENT MEASUREMENT MODE — deterministic teacher baseline (no dataset sanitizer)", flush=True)

    if STAGE_ID not in _RL_FIELD_AGENTS_BY_STAGE:
        raise ValueError(
            f"RL_MA_TEACHER_STAGE={STAGE_ID} not supported; use 6 (2v0) or 7 (3v0 / 3v3)."
        )
    n_rl_field = _RL_FIELD_AGENTS_BY_STAGE[STAGE_ID]
    active = _active_player_numbers(n_rl_field)
    _setup_pybh_active_players(active)
    shm_name = _ensure_visible_shm_name()
    print(f"[ma-teacher] stage={STAGE_ID} active_rl_players={active} (GK player {n_rl_field + 1} is BH)")
    parallel_slots = _use_visible_parallel_slots()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime: VisibleParallelRuntime | None = None
    if parallel_slots:
        print(
            f"[ma-teacher] visible_parallel_slots=on scene={VISIBLE_SCENE_NAME} "
            f"display_base={VISIBLE_DISPLAY_BASE} shm={shm_name}"
        )
        runtime = VisibleParallelRuntime(
            player_numbers=active,
            shm_name=shm_name,
            display_base=VISIBLE_DISPLAY_BASE,
            scene_name=VISIBLE_SCENE_NAME,
            simrobot_bin=SIMROBOT_BIN,
            simrobot_env={
                "PYBH_RL_OVERRIDE_TEAM": os.environ["PYBH_RL_OVERRIDE_TEAM"],
                "PYBH_RL_ACTIVE_PLAYERS": os.environ["PYBH_RL_ACTIVE_PLAYERS"],
            },
        )
        runtime.start()
    elif shm_name:
        print(f"[ma-teacher] visible_parallel_slots=off shm={shm_name}")

    # Spawn backends — player 1 is primary (launches SimRobot).
    backends: dict[int, PyBHBackend] = {}
    for player_number in active:
        backends[player_number] = _make_backend(
            player_number, is_primary=(player_number == active[0]), parallel_slots=parallel_slots,
        )
    for backend in backends.values():
        backend.start()

    encoder_env = _make_encoder_env()

    # Filter scenarios to those matching STAGE_ID; the rest run in another invocation.
    scenario_pool = team_defense_scenarios(EPISODES_PER_SCENARIO) if STAGE_ID == 8 else team_production_scenarios(EPISODES_PER_SCENARIO)
    scenarios = [s for s in scenario_pool if s.stage_id == STAGE_ID]
    if SCENARIO_FILTER:
        wanted = set(SCENARIO_FILTER)
        scenarios = [s for s in scenarios if s.name in wanted]
        missing = sorted(wanted - {s.name for s in scenarios})
        if missing:
            raise ValueError(f"RL_MA_TEACHER_SCENARIOS unknown for stage {STAGE_ID}: {missing}")
    if not scenarios:
        raise ValueError(f"No scenarios selected for stage {STAGE_ID}")
    print(f"[ma-teacher] {len(scenarios)} scenarios for stage {STAGE_ID}:")
    for s in scenarios:
        print(f"  - {s.name} ({s.episodes} eps, {s.obstacles})")

    output_dir = OUTPUT_ROOT / f"dataset_stage{STAGE_ID}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_stages: list[int] = []
    all_episode_ids: list[int] = []
    all_step_ids: list[int] = []
    all_weights: list[float] = []
    all_role_ids: list[int] = []
    all_coords: list[tuple[float, float]] = []
    all_label_consistent: list[int] = []
    scenario_summaries: list[dict[str, object]] = []
    next_episode_id = 0

    try:
        for scenario in scenarios:
            print(f"[ma-teacher] === {scenario.name} ===")
            stats = _collect_team_scenario(
                backends, encoder_env, scenario, output_dir,
                seed_base=SEED_BASE + next_episode_id, episode_offset=next_episode_id,
            )
            all_obs.extend(stats.pop("observations"))
            all_actions.extend(stats.pop("actions"))
            all_stages.extend(stats.pop("stages"))
            all_episode_ids.extend(stats.pop("episode_ids"))
            all_step_ids.extend(stats.pop("step_ids"))
            all_weights.extend(stats.pop("weights"))
            all_role_ids.extend(stats.pop("role_ids"))
            all_coords.extend(stats.pop("coordination_target_xy"))
            all_label_consistent.extend(stats.pop("label_consistent"))
            next_episode_id += int(stats["kept_episodes"])
            scenario_summaries.append(stats)
            print(
                f"[ma-teacher] {scenario.name}: kept={stats['kept_episodes']}/{stats['attempted_episodes']} "
                f"rows={len(all_obs)} drops={stats['drop_reasons']}"
            )
            _write_dataset_artifacts(
                output_dir,
                active=active,
                parallel_slots=parallel_slots,
                all_obs=all_obs,
                all_actions=all_actions,
                all_stages=all_stages,
                all_episode_ids=all_episode_ids,
                all_step_ids=all_step_ids,
                all_weights=all_weights,
                all_role_ids=all_role_ids,
                all_coords=all_coords,
                all_label_consistent=all_label_consistent,
                scenario_summaries=scenario_summaries,
                total_episodes=next_episode_id,
                partial=True,
            )
    except Exception:
        if all_obs:
            _write_dataset_artifacts(
                output_dir,
                active=active,
                parallel_slots=parallel_slots,
                all_obs=all_obs,
                all_actions=all_actions,
                all_stages=all_stages,
                all_episode_ids=all_episode_ids,
                all_step_ids=all_step_ids,
                all_weights=all_weights,
                all_role_ids=all_role_ids,
                all_coords=all_coords,
                all_label_consistent=all_label_consistent,
                scenario_summaries=scenario_summaries,
                total_episodes=next_episode_id,
                partial=True,
            )
        raise
    finally:
        encoder_env.close()
        for backend in backends.values():
            backend.close()
        if runtime is not None:
            runtime.close()

    _write_dataset_artifacts(
        output_dir,
        active=active,
        parallel_slots=parallel_slots,
        all_obs=all_obs,
        all_actions=all_actions,
        all_stages=all_stages,
        all_episode_ids=all_episode_ids,
        all_step_ids=all_step_ids,
        all_weights=all_weights,
        all_role_ids=all_role_ids,
        all_coords=all_coords,
        all_label_consistent=all_label_consistent,
        scenario_summaries=scenario_summaries,
        total_episodes=next_episode_id,
        partial=False,
    )
    if _EVAL_POLICY is not None or _MATCH_SUMMARY_ENABLED:
        _summarize_eval(output_dir)
    return 0


def _summarize_eval(output_dir: Path) -> None:
    """Cooperation / multi-agent behavior summary for matched policy/teacher runs."""
    import collections, json
    rows = _EVAL_ROWS
    if not rows:
        print("[ma-eval] no rows logged"); return
    # group by episode (scn, att)
    eps = collections.defaultdict(list)
    for r in rows:
        eps[(r["scn"], r["att"])].append(r)
    GOAL_X = 4500.0
    n_ep = len(eps)
    all3 = []           # % steps with all 3 distinct roles present simultaneously
    ballprog = []       # net ball advance toward opp goal (mm), per episode
    scored = conceded = 0
    pass_attempts = 0
    possession_steps = possession_total_steps = 0
    ball_recoveries = 0
    skill_by_role = collections.defaultdict(collections.Counter)
    skill_total = collections.Counter()
    role_steps = collections.Counter()
    opp_seen_steps = ball_seen_acc = total_agent_steps = 0
    _coll_steps = [0, 0]   # [total_steps, colliding_steps]
    for (scn, att), ers in eps.items():
        steps = collections.defaultdict(dict)   # step -> {aid: row}
        for r in ers:
            steps[r["step"]][r["aid"]] = r
            skill_by_role[r["role"]][r["skill"]] += 1
            skill_total[r["skill"]] += 1
            role_steps[r["role"]] += 1
            opp_seen_steps += r["opp_seen"]; ball_seen_acc += r["ball_seen"]; total_agent_steps += 1
            if r["skill"] == "pass":
                pass_attempts += 1
        # role co-occurrence
        co = [len({steps[s][a]["role"] for a in steps[s]}) for s in steps]
        all3.append(100.0 * np.mean([c >= 3 for c in co])) if co else None
        # teammate-collision proxy: any RL pair within 350mm at a step
        for s in steps:
            ps = [(r["rx"], r["ry"]) for r in steps[s].values()]
            close = any(math.hypot(ps[i][0]-ps[j][0], ps[i][1]-ps[j][1]) < 350.0
                        for i in range(len(ps)) for j in range(i+1, len(ps)))
            _coll_steps[0] += 1; _coll_steps[1] += int(close)
        # possession proxy: a controlled agent is either close enough to touch the ball,
        # or executes a ball-control skill while still within a plausible intervention radius.
        prev_control = False
        for s in sorted(steps):
            rs = list(steps[s].values())
            if not rs:
                continue
            min_ball_dist = min(float(r["ball_dist"]) for r in rs)
            active_control = any(
                r["skill"] in ("dribble", "pass", "shoot") and float(r["ball_dist"]) <= 1200.0
                for r in rs
            )
            controlled = min_ball_dist <= 650.0 or active_control
            possession_steps += int(controlled)
            possession_total_steps += 1
            if controlled and not prev_control:
                ball_recoveries += 1
            prev_control = controlled
        # ball progression: use the ball x from first and last step (any agent's team_ball)
        sk = sorted(steps)
        bx0 = next(iter(steps[sk[0]].values()))["bx"]
        bxN = next(iter(steps[sk[-1]].values()))["bx"]
        ballprog.append(bxN - bx0)
        maxbx = max(r["bx"] for r in ers); minbx = min(r["bx"] for r in ers)
        if maxbx > GOAL_X - 100: scored += 1
        if minbx < -GOAL_X + 100: conceded += 1
    print("\n" + "=" * 70)
    print(f"[ma-eval] MULTI-AGENT MATCH COOPERATION SUMMARY  episodes={n_ep}")
    print(f"  role co-occurrence (all 3 roles present): {np.mean(all3):.1f}% of steps")
    print(f"  role-step share: " + "  ".join(f"{k}={100*v/max(1,sum(role_steps.values())):.0f}%" for k, v in role_steps.items()))
    print(f"  ball progression toward opp goal: mean {np.mean(ballprog):+.0f} mm/episode  (>0 = team advances)")
    print(f"  pass attempts (striker->teammate): {pass_attempts}")
    print(f"  possession proxy: {100*possession_steps/max(1,possession_total_steps):.1f}% of episode steps")
    print(f"  ball recoveries proxy: {ball_recoveries/max(1,n_ep):.2f}/episode")
    print(f"  episodes ball reached opp goal (scored): {scored}/{n_ep}   own goal (conceded): {conceded}/{n_ep}")
    print(f"  perception: opp_seen={100*opp_seen_steps/max(1,total_agent_steps):.0f}%  mean_ball_seen={ball_seen_acc/max(1,total_agent_steps):.0f}%")
    print(f"  teammate-collision: {100*_coll_steps[1]/max(1,_coll_steps[0]):.1f}% of steps (any RL pair <350mm)  "
          f"[block_priority={_EVAL_BLOCK_PRIORITY} collision_avoid={_EVAL_COLLISION_AVOID}]")
    tot_skill = max(1, sum(skill_total.values()))
    print("  action distribution: " + "  ".join(f"{k}={100*v/tot_skill:.0f}%" for k, v in skill_total.most_common()))
    for role in ("striker", "open_support", "off_ball_support"):
        c = skill_by_role.get(role)
        if c:
            tot = sum(c.values())
            print(f"  {role:16s} skills: " + "  ".join(f"{k}={100*v/tot:.0f}%" for k, v in c.most_common()))
    out = output_dir / "eval_match_summary.json"
    out.write_text(json.dumps({
        "condition": "policy" if _EVAL_POLICY is not None else "deterministic_teacher",
        "policy_path": _EVAL_POLICY_PATH if _EVAL_POLICY is not None else "",
        "episodes": n_ep,
        "rows": len(rows),
        "role_co_occurrence_pct": float(np.mean(all3)),
        "ball_progression_mm": float(np.mean(ballprog)),
        "pass_attempts": int(pass_attempts),
        "possession_proxy_pct": float(100.0 * possession_steps / max(1, possession_total_steps)),
        "ball_recoveries_per_episode": float(ball_recoveries / max(1, n_ep)),
        "scored": int(scored),
        "conceded": int(conceded),
        "perception_opp_seen_pct": float(100.0 * opp_seen_steps / max(1, total_agent_steps)),
        "perception_ball_seen_mean_pct": float(ball_seen_acc / max(1, total_agent_steps)),
        "teammate_collision_proxy_pct": float(100.0 * _coll_steps[1] / max(1, _coll_steps[0])),
        "action_distribution": dict(skill_total),
        "skill_by_role": {k: dict(v) for k, v in skill_by_role.items()},
    }, indent=2))
    print(f"  wrote {out}")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
