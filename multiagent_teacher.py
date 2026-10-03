"""Multi-agent heuristic teacher — MapleTeacherPolicy.

Per-role labels for stage 6+. Each agent's call is independent and stateless;
all team-aware state (role assignment, EMA coordination targets, last pass target
decay) lives in the env and arrives via the info packet (§11.B.3.3).

R1-R4-R6 are implemented inline per §14.12. R5 is intentionally absent (teammate-
as-obstacle is solved by Block A nearest_teammate_front_dist + the BHuman walk
planner, not the teacher).

DO NOT add state to this module. DO NOT compute role here. DO NOT smooth targets here.

BHuman mapping (Bhuman-deterministic-agent-map.md):
  R1 assign_roles()     ← determineActiveAgent() / Behavior.cpp:L1065
  R2 smash_or_pass()    ← PlayBall::smashOrPass() / PlayBall.cpp:33-118
  R3 support_target()   ← Forward::rating() + Midfielder::rating() / PositionRoles/
  R4 hysteresis         ← TTRB stabilityOffset + decisionPenalty (embedded in R1/R2/R3)
  R6 proxy functions    ← PassEvaluationProvider + ExpectedGoalsProvider (embedded in R2)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from action import SKILL_ORDER, SkillType, StructuredSkillAction
from config import ACTION, FIELD_X_HALF, FIELD_Y_HALF, GOAL_X
from gate import GateDecision
from pybh_backend import BackendObservation
from teacher_policy import _action_from_anchor, absolute_to_residual_action, stage_aware_teacher_action

# ---------------------------------------------------------------------------
# §14.12.1 — Constants (single source of truth). Tune on Day 5.
# ---------------------------------------------------------------------------

# R1 (Role assignment)
ROLE_HYSTERESIS_FRAMES = 7          # §11.B.3.1; raised 5→7 per §14.24 after role_thrash(14-56) drops in 2026-06-10 long-cap probes
ROLE_MIN_DISTANCE_GAP_MM = 200.0    # candidate must be ≥ 200mm closer than current striker
TEAMMATE_FRESH_MAX_AGE_MS = 1000.0  # poses older than this are treated as stale

# R2 (smashOrPass — port of PlayBall.cpp:33-118)
# own_value above this → shoot directly (no pass auction). The 0.55 default was
# calibrated to the old binary's inflated ExpectedGoals scale; on the rebuilt
# binary armed windows run 0.30-0.58, so 0.55 made the striker pass away every
# armed shot (2026-06-10 stage-7 probe: 22/22 armed rows chose pass). Set to the
# gate's shoot_open_enter (0.30) for a deterministic finisher: armed ⇒ shoot,
# pass stays a mid-drive tool for shaded lanes. Env-tunable per regen run.
SHOOT_THRESHOLD = float(os.getenv("RL_MA_SHOOT_THRESHOLD", "0.55"))
MIN_PASS_DISTANCE_MM = 400.0        # PlayBall.cpp:p.minPassDistance (adapted: tighter)
MAX_PASS_DISTANCE_MM = 4000.0       # PlayBall.cpp:p.maxPassDistance
DECISION_PENALTY_MIN = 0.05         # mapToRange floor (close ball → big penalty)
DECISION_PENALTY_MAX = 0.20         # mapToRange ceiling (far ball → small penalty)
# Pass must beat the striker's own rating by this margin to commit. BHuman compares
# ratings directly, but its own-rating (ExpectedGoals) runs hotter mid-field than the
# rebuilt sim's deflated xG (~0.24-0.37 mid-drive); without a margin the teacher
# passes the instant a forward support edges ahead and the dribble drive vanishes
# from the dataset (2026-06-10 probe: dribble_frac 0.8%). Env-tunable for regen.
PASS_VALUE_MARGIN = float(os.getenv("RL_MA_PASS_VALUE_MARGIN", "0.15"))
BALL_DIST_MIN_PENALTY_MM = 500.0    # ball at this distance → DECISION_PENALTY_MAX
BALL_DIST_MAX_PENALTY_MM = 2500.0   # ball at this distance → DECISION_PENALTY_MIN
LAST_PASS_TARGET_LIFETIME_MS = 1500.0  # window where last_pass_target discounts new choices

# R3 (Support positioning — Tactic templates + Midfielder separation)
SIGMA_TEAM_MM = 800.0               # Midfielder.cpp:60-72 Gaussian σ
STAB_OFFSET_MM = 800.0              # Tactic.stabilityOffset analogue: keep slot if within
HOLD_TOLERANCE_OPEN_MM = 250.0      # §11.B.2.2 OPEN_LANE_HOLD_TOLERANCE_MM
HOLD_TOLERANCE_OFF_BALL_MM = 350.0  # §11.B.2.3 TRIANGLE_HOLD_TOLERANCE_MM
SUPPORT_SLOT_CANDIDATE_GRID = (     # field-frame xy candidates; y mirrored by ball_y_sign
    (+800.0, +1200.0),
    (+1200.0, +800.0),
    (+400.0, +1400.0),
    (-400.0, +1200.0),              # fall-back-deep slot for crowded forward zones
    # Forward receive slots (BHuman Tactic t13/t22 place forwards in the opponent
    # half; capping the grid at x=1200 made the support permanently unpassable —
    # goal_proximity 0.27 never beat the striker's own shoot rating). forward_rating
    # zeroes these when the ball is too far (pass_dist > MAX_PASS_DISTANCE_MM), so
    # they only activate as the ball advances.
    # y kept ≥1200 and x capped at 2800 so the support's slot stays clear of the
    # striker's drive corridor AND its finishing area (ball y ∈ ±450 in band
    # scenarios). 2026-06-10 probes: y=700-1100 lost 3/6 episodes to sustained
    # teammate-collision; adding a (3400,1000) slot lost 4/6 — the slot scorer
    # always picks the deepest slot, parking the support where the striker finishes.
    (+2000.0, +1300.0),
    (+2800.0, +1200.0),
)

# R6 (Pass quality proxies for BHuman's learned estimators)
CAN_SCORE_BOOST = 0.3               # own ball: + 0.3 × can_score_now added to shoot rating
OPPONENT_LANE_CLEARANCE_MM = 800.0  # nearest_opponent_front_dist must exceed this

# v3 (doc-35) contingent hold: a support stands ONLY when it is a live pass option —
# the striker controls the ball and the support's receive lane is open. Otherwise it
# repositions (walk), even if sitting on its slot. This ties "stand" to a REASON, which
# is what the geometric hold lacked; in contrastive worlds (teammate open vs covered) it
# produces DIFFERENT labels (hold vs reposition) on the same ego-slot — the contingency
# the policy needs to learn. Env-tunable; default ON for the v3 teacher.
CONTINGENT_HOLD = os.getenv("RL_MA_CONTINGENT_HOLD", "1") not in ("0", "false", "False", "")
SUPPORT_CONTROL_NEAR_MM = float(os.getenv("RL_MA_SUPPORT_CONTROL_NEAR_MM", "600.0"))   # teammate within this of team ball ⇒ "striker controls"
SUPPORT_LANE_CLEAR_MM = float(os.getenv("RL_MA_SUPPORT_LANE_CLEAR_MM", "700.0"))       # front opponent nearer than this ⇒ receive lane covered

# Exclusions
GK_EXCLUDE_X_MM = -2000.0          # teammate with absolute x < this is treated as GK

# Defensive phase / positioning (§30 handoff). Env-tunable so dataset, rollout,
# and probes can sweep without touching the 47-dim contract.
DEFENSE_PHASE_ENTER_X = float(os.getenv("RL_MA_DEF_PHASE_ENTER_X", "-300.0"))
DEFENSE_PHASE_EXIT_X = float(os.getenv("RL_MA_DEF_PHASE_EXIT_X", "300.0"))
DEFENDER_RADIUS_MM = float(os.getenv("RL_MA_DEF_RADIUS_MM", "1600.0"))
GK_STAND_RADIUS_MM = float(os.getenv("RL_MA_DEF_GK_STAND_RADIUS_MM", "400.0"))
OWN_GOAL_X_MM = -GOAL_X
OWN_GOAL_POST_Y_MM = 800.0
OWN_GOAL_AREA_LIMIT_X_MM = -3900.0
OWN_GOAL_AREA_PUSH_X_MM = -3650.0
OWN_GOAL_AREA_HALF_Y_MM = 1100.0
DEFENSE_SLOT_CANDIDATE_GRID = (
    (-2200.0, +1100.0),
    (-1400.0, +900.0),
    (-600.0, +1300.0),
    (-1000.0, +400.0),
)
DEFENDER_BLOCK_BALL_X_MAX_MM = float(os.getenv("RL_MA_DEF_BLOCK_BALL_X_MAX_MM", "-500.0"))
DEFENDER_BLOCK_BALL_DIST_MM = float(os.getenv("RL_MA_DEF_BLOCK_BALL_DIST_MM", "2200.0"))
DEFENDER_BLOCK_FRONT_OPP_MM = float(os.getenv("RL_MA_DEF_BLOCK_FRONT_OPP_MM", "1300.0"))
DEFENDER_BLOCK_FRONT_BALL_X_MM = float(os.getenv("RL_MA_DEF_BLOCK_FRONT_BALL_X_MM", "1800.0"))
DEFENDER_BLOCK_COVER_DIST_MM = float(os.getenv("RL_MA_DEF_BLOCK_COVER_DIST_MM", "900.0"))
# Mark a runner (not the ball-carrier): opponent in our front cone within this range
# while the ball itself is NOT in front. Threshold matches the train.py gate's
# mark_threat (nearest_opponent_front_dist < 1500mm) so no mark label is masked.
DEFENDER_MARK_FRONT_OPP_MM = float(os.getenv("RL_MA_DEF_MARK_FRONT_OPP_MM", "1500.0"))

# Recovery + defensive clearance (§Part2 D1-D3). When a defender wins the ball it
# must clear to safety or pass out — not keep blocking/holding. D1: physical ball
# control promotes any agent to striker immediately (bypassing role hysteresis).
# D2/D3: a striker deep in our own third prefers a safe up-field pass, else a
# clearance kick (shoot up-field, away from our goal), else drives the ball out.
BALL_CONTROL_RADIUS_MM = float(os.getenv("RL_MA_BALL_CONTROL_RADIUS_MM", "350.0"))
# Fix for the "disconnected defender": if the ball is LOOSE (no teammate controls it)
# and this agent is the closest field player within this radius, promote it to striker
# immediately (bypass the 7-frame hysteresis) so a defender contests winnable loose
# balls in its zone instead of holding cover. 0 disables.
RECOVERY_ENGAGE_RADIUS_MM = float(os.getenv("RL_MA_RECOVERY_ENGAGE_RADIUS_MM", "1100.0"))
DEFENSIVE_CLEAR_X_MM = float(os.getenv("RL_MA_DEF_CLEAR_X_MM", "-1500.0"))
CLEAR_PASS_MIN_UPFIELD_MM = float(os.getenv("RL_MA_DEF_CLEAR_MIN_UPFIELD_MM", "600.0"))
CLEAR_ESCAPE_TARGET_X_MM = float(os.getenv("RL_MA_DEF_CLEAR_ESCAPE_X_MM", "0.0"))
CLEAR_CENTRAL_DANGER_Y_MM = float(os.getenv("RL_MA_DEF_CLEAR_CENTRAL_Y_MM", "1500.0"))

# EMA smoothing for coordination targets (§11.B.3.2). Lives on the env (not the teacher);
# constants exposed here so the orchestrator imports a single source of truth.
# v3a (§Part1 P5): lowered alphas so the target a support is asked to hold stops
# drifting frame-to-frame. With the old 0.3/0.2 the open-lane/triangle centre tracked
# the ball closely enough that a support could never settle inside HOLD_RADIUS — low
# hold by construction. Slower EMA gives a still target to stand on. Env-tunable.
EMA_ALPHA_OPEN_LANE = float(os.getenv("RL_MA_EMA_ALPHA_OPEN_LANE", "0.15"))   # ~7-frame smoothing
EMA_ALPHA_TRIANGLE = float(os.getenv("RL_MA_EMA_ALPHA_TRIANGLE", "0.10"))     # ~10-frame smoothing


# ---------------------------------------------------------------------------
# §14.12 Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TeammateSlot:
    """One teammate as seen from Block B obs indices (sorted by proximity)."""
    slot_idx: int       # 0=nearest, 1=second, 2=third (Block B order)
    agent_id: int       # player number
    x_field: float      # absolute field x (mm)
    y_field: float      # absolute field y (mm)
    is_gk: bool


@dataclass
class TeamRoleHistory:
    """Lives on the env, one per episode. Carries R1 hysteresis state.

    env.reset() must call TeamRoleHistory() to get a fresh instance.
    env.step() calls assign_roles(agent_obs, team_ball, history) each step.
    """
    current_roles: dict[int, str] = field(default_factory=dict)
    # {agent_id: consecutive frames this agent has been proposed-but-not-committed striker}
    candidate_streak: dict[int, int] = field(default_factory=dict)


@dataclass
class TeamCoordinationState:
    """Per-env multi-agent runtime state.

    Owns the EMA-smoothed coordination targets, the per-striker last_pass_target
    decay, and the role hysteresis history. The teacher reads from this struct;
    the orchestrator (dataset script or env step loop) advances it each step.

    Reset semantics: instantiate fresh on env.reset() — all fields default-empty.
    """
    roles: TeamRoleHistory = field(default_factory=TeamRoleHistory)
    open_lane_ema: tuple[float, float] | None = None
    triangle_ema: tuple[float, float] | None = None
    # last_pass_target is the BHuman agent_id (player number), keyed by striker_id.
    # Decays in age each step; stale entries are ignored by smash_or_pass.
    last_pass_target_by_striker: dict[int, int] = field(default_factory=dict)
    last_pass_target_age_ms_by_striker: dict[int, float] = field(default_factory=dict)
    # Per-agent committed support slot for stabilityOffset (R3 / R4 hysteresis).
    support_committed_slot: dict[int, tuple[float, float]] = field(default_factory=dict)
    phase: str = "attack"
    defender_side_by_agent: dict[int, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.roles = TeamRoleHistory()
        self.open_lane_ema = None
        self.triangle_ema = None
        self.last_pass_target_by_striker.clear()
        self.last_pass_target_age_ms_by_striker.clear()
        self.support_committed_slot.clear()
        self.phase = "attack"
        self.defender_side_by_agent.clear()


def update_team_phase(state: TeamCoordinationState, team_ball: tuple[float, float, bool]) -> str:
    """Sticky attack/defense phase from team ball x.

    Uses a dead band around midfield so the phase cannot flicker when the ball is
    contested near x=0. If team ball is invalid, keep the previous phase.
    """
    ball_x, _ball_y, valid = team_ball
    if not valid:
        return state.phase
    if float(ball_x) < DEFENSE_PHASE_ENTER_X:
        state.phase = "defense"
    elif float(ball_x) > DEFENSE_PHASE_EXIT_X:
        state.phase = "attack"
    return state.phase


# ---------------------------------------------------------------------------
# R1 — Role assignment with 5-frame hysteresis (§14.12.2)
# Port of Behavior::determineActiveAgent() — Behavior.cpp:L1065
# ---------------------------------------------------------------------------

def assign_roles(
    agent_obs: dict[int, BackendObservation],
    team_ball: tuple[float, float, bool],
    history: TeamRoleHistory,
    phase: str = "attack",
) -> dict[int, str]:
    """R1: assign striker / open_support / off_ball_support to each RL agent.

    agent_obs:  {agent_id: obs} — only RL field players (GK excluded by caller)
    team_ball:  (x_mm, y_mm, valid) — global team ball estimate from obs[32:35]
    history:    TeamRoleHistory from env (mutated in-place)

    Returns {agent_id: "striker" | "open_support" | "off_ball_support"}.
    """
    agent_ids = sorted(agent_obs.keys())
    n = len(agent_ids)

    if n == 0:
        return {}

    team_ball_x, team_ball_y, team_ball_valid = team_ball

    # 1. Compute each agent's distance to ball.
    def ball_dist(agent_id: int) -> float:
        obs = agent_obs[agent_id]
        if team_ball_valid:
            bx, by = team_ball_x, team_ball_y
        else:
            bx = float(obs.robot_x) + float(obs.ball_rel_x)
            by = float(obs.robot_y) + float(obs.ball_rel_y)
        return math.hypot(float(obs.robot_x) - bx, float(obs.robot_y) - by)

    distances = {aid: ball_dist(aid) for aid in agent_ids}

    # 2. All-stale fallback: use freshest natural ball belief as striker.
    # NOTE: obs.time_since_ball_seen is already in milliseconds (matches env's TIME_NORM_MS=5000).
    # Bug 5 fix: do NOT multiply by 1000 — that would convert ms→µs and fire the fallback every frame.
    all_stale = all(
        float(agent_obs[aid].time_since_ball_seen) > TEAMMATE_FRESH_MAX_AGE_MS
        for aid in agent_ids
    )
    if all_stale and not team_ball_valid:
        freshest = min(agent_ids, key=lambda aid: float(agent_obs[aid].time_since_ball_seen))
        roles = {aid: ("striker" if aid == freshest else "off_ball_support") for aid in agent_ids}
        history.current_roles = roles
        history.candidate_streak.clear()
        return roles

    # 3. Proposed striker = agent with minimum distance (tie-break: lowest id).
    candidate = min(agent_ids, key=lambda aid: (distances[aid], aid))

    # 4. R4 Hysteresis (BHuman: TTRB stabilityOffset 1000ms → we use frame count).
    prev_striker = None
    for aid, role in history.current_roles.items():
        if role == "striker":
            prev_striker = aid
            break

    # D1: physical ball control overrides the TTRB hysteresis. Whoever actually has
    # the ball plays it — without this, an agent that just won a duel/recovery keeps
    # its prior off_ball/defender role for ROLE_HYSTERESIS_FRAMES and blocks/holds
    # instead of clearing or passing. Recovery is exactly the case the hysteresis
    # should NOT delay.
    candidate_has_ball = (
        math.hypot(float(agent_obs[candidate].ball_rel_x), float(agent_obs[candidate].ball_rel_y))
        < BALL_CONTROL_RADIUS_MM
        and float(agent_obs[candidate].time_since_ball_seen) <= TEAMMATE_FRESH_MAX_AGE_MS
    )

    # Loose-ball recovery: nobody controls the ball, and the closest agent (which may
    # be a deep defender) is within reach → promote it now, bypassing hysteresis, so it
    # engages the loose ball instead of holding cover. This closes the "dead zone".
    ball_loose = all(
        math.hypot(float(agent_obs[aid].ball_rel_x), float(agent_obs[aid].ball_rel_y)) >= BALL_CONTROL_RADIUS_MM
        for aid in agent_ids
    )
    candidate_should_recover = (
        RECOVERY_ENGAGE_RADIUS_MM > 0.0
        and ball_loose
        and distances[candidate] <= RECOVERY_ENGAGE_RADIUS_MM
    )

    committed_striker = candidate
    if (
        prev_striker is not None
        and prev_striker in agent_ids
        and prev_striker != candidate
        and not candidate_has_ball
        and not candidate_should_recover
    ):
        gap = distances[prev_striker] - distances[candidate]
        if gap < ROLE_MIN_DISTANCE_GAP_MM:
            # Gap too small — keep previous striker regardless of streak.
            committed_striker = prev_striker
            history.candidate_streak.clear()
        else:
            streak = history.candidate_streak.get(candidate, 0) + 1
            history.candidate_streak[candidate] = streak
            if streak < ROLE_HYSTERESIS_FRAMES:
                committed_striker = prev_striker
            else:
                committed_striker = candidate
                history.candidate_streak.clear()
    else:
        history.candidate_streak.clear()

    # 5. Assign remaining agents as support roles.
    non_strikers = [aid for aid in agent_ids if aid != committed_striker]

    if len(non_strikers) == 0:
        roles = {committed_striker: "striker"}
    elif len(non_strikers) == 1:
        # 2-RL: attack uses open_support; defense uses the remaining field
        # player as the deep defender.
        roles = {
            committed_striker: "striker",
            non_strikers[0]: "off_ball_support" if phase == "defense" else "open_support",
        }
    elif phase == "defense":
        defender = min(non_strikers, key=lambda aid: (float(agent_obs[aid].robot_x), aid))
        roles = {committed_striker: "striker"}
        for aid in non_strikers:
            roles[aid] = "off_ball_support" if aid == defender else "open_support"
    else:
        # 3-RL+: open_support = highest forward rating; rest = off_ball.
        striker_xy = (float(agent_obs[committed_striker].robot_x),
                      float(agent_obs[committed_striker].robot_y))
        if team_ball_valid:
            ball_xy = (team_ball_x, team_ball_y)
        else:
            obs_s = agent_obs[committed_striker]
            ball_xy = (float(obs_s.robot_x) + float(obs_s.ball_rel_x),
                       float(obs_s.robot_y) + float(obs_s.ball_rel_y))

        def fwd_score(aid: int) -> float:
            pos = (float(agent_obs[aid].robot_x), float(agent_obs[aid].robot_y))
            return forward_rating(pos, ball_xy)

        open_support = max(non_strikers, key=fwd_score)
        roles = {committed_striker: "striker"}
        for aid in non_strikers:
            roles[aid] = "open_support" if aid == open_support else "off_ball_support"

    history.current_roles = roles
    return roles


# ---------------------------------------------------------------------------
# R2 — smashOrPass (§14.12.3)
# Port of PlayBall::smashOrPass() — PlayBall.cpp:33-118
# ---------------------------------------------------------------------------

def smash_or_pass(
    striker_obs: BackendObservation,
    teammate_slots: list[TeammateSlot],
    last_pass_target: int | None,
    last_pass_target_age_ms: float,
) -> tuple[str, int]:
    """R2: decide shoot vs pass for the active striker.

    Returns (skill_name, pass_target_agent_id).
    pass_target_agent_id == -1 when skill_name != "pass".

    Bug 1 fix: pass_target is now the BHuman player number (agent_id), matching
    PlayBall.cpp:108 `SkillRequest::Builder::passTo(passTarget->number)`. The env's
    `last_pass_target` state MUST be the agent_id (not the Block-B slot_idx). Slots
    with agent_id < 0 are skipped — the env's mapping is incomplete for that slot
    and we cannot tell BHuman to pass to an unknown player.

    Direct line-by-line correspondence with PlayBall.cpp:
      PlayBall.cpp:39   → decision_penalty (R4 hysteresis on decision)
      PlayBall.cpp:44   → own_value (R6 shoot rating proxy)
      PlayBall.cpp:54-55 → penalise own_value when was passing last frame
      PlayBall.cpp:56-60 → shoot if own_value > threshold
      PlayBall.cpp:64-104 → loop teammates, pick best pass
      PlayBall.cpp:105-108 → commit best pass if found
      PlayBall.cpp:116-117 → fallback shoot
    """
    # PlayBall.cpp:39 — decision penalty: larger when ball is close (commitment zone).
    # Bug 4 fix: np.interp requires monotonically increasing xp. Previous call passed
    # xp=[2500, 500] which silently clamps to fp[0]=0.05, freezing the penalty at MIN
    # for every distance. Fixed by passing increasing xp [500, 2500] and decreasing fp
    # [MAX, MIN] so close ball → DECISION_PENALTY_MAX, far ball → DECISION_PENALTY_MIN.
    d_ball = math.hypot(float(striker_obs.ball_rel_x), float(striker_obs.ball_rel_y))
    decision_penalty = _map_to_range(
        d_ball,
        in_lo=BALL_DIST_MIN_PENALTY_MM, in_hi=BALL_DIST_MAX_PENALTY_MM,   # 500 → 2500
        out_lo=DECISION_PENALTY_MAX, out_hi=DECISION_PENALTY_MIN,         # 0.20 → 0.05
    )

    # PlayBall.cpp:44 — own shoot rating (R6 proxy for ExpectedGoals.getRating(ballPos)).
    own_value = (
        float(striker_obs.shot_opening_with_obstacles)
        + CAN_SCORE_BOOST * float(striker_obs.can_score_now)
    )

    # PlayBall.cpp:54-55 — penalise if last decision was a pass (stability anti-thrash).
    target_active = (
        last_pass_target is not None
        and last_pass_target_age_ms < LAST_PASS_TARGET_LIFETIME_MS
    )
    if target_active:
        own_value -= decision_penalty

    # PlayBall.cpp:56-60 — shoot directly if own rating is high enough.
    if own_value > SHOOT_THRESHOLD:
        return ("shoot", -1)

    # PlayBall.cpp:64-104 — score each teammate as a pass candidate.
    # PASS_VALUE_MARGIN: candidate must beat own rating by a clear margin (see
    # constant rationale) so passing complements rather than replaces the drive.
    best_agent_id = -1
    best_value = own_value + PASS_VALUE_MARGIN
    for slot in teammate_slots:
        if slot.is_gk:
            continue
        # Skip slots where the env did not provide an agent_id (mapping incomplete);
        # we cannot tell BHuman to pass to an unknown player.
        if slot.agent_id < 0:
            continue
        pass_dist = math.hypot(
            slot.x_field - float(striker_obs.robot_x),
            slot.y_field - float(striker_obs.robot_y),
        )
        if pass_dist < MIN_PASS_DISTANCE_MM or pass_dist > MAX_PASS_DISTANCE_MM:
            continue

        pr = _pass_reachable_proxy(striker_obs, slot)
        gp = _goal_proximity_proxy(slot)
        if pr == 0.0 or gp == 0.0:
            continue

        agent_value = pr * gp

        # PlayBall.cpp:94-98 — penalise switching pass target; keyed on agent_id.
        if target_active and last_pass_target != slot.agent_id:
            agent_value -= decision_penalty

        if agent_value > best_value:
            best_value = agent_value
            best_agent_id = slot.agent_id

    # PlayBall.cpp:105-117 — commit.
    if best_agent_id >= 0:
        return ("pass", best_agent_id)
    return ("shoot", -1)


# ---------------------------------------------------------------------------
# R6 — Pass quality proxies (§14.12.6)
# Embedded in R2; not a separate module.
# ---------------------------------------------------------------------------

def _pass_reachable_proxy(striker_obs: BackendObservation, target: TeammateSlot) -> float:
    """Binary proxy for PassEvaluationProvider.getRating(passLine).

    Returns 1.0 if the pass lane looks clear; 0.0 otherwise.
    Replaces BHuman's full geometric passTargetFree × passLineFree × targetInField check.
    """
    # Front-cone opponent distance as pass-line clearance proxy. Convention (matches
    # support_is_live_pass_option and the defender block gate): a non-positive value
    # means NO opponent detected in front → lane clear. Only a POSITIVE distance below
    # the clearance counts as covered — otherwise "no opponent" (0.0) would be wrongly
    # read as "opponent on top of us" and starve every clear-lane pass.
    front = float(striker_obs.nearest_opponent_front_dist)
    if 0.0 < front <= OPPONENT_LANE_CLEARANCE_MM:
        return 0.0
    return 1.0


def _goal_proximity_proxy(target: TeammateSlot) -> float:
    """Proxy for ExpectedGoals.getRating(teammatePos).

    BHuman: small NN(distance_to_goal, opening_angle) → xG.
    Our proxy: linear field-x position (closer to opponent goal = higher xG).
    """
    return float(np.clip(target.x_field / GOAL_X, 0.0, 1.0))


# ---------------------------------------------------------------------------
# R3 — Support positioning (§14.12.4)
# Port of Forward::rating() + Midfielder::rating() / PositionRoles/
# ---------------------------------------------------------------------------

def support_target(
    support_obs: BackendObservation,
    ball_field_xy: tuple[float, float],
    striker_field_xy: tuple[float, float],
    other_support_field_xy: tuple[float, float] | None,
    current_slot: tuple[float, float] | None,
    hold_tolerance_mm: float = HOLD_TOLERANCE_OPEN_MM,
    forced_side: float | None = None,
) -> tuple[tuple[float, float], str]:
    """R3: choose the best support slot and decide walk-or-stand.

    Returns (target_xy_field_mm, skill_name) where skill_name ∈ {"walk", "stand"}.

    Port of Tactic position assignment:
      - forward_rating() ← Forward::rating() pass×goal proximity
      - separation_penalty() ← Midfielder::rating() teamSeparation Gaussian
      - STAB_OFFSET_MM ← Tactic.Position.stabilityOffset

    forced_side: pass ±1.0 to pin the mirror side for the episode. Deriving the
    side from the ball's CURRENT y flips the candidates every frame when the ball
    is driven near the centerline (|y| < 450 is the common band-scenario case);
    the EMA then averages mirrored slots onto y≈0 and walks the support down the
    striker's corridor — same failure mode as compute_triangle_target's per-step
    side derivation. BHuman Tactic templates assign formation sides sticky, not
    per-frame.
    """
    # Mirror grid y by ball side so supports stay on the ball's side of the field.
    if forced_side is not None:
        ball_y_sign = 1.0 if forced_side >= 0.0 else -1.0
    else:
        ball_y_sign = 1.0 if ball_field_xy[1] >= 0.0 else -1.0
    candidates = [(x, y * ball_y_sign) for (x, y) in SUPPORT_SLOT_CANDIDATE_GRID]

    teammates_for_sep = [striker_field_xy]
    if other_support_field_xy is not None:
        teammates_for_sep.append(other_support_field_xy)

    scored = []
    for cand in candidates:
        fwd = forward_rating(cand, ball_field_xy)
        sep = separation_penalty(cand, teammates_for_sep)
        scored.append((fwd * sep, cand))

    scored.sort(reverse=True, key=lambda t: t[0])
    best_score, best_cand = scored[0]

    # stabilityOffset analogue: keep current slot unless alternative is meaningfully better.
    if current_slot is not None:
        cur_score = next((s for s, c in scored if c == current_slot), 0.0)
        # Score units are [0,1]; scale STAB_OFFSET relative to field half.
        stab_threshold = STAB_OFFSET_MM / (2.0 * FIELD_X_HALF)
        if best_score - cur_score < stab_threshold:
            best_cand = current_slot

    dx = best_cand[0] - float(support_obs.robot_x)
    dy = best_cand[1] - float(support_obs.robot_y)
    dist = math.hypot(dx, dy)
    skill = "stand" if dist < hold_tolerance_mm else "walk"
    return (best_cand, skill)


def forward_rating(pos: tuple[float, float], ball: tuple[float, float]) -> float:
    """Port of Forward::rating(pos) — PositionRoles/Forward.cpp:24-38.

    Original: passRating × goalRating × baseRating.
    Our proxy: pass_dist_ok(pos, ball) × goal_proximity(pos).
    baseRating = 1.0 (candidate grid is already shaped to valid receive poses).
    """
    pass_dist = math.hypot(pos[0] - ball[0], pos[1] - ball[1])
    if pass_dist < MIN_PASS_DISTANCE_MM or pass_dist > MAX_PASS_DISTANCE_MM:
        return 0.0
    return float(np.clip(pos[0] / GOAL_X, 0.0, 1.0))


def separation_penalty(pos: tuple[float, float], teammates: list[tuple[float, float]]) -> float:
    """Port of Midfielder::rating separation Gaussian — PositionRoles/Midfielder.cpp:60-72.

    rating = 1 - exp(-0.5 × minTeammateDistSq / σ²)

    Pos with no nearby teammates → ≈ 1.0 (clean).
    Pos stacked on a teammate → ≈ 0.0 (bad).
    """
    if not teammates:
        return 1.0
    min_dsq = min((pos[0] - t[0]) ** 2 + (pos[1] - t[1]) ** 2 for t in teammates)
    return 1.0 - math.exp(-0.5 * min_dsq / (SIGMA_TEAM_MM ** 2))


# ---------------------------------------------------------------------------
# Role branches: teacher_striker, teacher_open_support, teacher_off_ball_support
# ---------------------------------------------------------------------------

def teacher_striker(
    obs: BackendObservation,
    stage_id: int,
    prev_skill_name: str | None,
    gate_decision: GateDecision,
    teammate_slots: list[TeammateSlot],
    last_pass_target: int | None,
    last_pass_target_age_ms: float,
) -> StructuredSkillAction:
    """§11.B.2.1 + R2.

    Delegates to the existing single-agent teacher for all walk/dribble decisions.
    When the single-agent teacher would shoot AND both shoot_armed and pass_armed
    are True, R2 smash_or_pass() may override with a pass action.

    Bug 6 fix: hard-gate on pass_armed via getattr with default False. Until gate.py
    has been extended with the `pass_armed` field, this guard is structurally False
    and no pass labels are emitted — preserving Design-A consistency before the gate,
    obs, and mask extensions land. Once gate.py exposes pass_armed, the guard fires.

    last_pass_target is the BHuman agent_id (player number) from the prior frame,
    or None if no recent pass. The env owns this state.
    """
    # D2/D3: defensive-third clearance. When the striker (often a just-recovered
    # defender, see D1) controls the ball deep in our own third, real soccer says
    # clear to safety or pass out — NOT dribble across our own box toward the
    # opponent goal. Prefer a safe up-field pass; else clear (kick up-field, which
    # B-Human's goal-aware shoot does from deep); else drive the ball out toward
    # midfield. Fires only with ball control (shoot/dribble armed), so it's gate-
    # consistent; the clearance-shoot path relies on shoot_armed being set for
    # clearances (baked in the dataset, see teacher_multiagent_dataset / D4).
    controlled = bool(gate_decision.shoot_armed or gate_decision.dribble_armed)
    if controlled and float(obs.ball_x) < DEFENSIVE_CLEAR_X_MM:
        clear_pass_id = defensive_clear_pass_target(obs, teammate_slots)
        if clear_pass_id >= 0 and bool(getattr(gate_decision, "pass_armed", False)):
            return StructuredSkillAction(
                skill_type=SkillType.pass_,
                target_x=0.0, target_y=0.0, target_theta=0.0,
                pass_target=clear_pass_id,
            )
        if gate_decision.shoot_armed and defender_clear_armed(obs, teammate_slots):
            # Clearance kick up-field (away from our goal).
            return StructuredSkillAction(skill_type=SkillType.shoot)
        # Fallback: drive the ball out of the danger zone toward midfield/wing.
        side = 1.0 if float(obs.robot_y) >= 0.0 else -1.0
        escape_x = CLEAR_ESCAPE_TARGET_X_MM
        escape_y = float(np.clip(abs(float(obs.robot_y)) + 600.0, 0.0, 2000.0)) * side
        heading = math.atan2(escape_y - float(obs.robot_y), escape_x - float(obs.robot_x))
        drive_skill = SkillType.dribble if gate_decision.dribble_armed else SkillType.walk
        return StructuredSkillAction(
            skill_type=drive_skill,
            target_x=escape_x, target_y=escape_y, target_theta=heading,
        )

    decision = stage_aware_teacher_action(
        obs, stage_id, prev_skill_name, gate_decision=gate_decision,
    )
    # Defensive read — gate may not yet carry pass_armed (extension pending per §14.14).
    pass_armed = bool(getattr(gate_decision, "pass_armed", False))

    # PlayBall.cpp decides shoot-vs-pass-vs-dribble every frame while playing the
    # ball. Restricting the pass override to shoot decisions starved pass entirely:
    # mid-advance (where a forward support beats the striker's own xG) the
    # single-agent teacher is dribbling, not shooting. Override either decision
    # when the corresponding arm bit holds and R2 proposes a committed pass.
    ball_controlled = (
        (decision.skill_name == "shoot" and gate_decision.shoot_armed)
        or (decision.skill_name == "dribble" and gate_decision.dribble_armed)
    )
    if ball_controlled and pass_armed:
        skill, agent_id = smash_or_pass(
            obs, teammate_slots, last_pass_target, last_pass_target_age_ms,
        )
        if skill == "pass" and agent_id >= 0:
            return StructuredSkillAction(
                skill_type=SkillType.pass_,
                target_x=0.0,
                target_y=0.0,
                target_theta=0.0,
                pass_target=agent_id,
            )
    return decision


def support_is_live_pass_option(obs: BackendObservation) -> bool:
    """v3 (doc-35): is THIS support a live pass option right now?

    True when (a) the team ball is valid and a teammate is controlling it (within
    SUPPORT_CONTROL_NEAR_MM of the ball — i.e. the striker has it) and (b) the
    support's front receive lane isn't tightly covered by an opponent. This is the
    contingency that makes "hold" meaningful: stand when you're a real outlet,
    reposition otherwise.
    """
    if not bool(getattr(obs, "global_team_ball_valid", False)):
        return False
    bx = float(getattr(obs, "global_team_ball_x_mm", 0.0))
    by = float(getattr(obs, "global_team_ball_y_mm", 0.0))
    poses = getattr(obs, "teammate_poses", ()) or ()
    ages = getattr(obs, "teammate_ages_ms", ()) or ()
    controls = False
    for idx, pose in enumerate(poses):
        age = float(ages[idx]) if idx < len(ages) else float("inf")
        if age <= TEAMMATE_FRESH_MAX_AGE_MS and math.hypot(float(pose[0]) - bx, float(pose[1]) - by) < SUPPORT_CONTROL_NEAR_MM:
            controls = True
            break
    if not controls:
        return False
    front_opp = float(obs.nearest_opponent_front_dist)
    lane_covered = 0.0 < front_opp < SUPPORT_LANE_CLEAR_MM
    return not lane_covered


def teacher_open_support(
    obs: BackendObservation,
    stage_id: int,
    prev_skill_name: str | None,
    gate_decision: GateDecision,
    coordination_target: tuple[float, float],
    phase: str = "attack",
) -> StructuredSkillAction:
    """§11.B.2.2 — walk-or-stand toward the smoothed open-lane target.

    Design-A: always emits walk or stand. Mask for is_open_support forbids
    shoot and dribble. Gate consistency holds.
    coordination_target is the EMA-smoothed open-lane center (field mm), from env.

    v3 (doc-35): in ATTACK, hold (stand) only when at the slot AND a live pass
    option; otherwise reposition (walk) — keeps the support an active outlet
    instead of parking in the void. In DEFENSE the open bit is the defense-supporter
    (cover, not a pass outlet), so contingent hold does NOT apply — geometric hold.
    """
    tx, ty = coordination_target
    dx = tx - float(obs.robot_x)
    dy = ty - float(obs.robot_y)
    dist = math.hypot(dx, dy)
    at_slot = dist < HOLD_TOLERANCE_OPEN_MM
    contingent = CONTINGENT_HOLD and phase != "defense"
    if at_slot and (not contingent or support_is_live_pass_option(obs)):
        return StructuredSkillAction(skill_type=SkillType.stand)
    target_theta = math.atan2(dy, dx)
    return _support_walk_action(tx, ty, target_theta)


def teacher_off_ball_support(
    obs: BackendObservation,
    stage_id: int,
    prev_skill_name: str | None,
    gate_decision: GateDecision,
    coordination_target: tuple[float, float],
) -> StructuredSkillAction:
    """§11.B.2.3 — triangulation + unexpected-possession escape hatch.

    If dribble_armed fires (ball landed on this agent), act as striker for
    one frame instead of walking away. This handles unexpected possession.

    Design-A: normally emits walk or stand (forbid shoot, dribble allowed).
    Escape hatch may emit dribble — gate_decision.dribble_armed being True
    means the gate already permits it, so Design-A holds.
    """
    if gate_decision.dribble_armed:
        return stage_aware_teacher_action(
            obs, stage_id, prev_skill_name, gate_decision=gate_decision,
        )
    tx, ty = coordination_target
    dx = tx - float(obs.robot_x)
    dy = ty - float(obs.robot_y)
    dist = math.hypot(dx, dy)
    at_slot = dist < HOLD_TOLERANCE_OFF_BALL_MM
    # v3 (doc-35): contingent hold — stand only when at the triangle vertex AND a
    # live pass option; otherwise reposition.
    if at_slot and (not CONTINGENT_HOLD or support_is_live_pass_option(obs)):
        return StructuredSkillAction(skill_type=SkillType.stand)
    target_theta = math.atan2(dy, dx)
    return _support_walk_action(tx, ty, target_theta)


def teacher_defender(
    obs: BackendObservation,
    stage_id: int,
    prev_skill_name: str | None,
    gate_decision: GateDecision,
    coordination_target: tuple[float, float],
    team_ball_valid: bool = False,
    face_ball_xy: tuple[float, float] | None = None,
) -> StructuredSkillAction:
    """Defensive off-ball role.

    Ports Rafael's useful values without porting the obs26 actor or duplicate
    mask: block offset is ball_y / 900, stale own ball with valid team ball walks
    to cover instead of freezing, and hold/stand only happens near cover.
    """
    if gate_decision.dribble_armed or defender_block_armed(
        obs,
        team_ball_valid=team_ball_valid,
        face_ball_xy=face_ball_xy,
        coordination_target=coordination_target,
    ):
        return StructuredSkillAction(
            skill_type=SkillType.block,
            target_x=0.0,
            target_y=_signed_block_offset(obs),
            target_theta=0.0,
            pass_target=0,
        )
    if defender_mark_armed(
        obs,
        team_ball_valid=team_ball_valid,
        face_ball_xy=face_ball_xy,
        coordination_target=coordination_target,
    ):
        return StructuredSkillAction(
            skill_type=SkillType.mark,
            target_x=0.0,
            target_y=_signed_mark_offset(obs),
            target_theta=0.0,
            pass_target=0,
        )
    tx, ty = coordination_target
    dx = tx - float(obs.robot_x)
    dy = ty - float(obs.robot_y)
    dist = math.hypot(dx, dy)
    if (
        dist < HOLD_TOLERANCE_OFF_BALL_MM
        and not (team_ball_valid and float(obs.time_since_ball_seen) > TEAMMATE_FRESH_MAX_AGE_MS)
    ):
        return StructuredSkillAction(skill_type=SkillType.stand)
    face_x = float(obs.ball_x)
    face_y = float(obs.ball_y)
    if team_ball_valid and face_ball_xy is not None:
        face_x, face_y = face_ball_xy
    target_theta = math.atan2(face_y - float(obs.robot_y), face_x - float(obs.robot_x))
    return _support_walk_action(tx, ty, target_theta)


# ---------------------------------------------------------------------------
# Public dispatcher — §14.13
# ---------------------------------------------------------------------------

def stage_aware_team_teacher_action(
    obs: BackendObservation,
    stage_id: int,
    prev_skill_name: str | None = None,
    *,
    gate_decision: GateDecision,
    role: str,
    coordination_target: tuple[float, float],
    teammate_slots: list[TeammateSlot],
    last_pass_target: int | None,
    last_pass_target_age_ms: float,
    phase: str = "attack",
    team_ball_valid: bool = False,
    face_ball_xy: tuple[float, float] | None = None,
) -> StructuredSkillAction:
    """Stateless dispatcher. Pure function of its inputs.

    role:               "striker" | "open_support" | "off_ball_support"
    coordination_target: field-frame (x, y) in mm — ball for striker,
                         EMA open-lane centre for open_support,
                         EMA triangle vertex for off_ball_support.
    teammate_slots:     list[TeammateSlot] populated from obs[26:32] by caller.
    last_pass_target:   slot_idx from prior step, or None.
    last_pass_target_age_ms: ms elapsed since last pass decision, 0 if none.
    """
    if role == "striker":
        return teacher_striker(
            obs, stage_id, prev_skill_name, gate_decision,
            teammate_slots, last_pass_target, last_pass_target_age_ms,
        )
    if role == "open_support":
        return teacher_open_support(
            obs, stage_id, prev_skill_name, gate_decision, coordination_target, phase=phase,
        )
    if role == "off_ball_support":
        if phase == "defense":
            return teacher_defender(
                obs,
                stage_id,
                prev_skill_name,
                gate_decision,
                coordination_target,
                team_ball_valid=team_ball_valid,
                face_ball_xy=face_ball_xy,
            )
        return teacher_off_ball_support(
            obs, stage_id, prev_skill_name, gate_decision, coordination_target,
        )
    raise ValueError(f"Unknown role: {role!r}")


# ---------------------------------------------------------------------------
# Residual conversion + dataset payload (§11.B.2.1, Bug 7 fix)
#
# The single-agent pipeline (teacher_policy.absolute_to_residual_action) anchors
# walk/dribble residuals on the BALL. For supports, the absolute target is a slot
# AWAY from the ball, so the ball-anchored residual would encode "(slot - ball)"
# and the policy could not learn the support trajectory.
#
# Approach B (per user decision): mirror the residual conversion locally so
# `teacher_policy.py` is provably untouched. Striker delegates to the imported
# absolute_to_residual_action (bit-identical to single-agent). Supports anchor
# walk residuals on the coordination_target; supports' dribble (escape hatch on
# off-ball-support) anchors on the ball via the imported single-agent helper.
# ---------------------------------------------------------------------------


def _wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]. Local copy of teacher_policy._wrap_angle to avoid
    importing a private name and to keep this module self-contained."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def build_open_lane_anchor(
    obs: BackendObservation,
    coordination_target: tuple[float, float],
) -> dict[str, float]:
    """Anchor for open-support residuals: walk toward the smoothed open-lane
    target, facing it. Parallel to reward.build_fallback_action_targets but
    anchored on the coordination target instead of the ball."""
    tx, ty = coordination_target
    return {
        "target_x": float(tx),
        "target_y": float(ty),
        "target_theta": math.atan2(ty - float(obs.robot_y), tx - float(obs.robot_x)),
    }


def build_triangle_anchor(
    obs: BackendObservation,
    coordination_target: tuple[float, float],
) -> dict[str, float]:
    """Anchor for off-ball-support residuals. Identical math to the open-lane
    anchor; the env applies a different EMA α (0.2 vs 0.3) to the coordination
    target before passing it in, so the smoothing characteristic differs even
    though the anchor function is the same."""
    return build_open_lane_anchor(obs, coordination_target)


def _to_residual_role(
    action: StructuredSkillAction,
    obs: BackendObservation,
    role: str,
    coordination_target: tuple[float, float] | None,
) -> StructuredSkillAction:
    """Role-aware absolute → residual conversion.

    striker:           delegates to teacher_policy.absolute_to_residual_action.
                       Single-agent path bit-identical preserved.
    open_support:      walk residuals anchored on coordination_target.
    off_ball_support:  walk residuals anchored on coordination_target;
                       dribble residuals (escape hatch) ball-anchored via
                       the single-agent helper.
    non-walk/dribble:  zeros + preserve pass_target.
    """
    if role == "striker":
        return absolute_to_residual_action(action, obs)

    if action.skill_name not in {"walk", "dribble", "block", "mark"}:
        # stand / shoot / pass: zeros, preserve pass_target if any.
        return StructuredSkillAction(
            skill_type=action.skill_type,
            target_x=0.0,
            target_y=0.0,
            target_theta=0.0,
            pass_target=action.pass_target,
        )

    if action.skill_name == "dribble":
        # Off-ball-support escape hatch: action targets the ball, so anchor on
        # the ball via the existing single-agent helper.
        return absolute_to_residual_action(action, obs)

    if action.skill_name in {"block", "mark"}:
        # block/mark carry a 1-D lateral offset in target_y (param_mask (0,1,0,0));
        # target_x is masked. Pass the offset through unchanged so the defender's
        # block-the-ball-line / mark-the-runner aim survives into the BC label.
        return StructuredSkillAction(
            skill_type=action.skill_type,
            target_x=0.0,
            target_y=float(np.clip(action.target_y, -1.0, 1.0)),
            target_theta=0.0,
            pass_target=action.pass_target,
        )

    # Support walk — anchor on coordination_target.
    if coordination_target is None:
        raise ValueError(
            f"role={role!r} walking requires a coordination_target; got None"
        )
    if role == "open_support":
        anchors = build_open_lane_anchor(obs, coordination_target)
    elif role == "off_ball_support":
        anchors = build_triangle_anchor(obs, coordination_target)
    else:
        raise ValueError(f"Unknown role for residual conversion: {role!r}")

    residual_x = 0.0
    if abs(float(ACTION.residual_x_range)) > 1e-6:
        residual_x = float(np.clip(
            (float(action.target_x) - anchors["target_x"]) / ACTION.residual_x_range,
            -1.0, 1.0,
        ))
    residual_y = float(np.clip(
        (float(action.target_y) - anchors["target_y"]) / ACTION.residual_y_range,
        -1.0, 1.0,
    ))
    residual_theta = float(np.clip(
        _wrap_angle(float(action.target_theta) - anchors["target_theta"])
        / ACTION.residual_theta_range,
        -1.0, 1.0,
    ))
    return StructuredSkillAction(
        skill_type=action.skill_type,
        target_x=residual_x,
        target_y=residual_y,
        target_theta=residual_theta,
        pass_target=action.pass_target,
    )


def team_teacher_payload(
    obs: BackendObservation,
    stage_id: int,
    prev_skill_name: str | None,
    *,
    gate_decision: GateDecision,
    role: str,
    coordination_target: tuple[float, float],
    teammate_slots: list[TeammateSlot],
    last_pass_target: int | None,
    last_pass_target_age_ms: float,
    phase: str = "attack",
    team_ball_valid: bool = False,
    face_ball_xy: tuple[float, float] | None = None,
) -> tuple[dict[str, float], StructuredSkillAction, StructuredSkillAction]:
    """Multi-agent analogue of teacher_policy.teacher_policy_payload.

    Returns (payload, absolute_action, residual_action). The dataset generator
    writes residual_action; payload carries the skill one-hots + role_id for BC.

    Use this from teacher_production_dataset.py instead of teacher_policy_payload
    for stage 6+ scenarios. Stage <= 5 keeps using teacher_policy_payload for
    bit-identical single-agent labels.
    """
    absolute = stage_aware_team_teacher_action(
        obs,
        stage_id,
        prev_skill_name,
        gate_decision=gate_decision,
        role=role,
        coordination_target=coordination_target,
        teammate_slots=teammate_slots,
        last_pass_target=last_pass_target,
        last_pass_target_age_ms=last_pass_target_age_ms,
        phase=phase,
        team_ball_valid=team_ball_valid,
        face_ball_xy=face_ball_xy,
    )
    residual = _to_residual_role(absolute, obs, role, coordination_target)
    payload: dict[str, float] = residual.as_dict()
    payload["policy_skill_raw"] = float(residual.skill_type)
    for skill_name in SKILL_ORDER:
        payload[f"policy_skill_prob_{skill_name}"] = 1.0 if skill_name == residual.skill_name else 0.0
    payload["role_id"] = float({"striker": 0, "open_support": 1, "off_ball_support": 2}[role])
    return payload, absolute, residual


# ---------------------------------------------------------------------------
# Coordination target computation — called by env each step, result EMA-smoothed there
# ---------------------------------------------------------------------------

def compute_open_lane_target(
    open_support_obs: BackendObservation,
    ball_field_xy: tuple[float, float],
    striker_field_xy: tuple[float, float],
    forced_side: float | None = None,
) -> tuple[float, float]:
    """Compute the raw (pre-EMA) open-lane receive position for the open_support.

    Strategy: pick the highest-scoring slot from SUPPORT_SLOT_CANDIDATE_GRID.
    The EMA smoothing (α=0.3) is applied in the env after this returns.
    Returns absolute field coordinates (mm).

    Bug 2 fix: support_target returns (target_xy, skill_name). The xy tuple is the
    FIRST element. Previous code used `_, best_cand = support_target(...)` which
    assigned the skill string to best_cand and silently returned "walk"/"stand"
    instead of coordinates.
    """
    best_cand, _ = support_target(
        support_obs=open_support_obs,
        ball_field_xy=ball_field_xy,
        striker_field_xy=striker_field_xy,
        other_support_field_xy=None,
        current_slot=None,
        hold_tolerance_mm=HOLD_TOLERANCE_OPEN_MM,
        forced_side=forced_side,
    )
    return best_cand


def compute_triangle_target(
    off_ball_obs: BackendObservation,
    ball_field_xy: tuple[float, float],
    striker_field_xy: tuple[float, float],
    open_support_field_xy: tuple[float, float],
    forced_side: float | None = None,
) -> tuple[float, float]:
    """Compute the raw (pre-EMA) triangulation vertex for the off_ball_support.

    Mirrors SUPPORT_SLOT_CANDIDATE_GRID y to the *opposite* side from open_support
    so the two supports spread out. The EMA (α=0.2) is applied in the env.
    Returns absolute field coordinates (mm).

    forced_side: pass ±1.0 to pin the mirror side for the whole episode. Deriving
    the side from open_support's CURRENT y every step makes the candidates flip
    sign whenever the open support wobbles across the centerline; the EMA then
    averages the two mirrored targets onto y≈0 and walks the off_ball support
    straight down the striker's drive corridor (2026-06-10 collision_contexts:
    7/9 stage-7 collisions were off_ball near y≈0 at x 1600-2100).
    """
    # Flip y: off_ball goes to the side opposite of open_support.
    if forced_side is not None:
        ball_y_sign_flipped = 1.0 if forced_side >= 0.0 else -1.0
    else:
        ball_y_sign_flipped = -1.0 if open_support_field_xy[1] >= 0.0 else 1.0
    candidates = [(x, y * ball_y_sign_flipped) for (x, y) in SUPPORT_SLOT_CANDIDATE_GRID]

    teammates_for_sep = [striker_field_xy, open_support_field_xy]
    scored = []
    for cand in candidates:
        fwd = forward_rating(cand, ball_field_xy)
        sep = separation_penalty(cand, teammates_for_sep)
        scored.append((fwd * sep, cand))

    scored.sort(reverse=True, key=lambda t: t[0])
    _, best = scored[0]
    return best


def defender_cover_target(
    defender_obs: BackendObservation,
    ball_field_xy: tuple[float, float],
    *,
    current_side: float | None = None,
) -> tuple[tuple[float, float], float]:
    """Cover the uncovered own-goal gap on a defensive circle.

    Simplified B-Human Defender::position port: pick a sticky post side, derive
    the GK-covered edge, then intersect ball→cover_point with the defensive
    circle centered on own goal line.
    """
    side = _defender_side(ball_field_xy[1], current_side)
    gk_x, gk_y = _own_gk_pose(defender_obs)
    goal_post = (OWN_GOAL_X_MM, OWN_GOAL_POST_Y_MM * side)
    gk_edge = (gk_x, gk_y + GK_STAND_RADIUS_MM * side)
    cover_point = ((goal_post[0] + gk_edge[0]) * 0.5, (goal_post[1] + gk_edge[1]) * 0.5)
    target = _line_circle_intersection_near_ball(
        ball_field_xy,
        cover_point,
        (OWN_GOAL_X_MM, 0.0),
        DEFENDER_RADIUS_MM,
    )
    target = _clip_defender_target(target)
    return target, side


def defense_support_target(
    support_obs: BackendObservation,
    ball_field_xy: tuple[float, float],
    presser_field_xy: tuple[float, float],
    defender_field_xy: tuple[float, float],
    forced_side: float | None = None,
) -> tuple[float, float]:
    side = 1.0 if (forced_side if forced_side is not None else ball_field_xy[1]) >= 0.0 else -1.0
    candidates = [(x, y * side) for (x, y) in DEFENSE_SLOT_CANDIDATE_GRID]
    teammates_for_sep = [presser_field_xy, defender_field_xy]
    scored = []
    for cand in candidates:
        lane = _defense_lane_cover(cand, ball_field_xy)
        sep = separation_penalty(cand, teammates_for_sep)
        scored.append((lane * sep, cand))
    scored.sort(reverse=True, key=lambda item: item[0])
    return scored[0][1]


# ---------------------------------------------------------------------------
# TeammateSlot builder — convenience helper for callers
# ---------------------------------------------------------------------------

def teammate_slots_from_obs(
    self_obs: BackendObservation,
    agent_id_map: dict[int, int],
) -> list[TeammateSlot]:
    """Build the TeammateSlot list from raw teammate_poses + teammate_ages_ms.

    Bug 3 fix: BackendObservation does NOT carry `nearest_teammate_x_rel` etc. —
    those are positions in the *encoded* 41-dim obs vector (ObsIndex 26..31), not
    attributes on the dataclass. The raw dataclass exposes `teammate_poses` (field
    frame), `teammate_ages_ms`, and `teammate_activities`. This builder replicates
    the same age-filter + Euclidean-sort that environment.py:_encode_obs applies
    when populating Block B, so the slot order here is identical to Block B's.

    self_obs:     the calling agent's BackendObservation.
    agent_id_map: {slot_idx: agent_id} — the env must construct this map with the
                  SAME filter + sort as below, so map[slot_idx] is the BHuman
                  player number that ends up at that slot. Pass -1 for slots whose
                  agent_id is unknown — smash_or_pass skips them.

    Returns up to 3 TeammateSlot objects in distance-sorted order.
    """
    self_x = float(self_obs.robot_x)
    self_y = float(self_obs.robot_y)

    # Mirror environment.py:_encode_obs fresh-and-sort logic exactly.
    fresh: list[tuple[float, float]] = []
    for idx, pose in enumerate(self_obs.teammate_poses):
        age = (
            float(self_obs.teammate_ages_ms[idx])
            if idx < len(self_obs.teammate_ages_ms)
            else float("inf")
        )
        if age <= TEAMMATE_FRESH_MAX_AGE_MS:
            fresh.append((float(pose[0]), float(pose[1])))
    fresh.sort(key=lambda p: (p[0] - self_x) ** 2 + (p[1] - self_y) ** 2)

    slots: list[TeammateSlot] = []
    for slot_idx, (x_field, y_field) in enumerate(fresh[:3]):
        agent_id = agent_id_map.get(slot_idx, -1)
        is_gk = x_field < GK_EXCLUDE_X_MM
        slots.append(TeammateSlot(
            slot_idx=slot_idx,
            agent_id=agent_id,
            x_field=x_field,
            y_field=y_field,
            is_gk=is_gk,
        ))
    return slots


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _support_walk_action(
    target_x: float,
    target_y: float,
    target_theta: float,
) -> StructuredSkillAction:
    """Build a walk action toward an absolute field-frame (x, y) target.

    Unlike _action_from_anchor (which anchors on the ball), this anchor is
    the coordination_target itself. The caller has already computed target_theta.
    No obstacle-avoidance residual for supports (they are far from the ball cluster).
    """
    return StructuredSkillAction(
        skill_type=SkillType.walk,
        target_x=float(np.clip(target_x, -ACTION.x_range, ACTION.x_range)),
        target_y=float(np.clip(target_y, -ACTION.y_range, ACTION.y_range)),
        target_theta=float(np.clip(target_theta, -ACTION.theta_range, ACTION.theta_range)),
    )


def _signed_block_offset(obs: BackendObservation) -> float:
    return float(np.clip(float(obs.ball_rel_y) / 900.0, -1.0, 1.0))


def defender_block_armed(
    obs: BackendObservation,
    *,
    team_ball_valid: bool,
    face_ball_xy: tuple[float, float] | None,
    coordination_target: tuple[float, float],
) -> bool:
    """Synthetic high-level block gate for defensive BC labels.

    The train-time role mask currently allows `block` through the same obs bit as
    the unexpected-possession/dribble gate. A defender should also block when it
    is close to cover and the ball/opponent threat is in front of our goal.
    """
    if not team_ball_valid or face_ball_xy is None:
        return False
    ball_x, ball_y = float(face_ball_xy[0]), float(face_ball_xy[1])
    if ball_x > DEFENDER_BLOCK_BALL_X_MAX_MM:
        return False
    dist_to_cover = math.hypot(
        float(obs.robot_x) - float(coordination_target[0]),
        float(obs.robot_y) - float(coordination_target[1]),
    )
    if dist_to_cover > DEFENDER_BLOCK_COVER_DIST_MM:
        return False
    # Block = intercept the ball line. Arm only on the ball being in the front cone
    # (obs.ball_rel_x), which the train.py gate recomputes 1:1 as its block_threat
    # (ball_rel_x < 1800mm). The earlier team-ball-distance / opponent-front terms were
    # dropped here: the distance term was not recomputable from a stored obs column
    # (it read the team ball) so it could emit a block label the gate would mask, and
    # the opponent-front term now drives `mark` instead (man-mark the runner).
    ball_front = 0.0 < float(obs.ball_rel_x) < DEFENDER_BLOCK_FRONT_BALL_X_MM
    return bool(ball_front)


def defender_mark_armed(
    obs: BackendObservation,
    *,
    team_ball_valid: bool,
    face_ball_xy: tuple[float, float] | None,
    coordination_target: tuple[float, float],
) -> bool:
    """Synthetic high-level mark gate for defensive BC labels.

    Mark a runner: an opponent is in our front cone (within DEFENDER_MARK_FRONT_OPP_MM)
    while the ball itself is NOT in front (that case is a `block`). Arm condition reads
    only obs.nearest_opponent_front_dist + obs.ball_rel_x — the same stored columns the
    train.py gate uses for mark_threat (nearest_opponent_front_dist < 1500mm), so the
    gate is an exact superset and no mark label is masked. The our-half + near-cover
    guards mirror block and only narrow emission.
    """
    if not team_ball_valid or face_ball_xy is None:
        return False
    ball_x, ball_y = float(face_ball_xy[0]), float(face_ball_xy[1])
    if ball_x > DEFENDER_BLOCK_BALL_X_MAX_MM:
        return False
    dist_to_cover = math.hypot(
        float(obs.robot_x) - float(coordination_target[0]),
        float(obs.robot_y) - float(coordination_target[1]),
    )
    if dist_to_cover > DEFENDER_BLOCK_COVER_DIST_MM:
        return False
    opponent_front = 0.0 < float(obs.nearest_opponent_front_dist) < DEFENDER_MARK_FRONT_OPP_MM
    ball_front = 0.0 < float(obs.ball_rel_x) < DEFENDER_BLOCK_FRONT_BALL_X_MM
    return bool(opponent_front and not ball_front)


def _signed_mark_offset(obs: BackendObservation) -> float:
    """Lateral target_y offset toward the marked opponent (block/mark param_mask=(0,1,0,0)).

    Uses the nearest opponent's bearing+range to place the offset on the opponent's
    side. Mirrors _signed_block_offset's normalisation (÷900, clipped) so the C++
    SkillRequest::Builder::mark(Vector2f) target convention matches block exactly.
    """
    opp_rel_y = float(obs.nearest_opponent_dist) * math.sin(float(obs.nearest_opponent_angle))
    return float(np.clip(opp_rel_y / 900.0, -1.0, 1.0))


def defensive_clear_pass_target(
    striker_obs: BackendObservation,
    teammate_slots: list[TeammateSlot],
) -> int:
    """D2: pick the safest up-field outlet for a clearance pass (agent_id or -1).

    Unlike the attacking smash_or_pass (which scores teammates by goal proximity),
    a clearance wants the most UP-FIELD teammate relative to the ball — get it
    forward and out of our third, keeping possession. Lane must be clear and the
    pass within range; GK is never a clearance target.
    """
    ball_x = float(striker_obs.ball_x)
    best_id = -1
    best_x = ball_x + CLEAR_PASS_MIN_UPFIELD_MM  # must be at least this far up-field
    if float(striker_obs.nearest_opponent_front_dist) <= OPPONENT_LANE_CLEARANCE_MM:
        return -1  # front lane blocked — don't pass into pressure, clear/drive instead
    for slot in teammate_slots:
        if slot.is_gk or slot.agent_id < 0:
            continue
        pass_dist = math.hypot(
            slot.x_field - float(striker_obs.robot_x),
            slot.y_field - float(striker_obs.robot_y),
        )
        if pass_dist < MIN_PASS_DISTANCE_MM or pass_dist > MAX_PASS_DISTANCE_MM:
            continue
        if slot.x_field > best_x:
            best_x = slot.x_field
            best_id = slot.agent_id
    return best_id


def defender_clear_armed(
    obs: BackendObservation,
    teammate_slots: list[TeammateSlot],
) -> bool:
    """D3: synthetic clearance-shoot gate for deep, pressured, no-outlet situations.

    Mirrors defender_block_armed's role: the train-time mask allows shoot only when
    shoot_armed is set, so the dataset bakes shoot_armed=True when this fires (Design-A
    preserved). True when the striker is deep in our own third, centrally exposed, and
    has no safe up-field pass — i.e. the moment to hoof it clear rather than dribble
    through our own box.
    """
    if float(obs.ball_x) >= DEFENSIVE_CLEAR_X_MM:
        return False
    if defensive_clear_pass_target(obs, teammate_slots) >= 0:
        return False  # a safe pass exists — prefer it over a blind clearance
    central = abs(float(obs.ball_y)) < CLEAR_CENTRAL_DANGER_Y_MM
    return bool(central)


def _defender_side(ball_y: float, current_side: float | None) -> float:
    if float(ball_y) > 450.0:
        return 1.0
    if float(ball_y) < -450.0:
        return -1.0
    if current_side is not None:
        return 1.0 if current_side >= 0.0 else -1.0
    return 1.0 if float(ball_y) >= 0.0 else -1.0


def _own_gk_pose(obs: BackendObservation) -> tuple[float, float]:
    candidates: list[tuple[float, float]] = []
    for idx, pose in enumerate(obs.teammate_poses):
        age = float(obs.teammate_ages_ms[idx]) if idx < len(obs.teammate_ages_ms) else float("inf")
        x, y = float(pose[0]), float(pose[1])
        if age <= TEAMMATE_FRESH_MAX_AGE_MS and x < GK_EXCLUDE_X_MM:
            candidates.append((x, y))
    if candidates:
        return min(candidates, key=lambda p: abs(p[0] - OWN_GOAL_X_MM) + abs(p[1]))
    return (OWN_GOAL_X_MM + 200.0, 0.0)


def _line_circle_intersection_near_ball(
    p0: tuple[float, float],
    p1: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> tuple[float, float]:
    x0, y0 = p0
    x1, y1 = p1
    cx, cy = center
    dx = x1 - x0
    dy = y1 - y0
    a = dx * dx + dy * dy
    if a < 1e-6:
        return _clip_defender_target((OWN_GOAL_X_MM + radius, 0.0))
    fx = x0 - cx
    fy = y0 - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        # Project cover point direction onto circle.
        angle = math.atan2(y1 - cy, x1 - cx)
        return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    root = math.sqrt(disc)
    ts = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    points = [(x0 + t * dx, y0 + t * dy) for t in ts]
    return min(points, key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2)


def _clip_defender_target(target: tuple[float, float]) -> tuple[float, float]:
    x = float(np.clip(target[0], -FIELD_X_HALF + 250.0, FIELD_X_HALF - 250.0))
    y = float(np.clip(target[1], -FIELD_Y_HALF + 250.0, FIELD_Y_HALF - 250.0))
    if x < OWN_GOAL_AREA_LIMIT_X_MM and abs(y) < OWN_GOAL_AREA_HALF_Y_MM:
        x = OWN_GOAL_AREA_PUSH_X_MM
    return (x, y)


def _defense_lane_cover(candidate: tuple[float, float], ball: tuple[float, float]) -> float:
    # Reward being between the ball and own goal, but shifted to the second lane
    # instead of stacking on the deep defender.
    own_goal = (OWN_GOAL_X_MM, 0.0)
    bx, by = ball
    cx, cy = candidate
    vx = own_goal[0] - bx
    vy = own_goal[1] - by
    wx = cx - bx
    wy = cy - by
    seg = vx * vx + vy * vy
    if seg < 1e-6:
        return 0.0
    t = float(np.clip((wx * vx + wy * vy) / seg, 0.0, 1.0))
    px = bx + t * vx
    py = by + t * vy
    lane_dist = math.hypot(cx - px, cy - py)
    depth_ok = 1.0 if cx <= max(bx + 800.0, -300.0) else 0.35
    second_lane = math.exp(-0.5 * ((lane_dist - 900.0) / 700.0) ** 2)
    return float(depth_ok * second_lane)


def _map_to_range(
    value: float,
    in_lo: float,
    in_hi: float,
    out_lo: float,
    out_hi: float,
) -> float:
    """Linear interpolation from [in_lo, in_hi] → [out_lo, out_hi], clamped.

    Port of BHuman's mapToRange() utility. Used by R2 for decision_penalty.
    Note: in_lo < in_hi is NOT required — caller passes lo/hi in the order that
    makes the semantics clear; np.interp handles reversed axes.
    """
    return float(np.interp(value, [in_lo, in_hi], [out_lo, out_hi]))


# ---------------------------------------------------------------------------
# Orchestration helpers (§14.17) — used by the dataset script and the env step loop
# ---------------------------------------------------------------------------


def ema_step(prev: tuple[float, float] | None,
             new: tuple[float, float],
             alpha: float) -> tuple[float, float]:
    """One-step EMA on a 2D point. alpha=0.3 for open-lane, 0.2 for triangle (§11.B.3.2)."""
    if prev is None:
        return (float(new[0]), float(new[1]))
    return (
        alpha * float(new[0]) + (1.0 - alpha) * float(prev[0]),
        alpha * float(new[1]) + (1.0 - alpha) * float(prev[1]),
    )


def build_agent_id_map(
    self_agent_id: int,
    agent_positions: dict[int, tuple[float, float]],
    agent_pose_ages_ms: dict[int, float],
) -> dict[int, int]:
    """Build {slot_idx: agent_id} for one agent's view of its teammates.

    Mirrors environment.py:_encode_obs Block-B sort/filter so the resulting
    agent_id_map is consistent with the slot order seen by teammate_slots_from_obs.

    self_agent_id:      the calling agent's player number (excluded from teammates)
    agent_positions:    {agent_id: (x_field_mm, y_field_mm)} for ALL field agents
                        including the caller (will be filtered out)
    agent_pose_ages_ms: {agent_id: broadcast age ms}; stale entries filtered out

    Returns up to 3 mappings, sorted-by-distance from the caller.
    """
    self_pos = agent_positions.get(self_agent_id)
    if self_pos is None:
        return {}
    candidates = []
    for aid, pos in agent_positions.items():
        if aid == self_agent_id:
            continue
        age = float(agent_pose_ages_ms.get(aid, float("inf")))
        if age > TEAMMATE_FRESH_MAX_AGE_MS:
            continue
        candidates.append((aid, pos))
    candidates.sort(
        key=lambda kv: (kv[1][0] - self_pos[0]) ** 2 + (kv[1][1] - self_pos[1]) ** 2
    )
    return {slot_idx: aid for slot_idx, (aid, _) in enumerate(candidates[:3])}


def advance_pass_target_age(state: TeamCoordinationState, dt_ms: float) -> None:
    """Decay every striker's last_pass_target_age_ms by dt_ms. Stale entries are
    removed once they exceed LAST_PASS_TARGET_LIFETIME_MS so smash_or_pass treats
    them as cleared."""
    expired: list[int] = []
    for striker_id, age in state.last_pass_target_age_ms_by_striker.items():
        new_age = age + dt_ms
        if new_age >= LAST_PASS_TARGET_LIFETIME_MS:
            expired.append(striker_id)
        else:
            state.last_pass_target_age_ms_by_striker[striker_id] = new_age
    for striker_id in expired:
        state.last_pass_target_age_ms_by_striker.pop(striker_id, None)
        state.last_pass_target_by_striker.pop(striker_id, None)


def record_pass_target(state: TeamCoordinationState, striker_id: int, target_agent_id: int) -> None:
    """Called by the orchestrator after the striker commits a pass action.
    Resets the decay so R4 hysteresis applies during the next ~1.5 s window."""
    state.last_pass_target_by_striker[striker_id] = int(target_agent_id)
    state.last_pass_target_age_ms_by_striker[striker_id] = 0.0


def build_team_gate_decision(
    base_decision: GateDecision,
    *,
    role: str,
    pass_armed: bool = False,
    pass_arm_progress: float = 0.0,
    observe_armed: bool = False,
) -> GateDecision:
    """Augment a single-agent SkillGate.step() result with multi-agent fields.

    The single-agent gate computes shoot_armed/dribble_armed/shoot_arm_progress
    from the obs alone. The orchestrator runs smash_or_pass to determine
    pass_armed and writes the role one-hot here. The resulting GateDecision has
    obs_features_team() populated, which the env's _encode_obs appends at
    indices 41-46.

    role must be one of "striker", "open_support", "off_ball_support".
    """
    import dataclasses
    role_flags = {
        "is_striker": role == "striker",
        "is_open_support": role == "open_support",
        "is_off_ball_support": role == "off_ball_support",
    }
    if not any(role_flags.values()):
        raise ValueError(
            f"Unknown role: {role!r} (expected 'striker' | 'open_support' | 'off_ball_support')"
        )
    return dataclasses.replace(
        base_decision,
        pass_armed=bool(pass_armed),
        observe_armed=bool(observe_armed),
        pass_arm_progress=float(pass_arm_progress),
        **role_flags,
    )
