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
from dataclasses import dataclass, field

import numpy as np

from action import SKILL_ORDER, SkillType, StructuredSkillAction
from config import ACTION, FIELD_X_HALF, FIELD_Y_HALF, GOAL_X
from gate import GateDecision
from pybh_backend import BackendObservation
from teacher_policy import _action_from_anchor, stage_aware_teacher_action

# ---------------------------------------------------------------------------
# §14.12.1 — Constants (single source of truth). Tune on Day 5.
# ---------------------------------------------------------------------------

# R1 (Role assignment)
ROLE_HYSTERESIS_FRAMES = 5          # §11.B.3.1 — consecutive frames before swap commits
ROLE_MIN_DISTANCE_GAP_MM = 200.0    # candidate must be ≥ 200mm closer than current striker
TEAMMATE_FRESH_MAX_AGE_MS = 1000.0  # poses older than this are treated as stale

# R2 (smashOrPass — port of PlayBall.cpp:33-118)
SHOOT_THRESHOLD = 0.55              # own_value above this → shoot directly
MIN_PASS_DISTANCE_MM = 400.0        # PlayBall.cpp:p.minPassDistance (adapted: tighter)
MAX_PASS_DISTANCE_MM = 4000.0       # PlayBall.cpp:p.maxPassDistance
DECISION_PENALTY_MIN = 0.05         # mapToRange floor (close ball → big penalty)
DECISION_PENALTY_MAX = 0.20         # mapToRange ceiling (far ball → small penalty)
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
)

# R6 (Pass quality proxies for BHuman's learned estimators)
CAN_SCORE_BOOST = 0.3               # own ball: + 0.3 × can_score_now added to shoot rating
OPPONENT_LANE_CLEARANCE_MM = 800.0  # nearest_opponent_front_dist must exceed this

# Exclusions
GK_EXCLUDE_X_MM = -2000.0          # teammate with absolute x < this is treated as GK


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


# ---------------------------------------------------------------------------
# R1 — Role assignment with 5-frame hysteresis (§14.12.2)
# Port of Behavior::determineActiveAgent() — Behavior.cpp:L1065
# ---------------------------------------------------------------------------

def assign_roles(
    agent_obs: dict[int, BackendObservation],
    team_ball: tuple[float, float, bool],
    history: TeamRoleHistory,
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
    all_stale = all(
        float(agent_obs[aid].time_since_ball_seen) * 1000.0 > TEAMMATE_FRESH_MAX_AGE_MS
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

    committed_striker = candidate
    if prev_striker is not None and prev_striker in agent_ids and prev_striker != candidate:
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
        # 2-RL: exactly one open_support.
        roles = {
            committed_striker: "striker",
            non_strikers[0]: "open_support",
        }
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

    Returns (skill_name, pass_target_slot_idx).
    pass_target_slot_idx == -1 when skill_name != "pass".

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
    d_ball = math.hypot(float(striker_obs.ball_rel_x), float(striker_obs.ball_rel_y))
    decision_penalty = _map_to_range(
        d_ball,
        in_lo=BALL_DIST_MAX_PENALTY_MM, in_hi=BALL_DIST_MIN_PENALTY_MM,
        out_lo=DECISION_PENALTY_MIN, out_hi=DECISION_PENALTY_MAX,
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
    best_slot_idx = -1
    best_value = own_value
    for slot in teammate_slots:
        if slot.is_gk:
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

        # PlayBall.cpp:94-98 — penalise switching pass target.
        if target_active and last_pass_target != slot.slot_idx:
            agent_value -= decision_penalty

        if agent_value > best_value:
            best_value = agent_value
            best_slot_idx = slot.slot_idx

    # PlayBall.cpp:105-117 — commit.
    if best_slot_idx >= 0:
        return ("pass", best_slot_idx)
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
    # Use striker's front-cone opponent distance as pass-line clearance proxy.
    if float(striker_obs.nearest_opponent_front_dist) <= OPPONENT_LANE_CLEARANCE_MM:
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
) -> tuple[tuple[float, float], str]:
    """R3: choose the best support slot and decide walk-or-stand.

    Returns (target_xy_field_mm, skill_name) where skill_name ∈ {"walk", "stand"}.

    Port of Tactic position assignment:
      - forward_rating() ← Forward::rating() pass×goal proximity
      - separation_penalty() ← Midfielder::rating() teamSeparation Gaussian
      - STAB_OFFSET_MM ← Tactic.Position.stabilityOffset
    """
    # Mirror grid y by ball side so supports stay on the ball's side of the field.
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
    When the single-agent teacher would shoot and shoot_armed, R2 smash_or_pass()
    may override with a pass action.
    """
    decision = stage_aware_teacher_action(
        obs, stage_id, prev_skill_name, gate_decision=gate_decision,
    )
    if decision.skill_name == "shoot" and gate_decision.shoot_armed:
        skill, slot_idx = smash_or_pass(
            obs, teammate_slots, last_pass_target, last_pass_target_age_ms,
        )
        if skill == "pass" and slot_idx >= 0:
            return StructuredSkillAction(
                skill_type=SkillType.pass_,
                target_x=0.0,
                target_y=0.0,
                target_theta=0.0,
                pass_target=slot_idx,
            )
    return decision


def teacher_open_support(
    obs: BackendObservation,
    stage_id: int,
    prev_skill_name: str | None,
    gate_decision: GateDecision,
    coordination_target: tuple[float, float],
) -> StructuredSkillAction:
    """§11.B.2.2 — walk-or-stand toward the smoothed open-lane target.

    Design-A: always emits walk or stand. Mask for is_open_support forbids
    shoot and dribble. Gate consistency holds.
    coordination_target is the EMA-smoothed open-lane center (field mm), from env.
    """
    tx, ty = coordination_target
    dx = tx - float(obs.robot_x)
    dy = ty - float(obs.robot_y)
    dist = math.hypot(dx, dy)
    if dist < HOLD_TOLERANCE_OPEN_MM:
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
    if dist < HOLD_TOLERANCE_OFF_BALL_MM:
        return StructuredSkillAction(skill_type=SkillType.stand)
    target_theta = math.atan2(dy, dx)
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
            obs, stage_id, prev_skill_name, gate_decision, coordination_target,
        )
    if role == "off_ball_support":
        return teacher_off_ball_support(
            obs, stage_id, prev_skill_name, gate_decision, coordination_target,
        )
    raise ValueError(f"Unknown role: {role!r}")


# ---------------------------------------------------------------------------
# Coordination target computation — called by env each step, result EMA-smoothed there
# ---------------------------------------------------------------------------

def compute_open_lane_target(
    open_support_obs: BackendObservation,
    ball_field_xy: tuple[float, float],
    striker_field_xy: tuple[float, float],
) -> tuple[float, float]:
    """Compute the raw (pre-EMA) open-lane receive position for the open_support.

    Strategy: pick the highest-scoring slot from SUPPORT_SLOT_CANDIDATE_GRID.
    The EMA smoothing (α=0.3) is applied in the env after this returns.
    Returns absolute field coordinates (mm).
    """
    _, best_cand = support_target(
        support_obs=open_support_obs,
        ball_field_xy=ball_field_xy,
        striker_field_xy=striker_field_xy,
        other_support_field_xy=None,
        current_slot=None,
        hold_tolerance_mm=HOLD_TOLERANCE_OPEN_MM,
    )
    return best_cand


def compute_triangle_target(
    off_ball_obs: BackendObservation,
    ball_field_xy: tuple[float, float],
    striker_field_xy: tuple[float, float],
    open_support_field_xy: tuple[float, float],
) -> tuple[float, float]:
    """Compute the raw (pre-EMA) triangulation vertex for the off_ball_support.

    Mirrors SUPPORT_SLOT_CANDIDATE_GRID y to the *opposite* side from open_support
    so the two supports spread out. The EMA (α=0.2) is applied in the env.
    Returns absolute field coordinates (mm).
    """
    # Flip y: off_ball goes to the side opposite of open_support.
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


# ---------------------------------------------------------------------------
# TeammateSlot builder — convenience helper for callers
# ---------------------------------------------------------------------------

def teammate_slots_from_obs(
    self_obs: BackendObservation,
    agent_id_map: dict[int, int],
) -> list[TeammateSlot]:
    """Build the TeammateSlot list from Block B obs indices (26:32).

    self_obs:     the calling agent's BackendObservation.
    agent_id_map: {slot_idx: agent_id} — caller maps Block-B slots to known agent IDs.
                  If an agent_id is not known for a slot, pass -1.

    Returns list of up to 3 TeammateSlot objects (only populated slots are included).
    """
    slots_data = [
        (0, float(self_obs.nearest_teammate_x_rel), float(self_obs.nearest_teammate_y_rel)),
        (1, float(self_obs.second_teammate_x_rel), float(self_obs.second_teammate_y_rel)),
        (2, float(self_obs.third_teammate_x_rel), float(self_obs.third_teammate_y_rel)),
    ]
    slots = []
    for slot_idx, x_rel, y_rel in slots_data:
        # Sentinel: slot is zero-filled when no teammate is present.
        if x_rel == 0.0 and y_rel == 0.0:
            continue
        x_field = float(self_obs.robot_x) + x_rel
        y_field = float(self_obs.robot_y) + y_rel
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
