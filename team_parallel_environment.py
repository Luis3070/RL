"""Team-coordinated multi-slot env for stage 6/7 MAPPO training.

`TeamSlotSimRobotEnv` extends `ParallelSlotSimRobotEnv` with the §14.17 env-side
team layer, mirroring what `teacher_multiagent_dataset.py` does at dataset-gen
time so rollout observations match the BC corpus distribution:

  * coordinated resets — ONE team scenario world per episode, transformed per
    slot (each slot's controlled player gets its own pose; everyone else spawns
    as teammates/dummies in that slot's sim);
  * cross-sim team-state injection at BHuman comm cadence (teammate poses,
    activities, team ball from the ball-authority slot's perception channel);
  * role assignment (striker / open_support / off_ball_support) with hysteresis
    + per-robot spawn-side latches for the support slot mirrors;
  * team-augmented gate decisions (pass_armed via smashOrPass, role one-hot)
    so obs[41:47] is live during rollouts;
  * role-aware residual anchors (the slot env decodes support walk residuals
    against the EMA coordination target — see
    RealSimRobotEnv._decode_residual_policy_action).

train.py is untouched: train_multiagent.py monkey-patches
`train.ParallelSlotSimRobotEnv` to this class for the duration of a multi-agent
run. Single-agent training never sees this code path.
"""
from __future__ import annotations

import math
import os
import threading
from collections import Counter
from dataclasses import replace as dc_replace

import numpy as np

from config import EnvConfig, GOAL_X, GOAL_Y_HALF
from gate import GateDecision
from multiagent_teacher import (
    EMA_ALPHA_OPEN_LANE,
    EMA_ALPHA_TRIANGLE,
    TeamCoordinationState,
    assign_roles,
    build_agent_id_map,
    build_team_gate_decision,
    defender_cover_target,
    defense_support_target,
    GK_EXCLUDE_X_MM,
    compute_open_lane_target,
    compute_triangle_target,
    smash_or_pass,
    support_is_live_pass_option,
    teammate_slots_from_obs,
    update_team_phase,
)
from multiagent_teacher import CONTINGENT_HOLD as TEACHER_CONTINGENT_HOLD
from parallel_environment import ParallelSlotSimRobotEnv
from pybh_backend import BackendObservation, build_world_spec

# Same env vars (and semantics) as teacher_multiagent_dataset.py — rollouts must
# carry the identical comm-cadence texture the BC corpus was generated with.
TEAM_COMM_INTERVAL_MS = float(os.getenv("RL_MA_TEACHER_TEAM_COMM_INTERVAL_MS", "600.0"))
TEAM_BALL_VALID_MAX_MS = float(os.getenv("RL_MA_TEACHER_TEAM_BALL_VALID_MAX_MS", "2000.0"))
STEP_DT_MS = float(os.getenv("RL_MA_TEACHER_STEP_DT_MS", "33.0"))

# §14.20 team reward terms, applied at the TEAM layer (reward.py stays frozen —
# single-agent reward shaping is untouched; these are additive bonuses on the
# slot rewards for stages ≥6 only). Per-step hold bonuses are deliberately
# conservative vs the §14.20 headline numbers (0.5 per-EVENT there ≈ 0.05
# per-step here) so they don't dominate the existing per-step reward scale.
TEAM_GOAL_BONUS = float(os.getenv("RL_MA_TEAM_GOAL_BONUS", "2.5"))
TEAM_GOAL_CONCEDED_PENALTY = float(os.getenv("RL_MA_TEAM_GOAL_CONCEDED_PENALTY", "2.5"))
# v3a (§Part1): supports collapsed to walk (.86) vs the teacher's stand (.66) because
# the hold bonus paid for mere proximity, not for actually standing — walking around
# inside the radius earned the same credit. And the open/triangle split (0.05/0.02)
# mirrored the measured hold gap (open 0.159 / triangle 0.003): the triangle bonus was
# too small to matter. Fixes: (1) require stand to collect (SUPPORT_HOLD_REQUIRE_STAND),
# (2) close the open/triangle gap and lift both, (3) penalise drifting back out.
SUPPORT_OPEN_HOLD_BONUS = float(os.getenv("RL_MA_SUPPORT_OPEN_HOLD_BONUS", "0.07"))
SUPPORT_TRIANGLE_HOLD_BONUS = float(os.getenv("RL_MA_SUPPORT_TRIANGLE_HOLD_BONUS", "0.06"))
SUPPORT_HOLD_REQUIRE_STAND = os.getenv("RL_MA_SUPPORT_HOLD_REQUIRE_STAND", "1") not in ("0", "false", "False")
SUPPORT_DRIFT_PENALTY = float(os.getenv("RL_MA_SUPPORT_DRIFT_PENALTY", "0.04"))
# v3b: the v3a drift penalty fired on leaving the radius after mere proximity, which
# taught a walk-biased support to AVOID the radius (triangle_hold collapsed to 0.0).
# Reworked to fire only after the support actually SETTLED — stood inside the radius
# for >= DRIFT_HOLD_MIN_STEPS consecutive steps — so it penalises abandoning a held
# slot, not passing through one.
DRIFT_HOLD_MIN_STEPS = int(os.getenv("RL_MA_DRIFT_HOLD_MIN_STEPS", "5"))
# v4-S1: reward COMPLETED passes, not merely executed ones. v3/v3.1 showed a flat
# executed-pass credit doesn't make the policy pass (pass collapsed to 0). The reward
# now fires only when, after a pass, the ball actually reaches a (non-GK) teammate's
# position within PASS_COMPLETION_WINDOW steps — i.e. an accurate pass to the open
# teammate. A small attempt credit is kept so attempting isn't punished, but the bulk
# of the reward is on completion. (Old PASS_ATTEMPT_CREDIT default was 1.0 on execute.)
PASS_ATTEMPT_CREDIT = float(os.getenv("RL_MA_PASS_ATTEMPT_CREDIT", "0.1"))
PASS_COMPLETED_REWARD = float(os.getenv("RL_MA_PASS_COMPLETED_REWARD", "1.5"))
PASS_COMPLETION_RADIUS_MM = float(os.getenv("RL_MA_PASS_COMPLETION_RADIUS_MM", "600.0"))
PASS_COMPLETION_WINDOW = int(os.getenv("RL_MA_PASS_COMPLETION_WINDOW", "60"))
# Stage-3 RECEIVER-LEAD (env-gated, default OFF). Diagnostic (2026-06-17): supports hold
# their static slots (~2.3m from the ball) and passes fire, but the ball undershoots and
# the receiver never closes the last ~1.2m (pass_min_dist_to_target_mm 1238, completions 0).
# When a pass is in flight, lead the support nearest the ball's predicted END position
# toward that arrival point (instead of holding its slot) so it can actually receive.
# Inference-only override of the coordination target; layered on v4.2 with no retrain.
RECEIVER_LEAD = os.getenv("RL_MA_RECEIVER_LEAD", "0").lower() in ("1", "true", "yes")
HOLD_RADIUS_MM = float(os.getenv("RL_MA_HOLD_RADIUS_MM", "600.0"))
# v4.4: DENSE navigate-to-slot shaping. The hold bonus above is BINARY (pays nothing
# until the support is already inside HOLD_RADIUS_MM and standing), so PPO gets no
# gradient to WALK to the slot — and BC is open-loop, so it clones the `stand` marginal
# but not the closed-loop positioning. Diagnosis (corpus, 2026-06-16): the teacher
# off_ball_support sits a median 300mm from triangle_ema and every standing frame is
# within 600mm, so the target is reachable; the trained policy just never gets there.
# This term adds a continuous reward that rises linearly as the support nears its EMA
# slot (open_lane / triangle), giving PPO the missing navigation gradient. Env-gated:
# default 0.0 => OFF, so single-agent and all prior validated runs are unaffected.
SUPPORT_APPROACH_BONUS = float(os.getenv("RL_MA_SUPPORT_APPROACH_BONUS", "0.0"))
SUPPORT_APPROACH_RANGE_MM = float(os.getenv("RL_MA_SUPPORT_APPROACH_RANGE_MM", "2500.0"))
# v3a (§Part1): conversion was borderline (shoot_finish ~0.17-0.25) because the only
# finishing signal was the sparse one-shot goal bonus; the striker dribbles a lot and
# rarely commits a shot. A small DENSE reward for taking a QUALITY shot (in the validated
# 2.5-3m band with an open lane) pushes the policy to finish more than the BC teacher.
# Kept small relative to TEAM_GOAL_BONUS so it shapes, not dominates (the goal bonus at
# 5.0 traded away collision discipline in v2 — same failure mode to avoid here).
FINISH_QUALITY_BONUS = float(os.getenv("RL_MA_FINISH_QUALITY_BONUS", "0.15"))
FINISH_SHOT_OPEN_MIN = float(os.getenv("RL_MA_FINISH_SHOT_OPEN_MIN", "0.30"))
FINISH_BAND_MIN_MM = float(os.getenv("RL_MA_FINISH_BAND_MIN_MM", "1500.0"))
FINISH_BAND_MAX_MM = float(os.getenv("RL_MA_FINISH_BAND_MAX_MM", "3200.0"))

# Side-channel for the BC competence gate: train.py's eval summarizer has a
# fixed metric list, so the team metrics can't flow through episode_metrics
# without touching train.py. The team env publishes its latest episode metrics
# here; MultiAgentBCCompetenceGate reads them when the summary lacks the keys.
LAST_TEAM_METRICS: dict[str, float] = {}

_ACTIVITY_IDLE = 0
_ACTIVITY_GOING_TO_BALL = 1
_ACTIVITY_DRIBBLING = 2
_ACTIVITY_KICKING = 3

_ROLE_NAMES = ("striker", "open_support", "off_ball_support")


def _activity_for(skill_name: str | None, role: str | None) -> int:
    if skill_name in ("shoot", "pass"):
        return _ACTIVITY_KICKING
    if skill_name == "dribble":
        return _ACTIVITY_DRIBBLING
    if skill_name == "walk" and role == "striker":
        return _ACTIVITY_GOING_TO_BALL
    return _ACTIVITY_IDLE


def _world_player_pose(world_spec, player_number: int) -> tuple[float, float, float]:
    if int(player_number) == 1:
        return float(world_spec.robot_x), float(world_spec.robot_y), float(world_spec.robot_theta)
    for spec in world_spec.teammates:
        if int(spec.number) == int(player_number):
            return float(spec.x), float(spec.y), float(spec.theta)
    raise KeyError(f"player {player_number} not present in WorldSpec")


def _world_spec_for_slot(world_spec, controlled_player_number: int):
    """Per-slot WorldSpec: the controlled player takes `robot_*`; every other
    same-team body (including player 1 and the BH GK) becomes a teammate dummy."""
    from pybh_backend import WorldPlayerSpec

    rx, ry, rtheta = _world_player_pose(world_spec, controlled_player_number)
    teammates = []
    if int(controlled_player_number) != 1:
        teammates.append(WorldPlayerSpec(
            number=1,
            x=float(world_spec.robot_x), y=float(world_spec.robot_y),
            theta=float(world_spec.robot_theta),
        ))
    for spec in world_spec.teammates:
        if int(spec.number) == int(controlled_player_number):
            continue
        teammates.append(WorldPlayerSpec(
            number=int(spec.number), x=float(spec.x), y=float(spec.y), theta=float(spec.theta),
        ))
    return build_world_spec(
        ball_pos=np.asarray([float(world_spec.ball_x), float(world_spec.ball_y)], dtype=np.float32),
        robot_pos=np.asarray([rx, ry], dtype=np.float32),
        robot_theta=float(rtheta),
        teammates=tuple(teammates),
        opponents=tuple(world_spec.opponents),
    )


class TeamSlotSimRobotEnv(ParallelSlotSimRobotEnv):
    """ParallelSlotSimRobotEnv + the §14.17 team layer."""

    metadata = {"name": "team-slot-simrobot-v1"}

    def __init__(self, env_cfg: EnvConfig):
        super().__init__(env_cfg)
        self._team_lock = threading.Lock()
        self._raw_obs: dict[int, BackendObservation] = {}
        self._team_state = TeamCoordinationState()
        self._comm_snapshot: dict | None = None
        self._team_step_idx = 0
        self._ball_authority: int | None = None
        self._spawn_side: dict[int, float] = {}
        self._roles: dict[int, str] = {}
        self._prev_skill: dict[int, str | None] = {pn: None for pn in self.player_numbers}
        self._static_teammates: tuple[tuple[int, float, float, float], ...] = ()
        self._world_spec = None
        # Team episode metrics (hold accounting per §14.19 competence keys).
        self._ep_role_steps: Counter = Counter()
        self._ep_hold_steps: Counter = Counter()
        self._ep_role_swaps = 0
        self._ep_defense_steps = 0
        self._ep_defender_cover_hold_steps = 0
        self._ep_defender_gk_collision_steps = 0
        # v3b drift penalty (P2): per-agent consecutive "stood inside hold radius"
        # streak. Drift penalty fires only when a streak >= DRIFT_HOLD_MIN_STEPS ends
        # by leaving the radius (settled-then-abandoned), not on mere proximity.
        self._hold_streak: dict[int, int] = {pn: 0 for pn in self.player_numbers}
        # v4-S1 completed-pass tracking: {passer_pn: {"targets":[(x,y)..], "steps_left":int}}
        self._pending_pass: dict[int, dict] = {}
        # Stage-3 pass-completion metric: team-wide per-episode counts. A pass is
        # ATTEMPTED when a `pass` skill registers a pending pass; it COMPLETES when the
        # ball reaches a (non-GK) teammate target within the window. Surfaced as
        # pass_attempts / pass_completions / pass_completion_rate (see _team_metrics).
        self._ep_pass_attempts = 0
        self._ep_pass_completions = 0
        self._ep_pass_min_dists: list[float] = []

        # Install the per-slot hooks.
        for agent, player_number in zip(self.possible_agents, self.player_numbers, strict=True):
            env = self._envs[agent]
            env._team_hook = self._make_hook(int(player_number))

    # ------------------------------------------------------------------
    # Per-slot hook — runs inside slot step threads (must be thread-safe).
    # ------------------------------------------------------------------

    def _make_hook(self, player_number: int):
        def hook(env, obs: BackendObservation, base: GateDecision):
            with self._team_lock:
                self._raw_obs[player_number] = obs
                snapshot = self._comm_snapshot
                roles = dict(self._roles)
                team_state = self._team_state
                step_idx = self._team_step_idx
            role = roles.get(player_number, "striker" if len(self.player_numbers) == 1 else "off_ball_support")

            injected = self._inject_for(obs, player_number, snapshot, step_idx)

            pass_armed = False
            if role == "striker" and (base.shoot_armed or base.dribble_armed):
                agent_positions = self._positions_from_snapshot(snapshot)
                agent_positions[player_number] = (float(obs.robot_x), float(obs.robot_y))
                ages = {pn: 0.0 for pn in agent_positions}
                agent_id_map = build_agent_id_map(player_number, agent_positions, ages)
                slots = teammate_slots_from_obs(injected, agent_id_map)
                proposal, _target = smash_or_pass(
                    injected, slots,
                    team_state.last_pass_target_by_striker.get(player_number),
                    team_state.last_pass_target_age_ms_by_striker.get(player_number, 0.0),
                )
                pass_armed = proposal == "pass"

            decision = build_team_gate_decision(base, role=role, pass_armed=pass_armed)

            # Role + coordination target for residual anchor decoding next step.
            coord = None
            if role == "open_support":
                coord = team_state.open_lane_ema
            elif role == "off_ball_support":
                coord = team_state.triangle_ema
            if coord is None and snapshot is not None:
                coord = snapshot["ball"]
            # Stage-3 receiver-lead: while a pass is in flight, lead the support that is the
            # actual receiver (its slot nearest the ball's predicted END) toward that arrival
            # point so it closes the undershoot gap; the other support keeps its slot to hold
            # spacing. Predicted end is robot-relative -> rotate into field frame (inverse of
            # _field_to_robot: field = robot + R(theta)·rel).
            if (RECEIVER_LEAD and coord is not None and self._pending_pass
                    and role in ("open_support", "off_ball_support")):
                rth = float(obs.robot_theta)
                c, s = math.cos(rth), math.sin(rth)
                end_x = float(obs.robot_x) + c * float(obs.ball_end_rel_x) - s * float(obs.ball_end_rel_y)
                end_y = float(obs.robot_y) + s * float(obs.ball_end_rel_x) + c * float(obs.ball_end_rel_y)
                open_ema = team_state.open_lane_ema
                tri_ema = team_state.triangle_ema
                d_open = math.hypot(open_ema[0] - end_x, open_ema[1] - end_y) if open_ema else float("inf")
                d_tri = math.hypot(tri_ema[0] - end_x, tri_ema[1] - end_y) if tri_ema else float("inf")
                is_receiver = (
                    (role == "open_support" and d_open <= d_tri)
                    or (role == "off_ball_support" and d_tri < d_open)
                )
                if is_receiver:
                    coord = (end_x, end_y)
            env._team_role = role
            env._team_coordination_target = coord
            return injected, decision

        return hook

    def _positions_from_snapshot(self, snapshot: dict | None) -> dict[int, tuple[float, float]]:
        positions: dict[int, tuple[float, float]] = {}
        if snapshot is not None:
            for pn, pose in snapshot["poses"].items():
                positions[pn] = (float(pose[0]), float(pose[1]))
        for num, sx, sy, _ in self._static_teammates:
            positions[num] = (sx, sy)
        return positions

    def _inject_for(
        self,
        obs: BackendObservation,
        player_number: int,
        snapshot: dict | None,
        step_idx: int,
    ) -> BackendObservation:
        if snapshot is None:
            return obs
        elapsed_ms = max(0.0, (step_idx - int(snapshot["step"])) * STEP_DT_MS)
        ball_x, ball_y = snapshot["ball"]
        ball_age = float(snapshot["ball_age_ms"]) + elapsed_ms
        poses: list[tuple[float, float, float]] = []
        ages: list[float] = []
        acts: list[int] = []
        for pn in sorted(snapshot["poses"]):
            if pn == player_number:
                continue
            poses.append(snapshot["poses"][pn])
            ages.append(elapsed_ms)
            acts.append(int(snapshot["activities"].get(pn, _ACTIVITY_IDLE)))
        for _num, sx, sy, stheta in self._static_teammates:
            poses.append((sx, sy, stheta))
            ages.append(elapsed_ms)
            acts.append(_ACTIVITY_IDLE)
        return dc_replace(
            obs,
            teammate_poses=tuple(poses),
            teammate_ages_ms=tuple(ages),
            teammate_activities=tuple(acts),
            global_team_ball_x_mm=float(ball_x),
            global_team_ball_y_mm=float(ball_y),
            global_team_ball_age_ms=ball_age,
            global_team_ball_valid=bool(ball_age <= TEAM_BALL_VALID_MAX_MS),
        )

    # ------------------------------------------------------------------
    # Team state refresh — runs on the wrapper thread between slot steps.
    # ------------------------------------------------------------------

    def _refresh_team_state(self) -> None:
        with self._team_lock:
            raw = dict(self._raw_obs)
        if not raw:
            return

        if self._ball_authority is None or self._ball_authority not in raw:
            self._ball_authority = min(
                sorted(raw),
                key=lambda pn: math.hypot(
                    float(raw[pn].ball_x) - float(raw[pn].robot_x),
                    float(raw[pn].ball_y) - float(raw[pn].robot_y),
                ),
            )
        auth = raw[self._ball_authority]
        # Finite-guard the perception ball: during reset settling the wire can
        # carry NaN. Fall back to the exported ball, then to the previous
        # snapshot's ball.
        bx, by = float(auth.estimated_ball_x), float(auth.estimated_ball_y)
        if not (math.isfinite(bx) and math.isfinite(by)):
            bx, by = float(auth.ball_x), float(auth.ball_y)
        if not (math.isfinite(bx) and math.isfinite(by)) and self._comm_snapshot is not None:
            bx, by = self._comm_snapshot["ball"]
        ball_age_raw = float(auth.natural_time_since_ball_seen)
        if not math.isfinite(ball_age_raw):
            ball_age_raw = TEAM_BALL_VALID_MAX_MS + 1.0
        team_ball = (bx, by, ball_age_raw <= TEAM_BALL_VALID_MAX_MS)

        phase = update_team_phase(self._team_state, team_ball)
        prev_roles = dict(self._roles)
        roles = assign_roles(raw, team_ball, self._team_state.roles, phase=phase)
        swaps = sum(
            1 for pn, role in roles.items() if prev_roles.get(pn) is not None and prev_roles[pn] != role
        )
        self._ep_role_swaps += swaps

        positions = {pn: (float(o.robot_x), float(o.robot_y)) for pn, o in raw.items()}
        for num, sx, sy, _stheta in self._static_teammates:
            positions[num] = (sx, sy)
        striker_id = next((pn for pn, r in roles.items() if r == "striker"), None)
        open_id = next((pn for pn, r in roles.items() if r == "open_support"), None)
        off_id = next((pn for pn, r in roles.items() if r == "off_ball_support"), None)

        if off_id is not None and striker_id is not None and phase == "defense":
            raw_tri, side = defender_cover_target(
                raw[off_id],
                (team_ball[0], team_ball[1]),
                current_side=self._team_state.defender_side_by_agent.get(off_id),
            )
            self._team_state.defender_side_by_agent[off_id] = side
            ema = self._team_state.triangle_ema
            self._team_state.triangle_ema = (
                raw_tri if ema is None
                else (
                    EMA_ALPHA_TRIANGLE * raw_tri[0] + (1 - EMA_ALPHA_TRIANGLE) * ema[0],
                    EMA_ALPHA_TRIANGLE * raw_tri[1] + (1 - EMA_ALPHA_TRIANGLE) * ema[1],
                )
            )
        if open_id is not None and striker_id is not None:
            raw_open = (
                defense_support_target(
                    raw[open_id],
                    (team_ball[0], team_ball[1]),
                    positions[striker_id],
                    positions[off_id],
                    forced_side=self._spawn_side.get(open_id),
                )
                if phase == "defense" and off_id is not None
                else compute_open_lane_target(
                    raw[open_id], (team_ball[0], team_ball[1]), positions[striker_id],
                    forced_side=self._spawn_side.get(open_id),
                )
            )
            ema = self._team_state.open_lane_ema
            self._team_state.open_lane_ema = (
                raw_open if ema is None
                else (
                    EMA_ALPHA_OPEN_LANE * raw_open[0] + (1 - EMA_ALPHA_OPEN_LANE) * ema[0],
                    EMA_ALPHA_OPEN_LANE * raw_open[1] + (1 - EMA_ALPHA_OPEN_LANE) * ema[1],
                )
            )
        if off_id is not None and striker_id is not None and open_id is not None and phase != "defense":
            raw_tri = compute_triangle_target(
                raw[off_id], (team_ball[0], team_ball[1]),
                positions[striker_id], positions[open_id],
                forced_side=self._spawn_side.get(off_id),
            )
            ema = self._team_state.triangle_ema
            self._team_state.triangle_ema = (
                raw_tri if ema is None
                else (
                    EMA_ALPHA_TRIANGLE * raw_tri[0] + (1 - EMA_ALPHA_TRIANGLE) * ema[0],
                    EMA_ALPHA_TRIANGLE * raw_tri[1] + (1 - EMA_ALPHA_TRIANGLE) * ema[1],
                )
            )

        # Comm snapshot at BHuman cadence.
        refresh = (
            self._comm_snapshot is None
            or (self._team_step_idx - int(self._comm_snapshot["step"])) * STEP_DT_MS >= TEAM_COMM_INTERVAL_MS
        )
        snapshot = self._comm_snapshot
        if refresh:
            snapshot = {
                "step": self._team_step_idx,
                "poses": {
                    pn: (float(o.robot_x), float(o.robot_y), float(o.robot_theta))
                    for pn, o in raw.items()
                },
                "activities": {
                    pn: _activity_for(self._prev_skill.get(pn), roles.get(pn)) for pn in raw
                },
                "ball": (bx, by),
                "ball_age_ms": ball_age_raw if math.isfinite(ball_age_raw) else 0.0,
            }

        # Hold accounting (competence-gate metrics).
        for pn, role in roles.items():
            self._ep_role_steps[role] += 1
            target = None
            if self._team_state.phase == "defense":
                self._ep_defense_steps += 1
            if role == "open_support":
                target = self._team_state.open_lane_ema
            elif role == "off_ball_support":
                target = self._team_state.triangle_ema
            if target is not None:
                dist = math.hypot(positions[pn][0] - target[0], positions[pn][1] - target[1])
                if dist < 600.0:
                    self._ep_hold_steps[role] += 1
                    if role == "off_ball_support" and self._team_state.phase == "defense":
                        self._ep_defender_cover_hold_steps += 1
            if role == "off_ball_support" and self._team_state.phase == "defense":
                gk = self._own_gk_from_positions(positions)
                if gk is not None and math.hypot(positions[pn][0] - gk[0], positions[pn][1] - gk[1]) < 650.0:
                    self._ep_defender_gk_collision_steps += 1

        with self._team_lock:
            self._roles = roles
            self._comm_snapshot = snapshot

    def _team_metrics(self) -> dict[str, float]:
        open_steps = max(1, self._ep_role_steps.get("open_support", 0))
        off_steps = max(1, self._ep_role_steps.get("off_ball_support", 0))
        total = max(1, sum(self._ep_role_steps.values()))
        defense_steps = max(1, self._ep_defense_steps)
        return {
            "support_open_hold_rate": float(self._ep_hold_steps.get("open_support", 0)) / open_steps,
            "support_triangle_hold_rate": float(self._ep_hold_steps.get("off_ball_support", 0)) / off_steps,
            "defender_cover_hold_rate": float(self._ep_defender_cover_hold_steps) / defense_steps,
            "defender_gk_collision_rate": float(self._ep_defender_gk_collision_steps) / defense_steps,
            "defense_phase_step_frac": float(self._ep_defense_steps) / total,
            "role_thrash_rate": float(self._ep_role_swaps) / total,
            # Stage-3 pass-completion. pass_attempts/pass_completions are per-episode
            # COUNTS (averaged across eval episodes -> mean attempts/completions per
            # episode); the unbiased aggregate completion rate is mean(pass_completions)
            # / mean(pass_attempts). pass_completion_rate is the per-episode ratio
            # (0.0 when no pass was attempted) for a quick read.
            "pass_attempts": float(self._ep_pass_attempts),
            "pass_completions": float(self._ep_pass_completions),
            "pass_completion_rate": (
                float(self._ep_pass_completions) / float(self._ep_pass_attempts)
                if self._ep_pass_attempts > 0 else 0.0
            ),
            # Diag: mean closest-approach (mm) of the ball to a pass target across
            # resolved passes. ~radius => near-miss (widen radius / lead the receiver);
            # large => ball never travels to a teammate (pass_target/decode issue).
            "pass_min_dist_to_target_mm": (
                float(sum(self._ep_pass_min_dists)) / len(self._ep_pass_min_dists)
                if self._ep_pass_min_dists else 0.0
            ),
        }

    def _own_gk_from_positions(self, positions: dict[int, tuple[float, float]]) -> tuple[float, float] | None:
        candidates = [pos for pn, pos in positions.items() if pn not in self.player_numbers and pos[0] < -2000.0]
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p[0] + GOAL_X) + abs(p[1]))

    # ------------------------------------------------------------------
    # ParallelEnv overrides
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        # ONE world per team episode, sampled from the (team) production pool of
        # the first slot env — same machinery, same env-var weights.
        first_env = next(iter(self._envs.values()))
        if first_env._production_scenarios:
            scenario = first_env._sample_production_scenario()
            world = scenario.world_fn(np.random.default_rng(None if seed is None else int(seed)))
            stage_id = int(scenario.stage_id)
        else:
            raise RuntimeError(
                "TeamSlotSimRobotEnv requires RL_PRODUCTION_SCENARIO_RESETS=1 and "
                "RL_PRODUCTION_SCENARIO_SET=team (team scenario pool empty)."
            )

        self._world_spec = world
        self._static_teammates = tuple(
            (int(spec.number), float(spec.x), float(spec.y), float(spec.theta))
            for spec in world.teammates
            if int(spec.number) not in self.player_numbers
        )
        # Fresh per-episode team state. Initial roles from world geometry so the
        # reset hooks have a sane role before the first refresh.
        self._team_state = TeamCoordinationState()
        # Pseudo-snapshot from the WORLD SPEC so the reset-frame obs are injected
        # too: the raw wire carries NaN in the TeammatesBallModel fields (no team
        # comm in slot sims) and an un-injected reset frame poisons the policy
        # logits. The dataset rows are always injected; the env must match.
        self._comm_snapshot = {
            "step": 0,
            "poses": {
                pn: _world_player_pose(world, pn) for pn in self.player_numbers
            },
            "activities": {pn: _ACTIVITY_IDLE for pn in self.player_numbers},
            "ball": (float(world.ball_x), float(world.ball_y)),
            "ball_age_ms": 0.0,
        }
        self._team_step_idx = 0
        self._ball_authority = None
        self._ep_role_steps = Counter()
        self._ep_hold_steps = Counter()
        self._ep_role_swaps = 0
        self._ep_defense_steps = 0
        self._ep_defender_cover_hold_steps = 0
        self._ep_defender_gk_collision_steps = 0
        self._hold_streak = {pn: 0 for pn in self.player_numbers}
        self._pending_pass = {}
        self._ep_pass_attempts = 0
        self._ep_pass_completions = 0
        self._ep_pass_min_dists = []
        self._prev_skill = {pn: None for pn in self.player_numbers}
        poses = {pn: _world_player_pose(world, pn) for pn in self.player_numbers}
        self._spawn_side = {pn: (1.0 if poses[pn][1] >= 0.0 else -1.0) for pn in poses}
        by_ball_dist = sorted(
            self.player_numbers,
            key=lambda pn: math.hypot(poses[pn][0] - float(world.ball_x), poses[pn][1] - float(world.ball_y)),
        )
        phase = update_team_phase(self._team_state, (float(world.ball_x), float(world.ball_y), True))
        roles: dict[int, str] = {by_ball_dist[0]: "striker"}
        rest = list(by_ball_dist[1:])
        if phase == "defense" and rest:
            defender = min(rest, key=lambda pn: (poses[pn][0], pn))
            for pn in rest:
                roles[pn] = "off_ball_support" if pn == defender else "open_support"
        else:
            rest = sorted(rest, key=lambda pn: -poses[pn][0])
            if rest:
                roles[rest[0]] = "open_support"
            for pn in rest[1:]:
                roles[pn] = "off_ball_support"
        with self._team_lock:
            self._roles = roles
            self._raw_obs = {}

        # Reset every slot against its per-player transform of the SAME world.
        self.agents = list(self.possible_agents)
        observations: dict[str, object] = {}
        infos: dict[str, dict] = {}
        for agent, player_number in zip(self.possible_agents, self.player_numbers, strict=True):
            slot_options = dict(options)
            slot_options["world_spec"] = _world_spec_for_slot(world, int(player_number))
            slot_options["stage_id"] = stage_id
            obs, info = self._envs[agent].reset(seed=seed, options=slot_options)
            observations[agent] = obs["player_1"]
            infos[agent] = dict(info["player_1"])
            infos[agent]["parallel_agent_id"] = agent
            infos[agent]["team_role"] = roles.get(int(player_number), "")

        # First refresh so step 1 hooks see a real snapshot.
        self._refresh_team_state()
        return observations, infos

    def step(self, actions):
        self._team_step_idx += 1
        self._refresh_team_state()
        try:
            observations, rewards, terminations, truncations, infos = super().step(actions)
        except RuntimeError as exc:
            # Sim stall (pybh.rl_get_obs timeout — both the "stale visible frames"
            # and the generic "timed out in backend" variants, raised from step or
            # set_dynamic_world). Truncate the team episode instead of killing the
            # run — transient stalls recover on the next reset; hard freezes will
            # fail the reset and surface.
            msg = str(exc)
            if "rl_get_obs" not in msg and "stale visible frames" not in msg:
                raise
            print(f"[team-env] slot stall — truncating episode: {exc}", flush=True)
            self.agents = []
            obs = {
                agent: (env._last_obs if env._last_obs is not None
                        else np.zeros(env.observation_spaces["player_1"].shape, dtype=np.float32))
                for agent, env in self._envs.items()
            }
            zero_r = {agent: 0.0 for agent in self._envs}
            return (
                obs, zero_r,
                {agent: False for agent in self._envs},
                {agent: True for agent in self._envs},
                {agent: {"team_stall_truncation": True} for agent in self._envs},
            )

        # ---- §14.20 team reward terms (stages ≥6; additive on slot rewards) ----
        team_goal = False
        goal_conceded = False
        for agent in self.possible_agents:
            env = self._envs[agent]
            if terminations.get(agent) and bool(env._episode_tracker.get("goal", False)):
                team_goal = True
            raw = env._last_backend_obs
            if raw is not None and float(raw.ball_x) <= -GOAL_X and abs(float(raw.ball_y)) <= GOAL_Y_HALF:
                goal_conceded = True
        with self._team_lock:
            roles_now = dict(self._roles)
        for agent, player_number in zip(self.possible_agents, self.player_numbers, strict=True):
            if agent not in rewards:
                continue
            bonus = 0.0
            role = roles_now.get(int(player_number), "")
            env = self._envs[agent]
            raw = env._last_backend_obs
            info = infos.get(agent, {})
            executed_skill = (info.get("executed_skill_name") or "")
            if raw is not None and role in ("open_support", "off_ball_support"):
                target = (
                    self._team_state.open_lane_ema if role == "open_support"
                    else self._team_state.triangle_ema
                )
                if target is not None:
                    dist = math.hypot(float(raw.robot_x) - target[0], float(raw.robot_y) - target[1])
                    # v4.4: dense navigation gradient toward the slot (any skill), so PPO
                    # is pulled to WALK to the EMA target before the binary hold bonus can
                    # ever pay. Linear ramp: full at dist=0, zero at >= SUPPORT_APPROACH_RANGE_MM.
                    if SUPPORT_APPROACH_BONUS > 0.0:
                        bonus += SUPPORT_APPROACH_BONUS * max(0.0, 1.0 - dist / SUPPORT_APPROACH_RANGE_MM)
                    in_radius = dist < HOLD_RADIUS_MM
                    # P1: pay the hold bonus only when the support is actually holding
                    # position (executed stand), not merely passing through the radius.
                    is_standing = (not SUPPORT_HOLD_REQUIRE_STAND) or executed_skill == "stand"
                    # v3 (doc-35 §3): contingent hold — only reward holding when the
                    # support is a LIVE PASS OPTION (striker controls + lane open), so
                    # the reward matches the contingent teacher and stops paying for
                    # holding in the void. Attack phase only — in defense the off_ball/
                    # open roles are defender/defense-supporter (cover, not pass outlets),
                    # so they keep the geometric hold bonus.
                    contingent = TEACHER_CONTINGENT_HOLD and self._team_state.phase != "defense"
                    live_option = (not contingent) or support_is_live_pass_option(raw)
                    held = in_radius and is_standing and live_option
                    if held:
                        bonus += (
                            SUPPORT_OPEN_HOLD_BONUS if role == "open_support"
                            else SUPPORT_TRIANGLE_HOLD_BONUS
                        )
                    # P2 (v3b): drift penalty only after the support SETTLED — stood in
                    # the radius for >= DRIFT_HOLD_MIN_STEPS — then left. Penalising mere
                    # proximity (v3a) taught avoidance; this penalises abandoning a held
                    # slot. Walking through (in radius but not standing) just resets.
                    pn = int(player_number)
                    prev_streak = self._hold_streak.get(pn, 0)
                    if held:
                        self._hold_streak[pn] = prev_streak + 1
                    else:
                        self._hold_streak[pn] = 0
                        if (not in_radius) and prev_streak >= DRIFT_HOLD_MIN_STEPS:
                            bonus -= SUPPORT_DRIFT_PENALTY
                else:
                    self._hold_streak[int(player_number)] = 0
            else:
                self._hold_streak[int(player_number)] = 0
            # P6: dense finishing reward for the striker taking a quality shot in band.
            if role == "striker" and executed_skill == "shoot" and raw is not None:
                shot_open = float(getattr(raw, "shot_opening_with_obstacles", 0.0))
                ball_to_goal = math.hypot(float(GOAL_X) - float(raw.ball_x), float(raw.ball_y))
                if (
                    shot_open >= FINISH_SHOT_OPEN_MIN
                    and FINISH_BAND_MIN_MM <= ball_to_goal <= FINISH_BAND_MAX_MM
                ):
                    bonus += FINISH_QUALITY_BONUS
            # v4-S1: completed-pass reward. A pending pass completes if the ball reaches
            # within PASS_COMPLETION_RADIUS of any (non-GK) teammate position it was aimed
            # at, within PASS_COMPLETION_WINDOW steps; otherwise it expires (no reward).
            pn = int(player_number)
            if pn in self._pending_pass and raw is not None:
                p = self._pending_pass[pn]
                ball = (float(raw.ball_x), float(raw.ball_y))
                cur_min = min(math.hypot(ball[0] - tx, ball[1] - ty) for (tx, ty) in p["targets"])
                # Diag: closest approach of the ball to any pass target over the pass's life.
                # Distinguishes "near-miss (>radius)" from "ball never travels to a teammate".
                p["min_dist"] = min(p.get("min_dist", float("inf")), cur_min)
                reached = cur_min < PASS_COMPLETION_RADIUS_MM
                if reached:
                    bonus += PASS_COMPLETED_REWARD
                    self._ep_pass_completions += 1
                    self._ep_pass_min_dists.append(p["min_dist"])
                    del self._pending_pass[pn]
                else:
                    p["steps_left"] -= 1
                    if p["steps_left"] <= 0:
                        self._ep_pass_min_dists.append(p["min_dist"])
                        del self._pending_pass[pn]
            if executed_skill == "pass":
                bonus += PASS_ATTEMPT_CREDIT  # small attempt shaping; bulk reward is on completion
                if raw is not None and pn not in self._pending_pass:
                    targets = [(float(t[0]), float(t[1])) for t in getattr(raw, "teammate_poses", ())
                               if float(t[0]) >= GK_EXCLUDE_X_MM]  # non-GK injected teammate positions
                    if targets:
                        self._pending_pass[pn] = {"targets": targets, "steps_left": PASS_COMPLETION_WINDOW,
                                                  "min_dist": float("inf")}
                        self._ep_pass_attempts += 1
            if team_goal:
                bonus += TEAM_GOAL_BONUS
            if goal_conceded:
                bonus -= TEAM_GOAL_CONCEDED_PENALTY
            if bonus:
                rewards[agent] = float(rewards[agent]) + bonus
                infos.setdefault(agent, {})["team_reward_bonus"] = bonus

        # Track executed skills (for the activity encoding) and decay pass-target
        # ages. NOTE: pass-target recording (R4 hysteresis) is skipped at rollout
        # time — the executed pass_target is not surfaced in info. Effect is a
        # slightly weaker own-value penalty in smash_or_pass; the obs/gate
        # contract is unaffected.
        from multiagent_teacher import advance_pass_target_age

        advance_pass_target_age(self._team_state, STEP_DT_MS)
        for agent, player_number in zip(self.possible_agents, self.player_numbers, strict=True):
            info = infos.get(agent)
            if not info:
                continue
            skill = info.get("executed_skill_name") or info.get("policy_skill_name")
            if skill:
                self._prev_skill[int(player_number)] = str(skill)
            info["team_role"] = self._roles.get(int(player_number), "")

        done_any = any(terminations.values()) or any(truncations.values())
        if done_any:
            metrics = self._team_metrics()
            metrics["team_goal"] = float(team_goal)
            metrics["goals_conceded_rate"] = float(goal_conceded)
            for agent in infos:
                infos[agent].update(metrics)
                # P0: also fold the team metrics into episode_metrics so the eval
                # aggregator (which averages result["episode_metrics"][key] across
                # episodes) sees a per-episode hold-rate and reports the MEAN — not
                # the last-episode-only value the competence gate used to read from
                # LAST_TEAM_METRICS.
                ep_metrics = infos[agent].setdefault("episode_metrics", {})
                ep_metrics.update(metrics)
            # Publish for the BC competence gate (side-channel — see module docstring).
            LAST_TEAM_METRICS.clear()
            LAST_TEAM_METRICS.update(metrics)
        return observations, rewards, terminations, truncations, infos
