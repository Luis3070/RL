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
    compute_open_lane_target,
    compute_triangle_target,
    smash_or_pass,
    teammate_slots_from_obs,
    update_team_phase,
)
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
SUPPORT_OPEN_HOLD_BONUS = float(os.getenv("RL_MA_SUPPORT_OPEN_HOLD_BONUS", "0.05"))
SUPPORT_TRIANGLE_HOLD_BONUS = float(os.getenv("RL_MA_SUPPORT_TRIANGLE_HOLD_BONUS", "0.02"))
PASS_ATTEMPT_CREDIT = float(os.getenv("RL_MA_PASS_ATTEMPT_CREDIT", "1.0"))
HOLD_RADIUS_MM = float(os.getenv("RL_MA_HOLD_RADIUS_MM", "600.0"))

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
            if raw is not None and role in ("open_support", "off_ball_support"):
                target = (
                    self._team_state.open_lane_ema if role == "open_support"
                    else self._team_state.triangle_ema
                )
                if target is not None:
                    dist = math.hypot(float(raw.robot_x) - target[0], float(raw.robot_y) - target[1])
                    if dist < HOLD_RADIUS_MM:
                        bonus += (
                            SUPPORT_OPEN_HOLD_BONUS if role == "open_support"
                            else SUPPORT_TRIANGLE_HOLD_BONUS
                        )
            info = infos.get(agent, {})
            if (info.get("executed_skill_name") or "") == "pass":
                bonus += PASS_ATTEMPT_CREDIT
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
            # Publish for the BC competence gate (side-channel — see module docstring).
            LAST_TEAM_METRICS.clear()
            LAST_TEAM_METRICS.update(metrics)
        return observations, rewards, terminations, truncations, infos
