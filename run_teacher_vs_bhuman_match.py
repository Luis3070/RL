"""D2 live match harness: the HEURISTIC TEACHER (multiagent_teacher.team_teacher_payload)
vs real B-Human. A copy of run_v5_vs_bhuman_match.py with only the action source swapped.

Question it answers: is the teacher itself good enough to beat B-Human in a real,
continuous, shared-sim match? If the teacher also takes the ball only ~4-6% of the time,
imitating it harder cannot win, and the limit is the teacher / SkillRequest interface.

Everything else is identical to the v5 harness on purpose (scene, roles, EMA coordination
targets, gate + harness shoot-sanity clamp, 8 Hz loop, GameController score, the same
step-log schema) so the two batches are directly comparable with
tools/summarize_v5_vs_bhuman_matches.py. Differences:
  * the teacher's absolute action is executed; the v5 force-shoot / force-walk crutches
    are NOT applied (the teacher has no logits to override);
  * the v5 policy is still evaluated on every agent's encoded obs and logged as a SHADOW
    (`shadow_skill`, `shadow_logits`) -- it never acts. This gives live student-vs-teacher
    agreement on the states the live match actually produces (D1 on live states).

Original v5 harness description follows.

One process invocation drives ONE match. Players 2 and 3 on team 24 are controlled by
the v5 merged checkpoint (`runs/merge_v5_attack_defense/checkpoint_stage8_step10000.pt`,
the same weights exported to `ppo_team_hsl2026_v5_merged.onnx`, parity validated to
7.6e-6 per docs/paper production.md); player 1 (GK) and the entire opposing team run
unmodified B-Human/Sabana-Herons behavior.

This reuses the exact per-agent role/gate/policy-decode pipeline already proven in the
"eval mode" branch of teacher_multiagent_dataset.py::_collect_team_scenario (the same
code path that produced the 2026-08-05 paper_match_v5 run) — imported, not
reimplemented — but points it at a live, continuous, REAL-opponent scene instead of the
parallel-slot dummy-opponent training scenarios. The cross-sim team-comm injection used
there (_refresh_comm_snapshot/_inject_team_state) is intentionally NOT used here: in
this scene all robots share one sim with real B-Human team communication, so
obs.teammate_poses / obs.global_team_ball_* are already populated natively.

Usage:
  python run_teacher_vs_bhuman_match.py --launch --seconds 600 --output runs/.../match_1.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

RL_OVERRIDE_TEAM = 24
DEFAULT_ACTIVE_PLAYERS = (2, 3)
os.environ.setdefault("PYBH_RL_OVERRIDE_TEAM", str(RL_OVERRIDE_TEAM))
os.environ.setdefault("PYBH_RL_ACTIVE_PLAYERS", ",".join(str(p) for p in DEFAULT_ACTIVE_PLAYERS))

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import BHUMAN_DIR, SIMROBOT_BIN  # noqa: E402

# Importing teacher_multiagent_dataset transitively chdirs into BHUMAN_DIR and loads
# pybh.controller (see its module docstring / pybh_backend.py). Do this before any
# other RL-repo-relative path use.
import teacher_multiagent_dataset as tmd  # noqa: E402
from pybh_backend import BackendObservation  # noqa: E402
from action import SKILL_ORDER, SkillType, StructuredSkillAction  # noqa: E402
from train import gate_skill_mask_from_obs  # noqa: E402
from multiagent_teacher import (  # noqa: E402
    TeamCoordinationState,
    assign_roles,
    update_team_phase,
    defender_cover_target,
    defense_support_target,
    compute_open_lane_target,
    compute_triangle_target,
    defender_block_armed,
    defender_clear_armed,
    smash_or_pass,
    advance_pass_target_age,
    build_agent_id_map,
    teammate_slots_from_obs,
    build_team_gate_decision,
    team_teacher_payload,
    record_pass_target,
    BALL_CONTROL_RADIUS_MM,
    GOAL_X,
    OWN_GOAL_X_MM,
    OWN_GOAL_POST_Y_MM,
    EMA_ALPHA_TRIANGLE,
    EMA_ALPHA_OPEN_LANE,
)
from gate import SkillGate  # noqa: E402

import pybh.controller as pybh  # noqa: E402  (already loaded transitively; cheap re-import)


DEFAULT_SCENE = BHUMAN_DIR / "Config/Scenes/RLvsBH3v3_3D.ros2"
DEFAULT_CHECKPOINT = Path("/home/luis/GitHub/RL/runs/merge_v5_attack_defense/checkpoint_stage8_step10000.pt")
DEFAULT_LOOP_HZ = 8.0
DEFAULT_TIMEOUT_MS = 1200
DEFAULT_MAX_WAIT_S = 30.0

# Goal-line proxy (ball-crossing-line, per the agreed measurement method — there is no
# score field in the Python<->C++ bridge). GOAL_X/OWN_GOAL_X_MM/OWN_GOAL_POST_Y_MM come
# from multiagent_teacher.py, the same constants the defensive-clear teacher logic uses,
# so this proxy stays consistent with the rest of the pipeline's world-frame convention.
GOAL_LINE_MARGIN_MM = 100.0
GOAL_COOLDOWN_S = 3.0
KICKOFF_RETURN_RADIUS_MM = 700.0

_DIAG_PASS_PRINTS = 0  # one-off instrumentation: why does pass never arm? capped print count

# --- Inference-only repair overrides (strategy 1+3, no retraining, no checkpoint change) ---
# Live matches showed the policy's own shoot logit never wins the argmax even when the
# gate legitimately arms it (0% across ~150 armed steps in the 10-match batches), and
# open/off-ball support roles spend >50% of steps simply standing. Both are patched here
# at the wrapper level: the gate already guarantees a forced shoot is legal/safe (on-ball,
# aligned, in range, lane open) and a forced walk targets the same EMA coordination anchor
# the policy itself would have decoded to, so neither override can produce a nonsensical
# action -- they only stop the policy from dithering past an already-legal opportunity.
FORCE_SHOOT_AFTER_FRAMES = 1   # force on the first declined armed frame: measured against
# the existing 10x4v4 match batch, the armed window never survives more than 2 consecutive
# frames before self-disarming (the policy's own walk/stand action breaks the tight on-ball
# envelope on the very next step) -- a higher threshold would simply never fire.
FORCE_WALK_AFTER_FRAMES = 8    # ~1.0s of pure standing for a non-striker role

# --- Stuck-armed sanity clamp (bug found in the repair1_4v4 validation batch) ---
# gate.py's own range_enter/range_exit check (goal_dist <= ~3000-3200mm) should make it
# impossible for shoot_armed to be True while the ball is thousands of mm from goal, but
# match_2/agent_2 in repair1_4v4 stayed shoot_armed=True for a sustained 3+s window (24+
# consecutive control frames) while positioned 7000-7700mm from goal, deep in the RL
# team's OWN half, with no opponent even visible (nearest_opponent_dist pinned at the
# 9000mm "not seen" sentinel) -- 26 of the batch's 33 "shots" came from this one runaway
# episode. Root cause not fully isolated (would need robot_theta/can_score_now/
# natural_time_since_ball_seen logged per-step, which the harness doesn't currently
# record) in the time available. This is a redundant, independent distance check applied
# in the harness (not gate.py, which stays untouched/shared with training) that forces
# shoot_armed=False whenever the ball the harness itself measures is not actually near
# goal -- a correct result regardless of what caused the gate's internal state to stick.
SHOOT_SANITY_MAX_GOAL_DIST_MM = 3300.0  # shoot_goal_exit_mm (3200) + small margin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live match: v5 merged RL policy (players 2,3) vs real B-Human, RLvsBH3v3_3D."
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--simrobot-bin", type=Path, default=SIMROBOT_BIN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--launch", action="store_true", help="Launch SimRobot with the scene.")
    parser.add_argument("--seconds", type=float, default=600.0, help="Match duration (default: one SPL half).")
    parser.add_argument("--hz", type=float, default=DEFAULT_LOOP_HZ)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument(
        "--active-players", type=str,
        default=",".join(str(p) for p in DEFAULT_ACTIVE_PLAYERS),
        help="Comma-separated RL jersey numbers on team 24.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path for the raw step log.")
    parser.add_argument(
        "--teacher-stage", type=int, default=8,
        help="stage_id passed to team_teacher_payload (8 = attack + defense logic, as in the v5 corpus).",
    )
    parser.add_argument(
        "--bh-macro-radius-mm", type=float, default=0.0,
        help="D3: if > 0, the striker that is our nearest robot to a fresh ball within this radius plays "
             "B-Human's own ball macro (smash_or_pass -> shoot or passTo) instead of the teacher's gated "
             "walk/dribble. 0 = off (pure D2 teacher).",
    )
    return parser.parse_args()


def parse_player_numbers(raw: str) -> tuple[int, ...]:
    players: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        player = int(token)
        if not (1 <= player <= 20):
            raise ValueError(f"player numbers must be within 1..20, got {player}")
        players.append(player)
    if not players:
        raise ValueError("active player list cannot be empty")
    return tuple(dict.fromkeys(players))


def maybe_launch_simrobot(simrobot_bin: Path, scene: Path, active_players: tuple[int, ...]) -> subprocess.Popen | None:
    if not simrobot_bin.exists():
        raise FileNotFoundError(f"SimRobot binary not found: {simrobot_bin}")
    if not scene.exists():
        raise FileNotFoundError(f"Scene file not found: {scene}")
    env = os.environ.copy()
    env["PYBH_RL_OVERRIDE_TEAM"] = str(RL_OVERRIDE_TEAM)
    env["PYBH_RL_ACTIVE_PLAYERS"] = ",".join(str(p) for p in active_players)
    return subprocess.Popen(
        [str(simrobot_bin), str(scene)],
        cwd=BHUMAN_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def fetch_obs(player_number: int, timeout_ms: int) -> BackendObservation:
    return BackendObservation.from_payload(dict(pybh.rl_get_obs(player_number, timeout_ms)))


def wait_for_initial_observations(
    player_numbers: tuple[int, ...], timeout_ms: int, max_wait_s: float
) -> dict[int, BackendObservation]:
    deadline = time.monotonic() + max_wait_s
    ready: dict[int, BackendObservation] = {}
    while time.monotonic() < deadline:
        ready = {}
        for player in player_numbers:
            obs = fetch_obs(player, timeout_ms)
            if obs.obs_ready:
                ready[player] = obs
        if len(ready) == len(player_numbers):
            return ready
    missing = [str(p) for p in player_numbers if p not in ready]
    raise TimeoutError(f"Timed out waiting for initial observations for players: {', '.join(missing)}")


def send_action(player_number: int, action: StructuredSkillAction) -> None:
    """Mirrors PyBHBackend._set_action's skill-name translation (pybh_backend.py)."""
    skill_name = action.skill_name
    pybh_name = "walkTo" if skill_name == "walk" else skill_name
    kwargs = dict(
        player_number=player_number,
        skill=pybh_name,
        target_x=float(action.target_x),
        target_y=float(action.target_y),
        target_theta=float(action.target_theta),
    )
    if action.skill is SkillType.pass_:
        try:
            pybh.rl_set_action(pass_target=int(action.pass_target), **kwargs)
            return
        except TypeError:
            pass
    pybh.rl_set_action(**kwargs)


class GoalDetector:
    """Ball-crossing-goal-line proxy with a kickoff-return confirmation.

    A goal is only confirmed once the ball is seen crossing the goal line AND then
    returns near the center circle within a few seconds (the GameController kickoff
    reset) — this rejects the case where the ball merely rolls out near a post without
    a goal being given.
    """

    def __init__(self) -> None:
        self._pending_side: str | None = None  # "for" | "against"
        self._pending_since: float | None = None
        self._cooldown_until: float = 0.0
        self.scored_for = 0
        self.scored_against = 0
        self.events: list[dict] = []

    def update(self, *, t: float, ball_x: float, ball_y: float, valid: bool = True) -> None:
        if not valid or t < self._cooldown_until:
            return
        if self._pending_side is None:
            if ball_x >= GOAL_X - GOAL_LINE_MARGIN_MM and abs(ball_y) <= OWN_GOAL_POST_Y_MM:
                self._pending_side = "for"
                self._pending_since = t
            elif ball_x <= OWN_GOAL_X_MM + GOAL_LINE_MARGIN_MM and abs(ball_y) <= OWN_GOAL_POST_Y_MM:
                self._pending_side = "against"
                self._pending_since = t
            return
        # Waiting for kickoff confirmation (ball back near center).
        if math.hypot(ball_x, ball_y) <= KICKOFF_RETURN_RADIUS_MM:
            if self._pending_side == "for":
                self.scored_for += 1
            else:
                self.scored_against += 1
            self.events.append({"t": t, "side": self._pending_side})
            self._cooldown_until = t + GOAL_COOLDOWN_S
            self._pending_side = None
            self._pending_since = None
        elif self._pending_since is not None and (t - self._pending_since) > 15.0:
            # Ball never returned to center — treat as a false positive (e.g. it was
            # cleared back out) and drop the pending goal without counting it.
            self._pending_side = None
            self._pending_since = None


def _correct_is_gk(teammate_slots: list, obs: BackendObservation) -> list:
    """teammate_slots_from_obs infers is_gk from GK_EXCLUDE_X_MM (a training-time proxy
    tuned for the synthetic 3-RL-field-player setup) — in this 2-field-player live
    harness the sole support's own defensive coordination target legitimately dips
    below that threshold and gets misclassified as the goalkeeper, which then makes
    smash_or_pass skip it as a pass candidate every time. obs.teammate_is_gk carries
    B-Human's real isGoalkeeper flag (position-matched against obs.teammate_poses,
    since sorting order is not guaranteed to match); prefer it when present rather
    than editing the shared multiagent_teacher.py heuristic used by training."""
    if not obs.teammate_is_gk:
        return teammate_slots
    real_gk_by_pos = {
        (round(pose[0]), round(pose[1])): is_gk
        for pose, is_gk in zip(obs.teammate_poses, obs.teammate_is_gk)
    }
    corrected = []
    for slot in teammate_slots:
        real = real_gk_by_pos.get((round(slot.x_field), round(slot.y_field)))
        corrected.append(slot if real is None else replace(slot, is_gk=real))
    return corrected


def build_team_decision_for_agent(
    *,
    aid: int,
    role: str,
    phase: str,
    obs: BackendObservation,
    obs_by_agent: dict[int, BackendObservation],
    agent_positions: dict[int, tuple[float, float]],
    agent_pose_ages: dict[int, float],
    team_ball: tuple[float, float, bool],
    coord: tuple[float, float],
    gate: SkillGate,
    team_state: TeamCoordinationState,
):
    """Per-agent gate decision + role augmentation, mirroring
    teacher_multiagent_dataset.py::_collect_team_scenario lines ~1244-1304."""
    agent_id_map = build_agent_id_map(aid, agent_positions, agent_pose_ages)
    teammate_slots = teammate_slots_from_obs(obs, agent_id_map)
    teammate_slots = _correct_is_gk(teammate_slots, obs)

    base_decision = gate.step(obs)

    if base_decision.shoot_armed:
        goal_dist = math.hypot(GOAL_X - float(obs.ball_x), float(obs.ball_y))
        if goal_dist > SHOOT_SANITY_MAX_GOAL_DIST_MM:
            base_decision = replace(base_decision, shoot_armed=False, shoot_arm_progress=0.0)

    if role == "off_ball_support" and phase == "defense" and defender_block_armed(
        obs,
        team_ball_valid=bool(team_ball[2]),
        face_ball_xy=(team_ball[0], team_ball[1]),
        coordination_target=coord,
    ):
        from dataclasses import replace
        base_decision = replace(
            base_decision,
            dribble_armed=True,
            shoot_arm_progress=max(float(base_decision.shoot_arm_progress), 1.0),
        )

    pass_armed = False
    if role == "striker" and (base_decision.shoot_armed or base_decision.dribble_armed):
        skill_proposal, _agent_target = smash_or_pass(
            obs, teammate_slots,
            team_state.last_pass_target_by_striker.get(aid),
            team_state.last_pass_target_age_ms_by_striker.get(aid, 0.0),
        )
        pass_armed = skill_proposal == "pass"
        global _DIAG_PASS_PRINTS
        if _DIAG_PASS_PRINTS < 25:
            _DIAG_PASS_PRINTS += 1
            own_value = float(obs.shot_opening_with_obstacles) + 2.0 * float(obs.can_score_now)
            print(
                f"[pass-diag #{_DIAG_PASS_PRINTS}] aid={aid} shoot_armed={base_decision.shoot_armed} "
                f"dribble_armed={base_decision.dribble_armed} n_teammate_slots={len(teammate_slots)} "
                f"slots={[(s.agent_id, round(s.x_field), round(s.y_field), s.is_gk) for s in teammate_slots]} "
                f"striker_pos=({obs.robot_x:.0f},{obs.robot_y:.0f}) own_value~{own_value:.3f} "
                f"proposal={skill_proposal}",
                flush=True,
            )

    if (
        role == "striker"
        and not pass_armed
        and (base_decision.shoot_armed or base_decision.dribble_armed)
        and defender_clear_armed(obs, teammate_slots)
    ):
        from dataclasses import replace
        base_decision = replace(
            base_decision,
            shoot_armed=True,
            shoot_arm_progress=max(float(base_decision.shoot_arm_progress), 1.0),
        )

    team_decision = build_team_gate_decision(base_decision, role=role, pass_armed=pass_armed)
    return team_decision


def _resolve_repo_path(path: Path) -> Path:
    """Relative CLI paths are typed against the RL repo root, but importing
    teacher_multiagent_dataset (for pybh) chdirs the process into BHUMAN_DIR — so any
    relative path must be resolved against _SCRIPT_DIR (computed pre-chdir), not cwd.
    Same pitfall tools/export_ppo_policy_team.py already documents."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = _SCRIPT_DIR / path
    return path.resolve()


def main() -> int:
    args = parse_args()
    args.checkpoint = _resolve_repo_path(args.checkpoint)
    active_players = parse_player_numbers(args.active_players)
    simrobot_process = (
        maybe_launch_simrobot(args.simrobot_bin, args.scene, active_players) if args.launch else None
    )

    print("D2: heuristic TEACHER vs real B-Human — live match (v5 = shadow only)", flush=True)
    print(f"teacher_stage={args.teacher_stage}", flush=True)
    print(f"bh_macro_radius_mm={args.bh_macro_radius_mm}", flush=True)
    print(f"scene={args.scene}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"rl_active_players={','.join(str(p) for p in active_players)}", flush=True)

    steps: list[dict] = []
    goal_detector = GoalDetector()
    try:
        policy = tmd._load_eval_policy(str(args.checkpoint))
        encoder_env = tmd._make_encoder_env()
        gates = {aid: SkillGate() for aid in active_players}
        team_state = TeamCoordinationState()
        shoot_hold_streak = {aid: 0 for aid in active_players}
        prev_skill_name: dict[int, str | None] = {aid: None for aid in active_players}
        idle_stand_streak = {aid: 0 for aid in active_players}

        for player in active_players:
            send_action(player, StructuredSkillAction(skill_type=int(SkillType.stand)))

        obs_by_agent = wait_for_initial_observations(active_players, args.timeout_ms, DEFAULT_MAX_WAIT_S)
        print("observations_ready=1", flush=True)

        t0 = time.monotonic()
        deadline = t0 + args.seconds
        period_s = 1.0 / max(0.1, float(args.hz))
        step = 0

        while time.monotonic() < deadline:
            step += 1
            t = time.monotonic() - t0

            agent_positions = {aid: (float(o.robot_x), float(o.robot_y)) for aid, o in obs_by_agent.items()}
            agent_pose_ages = {aid: 0.0 for aid in obs_by_agent}
            team_ball = tmd._team_ball_from_obs(obs_by_agent)
            phase = update_team_phase(team_state, team_ball)
            roles = assign_roles(obs_by_agent, team_ball, team_state.roles, phase=phase)

            striker_id = next((aid for aid, r in roles.items() if r == "striker"), None)
            open_id = next((aid for aid, r in roles.items() if r == "open_support"), None)
            off_id = next((aid for aid, r in roles.items() if r == "off_ball_support"), None)

            if off_id is not None and striker_id is not None and phase == "defense":
                raw_tri, side = defender_cover_target(
                    obs_by_agent[off_id], (team_ball[0], team_ball[1]),
                    current_side=team_state.defender_side_by_agent.get(off_id),
                )
                team_state.defender_side_by_agent[off_id] = side
                team_state.triangle_ema = (
                    raw_tri if team_state.triangle_ema is None else (
                        EMA_ALPHA_TRIANGLE * raw_tri[0] + (1 - EMA_ALPHA_TRIANGLE) * team_state.triangle_ema[0],
                        EMA_ALPHA_TRIANGLE * raw_tri[1] + (1 - EMA_ALPHA_TRIANGLE) * team_state.triangle_ema[1],
                    )
                )
            if open_id is not None and striker_id is not None:
                raw_open = (
                    defense_support_target(
                        obs_by_agent[open_id], (team_ball[0], team_ball[1]),
                        agent_positions[striker_id], agent_positions.get(off_id, agent_positions[open_id]),
                    )
                    if phase == "defense" and off_id is not None
                    else compute_open_lane_target(
                        obs_by_agent[open_id], (team_ball[0], team_ball[1]), agent_positions[striker_id],
                    )
                )
                team_state.open_lane_ema = (
                    raw_open if team_state.open_lane_ema is None else (
                        EMA_ALPHA_OPEN_LANE * raw_open[0] + (1 - EMA_ALPHA_OPEN_LANE) * team_state.open_lane_ema[0],
                        EMA_ALPHA_OPEN_LANE * raw_open[1] + (1 - EMA_ALPHA_OPEN_LANE) * team_state.open_lane_ema[1],
                    )
                )
            if off_id is not None and striker_id is not None and open_id is not None and phase != "defense":
                raw_tri = compute_triangle_target(
                    obs_by_agent[off_id], (team_ball[0], team_ball[1]),
                    agent_positions[striker_id], agent_positions[open_id],
                )
                team_state.triangle_ema = (
                    raw_tri if team_state.triangle_ema is None else (
                        EMA_ALPHA_TRIANGLE * raw_tri[0] + (1 - EMA_ALPHA_TRIANGLE) * team_state.triangle_ema[0],
                        EMA_ALPHA_TRIANGLE * raw_tri[1] + (1 - EMA_ALPHA_TRIANGLE) * team_state.triangle_ema[1],
                    )
                )

            advance_pass_target_age(team_state, 1000.0 / max(0.1, float(args.hz)))

            # D3: which of our robots is nearest to a FRESH ball (own estimate, same
            # freshness rule as the step-row ball_fresh flag below).
            def _fresh_ball_dist(o: BackendObservation) -> float:
                if float(o.ball_seen_percentage) <= 0.0 or float(o.natural_time_since_ball_seen) > 1000.0:
                    return float("inf")
                return math.hypot(float(o.ball_x) - float(o.robot_x), float(o.ball_y) - float(o.robot_y))
            fresh_ball_dist = {aid: _fresh_ball_dist(o) for aid, o in obs_by_agent.items()}
            nearest_aid = min(fresh_ball_dist, key=fresh_ball_dist.get)

            actions_for_step: dict[int, StructuredSkillAction] = {}
            step_row = {"step": step, "t": t, "phase": phase, "agents": {}}

            for aid in active_players:
                role = roles.get(aid, "off_ball_support")
                obs = obs_by_agent[aid]
                coord = (
                    team_state.open_lane_ema if role == "open_support" and team_state.open_lane_ema is not None
                    else team_state.triangle_ema if role == "off_ball_support" and team_state.triangle_ema is not None
                    else (team_ball[0], team_ball[1])
                )
                if coord is None:
                    coord = agent_positions[aid]

                team_decision = build_team_decision_for_agent(
                    aid=aid, role=role, phase=phase, obs=obs, obs_by_agent=obs_by_agent,
                    agent_positions=agent_positions, agent_pose_ages=agent_pose_ages,
                    team_ball=team_ball, coord=coord, gate=gates[aid], team_state=team_state,
                )
                encoded = tmd._encode_for_agent(encoder_env, obs, team_decision)

                # SHADOW: what v5 would have chosen on this exact live state (never executed).
                enc_t = torch.as_tensor(encoded, dtype=torch.float32).reshape(1, -1)
                with torch.no_grad():
                    skill_logits, _param_mean = policy(enc_t)
                gate_mask = gate_skill_mask_from_obs(enc_t)
                masked_logits = skill_logits + (gate_mask - 1.0) * 1e9
                shadow_idx = int(masked_logits.argmax(1).item())

                # ACTOR: the heuristic teacher, exactly as in
                # teacher_multiagent_dataset._collect_team_scenario (non-eval branch).
                teammate_slots = _correct_is_gk(
                    teammate_slots_from_obs(obs, build_agent_id_map(aid, agent_positions, agent_pose_ages)),
                    obs,
                )
                _payload, absolute, residual = team_teacher_payload(
                    obs, args.teacher_stage, prev_skill_name[aid],
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
                # D3: hand ball play to B-Human's own macro (PlayBall port: shoot vs passTo),
                # bypassing the 380mm/aligned/3m shoot gate, exactly where B-Human uses it.
                bh_macro = (
                    args.bh_macro_radius_mm > 0.0
                    and role == "striker"
                    and aid == nearest_aid
                    and fresh_ball_dist[aid] <= args.bh_macro_radius_mm
                )
                if bh_macro:
                    proposal, pass_agent = smash_or_pass(
                        obs, teammate_slots,
                        team_state.last_pass_target_by_striker.get(aid),
                        team_state.last_pass_target_age_ms_by_striker.get(aid, 0.0),
                    )
                    if proposal == "pass" and pass_agent >= 0:
                        absolute = StructuredSkillAction(skill_type=int(SkillType.pass_), pass_target=int(pass_agent))
                    else:
                        absolute = StructuredSkillAction(skill_type=int(SkillType.shoot))
                    residual = absolute
                if residual.skill_name == "pass":
                    record_pass_target(team_state, aid, int(residual.pass_target))
                prev_skill_name[aid] = residual.skill_name
                actions_for_step[aid] = absolute

                teacher_idx = int(residual.skill_type)
                step_row["agents"][aid] = {
                    "role": role,
                    "skill": residual.skill_name,
                    "teacher_label_legal": bool(float(gate_mask[0, teacher_idx]) > 0.5),
                    "shadow_skill": SKILL_ORDER[shadow_idx],
                    "gate_mask": gate_mask[0].tolist(),
                    "raw_logits": skill_logits[0].tolist(),
                    "robot_x": float(obs.robot_x), "robot_y": float(obs.robot_y),
                    "nearest_opponent_dist": float(obs.nearest_opponent_dist),
                    "forced_shoot": False,
                    "forced_walk": False,
                    "bh_macro": bool(bh_macro),
                    "ball_dist": math.hypot(
                        float(obs.ball_x) - float(obs.robot_x), float(obs.ball_y) - float(obs.robot_y)
                    ),
                }

            for aid, action in actions_for_step.items():
                send_action(aid, action)

            # No ground-truth ball export exists in a live (non-training) match. team_ball
            # (obs[32:34]) was tried first but never validates in this scene (a 120s smoke
            # run showed global_team_ball_valid=False on 946/946 steps — real B-Human team
            # comm doesn't form a shared ball belief here). Falling back to the freshest
            # single robot's own BallModel estimate: _ball_authority already picks whichever
            # RL player is closest to its own perceived ball, and gating on natural
            # perception age rejects stale/hallucinated readings from whichever one currently
            # doesn't see it (matches the ball_reliable() freshness check
            # run_rl_vs_bhuman_3v3.py already uses for the same class of signal).
            auth_aid = tmd._ball_authority(obs_by_agent)
            auth_obs = obs_by_agent[auth_aid]
            ball_fresh = (
                float(auth_obs.ball_seen_percentage) > 0.0
                and float(auth_obs.natural_time_since_ball_seen) <= 1000.0
            )
            step_row["ball_x"] = float(auth_obs.ball_x)
            step_row["ball_y"] = float(auth_obs.ball_y)
            step_row["ball_authority"] = auth_aid
            step_row["ball_fresh"] = ball_fresh
            # Real GameController score (autoreferee ground truth), now exported by the
            # C++ bridge. The ball-crossing proxy below is kept only for backward
            # comparison with the pre-2026-08-11 batches -- gc_* is the authoritative
            # scoreline. All RL players see the same GameState, so player order is
            # irrelevant; take the max in case one slot's shared-mem frame lags.
            step_row["gc_goals_for"] = max(int(o.own_score) for o in obs_by_agent.values())
            step_row["gc_goals_against"] = max(int(o.opponent_score) for o in obs_by_agent.values())
            # TRUE possession inputs: distance from the ball to the nearest opponent, in
            # absolute field coordinates (team-fused GlobalOpponentsModel), alongside our
            # own nearest-to-ball distance. This is a real "who is closest to the ball"
            # comparison -- unlike the legacy egocentric nearest_opponent_dist proxy,
            # which answers "is an opponent near ME". Union the per-robot lists and
            # de-duplicate on a 200mm grid, since each robot reports its own fused view
            # of the same physical opponents.
            opp_pts: set[tuple[int, int]] = set()
            for o in obs_by_agent.values():
                for ox, oy in o.opponent_positions:
                    opp_pts.add((round(ox / 200.0), round(oy / 200.0)))
            opp_ball_dists = [
                math.hypot(gx * 200.0 - step_row["ball_x"], gy * 200.0 - step_row["ball_y"])
                for gx, gy in opp_pts
            ]
            step_row["opp_count"] = len(opp_pts)
            # B-Human behaviour log (D2 side-dataset): de-duplicated opponent field
            # positions (200mm grid) + each RL robot's ball estimate, so opponent
            # approach paths, speeds and ball-winning episodes can be reconstructed.
            step_row["opp_positions"] = sorted((gx * 200, gy * 200) for gx, gy in opp_pts)
            step_row["opp_nearest_ball_dist"] = min(opp_ball_dists) if opp_ball_dists else None
            step_row["rl_nearest_ball_dist"] = min(
                math.hypot(float(o.ball_x) - float(o.robot_x), float(o.ball_y) - float(o.robot_y))
                for o in obs_by_agent.values()
            )
            goal_detector.update(t=t, ball_x=step_row["ball_x"], ball_y=step_row["ball_y"], valid=ball_fresh)
            step_row["goals_for"] = goal_detector.scored_for
            step_row["goals_against"] = goal_detector.scored_against
            steps.append(step_row)

            if step % int(max(1.0, args.hz) * 30) == 0:
                print(
                    f"t={t:6.1f}s goals_for={goal_detector.scored_for} goals_against={goal_detector.scored_against} "
                    f"ball=({step_row['ball_x']:.0f},{step_row['ball_y']:.0f}) "
                    f"roles={ {a: r['role'] for a, r in step_row['agents'].items()} }",
                    flush=True,
                )

            time.sleep(period_s)
            obs_by_agent = {aid: fetch_obs(aid, args.timeout_ms) for aid in active_players}
    finally:
        for player in active_players:
            send_action(player, StructuredSkillAction(skill_type=int(SkillType.stand)))
        if simrobot_process is not None:
            simrobot_process.terminate()
            try:
                simrobot_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                simrobot_process.kill()

    output_path = args.output or Path(
        f"runs/paper_match_teacher_vs_bhuman_live/match_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path = _resolve_repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "scene": str(args.scene),
        "checkpoint": str(args.checkpoint),
        "controller": "teacher",
        "teacher_stage": args.teacher_stage,
        "bh_macro_radius_mm": args.bh_macro_radius_mm,
        "shadow_policy": str(args.checkpoint),
        "active_players": list(active_players),
        "seconds": args.seconds,
        "hz": args.hz,
        "goals_for": goal_detector.scored_for,
        "goals_against": goal_detector.scored_against,
        "goal_events": goal_detector.events,
        "gc_goals_for": (steps[-1]["gc_goals_for"] if steps else 0),
        "gc_goals_against": (steps[-1]["gc_goals_against"] if steps else 0),
        "steps": steps,
    }))
    gc_for = steps[-1]["gc_goals_for"] if steps else 0
    gc_against = steps[-1]["gc_goals_against"] if steps else 0
    print(f"wrote {output_path}", flush=True)
    print(f"FINAL SCORE (GameController)  teacher={gc_for}  bhuman={gc_against}", flush=True)
    print(f"FINAL SCORE (legacy proxy)    teacher={goal_detector.scored_for}  bhuman={goal_detector.scored_against}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
