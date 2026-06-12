from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

# Must be set before importing pybh-backed modules.
os.environ.setdefault("PYBH_RL_OVERRIDE_TEAM", "24")
os.environ.setdefault("PYBH_RL_ACTIVE_PLAYERS", "1,2")
os.environ.setdefault("PYBH_RL_SHM_NAME", f"/pybh_rl_ma_diag_{os.getpid()}")
os.environ.setdefault("RL_VISIBLE_STEP_OBS_TIMEOUT_MS", "2000")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action import SkillType, StructuredSkillAction
from config import SIMROBOT_BIN
from gate import GateConfig, SkillGate
from multiagent_teacher import (
    EMA_ALPHA_OPEN_LANE,
    TeamCoordinationState,
    advance_pass_target_age,
    assign_roles,
    build_agent_id_map,
    build_team_gate_decision,
    compute_open_lane_target,
    record_pass_target,
    smash_or_pass,
    team_teacher_payload,
    teammate_slots_from_obs,
)
from production_curriculum import team_production_scenarios
from pybh_backend import BackendObservation, PyBHBackend, WorldPlayerSpec, WorldSpec, build_world_spec
from visible_parallel_runtime import VisibleParallelRuntime


STAGE_ID = 6
ACTIVE = (1, 2)
STEP_DT_MS = 33.0
DISPLAY_BASE = int(os.environ.get("RL_MA_DIAG_DISPLAY_BASE", "170"))
MAX_STEPS = int(os.environ.get("RL_MA_DIAG_STEPS", "300"))
SEED = int(os.environ.get("RL_MA_DIAG_SEED", "880000"))


def _player_pose(world: WorldSpec, player_number: int) -> tuple[float, float, float]:
    if player_number == 1:
        return float(world.robot_x), float(world.robot_y), float(world.robot_theta)
    for spec in world.teammates:
        if int(spec.number) == int(player_number):
            return float(spec.x), float(spec.y), float(spec.theta)
    raise KeyError(f"player {player_number} not present in world spec")


def _slot_reset_args(world: WorldSpec, player_number: int) -> dict:
    x, y, theta = _player_pose(world, player_number)
    teammates: list[WorldPlayerSpec] = []
    if player_number != 1:
        teammates.append(WorldPlayerSpec(number=1, x=world.robot_x, y=world.robot_y, theta=world.robot_theta))
    for spec in world.teammates:
        if int(spec.number) != int(player_number):
            teammates.append(spec)
    return dict(
        ball_pos=np.asarray([world.ball_x, world.ball_y], dtype=np.float32),
        robot_pos=np.asarray([x, y], dtype=np.float32),
        robot_theta=float(theta),
        teammates=tuple(teammates),
        opponents=tuple(world.opponents),
    )


def _reset_all(backends: dict[int, PyBHBackend], world: WorldSpec) -> dict[int, BackendObservation]:
    obs: dict[int, BackendObservation] = {}
    for player_number, backend in backends.items():
        obs[player_number] = backend.reset(**_slot_reset_args(world, player_number))
    return obs


def _team_ball(obs_by_agent: dict[int, BackendObservation]) -> tuple[float, float, bool]:
    primary = obs_by_agent[1]
    if primary.global_team_ball_valid:
        return float(primary.global_team_ball_x_mm), float(primary.global_team_ball_y_mm), True
    return float(primary.ball_x), float(primary.ball_y), True


def _print_step(
    step: int,
    obs_by_agent: dict[int, BackendObservation],
    actions: dict[int, StructuredSkillAction],
    roles: dict[int, str],
    dribble_armed: int,
    shoot_armed: int,
    pass_armed: int,
) -> None:
    striker = obs_by_agent[1]
    action = actions.get(1, StructuredSkillAction(skill_type=SkillType.stand))
    dist = math.hypot(float(striker.ball_x) - float(striker.robot_x), float(striker.ball_y) - float(striker.robot_y))
    print(
        f"step={step:03d} role1={roles.get(1)} skill1={action.skill_name:<7} "
        f"dist_ball={dist:7.1f} robot=({striker.robot_x:7.1f},{striker.robot_y:7.1f}) "
        f"sim=({striker.simrobot_robot_x:7.1f},{striker.simrobot_robot_y:7.1f}) "
        f"ball=({striker.ball_x:7.1f},{striker.ball_y:7.1f}) "
        f"motion={striker.motion_request} provider={striker.provider_motion_request} "
        f"speed=({striker.motion_speed_x:.1f},{striker.motion_speed_y:.1f}) "
        f"arms(d/s/p)=({dribble_armed}/{shoot_armed}/{pass_armed})",
        flush=True,
    )


def main() -> int:
    scenarios = [s for s in team_production_scenarios(1) if s.stage_id == STAGE_ID]
    scenario = next(s for s in scenarios if s.name == "team_2v0_clean_band")
    world = scenario.world_fn(np.random.default_rng(SEED))
    print(
        f"[diag parallel] shm={os.environ['PYBH_RL_SHM_NAME']} display_base={DISPLAY_BASE} "
        f"world ball=({world.ball_x:.1f},{world.ball_y:.1f}) "
        f"p1=({world.robot_x:.1f},{world.robot_y:.1f})",
        flush=True,
    )

    runtime = VisibleParallelRuntime(
        player_numbers=ACTIVE,
        shm_name=os.environ["PYBH_RL_SHM_NAME"],
        display_base=DISPLAY_BASE,
        scene_name="RL1v0_3D_visible",
        simrobot_bin=SIMROBOT_BIN,
        simrobot_env={
            "PYBH_RL_OVERRIDE_TEAM": os.environ["PYBH_RL_OVERRIDE_TEAM"],
            "PYBH_RL_ACTIVE_PLAYERS": os.environ["PYBH_RL_ACTIVE_PLAYERS"],
        },
    )
    backends = {
        player: PyBHBackend(
            mode="simrobot_visible",
            player_number=player,
            simrobot_profile="3d_visible",
            simrobot_launch=False,
            visible_status_log_interval_s=5.0,
        )
        for player in ACTIVE
    }
    gates = {player: SkillGate(GateConfig.from_env()) for player in ACTIVE}
    team_state = TeamCoordinationState()
    dribble_armed = 0
    shoot_armed = 0
    pass_armed = 0
    skill_counts: dict[str, int] = {}

    try:
        runtime.start()
        for backend in backends.values():
            backend.start()
        for gate in gates.values():
            gate.reset()
        obs_by_agent = _reset_all(backends, world)

        for step in range(1, MAX_STEPS + 1):
            team_ball = _team_ball(obs_by_agent)
            roles = assign_roles(obs_by_agent, team_ball, team_state.roles)
            positions = {aid: (float(obs.robot_x), float(obs.robot_y)) for aid, obs in obs_by_agent.items()}
            striker_id = next((aid for aid, role in roles.items() if role == "striker"), None)
            open_id = next((aid for aid, role in roles.items() if role == "open_support"), None)
            if striker_id is not None and open_id is not None:
                raw_open = compute_open_lane_target(
                    obs_by_agent[open_id],
                    (team_ball[0], team_ball[1]),
                    positions[striker_id],
                )
                team_state.open_lane_ema = (
                    raw_open
                    if team_state.open_lane_ema is None
                    else (
                        EMA_ALPHA_OPEN_LANE * raw_open[0] + (1.0 - EMA_ALPHA_OPEN_LANE) * team_state.open_lane_ema[0],
                        EMA_ALPHA_OPEN_LANE * raw_open[1] + (1.0 - EMA_ALPHA_OPEN_LANE) * team_state.open_lane_ema[1],
                    )
                )
            advance_pass_target_age(team_state, STEP_DT_MS)

            actions: dict[int, StructuredSkillAction] = {}
            for aid in ACTIVE:
                role = roles.get(aid, "off_ball_support")
                obs = obs_by_agent[aid]
                agent_id_map = build_agent_id_map(aid, positions, {k: 0.0 for k in positions})
                teammate_slots = teammate_slots_from_obs(obs, agent_id_map)
                base_decision = gates[aid].step(obs)
                this_pass_armed = False
                if role == "striker" and base_decision.shoot_armed:
                    proposal, _target = smash_or_pass(
                        obs,
                        teammate_slots,
                        team_state.last_pass_target_by_striker.get(aid),
                        team_state.last_pass_target_age_ms_by_striker.get(aid, 0.0),
                    )
                    this_pass_armed = proposal == "pass"
                team_decision = build_team_gate_decision(base_decision, role=role, pass_armed=this_pass_armed)
                coord = (
                    team_state.open_lane_ema
                    if role == "open_support" and team_state.open_lane_ema is not None
                    else (team_ball[0], team_ball[1])
                )
                _payload, absolute, residual = team_teacher_payload(
                    obs,
                    STAGE_ID,
                    None,
                    gate_decision=team_decision,
                    role=role,
                    coordination_target=coord,
                    teammate_slots=teammate_slots,
                    last_pass_target=team_state.last_pass_target_by_striker.get(aid),
                    last_pass_target_age_ms=team_state.last_pass_target_age_ms_by_striker.get(aid, 0.0),
                )
                if aid == 1:
                    dribble_armed += int(bool(team_decision.dribble_armed))
                    shoot_armed += int(bool(team_decision.shoot_armed))
                    pass_armed += int(bool(this_pass_armed))
                    skill_counts[residual.skill_name] = skill_counts.get(residual.skill_name, 0) + 1
                    if residual.skill_name == "pass":
                        record_pass_target(team_state, aid, int(residual.pass_target))
                actions[aid] = absolute

            if step <= 20 or step % 25 == 0:
                _print_step(step, obs_by_agent, actions, roles, dribble_armed, shoot_armed, pass_armed)

            for aid, backend in backends.items():
                obs_by_agent[aid] = backend.step(actions.get(aid, StructuredSkillAction(skill_type=SkillType.stand)))

        striker = obs_by_agent[1]
        print(
            "[diag parallel] RESULT "
            f"skill_counts={skill_counts} dribble_armed={dribble_armed} "
            f"shoot_armed={shoot_armed} pass_armed={pass_armed} "
            f"final_dist={math.hypot(striker.ball_x - striker.robot_x, striker.ball_y - striker.robot_y):.1f} "
            f"final_robot=({striker.robot_x:.1f},{striker.robot_y:.1f}) "
            f"final_sim=({striker.simrobot_robot_x:.1f},{striker.simrobot_robot_y:.1f})",
            flush=True,
        )
        return 0 if dribble_armed > 0 and shoot_armed > 0 else 2
    finally:
        for backend in backends.values():
            backend.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
