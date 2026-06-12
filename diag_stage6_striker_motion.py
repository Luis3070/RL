from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

# The bridge can be read by BHuman at launch, so make the control contract true
# before importing pybh_backend through teacher_multiagent_dataset.
os.environ.setdefault("PYBH_RL_OVERRIDE_TEAM", "24")
os.environ.setdefault("PYBH_RL_ACTIVE_PLAYERS", "1,2")
os.environ.setdefault("RL_MA_TEACHER_STAGE", "6")
os.environ.setdefault("RL_MA_TEACHER_EPISODES", "1")
os.environ.setdefault("RL_MA_TEACHER_MAX_STEPS", "50")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action import SkillType, StructuredSkillAction
from gate import GateConfig, SkillGate
from multiagent_teacher import (
    EMA_ALPHA_OPEN_LANE,
    TeamCoordinationState,
    advance_pass_target_age,
    assign_roles,
    build_agent_id_map,
    build_team_gate_decision,
    compute_open_lane_target,
    smash_or_pass,
    team_teacher_payload,
    teammate_slots_from_obs,
)
from production_curriculum import team_production_scenarios
from teacher_multiagent_dataset import (
    STEP_DT_MS,
    _active_player_numbers,
    _coordinated_reset,
    _make_backend,
    _step_backends,
    _team_ball_from_obs,
)


def _dist_to_ball(obs) -> float:
    return math.hypot(float(obs.robot_x) - float(obs.ball_x), float(obs.robot_y) - float(obs.ball_y))


def main() -> int:
    active = _active_player_numbers(2)
    backends = {p: _make_backend(p, is_primary=(p == active[0])) for p in active}
    for backend in backends.values():
        backend.start()

    try:
        scenario = next(s for s in team_production_scenarios(1) if s.stage_id == 6)
        world = scenario.world_fn(np.random.default_rng(880000))
        print(
            "[diag] world "
            f"ball=({world.ball_x:.1f},{world.ball_y:.1f}) "
            f"striker_spawn=({world.robot_x:.1f},{world.robot_y:.1f}) "
            f"env_active={os.environ.get('PYBH_RL_ACTIVE_PLAYERS')}",
            flush=True,
        )

        gates = {aid: SkillGate(GateConfig.from_env()) for aid in active}
        team_state = TeamCoordinationState()
        obs_by_agent = _coordinated_reset(backends, active[0], world)
        prev_skill = {aid: None for aid in active}

        print(
            "step,p,role,pre_x,pre_y,post_x,post_y,dx,dy,ball_x,ball_y,"
            "dist_pre,dist_post,skill,target_x,target_y,target_theta,"
            "shoot_armed,dribble_armed,natural_seen_ms,team_seen_ms,frame_pre,frame_post,"
            "requested_skill,provider_calls,provider_motion,provider_target_x,provider_target_y,"
            "skill_behavior_calls,skill_behavior_motion,skill_behavior_target_x,skill_behavior_target_y,"
            "motion_speed_x,motion_speed_y,ground_contact,fall_state,"
            "sim_x,sim_y,corrected_pose,synthetic_pose,pose_align_err",
            flush=True,
        )

        for step in range(1, 51):
            agent_positions = {aid: (float(o.robot_x), float(o.robot_y)) for aid, o in obs_by_agent.items()}
            agent_pose_ages = {aid: 0.0 for aid in obs_by_agent}
            team_ball = _team_ball_from_obs(obs_by_agent)
            roles = assign_roles(obs_by_agent, team_ball, team_state.roles)
            striker_id = next((aid for aid, role in roles.items() if role == "striker"), None)
            open_id = next((aid for aid, role in roles.items() if role == "open_support"), None)

            if open_id is not None and striker_id is not None:
                raw_open = compute_open_lane_target(
                    obs_by_agent[open_id],
                    (team_ball[0], team_ball[1]),
                    agent_positions[striker_id],
                )
                team_state.open_lane_ema = (
                    raw_open if team_state.open_lane_ema is None
                    else (
                        EMA_ALPHA_OPEN_LANE * raw_open[0] + (1.0 - EMA_ALPHA_OPEN_LANE) * team_state.open_lane_ema[0],
                        EMA_ALPHA_OPEN_LANE * raw_open[1] + (1.0 - EMA_ALPHA_OPEN_LANE) * team_state.open_lane_ema[1],
                    )
                )

            advance_pass_target_age(team_state, STEP_DT_MS)
            actions = {}
            debug_rows = {}
            for aid in active:
                obs = obs_by_agent[aid]
                role = roles.get(aid, "off_ball_support")
                base = gates[aid].step(obs)
                pass_armed = False
                agent_id_map = build_agent_id_map(aid, agent_positions, agent_pose_ages)
                teammate_slots = teammate_slots_from_obs(obs, agent_id_map)
                if role == "striker" and base.shoot_armed:
                    skill_proposal, _ = smash_or_pass(
                        obs,
                        teammate_slots,
                        team_state.last_pass_target_by_striker.get(aid),
                        team_state.last_pass_target_age_ms_by_striker.get(aid, 0.0),
                    )
                    pass_armed = skill_proposal == "pass"
                gate = build_team_gate_decision(base, role=role, pass_armed=pass_armed)
                coord = team_state.open_lane_ema if role == "open_support" and team_state.open_lane_ema else (team_ball[0], team_ball[1])
                _payload, absolute, _residual = team_teacher_payload(
                    obs,
                    6,
                    prev_skill[aid],
                    gate_decision=gate,
                    role=role,
                    coordination_target=coord,
                    teammate_slots=teammate_slots,
                    last_pass_target=team_state.last_pass_target_by_striker.get(aid),
                    last_pass_target_age_ms=team_state.last_pass_target_age_ms_by_striker.get(aid, 0.0),
                )
                actions[aid] = absolute
                prev_skill[aid] = absolute.skill_name
                debug_rows[aid] = (role, obs, gate, absolute)

            pre_rows = {aid: (debug_rows[aid], _dist_to_ball(debug_rows[aid][1])) for aid in active}
            next_obs = _step_backends(backends, actions)
            for aid in active:
                (role, pre, gate, action), dist_pre = pre_rows[aid]
                post = next_obs[aid]
                print(
                    f"{step},{aid},{role},"
                    f"{float(pre.robot_x):.1f},{float(pre.robot_y):.1f},"
                    f"{float(post.robot_x):.1f},{float(post.robot_y):.1f},"
                    f"{float(post.robot_x)-float(pre.robot_x):.1f},{float(post.robot_y)-float(pre.robot_y):.1f},"
                    f"{float(pre.ball_x):.1f},{float(pre.ball_y):.1f},"
                    f"{dist_pre:.1f},{_dist_to_ball(post):.1f},"
                    f"{action.skill_name},{float(action.target_x):.1f},{float(action.target_y):.1f},{float(action.target_theta):.3f},"
                    f"{int(bool(gate.shoot_armed))},{int(bool(gate.dribble_armed))},"
                    f"{float(getattr(pre, 'natural_time_since_ball_seen', 0.0)):.1f},"
                    f"{float(getattr(pre, 'time_since_ball_seen', 0.0)):.1f},"
                    f"{int(pre.frame)},{int(post.frame)},"
                    f"{post.requested_skill},"
                    f"{int(post.provider_call_count)},{int(post.provider_motion_request)},"
                    f"{float(post.provider_target_x):.1f},{float(post.provider_target_y):.1f},"
                    f"{int(post.skill_behavior_call_count)},{int(post.skill_behavior_motion_request)},"
                    f"{float(post.skill_behavior_walk_target_x):.1f},{float(post.skill_behavior_walk_target_y):.1f},"
                f"{float(post.motion_speed_x):.1f},{float(post.motion_speed_y):.1f},"
                f"{int(bool(post.motion_engine_ground_contact))},{int(post.motion_engine_fall_state)},"
                f"{float(post.simrobot_robot_x):.1f},{float(post.simrobot_robot_y):.1f},"
                f"{int(bool(post.corrected_exported_robot_pose))},"
                f"{int(bool(post.synthetic_pose_observation))},"
                f"{float(post.pose_alignment_error_mm):.1f}",
                flush=True,
            )
            obs_by_agent = next_obs
    finally:
        for backend in backends.values():
            backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
