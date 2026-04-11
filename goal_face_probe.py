"""
goal_face_probe.py

Run a trained checkpoint in fixed near-goal scenarios to diagnose:
  - playBall takeover behavior close to the opponent goal
  - HOLD / pre-kick loops
  - BallTo / ExecTo consistency
  - whether odd front-of-goal behavior comes from policy, bridge telemetry,
    or the 2D executor layer
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

os.environ.setdefault("RL_EXECUTION_MODE", "bhuman_sidecar")

from config import CURRICULUM, FIELD, SIM
from environment import NaoSoccerEnv
from visualizer import load_policy


DEFAULT_CHECKPOINT = (
    r"C:\ALEJO\runs\practical_stage0_fovTrue_1775918841\checkpoints\mappo_ep000060.pt"
)


@dataclass(frozen=True)
class ProbeScenario:
    name: str
    ball_pos: tuple[float, float]
    agent_pos: tuple[float, float]
    agent_theta: float
    note: str


SCENARIOS: tuple[ProbeScenario, ...] = (
    ProbeScenario(
        name="center_finish",
        ball_pos=(2280.0, 90.0),
        agent_pos=(2060.0, -20.0),
        agent_theta=0.08,
        note="central finish, clean shooting lane",
    ),
    ProbeScenario(
        name="left_tight_angle",
        ball_pos=(2360.0, 520.0),
        agent_pos=(2140.0, 360.0),
        agent_theta=0.06,
        note="tight angle near upper post",
    ),
    ProbeScenario(
        name="right_tight_angle",
        ball_pos=(2360.0, -520.0),
        agent_pos=(2140.0, -360.0),
        agent_theta=-0.06,
        note="tight angle near lower post",
    ),
    ProbeScenario(
        name="ball_across_body",
        ball_pos=(2235.0, 240.0),
        agent_pos=(2120.0, -10.0),
        agent_theta=0.28,
        note="ball slightly across the body before goal shot",
    ),
)


def _pair(value) -> tuple[Optional[float], Optional[float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None, None
    return None, None


def _is_out_of_field(x: Optional[float], y: Optional[float]) -> bool:
    if x is None or y is None:
        return False
    return abs(x) > FIELD.x_opponent_ground or abs(y) > FIELD.y_left_sideline


def _zero_actions(env: NaoSoccerEnv) -> Dict[str, np.ndarray]:
    return {
        aid: np.zeros(env.action_space(aid).shape, dtype=np.float32)
        for aid in env.agents
    }


def _apply_scenario(env: NaoSoccerEnv, scenario: ProbeScenario) -> Dict[str, np.ndarray]:
    env._ball.pos[:] = np.array(scenario.ball_pos, dtype=np.float32)
    env._ball.vel[:] = 0.0
    env._ball.seen_age_ms = 0.0
    env._ball.last_touched_by = None

    for idx, aid in enumerate(env.agents):
        agent = env._agents_state[aid]
        if idx == 0:
            pos = np.array(scenario.agent_pos, dtype=np.float32)
            theta = float(scenario.agent_theta)
        else:
            pos = np.array(
                [FIELD.x_own_ground + 400.0, -900.0 + 450.0 * idx],
                dtype=np.float32,
            )
            theta = 0.0

        agent.pos = pos.copy()
        agent.theta = theta
        agent.vel = np.zeros(2, dtype=np.float32)
        agent.omega = 0.0
        agent.is_upright = True
        agent.walk_target = pos.copy()
        agent.speed_scalar = 0.0
        agent.is_active = False

    for idx, opp in enumerate(env._opponents_state):
        opp.pos = np.array(
            [FIELD.x_own_ground + 700.0, -1000.0 + 500.0 * idx],
            dtype=np.float32,
        )
        opp.theta = np.pi
        opp.vel = np.zeros(2, dtype=np.float32)
        opp.omega = 0.0
        opp.is_upright = True

    env._game.is_playing = True
    env._game.own_score = 0
    env._game.opp_score = 0
    env._game.step_count = 0
    env._game.time_elapsed = 0.0

    env._ball_out_this_step = False
    env._takeover_events = []
    env._episode_takeover_count = 0
    env._episode_takeover_success = 0
    env._episode_handoff_achieved = False
    env._episode_handoff_ready_frames = 0
    env._episode_playball_entry_count = 0
    env._episode_playball_achieved = False
    env._prev_playball_frame = False
    env._prev_handoff_ready = False

    env._handoff_mgr.reset()
    env._handoff_mgr.set_takeover_timeout_frames(env._stage.takeover_timeout_steps)
    if env._sidecar is not None:
        env._sidecar.reset()
    if env._sidecar_physics is not None:
        env._sidecar_physics.reset()

    env._select_active_agent()
    env._prev_ball_pos = env._ball.pos.copy()
    env._prev_ball_speed = 0.0
    env._prev_ball_dists = {
        aid: float(np.linalg.norm(env._agents_state[aid].pos - env._ball.pos))
        for aid in env.agents
    }
    env._prev_thetas = {aid: env._agents_state[aid].theta for aid in env.agents}
    env._prev_omegas = {aid: env._agents_state[aid].omega for aid in env.agents}
    env._prev_targets = {aid: env._agents_state[aid].pos.copy() for aid in env.agents}
    env._curr_targets = {}

    return env._build_observations()


def _policy_actions(policy, observations: Dict[str, np.ndarray], env: NaoSoccerEnv) -> Dict[str, np.ndarray]:
    if policy is None:
        return _zero_actions(env)

    actions: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for aid in env.agents:
            obs = observations.get(aid)
            if obs is None:
                obs = np.zeros(env.observation_space(aid).shape, dtype=np.float32)
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            out = policy.act({"observations": obs_t}, role="policy")
            actions[aid] = out[0].squeeze(0).cpu().numpy().astype(np.float32)
    return actions


def _scenario_rows(policy,
                   checkpoint_path: str,
                   stage_idx: int,
                   use_fov: bool,
                   scenario: ProbeScenario,
                   max_steps: int,
                   seed: int) -> List[Dict[str, object]]:
    env = NaoSoccerEnv(stage_cfg=CURRICULUM[stage_idx], use_fov=use_fov, seed=seed)
    try:
        env.reset(seed=seed)
        observations = _apply_scenario(env, scenario)
        rows: List[Dict[str, object]] = []

        for step_idx in range(max_steps):
            actions = _policy_actions(policy, observations, env)
            observations, rewards, terminateds, truncateds, infos = env.step(actions)
            info = next(iter(infos.values()), {})
            agent = env._agents_state[env.agents[0]]
            ball = env._ball

            walk_x, walk_y = _pair(info.get("sidecar_behavior_walking_to_field") or info.get("sidecar_motion_walk_target_field"))
            skill_x, skill_y = _pair(info.get("sidecar_skill_target"))
            ballto_x, ballto_y = _pair(info.get("sidecar_ball_target_field"))
            execto_x, execto_y = _pair(info.get("sidecar_effective_target"))

            rows.append(
                {
                    "checkpoint": checkpoint_path,
                    "scenario": scenario.name,
                    "note": scenario.note,
                    "stage": stage_idx,
                    "use_fov": int(use_fov),
                    "seed": seed,
                    "step": step_idx,
                    "agent_x": float(agent.pos[0]),
                    "agent_y": float(agent.pos[1]),
                    "agent_theta": float(agent.theta),
                    "ball_x": float(ball.pos[0]),
                    "ball_y": float(ball.pos[1]),
                    "ball_vx": float(ball.vel[0]),
                    "ball_vy": float(ball.vel[1]),
                    "reward_sum": float(sum(rewards.values())),
                    "control_mode": info.get("takeover_active") and "BHUMAN" or (info.get("handoff_ready") and "HANDOFF" or "RL"),
                    "macro_type": info.get("macro_type"),
                    "macro_end_reason": info.get("last_macro_end_reason"),
                    "sidecar_macro_type": info.get("sidecar_macro_type"),
                    "skill_request": info.get("sidecar_skill_request"),
                    "motion_request": info.get("sidecar_motion_request"),
                    "ball_action": info.get("sidecar_kick_type"),
                    "handoff_ready": int(bool(info.get("handoff_ready", False))),
                    "takeover_active": int(bool(info.get("takeover_active", False))),
                    "playball_entry_count": int(info.get("episode_playball_entry_count", 0) or 0),
                    "takeover_count": int(info.get("episode_takeover_count", 0) or 0),
                    "takeover_success": int(info.get("episode_takeover_success", 0) or 0),
                    "pending_kick_hold": int(bool(info.get("pending_kick_hold", False))),
                    "pending_kick_hold_frames": int(info.get("pending_kick_hold_frames", 0) or 0),
                    "post_kick_release_frames": int(info.get("post_kick_release_frames", 0) or 0),
                    "takeover_kick": int(bool(info.get("takeover_kick", False))),
                    "takeover_kick_proxy": int(bool(info.get("takeover_kick_proxy", False))),
                    "walk_field_x": walk_x,
                    "walk_field_y": walk_y,
                    "skill_target_x": skill_x,
                    "skill_target_y": skill_y,
                    "ballto_x": ballto_x,
                    "ballto_y": ballto_y,
                    "execto_x": execto_x,
                    "execto_y": execto_y,
                    "ballto_oob": int(_is_out_of_field(ballto_x, ballto_y)),
                    "execto_oob": int(_is_out_of_field(execto_x, execto_y)),
                }
            )

            if terminateds.get("__all__") or truncateds.get("__all__"):
                break

        return rows
    finally:
        env.close()


def _summarize_rows(rows: Iterable[Dict[str, object]]) -> List[str]:
    rows = list(rows)
    if not rows:
        return ["  no rows"]

    hold_frames = [int(r["pending_kick_hold_frames"]) for r in rows]
    ballto_oob = sum(int(r["ballto_oob"]) for r in rows)
    execto_oob = sum(int(r["execto_oob"]) for r in rows)
    pb = max(int(r["playball_entry_count"]) for r in rows)
    tk = max(int(r["takeover_success"]) for r in rows)
    takeover_count = max(int(r["takeover_count"]) for r in rows)

    skill_counts: Dict[str, int] = {}
    motion_counts: Dict[str, int] = {}
    end_counts: Dict[str, int] = {}
    for row in rows:
        skill = str(row["skill_request"] or "none")
        motion = str(row["motion_request"] or "none")
        end_reason = str(row["macro_end_reason"] or "none")
        skill_counts[skill] = skill_counts.get(skill, 0) + 1
        motion_counts[motion] = motion_counts.get(motion, 0) + 1
        end_counts[end_reason] = end_counts.get(end_reason, 0) + 1

    def _top(counter: Dict[str, int]) -> str:
        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        return ", ".join(f"{k}:{v}" for k, v in items) if items else "-"

    return [
        f"  rows={len(rows)} pb_entries={pb} takeover={takeover_count} tk_success={tk}",
        f"  max_hold={max(hold_frames) if hold_frames else 0} ballto_oob={ballto_oob} execto_oob={execto_oob}",
        f"  skills={_top(skill_counts)}",
        f"  motions={_top(motion_counts)}",
        f"  end_reasons={_top(end_counts)}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe near-goal behavior for a trained checkpoint")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to checkpoint .pt")
    parser.add_argument("--stage", type=int, default=0, help="Curriculum stage index")
    parser.add_argument("--steps", type=int, default=120, help="Max env steps per scenario")
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed")
    parser.add_argument("--no-fov", action="store_true", help="Disable opponent FOV filtering")
    parser.add_argument("--output-dir", default="", help="Optional output directory")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    policy = load_policy(str(checkpoint_path))
    if policy is None:
        raise SystemExit(f"Could not load policy from {checkpoint_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent.parent / "diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    csv_path = output_dir / f"goal_face_probe_stage{args.stage}_{ts}.csv"

    all_rows: List[Dict[str, object]] = []
    print("[GOAL PROBE]")
    print(f"  checkpoint = {checkpoint_path}")
    print(f"  stage      = {args.stage}")
    print(f"  use_fov    = {not args.no_fov}")
    print(f"  steps      = {args.steps}")
    print(f"  output     = {csv_path}")

    for idx, scenario in enumerate(SCENARIOS):
        rows = _scenario_rows(
            policy=policy,
            checkpoint_path=str(checkpoint_path),
            stage_idx=args.stage,
            use_fov=not args.no_fov,
            scenario=scenario,
            max_steps=args.steps,
            seed=args.seed + idx,
        )
        all_rows.extend(rows)
        print(f"\n[SCENARIO] {scenario.name} — {scenario.note}")
        for line in _summarize_rows(rows):
            print(line)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n[GOAL PROBE] CSV written to {csv_path}")


if __name__ == "__main__":
    main()
