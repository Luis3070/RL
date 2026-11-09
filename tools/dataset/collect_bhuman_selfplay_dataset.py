"""B-Human-labelled dataset from a live B-Human vs B-Human match (no RL override).

All team-24 robots play stock B-Human (PYBH_RL_ACTIVE_PLAYERS is left EMPTY, so
StrategyBehaviorControl never takes the external-override branch), but the bridge still
exports every team-24 robot's observation AND the SkillRequest B-Human's own behaviour
chose (`skill_behavior_skill_request`, written in SkillBehaviorControl.cpp for any
team-24 player). B-Human's SkillRequest vocabulary is our action space:

  SkillRequest::Type  none stand walk shoot pass dribble block mark observe interceptBall keeperDive
  SKILL_ORDER idx       -    0    1     2    3     4      5     6     7          -           -

So each row = (our exact 47-dim encoded obs, B-Human's skill label), i.e. B-Human as a
teacher expressed in the policy's own action interface. The 47-dim encoding reuses the
match harness pipeline verbatim (assign_roles / EMA coordination targets / SkillGate /
team gate decision / tmd._encode_for_agent), so rows are directly comparable with the
corpus and with rlfootball/evaluation/run_teacher_vs_bhuman_match.py logs.

Also answers the action-interface question: if B-Human, acting through the same
SkillRequests, controls the ball ~35-40% of the time, the SkillRequest interface is not
the ceiling -- decision content / timing is.

Writes <out>/samples.npz (+ summary.json). The label is RAW B-Human output: no gate
filtering, no episode survivorship filter (`label_legal_under_gate` is stored instead).

Usage:
  python tools/dataset/collect_bhuman_selfplay_dataset.py --launch --seconds 600 --out runs/bhuman_selfplay_dataset/match_1
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# MUST precede any import that does os.environ.setdefault("PYBH_RL_ACTIVE_PLAYERS", ...):
# an empty list means no team-24 robot is overridden -> everyone plays stock B-Human.
os.environ["PYBH_RL_OVERRIDE_TEAM"] = "24"
os.environ["PYBH_RL_ACTIVE_PLAYERS"] = ""
os.environ.pop("PYBH_PPO_ACTIVE_PLAYERS", None)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import rlfootball.evaluation.run_v5_vs_bhuman_match as h  # noqa: E402  (reuses its pipeline; chdirs via tmd import)
from rlfootball.evaluation.run_v5_vs_bhuman_match import (  # noqa: E402
    BHUMAN_DIR, SIMROBOT_BIN, DEFAULT_TIMEOUT_MS, tmd, pybh, BackendObservation, SkillGate,
    TeamCoordinationState, assign_roles, update_team_phase, defender_cover_target,
    defense_support_target, compute_open_lane_target, compute_triangle_target,
    advance_pass_target_age, EMA_ALPHA_TRIANGLE, EMA_ALPHA_OPEN_LANE,
    build_team_decision_for_agent, gate_skill_mask_from_obs,
)

BH_TO_SKILL_IDX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}  # SkillRequest.Type -> SKILL_ORDER
BH_SKILL_NAMES = ["none", "stand", "walk", "shoot", "pass", "dribble", "block", "mark", "observe",
                  "interceptBall", "keeperDive"]
ROLE_IDS = {"striker": 0, "open_support": 1, "off_ball_support": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", type=Path, default=BHUMAN_DIR / "Config/Scenes/RLvsBH4v4_3D.ros2")
    p.add_argument("--simrobot-bin", type=Path, default=SIMROBOT_BIN)
    p.add_argument("--launch", action="store_true")
    p.add_argument("--seconds", type=float, default=600.0)
    p.add_argument("--hz", type=float, default=8.0, help="Sampling rate (B-Human itself still decides at cognition rate).")
    p.add_argument("--players", type=str, default="2,3,4", help="Team-24 field players to record.")
    p.add_argument("--gk", type=int, default=1, help="Team-24 GK number, logged raw only (no 47-dim row).")
    p.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def raw_obs(player: int, timeout_ms: int) -> tuple[dict, BackendObservation]:
    payload = dict(pybh.rl_get_obs(player, timeout_ms))
    return payload, BackendObservation.from_payload(payload)


def main() -> int:
    args = parse_args()
    out = h._resolve_repo_path(args.out)
    players = h.parse_player_numbers(args.players)
    proc = None
    if args.launch:
        env = os.environ.copy()
        proc = subprocess.Popen([str(args.simrobot_bin), str(args.scene)], cwd=BHUMAN_DIR, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"B-Human self-play collection scene={args.scene} players={players} (no RL override)", flush=True)

    rows: dict[str, list] = {k: [] for k in (
        "observations", "bh_skill_raw", "label_skill_idx", "label_legal_under_gate", "role_ids",
        "player", "step_ids", "t", "bh_motion_request", "bh_walk_target", "bh_intercept",
        "zweikampf_active", "ball_dist", "nearest_opponent_dist", "robot_xy", "coordination_target_xy")}
    step_log: list[dict] = []
    try:
        encoder_env = tmd._make_encoder_env()
        gates = {aid: SkillGate() for aid in players}
        team_state = TeamCoordinationState()

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            obs_by_agent = {aid: raw_obs(aid, args.timeout_ms)[1] for aid in players}
            if all(o.obs_ready for o in obs_by_agent.values()):
                break
        else:
            raise TimeoutError("team-24 observations never became ready (is the bridge exporting without override?)")
        print("observations_ready=1", flush=True)

        t0 = time.monotonic()
        period = 1.0 / max(0.1, args.hz)
        step = 0
        while time.monotonic() - t0 < args.seconds:
            step += 1
            t = time.monotonic() - t0
            payloads, obs_by_agent = {}, {}
            for aid in players:
                payloads[aid], obs_by_agent[aid] = raw_obs(aid, args.timeout_ms)
            gk_payload = dict(pybh.rl_get_obs(args.gk, args.timeout_ms))

            # Same team-coordination bookkeeping as the match harness (needed only to
            # produce the role one-hot + gate bits of the 47-dim encoding).
            agent_positions = {aid: (float(o.robot_x), float(o.robot_y)) for aid, o in obs_by_agent.items()}
            agent_pose_ages = {aid: 0.0 for aid in obs_by_agent}
            team_ball = tmd._team_ball_from_obs(obs_by_agent)
            phase = update_team_phase(team_state, team_ball)
            roles = assign_roles(obs_by_agent, team_ball, team_state.roles, phase=phase)
            striker_id = next((a for a, r in roles.items() if r == "striker"), None)
            open_id = next((a for a, r in roles.items() if r == "open_support"), None)
            off_id = next((a for a, r in roles.items() if r == "off_ball_support"), None)
            bxy = (team_ball[0], team_ball[1])
            if off_id is not None and striker_id is not None and phase == "defense":
                raw_tri, side = defender_cover_target(obs_by_agent[off_id], bxy,
                                                      current_side=team_state.defender_side_by_agent.get(off_id))
                team_state.defender_side_by_agent[off_id] = side
                team_state.triangle_ema = raw_tri if team_state.triangle_ema is None else tuple(
                    EMA_ALPHA_TRIANGLE * r + (1 - EMA_ALPHA_TRIANGLE) * e for r, e in zip(raw_tri, team_state.triangle_ema))
            if open_id is not None and striker_id is not None:
                raw_open = (defense_support_target(obs_by_agent[open_id], bxy, agent_positions[striker_id],
                                                   agent_positions.get(off_id, agent_positions[open_id]))
                            if phase == "defense" and off_id is not None
                            else compute_open_lane_target(obs_by_agent[open_id], bxy, agent_positions[striker_id]))
                team_state.open_lane_ema = raw_open if team_state.open_lane_ema is None else tuple(
                    EMA_ALPHA_OPEN_LANE * r + (1 - EMA_ALPHA_OPEN_LANE) * e for r, e in zip(raw_open, team_state.open_lane_ema))
            if off_id is not None and striker_id is not None and open_id is not None and phase != "defense":
                raw_tri = compute_triangle_target(obs_by_agent[off_id], bxy, agent_positions[striker_id],
                                                  agent_positions[open_id])
                team_state.triangle_ema = raw_tri if team_state.triangle_ema is None else tuple(
                    EMA_ALPHA_TRIANGLE * r + (1 - EMA_ALPHA_TRIANGLE) * e for r, e in zip(raw_tri, team_state.triangle_ema))
            advance_pass_target_age(team_state, 1000.0 / max(0.1, args.hz))

            for aid in players:
                role = roles.get(aid, "off_ball_support")
                obs = obs_by_agent[aid]
                coord = (team_state.open_lane_ema if role == "open_support" and team_state.open_lane_ema is not None
                         else team_state.triangle_ema if role == "off_ball_support" and team_state.triangle_ema is not None
                         else bxy) or agent_positions[aid]
                decision = build_team_decision_for_agent(
                    aid=aid, role=role, phase=phase, obs=obs, obs_by_agent=obs_by_agent,
                    agent_positions=agent_positions, agent_pose_ages=agent_pose_ages,
                    team_ball=team_ball, coord=coord, gate=gates[aid], team_state=team_state)
                encoded = np.asarray(tmd._encode_for_agent(encoder_env, obs, decision), dtype=np.float32)
                mask = gate_skill_mask_from_obs(torch.as_tensor(encoded).reshape(1, -1))[0].numpy()
                p = payloads[aid]
                bh = int(p.get("skill_behavior_skill_request", -1))
                lab = BH_TO_SKILL_IDX.get(bh, -1)
                rows["observations"].append(encoded)
                rows["bh_skill_raw"].append(bh)
                rows["label_skill_idx"].append(lab)
                rows["label_legal_under_gate"].append(bool(lab >= 0 and mask[lab] > 0.5))
                rows["role_ids"].append(ROLE_IDS[role])
                rows["player"].append(aid)
                rows["step_ids"].append(step)
                rows["t"].append(t)
                rows["bh_motion_request"].append(int(p.get("skill_behavior_motion_request", -1)))
                rows["bh_walk_target"].append((float(p.get("skill_behavior_walk_target_x", 0.0)),
                                               float(p.get("skill_behavior_walk_target_y", 0.0)),
                                               float(p.get("skill_behavior_walk_target_theta", 0.0))))
                rows["bh_intercept"].append(bool(p.get("skill_behavior_should_intercept_ball", False)))
                rows["zweikampf_active"].append(bool(p.get("zweikampf_active", False)))
                rows["ball_dist"].append(math.hypot(float(obs.ball_x) - float(obs.robot_x),
                                                    float(obs.ball_y) - float(obs.robot_y)))
                rows["nearest_opponent_dist"].append(float(obs.nearest_opponent_dist))
                rows["robot_xy"].append((float(obs.robot_x), float(obs.robot_y)))
                rows["coordination_target_xy"].append((float(coord[0]), float(coord[1])))

            any_obs = next(iter(obs_by_agent.values()))
            step_log.append({
                "step": step, "t": t,
                "gc_goals_for": max(int(o.own_score) for o in obs_by_agent.values()),
                "gc_goals_against": max(int(o.opponent_score) for o in obs_by_agent.values()),
                "ball_x": float(any_obs.ball_x), "ball_y": float(any_obs.ball_y),
                "opp_positions": sorted({(round(x / 200) * 200, round(y / 200) * 200)
                                         for o in obs_by_agent.values() for x, y in o.opponent_positions}),
                "gk_skill_raw": int(gk_payload.get("skill_behavior_skill_request", -1)),
            })
            if step % int(max(1.0, args.hz) * 60) == 0:
                labs = np.asarray(rows["bh_skill_raw"])
                print(f"t={t:6.1f}s rows={len(labs)} score={step_log[-1]['gc_goals_for']}-{step_log[-1]['gc_goals_against']} "
                      f"skills={ {BH_SKILL_NAMES[k] if 0 <= k < len(BH_SKILL_NAMES) else k: int((labs == k).sum()) for k in np.unique(labs)} }",
                      flush=True)
            time.sleep(period)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "samples.npz", **{k: np.asarray(v) for k, v in rows.items()},
                        skill_order=np.asarray(tmd.SKILL_ORDER))
    labs = np.asarray(rows["bh_skill_raw"])
    summary = {
        "source": "bhuman_selfplay_no_override", "scene": str(args.scene), "players": list(players),
        "seconds": args.seconds, "hz": args.hz, "rows": int(len(labs)),
        "bh_skill_counts": {BH_SKILL_NAMES[k] if 0 <= k < len(BH_SKILL_NAMES) else str(k): int((labs == k).sum())
                            for k in np.unique(labs)} if len(labs) else {},
        "label_legal_under_gate_rate": float(np.mean(rows["label_legal_under_gate"])) if labs.size else None,
        "gc_final": [step_log[-1]["gc_goals_for"], step_log[-1]["gc_goals_against"]] if step_log else None,
        "obs_size": 47,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "steps.json").write_text(json.dumps(step_log))
    print(f"wrote {out}  {json.dumps(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
