"""Unified live-match probe: one harness, four controllers, identical instrumentation.

Scene: BHvsBH4v4_3D (scenario 4v4_BH3D). Both goalkeepers and all four opponents are
stock B-Human (enableEmbeddedGK=false). Team-24 field players 2,3,4 are driven by:

  --controller expert   teacher + B-Human ball macro (striker nearest a fresh ball within
                        --bh-macro-radius-mm plays smash_or_pass -> shoot / passTo)
  --controller teacher  plain heuristic teacher (multiagent_teacher.team_teacher_payload)
  --controller v5       v5 merged policy, gate-masked argmax, NO force-shoot/force-walk crutches
  --controller bhuman   no override: stock B-Human plays team 24 too (control arm C);
                        Python only observes

Every arm logs the SAME per-step record, so arms are compared with one instrument:
  * actor skill + three shadow labels on the same encoded state: teacher, expert, v5
    (+ B-Human's own executed SkillRequest from the bridge) and their legality under the
    CURRENT v5 gate mask  -> DAgger-ready relabel data;
  * per robot (incl. our B-Human GK, player 1): estimated pose, SIMULATOR ground-truth pose
    (simrobot_robot_*), ball estimate (field), ball velocity, seen/age, nearest opponent,
    fall state, executed motion, Zweikampf, contacts;
  * per step: GameController score, fused opponent positions.
Encoded 47-dim obs + label indices for the 3 field players go to a companion .npz.

Observation mode: PYBH_RL_OBS_MODE=estimated (set by default here). The bridge default
("corrected") replaces a stale ball (>1 s unseen) with the last reset position flagged as
freshly seen -- a training convenience that does not exist on the robot. B-Human's own
behaviour never reads it.

Freeze watchdog: if 20 consecutive control periods exceed 1 s (bridge stalled), the match
aborts with exit code 3 and the JSON is marked invalid, so the runner can replay it.
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

CONTROLLERS = ("expert", "teacher", "v5", "bhuman", "dagger", "hybrid")


def _early_controller() -> tuple[str, str]:
    ctrl, players = "expert", "2,3,4"
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--controller" and i + 1 < len(argv):
            ctrl = argv[i + 1]
        elif a.startswith("--controller="):
            ctrl = a.split("=", 1)[1]
        elif a == "--active-players" and i + 1 < len(argv):
            players = argv[i + 1]
        elif a == "--env" and i + 1 < len(argv) and "=" in argv[i + 1]:
            # per-match knobs read at import time (e.g. RL_MA_PASS_VALUE_MARGIN) must be set here,
            # before multiagent_teacher is imported
            k, v = argv[i + 1].split("=", 1)
            os.environ[k] = v
    return ctrl, players


_CTRL, _PLAYERS = _early_controller()
# Must precede the harness import (it setdefault()s these). bhuman arm: nobody overridden.
os.environ["PYBH_RL_OVERRIDE_TEAM"] = "24"
os.environ["PYBH_RL_ACTIVE_PLAYERS"] = "" if _CTRL == "bhuman" else _PLAYERS
os.environ.setdefault("PYBH_RL_OBS_MODE", "estimated")
os.environ.pop("PYBH_PPO_ACTIVE_PLAYERS", None)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import rlfootball.evaluation.run_v5_vs_bhuman_match as h  # noqa: E402  (chdirs into BHUMAN_DIR, loads pybh)
from rlfootball.evaluation.run_v5_vs_bhuman_match import (  # noqa: E402
    BHUMAN_DIR, SIMROBOT_BIN, tmd, pybh, BackendObservation, SkillGate, SkillType,
    StructuredSkillAction, TeamCoordinationState, assign_roles, update_team_phase,
    defender_cover_target, defense_support_target, compute_open_lane_target,
    compute_triangle_target, advance_pass_target_age, build_agent_id_map,
    teammate_slots_from_obs, smash_or_pass, EMA_ALPHA_TRIANGLE, EMA_ALPHA_OPEN_LANE,
    build_team_decision_for_agent, gate_skill_mask_from_obs, send_action, _correct_is_gk,
)
from rlfootball.core.action import SKILL_ORDER, action_param_mask  # noqa: E402
from rlfootball.core.observation import ObsIndex  # noqa: E402
from rlfootball.teachers.multiagent_teacher import team_teacher_payload, record_pass_target  # noqa: E402

SKILL_IDX = {name: i for i, name in enumerate(SKILL_ORDER)}
# B-Human SkillRequest::Type -> SKILL_ORDER index (none/interceptBall/keeperDive -> -1)
BH_TO_SKILL_IDX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
ROLE_IDS = {"striker": 0, "open_support": 1, "off_ball_support": 2}
SHOOT_ARMED, SHOOT_PROGRESS, PASS_ARMED = int(ObsIndex.shoot_armed), int(ObsIndex.shoot_arm_progress), int(ObsIndex.pass_armed)
FREEZE_DT_S = 1.0
FREEZE_CONSECUTIVE = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified live-match probe (see module docstring).")
    p.add_argument("--controller", choices=CONTROLLERS, required=True)
    p.add_argument("--scene", type=Path, default=BHUMAN_DIR / "Config/Scenes/BHvsBH4v4_3D.ros2")
    p.add_argument("--simrobot-bin", type=Path, default=SIMROBOT_BIN)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("runs/merge_v5_attack_defense/checkpoint_stage8_step10000.pt"))
    p.add_argument("--launch", action="store_true")
    p.add_argument("--seconds", type=float, default=600.0)
    p.add_argument("--hz", type=float, default=8.0)
    p.add_argument("--timeout-ms", type=int, default=1200)
    p.add_argument("--active-players", type=str, default="2,3,4")
    p.add_argument("--gk", type=int, default=1)
    p.add_argument("--learned-roles", type=str, default="striker",
                   help="hybrid/handover: roles the student drives; the rest are handed to B-Human each frame")
    p.add_argument("--sample", action="store_true",
                   help="RL rollouts: sample skill (gate-masked softmax) and the movable position params "
                        "(y, theta; Gaussian, std --param-std) instead of argmax; log-probs go to the npz")
    p.add_argument("--param-std", type=float, default=0.15)
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE set before imports (applied early)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--coord-checkpoint", type=Path, default=None,
                   help="learned support coordinator (rlfootball/training/learned_coordinator.py) replacing the rule-based target")
    p.add_argument("--team-field-players", type=str, default="2,3,4",
                   help="all team-24 field players; those not in --active-players are logged as 'others'")
    p.add_argument("--teacher-stage", type=int, default=8)
    p.add_argument("--bh-macro-radius-mm", type=float, default=1500.0)
    p.add_argument("--output", type=Path, required=True, help="match JSON path (npz written alongside)")
    return p.parse_args()


def raw(player: int, timeout_ms: int) -> tuple[dict, BackendObservation]:
    payload = dict(pybh.rl_get_obs(player, timeout_ms))
    return payload, BackendObservation.from_payload(payload)


def robot_record(p: dict, o: BackendObservation) -> dict:
    """Everything soccer-relevant the bridge exports for one team-24 robot."""
    return {
        "x": float(o.robot_x), "y": float(o.robot_y), "th": float(o.robot_theta),
        "gt_x": float(p.get("simrobot_robot_x", 0.0)), "gt_y": float(p.get("simrobot_robot_y", 0.0)),
        "gt_th": float(p.get("simrobot_robot_theta", 0.0)),
        "bx": float(p.get("estimated_ball_x", o.ball_x)), "by": float(p.get("estimated_ball_y", o.ball_y)),
        "bvx": float(o.ball_vel_x), "bvy": float(o.ball_vel_y),
        "b_seen": float(o.natural_ball_seen_percentage), "b_age": float(o.natural_time_since_ball_seen),
        "b_dist": math.hypot(float(o.ball_rel_x), float(o.ball_rel_y)),
        "opp_d": float(o.nearest_opponent_dist), "opp_front_d": float(o.nearest_opponent_front_dist),
        "fall": int(o.motion_engine_fall_state),
        "motion": int(p.get("skill_behavior_motion_request", -1)),
        "exec_motion": int(p.get("motion_request", -1)),
        "bh_skill": int(p.get("skill_behavior_skill_request", -1)),
        # B-Human's own walk target (MotionRequest.walkTarget, robot-relative) -> positioning labels
        "bh_wt_x": float(p.get("skill_behavior_walk_target_x", 0.0)),
        "bh_wt_y": float(p.get("skill_behavior_walk_target_y", 0.0)),
        "bh_wt_th": float(p.get("skill_behavior_walk_target_theta", 0.0)),
        "bh_pass_target": int(p.get("requested_pass_target", -1)),
        # DAgger shadow label: B-Human's would-be SkillRequest for an overridden robot (field frame)
        "shadow_skill": int(p.get("shadow_bh_skill", -1)),
        "shadow_tx": float(p.get("shadow_bh_target_x", 0.0)), "shadow_ty": float(p.get("shadow_bh_target_y", 0.0)),
        "shadow_tth": float(p.get("shadow_bh_target_theta", 0.0)),
        "shadow_pass": int(p.get("shadow_bh_pass_target", -1)), "shadow_frame": int(p.get("shadow_bh_frame", 0)),
        "zk": bool(o.zweikampf_active),
        "arm_c": bool(o.obstacle_arm_contact), "foot_c": bool(o.obstacle_foot_contact),
        "shot_open": float(o.shot_opening_with_obstacles), "can_score": bool(o.can_score_now),
    }


def ema(new, old, a):
    return new if old is None else tuple(a * n + (1 - a) * o for n, o in zip(new, old))


def main() -> int:
    args = parse_args()
    out_json = h._resolve_repo_path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    ckpt = h._resolve_repo_path(args.checkpoint)
    players = h.parse_player_numbers(args.active_players)
    other_players = [p for p in h.parse_player_numbers(args.team_field_players) if p not in players]
    acting = args.controller != "bhuman"

    proc = None
    sim_raw = out_json.with_suffix(".simraw.log")
    sim_fh = None
    if args.launch:
        # Capture SimRobot output (freeze diagnosis); condensed to .sim.log at the end.
        sim_fh = open(sim_raw, "w")
        proc = subprocess.Popen([str(args.simrobot_bin), str(args.scene)], cwd=BHUMAN_DIR,
                                env=os.environ.copy(), stdout=sim_fh, stderr=subprocess.STDOUT)
    print(f"probe controller={args.controller} scene={args.scene.name} players={players} "
          f"obs_mode={os.environ.get('PYBH_RL_OBS_MODE')} override={os.environ['PYBH_RL_ACTIVE_PLAYERS']!r}",
          flush=True)

    steps: list[dict] = []
    npz = {k: [] for k in ("obs", "player", "step", "t", "role", "actor", "teacher", "expert", "v5",
                           "bh", "legal_teacher", "legal_expert", "legal_v5", "macro",
                           "learned", "samp_skill", "samp_params", "logp")}
    rng = np.random.default_rng(args.seed)
    SAMPLED_DIMS = (1, 2)  # residual y, theta: the only movable position params (x range 0, pass target from PlayBall)
    invalid_reason = None
    try:
        policy = tmd._load_eval_policy(str(ckpt))
        learned_roles = {r.strip() for r in args.learned_roles.split(",") if r.strip()}
        coord_net = None
        if args.coord_checkpoint is not None:
            from rlfootball.training.learned_coordinator import load_coord_net, predict_field_target
            coord_net = load_coord_net(h._resolve_repo_path(args.coord_checkpoint))
        enc = tmd._make_encoder_env()
        gates = {aid: SkillGate() for aid in players}
        ts = TeamCoordinationState()
        prev_teacher = {aid: None for aid in players}

        if acting:
            for aid in players:
                send_action(aid, StructuredSkillAction(skill_type=int(SkillType.stand)))
        deadline = time.monotonic() + 60.0
        while True:
            obs_by = {aid: raw(aid, args.timeout_ms)[1] for aid in players}
            if all(o.obs_ready for o in obs_by.values()):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("observations never ready")
        print("observations_ready=1", flush=True)

        t0 = time.monotonic()
        period = 1.0 / max(0.1, args.hz)
        step, slow_streak, last_t = 0, 0, 0.0
        while time.monotonic() - t0 < args.seconds:
            step += 1
            t = time.monotonic() - t0
            slow_streak = slow_streak + 1 if (t - last_t) > FREEZE_DT_S and step > 1 else 0
            last_t = t
            if slow_streak >= FREEZE_CONSECUTIVE:
                invalid_reason = f"bridge_freeze_at_{t:.0f}s"
                print(f"FREEZE detected at t={t:.0f}s -> aborting match", flush=True)
                break

            payloads, obs_by = {}, {}
            for aid in players:
                payloads[aid], obs_by[aid] = raw(aid, args.timeout_ms)
            gk_p, gk_o = raw(args.gk, args.timeout_ms)
            others = {pid: raw(pid, args.timeout_ms) for pid in other_players}

            # --- team coordination bookkeeping (identical to the match harness) ---
            pos = {aid: (float(o.robot_x), float(o.robot_y)) for aid, o in obs_by.items()}
            ages = {aid: 0.0 for aid in obs_by}
            team_ball = tmd._team_ball_from_obs(obs_by)
            bxy = (team_ball[0], team_ball[1])
            phase = update_team_phase(ts, team_ball)
            roles = assign_roles(obs_by, team_ball, ts.roles, phase=phase)
            sid = next((a for a, r in roles.items() if r == "striker"), None)
            oid = next((a for a, r in roles.items() if r == "open_support"), None)
            fid = next((a for a, r in roles.items() if r == "off_ball_support"), None)
            if fid is not None and sid is not None and phase == "defense":
                tri, side = defender_cover_target(obs_by[fid], bxy, current_side=ts.defender_side_by_agent.get(fid))
                ts.defender_side_by_agent[fid] = side
                ts.triangle_ema = ema(tri, ts.triangle_ema, EMA_ALPHA_TRIANGLE)
            if oid is not None and sid is not None:
                op = (defense_support_target(obs_by[oid], bxy, pos[sid], pos.get(fid, pos[oid]))
                      if phase == "defense" and fid is not None
                      else compute_open_lane_target(obs_by[oid], bxy, pos[sid]))
                ts.open_lane_ema = ema(op, ts.open_lane_ema, EMA_ALPHA_OPEN_LANE)
            if fid is not None and sid is not None and oid is not None and phase != "defense":
                ts.triangle_ema = ema(compute_triangle_target(obs_by[fid], bxy, pos[sid], pos[oid]),
                                      ts.triangle_ema, EMA_ALPHA_TRIANGLE)
            advance_pass_target_age(ts, 1000.0 / max(0.1, args.hz))

            def fresh_d(o):
                if o.ball_seen_percentage <= 0.0 or o.natural_time_since_ball_seen > 1000.0:
                    return float("inf")
                return math.hypot(o.ball_x - o.robot_x, o.ball_y - o.robot_y)
            fd = {aid: fresh_d(o) for aid, o in obs_by.items()}
            nearest = min(fd, key=fd.get)

            row = {"step": step, "t": t, "phase": phase, "agents": {}, "gk": robot_record(gk_p, gk_o),
                   "others": {str(pid): robot_record(pp, oo) for pid, (pp, oo) in others.items()}}
            passthrough = set()
            actions = {}
            for aid in players:
                role = roles.get(aid, "off_ball_support")
                o = obs_by[aid]
                coord = (ts.open_lane_ema if role == "open_support" and ts.open_lane_ema is not None
                         else ts.triangle_ema if role == "off_ball_support" and ts.triangle_ema is not None
                         else bxy) or pos[aid]
                dec = build_team_decision_for_agent(
                    aid=aid, role=role, phase=phase, obs=o, obs_by_agent=obs_by, agent_positions=pos,
                    agent_pose_ages=ages, team_ball=team_ball, coord=coord, gate=gates[aid], team_state=ts)
                encoded = np.asarray(tmd._encode_for_agent(enc, o, dec), dtype=np.float32)
                coord_used = coord
                if coord_net is not None and role in ("open_support", "off_ball_support"):
                    coord_used = predict_field_target(coord_net, encoded)

                slots = _correct_is_gk(teammate_slots_from_obs(o, build_agent_id_map(aid, pos, ages)), o)
                _pl, t_abs, t_res = team_teacher_payload(
                    o, args.teacher_stage, prev_teacher[aid], gate_decision=dec, role=role,
                    coordination_target=coord, teammate_slots=slots,
                    last_pass_target=ts.last_pass_target_by_striker.get(aid),
                    last_pass_target_age_ms=ts.last_pass_target_age_ms_by_striker.get(aid, 0.0),
                    phase=phase, team_ball_valid=bool(team_ball[2]), face_ball_xy=bxy)
                prev_teacher[aid] = t_res.skill_name
                macro = (args.bh_macro_radius_mm > 0 and role == "striker" and aid == nearest
                         and fd[aid] <= args.bh_macro_radius_mm)
                e_abs = t_abs
                if macro:
                    prop, pa = smash_or_pass(o, slots, ts.last_pass_target_by_striker.get(aid),
                                             ts.last_pass_target_age_ms_by_striker.get(aid, 0.0))
                    e_abs = (StructuredSkillAction(skill_type=int(SkillType.pass_), pass_target=int(pa))
                             if prop == "pass" and pa >= 0 else StructuredSkillAction(skill_type=int(SkillType.shoot)))

                # dagger arm: Design-A baking identical to rlfootball/training/train_dagger_v5.py::bake --
                # macro state => shoot-armed (+progress); pass-armed when PlayBall proposes a pass.
                if args.controller in ("dagger", "hybrid") and macro:
                    encoded[SHOOT_ARMED] = 1.0
                    encoded[SHOOT_PROGRESS] = 1.0
                    if e_abs.skill_name == "pass":
                        encoded[PASS_ARMED] = 1.0
                et = torch.as_tensor(encoded).reshape(1, -1)
                with torch.no_grad():
                    logits, pmean = policy(et)
                mask = gate_skill_mask_from_obs(et)
                v5_idx = int((logits + (mask - 1.0) * 1e9).argmax(1).item())
                mk = mask[0].numpy()

                samp_skill, samp_params, logp = -1, np.zeros(4, np.float32), 0.0
                if args.controller in ("v5", "dagger", "hybrid"):
                    pr = pmean[0].numpy().astype(np.float64).copy()
                    skill_idx = v5_idx
                    if args.sample:
                        # Gate-masked softmax over skills; Gaussian on the movable params. The
                        # unclipped sample is logged (log-prob), the clipped one is executed.
                        lg = (logits + (mask - 1.0) * 1e9)[0].double()
                        probs = torch.softmax(lg, 0).numpy()
                        skill_idx = int(rng.choice(len(probs), p=probs / probs.sum()))
                        logp = float(np.log(max(probs[skill_idx], 1e-12)))
                        pm = action_param_mask(SKILL_ORDER[skill_idx])
                        noisy = pr.copy()
                        for d in SAMPLED_DIMS:
                            if pm[d] > 0:
                                noisy[d] = pr[d] + args.param_std * rng.standard_normal()
                                logp += float(-0.5 * ((noisy[d] - pr[d]) / args.param_std) ** 2
                                              - math.log(args.param_std * math.sqrt(2 * math.pi)))
                        samp_skill, samp_params = skill_idx, noisy.astype(np.float32)
                        pr = np.clip(noisy, -1.0, 1.0)
                    resid = StructuredSkillAction(
                        skill_type=skill_idx, target_x=float(pr[0]), target_y=float(pr[1]), target_theta=float(pr[2]),
                        pass_target=int(np.clip(round((float(pr[3]) + 1.0) * 0.5 * 2.0), 0, 2)))
                    enc._team_role = role
                    enc._team_coordination_target = coord_used
                    act = enc._decode_residual_policy_action(resid, obs=o)
                    # Expert pass targets were not logged, so the student's param head never
                    # learned them: in the dagger arm a pass goes to PlayBall's chosen receiver.
                    if args.controller in ("dagger", "hybrid") and act.skill_name == "pass" and e_abs.skill_name == "pass":
                        act = replace(act, pass_target=int(e_abs.pass_target))
                elif args.controller == "teacher":
                    act = t_abs
                elif args.controller == "expert":
                    act = e_abs
                else:
                    act = None
                # hybrid: the learned student keeps only the striker; the other robots are
                # handed back to B-Human's own strategy for this frame (C++ "bhuman" passthrough).
                if args.controller == "hybrid" and role not in learned_roles:
                    act = None
                    passthrough.add(aid)
                if act is not None:
                    actions[aid] = act
                    if act.skill_name == "pass":
                        record_pass_target(ts, aid, int(act.pass_target))

                bh_raw = int(payloads[aid].get("skill_behavior_skill_request", -1))
                t_i, e_i = SKILL_IDX[t_abs.skill_name], SKILL_IDX[e_abs.skill_name]
                a_i = SKILL_IDX[act.skill_name] if act is not None else BH_TO_SKILL_IDX.get(bh_raw, -1)
                rec = robot_record(payloads[aid], o)
                rec.update({
                    "role": role, "actor": SKILL_ORDER[a_i] if a_i >= 0 else "none",
                    "teacher": t_abs.skill_name, "expert": e_abs.skill_name, "v5": SKILL_ORDER[v5_idx],
                    "macro": bool(macro), "pass_target": int(act.pass_target) if act is not None and act.skill_name == "pass" else -1,
                    "legal_teacher": bool(mk[t_i] > 0.5), "legal_expert": bool(mk[e_i] > 0.5),
                    "ctrl": "bhuman" if (aid in passthrough or not acting) else "learned",
                    "coord_x": float(coord[0]), "coord_y": float(coord[1]),
                    "coord_used_x": float(coord_used[0]), "coord_used_y": float(coord_used[1]),
                })
                row["agents"][aid] = rec
                for k, v in (("obs", encoded), ("player", aid), ("step", step), ("t", t),
                             ("role", ROLE_IDS[role]), ("actor", a_i), ("teacher", t_i), ("expert", e_i),
                             ("v5", v5_idx), ("bh", bh_raw), ("legal_teacher", mk[t_i] > 0.5),
                             ("legal_expert", mk[e_i] > 0.5), ("legal_v5", True), ("macro", macro),
                             ("learned", act is not None), ("samp_skill", samp_skill if act is not None else -1),
                             ("samp_params", samp_params), ("logp", logp)):
                    npz[k].append(v)

            for aid, act in actions.items():
                send_action(aid, act)
            for aid in passthrough:
                pybh.rl_set_action(player_number=aid, skill="bhuman", target_x=0.0, target_y=0.0, target_theta=0.0)

            row["gc_for"] = max(int(o.own_score) for o in obs_by.values())
            row["gc_against"] = max(int(o.opponent_score) for o in obs_by.values())
            row["opp"] = sorted({(round(x / 200) * 200, round(y / 200) * 200)
                                 for o in list(obs_by.values()) + [gk_o] for x, y in o.opponent_positions})
            steps.append(row)
            if step % int(max(1.0, args.hz) * 60) == 0:
                print(f"t={t:6.1f}s score={row['gc_for']}-{row['gc_against']} "
                      f"roles={ {a: r['role'] for a, r in row['agents'].items()} }", flush=True)
            time.sleep(period)
    finally:
        if acting:
            for aid in players:
                try:
                    send_action(aid, StructuredSkillAction(skill_type=int(SkillType.stand)))
                except Exception:  # noqa: BLE001
                    pass
        if proc is not None:
            sim_rc = proc.poll()  # None = still alive (hung?) at match end, int = it had exited
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            sim_fh.close()
            try:
                lines = open(sim_raw, errors="replace").read().splitlines()
                keys = ("assert", "abort", "error", "exception", "segmentation", "terminate", "fatal",
                        "timeout", "warning: [rl]")
                keep = [ln for ln in lines if any(k in ln.lower() for k in keys)
                        and "attribute '" not in ln and "not found" not in ln]
                with open(out_json.with_suffix(".sim.log"), "w") as fh:
                    fh.write(f"# simrobot exit code at match end: {sim_rc} (None = still running)\n")
                    fh.write("# --- error/assert/timeout lines ---\n" + "\n".join(keep[-500:]) + "\n")
                    fh.write("# --- last 300 lines ---\n" + "\n".join(lines[-300:]) + "\n")
                os.remove(sim_raw)
            except OSError:
                pass

    dur = steps[-1]["t"] if steps else 0.0
    loop_hz = len(steps) / dur if dur > 0 else 0.0
    valid = invalid_reason is None and dur >= 0.95 * args.seconds and loop_hz >= 7.5
    if invalid_reason is None and not valid:
        invalid_reason = f"loop_{loop_hz:.2f}Hz_dur_{dur:.0f}s"
    meta = {
        "controller": args.controller, "scene": str(args.scene), "checkpoint": str(ckpt),
        "players": list(players), "gk": args.gk, "seconds": args.seconds, "hz": args.hz,
        "bh_macro_radius_mm": args.bh_macro_radius_mm, "teacher_stage": args.teacher_stage,
        "learned_roles": args.learned_roles, "coord_checkpoint": str(args.coord_checkpoint),
        "sample": args.sample, "param_std": args.param_std, "seed": args.seed, "env_overrides": args.env,
        "obs_mode": os.environ.get("PYBH_RL_OBS_MODE"),
        "gate_env": {k: v for k, v in os.environ.items() if k.startswith("RL_GATE_")},
        "loop_hz": loop_hz, "duration_s": dur, "valid": valid, "invalid_reason": invalid_reason,
        "gc_final": [steps[-1]["gc_for"], steps[-1]["gc_against"]] if steps else None,
    }
    out_json.write_text(json.dumps({"meta": meta, "steps": steps}))
    np.savez_compressed(out_json.with_suffix(".npz"),
                        **{k: np.asarray(v) for k, v in npz.items()},
                        skill_order=np.asarray(SKILL_ORDER))
    print(f"wrote {out_json} valid={valid} reason={invalid_reason} loop={loop_hz:.2f}Hz", flush=True)
    print(f"FINAL SCORE (GameController) team24={meta['gc_final']}", flush=True)
    return 0 if valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
