from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rlfootball.core.goalkeeper_action import GK_SKILL_ORDER
from rlfootball.teachers.goalkeeper_teacher import (
    GOALIE_LINE_X_MM,
    OWN_GOAL_HALF_Y_MM,
    OWN_GOAL_X_MM,
    encode_goalkeeper_obs,
    goalkeeper_teacher_action,
    goalkeeper_skill_mask_np,
)
from rlfootball.core.pybh_backend import BackendObservation
from rlfootball.training.train_goalkeeper_ppo import GoalkeeperPPOActor, _masked_logits
from rlfootball.core.config import TRAIN


DT_S = 0.10
MAX_STEPS = 80
SAVE_FREEZE_STEPS = 6
REBOUND_SPEED = 450.0
SAVE_COMMIT_STEPS = 8


@dataclass(frozen=True)
class AttackSpec:
    name: str
    start_x: float
    start_y_range: tuple[float, float]
    dribble_speed_range: tuple[float, float]
    shot_x: float
    target_y_range: tuple[float, float]
    shot_speed_range: tuple[float, float]
    lateral_cut_range: tuple[float, float] = (-120.0, 120.0)
    # When True the attacker dribbles the ball all the way toward an open goal
    # corner and NEVER releases a shot. The keeper must come off its line, cut
    # the angle and smother/clear/intercept the ball before it crosses. shot_x /
    # shot_speed_range are unused in this mode; target_y_range is the corner aim.
    is_dribble_in: bool = False
    # Stress knobs for dribble-in only. random_side: treat start_y_range and
    # target_y_range as POSITIVE magnitudes and mirror both to a random side per
    # episode (one scenario tests L and R). round_keeper: carry straight at the
    # keeper (aim y=0) until ball_x<=cut_x, then cut to the corner (beat-the-keeper).
    random_side: bool = False
    round_keeper: bool = False
    cut_x: float = -3500.0


ATTACKS = (
    AttackSpec("central_drive_finish", -1300.0, (-250.0, 250.0), (450.0, 750.0), -3200.0, (-80.0, 80.0), (850.0, 1350.0)),
    AttackSpec("left_low_finish", -1300.0, (250.0, 800.0), (450.0, 720.0), -3150.0, (120.0, 260.0), (900.0, 1450.0)),
    AttackSpec("right_low_finish", -1300.0, (-800.0, -250.0), (450.0, 720.0), -3150.0, (-260.0, -120.0), (900.0, 1450.0)),
    AttackSpec("left_post_jump", -1600.0, (450.0, 1000.0), (520.0, 820.0), -3350.0, (420.0, 690.0), (1050.0, 1700.0)),
    AttackSpec("right_post_jump", -1600.0, (-1000.0, -450.0), (520.0, 820.0), -3350.0, (-690.0, -420.0), (1050.0, 1700.0)),
    AttackSpec("diagonal_cutback_left", -900.0, (-750.0, -250.0), (550.0, 850.0), -3050.0, (250.0, 580.0), (850.0, 1300.0), (180.0, 360.0)),
    AttackSpec("diagonal_cutback_right", -900.0, (250.0, 750.0), (550.0, 850.0), -3050.0, (-580.0, -250.0), (850.0, 1300.0), (-360.0, -180.0)),
    AttackSpec("slow_roller", -1800.0, (-400.0, 400.0), (250.0, 430.0), -3400.0, (-180.0, 180.0), (450.0, 750.0)),
    AttackSpec("fast_blast", -2100.0, (-550.0, 550.0), (650.0, 950.0), -3600.0, (-500.0, 500.0), (1550.0, 2300.0)),
    AttackSpec("rebound_second_shot", -2300.0, (-500.0, 500.0), (500.0, 780.0), -3500.0, (-560.0, 560.0), (950.0, 1700.0)),
    AttackSpec("one_v_one_breakaway", -950.0, (-350.0, 350.0), (750.0, 1050.0), -3700.0, (-320.0, 320.0), (700.0, 1250.0)),
    AttackSpec("close_range_reaction", -2600.0, (-500.0, 500.0), (550.0, 850.0), -3900.0, (-360.0, 360.0), (950.0, 1700.0)),
    AttackSpec("chip_lob_proxy", -1800.0, (-800.0, 800.0), (450.0, 750.0), -3350.0, (-690.0, 690.0), (850.0, 1450.0)),
    AttackSpec("deflection_midflight", -1600.0, (-500.0, 500.0), (500.0, 850.0), -3300.0, (-620.0, 620.0), (900.0, 1650.0), (-420.0, 420.0)),
    AttackSpec("screened_late_shot", -2300.0, (-450.0, 450.0), (400.0, 700.0), -3850.0, (-420.0, 420.0), (1100.0, 2100.0)),
    # Pure dribble-into-goal: attacker carries the ball toward an open corner and
    # never shoots (shot_x / shot_speed unused). Symmetric L/R pair so the corner
    # angle is tested on both sides. target_y_range = the corner aim (|y|<800 posts).
    AttackSpec("dribble_in_left", -2700.0, (-250.0, 250.0), (380.0, 650.0), -4500.0, (340.0, 760.0), (0.0, 0.0), is_dribble_in=True),
    AttackSpec("dribble_in_right", -2700.0, (-250.0, 250.0), (380.0, 650.0), -4500.0, (-760.0, -340.0), (0.0, 0.0), is_dribble_in=True),
    # STRESS variants — each steps OUTSIDE the slow-corner head's trigger box to
    # probe how far the fix generalizes. random_side mirrors L/R within the scenario.
    #  fast      : carry > 900 mm/s -> breaks the head's speed<=0.30(norm) gate.
    #  offcenter : ball carried in a wide lane -> breaks central_carry |rel_y|<=0.12.
    #  round_keeper: straight at keeper then late cut to the open side (1v1 round).
    AttackSpec("dribble_in_fast", -2700.0, (0.0, 250.0), (1000.0, 1500.0), -4500.0, (340.0, 760.0), (0.0, 0.0), is_dribble_in=True, random_side=True),
    AttackSpec("dribble_in_offcenter", -2700.0, (550.0, 950.0), (430.0, 700.0), -4500.0, (470.0, 780.0), (0.0, 0.0), is_dribble_in=True, random_side=True),
    AttackSpec("dribble_in_round_keeper", -2500.0, (0.0, 200.0), (620.0, 980.0), -4500.0, (430.0, 760.0), (0.0, 0.0), is_dribble_in=True, random_side=True, round_keeper=True),
)


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return (_REPO / p).resolve() if not p.is_absolute() else p.resolve()


def _field_to_robot(robot_x: float, robot_y: float, robot_theta: float, x: float, y: float) -> tuple[float, float]:
    dx = x - robot_x
    dy = y - robot_y
    c = math.cos(-robot_theta)
    s = math.sin(-robot_theta)
    return dx * c - dy * s, dx * s + dy * c


def _obs(
    *,
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    pass_options: float = 0.0,
) -> BackendObservation:
    rel_x, rel_y = _field_to_robot(robot_x, robot_y, robot_theta, ball_x, ball_y)
    return BackendObservation(
        ball_x=float(ball_x),
        ball_y=float(ball_y),
        robot_x=float(robot_x),
        robot_y=float(robot_y),
        robot_theta=float(robot_theta),
        simrobot_robot_x=float(robot_x),
        simrobot_robot_y=float(robot_y),
        simrobot_robot_theta=float(robot_theta),
        frame=1,
        obs_ready=True,
        estimated_ball_x=float(ball_x),
        estimated_ball_y=float(ball_y),
        ball_rel_x=float(rel_x),
        ball_rel_y=float(rel_y),
        ball_end_rel_x=float(rel_x + 0.75 * ball_vx),
        ball_end_rel_y=float(rel_y + 0.75 * ball_vy),
        ball_vel_x=float(ball_vx),
        ball_vel_y=float(ball_vy),
        time_since_ball_seen=0.0,
        natural_time_since_ball_seen=0.0,
        ball_seen_percentage=100.0,
        ball_consistent_with_game_state=True,
        pass_options_count=float(pass_options),
        global_team_ball_x_mm=float(ball_x),
        global_team_ball_y_mm=float(ball_y),
        global_team_ball_age_ms=0.0,
        global_team_ball_valid=True,
        goal_opponent_x_mm=4500.0,
        goal_opponent_y_mm=0.0,
        nearest_opponent_dist=max(0.0, math.hypot(ball_x - robot_x, ball_y - robot_y) - 250.0),
        nearest_opponent_front_dist=max(0.0, ball_x - robot_x),
        motion_engine_ground_contact=True,
        motion_engine_fall_state=0,
    )


def _load_actor(checkpoint: Path, device: torch.device) -> GoalkeeperPPOActor:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt["policy_state_dict"]
    actor = GoalkeeperPPOActor(int(sd["trunk.0.weight"].shape[1]), TRAIN.hidden_size).to(device)
    actor.load_state_dict(sd, strict=False)
    actor.eval()
    return actor


@torch.no_grad()
def _policy_action(actor: GoalkeeperPPOActor, obs_vec: np.ndarray, device: torch.device) -> tuple[str, np.ndarray, np.ndarray]:
    obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).view(1, -1)
    logits, params, _ = actor(obs_t)
    masked = _masked_logits(logits, obs_t)
    skill_id = int(torch.argmax(masked, dim=-1).item())
    return GK_SKILL_ORDER[skill_id], params.detach().cpu().numpy().reshape(-1), torch.softmax(masked, dim=-1).detach().cpu().numpy().reshape(-1)


def _teacher_action(obs: BackendObservation) -> tuple[str, np.ndarray, np.ndarray]:
    action, _cognition = goalkeeper_teacher_action(obs)
    return action.skill_name, action.as_numpy()[1:], np.zeros(len(GK_SKILL_ORDER), dtype=np.float32)


def _shot_velocity(ball_x: float, ball_y: float, target_y: float, speed: float) -> tuple[float, float]:
    dx = OWN_GOAL_X_MM - ball_x
    dy = target_y - ball_y
    norm = max(1e-6, math.hypot(dx, dy))
    return speed * dx / norm, speed * dy / norm


def _crossing_y(ball_x: float, ball_y: float, vx: float, vy: float) -> float | None:
    return _line_crossing_y(OWN_GOAL_X_MM, ball_x, ball_y, vx, vy)


def _line_crossing_y(line_x: float, ball_x: float, ball_y: float, vx: float, vy: float) -> float | None:
    if vx >= -1e-6:
        return None
    t = (line_x - ball_x) / vx
    if t < 0.0:
        return None
    return ball_y + vy * t


def _save_radius(skill: str) -> float:
    if skill == "stand":
        return 95.0
    if skill == "walk_goalie_pose":
        return 170.0
    if skill == "intercept_center":
        return 85.0
    if skill == "intercept_lateral":
        return 145.0
    if skill in {"intercept_low_left", "intercept_low_right"}:
        return 240.0
    if skill in {"intercept_jump_left", "intercept_jump_right"}:
        return 650.0
    return 0.0


def _skill_matches_crossing(skill: str, crossing_y: float) -> bool:
    if skill == "stand":
        return abs(crossing_y) <= 95.0
    if skill == "walk_goalie_pose":
        return abs(crossing_y) <= 170.0
    if skill == "intercept_center":
        return abs(crossing_y) <= 85.0
    if skill == "intercept_lateral":
        return abs(crossing_y) <= 145.0
    if skill == "intercept_low_left":
        return 0.0 < crossing_y <= 240.0
    if skill == "intercept_low_right":
        return -240.0 <= crossing_y < 0.0
    if skill == "intercept_jump_left":
        return 120.0 < crossing_y <= 700.0
    if skill == "intercept_jump_right":
        return -700.0 <= crossing_y < -120.0
    return False


def _dribble_engage_reach(skill: str) -> float:
    """Distance (mm) within which the keeper stops a *dribbled* (slow, ground)
    ball given the chosen skill. Tighter than shot-save radii: you cannot smother
    a ball at full dive range, you must get your body/feet to it. Body-block on
    stand/walk lets the keeper stop balls that pass through its position."""
    if skill in {"intercept_jump_left", "intercept_jump_right"}:
        return 300.0
    if skill in {"intercept_low_left", "intercept_low_right"}:
        return 240.0
    if skill == "intercept_lateral":
        return 200.0
    if skill == "intercept_center":
        return 150.0
    if skill in {"clear", "dribble_out", "pass"}:
        return 260.0
    if skill == "walk_goalie_pose":
        return 150.0
    if skill == "stand":
        return 110.0
    return 0.0


def run_episode(
    policy,
    spec: AttackSpec,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, object]:
    robot_x = GOALIE_LINE_X_MM
    robot_y = float(rng.normal(0.0, 80.0))
    robot_theta = 0.0
    ball_x = float(spec.start_x)
    dribble_speed = float(rng.uniform(*spec.dribble_speed_range))
    if spec.random_side:
        side = 1.0 if rng.random() < 0.5 else -1.0
        ball_y = side * float(rng.uniform(*spec.start_y_range))
        target_y = side * float(rng.uniform(*spec.target_y_range))
    else:
        ball_y = float(rng.uniform(*spec.start_y_range))
        target_y = float(rng.uniform(*spec.target_y_range))
    shot_speed = float(rng.uniform(*spec.shot_speed_range))
    lateral_cut = float(rng.uniform(*spec.lateral_cut_range))
    shot_fired = False
    saved = False
    save_step = -1
    goal = False
    out = False
    freeze = 0
    skills: list[str] = []
    first_save_skill = ""
    min_ttc = float("inf")
    final_crossing = None
    committed_skill = ""
    committed_steps_left = 0

    for step in range(MAX_STEPS):
        if spec.is_dribble_in:
            # Attacker dribbles the ball toward the open corner (OWN_GOAL_X_MM,
            # aim_y) at dribble pace; no shot is ever released. round_keeper carries
            # straight at the keeper (aim_y=0) until close, then cuts to the corner.
            # Once the keeper has smothered/cleared (saved), the dribble pursuit
            # stops and the ball keeps the rebound velocity set in the save block —
            # otherwise the attacker would "re-acquire" a cleared ball every step.
            # NOTE: this branch must own the whole if/elif so a saved dribble-in
            # never falls through to the `not shot_fired` shot-dribble branch below.
            if not saved:
                aim_y = target_y
                if spec.round_keeper and ball_x > spec.cut_x:
                    aim_y = 0.0
                dx = OWN_GOAL_X_MM - ball_x
                dy = aim_y - ball_y
                nrm = max(1e-6, math.hypot(dx, dy))
                ball_vx = dribble_speed * dx / nrm
                ball_vy = dribble_speed * dy / nrm
        elif not shot_fired:
            ball_vx = -dribble_speed
            desired_y = ball_y + lateral_cut * DT_S
            ball_vy = float(np.clip((desired_y - ball_y) / DT_S, -450.0, 450.0))
            if ball_x <= spec.shot_x:
                shot_fired = True
                ball_vx, ball_vy = _shot_velocity(ball_x, ball_y, target_y, shot_speed)
        crossing = _crossing_y(ball_x, ball_y, ball_vx, ball_vy)
        final_crossing = crossing
        obs = _obs(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_theta=robot_theta,
            ball_x=ball_x,
            ball_y=ball_y,
            ball_vx=ball_vx,
            ball_vy=ball_vy,
        )
        encoded = encode_goalkeeper_obs(obs)
        skill, params, probs = policy(obs, encoded, device)
        if goalkeeper_skill_mask_np(encoded)[GK_SKILL_ORDER.index(skill)] <= 0.0:
            skill = "stand"
        skills.append(skill)
        if skill.startswith("intercept_"):
            committed_skill = skill
            committed_steps_left = SAVE_COMMIT_STEPS
        elif committed_steps_left > 0:
            committed_steps_left -= 1
        else:
            committed_skill = ""

        if skill == "walk_goalie_pose":
            target_x = float(np.clip(params[0], -1.0, 1.0) * 4500.0)
            target_y_walk = float(np.clip(params[1], -1.0, 1.0) * 3000.0)
            dx = np.clip(target_x - robot_x, -220.0 * DT_S, 220.0 * DT_S)
            dy = np.clip(target_y_walk - robot_y, -260.0 * DT_S, 260.0 * DT_S)
            robot_x += float(dx)
            robot_y += float(dy)
        elif skill == "stand":
            pass

        if shot_fired and crossing is not None:
            ttc = (OWN_GOAL_X_MM - ball_x) / ball_vx if ball_vx < 0.0 else float("inf")
            min_ttc = min(min_ttc, float(ttc))
            keeper_crossing = _line_crossing_y(robot_x, ball_x, ball_y, ball_vx, ball_vy)
            if keeper_crossing is None:
                keeper_crossing = crossing
            keeper_ttc = (robot_x - ball_x) / ball_vx if ball_vx < 0.0 else float("inf")
            rel_crossing = keeper_crossing - robot_y
            save_skill = skill if _skill_matches_crossing(skill, rel_crossing) else committed_skill
            if (
                not saved
                and 0.0 <= keeper_ttc <= 1.15
                and _skill_matches_crossing(save_skill, rel_crossing)
                and abs(rel_crossing) <= _save_radius(save_skill)
            ):
                saved = True
                save_step = step
                first_save_skill = save_skill
                freeze = SAVE_FREEZE_STEPS
                ball_x = robot_x + 250.0
                ball_y = robot_y + np.sign(rel_crossing or 1.0) * 220.0
                ball_vx = REBOUND_SPEED
                ball_vy = float(rng.normal(0.0, 180.0))

        if spec.is_dribble_in and not saved:
            # Keeper stops the dribbled ball when its body/reach gets to the ball.
            # Coming off the line early is required to reach a corner-bound dribble;
            # staying central only blocks central runs (body-block via stand/walk).
            engage_dist = math.hypot(ball_x - robot_x, ball_y - robot_y)
            if engage_dist <= _dribble_engage_reach(skill):
                saved = True
                save_step = step
                first_save_skill = skill
                freeze = SAVE_FREEZE_STEPS
                ball_x = robot_x + 250.0
                ball_y = robot_y
                ball_vx = REBOUND_SPEED
                ball_vy = float(rng.normal(0.0, 120.0))

        if freeze > 0:
            freeze -= 1
        else:
            ball_x += ball_vx * DT_S
            ball_y += ball_vy * DT_S

        if ball_x <= OWN_GOAL_X_MM:
            goal = abs(ball_y) <= OWN_GOAL_HALF_Y_MM
            out = not goal
            break
        if saved and ball_x > GOALIE_LINE_X_MM + 900.0:
            break
        if abs(ball_y) > 1800.0 and shot_fired:
            out = True
            break

    counts = Counter(skills)
    return {
        "scenario": spec.name,
        "goal": bool(goal),
        "saved": bool(saved),
        "out": bool(out),
        "shot_fired": bool(shot_fired),
        "steps": int(step + 1),
        "save_step": int(save_step),
        "save_skill": first_save_skill,
        "final_ball": [float(ball_x), float(ball_y)],
        "goal_crossing_y": None if final_crossing is None else float(final_crossing),
        "min_ttc": None if not math.isfinite(min_ttc) else float(min_ttc),
        "skill_counts": dict(sorted(counts.items())),
        "dominant_skill": counts.most_common(1)[0][0] if counts else "",
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario"])].append(row)
    scenario_summary = {}
    all_counts = Counter()
    for row in rows:
        all_counts.update(row.get("skill_counts", {}))
    for name, items in sorted(by_scenario.items()):
        n = len(items)
        goals = sum(1 for item in items if item["goal"])
        saves = sum(1 for item in items if item["saved"])
        outs = sum(1 for item in items if item["out"])
        skill_counts = Counter()
        save_skills = Counter()
        for item in items:
            skill_counts.update(item.get("skill_counts", {}))
            if item.get("save_skill"):
                save_skills[str(item["save_skill"])] += 1
        scenario_summary[name] = {
            "episodes": n,
            "goals": goals,
            "saves": saves,
            "outs": outs,
            "goal_rate": goals / max(1, n),
            "save_rate": saves / max(1, n),
            "dominant_skills": dict(skill_counts.most_common(6)),
            "save_skills": dict(save_skills),
        }
    goals = sum(1 for row in rows if row["goal"])
    saves = sum(1 for row in rows if row["saved"])
    outs = sum(1 for row in rows if row["out"])
    effective_goals_conceded = len(rows) - saves - outs
    return {
        "episodes": len(rows),
        "goals": goals,
        "effective_goals_conceded": effective_goals_conceded,
        "saves": saves,
        "outs": outs,
        "goal_rate": goals / max(1, len(rows)),
        "effective_goal_rate": effective_goals_conceded / max(1, len(rows)),
        "save_rate": saves / max(1, len(rows)),
        "out_rate": outs / max(1, len(rows)),
        "policy_skill_counts": dict(sorted(all_counts.items())),
        "scenarios": scenario_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("ppo", "teacher"), default="ppo")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--episodes", type=int, default=900)
    parser.add_argument("--seed", type=int, default=742619)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="")
    parser.add_argument("--dump-episodes", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    actor = None
    if args.policy == "ppo":
        if not args.checkpoint:
            parser.error("--checkpoint is required when --policy=ppo")
        actor = _load_actor(_resolve(args.checkpoint), device)

        def policy(obs: BackendObservation, encoded: np.ndarray, dev: torch.device):
            return _policy_action(actor, encoded, dev)

        policy_ref = str(_resolve(args.checkpoint))
    else:
        def policy(obs: BackendObservation, encoded: np.ndarray, dev: torch.device):
            return _teacher_action(obs)

        policy_ref = "goalkeeper_teacher_action"

    rows = []
    for i in range(args.episodes):
        spec = ATTACKS[i % len(ATTACKS)]
        rows.append(run_episode(policy, spec, rng, device))
    result = {
        "policy": args.policy,
        "checkpoint": policy_ref,
        "benchmark": "scripted_closed_loop_striker_dribble_shots_v2",
        "dt_s": DT_S,
        "summary": summarize(rows),
        "episodes": rows if args.dump_episodes else [],
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output:
        out = _resolve(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
