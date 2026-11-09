from __future__ import annotations

import argparse
import json
import math
import os
import time
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rlfootball.core.action import SkillType, StructuredSkillAction
from rlfootball.core.config import BHUMAN_DIR, SIMROBOT_BIN, TRAIN
from rlfootball.core.goalkeeper_action import GK_SKILL_ORDER
from rlfootball.teachers.goalkeeper_teacher import (
    GOALIE_LINE_X_MM,
    OWN_GOAL_HALF_Y_MM,
    OWN_GOAL_X_MM,
    encode_goalkeeper_obs,
    gk_action_to_native_bridge_name,
    goalkeeper_skill_mask_np,
)
from rlfootball.core.pybh_backend import PyBHBackend
from rlfootball.training.train_goalkeeper_ppo import GoalkeeperPPOActor, _masked_logits

import pybh.controller as pybh


DT_FRAMES = 3
MAX_STEPS = 120
DEFAULT_SCENE = BHUMAN_DIR / "Config/Scenes/RLvsBH3v3_3D.ros2"


@dataclass(frozen=True)
class ThreatSpec:
    name: str
    ball_x: float
    ball_y_range: tuple[float, float]
    target_y_range: tuple[float, float]
    speed_range: tuple[float, float]


THREATS = (
    ThreatSpec("central_drive_finish", -2600.0, (-250.0, 250.0), (-80.0, 80.0), (850.0, 1250.0)),
    ThreatSpec("close_range_reaction", -3400.0, (-420.0, 420.0), (-360.0, 360.0), (900.0, 1450.0)),
    ThreatSpec("left_post_jump", -3000.0, (350.0, 850.0), (430.0, 720.0), (1050.0, 1650.0)),
    ThreatSpec("right_post_jump", -3000.0, (-850.0, -350.0), (-720.0, -430.0), (1050.0, 1650.0)),
    ThreatSpec("fast_blast", -2400.0, (-500.0, 500.0), (-520.0, 520.0), (1500.0, 2200.0)),
    ThreatSpec("slow_roller", -2600.0, (-350.0, 350.0), (-220.0, 220.0), (420.0, 680.0)),
    ThreatSpec("dribble_in_left", -3000.0, (-160.0, 220.0), (420.0, 740.0), (430.0, 680.0)),
    ThreatSpec("dribble_in_right", -3000.0, (-220.0, 160.0), (-740.0, -420.0), (430.0, 680.0)),
)


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return (_REPO / p).resolve() if not p.is_absolute() else p.resolve()


def _load_actor(checkpoint: Path, device: torch.device) -> GoalkeeperPPOActor:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt["policy_state_dict"]
    actor = GoalkeeperPPOActor(int(sd["trunk.0.weight"].shape[1]), TRAIN.hidden_size).to(device)
    actor.load_state_dict(sd, strict=False)
    actor.eval()
    return actor


def _field_velocity(ball_x: float, ball_y: float, target_y: float, speed: float) -> tuple[float, float]:
    dx = OWN_GOAL_X_MM - ball_x
    dy = target_y - ball_y
    norm = max(1e-6, math.hypot(dx, dy))
    return speed * dx / norm, speed * dy / norm


def _set_ball_state(backend: PyBHBackend, x: float, y: float, vx: float, vy: float) -> None:
    if backend.mode == "simrobot_visible":
        request_id = int(pybh.rl_set_dynamic_world(backend.player_number, ball=(float(x), float(y), float(vx), float(vy)), own_pose=None, teammates=None, opponents=None))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            status = dict(pybh.rl_get_world_status(backend.player_number))
            if int(status.get("dynamic_applied_id", 0)) == request_id:
                if not bool(status.get("world_result_ok", True)):
                    raise RuntimeError(f"visible dynamic ball velocity failed: {status.get('world_result_error', '')}")
                return
            time.sleep(0.01)
        raise RuntimeError("timed out waiting for visible dynamic ball velocity request")
    scale = backend.simrobot_scene.position_scale
    if not pybh.simrobot_set_ball_state(float(x * scale), float(y * scale), float(vx * scale), float(vy * scale)):
        raise RuntimeError("simrobot_set_ball_state failed")
    pybh.rl_mark_embedded_dynamic_context(backend.player_number, (float(x), float(y)))


def _ball_field_state(backend: PyBHBackend) -> tuple[float, float, float, float]:
    if backend.mode == "simrobot_visible":
        payload = dict(pybh.rl_get_obs(backend.player_number, 1000))
        return (
            float(payload.get("ball_x", 0.0)),
            float(payload.get("ball_y", 0.0)),
            float(payload.get("ball_vel_x", 0.0)),
            float(payload.get("ball_vel_y", 0.0)),
        )
    state = pybh.simrobot_get_ball_state()
    if state is None:
        raise RuntimeError("simrobot_get_ball_state failed")
    scale = backend.simrobot_scene.position_scale
    return (
        float(state["x"]) / scale,
        float(state["y"]) / scale,
        float(state["vx"]) / scale,
        float(state["vy"]) / scale,
    )


@torch.no_grad()
def _send_rl_action(backend: PyBHBackend, actor: GoalkeeperPPOActor, device: torch.device) -> str:
    obs = backend.read_obs()
    encoded = encode_goalkeeper_obs(obs)
    obs_t = torch.as_tensor(encoded, dtype=torch.float32, device=device).view(1, -1)
    logits, params, _ = actor(obs_t)
    masked = _masked_logits(logits, obs_t)
    skill_id = int(torch.argmax(masked, dim=-1).item())
    if goalkeeper_skill_mask_np(encoded)[skill_id] <= 0.0:
        skill_id = 0
    skill = GK_SKILL_ORDER[skill_id]
    p = params.detach().cpu().numpy().reshape(-1)

    set_action = pybh.rl_set_visible_action if backend.mode == "simrobot_visible" else pybh.rl_set_action
    if skill == "walk_goalie_pose":
        set_action(1, "walkTo", float(np.clip(p[0], -1.0, 1.0) * 4500.0), float(np.clip(p[1], -1.0, 1.0) * 3000.0), float(np.clip(p[2], -1.0, 1.0) * math.pi), -1)
    elif skill == "observe":
        set_action(1, "observe", float(obs.global_team_ball_x_mm), float(obs.global_team_ball_y_mm), 0.0, -1)
    elif skill.startswith("intercept_"):
        # Native GK bridge names are now available in StrategyBehaviorControl external override.
        class _A:
            @property
            def skill(self):
                from rlfootball.core.goalkeeper_action import GKSkillType
                return GKSkillType.from_public_name(skill)
        set_action(1, gk_action_to_native_bridge_name(_A()), 0.0, 0.0, 0.0, -1)
    elif skill == "clear":
        set_action(1, "shoot", 0.0, 0.0, 0.0, -1)
    elif skill == "pass":
        set_action(1, "pass", 0.0, 0.0, 0.0, 2)
    elif skill == "dribble_out":
        set_action(1, "dribble", 0.0, 0.0, float(np.clip(p[2], -1.0, 1.0) * math.pi), -1)
    else:
        set_action(1, "stand", 0.0, 0.0, 0.0, -1)
    return skill


def _telemetry(player_number: int = 1) -> dict[str, object]:
    payload = dict(pybh.rl_get_obs(player_number, 1))
    return {
        "requested_skill": str(payload.get("requested_skill", "")),
        "skill_behavior_skill_request": int(payload.get("skill_behavior_skill_request", -1)),
        "motion_engine_input_request": int(payload.get("motion_engine_input_request", -1)),
        "motion_engine_effective_request": int(payload.get("motion_engine_effective_request", -1)),
        "motion_engine_dive_request": int(payload.get("motion_engine_dive_request", -1)),
        "fall_state": int(payload.get("motion_engine_fall_state", -1)),
    }


def run_episode(
    backend: PyBHBackend,
    policy: str,
    actor: GoalkeeperPPOActor | None,
    device: torch.device,
    spec: ThreatSpec,
    rng: np.random.Generator,
) -> dict[str, object]:
    ball_x = float(spec.ball_x)
    ball_y = float(rng.uniform(*spec.ball_y_range))
    target_y = float(rng.uniform(*spec.target_y_range))
    speed = float(rng.uniform(*spec.speed_range))
    vx, vy = _field_velocity(ball_x, ball_y, target_y, speed)

    obs = backend.reset(
        ball_pos=np.asarray([ball_x, ball_y], dtype=np.float32),
        robot_pos=np.asarray([GOALIE_LINE_X_MM, float(rng.normal(0.0, 50.0))], dtype=np.float32),
        robot_theta=0.0,
        teammates=(),
        opponents=(),
    )
    _set_ball_state(backend, ball_x, ball_y, vx, vy)
    set_action = pybh.rl_set_visible_action if backend.mode == "simrobot_visible" else pybh.rl_set_action
    set_action(1, "stand", 0.0, 0.0, 0.0, -1)

    skills: list[str] = []
    telemetry_counts: Counter[str] = Counter()
    goal = False
    out = False
    min_ball_x = ball_x
    last_ball = (ball_x, ball_y)

    for step in range(MAX_STEPS):
        if policy == "rl":
            assert actor is not None
            skills.append(_send_rl_action(backend, actor, device))
        else:
            skills.append("bhuman")

        if backend.mode == "simrobot":
            for _ in range(DT_FRAMES):
                pybh.simrobot_step(1)
        else:
            time.sleep(0.10)

        bx, by, bvx, bvy = _ball_field_state(backend)
        last_ball = (bx, by)
        min_ball_x = min(min_ball_x, bx)
        t = _telemetry()
        telemetry_counts[f"{t['skill_behavior_skill_request']}:{t['motion_engine_input_request']}:{t['motion_engine_dive_request']}"] += 1

        if bx <= OWN_GOAL_X_MM:
            goal = abs(by) <= OWN_GOAL_HALF_Y_MM
            out = not goal
            break
        if bx > GOALIE_LINE_X_MM + 900.0 and step > 8:
            break
        if abs(by) > 1800.0:
            out = True
            break
        if math.hypot(bvx, bvy) < 45.0 and step > 25:
            break

    return {
        "scenario": spec.name,
        "goal": bool(goal),
        "saved": not goal,
        "out": bool(out),
        "steps": int(step + 1),
        "final_ball": [float(last_ball[0]), float(last_ball[1])],
        "min_ball_x": float(min_ball_x),
        "skill_counts": dict(Counter(skills)),
        "telemetry_counts": dict(telemetry_counts.most_common(8)),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario"])].append(row)
    scenarios = {}
    for name, items in sorted(by_scenario.items()):
        n = len(items)
        goals = sum(1 for item in items if item["goal"])
        scenarios[name] = {
            "episodes": n,
            "goals": goals,
            "saves": n - goals,
            "save_rate": (n - goals) / max(1, n),
            "goal_rate": goals / max(1, n),
        }
    goals = sum(1 for row in rows if row["goal"])
    return {
        "episodes": len(rows),
        "goals": goals,
        "saves": len(rows) - goals,
        "save_rate": (len(rows) - goals) / max(1, len(rows)),
        "goal_rate": goals / max(1, len(rows)),
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Real SimRobot A/B: original B-Human GK vs RL GK on identical moving-ball threats.")
    parser.add_argument("--policy", choices=("bhuman", "rl"), required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--episodes-per-scenario", type=int, default=6)
    parser.add_argument("--seed", type=int, default=742619)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--backend", choices=("simrobot", "simrobot_visible"), default="simrobot_visible")
    parser.add_argument("--output", default="")
    parser.add_argument("--dump-episodes", action="store_true")
    args = parser.parse_args()

    os.environ["PYBH_RL_OVERRIDE_TEAM"] = "24"
    if args.policy == "rl":
        os.environ["PYBH_RL_ACTIVE_PLAYERS"] = "1"
    else:
        os.environ["PYBH_RL_ACTIVE_PLAYERS"] = "2"

    device = torch.device(args.device)
    actor = _load_actor(_resolve(args.checkpoint), device) if args.policy == "rl" else None
    rng = np.random.default_rng(args.seed)
    backend = PyBHBackend(
        mode=args.backend,
        simrobot_profile="3d",
        simrobot_scene_path=args.scene,
        simrobot_robot_name="robot1",
        simrobot_launch=args.backend == "simrobot_visible",
        simrobot_bin=SIMROBOT_BIN,
        frames_per_step=DT_FRAMES,
        obs_timeout_ms=2500,
        player_number=1,
    )
    rows: list[dict[str, object]] = []
    try:
        for spec in THREATS:
            for _ in range(int(args.episodes_per_scenario)):
                rows.append(run_episode(backend, args.policy, actor, device, spec, rng))
    finally:
        backend.close()

    result = {
        "policy": args.policy,
        "checkpoint": "" if not args.checkpoint else str(_resolve(args.checkpoint)),
        "scene": str(args.scene),
        "benchmark": "real_simrobot_strategy_bhuman_ab_moving_ball_v1",
        "episodes_per_scenario": int(args.episodes_per_scenario),
        "seed": int(args.seed),
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
