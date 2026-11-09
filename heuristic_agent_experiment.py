from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from action import SkillType, StructuredSkillAction
from config import DT, FIELD_X_HALF, FIELD_Y_HALF, GOAL_X, GOAL_Y_HALF, SIMROBOT_BIN
from curriculum import get_dynamic_world_update, get_stage_spec, sample_stage_world_spec
from pybh_backend import PyBHBackend
from teacher_policy import BALL_SEARCH_MAX_AGE_MS, stage_aware_teacher_action

HANDOFF_OFFSET_MM = np.array([-210.0, 0.0], dtype=np.float32)
APPROACH_THRESHOLD_MM = 350.0
FINE_DIVISOR_MM = 400.0
SHOOT_THRESHOLD_MM = 260.0
MAX_STEPS = 500


def _face_ball(robot_pos: np.ndarray, ball_pos: np.ndarray) -> float:
    delta = ball_pos - robot_pos
    return math.atan2(float(delta[1]), float(delta[0]))


def _heuristic_action(robot_pos: np.ndarray, ball_pos: np.ndarray) -> StructuredSkillAction:
    target = ball_pos + HANDOFF_OFFSET_MM
    delta = target - robot_pos
    distance = float(np.linalg.norm(delta))

    if distance <= SHOOT_THRESHOLD_MM:
        return StructuredSkillAction(skill_type=SkillType.shoot)

    divisor = 550.0 if distance > APPROACH_THRESHOLD_MM else FINE_DIVISOR_MM
    target_x = float(np.clip(delta[0] / divisor, -1.0, 1.0)) * divisor + float(robot_pos[0])
    target_y = float(np.clip(delta[1] / divisor, -1.0, 1.0)) * divisor + float(robot_pos[1])
    return StructuredSkillAction(
        skill_type=SkillType.walk,
        target_x=target_x,
        target_y=target_y,
        target_theta=_face_ball(robot_pos, ball_pos),
    )


def _stage_aware_teacher_action(obs, stage_id: int, prev_skill_name: str | None = None) -> StructuredSkillAction:
    return stage_aware_teacher_action(obs, stage_id, prev_skill_name)


def _goal_scored(ball_pos: np.ndarray) -> bool:
    return bool(ball_pos[0] >= GOAL_X and abs(ball_pos[1]) <= GOAL_Y_HALF)


def _run_episode(backend: PyBHBackend, seed: int, stage_id: int) -> dict[str, int | bool | None | float]:
    rng = np.random.default_rng(seed)
    world_spec = sample_stage_world_spec(stage_id, rng)
    obs = backend.reset(world_spec=world_spec)
    ball_pos = np.array([float(obs.ball_x), float(obs.ball_y)], dtype=np.float32)
    robot_pos = np.array([float(obs.robot_x), float(obs.robot_y)], dtype=np.float32)
    ball_pos[:] = (obs.ball_x, obs.ball_y)
    robot_pos[:] = (obs.robot_x, obs.robot_y)

    shoot_issued = False
    goal_scored = False
    handoff_ready = False
    first_shoot_step: int | None = None
    first_dribble_step: int | None = None
    walk_steps = 0
    dribble_steps = 0
    shoot_steps = 0
    stand_steps = 0
    stale_steps = 0
    prev_skill_name: str | None = None
    step = 0

    for step in range(1, MAX_STEPS + 1):
        action = _stage_aware_teacher_action(obs, stage_id, prev_skill_name)
        obs = backend.step(action)
        world_update = get_dynamic_world_update(
            stage_id,
            step,
            robot_x=obs.robot_x,
            robot_y=obs.robot_y,
            ball_x=obs.ball_x,
            ball_y=obs.ball_y,
        )
        if world_update:
            obs = backend.set_dynamic_world(**world_update)
        ball_pos[:] = (obs.ball_x, obs.ball_y)
        robot_pos[:] = (obs.robot_x, obs.robot_y)
        prev_skill_name = action.skill_name

        walk_steps += int(action.skill_name == "walk")
        dribble_steps += int(action.skill_name == "dribble")
        shoot_steps += int(action.skill_name == "shoot")
        stand_steps += int(action.skill_name == "stand")
        stale_steps += int(float(obs.time_since_ball_seen) > BALL_SEARCH_MAX_AGE_MS)

        if action.skill_name == "shoot" and not shoot_issued:
            shoot_issued = True
            first_shoot_step = step
        if action.skill_name == "dribble" and first_dribble_step is None:
            first_dribble_step = step
            handoff_ready = True

        if _goal_scored(ball_pos):
            goal_scored = True
            break
        if abs(ball_pos[0]) > FIELD_X_HALF or abs(ball_pos[1]) > FIELD_Y_HALF:
            break

    return {
        "seed": seed,
        "steps": step,
        "handoff_ready": handoff_ready,
        "shoot_issued": shoot_issued,
        "goal_scored": goal_scored,
        "first_shoot_step": first_shoot_step,
        "first_dribble_step": first_dribble_step,
        "walk_steps": walk_steps,
        "dribble_steps": dribble_steps,
        "shoot_steps": shoot_steps,
        "stand_steps": stand_steps,
        "stale_steps": stale_steps,
    }


def run_experiment(
    n_episodes: int,
    seed_base: int,
    *,
    stage_id: int,
    backend_mode: str,
    simrobot_profile: str,
    launch_simrobot: bool,
) -> list[dict[str, int | bool | None | float]]:
    backend = PyBHBackend(
        mode=backend_mode,
        frames_per_step=3 if backend_mode == "simrobot" else 1,
        obs_timeout_ms=1000,
        simrobot_profile=simrobot_profile,
        simrobot_launch=launch_simrobot,
        simrobot_bin=SIMROBOT_BIN,
    )
    backend.start()

    results: list[dict[str, int | bool | None | float]] = []
    try:
        for episode in range(n_episodes):
            results.append(_run_episode(backend, seed_base + episode, stage_id))
            if (episode + 1) % 10 == 0:
                goals = sum(int(result["goal_scored"]) for result in results)
                handoffs = sum(int(result["handoff_ready"]) for result in results)
                print(f"  [{episode + 1}/{n_episodes}] goals={goals} handoffs={handoffs}")
    finally:
        backend.close()

    return results


def print_summary(results: list[dict[str, int | bool | None | float]], label: str) -> None:
    total = len(results)
    goals = sum(int(result["goal_scored"]) for result in results)
    handoffs = sum(int(result["handoff_ready"]) for result in results)
    shoots = sum(int(result["shoot_issued"]) for result in results)
    first_shoot_steps = [int(result["first_shoot_step"]) for result in results if result["first_shoot_step"] is not None]
    first_dribble_steps = [
        int(result["first_dribble_step"]) for result in results if result["first_dribble_step"] is not None
    ]
    total_steps = max(1, sum(int(result["steps"]) for result in results))

    print()
    print("=" * 55)
    print(f"  HEURISTIC AGENT — {label} ({total} episodios)")
    print("=" * 55)
    print(f"  goal_rate      : {goals / total:.3f}  ({goals}/{total})")
    print(f"  handoff_rate   : {handoffs / total:.3f}  ({handoffs}/{total})")
    print(f"  shoot_rate     : {shoots / total:.3f}  ({shoots}/{total})")
    if first_dribble_steps:
        median_step = statistics.median(first_dribble_steps)
        print(f"  median_dribble_step: {median_step:.0f}  (~{median_step * DT:.2f}s)")
    if first_shoot_steps:
        median_step = statistics.median(first_shoot_steps)
        print(f"  median_shoot_step: {median_step:.0f}  (~{median_step * DT:.2f}s)")
    print(f"  mean_steps     : {statistics.mean(int(result['steps']) for result in results):.1f}")
    print(f"  skill_mix      : stand={sum(int(r['stand_steps']) for r in results) / total_steps:.3f} "
          f"walk={sum(int(r['walk_steps']) for r in results) / total_steps:.3f} "
          f"dribble={sum(int(r['dribble_steps']) for r in results) / total_steps:.3f} "
          f"shoot={sum(int(r['shoot_steps']) for r in results) / total_steps:.3f}")
    print(f"  stale_step_rate: {sum(int(r['stale_steps']) for r in results) / total_steps:.3f}")
    print("=" * 55)


def save_csv(results: list[dict[str, int | bool | None | float]], path: Path) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode",
                "seed",
                "steps",
                "handoff_ready",
                "shoot_issued",
                "goal_scored",
                "first_shoot_step",
                "first_dribble_step",
                "walk_steps",
                "dribble_steps",
                "shoot_steps",
                "stand_steps",
                "stale_steps",
            ],
        )
        writer.writeheader()
        for index, result in enumerate(results, start=1):
            writer.writerow({"episode": index, **result})
    print(f"  CSV guardado en {path}")


def plot_results(results: list[dict[str, int | bool | None | float]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no disponible — plot omitido)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    episodes = list(range(1, len(results) + 1))
    window = 20

    goals_roll = [
        sum(int(result["goal_scored"]) for result in results[max(0, index - window) : index + 1]) / min(index + 1, window)
        for index in range(len(results))
    ]
    axes[0].plot(episodes, goals_roll)
    axes[0].set_title("Rolling Goal Rate")
    axes[0].set_ylabel("Rate")
    axes[0].set_xlabel("Episode")

    axes[1].scatter(episodes, [int(result["steps"]) for result in results], s=4, alpha=0.6)
    axes[1].set_title("Steps per Episode")
    axes[1].set_xlabel("Episode")

    first_shoot_steps = [
        (index + 1, int(result["first_shoot_step"]))
        for index, result in enumerate(results)
        if result["first_shoot_step"] is not None
    ]
    if first_shoot_steps:
        xs, ys = zip(*first_shoot_steps)
        axes[2].scatter(xs, ys, s=4, alpha=0.6, color="orange")
    axes[2].set_title("First Shoot Step")
    axes[2].set_xlabel("Episode")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"  Plot guardado en {path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic agent experiment on real SimRobot backends")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=200)
    parser.add_argument("--stage-id", type=int, default=0, choices=(-1, 0, 1, 2, 3, 4))
    parser.add_argument("--backend", choices=("simrobot", "simrobot_visible"), default="simrobot")
    parser.add_argument("--simrobot-profile", choices=("2d", "3d"), default="2d")
    parser.add_argument("--launch-simrobot", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    stage = get_stage_spec(args.stage_id)
    label = f"{args.backend}:{args.simrobot_profile}:stage{args.stage_id}"
    print(
        f"Corriendo {args.episodes} episodios con seed_base={args.seed_base} "
        f"({label}, active_skills={stage.active_skill_names}) ..."
    )
    started_at = time.perf_counter()
    results = run_experiment(
        args.episodes,
        args.seed_base,
        stage_id=args.stage_id,
        backend_mode=args.backend,
        simrobot_profile=args.simrobot_profile,
        launch_simrobot=args.launch_simrobot,
    )
    elapsed = time.perf_counter() - started_at

    print_summary(results, label)
    print(f"  Tiempo total: {elapsed:.1f}s  ({elapsed / args.episodes:.2f}s/episodio)")

    save_csv(results, _SCRIPT_DIR / "heuristic_experiment_results.csv")
    if not args.no_plot:
        plot_results(results, _SCRIPT_DIR / "heuristic_experiment_plot.png")


if __name__ == "__main__":
    main()
