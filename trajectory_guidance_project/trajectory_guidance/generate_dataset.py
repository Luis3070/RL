"""
generate_dataset.py - Stage-aware BC dataset generation for trajectory_guidance.

This generator keeps the trajectory dataset aligned with the project-wide
BHuman-in-loop Option 1 baseline:
  - real backend
  - persistent_2d_cognition bridge mode
  - 2D / 4v4_Scaled environment
  - honest labels for stages 0-3
  - hybrid starts: natural reset seeds + deterministic launch templates

Stage intent policy:
  - Stage 0: control only
  - Stage 1: dribble only
  - Stage 2-3: dribble + pass_setup + shoot_setup
"""
from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    workspace_root = project_dir.parent
    sys.path = [p for p in sys.path if Path(p).resolve() != script_dir]
    sys.path.insert(0, str(workspace_root))
    sys.path.insert(1, str(project_dir))

from config import FIELD as _FIELD
from trajectory_guidance.dataset import export_snapshots_pkl
from trajectory_guidance.launch_templates import (
    get_stage_template_names,
    resolve_launch_template,
)
from trajectory_guidance.settings import ASTAR_CFG, DATASET_CFG, GUMBEL_CFG


Task = Tuple[
    int,
    str,
    int,
    int,
    int,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[float],
    Optional[float],
    float,
    str,
    Optional[str],
]


HYBRID_INCLUDE_RESET_SEEDS = True
HYBRID_INCLUDE_LAUNCH_TEMPLATES = True
HYBRID_MAX_ATTEMPT_MULTIPLIER = 12
FAIL_CLOSED_ON_INSUFFICIENT_ACCEPTED = True


def _field_bounds() -> Tuple[float, float, float, float]:
    return (
        float(_FIELD.x_own_ground),
        float(_FIELD.x_opponent_ground),
        float(_FIELD.y_right_sideline),
        float(_FIELD.y_left_sideline),
    )


def _clamp_point(x: float, y: float,
                 margin_x: float = 180.0,
                 margin_y: float = 160.0) -> np.ndarray:
    x_min, x_max, y_min, y_max = _field_bounds()
    return np.array([
        float(np.clip(x, x_min + margin_x, x_max - margin_x)),
        float(np.clip(y, y_min + margin_y, y_max - margin_y)),
    ], dtype=np.float32)


PASS_TARGETS = [
    np.array([1500.0,  1800.0], dtype=np.float32),
    np.array([1500.0, -1800.0], dtype=np.float32),
    np.array([2500.0,  1200.0], dtype=np.float32),
    np.array([2500.0, -1200.0], dtype=np.float32),
    np.array([3000.0,  1800.0], dtype=np.float32),
    np.array([3000.0, -1800.0], dtype=np.float32),
    np.array([1000.0,  2000.0], dtype=np.float32),
    np.array([1000.0, -2000.0], dtype=np.float32),
]


def _goal_targets() -> List[np.ndarray]:
    return [
        _clamp_point(_FIELD.x_opponent_ground, 0.0, margin_x=0.0, margin_y=0.0),
        _clamp_point(_FIELD.x_opponent_ground, _FIELD.y_left_goal * 0.70,
                     margin_x=0.0, margin_y=0.0),
        _clamp_point(_FIELD.x_opponent_ground, _FIELD.y_right_goal * 0.70,
                     margin_x=0.0, margin_y=0.0),
        _clamp_point(_FIELD.x_opponent_ground, _FIELD.y_left_goal * 0.40,
                     margin_x=0.0, margin_y=0.0),
        _clamp_point(_FIELD.x_opponent_ground, _FIELD.y_right_goal * 0.40,
                     margin_x=0.0, margin_y=0.0),
    ]


def _pass_targets() -> List[np.ndarray]:
    return PASS_TARGETS


def _approach_targets(stage: int) -> List[np.ndarray]:
    x_span = float(_FIELD.x_opponent_ground - _FIELD.x_own_ground)
    if stage == 0:
        return [
            _clamp_point(_FIELD.x_own_ground + 0.40 * x_span, 0.0),
            _clamp_point(_FIELD.x_own_ground + 0.42 * x_span, 0.16 * _FIELD.y_left_sideline),
            _clamp_point(_FIELD.x_own_ground + 0.42 * x_span, 0.16 * _FIELD.y_right_sideline),
            _clamp_point(_FIELD.x_own_ground + 0.48 * x_span, 0.28 * _FIELD.y_left_sideline),
            _clamp_point(_FIELD.x_own_ground + 0.48 * x_span, 0.28 * _FIELD.y_right_sideline),
        ]
    if stage == 1:
        return [
            _clamp_point(_FIELD.x_own_ground + 0.52 * x_span, 0.0),
            _clamp_point(_FIELD.x_own_ground + 0.55 * x_span, 0.22 * _FIELD.y_left_sideline),
            _clamp_point(_FIELD.x_own_ground + 0.55 * x_span, 0.22 * _FIELD.y_right_sideline),
            _clamp_point(_FIELD.x_own_ground + 0.60 * x_span, 0.36 * _FIELD.y_left_sideline),
            _clamp_point(_FIELD.x_own_ground + 0.60 * x_span, 0.36 * _FIELD.y_right_sideline),
        ]
    return [
        _clamp_point(_FIELD.x_own_ground + 0.68 * x_span, 0.0),
        _clamp_point(_FIELD.x_own_ground + 0.70 * x_span, 0.20 * _FIELD.y_left_sideline),
        _clamp_point(_FIELD.x_own_ground + 0.70 * x_span, 0.20 * _FIELD.y_right_sideline),
        _clamp_point(_FIELD.x_own_ground + 0.74 * x_span, 0.34 * _FIELD.y_left_sideline),
        _clamp_point(_FIELD.x_own_ground + 0.74 * x_span, 0.34 * _FIELD.y_right_sideline),
    ]


def _biases_for_stage(stage: int) -> List[float]:
    if stage >= 2:
        return [
            0.0,
            math.pi / 8,
            -math.pi / 8,
            math.pi / 6,
            -math.pi / 6,
            math.pi / 4,
            -math.pi / 4,
        ]
    return [
        0.0,
        math.pi / 10,
        -math.pi / 10,
        math.pi / 6,
        -math.pi / 6,
        math.pi / 5,
        -math.pi / 5,
    ]


def _resolve_family_counts(stage: int,
                           primary_episodes: int,
                           shoot_episodes: int,
                           pass_episodes: int) -> Dict[str, int]:
    if stage == 0:
        return {"control": max(primary_episodes, 0)}
    if stage == 1:
        return {"dribble": max(primary_episodes, 0)}
    return {
        "dribble": max(primary_episodes, 0),
        "shoot_setup": max(shoot_episodes, 0),
        "pass_setup": max(pass_episodes, 0),
    }


def _build_tasks(stage: int,
                 primary_episodes: int,
                 shoot_episodes: int,
                 pass_episodes: int,
                 base_seed: int,
                  lookahead: int) -> List[Task]:
    family_counts = _resolve_family_counts(
        stage=stage,
        primary_episodes=primary_episodes,
        shoot_episodes=shoot_episodes,
        pass_episodes=pass_episodes,
    )
    approach_targets = _approach_targets(stage)
    goal_targets = _goal_targets()
    pass_targets = _pass_targets()
    biases = _biases_for_stage(stage)
    shoot_kick_angles  = [20.0, 25.0, 30.0, 35.0, 40.0]
    shoot_kick_dists   = [800.0, 1000.0, 1200.0, 1500.0, 2000.0]
    shoot_dribble_bias = [0.0, math.pi/6, -math.pi/6,
                          math.pi/8, -math.pi/8]

    pass_kick_angles   = [20.0, 25.0, 30.0, 35.0]
    pass_kick_dists    = [800.0, 1200.0, 1500.0, 2000.0]
    pass_dribble_bias  = [0.0, math.pi/6, -math.pi/6,
                          math.pi/8, -math.pi/8]
    stage_templates = get_stage_template_names(stage)

    tasks: List[Task] = []
    global_ep = 0
    for intent_label, total in family_counts.items():
        max_attempts = max(total * HYBRID_MAX_ATTEMPT_MULTIPLIER, total)
        for family_ep in range(max_attempts):
            current_seed = base_seed + global_ep
            approach_target = approach_targets[family_ep % len(approach_targets)]
            dribble_bias = biases[family_ep % len(biases)]
            kick_target: Optional[np.ndarray] = None
            kick_angle: Optional[float] = None
            kick_dist: Optional[float] = None

            if intent_label == "shoot_setup":
                kick_target = goal_targets[family_ep % len(goal_targets)]
                kick_angle = shoot_kick_angles[family_ep % len(shoot_kick_angles)]
                kick_dist = shoot_kick_dists[family_ep % len(shoot_kick_dists)]
                dribble_bias = shoot_dribble_bias[family_ep % len(shoot_dribble_bias)]
            elif intent_label == "pass_setup":
                kick_target = pass_targets[family_ep % len(pass_targets)]
                kick_angle = pass_kick_angles[family_ep % len(pass_kick_angles)]
                kick_dist = pass_kick_dists[family_ep % len(pass_kick_dists)]
                dribble_bias = pass_dribble_bias[family_ep % len(pass_dribble_bias)]

            if HYBRID_INCLUDE_RESET_SEEDS:
                tasks.append((
                    family_ep,
                    intent_label,
                    stage,
                    lookahead,
                    current_seed,
                    approach_target,
                    kick_target,
                    kick_angle,
                    kick_dist,
                    dribble_bias,
                    "reset_seed",
                    None,
                ))

            if HYBRID_INCLUDE_LAUNCH_TEMPLATES and stage_templates:
                template_name = stage_templates[family_ep % len(stage_templates)]
                tasks.append((
                    family_ep,
                    intent_label,
                    stage,
                    lookahead,
                    current_seed,
                    approach_target,
                    kick_target,
                    kick_angle,
                    kick_dist,
                    dribble_bias,
                    "launch_template",
                    template_name,
                ))
            global_ep += 1
    return tasks


def generate_single_trajectory(args: Task) -> Dict[str, object]:
    """Isolated worker entrypoint to keep env state process-local."""
    (ep, intent_label, stage, lookahead, current_seed, approach_target,
     kick_target, kick_angle, kick_dist, dribble_bias,
     start_mode, template_name) = args

    from trajectory_guidance.bridge import EnvBridge, GumbelPlanSampler
    from trajectory_guidance.evaluate_solver import SolverEvaluator
    from trajectory_guidance.solver import AStarSolver

    bridge = EnvBridge(seed=current_seed)
    solver = AStarSolver(ASTAR_CFG)
    evaluator = SolverEvaluator(bridge=bridge, solver=solver)
    sampler = GumbelPlanSampler(GUMBEL_CFG)
    launch_template = None
    if start_mode == "launch_template" and template_name is not None:
        launch_template = resolve_launch_template(
            stage_id=stage,
            template_name=template_name,
            approach_target=approach_target,
            kick_target=kick_target,
        )

    world_snapshot, plans = evaluator.plan_from_current_reset(
        stage_id=stage,
        seed=current_seed,
        launch_template=launch_template,
    )
    if not plans:
        return {
            "status": "no_plans",
            "ep": ep,
            "intent": intent_label,
            "start_mode": start_mode,
            "template_name": template_name,
        }

    sampled = sampler.sample(plans)
    ref_plan = sampled.selected

    metrics, snapshots = evaluator.replay_closed_loop(
        stage_id=stage,
        initial_snapshot=world_snapshot,
        lookahead=lookahead,
        collect_snapshots=True,
        reference_plan=ref_plan,
        dribble_bias_angle=dribble_bias,
        kick_target=kick_target,
        kick_angle_deg=kick_angle if kick_angle is not None else 20.0,
        kick_phase_dist=kick_dist if kick_dist is not None else 1800.0,
        intent_label=intent_label,
        approach_target=approach_target,
    )

    if not metrics.valid_plan:
        return {
            "status": "invalid",
            "ep": ep,
            "intent": intent_label,
            "metrics": metrics,
            "approach_target": approach_target,
            "kick_target": kick_target,
            "angle": kick_angle,
            "dist": kick_dist,
            "bias": dribble_bias,
            "start_mode": start_mode,
            "template_name": template_name,
        }
    if metrics.quality_score < DATASET_CFG.min_quality_score:
        return {
            "status": "low_quality",
            "ep": ep,
            "intent": intent_label,
            "quality": metrics.quality_score,
            "bias": dribble_bias,
            "start_mode": start_mode,
            "template_name": template_name,
        }

    return {
        "status": "ok",
        "ep": ep,
        "intent": intent_label,
        "metrics": metrics,
        "snapshots": snapshots,
        "approach_target": approach_target,
        "kick_target": kick_target,
        "angle": kick_angle,
        "dist": kick_dist,
        "bias": dribble_bias,
        "start_mode": start_mode,
        "template_name": template_name,
    }


def _run_tasks(tasks: Iterable[Task], n_jobs: int) -> Iterable[Dict[str, object]]:
    task_list = list(tasks)
    if n_jobs <= 1:
        for task in task_list:
            yield generate_single_trajectory(task)
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(generate_single_trajectory, task)
                   for task in task_list]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()


def _format_target(target: Optional[np.ndarray]) -> str:
    if target is None:
        return "-"
    return f"({float(target[0]):.0f},{float(target[1]):.0f})"


def _print_result(result: Dict[str, object]) -> None:
    intent_label = str(result["intent"])
    ep = int(result["ep"])
    bias_rad = float(result.get("bias", 0.0))
    bias_deg = math.degrees(bias_rad)
    start_mode = str(result.get("start_mode", "reset_seed"))
    template_name = result.get("template_name")
    start_label = start_mode if not template_name else f"{start_mode}:{template_name}"

    if result["status"] == "no_plans":
        print(f"[SKIP] start={start_label} intent={intent_label} ep={ep} no plans")
        return

    if result["status"] == "low_quality":
        quality = float(result.get("quality", 0.0))
        print(
            f"[SKIP] start={start_label} intent={intent_label} ep={ep} "
            f"low quality={quality:.2f}"
        )
        return

    metrics = result["metrics"]
    kick_target = _format_target(result.get("kick_target"))
    approach_target = _format_target(result.get("approach_target"))
    tk_ready = int(getattr(metrics, "episode_handoff_achieved", 0) > 0)
    tk_succ = int(getattr(metrics, "episode_takeover_success", 0) > 0)
    control_steps = int(getattr(metrics, "control_steps", 0))
    semantic_dribble_steps = int(getattr(metrics, "dribble_steps", 0))

    if result["status"] == "invalid":
        print(
            f"[SKIP] start={start_label} intent={intent_label} ep={ep} invalid "
            f"approach={approach_target} kick={kick_target} bias={bias_deg:.0f}deg "
            f"touch={int(metrics.touched_ball)} control={control_steps} "
            f"dribble_sem={semantic_dribble_steps} "
            f"TkReady={tk_ready} TkSucc={tk_succ}"
        )
        return

    print(
        f"[OK] start={start_label} intent={intent_label} ep={ep} "
        f"approach={approach_target} kick={kick_target} bias={bias_deg:.0f}deg "
        f"steps={metrics.steps} control={control_steps} "
        f"dribble_sem={semantic_dribble_steps} "
        f"TkReady={tk_ready} TkSucc={tk_succ} goal={int(metrics.goal_scored)}"
    )


def main() -> None:
    # Configuracion editable localmente en vez de CLI.
    stage = 0
    episodes = 5
    shoot_episodes = 5
    pass_episodes = 5
    out = "trajectory_demos/demos_fixed0.pkl"
    seed = 42
    lookahead = ASTAR_CFG.greedy_lookahead
    n_jobs = DATASET_CFG.default_n_jobs

    tasks = _build_tasks(
        stage=stage,
        primary_episodes=episodes,
        shoot_episodes=shoot_episodes,
        pass_episodes=pass_episodes,
        base_seed=seed,
        lookahead=lookahead,
    )

    family_counts = _resolve_family_counts(
        stage=stage,
        primary_episodes=episodes,
        shoot_episodes=shoot_episodes,
        pass_episodes=pass_episodes,
    )
    stats = {
        intent_label: {
            "target": total,
            "attempted": 0,
            "acc": 0,
            "no_plans": 0,
            "inv": 0,
            "low_q": 0,
        }
        for intent_label, total in family_counts.items()
    }
    start_stats: Dict[str, Dict[str, int]] = {}
    all_snapshots = []

    print(
        f"\n[DATASET] stage={stage} tasks={len(tasks)} n_jobs={n_jobs} "
        f"families={family_counts} hybrid=(seed={HYBRID_INCLUDE_RESET_SEEDS} "
        f"template={HYBRID_INCLUDE_LAUNCH_TEMPLATES}) "
        f"attempt_multiplier={HYBRID_MAX_ATTEMPT_MULTIPLIER}\n"
    )

    for result in _run_tasks(tasks, n_jobs):
        intent_label = str(result["intent"])
        stats_row = stats[intent_label]
        if stats_row["acc"] >= stats_row["target"]:
            continue
        status = str(result["status"])
        stats_row["attempted"] += 1
        start_mode = str(result.get("start_mode", "reset_seed"))
        template_name = result.get("template_name")
        start_label = start_mode if not template_name else f"{start_mode}:{template_name}"
        start_stats.setdefault(
            start_label,
            {"total": 0, "acc": 0, "no_plans": 0, "inv": 0, "low_q": 0},
        )
        start_stats[start_label]["total"] += 1
        if status == "ok":
            stats_row["acc"] += 1
            start_stats[start_label]["acc"] += 1
            all_snapshots.extend(result["snapshots"])
        elif status == "no_plans":
            stats_row["no_plans"] += 1
            start_stats[start_label]["no_plans"] += 1
        elif status == "invalid":
            stats_row["inv"] += 1
            start_stats[start_label]["inv"] += 1
        elif status == "low_quality":
            stats_row["low_q"] += 1
            start_stats[start_label]["low_q"] += 1
        _print_result(result)

        if all(row["acc"] >= row["target"] for row in stats.values()):
            break

    out_path = Path(out)
    targets_met = all(row["acc"] >= row["target"] for row in stats.values())
    if targets_met:
        export_snapshots_pkl(all_snapshots, str(out_path))

    total_target = sum(row["target"] for row in stats.values())
    total_attempted = sum(row["attempted"] for row in stats.values())
    total_acc = sum(row["acc"] for row in stats.values())

    print(f"\n{'=' * 72}")
    print("DATASET GENERATION COMPLETE")
    print(f"  Stage:   {stage}")
    for intent_label, row in stats.items():
        print(
            f"  {intent_label:11s} "
            f"{row['acc']}/{row['target']} accepted "
            f"(attempted={row['attempted']} "
            f"no_plans={row['no_plans']} invalid={row['inv']} low_q={row['low_q']})"
        )
    print("  By start:")
    for start_label, row in start_stats.items():
        print(
            f"    {start_label:28s} "
            f"{row['acc']}/{row['total']} accepted "
            f"(no_plans={row['no_plans']} invalid={row['inv']} low_q={row['low_q']})"
        )
    print(
        f"  Total:   {total_acc}/{total_target} accepted "
        f"(attempted={total_attempted}), {len(all_snapshots)} snapshots"
    )
    if targets_met:
        print(f"  Output:  {out_path}")
    else:
        print("  Output:  NOT WRITTEN (insufficient accepted trajectories)")
    print(f"{'=' * 72}")

    if FAIL_CLOSED_ON_INSUFFICIENT_ACCEPTED and not targets_met:
        raise RuntimeError(
            "Dataset generation failed closed: not enough trajectories with real takeover success "
            f"to meet targets {family_counts}."
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
