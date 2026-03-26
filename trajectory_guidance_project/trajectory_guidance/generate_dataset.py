"""
generate_dataset.py — Genera datasets de demostraciones BC-ready por stage.

Cambios respecto a la versión anterior:
  FIX-A: Usa replay_closed_loop en lugar de replay_plan.
  FIX-B: Reestructurado con `ProcessPoolExecutor` para procesar N episodios en paralelo
         y reducir drásticamente el tiempo de recolección frente al alto coste
         computacional del planificador A* per-step.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import concurrent.futures
import math

if __package__ in (None, ""):
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    workspace_root = project_dir.parent
    sys.path = [p for p in sys.path if Path(p).resolve() != script_dir]
    sys.path.insert(0, str(workspace_root))
    sys.path.insert(1, str(project_dir))

from trajectory_guidance.settings import (
    ASTAR_CFG, GUMBEL_CFG, DATASET_CFG,
)
from trajectory_guidance.dataset import export_snapshots_pkl

# ── WORKER GLOBALS (Must be picklable for ProcessPool) ──────────────────
def generate_single_trajectory(args):
    """Generatriz aislada instanciando entorno por Worker para ThreadSafety."""
    ep, ep_type, stage, lookahead, current_seed, target, kick_angle, kick_dist, dribble_bias = args
    
    from trajectory_guidance.bridge import EnvBridge, GumbelPlanSampler
    from trajectory_guidance.evaluate_solver import SolverEvaluator
    from trajectory_guidance.solver import AStarSolver
    from trajectory_guidance.settings import ASTAR_CFG, GUMBEL_CFG, DATASET_CFG
    
    # Cada proceso tiene su propio entorno (Pygame oculto) y su Solver A*
    bridge = EnvBridge(seed=current_seed)
    solver = AStarSolver(ASTAR_CFG)
    evaluator = SolverEvaluator(bridge=bridge, solver=solver)
    sampler = GumbelPlanSampler(GUMBEL_CFG)

    world_snapshot, plans = evaluator.plan_from_current_reset(stage_id=stage, seed=current_seed)
    if not plans:
        return {"status": "no_plans", "ep": ep, "type": ep_type}
    
    sampled = sampler.sample(plans)
    ref_plan = sampled.selected

    metrics, snapshots = evaluator.replay_closed_loop(
        stage_id=stage,
        initial_snapshot=world_snapshot,
        lookahead=lookahead,
        collect_snapshots=True,
        reference_plan=ref_plan,
        dribble_bias_angle=dribble_bias,
        kick_target=target,
        kick_angle_deg=kick_angle if kick_angle is not None else 20.0,
        kick_phase_dist=kick_dist if kick_dist is not None else 1800.0,
    )

    if not metrics.valid_plan:
        return {
            "status": "invalid", "ep": ep, "type": ep_type, "metrics": metrics, 
            "target": target, "angle": kick_angle, "dist": kick_dist, "bias": dribble_bias
        }
    if metrics.quality_score < DATASET_CFG.min_quality_score:
        return {
            "status": "low_quality", "ep": ep, "type": ep_type, 
            "quality": metrics.quality_score, "bias": dribble_bias
        }

    return {
        "status": "ok", "ep": ep, "type": ep_type, "metrics": metrics, 
        "snapshots": snapshots, "target": target, "angle": kick_angle, 
        "dist": kick_dist, "bias": dribble_bias
    }


def main() -> None:
    # ── Configuración ────────────────────────────────────────────────────
    stage = 3
    episodes = 5        # episodios de dribble
    shoot_episodes = 5 # episodios de disparo al arco (kick_target = goal)
    pass_episodes = 5 # episodios de pase a zona de campo (kick_target = zona)
    out = "trajectory_demos/demos_fixed0.pkl"
    base_seed = 42
    lookahead = ASTAR_CFG.greedy_lookahead
    
    # PARAMETRIZACION DE PARALELISMO
    n_jobs = 8  # Subimos a 8 cores concurrentes de un solo envión!

    from config import FIELD as _FIELD
    GOAL_TARGETS = [
        np.array([_FIELD.x_opponent_ground, 0.0],                      dtype=np.float32),
        np.array([_FIELD.x_opponent_ground, _FIELD.y_left_goal * 0.7], dtype=np.float32),
        np.array([_FIELD.x_opponent_ground, _FIELD.y_right_goal * 0.7],dtype=np.float32),
        np.array([_FIELD.x_opponent_ground, _FIELD.y_left_goal * 0.4], dtype=np.float32),
        np.array([_FIELD.x_opponent_ground, _FIELD.y_right_goal * 0.4],dtype=np.float32),
    ]
    PASS_TARGETS = [
        np.array([2000.0,  1500.0], dtype=np.float32),
        np.array([2000.0, -1500.0], dtype=np.float32),
        np.array([3000.0,   600.0], dtype=np.float32),
        np.array([3000.0,  -600.0], dtype=np.float32),
        np.array([1000.0,  2000.0], dtype=np.float32),
        np.array([1000.0, -2000.0], dtype=np.float32),
    ]

    shoot_kick_angles  = [25.0, 30.0, 35.0, 40.0, 45.0]
    shoot_kick_dists   = [800.0, 1000.0, 1200.0, 1500.0, 2000.0]
    shoot_dribble_bias = [0.0, math.pi/6, -math.pi/6,
                          math.pi/4, -math.pi/4,
                          math.pi/8, -math.pi/8]

    pass_kick_angles   = [25.0, 30.0, 35.0, 40.0]
    pass_kick_dists    = [800.0, 1200.0, 1500.0, 2000.0, 2500.0]
    pass_dribble_bias  = [0.0, math.pi/6, -math.pi/6,
                          math.pi/4, -math.pi/4,
                          math.pi/8, -math.pi/8]

    if stage >= 3:
        dribble_angles = [
            0.0, math.pi / 6, -math.pi / 6, math.pi / 4, -math.pi / 4, 
            math.pi / 3, -math.pi / 3,
        ]
    else:
        dribble_angles = [
            0.0, math.pi / 6, -math.pi / 6, math.pi / 4, -math.pi / 4, 
            math.pi / 3, -math.pi / 3, math.pi / 2, -math.pi / 2, 
            2 * math.pi / 3, -2 * math.pi / 3,
        ]

    tasks = []
    
    # Generar args para DRIBBLE
    for ep in range(episodes):
        current_seed = base_seed + ep
        dribble_angle = dribble_angles[ep % len(dribble_angles)]
        tasks.append((ep, "dribble", stage, lookahead, current_seed, None, None, None, dribble_angle))

    # Generar args para SHOOT
    for ep in range(shoot_episodes):
        current_seed = base_seed + episodes + ep
        goal_target      = GOAL_TARGETS[ep % len(GOAL_TARGETS)]
        s_kick_angle     = shoot_kick_angles[ep % len(shoot_kick_angles)]
        s_kick_dist      = shoot_kick_dists[ep % len(shoot_kick_dists)]
        s_dribble_bias   = shoot_dribble_bias[ep % len(shoot_dribble_bias)]
        tasks.append((ep, "shoot", stage, lookahead, current_seed, goal_target, s_kick_angle, s_kick_dist, s_dribble_bias))

    # Generar args para PASS
    for ep in range(pass_episodes):
        current_seed = base_seed + episodes + shoot_episodes + ep
        pass_target    = PASS_TARGETS[ep % len(PASS_TARGETS)]
        p_kick_angle   = pass_kick_angles[ep % len(pass_kick_angles)]
        p_kick_dist    = pass_kick_dists[ep % len(pass_kick_dists)]
        p_dribble_bias = pass_dribble_bias[ep % len(pass_dribble_bias)]
        tasks.append((ep, "pass", stage, lookahead, current_seed, pass_target, p_kick_angle, p_kick_dist, p_dribble_bias))

    # Counters
    stats = {
        "dribble": {"total": episodes, "acc": 0, "no_plans": 0, "inv": 0, "low_q": 0},
        "shoot": {"total": shoot_episodes, "acc": 0, "no_plans": 0, "inv": 0, "low_q": 0},
        "pass": {"total": pass_episodes, "acc": 0, "no_plans": 0, "inv": 0, "low_q": 0},
    }
    all_snapshots = []

    print(f"\n[PARALELISMO] Iniciando pool con n_jobs={n_jobs}. Procesando {len(tasks)} trayectorias...\n")
    
    completed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(generate_single_trajectory, t) for t in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            ep_type = res["type"]
            ep = res["ep"]
            completed_count += 1
            st_dict = stats[ep_type]
            
            # Format prints
            if res["status"] == "no_plans":
                st_dict["no_plans"] += 1
                if ep_type == "dribble":
                    print(f"[SKIP] ep={ep} no plans")
                
            elif res["status"] == "invalid":
                st_dict["inv"] += 1
                metrics = res["metrics"]
                if ep_type == "dribble":
                    angle_deg = math.degrees(res["bias"])
                    print(f"[SKIP] ep={ep} ({angle_deg:.0f}°) invalid (t={metrics.touched_ball} d={metrics.dribble_steps} g={metrics.goal_scored})")
                else:
                    gx, gy = float(res["target"][0]), float(res["target"][1])
                    print(f"[SKIP-{ep_type[0].upper()}] ep={ep} invalid target=({gx:.0f},{gy:.0f}) angle={res['angle']}° dist={res['dist']:.0f} (d={metrics.dribble_steps} g={metrics.goal_scored})")
            
            elif res["status"] == "low_quality":
                st_dict["low_q"] += 1
                
            elif res["status"] == "ok":
                st_dict["acc"] += 1
                all_snapshots.extend(res["snapshots"])
                metrics = res["metrics"]
                
                if ep_type == "dribble":
                    angle_deg = math.degrees(res["bias"])
                    print(f"[OK] ep={ep} ({angle_deg:.0f}°) steps={metrics.steps} dribble={metrics.dribble_steps} goal={metrics.goal_scored}")
                else:
                    gx, gy = float(res["target"][0]), float(res["target"][1])
                    print(f"[OK-{ep_type[0].upper()}] ep={ep} target=({gx:.0f},{gy:.0f}) angle={res['angle']}° dist={res['dist']:.0f} steps={metrics.steps} dribble={metrics.dribble_steps} goal={metrics.goal_scored}")

    # ── Exportar ─────────────────────────────────────────────────────────
    out_path = Path(out)
    export_snapshots_pkl(all_snapshots, str(out_path))
    
    total_ep = sum(s["total"] for s in stats.values())
    total_acc = sum(s["acc"] for s in stats.values())
    
    print(f"\n{'='*60}")
    print(f"DATASET GENERATION COMPLETE")
    print(f"  Stage: {stage}")
    print(f"  Dribble:  {stats['dribble']['acc']}/{stats['dribble']['total']} aceptados")
    print(f"  Shoot:    {stats['shoot']['acc']}/{stats['shoot']['total']} aceptados")
    print(f"  Pass:     {stats['pass']['acc']}/{stats['pass']['total']} aceptados")
    print(f"  Total:    {total_acc}/{total_ep} episodios, {len(all_snapshots)} snapshots")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Required for spawn multiprocess logic safety in Windows
    multiprocessing.freeze_support() if hasattr(sys, 'frozen') else None
    main()