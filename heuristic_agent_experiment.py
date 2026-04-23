"""
heuristic_agent_experiment.py

Experimento standalone: agente heurístico que posiciona el robot detrás del
balón y deja que BHuman ejecute el kick via pybh.controller (SimRobot path).

Arquitectura:
  Python mantiene física del balón y posición del robot (2D simplificada).
  pybh.controller corre los módulos de BHuman (Cognition) en el mismo proceso.
  RLSkillProvider en C++ lee la acción RL y escribe la observación de vuelta.
  Cuando BHuman ejecuta "shoot", Python aplica el impulso al balón.

Política heurística de dos fases:
  Fase 1 (dist > APPROACH_THRESHOLD_MM): paso grande hacia offset de handoff.
  Fase 2 (dist ≤ APPROACH_THRESHOLD_MM): corrección fina + emitir shoot.

Uso:
  python heuristic_agent_experiment.py --episodes 100
  python heuristic_agent_experiment.py --episodes 3 --no-plot   # smoke rápido
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# pybh requires CWD = SabanaHerons2026 so getBHDir() finds Config correctly
_BHUMAN_DIR = str(Path(__file__).parent.parent / "SabanaHerons2026")
os.chdir(_BHUMAN_DIR)
import pybh.controller as pybh
os.chdir(Path(__file__).parent)  # restore RL dir for CSV/plot output

# ── Configuración campo (mm) ──────────────────────────────────────────────────
FIELD_X_HALF        = 4500.0   # mitad del largo
FIELD_Y_HALF        = 3000.0   # mitad del ancho
GOAL_X              = 4500.0   # línea de fondo rival
GOAL_Y_HALF         = 750.0    # mitad de la portería

# ── BHuman pybh ──────────────────────────────────────────────────────────────
TEAM_NUMBER         = 24
PLAYER_NUMBER       = 1
LOCATION            = "field_e"
SCENARIO            = "4v4_Scaled"
FIELD_PLAYER_COLOR  = pybh.TeamColor.black
GOALKEEPER_COLOR    = pybh.TeamColor.black

# ── Heurística ────────────────────────────────────────────────────────────────
HANDOFF_OFFSET_MM   = np.array([-210.0, 0.0], dtype=np.float32)
APPROACH_THRESHOLD_MM = 350.0
FINE_DIVISOR_MM     = 400.0
SHOOT_THRESHOLD_MM  = 260.0    # dist al offset → emitir shoot

# ── Física simplificada del balón ─────────────────────────────────────────────
KICK_SPEED_MM_S     = 3500.0   # velocidad inicial del kick (mm/s)
BALL_FRICTION       = 0.92     # factor de atenuación por frame
DT                  = 0.033    # s por frame (~30 Hz)
MAX_STEPS           = 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def _face_ball(robot_pos: np.ndarray, ball_pos: np.ndarray) -> float:
    d = ball_pos - robot_pos
    return math.atan2(float(d[1]), float(d[0]))


def _heuristic_action(robot_pos: np.ndarray, ball_pos: np.ndarray,
                      robot_theta: float) -> Dict:
    target = ball_pos + HANDOFF_OFFSET_MM
    delta  = target - robot_pos
    dist   = float(np.linalg.norm(delta))

    if dist <= SHOOT_THRESHOLD_MM:
        # En posición de handoff → shoot
        return {"skill": "shoot", "x": 0.0, "y": 0.0, "theta": 0.0}

    divisor = 550.0 if dist > APPROACH_THRESHOLD_MM else FINE_DIVISOR_MM
    tx = float(np.clip(delta[0] / divisor, -1.0, 1.0)) * divisor + robot_pos[0]
    ty = float(np.clip(delta[1] / divisor, -1.0, 1.0)) * divisor + robot_pos[1]
    tt = _face_ball(robot_pos, ball_pos)
    return {"skill": "walkTo", "x": tx, "y": ty, "theta": tt}


def _apply_kick(ball_pos: np.ndarray, robot_theta: float):
    """Devuelve velocidad inicial del balón tras el kick."""
    return np.array([
        math.cos(robot_theta) * KICK_SPEED_MM_S,
        math.sin(robot_theta) * KICK_SPEED_MM_S,
    ], dtype=np.float32)


def _step_ball(ball_pos: np.ndarray, ball_vel: np.ndarray):
    ball_pos = ball_pos + ball_vel * DT
    ball_vel = ball_vel * BALL_FRICTION
    return ball_pos, ball_vel


def _goal_scored(ball_pos: np.ndarray) -> bool:
    return ball_pos[0] >= GOAL_X and abs(ball_pos[1]) <= GOAL_Y_HALF


# ── Episodio ──────────────────────────────────────────────────────────────────

def _run_episode(ctrl: pybh.Controller, seed: int) -> Dict:
    rng = np.random.default_rng(seed)

    # Posición inicial aleatoria del balón y del robot
    ball_pos   = np.array([
        rng.uniform(0.0, FIELD_X_HALF * 0.5),
        rng.uniform(-FIELD_Y_HALF * 0.3, FIELD_Y_HALF * 0.3),
    ], dtype=np.float32)
    robot_pos  = ball_pos + np.array([-800.0, rng.uniform(-200.0, 200.0)], dtype=np.float32)
    robot_pos  = np.clip(robot_pos, [-FIELD_X_HALF, -FIELD_Y_HALF],
                                    [FIELD_X_HALF,  FIELD_Y_HALF]).astype(np.float32)
    robot_theta = _face_ball(robot_pos, ball_pos)
    ball_vel   = np.zeros(2, dtype=np.float32)

    shoot_issued  = False
    goal_scored   = False
    handoff_ready = False
    first_shoot_step = None
    step = 0

    for step in range(1, MAX_STEPS + 1):
        # 1. Inyectar estado en BHuman
        pybh.set_world_state(ctrl, 0,
                             float(ball_pos[0]), float(ball_pos[1]),
                             float(robot_pos[0]), float(robot_pos[1]),
                             float(robot_theta))

        # 2. Decidir acción heurística
        action = _heuristic_action(robot_pos, ball_pos, robot_theta)

        # 3. Enviar acción a BHuman
        pybh.rl_set_action(PLAYER_NUMBER, action["skill"],
                           action.get("x", 0.0),
                           action.get("y", 0.0),
                           action.get("theta", 0.0))

        # 4. Avanzar un frame de Cognition
        ctrl.update()

        # 5. Leer observación de vuelta
        obs = pybh.rl_get_obs(PLAYER_NUMBER)
        if obs["obs_ready"]:
            robot_pos[0]  = obs["robot_x"]
            robot_pos[1]  = obs["robot_y"]
            robot_theta   = obs["robot_theta"]

        # 6. Física del balón
        if shoot_issued:
            ball_pos, ball_vel = _step_ball(ball_pos, ball_vel)
            if _goal_scored(ball_pos):
                goal_scored = True
                break
            # fuera del campo → termina
            if abs(ball_pos[0]) > FIELD_X_HALF or abs(ball_pos[1]) > FIELD_Y_HALF:
                break
        elif action["skill"] == "shoot":
            if not shoot_issued:
                shoot_issued = True
                first_shoot_step = step
                handoff_ready = True
                ball_vel = _apply_kick(ball_pos, robot_theta)

        # 7. Mover robot (física simplificada) si walkTo
        if action["skill"] == "walkTo" and not obs["obs_ready"]:
            target = np.array([action["x"], action["y"]], dtype=np.float32)
            delta  = target - robot_pos
            dist   = float(np.linalg.norm(delta))
            step_mm = min(dist, 80.0)  # ~80 mm/frame máximo
            if dist > 1.0:
                robot_pos = robot_pos + (delta / dist) * step_mm
            robot_theta = _face_ball(robot_pos, ball_pos)

    return {
        "seed":            seed,
        "steps":           step,
        "handoff_ready":   handoff_ready,
        "shoot_issued":    shoot_issued,
        "goal_scored":     goal_scored,
        "first_shoot_step": first_shoot_step,
    }


# ── Experimento ───────────────────────────────────────────────────────────────

def run_experiment(n_episodes: int, seed_base: int) -> List[Dict]:
    ctrl = pybh.Controller()
    ctrl.add_player(
        name=f"Heron{PLAYER_NUMBER}",
        team_number=TEAM_NUMBER,
        field_player_color=FIELD_PLAYER_COLOR,
        goalkeeper_color=GOALKEEPER_COLOR,
        player_number=PLAYER_NUMBER,
        location=LOCATION,
        scenario=SCENARIO,
    )
    ctrl.start()

    results = []
    try:
        for i in range(n_episodes):
            r = _run_episode(ctrl, seed_base + i)
            results.append(r)
            if (i + 1) % 10 == 0:
                g = sum(x["goal_scored"] for x in results)
                h = sum(x["handoff_ready"] for x in results)
                print(f"  [{i+1}/{n_episodes}] goals={g}  handoffs={h}")
    finally:
        ctrl.stop()

    return results


def print_summary(results: List[Dict]) -> None:
    n = len(results)
    goals    = sum(r["goal_scored"]   for r in results)
    handoffs = sum(r["handoff_ready"] for r in results)
    shoots   = sum(r["shoot_issued"]  for r in results)
    fss      = [r["first_shoot_step"] for r in results if r["first_shoot_step"] is not None]

    print()
    print("=" * 55)
    print(f"  HEURISTIC AGENT — SimRobot path ({n} episodios)")
    print("=" * 55)
    print(f"  goal_rate      : {goals/n:.3f}  ({goals}/{n})")
    print(f"  handoff_rate   : {handoffs/n:.3f}  ({handoffs}/{n})")
    print(f"  shoot_rate     : {shoots/n:.3f}  ({shoots}/{n})")
    if fss:
        print(f"  median_shoot_step: {statistics.median(fss):.0f}  (~{statistics.median(fss)*DT:.2f}s)")
    print(f"  mean_steps     : {statistics.mean(r['steps'] for r in results):.1f}")
    print("=" * 55)


def save_csv(results: List[Dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "episode", "seed", "steps", "handoff_ready",
            "shoot_issued", "goal_scored", "first_shoot_step",
        ])
        w.writeheader()
        for i, r in enumerate(results):
            w.writerow({"episode": i + 1, **r})
    print(f"  CSV guardado en {path}")


def plot_results(results: List[Dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  (matplotlib no disponible — plot omitido)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ep = list(range(1, len(results) + 1))
    window = 20

    # Rolling goal rate
    goals_roll = [
        sum(r["goal_scored"] for r in results[max(0, i-window):i+1]) /
        min(i + 1, window)
        for i in range(len(results))
    ]
    axes[0].plot(ep, goals_roll)
    axes[0].set_title("Rolling Goal Rate")
    axes[0].set_ylabel("Rate")
    axes[0].set_xlabel("Episode")

    # Steps per episode
    axes[1].scatter(ep, [r["steps"] for r in results], s=4, alpha=0.6)
    axes[1].set_title("Steps per Episode")
    axes[1].set_xlabel("Episode")

    # First shoot step
    fss = [(i+1, r["first_shoot_step"]) for i, r in enumerate(results)
           if r["first_shoot_step"] is not None]
    if fss:
        xs, ys = zip(*fss)
        axes[2].scatter(xs, ys, s=4, alpha=0.6, color="orange")
    axes[2].set_title("First Shoot Step")
    axes[2].set_xlabel("Episode")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"  Plot guardado en {path}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic agent experiment (SimRobot path)")
    parser.add_argument("--episodes",  type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=200)
    parser.add_argument("--no-plot",   action="store_true")
    args = parser.parse_args()

    print(f"Corriendo {args.episodes} episodios con seed_base={args.seed_base} ...")
    t0 = time.perf_counter()
    results = run_experiment(args.episodes, args.seed_base)
    elapsed = time.perf_counter() - t0

    print_summary(results)
    print(f"  Tiempo total: {elapsed:.1f}s  ({elapsed/args.episodes:.2f}s/episodio)")

    out = Path(__file__).parent
    save_csv(results, out / "heuristic_experiment_results.csv")
    if not args.no_plot:
        plot_results(results, out / "heuristic_experiment_plot.png")


if __name__ == "__main__":
    main()
