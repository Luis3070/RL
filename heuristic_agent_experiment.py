"""
heuristic_agent_experiment.py

Experimento standalone: agente heurístico que posiciona el robot detrás del
balón y deja que BHuman ejecute el kick via pybh.controller (SimRobot path).

Arquitectura:
  pybh.controller corre los módulos de BHuman (Cognition) en el mismo proceso.
  RLSkillProvider en C++ lee la acción RL y escribe la observación de vuelta.
  Por defecto, la física 2D también corre en C++ dentro de SabanaHerons
  mediante un backend headless (`RLSim2D`) inspirado en SimulatedRobot2D.
  Con `--python-sim`, el script usa el fallback viejo con física local Python.
  Con `--simrobot-real`, el loop completo corre sobre SimRobot 2D real:
  acción RL → Cognition real → robot real del scene graph → observación real.

Política heurística de dos fases:
  Fase 1 (dist > APPROACH_THRESHOLD_MM): paso grande hacia offset de handoff.
  Fase 2 (dist ≤ APPROACH_THRESHOLD_MM): corrección fina + emitir shoot.

Uso:
  python heuristic_agent_experiment.py --episodes 100
  python heuristic_agent_experiment.py --episodes 3 --no-plot   # smoke rápido
  python heuristic_agent_experiment.py --episodes 3 --no-plot --simrobot-real
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from action import SkillType, StructuredSkillAction

# pybh resolves Config and related assets from the process CWD.
# Keep the process rooted in SabanaHerons2026 for the full controller lifetime.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BHUMAN_DIR = _SCRIPT_DIR.parent / "SabanaHerons2026"
os.chdir(_BHUMAN_DIR)
import pybh.controller as pybh

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

# ── Física SimRobot 2D portada a Python ──────────────────────────────────────
DT                  = 0.033    # s por frame (~30 Hz)
MAX_STEPS           = 500
BASE_WALK_PERIOD_S  = 0.25
MAX_ROT_SPEED       = math.radians(70.0)
MAX_FWD_SPEED_MM_S  = 250.0
MAX_SIDE_SPEED_MM_S = 180.0
MAX_BACK_SPEED_MM_S = 200.0
BALL_FRICTION       = -0.30    # m/(s*s), igual a ballSpecification.cfg
SIMROBOT_SHOOT_DISTANCE_MM = 6000.0
MM_PER_M            = 1000.0
SIMROBOT_SCENE_PATH = _BHUMAN_DIR / "Config/Scenes/RL1v0.ros2d"
SIMROBOT_ROBOT_NAME = "robot1"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _face_ball(robot_pos: np.ndarray, ball_pos: np.ndarray) -> float:
    d = ball_pos - robot_pos
    return math.atan2(float(d[1]), float(d[0]))


def _normalize_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def _rotate(v: np.ndarray, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)


def _world_to_robot(robot_pos: np.ndarray, robot_theta: float,
                    target_pos: np.ndarray) -> np.ndarray:
    return _rotate(target_pos - robot_pos, -robot_theta)


def _heuristic_action(robot_pos: np.ndarray, ball_pos: np.ndarray,
                      robot_theta: float) -> StructuredSkillAction:
    target = ball_pos + HANDOFF_OFFSET_MM
    delta  = target - robot_pos
    dist   = float(np.linalg.norm(delta))

    if dist <= SHOOT_THRESHOLD_MM:
        # En posición de handoff → shoot
        return StructuredSkillAction(skill_type=SkillType.shoot)

    divisor = 550.0 if dist > APPROACH_THRESHOLD_MM else FINE_DIVISOR_MM
    tx = float(np.clip(delta[0] / divisor, -1.0, 1.0)) * divisor + robot_pos[0]
    ty = float(np.clip(delta[1] / divisor, -1.0, 1.0)) * divisor + robot_pos[1]
    tt = _face_ball(robot_pos, ball_pos)
    return StructuredSkillAction(
        skill_type=SkillType.walk,
        target_x=tx,
        target_y=ty,
        target_theta=tt,
    )


def _apply_kick(ball_pos: np.ndarray, robot_theta: float):
    """Aproxima el kick 2D usando BallPhysics::velocityForDistance."""
    kick_speed = math.sqrt(2.0) * math.sqrt(-BALL_FRICTION * 1000.0 * SIMROBOT_SHOOT_DISTANCE_MM)
    return np.array([
        math.cos(robot_theta) * kick_speed,
        math.sin(robot_theta) * kick_speed,
    ], dtype=np.float32)


def _step_ball(ball_pos: np.ndarray, ball_vel: np.ndarray):
    speed = float(np.linalg.norm(ball_vel))
    if speed <= 1e-6:
        return ball_pos, np.zeros(2, dtype=np.float32)

    stop_time = (speed / 1000.0) / -BALL_FRICTION
    t = min(DT, stop_time)
    acc = -(ball_vel / speed) * (abs(BALL_FRICTION) * 1000.0)
    ball_pos = ball_pos + ball_vel * t + acc * (0.5 * t * t)
    if t >= stop_time:
        ball_vel = np.zeros(2, dtype=np.float32)
    else:
        ball_vel = ball_vel + acc * t
    return ball_pos, ball_vel


def _goal_scored(ball_pos: np.ndarray) -> bool:
    return ball_pos[0] >= GOAL_X and abs(ball_pos[1]) <= GOAL_Y_HALF


def _simrobot_reset(ball_pos: np.ndarray, robot_pos: np.ndarray, robot_theta: float) -> None:
    scene_ball_x = float(-ball_pos[0] / MM_PER_M)
    scene_ball_y = float(-ball_pos[1] / MM_PER_M)
    scene_robot_x = float(-robot_pos[0] / MM_PER_M)
    scene_robot_y = float(-robot_pos[1] / MM_PER_M)
    scene_robot_theta = float(_normalize_angle(robot_theta - math.pi))

    if not pybh.simrobot_set_ball_state(scene_ball_x,
                                        scene_ball_y,
                                        0.0, 0.0):
        raise RuntimeError("No se pudo resetear la pelota en SimRobot.")
    if not pybh.simrobot_set_robot_pose(SIMROBOT_ROBOT_NAME,
                                        scene_robot_x,
                                        scene_robot_y,
                                        scene_robot_theta):
        raise RuntimeError(f"No se pudo resetear {SIMROBOT_ROBOT_NAME} en SimRobot.")


def _rotation_range(is_left_phase: bool) -> tuple[float, float]:
    inside_turn_ratio = 0.33
    inner = 2.0 * inside_turn_ratio * MAX_ROT_SPEED * BASE_WALK_PERIOD_S
    outer = 2.0 * (1.0 - inside_turn_ratio) * MAX_ROT_SPEED * BASE_WALK_PERIOD_S
    if is_left_phase:
        return -inner, outer
    return -outer, inner


def _translation_rectangle(rotation: float) -> tuple[float, float, float, float]:
    no_translation_from_rotation = math.radians(24.0)
    t_factor = max(0.0, 1.0 - max(0.0, abs(rotation) / no_translation_from_rotation))
    back_x = max(-1000.0, t_factor * -MAX_BACK_SPEED_MM_S * BASE_WALK_PERIOD_S)
    back_y = max(-1000.0, t_factor * -2.0 * MAX_SIDE_SPEED_MM_S * BASE_WALK_PERIOD_S)
    front_x = min(1000.0, t_factor * MAX_FWD_SPEED_MM_S * BASE_WALK_PERIOD_S)
    front_y = min(1000.0, t_factor * 2.0 * MAX_SIDE_SPEED_MM_S * BASE_WALK_PERIOD_S)
    return min(back_x, -0.01), max(front_x, 0.01), min(back_y, -0.01), max(front_y, 0.01)


def _clip_to_rectangle(target: np.ndarray, back_x: float, front_x: float,
                       back_y: float, front_y: float) -> np.ndarray:
    if back_x <= target[0] <= front_x and back_y <= target[1] <= front_y:
        return target.astype(np.float32)

    norm = float(np.linalg.norm(target))
    if norm <= 1e-6:
        return np.zeros(2, dtype=np.float32)

    direction = target / norm
    limits = []
    if abs(direction[0]) > 1e-6:
        limits.append((front_x if direction[0] > 0.0 else back_x) / direction[0])
    if abs(direction[1]) > 1e-6:
        limits.append((front_y if direction[1] > 0.0 else back_y) / direction[1])
    scale = min(x for x in limits if x > 0.0)
    return (direction * scale).astype(np.float32)


def _simrobot_walk_step(robot_pos: np.ndarray, robot_theta: float,
                        is_left_phase: Optional[bool], target_pos: np.ndarray,
                        target_theta: float) -> tuple[np.ndarray, float, bool]:
    target_rel = _world_to_robot(robot_pos, robot_theta, target_pos)
    target_rot = _normalize_angle(target_theta - robot_theta)
    target_norm = float(np.linalg.norm(target_rel))

    mod_rot = target_rot
    if target_norm > 600.0 - abs(target_rot) / math.pi * 400.0:
        mod_rot = math.atan2(float(target_rel[1]), float(target_rel[0]))
    elif target_norm > 100.0:
        factor = (target_norm - 100.0) / 500.0
        target_dir = target_rel / target_norm
        facing = np.array([math.cos(target_rot), math.sin(target_rot)], dtype=np.float32)
        mixed = factor * target_dir + (1.0 - factor) * facing
        mod_rot = math.atan2(float(mixed[1]), float(mixed[0]))

    should_be_left = bool(target_rel[1] > 0.0) if abs(target_rel[1]) > 1e-6 else bool(mod_rot > 0.0)
    phase_left = should_be_left if is_left_phase is None else not is_left_phase
    rot_min, rot_max = _rotation_range(phase_left)
    step_rot = float(np.clip(mod_rot, rot_min, rot_max))
    back_x, front_x, back_y, front_y = _translation_rectangle(step_rot)
    step_trans = _clip_to_rectangle(target_rel, back_x, front_x, back_y, front_y)
    if phase_left == bool(step_trans[1] < 0.0):
        step_trans[1] = 0.0

    frac = DT / BASE_WALK_PERIOD_S
    delta_local = step_trans * frac
    robot_pos = robot_pos + _rotate(delta_local, robot_theta)
    robot_theta = _normalize_angle(robot_theta + step_rot * frac)
    return robot_pos.astype(np.float32), robot_theta, phase_left


# ── Episodio ──────────────────────────────────────────────────────────────────

def _run_episode(ctrl: Optional[pybh.Controller], seed: int, backend: str) -> Dict:
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
    walk_left_phase: Optional[bool] = None

    if backend == "cpp":
        pybh.rl_sim_reset(
            PLAYER_NUMBER,
            float(ball_pos[0]), float(ball_pos[1]),
            float(robot_pos[0]), float(robot_pos[1]),
            float(robot_theta),
        )
    elif backend == "python":
        pybh.rl_sim_set_enabled(PLAYER_NUMBER, False)
    else:
        _simrobot_reset(ball_pos, robot_pos, robot_theta)
        pybh.rl_set_action(PLAYER_NUMBER, "stand", 0.0, 0.0, 0.0)
        pybh.simrobot_step(1)
        obs = pybh.rl_get_obs(PLAYER_NUMBER, 1000)
        if not obs["obs_ready"]:
            raise RuntimeError("SimRobot no produjo observación inicial para RL.")
        ball_pos[0] = obs["ball_x"]
        ball_pos[1] = obs["ball_y"]
        robot_pos[0] = obs["robot_x"]
        robot_pos[1] = obs["robot_y"]
        robot_theta = obs["robot_theta"]

    for step in range(1, MAX_STEPS + 1):
        # 1. Sincronizar el estado visible por BHuman.
        if backend == "python":
            pybh.set_world_state(ctrl, 0,
                                 float(ball_pos[0]), float(ball_pos[1]),
                                 float(robot_pos[0]), float(robot_pos[1]),
                                 float(robot_theta))

        # 2. Decidir acción heurística
        action = _heuristic_action(robot_pos, ball_pos, robot_theta)

        # 3. Enviar acción a BHuman
        pybh.rl_set_action(
            PLAYER_NUMBER,
            "walkTo" if action.skill_name == "walk" else action.skill_name,
            float(action.target_x),
            float(action.target_y),
            float(action.target_theta),
        )

        # 4. Avanzar un frame
        if backend == "simrobot":
            pybh.simrobot_step(1)
        else:
            assert ctrl is not None
            ctrl.update()

        # 5. Leer observación de vuelta
        obs = pybh.rl_get_obs(PLAYER_NUMBER, 1000)
        if not obs["obs_ready"]:
            raise RuntimeError(f"No llegó observación RL en backend={backend} paso={step}.")
        ball_pos[0] = obs["ball_x"]
        ball_pos[1] = obs["ball_y"]
        robot_pos[0] = obs["robot_x"]
        robot_pos[1] = obs["robot_y"]
        robot_theta = obs["robot_theta"]

        # 6. Física local fallback cuando no se usa el backend C++.
        if backend == "python" and shoot_issued:
            ball_pos, ball_vel = _step_ball(ball_pos, ball_vel)
        elif action.skill_name == "shoot":
            if not shoot_issued:
                shoot_issued = True
                first_shoot_step = step
                handoff_ready = True
                if backend == "python":
                    ball_vel = _apply_kick(ball_pos, robot_theta)

        # 7. Física local fallback del cuerpo cuando no se usa el backend C++.
        if backend == "python" and action.skill_name == "walk":
            target = np.array([action.target_x, action.target_y], dtype=np.float32)
            robot_pos, robot_theta, walk_left_phase = _simrobot_walk_step(
                robot_pos, robot_theta, walk_left_phase, target, float(action.target_theta)
            )

        if _goal_scored(ball_pos):
            goal_scored = True
            break
        if abs(ball_pos[0]) > FIELD_X_HALF or abs(ball_pos[1]) > FIELD_Y_HALF:
            break

    return {
        "seed":            seed,
        "steps":           step,
        "handoff_ready":   handoff_ready,
        "shoot_issued":    shoot_issued,
        "goal_scored":     goal_scored,
        "first_shoot_step": first_shoot_step,
    }


# ── Experimento ───────────────────────────────────────────────────────────────

def run_experiment(n_episodes: int, seed_base: int, backend: str) -> List[Dict]:
    if backend == "simrobot":
        if not hasattr(pybh, "simrobot_load_scene"):
            raise RuntimeError("La versión instalada de pybh no expone simrobot_load_scene().")
        if not pybh.simrobot_load_scene(str(SIMROBOT_SCENE_PATH)):
            raise RuntimeError(f"No se pudo cargar la escena SimRobot: {pybh.simrobot_last_error()}")
        ctrl = None
    else:
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
            r = _run_episode(ctrl, seed_base + i, backend=backend)
            results.append(r)
            if (i + 1) % 10 == 0:
                g = sum(x["goal_scored"] for x in results)
                h = sum(x["handoff_ready"] for x in results)
                print(f"  [{i+1}/{n_episodes}] goals={g}  handoffs={h}")
    finally:
        if backend == "simrobot":
            pybh.simrobot_unload_scene()
        else:
            assert ctrl is not None
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
    parser.add_argument("--python-sim", action="store_true",
                        help="Usa la física portada a Python en vez del backend 2D en C++.")
    parser.add_argument("--simrobot-real", action="store_true",
                        help="Usa la escena 2D real de SimRobot para balón/scene graph.")
    args = parser.parse_args()

    backend = "simrobot" if args.simrobot_real else ("python" if args.python_sim else "cpp")
    print(f"Corriendo {args.episodes} episodios con seed_base={args.seed_base} (sim={backend}) ...")
    t0 = time.perf_counter()
    results = run_experiment(args.episodes, args.seed_base, backend=backend)
    elapsed = time.perf_counter() - t0

    print_summary(results)
    print(f"  Tiempo total: {elapsed:.1f}s  ({elapsed/args.episodes:.2f}s/episodio)")

    save_csv(results, _SCRIPT_DIR / "heuristic_experiment_results.csv")
    if not args.no_plot:
        plot_results(results, _SCRIPT_DIR / "heuristic_experiment_plot.png")


if __name__ == "__main__":
    main()
