from __future__ import annotations

import argparse
import sys
import time

from rlfootball.core.config import SIMROBOT_BIN
from rlfootball.core.pybh_backend import PyBHBackend, WorldPlayerSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual validation suite for visible SimRobot world reset and dynamic updates.")
    parser.add_argument("--simrobot-profile", choices=("2d", "3d"), default="2d")
    parser.add_argument("--launch-simrobot", action="store_true")
    parser.add_argument("--simrobot-bin", default=SIMROBOT_BIN)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--request-timeout-ms", type=int, default=None)
    parser.add_argument("--reset-obs-timeout-ms", type=int, default=None)
    parser.add_argument("--status-log-interval-s", type=float, default=1.0)
    parser.add_argument("--pause-seconds", type=float, default=3.0)
    parser.add_argument("--invalid-opponent-number", type=int, default=20)
    return parser.parse_args()


def _build_backend(args: argparse.Namespace) -> PyBHBackend:
    return PyBHBackend(
        mode="simrobot_visible",
        simrobot_profile=args.simrobot_profile,
        obs_timeout_ms=args.timeout_ms,
        visible_request_timeout_ms=args.request_timeout_ms,
        visible_reset_obs_timeout_ms=args.reset_obs_timeout_ms,
        visible_status_log_interval_s=args.status_log_interval_s,
        simrobot_launch=args.launch_simrobot,
        simrobot_bin=args.simrobot_bin,
    )


def _print_obs(prefix: str, obs) -> None:
    print(
        f"{prefix} "
        f"obs_ready={int(obs.obs_ready)} "
        f"robot=({obs.robot_x:.1f},{obs.robot_y:.1f},{obs.robot_theta:.2f}) "
        f"ball=({obs.ball_x:.1f},{obs.ball_y:.1f})",
        flush=True,
    )


def _run_reset_extended(backend: PyBHBackend) -> bool:
    print("[1/3] Reset extendido", flush=True)
    input("Presiona Enter para enviar el reset extendido...")
    try:
        obs = backend.reset(
            ball_pos=[0.0, 0.0],
            robot_pos=[-1200.0, 0.0],
            robot_theta=0.0,
            teammates=(
                WorldPlayerSpec(number=2, x=-400.0, y=700.0, theta=0.0),
                WorldPlayerSpec(number=3, x=-400.0, y=-700.0, theta=0.0),
            ),
            opponents=(
                WorldPlayerSpec(number=1, x=1200.0, y=500.0, theta=3.14),
                WorldPlayerSpec(number=2, x=1200.0, y=-500.0, theta=3.14),
            ),
        )
        _print_obs("PASS reset ok", obs)
        input("Verifica posiciones en la GUI y presiona Enter para continuar...")
        return True
    except Exception as exc:
        print(f"FAIL reset extendido: {exc}", flush=True)
        return False


def _run_invalid_slot(backend: PyBHBackend, invalid_opponent_number: int) -> bool:
    print("[2/3] Invalid slot", flush=True)
    input("Presiona Enter para enviar el reset invalido...")
    try:
        backend.reset(
            ball_pos=[0.0, 0.0],
            robot_pos=[-1000.0, 0.0],
            robot_theta=0.0,
            opponents=(
                WorldPlayerSpec(
                    number=invalid_opponent_number,
                    x=1000.0,
                    y=0.0,
                    theta=0.0,
                ),
            ),
        )
    except Exception as exc:
        print(f"PASS rechazo explicito: {exc}", flush=True)
        input("Confirma en la GUI que no hubo reset parcial y presiona Enter para continuar...")
        return True
    print("FAIL invalid slot: el reset debía fallar y no falló.", flush=True)
    return False


def _run_dynamic_world(backend: PyBHBackend, pause_seconds: float) -> bool:
    print("[3/3] Dynamic world", flush=True)
    input("Presiona Enter para enviar el reset inicial del dynamic world...")
    try:
        obs = backend.reset(
            ball_pos=[0.0, 0.0],
            robot_pos=[-1400.0, 0.0],
            robot_theta=0.0,
            opponents=(WorldPlayerSpec(number=1, x=900.0, y=0.0, theta=3.14),),
        )
        _print_obs("reset ok", obs)

        time.sleep(pause_seconds)
        obs = backend.set_dynamic_world(ball_pos=[300.0, 600.0])
        _print_obs("pelota movida", obs)

        time.sleep(pause_seconds)
        obs = backend.set_dynamic_world(opponents=(WorldPlayerSpec(number=1, x=300.0, y=-700.0, theta=3.14),))
        _print_obs("rival movido", obs)

        time.sleep(pause_seconds)
        obs = backend.set_dynamic_world(robot_pos=[-800.0, 400.0], robot_theta=0.5)
        _print_obs("robot RL movido", obs)

        input("Verifica en la GUI que no hubo reset completo y presiona Enter para cerrar...")
        print("PASS dynamic world", flush=True)
        return True
    except Exception as exc:
        print(f"FAIL dynamic world: {exc}", flush=True)
        return False


def main() -> None:
    args = parse_args()
    backend = _build_backend(args)
    backend.start()
    try:
        print("Abre SimRobot y vuelve aca cuando la escena visible ya este cargada.", flush=True)
        input("Cuando la simulacion ya este corriendo, presiona Enter para arrancar la suite...")

        results = {
            "reset_extended": _run_reset_extended(backend),
            "invalid_slot": _run_invalid_slot(backend, args.invalid_opponent_number),
            "dynamic_world": _run_dynamic_world(backend, args.pause_seconds),
        }

        print("", flush=True)
        print("Resumen visible suite", flush=True)
        for name, ok in results.items():
            print(f"  {name}: {'PASS' if ok else 'FAIL'}", flush=True)

        if not all(results.values()):
            raise SystemExit(1)
    finally:
        backend.close()


if __name__ == "__main__":
    sys.exit(main())
