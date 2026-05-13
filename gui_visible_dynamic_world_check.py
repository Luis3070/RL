from __future__ import annotations

import argparse
import time

from config import SIMROBOT_BIN
from pybh_backend import PyBHBackend, WorldPlayerSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check visible SimRobot dynamic world updates.")
    parser.add_argument("--simrobot-profile", choices=("2d", "3d"), default="2d")
    parser.add_argument("--launch-simrobot", action="store_true")
    parser.add_argument("--simrobot-bin", default=SIMROBOT_BIN)
    parser.add_argument("--pause-seconds", type=float, default=3.0)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--request-timeout-ms", type=int, default=None)
    parser.add_argument("--reset-obs-timeout-ms", type=int, default=None)
    parser.add_argument("--status-log-interval-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = PyBHBackend(
        mode="simrobot_visible",
        simrobot_profile=args.simrobot_profile,
        obs_timeout_ms=args.timeout_ms,
        visible_request_timeout_ms=args.request_timeout_ms,
        visible_reset_obs_timeout_ms=args.reset_obs_timeout_ms,
        visible_status_log_interval_s=args.status_log_interval_s,
        simrobot_launch=args.launch_simrobot,
        simrobot_bin=args.simrobot_bin,
    )
    backend.start()
    try:
        print("Abre SimRobot, presiona F5 y luego vuelve aca.", flush=True)
        input("Cuando la simulacion ya este corriendo, presiona Enter para enviar el reset inicial...")
        print("Enviando reset inicial y esperando confirmacion de SimRobot...", flush=True)
        obs = backend.reset(
            ball_pos=[0.0, 0.0],
            robot_pos=[-1400.0, 0.0],
            robot_theta=0.0,
            opponents=(WorldPlayerSpec(number=1, x=900.0, y=0.0, theta=3.14),),
        )
        print(
            "reset ok "
            f"obs_ready={int(obs.obs_ready)} "
            f"robot=({obs.robot_x:.1f},{obs.robot_y:.1f}) "
            f"ball=({obs.ball_x:.1f},{obs.ball_y:.1f})",
            flush=True,
        )

        time.sleep(args.pause_seconds)
        print("Enviando update dinamico: mover solo la pelota...", flush=True)
        obs = backend.set_dynamic_world(ball_pos=[300.0, 600.0])
        print(
            "movi solo la pelota "
            f"robot=({obs.robot_x:.1f},{obs.robot_y:.1f}) "
            f"ball=({obs.ball_x:.1f},{obs.ball_y:.1f})",
            flush=True,
        )

        time.sleep(args.pause_seconds)
        print("Enviando update dinamico: mover solo el rival...", flush=True)
        obs = backend.set_dynamic_world(opponents=(WorldPlayerSpec(number=1, x=300.0, y=-700.0, theta=3.14),))
        print(
            "movi solo el rival "
            f"robot=({obs.robot_x:.1f},{obs.robot_y:.1f}) "
            f"ball=({obs.ball_x:.1f},{obs.ball_y:.1f})",
            flush=True,
        )

        time.sleep(args.pause_seconds)
        print("Enviando update dinamico: mover solo el robot RL...", flush=True)
        obs = backend.set_dynamic_world(robot_pos=[-800.0, 400.0], robot_theta=0.5)
        print(
            "movi solo el robot RL "
            f"robot=({obs.robot_x:.1f},{obs.robot_y:.1f},{obs.robot_theta:.2f}) "
            f"ball=({obs.ball_x:.1f},{obs.ball_y:.1f})",
            flush=True,
        )

        input("Verifica en la GUI que no hubo reset completo y presiona Enter para cerrar...")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
