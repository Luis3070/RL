from __future__ import annotations

import argparse

from config import SIMROBOT_BIN
from pybh_backend import PyBHBackend, WorldPlayerSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that visible SimRobot fails cleanly on an invalid world slot.")
    parser.add_argument("--simrobot-profile", choices=("2d", "3d"), default="2d")
    parser.add_argument("--launch-simrobot", action="store_true")
    parser.add_argument("--simrobot-bin", default=SIMROBOT_BIN)
    parser.add_argument("--invalid-opponent-number", type=int, default=20)
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
        print("Abre SimRobot y vuelve aca cuando la escena visible ya este cargada.", flush=True)
        input("Cuando la simulacion ya este corriendo, presiona Enter para enviar el reset invalido...")
        try:
            print("Enviando reset invalido y esperando rechazo explicito de SimRobot...", flush=True)
            backend.reset(
                ball_pos=[0.0, 0.0],
                robot_pos=[-1000.0, 0.0],
                robot_theta=0.0,
                opponents=(
                    WorldPlayerSpec(
                        number=args.invalid_opponent_number,
                        x=1000.0,
                        y=0.0,
                        theta=0.0,
                    ),
                ),
            )
            print("ERROR: el reset debía fallar y no falló.", flush=True)
        except Exception as exc:
            print(f"ok: {exc}", flush=True)
        input("Confirma en la GUI que no hubo reset parcial y presiona Enter para cerrar...")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
