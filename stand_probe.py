import pybh.controller as pybh
import time
import subprocess
from pathlib import Path

BHUMAN_DIR = Path("/home/rafael/Documentos/Robocup2026/SabanaHerons2026")
scene = BHUMAN_DIR / "Config/Scenes/RL1v0_3D_extra_dummy_front.ros2"
simrobot = BHUMAN_DIR / "Build/Linux/SimRobot/Develop/SimRobot"

proc = subprocess.Popen([str(simrobot), str(scene)])
print("SimRobot abierto — presiona F5 para arrancar, Ctrl+C para salir")
time.sleep(3)

try:
    while True:
        pybh.rl_set_action(1, "stand", 0.0, 0.0, 0.0)
        time.sleep(0.1)
except KeyboardInterrupt:
    proc.terminate()
