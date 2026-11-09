from __future__ import annotations

import os
import sys
from pathlib import Path

from config import BHUMAN_DIR


DEFAULT_EXTRA_SCENE = BHUMAN_DIR / "Config/Scenes/RL1v0_3D_extra.ros2"


def _has_scene_arg(argv: list[str]) -> bool:
    return "--scene" in argv


def main() -> None:
    argv = sys.argv[1:]
    if not _has_scene_arg(argv):
        argv = [*argv, "--scene", str(DEFAULT_EXTRA_SCENE)]

    target = Path(__file__).with_name("live_simrobot_probe.py")
    os.execv(sys.executable, [sys.executable, str(target), *argv])


if __name__ == "__main__":
    main()
