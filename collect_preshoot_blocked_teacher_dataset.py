from __future__ import annotations

import os
from pathlib import Path

import teacher_production_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "teacher_production_dataset" / "preshoot_blocked_focus_v1"

# Focus the collector on the instability we actually saw in the multi-seed probe:
# - preshoot_bad_angle_boundary: contrast / do-not-shoot-yet behavior
# - blocked_corridor_* + static-lane support: lane repair under pressure
# Dynamic full-field is kept only as light support because it was already stable.
DEFAULT_SCENARIOS = ",".join(
    (
        "preshoot_bad_angle_boundary",
        "blocked_corridor_repair",
        "blocked_corridor_finish_topup",
        "finish_band_2p5_3m_static_lane",
        "finish_bridge_static_topup",
        "dynamic_finish_pressure_topup",
    )
)


def _setdefault(name: str, value: str) -> None:
    os.environ.setdefault(name, value)


def main() -> None:
    _setdefault("RL_PROD_TEACHER_BACKEND_MODE", "simrobot_visible")
    _setdefault("RL_PROD_TEACHER_SIMROBOT_PROFILE", "3d_visible")
    _setdefault("RL_PROD_TEACHER_SCENARIO_SET", "all")
    _setdefault("RL_PROD_TEACHER_SCENARIOS", DEFAULT_SCENARIOS)
    _setdefault("RL_PROD_TEACHER_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))
    _setdefault("RL_PROD_TEACHER_EPISODES", "6")
    _setdefault("RL_PROD_TEACHER_MAX_ATTEMPT_MULTIPLIER", "4")
    _setdefault("RL_PROD_TEACHER_MAX_STEPS", "700")
    _setdefault("RL_PROD_TEACHER_SUCCESS_FILTER", "1")
    _setdefault("RL_PROD_TEACHER_SUCCESS_METRIC", "clean_finish")
    teacher_production_dataset.main()


if __name__ == "__main__":
    main()
