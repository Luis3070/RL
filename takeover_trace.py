from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


TRACE_FILE_NAME = "takeover_trace.csv"
_TRUE_VALUES = {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _repo_dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return values

    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raw = _repo_dotenv().get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in _TRUE_VALUES


def _xy(value: Any) -> tuple[Optional[float], Optional[float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None, None
    return None, None


class TakeoverTraceLogger:
    COLUMNS = [
        "episode",
        "stage_id",
        "stage_name",
        "env_step",
        "active_agent_id",
        "handoff_ready",
        "takeover_active",
        "macro_type",
        "takeover_macro_type",
        "macro_duration_frames",
        "macro_end_reason",
        "last_macro_end_reason",
        "playball_entry",
        "ready_but_walk",
        "ready_touch_but_walk",
        "ball_out",
        "ball_x",
        "ball_y",
        "sidecar_backend",
        "sidecar_current_tactic",
        "sidecar_macro_type",
        "sidecar_skill_request",
        "sidecar_motion_request",
        "sidecar_behavior_status",
        "behavior_walk_x",
        "behavior_walk_y",
        "behavior_walk_field_x",
        "behavior_walk_field_y",
        "motion_walk_x",
        "motion_walk_y",
        "motion_walk_theta",
        "motion_walk_field_x",
        "motion_walk_field_y",
        "motion_target_direction",
        "skill_target_direction",
        "kick_length",
        "skill_target_x",
        "skill_target_y",
        "shoot_target_x",
        "shoot_target_y",
        "ball_target_x",
        "ball_target_y",
        "effective_target_x",
        "effective_target_y",
        "bridge_frame_id",
    ]

    def __init__(self, log_dir: str):
        self._path = Path(log_dir) / TRACE_FILE_NAME
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        self._fh.flush()

    @property
    def path(self) -> str:
        return str(self._path)

    @classmethod
    def maybe_create(cls, log_dir: str) -> Optional["TakeoverTraceLogger"]:
        if not _env_flag("TRACE_TAKEOVER_STEPS", default=True):
            return None
        return cls(log_dir)

    def should_log(self, info: dict[str, Any]) -> bool:
        if not info:
            return False
        if info.get("sidecar_backend") != "real":
            return False
        return bool(
            info.get("handoff_ready")
            or info.get("takeover_active")
            or str(info.get("macro_type", "none")) != "none"
            or bool(info.get("playball_entry"))
            or bool(info.get("ball_out"))
            or str(info.get("last_macro_end_reason", "none")) != "none"
        )

    def log_step(self,
                 episode: int,
                 stage_id: int,
                 stage_name: str,
                 info: dict[str, Any]) -> None:
        if not self.should_log(info):
            return

        behavior_walk_x, behavior_walk_y = _xy(info.get("sidecar_behavior_walking_to"))
        behavior_walk_field_x, behavior_walk_field_y = _xy(info.get("sidecar_behavior_walking_to_field"))
        motion_walk_x, motion_walk_y = _xy(info.get("sidecar_motion_walk_target"))
        motion_walk_field_x, motion_walk_field_y = _xy(info.get("sidecar_motion_walk_target_field"))
        motion_walk_theta = None
        motion_walk_target = info.get("sidecar_motion_walk_target")
        if isinstance(motion_walk_target, (list, tuple)) and len(motion_walk_target) >= 3:
            try:
                motion_walk_theta = float(motion_walk_target[2])
            except (TypeError, ValueError):
                motion_walk_theta = None
        skill_x, skill_y = _xy(info.get("sidecar_skill_target"))
        shoot_x, shoot_y = _xy(info.get("sidecar_behavior_shooting_to"))
        ball_target_x, ball_target_y = _xy(info.get("sidecar_ball_target_field"))
        effective_x, effective_y = _xy(info.get("sidecar_effective_target"))
        ball_x, ball_y = _xy(info.get("ball_pos"))

        row = {
            "episode": int(episode),
            "stage_id": int(stage_id),
            "stage_name": stage_name,
            "env_step": int(info.get("step", 0) or 0),
            "active_agent_id": str(info.get("active_agent_id", "")),
            "handoff_ready": int(bool(info.get("handoff_ready", False))),
            "takeover_active": int(bool(info.get("takeover_active", False))),
            "macro_type": str(info.get("macro_type", "none")),
            "takeover_macro_type": str(info.get("takeover_macro_type", "none")),
            "macro_duration_frames": int(info.get("macro_duration_frames", 0) or 0),
            "macro_end_reason": str(info.get("macro_end_reason", "none")),
            "last_macro_end_reason": str(info.get("last_macro_end_reason", "none")),
            "playball_entry": int(bool(info.get("playball_entry", False))),
            "ready_but_walk": int(bool(info.get("ready_but_walk", False))),
            "ready_touch_but_walk": int(bool(info.get("ready_touch_but_walk", False))),
            "ball_out": int(bool(info.get("ball_out", False))),
            "ball_x": ball_x,
            "ball_y": ball_y,
            "sidecar_backend": str(info.get("sidecar_backend", "")),
            "sidecar_current_tactic": str(info.get("sidecar_current_tactic", "")),
            "sidecar_macro_type": str(info.get("sidecar_macro_type", "none")),
            "sidecar_skill_request": str(info.get("sidecar_skill_request", "none")),
            "sidecar_motion_request": str(info.get("sidecar_motion_request", "none")),
            "sidecar_behavior_status": str(info.get("sidecar_behavior_status", "")),
            "behavior_walk_x": behavior_walk_x,
            "behavior_walk_y": behavior_walk_y,
            "behavior_walk_field_x": behavior_walk_field_x,
            "behavior_walk_field_y": behavior_walk_field_y,
            "motion_walk_x": motion_walk_x,
            "motion_walk_y": motion_walk_y,
            "motion_walk_theta": motion_walk_theta,
            "motion_walk_field_x": motion_walk_field_x,
            "motion_walk_field_y": motion_walk_field_y,
            "motion_target_direction": info.get("sidecar_motion_target_direction"),
            "skill_target_direction": info.get("sidecar_skill_target_direction"),
            "kick_length": info.get("sidecar_kick_length"),
            "skill_target_x": skill_x,
            "skill_target_y": skill_y,
            "shoot_target_x": shoot_x,
            "shoot_target_y": shoot_y,
            "ball_target_x": ball_target_x,
            "ball_target_y": ball_target_y,
            "effective_target_x": effective_x,
            "effective_target_y": effective_y,
            "bridge_frame_id": int(info.get("bridge_frame_id", -1) or -1),
        }
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
