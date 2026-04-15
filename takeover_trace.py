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


def _trace_all_steps() -> bool:
    """Si True, una fila por step con sidecar real (CSV grande; análisis completo)."""
    return _env_flag("TRACE_TAKEOVER_ALL_STEPS", default=False)


@lru_cache(maxsize=1)
def _trace_sample_every() -> int:
    """
    Si N>0, loguea 1 de cada N steps (sidecar real) aunque no haya handoff/takeover.
    Cubre curricula tipo stages 5–12 donde should_log quedaría vacío siempre.
    """
    raw = os.environ.get("TRACE_TAKEOVER_SAMPLE_EVERY", "").strip()
    if not raw:
        raw = _repo_dotenv().get("TRACE_TAKEOVER_SAMPLE_EVERY", "").strip()
    if not raw:
        return 0
    try:
        n = int(raw, 10)
        return max(0, n)
    except ValueError:
        return 0


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
        "control_owner",
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
        "takeover_kick",
        "takeover_kick_proxy",
        "pending_kick_hold",
        "pending_kick_hold_frames",
        "post_kick_release_frames",
        "takeover_control_frames",
        "takeover_progress_frames",
        "takeover_ball_progress_mm",
        "takeover_ball_progress_along_target_mm",
        "ball_out",
        "ball_x",
        "ball_y",
        "ball_vx",
        "ball_vy",
        "ball_speed",
        "ball_heading_deg",
        "agent_x",
        "agent_y",
        "agent_theta_rad",
        "agent_theta_deg",
        "agent_vx",
        "agent_vy",
        "agent_speed",
        "agent_omega",
        "agent_walk_target_x",
        "agent_walk_target_y",
        "agent_ball_distance_mm",
        "agent_ball_bearing_rad",
        "agent_ball_bearing_deg",
        "agent_ball_heading_alignment",
        "sidecar_backend",
        "sidecar_current_tactic",
        "sidecar_macro_type",
        "sidecar_macro_type_effective",
        "sidecar_skill_request",
        "sidecar_motion_request",
        "sidecar_motion_executed_phase",
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
        "kick_length_raw",
        "skill_target_x",
        "skill_target_y",
        "bridge_ball_target_x",
        "bridge_ball_target_y",
        "bridge_contact_x",
        "bridge_contact_y",
        "field_ball_x",
        "field_ball_y",
        "field_ball_intercept_x",
        "field_ball_intercept_y",
        "shoot_target_x",
        "shoot_target_y",
        "ball_target_x",
        "ball_target_y",
        "effective_target_x",
        "effective_target_y",
        "effective_motion_target_x",
        "effective_motion_target_y",
        "effective_target_theta",
        "bridge_frame_id",
        "diag_ball_step_jump_mm",
        "diag_ball_teleport_flag",
        "diag_active_agent_step_jump_mm",
        "diag_agent_teleport_flag",
        "diag_shoot_semantic_dot",
        "diag_shoot_semantic_error",
        "diag_shoot_backward_flag",
        "episode_diag_ball_teleport_frames",
        "episode_diag_agent_teleport_frames",
        "episode_diag_shoot_backward_frames",
        "diag_active_agent_theta_step_delta_deg",
        "diag_theta_snap_prep_flag",
        "diag_ball_agent_center_dist_mm",
        "episode_diag_theta_snap_prep_frames",
        # Instrumentación extendida (control / geometría / kicks / post-kick)
        "diag_prev_control_owner",
        "diag_current_control_owner",
        "diag_control_owner_changed",
        "diag_control_transition",
        "episode_diag_control_owner_changes",
        "diag_active_agent_changed_step",
        "diag_ball_speed_step_delta",
        "diag_ball_large_jump_flag",
        "diag_ball_jump_when_kick_proxy_mm",
        "diag_ball_agent_sep_mm",
        "diag_ball_agent_overlap_flag",
        "diag_ball_agent_penetration_flag",
        "diag_min_opp_dist_mm",
        "diag_opp_pressure_flag",
        "diag_motion_walk_theta_deg",
        "diag_motion_walk_theta_vs_body_deg",
        "diag_effective_target_theta_vs_body_deg",
        "diag_skill_target_dir_vs_body_deg",
        "diag_motion_target_dir_vs_body_deg",
        "diag_theta_snap_motion_prep_flag",
        "diag_frames_since_kick_proxy",
        "diag_rapid_kick_proxy_flag",
        "diag_sidecar_macro_neq_takeover_macro",
        "diag_handoff_ready_rl_control_flag",
        "diag_macro_active_without_takeover_flag",
        "diag_ball_vel_x_mm_s",
        "diag_shoot_vel_retrograde_x_flag",
        "diag_post_kick_vel_align_dot",
        "diag_post_kick_inversion_flag",
        "episode_diag_ball_overlap_frames",
        "episode_diag_ball_penetration_frames",
        "episode_diag_soft_ball_jump_frames",
        "episode_diag_motion_vs_body_mismatch_frames",
        "episode_diag_theta_snap_motion_prep_frames",
        "episode_diag_rapid_kick_proxy_frames",
        "episode_diag_post_kick_inversion_frames",
        "episode_diag_shoot_vel_retrograde_frames",
        "episode_diag_opp_pressure_frames",
        "episode_diag_ball_overlap_rate",
        "episode_diag_ball_penetration_rate",
        "episode_diag_soft_ball_jump_rate",
        "episode_diag_motion_vs_body_mismatch_rate",
        "episode_diag_theta_snap_motion_prep_rate",
        "episode_diag_rapid_kick_proxy_rate",
        "episode_diag_post_kick_inversion_rate",
        "episode_diag_shoot_vel_retrograde_rate",
        "episode_diag_opp_pressure_rate",
        "episode_diag_control_owner_change_rate",
        # Física balón, inactivos, activation graph (bridge), desglose reward
        "diag_ball_collision_tags",
        "diag_inactive_ball_dists_json",
        "diag_activation_graph_len",
        "diag_activation_states",
        "diag_activation_options",
        "diag_activation_zweikampf_flag",
        "diag_macro_from_activation_graph",
        "diag_macro_bridge_vs_activation_flag",
        "rw_sparse",
        "rw_approach_ball",
        "rw_handoff_alignment",
        "rw_handoff_ready_bonus",
        "rw_handoff_break_penalty",
        "rw_playball_entry_bonus",
        "rw_ready_but_walk_penalty",
        "rw_takeover_stability",
        "rw_takeover_success",
        "rw_takeover_failure",
        "rw_support_spacing",
        "rw_anti_blocking",
        "rw_out_of_bounds",
        "rw_fall_penalty",
        "rw_oscillation_penalty",
        "rw_ball_out_penalty",
        "rw_fov_bonus",
        "rw_total",
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
        logger = cls(log_dir)
        if _trace_all_steps():
            print(
                "[TakeoverTrace] TRACE_TAKEOVER_ALL_STEPS=1 → CSV con **todos** los steps "
                "(sidecar real); archivo grande pero reproducible sin visualizer."
            )
        se = _trace_sample_every()
        if se > 0:
            print(
                f"[TakeoverTrace] TRACE_TAKEOVER_SAMPLE_EVERY={se} → además se loguea 1/{se} "
                "steps en backend real (útil sin handoff / macro none)."
            )
        return logger

    def should_log(self, info: dict[str, Any]) -> bool:
        if not info:
            return False
        if info.get("sidecar_backend") != "real":
            return False
        if _trace_all_steps():
            return True
        se = _trace_sample_every()
        if se > 0:
            try:
                st = int(info.get("step", 0) or 0)
                if st % se == 0:
                    return True
            except (TypeError, ValueError):
                pass
        return bool(
            info.get("handoff_ready")
            or info.get("takeover_active")
            or str(info.get("macro_type", "none")) != "none"
            or bool(info.get("playball_entry"))
            or bool(info.get("pending_kick_hold"))
            or bool(info.get("takeover_kick"))
            or bool(info.get("takeover_kick_proxy"))
            or int(info.get("post_kick_release_frames", 0) or 0) > 0
            or bool(info.get("ball_out"))
            or str(info.get("last_macro_end_reason", "none")) != "none"
            or int(info.get("diag_ball_teleport_flag", 0) or 0) != 0
            or int(info.get("diag_agent_teleport_flag", 0) or 0) != 0
            or int(info.get("diag_shoot_backward_flag", 0) or 0) != 0
            or int(info.get("diag_theta_snap_prep_flag", 0) or 0) != 0
            or int(info.get("diag_control_owner_changed", 0) or 0) != 0
            or int(info.get("diag_active_agent_changed_step", 0) or 0) != 0
            or int(info.get("diag_ball_large_jump_flag", 0) or 0) != 0
            or int(info.get("diag_ball_agent_overlap_flag", 0) or 0) != 0
            or int(info.get("diag_ball_agent_penetration_flag", 0) or 0) != 0
            or int(info.get("diag_opp_pressure_flag", 0) or 0) != 0
            or int(info.get("diag_theta_snap_motion_prep_flag", 0) or 0) != 0
            or int(info.get("diag_rapid_kick_proxy_flag", 0) or 0) != 0
            or int(info.get("diag_sidecar_macro_neq_takeover_macro", 0) or 0) != 0
            or int(info.get("diag_handoff_ready_rl_control_flag", 0) or 0) != 0
            or int(info.get("diag_macro_active_without_takeover_flag", 0) or 0) != 0
            or int(info.get("diag_shoot_vel_retrograde_x_flag", 0) or 0) != 0
            or int(info.get("diag_post_kick_inversion_flag", 0) or 0) != 0
            or str(info.get("diag_ball_collision_tags", "none") or "none") != "none"
            or int(info.get("diag_activation_zweikampf_flag", 0) or 0) != 0
            or int(info.get("diag_macro_bridge_vs_activation_flag", 0) or 0) != 0
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
        bridge_ball_target_x, bridge_ball_target_y = _xy(info.get("sidecar_motion_ball_target_field"))
        bridge_contact_x, bridge_contact_y = _xy(info.get("sidecar_motion_contact_point_field"))
        field_ball_x, field_ball_y = _xy(info.get("sidecar_field_ball_position"))
        field_ball_intercept_x, field_ball_intercept_y = _xy(
            info.get("sidecar_field_ball_intercepted_position")
        )
        ball_vx, ball_vy = _xy(info.get("ball_vel"))
        agent_x, agent_y = _xy(info.get("active_agent_pos"))
        agent_vx, agent_vy = _xy(info.get("active_agent_vel"))
        agent_walk_target_x, agent_walk_target_y = _xy(info.get("active_agent_walk_target"))
        shoot_x, shoot_y = _xy(info.get("sidecar_behavior_shooting_to"))
        ball_target_x, ball_target_y = _xy(info.get("sidecar_ball_target_field"))
        effective_x, effective_y = _xy(info.get("sidecar_effective_target"))
        effective_motion_x, effective_motion_y = _xy(info.get("sidecar_effective_motion_target"))
        ball_x, ball_y = _xy(info.get("ball_pos"))

        row = {
            "episode": int(episode),
            "stage_id": int(stage_id),
            "stage_name": stage_name,
            "env_step": int(info.get("step", 0) or 0),
            "control_owner": str(info.get("control_owner", "")),
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
            "takeover_kick": int(bool(info.get("takeover_kick", False))),
            "takeover_kick_proxy": int(bool(info.get("takeover_kick_proxy", False))),
            "pending_kick_hold": int(bool(info.get("pending_kick_hold", False))),
            "pending_kick_hold_frames": int(info.get("pending_kick_hold_frames", 0) or 0),
            "post_kick_release_frames": int(info.get("post_kick_release_frames", 0) or 0),
            "takeover_control_frames": int(info.get("takeover_control_frames", 0) or 0),
            "takeover_progress_frames": int(info.get("takeover_progress_frames", 0) or 0),
            "takeover_ball_progress_mm": float(info.get("takeover_ball_progress_mm", 0.0) or 0.0),
            "takeover_ball_progress_along_target_mm": float(
                info.get("takeover_ball_progress_along_target_mm", 0.0) or 0.0
            ),
            "ball_out": int(bool(info.get("ball_out", False))),
            "ball_x": ball_x,
            "ball_y": ball_y,
            "ball_vx": ball_vx,
            "ball_vy": ball_vy,
            "ball_speed": info.get("ball_speed"),
            "ball_heading_deg": info.get("ball_heading_deg"),
            "agent_x": agent_x,
            "agent_y": agent_y,
            "agent_theta_rad": info.get("active_agent_theta"),
            "agent_theta_deg": info.get("active_agent_theta_deg"),
            "agent_vx": agent_vx,
            "agent_vy": agent_vy,
            "agent_speed": info.get("active_agent_speed"),
            "agent_omega": info.get("active_agent_omega"),
            "agent_walk_target_x": agent_walk_target_x,
            "agent_walk_target_y": agent_walk_target_y,
            "agent_ball_distance_mm": info.get("active_agent_ball_distance_mm"),
            "agent_ball_bearing_rad": info.get("active_agent_ball_bearing_rad"),
            "agent_ball_bearing_deg": info.get("active_agent_ball_bearing_deg"),
            "agent_ball_heading_alignment": info.get("active_agent_ball_heading_alignment"),
            "sidecar_backend": str(info.get("sidecar_backend", "")),
            "sidecar_current_tactic": str(info.get("sidecar_current_tactic", "")),
            "sidecar_macro_type": str(info.get("sidecar_macro_type", "none")),
            "sidecar_macro_type_effective": str(
                info.get("sidecar_macro_type_effective", "none")
            ),
            "sidecar_skill_request": str(info.get("sidecar_skill_request", "none")),
            "sidecar_motion_request": str(info.get("sidecar_motion_request", "none")),
            "sidecar_motion_executed_phase": str(
                info.get("sidecar_motion_executed_phase", "unavailable")
            ),
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
            "kick_length_raw": info.get("sidecar_kick_length_raw"),
            "skill_target_x": skill_x,
            "skill_target_y": skill_y,
            "bridge_ball_target_x": bridge_ball_target_x,
            "bridge_ball_target_y": bridge_ball_target_y,
            "bridge_contact_x": bridge_contact_x,
            "bridge_contact_y": bridge_contact_y,
            "field_ball_x": field_ball_x,
            "field_ball_y": field_ball_y,
            "field_ball_intercept_x": field_ball_intercept_x,
            "field_ball_intercept_y": field_ball_intercept_y,
            "shoot_target_x": shoot_x,
            "shoot_target_y": shoot_y,
            "ball_target_x": ball_target_x,
            "ball_target_y": ball_target_y,
            "effective_target_x": effective_x,
            "effective_target_y": effective_y,
            "effective_motion_target_x": effective_motion_x,
            "effective_motion_target_y": effective_motion_y,
            "effective_target_theta": info.get("sidecar_effective_target_theta"),
            "bridge_frame_id": int(info.get("bridge_frame_id", -1) or -1),
            "diag_ball_step_jump_mm": float(info.get("diag_ball_step_jump_mm", 0.0) or 0.0),
            "diag_ball_teleport_flag": int(info.get("diag_ball_teleport_flag", 0) or 0),
            "diag_active_agent_step_jump_mm": float(
                info.get("diag_active_agent_step_jump_mm", 0.0) or 0.0
            ),
            "diag_agent_teleport_flag": int(info.get("diag_agent_teleport_flag", 0) or 0),
            "diag_shoot_semantic_dot": info.get("diag_shoot_semantic_dot"),
            "diag_shoot_semantic_error": info.get("diag_shoot_semantic_error"),
            "diag_shoot_backward_flag": int(info.get("diag_shoot_backward_flag", 0) or 0),
            "episode_diag_ball_teleport_frames": int(
                info.get("episode_diag_ball_teleport_frames", 0) or 0
            ),
            "episode_diag_agent_teleport_frames": int(
                info.get("episode_diag_agent_teleport_frames", 0) or 0
            ),
            "episode_diag_shoot_backward_frames": int(
                info.get("episode_diag_shoot_backward_frames", 0) or 0
            ),
            "diag_active_agent_theta_step_delta_deg": info.get(
                "diag_active_agent_theta_step_delta_deg"
            ),
            "diag_theta_snap_prep_flag": int(info.get("diag_theta_snap_prep_flag", 0) or 0),
            "diag_ball_agent_center_dist_mm": float(
                info.get("diag_ball_agent_center_dist_mm", 0.0) or 0.0
            ),
            "episode_diag_theta_snap_prep_frames": int(
                info.get("episode_diag_theta_snap_prep_frames", 0) or 0
            ),
            "diag_prev_control_owner": str(info.get("diag_prev_control_owner", "") or ""),
            "diag_current_control_owner": str(info.get("diag_current_control_owner", "") or ""),
            "diag_control_owner_changed": int(info.get("diag_control_owner_changed", 0) or 0),
            "diag_control_transition": str(info.get("diag_control_transition", "") or ""),
            "episode_diag_control_owner_changes": int(
                info.get("episode_diag_control_owner_changes", 0) or 0
            ),
            "diag_active_agent_changed_step": int(
                info.get("diag_active_agent_changed_step", 0) or 0
            ),
            "diag_ball_speed_step_delta": float(info.get("diag_ball_speed_step_delta", 0.0) or 0.0),
            "diag_ball_large_jump_flag": int(info.get("diag_ball_large_jump_flag", 0) or 0),
            "diag_ball_jump_when_kick_proxy_mm": float(
                info.get("diag_ball_jump_when_kick_proxy_mm", 0.0) or 0.0
            ),
            "diag_ball_agent_sep_mm": float(info.get("diag_ball_agent_sep_mm", 0.0) or 0.0),
            "diag_ball_agent_overlap_flag": int(info.get("diag_ball_agent_overlap_flag", 0) or 0),
            "diag_ball_agent_penetration_flag": int(
                info.get("diag_ball_agent_penetration_flag", 0) or 0
            ),
            "diag_min_opp_dist_mm": float(info.get("diag_min_opp_dist_mm", 0.0) or 0.0),
            "diag_opp_pressure_flag": int(info.get("diag_opp_pressure_flag", 0) or 0),
            "diag_motion_walk_theta_deg": info.get("diag_motion_walk_theta_deg"),
            "diag_motion_walk_theta_vs_body_deg": info.get("diag_motion_walk_theta_vs_body_deg"),
            "diag_effective_target_theta_vs_body_deg": info.get(
                "diag_effective_target_theta_vs_body_deg"
            ),
            "diag_skill_target_dir_vs_body_deg": info.get("diag_skill_target_dir_vs_body_deg"),
            "diag_motion_target_dir_vs_body_deg": info.get("diag_motion_target_dir_vs_body_deg"),
            "diag_theta_snap_motion_prep_flag": int(
                info.get("diag_theta_snap_motion_prep_flag", 0) or 0
            ),
            "diag_frames_since_kick_proxy": int(info.get("diag_frames_since_kick_proxy", 999) or 999),
            "diag_rapid_kick_proxy_flag": int(info.get("diag_rapid_kick_proxy_flag", 0) or 0),
            "diag_sidecar_macro_neq_takeover_macro": int(
                info.get("diag_sidecar_macro_neq_takeover_macro", 0) or 0
            ),
            "diag_handoff_ready_rl_control_flag": int(
                info.get("diag_handoff_ready_rl_control_flag", 0) or 0
            ),
            "diag_macro_active_without_takeover_flag": int(
                info.get("diag_macro_active_without_takeover_flag", 0) or 0
            ),
            "diag_ball_vel_x_mm_s": float(info.get("diag_ball_vel_x_mm_s", 0.0) or 0.0),
            "diag_shoot_vel_retrograde_x_flag": int(
                info.get("diag_shoot_vel_retrograde_x_flag", 0) or 0
            ),
            "diag_post_kick_vel_align_dot": info.get("diag_post_kick_vel_align_dot"),
            "diag_post_kick_inversion_flag": int(info.get("diag_post_kick_inversion_flag", 0) or 0),
            "episode_diag_ball_overlap_frames": int(
                info.get("episode_diag_ball_overlap_frames", 0) or 0
            ),
            "episode_diag_ball_penetration_frames": int(
                info.get("episode_diag_ball_penetration_frames", 0) or 0
            ),
            "episode_diag_soft_ball_jump_frames": int(
                info.get("episode_diag_soft_ball_jump_frames", 0) or 0
            ),
            "episode_diag_motion_vs_body_mismatch_frames": int(
                info.get("episode_diag_motion_vs_body_mismatch_frames", 0) or 0
            ),
            "episode_diag_theta_snap_motion_prep_frames": int(
                info.get("episode_diag_theta_snap_motion_prep_frames", 0) or 0
            ),
            "episode_diag_rapid_kick_proxy_frames": int(
                info.get("episode_diag_rapid_kick_proxy_frames", 0) or 0
            ),
            "episode_diag_post_kick_inversion_frames": int(
                info.get("episode_diag_post_kick_inversion_frames", 0) or 0
            ),
            "episode_diag_shoot_vel_retrograde_frames": int(
                info.get("episode_diag_shoot_vel_retrograde_frames", 0) or 0
            ),
            "episode_diag_opp_pressure_frames": int(
                info.get("episode_diag_opp_pressure_frames", 0) or 0
            ),
            "episode_diag_ball_overlap_rate": float(
                info.get("episode_diag_ball_overlap_rate", 0.0) or 0.0
            ),
            "episode_diag_ball_penetration_rate": float(
                info.get("episode_diag_ball_penetration_rate", 0.0) or 0.0
            ),
            "episode_diag_soft_ball_jump_rate": float(
                info.get("episode_diag_soft_ball_jump_rate", 0.0) or 0.0
            ),
            "episode_diag_motion_vs_body_mismatch_rate": float(
                info.get("episode_diag_motion_vs_body_mismatch_rate", 0.0) or 0.0
            ),
            "episode_diag_theta_snap_motion_prep_rate": float(
                info.get("episode_diag_theta_snap_motion_prep_rate", 0.0) or 0.0
            ),
            "episode_diag_rapid_kick_proxy_rate": float(
                info.get("episode_diag_rapid_kick_proxy_rate", 0.0) or 0.0
            ),
            "episode_diag_post_kick_inversion_rate": float(
                info.get("episode_diag_post_kick_inversion_rate", 0.0) or 0.0
            ),
            "episode_diag_shoot_vel_retrograde_rate": float(
                info.get("episode_diag_shoot_vel_retrograde_rate", 0.0) or 0.0
            ),
            "episode_diag_opp_pressure_rate": float(
                info.get("episode_diag_opp_pressure_rate", 0.0) or 0.0
            ),
            "episode_diag_control_owner_change_rate": float(
                info.get("episode_diag_control_owner_change_rate", 0.0) or 0.0
            ),
            "diag_ball_collision_tags": str(info.get("diag_ball_collision_tags", "none") or "none"),
            "diag_inactive_ball_dists_json": str(
                info.get("diag_inactive_ball_dists_json", "{}") or "{}"
            ),
            "diag_activation_graph_len": int(info.get("diag_activation_graph_len", 0) or 0),
            "diag_activation_states": str(info.get("diag_activation_states", "") or ""),
            "diag_activation_options": str(info.get("diag_activation_options", "") or ""),
            "diag_activation_zweikampf_flag": int(
                info.get("diag_activation_zweikampf_flag", 0) or 0
            ),
            "diag_macro_from_activation_graph": str(
                info.get("diag_macro_from_activation_graph", "none") or "none"
            ),
            "diag_macro_bridge_vs_activation_flag": int(
                info.get("diag_macro_bridge_vs_activation_flag", 0) or 0
            ),
            "rw_sparse": float(info.get("rw_sparse", 0.0) or 0.0),
            "rw_approach_ball": float(info.get("rw_approach_ball", 0.0) or 0.0),
            "rw_handoff_alignment": float(info.get("rw_handoff_alignment", 0.0) or 0.0),
            "rw_handoff_ready_bonus": float(info.get("rw_handoff_ready_bonus", 0.0) or 0.0),
            "rw_handoff_break_penalty": float(info.get("rw_handoff_break_penalty", 0.0) or 0.0),
            "rw_playball_entry_bonus": float(info.get("rw_playball_entry_bonus", 0.0) or 0.0),
            "rw_ready_but_walk_penalty": float(info.get("rw_ready_but_walk_penalty", 0.0) or 0.0),
            "rw_takeover_stability": float(info.get("rw_takeover_stability", 0.0) or 0.0),
            "rw_takeover_success": float(info.get("rw_takeover_success", 0.0) or 0.0),
            "rw_takeover_failure": float(info.get("rw_takeover_failure", 0.0) or 0.0),
            "rw_support_spacing": float(info.get("rw_support_spacing", 0.0) or 0.0),
            "rw_anti_blocking": float(info.get("rw_anti_blocking", 0.0) or 0.0),
            "rw_out_of_bounds": float(info.get("rw_out_of_bounds", 0.0) or 0.0),
            "rw_fall_penalty": float(info.get("rw_fall_penalty", 0.0) or 0.0),
            "rw_oscillation_penalty": float(info.get("rw_oscillation_penalty", 0.0) or 0.0),
            "rw_ball_out_penalty": float(info.get("rw_ball_out_penalty", 0.0) or 0.0),
            "rw_fov_bonus": float(info.get("rw_fov_bonus", 0.0) or 0.0),
            "rw_total": float(info.get("rw_total", 0.0) or 0.0),
        }
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
