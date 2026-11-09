from __future__ import annotations

import os
from typing import Dict, Mapping

import numpy as np

from config import BackendConfig, WireAuditConfig


def _coerce_rate(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key, 0.0)
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return 0.0
    return rate if np.isfinite(rate) else 0.0


def expected_obs_mode_code(obs_mode: str) -> int:
    return 0 if obs_mode == "estimated" else 1


def build_wire_audit(metrics: Mapping[str, object], *, backend_cfg: BackendConfig, audit_cfg: WireAuditConfig) -> Dict[str, object]:
    failures: list[str] = []
    warnings: list[str] = []

    # Multi-agent team eval: support roles legitimately do NOT face the ball — they
    # track lanes/teammates and act on the injected team-ball estimate (obs[32:34]).
    # So the team-AGGREGATE ball_visible / obs-corrected rates are structurally
    # ~0.33-0.75, not the ~1.0 of a single ball-focused striker (verified: the same
    # checkpoint scores ball_visible=1.0 single-agent, ~0.5 with 2/3 supports). The
    # single-agent thresholds (0.90/0.99) therefore FALSE-FAIL every team eval. In
    # team mode we relax ONLY these two ball-visibility FAILURE thresholds to a floor;
    # every other check (synthetic, collapse, pose alignment, estimated-mode) is
    # unchanged. Single-agent runs never set the flag → bit-identical behaviour.
    team_mode = os.getenv("RL_WIRE_AUDIT_TEAM_MODE", "0") not in ("0", "false", "False", "")
    team_visibility_floor = float(os.getenv("RL_WIRE_AUDIT_TEAM_VISIBILITY_FLOOR", "0.30"))
    eff_min_expected_obs_mode_rate = (
        min(audit_cfg.min_expected_obs_mode_rate, team_visibility_floor)
        if team_mode else audit_cfg.min_expected_obs_mode_rate
    )
    eff_min_visible_ball_visible_rate = (
        min(audit_cfg.min_visible_ball_visible_rate, team_visibility_floor)
        if team_mode else audit_cfg.min_visible_ball_visible_rate
    )

    expected_mode_rate_key = f"obs_mode_{backend_cfg.obs_mode}_rate"
    expected_mode_rate = _coerce_rate(metrics, expected_mode_rate_key)

    if expected_mode_rate < eff_min_expected_obs_mode_rate:
        failures.append(
            f"{expected_mode_rate_key}={expected_mode_rate:.3f} < {eff_min_expected_obs_mode_rate:.3f}"
        )

    threshold_checks = (
        ("synthetic_pose_rate", audit_cfg.max_synthetic_pose_rate),
        ("synthetic_ball_rate", audit_cfg.max_synthetic_ball_rate),
        ("pose_alignment_warning_rate", audit_cfg.max_pose_alignment_warning_rate),
        ("ball_stale_rate", audit_cfg.max_ball_stale_rate),
        ("ball_collapsed_rate", audit_cfg.max_ball_collapsed_rate),
    )
    for metric_name, max_allowed in threshold_checks:
        value = _coerce_rate(metrics, metric_name)
        if value > max_allowed:
            failures.append(f"{metric_name}={value:.3f} > {max_allowed:.3f}")

    if backend_cfg.obs_mode == "estimated":
        corrected_pose_rate = _coerce_rate(metrics, "corrected_exported_robot_pose_rate")
        corrected_ball_rate = _coerce_rate(metrics, "corrected_exported_ball_rate")
        if corrected_pose_rate > audit_cfg.max_estimated_mode_corrected_pose_rate:
            failures.append(
                "corrected_exported_robot_pose_rate="
                f"{corrected_pose_rate:.3f} > {audit_cfg.max_estimated_mode_corrected_pose_rate:.3f}"
            )
        if corrected_ball_rate > audit_cfg.max_estimated_mode_corrected_ball_rate:
            failures.append(
                f"corrected_exported_ball_rate={corrected_ball_rate:.3f} > "
                f"{audit_cfg.max_estimated_mode_corrected_ball_rate:.3f}"
            )

    if backend_cfg.mode == "simrobot_visible":
        ball_visible_rate = _coerce_rate(metrics, "ball_visible_rate")
        ball_trusted_rate = _coerce_rate(metrics, "ball_trusted_rate")
        ball_rescued_rate = _coerce_rate(metrics, "ball_rescued_rate")
        misleading_seen_synthetic_rate = _coerce_rate(metrics, "misleading_seen_synthetic_rate")
        ball_confidence_score_mean = _coerce_rate(metrics, "ball_confidence_score_mean")

        if ball_visible_rate < eff_min_visible_ball_visible_rate:
            failures.append(
                f"ball_visible_rate={ball_visible_rate:.3f} < {eff_min_visible_ball_visible_rate:.3f}"
            )
        if ball_rescued_rate > audit_cfg.max_visible_ball_rescued_rate:
            failures.append(
                f"ball_rescued_rate={ball_rescued_rate:.3f} > {audit_cfg.max_visible_ball_rescued_rate:.3f}"
            )
        if misleading_seen_synthetic_rate > audit_cfg.max_visible_misleading_seen_synthetic_rate:
            failures.append(
                "misleading_seen_synthetic_rate="
                f"{misleading_seen_synthetic_rate:.3f} > {audit_cfg.max_visible_misleading_seen_synthetic_rate:.3f}"
            )

        if ball_trusted_rate < audit_cfg.min_visible_ball_trusted_rate_warning:
            warnings.append(
                f"ball_trusted_rate={ball_trusted_rate:.3f} < {audit_cfg.min_visible_ball_trusted_rate_warning:.3f}"
            )
        if ball_rescued_rate > audit_cfg.max_visible_ball_rescued_rate_warning:
            warnings.append(
                f"ball_rescued_rate={ball_rescued_rate:.3f} > {audit_cfg.max_visible_ball_rescued_rate_warning:.3f}"
            )
        if misleading_seen_synthetic_rate > audit_cfg.max_visible_misleading_seen_synthetic_rate_warning:
            warnings.append(
                "misleading_seen_synthetic_rate="
                f"{misleading_seen_synthetic_rate:.3f} > {audit_cfg.max_visible_misleading_seen_synthetic_rate_warning:.3f}"
            )
        if ball_confidence_score_mean < audit_cfg.min_visible_ball_confidence_score_mean_warning:
            warnings.append(
                "ball_confidence_score_mean="
                f"{ball_confidence_score_mean:.3f} < {audit_cfg.min_visible_ball_confidence_score_mean_warning:.3f}"
            )

    collision_rate = _coerce_rate(metrics, "collision_rate")
    if collision_rate <= 0.0:
        collision_rate = _coerce_rate(metrics, "collision_step_rate")
    if collision_rate > audit_cfg.max_collision_rate_warning:
        warnings.append(f"collision_rate={collision_rate:.3f} > {audit_cfg.max_collision_rate_warning:.3f}")

    status = "FAIL" if failures else "WARN" if warnings else "PASS"
    return {
        "wire_audit_status": status,
        "wire_audit_expected_obs_mode": backend_cfg.obs_mode,
        "wire_audit_expected_obs_mode_code": expected_obs_mode_code(backend_cfg.obs_mode),
        "wire_audit_expected_obs_mode_rate": expected_mode_rate,
        "wire_audit_failure_count": len(failures),
        "wire_audit_warning_count": len(warnings),
        "wire_audit_failures": " | ".join(failures),
        "wire_audit_warnings": " | ".join(warnings),
    }
