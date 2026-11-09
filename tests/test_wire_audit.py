from __future__ import annotations

from rlfootball.core.config import BackendConfig, TRAIN
from rlfootball.core.wire_audit import build_wire_audit


def test_wire_audit_passes_for_healthy_corrected_metrics():
    metrics = {
        "obs_mode_corrected_rate": 1.0,
        "synthetic_pose_rate": 0.0,
        "synthetic_ball_rate": 0.0,
        "pose_alignment_warning_rate": 0.0,
        "ball_stale_rate": 0.0,
        "ball_collapsed_rate": 0.0,
        "collision_rate": 0.0,
    }

    audit = build_wire_audit(
        metrics,
        backend_cfg=BackendConfig(mode="simrobot", obs_mode="corrected"),
        audit_cfg=TRAIN.wire_audit,
    )

    assert audit["wire_audit_status"] == "PASS"
    assert audit["wire_audit_failure_count"] == 0


def test_wire_audit_fails_if_estimated_mode_observes_corrected_exports():
    metrics = {
        "obs_mode_estimated_rate": 1.0,
        "synthetic_pose_rate": 0.0,
        "synthetic_ball_rate": 0.0,
        "pose_alignment_warning_rate": 0.0,
        "ball_stale_rate": 0.0,
        "ball_collapsed_rate": 0.0,
        "corrected_exported_robot_pose_rate": 0.25,
        "corrected_exported_ball_rate": 0.25,
    }

    audit = build_wire_audit(
        metrics,
        backend_cfg=BackendConfig(mode="simrobot", obs_mode="estimated"),
        audit_cfg=TRAIN.wire_audit,
    )

    assert audit["wire_audit_status"] == "FAIL"
    assert audit["wire_audit_failure_count"] >= 1


def test_wire_audit_visible_warns_for_semantically_weak_ball_signal():
    metrics = {
        "obs_mode_corrected_rate": 1.0,
        "synthetic_pose_rate": 0.0,
        "synthetic_ball_rate": 0.10,
        "pose_alignment_warning_rate": 0.0,
        "ball_stale_rate": 0.0,
        "ball_collapsed_rate": 0.0,
        "ball_visible_rate": 0.98,
        "ball_trusted_rate": 0.30,
        "ball_rescued_rate": 0.35,
        "misleading_seen_synthetic_rate": 0.28,
        "ball_confidence_score_mean": 0.32,
    }

    audit = build_wire_audit(
        metrics,
        backend_cfg=BackendConfig(mode="simrobot_visible", obs_mode="corrected"),
        audit_cfg=TRAIN.wire_audit,
    )

    assert audit["wire_audit_status"] == "WARN"
    assert audit["wire_audit_failure_count"] == 0
    assert audit["wire_audit_warning_count"] >= 1


def test_wire_audit_visible_fails_for_heavily_rescued_ball_signal():
    metrics = {
        "obs_mode_corrected_rate": 1.0,
        "synthetic_pose_rate": 0.0,
        "synthetic_ball_rate": 0.55,
        "pose_alignment_warning_rate": 0.0,
        "ball_stale_rate": 0.0,
        "ball_collapsed_rate": 0.0,
        "ball_visible_rate": 0.70,
        "ball_trusted_rate": 0.10,
        "ball_rescued_rate": 0.75,
        "misleading_seen_synthetic_rate": 0.65,
        "ball_confidence_score_mean": 0.20,
    }

    audit = build_wire_audit(
        metrics,
        backend_cfg=BackendConfig(mode="simrobot_visible", obs_mode="corrected"),
        audit_cfg=TRAIN.wire_audit,
    )

    assert audit["wire_audit_status"] == "FAIL"
    assert audit["wire_audit_failure_count"] >= 1
