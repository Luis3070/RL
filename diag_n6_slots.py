"""Per-slot sanity diagnostic for N=6 visible parallel training.

Runs DIAG_STEPS training-style steps with the full production config
(LP_NUM_THREADS=4, simrobot_visible, 3d_visible, obs_mode=corrected)
and reports wire metrics, obs-space stats, and action diversity for
EVERY slot independently.  Pass when all 6 slots look healthy.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import math
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SabanaHerons2026", "Make", "Python"))

from config import ENV, BackendConfig
from parallel_environment import ParallelSlotSimRobotEnv
from observation import ObsIndex
from action import SkillType
from config import ROOT_DIR
from pybh_backend import SYNTHETIC_BALL_WARMUP_STEPS, SYNTHETIC_POSE_WARMUP_STEPS
from visible_parallel_runtime import VisibleParallelRuntime

DIAG_STEPS   = 120   # ~5-6 seconds sim time per slot at 22 joint-steps/s
DIAG_WARMUP_STEPS = max(SYNTHETIC_POSE_WARMUP_STEPS, SYNTHETIC_BALL_WARMUP_STEPS)
PLAYER_NOS   = (1, 2, 3, 4, 5, 6)
SHM_NAME     = "/pybh_rl_shared_state_v3"
DISPLAY_BASE = 410
RUNS_DIR     = ROOT_DIR / "runs" / "diag_n6_slots"

# ── env config ───────────────────────────────────────────────────────────────
backend_cfg = replace(
    ENV.backend,
    mode="simrobot_visible",
    obs_mode="corrected",
    simrobot_profile="3d_visible",
    frames_per_step=3,
)
env_cfg = replace(ENV, backend=backend_cfg, parallel_player_numbers=PLAYER_NOS)


def _obs_to_numpy(obs_tensor) -> np.ndarray:
    if isinstance(obs_tensor, torch.Tensor):
        return obs_tensor.detach().cpu().numpy().flatten()
    return np.asarray(obs_tensor).flatten()


def _write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _all_close_to_zero(values: list[float], *, atol: float = 1e-6) -> bool:
    return bool(values) and all(abs(v) <= atol for v in values)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RUNS_DIR / f"diag_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = VisibleParallelRuntime(
        player_numbers=PLAYER_NOS,
        shm_name=SHM_NAME,
        display_base=DISPLAY_BASE,
    )
    runtime.start()

    env = ParallelSlotSimRobotEnv(env_cfg=env_cfg)
    agents = env.possible_agents  # ['player_1', ..., 'player_6']

    # per-slot accumulators
    wire_keys = [
        "synthetic_pose_observation",
        "synthetic_ball_observation",
        "corrected_exported_robot_pose",
        "corrected_exported_ball",
        "ball_export_fresh",
        "pose_alignment_warning",
        "ball_observation_stale",
        "ball_observation_collapsed",
        "bridge_obs_ready",
    ]
    counts:  Dict[str, Dict[str, int]]   = {a: defaultdict(int) for a in agents}
    obs_buf: Dict[str, list]             = {a: [] for a in agents}
    frames:  Dict[str, list]             = {a: [] for a in agents}
    raw_time_since_seen: Dict[str, list[float]] = {a: [] for a in agents}
    raw_time_since_disappeared: Dict[str, list[float]] = {a: [] for a in agents}
    skill_counts: Dict[str, Dict[str, int]] = {a: defaultdict(int) for a in agents}

    obs_dict, _ = env.reset()
    dummy_action = torch.zeros(1, 5)
    collected_steps = 0
    terminated_early = False

    print(f"\n[diag] running {DIAG_STEPS} steps across {len(agents)} slots …\n", flush=True)

    for step in range(DIAG_STEPS):
        actions = {a: dummy_action for a in agents}
        obs_dict, _, _, _, infos = env.step(actions)
        if len(infos) != len(agents):
            terminated_early = True
            print(
                f"[diag] stopping early at step {step + 1}: "
                f"expected {len(agents)} agent infos, got {len(infos)}",
                flush=True,
            )
            break

        for agent in agents:
            info = infos[agent]
            if step >= DIAG_WARMUP_STEPS:
                for k in wire_keys:
                    if info.get(k):
                        counts[agent][k] += 1

                obs_buf[agent].append(_obs_to_numpy(obs_dict[agent]))
                frames[agent].append(int(info.get("bridge_frame", 0)))
                raw_time_since_seen[agent].append(float(info.get("time_since_ball_seen", 0.0)))
                raw_time_since_disappeared[agent].append(float(info.get("raw_obs", {}).get("time_since_ball_disappeared", 0.0)))

                skill = info.get("executed_skill_name") or info.get("policy_skill_name") or "unknown"
                skill_counts[agent][skill] += 1
        if step >= DIAG_WARMUP_STEPS:
            collected_steps += 1

        if (step + 1) % 30 == 0:
            print(f"  step {step + 1}/{DIAG_STEPS}", flush=True)

    env.close()
    runtime.close()

    # ── report ────────────────────────────────────────────────────────────────
    n = collected_steps
    if n <= 0:
        raise RuntimeError("diag_n6_slots collected zero scored steps after warmup")
    print("\n" + "=" * 70)
    print("PER-SLOT WIRE SANITY")
    print("=" * 70)
    print(f"(scored window: {n} steps after {DIAG_WARMUP_STEPS} warmup steps)")
    header = f"{'metric':<36}" + "".join(f"  p{p}" for p in PLAYER_NOS)
    print(header)
    print("-" * len(header))
    for k in wire_keys:
        row = f"{k:<36}"
        for agent in agents:
            rate = counts[agent][k] / n
            row += f"  {rate:.2f}"
        print(row)

    print("\n" + "=" * 70)
    print("OBS-SPACE STATS (mean ± std, 23 features)")
    print("=" * 70)
    obs_arrs = {a: np.stack(obs_buf[a]) for a in agents}  # (N, 23)
    feat_names = [o.name for o in ObsIndex]

    print(f"\n{'feature':<32}", end="")
    for p in PLAYER_NOS:
        print(f"  p{p}μ    p{p}σ ", end="")
    print()
    print("-" * (32 + len(PLAYER_NOS) * 14))

    any_nan = False
    any_frozen = []
    for fi, name in enumerate(feat_names):
        row = f"{name:<32}"
        stds = []
        for agent in agents:
            col = obs_arrs[agent][:, fi]
            mu, sigma = col.mean(), col.std()
            if math.isnan(mu) or math.isnan(sigma):
                any_nan = True
            row += f"  {mu:+.3f}  {sigma:.3f}"
            stds.append(sigma)
        # flag if ALL slots have std ≈ 0 AND feature is not supposed to be fixed
        if all(s < 0.005 for s in stds):
            any_frozen.append(name)
        print(row)

    print("\n" + "=" * 70)
    print("INTER-SLOT DIVERSITY (std of per-slot means — should be > 0 for pose/ball)")
    print("=" * 70)
    slot_means = np.stack([obs_arrs[a].mean(axis=0) for a in agents])  # (6, 23)
    inter_std  = slot_means.std(axis=0)
    for fi, name in enumerate(feat_names):
        flag = "  ← low diversity" if inter_std[fi] < 0.01 and fi in (0, 1, 2, 3, 4) else ""
        print(f"  {name:<32}  inter_slot_std={inter_std[fi]:.4f}{flag}")

    print("\n" + "=" * 70)
    print("FRAME ADVANCEMENT PER SLOT (sanity: each slot advances independently)")
    print("=" * 70)
    for agent, p in zip(agents, PLAYER_NOS):
        f = frames[agent]
        delta = f[-1] - f[0] if len(f) > 1 else 0
        print(f"  player{p}: first_frame={f[0]}  last_frame={f[-1]}  delta={delta}ms")

    print("\n" + "=" * 70)
    print("ACTION DISTRIBUTION PER SLOT (dummy action → all walk, expected)")
    print("=" * 70)
    for agent, p in zip(agents, PLAYER_NOS):
        dist = dict(skill_counts[agent])
        print(f"  player{p}: {dist}")

    # ── pass/fail ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PASS / FAIL SUMMARY")
    print("=" * 70)
    failures = []
    warnings = []
    per_slot_summary_rows: list[dict[str, object]] = []

    for agent, p in zip(agents, PLAYER_NOS):
        r = {k: counts[agent][k] / n for k in wire_keys}
        per_slot_summary_rows.append(
            {
                "player_number": p,
                "obs_ready_rate": float(counts[agent]["bridge_obs_ready"] / n),
                "synthetic_pose_rate": float(r["synthetic_pose_observation"]),
                "synthetic_ball_rate": float(r["synthetic_ball_observation"]),
                "corrected_exported_robot_pose_rate": float(r["corrected_exported_robot_pose"]),
                "corrected_exported_ball_rate": float(r["corrected_exported_ball"]),
                "ball_export_fresh_rate": float(r["ball_export_fresh"]),
                "pose_alignment_warning_rate": float(r["pose_alignment_warning"]),
                "ball_stale_rate": float(r["ball_observation_stale"]),
                "ball_collapsed_rate": float(r["ball_observation_collapsed"]),
                "first_frame": int(frames[agent][0]) if frames[agent] else 0,
                "last_frame": int(frames[agent][-1]) if frames[agent] else 0,
                "frame_delta_ms": int(frames[agent][-1] - frames[agent][0]) if len(frames[agent]) > 1 else 0,
            }
        )

        if not counts[agent]["bridge_obs_ready"] / n > 0.99:
            failures.append(f"player{p}: obs_ready_rate={r['bridge_obs_ready']:.3f} < 0.99")
        if r["synthetic_pose_observation"] > 0.02:
            failures.append(f"player{p}: synthetic_pose_rate={r['synthetic_pose_observation']:.3f} > 0.02")
        if r["synthetic_ball_observation"] > 0.80:
            failures.append(f"player{p}: synthetic_ball_rate={r['synthetic_ball_observation']:.3f} > 0.80")
        if r["ball_export_fresh"] < 0.10:
            failures.append(f"player{p}: ball_export_fresh_rate={r['ball_export_fresh']:.3f} < 0.10")
        if r["ball_observation_collapsed"] > 0.10:
            warnings.append(f"player{p}: ball_collapsed_rate={r['ball_observation_collapsed']:.3f} > 0.10")

    if any_nan:
        failures.append("NaN detected in obs array")

    expected_frozen = {
        "ball_vel_x", "ball_vel_y", "can_score_now",
        "nearest_teammate_dist", "nearest_opponent_dist",
        "nearest_uncertain_obstacle_dist", "nearest_teammate_front_dist",
        "nearest_opponent_front_dist", "nearest_uncertain_front_dist",
        "pass_options_count",
    }
    dynamic_expected_frozen = set()
    no_stale_ball = all(counts[agent]["ball_observation_stale"] == 0 for agent in agents)
    if "time_since_ball_seen" in any_frozen:
        raw_seen_ranges = [
            max(raw_time_since_seen[agent]) - min(raw_time_since_seen[agent])
            for agent in agents
            if raw_time_since_seen[agent]
        ]
        raw_seen_max = max(
            (value for agent in agents for value in raw_time_since_seen[agent]),
            default=0.0,
        )
        # With TIME_NORM_MS=5000, sub-25ms variation can look frozen in normalized
        # space while still behaving correctly. Treat that as healthy variation.
        if raw_seen_ranges and all(delta >= 10.0 for delta in raw_seen_ranges):
            dynamic_expected_frozen.add("time_since_ball_seen")
        elif no_stale_ball and raw_seen_max <= 40.0:
            # Continuously refreshed perception can keep this timer pinned near
            # zero across the scored window without indicating a broken feature.
            dynamic_expected_frozen.add("time_since_ball_seen")

    if "time_since_ball_disappeared" in any_frozen:
        raw_disappeared_all = [
            value
            for agent in agents
            for value in raw_time_since_disappeared[agent]
        ]
        if no_stale_ball and _all_close_to_zero(raw_disappeared_all):
            dynamic_expected_frozen.add("time_since_ball_disappeared")

    unexpected_frozen = [f for f in any_frozen if f not in expected_frozen and f not in dynamic_expected_frozen]
    if unexpected_frozen:
        warnings.append(f"Unexpected frozen features (all slots, all steps): {unexpected_frozen}")

    if not failures and not warnings:
        print("\n  RESULT: PASS — all 6 slots healthy\n")
    else:
        if failures:
            print("\n  FAILURES:")
            for f in failures:
                print(f"    FAIL  {f}")
        if warnings:
            print("\n  WARNINGS:")
            for w in warnings:
                print(f"    WARN  {w}")
        print(f"\n  RESULT: {'FAIL' if failures else 'WARN'}\n")

    feature_rows: list[dict[str, object]] = []
    for fi, name in enumerate(feat_names):
        row: dict[str, object] = {"feature": name, "inter_slot_std": float(inter_std[fi])}
        for agent, p in zip(agents, PLAYER_NOS):
            col = obs_arrs[agent][:, fi]
            row[f"p{p}_mean"] = float(col.mean())
            row[f"p{p}_std"] = float(col.std())
        feature_rows.append(row)

    action_rows: list[dict[str, object]] = []
    for agent, p in zip(agents, PLAYER_NOS):
        for skill_name, count in sorted(skill_counts[agent].items()):
            action_rows.append(
                {
                    "player_number": p,
                    "skill_name": skill_name,
                    "count": int(count),
                    "rate": float(count / n),
                }
            )

    summary = {
        "timestamp": timestamp,
        "diag_steps": DIAG_STEPS,
        "warmup_steps_skipped": DIAG_WARMUP_STEPS,
        "scored_steps": n,
        "terminated_early": terminated_early,
        "player_numbers": list(PLAYER_NOS),
        "wire_keys": wire_keys,
        "result": "FAIL" if failures else "WARN" if warnings else "PASS",
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "unexpected_frozen_features": unexpected_frozen,
        "dynamically_expected_frozen_features": sorted(dynamic_expected_frozen),
        "any_nan": any_nan,
        "output_dir": str(output_dir),
    }

    _write_csv(
        output_dir / "per_slot_summary.csv",
        per_slot_summary_rows,
        fieldnames=list(per_slot_summary_rows[0].keys()) if per_slot_summary_rows else ["player_number"],
    )
    _write_csv(
        output_dir / "feature_stats.csv",
        feature_rows,
        fieldnames=list(feature_rows[0].keys()) if feature_rows else ["feature", "inter_slot_std"],
    )
    _write_csv(
        output_dir / "action_distribution.csv",
        action_rows,
        fieldnames=list(action_rows[0].keys()) if action_rows else ["player_number", "skill_name", "count", "rate"],
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"[diag_n6_slots] output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
