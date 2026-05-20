"""Per-slot sanity diagnostic for N=7 visible parallel training (production config).

Mirrors diag_n6_slots.py but for the 7-slot production configuration:
  - Verifies all 7 slots are healthy (wire metrics, obs-space stats, frame advancement)
  - Reports raw env throughput steps/sec and transitions/sec vs 6-slot baseline
  - LP_NUM_THREADS auto-computes to 3/slot (21 total render threads vs 24 at 6-slot)

Display base 420 avoids conflict with training runs that use base 410.
SHM name /pybh_rl_diag_n7 avoids conflict with /pybh_rl_shared_state_v3.

Player 8 was tested and exceeds BHuman's SPL player number cap (1-7).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import math
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SabanaHerons2026", "Make", "Python"))

from config import ENV, ROOT_DIR
from parallel_environment import ParallelSlotSimRobotEnv
from observation import ObsIndex
from pybh_backend import SYNTHETIC_BALL_WARMUP_STEPS, SYNTHETIC_POSE_WARMUP_STEPS
from visible_parallel_runtime import VisibleParallelRuntime

DIAG_STEPS        = 120
DIAG_WARMUP_STEPS = max(SYNTHETIC_POSE_WARMUP_STEPS, SYNTHETIC_BALL_WARMUP_STEPS)
PLAYER_NOS        = (1, 2, 3, 4, 5, 6, 7)
SHM_NAME          = "/pybh_rl_diag_n7"
DISPLAY_BASE      = 420   # players use :421 – :427; 6-slot diag uses :411 – :416
RUNS_DIR          = ROOT_DIR / "runs" / "diag_n7_slots"

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


def _write_csv(path: Path, rows: list[dict], *, fieldnames: list[str]) -> None:
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

    os.environ["PYBH_RL_SHM_NAME"] = SHM_NAME

    runtime = VisibleParallelRuntime(
        player_numbers=PLAYER_NOS,
        shm_name=SHM_NAME,
        display_base=DISPLAY_BASE,
    )
    runtime.start()

    env = ParallelSlotSimRobotEnv(env_cfg=env_cfg)
    agents = env.possible_agents

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
    counts:   Dict[str, Dict[str, int]]   = {a: defaultdict(int) for a in agents}
    obs_buf:  Dict[str, list]             = {a: [] for a in agents}
    frames:   Dict[str, list]             = {a: [] for a in agents}
    raw_time_since_seen: Dict[str, list[float]] = {a: [] for a in agents}
    raw_time_since_disappeared: Dict[str, list[float]] = {a: [] for a in agents}
    skill_counts: Dict[str, Dict[str, int]] = {a: defaultdict(int) for a in agents}

    obs_dict, _ = env.reset()
    dummy_action = torch.zeros(1, 5)
    collected_steps = 0
    terminated_early = False

    print(f"\n[diag_n7] running {DIAG_STEPS} steps across {len(agents)} slots …\n", flush=True)

    # Wall-clock throughput measurement (post-warmup only)
    t_start: float | None = None

    for step in range(DIAG_STEPS):
        actions = {a: dummy_action for a in agents}
        t0 = time.perf_counter()
        obs_dict, _, _, _, infos = env.step(actions)

        if len(infos) != len(agents):
            terminated_early = True
            print(
                f"[diag_n7] stopping early at step {step + 1}: "
                f"expected {len(agents)} agent infos, got {len(infos)}",
                flush=True,
            )
            break

        post_warmup = step >= DIAG_WARMUP_STEPS
        if post_warmup and t_start is None:
            t_start = t0

        for agent in agents:
            info = infos[agent]
            if post_warmup:
                for k in wire_keys:
                    if info.get(k):
                        counts[agent][k] += 1
                obs_buf[agent].append(_obs_to_numpy(obs_dict[agent]))
                frames[agent].append(int(info.get("bridge_frame", 0)))
                raw_time_since_seen[agent].append(float(info.get("time_since_ball_seen", 0.0)))
                raw_time_since_disappeared[agent].append(
                    float(info.get("raw_obs", {}).get("time_since_ball_disappeared", 0.0))
                )
                skill = info.get("executed_skill_name") or info.get("policy_skill_name") or "unknown"
                skill_counts[agent][skill] += 1

        if post_warmup:
            collected_steps += 1

        if (step + 1) % 30 == 0:
            elapsed = time.perf_counter() - (t_start or t0)
            if collected_steps > 0 and t_start is not None:
                sps = collected_steps / elapsed
                print(f"  step {step + 1}/{DIAG_STEPS}  {sps:.1f} env-steps/sec  ({collected_steps} scored)", flush=True)
            else:
                print(f"  step {step + 1}/{DIAG_STEPS}  (warmup)", flush=True)

    t_end = time.perf_counter()
    env.close()
    runtime.close()

    # ── throughput ───────────────────────────────────────────────────────────
    n = collected_steps
    if n <= 0:
        raise RuntimeError("diag_n7_slots collected zero scored steps after warmup")

    wall_time_s = (t_end - t_start) if t_start is not None else float("nan")
    env_steps_per_sec = n / wall_time_s if wall_time_s > 0 else float("nan")
    transitions_per_sec = env_steps_per_sec * len(PLAYER_NOS)
    ms_per_step = 1000.0 * wall_time_s / n if n > 0 else float("nan")

    print("\n" + "=" * 70)
    print("THROUGHPUT (8-slot vs 6-slot baseline)")
    print("=" * 70)
    print(f"  scored env-steps   : {n}")
    print(f"  wall time (scored) : {wall_time_s:.2f}s")
    print(f"  env-steps/sec      : {env_steps_per_sec:.1f}   (6-slot baseline ~30.3)")
    print(f"  ms/env-step        : {ms_per_step:.1f}   (should stay ≈33ms = 1 camera frame)")
    print(f"  transitions/sec    : {transitions_per_sec:.1f}   (6-slot baseline ~182)")
    print(f"  transitions ratio  : {transitions_per_sec / 182.0:.2f}×   (expected ≈1.17×)")

    # ── wire sanity ──────────────────────────────────────────────────────────
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

    # ── obs-space stats ──────────────────────────────────────────────────────
    obs_arrs = {a: np.stack(obs_buf[a]) for a in agents}
    feat_names = [o.name for o in ObsIndex]

    print("\n" + "=" * 70)
    print("OBS-SPACE STATS (mean ± std)")
    print("=" * 70)
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
            row += f"  {mu:+.3f}/{sigma:.3f}"
            stds.append(sigma)
        if all(s < 0.005 for s in stds):
            any_frozen.append(name)
        print(row)

    # ── frame advancement ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FRAME ADVANCEMENT PER SLOT")
    print("=" * 70)
    for agent, p in zip(agents, PLAYER_NOS):
        f = frames[agent]
        delta = f[-1] - f[0] if len(f) > 1 else 0
        print(f"  player{p}: first={f[0]}  last={f[-1]}  delta={delta}ms")

    # ── pass/fail ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PASS / FAIL SUMMARY")
    print("=" * 70)
    failures = []
    warnings = []
    per_slot_rows: list[dict] = []

    for agent, p in zip(agents, PLAYER_NOS):
        r = {k: counts[agent][k] / n for k in wire_keys}
        per_slot_rows.append({
            "player_number": p,
            "obs_ready_rate": float(r["bridge_obs_ready"]),
            "synthetic_pose_rate": float(r["synthetic_pose_observation"]),
            "synthetic_ball_rate": float(r["synthetic_ball_observation"]),
            "ball_export_fresh_rate": float(r["ball_export_fresh"]),
            "pose_alignment_warning_rate": float(r["pose_alignment_warning"]),
            "ball_stale_rate": float(r["ball_observation_stale"]),
            "ball_collapsed_rate": float(r["ball_observation_collapsed"]),
            "frame_delta_ms": int(frames[agent][-1] - frames[agent][0]) if len(frames[agent]) > 1 else 0,
        })
        if not r["bridge_obs_ready"] > 0.99:
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
    dynamic_expected_frozen: set[str] = set()
    no_stale_ball = all(counts[agent]["ball_observation_stale"] == 0 for agent in agents)
    if "time_since_ball_seen" in any_frozen:
        raw_seen_ranges = [
            max(raw_time_since_seen[agent]) - min(raw_time_since_seen[agent])
            for agent in agents if raw_time_since_seen[agent]
        ]
        raw_seen_max = max(
            (v for agent in agents for v in raw_time_since_seen[agent]), default=0.0
        )
        if raw_seen_ranges and all(d >= 10.0 for d in raw_seen_ranges):
            dynamic_expected_frozen.add("time_since_ball_seen")
        elif no_stale_ball and raw_seen_max <= 40.0:
            dynamic_expected_frozen.add("time_since_ball_seen")
    if "time_since_ball_disappeared" in any_frozen:
        raw_dis_all = [v for agent in agents for v in raw_time_since_disappeared[agent]]
        if no_stale_ball and _all_close_to_zero(raw_dis_all):
            dynamic_expected_frozen.add("time_since_ball_disappeared")

    unexpected_frozen = [f for f in any_frozen if f not in expected_frozen and f not in dynamic_expected_frozen]
    if unexpected_frozen:
        warnings.append(f"Unexpected frozen features (all slots, all steps): {unexpected_frozen}")

    if not failures and not warnings:
        print("\n  RESULT: PASS — all 7 slots healthy\n")
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

    summary = {
        "timestamp": timestamp,
        "n_slots": len(PLAYER_NOS),
        "diag_steps": DIAG_STEPS,
        "warmup_steps_skipped": DIAG_WARMUP_STEPS,
        "scored_steps": n,
        "terminated_early": terminated_early,
        "wall_time_s": round(wall_time_s, 3),
        "env_steps_per_sec": round(env_steps_per_sec, 2),
        "ms_per_step": round(ms_per_step, 2),
        "transitions_per_sec": round(transitions_per_sec, 2),
        "transitions_ratio_vs_6slot": round(transitions_per_sec / 182.0, 3),  # 7/6 ≈ 1.17×
        "player_numbers": list(PLAYER_NOS),
        "result": "FAIL" if failures else "WARN" if warnings else "PASS",
        "failures": failures,
        "warnings": warnings,
        "unexpected_frozen_features": unexpected_frozen,
        "any_nan": any_nan,
        "output_dir": str(output_dir),
    }

    _write_csv(output_dir / "per_slot_summary.csv", per_slot_rows,
               fieldnames=list(per_slot_rows[0].keys()) if per_slot_rows else ["player_number"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"[diag_n7_slots] output_dir={output_dir}", flush=True)
    print(f"[diag_n7_slots] env_steps_per_sec={env_steps_per_sec:.1f}  transitions_per_sec={transitions_per_sec:.1f}", flush=True)


if __name__ == "__main__":
    main()
