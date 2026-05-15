"""Per-slot sanity diagnostic for N=6 visible parallel training.

Runs DIAG_STEPS training-style steps with the full production config
(LP_NUM_THREADS=4, simrobot_visible, 3d_visible, obs_mode=corrected)
and reports wire metrics, obs-space stats, and action diversity for
EVERY slot independently.  Pass when all 6 slots look healthy.
"""
from __future__ import annotations

import os
import sys
import math
from collections import defaultdict
from dataclasses import replace
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
from visible_parallel_runtime import VisibleParallelRuntime

DIAG_STEPS   = 120   # ~5-6 seconds sim time per slot at 22 joint-steps/s
PLAYER_NOS   = (1, 2, 3, 4, 5, 6)
SHM_NAME     = "/pybh_rl_shared_state_v3"
DISPLAY_BASE = 410

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


def main() -> None:
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
    skill_counts: Dict[str, Dict[str, int]] = {a: defaultdict(int) for a in agents}

    obs_dict, _ = env.reset()
    dummy_action = torch.zeros(1, 5)

    print(f"\n[diag] running {DIAG_STEPS} steps across {len(agents)} slots …\n", flush=True)

    for step in range(DIAG_STEPS):
        actions = {a: dummy_action for a in agents}
        obs_dict, _, _, _, infos = env.step(actions)

        for agent in agents:
            info = infos[agent]

            for k in wire_keys:
                if info.get(k):
                    counts[agent][k] += 1

            obs_buf[agent].append(_obs_to_numpy(obs_dict[agent]))
            frames[agent].append(int(info.get("bridge_frame", 0)))

            skill = info.get("executed_skill_name") or info.get("policy_skill_name") or "unknown"
            skill_counts[agent][skill] += 1

        if (step + 1) % 30 == 0:
            print(f"  step {step + 1}/{DIAG_STEPS}", flush=True)

    env.close()
    runtime.close()

    # ── report ────────────────────────────────────────────────────────────────
    n = DIAG_STEPS
    print("\n" + "=" * 70)
    print("PER-SLOT WIRE SANITY")
    print("=" * 70)
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

    for agent, p in zip(agents, PLAYER_NOS):
        r = {k: counts[agent][k] / n for k in wire_keys}

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
    unexpected_frozen = [f for f in any_frozen if f not in expected_frozen]
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


if __name__ == "__main__":
    main()
