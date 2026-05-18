"""
ci=30 production verification.

Phase 1 — Speed: runs N_SPEED_STEPS with all 6 slots via ParallelSlotSimRobotEnv
           and reports observed steps/sec (target ≥ 25).
Phase 2 — Obs/action sanity: runs N_OBS_STEPS on players 1 and 4 via single-slot
           RealSimRobotEnv and checks per-feature bounds, ball/pose health, and
           action space shape.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np

from config import ENV
from environment import RealSimRobotEnv, _ball_observation_masks
from heuristic_agent_experiment import _heuristic_action
from parallel_environment import ParallelSlotSimRobotEnv
from visible_parallel_runtime import VisibleParallelRuntime

N_SPEED_WARMUP = 10
N_SPEED_STEPS  = 100
N_OBS_WARMUP   = 20
N_OBS_STEPS    = 300

SCENE_NAME   = "RL1v0_3D_visible"
DISPLAY_BASE = 410
PROFILE      = "3d_visible"

BOUNDS = [
    ("robot_x",                         -1.1,  1.1),
    ("robot_y",                         -1.1,  1.1),
    ("robot_theta",                     -1.05, 1.05),
    ("ball_rel_x",                      -2.5,  2.5),
    ("ball_rel_y",                      -2.5,  2.5),
    ("ball_end_rel_x",                  -2.5,  2.5),
    ("ball_end_rel_y",                  -2.5,  2.5),
    ("ball_vel_x",                      -1.5,  1.5),
    ("ball_vel_y",                      -1.5,  1.5),
    ("time_since_ball_seen",             0.0,  1.01),
    ("time_since_ball_disappeared",      0.0,  1.01),
    ("ball_seen_percentage",             0.0,  1.01),
    ("ball_consistent_with_game_state", -0.01, 1.01),
    ("can_score_now",                   -0.01, 1.01),
    ("shot_quality_no_obstacles",       -0.01, 1.01),
    ("shot_opening_with_obstacles",     -0.01, 1.01),
    ("pass_options_count",              -0.01, 1.01),
    ("nearest_teammate_dist",           -0.01, 1.01),
    ("nearest_opponent_dist",           -0.01, 1.01),
    ("nearest_uncertain_obstacle_dist", -0.01, 1.01),
    ("nearest_teammate_front_dist",     -0.01, 1.01),
    ("nearest_opponent_front_dist",     -0.01, 1.01),
    ("nearest_uncertain_front_dist",    -0.01, 1.01),
]


def _make_backend(player: int, launch: bool = False):
    return replace(
        ENV.backend,
        mode="simrobot_visible",
        player_number=player,
        simrobot_launch=launch,
        simrobot_profile=PROFILE,
    )


# ── Phase 1: parallel step rate ───────────────────────────────────────────────

def measure_speed() -> float:
    env_cfg = replace(
        ENV,
        backend=replace(ENV.backend, mode="simrobot_visible",
                        simrobot_launch=False, simrobot_profile=PROFILE),
        parallel_player_numbers=(1, 2, 3, 4, 5, 6),
    )
    env = ParallelSlotSimRobotEnv(env_cfg=env_cfg)
    walk = {f"player_{p}": np.array([0.0, 1000.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for p in (1, 2, 3, 4, 5, 6)}

    env.reset()
    for _ in range(N_SPEED_WARMUP):
        obs, rew, term, trunc, info = env.step(walk)
        if not env.agents:
            env.reset()

    t0 = time.perf_counter()
    steps = 0
    for _ in range(N_SPEED_STEPS):
        env.step(walk)
        if not env.agents:
            env.reset()
        steps += 1
    elapsed = time.perf_counter() - t0

    env.close()
    return steps / elapsed


# ── Phase 2: single-slot obs/action sanity ────────────────────────────────────

def _heuristic_flat(raw) -> np.ndarray:
    """Walk toward ball using the heuristic geometry. Always emits walk (compact idx 0)."""
    robot_pos = np.array([raw.robot_x, raw.robot_y], dtype=np.float32)
    ball_pos  = np.array([raw.ball_x,  raw.ball_y],  dtype=np.float32)
    action = _heuristic_action(robot_pos, ball_pos)
    # Map any non-walk skill (shoot) back to walk so compact idx stays 0
    if action.skill_name != "walk":
        theta = float(np.arctan2(ball_pos[1] - robot_pos[1], ball_pos[0] - robot_pos[0]))
        return np.array([0.0, ball_pos[0], ball_pos[1], theta, 0.0], dtype=np.float32)
    return np.array([0.0, action.target_x, action.target_y, action.target_theta, 0.0],
                    dtype=np.float32)


def obs_sanity(player: int) -> dict:
    cfg = replace(ENV, backend=_make_backend(player),
                  parallel_player_numbers=(player,))
    env = RealSimRobotEnv(env_cfg=cfg)

    obs_stack = []
    ball_trusted = ball_visible = 0
    synthetic_ball_obs = corrected_exported = synthetic_any = 0
    natural_ball_visible = 0
    corrected_mode = pose_warn = 0
    synthetic_pose = 0
    corrected_pose = 0
    ball_seen_pcts = []
    natural_ball_seen_pcts = []
    natural_times = []
    # "facing but not seeing" — ball in FOV by ground-truth but BHuman didn't detect it
    # NAO camera: ~60° horizontal FOV → ±30° half-angle; reliable detection < 2500 mm.
    # ball_rel_x/y come from the corrected obs (SimRobot GT when synthetic) so they
    # reflect true ball position even when BHuman's own model has lost the ball.
    fov_steps = 0          # steps where ball is inside estimated camera FOV
    fov_not_seen_steps = 0 # of those, BHuman natural_ball_seen_pct == 0

    env.reset()
    raw = env._last_backend_obs
    for i in range(N_OBS_WARMUP + N_OBS_STEPS):
        act = _heuristic_flat(raw)
        obs_vec, _, done, _, _ = env.step({"player_1": act})
        raw = env._last_backend_obs
        if done["player_1"]:
            env.reset()
            raw = env._last_backend_obs
            continue
        if i < N_OBS_WARMUP:
            continue

        obs_stack.append(obs_vec["player_1"])
        masks = _ball_observation_masks(raw)
        if masks["ball_trusted_mask"]: ball_trusted += 1
        if masks["ball_visible_mask"]: ball_visible += 1
        if raw.synthetic_ball_observation:   synthetic_ball_obs += 1
        if raw.corrected_exported_ball:      corrected_exported += 1
        if raw.synthetic_ball_observation or raw.corrected_exported_ball:
            synthetic_any += 1
        if raw.obs_export_mode == 1:                     corrected_mode += 1
        if raw.pose_alignment_warning:                   pose_warn += 1
        if raw.synthetic_pose_observation:               synthetic_pose += 1
        if raw.corrected_exported_robot_pose:            corrected_pose += 1
        nat_pct = float(raw.natural_ball_seen_percentage)
        if nat_pct > 0:                                  natural_ball_visible += 1
        ball_seen_pcts.append(float(raw.ball_seen_percentage))
        natural_ball_seen_pcts.append(nat_pct)
        natural_times.append(float(raw.natural_time_since_ball_seen))
        # FOV check: ball_rel_x/y are in robot frame from corrected (GT) obs.
        # Ball is "in FOV" when it's in the forward half-plane, within 2500mm,
        # and within the NAO camera's ~±30° horizontal half-angle.
        bx, by = float(raw.ball_rel_x), float(raw.ball_rel_y)
        ball_dist = math.hypot(bx, by)
        if bx > 100.0 and ball_dist < 2500.0 and abs(math.atan2(by, bx)) < math.radians(30):
            fov_steps += 1
            if nat_pct == 0.0:
                fov_not_seen_steps += 1

    env.close()
    arr = np.stack(obs_stack)
    n = max(1, len(obs_stack))
    return {
        "player": player, "n": n,
        "obs_mean": arr.mean(0), "obs_min": arr.min(0), "obs_max": arr.max(0),
        # ball
        "ball_trusted_rate":               ball_trusted / n,
        "ball_visible_rate":               ball_visible / n,          # corrected obs
        "natural_ball_visible_rate":       natural_ball_visible / n,  # BHuman saw it
        "synthetic_ball_obs_rate":         synthetic_ball_obs / n,
        "corrected_exported_ball_rate":    corrected_exported / n,
        "synthetic_any_rate":              synthetic_any / n,
        "ball_seen_pct_mean":              float(np.mean(ball_seen_pcts)) if ball_seen_pcts else 0.0,
        "natural_ball_seen_pct_mean":      float(np.mean(natural_ball_seen_pcts)) if natural_ball_seen_pcts else 0.0,
        "natural_time_mean_ms":            float(np.mean(natural_times)) if natural_times else 0.0,
        "natural_time_p95_ms":             float(np.percentile(natural_times, 95)) if natural_times else 0.0,
        # FOV diagnostic: distinguishes agent vs environment as root cause of low confidence
        # High → environment/wiring problem (BHuman not seeing ball that IS in camera FOV)
        # Low  → agent behavior (robot looking away, normal policy-dependent variation)
        "fov_steps":                       fov_steps,
        "fov_not_seen_steps":              fov_not_seen_steps,
        "fov_perception_failure_rate":     fov_not_seen_steps / max(1, fov_steps),
        # pose
        "corrected_mode_rate":             corrected_mode / n,
        "pose_warn_rate":                  pose_warn / n,
        "synthetic_pose_rate":             synthetic_pose / n,        # ground-truth fallback
        "corrected_exported_pose_rate":    corrected_pose / n,        # inter-frame corrected
    }


def action_space_info() -> None:
    cfg = replace(ENV, backend=_make_backend(1), parallel_player_numbers=(1,))
    env = RealSimRobotEnv(env_cfg=cfg)
    env.reset()
    asp = env.action_spaces["player_1"]
    print(f"  shape  : {asp.shape}", flush=True)
    print(f"  low    : {asp.low}", flush=True)
    print(f"  high   : {asp.high}", flush=True)
    print(f"  skills : {env._compact_skill_map}", flush=True)
    env.close()


def main():
    print(f"\n{'='*62}", flush=True)
    print(f"ci=30 PRODUCTION VERIFICATION  (scene={SCENE_NAME})", flush=True)
    print(f"{'='*62}\n", flush=True)

    runtime = VisibleParallelRuntime(
        player_numbers=(1, 2, 3, 4, 5, 6),
        shm_name="/pybh_rl_shared_state_v3",
        display_base=DISPLAY_BASE,
        scene_name=SCENE_NAME,
    )
    runtime.start()
    print("Slots started, settling 6s…", flush=True)
    time.sleep(6)

    try:
        # ── 1. Obs bounds (before speed test — slots in clean reset state) ────
        print(f"\n{'─'*62}", flush=True)
        print("PHASE 1 — OBS SANITY (players 1 and 4)", flush=True)
        results = []
        for p in (1, 4):
            print(f"  running player {p}…", flush=True)
            results.append(obs_sanity(p))

        # ── 2. Speed (after obs sanity, slots already warmed up) ─────────────
        print(f"\n{'─'*62}", flush=True)
        print("PHASE 2 — PARALLEL STEP RATE (6 slots, ThreadPoolExecutor)", flush=True)
        print(f"{'─'*62}", flush=True)
        rate = measure_speed()
        speed_ok = rate >= 25.0
        print(f"  Observed: {rate:.1f} steps/sec   target ≥ 25  {'✓' if speed_ok else '✗ FAIL'}", flush=True)

        print(f"\n{'─'*62}", flush=True)
        print(f"{'FEATURE':<38} {'MEAN':>7} {'MIN':>7} {'MAX':>7}  PASS", flush=True)
        print(f"{'─'*62}", flush=True)
        any_fail = False
        for ri in results:
            for i, (name, lo, hi) in enumerate(BOUNDS):
                mn = ri["obs_mean"][i]
                lo_ = ri["obs_min"][i]
                hi_ = ri["obs_max"][i]
                ok = (lo_ >= lo) and (hi_ <= hi)
                if not ok:
                    any_fail = True
                    print(
                        f"[p={ri['player']}] {name:<36} {mn:>7.3f} {lo_:>7.3f} {hi_:>7.3f}"
                        f"  ✗ FAIL  expect [{lo}, {hi}]",
                        flush=True,
                    )
        if not any_fail:
            print("  All feature bounds: PASS", flush=True)

        # ── 3. Ball/pose health ───────────────────────────────────────────────
        print(f"\n{'─'*62}", flush=True)
        print(f"{'METRIC':<38} {'p=1':>8} {'p=4':>8}  THRESHOLD", flush=True)
        print(f"{'─'*62}", flush=True)

        def row(label, key, thresh=None, cmp=">=", note=""):
            vals = [r[key] for r in results]
            v1, v4 = vals[0], vals[1]
            if thresh is not None:
                ok = all((v >= thresh if cmp == ">=" else v <= thresh) for v in vals)
                flag = "✓" if ok else "✗ FAIL"
                if not ok:
                    health_pass[0] = False
                print(f"  {label:<40} {v1:>7.3f} {v4:>7.3f}  {cmp}{thresh}  {flag}", flush=True)
            else:
                suffix = f"  ({note})" if note else ""
                print(f"  {label:<40} {v1:>7.3f} {v4:>7.3f}{suffix}", flush=True)

        health_pass = [True]
        print(f"  {'METRIC':<40} {'p=1':>7} {'p=4':>7}", flush=True)
        print(f"  {'─'*58}", flush=True)
        print(f"  BALL", flush=True)
        row("  corrected ball_visible_rate",      "ball_visible_rate",            0.80, ">=")
        row("  natural ball_visible_rate",        "natural_ball_visible_rate",    note="policy-dep")
        row("  ball_seen_pct corrected mean",     "ball_seen_pct_mean")
        row("  ball_seen_pct natural mean",       "natural_ball_seen_pct_mean",   note="policy-dep")
        row("  synthetic_ball_observation rate",  "synthetic_ball_obs_rate",      note="SimRobot GT fallback")
        row("  corrected_exported_ball rate",     "corrected_exported_ball_rate", note="inter-frame correction")
        row("  natural_time_since_seen mean ms",  "natural_time_mean_ms",         note="policy-dep")
        row("  natural_time_since_seen p95  ms",  "natural_time_p95_ms",          note="policy-dep")
        row("  ball_trusted_rate",                "ball_trusted_rate",            note="policy-dep; training: 0.59-0.75")
        print(f"  FOV DIAGNOSTIC  (ball in camera FOV by GT, did BHuman see it?)", flush=True)
        for r in results:
            p = r["player"]
            fov  = r["fov_steps"]
            fail = r["fov_not_seen_steps"]
            rate_ = r["fov_perception_failure_rate"]
            flag = "✓ env OK" if rate_ < 0.15 else "✗ env PROBLEM"
            print(
                f"    p={p}: fov_steps={fov:4d}  not_seen={fail:4d}"
                f"  failure_rate={rate_:.3f}  → {flag}",
                flush=True,
            )
            if rate_ >= 0.15:
                health_pass[0] = False
        print(f"  POSE  (training baseline: pose_warn=0.0, synthetic_pose=0.0)", flush=True)
        row("  corrected_mode_rate",              "corrected_mode_rate",          0.99, ">=")
        row("  pose_warn_rate",                   "pose_warn_rate",               note="policy-dep; training=0.0")
        row("  synthetic_pose_observation rate",  "synthetic_pose_rate",          note="policy-dep; training=0.0 (SelfLocator active)")
        row("  corrected_exported_pose rate",     "corrected_exported_pose_rate", note="policy-dep; training=0.0")

        # ── 4. Action space ───────────────────────────────────────────────────
        print(f"\n{'─'*62}", flush=True)
        print("ACTION SPACE", flush=True)
        action_space_info()

        # ── 5. Summary ────────────────────────────────────────────────────────
        print(f"\n{'='*62}", flush=True)
        print("SUMMARY", flush=True)
        print(f"  Step rate      : {'PASS' if speed_ok else 'FAIL'}  ({rate:.1f} steps/sec, target ≥ 25)", flush=True)
        print(f"  Feature bounds : {'PASS' if not any_fail else 'FAIL'}", flush=True)
        print(f"  Ball/pose health: {'PASS' if health_pass[0] else 'FAIL'}", flush=True)
        print(f"{'='*62}\n", flush=True)

    finally:
        runtime.close()


if __name__ == "__main__":
    main()
