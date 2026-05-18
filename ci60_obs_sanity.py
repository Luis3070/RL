"""
ci=60 obs/action space sanity check.

Launches 6 visible ci=60 SimRobot slots, runs N_STEPS with a walk action,
and reports per-feature statistics for pose, ball, and trust metrics.
Compares key bounds against known ci=30 baselines.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from config import ENV
from environment import RealSimRobotEnv
from visible_parallel_runtime import VisibleParallelRuntime

N_STEPS   = 300   # per slot
N_WARMUP  = 20
DISPLAY_BASE = 420
SCENE_NAME   = "RL1v0_3D_visible_ci60"

# ── expected ranges for each obs feature (ci-agnostic) ──────────────────────
# Format: (name, lo, hi, note)
BOUNDS = [
    # pose — normalized
    ("robot_x",                   -1.1,  1.1,  "field x in [-4500,4500]mm"),
    ("robot_y",                   -1.1,  1.1,  "field y in [-3000,3000]mm"),
    ("robot_theta",               -1.05, 1.05, "heading / pi"),
    # ball relative — may be large when ball unseen (synthetic / stale)
    ("ball_rel_x",               -2.5,  2.5,  "relative ball x"),
    ("ball_rel_y",               -2.5,  2.5,  "relative ball y"),
    ("ball_end_rel_x",           -2.5,  2.5,  "predicted end rel x"),
    ("ball_end_rel_y",           -2.5,  2.5,  "predicted end rel y"),
    # ball vel — normalised by 3000 mm/s
    ("ball_vel_x",               -1.5,  1.5,  "ball vx / 3000"),
    ("ball_vel_y",               -1.5,  1.5,  "ball vy / 3000"),
    # time features — normalised to [0,1] by 5000ms
    ("time_since_ball_seen",      0.0,  1.01, "0=just seen, 1=5s+ ago"),
    ("time_since_ball_disappeared",0.0, 1.01, "0=just gone"),
    ("ball_seen_percentage",       0.0,  1.01, "seen% / 100"),
    # binary/quality
    ("ball_consistent_with_game_state", -0.01, 1.01, "0 or 1"),
    ("can_score_now",              -0.01, 1.01, "0 or 1"),
    ("shot_quality_no_obstacles",  -0.01, 1.01, "0..1"),
    ("shot_opening_with_obstacles",-0.01, 1.01, "0..1"),
    ("pass_options_count",         -0.01, 1.01, "/ 5, clipped"),
    # distances — normalised to [0,1] by field half-x
    ("nearest_teammate_dist",      -0.01, 1.01, "0..1"),
    ("nearest_opponent_dist",      -0.01, 1.01, "0..1"),
    ("nearest_uncertain_obstacle_dist", -0.01, 1.01, "0..1"),
    ("nearest_teammate_front_dist",-0.01, 1.01, "0..1"),
    ("nearest_opponent_front_dist",-0.01, 1.01, "0..1"),
    ("nearest_uncertain_front_dist",-0.01, 1.01, "0..1"),
]

# Ball/pose health thresholds (from ci=30 baselines)
BALL_TRUSTED_RATE_MIN  = 0.50   # fraction of steps where ball is trusted
BALL_VISIBLE_RATE_MIN  = 0.80
SYNTHETIC_BALL_MAX     = 0.40
OBS_MODE_CORRECTED_MIN = 0.99   # all steps should use corrected mode


def run_slot(player_number: int) -> dict:
    """Run N_STEPS on a single slot and collect statistics."""
    from config import ENV, BackendConfig
    backend_cfg = replace(
        ENV.backend,
        mode="simrobot_visible",
        player_number=player_number,
        simrobot_launch=False,
        simrobot_profile="3d_visible_ci60",
    )
    slot_cfg = replace(ENV, backend=backend_cfg, parallel_player_numbers=(player_number,))
    env = RealSimRobotEnv(env_cfg=slot_cfg)

    obs_stack   = []
    ball_trusted = 0
    ball_visible = 0
    synthetic    = 0
    synthetic_ball_obs = 0
    corrected_exported_ball = 0
    corrected_mode = 0
    pose_warn    = 0
    ball_seen_pcts = []
    natural_time_since_seen = []

    env.reset()
    # walk action: skill=0 (walk), targets at ball position
    walk_action = np.array([0.0, 1000.0, 0.0, 0.0, 0.0], dtype=np.float32)

    for i in range(N_WARMUP + N_STEPS):
        result = env.step({"player_1": walk_action})
        obs_vec, _, done, _, info = result
        obs_np = obs_vec["player_1"]
        raw = env._last_backend_obs

        if done["player_1"]:
            env.reset()
            continue

        if i < N_WARMUP:
            continue

        obs_stack.append(obs_np)

        # ball trust
        from environment import _ball_observation_masks, BALL_TRUSTED_MAX_AGE_MS
        masks = _ball_observation_masks(raw)
        if masks["ball_trusted_mask"]:   ball_trusted += 1
        if masks["ball_visible_mask"]:   ball_visible += 1

        if raw.synthetic_ball_observation:   synthetic_ball_obs += 1
        if raw.corrected_exported_ball:      corrected_exported_ball += 1
        if raw.synthetic_ball_observation or raw.corrected_exported_ball:
            synthetic += 1
        if raw.obs_export_mode == 1:     # 1 = corrected
            corrected_mode += 1
        if raw.pose_alignment_warning:   pose_warn += 1
        ball_seen_pcts.append(float(raw.ball_seen_percentage))
        natural_time_since_seen.append(float(raw.natural_time_since_ball_seen))

    env.close()

    arr = np.stack(obs_stack, axis=0)   # (N_STEPS, OBS_SIZE)
    n = max(1, len(obs_stack))
    return {
        "player":          player_number,
        "n_steps":         n,
        "obs_mean":        arr.mean(axis=0),
        "obs_min":         arr.min(axis=0),
        "obs_max":         arr.max(axis=0),
        "obs_std":         arr.std(axis=0),
        "ball_trusted_rate":           ball_trusted / n,
        "ball_visible_rate":           ball_visible / n,
        "synthetic_rate":              synthetic / n,
        "synthetic_ball_obs_rate":     synthetic_ball_obs / n,
        "corrected_exported_ball_rate": corrected_exported_ball / n,
        "corrected_mode_rate":         corrected_mode / n,
        "pose_warn_rate":              pose_warn / n,
        "ball_seen_pct_mean":          float(np.mean(ball_seen_pcts)) if ball_seen_pcts else 0.0,
        "ball_seen_pct_max":           float(np.max(ball_seen_pcts))  if ball_seen_pcts else 0.0,
        "natural_time_since_seen_mean": float(np.mean(natural_time_since_seen)) if natural_time_since_seen else 0.0,
        "natural_time_since_seen_p95":  float(np.percentile(natural_time_since_seen, 95)) if natural_time_since_seen else 0.0,
        "natural_time_since_seen_max":  float(np.max(natural_time_since_seen))  if natural_time_since_seen else 0.0,
    }


def main():
    print(f"\n{'='*60}", flush=True)
    print(f"ci=60 OBS / ACTION SPACE SANITY CHECK", flush=True)
    print(f"Scene: {SCENE_NAME}  Steps/slot: {N_STEPS}", flush=True)
    print(f"{'='*60}\n", flush=True)

    runtime = VisibleParallelRuntime(
        player_numbers=(1, 2, 3, 4, 5, 6),
        shm_name="/pybh_rl_shared_state_v3",
        display_base=DISPLAY_BASE,
        scene_name=SCENE_NAME,
    )
    runtime.start()
    print("Slots started, waiting 6s for SimRobot to settle…", flush=True)
    time.sleep(6)

    try:
        # Run one slot (player 1) for detailed obs stats; check a second (player 4)
        # to catch per-slot divergence.
        results = []
        for p in (1, 4):
            print(f"\n--- Player {p} ---", flush=True)
            r = run_slot(p)
            results.append(r)

        # ── 1. Per-feature bounds check ───────────────────────────────────────
        print(f"\n{'─'*60}", flush=True)
        print(f"{'FEATURE':<35} {'MEAN':>7} {'MIN':>7} {'MAX':>7}  PASS", flush=True)
        print(f"{'─'*60}", flush=True)
        any_fail = False
        for ri in results:
            for i, (name, lo, hi, note) in enumerate(BOUNDS):
                mn = ri["obs_mean"][i]
                lo_ = ri["obs_min"][i]
                hi_ = ri["obs_max"][i]
                ok = (lo_ >= lo) and (hi_ <= hi)
                flag = "✓" if ok else "✗ FAIL"
                if not ok:
                    any_fail = True
                    print(
                        f"[p={ri['player']}] {name:<33} {mn:>7.3f} {lo_:>7.3f} {hi_:>7.3f}  {flag}  expect [{lo}, {hi}]",
                        flush=True,
                    )

        if not any_fail:
            print("All feature bounds: PASS", flush=True)

        # ── 2. Ball / pose health ─────────────────────────────────────────────
        print(f"\n{'─'*60}", flush=True)
        print(f"BALL / POSE HEALTH (ci=60 vs ci=30 baseline)", flush=True)
        print(f"{'─'*60}", flush=True)
        header = f"  {'METRIC':<35} {'p=1':>8} {'p=4':>8}  THRESHOLD"
        print(header, flush=True)

        metrics = [
            ("ball_trusted_rate",             BALL_TRUSTED_RATE_MIN,  ">="),
            ("ball_visible_rate",             BALL_VISIBLE_RATE_MIN,  ">="),
            ("synthetic_rate",                SYNTHETIC_BALL_MAX,     "<="),
            ("synthetic_ball_obs_rate",       None,                   "info"),
            ("corrected_exported_ball_rate",  None,                   "info"),
            ("corrected_mode_rate",           OBS_MODE_CORRECTED_MIN, ">="),
            ("pose_warn_rate",                0.10,                   "<="),
            ("ball_seen_pct_mean",            None,                   "info"),
            ("ball_seen_pct_max",             None,                   "info"),
            ("natural_time_since_seen_mean",  None,                   "info"),
            ("natural_time_since_seen_p95",   None,                   "info"),
            ("natural_time_since_seen_max",   None,                   "info"),
        ]
        health_pass = True
        for key, thresh, cmp in metrics:
            vals = [r[key] for r in results]
            v1, v4 = vals
            if thresh is not None:
                if cmp == ">=":
                    ok = all(v >= thresh for v in vals)
                else:
                    ok = all(v <= thresh for v in vals)
                flag = "✓" if ok else "✗ FAIL"
                if not ok:
                    health_pass = False
            else:
                flag = "–"
            print(f"  {key:<35} {v1:>8.3f} {v4:>8.3f}  {cmp} {thresh if thresh is not None else ''}  {flag}", flush=True)

        # ── 3. Action space ───────────────────────────────────────────────────
        print(f"\n{'─'*60}", flush=True)
        print("ACTION SPACE (unchanged from ci=30)", flush=True)
        r0 = results[0]
        backend_cfg = replace(ENV.backend, mode="simrobot_visible", player_number=1,
                              simrobot_launch=False, simrobot_profile="3d_visible_ci60")
        slot_cfg = replace(ENV, backend=backend_cfg, parallel_player_numbers=(1,))
        env_tmp = RealSimRobotEnv(env_cfg=slot_cfg)
        env_tmp.reset()
        asp = env_tmp.action_spaces["player_1"]
        print(f"  shape : {asp.shape}", flush=True)
        print(f"  low   : {asp.low}", flush=True)
        print(f"  high  : {asp.high}", flush=True)
        print(f"  skills: {env_tmp._compact_skill_map}  (0=walk,1=dribble,2=shoot)", flush=True)
        env_tmp.close()

        # ── 4. Summary ────────────────────────────────────────────────────────
        print(f"\n{'='*60}", flush=True)
        print("SUMMARY", flush=True)
        print(f"  Feature bounds : {'PASS' if not any_fail else 'FAIL'}", flush=True)
        print(f"  Ball/pose health: {'PASS' if health_pass else 'FAIL'}", flush=True)
        print(f"  ci=60 note: ball_seen_percentage window is ~1020ms (vs ~2000ms at ci=30)", flush=True)
        print(f"              ball_trusted uses natural_time_since_ball_seen ≤600ms → unaffected", flush=True)
        print(f"{'='*60}\n", flush=True)

    finally:
        runtime.close()


if __name__ == "__main__":
    main()
