"""Training loop phase timing benchmark (Option 1 prerequisite).

Runs N_ROLLOUTS full rollouts of the actual training infrastructure
(same env config, same MAPPO agent as train.py) and measures:

  T_collect  — time spent inside wrapped_env.step() per rollout
               (pure SimRobot IPC: the 6-slot parallel wait)
  T_update   — time spent inside agent.post_interaction() at the rollout
               boundary (the PPO gradient update)
  T_reset    — time spent inside wrapped_env.reset() when episodes end
  T_other    — everything else (act(), record_transition(), pre_interaction(),
               Python bookkeeping)

From these we compute:
  efficiency = T_collect / (T_collect + T_update + T_reset + T_other)
  async_speedup_upper_bound = 1 / (1 - T_update / T_total)
    (if we perfectly overlap T_update with T_collect for the next rollout)

Usage:
  cd /home/luis/GitHub/RL
  python bench_timing.py

Env vars honoured:
  PYBH_RL_SHM_NAME   — SHM name (default: /pybh_rl_bench_timing)
  RL_VISIBLE_DISPLAY_BASE — Xvfb display base (default: 430, uses :431-:436)
  BENCH_N_ROLLOUTS   — number of scored rollouts (default: 6)
  BENCH_ROLLOUT_STEPS — steps per rollout (default: matches TRAIN.rollout_steps)
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SabanaHerons2026", "Make", "Python"))

from config import ENV, TRAIN, ROOT_DIR
from visible_parallel_runtime import VisibleParallelRuntime
from train import build_env, build_mappo, make_env_config, _move_to_device, _agent_state_from_env

RUNS_DIR = ROOT_DIR / "runs" / "bench_timing"

SHM_NAME     = os.environ.get("PYBH_RL_SHM_NAME", "/pybh_rl_bench_timing")
DISPLAY_BASE = int(os.environ.get("RL_VISIBLE_DISPLAY_BASE", "430"))
N_ROLLOUTS   = int(os.environ.get("BENCH_N_ROLLOUTS", "6"))


def main() -> None:
    rollout_steps = int(os.environ.get("BENCH_ROLLOUT_STEPS", str(TRAIN.rollout_steps)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RUNS_DIR / f"bench_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PYBH_RL_SHM_NAME"] = SHM_NAME
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # RL_PARALLEL_PLAYER_NUMBERS overrides the default single-player config
    _raw_players = os.environ.get("RL_PARALLEL_PLAYER_NUMBERS", "")
    if _raw_players:
        player_numbers = tuple(int(p.strip()) for p in _raw_players.split(",") if p.strip())
    else:
        player_numbers = (1, 2, 3, 4, 5, 6)

    backend_cfg = replace(
        ENV.backend,
        mode="simrobot_visible",
        obs_mode="corrected",
        simrobot_profile="3d_visible",
        simrobot_launch=False,
    )
    env_cfg = replace(
        make_env_config(stage_id=0, backend_cfg=backend_cfg),
        parallel_player_numbers=player_numbers,
    )

    print(f"[bench_timing] SHM={SHM_NAME}  display_base={DISPLAY_BASE}", flush=True)
    print(f"[bench_timing] N_ROLLOUTS={N_ROLLOUTS}  rollout_steps={rollout_steps}  device={device}", flush=True)
    print(f"[bench_timing] parallel_player_numbers={player_numbers}", flush=True)
    runtime = VisibleParallelRuntime(
        player_numbers=player_numbers,
        shm_name=SHM_NAME,
        display_base=DISPLAY_BASE,
    )
    runtime.start()

    raw_env, wrapped_env, _ = build_env(env_cfg)
    agent, _ = build_mappo(wrapped_env, device, TRAIN, env_cfg.active_skill_names)

    from skrl.trainers.torch import SequentialTrainerCfg
    trainer_cfg = SequentialTrainerCfg(
        timesteps=N_ROLLOUTS * rollout_steps,
        headless=True,
        disable_progressbar=True,
        close_environment_at_exit=False,
    )
    agent.init(trainer_cfg=trainer_cfg)
    agent.enable_training_mode(True)

    agent_id = wrapped_env.possible_agents[0]
    observations, _ = wrapped_env.reset()
    total_steps = N_ROLLOUTS * rollout_steps

    # Per-step timing accumulators
    t_step_list:   list[float] = []   # wrapped_env.step() duration
    t_update_list: list[float] = []   # agent.post_interaction() duration at rollout boundary
    t_reset_list:  list[float] = []   # wrapped_env.reset() duration
    t_other_list:  list[float] = []   # everything else per step

    print(f"\n[bench_timing] starting {total_steps} steps ({N_ROLLOUTS} rollouts × {rollout_steps}) …\n", flush=True)
    t_wall_start = time.perf_counter()

    for timestep in range(total_steps):
        t_step_start = time.perf_counter()

        # ── pre + act ─────────────────────────────────────────────────────────
        obs_device = _move_to_device(observations, device)
        states_device = _agent_state_from_env(raw_env, agent_id, device)
        agent.pre_interaction(timestep=timestep, timesteps=total_steps)
        with torch.no_grad():
            actions, _ = agent.act(
                observations=obs_device,
                states=states_device,
                timestep=timestep,
                timesteps=total_steps,
            )
            safe_actions = {k: torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0) for k, v in actions.items()}

        t_before_step = time.perf_counter()

        # ── env step (SimRobot IPC) ───────────────────────────────────────────
        next_obs, rewards, terminated, truncated, infos = wrapped_env.step(
            _move_to_device(safe_actions, "cpu")
        )

        t_after_step = time.perf_counter()
        t_step_list.append(t_after_step - t_before_step)

        # ── record ───────────────────────────────────────────────────────────
        next_obs_device = _move_to_device(next_obs, device)
        next_states_device = _agent_state_from_env(raw_env, agent_id, device)
        agent.record_transition(
            observations=obs_device,
            states=states_device,
            actions=safe_actions,
            rewards=_move_to_device(rewards, device),
            next_observations=next_obs_device,
            next_states=next_states_device,
            terminated=_move_to_device(terminated, device),
            truncated=_move_to_device(truncated, device),
            infos=infos,
            timestep=timestep,
            timesteps=total_steps,
        )

        t_before_post = time.perf_counter()

        # ── post_interaction (PPO update fires at rollout boundary) ───────────
        agent.post_interaction(timestep=timestep, timesteps=total_steps)

        t_after_post = time.perf_counter()
        post_duration = t_after_post - t_before_post

        is_rollout_boundary = (timestep + 1) % rollout_steps == 0
        if is_rollout_boundary:
            t_update_list.append(post_duration)
        # Non-boundary post_interaction is bookkeeping only, lumped into t_other

        # ── reset (if episode done) ───────────────────────────────────────────
        t_reset_duration = 0.0
        if not wrapped_env.agents:
            t_before_reset = time.perf_counter()
            observations, _ = wrapped_env.reset()  # noqa: F841 — reassigned at loop top
            t_reset_duration = time.perf_counter() - t_before_reset
            t_reset_list.append(t_reset_duration)
        else:
            observations = next_obs

        # ── other overhead ────────────────────────────────────────────────────
        t_step_end = time.perf_counter()
        total_step_time = t_step_end - t_step_start
        t_other = total_step_time - (t_after_step - t_before_step) - post_duration - t_reset_duration
        t_other_list.append(max(0.0, t_other))

        if (timestep + 1) % rollout_steps == 0:
            rollout_idx = (timestep + 1) // rollout_steps
            update_s = t_update_list[-1] if t_update_list else 0.0
            recent_steps = t_step_list[-rollout_steps:]
            collect_s = sum(recent_steps)
            mean_step_ms = 1000.0 * np.mean(recent_steps)
            print(
                f"  rollout {rollout_idx}/{N_ROLLOUTS}  "
                f"T_collect={collect_s:.2f}s  T_update={update_s:.2f}s  "
                f"ms/step={mean_step_ms:.1f}",
                flush=True,
            )

    t_wall_end = time.perf_counter()
    wrapped_env.close()
    raw_env.close()
    runtime.close()

    # ── aggregate ─────────────────────────────────────────────────────────────
    n_slots = len(player_numbers)
    wall_s = t_wall_end - t_wall_start

    # Per rollout: collect = sum of rollout_steps env.step() calls
    # Build per-rollout T_collect from step-level data
    rollout_collect: list[float] = []
    rollout_other:   list[float] = []
    for r in range(N_ROLLOUTS):
        sl = t_step_list[r * rollout_steps: (r + 1) * rollout_steps]
        ot = t_other_list[r * rollout_steps: (r + 1) * rollout_steps]
        rollout_collect.append(sum(sl))
        rollout_other.append(sum(ot))

    mean_collect   = float(np.mean(rollout_collect))
    mean_update    = float(np.mean(t_update_list)) if t_update_list else 0.0
    mean_reset     = float(np.mean(t_reset_list))  if t_reset_list  else 0.0
    mean_other     = float(np.mean(rollout_other))
    mean_rollout   = mean_collect + mean_update + mean_other + (mean_reset * len(t_reset_list) / N_ROLLOUTS)
    efficiency     = mean_collect / mean_rollout if mean_rollout > 0 else 0.0

    # Async speedup: if T_update overlaps T_collect for next rollout
    # Effective per-rollout time = max(T_collect, T_update) + T_other + T_reset
    effective_async = max(mean_collect, mean_update) + mean_other + (mean_reset * len(t_reset_list) / N_ROLLOUTS)
    async_speedup   = mean_rollout / effective_async if effective_async > 0 else 1.0

    mean_step_ms    = 1000.0 * np.mean(t_step_list)
    env_steps_per_sec_raw   = 1.0 / np.mean(t_step_list) if t_step_list else 0.0
    env_steps_per_sec_wall  = total_steps / wall_s
    transitions_per_sec_raw = env_steps_per_sec_raw * n_slots
    transitions_per_sec_wall = env_steps_per_sec_wall * n_slots

    print("\n" + "=" * 70)
    print("TRAINING LOOP PHASE TIMING (Option 1 — async update diagnosis)")
    print("=" * 70)
    print(f"  rollout_steps            : {rollout_steps}")
    print(f"  N_ROLLOUTS               : {N_ROLLOUTS}")
    print(f"  n_slots                  : {n_slots}")
    print()
    print(f"  T_collect (per rollout)  : {mean_collect:.2f}s  ({1000*mean_collect/rollout_steps:.1f} ms/step avg)")
    print(f"  T_update  (per rollout)  : {mean_update:.2f}s  ← PPO gradient update")
    print(f"  T_reset   (per reset)    : {mean_reset:.2f}s  (N_resets={len(t_reset_list)})")
    print(f"  T_other   (per rollout)  : {mean_other:.2f}s  (act, record, bookkeeping)")
    print(f"  T_rollout (total)        : {mean_rollout:.2f}s")
    print()
    print(f"  env collection efficiency: {100 * efficiency:.1f}%  (T_collect / T_rollout)")
    print(f"  wall env-steps/sec       : {env_steps_per_sec_wall:.1f}  (incl. update pauses)")
    print(f"  raw  env-steps/sec       : {env_steps_per_sec_raw:.1f}  (pure env IPC only)")
    print(f"  wall transitions/sec     : {transitions_per_sec_wall:.1f}")
    print(f"  raw  transitions/sec     : {transitions_per_sec_raw:.1f}")
    print()
    print(f"  mean step latency        : {mean_step_ms:.1f} ms  (target ≈33ms = 1 camera frame)")
    print()
    print("  Async overlap estimate:")
    print(f"    effective T_rollout    : {effective_async:.2f}s  (max(T_collect,T_update)+T_other)")
    print(f"    async speedup (upper)  : {async_speedup:.2f}×")
    print(f"    async wall-steps/sec   : {env_steps_per_sec_wall * async_speedup:.1f}  (projected)")
    print()
    if mean_update > mean_collect:
        print("  → T_update > T_collect: async overlap is LEARNER-BOTTLENECKED")
        print("    Increasing collection (more slots) would help before async.")
    elif mean_update > 0.5 * mean_collect:
        print("  → T_update is significant fraction of T_collect: async worthwhile.")
    else:
        print("  → T_update << T_collect: async overhead may outweigh benefit.")
    print("=" * 70)

    summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "rollout_steps": rollout_steps,
        "n_rollouts": N_ROLLOUTS,
        "n_slots": n_slots,
        "wall_time_s": round(wall_s, 3),
        "mean_T_collect_s": round(mean_collect, 4),
        "mean_T_update_s": round(mean_update, 4),
        "mean_T_reset_s": round(mean_reset, 4),
        "mean_T_other_s": round(mean_other, 4),
        "mean_T_rollout_s": round(mean_rollout, 4),
        "efficiency_pct": round(100 * efficiency, 2),
        "env_steps_per_sec_wall": round(env_steps_per_sec_wall, 2),
        "env_steps_per_sec_raw": round(env_steps_per_sec_raw, 2),
        "transitions_per_sec_wall": round(transitions_per_sec_wall, 2),
        "transitions_per_sec_raw": round(transitions_per_sec_raw, 2),
        "mean_step_ms": round(mean_step_ms, 2),
        "async_speedup_upper": round(async_speedup, 3),
        "async_projected_steps_per_sec": round(env_steps_per_sec_wall * async_speedup, 2),
        "n_resets": len(t_reset_list),
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"\n[bench_timing] output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
