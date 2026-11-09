#!/usr/bin/env python3
"""Closed-loop team eval that AGGREGATES support hold-rate per episode.

The trainer only exposes support hold-rate via the module global
`team_parallel_environment.LAST_TEAM_METRICS`, which is overwritten every
episode — so neither eval_summary.csv nor eval_episodes.csv preserves a
per-episode hold-rate, and the competence gate sees only the last episode.

This driver wraps `TeamSlotSimRobotEnv._team_metrics` (additive, no behaviour
change) to append every episode's metrics to a list, then runs the validated
production launcher with minimal training so it warm-starts v2b@7500 and does
one rollout + one eval. After the run it dumps the accumulated per-episode
hold/thrash metrics to JSON so we can report a robust mean, not a last-episode
sample.

Resolves the single offline 🟡 from VERDICTO_v2b.md (supports walk vs hold).
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.getenv("HOLDRATE_OUT", str(REPO / "runs/offline_team_eval/holdrate_closed_loop")))
OUT.mkdir(parents=True, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
records: list[dict] = []


def _dump() -> None:
    path = OUT / f"holdrate_{STAMP}.json"
    if not records:
        path.write_text(json.dumps({"episodes": 0, "note": "no episodes captured"}, indent=2))
        print(f"[holdrate] NO episodes captured → {path}", flush=True)
        return
    import statistics as st

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in records if key in r]
        return st.mean(vals) if vals else float("nan")

    summary = {
        "episodes_captured": len(records),
        "support_open_hold_rate_mean": _mean("support_open_hold_rate"),
        "support_triangle_hold_rate_mean": _mean("support_triangle_hold_rate"),
        "role_thrash_rate_mean": _mean("role_thrash_rate"),
        "per_episode": records,
    }
    path.write_text(json.dumps(summary, indent=2))
    print(
        f"[holdrate] episodes={len(records)} "
        f"open_hold_mean={summary['support_open_hold_rate_mean']:.3f} "
        f"triangle_hold_mean={summary['support_triangle_hold_rate_mean']:.3f} "
        f"role_thrash_mean={summary['role_thrash_rate_mean']:.4f} → {path}",
        flush=True,
    )


atexit.register(_dump)


def _install_accumulator() -> None:
    from rlfootball.env import team_parallel_environment as tpe

    orig = tpe.TeamSlotSimRobotEnv._team_metrics

    def patched(self):  # noqa: ANN001
        metrics = orig(self)
        try:
            records.append({k: float(v) for k, v in metrics.items()})
        except Exception:  # never let accounting break the run
            pass
        return metrics

    tpe.TeamSlotSimRobotEnv._team_metrics = patched
    print("[holdrate] _team_metrics accumulator installed", flush=True)


def main() -> None:
    # Minimal-train config: warm-start v2b@7500, 1 rollout + 1 eval.
    os.environ.setdefault("RL_GATE_SHOOT_OPEN_ENTER", "0.30")
    os.environ.setdefault("RL_GATE_SHOOT_OPEN_EXIT", "0.20")
    os.environ.setdefault("RL_MA_PROD_TRAIN_STEPS", "256")
    os.environ.setdefault("RL_MA_PROD_EVAL_INTERVAL", "256")
    os.environ.setdefault("RL_MA_PROD_EVAL_EPISODES", "12")
    os.environ.setdefault("RL_MA_PROD_MAX_EPISODE_STEPS", "800")
    os.environ.setdefault("RL_MA_PROD_EXPERIMENT_NAME", "closed_loop_holdrate_v2b")
    os.environ.setdefault("RL_MA_PROD_CHECKPOINT_INTERVAL", "100000")  # don't write extra ckpts

    _install_accumulator()

    from rlfootball.training import train_multiagent_production as tmp

    sys.argv = [
        "team_holdrate_eval",
        "--stage-id", "7",
        "--checkpoint", str(REPO / "runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt"),
        "--dataset", str(REPO / "runs/teacher_multiagent_dataset/dataset_multiagent_train_20260611_full_v1/samples.npz"),
    ]
    args = tmp.parse_args()
    tmp.run(args)


if __name__ == "__main__":
    main()
