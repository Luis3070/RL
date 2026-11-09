"""Does the heuristic teacher WIN or CONCEDE contested balls?

This is the load-bearing assumption behind the 2026-08-11 reactive-opponent corpus
regeneration: demonstrations collected against genuinely reactive B-Human opponents were
supposed to teach the policy to contest, because the teacher was assumed to contest. That
assumption was never measured, and the resulting fine-tune left contest-conceding slightly
WORSE (-6.90 mm/step vs -4.06). If the teacher itself backs off contested balls, then BC
on any corpus -- reactive opponents or not -- teaches exactly the behaviour we are trying
to remove, and more corpus work is wasted effort until the teacher is fixed.

Measured directly from the stored demonstrations (no sim run): each row carries the
robot-frame ball vector and the nearest-opponent distance, so the per-step change in
distance-to-ball, bucketed by opponent proximity, is recoverable.

Decoding (environment.py::_encode_obs):
  obs[3] = ball_rel_x / ACTION.x_range(4500)   obs[4] = ball_rel_y / ACTION.y_range(3000)
  obs[18] = nearest_opponent_dist / 4500       (GT when RL_TRAIN_GT_OPPONENT_OBS=1)
  role_id: 0=striker 1=open_support 2=off_ball_support

The headline number is `% of contested steps closing`, which -- unlike mm/step -- is
directly comparable against the live-match figures despite the teacher collecting at a
different control period than the 8 Hz match loop.

Usage:
  python tools/eval/measure_teacher_duel_behavior.py <dataset_dir> [<dataset_dir> ...]
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

X_RANGE = 4500.0
Y_RANGE = 3000.0
BALL_REL_X, BALL_REL_Y, NEAREST_OPP = 3, 4, 18
STRIKER_ROLE_ID = 0


def analyze(dataset_dir: Path, role_filter: int | None = STRIKER_ROLE_ID) -> dict:
    data = np.load(dataset_dir / "samples.npz")
    obs = data["observations"]
    episodes = data["episode_ids"]
    steps = data["step_ids"]
    roles = data["role_ids"] if "role_ids" in data.files else np.zeros(len(obs), dtype=np.int8)

    d_ball = np.hypot(obs[:, BALL_REL_X] * X_RANGE, obs[:, BALL_REL_Y] * Y_RANGE)
    opp_d = obs[:, NEAREST_OPP] * X_RANGE

    # Group by (episode, role) so consecutive rows belong to one agent's trajectory;
    # the corpus interleaves all three agents' rows within an episode.
    traj: dict[tuple[int, int], list[tuple[int, float, float]]] = defaultdict(list)
    for i in range(len(obs)):
        if role_filter is not None and int(roles[i]) != role_filter:
            continue
        traj[(int(episodes[i]), int(roles[i]))].append((int(steps[i]), float(d_ball[i]), float(opp_d[i])))

    bands = {"opp<600": (0.0, 600.0), "opp600-1200": (600.0, 1200.0), "opp>1200": (1200.0, 1e9)}
    deltas: dict[str, list[float]] = {b: [] for b in bands}
    for key, rows in traj.items():
        rows.sort(key=lambda r: r[0])
        for (s0, d0, _o0), (s1, d1, o1) in zip(rows, rows[1:]):
            if s1 != s0 + 1:      # only true consecutive steps
                continue
            if d0 > 6000.0 or d1 > 6000.0:   # discard sentinel/blank-obs rows
                continue
            closing = d0 - d1     # positive == moved toward the ball
            for name, (lo, hi) in bands.items():
                if lo <= o1 < hi:
                    deltas[name].append(closing)
                    break

    out = {"dataset": dataset_dir.name, "bands": {}}
    for name in bands:
        vals = deltas[name]
        if not vals:
            continue
        out["bands"][name] = {
            "n": len(vals),
            "mean_mm_per_step": statistics.mean(vals),
            "pct_closing": 100.0 * sum(1 for v in vals if v > 0) / len(vals),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dirs", nargs="+", type=Path)
    ap.add_argument("--all-roles", action="store_true", help="Include support roles, not just striker.")
    args = ap.parse_args()

    for d in args.dataset_dirs:
        r = analyze(d, role_filter=None if args.all_roles else STRIKER_ROLE_ID)
        print(f"=== {r['dataset']} ===")
        for band, st in r["bands"].items():
            print(f"  {band:14s} n={st['n']:7d}  mean {st['mean_mm_per_step']:+7.2f} mm/step   "
                  f"closing on {st['pct_closing']:5.1f}% of steps")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
