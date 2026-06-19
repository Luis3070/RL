#!/usr/bin/env python3
"""Mirror-augment the GK BC dataset L<->R to enforce exact left/right symmetry.

Why: left_post_jump (77.7%) vs right_post_jump (100%) is a LEARNED-policy asymmetry — the
save model + scenarios + raw skill counts are already symmetric, so count-balance isn't the
issue. Reflecting every sample across the goal's y-axis (and swapping left/right skills) gives
the network a perfectly symmetric corpus, which removes the freedom to learn an L/R bias.

Reflection contract (must match goalkeeper_teacher.encode_goalkeeper_obs index order):
- obs y-signed dims negated: [1 robot_y, 2 robot_theta, 4 ball_rel_y, 6 ball_end_rel_y,
  8 ball_vel_y, 14 goalie_err_y, 15 goalie_err_theta, 17 goalie_y, 18 goalie_theta,
  21 intersection_y, 30 team_ball_y_rel]; near_left_post(22) <-> near_right_post(23) swapped.
- action [skill, target_x, target_y, target_theta, pass_target]: target_y(2) & target_theta(3)
  negated; skill swapped low_left(5)<->low_right(6), jump_left(7)<->jump_right(8).
Run: python tools/mirror_augment_gk_dataset.py --dataset <dir/samples.npz>
"""
import argparse, json
from datetime import datetime
from pathlib import Path
import numpy as np

NEG_OBS = [1, 2, 4, 6, 8, 14, 15, 17, 18, 21, 30]
SWAP_OBS = (22, 23)
SKILL_SWAP = {5: 6, 6: 5, 7: 8, 8: 7}   # low_l<->low_r, jump_l<->jump_r


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    m = obs.copy()
    m[:, NEG_OBS] *= -1.0
    a, b = SWAP_OBS
    m[:, [a, b]] = m[:, [b, a]]
    return m


def mirror_actions(act: np.ndarray) -> np.ndarray:
    m = act.copy()
    m[:, 2] *= -1.0   # target_y
    m[:, 3] *= -1.0   # target_theta
    sk = m[:, 0].astype(int)
    m[:, 0] = np.array([SKILL_SWAP.get(int(s), int(s)) for s in sk], dtype=m.dtype)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to samples.npz")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()
    src = Path(args.dataset)
    z = dict(np.load(src, allow_pickle=True))
    obs = np.asarray(z["observations"]); act = np.asarray(z["actions"])
    n = len(obs)
    m_obs = mirror_obs(obs); m_act = mirror_actions(act)
    ep = np.asarray(z["episode_ids"]); ep_off = int(ep.max()) + 1
    out = {
        "observations": np.concatenate([obs, m_obs], 0).astype(np.float32),
        "actions": np.concatenate([act, m_act], 0).astype(np.float32),
        "stages": np.concatenate([z["stages"], z["stages"]], 0),
        "episode_ids": np.concatenate([ep, ep + ep_off], 0),
        "step_ids": np.concatenate([z["step_ids"], z["step_ids"]], 0),
        "weights": np.concatenate([z["weights"], z["weights"]], 0),
        "label_consistent": np.concatenate([z["label_consistent"], z["label_consistent"]], 0),
        "scenario_ids": np.concatenate([z["scenario_ids"], z["scenario_ids"]], 0),
        "scenario_names": np.concatenate([z["scenario_names"], z["scenario_names"]], 0),
        "skill_order": z["skill_order"],
    }
    out_dir = Path(args.output_dir) if args.output_dir else \
        src.parent.parent / f"dataset_gk_mirror_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "samples.npz", **out)
    # carry a summary if present
    sp = src.parent / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text()); s["mirror_augmented"] = True; s["rows"] = 2 * n
        (out_dir / "summary.json").write_text(json.dumps(s, indent=2))
    print(f"[gk-mirror] {n} -> {2*n} rows  out={out_dir/'samples.npz'}")
    # sanity: skill histogram must be L/R symmetric now
    import collections
    sk = out["actions"][:, 0].astype(int); c = collections.Counter(sk.tolist())
    for l, r in [(5, 6), (7, 8)]:
        print(f"  skill[{l}]={c.get(l,0)}  skill[{r}]={c.get(r,0)}  (must be equal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
