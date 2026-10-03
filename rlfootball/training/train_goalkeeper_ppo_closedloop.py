"""Closed-loop GK PPO: optimize GOAL-ACTUALLY-PREVENTED, not teacher-skill-match.

Why this exists: the offline imitation trainer (`train_goalkeeper_ppo.py`) rewards
matching the teacher's per-step skill. That cannot instill the multi-step *commitment*
of coming off the line for a corner-bound dribble, and gives no symmetry signal — so a
stable L/R asymmetry survives and the fast/off-center/round-keeper dribbles stay low.

This trainer rolls out the *same* analytical attack episodes used by the closed-loop
benchmark (`tools/eval/benchmark_goalkeeper_closed_loop_shots.py`), with the policy SAMPLING
actions, and rewards the terminal outcome (save vs goal) plus light positioning shaping.
A symmetry loss ties each obs to its mirror so left/right are learned together.

Head is OFF during training (the scripted slow-corner head is coupled to the old weights);
deploy this candidate head-free.

Run:
  python train_goalkeeper_ppo_closedloop.py --bc-checkpoint <bc.pt> --updates 60 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from rlfootball.training import train_goalkeeper_ppo as T
import tools.eval.benchmark_goalkeeper_closed_loop_shots as B
from rlfootball.core.config import ROOT_DIR
from rlfootball.core.goalkeeper_action import GK_ACTION_PARAM_MASKS, GK_SKILL_ORDER
from rlfootball.teachers.goalkeeper_teacher import (
    GOALIE_LINE_X_MM,
    OWN_GOAL_HALF_Y_MM,
    OWN_GOAL_X_MM,
    encode_goalkeeper_obs,
    goalkeeper_skill_mask_np,
)

# Disable the scripted slow-corner head so the trunk learns everything itself.
T._apply_slow_corner_dribble_head = lambda obs, logits, params: (logits, params)

RUNS_DIR = ROOT_DIR / "runs"
DT = B.DT_S
MAX_STEPS = B.MAX_STEPS

# Mirror contract (must match tools/dataset/mirror_augment_gk_dataset.py).
NEG_OBS = [1, 2, 4, 6, 8, 14, 15, 17, 18, 21, 30]
SWAP_OBS = (22, 23)
SKILL_SWAP = {5: 6, 6: 5, 7: 8, 8: 7}
_SKILL_PERM = [SKILL_SWAP.get(i, i) for i in range(len(GK_SKILL_ORDER))]

# Train on the dribble family (incl. stress) + the full shot suite so shots are kept.
_DRIBBLE = [s for s in B.ATTACKS if s.is_dribble_in]
_SHOTS = [s for s in B.ATTACKS if not s.is_dribble_in]


@dataclass
class Env:
    spec: B.AttackSpec
    robot_x: float
    robot_y: float
    ball_x: float
    ball_y: float
    dribble_speed: float
    target_y: float
    shot_speed: float
    lateral_cut: float
    shot_fired: bool = False
    saved: bool = False
    done: bool = False
    outcome: float = 0.0          # +1 save, -1 goal, +0.3 out
    freeze: int = 0
    committed_skill: str = ""
    committed_steps_left: int = 0
    ball_vx: float = 0.0
    ball_vy: float = 0.0
    prev_gap: float = 0.0


def _new_env(spec: B.AttackSpec, rng: np.random.Generator) -> Env:
    dribble_speed = float(rng.uniform(*spec.dribble_speed_range))
    if spec.random_side:
        side = 1.0 if rng.random() < 0.5 else -1.0
        ball_y = side * float(rng.uniform(*spec.start_y_range))
        target_y = side * float(rng.uniform(*spec.target_y_range))
    else:
        ball_y = float(rng.uniform(*spec.start_y_range))
        target_y = float(rng.uniform(*spec.target_y_range))
    return Env(
        spec=spec,
        robot_x=GOALIE_LINE_X_MM,
        robot_y=float(rng.normal(0.0, 80.0)),
        ball_x=float(spec.start_x),
        ball_y=ball_y,
        dribble_speed=dribble_speed,
        target_y=target_y,
        shot_speed=float(rng.uniform(*spec.shot_speed_range)),
        lateral_cut=float(rng.uniform(*spec.lateral_cut_range)),
    )


def _crossing_gap(e: Env) -> float:
    """Distance between keeper y and the ball's predicted crossing of the keeper line."""
    cy = B._line_crossing_y(e.robot_x, e.ball_x, e.ball_y, e.ball_vx, e.ball_vy)
    if cy is None:
        cy = e.ball_y
    return abs(cy - e.robot_y)


def _step_dynamics(e: Env, skill: str, params: np.ndarray, rng: np.random.Generator) -> float:
    """Advance one env by the benchmark's dynamics. Returns the step reward."""
    spec = e.spec
    # 1) ball velocity
    if spec.is_dribble_in:
        if not e.saved:
            aim_y = e.target_y
            if spec.round_keeper and e.ball_x > spec.cut_x:
                aim_y = 0.0
            dx = OWN_GOAL_X_MM - e.ball_x
            dy = aim_y - e.ball_y
            nrm = max(1e-6, math.hypot(dx, dy))
            e.ball_vx = e.dribble_speed * dx / nrm
            e.ball_vy = e.dribble_speed * dy / nrm
    elif not e.shot_fired:
        e.ball_vx = -e.dribble_speed
        desired_y = e.ball_y + e.lateral_cut * DT
        e.ball_vy = float(np.clip((desired_y - e.ball_y) / DT, -450.0, 450.0))
        if e.ball_x <= spec.shot_x:
            e.shot_fired = True
            e.ball_vx, e.ball_vy = B._shot_velocity(e.ball_x, e.ball_y, e.target_y, e.shot_speed)
    crossing = B._crossing_y(e.ball_x, e.ball_y, e.ball_vx, e.ball_vy)

    # 2) commit tracking
    if skill.startswith("intercept_"):
        e.committed_skill = skill
        e.committed_steps_left = B.SAVE_COMMIT_STEPS
    elif e.committed_steps_left > 0:
        e.committed_steps_left -= 1
    else:
        e.committed_skill = ""

    # 3) keeper motion
    if skill == "walk_goalie_pose":
        target_x = float(np.clip(params[0], -1.0, 1.0) * 4500.0)
        target_y_walk = float(np.clip(params[1], -1.0, 1.0) * 3000.0)
        e.robot_x += float(np.clip(target_x - e.robot_x, -220.0 * DT, 220.0 * DT))
        e.robot_y += float(np.clip(target_y_walk - e.robot_y, -260.0 * DT, 260.0 * DT))

    # 4) save detection (shot)
    if e.shot_fired and crossing is not None:
        keeper_crossing = B._line_crossing_y(e.robot_x, e.ball_x, e.ball_y, e.ball_vx, e.ball_vy)
        if keeper_crossing is None:
            keeper_crossing = crossing
        keeper_ttc = (e.robot_x - e.ball_x) / e.ball_vx if e.ball_vx < 0.0 else float("inf")
        rel = keeper_crossing - e.robot_y
        save_skill = skill if B._skill_matches_crossing(skill, rel) else e.committed_skill
        if (not e.saved and 0.0 <= keeper_ttc <= 1.15 and B._skill_matches_crossing(save_skill, rel)
                and abs(rel) <= B._save_radius(save_skill)):
            e.saved = True
            e.freeze = B.SAVE_FREEZE_STEPS
            e.ball_x = e.robot_x + 250.0
            e.ball_y = e.robot_y + np.sign(rel or 1.0) * 220.0
            e.ball_vx = B.REBOUND_SPEED
            e.ball_vy = float(rng.normal(0.0, 180.0))

    # 5) save detection (dribble)
    if spec.is_dribble_in and not e.saved:
        if math.hypot(e.ball_x - e.robot_x, e.ball_y - e.robot_y) <= B._dribble_engage_reach(skill):
            e.saved = True
            e.freeze = B.SAVE_FREEZE_STEPS
            e.ball_x = e.robot_x + 250.0
            e.ball_y = e.robot_y
            e.ball_vx = B.REBOUND_SPEED
            e.ball_vy = float(rng.normal(0.0, 120.0))

    # 6) advance ball
    if e.freeze > 0:
        e.freeze -= 1
    else:
        e.ball_x += e.ball_vx * DT
        e.ball_y += e.ball_vy * DT

    # 7) terminal + reward
    reward = 0.0
    # positioning shaping: reward shrinking the crossing gap (encourages coming across)
    gap = _crossing_gap(e)
    reward += 0.03 * float(np.tanh((e.prev_gap - gap) / 300.0))
    e.prev_gap = gap

    if e.ball_x <= OWN_GOAL_X_MM:
        if abs(e.ball_y) <= OWN_GOAL_HALF_Y_MM:
            e.outcome = -1.0
        else:
            e.outcome = 0.3
        e.done = True
        reward += e.outcome
    elif e.saved and e.ball_x > GOALIE_LINE_X_MM + 900.0:
        e.outcome = 1.0
        e.done = True
        reward += e.outcome
    elif abs(e.ball_y) > 1800.0 and e.shot_fired:
        e.outcome = 0.3
        e.done = True
        reward += e.outcome
    return reward


def _sample_specs(n: int, rng: np.random.Generator, dribble_frac: float) -> list[B.AttackSpec]:
    out = []
    for _ in range(n):
        pool = _DRIBBLE if rng.random() < dribble_frac else _SHOTS
        out.append(pool[rng.integers(len(pool))])
    return out


@torch.no_grad()
def rollout(actor, value, n_envs, rng, device, dribble_frac):
    envs = [_new_env(s, rng) for s in _sample_specs(n_envs, rng, dribble_frac)]
    for e in envs:
        e.prev_gap = 0.0
    obs_b, skill_b, rawp_b, lp_b, val_b, rew_b, alive_b = [], [], [], [], [], [], []
    pmask_t = torch.as_tensor([GK_ACTION_PARAM_MASKS[n] for n in GK_SKILL_ORDER], dtype=torch.float32, device=device)
    for _step in range(MAX_STEPS):
        alive = np.array([not e.done for e in envs])
        if not alive.any():
            break
        enc = np.stack([encode_goalkeeper_obs(B._obs(
            robot_x=e.robot_x, robot_y=e.robot_y, robot_theta=0.0,
            ball_x=e.ball_x, ball_y=e.ball_y, ball_vx=e.ball_vx, ball_vy=e.ball_vy)) for e in envs]).astype(np.float32)
        obs_t = torch.as_tensor(enc, device=device)
        logits, mean, log_std = actor(obs_t)
        masked = T._masked_logits(logits, obs_t)
        sdist = Categorical(logits=masked)
        skill = sdist.sample()
        std = log_std.exp().expand_as(mean)
        pdist = Normal(mean, std)
        raw = pdist.rsample()
        pm = pmask_t[skill]
        lp = sdist.log_prob(skill) + (pdist.log_prob(raw) * pm).sum(-1)
        params = (torch.clamp(raw, -1.0, 1.0) * pm).cpu().numpy()
        skill_np = skill.cpu().numpy()
        v = value(obs_t)
        step_rew = np.zeros(n_envs, dtype=np.float32)
        for i, e in enumerate(envs):
            if e.done:
                continue
            step_rew[i] = _step_dynamics(e, GK_SKILL_ORDER[int(skill_np[i])], params[i], rng)
        obs_b.append(enc); skill_b.append(skill_np); rawp_b.append(raw.cpu().numpy())
        lp_b.append(lp.cpu().numpy()); val_b.append(v.cpu().numpy()); rew_b.append(step_rew); alive_b.append(alive)
    outcomes = np.array([e.outcome for e in envs])
    saved = np.array([e.saved and e.outcome > 0.5 for e in envs])
    return {
        "obs": np.stack(obs_b), "skill": np.stack(skill_b), "raw": np.stack(rawp_b),
        "logp": np.stack(lp_b), "val": np.stack(val_b), "rew": np.stack(rew_b), "alive": np.stack(alive_b),
        "outcomes": outcomes, "saved": saved,
        "specnames": [e.spec.name for e in envs],
    }


def _gae(rew, val, alive, gamma, lam):
    Tn, N = rew.shape
    adv = np.zeros_like(rew)
    last = np.zeros(N)
    next_val = np.zeros(N)
    for t in reversed(range(Tn)):
        nonterm = alive[t].astype(np.float32)
        delta = rew[t] + gamma * next_val * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
        next_val = val[t]
    ret = adv + val
    return adv, ret


def _mirror_obs_t(obs: torch.Tensor) -> torch.Tensor:
    m = obs.clone()
    m[:, NEG_OBS] = -m[:, NEG_OBS]
    a, b = SWAP_OBS
    m[:, [a, b]] = m[:, [b, a]]
    return m


# Non-attack behaviors the closed-loop rollout never exercises — must be anchored to
# the teacher or the policy forgets them (pass/observe/dribble-out for distribution).
_ANCHOR_SCENARIOS = {
    "controlled_pass", "controlled_clear", "controlled_dribble",
    "stale_search", "stale_no_team_ball", "misposition_recover",
    "moving_away", "outside_goal_corridor",
}


def _load_anchor(dataset: Path, device: torch.device):
    z = np.load(dataset, allow_pickle=True)
    names = list(z["scenario_names"]); sid = z["scenario_ids"]; w = z["weights"]
    keep_ids = {i for i, n in enumerate(names) if n in _ANCHOR_SCENARIOS}
    sel = np.array([(int(s) in keep_ids) and (wv > 0) for s, wv in zip(sid, w)])
    idx = np.where(sel)[0]
    obs = torch.as_tensor(z["observations"][idx].astype(np.float32), device=device)
    acts = z["actions"][idx].astype(np.float32)
    skill = torch.as_tensor(acts[:, 0].round().astype(np.int64), device=device)
    params = torch.as_tensor(np.clip(acts[:, 1:], -1.0, 1.0).astype(np.float32), device=device)
    pmask = torch.as_tensor(
        np.stack([GK_ACTION_PARAM_MASKS[GK_SKILL_ORDER[int(s)]] for s in acts[:, 0].round().astype(int)]).astype(np.float32),
        device=device)
    print(f"[cl-ppo] anchor rows={len(idx)} from {dataset}", flush=True)
    return obs, skill, params, pmask


def train(args):
    device = torch.device(args.device)
    actor = T.GoalkeeperPPOActor(64, T.TRAIN.hidden_size).to(device)
    value = T.GoalkeeperValue(64, T.TRAIN.hidden_size).to(device)
    T._load_bc(actor, Path(args.bc_checkpoint), device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(value.parameters()), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)
    a_obs = a_skill = a_params = a_pmask = None
    if args.anchor_dataset and args.anchor_coef > 0.0:
        a_obs, a_skill, a_params, a_pmask = _load_anchor(Path(args.anchor_dataset), device)
    perm = torch.as_tensor(_SKILL_PERM, dtype=torch.long, device=device)
    out = Path(args.output_root) / f"ppo_gk_closedloop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for upd in range(args.updates):
        roll = rollout(actor, value, args.envs, rng, device, args.dribble_frac)
        adv, ret = _gae(roll["rew"], roll["val"], roll["alive"], args.gamma, args.lam)
        mask = roll["alive"].reshape(-1)
        obs = torch.as_tensor(roll["obs"].reshape(-1, 64)[mask], device=device)
        skill = torch.as_tensor(roll["skill"].reshape(-1)[mask], dtype=torch.long, device=device)
        raw = torch.as_tensor(roll["raw"].reshape(-1, 4)[mask], device=device)
        old_lp = torch.as_tensor(roll["logp"].reshape(-1)[mask], device=device)
        adv_t = torch.as_tensor(adv.reshape(-1)[mask], dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret.reshape(-1)[mask], dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
        n = obs.shape[0]
        for _ep in range(args.epochs):
            idx = torch.randperm(n, device=device)
            for s in range(0, n, args.minibatch):
                mb = idx[s:s + args.minibatch]
                lp, ent, v = T._evaluate_actions(actor, value, obs[mb], skill[mb], raw[mb])
                ratio = (lp - old_lp[mb]).exp()
                pg = -torch.min(ratio * adv_t[mb], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t[mb]).mean()
                vl = F.mse_loss(v, ret_t[mb])
                # L/R symmetry: logits(obs) must equal skill-permuted logits(mirror(obs))
                lo, _, _ = actor(obs[mb])
                lm, _, _ = actor(_mirror_obs_t(obs[mb]))
                sym = F.mse_loss(lo, lm[:, perm])
                anchor = torch.zeros((), device=device)
                if a_obs is not None:
                    am = torch.randint(0, a_obs.shape[0], (args.minibatch,), device=device)
                    alog, amean, _ = actor(a_obs[am])
                    amask = T._masked_logits(alog, a_obs[am])
                    anchor = F.cross_entropy(amask, a_skill[am])
                    anchor = anchor + F.mse_loss(amean * a_pmask[am], a_params[am] * a_pmask[am])
                loss = pg + 0.5 * vl - args.ent_coef * ent.mean() + args.sym_coef * sym + args.anchor_coef * anchor
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(value.parameters()), 0.75)
                opt.step()
        # per-update report
        names = np.array(roll["specnames"])
        sr = {}
        for grp, sel in [("dribble", np.array([("dribble_in" in x) for x in names])),
                         ("shot", np.array([("dribble_in" not in x) for x in names]))]:
            if sel.any():
                sr[grp] = float(roll["saved"][sel].mean())
        row = {"update": upd, "save_overall": float(roll["saved"].mean()), **{f"save_{k}": v for k, v in sr.items()},
               "sym": float(sym.detach().cpu()), "value_loss": float(vl.detach().cpu())}
        history.append(row)
        print(f"[cl-ppo] upd={upd}/{args.updates} save={row['save_overall']:.3f} "
              f"drib={sr.get('dribble', float('nan')):.3f} shot={sr.get('shot', float('nan')):.3f} sym={row['sym']:.4f}", flush=True)
    ckpt = out / "checkpoint_gk_closedloop.pt"
    torch.save({"policy_state_dict": actor.state_dict(), "value_state_dict": value.state_dict(),
                "config": vars(args), "history": history}, ckpt)
    (out / "report.json").write_text(json.dumps({"history": history, "checkpoint": str(ckpt)}, indent=2))
    print(f"[cl-ppo] saved {ckpt}", flush=True)
    return ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-checkpoint", required=True)
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.004)
    p.add_argument("--sym-coef", type=float, default=0.5)
    p.add_argument("--dribble-frac", type=float, default=0.6)
    p.add_argument("--anchor-dataset", default="", help="dataset for BC-anchoring non-attack behaviors (pass/observe/dribble-out)")
    p.add_argument("--anchor-coef", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=260619)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-root", default=str(RUNS_DIR / "goalkeeper_ppo"))
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    raise SystemExit(main())
