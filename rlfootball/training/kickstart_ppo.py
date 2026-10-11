"""Kickstarted PPO learner for the all-learned team (step 5).

One call = one PPO iteration on a batch of sampled live matches (run_probe_match.py --sample):
  * rewards are computed from the match logs (team-shared, + own fall penalty):
      GameController goal +-R_GOAL, on-target shot +1, completed pass +0.5,
      B-Human shot heading on target -0.5, ball in opponent half +0.003/step,
      ball in our defensive third -0.003/step, own fall -0.3
  * GAE per robot trajectory (gamma 0.995 per 8 Hz step ~ 25 s horizon, lambda 0.95), bootstrapped
    with the critic at the match cut-off
  * loss = PPO-clip on the joint log-prob (gate-masked skill + Gaussian y/theta params)
         + vf_coef * Huber(value) - ent_coef * H(skill)
         + kl_coef * KL(anchor || student)                 (anchor = starting student)
         + ce_coef * CE(student, teacher)                  (kickstarting; decays per iteration)
    teacher label: striker -> expert (teacher + B-Human macro); supports -> B-Human shadow label
    (shoot/pass outside the macro condition -> walk, as in train_distill_hybrid); labels illegal
    under the baked gate mask are skipped (Design-A).
Only rows the student controlled and sampled (npz `learned` & samp_skill >= 0) are used.

Usage:
  python -m rlfootball.training.kickstart_ppo --json "runs/kickstart/.../iter_003/*.json" --policy cur.pt \
      --anchor runs/distill_hybrid/round3.pt --iteration 3 --out next.pt
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rlfootball.core.action import SKILL_ORDER, ACTION_PARAM_MASKS
from rlfootball.core.config import ROOT_DIR
from rlfootball.core.gate import gate_skill_mask_from_obs
from rlfootball.training.train_dagger_v5 import Policy

HZ = 8.0
R_GOAL, R_SHOT_ON, R_PASS_OK, R_OPP_SHOT_ON = 10.0, 1.0, 0.5, -0.5
R_TERR_OPP, R_TERR_DEF, R_FALL = 0.003, -0.003, -0.3
R_FINAL_THIRD = 0.0   # extra per-step bonus while the ball is in the opponent final third (x > 1500)
GOAL_X, POST_Y, FRESH_MS = 4500.0, 800.0, 500.0
KICK_SPEED, KICK_NEAR, RECEIVE_MM, FALLEN = 900.0, 800.0, 450.0, 4
BH_TO_IDX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
WALK, SHOOT, PASS = SKILL_ORDER.index("walk"), SKILL_ORDER.index("shoot"), SKILL_ORDER.index("pass")
DIMS = (1, 2)
PMASK = torch.tensor([[ACTION_PARAM_MASKS[n][d] for d in DIMS] for n in SKILL_ORDER])  # [8, 2]


class Value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(47, 128), torch.nn.ELU(), torch.nn.Linear(128, 128),
                                       torch.nn.ELU(), torch.nn.Linear(128, 1))

    def forward(self, o):
        return self.net(o).squeeze(-1)


# ---------------------------------------------------------------- rewards from a match log
def _records(row):
    d = dict(row["agents"]); d.update(row.get("others", {}))
    return d


def _ball(row):
    pts = [(r["bx"], r["by"]) for r in list(_records(row).values()) + [row["gk"]] if r["b_seen"] > 0 and r["b_age"] <= FRESH_MS]
    return (float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts]))) if pts else None


def _vel(row):
    best = None
    for r in list(_records(row).values()) + [row["gk"]]:
        if r["b_seen"] > 0 and r["b_age"] <= FRESH_MS:
            sp = math.hypot(r["bvx"], r["bvy"])
            if best is None or sp > best[0]:
                c, s = math.cos(r["th"]), math.sin(r["th"])
                best = (sp, c * r["bvx"] - s * r["bvy"], s * r["bvx"] + c * r["bvy"])
    return best


def _on_target(b, vx, vy, gx):
    if (gx - b[0]) * vx <= 0 or abs(vx) < 1e-6:
        return False
    return abs(b[1] + vy * (gx - b[0]) / vx) < POST_Y + 100


def team_rewards(S):
    """Per-step team reward + per-(step, player) fall penalty + component totals."""
    n = len(S)
    r = np.zeros(n)
    comp = {"goal_for": 0, "goal_against": 0, "shot_on": 0, "pass_ok": 0, "opp_shot_on": 0, "falls": 0}
    for k in range(1, n):
        df = S[k]["gc_for"] - S[k - 1]["gc_for"]; da = S[k]["gc_against"] - S[k - 1]["gc_against"]
        r[k] += R_GOAL * (df - da); comp["goal_for"] += max(df, 0); comp["goal_against"] += max(da, 0)
    balls = [_ball(s) for s in S]
    pos = [{int(p): (q["gt_x"], q["gt_y"]) for p, q in _records(s).items()} for s in S]
    for k, b in enumerate(balls):
        if b is not None:
            r[k] += R_TERR_OPP if b[0] > 0 else (R_TERR_DEF if b[0] < -1500 else 0.0)
            if b[0] > 1500:
                r[k] += R_FINAL_THIRD
    k, W = 0, int(1.5 * HZ)
    while k < n:
        v, b = _vel(S[k]), balls[k]
        if v is None or b is None or v[0] < KICK_SPEED:
            k += 1; continue
        k0 = max(0, k - W)
        kicker, kick_skill, near_us = None, None, False
        for pid in _records(S[k]):
            near = any(balls[j] is not None and math.hypot(pos[j][int(pid)][0] - balls[j][0], pos[j][int(pid)][1] - balls[j][1]) <= KICK_NEAR
                       for j in range(k0, k + 1))
            if near:
                near_us = True
                ex = {_records(S[j])[pid]["bh_skill"] for j in range(k0, min(n, k + 4))}
                if 4 in ex or 3 in ex:
                    kicker, kick_skill = int(pid), ("pass" if 4 in ex else "shoot")
        opp_near = any(math.hypot(x - b[0], y - b[1]) <= KICK_NEAR for x, y in S[k]["opp"])
        ours = (v[1] > 0) if (near_us and opp_near) else near_us
        if ours and kick_skill == "shoot" and _on_target(b, v[1], v[2], GOAL_X):
            r[k] += R_SHOT_ON; comp["shot_on"] += 1
        elif ours and kick_skill == "pass":
            for j in range(k + int(0.5 * HZ), min(n, k + int(6 * HZ))):
                bj = balls[j]
                if bj is None:
                    continue
                if any(p not in (kicker, 1) and math.hypot(q[0] - bj[0], q[1] - bj[1]) < RECEIVE_MM for p, q in pos[j].items()):
                    r[j] += R_PASS_OK; comp["pass_ok"] += 1; break
                if any(math.hypot(x - bj[0], y - bj[1]) < RECEIVE_MM for x, y in S[j]["opp"]):
                    break
        elif not ours and opp_near and _on_target(b, v[1], v[2], -GOAL_X):
            r[k] += R_OPP_SHOT_ON; comp["opp_shot_on"] += 1
        k += int(2 * HZ)
    falls = {}
    for pid in S[0]["agents"]:
        fl = [s["agents"][pid]["fall"] for s in S]
        for j in range(1, n):
            if fl[j] == FALLEN and fl[j - 1] != FALLEN:
                falls[(j, int(pid))] = R_FALL; comp["falls"] += 1
    return r, falls, comp


# ---------------------------------------------------------------- batch loading
def load_match(jpath):
    J = json.load(open(jpath))
    S = J["steps"]
    if len(S) < 100:
        return None
    with np.load(jpath[:-5] + ".npz") as zf:
        z = {k: zf[k] for k in zf.files}
    if "learned" not in z:
        return None
    r_team, falls, comp = team_rewards(S)
    comp["gd"] = S[-1]["gc_for"] - S[-1]["gc_against"]
    comp["valid"] = bool(J["meta"]["valid"])
    keep = z["learned"].astype(bool) & (z["samp_skill"] >= 0)
    trajs = []
    for pid in np.unique(z["player"][keep]):
        idx = np.nonzero(keep & (z["player"] == pid))[0]
        idx = idx[np.argsort(z["step"][idx])]
        steps = z["step"][idx]
        rew = np.array([r_team[s - 1] + falls.get((s - 1, int(pid)), 0.0) for s in steps], np.float32)
        teach = []
        for i in idx:
            rec = S[int(z["step"][i]) - 1]["agents"][str(int(pid))]
            role, macro = int(z["role"][i]), bool(z["macro"][i])
            if role == 0:
                lab = int(z["expert"][i])
            else:
                lab = BH_TO_IDX.get(int(rec.get("shadow_skill", -1)), -1)
                if lab in (SHOOT, PASS) and not macro:
                    lab = WALK
            teach.append(lab)
        trajs.append({"obs": z["obs"][idx].astype(np.float32), "skill": z["samp_skill"][idx].astype(np.int64),
                      "params": z["samp_params"][idx][:, list(DIMS)].astype(np.float32),
                      "logp_old": z["logp"][idx].astype(np.float32), "rew": rew, "teach": np.array(teach, np.int64)})
    return trajs, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--policy", required=True, help="checkpoint that generated the batch")
    ap.add_argument("--anchor", required=True, help="starting student (KL anchor + teacher reference)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--param-std", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--ce0", type=float, default=0.3)
    ap.add_argument("--ce-decay", type=float, default=0.7)
    ap.add_argument("--critic-warmup-epochs", type=int, default=0, help="extra value-only epochs (first iteration)")
    ap.add_argument("--r-shot-on", type=float, default=R_SHOT_ON)
    ap.add_argument("--r-opp-shot-on", type=float, default=R_OPP_SHOT_ON)
    ap.add_argument("--r-final-third", type=float, default=R_FINAL_THIRD)
    args = ap.parse_args()
    globals().update(R_SHOT_ON=args.r_shot_on, R_OPP_SHOT_ON=args.r_opp_shot_on, R_FINAL_THIRD=args.r_final_third)
    rp = lambda p: str(p if Path(p).is_absolute() else ROOT_DIR / p)  # noqa: E731
    files = sorted({f for g in args.json for f in glob.glob(rp(g)) if f.endswith(".json")})
    loaded = [m for m in (load_match(f) for f in files) if m is not None]
    if not loaded:
        raise SystemExit("no sampled matches in batch")
    trajs = [t for m in loaded for t in m[0]]
    comps = [m[1] for m in loaded]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(rp(args.policy), map_location="cpu")
    anchor_ck = torch.load(rp(args.anchor), map_location="cpu")
    pol = Policy(ck["policy_state_dict"]).to(dev)
    anc = Policy(anchor_ck["policy_state_dict"]).to(dev).eval()
    val = Value().to(dev)
    val.load_state_dict(ck["value_state_dict"])

    # GAE per trajectory
    obs_l, sk_l, pa_l, lo_l, adv_l, ret_l, te_l = [], [], [], [], [], [], []
    with torch.no_grad():
        for t in trajs:
            o = torch.as_tensor(t["obs"], device=dev)
            v = val(o).cpu().numpy()
            T = len(v); adv = np.zeros(T, np.float32); last = 0.0
            for i in reversed(range(T)):
                nv = v[i + 1] if i + 1 < T else v[i]   # cut-off: bootstrap
                delta = t["rew"][i] + args.gamma * nv - v[i]
                last = delta + args.gamma * args.lam * last
                adv[i] = last
            obs_l.append(t["obs"]); sk_l.append(t["skill"]); pa_l.append(t["params"]); lo_l.append(t["logp_old"])
            adv_l.append(adv); ret_l.append(adv + v); te_l.append(t["teach"])
    O = torch.as_tensor(np.concatenate(obs_l), device=dev)
    SK = torch.as_tensor(np.concatenate(sk_l), device=dev)
    PA = torch.as_tensor(np.concatenate(pa_l), device=dev)
    LO = torch.as_tensor(np.concatenate(lo_l), device=dev)
    ADV = torch.as_tensor(np.concatenate(adv_l), device=dev)
    RET = torch.as_tensor(np.concatenate(ret_l), device=dev)
    TE = torch.as_tensor(np.concatenate(te_l), device=dev)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    M = gate_skill_mask_from_obs(O)
    te_ok = (TE >= 0) & (M[torch.arange(len(TE), device=dev), TE.clamp(min=0)] > 0.5)
    pm = PMASK.to(dev)[SK]  # [N, 2] which param dims were sampled
    with torch.no_grad():
        a_logp = F.log_softmax(anc(O)[0] + (M - 1.0) * 1e9, 1)
    ce_coef = args.ce0 * (args.ce_decay ** args.iteration)
    std = args.param_std
    opt = torch.optim.Adam(list(pol.parameters()) + list(val.parameters()), lr=args.lr)
    vopt = torch.optim.Adam(val.parameters(), lr=1e-3)
    for _ in range(args.critic_warmup_epochs):
        for b in torch.randperm(len(O), device=dev).split(args.minibatch):
            loss = F.smooth_l1_loss(val(O[b]), RET[b]); vopt.zero_grad(); loss.backward(); vopt.step()

    stats = {"clipfrac": [], "approx_kl": [], "kl_anchor": [], "ce": [], "entropy": [], "vloss": []}
    for _ in range(args.epochs):
        for b in torch.randperm(len(O), device=dev).split(args.minibatch):
            lg, pmean = pol(O[b])
            logp_all = F.log_softmax(lg + (M[b] - 1.0) * 1e9, 1)
            lp = logp_all.gather(1, SK[b][:, None]).squeeze(1)
            mu = pmean[:, list(DIMS)]
            gauss = -0.5 * ((PA[b] - mu) / std) ** 2 - math.log(std * math.sqrt(2 * math.pi))
            lp = lp + (gauss * pm[b]).sum(1)
            ratio = torch.exp(lp - LO[b])
            pg = -torch.min(ratio * ADV[b], ratio.clamp(1 - args.clip, 1 + args.clip) * ADV[b]).mean()
            vl = F.smooth_l1_loss(val(O[b]), RET[b])
            p = logp_all.exp()
            ent = -(p * logp_all.clamp(min=-30)).sum(1).mean()
            kl = (a_logp[b].exp() * (a_logp[b] - logp_all)).sum(1).mean()
            ok = te_ok[b]
            ce = F.nll_loss(logp_all[ok], TE[b][ok]) if ok.any() else torch.zeros((), device=dev)
            loss = pg + args.vf_coef * vl - args.ent_coef * ent + args.kl_coef * kl + ce_coef * ce
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pol.parameters()) + list(val.parameters()), 0.5)
            opt.step()
            with torch.no_grad():
                stats["clipfrac"].append(((ratio - 1).abs() > args.clip).float().mean().item())
                stats["approx_kl"].append((LO[b] - lp).mean().item())
                stats["kl_anchor"].append(kl.item()); stats["ce"].append(ce.item())
                stats["entropy"].append(ent.item()); stats["vloss"].append(vl.item())

    new = copy.copy(ck)
    new["policy_state_dict"] = pol.export_into(ck["policy_state_dict"])
    new["value_state_dict"] = {k: v.cpu() for k, v in val.state_dict().items()}
    summ = {k: float(np.mean(v)) for k, v in stats.items()}
    batch = {"matches": len(comps), "valid_matches": sum(c["valid"] for c in comps),
             "gd_mean": float(np.mean([c["gd"] for c in comps])),
             "rows": int(len(O)), "mean_reward_per_step": float(np.concatenate([t["rew"] for t in trajs]).mean()),
             **{k: int(sum(c[k] for c in comps)) for k in ("goal_for", "goal_against", "shot_on", "pass_ok", "opp_shot_on", "falls")}}
    new["kickstart"] = {"iteration": args.iteration, "policy_in": args.policy, "anchor": args.anchor, "ce_coef": ce_coef,
                        "rewards": {"goal": R_GOAL, "shot_on": R_SHOT_ON, "pass_ok": R_PASS_OK, "opp_shot_on": R_OPP_SHOT_ON,
                                    "terr_opp": R_TERR_OPP, "terr_def": R_TERR_DEF, "final_third": R_FINAL_THIRD, "fall": R_FALL},
                        "batch": batch, "train": summ, "files": files}
    out = Path(rp(args.out)); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, out)
    json.dump(new["kickstart"], open(out.with_suffix(".json"), "w"), indent=1, default=str)
    print(json.dumps({"iteration": args.iteration, "ce_coef": round(ce_coef, 4), "batch": batch,
                      "train": {k: round(v, 4) for k, v in summ.items()}}), flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
