"""DAgger distillation round for v5: fit the expert's skill labels on states v5 visited.

Inputs: probe .npz files (run_probe_match.py) whose rows carry the encoded 47-dim obs, the
expert label (teacher + B-Human macro) and the macro flag. Output: a checkpoint in the same
format as the v5 checkpoint (policy_state_dict replaced), loadable by tmd._load_eval_policy.

Design-A (simulation-only, no gate.py / train.py / C++ change): on MACRO rows the shoot-armed
bit (+ progress) is baked to 1, and the pass-armed bit to 1 when the expert passed. The
probe's `dagger` controller applies the identical rule at inference, so obs bits, mask and
labels agree. `bake()` is idempotent, so already-baked dagger-arm rows can be re-used.

Loss per row (gate-masked, from the BAKED obs):
  CE(student, expert label)                      -- the DAgger correction
  + kl_w * KL(v5 || student)                     -- anchor: keep v5 where it already agrees
  + param_w * ||param_head(student) - param_head(v5)||^2   -- expert targets were not logged,
                                                    so the position head is held at v5's
Usage:
  python tools/train_dagger_v5.py --npz runs/.../V_*_a0.npz --out runs/dagger_v5/round1.pt
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from train import gate_skill_mask_from_obs  # noqa: E402
from observation import ObsIndex  # noqa: E402
from action import SKILL_ORDER  # noqa: E402

PASS_IDX = SKILL_ORDER.index("pass")
SHOOT_ARMED, SHOOT_PROGRESS, PASS_ARMED = int(ObsIndex.shoot_armed), int(ObsIndex.shoot_arm_progress), int(ObsIndex.pass_armed)


def bake(obs: np.ndarray, macro: np.ndarray, expert: np.ndarray) -> np.ndarray:
    o = obs.copy()
    o[macro, SHOOT_ARMED] = 1.0
    o[macro, SHOOT_PROGRESS] = 1.0
    o[macro & (expert == PASS_IDX), PASS_ARMED] = 1.0
    return o


class Policy(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.t0 = torch.nn.Linear(47, 128)
        self.t2 = torch.nn.Linear(128, 128)
        self.skill = torch.nn.Linear(128, 8)
        self.param = torch.nn.Linear(128, 4)
        for mod, key in ((self.t0, "trunk.0"), (self.t2, "trunk.2"), (self.skill, "skill_head"), (self.param, "param_head")):
            mod.weight.data.copy_(sd[f"{key}.weight"])
            mod.bias.data.copy_(sd[f"{key}.bias"])

    def forward(self, o):
        h = F.elu(self.t2(F.elu(self.t0(o))))
        return self.skill(h), torch.tanh(self.param(h))

    def export_into(self, sd):
        sd = dict(sd)
        for mod, key in ((self.t0, "trunk.0"), (self.t2, "trunk.2"), (self.skill, "skill_head"), (self.param, "param_head")):
            sd[f"{key}.weight"] = mod.weight.detach().cpu().clone()
            sd[f"{key}.bias"] = mod.bias.detach().cpu().clone()
        return sd


def load_rows(paths):
    keep = ("obs", "expert", "macro", "role", "actor", "step")
    parts = {k: [] for k in keep}
    match_id = []
    for i, p in enumerate(paths):
        z = np.load(p)
        for k in keep:
            parts[k].append(z[k])
        match_id.append(np.full(len(z["expert"]), i))
    d = {k: np.concatenate(v) for k, v in parts.items()}
    d["match"] = np.concatenate(match_id)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True, help="probe npz files / globs")
    ap.add_argument("--base", default="runs/merge_v5_attack_defense/checkpoint_stage8_step10000.pt",
                    help="initialisation AND anchor checkpoint")
    ap.add_argument("--init", default=None, help="optional different initialisation (e.g. previous round)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--kl-w", type=float, default=0.3)
    ap.add_argument("--param-w", type=float, default=1.0)
    ap.add_argument("--holdout-match", type=int, default=-1, help="index of match held out for validation (-1 = last)")
    args = ap.parse_args()

    # importing train may chdir; resolve every relative path against the repo root
    rp = lambda p: str(p if Path(p).is_absolute() else REPO / p)  # noqa: E731
    paths = sorted({p for g in args.npz for p in glob.glob(rp(g))})
    args.base, args.out = rp(args.base), rp(args.out)
    args.init = rp(args.init) if args.init else None
    if not paths:
        raise SystemExit(f"no npz matched {args.npz}")
    d = load_rows(paths)
    obs = bake(d["obs"].astype(np.float32), d["macro"].astype(bool), d["expert"].astype(int))
    y = d["expert"].astype(int)
    hold = (len(paths) - 1) if args.holdout_match < 0 else args.holdout_match
    tr, va = d["match"] != hold, d["match"] == hold
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    base_ck = torch.load(args.base, map_location="cpu")
    init_ck = torch.load(args.init, map_location="cpu") if args.init else base_ck
    anchor = Policy(base_ck["policy_state_dict"]).to(dev).eval()
    student = Policy(init_ck["policy_state_dict"]).to(dev)

    O = torch.as_tensor(obs, device=dev)
    Y = torch.as_tensor(y, device=dev)
    R = torch.as_tensor(d["role"].astype(int), device=dev)
    MAC = torch.as_tensor(d["macro"].astype(bool), device=dev)
    M = gate_skill_mask_from_obs(O)
    legal = M[torch.arange(len(Y), device=dev), Y] > 0.5
    print(f"rows {len(y)} from {len(paths)} files | train {tr.sum()} val {va.sum()} (match {hold}) | "
          f"expert label legal under baked mask: {100 * legal.float().mean():.1f}% | macro rows {100 * d['macro'].mean():.1f}%",
          flush=True)
    keep = legal.cpu().numpy()  # illegal labels (if any) are dropped -- Design-A
    with torch.no_grad():
        a_logits, a_param = anchor(O)
        a_logp = F.log_softmax(a_logits + (M - 1.0) * 1e9, dim=1)

    def evaluate(model, sel):
        with torch.no_grad():
            lg, _ = model(O[sel])
            pred = (lg + (M[sel] - 1.0) * 1e9).argmax(1)
            yy, rr, mm = Y[sel], R[sel], MAC[sel]
            out = {"agree": (pred == yy).float().mean().item()}
            for r, nm in ((0, "striker"), (1, "open"), (2, "off_ball")):
                s = rr == r
                out[nm] = (pred[s] == yy[s]).float().mean().item() if s.any() else float("nan")
            out["macro"] = (pred[mm] == yy[mm]).float().mean().item() if mm.any() else float("nan")
            return out

    vsel = torch.as_tensor(va & keep, device=dev)
    print("before:", {k: round(100 * v, 1) for k, v in evaluate(student, vsel).items()}, flush=True)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    idx_all = np.nonzero(tr & keep)[0]
    for ep in range(args.epochs):
        np.random.shuffle(idx_all)
        tot = 0.0
        for i in range(0, len(idx_all), args.batch):
            b = torch.as_tensor(idx_all[i:i + args.batch], device=dev)
            lg, pm = student(O[b])
            logp = F.log_softmax(lg + (M[b] - 1.0) * 1e9, dim=1)
            ce = F.nll_loss(logp, Y[b])
            kl = (a_logp[b].exp() * (a_logp[b] - logp)).sum(1).mean()
            pl = ((pm - a_param[b]) ** 2).mean()
            loss = ce + args.kl_w * kl + args.param_w * pl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        print(f"epoch {ep + 1}: loss {tot / len(idx_all):.4f} | val", {k: round(100 * v, 1) for k, v in evaluate(student, vsel).items()}, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ck = copy.copy(base_ck)
    ck["policy_state_dict"] = student.export_into(base_ck["policy_state_dict"])
    ck["dagger"] = {"npz": paths, "epochs": args.epochs, "kl_w": args.kl_w, "param_w": args.param_w,
                    "lr": args.lr, "init": args.init or args.base, "holdout": hold,
                    "val_after": evaluate(student, vsel)}
    torch.save(ck, out)
    json.dump(ck["dagger"], open(out.with_suffix(".json"), "w"), indent=1, default=str)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
