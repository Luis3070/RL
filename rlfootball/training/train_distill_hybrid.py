"""Distil the hybrid team (learned striker + B-Human supports) into ONE learned policy.

Inputs: probe match JSON+NPZ pairs from run_probe_match.py (any controller). Per row
(one field robot, one step) the label source is chosen by who controlled the robot:
  * ctrl == "learned" (striker under the DAgger student): expert label, macro bits baked
    exactly as rlfootball/training/train_dagger_v5.py::bake (Design-A, simulation-only).
  * ctrl == "bhuman" (support handed to B-Human, or the whole team in the bhuman arm):
    B-Human's executed SkillRequest. A shoot/pass issued outside the macro condition is
    B-Human's approach-and-kick macro at range -> translated to `walk` toward the ball,
    because our gate cannot legalise a kick there.
  Labels still illegal under the (baked) gate mask are dropped (Design-A).

Position labels (support walk rows only): the policy can only move a support target
+-residual_y_range (450 mm) sideways and +-residual_theta_range in heading around the
coordinator anchor (residual_x_range = 0 by contract). B-Human's walk target (robot-relative
MotionRequest.walkTarget, mapped to the field with the robot's own pose estimate) is
converted to those residuals and clipped; the clipped share is reported, since it measures
how much of B-Human's positioning the current action contract cannot express.

Loss: masked CE(skill) + param_w * MSE(y/theta residual, support walk rows)
      + kl_w * KL(anchor || student) on all rows (anchor = initial checkpoint).
Usage:
  python -m rlfootball.training.train_distill_hybrid --json "runs/.../HD_*.json" --init runs/dagger_v5/round2.pt \
      --out runs/distill_hybrid/round1.pt
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

from rlfootball.core.action import SKILL_ORDER
from rlfootball.core.config import ACTION, ROOT_DIR
from rlfootball.core.gate import gate_skill_mask_from_obs
from rlfootball.training.train_dagger_v5 import bake, Policy

BH_TO_IDX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
WALK, SHOOT, PASS = SKILL_ORDER.index("walk"), SKILL_ORDER.index("shoot"), SKILL_ORDER.index("pass")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_match(jpath):
    J = json.load(open(jpath))
    if not J["meta"]["valid"]:
        return None
    # Load every npz array ONCE. Indexing the lazy NpzFile per row (z["obs"][i]) re-decompresses
    # the whole array each time and the row view pins that full copy -> ~20 GB for one match
    # (the 2026-10-06 OOM). Rows below are copied out of these in-memory arrays.
    with np.load(jpath[:-5] + ".npz") as zf:
        z = {k: zf[k] for k in zf.files}
    S = J["steps"]
    rows = {k: [] for k in ("obs", "label", "macro", "role", "ctrl_bh", "res_y", "res_th", "has_param", "clipped")}
    for i in range(len(z["step"])):
        step, pid = int(z["step"][i]), int(z["player"][i])
        rec = S[step - 1]["agents"].get(str(pid))
        if rec is None:
            continue
        macro = bool(z["macro"][i])
        role = int(z["role"][i])
        # Label source. Striker driven by the student -> expert (DAgger as before). Support
        # driven by the student -> B-Human SHADOW label (what B-Human would do on the
        # student's own state; real DAgger). Robot driven by B-Human -> its executed request.
        target_field = None
        if rec.get("ctrl", "learned") == "learned" and role == 0:
            lab, ctrl_bh = int(z["expert"][i]), False
        elif rec.get("ctrl", "learned") == "learned":
            if rec.get("shadow_skill", -1) <= 0:
                lab, ctrl_bh = int(z["expert"][i]), False   # pre-shadow data: expert fallback
            else:
                ctrl_bh = True
                lab = BH_TO_IDX.get(int(rec["shadow_skill"]), -1)
                target_field = (rec["shadow_tx"], rec["shadow_ty"], rec["shadow_tth"])
        else:
            ctrl_bh = True
            lab = BH_TO_IDX.get(int(rec["bh_skill"]), -1)
            if "bh_wt_x" in rec:
                c, s_ = math.cos(rec["th"]), math.sin(rec["th"])
                target_field = (rec["x"] + c * rec["bh_wt_x"] - s_ * rec["bh_wt_y"],
                                rec["y"] + s_ * rec["bh_wt_x"] + c * rec["bh_wt_y"],
                                wrap(rec["th"] + rec["bh_wt_th"]))
        if ctrl_bh and lab in (SHOOT, PASS) and not macro:
            lab = WALK  # approach-and-kick at range -> walk to the ball
        res_y = res_th = 0.0
        has_param = clipped = False
        if ctrl_bh and lab == WALK and role in (1, 2) and target_field is not None and "coord_x" in rec:
            fx, fy, fth = target_field
            cx = rec.get("coord_used_x", rec["coord_x"]); cy = rec.get("coord_used_y", rec["coord_y"])
            anchor_th = math.atan2(cy - rec["y"], cx - rec["x"])
            ry = (fy - cy) / ACTION.residual_y_range
            rt = wrap(fth - anchor_th) / ACTION.residual_theta_range
            clipped = abs(ry) > 1 or abs(rt) > 1
            res_y, res_th = float(np.clip(ry, -1, 1)), float(np.clip(rt, -1, 1))
            has_param = True
        for k, v in (("obs", z["obs"][i].copy()), ("label", lab), ("macro", macro), ("role", role), ("ctrl_bh", ctrl_bh),
                     ("res_y", res_y), ("res_th", res_th), ("has_param", has_param), ("clipped", clipped)):
            rows[k].append(v)
    return {k: np.asarray(v) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True, help="probe match JSON files / globs (npz alongside)")
    ap.add_argument("--init", default="runs/dagger_v5/round2.pt", help="initialisation and KL anchor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--kl-w", type=float, default=0.2)
    ap.add_argument("--param-w", type=float, default=1.0)
    args = ap.parse_args()
    rp = lambda p: str(p if Path(p).is_absolute() else ROOT_DIR / p)  # noqa: E731
    files = sorted({f for g in args.json for f in glob.glob(rp(g))})
    parts = [m for m in (load_match(f) for f in files) if m is not None and len(m["label"])]
    if not parts:
        raise SystemExit("no valid matches")
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    match_id = np.concatenate([np.full(len(p["label"]), i) for i, p in enumerate(parts)])
    obs = bake(d["obs"].astype(np.float32), d["macro"].astype(bool) & ~d["ctrl_bh"].astype(bool),
               np.where(d["label"] < 0, 0, d["label"]).astype(int))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    O = torch.as_tensor(obs, device=dev)
    Y = torch.as_tensor(np.where(d["label"] < 0, 0, d["label"]), device=dev)
    M = gate_skill_mask_from_obs(O)
    legal = (M[torch.arange(len(Y), device=dev), Y] > 0.5) & torch.as_tensor(d["label"] >= 0, device=dev)
    bh = torch.as_tensor(d["ctrl_bh"].astype(bool), device=dev)
    hp = torch.as_tensor(d["has_param"].astype(bool), device=dev)
    PT = torch.as_tensor(np.stack([d["res_y"], d["res_th"]], 1).astype(np.float32), device=dev)
    print(f"rows {len(Y)} from {len(parts)} matches | B-Human-labelled {bh.float().mean() * 100:.1f}% | "
          f"legal {legal.float().mean() * 100:.1f}% (B-Human rows {legal[bh].float().mean() * 100:.1f}%) | "
          f"support position labels {hp.sum().item()} (clipped by contract: {d['clipped'][d['has_param']].mean() * 100:.1f}%)",
          flush=True)
    print("B-Human support label mix:", {SKILL_ORDER[k]: round(100 * float(v), 1) for k, v in
          zip(*np.unique(d["label"][d["ctrl_bh"] & (d["label"] >= 0)], return_counts=True))} if bh.any() else {}, flush=True)

    ck = torch.load(rp(args.init), map_location="cpu")
    anchor = Policy(ck["policy_state_dict"]).to(dev).eval()
    student = Policy(ck["policy_state_dict"]).to(dev)
    with torch.no_grad():
        alog, _ = anchor(O)
        a_logp = F.log_softmax(alog + (M - 1.0) * 1e9, 1)
    hold = len(parts) - 1
    tr = torch.as_tensor(match_id != hold, device=dev)
    va = (~tr) & legal

    def ev():
        with torch.no_grad():
            lg, pm = student(O[va])
            pred = (lg + (M[va] - 1.0) * 1e9).argmax(1)
            out = {"agree": (pred == Y[va]).float().mean().item()}
            for nm, sel in (("bh_rows", bh[va]), ("learned_rows", ~bh[va])):
                out[nm] = (pred[sel] == Y[va][sel]).float().mean().item() if sel.any() else float("nan")
            sp = hp[va]
            out["pos_mse"] = ((pm[sp][:, 1:3] - PT[va][sp]) ** 2).mean().item() if sp.any() else float("nan")
            return {k: round(100 * v, 1) if k != "pos_mse" else round(v, 4) for k, v in out.items()}

    print("before:", ev(), flush=True)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    idx = torch.nonzero(tr & legal).squeeze(1)
    for ep in range(args.epochs):
        perm = idx[torch.randperm(len(idx), device=dev)]
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            lg, pm = student(O[b])
            logp = F.log_softmax(lg + (M[b] - 1.0) * 1e9, 1)
            loss = F.nll_loss(logp, Y[b]) + args.kl_w * (a_logp[b].exp() * (a_logp[b] - logp)).sum(1).mean()
            sel = hp[b]
            if sel.any():
                loss = loss + args.param_w * ((pm[sel][:, 1:3] - PT[b][sel]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {ep + 1}: val", ev(), flush=True)
    out = Path(rp(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    new = copy.copy(ck)
    new["policy_state_dict"] = student.export_into(ck["policy_state_dict"])
    new["distill_hybrid"] = {"files": files, "init": args.init, "epochs": args.epochs, "kl_w": args.kl_w,
                             "param_w": args.param_w, "val_after": ev()}
    torch.save(new, out)
    json.dump(new["distill_hybrid"], open(out.with_suffix(".json"), "w"), indent=1, default=str)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
