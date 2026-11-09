"""Learned support coordinator: predicts where a support robot should go, imitating B-Human.

Why: the policy contract lets a support move only +-450 mm sideways (and 0 mm along x)
around the RULE-BASED coordinator's target; 96.6% of B-Human's support positions fall
outside that band (runs/distill_hybrid/round1.json, 2026-10-06). Replacing the rule-based
coordination target with a learned one keeps the whole team learned and leaves the 47-dim
policy contract untouched (the policy's residual is still applied on top).

Model: 47-dim encoded obs (role one-hot included) -> normalised field target (x/4500, y/3000).
Labels: B-Human support walk targets in field coordinates, from
  * shadow labels (shadow_bh_* = SkillRequest.target, field frame) when present, else
  * the executed MotionRequest walk target (robot-relative bh_wt_*) mapped with the robot's pose.

Usage:
  python tools/learned_coordinator.py --json "runs/.../*.json" --out runs/learned_coord/coord_v1.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
X_RANGE, Y_RANGE = 4500.0, 3000.0
BH_WALK = 2          # SkillRequest::Type walk
SUPPORT_ROLES = (1, 2)


class CoordNet(torch.nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(47, hidden), torch.nn.ELU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.ELU(),
                                       torch.nn.Linear(hidden, 2), torch.nn.Tanh())

    def forward(self, o):
        return self.net(o)


def load_coord_net(path: str | Path, device: str = "cpu") -> CoordNet:
    ck = torch.load(path, map_location=device)
    net = CoordNet(ck.get("hidden", 128))
    net.load_state_dict(ck["state_dict"])
    return net.eval()


def predict_field_target(net: CoordNet, encoded: np.ndarray) -> tuple[float, float]:
    with torch.no_grad():
        x, y = net(torch.as_tensor(encoded, dtype=torch.float32).reshape(1, -1))[0].tolist()
    return x * X_RANGE, y * Y_RANGE


def field_target(rec: dict) -> tuple[float, float] | None:
    """B-Human's intended support position for this row, field frame, or None."""
    if rec.get("shadow_skill", -1) == BH_WALK:
        return rec["shadow_tx"], rec["shadow_ty"]
    if rec.get("ctrl") == "bhuman" and rec.get("bh_skill") == BH_WALK and "bh_wt_x" in rec:
        c, s = math.cos(rec["th"]), math.sin(rec["th"])
        return rec["x"] + c * rec["bh_wt_x"] - s * rec["bh_wt_y"], rec["y"] + s * rec["bh_wt_x"] + c * rec["bh_wt_y"]
    return None


def load_rows(jpath: str):
    J = json.load(open(jpath))
    if not J["meta"]["valid"]:
        return None
    with np.load(jpath[:-5] + ".npz") as zf:
        z = {k: zf[k] for k in ("obs", "step", "player", "role")}
    S = J["steps"]
    X, Y = [], []
    for i in range(len(z["step"])):
        if int(z["role"][i]) not in SUPPORT_ROLES:
            continue
        rec = S[int(z["step"][i]) - 1]["agents"].get(str(int(z["player"][i])))
        if rec is None:
            continue
        t = field_target(rec)
        if t is None:
            continue
        X.append(z["obs"][i].copy())
        Y.append((np.clip(t[0] / X_RANGE, -1, 1), np.clip(t[1] / Y_RANGE, -1, 1)))
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32)) if X else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="continue from an existing coordinator")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    rp = lambda p: str(p if Path(p).is_absolute() else REPO / p)  # noqa: E731
    files = sorted({f for g in args.json for f in glob.glob(rp(g))})
    parts = [r for r in (load_rows(f) for f in files) if r is not None]
    if not parts:
        raise SystemExit("no support-position rows")
    X = np.concatenate([p[0] for p in parts]); Y = np.concatenate([p[1] for p in parts])
    mid = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(parts)])
    hold = mid == len(parts) - 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt, Yt = torch.as_tensor(X, device=dev), torch.as_tensor(Y, device=dev)
    tr, va = torch.as_tensor(~hold, device=dev), torch.as_tensor(hold, device=dev)
    net = (load_coord_net(rp(args.init), dev) if args.init else CoordNet()).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def err_mm(sel):
        with torch.no_grad():
            d = (net(Xt[sel]) - Yt[sel]) * torch.tensor([X_RANGE, Y_RANGE], device=dev)
            return float(d.norm(dim=1).median())
    base = float(((Yt[va] - Yt[tr].mean(0)) * torch.tensor([X_RANGE, Y_RANGE], device=dev)).norm(dim=1).median())
    print(f"rows {len(X)} from {len(parts)} matches | val median error: predict-mean baseline {base:.0f} mm, before {err_mm(va):.0f} mm", flush=True)
    idx = torch.nonzero(tr).squeeze(1)
    for ep in range(args.epochs):
        perm = idx[torch.randperm(len(idx), device=dev)]
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]
            loss = F.smooth_l1_loss(net(Xt[b]), Yt[b], beta=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep + 1}: val median error {err_mm(va):.0f} mm", flush=True)
    out = Path(rp(args.out)); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.cpu().state_dict(), "hidden": 128, "files": files,
                "val_median_err_mm": err_mm(va) if False else None}, out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
