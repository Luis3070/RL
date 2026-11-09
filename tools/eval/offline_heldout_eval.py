"""Held-out offline evaluation of the imitation (DAgger / distillation) stage, for the paper.

Same data, initialisation, loss and hyper-parameters as rlfootball/training/train_distill_hybrid.py (round 3),
but evaluated with a GROUPED split: the matches (not rows) are split into K folds, so no step of a
held-out match is ever seen in training. For every fold x seed a fresh student is trained from the
init checkpoint and scored on the held-out matches only.
Metrics (held-out, legal rows): skill agreement (all / B-Human-labelled rows / expert-labelled rows),
per-skill recall + precision, macro-F1 over skills present, support-position MSE; plus the same for
the untrained init ("before") and a majority-class baseline. Per-epoch train loss and held-out
agreement are logged for the loss-curve figure.
Outputs: <out>/runs.csv (one row per fold x seed), <out>/curves.csv, <out>/summary.json
Usage: python tools/eval/offline_heldout_eval.py --files-from runs/distill_hybrid/round3.json --out runs/paper_offline/distill
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from rlfootball.core.gate import gate_skill_mask_from_obs  # noqa: E402
from rlfootball.core.action import SKILL_ORDER  # noqa: E402
from rlfootball.training.train_dagger_v5 import bake, Policy  # noqa: E402
from rlfootball.training.train_distill_hybrid import load_match  # noqa: E402


def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return [float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")]
    from scipy import stats
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-from", required=True, help="json with 'files' and 'init' (e.g. runs/distill_hybrid/round3.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=4); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=2048); ap.add_argument("--kl-w", type=float, default=0.2)
    ap.add_argument("--param-w", type=float, default=1.0)
    a = ap.parse_args()
    rp = lambda p: Path(p) if Path(p).is_absolute() else REPO / p  # noqa: E731
    spec = json.load(open(rp(a.files_from)))
    out = rp(a.out); out.mkdir(parents=True, exist_ok=True)
    parts, names = [], []
    for f in spec["files"]:
        m = load_match(f)
        if m is not None and len(m["label"]):
            parts.append(m); names.append(Path(f).name)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    mid = np.concatenate([np.full(len(p["label"]), i) for i, p in enumerate(parts)])
    lab = np.where(d["label"] < 0, 0, d["label"]).astype(int)
    obs = bake(d["obs"].astype(np.float32), d["macro"].astype(bool) & ~d["ctrl_bh"].astype(bool), lab)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    O = torch.as_tensor(obs, device=dev); Y = torch.as_tensor(lab, device=dev)
    M = gate_skill_mask_from_obs(O)
    legal = (M[torch.arange(len(Y), device=dev), Y] > 0.5) & torch.as_tensor(d["label"] >= 0, device=dev)
    bh = torch.as_tensor(d["ctrl_bh"].astype(bool), device=dev)
    hp = torch.as_tensor(d["has_param"].astype(bool), device=dev)
    PT = torch.as_tensor(np.stack([d["res_y"], d["res_th"]], 1).astype(np.float32), device=dev)
    ck = torch.load(rp(spec["init"]), map_location="cpu")
    with torch.no_grad():
        anchor = Policy(ck["policy_state_dict"]).to(dev).eval()
        alog, _ = anchor(O)
        a_logp = F.log_softmax(alog + (M - 1.0) * 1e9, 1)
    # fold assignment: matches shuffled once with a fixed seed, balanced across normal/swapped sides
    rng = np.random.default_rng(0)
    order = rng.permutation(len(parts)); fold_of = np.empty(len(parts), int)
    fold_of[order] = np.arange(len(parts)) % a.folds
    print(f"{len(parts)} matches, {len(Y)} rows, legal {legal.float().mean().item() * 100:.1f}% | folds:",
          {k: [names[i] for i in np.flatnonzero(fold_of == k)] for k in range(a.folds)}, flush=True)
    fold_t = torch.as_tensor(fold_of[mid], device=dev)

    def score(net, va):
        with torch.no_grad():
            lg, pm = net(O[va])
            pred = (lg + (M[va] - 1.0) * 1e9).argmax(1)
        return metrics(pred, va, pm)

    def metrics(pred, va, pm=None):
        y = Y[va]; r = {"agree": (pred == y).float().mean().item()}
        for nm, sel in (("agree_bh_rows", bh[va]), ("agree_expert_rows", ~bh[va])):
            r[nm] = (pred[sel] == y[sel]).float().mean().item() if sel.any() else float("nan")
        f1s = []
        for k, s in enumerate(SKILL_ORDER):
            tp = ((pred == k) & (y == k)).sum().item(); n_true = (y == k).sum().item(); n_pred = (pred == k).sum().item()
            rec = tp / n_true if n_true else float("nan"); prec = tp / n_pred if n_pred else float("nan")
            r[f"recall_{s}"], r[f"precision_{s}"], r[f"n_{s}"] = rec, prec, n_true
            if n_true:
                f1s.append(0.0 if not tp else 2 * rec * prec / (rec + prec))
        r["macro_f1"] = float(np.mean(f1s))
        if pm is not None:
            sp = hp[va]
            r["pos_mse"] = ((pm[sp][:, 1:3] - PT[va][sp]) ** 2).mean().item() if sp.any() else float("nan")
        return r

    runs, curves = [], []
    for k in range(a.folds):
        va = (fold_t == k) & legal; tr = (fold_t != k) & legal
        maj = torch.bincount(Y[tr], minlength=len(SKILL_ORDER)).argmax()
        base = {"fold": k, "seed": -1, "n_val_rows": int(va.sum().item()), "n_val_matches": int((fold_of == k).sum())}
        runs.append({**base, "model": "majority", **metrics(torch.full_like(Y[va], maj), va)})
        runs.append({**base, "model": "init_before", **score(anchor, va)})
        idx = torch.nonzero(tr).squeeze(1)
        for s in range(a.seeds):
            torch.manual_seed(1000 + s)
            student = Policy(ck["policy_state_dict"]).to(dev)
            opt = torch.optim.Adam(student.parameters(), lr=a.lr)
            for ep in range(a.epochs):
                perm = idx[torch.randperm(len(idx), device=dev)]; tot = n = 0
                for i in range(0, len(perm), a.batch):
                    b = perm[i:i + a.batch]
                    lg, pm = student(O[b])
                    logp = F.log_softmax(lg + (M[b] - 1.0) * 1e9, 1)
                    ce = F.nll_loss(logp, Y[b]); kl = (a_logp[b].exp() * (a_logp[b] - logp)).sum(1).mean()
                    loss = ce + a.kl_w * kl
                    sel = hp[b]
                    if sel.any():
                        loss = loss + a.param_w * ((pm[sel][:, 1:3] - PT[b][sel]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item() * len(b); n += len(b)
                with torch.no_grad():
                    lg, _ = student(O[va]); vl = F.nll_loss(F.log_softmax(lg + (M[va] - 1.0) * 1e9, 1), Y[va]).item()
                curves.append({"fold": k, "seed": s, "epoch": ep + 1, "train_loss": tot / n, "val_ce": vl,
                               "val_agree": score(student, va)["agree"]})
            r = {**base, "seed": s, "model": "student", **score(student, va)}
            runs.append(r)
            print(f"fold {k} seed {s}: agree {r['agree'] * 100:.1f}  bh {r['agree_bh_rows'] * 100:.1f}  "
                  f"expert {r['agree_expert_rows'] * 100:.1f}  macroF1 {r['macro_f1']:.3f}", flush=True)
    keys = sorted({k for r in runs for k in r})
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(runs)
    with open(out / "curves.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    summ = {"split": f"grouped {a.folds}-fold by match ({len(parts)} matches, {len(Y)} rows); "
                     f"{a.seeds} seeds per fold; CI = 95% t-interval over fold means (seed-averaged)",
            "files": spec["files"], "init": spec["init"], "models": {}}
    for model in ("majority", "init_before", "student"):
        rs = [r for r in runs if r["model"] == model]
        summ["models"][model] = {}
        for key in [k for k in keys if k not in ("fold", "seed", "model") and not k.startswith("n_")]:
            fold_means = [np.nanmean([r.get(key, np.nan) for r in rs if r["fold"] == f]) for f in range(a.folds)]
            seed_sd = (float(np.nanmean([np.nanstd([r.get(key, np.nan) for r in rs if r["fold"] == f], ddof=1)
                                         for f in range(a.folds)])) if model == "student" and a.seeds > 1 else None)
            summ["models"][model][key] = {"mean_ci95": ci95(fold_means), "within_fold_seed_sd": seed_sd}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    for model in ("majority", "init_before", "student"):
        m = summ["models"][model]
        print(model, {k: [round(v, 3) for v in m[k]["mean_ci95"]] for k in ("agree", "agree_bh_rows", "agree_expert_rows", "macro_f1")})


if __name__ == "__main__":
    main()
