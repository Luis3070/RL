"""Held-out offline evaluation of the training-ablation checkpoints (paper).

Held-out set: the 2026-10-03 full-probe corpus (runs/full_probe, live 4v4 matches vs real B-Human,
estimated observations, every field-robot step labelled by the expert = teacher + B-Human macro).
No ablation arm, nor the v5 recipe, ever trained on it. Macro states are baked into the gate bits
exactly as in rlfootball/training/train_dagger_v5.py; the Design-A mask (gate_skill_mask_from_obs) is applied.
For every run (arm x seed) and every saved checkpoint (steps 0..10000): skill agreement with the
expert (all rows / per role), per-skill recall for walk/shoot/pass/dribble/block/mark, and the entropy
of the predicted skill distribution (collapse indicator). Aggregates mean +/- 95% t-CI over seeds.
Outputs: <out>/per_checkpoint.csv, <out>/summary.json, printed table at the final step.
"""
import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from rlfootball.core.gate import gate_skill_mask_from_obs  # noqa: E402
from rlfootball.core.action import SKILL_ORDER  # noqa: E402
from rlfootball.training.train_dagger_v5 import bake, Policy  # noqa: E402
from scipy import stats  # noqa: E402

ROLES = {0: "striker", 1: "open_support", 2: "off_ball"}
SK = ["stand", "walk", "shoot", "pass", "dribble", "block", "mark"]


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return [float(x.mean()), float(h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/ablation_paper_*_s*")
    ap.add_argument("--heldout", default="runs/full_probe/**/*.npz")
    ap.add_argument("--extra", nargs="*", default=["V5=runs/merge_v5_attack_defense/checkpoint_stage8_step10000.pt"])
    ap.add_argument("--out", default="runs/paper_offline/ablation")
    a = ap.parse_args()
    out = REPO / a.out; out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(REPO / a.heldout), recursive=True))
    parts = [np.load(f) for f in files]
    obs = np.concatenate([p["obs"] for p in parts]).astype(np.float32)
    exp = np.concatenate([p["expert"] for p in parts]).astype(int)
    mac = np.concatenate([p["macro"] for p in parts]).astype(bool)
    role = np.concatenate([p["role"] for p in parts]).astype(int)
    legal = np.concatenate([p["legal_expert"] for p in parts]).astype(bool)
    keep = legal & (exp >= 0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    O = torch.as_tensor(bake(obs, mac, exp)[keep], device=dev)
    Y = torch.as_tensor(exp[keep], device=dev); R = role[keep]
    M = gate_skill_mask_from_obs(O)
    print(f"held-out: {len(files)} matches, {len(Y)} legal expert-labelled rows; label mix",
          {SKILL_ORDER[k]: int((Y == k).sum()) for k in range(len(SKILL_ORDER)) if (Y == k).any()}, flush=True)

    def score(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = Policy(ck["policy_state_dict"]).to(dev).eval()
        with torch.no_grad():
            lg, _ = net(O)
            pred = (lg + (M - 1.0) * 1e9).argmax(1)
        r = {"agree": (pred == Y).float().mean().item()}
        for k, nm in ROLES.items():
            s = torch.as_tensor(R == k, device=dev)
            r[f"agree_{nm}"] = (pred[s] == Y[s]).float().mean().item()
        for s in SK:
            k = SKILL_ORDER.index(s); n = (Y == k).sum().item()
            r[f"recall_{s}"] = ((pred == k) & (Y == k)).sum().item() / n if n else float("nan")
        r["balanced_acc"] = float(np.nanmean([r[f"recall_{s}"] for s in SK]))
        p = torch.bincount(pred, minlength=len(SKILL_ORDER)).float(); p = p / p.sum()
        r["pred_entropy_bits"] = float(-(p[p > 0] * torch.log2(p[p > 0])).sum())
        return r

    rows = []
    for d in sorted(glob.glob(str(REPO / a.runs))):
        m = re.search(r"ablation_paper_([A-Z]+)_s(\d+)$", d)
        if not m:
            continue
        for ckp in sorted(glob.glob(d + "/checkpoint_stage8_step*.pt")):
            step = int(re.search(r"step(\d+)", ckp).group(1))
            rows.append({"arm": m.group(1), "seed": int(m.group(2)), "step": step, **score(ckp)})
    for e in a.extra:
        lab, p = e.split("=", 1)
        rows.append({"arm": lab, "seed": 0, "step": -1, **score(str(REPO / p))})
    keys = list(rows[0])
    with open(out / "per_checkpoint.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summ = {"heldout": f"{len(files)} live 4v4 matches vs B-Human (runs/full_probe), {len(Y)} rows", "arms": {}}
    metrics = [k for k in keys if k not in ("arm", "seed", "step")]
    for arm in sorted({r["arm"] for r in rows}):
        summ["arms"][arm] = {}
        for step in sorted({r["step"] for r in rows if r["arm"] == arm}):
            rs = [r for r in rows if r["arm"] == arm and r["step"] == step]
            summ["arms"][arm][step] = {k: ci([r[k] for r in rs]) for k in metrics} | {"n": len(rs)}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    show = ["balanced_acc", "agree", "agree_striker", "agree_open_support", "agree_off_ball", "recall_shoot", "recall_pass",
            "recall_dribble", "recall_block", "recall_mark", "pred_entropy_bits"]
    print("| arm (final step, mean ± 95% CI over seeds) | " + " | ".join(show) + " |")
    for arm, by in summ["arms"].items():
        st = max(by); s = by[st]
        print(f"| {arm} step {st} n={s['n']} | " + " | ".join(f"{s[k][0]:.3f}±{s[k][1]:.3f}" for k in show) + " |")
    print("step-0 (warm start / random init):")
    for arm, by in summ["arms"].items():
        if 0 in by:
            print(f"  {arm}: agree {by[0]['agree'][0]:.3f}  entropy {by[0]['pred_entropy_bits'][0]:.2f}")


if __name__ == "__main__":
    main()
