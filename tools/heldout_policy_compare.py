"""Held-out comparison of the policy lineage (paper, claims 3 and 5).

Held-out data: runs/full_probe (2026-10-03, 37 live 4v4 matches vs real B-Human, estimated observations), used by
none of the compared policies. Two label sets:
  expert : every field-robot step labelled by the expert (teacher + B-Human macro), all 37 matches
  bhuman : B-Human's own executed skill on the 8 matches where B-Human drove our seat (C_*); shoot/pass issued outside
           the macro condition -> walk, exactly as tools/train_distill_hybrid.py translates them
Same bake + Design-A mask as training. Metrics per policy: agreement (all / per role), per-skill recall, balanced
accuracy, attack subset (striker rows) and defence subset (rows labelled block/mark). 95% CIs by bootstrap over
MATCHES (2000 resamples); paired differences (e.g. v5 - v4.2) use the same resampled matches.
Output: runs/paper_offline/heldout_policies/{summary.json, table.md}
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools"))
from train import gate_skill_mask_from_obs  # noqa: E402
from action import SKILL_ORDER  # noqa: E402
from train_dagger_v5 import bake, Policy  # noqa: E402

MODELS = {"v4.2": "runs/multiagent_v4_2/checkpoint_stage7_step12000.pt",
          "v5": "runs/merge_v5_attack_defense/checkpoint_stage8_step10000.pt",
          "I (imitation)": "runs/distill_hybrid/round3.pt",
          "G (final team)": "runs/kickstart/loop_20261006_224436/iter_12/policy.pt"}
PAIRS = [("v5", "v4.2"), ("G (final team)", "v5"), ("G (final team)", "I (imitation)")]
SK = ["stand", "walk", "shoot", "pass", "dribble", "block", "mark"]
WALK, SHOOT, PASS = (SKILL_ORDER.index(s) for s in ("walk", "shoot", "pass"))
ROLES = {0: "striker", 1: "open_support", 2: "off_ball"}
B = 2000


def load(label_set):
    O, Y, R, Mid = [], [], [], []
    files = sorted(glob.glob(str(REPO / "runs/full_probe/**/*.npz"), recursive=True))
    if label_set == "bhuman":
        files = [f for f in files if Path(f).name.startswith("C_")]
    for i, f in enumerate(files):
        z = np.load(f)
        mac = z["macro"].astype(bool)
        if label_set == "expert":
            y = z["expert"].astype(int); ok = z["legal_expert"].astype(bool) & (y >= 0)
            o = bake(z["obs"].astype(np.float32), mac, np.where(y < 0, 0, y))
        else:
            y = z["bh"].astype(int) - 1                      # B-Human enum 1..8 -> SKILL_ORDER index
            y = np.where(((y == SHOOT) | (y == PASS)) & ~mac, WALK, y)
            ok = (y >= 0) & (y < len(SKILL_ORDER))
            o = bake(z["obs"].astype(np.float32), mac & ((y == SHOOT) | (y == PASS)), np.where(ok, y, 0))
        O.append(o[ok]); Y.append(y[ok]); R.append(z["role"].astype(int)[ok]); Mid.append(np.full(ok.sum(), i))
    return np.concatenate(O), np.concatenate(Y), np.concatenate(R), np.concatenate(Mid), len(files)


def per_match_stats(pred, y, r, mid, n_match):
    """sufficient statistics per match so bootstrap over matches is cheap"""
    S = {}
    for m in range(n_match):
        s = mid == m
        d = {"n": s.sum(), "hit": (pred[s] == y[s]).sum()}
        for k, nm in ROLES.items():
            t = s & (r == k); d[f"n_{nm}"] = t.sum(); d[f"hit_{nm}"] = (pred[t] == y[t]).sum()
        for sk in SK:
            k = SKILL_ORDER.index(sk); t = s & (y == k)
            d[f"n_{sk}"] = t.sum(); d[f"hit_{sk}"] = (pred[t] == k).sum()
        t = s & ((y == SKILL_ORDER.index("block")) | (y == SKILL_ORDER.index("mark")))
        d["n_def"] = t.sum(); d["hit_def"] = (pred[t] == y[t]).sum()
        S[m] = d
    return S


def metrics(S, idx):
    tot = lambda k: sum(S[i][k] for i in idx)  # noqa: E731
    r = {"agree": tot("hit") / max(tot("n"), 1)}
    for nm in ROLES.values():
        r[f"agree_{nm}"] = tot(f"hit_{nm}") / tot(f"n_{nm}") if tot(f"n_{nm}") else np.nan
    rec = []
    for sk in SK:
        n = tot(f"n_{sk}"); r[f"recall_{sk}"] = tot(f"hit_{sk}") / n if n else np.nan
        if n:
            rec.append(r[f"recall_{sk}"])
    r["balanced_acc"] = float(np.mean(rec))
    r["defence_agree"] = tot("hit_def") / tot("n_def") if tot("n_def") else np.nan
    return r


def main():
    out = REPO / "runs/paper_offline/heldout_policies"; out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    summary, md = {}, []
    for ls in ("expert", "bhuman"):
        O, Y, R, Mid, nm_ = load(ls)
        Ot = torch.as_tensor(O, device=dev); M = gate_skill_mask_from_obs(Ot)
        boots = [rng.integers(0, nm_, nm_) for _ in range(B)]
        S = {}
        for name, p in MODELS.items():
            ck = torch.load(REPO / p, map_location="cpu", weights_only=False)
            with torch.no_grad():
                lg, _ = Policy(ck["policy_state_dict"]).to(dev).eval()(Ot)
                pred = (lg + (M - 1.0) * 1e9).argmax(1).cpu().numpy()
            S[name] = per_match_stats(pred, Y, R, Mid, nm_)
        res = {"n_matches": nm_, "n_rows": int(len(Y)),
               "label_mix": {SKILL_ORDER[k]: int((Y == k).sum()) for k in range(len(SKILL_ORDER)) if (Y == k).any()},
               "models": {}, "paired": {}}
        keys = list(metrics(S["v5"], range(nm_)))
        for name in MODELS:
            point = metrics(S[name], range(nm_)); bs = [metrics(S[name], b) for b in boots]
            res["models"][name] = {k: [point[k], float(np.nanpercentile([x[k] for x in bs], 2.5)),
                                       float(np.nanpercentile([x[k] for x in bs], 97.5))] for k in keys}
        for a, b in PAIRS:
            pa, pb = metrics(S[a], range(nm_)), metrics(S[b], range(nm_))
            d = [{k: metrics(S[a], bb)[k] - metrics(S[b], bb)[k] for k in keys} for bb in boots[:500]]
            res["paired"][f"{a} - {b}"] = {k: [pa[k] - pb[k], float(np.nanpercentile([x[k] for x in d], 2.5)),
                                               float(np.nanpercentile([x[k] for x in d], 97.5))] for k in keys}
        summary[ls] = res
        show = ["balanced_acc", "agree", "agree_striker", "agree_open_support", "agree_off_ball", "recall_shoot",
                "recall_pass", "recall_dribble", "recall_block", "recall_mark", "defence_agree"]
        f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"  # noqa: E731
        md += [f"### Labels: {ls} ({nm_} matches, {len(Y)} rows; mix {res['label_mix']})", "",
               "| metric | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
        md += [f"| {k} | " + " | ".join(f(res['models'][m][k]) for m in MODELS) + " |" for k in show]
        md += ["", "| paired difference | " + " | ".join(res["paired"]) + " |", "|---|" + "---|" * len(res["paired"])]
        md += [f"| {k} | " + " | ".join(f(res['paired'][p][k]) for p in res["paired"]) + " |" for k in show] + [""]
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    open(out / "table.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
