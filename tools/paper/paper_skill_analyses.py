"""Interpretable skill analyses for the final paper figures (2026-10-10).

1. confusion : for each training-ablation variant at 10k steps (and PPO alone at 30k), the row-normalised confusion
               matrix expert-skill -> chosen skill on the held-out live corpus (runs/full_probe), averaged over seeds.
2. share     : share of each chosen skill on the held-out states at every saved checkpoint (seed mean), per variant.
3. roles     : executed-skill share per role in the evaluation matches for the final team (G) and stock B-Human in
               the same seat (C). B-Human's shoot/pass requests issued outside the kick condition (macro) are
               approach-and-kick commands and are counted as walk, the same translation used for training labels.
Output: runs/paper_offline/skills/{confusion.json, share.json, roles.json}
"""
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from rlfootball.core.gate import gate_skill_mask_from_obs  # noqa: E402
from rlfootball.core.action import SKILL_ORDER  # noqa: E402
from rlfootball.training.train_dagger_v5 import bake, Policy  # noqa: E402

SK = ["stand", "walk", "shoot", "pass", "dribble", "block", "mark"]
K = [SKILL_ORDER.index(s) for s in SK]


def main():
    out = REPO / "runs/paper_offline/skills"; out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(REPO / "runs/full_probe/**/*.npz"), recursive=True))
    P = [np.load(f) for f in files]
    exp = np.concatenate([p["expert"] for p in P]).astype(int)
    keep = np.concatenate([p["legal_expert"] for p in P]).astype(bool) & (exp >= 0)
    obs = bake(np.concatenate([p["obs"] for p in P]).astype(np.float32), np.concatenate([p["macro"] for p in P]).astype(bool), exp)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    O = torch.as_tensor(obs[keep], device=dev); Y = exp[keep]; M = gate_skill_mask_from_obs(O)

    def predict(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            lg, _ = Policy(ck["policy_state_dict"]).to(dev).eval()(O)
            return (lg + (M - 1.0) * 1e9).argmax(1).cpu().numpy()

    conf, share = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for d in sorted(glob.glob(str(REPO / "runs/ablation_paper_*_s*"))):
        m = re.search(r"ablation_paper_([A-Z]+)_s(\d+)$", d)
        if not m:
            continue
        arm = m.group(1)
        for ckp in sorted(glob.glob(d + "/checkpoint_stage8_step*.pt")):
            step = int(re.search(r"step(\d+)", ckp).group(1))
            pred = predict(ckp)
            share[arm][step].append([float((pred == k).mean()) for k in range(len(SKILL_ORDER))])
            if step == (30000 if arm == "PPOLONG" else 10000):
                C = np.zeros((len(SK), len(SKILL_ORDER)))
                for i, k in enumerate(K):
                    r = pred[Y == k]
                    C[i] = np.bincount(r, minlength=len(SKILL_ORDER)) / max(len(r), 1)
                conf[arm].append(C.tolist())
        print("done", d, flush=True)
    json.dump({"rows": SK, "cols": list(SKILL_ORDER), "n_rows": {s: int((Y == k).sum()) for s, k in zip(SK, K)},
               "arms": {a: np.mean(v, 0).tolist() for a, v in conf.items()}, "n_seeds": {a: len(v) for a, v in conf.items()}},
              open(out / "confusion.json", "w"), indent=1)
    json.dump({"cols": list(SKILL_ORDER), "arms": {a: {s: np.mean(v, 0).tolist() for s, v in sorted(st.items())}
                                                    for a, st in share.items()}}, open(out / "share.json", "w"), indent=1)

    roles = {}
    for arm in ("G", "C"):
        per = defaultdict(list)
        for f in sorted(glob.glob(str(REPO / f"runs/paper_eval/block*/{arm}_*.json"))):
            J = json.load(open(f))
            if not J["meta"]["valid"]:
                continue
            c = defaultdict(Counter)
            for s in J["steps"]:
                for a in s["agents"].values():
                    sk = a.get("actor")
                    if arm == "C" and sk in ("shoot", "pass") and not a.get("macro"):
                        sk = "walk"
                    c[a.get("role")][sk] += 1
            for r, cnt in c.items():
                n = sum(cnt.values())
                per[r].append({k: cnt[k] / n for k in SKILL_ORDER})
        roles[arm] = {r: {k: [float(np.mean([m[k] for m in v])), len(v)] for k in SKILL_ORDER} for r, v in per.items() if r}
    json.dump(roles, open(out / "roles.json", "w"), indent=1)
    print(json.dumps({a: {r: {k: round(v[0], 3) for k, v in d.items() if v[0] > 0.005} for r, d in rr.items()} for a, rr in roles.items()}, indent=1))


if __name__ == "__main__":
    main()
