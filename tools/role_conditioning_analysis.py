"""Claim 3 evidence for the final all-learned team: one network, three role-specific behaviours.

(a) In-match behaviour per role (runs/paper_eval block1/block2, arms P/G/I): executed-skill mix, distance to the ball,
    field x and lateral spread per role; mean per match with 95% bootstrap CI over matches.
(b) Counterfactual role input on the same live states (npz obs of the G matches): the role one-hot (obs[44:47]) is
    replaced by each other role and the decision recomputed, (i) network only (mask of the true role kept) and
    (ii) full pipeline (mask recomputed with the swapped role). Reports the share of steps whose skill changes and the
    mean change in the support-position residuals (y, theta).
Output: runs/paper_offline/role_conditioning/{summary.json, table.md}
"""
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools"))
from train import gate_skill_mask_from_obs  # noqa: E402
from action import SKILL_ORDER  # noqa: E402
from train_dagger_v5 import Policy  # noqa: E402

ROLES = ["striker", "open_support", "off_ball_support"]
ROLE_IDX = 44
CKPT = "runs/kickstart/loop_20261006_224436/iter_12/policy.pt"
rng = np.random.default_rng(0)


def bci(x):
    x = np.asarray(x, float)
    bs = rng.choice(x, (2000, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def in_match(arms=("P", "G", "I")):
    res = {}
    for arm in arms:
        files = sorted(glob.glob(str(REPO / f"runs/paper_eval/block*/{arm}_*.json")))
        per = {r: {"skill": [], "ball_d": [], "x": [], "absy": []} for r in ROLES}
        for f in files:
            J = json.load(open(f))
            if not J["meta"]["valid"]:
                continue
            acc = {r: {"skill": Counter(), "ball_d": [], "x": [], "absy": []} for r in ROLES}
            for s in J["steps"]:
                for a in s["agents"].values():
                    r = a.get("role")
                    if r not in acc:
                        continue
                    acc[r]["skill"][a["actor"]] += 1
                    acc[r]["ball_d"].append(math.hypot(a["gt_x"] - a["bx"], a["gt_y"] - a["by"]))
                    acc[r]["x"].append(a["gt_x"]); acc[r]["absy"].append(abs(a["gt_y"]))
            for r in ROLES:
                n = sum(acc[r]["skill"].values())
                if not n:
                    continue
                per[r]["skill"].append({k: acc[r]["skill"][k] / n for k in SKILL_ORDER})
                for k in ("ball_d", "x", "absy"):
                    per[r][k].append(float(np.median(acc[r][k])))
        res[arm] = {"n_matches": len(per["striker"]["x"]), "roles": {}}
        for r in ROLES:
            d = {f"skill_{k}": bci([m[k] for m in per[r]["skill"]]) for k in SKILL_ORDER}
            d.update({k: bci(per[r][k]) for k in ("ball_d", "x", "absy")})
            res[arm]["roles"][r] = d
    return res


def counterfactual():
    files = sorted(glob.glob(str(REPO / "runs/paper_eval/block2/G_*.npz")))
    O = np.concatenate([np.load(f)["obs"] for f in files]).astype(np.float32)
    O = O[O[:, ROLE_IDX:ROLE_IDX + 3].sum(1) > 0.5]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Policy(torch.load(REPO / CKPT, map_location="cpu", weights_only=False)["policy_state_dict"]).to(dev).eval()
    Ot = torch.as_tensor(O, device=dev)
    true_role = Ot[:, ROLE_IDX:ROLE_IDX + 3].argmax(1)
    M0 = gate_skill_mask_from_obs(Ot)
    with torch.no_grad():
        lg0, p0 = net(Ot)
    pred0 = (lg0 + (M0 - 1) * 1e9).argmax(1)
    out = {"n_steps": int(len(O)), "pairs": {}}
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            sel = true_role == a
            X = Ot[sel].clone(); X[:, ROLE_IDX:ROLE_IDX + 3] = 0; X[:, ROLE_IDX + b] = 1
            with torch.no_grad():
                lg, p = net(X)
            net_only = (lg + (M0[sel] - 1) * 1e9).argmax(1)
            full = (lg + (gate_skill_mask_from_obs(X) - 1) * 1e9).argmax(1)
            out["pairs"][f"{ROLES[a]} -> {ROLES[b]}"] = {
                "n": int(sel.sum()),
                "skill_changed_network_only": float((net_only != pred0[sel]).float().mean()),
                "skill_changed_full_pipeline": float((full != pred0[sel]).float().mean()),
                "mean_abs_delta_residual_y": float((p[:, 1] - p0[sel][:, 1]).abs().mean()),
                "mean_abs_delta_residual_theta": float((p[:, 2] - p0[sel][:, 2]).abs().mean()),
                "skill_mix_after_full": {SKILL_ORDER[k]: float((full == k).float().mean()) for k in range(len(SKILL_ORDER))
                                         if (full == k).any()},
                "skill_mix_before": {SKILL_ORDER[k]: float((pred0[sel] == k).float().mean()) for k in range(len(SKILL_ORDER))
                                     if (pred0[sel] == k).any()}}
    return out


def main():
    out = REPO / "runs/paper_offline/role_conditioning"; out.mkdir(parents=True, exist_ok=True)
    S = {"in_match": in_match(), "counterfactual": counterfactual()}
    json.dump(S, open(out / "summary.json", "w"), indent=1)
    f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"  # noqa: E731
    md = []
    for arm, d in S["in_match"].items():
        md += [f"### In-match behaviour per role, arm {arm} (n={d['n_matches']} matches; shares and medians per match, 95% CI)", "",
               "| | " + " | ".join(ROLES) + " |", "|---|---|---|---|"]
        for k in ["skill_walk", "skill_stand", "skill_shoot", "skill_pass", "skill_dribble", "skill_observe", "ball_d", "x", "absy"]:
            md.append(f"| {k} | " + " | ".join(f(d['roles'][r][k]) for r in ROLES) + " |")
        md.append("")
    md += ["### Counterfactual role input (final-team network, live states)", "",
           "| true -> given role | n | skill changed (network only) | skill changed (full pipeline) | mean abs delta residual y | delta theta |",
           "|---|---|---|---|---|---|"]
    for k, v in S["counterfactual"]["pairs"].items():
        md.append(f"| {k} | {v['n']} | {v['skill_changed_network_only']:.3f} | {v['skill_changed_full_pipeline']:.3f} | "
                  f"{v['mean_abs_delta_residual_y']:.3f} | {v['mean_abs_delta_residual_theta']:.3f} |")
    open(out / "table.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
