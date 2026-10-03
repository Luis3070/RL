#!/usr/bin/env python3
"""v3.1 teammate-ablation skill-flip (replicates the v3 probe). No sim.

For each checkpoint: masked-argmax skill on the corpus obs (orig) vs with obs[:,26:39]=0
(teammate dims zeroed, goal cols 39:40 preserved). flip_rate = fraction whose argmax skill
changes. Higher = the policy conditions more on teammates. v2b=7.9%, v3=~2%; gate >=20%.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO.parent / "SabanaHerons2026" / "Make" / "Python"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import numpy as np, torch, torch.nn.functional as F
from action import SKILL_ORDER
from train import gate_skill_mask_from_obs

CORPUS = str(_REPO / "runs/teacher_multiagent_dataset/dataset_v31_backbone_plus_overlay/samples.npz")
RUN = _REPO / "runs/multiagent_v31"
ROLE = {0: "striker", 1: "open_support", 2: "off_ball_support"}

z = np.load(CORPUS, allow_pickle=True)
obs_np = z["observations"].astype(np.float32)
roles = z["role_ids"].astype(int); stages = z["stages"].astype(int)
obs = torch.from_numpy(obs_np)
obs_abl = obs.clone(); obs_abl[:, 26:39] = 0.0


def actor_logits(sd, x):
    h = F.elu(F.linear(x, sd["trunk.0.weight"], sd["trunk.0.bias"]))
    h = F.elu(F.linear(h, sd["trunk.2.weight"], sd["trunk.2.bias"]))
    return F.linear(h, sd["skill_head.weight"], sd["skill_head.bias"])


def masked_argmax(sd, x):
    with torch.no_grad():
        lg = actor_logits(sd, x)
        m = gate_skill_mask_from_obs(x).to(lg.dtype)
        return lg.masked_fill(m <= 0.0, -1e9).argmax(1).numpy()


out = {"ablation_slice": "obs[:,26:39]=0", "corpus": CORPUS, "rows": int(len(obs)), "checkpoints": {}}
for ck in sorted(RUN.glob("checkpoint_stage7_step*.pt")):
    if ck.name.endswith("step0.pt"):
        continue
    sd = torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"]
    a_o = masked_argmax(sd, obs); a_a = masked_argmax(sd, obs_abl)
    flip = (a_o != a_a)
    rec = {"flip_rate": float(flip.mean())}
    s7 = stages == 7
    rec["flip_rate_stage7"] = float(flip[s7].mean()) if s7.any() else 0.0
    rec["by_role"] = {ROLE[r]: float(flip[(roles == r)].mean()) for r in sorted(set(roles)) if (roles == r).any()}
    out["checkpoints"][ck.name] = rec
    print(f"{ck.name}: flip_all={rec['flip_rate']:.3f} flip_stage7={rec['flip_rate_stage7']:.3f} "
          f"by_role={ {k: round(v,3) for k,v in rec['by_role'].items()} }")

(RUN / "v31_teammate_ablation.json").write_text(json.dumps(out, indent=1))
best = max(c["flip_rate"] for c in out["checkpoints"].values())
print(f"\n[v31-ablation] best flip_rate={best:.3f}  (v2b=0.079, v3=0.020, gate>=0.20)  -> {RUN/'v31_teammate_ablation.json'}")
