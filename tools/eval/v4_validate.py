#!/usr/bin/env python3
"""v4 no-sim validation: teammate-ablation skill-flip + DECISIVE policy PASS RATE.

For each v4 checkpoint, on the v4 corpus obs:
  - flip_rate: masked-argmax skill changes when obs[:,26:39]=0 (teammate dims zeroed).
  - pass_rate_all: fraction of rows whose masked-argmax skill == pass.
  - pass_rate_striker: same, restricted to role 0 (striker = the passer).
  - pass_rate_pass_armed: among rows where pass is ARMED (obs[:,41]>=0.5), fraction
    whose masked-argmax == pass. THE decisive metric — when a pass is available, does
    the policy choose it? v3.1 had pass_rate ~0.0 everywhere; v4 must be clearly >0 (>=5%).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO.parent / "SabanaHerons2026" / "Make" / "Python"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import numpy as np, torch, torch.nn.functional as F
from rlfootball.core.action import SKILL_ORDER
from rlfootball.training.train import gate_skill_mask_from_obs

CORPUS = str(_REPO / "runs/teacher_multiagent_dataset/dataset_v4_backbone_plus_overlay/samples.npz")
RUN = _REPO / "runs/multiagent_v4"
ROLE = {0: "striker", 1: "open_support", 2: "off_ball_support"}
PASS = list(SKILL_ORDER).index("pass")
PASS_ARMED_COL = 41  # obs[41] = pass_armed (47-dim contract [41:43])

z = np.load(CORPUS, allow_pickle=True)
obs_np = z["observations"].astype(np.float32)
roles = z["role_ids"].astype(int)
obs = torch.from_numpy(obs_np)
obs_abl = obs.clone(); obs_abl[:, 26:39] = 0.0
pass_armed = obs_np[:, PASS_ARMED_COL] >= 0.5
striker = roles == 0
print(f"corpus rows={len(obs)}  pass_idx={PASS}  pass_armed_rows={int(pass_armed.sum())}  striker_rows={int(striker.sum())}")


def actor_logits(sd, x):
    h = F.elu(F.linear(x, sd["trunk.0.weight"], sd["trunk.0.bias"]))
    h = F.elu(F.linear(h, sd["trunk.2.weight"], sd["trunk.2.bias"]))
    return F.linear(h, sd["skill_head.weight"], sd["skill_head.bias"])


def masked_argmax(sd, x):
    with torch.no_grad():
        lg = actor_logits(sd, x)
        m = gate_skill_mask_from_obs(x).to(lg.dtype)
        return lg.masked_fill(m <= 0.0, -1e9).argmax(1).numpy()


def frac(mask):
    return float(mask.mean()) if mask.size else 0.0


out = {"corpus": CORPUS, "rows": int(len(obs)), "pass_idx": PASS, "checkpoints": {}}
for ck in sorted(RUN.glob("checkpoint_stage7_step*.pt")):
    if ck.name.endswith("step0.pt"):
        continue
    sd = torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"]
    a_o = masked_argmax(sd, obs); a_a = masked_argmax(sd, obs_abl)
    flip = (a_o != a_a)
    is_pass = a_o == PASS
    rec = {
        "flip_rate": frac(flip),
        "pass_rate_all": frac(is_pass),
        "pass_rate_striker": frac(is_pass[striker]),
        "pass_rate_pass_armed": frac(is_pass[pass_armed]),
        "by_role_flip": {ROLE[r]: frac(flip[roles == r]) for r in sorted(set(roles)) if (roles == r).any()},
    }
    out["checkpoints"][ck.name] = rec
    print(f"{ck.name}: flip={rec['flip_rate']:.3f}  PASS_all={rec['pass_rate_all']:.4f} "
          f"PASS_striker={rec['pass_rate_striker']:.4f}  PASS_armed={rec['pass_rate_pass_armed']:.4f}")

(RUN / "v4_validation.json").write_text(json.dumps(out, indent=1))
best_pass = max(c["pass_rate_pass_armed"] for c in out["checkpoints"].values())
best_flip = max(c["flip_rate"] for c in out["checkpoints"].values())
print(f"\n[v4-validate] best flip={best_flip:.3f} (v2b 0.079, v3.1 0.004)  "
      f"best PASS_armed={best_pass:.4f} (v3.1 0.000; DECISIVE target >=0.05) -> {RUN/'v4_validation.json'}")
