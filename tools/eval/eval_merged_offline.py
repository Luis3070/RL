#!/usr/bin/env python3
"""Publication-grade OFFLINE evaluation of the merged 47-dim brain.

Gate-aware, deterministic, no SimRobot (the visible 3-slot trio is an unusable
instrument). Covers: sanity, per-role skill distribution, diversity/entropy,
input-noise robustness, role-conditioning, and baseline agreement vs v4.2 (attack
ref), models/defender/checkpoints (defense ref) and v2b@7500 (pre-merge base).

Run: PYTHONPATH=/home/luis/GitHub/RL:.../Make/Python python tools/eval/eval_merged_offline.py
"""
import numpy as np, torch, torch.nn.functional as F
R = "/home/luis/GitHub/RL/"
from rlfootball.core.action import SKILL_ORDER
from rlfootball.core.observation import ObsIndex, OBS_SIZE
from rlfootball.training import train as T

torch.manual_seed(0); np.random.seed(0)
S = {n: i for i, n in enumerate(SKILL_ORDER)}
ROLE = {"striker": int(ObsIndex.is_striker), "open": int(ObsIndex.is_open_support),
        "offball": int(ObsIndex.is_off_ball_support)}


class Pol(torch.nn.Module):
    def __init__(s, sd):
        super().__init__()
        for k in ("trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
                  "skill_head.weight", "skill_head.bias", "param_head.weight", "param_head.bias"):
            setattr(s, k.replace(".", "_"), sd[k])

    def forward(s, o):
        h = F.elu(F.linear(o, s.trunk_0_weight, s.trunk_0_bias))
        h = F.elu(F.linear(h, s.trunk_2_weight, s.trunk_2_bias))
        return F.linear(h, s.skill_head_weight, s.skill_head_bias), torch.tanh(
            F.linear(h, s.param_head_weight, s.param_head_bias))


def load(p): return Pol(torch.load(R + p, map_location="cpu")["policy_state_dict"])


NETS = {
    "merged": "runs/merge_v5_attack_defense/checkpoint_stage8_step10000.pt",
    "v4.2":   "runs/multiagent_v4_2/checkpoint_stage7_step12000.pt",
    "pt_def47": "models/defender/checkpoints/checkpoint_stage8_step2500.pt",
    "v2b":    "runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt",
}
nets = {k: load(v) for k, v in NETS.items()}

z = np.load(R + "runs/teacher_multiagent_dataset/dataset_merged_v5_20260618/samples.npz", allow_pickle=True)
OBS = torch.tensor(np.asarray(z["observations"]), dtype=torch.float32)
stages = np.asarray(z["stages"]).astype(int)
roles = np.asarray(z["role_ids"]).astype(int)
lab = np.asarray(z["actions"])[:, 0].astype(int)
N = len(OBS)


def gated_pred(net, obs):
    with torch.no_grad():
        lg, pm = net(obs)
    mask = T.gate_skill_mask_from_obs(obs)
    lg = lg + (mask - 1.0) * 1e9
    return lg.argmax(1).numpy(), pm.numpy(), mask.numpy()


def dist(pred, idx):
    n = len(idx);
    if n == 0: return {}
    c = np.bincount(pred[idx], minlength=len(SKILL_ORDER))
    return {SKILL_ORDER[i]: round(100 * c[i] / n, 1) for i in range(len(SKILL_ORDER)) if c[i] > 0}


def entropy(pred, idx):
    if len(idx) == 0: return 0.0
    c = np.bincount(pred[idx], minlength=len(SKILL_ORDER)).astype(float)
    p = c[c > 0] / c.sum()
    return round(float(-(p * np.log2(p)).sum()), 3)


print("=" * 78)
print(f"OFFLINE EVAL  merged@10000   rows={N}  obs={OBS.shape[1]}  skills={len(SKILL_ORDER)}")
print(f"  stage rows: attack(7)={int((stages==7).sum())}  defense(8)={int((stages==8).sum())}")
print("=" * 78)

pm_pred, pm_par, pm_mask = gated_pred(nets["merged"], OBS)

# ---- 1. SANITY ----
print("\n[1] SANITY")
with torch.no_grad():
    lg, par = nets["merged"](OBS)
print(f"  logits finite: {bool(torch.isfinite(lg).all())}   params finite: {bool(torch.isfinite(par).all())}")
print(f"  param range (tanh): min={float(par.min()):.3f} max={float(par.max()):.3f}  (expect [-1,1])")
# gate violation: did gated argmax ever select a masked-off (disallowed) skill?
viol = int((pm_mask[np.arange(N), pm_pred] < 0.5).sum())
print(f"  gate violations (picked a forbidden skill): {viol}  (expect 0)")
# determinism
p2, _, _ = gated_pred(nets["merged"], OBS)
print(f"  determinism (re-run argmax match): {100*(p2==pm_pred).mean():.1f}%")

# ---- 2. SKILL DISTRIBUTION per role (merged) ----
print("\n[2] SKILL DISTRIBUTION  (gated argmax, merged) per role one-hot")
for rn, rb in ROLE.items():
    idx = np.where(roles == (0 if rn == "striker" else 1 if rn == "open" else 2))[0]
    print(f"  {rn:8s} n={len(idx):6d}  H={entropy(pm_pred,idx)} bits  {dist(pm_pred, idx)}")

# ---- 3. DIVERSITY: param spread (degenerate-target check) ----
print("\n[3] DIVERSITY  (action-param spread; degenerate-target guard)")
for rn, rid in [("striker", 0), ("open", 1), ("offball", 2)]:
    idx = roles == rid
    p = pm_par[idx]
    print(f"  {rn:8s} target_x std={p[:,0].std():.3f}  target_y std={p[:,1].std():.3f}  "
          f"theta std={p[:,2].std():.3f}  (near-0 std => degenerate)")

# ---- 4. ROBUSTNESS: perception-noise input perturbation ----
print("\n[4] ROBUSTNESS  (Gaussian noise on perception dims [0:23]+[26:44]; skill-flip & param drift)")
pdims = list(range(0, 23)) + list(range(26, 44))   # skip gate bits[23:26] & role[44:47]
for sigma in (0.02, 0.05, 0.10):
    flips, drifts = [], []
    for _ in range(3):
        o = OBS.clone()
        noise = torch.zeros_like(o); noise[:, pdims] = torch.randn(len(OBS), len(pdims)) * sigma
        pr, pa, _ = gated_pred(nets["merged"], o + noise)
        flips.append(100 * (pr != pm_pred).mean())
        drifts.append(float(np.abs(pa - pm_par).mean()))
    print(f"  sigma={sigma:.2f}: skill-flip={np.mean(flips):.1f}%   mean|Δparam|={np.mean(drifts):.4f}")

# ---- 5. ROLE-CONDITIONING: same obs, swap role one-hot ----
print("\n[5] ROLE-CONDITIONING  (same world obs, force role one-hot; skill distribution must shift)")
def force_role(obs, role):
    o = obs.clone()
    for b in ROLE.values(): o[:, b] = 0.0
    o[:, ROLE[role]] = 1.0
    return o
atk = np.where(stages == 7)[0]
dfn = np.where(stages == 8)[0]
for srcname, src in [("attack-obs", atk), ("defense-obs", dfn)]:
    sub = OBS[src]
    for role in ("striker", "offball"):
        pr, _, _ = gated_pred(nets["merged"], force_role(sub, role))
        print(f"  {srcname} forced->{role:8s}: {dist(pr, np.arange(len(pr)))}")

# ---- 6. BASELINE AGREEMENT ----
print("\n[6] BASELINE AGREEMENT  (gated argmax vs baselines)")
preds = {k: gated_pred(v, OBS)[0] for k, v in nets.items()}
print(f"  ATTACK rows (stage7): merged vs v4.2  = {100*(preds['merged'][atk]==preds['v4.2'][atk]).mean():.1f}%   "
      f"vs v2b = {100*(preds['merged'][atk]==preds['v2b'][atk]).mean():.1f}%")
print(f"  DEFENSE rows (stage8): merged vs pt_def47 = {100*(preds['merged'][dfn]==preds['pt_def47'][dfn]).mean():.1f}%   "
      f"vs v4.2 = {100*(preds['merged'][dfn]==preds['v4.2'][dfn]).mean():.1f}%")

# ---- 7. DEFENSE recall + what-instead ----
print("\n[7] DEFENSE skill recall (teacher-labeled rows) + merged's alternative when it misses")
for nm in ("mark", "block"):
    i = S[nm]; t = np.where((stages == 8) & (lab == i))[0]
    if len(t) == 0: continue
    got = pm_pred[t] == i
    alt = dist(pm_pred, t[~got]) if (~got).any() else {}
    print(f"  {nm:6s} teacher={len(t):5d}  merged_recall={100*got.mean():.1f}%   when-miss->{alt}")

# ---- 8. ATTACK skill recall ----
print("\n[8] ATTACK skill recall (teacher-labeled rows, stage7)")
for nm in ("pass", "shoot", "dribble"):
    i = S[nm]; t = np.where((stages == 7) & (lab == i))[0]
    if len(t) == 0: continue
    print(f"  {nm:7s} teacher={len(t):5d}  merged_recall={100*(pm_pred[t]==i).mean():.1f}%   "
          f"v4.2_recall={100*(preds['v4.2'][t]==i).mean():.1f}%")
print("\n" + "=" * 78)
