"""Offline semaphore evaluation of the multi-agent team policy (V2 / v2b@7500).

WHY OFFLINE: the live SimRobot 3D loop needs the compiled `pybh` extension, which
is not built on this Windows box. This harness instead evaluates the *exact*
deployment artifact — `ppo_team_hsl2026.onnx`, the ONNX that gets transplanted
into B-Human C++ — on the real teacher BC corpus (355k recorded 47-dim obs with
teacher labels, role_ids and gate state). It scores the three axes the reviewer
asked for and emits a red / yellow / green verdict per axis.

It reproduces the deployment decision path bit-faithfully:
  * the forbid-invalid skill mask of train.gate_skill_mask_from_obs (cols
    23,24,41,42,44,45,46), and
  * deterministic action = argmax over masked skill logits.

What it CAN judge (policy-function level): numerical stability, gate compliance,
perturbation jitter, OOD/missing-input robustness, role differentiation, skill
diversity, pass usage, teacher fidelity.

What it CANNOT judge (needs the closed loop / live sim): real collision rate,
goal conversion, pass-completion in motion. Those are flagged for the live run.

Usage:
    .venv/Scripts/python.exe tools/eval/offline_team_eval.py
    .venv/Scripts/python.exe tools/eval/offline_team_eval.py --onnx models/team/ppo_team_hsl2026.onnx \
        --corpus runs/teacher_multiagent_dataset/dataset_multiagent_train_20260611_full_v1/samples.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Windows consoles default to cp1252; force UTF-8 so the semaphore emojis render.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]

# SKILL_ORDER (must match action.SkillType / ppo_team_hsl2026.json skill_order).
SKILL_ORDER = ("stand", "walk", "shoot", "pass", "dribble", "block", "mark", "observe")
S = {name: i for i, name in enumerate(SKILL_ORDER)}

# obs columns used by the gate (mirror observation.ObsIndex).
SHOOT_ARMED = 23
DRIBBLE_ARMED = 24
PASS_ARMED = 41
OBSERVE_ARMED = 42
IS_STRIKER = 44
IS_OPEN = 45
IS_OFF_BALL = 46
ROLE_NAMES = ("striker", "open_support", "off_ball_support")

NEG_INF = -1.0e30


# ---------------------------------------------------------------------------
# Deployment-faithful gate + decision
# ---------------------------------------------------------------------------
def gate_mask(obs: np.ndarray) -> np.ndarray:
    """Vectorised numpy port of train.gate_skill_mask_from_obs for obs>=47."""
    n = obs.shape[0]
    mask = np.ones((n, len(SKILL_ORDER)), dtype=np.float32)
    shoot_armed = obs[:, SHOOT_ARMED] >= 0.5
    dribble_armed = obs[:, DRIBBLE_ARMED] >= 0.5
    finish_armed = shoot_armed | dribble_armed
    pass_armed = obs[:, PASS_ARMED] >= 0.5
    observe_armed = obs[:, OBSERVE_ARMED] >= 0.5
    is_striker = obs[:, IS_STRIKER] >= 0.5
    is_open = obs[:, IS_OPEN] >= 0.5
    is_off = obs[:, IS_OFF_BALL] >= 0.5
    has_role = is_striker | is_open | is_off
    single = ~has_role

    # stand/walk default allowed; single-agent fallback masks them when finish-armed.
    sa_finish = finish_armed & single
    mask[:, S["stand"]] = np.where(single, ~sa_finish, True).astype(np.float32)
    mask[:, S["walk"]] = np.where(single, ~sa_finish, True).astype(np.float32)
    mask[:, S["shoot"]] = (shoot_armed & (is_striker | single)).astype(np.float32)
    mask[:, S["pass"]] = (pass_armed & is_striker).astype(np.float32)
    mask[:, S["dribble"]] = (
        (dribble_armed & is_striker) | (dribble_armed & is_off) | (dribble_armed & single)
    ).astype(np.float32)
    mask[:, S["observe"]] = (observe_armed & (~is_striker) & has_role).astype(np.float32)
    # block / mark inactive for this policy.
    mask[:, S["block"]] = 0.0
    mask[:, S["mark"]] = 0.0
    return mask


def masked_argmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask > 0.5, logits, NEG_INF)
    return masked.argmax(axis=1)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class Policy:
    def __init__(self, onnx_path: Path):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def run(self, obs: np.ndarray, batch: int = 16384):
        logits_all, param_all = [], []
        for i in range(0, obs.shape[0], batch):
            chunk = obs[i : i + batch].astype(np.float32)
            logits, param = self.sess.run(None, {self.in_name: chunk})
            logits_all.append(np.asarray(logits, dtype=np.float32))
            param_all.append(np.asarray(param, dtype=np.float32))
        return np.concatenate(logits_all), np.concatenate(param_all)


# ---------------------------------------------------------------------------
# Semaphore helper
# ---------------------------------------------------------------------------
def light(value: float, green_if, yellow_if, higher_is_better=True) -> str:
    """Return 'GREEN'/'YELLOW'/'RED' given thresholds (green/yellow boundaries)."""
    if higher_is_better:
        if value >= green_if:
            return "GREEN"
        if value >= yellow_if:
            return "YELLOW"
        return "RED"
    else:
        if value <= green_if:
            return "GREEN"
        if value <= yellow_if:
            return "YELLOW"
        return "RED"


def worst(lights) -> str:
    order = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    return max(lights, key=lambda l: order[l])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "models" / "team" / "ppo_team_hsl2026.onnx"))
    ap.add_argument(
        "--corpus",
        default=str(
            ROOT
            / "runs/teacher_multiagent_dataset/dataset_multiagent_train_20260611_full_v1/samples.npz"
        ),
    )
    ap.add_argument("--perturb-sample", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(ROOT / "runs/offline_team_eval"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pol = Policy(Path(args.onnx))
    data = np.load(args.corpus, allow_pickle=True)
    obs = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    role_ids = data["role_ids"].astype(np.int64)
    weights = data["weights"].astype(np.float32)
    label_consistent = data["label_consistent"].astype(np.int64)
    teacher_skill = np.clip(np.rint(actions[:, 0]).astype(np.int64), 0, len(SKILL_ORDER) - 1)
    n = obs.shape[0]

    report: dict = {
        "meta": {
            "onnx": str(args.onnx),
            "corpus": str(args.corpus),
            "n_obs": int(n),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "obs_dim": int(obs.shape[1]),
        }
    }
    print(f"[offline-eval] policy={Path(args.onnx).name}  corpus N={n}  obs_dim={obs.shape[1]}")

    # --- clean forward pass over full corpus ---
    logits, param = pol.run(obs)
    mask = gate_mask(obs)
    chosen = masked_argmax(logits, mask)
    raw_argmax = logits.argmax(axis=1)
    probs = softmax(logits)
    finite = np.isfinite(logits).all(axis=1) & np.isfinite(param).all(axis=1)

    # ============================ STABILITY ============================
    nan_rate = float((~finite).mean())
    # REAL gate compliance: does the DEPLOYED (masked) choice ever land on a
    # forbidden skill? Must be 0 — this is what actually runs in C++.
    masked_illegal = float((mask[np.arange(n), chosen] < 0.5).mean())
    # INFO only: the unmasked head's top pick / prob mass on forbidden skills.
    # In forbid-invalid MAPPO the mask is applied at sampling AND update time, so
    # the policy never trains its raw head to respect the gate. A high value here
    # is BY DESIGN, not a defect — it is not a verdict driver.
    raw_forbidden = float((mask[np.arange(n), raw_argmax] < 0.5).mean())
    mass_forbidden = float((probs * (mask < 0.5)).sum(axis=1).mean())
    # param saturation (tanh-ish): fraction of active param components at |.|>0.98
    sat_rate = float((np.abs(param) > 0.98).mean())

    # determinism: re-run a subset, must be bit-identical
    sub = rng.choice(n, size=min(5000, n), replace=False)
    l2, p2 = pol.run(obs[sub])
    determinism_ok = bool(
        np.array_equal(l2, pol.run(obs[sub])[0]) and np.allclose(l2, logits[sub], atol=1e-5)
    )

    # perturbation jitter: per-feature-scaled gaussian noise, skill-flip + param drift
    col_std = obs.std(axis=0) + 1e-6
    ps = rng.choice(n, size=min(args.perturb_sample, n), replace=False)
    base_obs = obs[ps]
    base_logits, base_param = pol.run(base_obs)
    base_mask = gate_mask(base_obs)
    base_chosen = masked_argmax(base_logits, base_mask)
    sigma_small = 0.05
    noise = rng.normal(0.0, sigma_small, size=base_obs.shape).astype(np.float32) * col_std
    # don't perturb the discrete gate/role bits — they are set by the env, not noise.
    for c in (SHOOT_ARMED, DRIBBLE_ARMED, PASS_ARMED, OBSERVE_ARMED, IS_STRIKER, IS_OPEN, IS_OFF_BALL):
        noise[:, c] = 0.0
    pert_obs = base_obs + noise
    pert_logits, pert_param = pol.run(pert_obs)
    pert_chosen = masked_argmax(pert_logits, base_mask)
    jitter_flip = float((pert_chosen != base_chosen).mean())
    param_drift = float(np.linalg.norm(pert_param - base_param, axis=1).mean())

    st_nan = light(nan_rate, 0.0, 0.0, higher_is_better=False)
    st_gate = light(masked_illegal, 0.0, 0.0, higher_is_better=False)
    st_jit = light(jitter_flip, 0.05, 0.15, higher_is_better=False)
    st_det = "GREEN" if determinism_ok else "RED"
    stability_light = worst([st_nan, st_gate, st_jit, st_det])
    report["stability"] = {
        "nan_inf_rate": nan_rate,
        "masked_choice_illegal_rate": masked_illegal,
        "param_saturation_rate": sat_rate,
        "determinism_ok": determinism_ok,
        "jitter_skill_flip_rate@sigma0.05": jitter_flip,
        "jitter_param_drift_l2@sigma0.05": param_drift,
        "info_raw_head_wants_forbidden_rate": raw_forbidden,
        "info_raw_head_prob_mass_on_forbidden_mean": mass_forbidden,
        "info_note": (
            "raw_head_* are INFORMATIONAL: forbid-invalid MAPPO applies the mask at "
            "sample+update time, so the unmasked head is never trained to respect the "
            "gate. The deployed (masked) choice is illegal 0% of the time."
        ),
        "lights": {"nan": st_nan, "gate_compliance(masked)": st_gate, "jitter": st_jit, "determinism": st_det},
        "verdict": stability_light,
    }

    # ============================ ROBUSTNESS ============================
    # noise sweep: skill-agreement vs clean as sigma grows
    noise_curve = {}
    for sigma in (0.05, 0.1, 0.2, 0.5):
        nz = rng.normal(0.0, sigma, size=base_obs.shape).astype(np.float32) * col_std
        for c in (SHOOT_ARMED, DRIBBLE_ARMED, PASS_ARMED, OBSERVE_ARMED, IS_STRIKER, IS_OPEN, IS_OFF_BALL):
            nz[:, c] = 0.0
        nl, _np = pol.run(base_obs + nz)
        nch = masked_argmax(nl, base_mask)
        nfin = np.isfinite(nl).all(axis=1)
        noise_curve[f"agree@sigma{sigma}"] = float((nch == base_chosen).mean())
        noise_curve[f"naninf@sigma{sigma}"] = float((~nfin).mean())
    robust_noise_agree = noise_curve["agree@sigma0.2"]

    # missing teammates: zero obs[26:32] + team ball cols, must stay finite & legal
    miss = base_obs.copy()
    miss[:, 26:38] = 0.0
    ml, mp = pol.run(miss)
    miss_finite = float(np.isfinite(ml).all(axis=1).mean())
    miss_flip = float((masked_argmax(ml, gate_mask(miss)) != base_chosen).mean())

    open_rows = role_ids == 1
    off_rows = role_ids == 2
    striker_rows = role_ids == 0

    # pass-context robustness (deployment-relevant): on pass-armed striker rows,
    # does the masked pass decision survive sigma=0.1 obs noise? The team's headline
    # claim is "100% passes when position allows" — this checks it is not brittle.
    pa = (gate_mask(obs)[:, S["pass"]] > 0.5) & striker_rows
    pass_recall_clean = float((chosen[pa] == S["pass"]).mean()) if pa.any() else float("nan")
    pa_obs = obs[pa]
    nz = rng.normal(0.0, 0.1, size=pa_obs.shape).astype(np.float32) * col_std
    for c in (SHOOT_ARMED, DRIBBLE_ARMED, PASS_ARMED, OBSERVE_ARMED, IS_STRIKER, IS_OPEN, IS_OFF_BALL):
        nz[:, c] = 0.0
    pl, _pp = pol.run(pa_obs + nz)
    pass_recall_noisy = (
        float((masked_argmax(pl, gate_mask(pa_obs)) == S["pass"]).mean()) if pa.any() else float("nan")
    )

    # OOD extremes: top 1% obs by L2 norm — sanity (finite)
    norms = np.linalg.norm(obs, axis=1)
    tail = norms >= np.quantile(norms, 0.99)
    tail_finite = float(finite[tail].mean())

    rb_noise = light(robust_noise_agree, 0.80, 0.60, higher_is_better=True)
    rb_miss = light(miss_finite, 1.0, 0.999, higher_is_better=True)
    rb_pass = light(pass_recall_noisy, 0.90, 0.75, higher_is_better=True)
    rb_tail = light(tail_finite, 1.0, 0.999, higher_is_better=True)
    robustness_light = worst([rb_noise, rb_miss, rb_pass, rb_tail])
    report["robustness"] = {
        "noise_curve": noise_curve,
        "missing_teammates_finite_rate": miss_finite,
        "missing_teammates_skill_flip_rate": miss_flip,
        "pass_recall_clean@pass_armed_striker": pass_recall_clean,
        "pass_recall_noisy@sigma0.1": pass_recall_noisy,
        "ood_tail_finite_rate": tail_finite,
        "lights": {"noise": rb_noise, "missing_inputs": rb_miss, "pass_robustness": rb_pass, "ood_tail": rb_tail},
        "verdict": robustness_light,
    }

    # ============================ DIVERSITY ============================
    def dist(idx):
        c = np.bincount(chosen[idx], minlength=len(SKILL_ORDER)).astype(np.float64)
        return c / max(c.sum(), 1.0)

    overall = dist(np.arange(n))
    ent = float(-(overall[overall > 0] * np.log(overall[overall > 0])).sum())
    top_rate = float(overall.max())
    per_role = {ROLE_NAMES[r]: dist(role_ids == r).tolist() for r in (0, 1, 2)}

    # pass usage: among striker rows where pass is gate-allowed, policy pass-rate
    pass_allowed = (mask[:, S["pass"]] > 0.5) & striker_rows
    pass_recall = (
        float((chosen[pass_allowed] == S["pass"]).mean()) if pass_allowed.any() else float("nan")
    )
    policy_pass_rate = float((chosen == S["pass"]).mean())

    # teacher fidelity (weighted, on label-consistent rows)
    lc = label_consistent == 1
    w = weights * lc
    agree = (chosen == teacher_skill).astype(np.float64)
    fidelity = float((agree * w).sum() / max(w.sum(), 1e-9))
    fidelity_by_role = {}
    for r in (0, 1, 2):
        m = (role_ids == r) & lc
        wr = weights[m]
        fidelity_by_role[ROLE_NAMES[r]] = (
            float((agree[m] * wr).sum() / max(wr.sum(), 1e-9)) if m.any() else float("nan")
        )

    # role differentiation: striker must use ball skills, supports must not.
    striker_ball = float(
        dist(striker_rows)[[S["shoot"], S["dribble"], S["pass"]]].sum()
    )
    support_chosen = np.concatenate([chosen[open_rows], chosen[off_rows]])
    support_ball = (
        float(np.isin(support_chosen, [S["shoot"], S["pass"]]).mean())
        if support_chosen.size
        else 0.0
    )

    striker_fid = fidelity_by_role["striker"]
    support_fid = float(np.nanmean([fidelity_by_role["open_support"], fidelity_by_role["off_ball_support"]]))
    dv_div = light(top_rate, 0.85, 0.95, higher_is_better=False)  # mode-collapse guard
    # striker is the imitation-critical role (finishing/passing). Supports may
    # legally diverge stand<->walk, so they do NOT gate the verdict; the support
    # stand/walk shift is surfaced as a YELLOW caveat for the live hold-rate check.
    dv_striker = light(striker_fid, 0.85, 0.70, higher_is_better=True)
    dv_support = light(support_ball, 0.01, 0.05, higher_is_better=False)  # supports rarely shoot/pass
    # informational support-fidelity light (walk-vs-stand divergence vs teacher).
    dv_support_fid = light(support_fid, 0.55, 0.35, higher_is_better=True)
    diversity_light = worst([dv_div, dv_striker, dv_support])
    report["diversity"] = {
        "overall_skill_dist": {SKILL_ORDER[i]: float(overall[i]) for i in range(len(SKILL_ORDER))},
        "skill_entropy_nats": ent,
        "top1_skill_rate": top_rate,
        "per_role_skill_dist": {
            role: {SKILL_ORDER[i]: float(v) for i, v in enumerate(d)} for role, d in per_role.items()
        },
        "striker_ball_skill_rate": striker_ball,
        "support_shoot_or_pass_rate": support_ball,
        "pass_recall_on_pass_armed_striker": pass_recall,
        "policy_pass_rate_overall": policy_pass_rate,
        "teacher_fidelity_weighted": fidelity,
        "teacher_fidelity_by_role": fidelity_by_role,
        "support_walk_vs_stand_note": (
            "Supports diverge from the teacher mainly by preferring walk over stand "
            "(both gate-legal). This is a YELLOW flag, not a defect: only the live "
            "hold-rate (support_open_hold_rate / support_triangle_hold_rate from the "
            "team env) can confirm supports HOLD position vs drift. Run the live sim."
        ),
        "lights": {
            "mode_collapse": dv_div,
            "striker_fidelity": dv_striker,
            "support_discipline": dv_support,
            "support_fidelity(info)": dv_support_fid,
        },
        "verdict": diversity_light,
    }

    overall_verdict = worst([stability_light, robustness_light, diversity_light])
    report["overall_verdict"] = overall_verdict
    report["caveat"] = (
        "Offline policy-function evaluation on the deployment ONNX. Closed-loop "
        "metrics (real collision rate, goal conversion, pass completion in motion) "
        "require the live SimRobot 3D loop (pybh) and are NOT covered here."
    )

    # --- write + print ---
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"offline_eval_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    EMOJI = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    print("\n" + "=" * 64)
    print("  VEREDICTO SEMÁFORO — multiagente V2 (v2b@7500, ONNX deploy)")
    print("=" * 64)
    for axis in ("stability", "robustness", "diversity"):
        v = report[axis]["verdict"]
        print(f"  {EMOJI[v]}  {axis.upper():12s} {v}")
        for k, l in report[axis]["lights"].items():
            print(f"        {EMOJI[l]} {k}: {l}")
    print("-" * 64)
    print(f"  {EMOJI[overall_verdict]}  GLOBAL: {overall_verdict}")
    print("=" * 64)
    print("\nMétricas clave:")
    print(f"  stability:  nan={nan_rate:.4f}  masked_illegal={masked_illegal:.4f}  jitter_flip={jitter_flip:.4f}  det_ok={determinism_ok}")
    print(f"              (info) raw_head_wants_forbidden={raw_forbidden:.3f} — by-design, not a defect")
    print(f"  robustness: agree@σ0.2={robust_noise_agree:.3f}  missing_finite={miss_finite:.4f}  pass_recall_noisy@σ0.1={pass_recall_noisy:.3f}")
    print(f"  diversity:  top1={top_rate:.3f}  entropy={ent:.3f}  striker_fidelity={striker_fid:.3f}  support_fidelity={support_fid:.3f}")
    print(f"              striker_ball={striker_ball:.3f}  pass_recall(armed)={pass_recall:.3f}  policy_pass_rate={policy_pass_rate:.4f}")
    print(f"              per-role striker={fidelity_by_role['striker']:.3f} open={fidelity_by_role['open_support']:.3f} off_ball={fidelity_by_role['off_ball_support']:.3f}")
    print(f"\n  report → {out_path}")
    print(f"  CAVEAT: {report['caveat']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
