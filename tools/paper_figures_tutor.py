"""Figures answering the tutor's review (2026-10-09). Reads only finished result files:
  runs/paper_eval/stats_final/stats.json          match-level eval (pre-registered)
  runs/paper_offline/ablation/per_checkpoint.csv  held-out offline eval of the seeded training ablation
  runs/paper_offline/ablation/train_curves.csv    training curves of the ablation runs (tensorboard)
  runs/paper_offline/distill/curves.csv           imitation-stage curves (grouped k-fold x seeds)
  runs/kickstart/*/iter_*/learner.log             RL (kickstart PPO) stage, per iteration
Writes .docs/FIGURES/tutor_*.pdf/.png. Palette validated with the dataviz validator (light, 5 slots).
"""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / ".docs/FIGURES"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MK = ["o", "s", "^", "D", "v"]
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2, "savefig.dpi": 200})


def tci(x):
    x = np.asarray(x, float)
    return x.mean(), (stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


ARMS = [("I", "Imitation only (DAgger)"), ("G", "Imitation + RL (kickstart PPO)"),
        ("P", "Imitation + RL + pass rule (final)"), ("C", "B-Human (heuristic) in same seat")]


def fig_match():
    S = json.load(open(REPO / "runs/paper_eval/stats_final/stats.json"))["arms"]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for i, (k, name) in enumerate(ARMS):
        m, lo, hi = S[k]["goal_diff"]
        c = SER[i]
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt=MK[i], color=c, ms=7, capsize=3, lw=2)
        ax.text(hi + 0.15, i, f"{S[k]['W']}-{S[k]['D']}-{S[k]['L']}  (n={S[k]['n']})", va="center", color=INK2, fontsize=8)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(ARMS)), [n for _, n in ARMS]); ax.invert_yaxis()
    ax.set_xlabel("Goal difference per match vs B-Human (mean, 95% CI)")
    ax.set_xlim(-3.2, 3.6)
    save(fig, "tutor_fig_match_goal_difference")

    mets = [("possession_pct", "Possession %"), ("recoveries", "Recoveries"), ("falls", "Falls"),
            ("ball_in_final_third_pct", "Ball in final third %"), ("shots_on_target", "Shots on target"),
            ("goals_for", "Goals for")]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.0))
    for ax, (k, name) in zip(axs.flat, mets):
        for i, (a, _) in enumerate(ARMS):
            m, lo, hi = S[a][k]
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt=MK[i], color=SER[i], ms=6, capsize=3)
        ax.set_title(name, fontsize=9, color=INK); ax.set_xticks(range(len(ARMS)), [a for a, _ in ARMS])
        ax.set_xlim(-0.5, len(ARMS) - 0.5)
    fig.legend([plt.Line2D([], [], marker=MK[i], color=SER[i], ls="") for i in range(len(ARMS))],
               [f"{a}: {n}" for a, n in ARMS], loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    save(fig, "tutor_fig_match_soccer_stats")


ABL = [("FULL", "Pipeline: warm start + BC + KL"), ("NOKL", "Pipeline without KL anchor"),
       ("NOBC", "Pipeline without BC loss (KL only)"), ("WSONLY", "Warm start + PPO (no BC, no KL)"),
       ("PPO", "PPO alone (random init)")]


def fig_ablation():
    rows = list(csv.DictReader(open(REPO / "runs/paper_offline/ablation/per_checkpoint.csv")))
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], int(r["step"]))].append(r)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, key, lab in ((axs[0], "balanced_acc", "Balanced accuracy vs expert (held-out)"),
                         (axs[1], "pred_entropy_bits", "Entropy of chosen skills (bits)")):
        for i, (a, name) in enumerate(ABL):
            steps = sorted(s for (aa, s) in by if aa == a)
            mc = [tci([float(r[key]) for r in by[(a, s)]]) for s in steps]
            m = np.array([x[0] for x in mc]); h = np.array([x[1] for x in mc])
            ax.plot(steps, m, color=SER[i], marker=MK[i], ms=5, label=name)
            ax.fill_between(steps, np.maximum(m - h, 0), m + h, color=SER[i], alpha=0.15, lw=0)
        steps = sorted(s for (aa, s) in by if aa == "PPOLONG")
        if steps:
            mc = [tci([float(r[key]) for r in by[("PPOLONG", s)]]) for s in steps]
            ax.plot(steps, [x[0] for x in mc], color=SER[4], ls="--", lw=1.5, label="PPO alone, 30k steps")
        ax.set_xlabel("PPO training step"); ax.set_title(lab, fontsize=9, color=INK)
    axs[0].legend(fontsize=6.5, loc="center right")
    save(fig, "tutor_fig_ablation_heldout_curves")

    skills = ["stand", "walk", "shoot", "pass", "dribble", "block", "mark"]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = 0.15
    for i, (a, name) in enumerate(ABL):
        rs = by[(a, 10000)]
        mc = [tci([float(r[f"recall_{s}"]) for r in rs]) for s in skills]
        x = np.arange(len(skills)) + (i - 2) * w
        ax.errorbar(x, [m for m, _ in mc], yerr=[h for _, h in mc], fmt=MK[i], color=SER[i], ms=5, capsize=2, label=name)
    ax.set_xticks(range(len(skills)), skills); ax.set_ylabel("Recall at step 10k"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save(fig, "tutor_fig_ablation_skill_recall")

    tr = list(csv.DictReader(open(REPO / "runs/paper_offline/ablation/train_curves.csv")))
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6))
    for ax, tag, lab in ((axs[0], "Reward / Instantaneous reward (mean)", "Mean step reward"),
                         (axs[1], "Loss / Policy loss (player_1)", "Policy loss"),
                         (axs[2], "Loss / Value loss (player_1)", "Value loss")):
        for i, (a, name) in enumerate(ABL):
            seeds = sorted({r["seed"] for r in tr if r["arm"] == a})
            grid = np.arange(256, 10001, 512)
            ys = []
            for s in seeds:
                pts = sorted((int(r["step"]), float(r["value"])) for r in tr if r["arm"] == a and r["seed"] == s and r["tag"] == tag)
                if len(pts) > 1:
                    ys.append(np.interp(grid, [p[0] for p in pts], [p[1] for p in pts]))
            if len(ys) < 2:
                continue
            ys = np.array(ys); m = ys.mean(0)
            h = stats.t.ppf(0.975, len(ys) - 1) * ys.std(0, ddof=1) / np.sqrt(len(ys))
            ax.plot(grid, m, color=SER[i], lw=1.5, label=name)
            ax.fill_between(grid, m - h, m + h, color=SER[i], alpha=0.12, lw=0)
        ax.set_title(lab, fontsize=9, color=INK); ax.set_xlabel("step")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    save(fig, "tutor_fig_ablation_training_curves")


def fig_training_stages():
    cur = list(csv.DictReader(open(REPO / "runs/paper_offline/distill/curves.csv")))
    it = []
    for d in sorted(glob.glob(str(REPO / "runs/kickstart/loop_20261006_224436/iter_*"))) + \
            sorted(glob.glob(str(REPO / "runs/kickstart/round2_20261008_013711/iter_*"))):
        ls = [l for l in open(d + "/learner.log") if l.startswith("{")] if Path(d + "/learner.log").exists() else []
        if ls:
            it.append(json.loads(ls[-1]))
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6))
    ep = sorted({int(r["epoch"]) for r in cur})
    for j, (k, lab) in enumerate((("train_loss", "train loss (CE + KL + param)"), ("val_ce", "held-out skill CE"))):
        mc = [tci([float(r[k]) for r in cur if int(r["epoch"]) == e]) for e in ep]
        m = np.array([x[0] for x in mc]); h = np.array([x[1] for x in mc])
        axs[0].plot(ep, m, color=SER[j], marker=MK[j], ms=4, label=lab)
        axs[0].fill_between(ep, m - h, m + h, color=SER[j], alpha=0.15, lw=0)
    axs[0].set_title("Imitation stage (4 folds x 3 seeds)", fontsize=9, color=INK); axs[0].set_xlabel("epoch"); axs[0].legend(fontsize=7)
    x = [j["iteration"] for j in it]
    for j, (k, lab) in enumerate((("ce", "teacher CE"), ("entropy", "policy entropy"), ("vloss", "value loss"))):
        axs[1].plot(x, [r["train"][k] for r in it], color=SER[j], marker=MK[j], ms=4, label=lab)
    axs[1].axvline(12.5, color=INK2, lw=1, ls=":"); axs[1].set_title("RL stage (kickstart PPO) losses", fontsize=9, color=INK)
    axs[1].set_xlabel("iteration (dotted: delivered ckpt)"); axs[1].legend(fontsize=7)
    axs[2].bar(x, [r["batch"]["gd_mean"] for r in it], color=SER[0], width=0.7)
    for xi, r in zip(x, it):
        axs[2].text(xi, 0.15, str(r["batch"]["valid_matches"]), ha="center", fontsize=6, color=INK2)
    axs[2].axhline(0, color=INK2, lw=1)
    axs[2].set_title("RL batch goal difference (n matches)", fontsize=9, color=INK); axs[2].set_xlabel("iteration")
    save(fig, "tutor_fig_training_stages")


LIN = ["v4.2", "v5", "I (imitation)", "G (final team)"]


def fig_lineage():
    S = json.load(open(REPO / "runs/paper_offline/heldout_policies/summary.json"))
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    E = S["expert"]["models"]
    keys = ["balanced_acc", "recall_shoot", "recall_pass", "recall_dribble", "recall_block", "recall_mark"]
    labs = ["balanced\nacc.", "shoot", "pass", "dribble", "block", "mark"]
    w = 0.18
    for i, m in enumerate(LIN):
        x = np.arange(len(keys)) + (i - 1.5) * w
        v = np.array([E[m][k] for k in keys])
        axs[0].errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=MK[i], color=SER[i], ms=5, capsize=2, label=m)
    axs[0].set_xticks(range(len(keys)), labs); axs[0].set_ylim(-0.05, 1.05)
    axs[0].set_title("Held-out live data, expert labels (37 matches)", fontsize=9, color=INK)
    Bh = S["bhuman"]["models"]
    keys = ["agree_striker", "agree_open_support", "agree_off_ball"]
    for i, m in enumerate(LIN):
        x = np.arange(len(keys)) + (i - 1.5) * w
        v = np.array([Bh[m][k] for k in keys])
        axs[1].errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=MK[i], color=SER[i], ms=5, capsize=2)
    axs[1].set_xticks(range(3), ["striker", "open\nsupport", "off-ball"]); axs[1].set_ylim(-0.05, 1.05)
    axs[1].set_title("Agreement with B-Human's own skills (8 matches)", fontsize=9, color=INK)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    save(fig, "tutor_fig_heldout_lineage")


def fig_roles():
    S = json.load(open(REPO / "runs/paper_offline/role_conditioning/summary.json"))
    R = S["in_match"]["G"]["roles"]
    roles = ["striker", "open_support", "off_ball_support"]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.7))
    for i, sk in enumerate(["walk", "shoot", "pass"]):
        x = np.arange(3) + (i - 1) * 0.22
        v = np.array([R[r][f"skill_{sk}"] for r in roles])
        axs[0].bar(x, v[:, 0], width=0.2, color=SER[i], label=sk, edgecolor="white", linewidth=1)
    axs[0].set_xticks(range(3), ["striker", "open support", "off-ball"]); axs[0].set_ylabel("share of steps")
    axs[0].set_title("Executed skills per role (final team, 20 matches)", fontsize=9, color=INK); axs[0].legend(fontsize=7)
    C = S["counterfactual"]["pairs"]
    pairs = [k for k in C]
    x = np.arange(len(pairs))
    axs[1].bar(x - 0.2, [C[k]["skill_changed_network_only"] for k in pairs], width=0.38, color=SER[0], label="network only (mask fixed)")
    axs[1].bar(x + 0.2, [C[k]["skill_changed_full_pipeline"] for k in pairs], width=0.38, color=SER[1], label="network + role mask")
    short = {"striker": "S", "open_support": "O", "off_ball_support": "F"}
    axs[1].set_xticks(x, [" -> ".join(short[p] for p in k.split(" -> ")) for k in pairs])
    axs[1].set_ylabel("share of steps whose skill changes"); axs[1].legend(fontsize=7)
    axs[1].set_title("Counterfactual role input (S/O/F = striker/open/off-ball)", fontsize=9, color=INK)
    save(fig, "tutor_fig_role_conditioning")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_match(); fig_ablation(); fig_training_stages(); fig_lineage(); fig_roles()
