"""ICRA/IEEE-style figures for the final all-learned team (2026-10-10).

Conventions: IEEE column widths (3.5 in single, 7.16 in double), Times-metric serif (Nimbus Roman) 8 pt, Type-42 fonts
(PDF eXpress), no in-figure titles (captions carry the text; see results/paper/figures/icra_captions.md), panel labels (a)(b),
one fixed colour per arm across all figures (validated categorical palette, CVD-safe), stock B-Human always neutral grey.
Inputs are the finished result files only (see tools/paper/paper_figures_tutor.py for paths). Output: results/paper/figures/icra_fig*.pdf|png
"""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/paper/figures"
OUT.mkdir(parents=True, exist_ok=True)
SINGLE, DOUBLE = 3.5, 7.16
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY, INK, INK2, GRID = "#6b6a64", "#1f1f1e", "#4a4944", "#e4e3de"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "legend.handlelength": 1.6,
    "lines.linewidth": 1.4, "lines.markersize": 4, "errorbar.capsize": 0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

# fixed identities
MATCH = [("I", "Imitation only", BLUE, "o"), ("G", "Imitation + RL (final)", ORANGE, "s"),
         ("P", "Final + pass rule", AQUA, "^"), ("C", "B-Human (heuristic)", GREY, "D")]
ABL = [("FULL", "Warm start + BC + KL", BLUE, "o"), ("NOKL", "Warm start + BC (no KL)", ORANGE, "s"),
       ("NOBC", "Warm start + KL (no BC)", AQUA, "^"), ("WSONLY", "Warm start + PPO", YELLOW, "D"),
       ("PPO", "PPO alone", MAGENTA, "v")]
LIN = [("v4.2", "v4.2 (attack)", BLUE, "o"), ("v5", "v5 (attack+defence)", AQUA, "^"),
       ("G (final team)", "Final team", ORANGE, "s")]


def tci(x):
    x = np.asarray(x, float)
    return x.mean(), (stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)


def plabel(ax, s, x=-0.16, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# ---------------------------------------------------------------- Fig. A: match evaluation (double column)
def fig_match():
    S = json.load(open(REPO / "runs/paper_eval/stats_final/stats.json"))
    A, Cmp = S["arms"], S["comparisons"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.25), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.55})
    for i, (k, name, c, m) in enumerate(MATCH):
        v = A[k]["goal_diff"]
        a.errorbar(v[0], i, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=m, color=c, mfc=c, ms=5, lw=1.4)
        a.text(3.25, i, f"{A[k]['W']}–{A[k]['D']}–{A[k]['L']}", va="center", ha="right", fontsize=7, color=INK2)
    a.text(3.25, -0.85, "W–D–L", ha="right", fontsize=7, color=INK2, style="italic")
    a.axvline(0, color=INK2, lw=0.6)
    a.set_yticks(range(len(MATCH)), [f"{n} ($n$={A[k]['n']})" for k, n, _, _ in MATCH])
    a.set_ylim(len(MATCH) - 0.5, -0.9); a.set_xlim(-3, 3.3); a.grid(axis="y", visible=False)
    a.set_xlabel("Goal difference per match")
    plabel(a, "(a)", x=-0.62)

    mets = [("possession_pct", "Possession"), ("recoveries", "Recoveries"), ("duels_won_pct", "Duels won"),
            ("falls", "Falls (fewer = better)"), ("goals_against", "Goals against"), ("ball_in_opp_half_pct", "Ball in opp. half"),
            ("ball_in_final_third_pct", "Ball in final third"), ("shots_on_target", "Shots on target"), ("goals_for", "Goals for")]
    arms = MATCH[:3]
    off = {"I": -0.22, "G": 0.0, "P": 0.22}
    for j, (key, lab) in enumerate(mets):
        base = A["C"][key][0]
        for k, name, c, m in arms:
            d, lo, hi, p = Cmp[f"{k}-C"][key]
            r, rl, rh = 100 * d / base, 100 * lo / base, 100 * hi / base
            sig = lo > 0 or hi < 0
            b.errorbar(r, j + off[k], xerr=[[r - rl], [rh - r]], fmt=m, color=c, mfc=c if sig else "white", mew=1.0, ms=4, lw=1.1)
    b.axvline(0, color=GREY, lw=0.8)
    b.set_yticks(range(len(mets)), [l for _, l in mets]); b.set_ylim(len(mets) - 0.5, -0.6)
    b.set_xlabel("Difference to B-Human in same seat (%)")
    b.grid(axis="y", visible=False)
    h = [Line2D([], [], marker=m, color=c, ls="", ms=4) for _, _, c, m in arms] + \
        [Line2D([], [], marker="o", color=INK2, mfc="white", ls="", ms=4)]
    b.legend(h, [n for _, n, _, _ in arms] + ["hollow: 95% CI includes 0"], loc="lower right", fontsize=6.5,
             handletextpad=0.3, borderaxespad=0.2)
    plabel(b, "(b)", x=-0.42)
    save(fig, "icra_fig_match_evaluation")


# ---------------------------------------------------------------- Fig. B: training ablation (double column)
def fig_ablation():
    rows = list(csv.DictReader(open(REPO / "runs/paper_offline/ablation/per_checkpoint.csv")))
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], int(r["step"]))].append(r)
    fig = plt.figure(figsize=(DOUBLE, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45], wspace=0.42)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, key, ylab in ((axs[0], "balanced_acc", "Balanced accuracy (held-out)"),
                          (axs[1], "pred_entropy_bits", "Skill entropy (bits)")):
        for a, name, c, m in ABL:
            steps = sorted(s for (aa, s) in by if aa == a)
            mc = [tci([float(r[key]) for r in by[(a, s)]]) for s in steps]
            mu = np.array([x[0] for x in mc]); h = np.array([x[1] for x in mc])
            ax.fill_between(np.array(steps) / 1e3, np.maximum(mu - h, 0), mu + h, color=c, alpha=0.18, lw=0)
            ax.plot(np.array(steps) / 1e3, mu, color=c, marker=m, ms=3.5, label=name)
        steps = sorted(s for (aa, s) in by if aa == "PPOLONG")
        mc = [tci([float(r[key]) for r in by[("PPOLONG", s)]]) for s in steps]
        ax.plot(np.array(steps) / 1e3, [x[0] for x in mc], color=MAGENTA, ls=(0, (3, 2)), lw=1.0)
        ax.set_xlabel("PPO steps ($\\times 10^3$)"); ax.set_ylabel(ylab); ax.set_xlim(-1, 31)
    axs[0].axhline(0.315, color=GREY, lw=0.7, ls=":")
    axs[0].text(30.5, 0.33, "mask-only chance", ha="right", fontsize=6.5, color=GREY)
    axs[0].text(30.5, 0.165, "PPO alone, 30k", ha="right", fontsize=6.5, color=MAGENTA)
    axs[0].set_ylim(0.1, 0.65)
    # significance at 10k
    g = lambda a: [float(r["balanced_acc"]) for r in by[(a, 10000)]]  # noqa: E731
    p1 = stats.ttest_ind(g("NOKL"), g("FULL"), equal_var=False).pvalue
    y0, y1 = np.mean(g("FULL")), np.mean(g("NOKL"))
    axs[0].plot([11.2, 11.6, 11.6, 11.2], [y0, y0, y1, y1], color=INK2, lw=0.6, clip_on=False)
    axs[0].text(11.9, (y0 + y1) / 2, stars(p1), fontsize=6.5, va="center", color=INK2)
    plabel(axs[0], "(a)", x=-0.3); plabel(axs[1], "(b)", x=-0.3)

    skills = ["stand", "walk", "shoot", "pass", "dribble", "block", "mark"]
    ax = axs[2]
    for i, (a, name, c, m) in enumerate(ABL):
        mc = [tci([float(r[f"recall_{s}"]) for r in by[(a, 10000)]]) for s in skills]
        x = np.arange(len(skills)) + (i - 2) * 0.15
        ax.errorbar(x, [v for v, _ in mc], yerr=[h for _, h in mc], fmt=m, color=c, ms=3.5, lw=1.0)
    ax.set_xticks(range(len(skills)), skills, rotation=30, ha="right", rotation_mode="anchor"); ax.set_ylabel("Recall at 10k steps"); ax.set_ylim(-0.05, 1.05)
    for xx in np.arange(len(skills) - 1) + 0.5:
        ax.axvline(xx, color=GRID, lw=0.5)
    ax.grid(axis="x", visible=False)
    plabel(ax, "(c)", x=-0.2)
    h = [Line2D([], [], color=c, marker=m, ms=3.5) for _, _, c, m in ABL] + [Line2D([], [], color=MAGENTA, ls=(0, (3, 2)), lw=1.0)]
    fig.legend(h, [n for _, n, _, _ in ABL] + ["PPO alone, 30k steps"], loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.09), columnspacing=1.2, handletextpad=0.4)
    save(fig, "icra_fig_ablation")


# ---------------------------------------------------------------- Fig. C: policy lineage on held-out data (single column)
def fig_lineage():
    S = json.load(open(REPO / "runs/paper_offline/heldout_policies/summary.json"))
    E, Bh = S["expert"]["models"], S["bhuman"]["models"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(SINGLE, 3.3), gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.55})
    keys = ["balanced_acc", "recall_shoot", "recall_pass", "recall_dribble", "recall_block", "recall_mark"]
    labs = ["Bal. acc.", "Shoot", "Pass", "Dribble", "Block", "Mark"]
    for i, (k, name, c, m) in enumerate(LIN):
        v = np.array([E[k][q] for q in keys]); x = np.arange(len(keys)) + (i - 1) * 0.22
        a.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=m, color=c, ms=4, lw=1.0, label=name)
    a.hlines(0.315, -0.4, 0.4, color=GREY, lw=0.8, ls=":")
    a.text(0, 0.24, "chance", ha="center", fontsize=6, color=GREY)
    a.set_xticks(range(len(keys)), labs); a.set_ylim(-0.05, 1.08); a.set_ylabel("vs expert labels")
    for xx in np.arange(len(keys) - 1) + 0.5:
        a.axvline(xx, color=GRID, lw=0.5)
    a.grid(axis="x", visible=False)
    a.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.02), handletextpad=0.3, columnspacing=0.9)
    plabel(a, "(a)", x=-0.2, y=1.13)
    keys = ["agree_striker", "agree_open_support", "agree_off_ball"]
    for i, (k, name, c, m) in enumerate(LIN):
        v = np.array([Bh[k][q] for q in keys]); x = np.arange(3) + (i - 1) * 0.22
        b.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=m, color=c, ms=4, lw=1.0)
    b.set_xticks(range(3), ["Striker", "Open support", "Off-ball support"]); b.set_ylim(-0.05, 1.05)
    b.set_ylabel("vs B-Human's choices")
    for xx in (0.5, 1.5):
        b.axvline(xx, color=GRID, lw=0.5)
    b.grid(axis="x", visible=False)
    plabel(b, "(b)", x=-0.2, y=1.06)
    save(fig, "icra_fig_lineage_heldout")


# ---------------------------------------------------------------- Fig. D: role conditioning (single column)
def fig_roles():
    S = json.load(open(REPO / "runs/paper_offline/role_conditioning/summary.json"))
    R = S["in_match"]["G"]["roles"]
    roles = [("striker", "Striker"), ("open_support", "Open support"), ("off_ball_support", "Off-ball support")]
    fig, (a, b) = plt.subplots(2, 1, figsize=(SINGLE, 3.0), gridspec_kw={"height_ratios": [0.9, 1.1], "hspace": 0.75})
    left = np.zeros(3)
    for sk, c in (("walk", BLUE), ("shoot", ORANGE), ("pass", AQUA)):
        v = np.array([R[r][f"skill_{sk}"][0] for r, _ in roles])
        a.barh(range(3), v, left=left, color=c, height=0.6, edgecolor="white", linewidth=1.0, label=sk.capitalize())
        for i, (vv, ll) in enumerate(zip(v, left)):
            if vv > 0.07:
                a.text(ll + vv / 2, i, f"{100 * vv:.0f}%", ha="center", va="center", fontsize=6.5, color="white")
        left += v
    a.set_yticks(range(3), [n for _, n in roles]); a.invert_yaxis(); a.set_xlim(0, 1)
    a.set_xlabel("Share of executed skills in matches"); a.grid(axis="y", visible=False)
    a.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.33), handletextpad=0.3)
    plabel(a, "(a)", x=-0.33, y=1.12)
    C = S["counterfactual"]["pairs"]
    short = {"striker": "S", "open_support": "O", "off_ball_support": "F"}
    pairs = list(C)
    x = np.arange(len(pairs))
    b.bar(x - 0.19, [100 * C[k]["skill_changed_network_only"] for k in pairs], width=0.36, color=BLUE, label="Network only")
    b.bar(x + 0.19, [100 * C[k]["skill_changed_full_pipeline"] for k in pairs], width=0.36, color=ORANGE, label="Network + role mask")
    b.set_xticks(x, [f"{short[k.split(' -> ')[0]]}$\\rightarrow${short[k.split(' -> ')[1]]}" for k in pairs])
    b.set_ylabel("Skill changed (%)"); b.set_xlabel("Role input swapped (S/O/F = striker/open/off-ball)")
    b.legend(loc="upper right", handletextpad=0.3); b.grid(axis="x", visible=False)
    plabel(b, "(b)", x=-0.17, y=1.04)
    save(fig, "icra_fig_role_conditioning")


# ---------------------------------------------------------------- Fig. E: training curves (double column, appendix)
def fig_training():
    cur = list(csv.DictReader(open(REPO / "runs/paper_offline/distill/curves.csv")))
    it = []
    for d in sorted(glob.glob(str(REPO / "runs/kickstart/loop_20261006_224436/iter_*"))) + \
            sorted(glob.glob(str(REPO / "runs/kickstart/round2_20261008_013711/iter_*"))):
        p = Path(d + "/learner.log")
        ls = [l for l in open(p) if l.startswith("{")] if p.exists() else []
        if ls:
            it.append(json.loads(ls[-1]))
    tr = list(csv.DictReader(open(REPO / "runs/paper_offline/ablation/train_curves.csv")))
    fig, axs = plt.subplots(1, 4, figsize=(DOUBLE, 1.9), gridspec_kw={"wspace": 0.5})
    ep = sorted({int(r["epoch"]) for r in cur})
    for k, lab, c, m in (("train_loss", "Train loss", BLUE, "o"), ("val_ce", "Held-out CE", ORANGE, "s")):
        mc = [tci([float(r[k]) for r in cur if int(r["epoch"]) == e]) for e in ep]
        mu = np.array([x[0] for x in mc]); h = np.array([x[1] for x in mc])
        axs[0].fill_between(ep, mu - h, mu + h, color=c, alpha=0.2, lw=0)
        axs[0].plot(ep, mu, color=c, marker=m, ms=3, label=lab)
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].legend(loc="center right"); axs[0].set_xticks([1, 4, 8])
    x = [j["iteration"] for j in it]
    for k, lab, c, m in (("ce", "Teacher CE", BLUE, "o"), ("entropy", "Entropy", ORANGE, "s"), ("vloss", "Value loss", AQUA, "^")):
        axs[1].plot(x, [r["train"][k] for r in it], color=c, marker=m, ms=3, lw=1.0, label=lab)
    axs[1].axvline(12, color=GREY, lw=0.7, ls=":")
    axs[1].set_xlabel("RL iteration"); axs[1].set_ylabel("Loss / entropy"); axs[1].set_ylim(0.05, 0.25)
    axs[1].legend(loc="upper center", ncol=3, fontsize=5.5, handlelength=1.0, columnspacing=0.6, handletextpad=0.3)
    gd = [r["batch"]["gd_mean"] for r in it]; n = [r["batch"]["valid_matches"] for r in it]
    axs[2].bar(x, gd, color=GREY, width=0.7)
    axs[2].axhline(0, color=INK2, lw=0.6); axs[2].axvline(12, color=GREY, lw=0.7, ls=":")
    axs[2].set_xlabel("RL iteration"); axs[2].set_ylabel("Batch goal diff."); axs[2].grid(axis="x", visible=False)
    for a, name, c, m in ABL:
        seeds = sorted({r["seed"] for r in tr if r["arm"] == a})
        grid = np.arange(256, 10001, 512); ys = []
        for s in seeds:
            pts = sorted((int(r["step"]), float(r["value"])) for r in tr
                         if r["arm"] == a and r["seed"] == s and r["tag"] == "Reward / Instantaneous reward (mean)")
            if len(pts) > 1:
                ys.append(np.interp(grid, [p[0] for p in pts], [p[1] for p in pts]))
        if len(ys) > 1:
            ys = np.array(ys); mu = ys.mean(0)
            axs[3].plot(grid / 1e3, np.convolve(mu, np.ones(3) / 3, mode="same"), color=c, lw=1.0)
    axs[3].set_xlabel("PPO steps ($\\times 10^3$)"); axs[3].set_ylabel("Mean step reward")
    fig.legend([Line2D([], [], color=c, lw=1.2) for _, _, c, _ in ABL], [n for n, *_ in [(x[1],) for x in ABL]],
               loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.12), fontsize=6.5, columnspacing=1.0)
    for ax, s in zip(axs, "abcd"):
        plabel(ax, f"({s})", x=-0.38)
    save(fig, "icra_fig_training_curves")


if __name__ == "__main__":
    fig_match(); fig_ablation(); fig_lineage(); fig_roles(); fig_training()
