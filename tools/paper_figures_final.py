"""Figures for 'For the final version/figures' (2026-10-10). Same IEEE style as tools/paper_figures_icra.py.

New here (interpretable skill, reward and loss figures):
  final_fig_skill_confusion   what each training variant chooses when the expert would choose X (held-out)
  final_fig_skill_share       share of chosen skills over training, full pipeline vs PPO alone
  final_fig_skill_roles       executed skills per role: final team vs stock B-Human in the same seat
  final_fig_rewards           reward during PPO (ablation, 6 seeds), evaluation return, RL stage of the final team
  final_fig_losses            imitation, PPO (policy/value/entropy, 6 seeds), coordinator and RL-stage losses
The ICRA figures (match evaluation, ablation, lineage, role conditioning, training curves) are regenerated and copied.
"""
import csv
import glob
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figures_icra as I  # noqa: E402  (applies the rcParams)
from paper_figures_icra import plt, Line2D, REPO, SINGLE, DOUBLE, ABL, GREY, INK, INK2, GRID, BLUE, ORANGE, AQUA, YELLOW, MAGENTA, plabel, tci  # noqa: E402,E501

FIN = REPO / "For the final version/figures"
SKC = {"stand": "#4a3aa7", "walk": BLUE, "shoot": ORANGE, "pass": AQUA, "dribble": YELLOW, "block": MAGENTA, "mark": "#008300"}
SKILLS = ["stand", "walk", "shoot", "pass", "dribble", "block", "mark"]
ABLNAME = {a: n for a, n, _, _ in ABL} | {"PPOLONG": "PPO alone, 30k steps"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIN / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def fig_confusion():
    C = json.load(open(REPO / "runs/paper_offline/skills/confusion.json"))
    rows, cols = C["rows"], C["cols"]
    keep = [c for c in cols if c != "observe"]
    ci = [cols.index(c) for c in keep]
    arms = ["FULL", "NOKL", "NOBC", "WSONLY", "PPO", "PPOLONG"]
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE, 4.3), gridspec_kw={"wspace": 0.12, "hspace": 0.42})
    cmap = plt.get_cmap("Blues")
    for ax, a, lab in zip(axs.flat, arms, "abcdef"):
        Mx = np.array(C["arms"][a])[:, ci]
        ax.imshow(Mx, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                v = Mx[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if v > 0.55 else INK)
        for i in range(len(rows)):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK2, lw=0.6))
        ax.set_xticks(range(len(keep)), keep, rotation=40, ha="right", rotation_mode="anchor", fontsize=6.5)
        ax.set_yticks(range(len(rows)), rows if lab in "ad" else [""] * len(rows), fontsize=6.5)
        ax.grid(False); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"({lab}) {ABLNAME[a]}  (n={C['n_seeds'][a]} seeds)", fontsize=7.5, color=INK, loc="left")
    fig.text(0.06, 0.5, "Expert's skill (held-out live states)", rotation=90, va="center", ha="center", fontsize=8)
    fig.text(0.5, 0.0, "Skill chosen by the trained policy (row %, diagonal boxed = agreement)", ha="center", fontsize=8)
    save(fig, "final_fig_skill_confusion")


def fig_share():
    S = json.load(open(REPO / "runs/paper_offline/skills/share.json"))
    cols = S["cols"]
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.1), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, a, lab in zip(axs, ["FULL", "NOKL", "PPOLONG"], "abc"):
        st = sorted(S["arms"][a], key=int)
        x = np.array([int(s) for s in st]) / 1e3
        Y = np.array([[S["arms"][a][s][cols.index(k)] for s in st] for k in SKILLS])
        ax.stackplot(x, Y, colors=[SKC[k] for k in SKILLS], edgecolor="white", linewidth=0.4)
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(0, 1); ax.grid(False)
        ax.set_xlabel("PPO steps ($\\times 10^3$)")
        ax.set_title(f"({lab}) {ABLNAME[a]}", fontsize=7.5, loc="left", color=INK)
    axs[0].set_ylabel("Share of chosen skills")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=SKC[k]) for k in SKILLS], SKILLS, loc="upper center", ncol=7,
               bbox_to_anchor=(0.5, 1.1), handlelength=1.0, columnspacing=1.0)
    save(fig, "final_fig_skill_share")


def fig_roles_vs_bhuman():
    R = json.load(open(REPO / "runs/paper_offline/skills/roles.json"))
    roles = [("striker", "Striker"), ("open_support", "Open support"), ("off_ball_support", "Off-ball support")]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
    ylab, y = [], 0
    for r, rn in roles:
        for arm, an in (("G", "Final team"), ("C", "B-Human")):
            left = 0
            for k in ("stand", "walk", "shoot", "pass", "dribble"):
                v = R[arm][r][k][0]
                ax.barh(y, v, left=left, color=SKC[k], height=0.72, edgecolor="white", linewidth=0.8)
                if v > 0.08:
                    ax.text(left + v / 2, y, f"{100 * v:.0f}%", ha="center", va="center", fontsize=6, color="white")
                left += v
            ylab.append(f"{rn} – {an}"); y += 1
        y += 0.5
    ax.set_yticks([0, 1, 2.5, 3.5, 5, 6], ylab, fontsize=6.5); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(False)
    ax.set_xlabel("Share of executed skills in evaluation matches")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=SKC[k]) for k in ("stand", "walk", "shoot", "pass")],
              ["stand", "walk", "shoot", "pass"], loc="upper center", ncol=4, bbox_to_anchor=(0.4, 1.2), handlelength=1.0)
    save(fig, "final_fig_skill_roles")


def seed_curves(tag, arms, grid):
    tr = list(csv.DictReader(open(REPO / "runs/paper_offline/ablation/train_curves.csv")))
    by = defaultdict(list)
    for r in tr:
        if r["tag"] == tag:
            by[(r["arm"], r["seed"])].append((int(r["step"]), float(r["value"])))
    out = {}
    for a in arms:
        ys = []
        for (aa, s), pts in by.items():
            if aa == a and len(pts) > 2:
                pts.sort(); ys.append(np.interp(grid, [p[0] for p in pts], [p[1] for p in pts]))
        if len(ys) > 1:
            ys = np.array(ys)
            k = np.ones(5) / 5
            ys = np.array([np.convolve(np.pad(y, 2, mode="edge"), k, mode="valid") for y in ys])
            mu = ys.mean(0); h = stats.t.ppf(0.975, len(ys) - 1) * ys.std(0, ddof=1) / np.sqrt(len(ys))
            out[a] = (mu, h, len(ys))
    return out


def kick_iters():
    it = []
    for d in sorted(glob.glob(str(REPO / "runs/kickstart/loop_20261006_224436/iter_*"))) + \
            sorted(glob.glob(str(REPO / "runs/kickstart/round2_20261008_013711/iter_*"))):
        p = Path(d + "/learner.log")
        ls = [l for l in open(p) if l.startswith("{")] if p.exists() else []
        if ls:
            it.append(json.loads(ls[-1]))
    return it


def fig_rewards():
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.1), gridspec_kw={"wspace": 0.42})
    grid = np.arange(256, 10001, 256)
    cur = seed_curves("Reward / Total reward (mean)", [a for a, *_ in ABL], grid)
    for a, n, c, m in ABL:
        if a in cur:
            mu, h, _ = cur[a]
            axs[0].fill_between(grid / 1e3, mu - h, mu + h, color=c, alpha=0.15, lw=0)
            axs[0].plot(grid / 1e3, mu, color=c, lw=1.1)
    axs[0].set_xlabel("PPO steps ($\\times 10^3$)"); axs[0].set_ylabel("Training episode return")
    plabel(axs[0], "(a)", x=-0.3)
    ev = defaultdict(list)
    for f in glob.glob(str(REPO / "runs/ablation_paper_*_s*/telemetry/*/csv/eval_summary.csv")):
        a = re.search(r"ablation_paper_([A-Z]+)_s\d+", f).group(1)
        for r in csv.DictReader(open(f)):
            ev[(a, int(r["timestep"]))].append(float(r["episode_reward_mean"]))
    for i, (a, n, c, m) in enumerate(ABL):
        st = sorted(s for (aa, s) in ev if aa == a)
        mc = [tci(ev[(a, s)]) for s in st]
        axs[1].errorbar(np.array(st) / 1e3 + (i - 2) * 0.18, [x[0] for x in mc], yerr=[x[1] for x in mc], color=c,
                        marker=m, ms=3.5, lw=1.0)
    axs[1].axhline(0, color=GREY, lw=0.6)
    axs[1].set_xlabel("PPO steps ($\\times 10^3$)"); axs[1].set_ylabel("Evaluation return (6 ep.)")
    plabel(axs[1], "(b)", x=-0.3)
    it = kick_iters()
    x = [j["iteration"] for j in it]
    axs[2].bar(x, [1e3 * j["batch"]["mean_reward_per_step"] for j in it], color=GREY, width=0.7)
    axs[2].axhline(0, color=INK2, lw=0.6); axs[2].axvline(12, color=ORANGE, lw=0.8, ls=":")
    axs[2].text(12.3, 1.6, "final", color=ORANGE, fontsize=6.5)
    axs[2].set_xlabel("RL iteration (final team)"); axs[2].set_ylabel("Mean step reward ($\\times 10^{-3}$)")
    axs[2].grid(axis="x", visible=False)
    plabel(axs[2], "(c)", x=-0.3)
    fig.legend([Line2D([], [], color=c, marker=m, ms=3.5) for _, _, c, m in ABL], [n for _, n, _, _ in ABL],
               loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.1), columnspacing=1.0)
    save(fig, "final_fig_rewards")


def coord_curve(log):
    pts = []
    for l in open(log):
        m = re.search(r"before (\d+) mm", l)
        if m:
            pts.append((0, int(m.group(1)))); base = int(re.search(r"baseline (\d+) mm", l).group(1))
        m = re.search(r"epoch (\d+): val median error (\d+) mm", l)
        if m:
            pts.append((int(m.group(1)), int(m.group(2))))
    return pts, base


def fig_losses():
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE, 3.9), gridspec_kw={"wspace": 0.45, "hspace": 0.6})
    cur = list(csv.DictReader(open(REPO / "runs/paper_offline/distill/curves.csv")))
    ep = sorted({int(r["epoch"]) for r in cur})
    for k, lab, c, m in (("train_loss", "train (CE+KL+pos.)", BLUE, "o"), ("val_ce", "held-out skill CE", ORANGE, "s")):
        mc = [tci([float(r[k]) for r in cur if int(r["epoch"]) == e]) for e in ep]
        mu = np.array([x[0] for x in mc]); h = np.array([x[1] for x in mc])
        axs[0, 0].fill_between(ep, mu - h, mu + h, color=c, alpha=0.2, lw=0)
        axs[0, 0].plot(ep, mu, color=c, marker=m, ms=3, label=lab)
    axs[0, 0].set_xlabel("Epoch"); axs[0, 0].set_ylabel("Loss"); axs[0, 0].set_xticks([1, 4, 8])
    axs[0, 0].legend(loc="center right", fontsize=6.5)
    grid = np.arange(256, 10001, 256)
    for ax, tag, lab in ((axs[0, 1], "Loss / Policy loss (player_1)", "PPO policy loss"),
                         (axs[0, 2], "Loss / Value loss (player_1)", "PPO value loss"),
                         (axs[1, 0], "Loss / Entropy loss (player_1)", "PPO entropy loss")):
        cur2 = seed_curves(tag, [a for a, *_ in ABL], grid)
        for a, n, c, m in ABL:
            if a in cur2:
                mu, h, _ = cur2[a]
                ax.fill_between(grid / 1e3, mu - h, mu + h, color=c, alpha=0.12, lw=0)
                ax.plot(grid / 1e3, mu, color=c, lw=1.0)
        ax.set_xlabel("PPO steps ($\\times 10^3$)"); ax.set_ylabel(lab)
    for (log, lab, c, m) in ((REPO / "runs/handover/pipeline_20261006_222423/train_round2_coord.log", "coord v1", BLUE, "o"),
                             (REPO / "runs/handover/pipeline_20261006_222423/train_round3_coord.log", "coord v2 (final)", ORANGE, "s")):
        pts, base = coord_curve(log)
        axs[1, 1].plot([p[0] for p in pts], [p[1] / 1e3 for p in pts], color=c, marker=m, ms=3, label=lab)
    axs[1, 1].axhline(base / 1e3, color=GREY, lw=0.7, ls=":")
    axs[1, 1].text(30, base / 1e3 + 0.03, "predict-mean baseline", ha="right", fontsize=6, color=GREY)
    axs[1, 1].set_xlabel("Epoch"); axs[1, 1].set_ylabel("Held-out median error (m)"); axs[1, 1].legend(loc="upper right", fontsize=6.5)
    it = kick_iters(); x = [j["iteration"] for j in it]
    for k, lab, c, m in (("ce", "teacher CE", BLUE, "o"), ("entropy", "entropy", ORANGE, "s"), ("vloss", "value loss", AQUA, "^")):
        axs[1, 2].plot(x, [r["train"][k] for r in it], color=c, marker=m, ms=2.5, lw=0.9, label=lab)
    axs[1, 2].axvline(12, color=GREY, lw=0.7, ls=":"); axs[1, 2].set_ylim(0.05, 0.24)
    axs[1, 2].set_xlabel("RL iteration (final team)"); axs[1, 2].set_ylabel("Loss / entropy")
    axs[1, 2].legend(loc="upper center", ncol=3, fontsize=5.5, handlelength=1.0, columnspacing=0.5, handletextpad=0.3)
    for ax, s in zip(axs.flat, "abcdef"):
        plabel(ax, f"({s})", x=-0.36)
    fig.legend([Line2D([], [], color=c, lw=1.2) for _, _, c, _ in ABL], [f"PPO ablation: {n}" for _, n, _, _ in ABL],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=6.5, columnspacing=1.0)
    save(fig, "final_fig_losses")


if __name__ == "__main__":
    FIN.mkdir(parents=True, exist_ok=True)
    I.fig_match(); I.fig_ablation(); I.fig_lineage(); I.fig_roles(); I.fig_training()
    for f in glob.glob(str(REPO / ".docs/FIGURES/icra_fig_*")):
        shutil.copy(f, FIN / Path(f).name.replace("icra_fig_", "final_fig_"))
    fig_confusion(); fig_share(); fig_roles_vs_bhuman(); fig_rewards(); fig_losses()
