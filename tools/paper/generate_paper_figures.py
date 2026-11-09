#!/usr/bin/env python3
"""Generate paper-ready evaluation figures for the v5 RL football agent notes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[2] / "results/paper/figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def savefig(name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close()


def pareto_skill_labels() -> None:
    skills = ["walk", "stand", "dribble", "block", "shoot", "pass", "mark"]
    counts = np.array([62323, 57667, 11563, 2817, 2565, 2015, 673], dtype=float)
    total = counts.sum()
    pct = counts / total * 100
    cumulative = np.cumsum(pct)

    fig, ax = plt.subplots(figsize=(6.75, 2.85))
    x = np.arange(len(skills))
    bars = ax.bar(x, counts, color="#5b5f66", edgecolor="#222222", linewidth=0.55)
    ax.set_ylabel("Positive skill-label count")
    ax.set_xlabel("Skill")
    ax.set_xticks(x)
    ax.set_xticklabels(skills)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.45)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(x, cumulative, color="#111111", marker="o", linewidth=1.45, markersize=3.5)
    ax2.axhline(80, color="#777777", linewidth=0.85, linestyle="--")
    ax2.set_ylabel("Cumulative percentage")
    ax2.set_ylim(0, 105)
    ax2.spines["top"].set_visible(False)

    for i in [0, 1]:
        ax.text(
            bars[i].get_x() + bars[i].get_width() / 2,
            counts[i] + total * 0.012,
            f"{pct[i]:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.4,
        )

    ax2.annotate(
        "walk + stand = 85.94%",
        xy=(1, cumulative[1]),
        xytext=(1.8, 67),
        arrowprops=dict(arrowstyle="->", linewidth=0.7, color="#333333"),
        fontsize=7.4,
    )
    ax2.annotate(
        "pass + shoot + mark = 3.76%",
        xy=(5.55, cumulative[5]),
        xytext=(3.9, 47),
        arrowprops=dict(arrowstyle="->", linewidth=0.7, color="#333333"),
        fontsize=7.4,
    )
    ax.set_title("Skill-Label Distribution in the v5 Merged Training Corpus")
    fig.tight_layout()
    savefig("fig_pareto_v5_skill_labels")


def attack_preservation() -> None:
    labels = ["Skill\nagreement", "Pass\nrecall", "Shoot\nrecall", "Dribble\nrecall"]
    values = [96.7, 100.0, 99.9, 96.4]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(5.8, 2.85))
    bars = ax.bar(x, values, color=["#4f6d7a", "#6b8f71", "#6b8f71", "#6b8f71"], edgecolor="#222222", linewidth=0.55)
    ax.set_ylim(90, 101)
    ax.set_ylabel("Percent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.45)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.22, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title("v5 Attack Preservation Against v4.2 Reference Policy")
    ax.text(
        0.02,
        0.04,
        "Reference corpus: 105,624 Stage-7 attack rows",
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout()
    savefig("fig_attack_preservation_v5_vs_v42")


def combined_results() -> None:
    fig, axs = plt.subplots(2, 2, figsize=(7.15, 4.35))
    fig.suptitle("Summary Validation of the v5 Merged ONNX Policy", y=0.995, fontsize=10)

    ax = axs[0, 0]
    labels = ["agreement", "pass", "shoot", "dribble"]
    vals = [96.7, 100.0, 99.9, 96.4]
    ax.bar(labels, vals, color="#5f7f95", edgecolor="#222222", linewidth=0.5)
    ax.set_ylim(90, 101)
    ax.set_title("A. Attack preservation")
    ax.set_ylabel("Percent")
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=7)

    ax = axs[0, 1]
    labels = ["mark\nrecall", "defender\nscore", "goal-side\nrate"]
    vals = [93.2, 76.0, 90.0]
    ax.bar(labels, vals, color="#748c69", edgecolor="#222222", linewidth=0.5)
    ax.set_ylim(0, 105)
    ax.set_title("B. Defense insertion")
    ax.set_ylabel("Percent / scaled score")
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 2.0, f"{v:.1f}", ha="center", fontsize=7)

    ax = axs[1, 0]
    labels = ["role\nco-occur.", "striker\nshare", "open\nshare", "off-ball\nshare", "forced\noff-ball mark"]
    vals = [100.0, 33.3, 33.3, 33.3, 61.0]
    colors = ["#6d737a", "#9a9a9a", "#9a9a9a", "#9a9a9a", "#748c69"]
    ax.bar(labels, vals, color=colors, edgecolor="#222222", linewidth=0.5)
    ax.set_ylim(0, 105)
    ax.set_title("C. Role-conditioned execution")
    ax.set_ylabel("Percent")
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 2.0, f"{v:.1f}", ha="center", fontsize=7)

    ax = axs[1, 1]
    labels = ["gate\nviolations", "skill flips\nnoise σ=0.05", "logit\nmismatch ×1e6"]
    vals = [0.0, 4.5, 7.6]
    ax.bar(labels, vals, color=["#6d737a", "#a37c40", "#6d737a"], edgecolor="#222222", linewidth=0.5)
    ax.set_ylim(0, 10)
    ax.set_title("D. Deployment parity and robustness")
    ax.set_ylabel("Value")
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    for i, v in enumerate(vals):
        label = "0" if v == 0 else f"{v:.1f}"
        ax.text(i, v + 0.25, label, ha="center", fontsize=7)
    ax.text(0.02, 0.92, "Deterministic inference; residuals bounded", transform=ax.transAxes, fontsize=7)

    for ax in axs.ravel():
        ax.tick_params(axis="x", rotation=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    savefig("fig_combined_v5_validation")


def version_timeline() -> None:
    versions = ["v1", "v2b", "v3/v3.1", "v4", "v4.2", "Stage-8", "v5"]
    x = np.arange(len(versions))
    notes = [
        "obs26\n0/5 eval",
        "obs47\nclean finish 1.000",
        "audit gaps\n204k/262k rows",
        "pass overlay\n434,639 rows",
        "attack anchor\npass ~0.99",
        "defense source\ndef. score 0.767",
        "merged ONNX\n96.7% atk, 93.2% mark",
    ]

    fig, ax = plt.subplots(figsize=(7.3, 2.55))
    ax.plot(x, np.zeros_like(x), color="#222222", linewidth=1.0)
    ax.scatter(x, np.zeros_like(x), s=65, color="#4f6d7a", edgecolor="#222222", zorder=3)
    for i, (ver, note) in enumerate(zip(versions, notes)):
        y = 0.28 if i % 2 == 0 else -0.34
        va = "bottom" if y > 0 else "top"
        ax.plot([i, i], [0, y * 0.72], color="#777777", linewidth=0.65)
        ax.text(i, y, f"{ver}\n{note}", ha="center", va=va, fontsize=7.2)
    ax.set_ylim(-0.72, 0.68)
    ax.set_xlim(-0.45, len(versions) - 0.55)
    ax.axis("off")
    ax.set_title("Development Timeline from Individual obs26 Policy to v5 Merged Agent", pad=8)
    fig.tight_layout()
    savefig("fig_version_timeline_v1_to_v5")


def dataset_evolution() -> None:
    corpora = ["v1\nobs26", "v3\nattack", "v3.1\nattack", "v4\nattack", "Stage-7\nfinal", "Stage-8\ndefense", "v5\nmerged"]
    rows = np.array([17815, 204474, 261646, 434639, 105624, 33999, 139623])
    episodes = ["ep n/r", "103 ep", "144 ep", "291 ep", "45 ep", "29 ep", "74 ep"]

    fig, ax = plt.subplots(figsize=(7.0, 3.35))
    x = np.arange(len(corpora))
    bars = ax.bar(x, rows / 1000, color="#6d737a", edgecolor="#222222", linewidth=0.5)
    ax.set_ylabel("Rows (thousands)")
    ax.set_xticks(x)
    ax.set_xticklabels(corpora)
    ax.set_ylim(0, 520)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.45)
    ax.set_axisbelow(True)
    for bar, row, ep in zip(bars, rows, episodes):
        y = row / 1000
        label = f"{row:,}\n{ep}"
        if y > 380:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y - 18,
                label,
                ha="center",
                va="top",
                fontsize=6.8,
                color="white",
            )
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, y + 10, label, ha="center", va="bottom", fontsize=6.8)
    ax.set_title("Dataset Evolution and Scale Across Agent Versions", pad=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig("fig_dataset_evolution")


def limitations_matrix() -> None:
    labels = [
        "Attack\npreservation",
        "Defense\ninsertion",
        "Role\nconditioning",
        "ONNX\nparity",
        "Full contested\nmatch",
        "Live pass\ncompletion",
        "Goals\nconceded",
        "Real-match\nperception",
        "Block/mark\nexecution",
    ]
    status = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0]).reshape(1, -1)
    colors = np.array([[0.45, 0.58, 0.43], [0.72, 0.72, 0.72]])

    fig, ax = plt.subplots(figsize=(7.2, 1.75))
    img = colors[1 - status]
    ax.imshow(img, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks([])
    ax.set_title("Validation Coverage and Remaining Experimental Gaps")
    for i, s in enumerate(status.ravel()):
        ax.text(i, 0, "validated" if s else "future", ha="center", va="center", fontsize=7, color="#111111")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    savefig("fig_validation_coverage_matrix")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pareto_skill_labels()
    attack_preservation()
    combined_results()
    version_timeline()
    dataset_evolution()
    limitations_matrix()


if __name__ == "__main__":
    main()
