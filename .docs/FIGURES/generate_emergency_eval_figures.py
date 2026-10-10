#!/usr/bin/env python3
"""Generate the emergency IEEE-style evaluation figures/tables.

Inputs are repository-local telemetry artifacts:
- v5 merged training telemetry.
- PPO-only ablation telemetry, generated during the paper sprint.
- matched SimRobot match-summary JSON files for v5 and deterministic teacher runs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

PIPELINE_CSV = ROOT / "runs/merge_v5_attack_defense/telemetry/multiagent_stage8_finetune_train_stage8_20260618_215442/csv/eval_summary.csv"
DEFENDER_PIPELINE_CSV = ROOT / "runs/defender_47_v2_blockmark/telemetry/multiagent_stage8_finetune_train_stage8_20260618_071852/csv/eval_summary.csv"
ABLATION_ROOT = ROOT / "runs/ablation_v5_ppo_only_no_bc_kl"
V5_MATCH_ROOT = ROOT / "runs/paper_match_v5"
TEACHER_MATCH_ROOT = ROOT / "runs/paper_match_bhuman_heuristic"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close()


def _read_csv(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        return []
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            clean: dict[str, float | str] = {}
            for k, v in row.items():
                if v is None or v == "":
                    clean[k] = ""
                    continue
                try:
                    clean[k] = float(v)
                except ValueError:
                    clean[k] = v
            rows.append(clean)
    return rows


def _latest_eval_csv(root: Path) -> Path | None:
    candidates = sorted(root.glob("telemetry/*/csv/eval_summary.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _latest_match_summary(root: Path) -> Path | None:
    candidates = sorted(root.glob("dataset_stage8_*/eval_match_summary.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _col(rows: Iterable[dict[str, float | str]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        val = row.get(key, np.nan)
        vals.append(float(val) if val != "" else np.nan)
    return np.asarray(vals, dtype=float)


def pipeline_vs_ppo_ablation() -> None:
    """Reward/convergence comparison between the proposed pipeline and PPO-only ablation."""
    proposed = _read_csv(PIPELINE_CSV)
    proposed_def = _read_csv(DEFENDER_PIPELINE_CSV)
    ablation_csv = _latest_eval_csv(ABLATION_ROOT)
    ablation = _read_csv(ablation_csv) if ablation_csv else []

    fig, axs = plt.subplots(1, 2, figsize=(7.15, 2.95))

    ax = axs[0]
    if proposed_def:
        ax.plot(
            _col(proposed_def, "timestep"),
            _col(proposed_def, "episode_reward_mean"),
            marker="^",
            linewidth=1.35,
            color="#52734d",
            label="Teacher-student defense pipeline",
        )
    if proposed:
        ax.plot(
            _col(proposed, "timestep"),
            _col(proposed, "episode_reward_mean"),
            marker="o",
            linewidth=1.15,
            linestyle="--",
            color="#1f4e79",
            label="v5 attack-defense merge",
        )
    if ablation:
        ax.plot(
            _col(ablation, "timestep"),
            _col(ablation, "episode_reward_mean"),
            marker="s",
            linewidth=1.4,
            color="#8a3d2b",
            label="PPO-only fine-tuning ablation",
        )
    ax.axhline(0.0, color="#666666", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Fine-tuning timestep")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("A. Reward during Stage-8 fine-tuning")
    ax.grid(axis="y", color="#dddddd", linewidth=0.45)
    ax.legend(frameon=False, loc="best")
    if ablation:
        ax.text(
            0.02,
            -0.30,
            "PPO-only run: no BC/KL/replay; stopped after two complete evals for the timed paper sprint.",
            transform=ax.transAxes,
            fontsize=6.8,
            color="#333333",
        )

    ax = axs[1]
    if proposed:
        ax.plot(
            _col(proposed, "timestep"),
            _col(proposed, "defender_score"),
            marker="o",
            linewidth=1.4,
            color="#1f4e79",
            label="v5 merge pipeline",
        )
    if proposed_def:
        ax.plot(
            _col(proposed_def, "timestep"),
            _col(proposed_def, "defender_score"),
            marker="^",
            linewidth=1.15,
            color="#52734d",
            label="defense specialist source",
        )
    if ablation:
        ax.plot(
            _col(ablation, "timestep"),
            _col(ablation, "defender_score"),
            marker="s",
            linewidth=1.4,
            color="#8a3d2b",
            label="PPO-only ablation",
        )
    ax.set_ylim(0.55, 1.03)
    ax.set_xlabel("Fine-tuning timestep")
    ax.set_ylabel("Defender score")
    ax.set_title("B. Defensive competence stability")
    ax.grid(axis="y", color="#dddddd", linewidth=0.45)
    ax.legend(frameon=False, loc="best")

    fig.suptitle("Teacher-Student Pipeline vs PPO-Only Ablation", y=1.04, fontsize=10)
    fig.tight_layout()
    _save("fig_pipeline_vs_ppo_ablation_reward")


def _flatten_action_pct(summary: dict) -> dict[str, float]:
    counts = {str(k): float(v) for k, v in summary.get("action_distribution", {}).items()}
    if not counts:
        for role_counts in summary.get("skill_by_role", {}).values():
            for k, v in role_counts.items():
                counts[str(k)] = counts.get(str(k), 0.0) + float(v)
    total = sum(counts.values()) or 1.0
    return {k: 100.0 * v / total for k, v in counts.items()}


def _load_match_summary(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def match_metrics_table_and_actions() -> None:
    """Matched SimRobot benchmark table and action-distribution figure."""
    v5_path = _latest_match_summary(V5_MATCH_ROOT)
    teacher_path = _latest_match_summary(TEACHER_MATCH_ROOT)
    v5 = _load_match_summary(v5_path)
    teacher = _load_match_summary(teacher_path)
    if not v5 or not teacher:
        missing = []
        if not v5:
            missing.append(str(V5_MATCH_ROOT / "dataset_stage8_*/eval_match_summary.json"))
        if not teacher:
            missing.append(str(TEACHER_MATCH_ROOT / "dataset_stage8_*/eval_match_summary.json"))
        print("[figures] match summaries missing; skipped match figures:")
        for item in missing:
            print(f"  - {item}")
        return

    labels = ["v5 merged RL", "B-Human/Sabana-Herons\nheuristic baseline"]
    summaries = [v5, teacher]

    table_rows = [
        ("Episodes", "episodes", "{:.0f}"),
        ("Goal proxy", "scored", "{:.0f}"),
        ("Conceded proxy", "conceded", "{:.0f}"),
        ("Ball progression (m/ep.)", "ball_progression_mm", "{:.2f}"),
        ("Pass attempts / ep.", "pass_attempts", "{:.2f}"),
        ("Possession proxy (%)", "possession_proxy_pct", "{:.1f}"),
        ("Ball recoveries / ep.", "ball_recoveries_per_episode", "{:.2f}"),
        ("Role co-occurrence (%)", "role_co_occurrence_pct", "{:.1f}"),
        ("Collision proxy (%)", "teammate_collision_proxy_pct", "{:.1f}"),
    ]

    def metric(summary: dict, key: str) -> float:
        val = float(summary.get(key, 0.0))
        if key == "ball_progression_mm":
            return val / 1000.0
        if key == "pass_attempts":
            return val / max(1.0, float(summary.get("episodes", 1.0)))
        return val

    csv_path = OUT / "table_match_metrics_v5_vs_bhuman.csv"
    md_path = OUT / "table_match_metrics_v5_vs_bhuman.md"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Metric", "v5 merged RL", "B-Human/Sabana-Herons heuristic baseline"])
        for name, key, fmt in table_rows:
            writer.writerow([name] + [fmt.format(metric(s, key)) for s in summaries])
    with md_path.open("w") as fh:
        fh.write("| Metric | v5 merged RL | B-Human/Sabana-Herons heuristic baseline |\n")
        fh.write("|---|---:|---:|\n")
        for name, key, fmt in table_rows:
            fh.write(f"| {name} | {fmt.format(metric(v5, key))} | {fmt.format(metric(teacher, key))} |\n")
        fh.write("\n")
        fh.write(f"Source v5: `{v5_path.relative_to(ROOT)}`\n\n")
        fh.write(f"Source baseline: `{teacher_path.relative_to(ROOT)}`\n")

    v5_actions = _flatten_action_pct(v5)
    teacher_actions = _flatten_action_pct(teacher)
    match_metric_labels = [
        "Possession\nproxy",
        "Recoveries\nx10",
        "Progress\nm x100",
        "Collision\nproxy",
        "Mark+block\nactions",
    ]
    v5_metric_values = [
        metric(v5, "possession_proxy_pct"),
        metric(v5, "ball_recoveries_per_episode") * 10.0,
        metric(v5, "ball_progression_mm") * 100.0,
        metric(v5, "teammate_collision_proxy_pct"),
        v5_actions.get("mark", 0.0) + v5_actions.get("block", 0.0),
    ]
    teacher_metric_values = [
        metric(teacher, "possession_proxy_pct"),
        metric(teacher, "ball_recoveries_per_episode") * 10.0,
        metric(teacher, "ball_progression_mm") * 100.0,
        metric(teacher, "teammate_collision_proxy_pct"),
        teacher_actions.get("mark", 0.0) + teacher_actions.get("block", 0.0),
    ]
    x = np.arange(len(match_metric_labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.15, 3.05))
    b1 = ax.bar(x - width / 2, v5_metric_values, width, color="#1f4e79", edgecolor="#222222", linewidth=0.45, label="v5 merged RL")
    b2 = ax.bar(x + width / 2, teacher_metric_values, width, color="#8a8f96", edgecolor="#222222", linewidth=0.45, label="Heuristic baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(match_metric_labels)
    ax.set_ylabel("Scaled value")
    ax.set_title("Matched Stage-8 Benchmark Metrics")
    ax.grid(axis="y", color="#dddddd", linewidth=0.45)
    ax.legend(frameon=False, loc="upper right")
    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=6.8)
    ax.text(
        0.0,
        -0.27,
        "Scaled columns: recoveries x10; progression in meters x100. Goal and conceded proxies were 0/15 for both.",
        transform=ax.transAxes,
        fontsize=6.8,
        color="#333333",
    )
    fig.tight_layout()
    _save("fig_match_metrics_v5_vs_bhuman")

    fig, ax = plt.subplots(figsize=(7.15, 2.95))
    ax.axis("off")
    cell_text = []
    for name, key, fmt in table_rows:
        cell_text.append([name] + [fmt.format(metric(s, key)) for s in summaries])
    tab = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "v5 merged RL", "B-Human/Sabana-Herons heuristic baseline"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.38, 0.22, 0.34],
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.4)
    tab.scale(1.0, 1.35)
    for (row, col), cell in tab.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#555555")
        if row == 0:
            cell.set_facecolor("#dfe6eb")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(ha="left")
    ax.set_title("Matched SimRobot Stage-8 Benchmark: v5 RL vs Deterministic Baseline", pad=10)
    ax.text(
        0.0,
        -0.08,
        "Goal/conceded are field-crossing proxies; possession/recovery use ball-distance and ball-control skill proxies.",
        transform=ax.transAxes,
        fontsize=7,
        color="#333333",
    )
    _save("table_match_metrics_v5_vs_bhuman")

    action_pcts = [_flatten_action_pct(s) for s in summaries]
    skills = sorted(set(action_pcts[0]) | set(action_pcts[1]), key=lambda k: -max(d.get(k, 0.0) for d in action_pcts))
    colors = {
        "walk": "#6d737a",
        "stand": "#b4b4b4",
        "dribble": "#4f6d7a",
        "shoot": "#8a3d2b",
        "pass": "#52734d",
        "block": "#946f3b",
        "mark": "#7b5a7a",
        "observe": "#999999",
    }
    fig, ax = plt.subplots(figsize=(6.35, 2.95))
    bottoms = np.zeros(2)
    x = np.arange(2)
    for skill in skills:
        vals = np.array([d.get(skill, 0.0) for d in action_pcts])
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=skill,
            color=colors.get(skill, "#cccccc"),
            edgecolor="#222222",
            linewidth=0.35,
        )
        for i, val in enumerate(vals):
            if val >= 7.0:
                ax.text(i, bottoms[i] + val / 2.0, f"{skill}\n{val:.0f}%", ha="center", va="center", fontsize=6.5)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Executed action share (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#dddddd", linewidth=0.45)
    ax.legend(ncol=min(4, max(1, len(skills))), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_title("Action Distribution in the Matched Stage-8 Benchmark")
    fig.tight_layout()
    _save("fig_action_distribution_v5_vs_bhuman")


def main() -> int:
    pipeline_vs_ppo_ablation()
    match_metrics_table_and_actions()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
