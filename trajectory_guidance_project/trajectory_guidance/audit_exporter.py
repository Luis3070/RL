"""
audit_exporter.py — Auditoría estática + validación de dataset para trajectory_guidance.

Actualizado para verificar:
  FIX-A: Que evaluate_solver usa replay_closed_loop (no replay_plan open-loop)
  FIX-B: Que quality está normalizada por horizonte
  FIX-C: Que min_quality_score es suficientemente bajo para stages largos
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Finding:
    severity: str
    title: str
    file: str
    evidence: str
    why_it_matters: str
    recommendation: str


@dataclass
class AuditReport:
    project_root: str
    findings: List[Finding] = field(default_factory=list)
    dataset_summary: Dict[str, Any] = field(default_factory=dict)

    def add(self, **kwargs: Any) -> None:
        self.findings.append(Finding(**kwargs))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_root": self.project_root,
            "findings": [asdict(f) for f in self.findings],
            "dataset_summary": self.dataset_summary,
        }


class SourceInspector:
    def __init__(self, project_root: Path):
        self.root = project_root
        self.files = {
            "generate_dataset": self.root / "generate_dataset.py",
            "evaluate_solver": self.root / "evaluate_solver.py",
            "settings": self.root / "settings.py",
            "bridge": self.root / "bridge.py",
            "dataset": self.root / "dataset.py",
            "solver": self.root / "solver.py",
        }
        self.text = {
            name: path.read_text(encoding="utf-8")
            for name, path in self.files.items()
            if path.exists()
        }

    def exists(self, name: str) -> bool:
        return name in self.text

    def get(self, name: str) -> str:
        return self.text[name]


class DatasetInspector:
    def __init__(self, path: Optional[Path]):
        self.path = path
        self.samples: List[dict] = []
        if path and path.exists():
            with path.open("rb") as f:
                raw = pickle.load(f)
            if isinstance(raw, list):
                self.samples = raw

    def summarize(self) -> Dict[str, Any]:
        if not self.samples:
            return {
                "dataset_path": str(self.path) if self.path else None,
                "exists": bool(self.path and self.path.exists()),
                "num_samples": 0,
            }

        by_traj: Dict[str, List[dict]] = {}
        by_stage: Dict[int, List[dict]] = {}
        for s in self.samples:
            tid = str(s.get("trajectory_id", "<missing>"))
            by_traj.setdefault(tid, []).append(s)
            by_stage.setdefault(int(s.get("stage_id", -1)), []).append(s)

        traj_rows = []
        ended_on_touch_only = 0
        missing_seed_info = 0
        for tid, rows in by_traj.items():
            rows_sorted = sorted(rows, key=lambda r: int(r.get("t", 0)))
            last = rows_sorted[-1]
            touched_any = any(
                bool((r.get("event", {}) or {}).get("ball_touched", False))
                for r in rows_sorted
            )
            dribble_steps = sum(
                1 for r in rows_sorted
                if bool((r.get("event", {}) or {}).get("dribbling", False))
            )
            goal_any = any(
                bool((r.get("event", {}) or {}).get("goal_scored", False))
                for r in rows_sorted
            )
            touch_only = touched_any and not goal_any and dribble_steps == 0
            if touch_only:
                ended_on_touch_only += 1
            if "reset_seed" not in last and "world_snapshot" not in last:
                missing_seed_info += 1
            traj_rows.append({
                "trajectory_id": tid,
                "stage_id": int(last.get("stage_id", -1)),
                "num_steps": len(rows_sorted),
                "max_t": int(last.get("t", 0)),
                "done_last": bool(last.get("done", False)),
                "touched_any": touched_any,
                "dribble_steps": dribble_steps,
                "goal_any": goal_any,
                "quality_score": float(last.get("quality_score", 0.0)),
                "touch_only_signature": touch_only,
            })

        stage_summary = {}
        for stage_val, rows in by_stage.items():
            traj_ids = {
                str(r.get("trajectory_id", "<missing>")) for r in rows}
            stage_summary[str(stage_val)] = {
                "num_samples": len(rows),
                "num_trajectories": len(traj_ids),
            }

        return {
            "dataset_path": str(self.path),
            "exists": True,
            "num_samples": len(self.samples),
            "num_trajectories": len(by_traj),
            "touch_only_trajectories": ended_on_touch_only,
            "trajectories_missing_seed_or_world_snapshot": missing_seed_info,
            "stage_summary": stage_summary,
            "trajectory_preview": traj_rows[:20],
        }


class ExportAuditor:
    def __init__(self, project_root: Path,
                 dataset_path: Optional[Path] = None):
        self.root = project_root
        self.src = SourceInspector(project_root)
        self.dataset = DatasetInspector(dataset_path)
        self.report = AuditReport(project_root=str(project_root))

    def run(self) -> AuditReport:
        self._check_closed_loop_fix()
        self._check_quality_normalization()
        self._check_quality_threshold()
        self._check_stage0_dribble_primitives()
        self._check_snapshot_reproducibility_gap()
        self.report.dataset_summary = self.dataset.summarize()
        self._check_dataset_touch_only_signature()
        return self.report

    # ── FIX-A verification ───────────────────────────────────────────────

    def _check_closed_loop_fix(self) -> None:
        """Verifica que el replay usa closed-loop, no open-loop."""
        if not self.src.exists("evaluate_solver"):
            return

        text = self.src.get("evaluate_solver")
        has_closed_loop = "replay_closed_loop" in text
        has_open_loop = "for wp in plan.waypoints" in text

        if has_open_loop and not has_closed_loop:
            self.report.add(
                severity="critical",
                title="Replay uses open-loop waypoint following",
                file="trajectory_guidance/evaluate_solver.py",
                evidence=(
                    "Found 'for wp in plan.waypoints' pattern, which executes "
                    "pre-computed waypoints without re-planning from actual state."
                ),
                why_it_matters=(
                    "The solver's kinematic model (180mm/step) diverges from "
                    "the environment physics (25mm/step + EMA). After 3 steps, "
                    "the agent follows waypoints aimed at the goal instead of "
                    "the ball."
                ),
                recommendation=(
                    "Replace with replay_closed_loop that calls "
                    "solve_greedy_step(live_state) each step."
                ),
            )
        elif has_closed_loop:
            self.report.add(
                severity="info",
                title="Closed-loop replay confirmed",
                file="trajectory_guidance/evaluate_solver.py",
                evidence="Found replay_closed_loop method with MPC pattern.",
                why_it_matters="Fix A is applied correctly.",
                recommendation="No action needed.",
            )

    # ── FIX-B verification ───────────────────────────────────────────────

    def _check_quality_normalization(self) -> None:
        """Verifica que quality está normalizada por horizonte."""
        if not self.src.exists("solver"):
            return

        text = self.src.get("solver")
        has_normalized = "cost_per_step" in text or "/ n_steps" in text
        has_raw = "0.001 * self.planner_cost" in text and "n_steps" not in text

        if has_raw:
            self.report.add(
                severity="critical",
                title="Quality not normalized by trajectory length",
                file="trajectory_guidance/solver.py",
                evidence=(
                    "PlannedTrajectory.quality uses raw planner_cost without "
                    "dividing by number of steps."
                ),
                why_it_matters=(
                    "Stages 1-3 with longer trajectories accumulate more cost, "
                    "driving quality negative and causing 0 accepted episodes."
                ),
                recommendation="Normalize: cost_per_step = planner_cost / n_steps.",
            )
        elif has_normalized:
            self.report.add(
                severity="info",
                title="Quality normalization confirmed",
                file="trajectory_guidance/solver.py",
                evidence="Quality divides planner_cost by trajectory length.",
                why_it_matters="Fix B is applied correctly.",
                recommendation="No action needed.",
            )

    def _check_quality_threshold(self) -> None:
        """Verifica que min_quality_score es razonable."""
        if not self.src.exists("settings"):
            return
        text = self.src.get("settings")
        match = re.search(
            r"min_quality_score:\s*float\s*=\s*(-?[0-9.]+)", text)
        if match:
            threshold = float(match.group(1))
            if threshold >= 0.05:
                self.report.add(
                    severity="warning",
                    title="min_quality_score may be too high for long stages",
                    file="trajectory_guidance/settings.py",
                    evidence=f"min_quality_score = {threshold:.2f}",
                    why_it_matters=(
                        "Even with normalized quality, long-horizon stages "
                        "can produce modest quality scores. A threshold of "
                        f"{threshold:.2f} may reject valid trajectories."
                    ),
                    recommendation=(
                        "Lower to -5.0 or less during dataset bootstrapping."
                    ),
                )

    # ── Existing checks (maintained) ─────────────────────────────────────

    def _check_stage0_dribble_primitives(self) -> None:
        if not self.src.exists("settings"):
            return
        text = self.src.get("settings")
        if ("stage_id=0" in text
                and "dribble_safe" not in text.split("stage_id=0")[1].split(
                    "stage_id=1")[0] if "stage_id=1" in text else ""):
            self.report.add(
                severity="critical",
                title="Stage 0 missing dribble primitives",
                file="trajectory_guidance/settings.py",
                evidence="Stage 0 primitive_modes does not include dribble_safe.",
                why_it_matters="Without dribble primitives, the planner cannot generate dribble behavior.",
                recommendation="Add dribble_safe to Stage 0 primitive_modes.",
            )

    def _check_snapshot_reproducibility_gap(self) -> None:
        if not self.src.exists("dataset"):
            return
        text = self.src.get("dataset")
        has_seed = "reset_seed" in text
        has_world_snapshot = "world_snapshot" in text
        if not (has_seed or has_world_snapshot):
            self.report.add(
                severity="warning",
                title="Snapshots cannot reconstruct original reset",
                file="trajectory_guidance/dataset.py",
                evidence="No reset_seed or world_snapshot in DemoSnapshot.",
                why_it_matters="Cannot replay exactly for debugging.",
                recommendation="Store reset_seed and initial poses.",
            )

    def _check_dataset_touch_only_signature(self) -> None:
        ds = self.report.dataset_summary
        if not ds or ds.get("num_trajectories", 0) == 0:
            return
        touch_only = int(ds.get("touch_only_trajectories", 0))
        num_traj = int(ds.get("num_trajectories", 0))
        if num_traj > 0 and touch_only > 0:
            self.report.add(
                severity="warning",
                title="Dataset contains touch-only trajectories",
                file=str(ds.get("dataset_path")),
                evidence=(
                    f"{touch_only}/{num_traj} trajectories touch ball but "
                    f"show zero dribble steps and zero goals."
                ),
                why_it_matters=(
                    "Consistent with open-loop divergence or validation "
                    "accepting on first touch only."
                ),
                recommendation="Re-export with closed-loop replay.",
            )


def render_text(report: AuditReport) -> str:
    lines = []
    lines.append("TRAJECTORY EXPORT AUDIT REPORT")
    lines.append(f"Project root: {report.project_root}")
    lines.append("")
    lines.append(f"Findings: {len(report.findings)}")
    lines.append("")
    for idx, finding in enumerate(report.findings, start=1):
        lines.append(
            f"[{idx}] {finding.severity.upper()} - {finding.title}")
        lines.append(f"    File: {finding.file}")
        lines.append(f"    Evidence: {finding.evidence}")
        lines.append(f"    Why it matters: {finding.why_it_matters}")
        lines.append(f"    Recommendation: {finding.recommendation}")
        lines.append("")

    if report.dataset_summary:
        lines.append("DATASET SUMMARY")
        for key, value in report.dataset_summary.items():
            if key == "trajectory_preview":
                lines.append("  trajectory_preview:")
                for row in value:
                    lines.append(f"    - {row}")
            else:
                lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parent
    dataset_path: Optional[str] = "trajectory_demos/demos_fixed0.pkl"
    out_json: Optional[str] = None
    out_txt: Optional[str] = "trajectory_demos/audit_report.txt"

    auditor = ExportAuditor(
        project_root, Path(dataset_path) if dataset_path else None)
    report = auditor.run()

    text = render_text(report)
    print(text)

    if out_json:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report.to_dict(), indent=2,
                                ensure_ascii=False),
                     encoding="utf-8")

    if out_txt:
        p = Path(out_txt)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()