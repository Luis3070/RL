"""
audit_exporter.py - Static and dataset audit for trajectory_guidance exports.

The audit focuses on BHuman-in-loop Option 1 invariants for BC datasets:
  - closed-loop replay
  - stage-aware intents
  - real backend provenance
  - persistent_2d_cognition bridge mode
  - no contract-validation fallback in accepted trajectories
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


ALLOWED_INTENTS_BY_STAGE: Dict[int, Set[str]] = {
    0: {"control"},
    1: {"dribble"},
    2: {"dribble", "pass_setup", "shoot_setup"},
    3: {"dribble", "pass_setup", "shoot_setup"},
}


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
        by_intent: Dict[str, int] = {}
        invalid_intent_traj: List[Dict[str, Any]] = []
        nonreal_backend_trajectories = 0
        wrong_bridge_mode_trajectories = 0
        contract_validation_trajectories = 0
        touch_only_trajectories = 0
        missing_seed_info = 0
        handoff_successful_trajectories = 0
        takeover_successful_trajectories = 0

        for sample in self.samples:
            tid = str(sample.get("trajectory_id", "<missing>"))
            by_traj.setdefault(tid, []).append(sample)
            by_stage.setdefault(int(sample.get("stage_id", -1)), []).append(sample)
            intent_label = str(sample.get("intent", "<missing>"))
            by_intent[intent_label] = by_intent.get(intent_label, 0) + 1

        trajectory_preview = []
        stage_summary: Dict[str, Dict[str, Any]] = {}
        stage_intent_summary: Dict[str, Dict[str, int]] = {}

        for tid, rows in by_traj.items():
            rows_sorted = sorted(rows, key=lambda r: int(r.get("t", 0)))
            last = rows_sorted[-1]
            stage_id = int(last.get("stage_id", -1))
            intent_label = str(last.get("intent", "<missing>"))
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
            handoff_achieved = int(last.get("episode_handoff_achieved", 0)) > 0
            takeover_success = int(last.get("episode_takeover_success", 0)) > 0
            backend = str(last.get("sidecar_backend", "<missing>"))
            bridge_mode = str(last.get("sidecar_bridge_mode", "<missing>"))
            contract_mode = str(last.get("sidecar_contract_validation_mode", "<missing>"))
            allowed_intents = ALLOWED_INTENTS_BY_STAGE.get(stage_id, set())
            touch_only = touched_any and not goal_any and dribble_steps == 0

            if touch_only:
                touch_only_trajectories += 1
            if "reset_seed" not in last and "world_snapshot" not in last:
                missing_seed_info += 1
            if backend != "real":
                nonreal_backend_trajectories += 1
            if bridge_mode != "persistent_2d_cognition":
                wrong_bridge_mode_trajectories += 1
            if contract_mode not in {"", "none", "disabled", "False", "false"}:
                contract_validation_trajectories += 1
            if handoff_achieved:
                handoff_successful_trajectories += 1
            if takeover_success:
                takeover_successful_trajectories += 1
            if allowed_intents and intent_label not in allowed_intents:
                invalid_intent_traj.append({
                    "trajectory_id": tid,
                    "stage_id": stage_id,
                    "intent": intent_label,
                })

            stage_summary.setdefault(str(stage_id), {
                "num_samples": 0,
                "num_trajectories": 0,
                "handoff_achieved_trajectories": 0,
                "takeover_success_trajectories": 0,
            })
            stage_summary[str(stage_id)]["num_trajectories"] += 1
            stage_summary[str(stage_id)]["handoff_achieved_trajectories"] += int(handoff_achieved)
            stage_summary[str(stage_id)]["takeover_success_trajectories"] += int(takeover_success)
            stage_intent_summary.setdefault(str(stage_id), {})
            stage_intent_summary[str(stage_id)][intent_label] = (
                stage_intent_summary[str(stage_id)].get(intent_label, 0) + 1
            )

            trajectory_preview.append({
                "trajectory_id": tid,
                "stage_id": stage_id,
                "intent": intent_label,
                "num_steps": len(rows_sorted),
                "done_last": bool(last.get("done", False)),
                "touched_any": touched_any,
                "dribble_steps": dribble_steps,
                "goal_any": goal_any,
                "quality_score": float(last.get("quality_score", 0.0)),
                "handoff_achieved": handoff_achieved,
                "takeover_success": takeover_success,
                "backend": backend,
                "bridge_mode": bridge_mode,
                "contract_validation_mode": contract_mode,
                "touch_only_signature": touch_only,
            })

        for stage_val, rows in by_stage.items():
            stage_summary.setdefault(str(stage_val), {
                "num_samples": 0,
                "num_trajectories": 0,
                "handoff_achieved_trajectories": 0,
                "takeover_success_trajectories": 0,
            })
            stage_summary[str(stage_val)]["num_samples"] = len(rows)

        return {
            "dataset_path": str(self.path),
            "exists": True,
            "num_samples": len(self.samples),
            "num_trajectories": len(by_traj),
            "intent_counts": by_intent,
            "invalid_intent_trajectories": invalid_intent_traj[:50],
            "nonreal_backend_trajectories": nonreal_backend_trajectories,
            "wrong_bridge_mode_trajectories": wrong_bridge_mode_trajectories,
            "contract_validation_trajectories": contract_validation_trajectories,
            "touch_only_trajectories": touch_only_trajectories,
            "trajectories_missing_seed_or_world_snapshot": missing_seed_info,
            "handoff_achieved_trajectories": handoff_successful_trajectories,
            "takeover_success_trajectories": takeover_successful_trajectories,
            "stage_summary": stage_summary,
            "stage_intent_summary": stage_intent_summary,
            "trajectory_preview": trajectory_preview[:20],
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
        self._check_stage_aware_generator()
        self.report.dataset_summary = self.dataset.summarize()
        self._check_dataset_touch_only_signature()
        self._check_dataset_stage_intents()
        self._check_dataset_backend_provenance()
        self._check_takeover_presence()
        return self.report

    def _check_closed_loop_fix(self) -> None:
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
                    "The solver kinematics can diverge from the environment physics, "
                    "which breaks BC data quality."
                ),
                recommendation=(
                    "Use replay_closed_loop and solve_greedy_step on live state."
                ),
            )
        elif has_closed_loop:
            self.report.add(
                severity="info",
                title="Closed-loop replay confirmed",
                file="trajectory_guidance/evaluate_solver.py",
                evidence="Found replay_closed_loop method with MPC pattern.",
                why_it_matters="Closed-loop replay is required for honest trajectory export.",
                recommendation="No action needed.",
            )

    def _check_quality_normalization(self) -> None:
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
                evidence="Raw planner_cost appears in quality without horizon normalization.",
                why_it_matters="Longer stages would be rejected for numeric reasons instead of behavior.",
                recommendation="Normalize planner cost by number of steps.",
            )
        elif has_normalized:
            self.report.add(
                severity="info",
                title="Quality normalization confirmed",
                file="trajectory_guidance/solver.py",
                evidence="Quality divides planner_cost by trajectory length.",
                why_it_matters="Long-horizon stages can now be filtered behaviorally, not by length bias.",
                recommendation="No action needed.",
            )

    def _check_quality_threshold(self) -> None:
        if not self.src.exists("settings"):
            return
        text = self.src.get("settings")
        match = re.search(r"min_quality_score:\s*float\s*=\s*(-?[0-9.]+)", text)
        if not match:
            return

        threshold = float(match.group(1))
        if threshold >= 0.05:
            self.report.add(
                severity="warning",
                title="min_quality_score may be too high for long stages",
                file="trajectory_guidance/settings.py",
                evidence=f"min_quality_score = {threshold:.2f}",
                why_it_matters="A high threshold may reject valid long-horizon trajectories.",
                recommendation="Keep the threshold relaxed and rely on behavioral validity checks.",
            )

    def _check_stage0_dribble_primitives(self) -> None:
        if not self.src.exists("settings"):
            return
        text = self.src.get("settings")
        stage0_slice = text.split("0: StageObjective(", 1)
        if len(stage0_slice) < 2:
            return
        stage0_block = stage0_slice[1].split("1: StageObjective(", 1)[0]
        if "dribble_safe" not in stage0_block:
            self.report.add(
                severity="critical",
                title="Stage 0 missing short-control primitive",
                file="trajectory_guidance/settings.py",
                evidence="Stage 0 objective block does not include dribble_safe.",
                why_it_matters="Stage 0 needs short control to support a stable handoff before takeover.",
                recommendation="Keep dribble_safe/recover_contact available in stage 0.",
            )

    def _check_snapshot_reproducibility_gap(self) -> None:
        if not self.src.exists("dataset"):
            return
        text = self.src.get("dataset")
        has_seed = "reset_seed" in text
        has_world_snapshot = "world_snapshot" in text
        if not (has_seed and has_world_snapshot):
            self.report.add(
                severity="warning",
                title="Snapshots cannot fully reconstruct original reset",
                file="trajectory_guidance/dataset.py",
                evidence="DemoSnapshot is missing reset_seed and/or world_snapshot.",
                why_it_matters="Trajectory debugging becomes weaker when a bad export cannot be replayed faithfully.",
                recommendation="Store both reset_seed and world_snapshot in exported samples.",
            )

    def _check_stage_aware_generator(self) -> None:
        if not self.src.exists("generate_dataset"):
            return
        text = self.src.get("generate_dataset")
        missing = []
        for token in ["control", "pass_setup", "shoot_setup",
                      "_resolve_family_counts", "default_n_jobs"]:
            if token not in text:
                missing.append(token)
        if missing:
            self.report.add(
                severity="warning",
                title="Dataset generator may not be stage-aware",
                file="trajectory_guidance/generate_dataset.py",
                evidence=f"Missing generator markers: {', '.join(missing)}",
                why_it_matters="The export pipeline could create semantically false labels for early stages.",
                recommendation="Use a stage-aware generator with honest intents and serial-by-default real backend execution.",
            )
        else:
            self.report.add(
                severity="info",
                title="Stage-aware generator markers detected",
                file="trajectory_guidance/generate_dataset.py",
                evidence="Found stage-aware intents, family resolver, and default_n_jobs usage.",
                why_it_matters="The generator is structurally aligned with Option 1 dataset boundaries.",
                recommendation="No action needed.",
            )

    def _check_dataset_touch_only_signature(self) -> None:
        ds = self.report.dataset_summary
        if not ds or ds.get("num_trajectories", 0) == 0:
            return
        touch_only = int(ds.get("touch_only_trajectories", 0))
        num_traj = int(ds.get("num_trajectories", 0))
        if touch_only > 0:
            self.report.add(
                severity="warning",
                title="Dataset contains touch-only trajectories",
                file=str(ds.get("dataset_path")),
                evidence=f"{touch_only}/{num_traj} trajectories touch the ball but show zero dribble steps and zero goals.",
                why_it_matters="This pattern is inconsistent with healthy control/handoff demonstrations.",
                recommendation="Re-export after checking closed-loop replay and validity gates.",
            )

    def _check_dataset_stage_intents(self) -> None:
        ds = self.report.dataset_summary
        if not ds or ds.get("num_trajectories", 0) == 0:
            return
        invalid = ds.get("invalid_intent_trajectories", []) or []
        if invalid:
            self.report.add(
                severity="critical",
                title="Dataset violates stage intent policy",
                file=str(ds.get("dataset_path")),
                evidence=f"Found {len(invalid)} trajectory summaries with intents outside the allowed stage policy.",
                why_it_matters="Stages 0-1 would contain semantically false labels for BC.",
                recommendation="Re-export with stage 0=control, stage 1=dribble, stage 2-3=dribble/pass_setup/shoot_setup.",
            )

    def _check_dataset_backend_provenance(self) -> None:
        ds = self.report.dataset_summary
        if not ds or ds.get("num_trajectories", 0) == 0:
            return
        nonreal = int(ds.get("nonreal_backend_trajectories", 0))
        wrong_bridge = int(ds.get("wrong_bridge_mode_trajectories", 0))
        contract_validation = int(ds.get("contract_validation_trajectories", 0))
        if nonreal > 0 or wrong_bridge > 0 or contract_validation > 0:
            self.report.add(
                severity="critical",
                title="Dataset provenance breaks Option 1 baseline",
                file=str(ds.get("dataset_path")),
                evidence=(
                    f"nonreal_backend={nonreal}, wrong_bridge_mode={wrong_bridge}, "
                    f"contract_validation={contract_validation}"
                ),
                why_it_matters="BC would be trained on trajectories that do not come from the required real baseline.",
                recommendation="Reject and regenerate trajectories until every accepted sample comes from real backend and persistent_2d_cognition without contract fallback.",
            )

    def _check_takeover_presence(self) -> None:
        ds = self.report.dataset_summary
        if not ds or ds.get("num_trajectories", 0) == 0:
            return
        handoff = int(ds.get("handoff_achieved_trajectories", 0))
        takeover = int(ds.get("takeover_success_trajectories", 0))
        num_traj = int(ds.get("num_trajectories", 0))
        if handoff == 0 or takeover == 0:
            self.report.add(
                severity="warning",
                title="Dataset lacks takeover evidence",
                file=str(ds.get("dataset_path")),
                evidence=f"handoff_achieved={handoff}/{num_traj}, takeover_success={takeover}/{num_traj}",
                why_it_matters="The dataset may be planner-centric instead of BHuman-handoff-centric.",
                recommendation="Review validity gates and require episode_handoff_achieved and episode_takeover_success in accepted trajectories.",
            )


def render_text(report: AuditReport) -> str:
    lines = []
    lines.append("TRAJECTORY EXPORT AUDIT REPORT")
    lines.append(f"Project root: {report.project_root}")
    lines.append("")
    lines.append(f"Findings: {len(report.findings)}")
    lines.append("")
    for idx, finding in enumerate(report.findings, start=1):
        lines.append(f"[{idx}] {finding.severity.upper()} - {finding.title}")
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
    # Configuracion editable localmente en vez de CLI.
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
        p.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                     encoding="utf-8")

    if out_txt:
        p = Path(out_txt)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
