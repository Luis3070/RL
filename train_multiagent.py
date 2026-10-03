"""Multi-agent MAPPO training entrypoint — sibling of `train.py`.

Mirrors train.py's surface (ScriptRunConfig, run_smoke, run_training, main) but
swaps in two role-aware BC components without touching train.py:

  * `MultiAgentBCReplayRegularizer`  — role-stratified sampling on (stage, role, skill)
                                       per §11.A.5 of docs/notes/10-graph-coordination-post-production-roadmap.md. Adds the role_id column
                                       missing from the legacy `(stage, skill)` keying so
                                       striker-shoot rows are not diluted by support-walk
                                       rows.

  * `MultiAgentBCCompetenceGate`     — competence = min(striker, open_support, off_ball_support)
                                       per §11.B.5.2 of docs/notes/10-graph-coordination-post-production-roadmap.md. The legacy gate decays β
                                       on striker progress alone; with three roles, that
                                       would shut BC off while supports were still
                                       incompetent. Min ensures BC stays alive until every
                                       role clears the bar.

train.py stays untouched (`runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt`
single-agent fine-tuning still uses train_production.py). The two overrides are installed
via temporary monkey-patches on the train module for the duration of multiagent_main(),
then restored — so calling either train.main() or train_multiagent.main() in the same
Python process behaves correctly each time.

Run via train_multiagent_production.py (sibling of train_production.py) which sets the
§14.19 retunings (bc_beta_max=0.08, bc_competence_half_life_evals=25, etc.) and points
at the multi-agent dataset / KL anchor checkpoint.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

import train
from action import SKILL_ORDER
from observation import OBS_SIZE
from train import (
    BCCompetenceGate,
    BCReplayRegularizer,
    IDE_RUN,
    PolicyNetwork,
    ScriptRunConfig,
    TelemetryPaths,
    _bc_param_mask,
    _latest_teacher_bc_dataset,
    _stage_skill_logits,
    bc_ce_safe_logits,
)

# Role enum used in the multi-agent dataset's `role_ids` column.
_ROLE_NAMES = ("striker", "open_support", "off_ball_support")
_ROLE_TO_ID = {name: i for i, name in enumerate(_ROLE_NAMES)}


# ---------------------------------------------------------------------------
# Role-stratified BC replay regularizer (§11.A.5, §11.B.5.1)
# ---------------------------------------------------------------------------


class MultiAgentBCReplayRegularizer(BCReplayRegularizer):
    """BC replay with role-aware sampling.

    Indexing key changes from (stage, skill) → (stage, role, skill). Sampling mix
    explicitly preserves striker-shoot rows even when support-walk rows dominate
    the natural class distribution.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        reference_policy: PolicyNetwork,
        *,
        dataset_path: str | Path | None,
        device: str,
        learning_rate: float,
        batch_size: int,
        beta_max: float,
        beta_min: float,
        skill_loss_scale: float,
        param_loss_scale: float,
        updates_per_rollout: int,
        seed: int,
    ):
        super().__init__(
            policy=policy,
            reference_policy=reference_policy,
            dataset_path=dataset_path,
            device=device,
            learning_rate=learning_rate,
            batch_size=batch_size,
            beta_max=beta_max,
            beta_min=beta_min,
            skill_loss_scale=skill_loss_scale,
            param_loss_scale=param_loss_scale,
            updates_per_rollout=updates_per_rollout,
            seed=seed,
        )
        # Re-index by (stage, role, skill) using the role_ids column produced by
        # teacher_multiagent_dataset.py. Falls back to the legacy single-agent
        # (stage, skill) indexing if the dataset lacks role_ids (single-agent
        # back-compat).
        data = np.load(self.path, allow_pickle=True)
        if "role_ids" not in data:
            self.indices_by_stage_role_skill: dict[tuple[int, int, int], np.ndarray] = {}
            self._has_role_axis = False
            return
        role_ids = data["role_ids"].astype(np.int8)
        weights_np = data["weights"].astype(np.float32)
        actions_np = data["actions"].astype(np.float32)
        stages_np = data["stages"].astype(np.int64)
        valid = np.flatnonzero(weights_np > 0.0)
        skill_ids = np.clip(np.rint(actions_np[:, 0]).astype(np.int64), 0, len(SKILL_ORDER) - 1)
        self.indices_by_stage_role_skill: dict[tuple[int, int, int], np.ndarray] = {}
        for stage_id in sorted(set(stages_np[valid].tolist())):
            for role_id in sorted(set(role_ids[valid].tolist())):
                for skill_id in sorted(set(skill_ids[valid].tolist())):
                    mask = (
                        (stages_np[valid] == stage_id)
                        & (role_ids[valid] == role_id)
                        & (skill_ids[valid] == skill_id)
                    )
                    idx = valid[mask]
                    if idx.size:
                        self.indices_by_stage_role_skill[(int(stage_id), int(role_id), int(skill_id))] = idx
        self._has_role_axis = True
        self._available_role_stages = sorted({(stage_id, role_id) for stage_id, role_id, _ in self.indices_by_stage_role_skill})
        # v4-S2: decision-window filter. A "decision window" row is one where the gate has
        # armed a real choice (shoot/dribble/pass). When RL_MA_BC_DECISION_WINDOW_ONLY=1,
        # walk/stand candidates are restricted to decision-window rows — so BC trains on the
        # moments that matter instead of the ~80% idle walk. Armed skills are unaffected.
        self._decision_window_only = os.getenv("RL_MA_BC_DECISION_WINDOW_ONLY", "0") not in ("0", "false", "False", "")
        obs_np = data["observations"]
        try:
            from observation import ObsIndex as _OI
            armed = (
                (obs_np[:, int(_OI.shoot_armed)] >= 0.5)
                | (obs_np[:, int(_OI.dribble_armed)] >= 0.5)
                | (obs_np[:, int(_OI.pass_armed)] >= 0.5)
            )
        except Exception:
            armed = np.ones(len(obs_np), dtype=bool)
        self._decision_window_set = set(np.flatnonzero(armed).tolist())
        self._WALK_ID = SKILL_ORDER.index("walk")
        self._STAND_ID = SKILL_ORDER.index("stand")

    @staticmethod
    def _role_skill_mix(stage_id: int) -> tuple[tuple[str, str, float], ...]:
        """Per-role-per-skill batch mix for stages 6-7 (§14.19 / §11.A.5).

        Striker shoot/dribble/walk preserved as the dominant signal so single-agent
        finishing capability does not erode; support-walk and -stand sampled at
        proportions that drive role emergence without dilution.
        """
        if stage_id <= 5:
            # Single-agent stages: legacy mix via parent _skill_mix.
            return ()
        # Stages 6-7 use the same role/skill mix; small per-stage tweaks possible
        # but unnecessary at the sprint horizon.
        # Pass bumped 0.04→0.10 (striker walk 0.12→0.06): the 2026-06-12 ablation
        # showed pass competence is context-correct (P(pass)=0.71 argmax 96.8% on
        # teacher-pass contexts) but eval rollouts under-visit those contexts —
        # heavier BC sampling keeps the skill sharp while team rewards create the
        # contexts.
        # P7 (conversion): striker shoot up, dribble down vs the original 0.20/0.18 —
        # the closed-loop striker over-dribbles and under-shoots. v3a cut dribble too
        # hard (0.15): conversion stabilised at 0.25 but the dynamic_full_field floor
        # scenario regressed (drive too weak to finish under pressure, 4/5→3/5). v3b
        # softens the cut to 0.22/0.17 — keep the conversion gain, restore the drive.
        # P4 (support hold): supports collapsed to walk (.86 vs teacher .66 stand),
        # so flip the walk/stand mix — open walk 0.16→0.10 / stand 0.08→0.13,
        # off_ball walk 0.14→0.08 / stand 0.06→0.12 — so BC pulls supports to hold.
        # (Sums to 1.00; off_ball dribble 0.02 keeps the unexpected-possession hatch.)
        # v4-S2: striker pass 0.10→0.30. v3/v3.1 collapsed pass→0 because pass is only ~3%
        # of the corpus and the 0.10 mix was too weak to keep it alive through PPO. Heavy
        # oversampling (with replacement) of the rare pass rows is the point. Finishing
        # (shoot/dribble) trimmed accordingly; the completed-pass + finishing PPO rewards
        # rebalance at training time. Env-tunable via RL_MA_BC_STRIKER_PASS_FRAC.
        pass_frac = float(os.getenv("RL_MA_BC_STRIKER_PASS_FRAC", "0.30"))
        return (
            ("striker", "shoot", 0.18),
            ("striker", "dribble", 0.12),
            ("striker", "walk", 0.04),
            ("striker", "pass", pass_frac),
            ("open_support", "walk", 0.08),
            ("open_support", "stand", 0.12),
            ("off_ball_support", "walk", 0.06),
            ("off_ball_support", "stand", 0.08),
            ("off_ball_support", "dribble", 0.02),
        )

    def _sample_indices(self, stage_id: int) -> np.ndarray:
        # Single-agent stages: defer to parent (preserves the validated mix).
        if stage_id <= 5 or not self._has_role_axis:
            return super()._sample_indices(stage_id)

        mix = self._role_skill_mix(stage_id)
        if not mix:
            return super()._sample_indices(stage_id)

        # If the requested stage is missing from the dataset, fall back to the
        # nearest available multi-agent stage.
        ma_stages = sorted({s for s, _ in self._available_role_stages if s >= 6})
        if not ma_stages:
            return super()._sample_indices(stage_id)
        if stage_id not in ma_stages:
            earlier = [s for s in ma_stages if s <= stage_id]
            stage_id = earlier[-1] if earlier else ma_stages[0]

        chunks: list[np.ndarray] = []
        remaining = self.batch_size
        for role_name, skill_name, fraction in mix:
            role_id = _ROLE_TO_ID[role_name]
            try:
                skill_id = SKILL_ORDER.index(skill_name)
            except ValueError:
                continue
            candidates = self.indices_by_stage_role_skill.get((stage_id, role_id, skill_id))
            if candidates is None or candidates.size == 0:
                continue
            # v4-S2: in decision-window mode, restrict idle walk/stand to rows where a
            # gate decision was actually live (drop the ~80% idle filler). Armed skills
            # (shoot/dribble/pass) are decision windows by definition → left as-is.
            if self._decision_window_only and skill_id in (self._WALK_ID, self._STAND_ID):
                filtered = np.array([i for i in candidates.tolist() if i in self._decision_window_set], dtype=candidates.dtype)
                if filtered.size:
                    candidates = filtered
            count = max(1, int(round(self.batch_size * fraction)))
            count = min(count, remaining) if remaining > 0 else count
            probs = self.weights[candidates].detach().cpu().numpy().astype(np.float64)
            probs = probs / max(float(probs.sum()), 1e-12)
            chunks.append(self.rng.choice(candidates, size=count, replace=True, p=probs))
            remaining -= count

        if not chunks:
            return super()._sample_indices(stage_id)

        batch = np.concatenate(chunks)
        if batch.size < self.batch_size:
            extra = self.rng.choice(batch, size=self.batch_size - batch.size, replace=True)
            batch = np.concatenate([batch, extra])
        self.rng.shuffle(batch)
        return batch[: self.batch_size]


# ---------------------------------------------------------------------------
# Role-aware BC competence gate (§11.B.5.2)
# ---------------------------------------------------------------------------


class MultiAgentBCCompetenceGate(BCCompetenceGate):
    """Competence = min across roles.

    Per-role competence keys (any one missing → treated as 0.0 so β stays high):
      - striker:          summary["clean_finish_rate"] or "dribble_compat_rate"
      - open_support:     summary["support_open_hold_rate"]
      - off_ball_support: summary["support_triangle_hold_rate"]

    The min is a conservative aggregator: BC β stays high until *every* role
    clears the bar; striker progress alone cannot decay it.
    """

    def update(self, summary: Mapping[str, object]) -> dict[str, float]:
        if int(float(summary.get("stage_id", 0.0))) == 8:
            defender_competence = float(summary.get("defender_competence_rate", summary.get("defender_score", 0.0)))
            defender_competence = float(np.clip(defender_competence, 0.0, 1.0))
            if not self.initialized:
                self.competence_ema = defender_competence
                self.initialized = True
            else:
                self.competence_ema = (1.0 - self.alpha) * self.competence_ema + self.alpha * defender_competence
            return {
                "bc_competence_striker": float(summary.get("takeover_success_rate", 0.0)),
                "bc_competence_open_support": float(summary.get("defender_cover_corridor_rate", 0.0)),
                "bc_competence_off_ball_support": defender_competence,
                "bc_competence": defender_competence,
                "bc_competence_ema": float(self.competence_ema),
                "bc_beta": self.beta(),
            }

        striker_competence = float(summary.get("clean_finish_rate", 0.0))
        if striker_competence <= 0.0:
            striker_competence = float(summary.get("dribble_compat_rate", 0.0))
        striker_competence = float(np.clip(striker_competence, 0.0, 1.0))

        # Support hold rates don't flow through train.py's fixed eval metric list;
        # the team env publishes them via a side-channel at episode end. Without
        # this, the min-across-roles is permanently 0 and β never decays (the
        # 2026-06-11 run trained with β pinned at beta_max the whole way).
        def _team_metric(key: str) -> float:
            value = summary.get(key)
            if value is None:
                from team_parallel_environment import LAST_TEAM_METRICS
                value = LAST_TEAM_METRICS.get(key, 0.0)
            return float(value)

        open_competence = float(np.clip(_team_metric("support_open_hold_rate"), 0.0, 1.0))
        off_ball_competence = float(np.clip(_team_metric("support_triangle_hold_rate"), 0.0, 1.0))

        # Min-across-roles: worst-role drives β decay.
        competence = float(min(striker_competence, open_competence, off_ball_competence))

        if not self.initialized:
            self.competence_ema = competence
            self.initialized = True
        else:
            self.competence_ema = (1.0 - self.alpha) * self.competence_ema + self.alpha * competence
        return {
            "bc_competence": competence,
            "bc_competence_striker": striker_competence,
            "bc_competence_open_support": open_competence,
            "bc_competence_off_ball_support": off_ball_competence,
            "bc_competence_ema": float(self.competence_ema),
            "bc_beta": self.beta(),
        }


# ---------------------------------------------------------------------------
# Public entry points — same shape as train.run_smoke / train.run_training / train.main
# ---------------------------------------------------------------------------


def _install_multiagent_components() -> tuple:
    """Monkey-patch the train module to use the role-aware BC components and the
    team-coordinated multi-slot env (§14.17 env-side team layer).

    Returns the originals so callers can restore on exit. We never modify
    train.py source.
    """
    from team_parallel_environment import TeamSlotSimRobotEnv

    # Tell the wire audit this is a multi-agent team run, so it relaxes the two
    # ball-visibility failure thresholds (support roles don't face the ball — the
    # single-agent 0.90/0.99 thresholds false-fail every team eval). setdefault so
    # an explicit override is respected.
    os.environ.setdefault("RL_WIRE_AUDIT_TEAM_MODE", "1")

    orig_replay = train.BCReplayRegularizer
    orig_gate = train.BCCompetenceGate
    orig_parallel_env = train.ParallelSlotSimRobotEnv
    train.BCReplayRegularizer = MultiAgentBCReplayRegularizer
    train.BCCompetenceGate = MultiAgentBCCompetenceGate
    train.ParallelSlotSimRobotEnv = TeamSlotSimRobotEnv
    return orig_replay, orig_gate, orig_parallel_env


def _restore_components(orig_replay, orig_gate, orig_parallel_env) -> None:
    train.BCReplayRegularizer = orig_replay
    train.BCCompetenceGate = orig_gate
    train.ParallelSlotSimRobotEnv = orig_parallel_env


def run_smoke(run_cfg: ScriptRunConfig) -> None:
    """Multi-agent smoke run. Delegates to train.run_smoke with role-aware BC swapped in."""
    orig = _install_multiagent_components()
    try:
        train.run_smoke(run_cfg)
    finally:
        _restore_components(*orig)


def run_training(run_cfg: ScriptRunConfig) -> None:
    """Multi-agent PPO training. Delegates to train.run_training with role-aware BC swapped in.

    The mask path (`train.gate_skill_mask_from_obs`) already auto-switches between
    the single-agent and role-conditioned forbid contracts based on whether the
    role one-hot is set in the 47-dim obs — no override needed there.
    """
    orig = _install_multiagent_components()
    try:
        train.run_training(run_cfg)
    finally:
        _restore_components(*orig)


def main(run_cfg: ScriptRunConfig = IDE_RUN) -> None:
    """Multi-agent training entrypoint, mirroring train.main."""
    orig = _install_multiagent_components()
    try:
        train.main(run_cfg)
    finally:
        _restore_components(*orig)


if __name__ == "__main__":
    main(train._run_cfg_from_env())
