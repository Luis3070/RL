# CODEX TASK — Merge striker(v4.2) + defender into ONE 47-dim ONNX brain

## GOAL
Train a SINGLE 47-dim checkpoint (→ one ONNX) that handles ALL three roles
(striker / open_support / off_ball_support-defender) driven by the role one-hot
(`obs[44:46]`), so deployment needs no per-role dispatch. It must (a) PRESERVE v4.2's
attack quality (pass ~0.99, eval floor ≥4/5) and (b) ADD defense (block/mark/positioning)
without regressing attack.

## HARD PREREQUISITE — DO NOT START WITHOUT THIS
Off-ball opponent perception must be FIXED first (see `CODEX_OFFBALL_PERCEPTION_HANDOFF.md`).
Verified this session: in the multi-agent scene the off-ball defender perceives opponents
~0% (gaze faces the ball). If you retrain defense in that state, the policy gets ZERO
opponent/mark signal and will NOT learn defense — you'll burn days and regress the pass for
nothing. Gate: a trío eval must show `opponent_seen_rate > 0.30` BEFORE any merge training.

## APPROACH (single combined fine-tune, warm-started)
- **Warm-start + KL-anchor** from the v4.2 checkpoint (`runs/multiagent_v4_2/...`, the
  validated striker/pass brain) — NOT from scratch. The KL anchor protects the pass.
- **Combined BC corpus** = the v4.2 team/attack dataset (the 0.99-pass backbone) UNION a
  fresh DEFENSE dataset (block/mark/positioning + recover→clear/pass-out). Mix so attack
  stays dominant (defense is the minority skill being added), e.g. attack:defense ≈ 70:30 —
  TUNE so pass does not drop.
- **PPO scenario pool** = team attack scenarios (team_3v0*, team_3v3_full_dynamic,
  preshoot_bad_angle_boundary) UNION the defense scenarios (defense_3v3_full,
  defense_2v3_dynamic, defense_recovery_clear, transition_midfield). Weight defense in
  gradually.
- Reuse the existing trainer `train_multiagent_production.py` (stage 7 corpus + stage-8
  defense scenarios). The corrected Design-A gate in `train.py::gate_skill_mask_from_obs`
  already arms block/mark for off_ball and pass/shoot for striker — keep it as the SINGLE
  mask source for BC labels, env step, and PPO recompute (do not fork it).

## HARD INVARIANTS (instant fail if broken)
- `OBS_SIZE = 47`, `ObsIndex`, `SKILL_ORDER` unchanged.
- `gate_skill_mask_from_obs` is the single mask source; do not compute masks elsewhere.
- Episode steps stay **800**. Shoot band stays **2.5–3.0 m**.
  `RL_GATE_SHOOT_OPEN_ENTER=0.30 / EXIT=0.20` everywhere.
- Do NOT overwrite the deployed `ppo_team_hsl2026.onnx` (v2b) or the v4.1/v4.2 ONNX —
  export the merged brain under a NEW name (e.g. `ppo_team_hsl2026_v5_merged.onnx`).
- Do not commit/push without approval.

## DATASET / GATE DISCIPLINE (this repo's recurring failure mode)
- Teacher labels MUST pass through the REAL `train.gate_skill_mask_from_obs` (the
  `teacher_multiagent_dataset._team_label_allowed` path already does this) — a stale
  hand-mirrored mask silently drops labels ("gate mark-leak" class of bug).
- Verify with `tools/verify_defender_gate_contract.py` (must print ALL-PASS) before training.
- Confirm the defense dataset on disk actually contains block/mark rows with nonzero
  `target_y` (and that opponents were PERCEIVED during collection — see prerequisite).

## EVAL GATES (all must hold to accept the merged brain)
1. Pass NOT regressed: striker `executed_pass_rate`/pass-completion within ~5% of v4.2.
2. Attack floor ≥ 4/5 on the behavioral/team eval (no worse than v4.2).
3. Defense WORKS: in a trío with fixed off-ball perception, `executed_block_rate > 0` AND
   `executed_mark_rate > 0`, `goals_conceded` ≤ family/baseline.
4. No degenerate collapse (shoot_rate not 0; no stand/idle lock; collision not inflated).

## RECOMMENDED SEQUENCING
1. Land the off-ball gaze fix; prove `opponent_seen_rate>0.30`.
2. Regenerate a clean defense dataset WITH perceived opponents; verify gate ALL-PASS.
3. Short merge fine-tune (warm-start v4.2 + KL anchor, attack-dominant BC); eval gates above.
4. Only if gates pass: export `ppo_team_hsl2026_v5_merged.onnx` (NEW name).

## FALLBACK
If the merge regresses the pass and can't be recovered, STOP and fall back to the
per-role FAMILY (v4.2 for striker/support + pt_defender_47 for defender, dispatched by
role one-hot) — that path needs no retrain and preserves v4.2 exactly. The family is the
safe baseline the merge must BEAT before replacing it.
