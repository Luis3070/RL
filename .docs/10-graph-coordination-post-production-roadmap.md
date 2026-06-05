# Graph Coordination — 1-Week Post-Production Sprint

**Status:** plan for the 1 week we have after the production single-agent checkpoint ships.
**Hard constraint:** 7 calendar days. Not 2 weeks, not 4 weeks. One week.
**What we ship:** an inference-only, hard-mask version of the graph idea. NOT a retrained, learned-coordination version.
**Audience:** new contributor to the RL pipeline. This document assumes you know Python and basic RL vocabulary but not this codebase.

---

## 0. Read this section first or you will misunderstand the whole document

There are two versions of the graph coordination idea:

- **Version A (the full one).** OBS_SIZE grows to carry graph state, teacher gets per-role branches, dataset is re-collected, BC and curriculum get new reward terms, policy is retrained for 60k+ steps to learn graph-conditioned behavior. **Realistic effort: 8-10 weeks.**
- **Version B (the 1-week one).** OBS_SIZE stays at 26. No retraining. The adjacency matrix is computed in the Python wrapper at inference time and used as a **hard skill mask + residual override** on top of the existing single-agent policy. The graph is a *coordination structure imposed from outside*, not a feature the policy has learned. **Realistic effort: 7 days.**

**This document is the plan for Version B.** Version A is documented as a v2 appendix at §11 so the choice is recorded, not as a roadmap to start on.

If you came here expecting "8 weeks of careful staged training," you are reading the wrong document. The 1-week window does not buy us a learned graph policy. What it buys us is an inference layer that uses the graph to *coordinate the policy we already trained*. The end behavior visible on the field can be very similar; the engineering posture is completely different.

---

## 1. Glossary — terms you will see in this document

If you have not worked on this codebase before, read this section before anything else.

- **NAO** — the humanoid robot platform. Four NAOs per RoboCup SPL team.
- **PPO / MAPPO** — Proximal Policy Optimization / Multi-Agent PPO. The RL algorithms we use. We will not be touching them in this sprint.
- **Policy** — the neural network that maps observation → action. Has a categorical skill head (chooses from `stand / walk / dribble / shoot`) and a continuous residual head (3 floats that adjust where to aim the skill). **We do not retrain it in this sprint.**
- **Skill** — discrete primitive the BHuman C++ stack executes. Four production skills: `stand`, `walk`, `dribble`, `shoot`.
- **Residual / anchor** — the policy emits *deltas* relative to a sensible default (the anchor), not absolute coordinates. Walk anchor = "walk toward the ball"; residuals shift it.
- **Observation (obs)** — 26-float vector the policy reads each step. Includes pose, ball pose, nearest-teammate/opponent distances, shot-quality features, and three gate bits at indices 23/24/25. **OBS_SIZE = 26 and stays 26 in this sprint.**
- **Skill gate** — `gate.py`. Decides per step which skills are "armed" (shoot_armed, dribble_armed). Bakes those bits into the obs at 23/24/25.
- **Design-A** — invariant that the policy's action mask, the gate's bits in obs, and the teacher's labels all agree on what's allowed. **Hard mask coordination in this sprint preserves Design-A because we narrow the mask, not extend it.**
- **BC (behavioral cloning), BC replay, teacher, teacher dataset** — these all exist for the training-time loop. **They are irrelevant in this sprint** because we are not training. We list them here only so you recognize them in the codebase.
- **Curriculum / stage** — `curriculum.py`. Training stages 0-4. Irrelevant here; we are post-training.
- **Scenario / world** — `production_curriculum.py`. Named situations with deterministic placement. Used by `behavioral_evaluator.py` to grade a checkpoint.
- **Behavioral evaluator** — `behavioral_evaluator.py`. The production gate. Returns exit code 0 iff all scenarios pass their pass_fn.
- **MultiAgentCoordinator** — the Python wrapper introduced by the 4v4 sprint (`.docs/09`). Assigns roles (striker = closest to ball, supports = others), applies per-role skill masks and residual overrides on top of policy outputs. **This is where Version B's graph logic lives.**
- **Production band** — 2.5-3.0 m from goal where shoot can credibly score. Hard field constraint.
- **The production checkpoint** — the single-agent striker that just passed `behavioral_evaluator.py`. This is our starting checkpoint. We do not modify it.

---

## 2. The graph coordination idea, explained from scratch

### 2.1 The intuition

Soccer is a team sport. When the striker has the ball and a teammate is in a clean shot lane, the right behavior is for the striker to drop the ball into that lane and the teammate to be there to take the shot. The current single-agent policy was trained without any notion of "teammate" — each robot acts as if it were the only RL agent.

The graph idea: treat the team as a graph (robots = nodes, relations like "open for a shot" = edges) and use the graph to drive coordination behavior on top of the trained policy.

### 2.2 The data structure

A 4×4 adjacency matrix over our 4 robots (3 RL + 1 BH GK). Each edge carries typed relations:

- `has_ball_to_open_lane`: source has the ball; destination is in a clean shot lane.
- `passable`: a pass from source to destination is geometrically feasible.
- `nearest_to_ball`: exactly one node is the striker — the one closest to the ball.

The GK node is **read-only** (BH controls it). We observe its state; we never bias or constrain it.

### 2.3 What Version B does with the graph

Two things, both at **inference time inside the MultiAgentCoordinator**:

**(a) Telemetry.** The adjacency matrix is computed each step and written to a CSV alongside the per-step action and observation. This makes coordination behavior *visible* — post-game you can read the CSV and see exactly when the team was in a passable configuration and what each robot did about it.

**(b) Hard mask + residual override.** When the graph says "robot X is the open destination of a `has_ball_to_open_lane` edge":
- Robot X's skill mask is narrowed to `{walk}`. Even if the trained policy wanted to emit `shoot` or `dribble`, the wrapper overrides it.
- Robot X's walk residual is replaced with one that points at the open-lane geometric center, not at the ball.

This is **not learning.** The policy's weights are unchanged. The wrapper is forcing the policy to do the coordinated thing in the situations where coordination matters, using the trained policy's walk skill as the execution mechanism.

### 2.4 What Version B does NOT do

- Does not give the support agent a positive reward for being open (no shaping signal, because no training).
- Does not give the team a positive reward for completed passes (same reason).
- Does not let the policy *learn* when to coordinate; the wrapper *imposes* coordination.

The behavior visible on the field can still be reasonable. The difference is that we are not generalizing — every coordination decision is hand-coded in the wrapper. If the field changes (new opponent strategy, new scenario), we update the wrapper, not the policy.

---

## 3. Why we cannot do the full Version A in 1 week — the four-lens analysis

This section explains the engineering reason Version B exists. Read it before §4 so the scope of Version B feels justified, not arbitrary.

The current pipeline is structurally hostile to the full graph idea. The hostility is concentrated in four places, and most of it produces **silent failures** — the pipeline will report "healthy" while the policy ignores the graph signal.

### 3.1 Lens 1 — Teacher policy is single-agent (file: `teacher_policy.py`)

`stage_aware_teacher_action` assumes "I am the controlled robot, I drive the ball, I shoot when armed." It has no role concept, no team-aware gate predicate ("am I in an open lane for the ball-holder?"), and no anchor for "walk to the open lane" (existing anchors are ball-centric).

To make graph behavior teachable you'd extend the teacher with a role branch, add an `open_for_pass_armed` gate predicate in `gate.py` (team-aware), and add a parallel anchor system for open-lane geometry. That's 3-5 days of careful design work.

### 3.2 Lens 2 — Teacher dataset filters drop the right answers (file: `teacher_production_dataset.py`)

`_episode_kept` drops any episode where `dribble == 0 AND shoot == 0` (calls it `pure_walk_stall`). But a correctly positioned support has zero dribbles and zero shoots — it walked into the open lane and stood ready. The filter would drop the cleanest examples of graph-coordinated behavior.

The Design-A audit `_gate_label_consistent` only checks local gate bits; a support's `walk` with `none_armed` passes the audit trivially and carries no graph-conditional content. The dataset would self-certify as healthy and be empty of graph signal.

Re-collecting with the right filters and per-role coverage takes 1-1.5 weeks wallclock on visible SimRobot, after Phase 1 lands.

### 3.3 Lens 3 — BC machinery actively anti-learns the new feature (file: `train.py`)

This is the most subtle failure mode. The warm-start zero-pad in `_load_state_dict_obs_padded` (`train.py:1206`) handles obs growth at load time — that's fine. But the continuous BC replay is hostile:

- The **KL anchor** in `BCReplayRegularizer._loss` (`train.py:1441`) computes divergence against a frozen clone of the just-warm-started policy. After zero-pad, the clone's weights on any new columns are zero — i.e. graph-blind. Every PPO update that learns graph-conditioned behavior raises the KL and gets damped back toward graph-blindness.
- The **competence gate** (`BCCompetenceGate`, `train.py:1508`) decays `bc_beta` based on `dribble_compat_rate` / `clean_takeover_rate` — both striker-only metrics. β cuts off before any graph behavior is learned, handing the new feature to pure PPO under sparse reward.

You'd need lower `bc_beta_max`, much longer `bc_competence_half_life_evals`, and new coordination metrics plumbed into the competence gate's input. Tuning those without regressing the single-agent baseline is 1-2 weeks of fine-tune iteration.

### 3.4 Lens 4 — Curriculum is the friendliest lens but still broad (files: `curriculum.py`, `reward.py`, `environment.py`)

This one is structurally additive. You add reward terms (`graph_open_bias`, `pass_credit`), add a stage 5 spec, add coordination metrics through `episode_metrics`, and tune one knob at a time against the evaluator. 3-5 days for the plumbing, plus iteration.

But there's no pass skill in `ACTIVE_SKILLS` — the closest proxy for `pass_credit` is "ball ownership transitioned between RL agents in ≤ k frames," which is fragile under perception flicker.

### 3.5 The silent failure property

All four lenses combine to produce a dangerous property: **the surface signals all report healthy while the policy ignores the graph entirely.** Dataset audit passes, BC loss decreases, competence gate decays β cleanly, curriculum promotes — and the policy ships graph-blind. You only catch it with a multi-agent evaluator rubric that does not exist yet.

### 3.6 Net implication for the 1-week budget

The minimum coherent Version A is: gate extension (1 week) → teacher rewrite (1-1.5 weeks) → multi-agent worlds (3-5 days) → dataset re-collection (1-1.5 weeks wallclock) → reward + stage (3-5 days) → fine-tune (2-3 weeks). **8-10 weeks minimum.** With 7 days, every single piece would be hurried, and the silent failure modes would not be caught.

**Therefore we ship Version B.** Hard mask + telemetry. No retraining. The next sections describe what that actually looks like.

---

## 4. What Version B is, concretely

### 4.1 The pieces that change

Three files get edits:

- **`MultiAgentCoordinator`** (introduced by the 4v4 sprint, `.docs/09`). The graph computation and the hard-mask logic land here.
- **`behavioral_evaluator.py`**. Three new pass functions for team coordination metrics. Three new scenario entries.
- **`production_curriculum.py`**. Three new team scenarios (3 RL agents spawned, opponents placed to favor coordination).

Two files do **not** change:

- **The trained policy `.pt`**. We use the production checkpoint as-is.
- **OBS_SIZE / config.py**. 26 stays 26.

Files we explicitly do not touch:

- `teacher_policy.py`, `teacher_production_dataset.py`, `train.py`, `train_production.py`, `curriculum.py`, `reward.py`, `gate.py`.

(`gate.py` may get *one* new public function for computing `is_open_for_pass(obs, ball_holder_pose)` — a pure-Python predicate that does not change the existing gate, does not change OBS_SIZE, and is consumed by the wrapper only. This is on the boundary; if the predicate fits naturally in the wrapper file, put it there instead.)

### 4.2 The graph layer's contract

Per step, inside `MultiAgentCoordinator.step(observations_per_agent)`:

1. **Compute the adjacency.** For each ordered pair `(i, j)` of RL agents, compute the edge struct:
   ```
   edge = {
       has_ball_to_open_lane: bool,
       passable: bool,
   }
   ```
   Plus a per-node `is_striker: bool` (exactly one true — the agent closest to the ball with possession indicators consistent).

2. **Run the trained policy** on each agent's obs, exactly as the 4v4 wrapper already does. Get a proposed action per agent.

3. **Apply the hard mask.** For each agent:
   - If `is_striker`: pass the policy's action through unchanged (striker uses the trained behavior).
   - If `not is_striker` (i.e. support) and an edge `has_ball_to_open_lane` points *to* this agent: force `skill = walk`, residual = open-lane anchor.
   - If `not is_striker` and no edge points to this agent: fall back to the 4v4 wrapper's default support behavior (the hand-coded support position from `.docs/09`).

4. **Emit the action.** Return `{agent_id: payload}` exactly as the wrapper already does.

5. **Log telemetry.** Append a row to `runs/<run_id>/graph_telemetry.csv` with the adjacency state + each agent's override decision.

The contract preserves Design-A because every mask narrowing is a strict subset of what the gate already allowed. The wrapper never *adds* a skill that the gate would have forbidden; it only restricts.

**Note on policy taxonomy:** the wrapper above implements two role behaviors (striker, open-support) and falls back to the 4v4 wrapper default for off-ball-support. The graph idea actually induces seven distinct coordination policies (striker, open-support, off-ball/triangulation, defensive-recovery, spacing, strong/weak-side, pass-lane anticipation). Version B implements the first two as hard masks because they're the highest-value pair and the 1-week budget can't carry more. For the full taxonomy, the convergence engineering rationale, and the obs-gate design that adds them to Version A's input, see **§11.A**.

### 4.3 The open-for-pass predicate (Version B definition)

Conservative implementation, no team-aware perception, no new gate state:

```python
def is_open_for_pass(my_obs, ball_holder_obs):
    # Both obs are BackendObservation instances.
    # I am "open for pass" iff:
    #   1. The shot lane from MY position to the goal is clean
    #      (my_obs.shot_opening_with_obstacles > 0.6).
    #   2. I am in the production band 2.5 - 3.0 m from goal.
    #   3. The line from ball_holder to me is clear of opponents
    #      (rough geometric check against ball_holder_obs's nearest
    #      opponent features).
    #   4. I am not the ball holder.
```

Conservative because we err on the side of *not* firing the override. If `shot_opening_with_obstacles` is uncertain, the predicate returns false and the support falls back to its default position. False negatives (we fail to coordinate) are recoverable; false positives (we force walk when we shouldn't) can break a play.

---

## 5. Prerequisites — must be true before Day 1

If any of these is false, the 7-day plan does not start.

- **Production single-agent checkpoint exists and passes `behavioral_evaluator.py`.** This is the policy we use for inference. We do not have a Version B without a working single-agent policy.
- **4v4 wrapper baseline (`.docs/09`) is built and runnable.** `MultiAgentCoordinator` exists, role assignment works, hard-coded support positions exist. We're adding the graph layer on top.
- **One visible-SimRobot host is available for evaluation.** ~30 min per behavioral_evaluator run; we need 4-5 runs across the week.
- **The user owns the production push.** We don't push without authorization.

---

## 6. Day-by-day plan

7 calendar days. Each day has one focused goal, one deliverable, one validation gate.

### Day 1 — Adjacency computation + telemetry only

**Goal:** make the graph state visible. No behavior change yet.

**Tasks:**
- Add `compute_adjacency(observations_per_agent)` to `MultiAgentCoordinator` (or a new module `team_graph.py` if the wrapper is getting crowded). Pure function: takes the per-agent obs dict, returns the 4×4 adjacency.
- Implement `is_open_for_pass` per §4.3.
- Compute the adjacency every step in the wrapper. **Do not consume it yet** — the policy and the 4v4 wrapper behave exactly as before.
- Dump the adjacency to `runs/<run_id>/graph_telemetry.csv` per step. Columns: `step, agent_id, role, is_striker, edges_in (json), edges_out (json), shot_opening, ball_distance_to_goal`.

**Deliverable:** PR with the adjacency computation + telemetry dump. No behavior change vs. the 4v4 wrapper baseline.

**Validation gate (must pass to start Day 2):**
- Run the 4v4 wrapper against `production_scenarios` with the new telemetry on. Confirm the existing single-agent `_pass_finish` etc. still pass. (Nothing should change — the adjacency is unused. This is a regression check.)
- Open `graph_telemetry.csv` from one episode. Spot-check 10 rows. Confirm `is_striker` matches what you'd expect from the ball position; confirm `is_open_for_pass` fires occasionally when a support is in the band.

### Day 2 — Hard mask + open-lane anchor

**Goal:** force support behavior when the graph says "I am the open destination."

**Tasks:**
- Add an open-lane anchor: `compute_open_lane_target(my_pose, ball_holder_pose)`. Returns the (x, y) the support should walk to. v1 implementation: midpoint between me and the goal, offset perpendicular toward the cleaner lane.
- In `MultiAgentCoordinator.step`, after the policy emits its action and before the env step:
  - If agent is a support AND an edge `has_ball_to_open_lane` points to this agent:
    - Override `skill_id := walk`.
    - Override residuals to point at `compute_open_lane_target`.
  - Else: existing 4v4 wrapper behavior (default support position).
- Add a per-step boolean `graph_override_applied` to the telemetry CSV so you can grep for overrides post-hoc.

**Deliverable:** PR with the hard mask + anchor + override logging.

**Validation gate:**
- Run the 4v4 wrapper against `production_scenarios` (single-agent regression check). `_pass_finish` etc. must still pass.
- Run against one team scenario you'll build on Day 3-4 (or a manual hand-set world spec). Confirm `graph_override_applied` is true for ≥ 5% of support steps and that the override moves the support toward a credible open-lane position. Eyeball one episode in visible SimRobot.

### Day 3 — Team scenarios in `production_curriculum.py`

**Goal:** create the worlds the team coordination will be graded on.

**Tasks:**
- Add a `_team_drive_finish_world(rng, *, goal_distance_mm, obstacles, support_offset_mm)` builder that spawns 3 RL agents:
  - Striker positioned to approach the ball as in single-agent.
  - Left support positioned on a credible open lane to the left of the striker, offset from the ideal by `support_offset_mm`.
  - Right support mirrors right.
- Add 3 team scenarios:
  - `team_finish_band_2p5_3m_clean` — ball in band, supports near-ideal, no opponents.
  - `team_blocked_lane_repair` — striker's direct lane is blocked by an opponent pinch, forcing the team to use the support's lane.
  - `team_dynamic_pressure` — 3 RL agents with dynamic opponents.
- Add `team_production_scenarios(episodes)` returning the new list. Leave `production_scenarios` untouched.

**Deliverable:** PR with the team world builder + 3 scenarios.

**Validation gate:**
- Spawn each team scenario once in visible SimRobot via `behavioral_evaluator.py`. Confirm 3 RL agents spawn at sensible positions, opponents place correctly, < 5% pose-warning rate over 4 episodes.

### Day 4 — Multi-agent evaluator pass functions

**Goal:** define what "good Version B coordination" means in measurable terms.

**Tasks:**
- Add metrics to `environment.py episode_metrics`:
  - `graph_override_rate`: fraction of support steps where the hard mask fired.
  - `support_open_hold_steps_mean`: avg steps a support held a position with `is_open_for_pass == true`.
  - `team_goal_rate`: episodes where any RL agent scored.
  - `role_thrash_rate`: episodes where role assignment flipped > T times (T = 5 over a 400-step episode).
- Wire those through `train._aggregate_eval_metrics` so they show up in `eval_summary` (this is just adding the keys to the aggregator — `_aggregate_eval_metrics` already passes through unknown keys).
- Add 3 pass functions to `behavioral_evaluator.py`:
  - `_pass_team_coordination`: `team_goal_rate >= 0.05`, `graph_override_rate >= 0.10`, `role_thrash_rate <= 0.10`.
  - `_pass_support_positioning`: `support_open_hold_steps_mean >= 30`, `collision_rate <= 0.12`.
  - `_pass_team_no_regression`: re-run single-agent scenarios and confirm `_pass_finish` etc. still pass.
- Register the team scenarios in `_PASS_FNS`.

**Deliverable:** PR with the 3 pass functions and the metric plumbing.

**Validation gate:**
- Run `behavioral_evaluator.py` against the production checkpoint **with the wrapper graph layer turned off** (toggle: `graph_override_applied` never set). Confirm:
  - `_pass_team_no_regression` passes (single-agent unchanged).
  - `_pass_team_coordination` *fails* (no coordination yet — this is the baseline).
  - `_pass_support_positioning` likely fails.
- Numbers are well-formed (no NaN, sensible ranges). This is your baseline you'll compare against.

### Day 5 — End-to-end run with the graph layer on

**Goal:** measure the lift from the hard mask.

**Tasks:**
- Turn the graph layer on (it's been computing telemetry all week but Day 2's override logic should be conditionally gated on a `RL_GRAPH_OVERRIDE=1` env var or a wrapper flag — flip it).
- Run `behavioral_evaluator.py --episodes 8` against the full scenario list (single-agent + team).
- Read the output:
  - Single-agent rubric: still passing? If not, Day 6 is debugging, not iterating.
  - Team rubric: how many of the 3 team pass functions pass?
- Open `graph_telemetry.csv` from one team episode. Manually trace 10 steps where `graph_override_applied = true`. Did the override produce the intended support motion?

**Deliverable:** a run summary log + first version of the day's findings.

**Validation gate:** at least one of `_pass_team_coordination` / `_pass_support_positioning` passes. Single-agent rubric does not regress. If both team pass functions fail, Day 6 is calibration; if both pass cleanly, Day 6 is robustness sweeps.

### Day 6 — Threshold calibration

**Goal:** tune the `is_open_for_pass` predicate and the open-lane anchor based on Day 5 findings.

**Calibration knobs (one at a time, validate after each):**

- `shot_opening_with_obstacles` threshold in `is_open_for_pass`. Default 0.6. If override fires too rarely → lower to 0.5. If override fires falsely → raise to 0.7.
- Production band tolerance in `is_open_for_pass`. Default 2.5-3.0 m. Can widen to 2.4-3.1 m to catch edge cases; do not narrow.
- Open-lane anchor offset. The perpendicular offset from "midpoint between me and goal." If supports stack on top of the striker → increase offset; if supports stray off the field → decrease.
- Role thrash mitigation. If `role_thrash_rate` is high, add a 5-step hysteresis to the role assignment in `MultiAgentCoordinator` (keep current role unless the new candidate has been closer to the ball for ≥ 5 frames).

After each change: re-run `behavioral_evaluator.py --episodes 4` against the team scenarios. Single-agent rubric must keep passing.

**Deliverable:** final values of all calibration knobs, baked into the wrapper code.

**Validation gate:** the team rubric is in the best state you can get by end of Day 6. At minimum `_pass_team_no_regression` passes; at minimum one of the two team pass functions passes.

### Day 7 — Final validation + handoff

**Goal:** confirm shippability and document.

**Tasks:**
- Final eval run with `--episodes 8` across the full scenario set. Read every line of `summary.csv`.
- Cross-check the telemetry: `graph_telemetry.csv` from 3 different episodes (clean band, blocked lane, dynamic pressure). Confirm the graph state makes sense each time.
- Write `GRAPH_LAYER.md` at repo root containing:
  - Launch command (`MultiAgentCoordinator` with `RL_GRAPH_OVERRIDE=1`).
  - The `is_open_for_pass` definition + the calibrated thresholds.
  - The open-lane anchor definition + offset value.
  - The role-thrash hysteresis value if added.
  - Verification gates (what numbers in `eval_summary.csv` indicate the layer is functioning).
  - The list of files touched.
  - The list of files NOT touched (so a future reader knows the policy + dataset + training pipeline are untouched).
- **Do not push.** Commit locally. The user owns the push decision.

**Deliverable:** final commit + `GRAPH_LAYER.md`.

**Validation gate:** the user approves the commit for the deployment pipeline.

---

## 7. Daily validation gates summary

Use this as a single-page reference during the sprint.

| Day | Must-pass before next day | Likely failure to plan for |
|---|---|---|
| 1 | Single-agent regression check passes; telemetry CSV reads sensibly | `is_striker` flips between agents — fix role assignment in wrapper |
| 2 | Single-agent still passes; override fires ≥ 5% of support steps on test scenario | Override fires but support walks in a weird direction — anchor bug |
| 3 | Team scenarios spawn cleanly; < 5% pose-warning rate | Pose warnings — likely spawn coordinates conflict with field bounds |
| 4 | Single-agent rubric passes; team rubric baseline fails (expected); metrics well-formed | New metric returns NaN — wiring through `_aggregate_eval_metrics` |
| 5 | Single-agent rubric still passes; ≥ 1 team pass function passes | Both team pass functions fail — go to Day 6 calibration |
| 6 | Team rubric improved over Day 5; single-agent never regresses | Single-agent regressed during calibration — revert and re-tune |
| 7 | Final 8-episode run passes; documentation complete | — |

---

## 8. Risks and mitigations

### Risk 1: Single-agent regression from the override

**Symptom:** `_pass_finish` fails after adding the graph layer.

**Cause:** the override fires too aggressively. A support who was about to take a useful action gets forced into a worse position. Or `is_striker` is unstable and the wrapper occasionally treats the actual striker as a support.

**Mitigation:**
- The override is gated on `is_striker == False`. Make the striker assignment conservative (5-step hysteresis from Day 6).
- The override only fires when the predicate is *strictly* true. Conservative thresholds in `is_open_for_pass`.
- If regression appears mid-week, immediately roll back the override (env var `RL_GRAPH_OVERRIDE=0`) and re-run. If single-agent passes with override off, the bug is in the override logic, not the policy.

### Risk 2: Role thrash

**Symptom:** `role_thrash_rate` is high; supports keep swapping mid-episode; behavior looks chaotic.

**Cause:** ball position is noisy on visible SimRobot. The closest-to-ball calculation flips between two agents who are roughly equidistant.

**Mitigation:**
- Hysteresis on role assignment: keep current striker unless another agent has been ≥ 200mm closer to the ball for ≥ 5 consecutive frames.
- If thrash is still high, reduce to 2 supports (drop one to a fixed defensive role) for the team scenarios.

### Risk 3: `is_open_for_pass` is too conservative — never fires

**Symptom:** `graph_override_rate < 0.05` even on `team_finish_band_2p5_3m_clean`.

**Cause:** `shot_opening_with_obstacles` is below 0.6 for most support positions even in clean scenarios.

**Mitigation:**
- Day 6 calibration: lower threshold to 0.5 or 0.45.
- Inspect `shot_opening_with_obstacles` distribution on the support obs from the telemetry CSV. Pick a threshold at the 60th percentile of the observed values.

### Risk 4: Visible SimRobot wallclock blows past the 7-day budget

**Symptom:** behavioral_evaluator runs taking >> 30 min; slot wedges.

**Cause:** historical. 4 slots is the safe ceiling; 6+ wedges. Reset budgets matter.

**Mitigation:**
- Stay at 4 slots. Do not be tempted to scale up.
- Keep the production reset budget envvars: `RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS=4000`, `RL_VISIBLE_BALL_SETTLE_MAX_REPLACES=5`, `RL_VISIBLE_RESET_MAX_ATTEMPTS=3`, `RL_VISIBLE_STEP_OBS_TIMEOUT_MS=1000`.
- Run evaluations overnight when possible.

### Risk 5: The hard-mask coordination doesn't actually score goals

**Symptom:** all metrics improve (`graph_override_rate` high, `support_open_hold_steps` high) but `team_goal_rate` is still ~0.

**Cause:** the support holds the open position, but the striker never drops the ball — no pass actually happens because the striker policy was never trained to pass. The closest behavior is "striker dribbles past the support."

**Mitigation:**
- This is a real limitation of Version B. Document it honestly in `GRAPH_LAYER.md`. The hard mask coordinates support positioning; striker pass behavior would require either retraining (Version A) or a hard-coded "drop ball when support is open" rule in the striker side of the wrapper.
- Day 5-6: if `team_goal_rate` stays at 0, add a striker-side rule: if `is_striker AND there exists a support with is_open_for_pass`, override striker `skill := dribble` with residual pointing toward the support (this is a deliberate ball-drop). This is more aggressive than the support-side override; treat it as a Day 6 experiment, not a Day 2 baseline.

---

## 9. Anti-patterns — what you should NOT do

1. **Do not retrain anything.** No training script gets launched this week. `train_production.py`, `train_bc.py`, `train.py` are read-only.

2. **Do not grow OBS_SIZE.** The wrapper consumes the existing 26-dim obs. New information lives in *wrapper state*, not in the policy's input.

3. **Do not modify `teacher_policy.py` or `teacher_production_dataset.py`.** They are for training-time. We're not training.

4. **Do not add reward terms to `reward.py`.** Same reason. The reward function is consulted at training time; we are not training.

5. **Do not add a stage 5 to `curriculum.py`.** Same reason.

6. **Do not skip the single-agent regression check.** Every day's validation gate includes "single-agent rubric still passes." Skip it once and you ship a Version B that's worse than the baseline.

7. **Do not extend `gate.py`'s `SkillGate`.** The wrapper can call its own pure-function `is_open_for_pass` predicate; it does not need to be part of the training-time gate.

8. **Do not raise more than 4 visible SimRobot slots.** Historical wedge precedent.

9. **Do not push to main without authorization.** User owns the push decision.

10. **Do not commit `graph_telemetry.csv` files.** They live under `runs/` which is gitignored; force-adding them is wrong here.

---

## 10. What this does NOT accomplish

Be honest with stakeholders. Version B delivers:

- **Coordinated support positioning.** Supports walk to and hold open lanes when the graph says they should.
- **Telemetry that explains team behavior.** Post-game CSV showing exactly when coordination decisions fired.
- **No single-agent regression.** The striker behaves as it does in production.

Version B does **not** deliver:

- **A pass-trained policy.** The striker does not know how to drop the ball cleanly for a support; passes that work look like coordinated dribbles.
- **Learned coordination generalization.** Every coordination decision is hand-coded thresholds. New scenario types may need wrapper updates.
- **Pass-completion rewards or shaping.** The policy gets no learning signal from coordination.
- **The graph as a *learned feature*.** The policy never sees the adjacency.

If on the field you need full team coordination with learned passing, Version A is the project. §11 describes it.

---

## 11. Future work — Version A (the full retrain) as appendix

For the future sprint when we have 8-10 weeks. Recorded so the choice is documented; **not a roadmap to start now.**

Prerequisites: production single-agent + 4v4 wrapper shipped (same as Version B), plus Version B shipped (so we have the telemetry baseline).

### Phased outline

- **Phase 1 (1 week):** extend `gate.py` with the `open_for_pass_armed` predicate. Add the role one-hot + soft progress floats to the obs per the design spec in §11.A (OBS_SIZE 26 → 31, 5 new dims at indices 26-30). Extend `gate_skill_mask_from_obs` with role-conditioned forbid-invalid per the contract in §11.A.4. Validate that warm-starting the production checkpoint at OBS_SIZE=31 via the existing `_load_state_dict_obs_padded` (`train.py:1206`) does not regress single-agent. Follow the staged addition order in §11.A.6 — land the role one-hot first and verify before adding the progress floats.
- **Phase 2 (3-5 days):** extend `behavioral_evaluator.py` with team metrics in `episode_metrics`. Add 3 team pass functions. Confirm the baseline (no retraining yet) fails the team rubric in a measurable way (gives us the calibration target).
- **Phase 3 (1-1.5 weeks):** rewrite `teacher_policy.py` with per-role branches per §11.B.2 (striker / open_support / off_ball_support). Add `compute_open_lane_target` and `build_open_lane_anchor` per §11.B.3. Centralize team-aware state (role assignment with 5-frame hysteresis, EMA-smoothed coordination targets) in the env per §11.B.3 — the teacher stays stateless. Unit test that striker labels are bit-identical to today's single-agent labels, open-support labels walk toward smoothed open-lane targets with tolerance-based stand-vs-walk, and off-ball-support labels triangulate with an unexpected-possession escape hatch.
- **Phase 4 (3-5 days):** add team scenarios to `production_curriculum.py` (the Version B team scenarios are the starting point; team-trained scenarios need more variation).
- **Phase 5 (1-1.5 weeks wallclock):** re-collect dataset at OBS_SIZE=31 with per-role teacher labels. Per-role success filter (replace `_episode_kept`). Per-role quality weight. Record per-row `role_id` so `BCReplayRegularizer._sample_indices` can stratify by `(stage, role, skill)` (see §11.A.5). New audit `positive_weight_graph_label_consistent`.
- **Phase 6 (3-5 days):** add reward terms per §11.B.6.1 (`support_open_hold_bonus`, `support_triangle_hold_bonus`, `team_goal_bonus`, `pass_attempt_credit`). Add stage 5 spec in `curriculum.py` per §11.B.6.1 — sequential, NOT in-place re-spec of stage 4. Extend `BCCompetenceGate` to take the **min** across (striker, open-support, off-ball-support) competence per §11.B.5.2 (not the mean — BC must stay alive until every role is competent). Add `stage4_finish_interleave_probability = 0.2` for single-agent retention.
- **Phase 7 (2-3 weeks):** fine-tune `train_team_production.py` (sibling to `train_production.py`, full config in §11.B.6.3). **Critical retunings:** `bc_beta_max = 0.08`, `bc_competence_half_life_evals = 25`, dual-rubric eval at every checkpoint, per-eval auto-rollback when single-agent regresses (§11.B.6.4). KL anchor must be the production single-agent checkpoint loaded via `_load_state_dict_obs_padded` — **not** a fresh warm-start — per §11.B.5.3. Role-stratified BC sampling per §11.B.5.1. Calibrate reward terms one at a time vs `behavioral_evaluator.py`.
- **Phase 8 (1 week):** final validation, deployment, handoff doc.

**Total: 8-10 weeks** if all goes right. The silent-failure modes from §3 are real risks at every phase; build the rubric (Phase 2) before the dataset (Phase 5) or you will collect graph-blind data.

---

## 11.A Coordination policies and obs-gate design spec

This subsection is the design detail for §11 Phase 1-3. It defines (a) the behavioral policies the graph induces, (b) which of those policies become obs-gate columns in the policy's input, and (c) the convergence engineering rationale for the chosen set. **Version B implements a subset of these as wrapper rules; Version A implements them as obs-gated learned behaviors.** Both versions share the same underlying policy taxonomy — the difference is where the logic lives (wrapper vs. obs+mask+learned-policy).

### 11.A.1 Policies vs. obs-gates — different layers

- **Coordination policies** are behavioral rules derived from graph state. In Version B they live in `MultiAgentCoordinator` as Python functions. In Version A they're produced by per-role teacher branches and learned by the policy through BC + PPO.
- **Obs-gates** are float/bit columns in the observation that the policy reads and that `gate_skill_mask_from_obs` consults to derive the forbid-invalid action mask.
- Every policy has a candidate obs-gate twin, but adding obs columns is expensive (KL anchor anti-learning, dataset re-collection, all four lenses from §3). Prune the obs-gate set ruthlessly to the bits that most reduce convergence variance — not every policy earns a gate bit.

### 11.A.2 The seven coordination policies the graph induces

| # | Policy | Trigger | Behavior | Mask implication | Obs-gate twin |
|---|---|---|---|---|---|
| 1 | Striker | Closest to ball AND possession indicators (ball confidence high, within takeover envelope) | Drive ball to goal, shoot when armed. Unchanged from today's single-agent behavior. | Standard single-agent gate. | `is_striker` bit (recommended) |
| 2 | Open support | Not striker AND graph edge `has_ball_to_open_lane` → me | Walk to open-lane geometric center, hold position. | Forbid shoot, dribble. | `is_open_support` bit (recommended) |
| 3 | Off-ball support (triangulation) | Not striker AND not open support | Hold triangulation vertex with striker + other support (opposite side of field from open support, equidistant). | Forbid shoot; dribble allowed for unexpected possession. | `is_off_ball_support` bit (recommended) |
| 4 | Defensive recovery | Opponent has ball AND ball in our defensive half | Closest-RL intercepts; others hold cover positions on most-threatening pass lanes. | Forbid shoot for all. | `defensive_mode_armed` bit (defer to v2) |
| 5 | Spacing / no-overlap | Teammate within ~800-1000 mm | Walk perpendicular to teammate heading until spacing restored. Applied only to the lower-priority agent of the overlapping pair to avoid oscillation. | None (residual override only). | None — wrapper-only |
| 6 | Strong/weak side balance | Passive (derived from ball y-position) | At any moment, one support on the ball's y-half ("strong"), one on opposite ("weak"). | None (soft). | None — captured implicitly by triangulation |
| 7 | Pass-lane anticipation | Off-ball-support AND a future open lane (≤ 2-3 s) is forming | Pre-position toward predicted lane center; pre-rotate to face it. | None. | Defer — continuous `pass_anticipation_progress` for v3 |

Per-policy robustness notes:

- **Open support (Policy 2)** is the highest-value coordination policy — it creates the scoring opportunity. Dominant failure mode: open-lane geometric center jitters on noisy ball perception. Mitigation: smooth the target position over 3-5 frames so the support doesn't oscillate.
- **Off-ball support (Policy 3)** failure mode: role thrash. When role assignment flips, the triangulation target moves discontinuously. Mitigation: 5-frame hysteresis on role assignment (do not flip role unless the new candidate has been closer to the ball for ≥ 5 consecutive frames).
- **Defensive recovery (Policy 4)** is real but v2 work; not on the critical convergence path for offensive coordination. Defer until offensive team-rubric clears.
- **Spacing (Policy 5)** is mandatory but doesn't need an obs bit. Wrapper-side residual override suffices. The key is applying the override only to the lower-priority agent — if both apply simultaneously they oscillate.
- **Strong/weak side (Policy 6)** is captured implicitly by triangulation's "opposite side from open support" component. No additional structure needed for v1.

### 11.A.3 Minimal obs-gate set — 5 new dimensions, OBS_SIZE 26 → 31

Of the seven policies, only the first three merit obs-gate representation in v1. Policies 4-7 are wrapper-only or future work.

**Hard role one-hot (3 bits):**

| Index | Name | Semantics |
|---|---|---|
| 26 | `is_striker` | Closest-to-ball AND possession indicators consistent |
| 27 | `is_open_support` | Not striker AND graph edge `has_ball_to_open_lane` → me |
| 28 | `is_off_ball_support` | Not striker AND not open support (and I'm not GK) |

Exactly one of the three is true for an RL agent on each step. The GK reads false on all three from the RL side (we never train the GK).

**Why 3-bit one-hot, not 2-bit "is_support" + "is_open":** each role is a distinct policy regime (striker drives + shoots, open-support holds the open lane, off-ball triangulates). The 3-bit encoding lets the policy network learn three role-conditioned sub-policies cleanly rather than having to decode "support AND NOT open = off-ball" via internal feature combinations. Same parameter cost, cleaner learning.

**Soft progress floats (2 floats):**

| Index | Name | Range | Semantics |
|---|---|---|---|
| 29 | `open_lane_progress` | [0, 1] | Gradient of pass-lane cleanness toward me. Equal to the smoothed `is_open_support` predicate before thresholding. Rises continuously as I approach the open-lane geometric center; non-zero even when `is_open_support == 0`. |
| 30 | `support_spacing_progress` | [0, 1] | 1.0 when I am well-spaced from the other support (distance ≥ overlap threshold); decays linearly to 0 within stacking distance. |

**Total new dims: 5. OBS_SIZE: 31.** (Phase 1 of §11 updated to reflect this.)

### 11.A.4 Extended `gate_skill_mask_from_obs` contract

```python
def gate_skill_mask_from_obs(obs):
    base_mask = single_agent_gate_mask(obs)  # existing logic at obs[23:26]
    if obs[is_striker]:
        role_mask = {stand, walk, dribble, shoot}    # standard single-agent gate
    elif obs[is_open_support]:
        role_mask = {stand, walk}                    # walk to lane, then stand
    elif obs[is_off_ball_support]:
        role_mask = {stand, walk, dribble}           # mostly walk; dribble for unexpected possession
    else:
        role_mask = {stand, walk}                    # fallback (should not occur in practice)
    return base_mask AND role_mask
```

The role mask is ANDed with the existing single-agent gate mask. Net effect: the existing `shoot_armed` / `dribble_armed` logic applies *only* to the striker; supports are structurally prevented from emitting shoot regardless of how the underlying single-agent gate sees them.

**Design choices in the contract:**

- **Open-support allows `stand`** in addition to `walk`. A perfectly positioned support holding still should be allowed to either stand or walk-with-zero-residual. Allowing both gives the policy expressive flexibility at no cost.
- **Off-ball-support allows `dribble`** for the unexpected-possession edge case. If an off-ball-support takes ball control mid-frame (a deflection bouncing toward them), they should be able to react. On the next frame the role assigner promotes them to striker; the momentary dribble is a clean transition.
- **Open-support forbids `dribble`** because the wrapper's residual is forcing them toward the open-lane geometric target, not toward the ball. Allowing dribble would create an action whose residual anchor doesn't make sense in the open-support frame of reference.
- **Everyone except striker forbids `shoot`**. Hard. There is no scenario where a support shooting from a non-striker role is the right call; the role assigner would have promoted them to striker first if it were.

This is what produces the *encarrilamiento* effect — the support roles cannot express bad behaviors because the action space is narrowed at the mask level. PPO never wastes exploration on "support emits shoot from 2.5 m."

### 11.A.5 Convergence engineering rationale

Why these specific 5 dims and not others, in convergence terms:

- **3-bit role one-hot.** Each role bit removes a large action-space region from the wrong-role agent. Expected support-role convergence speedup: **3-5× vs. a no-gate retrain.** Credit-assignment cleanliness: without the mask, a support that randomly emits `shoot` in early training mixes "bad role choice" with "bad shoot" reward — the gradient PPO receives is noisy because two failure modes are entangled. With the mask, the support's reward reflects *only* its positioning decisions. Cleaner gradient → faster convergence.

- **`open_lane_progress`.** Smooths the reward landscape. The discrete "I am open support" signal is a step function PPO has to stumble onto by chance. A continuous float that rises as the support approaches the open lane gives a learnable gradient at every step. This is the same engineering pattern as the existing `shoot_arm_progress` at index 25 — `shoot_armed` is the step function, `shoot_arm_progress` is the gradient that lets the policy *anticipate* arming.

- **`support_spacing_progress`.** Prevents the stacking failure mode (both supports collapse onto the same open lane) without requiring a new reward term in `reward.py`. The policy sees in its obs that spacing matters and develops a spacing preference implicitly through value-function learning. Adding it as a reward term would require calibrating against the existing reward magnitudes (where `good_walk ≈ 96 weighted/ep` dominates); adding it as an obs feature lets the policy weight it organically.

- **BC sampling stratification.** Recording per-row `role_id` in the dataset (Phase 5) lets `BCReplayRegularizer._sample_indices` (`train.py:1404`) stratify by `(stage, role, skill)` instead of just `(stage, skill)`. This solves the §3.2(d) class-imbalance problem: striker shoot rows and support walk rows are sampled with role-balanced weights, preventing support-walks from diluting striker-shoots.

**What we deliberately exclude and why:**

- **Full adjacency flatten (12+ floats):** too much capacity; the role one-hot is the compressed sufficient statistic. Adding the raw matrix forces the policy to learn the role assignment from scratch.
- **Ball-holder identity one-hot:** derivable from `is_striker AND ball confidence`. No new information; redundant column.
- **Open-lane target coordinates (x, y):** wrapper-internal anchor data. The policy doesn't need the geometric coordinates, only the *gradient toward "I should walk there"* — already encoded by the existing residual anchor mechanism (the teacher computes residuals relative to the anchor; the policy emits residuals relative to its inferred anchor).
- **Striker progress float:** derivable from existing `nearest_teammate_front_dist` and pose features. PPO learns it for free from the existing 26 dims; an explicit column is redundant and adds dataset-shape pressure for no convergence benefit.
- **Defensive mode / side balance:** important but v2. Adding them in v1 expands the dataset re-collection scope (Phase 5) and the calibration burden (Phase 7) without contributing to v1's offensive-coordination goal.

### 11.A.6 Recommended addition order

Do not add all 5 dims at once. Land the role one-hot first; verify; then add progress floats. This is the same "one knob at a time" discipline `FOR_TRAIN_PRODUCTION.md §6` prescribes for reward tuning.

1. **Step 1: role one-hot (3 bits at indices 26-28).** Extend `gate_skill_mask_from_obs` per §11.A.4. Run a short fine-tune (10-15k steps) from the production single-agent checkpoint with the new mask but BC β tuned per Phase 7. Validation gates:
   - Single-agent `_pass_finish` does not regress.
   - In team-scenario eval, supports' `policy_shoot_rate` drops to 0 (the mask is doing its job).
   - `role_thrash_rate ≤ 0.10`.
2. **Step 2: `open_lane_progress` (1 float at index 29).** Re-collect a small dataset addendum with this column populated (or pad the Step 1 dataset and rely on PPO to discover the gradient). Short fine-tune. Validation gate: `support_open_hold_steps_mean` rises measurably on the team rubric vs. Step 1.
3. **Step 3: `support_spacing_progress` (1 float at index 30).** Same pattern. Validation gate: stacking failures (two supports within 800 mm for ≥ 50 consecutive frames in an episode) drop to near zero.

Total fine-tune budget across the three steps: ~50k steps. **If Step 1 doesn't measurably accelerate convergence on the team rubric, Steps 2-3 are not worth adding** — the obs growth tax exceeds the convergence benefit.

### 11.A.7 Anti-patterns specific to this gate design

- **Do not encode role as an integer (0/1/2) instead of a one-hot.** Forces the policy to learn a false ordinal relationship between roles ("role 2 is more like role 1 than role 0"). Use the 3-bit one-hot; identical parameter cost.
- **Do not couple `is_open_support` to a single-frame `shot_opening_with_obstacles`.** Smooth over 3-5 frames. Role thrash is the dominant practical failure mode and a single-frame predicate guarantees thrash on noisy perception.
- **Do not make `is_striker` purely "closest to ball."** Add a possession indicator (ball confidence high AND within takeover envelope). Otherwise the striker bit flips every time the ball bounces away from the controlling robot.
- **Do not add gates for behaviors derivable from existing obs.** Striker progress is one such — don't add a redundant `striker_progress` float when the network can compute it from `nearest_teammate_front_dist` and pose features.
- **Do not let the open-support mask forbid `stand`.** A perfectly positioned support that wants to hold still should be allowed both `stand` and `walk` with zero residual. Forbidding `stand` would force the policy to emit walk+zero-residual which adds noise via the residual head's exploration variance.
- **Do not let the open-support mask allow `dribble`.** The residual anchor for open-support is the open-lane target, not the ball. Dribble residuals relative to that anchor are nonsensical. Off-ball-support is different (its residual anchor is the triangulation target, which is closer to a generic "be in this region" target that dribble can semi-sensibly extend).
- **Do not split `is_open_support` into multiple soft floats** (e.g. `lane_cleanness`, `band_position`, `angle_to_goal`). Each adds an obs column; the policy can derive these from the single `open_lane_progress` float plus existing obs.
- **Do not skip the staged addition order.** Adding all 5 dims at once and running a 60k-step fine-tune blocks the per-step convergence attribution. When something breaks you cannot tell which dim caused it.

---

## 11.B Multi-agent teacher policies, dataset, BC, and training protocol

This subsection is the design detail for §11 Phases 3, 5, 6, and 7. It defines (a) the teacher's per-role branches, (b) the env-side team-aware state that feeds the teacher, (c) the dataset's per-role contracts, (d) the BC machinery adaptations, and (e) the stage-5 training protocol. **Design thesis (the whole subsection in one sentence): centralize team-aware state in the env, keep the teacher per-agent and stateless, and let the existing Design-A invariant carry the role conditioning end-to-end.**

### 11.B.1 What changes vs. single-agent — minimal surface

The teacher's per-agent contract today:

```
input:  obs (BackendObservation), stage_id, prev_skill_name, gate_decision
output: StructuredSkillAction (skill_type + 3 residuals + pass_target)
```

The multi-agent contract:

```
input:  obs, stage_id, prev_skill_name, gate_decision, role, coordination_target
output: StructuredSkillAction
```

Two new inputs: `role` (one of `striker` / `open_support` / `off_ball_support`) and `coordination_target` (the geometric point this role should anchor against — the ball for striker, the open-lane center for open-support, the triangulation vertex for off-ball).

**Critical: the teacher does not compute `role` or `coordination_target` itself.** Both come from the env. Reasoning:

- Role assignment requires team-wide state (positions, possession, ball-distances). Computing it inside the teacher means duplicating that state per agent — three teachers, three independent role assignments, three potentially-disagreeing answers.
- `coordination_target` requires team geometry (open-lane center depends on striker position + opponents + goal). Same argument.
- Stateless teacher = teacher is purely a function of `(obs, role, target)`. Trivially testable, no synchronization problems, no risk of the teacher having a different view of the team than the env.

### 11.B.2 The three teacher branches

#### 11.B.2.1 Striker — bit-identical to today

```python
def teacher_striker(obs, stage_id, prev_skill_name, gate_decision):
    return stage_aware_teacher_action(
        obs, stage_id, prev_skill_name, gate_decision=gate_decision
    )
```

**Why unchanged:** the production single-agent striker policy is what just shipped. The BC anchor for stage-5 fine-tuning is *this* checkpoint. Changing the striker teacher means relabeling striker rows away from what the production policy already does well, which fights the KL anchor and risks regressing finishing. **Hard rule: striker labels at stage 5 are bit-identical to striker labels at stage 4.**

#### 11.B.2.2 Open-support — walk-or-stand with tolerance

```python
def teacher_open_support(obs, stage_id, prev_skill_name, gate_decision, open_lane_target):
    target_dx = open_lane_target.x - obs.robot_x
    target_dy = open_lane_target.y - obs.robot_y
    dist_to_target_mm = sqrt(target_dx**2 + target_dy**2)

    # If at target (within tolerance), hold position.
    if dist_to_target_mm < OPEN_LANE_HOLD_TOLERANCE_MM:  # e.g. 250mm
        return StructuredSkillAction(
            skill_type=SkillType.stand,
            target_x=0.0, target_y=0.0, target_theta=0.0,
        )

    # Else walk toward target, face it.
    target_theta = atan2(target_dy, target_dx)
    return _action_from_open_lane_anchor(obs, open_lane_target, target_theta, stage_id=stage_id)
```

`_action_from_open_lane_anchor` is a new helper analogous to today's `_action_from_anchor` but anchored against `build_open_lane_anchor(obs, target)` (the open-lane counterpart of `build_fallback_action_targets`).

**Gate consistency:** always emits `walk` or `stand`. Mask in §11.A.4 allows `{stand, walk}` for `is_open_support`. Design-A holds.

**Robustness:** the tolerance-based stand-vs-walk decision (`OPEN_LANE_HOLD_TOLERANCE_MM = 250mm`) eliminates the dominant failure mode of open-support — oscillating "walk-stop-walk" because the residual never quite zeroes out. The teacher decides discretely; the policy learns the discrete pattern through BC.

#### 11.B.2.3 Off-ball-support — triangulation with possession escape hatch

```python
def teacher_off_ball_support(obs, stage_id, prev_skill_name, gate_decision, triangle_target):
    # Unexpected possession: promote to striker logic for this step.
    if gate_decision.dribble_armed:
        return teacher_striker(obs, stage_id, prev_skill_name, gate_decision)

    target_dx = triangle_target.x - obs.robot_x
    target_dy = triangle_target.y - obs.robot_y
    dist_to_target_mm = sqrt(target_dx**2 + target_dy**2)

    if dist_to_target_mm < TRIANGLE_HOLD_TOLERANCE_MM:  # e.g. 350mm
        return StructuredSkillAction(skill_type=SkillType.stand)

    target_theta = atan2(target_dy, target_dx)
    return _action_from_triangle_anchor(obs, triangle_target, target_theta, stage_id=stage_id)
```

**The unexpected-possession escape hatch** is the key robustness element. If an off-ball-support takes ball control mid-frame (a deflection bouncing toward them), the gate-driven check promotes them to striker logic immediately. The env's role assigner catches up next frame; the single-frame mismatch is bounded.

**Larger tolerance (350mm vs 250mm):** off-ball positioning is less precise — the triangulation vertex is "be roughly here," not "be exactly here." Tighter tolerance produces unnecessary micro-walking.

**Gate consistency:** emits `walk`, `stand`, or `dribble` (only via the escape hatch). Mask in §11.A.4 allows `{stand, walk, dribble}`. Holds.

#### 11.B.2.4 Unified dispatch

```python
def stage_aware_team_teacher_action(obs, stage_id, prev_skill_name, *,
                                     gate_decision, role, coordination_target):
    if role == "striker":
        return teacher_striker(obs, stage_id, prev_skill_name, gate_decision)
    elif role == "open_support":
        return teacher_open_support(obs, stage_id, prev_skill_name, gate_decision, coordination_target)
    elif role == "off_ball_support":
        return teacher_off_ball_support(obs, stage_id, prev_skill_name, gate_decision, coordination_target)
    raise ValueError(f"Unknown role: {role}")
```

Three branches. No shared mutable state. Each branch is testable in isolation against hand-constructed `(obs, gate_decision, role, target)` tuples. **This is the structural reliability win** — the teacher cannot disagree with itself across agents because each agent's call is independent.

### 11.B.3 Env-side responsibilities — team-aware state centralization

#### 11.B.3.1 Role assignment with hysteresis (5 frames)

```python
def assign_roles(prev_roles, agent_obs_dict, ball_state, history):
    distances = {agent_id: distance_to_ball(obs, ball_state)
                 for agent_id, obs in agent_obs_dict.items()}
    striker_candidate = min(distances, key=distances.get)
    if not has_possession_indicators(agent_obs_dict[striker_candidate]):
        striker_candidate = None  # Ball loose; striker undefined.

    # Hysteresis: only flip striker if candidate has been closer for >= 5 consecutive frames.
    if prev_roles.striker and striker_candidate != prev_roles.striker:
        history.record_candidate(striker_candidate)
        if history.candidate_consecutive_frames(striker_candidate) < 5:
            striker_candidate = prev_roles.striker  # Keep previous.

    open_support = compute_open_support(agent_obs_dict, striker_candidate, ball_state)
    off_ball_support = the_other_one(agent_obs_dict, striker_candidate, open_support)
    return {striker_candidate: "striker",
            open_support: "open_support",
            off_ball_support: "off_ball_support"}
```

The 5-frame hysteresis is **mandatory.** Without it, BC labels are noisy because the role assigned to a sample depends on the same-frame distance calculation, which is fragile under perception noise. With it, BC sees stable role attributions and learns the post-hysteresis behavior, which is what production will see.

#### 11.B.3.2 Coordination target computation with EMA smoothing

```python
def compute_open_lane_target(open_support_pose, ball_pose, opponents, goal):
    # Ideal open-lane vertex such that:
    #   1. Line from ball_pose to target is clear of opponents.
    #   2. Line from target to goal is clear of opponents.
    #   3. Target is within production band (2.5-3.0m from goal).
    #   4. Target is on the open-support's side of the field.
    candidate = midpoint(ball_pose, goal) + perpendicular_offset(toward open_support side)
    return ema_filter(candidate, alpha=0.3)  # 3-frame EMA.

def compute_triangle_target(off_ball_pose, striker_pose, open_support_pose, ball_pose, goal):
    candidate = compute_triangle_vertex(...)
    return ema_filter(candidate, alpha=0.2)  # 5-frame EMA (slower).
```

EMA smoothing on coordination targets is **mandatory.** Without it, the target jitters on noisy ball perception and the support's walk residual oscillates; with it, the policy sees a stable target and learns to chase it smoothly. The off-ball target uses a slower EMA (α=0.2) because triangulation is naturally less critical.

#### 11.B.3.3 The new `info` payload

Each step, the env packs into `info[agent_id]`:

```python
info[agent_id] = {
    ...existing fields...,
    "raw_obs": backend_obs_dict,
    "gate_decision": gate_decision,
    "role": "striker" | "open_support" | "off_ball_support",
    "coordination_target": (x_mm, y_mm),
    "team_state": {
        "striker_id": ...,
        "open_support_id": ...,
        "off_ball_support_id": ...,
        "ball_holder_id": ... | None,
    },
}
```

The teacher reads `role`, `coordination_target` from this payload. The dataset collector records all four fields per sample.

### 11.B.4 Dataset contracts — per-role filter, weight, audit

Three changes to `teacher_production_dataset.py`. Each preserves Design-A while extending per-role coverage.

#### 11.B.4.1 Per-role success filter (replaces `_episode_kept`)

```python
def _episode_kept_v2(scenario, episode_metrics, episode_skill_counts,
                     episode_role_counts, episode_coordination_stats):
    if scenario.is_contrast:
        return True, "contrast_engaged"

    striker_steps = episode_role_counts.get("striker", 0)
    open_steps = episode_role_counts.get("open_support", 0)
    off_ball_steps = episode_role_counts.get("off_ball_support", 0)

    striker_ok = (
        striker_steps == 0
        or episode_metrics.get("clean_finish", 0.0) > 0.0
        or episode_metrics.get("shoot_finish", 0.0) > 0.0
    )
    open_ok = (
        open_steps == 0
        or episode_coordination_stats.get("open_lane_hold_steps", 0) >= 20
    )
    off_ball_ok = (
        off_ball_steps == 0
        or episode_coordination_stats.get("triangle_hold_steps", 0) >= 30
    )

    kept = striker_ok and open_ok and off_ball_ok
    return kept, "kept" if kept else "role_quality_fail"
```

**Critical fix vs. today's filter:** today's `pure_walk_stall` check would drop every episode where the support correctly held position (no dribble + no shoot = stall). The v2 filter inverts this — for support roles, "walked into position and held it for ≥ K frames" is the *success* condition. Per-role thresholds (20 for open, 30 for triangle) are calibration knobs.

#### 11.B.4.2 Per-role quality weight

```python
def _sample_quality_weight_v2(info, role, coordination_state):
    perception_weight = _sample_quality_weight(info)  # Existing.
    if perception_weight == 0.0:
        return 0.0

    if role == "striker":
        return perception_weight  # Unchanged.
    elif role == "open_support":
        if coordination_state["dist_to_open_target_mm"] < 250:
            return perception_weight  # Full weight at target.
        elif coordination_state["delta_dist_to_open_target_mm"] < 0:
            return perception_weight * 0.75  # Closing — 3/4 weight.
        else:
            return perception_weight * 0.25  # Drifting away — 1/4 weight.
    elif role == "off_ball_support":
        if coordination_state["dist_to_triangle_target_mm"] < 400:
            return perception_weight
        elif coordination_state["delta_dist_to_triangle_target_mm"] < 0:
            return perception_weight * 0.5
        else:
            return perception_weight * 0.2
```

The "going-wrong-direction" near-zero weight is critical: it tells BC that a sample where the support walked *away* from the target is poor data (the teacher should have produced a better label, but didn't, so down-weight the consequence).

#### 11.B.4.3 Extended Design-A audit

```python
def _gate_label_consistent_v2(bucket, skill_name, role):
    if not _gate_label_consistent(bucket, skill_name):
        return False  # Existing gate-bucket check.
    # Role-mask check (mirrors §11.A.4).
    if role == "striker":
        return skill_name in {"stand", "walk", "dribble", "shoot"}
    elif role == "open_support":
        return skill_name in {"stand", "walk"}
    elif role == "off_ball_support":
        return skill_name in {"stand", "walk", "dribble"}
    return False
```

The audit is now *both* gate-bucket consistent AND role-mask consistent. The dataset's `summary.json` gets a new field `positive_weight_role_mask_consistent`. **Both `positive_weight_gate_label_consistent` and `positive_weight_role_mask_consistent` must be `true`** for the dataset to be promotable.

#### 11.B.4.4 Dataset schema additions

`samples.npz` gains two new arrays:

- `role_ids[N]` (int8) — 0=striker, 1=open_support, 2=off_ball_support.
- `coordination_target_xy[N, 2]` (float32) — the target the teacher anchored against. Telemetry; not consumed by BC directly (residuals already encode anchor-relative deltas).

`OBS_SIZE = 31` per §11.A.3. Role one-hot is baked into obs by the env before recording; `role_ids` is a parallel scalar for BC stratification convenience.

### 11.B.5 BC machinery adaptations

#### 11.B.5.1 Role-stratified sampling

```python
class BCReplayRegularizer:
    def __init__(self, ...):
        ...
        # Index by (stage_id, role_id, skill_id) instead of (stage_id, skill_id).
        self.indices_by_stage_role_skill = {}
        skill_ids = np.clip(np.rint(actions[:, 0]).astype(np.int64), 0, len(SKILL_ORDER) - 1)
        for stage_id in sorted(set(stages[valid].tolist())):
            for role_id in sorted(set(role_ids[valid].tolist())):
                for skill_id in sorted(set(skill_ids[valid].tolist())):
                    idx = valid[
                        (stages[valid] == stage_id)
                        & (role_ids[valid] == role_id)
                        & (skill_ids[valid] == skill_id)
                    ]
                    if idx.size:
                        self.indices_by_stage_role_skill[(int(stage_id), int(role_id), int(skill_id))] = idx

    def _role_skill_mix(self, stage_id):
        if stage_id == 5:
            return {
                ("striker", "shoot"): 0.20,
                ("striker", "dribble"): 0.18,
                ("striker", "walk"): 0.12,
                ("open_support", "walk"): 0.16,
                ("open_support", "stand"): 0.08,
                ("off_ball_support", "walk"): 0.14,
                ("off_ball_support", "stand"): 0.08,
                ("off_ball_support", "dribble"): 0.04,
            }
        return self._skill_mix(stage_id)  # Stages ≤ 4: existing skill-only mix.
```

**The role+skill mix is the key convergence engineering element.** Without it, dataset's natural class distribution over-represents support walks and dilutes striker shoots. The explicit mix forces 20% of each batch to be striker-shoot rows, preserving the striker's BC signal even as supports get equal representation.

#### 11.B.5.2 Role-aware competence gate — take the MIN

```python
class BCCompetenceGate:
    def update(self, summary):
        # MIN across roles — β is driven by worst-role performance.
        striker_competence = float(summary.get("clean_finish_rate", 0.0))
        open_competence = float(summary.get("support_open_hold_rate", 0.0))
        off_ball_competence = float(summary.get("support_triangle_hold_rate", 0.0))
        competence = min(striker_competence, open_competence, off_ball_competence)
        ...
```

**Min, not mean.** If striker is at 0.9 and supports are at 0.1, mean = 0.37 and β decays. But supports aren't ready. Min ensures BC stays alive until *every* role is competent. Conservative; what we want.

#### 11.B.5.3 KL anchor — production checkpoint, not fresh warm-start

```python
def load_bc_warm_start_v2(models, *, production_checkpoint_path, device, active_skill_names):
    """For stage 5: warm-start from the production single-agent checkpoint.

    Critical: the frozen reference policy used by BCReplayRegularizer's KL anchor
    must be THIS production checkpoint, loaded via _load_state_dict_obs_padded
    (train.py:1206) so the new role + progress columns zero-init. The production
    checkpoint is a known-good finishing striker; the KL pull preserves finishing
    skill while PPO learns supports. A fresh zero-padded warm-start would be
    graph-blind, and KL would actively un-learn the role conditioning.
    """
    return load_bc_warm_start(models, checkpoint_path=production_checkpoint_path,
                              device=device, active_skill_names=active_skill_names)
```

This is a one-line semantic change but the **largest convergence-and-robustness lever** in the multi-agent training. The production checkpoint is the reference. Its KL pull preserves single-agent finishing as we add multi-agent shaping. Without this, the §3.3 anti-learning failure would dominate.

### 11.B.6 Curriculum and training protocol

#### 11.B.6.1 Stage 5 spec

```python
STAGE_5_SPEC = StageSpec(
    stage_id=5,
    active_skill_names=("stand", "walk", "dribble", "shoot"),  # Unchanged.
    scene_path=...,  # Multi-agent scene.
    reward_weights={
        **STAGE_4_SPEC.reward_weights,  # Inherit all existing terms.
        "support_open_hold_bonus": 0.5,    # +0.5/step when open_lane_progress >= 0.7.
        "support_triangle_hold_bonus": 0.2,
        "team_goal_bonus": 5.0,             # Distributed across RL agents on team goal.
        "pass_attempt_credit": 1.0,         # On detected handoff event.
    },
    active_scenarios=team_production_scenarios,
    interleave_with_previous_stage=0.2,  # 20% of resets fall back to stage 4 single-agent.
)
```

#### 11.B.6.2 Promotion gate

```python
def promote_to_stage_5(metrics, stage_history):
    # Single-agent must already pass _pass_finish.
    if not stage_history.get(4, {}).get("clean_finish_rate", 0.0) >= 0.60:
        return False
    if not metrics.get("policy_shoot_rate", 0.0) >= 0.02:
        return False
    return True

def promotion_complete_for_ship(metrics):
    return (
        _pass_finish(metrics)[0]
        and _pass_team_coordination(metrics)[0]
        and _pass_support_positioning(metrics)[0]
    )
```

The "must already pass _pass_finish" guard means we don't enter stage 5 with a non-finishing checkpoint — exactly the prerequisite from §4.

#### 11.B.6.3 Training protocol — `train_team_production.py`

Sibling to `train_production.py`. Key `ScriptRunConfig` differences:

```python
ScriptRunConfig(
    stage_id=5,
    max_stage_id=5,                    # Pin at stage 5; no further promotion.
    train_steps=80_000,
    bc_warm_start=True,
    bc_checkpoint_path=PRODUCTION_CHECKPOINT_PATH,  # Single-agent finisher.
    bc_replay=True,
    bc_replay_dataset_path=TEAM_DATASET_PATH,
    # Critical BC tuning:
    bc_beta_max=0.08,                  # Softer than production's 0.15.
    bc_beta_min=0.02,
    bc_competence_half_life_evals=25,  # Longer than production's 10.
    bc_replay_weight=0.10,
    bc_replay_updates=2,
    bc_skill_loss_scale=0.40,
    bc_param_loss_scale=0.50,
    parallel_player_numbers=(1, 2, 3, 4),
    # Single-agent retention.
    stage4_finish_interleave_probability=0.2,
    eval_episodes=8,
    max_episode_steps=400,
    # NEW: dual rubric evaluation.
    eval_rubrics=("single_agent", "team_coordination", "support_positioning"),
)
```

**Dual rubric evaluation is the production safeguard.** Every checkpoint runs through both the single-agent rubric (`production_scenarios`) and the team rubric (`team_production_scenarios`). Promotion to "shipped" requires both passing for 2 consecutive evals.

#### 11.B.6.4 Per-eval auto-rollback

```python
def post_eval_health_check(timestep, summary, history):
    if not _pass_finish(summary)[0]:
        if history.last_passing_checkpoint:
            print(f"[health-check] step={timestep} _pass_finish failed; "
                  f"reloading from {history.last_passing_checkpoint}")
            return ("rollback", history.last_passing_checkpoint)
    return ("continue", None)
```

The "no quiet regression" rule. The moment single-agent stops passing, we roll back and re-tune rather than letting the run continue degrading.

### 11.B.7 Convergence engineering summary

Mechanisms that produce *faster convergence*:

1. **Role mask eliminates the wrong-skill exploration tax.** Supports never emit shoot; PPO never wastes rollouts there.
2. **Per-role anchor + residual.** Teacher's labels and policy's outputs live in the same residual space per role. BC is direct supervision; no representation mismatch.
3. **Progress floats give learnable gradients.** `open_lane_progress` and `support_spacing_progress` smooth the reward landscape. PPO climbs them at every step.
4. **Role+skill BC sampling preserves the striker-shoot signal.** Without explicit mixing, support walks would dilute striker shoots; production capability would erode.
5. **Role-aware competence keeps BC alive longer.** β decays based on the worst role, not the best. Supports get full BC distillation budget until they're competent.
6. **Production-checkpoint KL anchor preserves single-agent.** KL pulls toward a known-finishing policy. Stage-5 PPO learns supports *in addition to* finishing, not *instead of*.
7. **Sequential stage 5, not in-place stage 4.** Single-agent stages are frozen; multi-agent additions live in their own stage with their own scenario set.

### 11.B.8 Robustness engineering summary

Mechanisms that produce *production robustness*:

1. **Env-side role hysteresis (5 frames).** Eliminates the dominant role-thrash failure mode. Labels stable; policy learns stable behaviors.
2. **EMA-smoothed coordination targets (3 frames open-lane, 5 frames triangulation).** Eliminates the dominant target-jitter failure mode. Residuals stable; walks smooth.
3. **Tolerance-based stand-vs-walk in the teacher.** No oscillation at target. Policy learns discrete behaviors with discrete signals.
4. **Unexpected-possession escape hatch in off-ball-support.** Deflection-to-support handled gracefully; support promotes locally, env catches up next frame.
5. **Per-role mask is structural, not learned.** Worst-case support behavior is "walked somewhere suboptimal" — recoverable. Without mask, worst case is "support shot wildly" — catastrophic.
6. **Dual rubric evaluation at every checkpoint.** Single-agent and team rubrics both checked. Promotion requires 2 consecutive passes on both.
7. **Per-eval auto-rollback.** Single-agent regression triggers immediate reload from last good checkpoint. No silent drift.
8. **Independent per-agent inference.** Each robot's policy call is independent. One robot's perception failure does not propagate.
9. **Telemetry includes role + coordination target per step.** Production failures are interpretable: read the CSV, see which role was assigned and which target was expected, attribute the failure cleanly.

### 11.B.9 Anti-patterns specific to multi-agent teacher design

- **Do not let the teacher compute its own role.** Centralize in the env. Three teachers computing roles independently risk disagreement.
- **Do not skip role-assignment hysteresis.** Single-frame role flips poison the dataset because BC sees noisy labels for the same situation.
- **Do not smooth the coordination target inside the teacher.** Smooth in the env. Teacher should be stateless; smoothing introduces hidden state.
- **Do not write per-stage role-conditioned teacher branches at stages 0-4.** Those stages are single-agent. The role branch only exists at stage 5.
- **Do not anchor BC's KL to a fresh warm-start at OBS_SIZE=31.** Anchor to the production single-agent checkpoint loaded via `_load_state_dict_obs_padded`. The production policy is a finishing striker; KL preserves that.
- **Do not change `BCCompetenceGate` to take the mean across roles.** Take the min. Otherwise BC turns off while supports are still incompetent.
- **Do not let the dataset filter accept episodes where supports never reached their targets.** Per-role hold-steps check is the dataset's only defense against teaching "support walked around uselessly."
- **Do not stratify BC sampling by (stage, role) alone, omitting skill.** You'd over-sample whichever skill dominates per role. Stratify by `(stage, role, skill)` for explicit balance.
- **Do not omit the stage-4 interleave at stage 5.** Single-agent retention is fragile; explicit interleave forces the policy to keep finishing while it learns supports.
- **Do not skip the per-eval health check.** A 5k-step degradation is easier to roll back than a 20k-step run that ships a regressed checkpoint.
- **Do not let the off-ball escape hatch promote to striker logic without the role bit changing.** The escape hatch is a one-step teacher behavior; the role assignment must catch up next step or you'll have an off-ball-support emitting striker-shaped labels for many frames in a row (poisoning the off-ball dataset).
- **Do not collect dataset rows when ball_holder_id is unstable.** During possession transitions (ball loose), no role's behavior is well-defined. Skip these frames in the dataset (or weight them to zero).

---

## 12. File:line cheat sheet

For when you're debugging and want to know "where is X."

### Wrapper (where Version B lives)
- `MultiAgentCoordinator` (location TBD from `.docs/09` — the 4v4 sprint introduces this).
- The graph logic goes into either the coordinator or a sibling `team_graph.py`.

### Evaluator (extended in Days 3-4)
- `behavioral_evaluator.py:49-94` — existing pass functions, model for the new ones.
- `behavioral_evaluator.py:98-104` — `_PASS_FNS` registry to extend.
- `behavioral_evaluator.py:107` — `_situations` binds pass functions to scenarios.

### Production curriculum (extended in Day 3)
- `production_curriculum.py:227` — `production_scenarios`, model for `team_production_scenarios`.
- `production_curriculum.py:63` — `_drive_finish_world`, model for `_team_drive_finish_world`.
- `production_curriculum.py:56` — `_support_players_far`, the parked-teammate helper. Replace with role-positioned active supports.

### Environment (extended in Day 4)
- `environment.py:1191-1349` — where production scenario state lands in `info`. Add the team metrics here.
- `environment.py:215` — production scenario sampling. The team scenarios plug in via the same `RL_PRODUCTION_SCENARIO_SET` mechanism.

### Things we do NOT touch (read-only this week)
- `teacher_policy.py`, `teacher_production_dataset.py`, `teacher_dataset.py`.
- `train.py`, `train_production.py`, `train_bc.py`.
- `curriculum.py`, `reward.py`.
- `gate.py` (mostly — one pure helper function is OK if it makes sense to colocate; otherwise put it in the wrapper).
- `config.py` (OBS_SIZE stays 26).
- The production policy `.pt` file.

---

## 13. Closing thought

The week we have is enough to ship coordination behavior that is **observably better** than the 4v4 baseline. It is not enough to ship a policy that has **learned** coordination. Those are different products and they look different in the engineering posture: one is a wrapper layer; the other is an 8-week training program.

The graph as a *learned* feature is the right destination. Version B is the responsible thing to ship in 7 days while we get there. Hard mask + telemetry; no retrain; no silent failure modes; no risk to the production single-agent checkpoint.

If you take one lesson from this document: **the 1-week budget rules out Version A.** Read §3 if you doubt this. The pipeline is structurally hostile to the graph signal in ways that are invisible until the rubric runs, and you cannot build a trustworthy rubric in 7 days while also building the dataset, the teacher, and the fine-tune. Ship Version B, document it honestly, and put Version A in the backlog with §11 as the starting plan.
