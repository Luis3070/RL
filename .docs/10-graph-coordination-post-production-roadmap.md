# Graph Coordination — 1-Week Post-Production Sprint

> ### 2026-06-08 STATUS UPDATE — read §14 first
>
> This doc was written when (a) the obs was still 26-dim, (b) the stabilization repair had not landed, (c) the project still implicitly accepted BHuman tactical logic as a production fallback. **All three premises are now false.**
>
> - **Substrate shipped (Laura + Rafael, 2026-06-05 → 06-08):** 41-dim obs, MAPPO wire + tests, behavior-preservation contract, fallen-stuck termination, upright-streak reset, 4/5-stabilized checkpoint, ONNX export. Phase 1 of §11 (obs growth + warm-start) is done at 41 dims rather than 31.
> - **Goal locked (2026-06-08):** RL-dominant production. The deployed brain on each NAO is the RL ONNX. BHuman C++ tactical logic (`StrategyStatus.role`, `Tactic`, pass, support positioning) is a *teacher / heuristic predecessor* for our pipeline, NOT a production runtime. Wrappers are training-time scaffolding only — they ship nothing.
> - **Version B is dead for this sprint.** Wrapper-only coordination violates the locked goal. Version A design specs in §11.A and §11.B remain canonical; we execute a compressed subset of them as a 6-day sprint.
> - **Real-NAO demo: 2026-06-15 (next Monday).** Six calendar days. See §14 for the integrated build plan, BHuman salvage map, and per-day owner table.
>
> §0–§13 below are kept for historical reference and for the design specs in §11.A / §11.B which remain accurate. **For the current plan, jump to §14.**

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

Do not add all 5 dims at once. Land the role one-hot first; verify; then add progress floats. This is the same "one knob at a time" discipline `FOR_TRAIN_PRODUCTION.md §6` [removed from this branch; available on rl-v4.2-multiagent-gk] prescribes for reward tuning.

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

---

## 14. 2026-06-08 UPDATE — Substrate shipped, RL-dominant goal locked, integrated 6-day multi-agent sprint

This section **supersedes §6 (Version B day-by-day) for the current sprint** and **operationalizes a compressed subset of §11 (Version A)** against the substrate Laura and Rafael shipped between original drafting and 2026-06-08. The structural analysis in §3 and the design specs in §11.A and §11.B remain canonical — read them. This section says which subset of those specs we execute in 6 calendar days and who owns each piece.

### 14.1 Substrate state — what shipped between original doc and now

Eight commits materially changed the picture; the §11 Phase 1 work is partly done.

- `22a4631` (Laura) — Obs space extended 26 → 41 dims. Block B teammate slot positions (26-31), Block C team-ball x/y/age (32-34), Block D engagement bits (35-37, hardcoded to zero at this commit), Block E consensus bit (38), Block F opponent goal x/y (39-40). Backend wire and `_encode_obs` updated. Zero-pad warm-start preserved.
- `a4424fe` (Laura) — Behavioral evaluator adapted to 41-dim shape.
- `91f3885` (Rafael) — PPO stability repair: fallen-stuck early termination (stage ≥ 3, 24 consecutive frames in `_update_fallen_stuck_counter`) + 3-frame upright reset streak in pybh_backend. New `behavioral_stability_probe.py` with named seed set `11,23,37,51,67,83`. Docs 26 (transplant implementation, 890 lines) + 27 (stability evaluator rationale).
- `9a97634` (Rafael) — Four production-stabilized checkpoints in `runs/production_stability_repair_v1/`. The 5000-step one is the operative baseline.
- `a733c39` (Rafael) — MAPPO observation wire probes + tests. `live_multiagent_obs_probe.py` [removed from this branch; available on rl-v4.2-multiagent-gk] (470 LOC), `test_mappo_observation_space.py` (175 LOC), `test_obs_migration_contract.py` (zero-pad bit-identity for policy + value). Engagement-bit logic implemented: D = kicking ∨ dribbling ∨ (goingToBall ∧ within 700mm of team ball). Doc 25 (MAPPO wire findings — read it for Block C age-proxy and Block E noise modes).
- `b2f62ae` (Laura) — ONNX exported from `checkpoint_stage4_step5000.pt`. `tools/export_ppo_policy.py` infrastructure.
- BHuman side (`SkillBehaviorControl.cpp:61-85`) — adds `encodeTeammateActivity()` proxy and `RLTeammateActivity` enum (idle=0 / goingToBall=1 / dribbling=2 / kicking=3). This is the derived proxy doc 25 § 1 explains BHuman has no native `BehaviorStatus.activity` for.

Mapping to §11 phases:
- **Phase 1 (obs growth, zero-pad warm-start, mask test) — DONE at 41 dims** (originally specified at 31). Extending to 47 dims for the role one-hot reuses the same machinery.
- **Phase 2 (multi-agent evaluator) — partial.** `behavioral_evaluator.py` accepts the 41-dim obs; team metrics in `episode_metrics` are not yet added.
- **Phases 3, 4, 5, 6, 7 — not started.**

### 14.2 Goal locked — RL-dominant production

Two clarifications from the 2026-06-08 conversation:

1. **The deployed brain on each NAO is the RL ONNX policy.** BHuman C++ tactical/strategic logic (`StrategyStatus.role`, `Tactic`, pass, support positioning) is a **teacher / heuristic predecessor** for our pipeline, NOT the production decision-maker. Multi-agent coordination is emergent from same ONNX on every NAO + the 41-dim obs blocks B-F.
2. **Wrappers (`MultiAgentCoordinator`) are training-time scaffolding only.** They ship nothing. Production is `pure ONNX + BHuman observation wire`.

Consequence: **Version B (wrapper-only inference layer) is dead for this sprint.** Coordination must live in the policy. The current sprint executes a compressed subset of Version A.

**One exception:** GK stays BHuman C++ for this sprint (time/resource constraint, not philosophy). All other roles emerge from the policy.

### 14.3 BHuman salvage map — what to port for the teacher / gate / dataset / mask

Audit of BHuman tactical logic against the actual C++ surface, with destination in our pipeline.

| BHuman component | What it does (BHuman semantics) | Where it lands in our pipeline |
|---|---|---|
| `PlayBall::smashOrPass()` (`ActiveRoles/PlayBall.cpp:33-118`) | Shoot-vs-pass decision with `decisionPenalty` hysteresis (lines 38-39, 53-55, 94-98) | `teacher_striker` shoot/pass branch — extends §11.B.2.1. The `decisionPenalty` pattern unifies R2+R4+R6 of the lower-scope teacher discussion. |
| `Forward::rating(pos)` (`PositionRoles/Forward.cpp:24-38`) | Receive-position rating `passEval × goal × base_gauss` | `teacher_open_support` open-lane target — extends §11.B.2.2 |
| `Midfielder::rating` separation Gaussian (`PositionRoles/Midfielder.cpp:60-72`) | `1 - exp(-dist²/σ²)` keep-distance from teammates | `teacher_off_ball_support` triangulation + spacing reward — extends §11.B.2.3 |
| `Defender::position(side, ...)` (`PositionRoles/Defender.cpp:25-49`) | Between own goal and ball, side-aware, GK-aware | Defender role for 4v4 (DEFERRED to Stage 7+ — out of sprint scope) |
| `Tactic` enum + position templates (`Tools/.../Tactic.h:17-133`) | Named formation templates (`t13_3v3_scaled`, `t22_3v3_scaled`, `t121_4v4_scaled`) with `Pose2f` poses and `stabilityOffset` | Support slot templates (Python-hardcoded, not read at runtime) + role-position hysteresis pattern |
| `ActiveRole::Type` enum (`Tools/.../ActiveRole.h:17-23`) | playBall / freeKickWall / closestToTeammatesBall / startSetPlay | Role taxonomy reference; we use the 3-role taxonomy from §11.A.3 |
| `BallSearch::decidePositionForSearch` (`BallSearch.cpp`) | Voronoi-tiled search per agent when team ball is lost | `observe_armed` gate bit (DEFERRED to Stage 8+ — out of sprint scope) |
| `StrategyStatus` (`Representations/.../StrategyStatus.h`) | Per-robot broadcast: proposed/accepted tactic, set play, position, role | Reference for what BHuman publishes; we **do NOT add to obs wire** — policy infers role from blocks B/C/D/E |
| `encodeTeammateActivity()` (`SkillBehaviorControl.cpp:61-85`) | Activity proxy (idle/goingToBall/dribbling/kicking) | Already in Block D via `teammate_activity` — see doc 25 § 1 |
| `TeammatesBallModelProvider` thresholds (`inactivityInvalidationTimeSpan=1000`, `minSendInterval=1000`) | Validity gating for team ball | Already informs `TEAMMATE_STALE_THRESHOLD_MS = 1000.0` in `environment.py` |

**Cheap proxies for BHuman's learned estimators.** We do NOT port `PassEvaluation` or `ExpectedGoals`; we use proxies built from the 41-dim obs.

| BHuman learned component | Cheap proxy | Source in 41-dim obs |
|---|---|---|
| `expectedGoals.getRating(ballPos)` | `shot_opening_with_obstacles + 0.3 * can_score_now` | dims 15, 13 |
| `expectedGoals.getRating(teammatePos)` | `linear_goal_proximity = clip(teammate_x_rel / goal_x_rel, 0, 1)` | dims 26/28/30 + 39 |
| `passEvaluation.getRating(line)` | binary `passDist ∈ [400, 4000] AND nearest_opponent_front_dist > 800` | dims 21 + Block B |
| `Tactic::Position::stabilityOffset` | `last_pass_target` lifetime decay (1.5 s) + 5-frame role hysteresis (§11.B.3.1) | new GateState fields |

### 14.4 Encarrilamiento subset we ship — Step 1 of §11.A.6 only

From the staged addition order in §11.A.6:

- **STEP 1 (THIS SPRINT): role one-hot, 3 bits.** Indices 44/45/46 (extending the shipped 41-dim obs, not 26→31 as the doc originally specified). `is_striker`, `is_open_support`, `is_off_ball_support`. Plus the extended mask per §11.A.4: open_support forbids shoot+dribble; off_ball_support forbids shoot.
- **STEP 2 (DEFER post-Monday): `open_lane_progress`** at index 47.
- **STEP 3 (DEFER post-Monday): `support_spacing_progress`** at index 48.

§11.A.6's own final paragraph says: *"If Step 1 doesn't measurably accelerate convergence on the team rubric, Steps 2-3 are not worth adding."* That gate cannot be evaluated in 6 days. Therefore Steps 2-3 are deferred and we do not pay their dataset-rebuild cost.

The 3-5× convergence speedup §11.A.5 attributes to the role one-hot is the load-bearing element for our 1-2 day fine-tune budget.

We also extend the gate with **3 skill-arm bits** (parallel to existing `shoot_armed`/`dribble_armed`/`shoot_arm_progress` at indices 23/24/25):
- `pass_armed` (smashOrPass picked "pass" AND decisionPenalty-stable)
- `observe_armed` (ball lost beyond threshold AND a teammate has team-ball belief)
- `pass_arm_progress` (debounce fill toward arming)

These three sit at indices 41/42/43. Role one-hot at 44/45/46. **Final obs growth this sprint: 41 → 47.**

The behavior-preservation contract in `test_obs_migration_contract.py` extends with a 26→47 zero-pad bit-identity assertion. Single-agent floor cannot regress structurally.

### 14.5 Operative §11.B retunings for this sprint

Copying from §11.B.5 and §11.B.6.3, the values that apply unchanged:

- `bc_beta_max = 0.08` (softer than production's 0.15)
- `bc_beta_min = 0.02`
- `bc_competence_half_life_evals = 25` (longer than production's 10)
- `bc_replay_weight = 0.10`
- `bc_replay_updates = 2`
- `bc_skill_loss_scale = 0.40`
- `bc_param_loss_scale = 0.50`
- `parallel_player_numbers = (1, 2, 3)` for stage 6 (2v0 + BH GK = 3 total), `(1, 2, 3, 4)` for stage 7 (3v0 + BH GK = 4 total)
- `stage4_finish_interleave_probability = 0.2`
- `eval_episodes = 8`
- `max_episode_steps = 400`
- BC competence gate takes **min** across (striker, open_support, off_ball_support) per §11.B.5.2
- KL anchor = `runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt` loaded via `_load_state_dict_obs_padded`, per §11.B.5.3 (this is the production checkpoint §11.B.5.3 referenced as a placeholder)
- Per-eval auto-rollback per §11.B.6.4 (any single-agent regression → reload last passing checkpoint)
- Dual rubric evaluation per §11.B.6.3 (single-agent + team)

### 14.6 6-day sprint plan with owners

Today is **Monday 2026-06-08 near midnight**. Real-NAO demo is **Monday 2026-06-15**. Tue–Sun = 6 working days.

Owners: **Luis** = multi-agent pipeline (teacher, gate, env, mask, dataset, BC, PPO). **Rafael** = stability + transplant + real-NAO validation. **Laura** = dataset/eval support + ONNX (already shipped baseline).

| Day | Owner | Build | Validation gate | Reference |
|---|---|---|---|---|
| **Tue 06-09** | Luis | `multiagent_teacher.py` with three role branches (BHuman salvage per §14.3 + §11.B.2). Extend `gate.py`: `pass_armed`, `observe_armed`, `pass_arm_progress`, `is_striker`, `is_open_support`, `is_off_ball_support`. Extend obs to 47 dims. Extend `gate_skill_mask_from_obs` per §11.A.4. | `test_obs_migration_contract` passes at 47 dims (26→47 bit-identity). Teacher labels for a hand-constructed 3-agent obs match expectations. Single-agent behavioral_evaluator unchanged (regression check). | §11.A.3, §11.A.4, §11.B.2, §14.3, §14.4 |
| **Tue 06-09** | Rafael | Verify the existing ONNX runs cleanly inside BHuman on a real NAO: model loads, inference < 30 ms, no crash, walk/dribble/shoot skills callable. | Latency < 50 ms ceiling. No crash over 10-minute run. | doc 23, doc 26 |
| **Wed 06-10** | Luis | Env-side role assignment with 5-frame hysteresis + EMA-smoothed coordination targets (α=0.3 open-lane, α=0.2 triangle) per §11.B.3. Add stages 6 (2v0 + BH GK) and 7 (3v0 + BH GK) to curriculum. Multi-agent dataset gen with per-row `role_id` for BC stratification per §11.A.5. Sanitizer filters extended (per-role hold-steps; role-thrash drop; teammate-teammate-collision drop). | Dataset audit `positive_weight_gate_label_consistent: true` AND new `positive_weight_role_mask_consistent: true`. ≥ 100 episodes per stage. | §11.B.3, §11.B.4 |
| **Wed 06-10** | Rafael | Single real NAO with RL striker drives the ball in 1v0 (hardware reproduction of single-agent behavior). | Striker completes ≥ 3 of 5 single-agent behavioral_evaluator scenarios on hardware. | doc 27 |
| **Thu 06-11** | Luis | BC train from new dataset, warm-start from `checkpoint_stage4_step5000.pt` via zero-pad (26→47). Role-stratified BC sampling per §11.A.5. Reward terms extended (conservative) per §11.B.6.1: `support_open_hold_bonus = 0.5`, `support_triangle_hold_bonus = 0.2`, `team_goal_bonus = 5.0`, `pass_attempt_credit = 1.0`. KL anchor wired per §11.B.5.3. | Single-agent behavioral_evaluator floor ≥ 3/5 on new BC checkpoint. BC loss decreases monotonically across all three roles. | §11.B.5, §11.B.6.1 |
| **Thu 06-11** | Laura | Multi-agent evaluator scaffolding: add `team_goal_rate`, `role_thrash_rate`, `support_open_hold_steps_mean`, `support_triangle_hold_steps_mean` to `environment.py episode_metrics`. Pass functions: `_pass_team_coordination`, `_pass_support_positioning`, `_pass_team_no_regression` per §11.A.5 + §11 Phase 2. Register team scenarios in `_PASS_FNS`. | Single-agent rubric still passes against `checkpoint_stage4_step5000.pt`. Team rubric baseline fails in measurable way (no NaN, sensible ranges). | §11.A.5, §11 Phase 2 |
| **Fri 06-12** | Luis | PPO fine-tune stage 6 with §14.5 retunings + per-eval auto-rollback per §11.B.6.4. If stage 6 promotes, run stage 7. | Stage 6 promotion gate (clean_finish ≥ 0.6 AND policy_shoot_rate ≥ 0.02 AND team_goal_rate > 0) OR explicit decision to stay on 6. Single-agent never regresses. | §11.B.6.2, §11.B.6.4 |
| **Sat 06-13** | Luis | Multi-agent stability probe (Rafael seed set, team metrics). ONNX export from best checkpoint. SimRobot 3D demos at 2v0 and 3v0. | Stability: ≥ 3/5 single-agent, ≥ 1 team goal in 4-episode 2v0 probe, no teammate-teammate collisions. ONNX runs in BHuman. | doc 27 |
| **Sat 06-13** | Laura | Behavioral stability probe extension for multi-agent (sibling `behavioral_stability_multiagent_probe.py`, same seed contract). | Per-seed multi-agent rubric reports pass_rate without NaN. | doc 27 |
| **Sun 06-14** | All | Buffer. Dry-run real-NAO demo end-to-end with Rafael's deployment. Address whatever surfaces. | Demo works on hardware. | — |
| **Mon 06-15** | All | Real-NAO demo. | — | — |

### 14.7 Anti-patterns updated for this sprint

In addition to §9 (still applies — read it):

- **Do not add `open_lane_progress` or `support_spacing_progress` this sprint.** §11.A.6 requires Step-1 verification first; we cannot verify in 6 days. Defer.
- **Do not extend obs beyond 47 dims this sprint.** Any feature beyond role one-hot adds dataset-shape pressure we cannot validate.
- **Do not skip the per-eval auto-rollback** (§11.B.6.4). Multi-agent fine-tunes regress quietly; the rollback is the only defense.
- **Do not write Python wrapper coordination logic.** Violates the RL-dominant goal locked in §14.2. Coordination lives in the policy via role one-hot + obs blocks B-F.
- **Do not unmask `mark` or `block`.** Both require opponent positions in obs (intentionally excluded per spec § 5 of doc 24).
- **Do not change the GK to RL** this sprint. Confirmed scope.
- **Do not modify `checkpoint_stage4_step5000.pt`.** It is the KL anchor + fallback. Treat as immutable.
- **Do not skip role-assignment hysteresis (5 frames)**, even "temporarily for debugging." Single-frame role flips poison the dataset. §11.B.3.1.
- **Do not smooth coordination target inside the teacher.** Smooth in the env. Stateless teacher. §11.B.9.

### 14.8 What this sprint does NOT accomplish

Honest scoping, extending §10:

- Does not learn `mark`/`block`/defensive recovery. Opponents not in obs.
- Does not learn complex set plays. Game state PLAYING only.
- Does not include Steps 2-3 of §11.A.6 (progress floats). Deferred.
- Does not finish §11 as scoped (Phase 5 dataset re-collection at OBS_SIZE=31, Phase 7 full 80k fine-tune). We do a compressed 10-20k-step fine-tune at OBS_SIZE=47 from a stabilized checkpoint.
- Does not replace BHuman GK.
- Does not deliver pass-receive learning beyond the smashOrPass teacher label distribution. The receiver behavior is "be in the open-lane slot when team ball is mine" via the open-support mask.

Post-Monday natural follow-ups: §11.A.6 Steps 2-3, full §11 Phase 7 fine-tune at the proper step budget, §11.A.2 policy expansion (defensive recovery, pass-lane anticipation), RL GK as separate policy.

### 14.9 Related artifacts (canonical files to read before starting)

- `runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt` — KL anchor for stage-5+ fine-tunes. The 4/5 stabilized single-agent baseline. Immutable this sprint.
- `ppo_striker_hsl2026.onnx` + `.onnx.data` — ONNX export of the above. Real-NAO deployment artifact.
- `tools/export_ppo_policy.py` — ONNX export infrastructure (re-export after sprint).
- `test_obs_migration_contract.py` — extend with 26→47 assertion. Behavior-preservation contract.
- `test_mappo_observation_space.py` — extend with role-mask assertions and 47-dim shape.
- `behavioral_stability_probe.py` — sibling `behavioral_stability_multiagent_probe.py` for team metrics.
- `docs/24-multiagent-obs-extension-26-to-41-spec.md` — Block A-F spec, canonical for the 41-dim obs.
- `docs/25-mappo-wire-findings.md` — MAPPO wire reliability notes (Block D activity proxy, Block C age proxy, Block E noise modes).
- `docs/26-implementacion-ppo-a-transplantar.md` — Rafael's 890-line transplant implementation guide.
- `docs/27-estabilidad-behavioral-evaluator.md` — stability evaluator rationale and seed contract.
- `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp` — smashOrPass reference for the striker teacher branch.
- `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/PositionRoles/{Forward,Midfielder,Defender}.cpp` — rating-function references.
- `SabanaHerons2026/Src/Tools/BehaviorControl/Strategy/Tactic.h` — formation templates (`t13_3v3_scaled`, `t22_3v3_scaled`).
- `SabanaHerons2026/Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.cpp:61-85` — the activity proxy already in our wire.

### 14.10 Memory pointer

The Claude memory at `~/.claude/projects/-home-luis-GitHub-RL/memory/multiagent-rl-dominant-production-goal.md` captures the RL-dominant goal locked in §14.2 as durable cross-session memory. The memory at `graph-coordination-pipeline-hostility.md` predates the substrate work and is **stale on timeline** (it estimated 3-4 weeks for the graph approach; with the substrate shipped, the encarrilamiento Step-1 subset fits 6 days). The memory's structural analysis remains accurate — only the budget changed.

### 14.11 First commit of the sprint

The Tuesday morning starting point is one file: `multiagent_teacher.py`. Its imports include `stage_aware_teacher_action` from the existing `teacher_policy.py` (for the striker branch, bit-identical per §11.B.2.1), the new role enums, and a port of the `smashOrPass` decisionPenalty pattern from `PlayBall.cpp`. The gate extension lands in the same commit so the obs encoding stays consistent. The behavior-preservation test extension lands in the same commit so single-agent floor is guaranteed before any multi-agent code paths execute.

---

### 14.12 The five R-rules — fully specified

The heuristic teacher is built from five rules. R1-R3 are positive ("what to do"); R4 is hysteresis (applied inside R1, R2, R3); R6 is pass quality scoring (embedded inside R2 — there is no separate R6 module, it's the scoring function R2 calls). **R5 from the earlier discussion was dropped — teammate-as-obstacle is already handled by Block A `nearest_teammate_front_dist` + BHuman's walk/dribble trajectory planner; it does not need a teacher rule.**

All five rules together fit in ~400 lines of Python in `multiagent_teacher.py` + ~80 lines of env-side state in `environment.py`. Constants below are first-pass; calibrate after Day 5.

#### 14.12.1 Constants (single source of truth)

```python
# multiagent_teacher.py — top-of-module constants. Tune on Day 5.

# R1 (Role assignment)
ROLE_HYSTERESIS_FRAMES = 5             # §11.B.3.1 — 5 consecutive frames before swap
ROLE_MIN_DISTANCE_GAP_MM = 200.0       # candidate must be ≥ 200mm closer to ball
TEAMMATE_FRESH_MAX_AGE_MS = 1000.0     # same as TEAMMATE_STALE_THRESHOLD_MS in env

# R2 (smashOrPass, port of PlayBall.cpp:33-118)
SHOOT_THRESHOLD = 0.55                 # own_value above this → shoot
MIN_PASS_DISTANCE_MM = 400.0           # PlayBall.cpp:p.minPassDistance
MAX_PASS_DISTANCE_MM = 4000.0          # PlayBall.cpp:p.maxPassDistance
DECISION_PENALTY_MIN = 0.05            # mapToRange floor
DECISION_PENALTY_MAX = 0.20            # mapToRange ceiling
BALL_DIST_MIN_PENALTY_MM = 500.0       # tighter ball → larger penalty
BALL_DIST_MAX_PENALTY_MM = 2500.0
LAST_PASS_TARGET_LIFETIME_MS = 1500.0  # R4 decay window

# R3 (Support positioning)
SIGMA_TEAM_MM = 800.0                  # Midfielder separation Gaussian σ
STAB_OFFSET_MM = 800.0                 # Tactic.Position.stabilityOffset analogue
HOLD_TOLERANCE_OPEN_MM = 250.0         # §11.B.2.2 OPEN_LANE_HOLD_TOLERANCE_MM
HOLD_TOLERANCE_OFF_BALL_MM = 350.0     # §11.B.2.3 TRIANGLE_HOLD_TOLERANCE_MM
SUPPORT_SLOT_CANDIDATE_GRID = (        # field-frame xy candidates per side, mirrored by ball_y_sign
    (+800.0, +1200.0),
    (+1200.0, +800.0),
    (+400.0, +1400.0),
    (-400.0, +1200.0),                 # fall-back-deep slot
)

# R6 (Pass quality — proxies for BHuman's learned estimators)
SHOOT_OPEN_THRESHOLD_FOR_LANE = 0.5    # for own ball shoot rating
CAN_SCORE_BOOST = 0.3                  # own ball: + 0.3 × can_score_now
OPPONENT_LANE_CLEARANCE_MM = 800.0     # pass lane: nearest_opponent_front_dist > 800mm

# Env-side, but listed here for context
EMA_ALPHA_OPEN_LANE = 0.3              # 3-frame EMA on open-lane target
EMA_ALPHA_TRIANGLE = 0.2               # 5-frame EMA on triangle target
GK_EXCLUDE_X_MM = -2000.0              # teammate with x_field < this is treated as GK
```

#### 14.12.2 R1 — Role assignment with 5-frame hysteresis

```python
@dataclass
class TeamRoleHistory:
    """Lives on the env, one per episode. Carries the hysteresis state."""
    current_roles: dict[int, str]               # {agent_id: "striker"|"open_support"|"off_ball_support"}
    candidate_streak: dict[int, int]            # {candidate_id: consecutive frames as proposed striker}

def assign_roles(
    agent_obs: dict[int, BackendObservation],   # only RL agents (GK excluded by caller)
    team_ball: tuple[float, float, bool],       # (x_mm, y_mm, valid)
    history: TeamRoleHistory,
) -> dict[int, str]:
    """
    Returns {agent_id: role}. Deterministic. Side-effects history in-place.

    Rules:
      1. Distance to ball = sqrt((robot_x - ball_x)^2 + (robot_y - ball_y)^2).
         Ball coords: prefer obs.global_team_ball if team_ball[2] else (robot_x + ball_rel_x, robot_y + ball_rel_y) of the closest-claiming agent.
      2. Filter agents with valid pose (broadcast age <= TEAMMATE_FRESH_MAX_AGE_MS — for the self agent always true).
      3. Lowest distance = striker_candidate; tie-break by lowest agent_id (deterministic).
      4. Hysteresis: if history.current_roles[agent].striker == prev_striker AND candidate != prev_striker:
           if (prev.distance - candidate.distance) < ROLE_MIN_DISTANCE_GAP_MM: keep prev as striker.
           else increment candidate_streak[candidate]; if < ROLE_HYSTERESIS_FRAMES: keep prev.
           else: commit swap, zero all streaks.
      5. open_support = the_other_RL with highest forward_rating(pos, ball) — see R3.
         For 2 RL (after striker chosen, exactly one remains) it is that one.
      6. off_ball_support = the remaining RL (in 3v0 only).
      7. Update history.current_roles, return mapping.
    """
```

**Edge cases:**
- 2-RL (stage 6): exactly one striker + one open_support. No off_ball.
- 3-RL (stage 7): one striker + one open_support + one off_ball.
- All RL agents in "ball lost" state (team_ball invalid AND each agent's own ball is stale): the agent with the freshest natural ball belief is striker; others off_ball_support.

#### 14.12.3 R2 — smashOrPass (port of PlayBall.cpp:33-118)

```python
def smash_or_pass(
    striker_obs: BackendObservation,
    teammate_slots: list[TeammateSlot],          # sorted by Block B order: [(slot_idx, agent_id, x_field, y_field, is_gk)]
    last_pass_target: int | None,                # None or slot_idx of last frame's pass
    last_pass_target_age_ms: float,              # 0 if no last pass; decays naturally
) -> tuple[str, int]:
    """
    Returns (skill_name, pass_target_slot). pass_target_slot = -1 if skill != "pass".

    Port of PlayBall::smashOrPass(). Direct line-by-line correspondence:
      PlayBall.cpp:39  → decision_penalty
      PlayBall.cpp:44  → own_value
      PlayBall.cpp:54-55 → own_value -= decision_penalty when last_pass_target was active
      PlayBall.cpp:56-60 → if own_value > shoot_threshold: return ("shoot", -1)
      PlayBall.cpp:64-104 → loop over teammates, compute agent_value, pick best
      PlayBall.cpp:105-108 → if best: return ("pass", best.slot)
      PlayBall.cpp:116-117 → fallback: return ("shoot", -1)
    """
    # 1. Decision penalty ramps with ball distance — close ball, big penalty (don't thrash).
    d_ball = math.hypot(striker_obs.ball_rel_x, striker_obs.ball_rel_y)
    decision_penalty = map_to_range(
        d_ball, BALL_DIST_MAX_PENALTY_MM, BALL_DIST_MIN_PENALTY_MM,
        DECISION_PENALTY_MAX, DECISION_PENALTY_MIN,
    )

    # 2. Own shoot rating from obs proxies (R6).
    own_value = float(striker_obs.shot_opening_with_obstacles) + CAN_SCORE_BOOST * float(striker_obs.can_score_now)
    if last_pass_target is not None and last_pass_target_age_ms < LAST_PASS_TARGET_LIFETIME_MS:
        own_value -= decision_penalty

    # 3. Direct shoot if own rating high enough — PlayBall.cpp:56-60.
    if own_value > SHOOT_THRESHOLD:
        return ("shoot", -1)

    # 4. Score each teammate as pass candidate — PlayBall.cpp:64-104.
    best_slot = -1
    best_value = own_value
    for slot in teammate_slots:
        if slot.is_gk:                                   # PlayBall.cpp:66-67 (skip GK)
            continue
        pass_dist = math.hypot(
            slot.x_field - striker_obs.robot_x,
            slot.y_field - striker_obs.robot_y,
        )
        if pass_dist < MIN_PASS_DISTANCE_MM or pass_dist > MAX_PASS_DISTANCE_MM:
            continue                                     # PlayBall.cpp:70-72

        # R6 pass quality (proxies replace BHuman's learned PassEvaluation/ExpectedGoals).
        pass_reachable = pass_reachable_proxy(striker_obs, slot)
        goal_proximity = goal_proximity_proxy(slot)
        if pass_reachable == 0.0 or goal_proximity == 0.0:
            continue

        agent_value = pass_reachable * goal_proximity
        if last_pass_target is not None and last_pass_target != slot.slot_idx \
                and last_pass_target_age_ms < LAST_PASS_TARGET_LIFETIME_MS:
            agent_value -= decision_penalty              # PlayBall.cpp:94-98

        if agent_value > best_value:
            best_value = agent_value
            best_slot = slot.slot_idx

    # 5. Commit decision — PlayBall.cpp:105-117.
    if best_slot >= 0:
        return ("pass", best_slot)
    return ("shoot", -1)


def pass_reachable_proxy(striker_obs: BackendObservation, target_slot: TeammateSlot) -> float:
    """R6 proxy for PassEvaluation.getRating(line). Binary 0/1."""
    pass_dist = math.hypot(
        target_slot.x_field - striker_obs.robot_x,
        target_slot.y_field - striker_obs.robot_y,
    )
    if pass_dist < MIN_PASS_DISTANCE_MM or pass_dist > MAX_PASS_DISTANCE_MM:
        return 0.0
    # Opponent-on-lane proxy: nearest opponent front distance from striker.
    # We lack per-line opponent geometry, so we use the front-cone proxy.
    if float(striker_obs.nearest_opponent_front_dist) <= OPPONENT_LANE_CLEARANCE_MM:
        return 0.0
    return 1.0


def goal_proximity_proxy(target_slot: TeammateSlot) -> float:
    """R6 proxy for ExpectedGoals.getRating(teammatePos). Linear in x toward opponent goal."""
    # Goal at +FIELD_X (e.g. +4500mm). Teammate closer to goal → higher rating.
    return float(np.clip(target_slot.x_field / GOAL_X, 0.0, 1.0))
```

#### 14.12.4 R3 — Support positioning (Tactic templates + Midfielder separation)

```python
def support_target(
    support_obs: BackendObservation,
    ball_field_xy: tuple[float, float],
    striker_field_xy: tuple[float, float],
    other_support_field_xy: tuple[float, float] | None,
    current_slot: tuple[float, float] | None,       # last frame's committed slot, for stabilityOffset
) -> tuple[tuple[float, float], str]:
    """
    Returns (target_xy_field, skill_name) where skill_name ∈ {"walk", "stand"}.

    Rule:
      1. Build candidate slot positions from SUPPORT_SLOT_CANDIDATE_GRID, mirroring y by ball y sign.
      2. Score each candidate: forward_rating(slot, ball) × separation_penalty(slot, teammates).
      3. Apply stabilityOffset: if current_slot has score within STAB_OFFSET_MM of best, keep current.
      4. Dist to chosen slot:
           < HOLD_TOLERANCE → ("stand")
           else → ("walk")
    """
    # Mirror grid by ball y sign so supports are on the same side as ball.
    ball_y_sign = 1.0 if ball_field_xy[1] >= 0.0 else -1.0
    candidates = [(x, y * ball_y_sign) for (x, y) in SUPPORT_SLOT_CANDIDATE_GRID]

    scored = []
    for cand in candidates:
        fwd = forward_rating(cand, ball_field_xy)
        teammates_for_sep = [striker_field_xy]
        if other_support_field_xy is not None:
            teammates_for_sep.append(other_support_field_xy)
        sep = separation_penalty(cand, teammates_for_sep)
        score = fwd * sep
        scored.append((score, cand))

    scored.sort(reverse=True)
    best_score, best_cand = scored[0]

    # stabilityOffset analogue: prefer the current slot if it's within STAB_OFFSET of best.
    if current_slot is not None:
        cur_score = next((s for (s, c) in scored if c == current_slot), 0.0)
        if best_score - cur_score < (STAB_OFFSET_MM / 1000.0):   # score units, scale matches
            best_cand = current_slot

    # Tolerance: stand if close enough; else walk.
    self_x = float(support_obs.robot_x)
    self_y = float(support_obs.robot_y)
    dist = math.hypot(best_cand[0] - self_x, best_cand[1] - self_y)
    skill = "stand" if dist < HOLD_TOLERANCE_OPEN_MM else "walk"
    return (best_cand, skill)


def forward_rating(pos: tuple[float, float], ball: tuple[float, float]) -> float:
    """
    Port of Forward::rating(pos) (PositionRoles/Forward.cpp:24-38).

    Original:
        passRating * goalRating * baseRating
    Our proxy:
        pass_reachable_from_ball × goal_proximity × base_gauss(pos − ideal_receive_pose)

    For sprint simplicity, baseRating is computed from a fixed "ideal_receive_pose" for the
    slot, OR replaced by 1.0 if the candidate grid is already at the right poses. Use:
    """
    # Pass-reachable from the ball position.
    pass_dist = math.hypot(pos[0] - ball[0], pos[1] - ball[1])
    if pass_dist < MIN_PASS_DISTANCE_MM or pass_dist > MAX_PASS_DISTANCE_MM:
        return 0.0
    # Goal proximity: closer to opponent goal is better.
    goal = float(np.clip(pos[0] / GOAL_X, 0.0, 1.0))
    # Base: candidate grid is already shaped; constant 1.0 is acceptable.
    return goal


def separation_penalty(pos: tuple[float, float], teammates: list[tuple[float, float]]) -> float:
    """
    Port of Midfielder::rating separation Gaussian (PositionRoles/Midfielder.cpp:60-72).

    rating = 1 - exp(-0.5 * minTeammateDistSq / σ²)

    Pos with no nearby teammates ≈ 1.0 (clean). Pos stacked on a teammate ≈ 0.0.
    """
    if not teammates:
        return 1.0
    min_dsq = min((pos[0] - t[0]) ** 2 + (pos[1] - t[1]) ** 2 for t in teammates)
    return 1.0 - math.exp(-0.5 * min_dsq / (SIGMA_TEAM_MM ** 2))
```

#### 14.12.5 R4 — Hysteresis (not a separate rule, but enumerated explicitly)

R4 is **three places** where temporal commit is enforced:

| Location | Mechanism | Constant | Purpose |
|---|---|---|---|
| R1 role assignment | `candidate_streak < ROLE_HYSTERESIS_FRAMES` keeps previous striker | `ROLE_HYSTERESIS_FRAMES = 5` | Eliminates role thrash (§11.B.3.1) |
| R2 smashOrPass | `decision_penalty` subtracted from rival options when switching last_pass_target | `DECISION_PENALTY_MIN/MAX` | Eliminates shoot↔pass thrash (PlayBall.cpp:38-39) |
| R3 support positioning | `STAB_OFFSET_MM` keeps current slot if alternative scores within margin | `STAB_OFFSET_MM = 800` | Eliminates support-slot oscillation (Tactic.stabilityOffset) |

No new code module — R4 lives inside R1/R2/R3 implementations above. **Do not separate it out into its own function**; the BHuman pattern works because the hysteresis is local to each decision, not a global meta-rule.

#### 14.12.6 R6 — Pass quality scoring (embedded in R2)

R6 is the **two proxy functions** `pass_reachable_proxy` and `goal_proximity_proxy` defined above in §14.12.3. They replace BHuman's `PassEvaluation` and `ExpectedGoals` learned estimators. The math (`pass_reachable × goal_proximity`) lives inside the smashOrPass loop at the `agent_value = ...` line. There is **no separate R6 module**.

### 14.13 `multiagent_teacher.py` module skeleton

```python
"""Multi-agent heuristic teacher.

Per-role labels for stage 6+. Each agent's call is independent and stateless;
all team-aware state (role assignment, EMA coordination targets, last pass target
decay) lives in the env and arrives via the info packet (§11.B.3.3).

R1-R4-R6 are implemented inline per §14.12. R5 is intentionally absent (teammate-
as-obstacle is solved by Block A `nearest_teammate_front_dist` + the BHuman walk
planner, not the teacher).

DO NOT add state to this module. DO NOT compute role here. DO NOT smooth targets here.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from action import SkillType, StructuredSkillAction
from config import ACTION, GOAL_X
from gate import GateDecision
from pybh_backend import BackendObservation
from teacher_policy import stage_aware_teacher_action  # striker branch reuse

# Constants per §14.12.1 go here. Keep them at module top — single source of truth.
# ... (omitted for brevity in this skeleton)


@dataclass(frozen=True)
class TeammateSlot:
    slot_idx: int        # Block B sort order (0=nearest, 1=second, 2=third)
    agent_id: int        # player number
    x_field: float
    y_field: float
    is_gk: bool


def teacher_striker(obs, stage_id, prev_skill_name, gate_decision,
                    teammate_slots, last_pass_target, last_pass_target_age_ms):
    """§11.B.2.1 + R2 smashOrPass. Calls existing single-agent teacher for walk/dribble
    decisions; overrides shoot vs pass via smash_or_pass when both arms are armed.
    """
    decision = stage_aware_teacher_action(
        obs, stage_id, prev_skill_name, gate_decision=gate_decision,
    )
    # If single-agent teacher chose shoot or dribble, we may want pass instead.
    if decision.skill_name == "shoot" and gate_decision.shoot_armed:
        skill, slot = smash_or_pass(obs, teammate_slots, last_pass_target, last_pass_target_age_ms)
        if skill == "pass":
            return StructuredSkillAction(
                skill_type=SkillType.pass_,
                target_x=0.0, target_y=0.0, target_theta=0.0,
                pass_target=slot,
            )
    return decision


def teacher_open_support(obs, stage_id, prev_skill_name, gate_decision,
                         coordination_target):
    """§11.B.2.2 — walk-or-stand-with-tolerance toward smoothed open-lane target."""
    tx, ty = coordination_target
    dx = tx - float(obs.robot_x)
    dy = ty - float(obs.robot_y)
    dist = math.hypot(dx, dy)
    if dist < HOLD_TOLERANCE_OPEN_MM:
        return StructuredSkillAction(skill_type=SkillType.stand)
    target_theta = math.atan2(dy, dx)
    return _open_lane_anchor_action(obs, (tx, ty), target_theta, stage_id=stage_id)


def teacher_off_ball_support(obs, stage_id, prev_skill_name, gate_decision,
                             coordination_target):
    """§11.B.2.3 — triangulation + unexpected-possession escape hatch."""
    if gate_decision.dribble_armed:
        # Escape hatch: ball is on me; act as striker for this one frame.
        return stage_aware_teacher_action(
            obs, stage_id, prev_skill_name, gate_decision=gate_decision,
        )
    tx, ty = coordination_target
    dx = tx - float(obs.robot_x)
    dy = ty - float(obs.robot_y)
    dist = math.hypot(dx, dy)
    if dist < HOLD_TOLERANCE_OFF_BALL_MM:
        return StructuredSkillAction(skill_type=SkillType.stand)
    target_theta = math.atan2(dy, dx)
    return _triangle_anchor_action(obs, (tx, ty), target_theta, stage_id=stage_id)


def stage_aware_team_teacher_action(
    obs: BackendObservation,
    stage_id: int,
    prev_skill_name: str | None,
    *,
    gate_decision: GateDecision,
    role: str,                                    # "striker" | "open_support" | "off_ball_support"
    coordination_target: tuple[float, float],     # field-frame xy
    teammate_slots: list[TeammateSlot],
    last_pass_target: int | None,
    last_pass_target_age_ms: float,
) -> StructuredSkillAction:
    """§11.B.2.4 dispatcher. Stateless. Pure function of inputs."""
    if role == "striker":
        return teacher_striker(obs, stage_id, prev_skill_name, gate_decision,
                               teammate_slots, last_pass_target, last_pass_target_age_ms)
    if role == "open_support":
        return teacher_open_support(obs, stage_id, prev_skill_name, gate_decision,
                                    coordination_target)
    if role == "off_ball_support":
        return teacher_off_ball_support(obs, stage_id, prev_skill_name, gate_decision,
                                        coordination_target)
    raise ValueError(f"Unknown role: {role!r}")


# --- helpers ---

def map_to_range(value, in_max, in_min, out_max, out_min):
    """BHuman mapToRange(value, ballDistMaxPenalty, ballDistMinPenalty, maxPenalty, minPenalty).
    Linear interp clamped to [out_min, out_max]."""
    if in_max == in_min:
        return out_min
    t = (value - in_min) / (in_max - in_min)
    t = max(0.0, min(1.0, t))
    return out_min + t * (out_max - out_min)


def smash_or_pass(...): ...                       # §14.12.3 body

def pass_reachable_proxy(...): ...                # §14.12.3 body

def goal_proximity_proxy(...): ...                # §14.12.3 body

def support_target(...): ...                       # §14.12.4 body

def forward_rating(...): ...                       # §14.12.4 body

def separation_penalty(...): ...                   # §14.12.4 body

def _open_lane_anchor_action(obs, target_xy, target_theta, *, stage_id):
    """Analogous to _action_from_anchor in teacher_policy.py but anchored on
    target_xy (open-lane center) instead of the ball. Returns walk skill with
    residuals computed against the open-lane anchor."""

def _triangle_anchor_action(obs, target_xy, target_theta, *, stage_id):
    """Same idea for the triangle target."""
```

### 14.14 Gate extension (`gate.py`)

Add three skill-arm bits + three role one-hot bits to `GateDecision`. Bake into obs per Design A.

```python
# gate.py additions (preserve all existing fields).

_GATED_SKILLS = ("shoot", "dribble", "pass", "observe")   # add pass, observe

@dataclass(frozen=True)
class GateConfig:
    # ... all existing fields unchanged ...

    # --- pass gate ---
    pass_enter_open_threshold: float = 0.5       # smashOrPass picked pass AND open
    pass_exit_open_threshold: float = 0.35
    pass_arm_frames: int = 2

    # --- observe gate (Stage 8+, present from sprint for future use) ---
    observe_enter_ball_lost_age_ms: float = 1500.0
    observe_exit_ball_lost_age_ms: float = 1000.0
    observe_arm_frames: int = 3


@dataclass
class GateState:
    shoot_armed: bool = False
    dribble_armed: bool = False
    pass_armed: bool = False                 # NEW
    observe_armed: bool = False              # NEW
    shoot_streak: int = 0
    dribble_streak: int = 0
    pass_streak: int = 0                     # NEW
    observe_streak: int = 0                  # NEW


@dataclass(frozen=True)
class GateDecision:
    shoot_armed: bool
    dribble_armed: bool
    shoot_arm_progress: float
    pass_armed: bool                         # NEW
    observe_armed: bool                      # NEW
    pass_arm_progress: float                 # NEW
    is_striker: bool                         # NEW (set by env, NOT computed here)
    is_open_support: bool                    # NEW (set by env)
    is_off_ball_support: bool                # NEW (set by env)
    allowed_skills: tuple[str, ...]

    def obs_features(self) -> list[float]:
        """Ordered to match ObsIndex 23..46. CRITICAL: order is the contract."""
        return [
            1.0 if self.shoot_armed else 0.0,           # 23
            1.0 if self.dribble_armed else 0.0,         # 24
            float(self.shoot_arm_progress),             # 25
            # NEW gate bits — indices 41-43.
            1.0 if self.pass_armed else 0.0,            # 41
            1.0 if self.observe_armed else 0.0,         # 42
            float(self.pass_arm_progress),              # 43
            # NEW role one-hot — indices 44-46.
            1.0 if self.is_striker else 0.0,            # 44
            1.0 if self.is_open_support else 0.0,       # 45
            1.0 if self.is_off_ball_support else 0.0,   # 46
        ]
```

**Important:** The gate **does not compute** `is_striker` / `is_open_support` / `is_off_ball_support`. The env computes them via `assign_roles` (§14.12.2) and writes them into the GateDecision before `obs_features()` is called. The gate computes only the skill-arm bits.

`SkillGate.step()` extends with:
```python
# After existing shoot + dribble logic, add:
pass_ready = (
    self._pass_signal_from_smash_or_pass(obs)        # set by env-side hook
    >= cfg.pass_enter_open_threshold
)
# arming follows same enter/exit/debounce pattern as shoot
# ... (mirror existing shoot_armed block)
```

The actual `pass_armed` decision is informed by the smashOrPass call from the env (the env runs `smash_or_pass` once per striker per step and writes the result into a shared field the gate reads). Pattern matches how `gate_decision` is shared today.

### 14.15 Observation extension (`observation.py`)

```python
class ObsIndex(IntEnum):
    # ... existing 0-40 unchanged ...
    # NEW gate bits (Block A extension):
    pass_armed = 41
    observe_armed = 42
    pass_arm_progress = 43
    # NEW role one-hot (Block G):
    is_striker = 44
    is_open_support = 45
    is_off_ball_support = 46


OBS_FEATURE_NAMES = tuple(index.name for index in ObsIndex)
OBS_SIZE = len(OBS_FEATURE_NAMES)        # = 47
```

`environment.py:_encode_obs` appends six floats from `gate_feats` (already returns the right ordered list after §14.14).

### 14.16 Mask extension (`train.py:gate_skill_mask_from_obs`)

```python
_PASS_SKILL_IDX = SKILL_ORDER.index("pass")
_OBSERVE_SKILL_IDX = SKILL_ORDER.index("observe")

def gate_skill_mask_from_obs(observations: torch.Tensor) -> torch.Tensor:
    """Forbid-invalid skill mask. Pure function of obs columns.
    Reads only: 23 (shoot_armed), 24 (dribble_armed), 41 (pass_armed),
                42 (observe_armed), 44 (is_striker), 45 (is_open_support),
                46 (is_off_ball_support).
    """
    bsz = observations.shape[0]
    mask = torch.zeros(
        (bsz, len(SKILL_ORDER)), dtype=observations.dtype, device=observations.device
    )

    # walk and stand never forbidden — fallback always available.
    mask[:, SKILL_ORDER.index("stand")] = 1.0
    mask[:, SKILL_ORDER.index("walk")] = 1.0

    shoot_armed = observations[:, 23] > 0.5
    dribble_armed = observations[:, 24] > 0.5
    pass_armed = observations[:, 41] > 0.5
    observe_armed = observations[:, 42] > 0.5

    is_striker = observations[:, 44] > 0.5
    is_open = observations[:, 45] > 0.5
    is_off_ball = observations[:, 46] > 0.5

    # Striker: full skill-arm gating, plus pass.
    mask[:, _SHOOT_SKILL_IDX] = (shoot_armed & is_striker).to(mask.dtype)
    mask[:, _PASS_SKILL_IDX] = (pass_armed & is_striker).to(mask.dtype)
    mask[:, _DRIBBLE_SKILL_IDX] = (dribble_armed & is_striker).to(mask.dtype)

    # Open-support: forbid shoot+dribble+pass entirely (per §11.A.4).
    # off-ball-support: dribble allowed for unexpected-possession escape hatch.
    mask[:, _DRIBBLE_SKILL_IDX] = (
        mask[:, _DRIBBLE_SKILL_IDX].bool() | (dribble_armed & is_off_ball)
    ).to(mask.dtype)

    # Observe is allowed for any non-striker when observe_armed (Stage 8+ scenarios).
    mask[:, _OBSERVE_SKILL_IDX] = (observe_armed & ~is_striker).to(mask.dtype)

    return mask
```

**Behavior-preservation contract extension** (`test_obs_migration_contract.py`):
```python
def test_obs_padding_preserves_policy_value_outputs_26_to_47():
    # Same shape as the existing 26→41 test, but at 47 dims.
    # When obs[:, 26:] = 0 the policy must output identical skill_logits, param_mean,
    # log_std, and values as the 26-dim baseline.
    ...

def test_gate_skill_mask_depends_only_on_specified_columns():
    # mask depends only on cols 23, 24, 41, 42, 44, 45, 46. Perturb every other
    # column and assert the mask is unchanged.
    ...
```

### 14.17 Environment extension (`environment.py`)

#### 14.17.1 Per-env team state

```python
class RealSimRobotEnv(ParallelEnv):
    def __init__(self, ...):
        # ... existing ...
        self._team_role_history = TeamRoleHistory(current_roles={}, candidate_streak={})
        self._open_lane_ema: tuple[float, float] | None = None
        self._triangle_ema: tuple[float, float] | None = None
        self._last_pass_target: int | None = None
        self._last_pass_target_age_ms: float = 0.0
        self._support_committed_slot: dict[int, tuple[float, float]] = {}
        # Per-agent fallen-stuck counters (extends Rafael's single-agent counter)
        self._fallen_consecutive_steps_per_agent: dict[int, int] = {}
```

Reset zeros all of these.

#### 14.17.2 Per-step orchestration

```python
def step(self, actions):
    # 1. Run physics → get per-agent BackendObservation.
    # 2. Compute team-aware state:
    #    a. Identify GK by exclusion (teammate with x_field < GK_EXCLUDE_X_MM).
    #    b. Assemble agent_obs dict for RL agents only.
    #    c. roles = assign_roles(agent_obs, team_ball, self._team_role_history)
    #    d. open_lane_target_raw = compute_open_lane_target(open_support_pose, ball_pose, ...)
    #    e. self._open_lane_ema = ema(self._open_lane_ema, open_lane_target_raw, EMA_ALPHA_OPEN_LANE)
    #    f. same for triangle target.
    # 3. Per agent: compute its GateDecision with role flags set.
    # 4. Per agent: build info packet (see 14.17.3).
    # 5. Update last_pass_target_age_ms += dt; reset on new pass.
    # 6. Bake gate.obs_features() into the encoded obs (existing flow).
    # 7. Apply per-agent fallen-stuck termination (Rafael pattern, scoped to ANY agent).
    # 8. Standard reward + done propagation.
```

#### 14.17.3 Info packet per §11.B.3.3

```python
info[agent_id] = {
    # ... existing fields ...
    "raw_obs": backend_obs_dict,
    "gate_decision": gate_decision,
    "role": role_str,                                     # str: "striker" | "open_support" | "off_ball_support"
    "role_id": role_id_int,                               # int: 0 / 1 / 2 — for dataset storage
    "coordination_target": (x_mm, y_mm),                  # EMA-smoothed by env
    "teammate_slots": [TeammateSlot(...), ...],           # for striker's smashOrPass call
    "last_pass_target": self._last_pass_target,
    "last_pass_target_age_ms": self._last_pass_target_age_ms,
    "team_state": {
        "striker_id": ...,
        "open_support_id": ...,
        "off_ball_support_id": ...,
        "ball_holder_id": ... | None,
    },
}
```

#### 14.17.4 EMA helper

```python
def _ema(prev: tuple[float, float] | None, new: tuple[float, float], alpha: float) -> tuple[float, float]:
    if prev is None:
        return new
    return (alpha * new[0] + (1 - alpha) * prev[0],
            alpha * new[1] + (1 - alpha) * prev[1])
```

#### 14.17.5 Per-agent fallen-stuck (Rafael pattern, scope = team)

```python
# Extend existing _update_fallen_stuck_counter:
def _update_fallen_stuck_counter_team(self, agent_obs_dict) -> bool:
    """Any agent stuck for FALLEN_STUCK_TERMINATE_STEPS terminates the team episode."""
    any_stuck = False
    for agent_id, obs in agent_obs_dict.items():
        if int(obs.motion_engine_fall_state) > UPRIGHT_MAX_FALL_STATE:
            self._fallen_consecutive_steps_per_agent[agent_id] = \
                self._fallen_consecutive_steps_per_agent.get(agent_id, 0) + 1
        else:
            self._fallen_consecutive_steps_per_agent[agent_id] = 0
        if (self._episode_stage_id >= FALLEN_STUCK_MIN_STAGE
            and FALLEN_STUCK_TERMINATE_STEPS > 0
            and self._fallen_consecutive_steps_per_agent[agent_id] >= FALLEN_STUCK_TERMINATE_STEPS):
            any_stuck = True
    return any_stuck
```

### 14.18 Dataset generation extensions (`teacher_production_dataset.py`)

Use the v2 contracts from §11.B.4. Specifically:

- Replace `_episode_kept` with `_episode_kept_v2` per §11.B.4.1. Per-role hold-steps requirements.
- Replace `_sample_quality_weight` with `_sample_quality_weight_v2` per §11.B.4.2. Down-weight "going wrong direction."
- Replace `_gate_label_consistent` with `_gate_label_consistent_v2` per §11.B.4.3. Both gate-bucket AND role-mask consistent. Add `positive_weight_role_mask_consistent` to `summary.json`.
- Add to `samples.npz`:
  - `role_ids[N]` (int8) — 0=striker / 1=open_support / 2=off_ball_support
  - `coordination_target_xy[N, 2]` (float32) — telemetry only
- Additional filter (sprint-specific):
  - Drop episodes with **any** teammate-teammate collision.
  - Drop episodes with role-thrash > 5 swaps over 400 steps (the same `role_thrash_rate` metric the evaluator computes).
  - Drop episodes where `pass` was emitted and the ball was intercepted (`ball_holder_id` becomes None or opponent within 1.5 s of pass).

The dataset's recorded per-step gate flags (cols 23/24/25 today; add 41/42/43/44/45/46) come from the gate's `obs_features()` at the time the teacher labeled the row — Design A automatic.

### 14.19 BC + PPO retunings (`train_team_production.py`)

New file, sibling of `train_production.py`. Differs only in:

```python
ScriptRunConfig(
    stage_id=6,                                # start at stage 6 (2v0 + GK)
    max_stage_id=7,                            # allow promotion to 7
    train_steps=20_000,                        # sprint budget; full §11.B.6.3 was 80k
    bc_warm_start=True,
    bc_checkpoint_path="runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt",
    bc_replay=True,
    bc_replay_dataset_path=TEAM_DATASET_PATH,
    bc_beta_max=0.08,
    bc_beta_min=0.02,
    bc_competence_half_life_evals=25,
    bc_replay_weight=0.10,
    bc_replay_updates=2,
    bc_skill_loss_scale=0.40,
    bc_param_loss_scale=0.50,
    parallel_player_numbers=(1, 2, 3),         # stage 6; (1, 2, 3, 4) for stage 7
    stage4_finish_interleave_probability=0.2,
    eval_episodes=8,
    max_episode_steps=400,
    eval_rubrics=("single_agent", "team_coordination", "support_positioning"),
    per_eval_auto_rollback=True,               # §11.B.6.4
)
```

`BCReplayRegularizer` extends per §11.B.5.1: indices keyed by `(stage, role, skill)`. `BCCompetenceGate` extends per §11.B.5.2: `competence = min(striker_competence, open_competence, off_ball_competence)`.

### 14.20 Reward terms (`reward.py`)

```python
# Conservative starting weights — tune Day 5. Each rides on top of existing weights.
@dataclass
class RewardWeightsTeam(RewardWeights):
    # ... inherit all existing fields ...
    support_open_hold_bonus: float = 0.5      # per step when open_lane_progress equivalent >= 0.7
    support_triangle_hold_bonus: float = 0.2  # per step when at triangle within tolerance
    team_goal_bonus: float = 5.0              # distributed across RL agents on team goal
    pass_attempt_credit: float = 1.0          # on detected pass attempt regardless of outcome
    pass_completed_bonus: float = 3.0         # ball reached intended receiver within 1.5s
    pass_intercepted_penalty: float = -2.0    # opponent intercepted
    role_thrash_penalty: float = -1.0         # >5 role swaps in episode (terminal penalty)
    teammate_separation_penalty: float = -0.2 # per step both supports within 800mm of each other
```

Detection of pass events (`pass_attempt`, `pass_completed`, `pass_intercepted`):
- The env tracks `ball_holder_id` per step (the RL agent whose `ball_rel` is within takeover envelope, OR None).
- A `pass_attempt` is logged when an agent's selected skill in step t was `pass`.
- A `pass_completed` is logged if within 1.5 s after a `pass_attempt`, `ball_holder_id` becomes the receiver agent_id targeted by `pass_target` slot.
- A `pass_intercepted` is logged if within 1.5 s, an opponent intercepts (ball moves into a region near a detected opponent; for sprint, proxy = no RL agent claims the ball AND `nearest_opponent_dist < 500mm` for ≥ 5 frames).

### 14.21 Curriculum stage specs (`production_curriculum.py`) — LOCKED for Monday 2026-06-15 live BH demo

**Demo configuration:** 1 BHuman GK + 3 RL field (= 4 NAOs our side) vs full BH (= 4 NAOs their side). **All scenario weights below match this distribution.**

(If demo confirmed to be 2 RL field instead of 3, swap Stage 6 / Stage 7 emphasis per §14.21.4. **Confirm with Rafael before Tue.**)

#### 14.21.1 Skill sets

```python
ACTIVE_SKILLS_MA = ("stand", "walk", "dribble", "shoot", "pass")
# Note: mark, block, observe stay masked off this sprint — see §14.7 anti-patterns.
```

#### 14.21.2 Stage 6 — `multiagent_warmup_2v0` (sanity, NOT the workhorse)

```python
STAGE_6_SPEC = StageSpec(
    stage_id=6,
    name="multiagent_warmup_2v0",
    description="1 BH GK + 2 RL field. Warmup for the role one-hot + pass head; minimum opponent exposure.",
    ball_radius_min_mm=2500.0, ball_radius_max_mm=3000.0,
    full_field=False, obstacle_mode="mixed",
    active_skill_names=ACTIVE_SKILLS_MA,
    reward_weights=RewardWeightsTeam(
        **STAGE_4_REWARD_WEIGHTS_DICT,
        support_open_hold_bonus=0.5,
        support_triangle_hold_bonus=0.0,         # no off-ball-support in 2-RL-field
        team_goal_bonus=5.0,
        pass_attempt_credit=1.0,
        pass_completed_bonus=3.0,
        pass_intercepted_penalty=-1.0,            # mild — only static opponents at this stage
        role_thrash_penalty=-1.0,
        teammate_separation_penalty=-0.2,
    ),
    max_episode_steps=400,
)
```

**Scenarios** (weights total = 4.0):

| Scenario | Weight | Spawn |
|---|---|---|
| `team_2v0_clean_band` | 3.0 | 1 BH GK at own goal + 2 RL field, ball at 2.5-3.0m from opponent goal, no opponents on field |
| `team_2v0_static_blocker` | 1.0 | Same as above + 1 static BH opponent placed in the striker's direct shot lane |

**PPO budget: ≤ 5k stage-6 steps.** This stage's job is to warm role one-hot weights and validate the pass-receive pattern exists. **Hard stop and move to Stage 7** if BC + 5k PPO doesn't promote — do not grind here; Monday's demo is at Stage 7.

#### 14.21.3 Stage 7 — `multiagent_adversarial_3v3` (workhorse, matches demo distribution)

```python
STAGE_7_SPEC = StageSpec(
    stage_id=7,
    name="multiagent_adversarial_3v3",
    description="1 BH GK + 3 RL field vs full BH. Monday demo configuration. Heavy adversarial weighting.",
    ball_radius_min_mm=2500.0, ball_radius_max_mm=3500.0,
    full_field=True, obstacle_mode="mixed",
    active_skill_names=ACTIVE_SKILLS_MA,
    reward_weights=RewardWeightsTeam(
        **STAGE_4_REWARD_WEIGHTS_DICT,
        support_open_hold_bonus=0.5,
        support_triangle_hold_bonus=0.2,
        team_goal_bonus=5.0,
        pass_attempt_credit=1.0,
        pass_completed_bonus=3.0,
        pass_intercepted_penalty=-2.0,            # full penalty under live opponents
        role_thrash_penalty=-1.0,
        teammate_separation_penalty=-0.2,
    ),
    max_episode_steps=400,
)
```

**Scenarios** (weights total = 7.0):

| Scenario | Weight | Spawn | Why this weight |
|---|---|---|---|
| `team_3v0_clean` | 0.5 | 1 GK + 3 RL field, no opponents | Single-agent retention reference only — barely-weighted floor signal |
| `team_3v0_static_pinch` | 1.5 | 1 GK + 3 RL field + 2 static BH opponents pinching the central lane | Opponent-pressure signal at 2-opponent intensity, no dynamic confusion |
| `team_3v0_dynamic` | 2.0 | 1 GK + 3 RL field + 2-3 dynamic opponents using BH default behavior | Partial demo conditions — opponents move but it's not full BH |
| **`team_3v3_full_dynamic`** | **3.0** | **1 GK + 3 RL field vs full BH side (1 BH GK + 3 BH field running standard tactic)** | **Monday demo distribution. Non-negotiable weight. The only scenario where the policy sees exact perceptual conditions of the demo.** |

`team_3v3_full_dynamic` is the most informative scenario and also the slowest at ~1/8 wallclock efficiency per `[[production-finetune-v1-drag-and-throughput-fix]]`. Run more slots in parallel (per `parallel_player_numbers=(1,2,3)`) to amortize.

**PPO budget: 10-15k stage-7 steps.** This is the bulk of the sprint's compute.

#### 14.21.4 If demo is 2 RL field instead of 3

Swap weights: drop `team_3v3_full_dynamic` to 0.5, raise a new `team_2v2_full_dynamic` (1 GK + 2 RL field vs 1 BH GK + 2 BH field) to 3.0 in Stage 6, and use Stage 7 as a stretch goal not a deliverable. Confirm demo configuration with Rafael Tuesday morning before generating the dataset.

#### 14.21.5 Promotion gates (RELAXED for adversarial scope vs §11.B.6.2)

```python
def stage_6_promotion(metrics) -> bool:
    return (
        metrics["clean_finish_rate"] >= 0.60    # single-agent retention via interleave
        and metrics["policy_shoot_rate"] >= 0.02
        and metrics["team_goal_rate"] >= 0.10   # at least 1 goal per 10 episodes in 2v0 with static
        and metrics["role_thrash_rate"] <= 0.10
    )

def stage_7_promotion(metrics) -> bool:
    return (
        metrics["clean_finish_rate"] >= 0.45    # RELAXED from 0.60 — adversarial 3v3 is harder
        and metrics["team_goal_rate"] >= 0.05   # at least 1 goal per ~20 episodes vs full BH
        and metrics["role_thrash_rate"] <= 0.15 # RELAXED — dynamic opponents jitter ball
        and metrics["pass_completed_rate"] >= 0.15  # NEW — passes must work under pressure
    )
```

The clean_finish_rate floor at stage 7 drops because adversarial 3v3 vs full BH is genuinely harder than the single-agent baseline; demanding 0.60 is unrealistic for a first-pass policy. **0.45 + a positive team_goal_rate** is the right honest bar.

#### 14.21.6 World builders — what each scenario must spawn

For each new scenario, `_team_*_world(rng, ...)` returns a `WorldSpec` (per existing `production_curriculum.py:63` `_drive_finish_world` pattern). The pattern:

```python
def _team_3v3_full_dynamic_world(rng):
    # 1. Ball: production band (2.5-3.0 m from opponent goal), with small y jitter.
    ball_xy = sample_ball_pose_in_band(rng, x_min=2500.0, x_max=3000.0, y_jitter=600.0)
    # 2. Our team (4 NAOs):
    #    - GK (player 4, BHuman behavior): own goal area.
    #    - Striker spawn (player 1, RL): close to ball but not in shooting envelope yet.
    #    - Open support (player 2, RL): forward, on cleaner lane side.
    #    - Off-ball support (player 3, RL): triangle vertex, opposite side.
    # 3. Opponents (4 NAOs, full BH side, dynamic):
    #    - Their GK: their own goal area.
    #    - Their striker: mirror x of our striker (chases ball).
    #    - Their 2 supports: defensive positions.
    # 4. Call build_world_spec(...) to assemble.
    # 5. Set obstacle_mode="dynamic" so opponents run BH default tactic, not parked.
    return build_world_spec(...)
```

Reuse `_team_3v0_clean_world` and `_team_3v0_dynamic_world` as references — they share most logic; the full-dynamic variant just adds the opponent side as live BH agents instead of static or empty.

#### 14.21.7 Honest Monday demo behavior forecast

Be ready for what the policy WILL and WON'T do:

| Will do | Won't do |
|---|---|
| React to opponent proximity via summary scalars (`nearest_opponent_*_dist`) — shoot/pass urgency when pressed | Mark a specific opponent strategically |
| Hold support position when team-ball valid + nearest_opponent_front_dist large | Position between an opponent and goal (defensive recovery) |
| Pass through cleanly when lane and receiver score high | Anticipate which opponent will press next |
| Avoid teammate collision via Block A + BHuman walk planner | Avoid an opponent's predicted intercept (no opponent position in obs) |
| Walk-to-support when not striker | Switch to defensive mode when team loses possession |

**Demo signature:** organized in possession (passes, support holds, occasional goals), reactive-but-not-strategic out of possession. **Expect 0.05-0.15 team_goal_rate against full BH.** Above 0.20 means the BH side under-performed; below 0.03 means our policy's pressure-handling regressed and we should pull the rollback.

This is the honest first-pass bar. The "beat BH" target is §14.26.

### 14.22 Tests to add or extend

| File | Test | Purpose |
|---|---|---|
| `test_obs_migration_contract.py` | `test_obs_padding_preserves_policy_value_outputs_26_to_47` | Zero-pad bit-identity at 47 dims (extends the existing 26→41 test) |
| `test_obs_migration_contract.py` | `test_gate_skill_mask_depends_only_on_specified_columns` | Mask reads only 23/24/41/42/44/45/46 — perturb every other column, mask must not change |
| `test_mappo_observation_space.py` | `test_role_one_hot_round_trip` | When env writes role to GateDecision, obs encoding round-trips correctly to indices 44/45/46 |
| `test_mappo_observation_space.py` | `test_role_assignment_hysteresis_5_frames` | Hand-construct a 5-frame sequence with striker candidate flipping; assert role only changes after streak |
| `test_mappo_observation_space.py` | `test_ema_open_lane_target` | Per-step `compute_open_lane_target` jitters; assert EMA smooths within stated alpha |
| `test_multiagent_teacher.py` (NEW) | `test_smash_or_pass_shoots_when_own_value_high` | Hand-construct obs with `shot_opening_with_obstacles=0.8`; assert returns "shoot" |
| `test_multiagent_teacher.py` (NEW) | `test_smash_or_pass_passes_when_teammate_better` | Striker far from goal, teammate at goal x; assert pass with teammate's slot |
| `test_multiagent_teacher.py` (NEW) | `test_smash_or_pass_decision_penalty_prevents_thrash` | Last frame chose teammate A by 0.10 margin; this frame teammate B is 0.05 better; assert still A due to penalty |
| `test_multiagent_teacher.py` (NEW) | `test_smash_or_pass_excludes_gk` | One of teammate slots marked `is_gk=True`; assert never selected |
| `test_multiagent_teacher.py` (NEW) | `test_support_target_uses_mirrored_grid` | Ball y > 0 → grid uses +y slots; ball y < 0 → grid uses -y slots |
| `test_multiagent_teacher.py` (NEW) | `test_support_target_separation_avoids_stacking` | Striker at slot A; assert support picks a different slot due to separation penalty |
| `test_multiagent_teacher.py` (NEW) | `test_stage_aware_team_teacher_action_dispatch` | Three calls with different `role` strings; assert correct branch executes |
| `test_multiagent_teacher.py` (NEW) | `test_off_ball_escape_hatch` | `gate_decision.dribble_armed=True`, role=off_ball_support; assert striker-style action returned |
| `test_dataset_filter_v2.py` (NEW) | `test_episode_kept_v2_per_role_thresholds` | Open-support steps 25, hold steps 18 → drop (< 20). Open-support steps 25, hold steps 22 → keep. |
| `test_dataset_filter_v2.py` (NEW) | `test_quality_weight_v2_directional` | Closing on target → 0.75; drifting away → 0.25 |

### 14.23 Acceptance checklist — when is a step "done"

#### Tue commit acceptance (Luis)

- [ ] `multiagent_teacher.py` exists with all functions in §14.13 skeleton.
- [ ] `gate.py` extended with new fields per §14.14 (GateConfig, GateState, GateDecision, obs_features ordered correctly).
- [ ] `observation.py` extended to 47 dims per §14.15.
- [ ] `train.py:gate_skill_mask_from_obs` extended per §14.16.
- [ ] `test_obs_migration_contract.py` has `test_obs_padding_preserves_policy_value_outputs_26_to_47` passing.
- [ ] `test_obs_migration_contract.py` has `test_gate_skill_mask_depends_only_on_specified_columns` passing.
- [ ] `test_multiagent_teacher.py` has all 7 tests in §14.22 passing.
- [ ] Single-agent `behavioral_evaluator.py` run on `checkpoint_stage4_step5000.pt` with new 47-dim obs (zero-pad) → at least 3/5 — confirms regression preserved.

#### Wed commit acceptance (Luis)

- [ ] `environment.py` carries the per-env team state from §14.17.1.
- [ ] Per-step orchestration § 14.17.2 wires roles + EMA + last_pass into info per § 14.17.3.
- [ ] Per-agent fallen-stuck § 14.17.5 terminates team episode correctly.
- [x] `teacher_multiagent_dataset.py` runs the multi-agent dataset gen and produces `samples.npz` with `role_ids` and `coordination_target_xy` arrays.
- [x] Audit `summary.json` shows `positive_weight_gate_label_consistent: true` AND `positive_weight_role_mask_consistent: true`.
- [ ] Stages 6 and 7 added to `production_curriculum.py` per §14.21.
- [ ] At least 100 episodes per stage in the dataset.
- [ ] `test_dataset_filter_v2.py` tests passing.

2026-06-10 handoff update:
- Quarantined: `runs/teacher_multiagent_dataset/dataset_multiagent_train_20260610_0805/samples.npz`. The audit missed a silent failure: striker rows were walk-only (`dribble_armed=0`, `shoot_armed=0`) and must not be used for BC.
- Root cause isolated: the single multi-real `RLMAStage{6,7}_3D` scenes accepted BHuman walk commands but did not produce reliable visible locomotion. The collector now uses the production-style `VisibleParallelRuntime` slot topology (`RL1v0_3D_visible` per RL field player).
- Train-start sane dataset: `runs/teacher_multiagent_dataset/dataset_multiagent_train_20260610_sane_v1/samples.npz` — stages 6+7, 11 kept episodes, 9,637 rows, 47-dim obs, `role_ids`, `coordination_target_xy`, `label_consistent`.
- Audit result: `runs/teacher_multiagent_dataset/dataset_multiagent_train_20260610_sane_v1/audit.json` passed with zero role-mask bad rows, `striker_dribble_armed_rows=861`, `striker_shoot_armed_rows=154`, all six scenarios represented.
- Mechanical BC check: `train_multiagent.MultiAgentBCReplayRegularizer` loads the dataset, builds stage/role/skill buckets, samples stages 6 and 7, and runs one CPU BC update.
- Limitation: pass labels are still absent (`pass_armed_rows=0`), so this is a safe train-start corpus for support positioning plus striker walk/dribble/shoot retention, not a pass-rich final teacher corpus.

#### Thu commit acceptance (Luis + Laura)

- [ ] BC train on new dataset from `checkpoint_stage4_step5000.pt` with zero-pad warm-start completes.
- [ ] `behavioral_evaluator.py` on BC checkpoint: single-agent floor ≥ 3/5.
- [ ] BC loss decreases monotonically over the first 5k steps for each role (per-role loss surfaced in tensorboard / logs).
- [ ] `behavioral_evaluator.py` has new `_pass_team_coordination`, `_pass_support_positioning`, `_pass_team_no_regression` registered.
- [ ] Baseline run (BC only, no PPO) reports team rubric metrics without NaN.

#### Fri commit acceptance (Luis)

- [ ] `train_team_production.py` runs without crash for at least 5k PPO steps.
- [ ] Per-eval auto-rollback fires correctly when single-agent regresses (force a regression in a unit-style test or observe in run logs).
- [ ] Stage 6 promotion gate: clean_finish ≥ 0.6 AND policy_shoot_rate ≥ 0.02 AND team_goal_rate > 0 AND role_thrash_rate ≤ 0.10.

#### Sat commit acceptance (Luis + Laura)

- [ ] Multi-agent stability probe runs against final checkpoint with seed set `11,23,37,51,67,83`.
- [ ] Per-seed pass_rate logged for single-agent rubric (≥ 0.6) and team rubric.
- [ ] ONNX exported from final checkpoint at 47 dims.
- [ ] BHuman C++ loads new ONNX without error (Rafael verifies).

#### Mon demo acceptance

- [ ] 3 NAOs (2 RL striker+open-support + 1 BHuman GK) run for at least 5 minutes without crash.
- [ ] At least one observed pass attempt.
- [ ] At least one observed support-position hold (support stays in slot for ≥ 3 s while striker has ball).
- [ ] No teammate-teammate collisions over 5-minute run.

### 14.24 Failure modes and what to do

| If you see | Do |
|---|---|
| `test_obs_padding_preserves_policy_value_outputs_26_to_47` fails | A new column in obs_features() is not zero by default. Check §14.14 GateDecision defaults. |
| Single-agent floor drops below 3/5 on Tue (after gate/obs/mask only, no training) | Mask is forbidding a skill it shouldn't. Re-check §14.16 against §11.A.4 contract. |
| BC loss diverges after 1k steps | KL anchor is wrong checkpoint. Verify `bc_checkpoint_path = "runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt"`. |
| BC loss decreases for striker but not for supports | `BCCompetenceGate` is taking mean not min. Verify §14.19 / §11.B.5.2. |
| `role_thrash_rate > 0.20` in dataset audit | `ROLE_HYSTERESIS_FRAMES = 5` is too low for the scenario noise. Raise to 7, regen one scenario, re-check. |
| Support oscillates between two slots | `STAB_OFFSET_MM = 800` is too low. Raise to 1200, re-test. |
| `pass_completed_bonus` never fires | Detection logic in §14.20 is too strict. Loosen the 1.5 s receiver-claim window or the pass_target slot match. |
| Single-agent regresses mid-PPO | `per_eval_auto_rollback` is doing its job — investigate which reward term dominates and lower its weight. |
| Team-goal-rate stays 0 despite supports holding position | Striker is dribbling past the support without dropping ball. Check `decision_penalty` is being applied; lower `SHOOT_THRESHOLD` to 0.45 to bias toward pass. |
| ONNX export fails at 47 dims | `tools/export_ppo_policy.py` was hard-coded to 41. Update its obs_size argument. |

### 14.25 Glossary of new symbols introduced by this sprint

| Symbol | Where | Meaning |
|---|---|---|
| `R1`-`R6` | §14.12 | The five teacher rules (R5 dropped) |
| `smashOrPass` | §14.12.3, PlayBall.cpp:33 | BHuman's shoot-vs-pass decision; we port the structure |
| `decisionPenalty` | §14.12.3, PlayBall.cpp:38-39 | Subtracted from rival options on decision switch — R4 mechanism |
| `stabilityOffset` | §14.12.4, Tactic.h | Slot-commit hysteresis — R4 mechanism |
| `Tactic::Position::Type` | BHuman, Tactic.h:42-57 | Named role positions; we use only as template inspiration |
| `is_striker` / `is_open_support` / `is_off_ball_support` | §14.14, 14.15 | Role one-hot, obs indices 44/45/46 |
| `pass_armed` / `observe_armed` / `pass_arm_progress` | §14.14 | New skill-arm bits, obs indices 41/42/43 |
| `TeamRoleHistory` | §14.12.2 | Per-env hysteresis state for role assignment |
| `TeammateSlot` | §14.13 | Block-B-aligned teammate descriptor passed to smashOrPass |
| `forward_rating` / `separation_penalty` | §14.12.4 | Proxies for BHuman's Forward and Midfielder rating functions |
| `pass_reachable_proxy` / `goal_proximity_proxy` | §14.12.3 | Cheap replacements for BHuman's learned PassEvaluation / ExpectedGoals |
| `EMA_ALPHA_OPEN_LANE` / `EMA_ALPHA_TRIANGLE` | §14.12.1, 14.17.4 | Coordination target smoothing — §11.B.3.2 |
| `last_pass_target_age_ms` | §14.13, 14.17.1 | Decay counter for R4 decision-penalty in smashOrPass |
| Per-role role_id (0/1/2) | §14.18 | Dataset stratification key — §11.A.5 |

---

### 14.26 Post-Monday roadmap — toward "a team that can beat BHuman"

**Target deliverable:** an RL team that is structurally capable of competing with BHuman in 4v4 — not just possessing and shooting reactively, but **defending, marking, and intercepting strategically**. This is a separate engineering problem from §14.6's sprint.

**Stakeholder ambition (2026-06-08):** demonstrable progress by Friday after Monday's demo. The realistic, honest version of that ambition is laid out below; **what is achievable in 5 days vs 11 days vs 4 weeks differs sharply.** Read 14.26.5 before committing to a date.

#### 14.26.1 Why the Monday policy cannot beat BH

The Monday policy ships with opponents represented only as summary scalars in Block A (`nearest_opponent_dist`, `nearest_opponent_front_dist`, `nearest_uncertain_*`). Three consequences:

1. **No opponent-aware steering.** Policy can react to "an opponent is close" but cannot choose which opponent to avoid or where to step to break their pressure.
2. **No defensive role.** `mark` and `block` are still masked out (stage curriculum). Even if unmasked, the policy has no obs signal to choose mark targets or block positions.
3. **No anticipation.** No opponent velocity or pose history → no learnable "they'll be here in 1s" signal.

These are obs-architecture problems, not training problems. No amount of additional PPO on the Monday curriculum will fix them. The fix is **growing the obs** and rebuilding the dataset / mask / teacher pipeline on the new substrate.

#### 14.26.2 What changes between Monday and beat-BH-ready

| Layer | Monday | Beat-BH |
|---|---|---|
| Obs size | 47 dims | **53 dims** (Block H: 3 nearest opponents, x_rel + y_rel = 6 dims) |
| Skill set | `stand, walk, dribble, shoot, pass` (5 active) | + `mark`, `block`, `observe` (8 active) |
| Stage mask | role-conditioned + skill-armed | + opponent-presence-conditioned for defensive arming |
| Teacher branches | `striker, open_support, off_ball_support` (3) | + `defender` (4) — the §11.A.2 Policy 4 from doc original |
| Stages | 6, 7 | + 8 (3v3 + defensive skill), + 9 (4v4 full match if time) |
| Reward terms | offensive (pass_completed, support_hold, team_goal) | + defensive (mark_close, block_lane, ball_recovered, opponent_intercept_prevented) |
| KL anchor | `checkpoint_stage4_step5000.pt` (4/5 striker) | **Monday checkpoint** — the 47-dim multi-agent policy |
| Dataset | 47-dim role-stratified | re-collected at 53 dims with 4-role stratification |

#### 14.26.3 New obs Block H — opponent positions

Following the same pattern as Block B (teammates), but for opponents and constrained to nearest 3:

| Index | Name | Source in BHuman | Zero-default meaning |
|---|---|---|---|
| 47 | `nearest_opponent_x_rel` | nearest tracked opponent from `ObstacleModel.obstacles` where `isOpponent()`, transformed to robot frame | no opponent visible |
| 48 | `nearest_opponent_y_rel` | same, y component | same |
| 49 | `second_opponent_x_rel` | second-nearest opponent | only 1 opponent or fewer visible |
| 50 | `second_opponent_y_rel` | same, y | same |
| 51 | `third_opponent_x_rel` | third-nearest opponent | ≤ 2 opponents visible |
| 52 | `third_opponent_y_rel` | same, y | same |

Sort by Euclidean distance to me (same convention as Block B). Stale-or-uncertain opponents (`Obstacle.lastSeen > 1000ms` OR `isUnknown()` with low confidence) excluded — same `TEAMMATE_STALE_THRESHOLD_MS = 1000` analogue.

Wire change: `Module.cpp:630-640` extends with `opponent_x_mm`, `opponent_y_mm`, `opponent_age_ms` lists. Mirror the teammate wire.

**Zero-pad warm-start contract:** 47 → 53 must preserve Monday checkpoint behavior bit-for-bit when opponent dims are zero. Extend `test_obs_migration_contract.py` with `test_obs_padding_preserves_policy_value_outputs_47_to_53`.

#### 14.26.4 New skill mask logic for `mark` and `block`

Both `mark` and `block` need opponent-presence to be useful. New skill-arm bits at indices 53+:

```python
def gate_skill_mask_from_obs(obs):
    # ... existing logic preserved ...
    # NEW: mark_armed when an opponent is within mark-range AND I am not the striker.
    mark_armed = (obs[:, 47].abs() < MARK_RANGE_REL  # nearest opponent x_rel near
                  & obs[:, 48].abs() < MARK_RANGE_REL  # and y_rel near
                  & ~is_striker)
    mask[:, _MARK_SKILL_IDX] = mark_armed.to(mask.dtype)
    # block_armed when I am between an opponent and own goal.
    block_armed = ...    # geometric condition; left as exercise
    mask[:, _BLOCK_SKILL_IDX] = block_armed.to(mask.dtype)
    return mask
```

`MARK_RANGE_REL` ≈ `1200mm / ACTION.x_range`. The geometric condition for block requires comparing self position to (opponent, own_goal) line; ~10 lines of arithmetic.

#### 14.26.5 Three honest timelines depending on stakeholder interpretation of "next Friday"

The user said "next friday is a team which can beat Bhuman." Three interpretations of "next Friday," each with a different realistic outcome:

**Interpretation A — Friday 2026-06-19 (4 days post-Monday).** Not achievable. Even with substrate-machinery momentum (Laura proved 26→41 obs growth + Rafael proved stability), the minimum coherent beat-BH pipeline is:
- Day 1 (Tue post-demo): obs 47→53 wire + zero-pad warm-start + behavior-preservation test extension.
- Day 2 (Wed): defensive teacher branch + `mark_armed`/`block_armed` gates.
- Day 3 (Thu): dataset re-gen at 53 dims.
- Day 4 (Fri): BC train. **No PPO yet.** Demo is BC-only behavior — no learned defense.

Honest outcome at Friday 06-19: a structurally-defensive policy that hasn't been fine-tuned. Will mark/block but unreliably; may lose more matches than the Monday policy.

**Interpretation B — Friday 2026-06-26 (11 days post-Monday).** Achievable as "competitive," not "wins consistently." Pipeline:
- Days 1-2 (Tue-Wed): obs 47→53 + defensive teacher + mark/block gates.
- Days 3-4 (Thu-Fri): dataset re-gen + BC train.
- Days 5-6 (Sat-Sun): PPO Stage 8 (3v3 with full skill set).
- Days 7-8 (Mon-Tue): PPO Stage 8 continued, promote if metrics clear.
- Days 9-10 (Wed-Thu): Stage 9 (4v4) if Stage 8 promoted; otherwise extra Stage 8 time.
- Day 11 (Fri): stability + ONNX + demo.

Honest outcome at Friday 06-26: a policy that **scores against BH at rates comparable to or better than BH scores against us in selected scenarios**. "Beats BH 50%+ of matches" is a stretch; "competes with BH" is a fair claim.

**Interpretation C — late June / early July (~4 weeks).** Achievable as "wins consistently." Pipeline matches §11 Version A's full Phase 5-7 timing applied to the multi-agent substrate. The 8-10 weeks §11 originally estimated assumed starting from 26 dims with no multi-agent infrastructure; starting from the Monday 47-dim checkpoint cuts ~3-4 weeks of that.

#### 14.26.6 Recommended commitment

**Default plan: Interpretation B + honest scoping.** Tell stakeholders:

> By Friday 2026-06-26, an RL team that competes with BHuman in 4v4 SimRobot 3D demos — measurably better than the Monday demo, attempting defensive behaviors (mark, block), with team-goal-rate against BH in the 0.20-0.35 range. Consistent wins are a 3-week target, not a 2-week target.

If stakeholders demand Interpretation A, push back with §14.26.5 above and counter-offer: "Friday 06-19 we ship the substrate (53-dim obs + mark/block + defensive teacher), demo runs with BC-only behavior so it's qualitatively different from Monday but not measurably better in match performance. The 'beat' claim needs the 06-26 timing."

#### 14.26.7 Post-Monday day-by-day skeleton (Interpretation B)

| Day | Owner | Build | Validation gate |
|---|---|---|---|
| Tue 06-16 | Multi-agent owner + transplant owner | Wire opponent positions (Block H) from `ObstacleModel.obstacles` through `pybh_backend.py` to `environment.py`. Extend obs to 53 dims. Behavior-preservation test 47→53. | `test_obs_migration_contract.py:test_obs_padding_preserves_47_to_53` passes. Monday ONNX still loads + scores on baseline. |
| Wed 06-17 | Multi-agent owner | Defensive teacher branch (`teacher_defender`). Mark/block gates in `gate.py`. Extend `gate_skill_mask_from_obs` per §14.26.4. New role one-hot: 4 bits at 53/54/55/56 (rolling indices: was 44-46, now extend). **Note: this is a second role-mask expansion**, so the obs index plan is `is_striker, is_open_support, is_off_ball_support, is_defender` at 53/54/55/56 if Block H goes 47-52. Or move role bits to end. Implementer chooses; document the choice in test. | Teacher test cases for defender pass; mask test cases for mark/block pass. |
| Thu 06-18 | Multi-agent owner + Laura | Re-collect multi-agent dataset at 53 dims (or correctly-sized obs after role-bit relocation) with defensive scenarios in Stages 8+. Per-role stratification extended to 4 roles. Extend dataset audit. | Audit shows `positive_weight_role_mask_consistent: true` across 4 roles. |
| Fri 06-19 | Multi-agent owner | BC train from new dataset, warm-start from Monday checkpoint. (Per §11.B.5.3: KL anchor = MONDAY checkpoint, not original single-agent.) Initial behavioral_evaluator pass. **Demo-eligible BC-only stop point.** | Single-agent floor not regressed. Defensive role compatibility rate > 0 on a defensive scenario. |
| Sat 06-20 | Multi-agent owner | PPO Stage 8 (3v3 with full skill set including mark/block). | Stage 8 metrics tracked; per-eval auto-rollback on regression. |
| Sun 06-21 | Multi-agent owner | PPO Stage 8 continued. | Stage 8 promotion gate: team_goal_rate >= 0.15 vs full BH, opponent_intercept_prevented_rate >= 0.10. |
| Mon 06-22 | Multi-agent owner | PPO Stage 9 (4v4 full match) if Stage 8 promoted. Else extra Stage 8. | — |
| Tue 06-23 | Multi-agent owner | Continued. | — |
| Wed 06-24 | Multi-agent owner | Final PPO. | — |
| Thu 06-25 | Multi-agent owner + transplant owner | Stability probe + ONNX export at 53 dims. SimRobot 3D 4v4 demo dry-run. | ONNX loads on NAO under 50ms inference. |
| Fri 06-26 | All | "Compete-with-BH" demo. | — |

#### 14.26.8 What this roadmap does NOT solve

- **Score state in obs** (winning vs losing) — game-state-aware strategy (run-out-the-clock vs aggressive chase) requires `score_diff` + `time_remaining` in obs. Not in this roadmap.
- **Set plays** (free kicks, corner kicks, kick-offs) — game state branches require their own teacher logic. Defer.
- **Goalie as RL** — GK keeps being BH. Replacing GK is its own 2-3 week sub-project per §11.A.2 + GK-specific kinematics.
- **Pass-lane anticipation** (Policy 7 from §11.A.2) — requires predictive features in obs. Defer.

These are tracked for late-July / RoboCup-readiness work.

#### 14.26.9 Memory updates triggered by this roadmap

After Monday demo lands:
- Update memory `[[multiagent-rl-dominant-production-goal]]` with "Monday delivered" + measured team_goal_rate.
- Promote memory `[[obs-growth-zero-pad-pipeline]]` from 3-6 week estimate to "47→53 fits 2 days with proven substrate" based on Laura's 26→41 work as evidence.
- Create new memory `multiagent-defensive-obs-extension` capturing Block H spec, mark/block gating, defender role.
- Update `[[graph-coordination-pipeline-hostility]]` to mark Phase 1+2+3 complete after the 53-dim substrate lands.

This memory hygiene keeps future sessions from re-litigating decisions captured here.
