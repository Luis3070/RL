# 38 — Merged striker+defender brain, and the NAOv6 production transplant

Date: 2026-06-19. Artifact: **`ppo_team_hsl2026_v5_merged.onnx`** (one 47-dim role-conditioned
policy = striker + open_support + off_ball_support/defender). Read **doc 37 first** (the striker
contract — obs/gate/role/action layers); this doc adds the defense skills, the merge method +
results, and the **complete multi-agent deployment** on NAOv6. Nothing here is optional: every
item is required to reproduce in production what we got in SimRobot.

> SimRobot is treated as an accurate physical + perceptual proxy. A *numerically faithful* port
> (parity checklist §6) reproduces the measured behavior on hardware. The failure mode is not
> sim2real — it is a C++/Python contract drift that runs but behaves wrong, silently.

---

## 1. What the merged brain is

One ONNX, one weight set. **Role is an INPUT** (`obs[44:46]` one-hot), not an output. A rule-based
**coordinator** elects each robot's role every step and writes the one-hot; the shared policy
conditions on it. This is CTDE: *centralized role election + a single decentralized role-conditioned
policy*. Proven by the role-conditioning test — flipping the role bit on identical world obs flips
the behavior striker↔defense (defense-obs→off_ball: mark 61%).

Skills by role (gate-enforced): striker = walk/dribble/shoot/pass; open_support = walk(+dribble/block
escape); off_ball_support/defender = walk/block/mark(+dribble escape). `stand`/`walk` always legal.

## 2. How it was built (merge fine-tune)

1. **GT opponent-obs training flag** (`RL_TRAIN_GT_OPPONENT_OBS`, `environment.py`, default OFF,
   training-only): fills `obs[18]/obs[21]` from ground-truth opponent positions during collection +
   PPO. Wired into BOTH the PPO env (reset world_spec) and the BC collector (encoder env world_spec
   in `teacher_multiagent_dataset._collect_team_scenario`). Reason: in-sim off-ball opponent perception
   collapses (gaze/detector geometry), which would starve defense learning. **At deploy this is OFF —
   `obs[21]` comes from real vision**; the obs *contract* is identical, only the training-time source
   differs. The defense regime is close-range (mark gate < 1.5 m), exactly where real vision works
   (validated: roomy single-slot real-vision opponent perception 0.43; mark gate arms correctly).
2. **Defense BC corpus:** `tools/dataset/collect_defense_gt.sh` over the 5 stable stage-8 scenarios
   (`defense_3v3_full` EXCLUDED — it crashes the visible sim) → `dataset_stage8_20260618_211245`
   (33 999 rows, mark 673 / block 2817, roles balanced).
3. **Merge:** `merge_multiagent_teacher_datasets.py` UNION attack corpus
   `dataset_stage7_20260611_182346` (105 624 rows, pass 2015 / shoot 1677) → `dataset_merged_v5_20260618`
   (139 623 rows, ~76:24 attack:defense).
4. **Fine-tune:** `tools/run/train_merge_v5.sh` = `train_multiagent_production.py --stage-id 8`,
   **warm-start AND KL anchor = v4.2 step12000** (the KL ref is the warm-start ckpt, `train.py:2102`),
   dataset = merged_v5, BC β 0.08→0.02, **single slot** (stage-8 default = stable; 3-slot starves),
   PPO scenario weights exclude `defense_3v3_full`, 10 000 steps → `checkpoint_stage8_step10000.pt`.
5. **Export:** `tools/export/export_ppo_policy_team.py --active-skills stand,walk,dribble,shoot,pass,block,mark
   --specialization merged --output models/team/ppo_team_hsl2026_v5_merged.onnx`.

## 3. Results (how it was validated — the visible 3-slot eval is unusable, so OFFLINE + single-slot)

- **Attack preserved:** merged vs v4.2 skill agreement **96.7%** (105k rows); pass/shoot/dribble recall
  **100 / 99.9 / 96.4%** (identical to v4.2).
- **Defense added:** mark recall **93.2%** (≈ pt_def47 95.2%); in-sim defender_score **0.76**, goal_side 0.90.
- **Multi-agent match (shared-sim harness, 22 ep):** role co-occurrence **100%** (team always holds all
  3 roles), role share 33/33/33, **live off-ball mark 63%**, opp perception 81%.
- **Sanity:** 0 gate violations, deterministic, params bounded. **Robustness:** input-noise σ0.05 → 4.5%
  skill-flip (linear, no cliff). **block↔mark wrapper** (#3): block 11→26% when both armed and the ball
  is the nearer threat.

Known gaps (do NOT regress-test against these in prod, they are sim-harness limits): live passing &
goals-conceded unmeasured (the 3-opponent contested scene crashes the visible sim — needs a headless
harness); defense is BC-level, not PPO-refined; off-ball ball_seen low (29%, gaze).

## 4. NAOv6 production transplant — the COMPLETE contract

Everything in **doc 37 §B (B.0–B.7) applies unchanged** (same obs47, same ONNX I/O signature, same
normalization, same residual→absolute math). The merged ONNX only differs in the active-skills set
and the role-conditioned gate. Below is **what doc 37 did not cover** and is mandatory for the merged
multi-agent deployment.

### 4.0 The complete per-frame inference loop (read this first)
Run this **once per cognition frame, per RL field robot** (the GK runs its own separate policy,
§4.6). `obs_raw` = the existing B-Human world model (pose, ball, obstacles, team comm).
```
# --- once per frame, shared across the 3 RL robots (team-level) ---
team_state  = update_team_state(all_rl_teammate_poses, team_ball)      # team ball + histories
roles       = assign_roles(team_state, agent_obs, history)             # §4.2  -> {robot: role}
coord_tgts  = coordination_targets(roles, team_state)                  # §4.3  EMA open-lane/triangle/cover

# --- per robot ---
role      = roles[this_robot];  coord = coord_tgts[this_robot]
gate_bits = SkillGate.step(obs_raw)             # §4.4  arming -> shoot/dribble/pass armed + progress
obs47     = encode(obs_raw, gate_bits, onehot(role), coord)            # §4.1 + doc 29  (47 floats)
logits, residual = ONNX.run(obs47)                                     # skill_logits[8], param_mean[4]
mask      = gate_skill_mask_from_obs(obs47)                            # §4.4  role-conditioned forbid-invalid
skill     = argmax(logits + (mask - 1)*1e9)                            # deterministic gated argmax
skill     = block_priority_override(skill, mask, obs47)   # §4.8  OPTIONAL (recommended ON)
absolute  = decode_residual(skill, residual, obs_raw, role, coord)     # §4.3  role-aware anchor + residual
absolute  = project_action(absolute, active_skills)                   # §4.9 step 2
absolute  = repair_targets(absolute, obs_raw, role, coord)            # §4.9 step 3  (LOAD-BEARING)
absolute  = collision_avoid(absolute, obs_raw)            # §4.8  OPTIONAL
emit SkillRequest(skill, absolute)                                     # §4.5
```
Every box maps to a section below. Skip none. The two `OPTIONAL` boxes are wrappers (§4.8);
everything else is mandatory.

### 4.1 ONNX
`ppo_team_hsl2026_v5_merged.onnx`, onnx sha256 `abee6a5cff3966d6…`, ckpt sha256 `81e5555c4d49303e…`
(manifest `ppo_team_hsl2026_v5_merged.json`). Same I/O as v4.2 (`obs[1,47]`→`skill_logits[1,8]`,
`param_mean[1,4]`). 8 skill logits; `active_skills = {stand,walk,dribble,shoot,pass,block,mark}`.

### 4.2 The role coordinator MUST run on-team in C++ (this is the cooperation)
Port `multiagent_teacher.assign_roles` into `StrategyBehaviorControl` (runs every cognition frame):
- Inputs: each RL teammate's pose + the **team ball** estimate (already in `obs[32:34]` / team
  communication). GK is excluded from the election.
- **Striker** = min distance to team-ball (tie-break lowest player number).
- **Ball-control override (D1):** any robot within `BALL_CONTROL_RADIUS_MM` of the ball AND fresh ball
  → becomes striker immediately (so a defender who wins the ball flips to striker → clears/passes out).
- **TTRB hysteresis:** a striker flip requires a sustained distance gap (`ROLE_MIN_DISTANCE_GAP_MM`)
  held for `ROLE_HYSTERESIS_FRAMES` frames — prevents role thrash (constants in `multiagent_teacher.py`).
- **Phase (attack/defense)** from ball position/possession; in **defense** the deepest field player
  (min x) = `off_ball_support` (the defender), the rest `open_support`; in attack `open_support` =
  best forward rating, rest `off_ball_support`.
- Write the resulting one-hot into `obs[44:46]` for each robot before its inference.

### 4.3 Coordination targets MUST be computed (support/defender anchors)
The residual→absolute conversion for **support/defender WALK** is NOT ball-anchored — it is anchored on
an EMA coordination target. Port these (`multiagent_teacher.py`): `build_open_lane_anchor` /
`compute_open_lane_target` (open_support), `compute_triangle_target` (off_ball attack),
`defender_cover_target` (off_ball defense), each EMA-smoothed (`EMA_ALPHA_OPEN_LANE`,
`EMA_ALPHA_TRIANGLE`). Then in `_decode_residual_policy_action`: if role ∈ {open_support,
off_ball_support} and skill == walk → `anchors = build_open_lane_anchor(obs, coordination_target)`,
else `build_fallback_action_targets(skill, obs)`. **If you skip this, support/defender walks decode
ball-anchored and the robot freezes / walks at the ball instead of its slot** (the doc-28 "walk chosen
but frozen" failure). This was the load-bearing fix for triangle/cover hold.

### 4.4 Gate mask — role-conditioned (`gate_skill_mask_from_obs`), MANDATORY, apply before argmax
Port exactly (it is a pure function of the obs). Per role:
- **striker:** shoot iff `obs[23]`; pass iff `obs[41]`; dribble iff `obs[24]`; block/mark/observe forbidden.
- **off_ball_support (defender):** `block` iff `block_threat = 0 < obs[3] (ball_rel_x) < 1800/4500`
  OR `dribble_armed`; `mark` iff `mark_threat = 0 < obs[21] (opp_front) < 1500/4500`; dribble iff
  `dribble_armed` (escape); observe iff `obs[42]`; shoot/pass forbidden.
- **open_support:** block iff `dribble_armed`; observe iff `obs[42]`; (mostly walk/positioning).
- `stand`/`walk` always allowed for every role. Thresholds: **block 1800 mm, mark 1500 mm**, both
  normalized by `X_RANGE=4500`. Shoot band 2.5–3.0 m, `shoot_open enter 0.30 / exit 0.20`.

### 4.5 Defender skills → B-Human SkillRequests (NEW vs striker)
`block` and `mark` must map to real B-Human behaviors in `RLSkillProvider`:
- **block** = position the robot on the ball→own-goal line to intercept (a goal-side intercept walk to
  the computed block point). target from the action conversion.
- **mark** = goal-side man-mark of the nearest front opponent (stand between opponent and own goal,
  facing the opponent). The action's `target` is the mark spot; head looks at the mark target.
If these SkillRequests don't exist yet in C++, add them (they're the only new motor behaviors vs the
striker deploy). Until then, block/mark fall back to walk-to-target.

### 4.6 GK is the one exception
The merged brain is for the 3 **field** RL players. The goalkeeper is separate (own policy, doc TBD;
B-Human GK is being replaced — see the GK candidate). The GK reads `false` on all three role bits from
the RL side and is never elected striker by the field coordinator. Do not feed GK obs to this ONNX.

### 4.7 GT flag OFF at deploy
`RL_TRAIN_GT_OPPONENT_OBS` is training-only. In production `obs[18]/obs[21]` come from the real
`ObstacleModel` (vision). The encoder is identical; nothing to port for this — just do NOT inject GT
opponents at runtime (that would be the rejected sim2real shortcut, and it's unnecessary on hardware).

### 4.8 Optional inference wrappers (flag-gated; deploy alongside, ABOVE the ONNX argmax)
Validated this session, port if desired (they are thin post-processing, not retraining):
- **block-priority (#3):** if role==off_ball and argmax==mark and block is gate-allowed and
  `obs[3] (ball_front) < obs[21] (opp_front)` → choose block. Measured block 11→26%, mark 50→33%,
  ball progression +206→+583 mm/ep. Recommended ON.
- **collision-avoid (#4):** if skill∈{walk,dribble} and nearest opponent < ~500 mm, deflect the target
  laterally away from the opponent bearing. Built; benefit on opponent-pushing unmeasured (teammate
  collisions were already low). Optional.

### 4.9 Execution-path order + repair (LOAD-BEARING — was missing, do not skip)
`env.step` applies a fixed sequence after the policy emits (skill, residual). The C++ MUST
reproduce this exact order, or support/defender walks freeze (the triangle/cover-hold failure):
1. `_decode_residual_policy_action(skill, residual, obs, role, coord)` → absolute target (§4.3).
2. `project_action`: if the chosen skill is not in the stage's active-skill set, **project to `walk`**
   with `build_fallback_action_targets("walk", obs)` (telemetry honesty; rarely fires at deploy).
3. **`_repair_execution_targets` (always-on, load-bearing):** for `walk`/`dribble` only, recompute the
   role-aware anchor (`build_open_lane_anchor(obs, coord)` for support/defender walk, else
   `build_fallback_action_targets`). If the decoded target is **farther than the residual radius**
   from that anchor — `radius = hypot(RESIDUAL_X_RANGE, RESIDUAL_Y_RANGE)*1.05`, `theta_err ≤
   RESIDUAL_THETA_RANGE*1.05` — **snap the target back to the anchor**. This catches a policy target
   that drifted out of the legal residual ball (the closed-loop cause of `triangle_hold=0.000`
   before it was added). Without it, supports walk to garbage targets and stall.
4. Execute the resulting absolute action as the SkillRequest (§4.5).

### 4.10 Port-verbatim checklist (so NOTHING is lost)
The docs give the *contract*; these Python functions carry algorithms that must be ported **verbatim**
(not paraphrased) and parity-checked, because their exact numerics are part of the trained behavior:
- `gate.py::SkillGate` — the shoot/dribble **arming state machine** (hysteresis enter 0.30/exit 0.20,
  2.5–3.0 m band, `shoot_arm_progress` ramp, geometric takeover) → produces `obs[23:25]`.
- `multiagent_teacher.smash_or_pass` — striker `pass_armed` (`obs[41]`) + which outlet.
- `multiagent_teacher.assign_roles` (+ `TeamRoleHistory`, constants `BALL_CONTROL_RADIUS_MM`,
  `ROLE_MIN_DISTANCE_GAP_MM`, `ROLE_HYSTERESIS_FRAMES`) — role election (§4.2).
- `multiagent_teacher`: `compute_open_lane_target`, `compute_triangle_target`, `defender_cover_target`,
  `build_open_lane_anchor`, `build_triangle_anchor` + EMA alphas — coordination targets (§4.3).
- phase (attack/defense) selection: ball-x / possession test that picks the support layout (§4.2).
- `environment.py`: `_decode_residual_policy_action`, `project_action`, `_repair_execution_targets`,
  `build_fallback_action_targets`, `_wrap_angle` — the full action path (§4.3, §4.9).
- `train.py::gate_skill_mask_from_obs` — the role-conditioned forbid-invalid mask (§4.4).
- `observation.py::ObsIndex` + doc 29 encoder — the 47-dim obs.
- **Constants:** `X_RANGE=4500, Y_RANGE=3000, BALL_VEL_NORM=3000, TIME_NORM_MS=5000,
  RESIDUAL_X_RANGE=0.0, RESIDUAL_Y_RANGE=450.0, RESIDUAL_THETA_RANGE=0.75`; `ACTION_PARAM_MASKS` per skill.
- `pass_target` decode: the 4th param → teammate selection. **Currently approximate** (round to a
  teammate slot); a clean deploy should map it through the same teammate-slot order the encoder used
  (`build_agent_id_map`), or keep pass_target heuristic until a normalized target is trained.
Anything not in this list is either in doc 37 §B or doc 29. If a function here is paraphrased instead
of ported, treat it as a transplant risk and parity-check it explicitly.

## 5. Files to deploy together
`ppo_team_hsl2026_v5_merged.onnx` + its `.json` manifest. Python references to mirror in C++:
`observation.py` (obs47), `gate.py` + `train.py::gate_skill_mask_from_obs` (gate bits + mask),
`multiagent_teacher.py` (assign_roles + coordination anchors), `action.py` +
`environment.py::_decode_residual_policy_action` (residual→absolute, param masks, ranges). Doc 29 =
the 26→47 encoder plumbing.

## 5b. Common transplant pitfalls (every one of these has bitten us)
- **Forgot the gate bits in the obs** (`obs[23:25]`, `obs[41:43]` left 0): the policy was *trained*
  with them; zeros → it never shoots/passes. They are encoder outputs, not optional.
- **Applied argmax without the mask:** the GK case proved this — raw argmax collapses to one skill
  (intercept_center). Always `logits + (mask-1)*1e9` then argmax.
- **Decoded support/defender walk ball-anchored:** §4.3 — must use `build_open_lane_anchor(obs, coord)`,
  else the robot walks at the ball / freezes (triangle_hold=0).
- **Skipped `repair_targets`:** §4.9 — drifted targets never snap back → supports stall.
- **`residual_x_range` treated as nonzero:** it is **0.0**. `target_x` comes entirely from the anchor.
- **Role one-hot all-zero or two-hot:** exactly one bit set per RL robot. All-zero triggers the
  single-agent fallback gate (wrong for a field robot). The GK is all-zero *on purpose* (it's not this policy).
- **Wrong ONNX file:** use `ppo_team_hsl2026_v5_merged.onnx` (sha `abee6a5c…`). Verify the sha256.
- **Coordinator not run / run per-robot inconsistently:** `assign_roles` must run once on shared team
  state so the 3 robots see a consistent role assignment (no two strikers).
- **Angle wrap:** `target_theta` must be wrapped to (-π, π] (`_wrap_angle`).

## 6. Parity checklist (gate before trusting hardware)
Same 4 checks as doc 37 §B.7, **per role**: run each of the 3 roles through (encoder→ONNX→gate→argmax
→residual decode) and match Python to <1e-4 (logits), exact (argmax), <1 mm/<1e-3 rad (targets).
Additionally verify the **role coordinator** in C++ produces the same one-hot sequence as
`assign_roles` on a recorded 3-robot trajectory (role co-occurrence should be ~100%, no thrash).
