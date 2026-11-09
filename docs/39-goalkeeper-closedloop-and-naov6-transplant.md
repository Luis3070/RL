# Doc 39 — Goalkeeper: closed-loop RL build + NAOv6 C++ transplant

**Audience:** a junior engineer doing the C++ transplant with *no room for questions*.
**Scope:** everything done to produce the deployable RL goalkeeper, the exact obs/skill/mask/action
contract, and the step-by-step NAOv6 transplant. SimRobot is treated as an accurate physical/perceptual
proxy — do **not** add sim2real fudge factors; port the contract verbatim.

> The goalkeeper is a **separate network** from the merged striker/defender ONNX (doc 38) and the
> striker (doc 37). Different obs size (**64** vs 47), different skills (**12** vs 8), different mask.
> There is **no weight transfer** between them. The GK runs only on the keeper robot.

---

## 0. The deployable artifact

| | |
|---|---|
| ONNX | `ppo_goalkeeper_hsl2026_gk_closedloop_elite.onnx` |
| sha256 | `6475c4528f76ac8fffde07fed404a57e5d8452e9ee197821d07262597408ff62` |
| obs input | name `obs`, shape `[batch, 64]`, float32 |
| outputs | `skill_logits` `[batch, 12]`, `param_mean` `[batch, 4]` (both float32) |
| head | **HEAD-FREE** — no scripted post-processing inside or outside the graph |
| parity | max\|onnx−torch\| = 3.05e-5, argmax 100%, finite, 0 mask violations |

**This is the file to deploy.** Ignore the older GK ONNX in the tree
(`*_dribblefix_elite`, `*_elite_multiagent`, `*_ppo_validated`, `*_elite_candidate`, …); they are
earlier candidates kept only for provenance. The closed-loop file supersedes them.

### Closed-loop benchmark (analytical, 3600 episodes, deterministic argmax, head-free)
Overall save **93.3 %** across 20 scenarios. Highlights:

- **Dribble-into-goal** (the dominant real concession path): left/right/round-the-keeper **100 %**,
  off-center **94 %**, fast carry **81 %**. (The previous scripted-head GK was 98/98/**20/7/5.6** here.)
- **Distribution / idle:** controlled pass / clear / dribble-out / observe(search) / reposition all **100 %**.
- **Shots:** 12 of 15 scenarios at **96–100 %** (close-range 100, slow-roller 100, both posts 100,
  both low-finishes ≥99, cutbacks ≥99, one-v-one 99, screened 97).
- **Soft spots (known):** chip-lob 68 %, fast-blast 73 %, deflection-mid-flight 79 %. These are the
  rarer power/awkward shots; the trade is threat-aligned (most match goals are dribbles + short shots).
- **L/R symmetry:** learned (not scripted) — left == right on every mirror-paired scenario.

---

## 1. How it was built (pipeline — heuristic → dataset → BC → closed-loop PPO)

All synthetic (no SimRobot needed for training/eval; the closed-loop benchmark is analytical).

1. **Heuristic teacher** `goalkeeper_teacher.goalkeeper_teacher_action(obs)` — the expert. Decides
   stand / walk-to-pose / observe / the correct intercept (center/lateral/low/jump by predicted
   crossing) / clear / pass / dribble-out, all consistent with the legal mask.
2. **Dataset generation** `teacher_goalkeeper_dataset.py --synthetic --episodes 80 --steps 80`.
   Scenarios include the shot battery, the control/idle behaviors, and **dribble-into-goal**. The
   dribble scenarios were **broadened** so each episode is randomly *slow-central / fast (1000–1500 mm/s)
   / off-center (wide lane) / round-the-keeper (carry straight then late cut)* — labelled come-out
   (walk to a smother point) + clear when in reach.
3. **Mirror augmentation** `tools/dataset/mirror_augment_gk_dataset.py` — reflect every sample L↔R (negate the
   y-signed obs dims, swap near-left/right and the low/jump-left/right skills) so the corpus is exactly
   symmetric.
4. **Behavioral cloning** `train_goalkeeper_bc.py` — supervised warm-start of skill + param heads.
5. **Closed-loop PPO** `train_goalkeeper_ppo_closedloop.py` — **the key step.** Instead of rewarding
   "match the teacher's skill," it rolls out the *analytical attack episodes* (the same dynamics as the
   benchmark) with the policy sampling, and rewards **the goal actually being prevented**
   (+1 save / −1 goal / +0.3 forced-wide) plus a light crossing-gap shaping term. Two additions make it
   ship-quality:
   - **L/R symmetry loss:** MSE(logits(obs), skill-permuted logits(mirror(obs))) — drives the residual
     left/right asymmetry to zero.
   - **BC anchor on non-attack behaviors:** the attack-only rollout otherwise *catastrophically forgets*
     distribution (pass→clear, observe→walk). A cross-entropy/param anchor on
     `controlled_pass/clear/dribble`, `stale_search/no_team`, `misposition_recover`, `moving_away`,
     `outside_goal_corridor` keeps those at 100 %.
   Final recipe: `--updates 200 --envs 320 --dribble-frac 0.38 --anchor-coef 0.5 --sym-coef 0.5
   --lr 5e-5` warm-started from a from-scratch BC on the mirror dataset.
6. **Head-free export** `tools/export/export_goalkeeper_policy.py --no-head` (with `RL_GK_DISABLE_HEAD=1`).

> **Deprecated:** an earlier candidate (`*_dribblefix_elite`) used a *scripted slow-corner head* baked
> into the graph to patch slow-central dribbles. It is **coupled to the old weights** (it regresses
> shots on retrained weights) and only covers the slow-central regime. The closed-loop policy **learns**
> general dribble defense, so the head is removed. **Do not port the head.**

---

## 2. THE CONTRACT — must match exactly or the policy silently misbehaves

### 2.1 Observation vector — 64 floats (`goalkeeper_teacher.encode_goalkeeper_obs`)
Indices 0–37 are active features; **38–63 are zero-padded** (reserved). Constants:
`X = 4500 mm` (`ACTION.x_range`/half-field-x), `Y = 3000 mm` (`ACTION.y_range`/half-field-y),
`VEL = 2500 mm/s`, `TIME = 5000 ms`, `TTI = 4.0 s`, `ISECT = 1200 mm`
(`BALL_CATCH_MAX_WALK_DISTANCE_MM`), `FALL = 5.0`.

| idx | feature | formula (clip range) |
|----|---------|----------------------|
| 0  | robot_x | robot_x / X (−1,1) |
| 1  | robot_y | robot_y / Y (−1,1) |
| 2  | robot_theta | θ / π (−1,1) |
| 3  | ball_rel_x | ball_rel_x / X (−1,1) |
| 4  | ball_rel_y | ball_rel_y / Y (−1,1) |
| 5  | ball_end_rel_x | ball_end_rel_x / X (−1,1) |
| 6  | ball_end_rel_y | ball_end_rel_y / Y (−1,1) |
| 7  | ball_vel_x | ball_vel_x / VEL (−1,1) |
| 8  | ball_vel_y | ball_vel_y / VEL (−1,1) |
| 9  | time_since_ball_seen | ms / TIME (0,1) |
| 10 | natural_time_since_ball_seen | ms / TIME (0,1) |
| 11 | ball_seen_percentage | % / 100 (0,1) |
| 12 | ball_consistent_with_game_state | bool |
| 13 | goalie_error_x | (cognition, robot-frame) / X (−1,1) |
| 14 | goalie_error_y | / Y (−1,1) |
| 15 | goalie_error_theta | / π (−1,1) |
| 16 | goalie_x (target pose) | / X (−1,1) |
| 17 | goalie_y | / Y (−1,1) |
| 18 | goalie_theta | / π (−1,1) |
| 19 | rolling_towards_own_goal | bool |
| 20 | time_to_own_y_axis | tti_s / TTI (0,1); ∞→1.0 |
| 21 | intersection_y | crossing_y / ISECT (−1,1) |
| 22 | near_left_post | bool |
| 23 | near_right_post | bool |
| 24 | inside_penalty_area | bool |
| 25 | ball_controlled | bool |
| 26 | pass_options_count | n / 4 (0,1) |
| 27 | fresh_ball | bool |
| 28 | stale_ball | bool |
| 29 | team_ball_x − robot_x | / X (−1,1) |
| 30 | team_ball_y − robot_y | / Y (−1,1) |
| 31 | team_ball_age | ms / TIME (0,1) |
| 32 | team_ball_valid | bool |
| 33 | nearest_opponent_dist | mm / X (0,1) |
| 34 | nearest_opponent_front_dist | mm / X (0,1) |
| 35 | nearest_teammate_dist | mm / X (0,1) |
| 36 | motion ground_contact | bool |
| 37 | motion fall_state | state / FALL (0,1) |
| 38–63 | reserved | 0.0 |

### 2.2 The cognition the C++ must compute (obs[13..25,27,28])
These are derived in `goalkeeper_teacher.build_gk_cognition` / `_goalie_pose` / `_time_to_own_y_axis`.
**Port these functions verbatim.** Key constants (all overridable by `RL_GK_*` env, defaults shown):

- Goal geometry: `OWN_GOAL_X_MM=-4500`, `GOALIE_LINE_X_MM=-4300`, `OWN_GOAL_HALF_Y_MM=800`,
  `GOALIE_LINE_Y_LIMIT_MM=700`, `PENALTY_X_LIMIT_MM=-3900`, `PENALTY_HALF_Y_MM=1100`.
- `goalie pose` = BOB-line fallback: `gy = clip(ball_y · (|GOALIE_LINE−OWN_GOAL|/(ball_x−OWN_GOAL)),
  ±GOALIE_LINE_Y_LIMIT)`, on x=`GOALIE_LINE_X`, θ toward the ball. (Production may instead use
  B-Human's GoaliePoseProvider — but the **obs must be encoded from whatever pose the keeper is told to
  hold**, i.e. `goalie_error_*` = that target pose minus the actual robot pose in robot frame.)
- `time_to_own_y_axis`: in **robot frame**, with vx=ball_vel_x: if vx ≥ 0 or ball_rel_x < 0 → (∞, 0);
  else t = −ball_rel_x/vx, intersection_y = ball_rel_y + ball_vel_y·t.
- `rolling_towards_own_goal` = speed>80 mm/s AND ball_vel_x<0 AND |intersection_y|≤(OWN_GOAL_HALF_Y+250)
  AND t finite.
- `ball_controlled` = hypot(ball_rel_x, ball_rel_y) < `CONTROL_RADIUS_MM=450`.
- `fresh_ball` = natural_time_since_seen ≤ `FRESH_BALL_MAX_MS=150`; `stale_ball` = > `STALE_BALL_MS=1000`.
- `near_left/right_post` = within 450 mm of the respective post; `inside_penalty_area` from the limits above.

### 2.3 Skill order — 12 GK skills (`goalkeeper_action.GKSkillType`, logits index = enum value)
`0 stand, 1 walk_goalie_pose, 2 observe, 3 intercept_center, 4 intercept_lateral,
5 intercept_low_left, 6 intercept_low_right, 7 intercept_jump_left, 8 intercept_jump_right,
9 clear, 10 pass, 11 dribble_out`.

### 2.4 Legal-action mask — `goalkeeper_teacher.goalkeeper_skill_mask_np` (Design-A; **apply THEN argmax**)
Compute the mask from the **encoded obs** and set illegal logits to −∞ **before** argmax. Constants:
`STAND_RADIUS_MM=60`, `WALK_RADIUS_MM=125`, `GENUFLECT_RADIUS_MM=200`, `JUMP_RADIUS_MM=600`,
`INTERCEPT_TIME_MIN_S=0.02`, `INTERCEPT_TIME_MAX_S=3.0`, `KEEPER_JUMPING_ON=1`.

- `stand`, `walk_goalie_pose`: always legal.
- `observe`: legal iff `stale_ball` (obs[28]≥.5) **or** `team_ball_valid` (obs[32]≥.5).
- Let `y = obs[21] · 1200` (de-normalized crossing). `intercept_base` = `rolling`(obs19)
  AND `t_min/4 ≤ obs[20] ≤ t_max/4` (timing window in obs's /4s units) AND `fresh`(obs27)
  AND `stable` (obs[37] ≤ 0.4, i.e. not falling).
  - `intercept_center`: base AND |y| ≤ 60.
  - `intercept_lateral`: base AND |y| ≤ 125.
  - `intercept_low_left`: base AND 0 < y ≤ 200; `intercept_low_right`: base AND −200 ≤ y < 0.
  - `intercept_jump_left`: jumping-on AND base AND inside_penalty AND 0 < y ≤ 600 AND not near_left_post.
  - `intercept_jump_right`: jumping-on AND base AND inside_penalty AND −600 ≤ y < 0 AND not near_right_post.
- `clear`, `pass`, `dribble_out`: legal iff `ball_controlled` (obs[25]≥.5).

> The **intercept timing window is in obs[20]'s normalized units** (seconds ÷ 4). Do not compare raw
> seconds to obs[20]. Getting this wrong silently disarms late close-range saves (the original
> shipped-mask bug).

### 2.5 Action decode — `param_mean` [4] → field targets (`goalkeeper_action`)
The 4 params are `[target_x, target_y, target_theta, (unused)]` in **[−1,1]**, active per skill via
`GK_ACTION_PARAM_MASKS`. Denormalize only the active ones:

- `target_x = param[0] · 4500` mm, `target_y = param[1] · 3000` mm, `target_theta = param[2] · π` rad.
- Active params per skill: `walk_goalie_pose`→(x,y,θ); `observe`→(x,y); every `intercept_*`→(y only);
  `dribble_out`→(x,y); `stand`/`clear`/`pass`→none.
- `pass_target` is **not** policy-controlled (param head is masked off for `pass`); the runtime/bridge
  chooses the teammate. Do not optimize it as a continuous residual.

---

## 3. C++ transplant — step by step (NAOv6)

**T.0 — Load & run ONNX.** Load `ppo_goalkeeper_hsl2026_gk_closedloop_elite.onnx`. Per frame on the
keeper: build the 64-float `obs`, run the graph → `skill_logits[12]`, `param_mean[4]`.

**T.1 — Build obs (§2.1).** Feed perception: robot pose, ball relative pos/end-pos/velocity, ball
seen%/age/consistency, team ball, nearest opponent/teammate distances, motion ground-contact/fall-state.
Compute the cognition block (§2.2) by porting `build_gk_cognition` / `_goalie_pose` /
`_time_to_own_y_axis` verbatim. Normalize exactly per the table. Zero indices 38–63.

**T.2 — Mask then argmax (§2.4).** Build the legal mask from the encoded obs, set illegal logits to
−∞, take `argmax` → skill id. (Deterministic; no sampling at deploy.)

**T.3 — Decode params (§2.5).** Denormalize the active params for the chosen skill.

**T.4 — Map skill → keeper behavior / SkillRequest.** Bind each GK skill to the corresponding
SabanaHerons/B-Human keeper motion (names below are the *intended behavior*; bind to your existing
keeper skills):

| GK skill | behavior to issue |
|----------|-------------------|
| stand | hold stance, head active (set-position idle) |
| walk_goalie_pose | walk to `(target_x, target_y, target_theta)` — reposition on/near the goal line |
| observe | hold + active head scan (ball lost / searching) |
| intercept_center | block centrally at the line (genuflect/stand-wide, central) |
| intercept_lateral | small lateral step to block within ±125 mm |
| intercept_low_left / _right | genuflect / spread low to that side (≤200 mm) |
| intercept_jump_left / _right | dive to that side (≤600 mm) — B-Human GoalkeeperDive / KeeperJump |
| clear | kick the controlled ball clear upfield |
| pass | pass to a teammate (runtime/bridge picks the target) |
| dribble_out | dribble the controlled ball out toward `(target_x, target_y)` |

For intercepts, the **side and reach** are already implied by the chosen skill (the mask only exposes
the skill whose reach band matches the predicted crossing `y`); the `target_y` param is the fine aim.

**T.5 — Parity gate before trusting the port.** Feed ≥256 identical obs vectors through the C++ ONNX
path and the Python reference (`tools/export/export_goalkeeper_policy.py` already verifies torch↔onnx at
3e-5). Require: argmax identical on every vector, max\|logit\| diff < 1e-4, all outputs finite, **zero**
cases where argmax picks a masked-illegal skill.

---

## 4. Pitfalls (read before you ship)

1. **Head-free.** Do not add any scripted slow-corner head. Dribble defense is in the weights.
2. **Mask BEFORE argmax** — always. Raw argmax collapses to `intercept_center`.
3. **Intercept timing is in /4-second units** (obs[20]); compare the window in the same units (§2.4).
4. **`stable` gate = fall_state ≤ 0.4** — a falling/unstable keeper must not be issued an intercept.
5. **`KEEPER_JUMPING_ON`** gates dives entirely; if your hardware disables diving, set it off and the
   policy degrades gracefully to low/lateral blocks.
6. **Zero-pad 38–63.** The contract is 64 even though only 38 are active. Do not shrink the input.
7. **The benchmark is analytical.** Save outcomes assume reach windows (`_save_radius` /
   `_dribble_engage_reach`: center 85/150, lateral 145/200, low 240, jump 650/300 mm; ttc ≤ 1.15 s).
   Before treating 93 % as a real-robot number, **calibrate these windows against B-Human's actual GK
   dive/genuflect coverage** on NAOv6. This is the one real-robot validation still owed.

---

## 5. Reproduce / validate

```bash
export PYTHONPATH=/path/to/RL:/path/to/SabanaHerons2026/Make/Python
PY=~/anaconda3/envs/RLNAO/bin/python

# dataset (broadened dribble modes) -> mirror -> BC -> closed-loop PPO
$PY teacher_goalkeeper_dataset.py --synthetic --episodes 80 --steps 80 --output-dir <abs>/ds
$PY tools/dataset/mirror_augment_gk_dataset.py --dataset <abs>/ds/samples.npz --output-dir <abs>/ds_mirror
$PY train_goalkeeper_bc.py --dataset <abs>/ds_mirror/samples.npz --epochs 32 --device cuda:0
$PY train_goalkeeper_ppo_closedloop.py --bc-checkpoint <bc.pt> \
   --anchor-dataset <abs>/ds_mirror/samples.npz --anchor-coef 0.5 \
   --updates 200 --envs 320 --dribble-frac 0.38 --sym-coef 0.5 --lr 5e-5 --device cuda:0

# export head-free + benchmark head-free
RL_GK_DISABLE_HEAD=1 $PY tools/export/export_goalkeeper_policy.py --checkpoint <ckpt.pt> --no-head \
   --output models/goalkeeper/ppo_goalkeeper_hsl2026_gk_closedloop_elite.onnx
RL_GK_DISABLE_HEAD=1 $PY tools/eval/benchmark_goalkeeper_closed_loop_shots.py --policy ppo \
   --checkpoint <ckpt.pt> --episodes 3600 --seed 742619 --device cpu
# production sanity (clean+noisy skill acc, mask violations, finite, diversity)
$PY tools/eval/validate_goalkeeper_policy_production.py --checkpoint <ckpt.pt> --dataset <ds_mirror>/samples.npz
```

---

## 6. Files in this commit

- `ppo_goalkeeper_hsl2026_gk_closedloop_elite.onnx` — the deployable, head-free GK.
- `goalkeeper_action.py` — `GKSkillType` (12), `GK_ACTION_PARAM_MASKS`, `GKStructuredAction` (action decode).
- `goalkeeper_teacher.py` — obs contract (`encode_goalkeeper_obs`, 64-dim), cognition
  (`build_gk_cognition` + geometry), legal mask (`goalkeeper_skill_mask_np/torch`), heuristic teacher.
- `teacher_goalkeeper_dataset.py` — synthetic dataset gen incl. broadened dribble modes.
- `tools/dataset/mirror_augment_gk_dataset.py` — L/R symmetric augmentation.
- `train_goalkeeper_bc.py` — behavioral cloning.
- `train_goalkeeper_ppo.py` — offline imitation PPO + `_apply_slow_corner_dribble_head`
  (deprecated; `RL_GK_DISABLE_HEAD=1` switch).
- `train_goalkeeper_ppo_closedloop.py` — **the closed-loop goal-prevention PPO** (reward + symmetry + anchor).
- `tools/export/export_goalkeeper_policy.py` — ONNX export (`--no-head`) + parity manifest.
- `tools/eval/benchmark_goalkeeper_closed_loop_shots.py` — analytical save benchmark (shots + dribble + stress).
- `tools/eval/validate_goalkeeper_policy_production.py` — sanity/diversity/robustness gate.
- `docs/39-goalkeeper-closedloop-and-naov6-transplant.md` — this document.

---

## 7. Open items (honest)

1. **3 hard shots** (chip 68 %, fast-blast 73 %, deflection 79 %) — next polish is to oversample them in
   the closed-loop rollout to lift toward 90 % without losing dribble/distribution gains.
2. **Reach-window calibration** — confirm `_save_radius` / `_dribble_engage_reach` against real NAOv6
   keeper motion (§4.7) before treating the analytical save-rate as the on-robot number.
3. **Multiagent passing** — `pass_target` is runtime-chosen today; a learnable normalized pass target
   (and per-teammate obs) is the path to build-from-the-back distribution.
