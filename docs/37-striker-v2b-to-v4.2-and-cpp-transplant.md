# 37 — Striker v2b → v4.2, and the C++ transplant of the striker agent

Date: 2026-06-19. Scope: the **striker/team policy only** (`ppo_team_hsl2026_v4_2.onnx`),
NOT the merged defender brain (see doc 38). This documents (A) how v2b became v4.2 and
(B) exactly what C++ must replicate to run v4.2 on a NAOv6 with identical behavior.

> Canonical Python reference for every numeric contract below: `RL/environment.py::_encode_obs`
> (observation), `RL/train.py::gate_skill_mask_from_obs` (mask), `RL/environment.py::_decode_residual_policy_action`
> (action). The C++ must be **numerically equivalent** to these. The 26→47 obs encoder is
> already specified for Rafael in **doc 29**; this doc adds the gate, role, and action layers.

---

## A. How v2b became v4.2 (striker)

All steps are stage-7 (3 RL field + 1 BH GK), 47-dim obs, `SKILL_ORDER` unchanged.

| Stage | Artifact | What changed |
|-------|----------|--------------|
| **v2b (base)** | `runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt` | team-reward MAPPO; competent walk/dribble but pass rare at eval. The canonical warm-start + KL anchor for everything after. |
| **Gate fix** | `train.py::gate_skill_mask_from_obs` | The "gate mark-leak" bug: the BC label filter was a hand-mirror that silently dropped `pass`/`mark` labels. Replaced by calling the REAL gate on the encoded obs (single source). This *revealed* that v2b/v3.1 already passed ~99% under the corrected mask. |
| **v3 / v3.1** | `dataset_multiagent_v3_attack`, `dataset_v31_backbone_plus_overlay` | Contrastive **give-and-go / open-support** scenarios added as a BC overlay on a v2b subsample backbone (`run_v3_attack.sh`, `run_v31.sh`). Teaches the striker to release the ball to a forward outlet. |
| **v4** | `dataset_v4_backbone_plus_overlay` | Pass-rich overlay (`giveandgo_outlet_present`, `pass_forcing`) merged onto the v2b subsample (`run_v4.sh`). |
| **v4.2 (final)** | `runs/multiagent_v4_2/checkpoint_stage7_step12000.pt` → `ppo_team_hsl2026_v4_2.onnx` | `train_multiagent_production.py --stage-id 7`, **warm-start + KL anchor = v2b@7500**, dataset = `dataset_v4_backbone_plus_overlay`, **BC β 0.30→0.05**, 12 000 steps, eval/ckpt every 3 000. PPO scenario pool weights: `team_3v0_clean:1, team_3v0_static_pinch:2, team_3v0_dynamic:2, team_3v3_full_dynamic:3, preshoot_bad_angle_boundary:5`. Result: **pass ≈ 0.99, eval floor 4/5**. |

Key levers that mattered: (1) the **corrected gate** (without it pass is invisible), (2) the
**pass-rich contrastive overlay** unioned with a v2b backbone (don't train pass from scratch),
(3) **strong BC (0.30) + KL anchor to v2b** so attack imitation stays alive while PPO explores.

Reproduce: `bash tools/run_v4.sh` (build the dataset) then `bash tools/run_v4_2.sh` (train + the
export is `tools/export_ppo_policy_team.py --specialization team`).

---

## B. C++ transplant of the striker (v4.2)

The striker is the merged brain's *striker role*; this section is the subset that doc 38 reuses.
Five layers must all match Python. **If any one drifts, behavior breaks silently** (the policy
still runs, just wrong).

### B.0 ONNX I/O contract
- File: `ppo_team_hsl2026_v4_2.onnx`, sha256 `1de359e735d997ea…` (manifest `ppo_team_hsl2026_v4_2.json`).
- Input: `obs` float32 `[1, 47]`. Outputs: `skill_logits` `[1, 8]`, `param_mean` `[1, 4]` (already `tanh`-ed).
- `SKILL_ORDER` (logit index → skill): `0 stand, 1 walk, 2 shoot, 3 pass, 4 dribble, 5 block, 6 mark, 7 observe`.
- Deterministic: inference = **gate-mask then argmax** over skill_logits (no sampling at deploy).
- `param_mean` = **residual** action `[res_x, res_y, res_theta, pass_target]` in `[-1,1]`, NOT absolute targets.

### B.1 Observation (47 floats) — doc 29 is the spec
Build the 47-vector exactly per `_encode_obs`. Normalization constants (identical to Python):
`X_RANGE=4500, Y_RANGE=3000, BALL_VEL_NORM=3000, TIME_NORM_MS=5000`. Indices: see `observation.py::ObsIndex`
(robot pose 0-2, ball 3-12, scoring 13-16, proximity 17-22, **gate bits 23-25**, multi-agent block
26-40, **pass/observe arm bits 41-43**, **role one-hot 44-46**). Doc 29 covers 26→47 plumbing.

### B.2 Gate bits baked into obs (23-25, 41-43) — Design-A, MANDATORY
The forbid-invalid mask is a pure function of the stored obs, so these bits MUST be computed in
the encoder before inference (they are inputs the policy was trained on):
- `obs[23] shoot_armed`, `obs[24] dribble_armed`, `obs[25] shoot_arm_progress` — from `SkillGate`
  (gate.py): shoot arms on `shot_opening_with_obstacles` crossing **enter 0.30 / exit 0.20**
  (hysteresis) within the **2.5–3.0 m** band; dribble arms on geometric ball takeover.
- `obs[41] pass_armed`, `obs[42] observe_armed`, `obs[43] pass_arm_progress` — striker `pass_armed`
  is set when the striker has ball control (shoot OR dribble armed) AND `smash_or_pass` prefers a
  forward outlet (`multiagent_teacher.smash_or_pass`).

### B.3 Role one-hot (44-46) — the coordinator, NOT the network
Exactly one of `is_striker/is_open_support/is_off_ball_support` is 1.0 per RL agent per step. The
**C++ StrategyBehaviorControl runs the role election** (`multiagent_teacher.assign_roles`) and writes
the one-hot. Election: striker = closest to team-ball (+ ball-control override: whoever is within
ball-control radius becomes striker immediately) + TTRB hysteresis (sustained distance gap over N
frames before flipping). For a striker-only deploy, force `is_striker=1`.

### B.4 Gate mask (apply BEFORE argmax) — `gate_skill_mask_from_obs`
For the striker role: `stand`/`walk` always allowed; `shoot` iff `shoot_armed`; `pass` iff
`pass_armed`; `dribble` iff `dribble_armed`; `block`/`mark`/`observe` forbidden for striker.
Forbidden skills get logit `-1e9` (or set prob 0) so argmax never picks them.

### B.5 Residual → absolute action — `_decode_residual_policy_action`
The network outputs a residual; convert to an absolute target the same way:
```
res_x = clip(param[0],-1,1) * param_mask[0]   # NOTE residual_x_range = 0.0  → x is anchor-only
res_y = clip(param[1],-1,1) * param_mask[1]
res_theta = clip(param[2],-1,1) * param_mask[2]
anchors = build_fallback_action_targets(skill, obs)        # ball-anchored for striker skills
target_x = clip(anchors.x + res_x*RESIDUAL_X_RANGE, -4500, 4500)   # RESIDUAL_X_RANGE=0.0
target_y = clip(anchors.y + res_y*RESIDUAL_Y_RANGE, -3000, 3000)   # RESIDUAL_Y_RANGE=450.0
target_theta = wrap(anchors.theta + res_theta*RESIDUAL_THETA_RANGE) # RESIDUAL_THETA_RANGE=0.75
pass_target = round(param[3]→teammate) only if param_mask[3]>0
```
`param_mask = action.ACTION_PARAM_MASKS[skill]`. **`residual_x_range = 0.0`** is intentional — the
ball/skill anchor sets `target_x`; the residual only nudges `y` (±450 mm) and `theta` (±0.75 rad).
The anchor functions (`build_fallback_action_targets`) are in `action.py`/`environment.py`; port them
verbatim. (Support roles use `build_open_lane_anchor` instead — see doc 38.)

### B.6 Skill → B-Human SkillRequest
Map the chosen skill to the existing `RLSkillProvider` SkillRequests: `walk→walkTo`, `dribble→dribble`,
`shoot→shoot/kick`, `pass→pass(target)`, `stand→stand`. (block/mark/observe are defender/support and
not emitted by the striker.) This mapping already exists in `Src/.../RLSkillProvider.cpp`.

### B.7 Parity checklist (do this before trusting the port)
1. Dump 64 obs vectors from a Python rollout; feed the C++ encoder the same world → obs must match to < 1e-4 per element.
2. Feed those obs to the ONNX in C++ → `skill_logits` must match Python torch to < 1e-4 (we verified 7.6e-6 for the merged brain).
3. Apply the C++ gate → masked argmax must equal Python's for all 64.
4. Convert residual→absolute in C++ → targets must match `_decode_residual_policy_action` to < 1 mm / < 1e-3 rad.
Only after all four pass is the transplant faithful. SimRobot is an accurate physical/perceptual
proxy, so a faithful port reproduces the sim behavior on hardware.
