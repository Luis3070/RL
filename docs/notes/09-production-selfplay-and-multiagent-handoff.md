# 09 — Production self-play, multi-agent obs space, and 4v4 wrapper-coordinated deployment

Handoff for the next colleague. **Hard deadline: 4v4 RL (BH goalkeeper + 3 RL field) vs full BH in under 2 weeks.**

Three deliverables in priority order:

1. Take the current PPO single-agent policy to production-ready as a **self-play / 4v4 candidate**.
2. Accommodate the policy for the **multi-agent observation space** without breaking the 26-dim single-agent contract — sanity-first, fast. For the 2-week sprint this stays optional; the default plan keeps OBS_SIZE=26 and pushes coordination into a Python wrapper.
3. Deploy a **wrapper-coordinated 4v4 multi-agent** path that reuses the existing single-agent PPO checkpoint as-is and adds analytic role assignment + skill masking + residual overrides in Python. This is the only path that meets the 2-week budget. MAPPO and a from-scratch multi-agent policy are explicitly out of scope.

Time-budget strategy in one sentence: **change the wrapper, not the policy.** The PPO is already a competent walk-dribble-shoot agent in 26-dim obs; coordination becomes a runtime constraint over which skill it is allowed to choose and where its residual targets point, derived from role assignment. No new dataset, no new training run (one optional narrow fine-tune in week 2 if metrics require it).

Cross-references already in repo:
- Single-agent obs contract: `observation.py` (26-dim, see `ObsIndex`), `docs/09-implementacion-ppo-observation-space-23.md` (extension notes for MAPPO).
- 3v3 hybrid runner already exists: `run_rl_vs_bhuman_3v3.py`, scene `SabanaHerons2026/Config/Scenes/RLvsBH3v3_3D.ros2`, doc `docs/21-simrobot-estable-y-rl-vs-bhuman.md`.
- Teacher ideology: `teacher_policy.py` (gate-driven skill choice) + `teacher_production_dataset.py` (Design-A consistency + per-scenario coverage).
- Production pipeline you are inheriting: `train_production.py`, dataset `runs/fine_tuning_production_dataset/production_replay_stage4/samples.npz`.
- Drag / throughput context: memory `production-finetune-v1-drag-and-throughput-fix.md`.

Hard constraints (do NOT relax — see `docs/05-premisas-irrompibles.md`):
- Do NOT edit B-Human C++ (`SabanaHerons2026/Src` read-only).
- Use **visible** SimRobot only (`simrobot_visible` / `3d_visible`); embedded preflight crashes at reset.
- Shoot band stays **2.5–3.0 m** (field constraint).
- Do NOT use `conda run`; always:
  `source ~/anaconda3/etc/profile.d/conda.sh && conda activate RLNAO && cd  && PYTHONPATH= python <script>`

---

## 1. Production-ready single-agent (the immediate path)

State at handoff:
- Last completed fine-tune: `runs/production_finetune_v1/checkpoint_stage4_step35000.pt`. Stage-4 random eval reads clean, but production-scenario eval (`runs/behavioral_evaluator/behavior_20260528_073808/`) was **0/5 passed**; agent camps in the band (~46% of steps) and only shoots ~1.8%.
- v2 attempt (`runs/production_readiness_v2`) crashed at first reset on slot 6.
- `train_production.py` has been retuned (2026-05-29) to fix both drag and throughput. Defaults now: 4 slots, eval_episodes=8, max_episode_steps=400, reset budgets back to 4s/5/3, BC peak β=0.15, half-life=10 evals, replay_weight=0.10, skill/param=0.4/0.5, `dynamic_full_field_pressure` weight dropped to 0.

### 1.1 Launch the corrected fine-tune

```
source ~/anaconda3/etc/profile.d/conda.sh && conda activate RLNAO && \
cd  && \
PYTHONPATH= python train_production.py
```

Optional overrides (env vars; all defaults set in the wrapper):
- `RL_PROD_PLAYERS=1,2,3,4` — keep at 4; 6 saturated render threads in v2.
- `RL_PROD_TRAIN_STEPS=80000` — finish the original budget.
- `RL_PROD_BC_REPLAY_WEIGHT=0.10` — raise to 0.15 if shoot still doesn't take off.

### 1.2 Verification gates (do NOT promote past these without proof)

Per checkpoint at `eval_interval=5000`:

| Gate | Target by step | Where to read |
|---|---|---|
| `bc_beta` ≥ 0.05 | through ≥25k | `eval_summary.csv` column `bc_beta` |
| `policy_shoot_rate` ≥ 0.05 in band | by 20k | `eval_summary.csv` columns `policy_shoot_rate`, `shoot_distance_ready_rate` |
| `goal_rate` leaves noise (>0.05) | by 40k | `eval_summary.csv` |
| Wallclock ≤ 13 min per 5k steps | from step 0 | checkpoint mtimes |
| `behavioral_evaluator.py` ≥ 3/5 scenarios passed | at 40k+ | `runs/behavioral_evaluator/behavior_<ts>/summary.csv` |

If `bc_beta` collapses to <0.02 before 15k, the BC half-life is still too short — bump `RL_PROD_BC_HALF_LIFE_EVALS=15`. If `policy_shoot_rate` is still <0.02 at 25k with `bc_beta`>=0.08, you need a reward weight change (last resort, see §1.4).

### 1.3 Production behavioral eval (mandatory before any self-play step)

```
PYTHONPATH= python behavioral_evaluator.py
```
Reads checkpoint from `CHECKPOINT_PATH` inside `behavioral_evaluator.py` (no argparse). Update to the new checkpoint before running.

Pass criteria are scenario-specific (see `behavioral_evaluator.py` situation table). For production-ready: at least `finish_band_2p5_3m_clean`, `finish_band_2p5_3m_static_lane`, and `preshoot_bad_angle_boundary` must pass.

### 1.4 Last-resort reward levers (touch only if §1.2 still fails)

Edits go in `curriculum.py` stage 4 `RewardWeights`:
- Raise `goal` 7.0 → 12.0 (currently dwarfed by walk income).
- Raise `good_shoot` 2.0 → 3.5.
- Drop `good_walk` 0.2 → 0.10 (removes the camp incentive).

These are stage-global. They affect any future training, not just production fine-tune. Commit and tag the change so older checkpoints are still reproducible.

---

## 2. Multi-agent observation space — sanity-first accommodation

### 2.1 Current single-agent contract (must not break)

`OBS_SIZE = 26`. Layout (from `observation.py:ObsIndex`):

```
0   robot_x                       13  can_score_now
1   robot_y                       14  shot_quality_no_obstacles
2   robot_theta                   15  shot_opening_with_obstacles
3   ball_rel_x                    16  pass_options_count
4   ball_rel_y                    17  nearest_teammate_dist
5   ball_end_rel_x                18  nearest_opponent_dist
6   ball_end_rel_y                19  nearest_uncertain_obstacle_dist
7   ball_vel_x                    20  nearest_teammate_front_dist
8   ball_vel_y                    21  nearest_opponent_front_dist
9   time_since_ball_seen          22  nearest_uncertain_front_dist
10  time_since_ball_disappeared   23  shoot_armed       (gate)
11  ball_seen_percentage          24  dribble_armed     (gate)
12  ball_consistent_w_game_state  25  shoot_arm_progress (gate)
```

Indices 17-22 already carry compressed nearby-body context (just `nearest_*` scalars). That is enough for single-agent self-play, but NOT enough for role-conditioned tactics: a striker needs to know the support's pose, not "distance to nearest teammate".

### 2.2 Sprint decision: keep OBS_SIZE=26, push coordination into the wrapper

**For the 2-week 4v4 deliverable, do NOT extend the obs space.** Reasoning:

- An obs-space change is a checkpoint break: every existing PPO/BC checkpoint becomes incompatible (`docs/09-implementacion-ppo-observation-space-23.md:283` already calls this out: "Cualquier checkpoint PPO entrenado con 9 variables queda incompatible con el nuevo `observation_space`"). The same applies to going 26 → 32.
- A new contract requires a new BC dataset, a new fine-tune, and a new behavioral evaluator profile. That alone is 1+ week of work — already over budget.
- The `nearest_teammate_dist`/`nearest_opponent_dist`/`nearest_*_front_dist` columns (indices 17–22) plus the gate columns (23–25) carry enough nearby-body context for a wrapper-coordinated policy to perform sanely. The policy already learned to respect them.

The "where to put the multi-agent state" question moves out of the obs and into Python state in `wrapper.py` and the 4v4 runtime (§3). Role assignment, teammate intent, and ball-ownership are tracked in a `MultiAgentCoordinator` object, NOT injected into observations the policy was not trained on.

### 2.3 Optional 32-dim extension (stretch, post-deadline)

Keep this on file as a contract for the follow-up cycle, but do NOT build it during the 2-week sprint:

```
26  teammate_rel_x            (signed, 0.0 if no teammate / single-agent mode)
27  teammate_rel_y            (signed)
28  teammate_role_id          (small int: 0=striker, 1=support, 2=defender; -1 = unassigned)
29  teammate_intent_skill_id  (SkillType int of teammate's last executed skill; -1 if unknown)
30  self_role_id              (same enum; the policy's role this episode)
31  ball_owned_by_team        (0/1: any teammate has dribble_armed in last K=8 steps)
```

- `OBS_SIZE` becomes 32 only when `RL_MULTIAGENT_OBS=1`.
- When disabled, OBS_SIZE stays 26 — existing checkpoints + dataset load unchanged.
- New 32-dim dataset goes under `runs/teacher_multiagent_dataset/`; do NOT mix with the 26-dim canonical replay.

This extension is only worth building if §3's wrapper-coordinated 4v4 (week 1–2) plateaus and we have a 2–3 week follow-up cycle for a true role-conditioned fine-tune.

### 2.4 Sanity checks before deploying 4v4

These apply whether or not we extend the obs:

- `pybh_backend.py` already exports each robot's own pose + ball + nearby-body scalars. Per-robot obs are independent; the wrapper coordinates side-channel.
- `gate.py` stays per-robot. Do NOT feed teammate state into the gate at any point.
- The 4v4 runner needs to instantiate the **same 26-dim PPO checkpoint** in three slots (player_2/3/4 of team 24); player_1 is the BH goalkeeper, untouched.

---

## 3. 4v4 wrapper-coordinated multi-agent (the 2-week sprint)

Target: **team 24 = 1 BH goalkeeper (player_1) + 3 RL field players (player_2, 3, 4)** vs **team N = 4 B-Human players**, full match scene. Hard deadline ≤ 2 weeks from start of week 1.

The only multi-agent strategy compatible with this deadline is "**change the wrapper, not the policy**": deploy the existing single-agent PPO checkpoint three times in parallel, with a thin Python layer that (a) assigns roles, (b) restricts each robot's skill set per role, and (c) overrides residual targets per role when the policy would otherwise wander. The PPO sees the same 26-dim obs it was trained on; coordination is a runtime constraint, not a learned behavior.

This works because the teacher_policy/teacher_dataset ideology already encodes "gate state → correct skill". The wrapper layer turns the same idea into "(gate state, role) → allowed skills + target override". No new dataset. No new training (one optional narrow fine-tune in week 2 if metrics demand it). MAPPO and OBS_SIZE=32 stay out of scope.

### 3.1 Architecture (one diagram, three boxes)

```
              +----------------------------------------+
              |  MultiAgentCoordinator (Python, per-step) |
              |  - reads team-wide state (ball, poses)     |
              |  - assigns roles with hysteresis           |
              |  - publishes self_role + teammate_intent   |
              +-------+--------+--------+--------+-------+
                      |        |        |        |
                      v        v        v        v
              player_2 wrap  p_3 wrap  p_4 wrap   (BH GK untouched)
              role=striker   role=L    role=R
              skill_mask     skill_mask  skill_mask
              target_override target_o.  target_o.
                      |         |         |
                      v         v         v
              (same 26-dim PPO checkpoint, three forward passes)
```

The coordinator runs once per RL tick (~30 Hz). It pulls each RL robot's pose, the ball pose, and the gate decisions from `env.last_gate_decision`, computes the role assignment, then publishes back a per-robot `RolePolicy`. The wrappers consume `RolePolicy` and constrain the policy's output before it becomes a `SkillRequest`.

### 3.2 Role catalog for 4v4 (3 field players → 3 roles)

| Role | Selection rule | Allowed skills | Residual override |
|---|---|---|---|
| **striker** | closest field player to ball, with hysteresis (1.15× ratio over previous striker) | walk, dribble, shoot (full active set) | none — PPO targets passed through |
| **left_support** | of remaining two field players, the one with `y > striker.y` | walk only (mask dribble + shoot to 0) | target = ball + ( -1200mm goalward, +700mm lateral); θ toward ball |
| **right_support** | of remaining two field players, the one with `y < striker.y` | walk only | target = ball + ( -1200mm goalward, -700mm lateral); θ toward ball |
| (fallback) **defender** | activated when ball is in own half AND striker is not the closest to own goal | walk only | target = midpoint(ball, own_goal_center); θ toward ball |

Role assignment runs every 20 RL ticks (~0.7 s) **or** immediately on a `gate.dribble_armed` flip (a takeover transfer). The defender fallback only triggers in own half — in the opponent half the second-and-third closest players are always supports.

Why "walk only" for supports: the BC dataset is walk-dominant by 75% and the policy is already strong at walk. Skill masking is the same pattern as the existing `active_skill_mask` (`action.py:135`), with role-derived allowed sets. The policy never picks an out-of-mask skill — categorical head is masked to -inf logits exactly like the gate's forbid-invalid contract. PPO update consistency is preserved because the mask is computed deterministically from `(obs, role)` and re-derivable at update time (if we re-train).

### 3.3 Skill mask wiring (no policy change)

Add a `RolePolicy` dataclass and route every PPO action through it before it becomes a `SkillRequest`:

```python
# new file: role_policy.py
@dataclass(frozen=True)
class RolePolicy:
    role: str                       # "striker" / "left_support" / "right_support" / "defender"
    allowed_skills: tuple[str, ...] # subset of ACTIVE_SKILLS
    target_override: tuple[float, float, float] | None  # (x, y, theta) in field frame; None = pass through

def apply_role(policy_action: StructuredSkillAction,
               role: RolePolicy,
               obs: np.ndarray) -> StructuredSkillAction:
    # 1) Skill mask: if policy picked a skill not in role.allowed_skills, fall back to walk.
    # 2) Target override: replace target_x/y/theta when role demands it.
    # 3) pass_target stays from policy.
    ...
```

The override is applied **after** the policy forward pass — the policy still outputs its preferred skill + residuals, but the wrapper coerces. From PPO's perspective the action distribution is unchanged; from the env's perspective the executed action is role-correct.

Sanity: log when policy-preferred skill ≠ executed skill, per role. If supports get coerced from `dribble` to `walk` >30% of steps, the policy is fighting the role assignment — that's the signal to do the optional fine-tune (§3.6).

### 3.4 4v4 runner (extend the 3v3 pattern)

Reuse `run_rl_vs_bhuman_3v3.py` as the template. Three changes:

1. New scene `SabanaHerons2026/Config/Scenes/RLvsBH4v4_3D.ros2` derived from `RLvsBH3v3_3D.ros2`: add one more field robot per team. Scene files are config, NOT C++ source, so the read-only rule does not apply.
2. Set `RL_OVERRIDE_TEAM=24`, `ACTIVE_PLAYERS=(2, 3, 4)` (3 RL field players), leave player_1 as B-Human goalkeeper. Confirm via `PYBH_RL_OVERRIDE_TEAM` + `PYBH_RL_ACTIVE_PLAYERS` (already supported, see `docs/21-simrobot-estable-y-rl-vs-bhuman.md`).
3. Wire the `MultiAgentCoordinator` into the loop: one coordinator per tick, three role policies pushed to the per-player wrappers, three independent PPO forward passes.

Throughput: one visible SimRobot process with 4 RL robots in it is **cheaper** than the training setup (which spawns 4–6 visible processes). Match runs at ~real time on a 32-core machine; no Qt/OpenGL multiplication.

### 3.5 Robustness levers (what to tune when 4v4 misbehaves)

Each lever is a Python-side knob; none requires retraining.

| Symptom | Lever | Default | Override |
|---|---|---|---|
| Striker doesn't commit to shoot when armed | raise BC betas on inference (impossible; checkpoint already trained) → **use shoot-arm hold**: if `gate.shoot_armed` for ≥3 consecutive ticks, coerce skill to shoot | off | `RL_4V4_FORCE_SHOOT_HOLD_TICKS=3` |
| Two supports collide at midfield | raise lateral offset 700 → 1100 mm | 700 | `RL_4V4_SUPPORT_LATERAL_MM=1100` |
| Role flicker when both supports equidistant from ball | raise hysteresis ratio 1.15 → 1.30 | 1.15 | `RL_4V4_ROLE_HYSTERESIS=1.30` |
| Striker chases ball into own corner (no shot ever) | activate defender fallback at own-half entry | off | `RL_4V4_DEFENDER_FALLBACK=1` |
| Policy spam-switches walk↔dribble while supporting | mask dribble for supports for full episode (not just per-tick) | per-tick | `RL_4V4_SUPPORT_LOCK_WALK=1` |
| Goal pile-up near gate | force striker to back off 200 mm if `nearest_opponent_dist < 400` | off | `RL_4V4_STRIKER_BACKOFF=200` |

All levers ship disabled by default. Turn them on one at a time from the 4v4 eval log; do not stack changes blindly.

### 3.6 Optional narrow fine-tune (week 2, only if metrics demand)

If the wrapper-coordinated 4v4 reaches goal-rate >0.3 per match in week-1 eval, **ship as-is**. No retraining.

If supports get coerced from dribble→walk >30% of steps OR striker shoot rate stays <0.05 in 4v4 mode, do a 10–20k step fine-tune of the existing checkpoint with the wrapper coordinator active:

- Wrap `parallel_environment.py` so each slot exposes its `RolePolicy` at obs time AND applies the skill mask before forwarding to the env. This is the same MAPPO PettingZoo loop but with role-derived skill masks plugged into the gate-mask pathway (Design A still applies — masks become a function of obs + role, both stored).
- BC replay still uses the existing 26-dim dataset; the skill mask coerces support steps to walk, which matches dataset bias.
- 20k steps at the new throughput (~2 h on 4 visible slots).
- Verify: support's coerced-walk fraction drops to <10%, striker shoot rate ≥0.10 in band.

This fine-tune is the only training in the 2-week plan. Skip it if §3.5 levers alone hit the eval threshold.

### 3.7 Sanity-first defaults the new colleague MUST keep

- **One source of truth for roles**: the coordinator. Never let a wrapper pick its own role from local obs.
- **Skill mask is forbid-invalid, not preferred-only**: an out-of-mask categorical logit becomes -inf at the wrapper, not "discouraged". Mirrors the gate's existing contract.
- **No teammate state in the gate**: gate inputs stay per-robot (ball, pose, shoot_open). The coordinator reads gate decisions; it does not mutate them.
- **Goalkeeper untouched**: player_1 is B-Human. Coordinator does not produce a role for it.
- **Eval matrix is fixed before week 1 ends**: 6 seeds × 2-minute matches vs full BH. Re-run after every lever change. Track goal-rate, conceded-goal-rate, mean-time-to-shoot, support-coerce-rate.

---

## 4. Two-week sprint timeline (calendar-day, not week-of)

Day 0 is the day the new colleague starts. Each day has one deliverable and one verification step. Slippage on any day cascades — if a day's verify fails, the next day is debug, NOT advance.

| Day | Deliverable | Verify |
|---|---|---|
| 0 | Read this manifest + run §1.1 fine-tune in background. | `bc_beta` ≥ 0.05 at step 5k; throughput ~13 min/5k. |
| 1 | Single-agent fine-tune at 20k. Implement `role_policy.py` + `MultiAgentCoordinator` skeleton (no env yet, just unit tests). | Unit tests: role assignment + hysteresis + skill-mask coerce. |
| 2 | Fine-tune at 40k. Build `RLvsBH4v4_3D.ros2` scene from the 3v3 template; verify it loads and plays in visible SimRobot. | Scene runs 60s with 4 BH + 4 BH (no RL yet); no freeze, no missing-body errors. |
| 3 | Fine-tune at 60k. Extend `run_rl_vs_bhuman_3v3.py` → `run_rl_vs_bhuman_4v4.py` with `PYBH_RL_ACTIVE_PLAYERS=2,3,4`. Plumb `MultiAgentCoordinator` into the loop. | One RL player runs in 4v4 scene end-to-end (others B-Human). No coordinator yet — single-agent baseline. |
| 4 | Fine-tune at 80k (done). Behavioral eval on the new checkpoint. Wire 3 RL players into 4v4 with all three roles active. | Behavioral eval ≥3/5 scenarios pass. 4v4 runs a full match (10 minutes) without crash. |
| 5 | First 4v4 vs BH eval matrix: 6 seeds × 2 minutes, default coordinator settings. | Goal-rate, conceded-rate, mean-time-to-shoot, support-coerce-rate captured per seed. |
| 6 | Apply §3.5 levers based on day-5 metrics. Re-run the eval matrix. | One lever change improves goal-rate or coerce-rate without regressing the other. |
| 7 | **Sprint mid-point gate.** If goal-rate ≥0.3/match → ship path. Otherwise plan the optional fine-tune (§3.6). | Decision logged; either § 3.5-tuning continues or fine-tune launched. |
| 8–9 | Either: continue §3.5 tuning OR run §3.6 narrow fine-tune (10–20k steps with role masks). | If fine-tune: support-coerce-rate <10%, striker shoot-rate ≥0.10 in band. |
| 10 | Re-run eval matrix on tuned/fine-tuned checkpoint. | Goal-rate ≥0.3/match AND conceded-rate ≤0.5/match. |
| 11 | Adversarial seeds: vary BH opponent strategy (defensive vs aggressive), test corner cases. | No seed has goal-rate=0; no seed has conceded ≥1.5×median. |
| 12 | Telemetry / handoff: produce eval CSV, lever settings, checkpoint path, 4v4 scene path. | Deliverables file under `runs/production_4v4_v1/` with manifest. |
| 13 | Buffer day: known crash modes, robustness against bad goalie passes, edge cases. | Production-ready checkpoint frozen; handoff doc updated with what was changed. |

If day 4's behavioral eval fails (<3/5), STOP and apply §1.4 reward levers — do not build the 4v4 wrapper on a broken single-agent base. The wrapper amplifies what the policy already does; it cannot fix a policy that does not shoot.

---

## What is already on disk (so the new colleague does not re-derive)

- 26-dim sanitized BC dataset: `runs/fine_tuning_production_dataset/dataset_20260527_083253_sanitized/samples.npz` (clean, 0 positive-weight gate mismatches).
- Stage-4 replay view: `runs/fine_tuning_production_dataset/production_replay_stage4/samples.npz` (stages normalized to 4 for `BCReplayRegularizer`).
- Last fine-tune checkpoint: `runs/production_finetune_v1/checkpoint_stage4_step35000.pt` (single-agent; the v2 80k run is in flight after the 2026-05-29 fix).
- 3v3 hybrid runner + scene: `run_rl_vs_bhuman_3v3.py`, `RLvsBH3v3_3D.ros2` — template for the 4v4 deliverable.
- Behavioral evaluator: `behavioral_evaluator.py` (5 production scenarios from `production_curriculum.py`).
- Teacher pipeline: `teacher_policy.py`, `teacher_production_dataset.py` — both gate-driven, Design-A clean. **Read-only in the sprint:** do NOT regenerate datasets for the 4v4 wrapper path.

## What NOT to do (sprint-scoped)

- Do not extend the obs space (do NOT ship `RL_MULTIAGENT_OBS=1`). The 32-dim contract is on file (§2.3) for a follow-up cycle only.
- Do not regenerate any BC dataset. The wrapper-coordinated 4v4 reuses the canonical 26-dim replay.
- Do not edit `teacher_policy.py`. Role logic lives in a new `role_policy.py` + `MultiAgentCoordinator`; the gate stays per-robot.
- Do not use MAPPO. The §3.6 optional fine-tune is decentralized PPO with shared parameters; skill masking is the existing gate-mask pathway with role inputs.
- Do not change the shoot band (2.5–3.0 m) or the gate's `shoot_goal_enter_mm=2500` — field constraint.
- Do not commit/push without an explicit OK from the maintainer.
- Do not skip the day-4 behavioral eval gate. Building 4v4 on a broken single-agent makes lever-tuning impossible to interpret.

## Open questions for the maintainer

1. The new 4v4 scene `RLvsBH4v4_3D.ros2` — do we commit it under `SabanaHerons2026/Config/Scenes/`, or keep it under `RL/scenes/` and copy on launch? (Scene files are config not C++, so committing should be fine — confirm.)
2. Eval opponent: is the BH team the current `master` strategy, or a frozen snapshot for reproducibility?
3. If §3.6 fine-tune is needed, does the wrapper's skill-mask coerce path qualify as "Design A consistent" enough for BC replay, or do we need to regenerate the BC dataset with masks active? (Recommended: it qualifies — mask is `f(obs, role)` and role is `f(team_state)` — both reproducible at update time.)
4. Defender fallback (§3.2): on or off by default? Recommend off in week 1, on if conceded-rate is the blocker.
5. If reward levers in §1.4 are needed, do we commit them under a new stage 4b spec (`curriculum.py`) so existing stage 4 checkpoints remain reproducible?
