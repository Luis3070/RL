# FOR TRAIN PRODUCTION — fast-path to a shippable model

**Deadline: Saturday 2026-06-06.** Today is 2026-06-03. ~72 h of wallclock to land a
behavioral_evaluator ≥ 3/5 single-agent policy. This file is the minimum-context brief
to launch `train_production.py` and triage results without reading the rest of the repo.

If you only have time for one section, read **§1**. Everything else is reference.

---

## 1. Launch (one command)

```bash
source /home/luis/anaconda3/etc/profile.d/conda.sh
conda activate RLNAO
cd /home/luis/GitHub/RL
PYTHONPATH=/home/luis/GitHub/RL python train_production.py
```

That is the entire production fine-tune. It will:

1. Load `runs/production_finetune_v1/checkpoint_stage4_step35000.pt` as warm-start.
2. Stage-normalize the BC dataset to stage 4 (writes `runs/fine_tuning_production_dataset/production_replay_stage4/samples.npz` on every run).
3. Spin up 4 visible SimRobot slots (players 1–4) on display base :430.
4. Run stage-4 fine-tune for `RL_PROD_TRAIN_STEPS` (default 80 000), evaluating every 5 000.

Wallclock target: **≤ 13 min per 5 k steps** under the patched defaults (4 slots, eval_eps 8, max_episode_steps 400). If a 5 k window takes >25 min, throughput is wedged — see §4.3.

---

## 2. What is in this commit

| Path | Size | Role |
|------|------|------|
| `runs/production_finetune_v1/checkpoint_stage4_step35000.pt` | 176 KB | **Primary warm-start.** Last `.pt` from v1 (stage-4 fine-tune, BC-distilled, walk-dribble competent, **does not shoot**). |
| `runs/stage4_candidate/checkpoints/stage4_candidate_checkpoint_stage3_step55000.pt` | 176 KB | **Deeper fallback warm-start.** Stage-3 BC warm-start baseline. Use only if v1 step35000 has been corrupted; the v1 checkpoint already inherits this one's weights through BC warm-start. |
| `runs/fine_tuning_production_dataset/dataset_20260527_220501_production_merged/samples.npz` | 660 KB | **BC replay dataset.** 17 815 rows, 26-dim obs, 5-dim actions, gate-label sanitized. Skill mix (positive samples): walk 5 858 / dribble 6 332 / shoot 4 212. |
| `runs/fine_tuning_production_dataset/dataset_20260527_220501_production_merged/summary.json` | 20 KB | Dataset metadata. Read by `train_production.prepare_stage4_replay_dataset()` to copy provenance into the replay summary. |
| `FOR_TRAIN_PRODUCTION.md` | this file | — |

All four paths are referenced by hard-coded constants in `train_production.py:15-24`. Do not move them.

---

## 3. Logic of the `.pt`

### 3.1 `checkpoint_stage4_step35000.pt` — what it knows, what it does not

State after 35 000 stage-4 PPO updates (BC-distilled), measured on `production_finetune_v1` eval CSV last row:

**Knows:**
- `walk_in_walk_zone_rate` = 0.91 — picks walk inside the walk zone.
- `dribble_in_dribble_zone_rate` = 0.90 — picks dribble in the dribble zone.
- `clean_finish_rate` = 1.0 on band scenarios that the agent manages to reach.
- `wire_audit_status` = PASS (obs/gate plumbing is clean).
- `shoot_zone_rate` = 0.104 — does enter the 2.5–3.0 m shoot band ~10 % of steps.

**Does not:**
- `policy_shoot_rate` = 0.000 — **literally never selects shoot**. Inside the shoot zone it picks walk or dribble.
- `shoot_in_shoot_zone_rate` = 0.000.
- `goal_rate` = 0.0026 (≈ 0).
- `behavioral_evaluator` = **0/5** on the canonical production gate.

### 3.2 Why it does not shoot (the local optimum)

Stage-4 weighted-reward bookkeeping per episode at step 35 k:

| Term | Weighted/ep | Comment |
|------|-------------|---------|
| `good_walk` | ≈ 96 | walking pays continuously |
| `walk_persistence` | ≈ 63 | reinforces sticking with walk |
| `good_dribble` | ≈ 31 | dribble pays when ball is on |
| `good_shoot` | 0 | never fires → never collects |
| `goal` | 0 | follows from no-shoot |

PPO found a stable walk-dribble policy that earns ~30× the per-episode reward of a finisher. **The reward signal alone will not pull it out.** BC was supposed to bias toward shoot via the 4 212 shoot rows in the dataset, but in v1 `bc_beta` collapsed from 0.08 → 0.011 by step 5 000 (competence ramped fast, half-life was 3 evals), turning BC off before shoot ever stuck.

### 3.3 What the patched `train_production.py` (already in this commit) changes

All changes are **localized to `train_production.py`** — no curriculum/reward/dataset edit:

| Knob | OLD (v1) | NEW (this commit) | Why |
|------|----------|-------------------|-----|
| `bc_beta_max` | 0.08 | **0.15** | stronger peak BC pull |
| `bc_competence_half_life_evals` | 3.0 | **10.0** | BC stays alive 3× longer |
| `bc_replay_weight` | 0.03 | **0.10** | shoot rows pull 3× harder per replay batch |
| `bc_skill_loss_scale` | 0.20 | **0.40** | skill head listens to BC twice as hard |
| `bc_param_loss_scale` | 0.35 | **0.50** | residual head likewise |
| `bc_replay_updates` | 4 | 2 | redundant once competence saturates |
| `eval_episodes` | 24 | **8** | biggest wallclock win |
| `max_episode_steps` | 600 | **400** | band scenarios converge well under 400 |
| `parallel_player_numbers` | 1..6 | **1..4** | 6 slots saturate Qt/OpenGL, caused v2's slot-6 timeout wedge |
| `dynamic_full_field_pressure` weight | 1 | **0** | converted 0/8 in eval, most expensive scenario |
| `RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS` | 10 000 | **4 000** | inflated budgets deepened stalls, did not prevent |
| `RL_VISIBLE_BALL_SETTLE_MAX_REPLACES` | 10 | **5** | same |
| `RL_VISIBLE_RESET_MAX_ATTEMPTS` | 8 | **3** | same |

`v1/step35000.pt` was trained under the OLD defaults. **No `.pt` on disk has seen the NEW defaults end-to-end.** That gap is the work between you and Saturday.

### 3.4 `stage3_step55000.pt` — when to use it

Only if `step35000.pt` is corrupted/unloadable. Otherwise leave it — `step35000.pt` already
inherits this checkpoint's weights through BC warm-start, so starting from `stage3_step55000.pt`
just loses 35 k steps of walk/dribble fluency that you would have to rebuild.

---

## 4. Logic of the dataset

### 4.1 Shape and content (`samples.npz`)

- `observations`: `[17815, 26]` float32. The 26-dim layout follows `observation.ObsIndex` (gate bits at indices 23/24/25: `shoot_armed`, `dribble_armed`, `shoot_arm_progress`).
- `actions`: `[17815, 5]` float32 — `[skill_id, target_x, target_y, target_theta, param4]`.
- `stages`: `[17815]` int16 — original collection stages 1/2/4. `train_production.py` overwrites all to 4 at load time so `BCReplayRegularizer` (which keys on current PPO stage) surfaces every row during stage-4 fine-tune.
- `weights`: `[17815]` float32. 16 402 positive (`> 0`), 14 792 full-weight. Gate-label sanitized: 221 zero-weight rows whose teacher label was inconsistent with the baked gate bucket were zeroed in the source.
- `skill_order`: matches `action.SKILL_ORDER`. Active skills = (stand, walk, dribble, shoot).

Positive-sample skill mix:

| Skill | Count | % of positive |
|-------|-------|---------------|
| walk | 5 858 | 35.7 % |
| dribble | 6 332 | 38.6 % |
| shoot | 4 212 | **25.7 %** |

Shoot at 25.7 % is healthy headroom — the dataset is **not** the reason the policy doesn't shoot. BC just needs to stay on long enough for these rows to bias PPO, which is exactly what §3.3's patch is designed to do.

### 4.2 Gate-consistency invariant (the "Design-A" contract)

Each row's `(gate_bucket, skill_label)` pair is forced into one of the three legal combinations: `none_armed → walk`, `dribble_armed → dribble`, `shoot_armed → shoot`. The replay regularizer reproduces the same forbid-invalid skill mask the PPO env uses at update time, so BC and PPO push on the **same** policy distribution — no fight at the gradient level.

Do not regenerate this dataset before Saturday. It is correct and sanitized.

### 4.3 What gets auto-built at run start (not committed, do not look for it)

`runs/fine_tuning_production_dataset/production_replay_stage4/samples.npz` is rebuilt on every `train_production.py` launch from the source dataset. Stages are overwritten to 4 there. If you see a complaint that this file is missing, just re-launch — it is the script's first step.

---

## 5. Verification gates (watch these three numbers)

Eval CSV path during the run:

```
runs/production_readiness_v2/telemetry/production_finetune_train_stage4_*/csv/eval_summary.csv
```

Open the **last row** at each eval boundary (5 000-step cadence).

| Step | Metric | Required | What if it fails |
|------|--------|----------|------------------|
| 5 000 | `bc_beta` | ≥ 0.05 | BC decayed too fast → raise `RL_PROD_BC_HALF_LIFE_EVALS` to 15.0 and relaunch |
| 20 000 | `policy_shoot_rate` | ≥ 0.05 | shoot still not firing → escalate to §6 reward levers |
| 20 000 | `shoot_in_shoot_zone_rate` | ≥ 0.05 | same as above |
| 40 000 | `goal_rate` | ≥ 0.05 | almost-there → push to 60 k or escalate to §6 |
| any | `wire_audit_status` | `PASS` (`WARN` ok if `collision_rate > 0.25` only) | not `FAIL` |

**Production gate:** behavioral_evaluator ≥ 3/5 scenarios. Run it manually after the fine-tune (or use the in-loop production-scenario eval). Required passes:
- `finish_band_2p5_3m_clean` (load-bearing).
- `finish_band_2p5_3m_static_lane` OR `blocked_corridor_repair`.
- Any third.

`dynamic_full_field_pressure` is OK to fail — it's de-weighted to 0 in the scenario set.

---

## 6. If §5 fails: reward levers (last resort)

Edit `curriculum.py` stage-4 weights, **one lever at a time**, re-run, re-eval. Order matters.

| Lever | Default | Try | Effect |
|-------|---------|-----|--------|
| `goal` | 7.0 | **14.0** | doubles goal reward — pulls hardest on the missing signal |
| `good_shoot` | 2.0 | **4.0** | rewards shoot frames more — works in tandem with `goal` |
| `good_walk` | 0.2 | **0.05** | cuts walk subsidy — removes the local optimum's draw |
| `timeout` | -3.0 | **-5.0** | raises cost of not finishing |

Apply 1 → 2 only if the previous lever did not move `policy_shoot_rate` by ≥ 0.02 in 10 k steps. **Do not stack all four at once** — overshoot makes the policy spam shoot from any distance and the behavioral evaluator collapses on `preshoot_bad_angle_boundary`.

---

## 7. Decision tree to Saturday

```
Day 0 (Wed): launch from step35000.pt. Smoke-check at step 5 k (bc_beta ≥ 0.05).
Day 1 (Thu): step 20 k checkpoint. Decide:
            ├── policy_shoot_rate ≥ 0.05  → keep going to 40-60 k
            └── policy_shoot_rate < 0.05  → §6 lever 1 (goal 7→14), restart from step20k
Day 2 (Fri): step 40 k or post-lever step 20 k.
            ├── behavioral_evaluator ≥ 3/5  → STOP, this is the production .pt
            └── 1/5 or 2/5                  → §6 lever 2 (good_shoot 2→4) OR §6 lever 3
Day 3 (Sat): final eval. Production .pt is whichever step crossed ≥ 3/5 first.
```

If by Friday EOD nothing crosses ≥ 2/5, fall back to the behavioral_evaluator's
`finish_band_2p5_3m_clean`-only checkpoint and ship that as a striker-only model — the
4v4 wrapper (see `.docs/09 §3`) can still produce useful play with a striker who only
finishes the clean band.

---

## 8. What NOT to do (sprint-scoped)

- Do not edit B-Human C++ (`SabanaHerons2026/Src` is read-only).
- Do not regenerate the BC dataset. The 17 815-row merged dataset is gate-consistent and skill-balanced for shoot. The bottleneck is BC β decay, not the rows.
- Do not change `OBS_SIZE` or `SKILL_ORDER`. Both invalidate the warm-start `.pt`.
- Do not run the embedded backend. Visible SimRobot only (`simrobot_visible` + `3d_visible`).
- Do not use 6 parallel slots. v2 wedged on slot 6 timeouts and burned a night.
- Do not push to remote without explicit OK.

---

## 9. Files you will end up reading anyway

- `train_production.py` — entry script; everything tunable lives here.
- `train.py` — heavy lifter; do **not** edit for a sprint fix, only read.
- `curriculum.py` — stage-4 reward weights (§6 lives here, around line 750+).
- `gate.py` — `SkillGate` (shoot band 2 500–2 900 mm, align 30°). Do not touch.
- `production_curriculum.py` — the 5 production scenarios + the `*_topup` variants.
- `behavioral_evaluator.py` — the canonical 5-scenario production gate.
- `.docs/09-production-selfplay-and-multiagent-handoff.md` — full single-agent + 4v4 wrapper plan.

That is the whole production path. Run §1, watch §5, fall back to §6 if needed, ship Saturday.
