# CODEX TASK — Take the goalkeeper to ELITE, production-ready (4-hour, run-everything, report-only)

You (Codex) built the GK framework. Now finish it to **elite production quality**. Operating contract:

- **RUN EVERYTHING YOURSELF.** Train, evaluate, benchmark, export — all of it. Do **not** hand
  partial work back, do **not** ask Luis questions mid-task. When done (or at the 4h mark), send Luis
  **one results report** (format at the bottom). Results only — no code dumps.
- **HARD TIME LIMIT: 4 hours.** Work the priority order below. If you run out, ship what's validated +
  the full per-scenario table + a one-line verdict on the remaining bar.
- Env: `PYTHONPATH=/home/luis/GitHub/RL:/home/luis/GitHub/SabanaHerons2026/Make/Python`,
  python = `~/anaconda3/envs/RLNAO/bin/python`. The closed-loop benchmark is **analytical (no
  SimRobot)** — fast; you can run thousands of shots. If you touch the visible sim, `pgrep/pkill -x
  SimRobot` first (single session).
- Branch: `rl-v4.2-multiagent-gk`. Do not touch the merged-ONNX / defender files.

---

## 0. Current state — DO NOT REDO, only finish/validate

- **Candidate:** `runs/goalkeeper_ppo/ppo_gk_20260618_231005/checkpoint_gk_ppo_step180000.pt` — **93.63%
  save** (reproduced & validated). Holes: **`left_post_jump` 77.7%**, **`fast_blast` 82%**.
- **Contract:** 64-dim obs (`goalkeeper_teacher.encode_goalkeeper_obs`), 12 skills
  (`goalkeeper_action.GKSkillType`), per-step legal mask `goalkeeper_teacher.goalkeeper_skill_mask_np`
  (Design-A — intercepts gated by shot threat/timing, clear/pass/dribble by ball control). **Inference =
  apply the mask THEN argmax** (raw argmax collapses to intercept_center — verified).
- **Pipeline:** `teacher_goalkeeper_dataset.py` → `train_goalkeeper_bc.py` → `train_goalkeeper_ppo.py`
  → `tools/benchmark_goalkeeper_closed_loop_shots.py`. Mirror tool: `tools/mirror_augment_gk_dataset.py`.
- **Already produced this session (use, don't recreate):**
  - Mirror-augmented dataset (L↔R symmetric, 96k→192k): `runs/teacher_goalkeeper_dataset/dataset_gk_mirror_20260619_010831/samples.npz`.
  - Mirror-BC: `runs/goalkeeper_bc/bc_gk_20260619_010920/checkpoint_gk_bc_step23760.pt` (skill_acc 0.994).
  - **Re-PPO launched** from mirror-BC on the symmetric corpus: `runs/goalkeeper_ppo/ppo_gk_20260619_011330/`
    (180k steps). **Check if it finished; if not, finish it.**
  - Diagnosis already done: mirror-BC benchmark showed `left_post_jump` 77.7→**90.7%** (asymmetry was a
    LEARNED bias; symmetric corpus fixes it). BC-only overall is 79% — **PPO is what recovers the 90–100%
    on slow_roller/low_finish**, so the mirror **re-PPO** (above) is the policy to validate.
- **Deploy hazard:** 3 GK ONNX exist; only the parity-faithful one is deployable. Always parity-check
  before declaring a file deployable.

---

## 1. WORK TO DO — priority order, run all

### Track A — asymmetry (mostly done; finish + validate)
- Confirm `ppo_gk_20260619_011330` completed; benchmark it (`--episodes 2700`).
- **PASS:** `left_post_jump` within **≤3 pt** of `right_post_jump` AND overall save **≥ 93.63%** (symmetry
  not bought at the cost of save rate). If overall regressed, raise PPO steps / tune KL and re-run.

### Track B — `fast_blast` (and any clean scenario < 90%)
- **Root cause (diagnosed):** reaction/reach at high speed. `fast_blast` shots are central, speed
  1550–2300 mm/s. The mask floor `INTERCEPT_TIME_MIN_S = 0.10` (`RL_GK_INTERCEPT_TIME_MIN_S`) plus late
  detection can drop time-to-line under the floor → intercept **disarms** → goal; and the dive may not
  complete within `ttc ≤ 1.15`.
- **Try, measure each:** (a) lower `RL_GK_INTERCEPT_TIME_MIN_S` (e.g. 0.05) so fast shots stay armed;
  (b) earlier **pre-commit** in the teacher when `|ball_vel|` is high (fire the intercept sooner, before
  the ball is close); (c) make sure the selected intercept skill's reach covers the predicted crossing
  (`_skill_matches_crossing` / `_save_radius` in the benchmark define the reach windows — center 85,
  lateral 145, low 240, jump 650). Retrain (BC+PPO) if the fix is in the teacher/data.
- **PASS:** `fast_blast` ≥ **90%**.

### Track C — REALISM (the production gap — author in BOTH teacher AND benchmark, then retrain)
The current 9 scenarios (post jumps L/R, low finishes L/R, central drive, slow roller, fast blast,
diagonal cutbacks L/R) are clean, single-shot. An **elite production** keeper must also handle:
- **rebound / second-shot** (ball bounces off keeper → immediate follow-up).
- **1v1 breakaway** (attacker dribbles in close, keeper must narrow the angle / close down).
- **close-range reaction** (shot from inside the box, very short ttc).
- **chip / lob** (high ball over the keeper).
- **deflection** (shot changes direction mid-flight).
- **screened / late-appearing shot** (perception delay — ball seen late).
Add each to BOTH: `teacher_goalkeeper_dataset.py` (so the GK **trains** on them) and the benchmark
`AttackSpec` list (so they're **measured**). Then re-collect → re-BC (warm) → re-PPO → benchmark.
- **PASS:** each new scenario ≥ ~**85–90%** (1v1 / close-range may be lower — report honestly, don't hide).

---

## 2. DIAGNOSE / EVALUATE / VALIDATE — run all, on the FINAL policy

1. **Closed-loop benchmark** (`--episodes 2700`): overall save + per-scenario save% + goals + **save_skills**
   (must prove it actually **dives** (`intercept_jump_l/r`), **spreads low** (`intercept_low/lateral`),
   **blocks center** (`intercept_center`), and **stands / walk_goalie_pose** when there's no shot).
2. **Offline gate-aware battery** (load policy, apply `goalkeeper_skill_mask` then argmax on a held-out
   dataset): **0 gate violations**, deterministic, params bounded; **BC skill-recall vs teacher ≥ 99%**;
   **skill entropy** (uses all 12 skills, ≥ ~3.0/3.58 bits); **input-noise skill-flip** at σ 0.02/0.05/0.10
   must be **linear, no cliff** (reference: ~1/3.4/9.6%).
3. **ONNX parity:** export the FINAL policy; verify `max|onnx−torch| < 1e-4` and 100% argmax over ≥64
   obs. Give it a **clear new name + sha256**; confirm it does NOT collide with the stale GK onnx.

---

## 3. THE "ELITE / PRODUCTION-READY" BAR — ALL must hold to declare ready
1. **Overall save ≥ 96%** (stretch ≥ 97%); **no clean-shot scenario < 90%**.
2. **L/R symmetric:** `|left_* − right_*| ≤ 3 pt` for post_jump, low_finish, cutback.
3. **`fast_blast` ≥ 90%.**
4. **Realism scenarios exist, are TRAINED on, and each ≥ ~85%** (note any structurally-hard ones).
5. **Sanity** clean (0 gate violations, deterministic, bounded). **Diversity:** all 12 skills used
   context-appropriately. **Robustness:** noise skill-flip linear, no cliff.
6. **ONNX parity < 1e-4**, single clearly-named deployable file (+ sha256).
7. **Behaviors verified by `save_skills`:** dives L/R, genuflects/spreads, blocks center, stands &
   walks-to-pose when idle, clears/passes/dribbles-out under control.

If every bar holds → declare **ELITE-READY**. Otherwise → **NOT-YET** + name the exact failing bar(s).

---

## 4. REPORT TO LUIS (results only, one message)
- Overall save% (final policy) and a **full per-scenario table**: scenario · save% · goals · dominant save skill.
- **L/R symmetry deltas** (post_jump / low_finish / cutback).
- **Realism-scenario results** (each new type's save% + whether it was trained on).
- Sanity / diversity / robustness lines + **ONNX parity + final filename + sha256**.
- One-line **verdict**: `ELITE-READY` or `NOT-YET — failing: <bar(s)>`.
- What you changed (teacher/mask/scenarios/training), in 3–5 bullets. No code, no logs.

## 5. FILES
GK code: `goalkeeper_action.py`, `goalkeeper_teacher.py`, `teacher_goalkeeper_dataset.py`,
`train_goalkeeper_bc.py`, `train_goalkeeper_ppo.py`, `run_goalkeeper_production.py`,
`tools/benchmark_goalkeeper_closed_loop_shots.py`, `tools/mirror_augment_gk_dataset.py`,
`tools/export_goalkeeper_policy.py`, `tools/verify_goalkeeper_dataset_quality.py`.
Mask/obs index reference: `goalkeeper_teacher.encode_goalkeeper_obs` (64-dim order) +
`goalkeeper_skill_mask_np`. Benchmark reach windows: `_save_radius` / `_skill_matches_crossing`.
