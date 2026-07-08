# CODEX TASK — Fix trío reset spawn-stability (RLMAStage7_3D defender)

## ONE-SENTENCE GOAL
Make the 3-slot trío eval (`tools/test_rlmastage7_trio.sh` with `RL_MA_PROD_PLAYERS=1,2,3`)
reach the `[eval]` stage and write an `eval_summary.csv`, by eliminating the
**systematic `fall_state=4` (robot topples on teleport) at the defender deep-spawn**
during visible SimRobot reset. **Do not change any agent behavior, policy, gate,
reward, or observation.** This is reset/test plumbing ONLY.

## BACKGROUND (verified facts — do not re-litigate)
- Scene `Config/Scenes/RLMAStage7_3D.ros2` has 7 real Nao bodies (robot1-4 team +
  robot21/22/23 opponents). It is bound to scenario `4v4_RL3D` and uses the grayscale
  `BallPerceptor`. Perception WORKS: single-slot runs already measured
  `natural_ball_seen_rate≈0.98`, `opponent_seen_rate≈0.65–0.80`.
- `block`/`mark` only appear when the controlled agent is the OFF-BALL defender, which
  only happens in the multi-agent trío (players=1,2,3). A lone agent always engages the
  ball and role-flips to striker.
- The trío eval currently never reaches `[eval]`: at reset, the slot controlling the
  DEFENDER (deep spawn, x≈-3130) repeatedly fails convergence with `fall_state=4`
  (fully fallen) and occasional `world request timed out`. Single-slot grinds through;
  the trío is too flaky (falls amplified by 2-3 heavy sims contending for the GPU).
- The reset acceptance gate lives in `pybh_backend.py`:
  `VISIBLE_RESET_POSE_TOL_MM` (default 150), `VISIBLE_RESET_MAX_FALL_STATE` (default 1),
  `VISIBLE_RESET_UPRIGHT_STREAK` (default 3); convergence loop is the function that emits
  `"Visible SimRobot reset obs did not converge to requested pose within timeout"`
  (around lines 1350–1400). The runner already relaxes these via env to 400 / 3 / 2,
  which fixed the "upright-but-drifted ~200mm" rejections but NOT the falls.
- The defender spawn is in `production_curriculum.py::_defense_team_world`:
  ```python
  defender = (
      float(rng.uniform(-3400.0, -3000.0)),
      _clip_y(ball_y * 0.45),
  )
  ```
  GK is `WorldPlayerSpec(number=4, x=-4300.0, y=0.0)`. Opponents are at
  `(ball_x+400, ball_y)`, `(ball_x+1100, -ball_y)`, `(GOAL_X-300, 0)` with `GOAL_X=4500`.

## STEP 1 — DIAGNOSE BEFORE FIXING (mandatory)
Add **temporary** debug logging (guard behind an env flag, e.g. `RL_RESET_DEBUG=1`, and
remove before finishing) in the visible-reset convergence loop in `pybh_backend.py` to
capture, for each failed attempt on the falling slot:
1. the requested pose of the controlled robot,
2. the requested poses of ALL other spawned bodies in that slot's WorldSpec,
3. the controlled robot's `fall_state` over time after teleport.

Run single-slot defender repro: `RL_MA_PROD_PLAYERS=2 RL_RESET_DEBUG=1 bash tools/test_rlmastage7_trio.sh`
Determine which is true:
- (A) the defender is teleported within <600mm of another body (collision → topple), or
- (B) the defender topples with no nearby body (teleport pose / motion-state instability), or
- (C) it only falls under multi-slot contention (single-slot is reliably upright).

State your conclusion in the PR/commit description.

## STEP 2 — APPLY THE MINIMAL FIX (pick based on STEP 1)
**Preferred fix (A or B): adjust ONLY the defender spawn** in `_defense_team_world` so it
lands in a stable spot. Constraints the new defender spawn MUST satisfy (keep it a valid
deep defender):
- `x ∈ [-3400, -2600]`, on-field, `|y| ≤ 1500` (use existing `_clip_y`),
- ≥ 800mm from the GK at (-4300, 0),
- ≥ 700mm from every opponent and teammate spawn AND from the ball, for all sampled
  `ball_x/ball_y` in this scenario's ranges (add a small clamp/rejection if needed),
- still goal-side of the ball (defensive). Keep the `theta` facing-the-ball formula.

**Fallback fix (C only): post-teleport stabilization** in the `pybh_backend.py` reset loop
ONLY: when `fall_state` indicates falling, re-issue the teleport (reset the body to the
requested pose with zero velocity) and continue waiting, instead of counting it as a
failed attempt. Do NOT change the WorldSpec schema, the obs contract, the message
protocol, or any code outside the reset-retry loop.

Keep changes as small as possible. Prefer the spawn adjustment.

## ALLOWED FILES (you may modify ONLY these)
- `production_curriculum.py` — ONLY the `defender = (...)` spawn tuple (and a small local
  helper/clamp if needed) inside `_defense_team_world`.
- `pybh_backend.py` — ONLY the visible-reset retry/convergence loop (the function that
  prints the "did not converge" message) and the temporary `RL_RESET_DEBUG` logging.
- `tools/test_rlmastage7_trio.sh` — env knobs only.

## FORBIDDEN — DO NOT TOUCH (hard invariants; instant fail if changed)
- `train.py` (especially `gate_skill_mask_from_obs`), `gate.py`, `reward.py`.
- `observation.py` (`OBS_SIZE=41/47`, `ObsIndex`), `action.py` (`SKILL_ORDER`).
- `multiagent_teacher.py`, any teacher/dataset logic, any `.npz`.
- Any policy weights / checkpoints (`*.pt`, `*.onnx`), anything under `runs/`.
- The `.ros2` scene files (body counts, robot definitions) and `.con` files.
- The striker, v4.2, or any non-defense scenario world_fn.
- Do NOT retrain. Do NOT export ONNX. Do NOT commit or push.

## ENVIRONMENT / HOW TO RUN (single sim session only)
- `export PYTHONPATH=/home/luis/GitHub/RL:/home/luis/GitHub/SabanaHerons2026/Make/Python`
- `export DISPLAY=:1`
- Python: `~/anaconda3/envs/RLNAO/bin/python`
- BEFORE every launch: `pgrep -x SimRobot` must be empty; if not, `pkill -x SimRobot`.
  Only ONE SimRobot session at a time.
- Repro single-slot: `RL_MA_PROD_PLAYERS=2 bash tools/test_rlmastage7_trio.sh`
- Target: `RL_MA_PROD_PLAYERS=1,2,3 bash tools/test_rlmastage7_trio.sh`
- Each run is long (visible 3D sim). Use background + log tail; do not poll tightly.

## ACCEPTANCE CRITERIA (all must hold)
1. `RL_MA_PROD_PLAYERS=1,2,3 bash tools/test_rlmastage7_trio.sh` reaches `[eval]` and
   prints the final `Metrics:` block with a written `eval_summary.csv`.
2. At least 6 of 8 eval episodes complete (resets converge within ≤15 attempts each).
3. `opponent_seen_rate > 0.30` (opponents still perceived — proves you didn't "fix" it by
   hiding opponents) and `natural_ball_seen_rate > 0.80`.
4. Single-slot (`RL_MA_PROD_PLAYERS=2`) still completes (no regression).
5. Report the printed `executed_block_rate`, `executed_mark_rate`, `policy_mark_rate`,
   `policy_block_rate`, `defender_score`, `goals_conceded_rate`.
6. The `RL_RESET_DEBUG` logging is removed (or fully env-gated and off by default).
7. `git diff` touches only the three allowed files. `episode_steps` stays 800.

## DELIVERABLE
A short report: STEP-1 conclusion (A/B/C), the exact diff, and the STEP-5 metrics —
especially whether `executed_mark_rate > 0` (the whole point: confirm mark fires live with
a real defender role + perceived opponents). Leave all changes uncommitted.
