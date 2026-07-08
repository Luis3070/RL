# CODEX SESSION LESSONS — defender perception/mark investigation (2026-06-18)

Read this BEFORE touching the defender / multi-agent sim. It encodes dead-ends already
paid for this session so you don't repeat them. Each item = symptom → cause → what to do.

## 0. Golden rules (operational)
- **ONE SimRobot session at a time.** ALWAYS `pgrep -x SimRobot` and `pkill -x SimRobot`
  before launching. Two visible sims of a heavy scene also starve each other.
- `export DISPLAY=:1`. Python = `~/anaconda3/envs/RLNAO/bin/python`.
  `PYTHONPATH=/home/luis/GitHub/RL:/home/luis/GitHub/SabanaHerons2026/Make/Python`.
- `train.py` chdirs into `SabanaHerons2026/Make/Python` on import → use ABSOLUTE paths for
  checkpoints/datasets.
- Visible-sim runs are long; launch in background + tail logs, don't poll tightly.

## 1. Sim won't start: "Module YoloBallDetector/CameraStreamer is unknown" or
##    "BallPerceptor.cpp ASSERT(encoder.input(0).dims(2)==1) failed"
- CAUSE: HEAD migrated ball detection to the **Trionda** stack (commits 712b3a41 +
  be4459fe). It set `4v4_RL3D` to `YoloBallDetector`/`CameraStreamer` and a **color**
  BallPerceptor encoder (`encoder_color.h5`). Trionda is **PRODUCTION-ONLY**; the SimRobot
  pybh binary (Jun-10) only has the grayscale `BallPerceptor` (`dims(2)==1`).
- FIX (no rebuild): restore the pre-Trionda ball config for the RL scenario:
  `git -C /home/luis/GitHub/SabanaHerons2026 checkout 712b3a41~1 -- \
   Config/Scenarios/4v4_RL3D/threads.cfg \
   Config/Scenarios/4v4_RL3D/ballPerceptor.cfg \
   Config/Scenarios/4v4_RL3D/ballPerceptFilter.cfg`
  (grayscale `encoder.h5`/`classify.h5` already on disk). This is Luis's intent:
  **SimRobot uses BallPerceptor; Trionda is production-only.** Already applied (held
  uncommitted) — verify it's still in place before blaming anything else.

## 2. Perception is NOT broken — do not "fix" it with an oracle
- The instinct to route obstacles through `OracledPerceptsProvider` (GT opponents) was
  TRIED and REJECTED: it creates a sim2real gap (perfect opponent perception the real NAO
  won't have). Even the "noisy oracle" path was abandoned. Reverted.
- Opponents ARE perceived via the normal vision NN when they are REAL bodies in FoV.
  Do not touch the `RobotDetector`/`ObstacleModel` provider wiring.

## 3. Scene choice for opponent/mark tests
- `RL1v0_3D_visible` has **1 robot body** → opponents can only be DUMMY-injected and are
  NEVER perceived. Do NOT use it for opponent/mark tests (this caused the original "mark=0"
  red herring). It's fine for ball/striker single-agent.
- `RLMAStage7_3D` has **7 real bodies** (robot1-4 + 21/22/23 opponents) → opponents ARE
  perceivable. It binds (now) to scenario `4v4_RL3D` and has a `.con` (`ci 30`, `gc playing`)
  created this session (needed for render + play state). It was originally bound to the
  Trionda `3v3_Scaled` scenario — keep it on `4v4_RL3D` for SimRobot.

## 4. block/mark are GATE-DRIVEN, and need the OFF-BALL role
- The block/mark "fine-tune" (`runs/defender_47_v2_blockmark`) was a near no-op: an offline
  probe showed Bryam's original `pt_defender_47@2500` AND even bare v2b already pick
  block(0.73)/mark(0.96) on the right obs once the GATE arms them. The deliverable is the
  **corrected gate** (block armed from `obs[3] ball-in-front`, mark from
  `obs[21] opp-in-front`, off_ball role), not new weights.
- A LONE RL agent always ends up on the ball (dribbles) → D1 role-flip to striker → block/
  mark never arm. block/mark only appear when the agent is OFF-BALL, i.e. in the
  multi-agent trío (`players=1,2,3`) where other agents handle the ball.

## 5. THE REAL BLOCKER: off-ball opponent perception collapses in multi-agent
- Trío: `opponent_seen_rate` = 0.09 (3-slot) / 0.00 (2-slot) vs 0.65–0.80 single-slot.
  → `executed_mark_rate=0` is a PERCEPTION artifact, NOT a defender failure (block still
  fired 0.06–0.085).
- CAUSE: head runs `LookActive(withBall=true)` → off-ball defender stares at the ball while
  opponents are upfield/around it, outside FoV. Reducing slots did NOT help → not GPU
  contention; it's gaze geometry. See `CODEX_OFFBALL_PERCEPTION_HANDOFF.md` (the fix).
- On a REAL NAO this is less severe (real vision + head scan) — mark is expected to fire in
  deployment; the collapse is a sim multi-agent artifact.

## 6. Multi-agent reset stability in RLMAStage7 (already fixed by Codex)
- Symptom: visible reset never converges; `fall_state=4` (robot topples on teleport) at the
  defender deep-spawn; or "world request timed out" with 2-3 slots (heavy render).
- FIX applied: `_defense_team_world` defender spawn → `x=clip(ball_x-700,-3400,-2600)`,
  `y=ball_y-side*700` (in-bounds deep defender). Plus the runner relaxes reset acceptance
  (`RL_VISIBLE_RESET_POSE_TOL_MM=400`, `RL_VISIBLE_RESET_MAX_FALL_STATE=3`,
  `RL_VISIBLE_RESET_UPRIGHT_STREAK=2`) — robots walk ~200mm off-pose in `gc playing`.
- WARNING: the relaxed acceptance can START episodes in marginal/colliding poses
  (`collision_rate` 0.25–0.50, low `ball_trusted`, `wire=FAIL`). It's a tradeoff: strict =
  can't reset; relaxed = noisier episodes. Don't over-relax for a real training run.

## 7. Hard invariants (from CLAUDE.md + this line) — never break
- `OBS_SIZE=47`, `ObsIndex`, `SKILL_ORDER` immutable. Checkpoints are shape-locked.
- `gate_skill_mask_from_obs` (train.py) is the SINGLE skill-mask source — env step, BC
  labels, PPO recompute must all agree (Design-A). Teacher labels go through the REAL gate
  (`teacher_multiagent_dataset._team_label_allowed`); a stale hand-mirror silently drops
  labels = the "gate mark-leak" bug.
- Episode steps **800**. Shoot band **2.5–3.0m**. `RL_GATE_SHOOT_OPEN_ENTER/EXIT=0.30/0.20`.
- NEVER overwrite the deployed `ppo_team_hsl2026.onnx` (v2b) or v4.1/v4.2 ONNX. New brains
  get NEW filenames.
- v2b@7500 (`runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt`) is the
  canonical warm-start/KL anchor — don't swap it for an untested checkpoint.
- Max 4 visible SimRobot slots (6 crashes Qt/GL). In practice 3 heavy slots already starve
  perception/resets — prefer 1 for clean perception.

## 8. Useful artifacts created this session (held uncommitted)
- Eval columns added: `opponent_seen_rate` (environment.py, from `nearest_opponent_dist
  < 4500`), `executed_mark_rate` + `policy_mark_rate` (environment.py + train.py column list).
- `tools/test_rlmastage7_trio.sh` (RL_MA_PROD_PLAYERS overridable; relaxed-reset env),
  `tools/test_rlmastage7_perception.sh`, `tools/run_defense_3v3_perception.sh`,
  `tools/verify_defender_gate_contract.py` (gate ALL-PASS check).
- `train_multiagent_production.py`: `RL_MA_PROD_FORCE_PERCEPTION_SCENE=1` bypasses the
  stage-8 forced `RLMAStage7_3D` (lets you force a different scene); stage 8 defaults to
  `RLMAStage7_3D`.

## 9. Status at end of session
- Defender = good on evidence (perceives ball 0.99 / opp 0.80 single-slot; defender_score
  0.82; block fires; picks mark 0.96 offline). Deployment-ready via corrected gate + real
  vision. Outstanding ONLY: live in-sim mark % (blocked by item 5 — needs the gaze fix).
- Striker = v4.2, mature (pass 0.99, floor 4/5).
- Next: (a) off-ball gaze fix → re-run live mark proof; (b) merged ONNX (see its handoff;
  prerequisite = the gaze fix). Family (v4.2 + pt_defender_47 dispatched by role) is the
  safe no-retrain baseline.
