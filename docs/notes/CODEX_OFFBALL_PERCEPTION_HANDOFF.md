# CODEX TASK — Fix off-ball defender OPPONENT perception (multi-agent gaze)

## ONE-SENTENCE GOAL
Make the OFF-BALL defender perceive opponents in the multi-agent scene by fixing its
**head/gaze** (it currently stares at the ball, so upfield/surrounding opponents fall
outside its camera FoV and `ObstacleModel` stays empty). Raise `opponent_seen_rate` for
the off-ball defender from ~0.0 to **>0.30** in the trío, WITHOUT changing perception
modules, obs, gate, reward, policy, or ball perception.

## WHY (verified this session — do not re-derive)
- Opponent perception is VISION-based: `ObstacleModel` ← `RobotDetector`/`PlayersDeeptector`
  NN on the rendered camera image (needs line-of-sight + the opponent in FoV).
- `nearestOpponentDist`/`nearest_opponent_front_dist` derive from `theObstacleModel`
  (`Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.cpp` ~lines
  276/402). Head currently runs `LookActive(withBall=true)` → gaze locked on the ball.
- Measured: a striker/near-ball agent perceives opponents 65–80% (`opponent_seen_rate`).
  The OFF-BALL defender (deep, facing the ball, opponents upfield/around it) perceives
  **0–9%**. Reducing sim slots 3→2 did NOT help → it is NOT GPU contention; it is **gaze
  geometry** (opponents are not in the FoV the head is pointing at).
- Single-robot scenes perceive fine because that agent is near the action. The issue is
  specific to the off-ball role in the multi-agent scene (`RLMAStage7_3D`, 7 real bodies).
- The defender's MARK skill (gate-armed from `obs[21] nearest_opponent_front_dist`) can
  never fire if opponents are never perceived → this gaze fix is the unblocker for the
  live mark proof AND for any defense retrain.

## STEP 1 — DIAGNOSE (mandatory, behind an env flag, removed before finishing)
Instrument and log, for the OFF-BALL defender agent during a trío run:
- head yaw/pitch over time,
- number of opponents currently in the camera FoV (from GroundTruthWorldState vs FoV),
- `obstacle_opponent_count` / `nearest_opponent_dist` per step.
Run: `RL_MA_PROD_PLAYERS=1,2,3 bash tools/eval/test_rlmastage7_trio.sh` (defender is the off-ball
slot). Confirm the hypothesis: head points at the ball; opponents are outside FoV.

## STEP 2 — FIX (smallest change that works)
Give the OFF-BALL role a head/gaze behavior that periodically SCANS the danger zone
(toward our goal / the opponents) instead of locking on the ball. Likely locations
(investigate; the fix is almost certainly C++ in SabanaHerons2026):
- The head-control skill the RL robots use (search for `LookActive`, `LookAtPoint`,
  `LookLeftAndRight`, head skill calls in `SkillBehaviorControl` and in the RL skill path
  `Src/Modules/BehaviorControl/.../RLSkillProvider*` / `RLBehavior`).
- Make gaze role-aware: striker/on-ball keeps `LookActive(withBall=true)`; the off-ball
  defender uses a scanning gaze (e.g. `LookActive(withBall=false)` or an alternating
  scan covering the cone toward our goal where attackers come from).
- If head control is reachable from the RL/Python side, prefer the smallest hook there;
  otherwise patch the C++ head behavior gated to the off-ball/defender role.

If the fix is C++: rebuild only what's needed (`pybh` extension and/or SimulatedNao) per
`CLAUDE.md` [removed from this branch; available on rl-v4.2-multiagent-gk] "Rebuilding the C++ Bridge". Note current pybh binary is from Jun-10; if you
rebuild, also see `CODEX_SESSION_LESSONS_20260618.md` [removed from this branch; available on rl-v4.2-multiagent-gk] about the Trionda ball cfg.

## ALLOWED TO CHANGE
- Head/gaze behavior for the off-ball/defender role (C++ behavior or a Python head hook).
- Temporary diagnostic logging (env-gated, default off).
- `tools/eval/test_rlmastage7_trio.sh` env knobs only.

## FORBIDDEN — DO NOT TOUCH
- `observation.py` (`OBS_SIZE=47`, `ObsIndex`), `action.py` (`SKILL_ORDER`).
- `train.py::gate_skill_mask_from_obs`, `gate.py`, `reward.py`.
- The perception MODULE wiring for opponents (`RobotDetector`/`ObstacleModel` providers in
  `threads.cfg`) — do NOT swap in an oracle/ground-truth opponent source (that was tried and
  rejected: sim2real gap; see lessons doc). The fix is GAZE, not the detector.
- Ball perception (`BallPerceptor`), the striker's gaze, any policy/checkpoint, `runs/`.
- No retrain, no ONNX export, no commit.

## ACCEPTANCE
1. Trío run (`RL_MA_PROD_PLAYERS=1,2,3 bash tools/eval/test_rlmastage7_trio.sh`) reaches `[eval]`.
2. `opponent_seen_rate > 0.30` (up from ~0.0) — the off-ball defender now sees opponents.
3. `natural_ball_seen_rate` for on-ball agents NOT regressed (striker still perceives ball).
4. Report `executed_mark_rate` / `executed_block_rate` / `opponent_seen_rate` / `defender_score`.
5. Diagnostic logging removed/env-gated-off. Changes uncommitted.

## ENV (single sim session)
PYTHONPATH=/home/luis/GitHub/RL:/home/luis/GitHub/SabanaHerons2026/Make/Python ; DISPLAY=:1 ;
python=~/anaconda3/envs/RLNAO/bin/python ; ALWAYS `pgrep -x SimRobot` empty before launch.
