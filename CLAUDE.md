# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Python RL training system (MAPPO via `skrl`) that controls NAO soccer robots through B-Human's SimRobot simulator. The Python policy replaces B-Human's strategy layer, sending `SkillRequest` commands (walk/dribble/shoot/pass/…) via a C++ shared-memory bridge (`pybh`). B-Human handles all motion execution. The twin repos must stay in sync: this repo (`RL/`) contains Python; `SabanaHerons2026/` contains C++ and SimRobot.

## Environment Setup

**Conda** (recommended):
```bash
conda env create -f environment.yml   # creates "RLRobocup"
conda activate RLRobocup
```

`pybh` is a compiled C++ extension that must be on `PYTHONPATH`. On Linux:
```bash
export PYTHONPATH=/path/to/SabanaHerons2026/Make/Python
```

On Windows (this machine — see note below):
```powershell
$env:PYTHONPATH="C:\Users\MSI\Desktop\Programacion\SabanaHerons2026\Make\Python"
```

**Windows note:** `config.py:12` hard-codes the Linux SimRobot binary path. If running on Windows, override `SIMROBOT_BIN` or patch `config.py` to point to `Build/Windows/SimRobot/Develop/SimRobot.exe`.

## Key Commands

```bash
# Smoke test — validates bridge + MAPPO integration (4 steps, fast)
python train.py --backend simrobot --mode smoke --smoke-steps 4

# Smoke with visible SimRobot window
python train.py --mode smoke --backend simrobot_visible --simrobot-profile 3d --launch-simrobot --smoke-steps 4

# Short training run (stage 0)
python train.py --backend simrobot --mode train --train-steps 8000 --stage-id 0

# Production fine-tune (loads checkpoint, runs stage-4 for 80k steps with 4 parallel slots)
python train_production.py

# Behavioral Cloning pre-training (warm-start)
python train_bc.py

# Heuristic agent experiment (no RL, validates bridge + reward)
python heuristic_agent_experiment.py --episodes 10 --no-plot --backend simrobot --simrobot-profile 2d

# Live visible probe with heuristic policy
python live_simrobot_probe.py --launch --policy heuristic

# Evaluate a checkpoint against 5 production scenarios
python behavioral_evaluator.py --checkpoint runs/.../checkpoint.pt

# Bridge smoke (all 8 skills, one-by-one)
python smoke_test_skill_requests.py --backend simrobot_visible --simrobot-profile 3d --launch-simrobot

# Wire audit (obs/gate plumbing health check)
python test_wire_audit.py
```

## Architecture

### Data Flow

```
Python policy (train.py)
  → StructuredSkillAction (action.py)
  → PyBHBackend.step() (pybh_backend.py)
  → pybh.rl_set_action()          ← shared memory boundary
  → RLSkillProvider.cpp
  → SkillRequest → B-Human SkillBehaviorControl → MotionEngine → SimRobot
```

Observations travel the reverse path: `pybh.rl_get_obs()` → `BackendObservation` → `RealSimRobotEnv.step()` → 41-float obs vector → policy.

### Core Files

| File | Role |
|------|------|
| `config.py` | Central constants: paths, field dims, `EnvConfig`, `TrainConfig`, `ActionConfig`, scene presets |
| `action.py` | `StructuredSkillAction`, `SkillType` enum, `SKILL_ORDER`, masks, numpy conversion |
| `observation.py` | `ObsIndex` enum, `OBS_SIZE = 41` |
| `gate.py` | `SkillGate` — forbid-invalid masking; bakes `shoot_armed`/`dribble_armed` into obs[23:26] |
| `environment.py` | `RealSimRobotEnv(ParallelEnv)` — PettingZoo env, wraps backend + reward + gate |
| `pybh_backend.py` | `PyBHBackend` — manages SimRobot process/shared-mem, `WorldSpec` for resets |
| `reward.py` | `RewardWeights`, `evaluate_step()` — 33-term dense shaping |
| `curriculum.py` | `StageSpec` 0–4, `get_stage_spec()`, dynamic obstacle logic |
| `production_curriculum.py` | 5 named `ProductionScenario` objects used by evaluator and BC dataset |
| `train.py` | MAPPO training loop, `PolicyNetwork`, `ValueNetwork`, `gate_skill_mask_from_obs()` |
| `train_bc.py` | Behavioral Cloning warm-start training |
| `train_production.py` | Stage-4 fine-tune entry point; tunable BC/PPO knobs at top of file |
| `teacher_policy.py` | Heuristic expert; `stage_aware_teacher_action()` generates BC demonstrations |
| `teacher_production_dataset.py` | Runs teacher on hard scenarios, filters bad episodes, writes `samples.npz` |
| `behavioral_evaluator.py` | Grades checkpoints; 5 scenarios, pass/fail per scenario |
| `wrapper.py` | `collect_episode()`, thin skrl-compat wrapper |

### Observation Vector (`OBS_SIZE = 41`)

- [0:3] Robot pose: `robot_x`, `robot_y`, `robot_theta`
- [3:13] Ball perception: relative pos, predicted end pos, velocity, time-since-seen, seen%, trust flags
- [13:17] Scoring: `can_score_now`, `shot_quality_no_obstacles`, `shot_opening_with_obstacles`, `pass_options_count`
- [17:23] Proximity: nearest teammate/opponent/uncertain distances (front and all-directions)
- [23:26] **Skill-gate bits (Design-A):** `shoot_armed`, `dribble_armed`, `shoot_arm_progress`
- [26:41] Multi-agent context: 3 teammate relative positions, team ball estimate, teammate engagement flags, goal vector

### Curriculum Stages

| Stage | Skills active | Scenario |
|-------|--------------|---------|
| 0 | walk | Walk to ball, clean handoff (no shoot) |
| 1 | walk, dribble, shoot | Full chain without opponents |
| 2 | walk, dribble, shoot | Static opponents |
| 3 | walk, dribble, shoot | Dynamic/scripted opponents |
| 4 | walk, dribble, shoot | Full production field (deployment target) |

## Hard Invariants — Do Not Break

**Design-A:** The forbid-invalid skill mask used at PPO update time is derived purely from `obs[23:26]`. The gate bits in the observation, the action mask in `env.step()`, and the teacher dataset labels must all agree. `gate_skill_mask_from_obs()` is the single source — do not compute the mask elsewhere.

**`OBS_SIZE = 41` and `SKILL_ORDER`** cannot change without invalidating any loaded checkpoint (`.pt`). The warm-start checkpoint at `runs/production_finetune_v1/checkpoint_stage4_step35000.pt` was trained on the 26-dim layout (multi-agent dims were added after); padding logic in training handles the mismatch.

**No retraining during the graph-coordination sprint** (see `.docs/10-graph-coordination-post-production-roadmap.md`): the `MultiAgentCoordinator` wrapper logic must work on top of the existing policy without changing `OBS_SIZE`, `reward.py`, or `gate.py`.

**Max 4 visible SimRobot slots.** 6 slots cause Qt/OpenGL timeouts and crash the run.

## Active Sprint: Graph Coordination (Version B — Wrapper Only)

See `.docs/10-graph-coordination-post-production-roadmap.md`. The goal is a `MultiAgentCoordinator` that:
1. Computes an adjacency matrix from robot positions each step.
2. Applies a hard skill mask (`walk`-only) to support robots in open passing lanes.
3. Applies a residual direction override to move supports to open positions (not toward the ball).

This is entirely in Python inference — no retraining. The single-agent policy weights are frozen.

## Rebuilding the C++ Bridge

Required whenever `RLSharedState`, `RLSkillProvider.cpp`, or `Module.cpp` in `SabanaHerons2026/Src/` change. If Python and SimRobot have mismatched shared-memory layouts, `obs_ready` will never be `True`.

```bash
# Rebuild pybh Python extension
cd /path/to/SabanaHerons2026/Make/Python
python3 setup.py build_ext --inplace

# Rebuild SimulatedNao (if C++ RL bridge files changed)
cmake --build /path/to/SabanaHerons2026/Build/Linux/CMake/Develop --target SimulatedNao
```

On Windows: open `SabanaHerons2026/Make/Windows/B-Human.sln` in Visual Studio, build **SimulatedNao** in **Develop** config.

## Production Checkpoint Status

`runs/production_finetune_v1/checkpoint_stage4_step35000.pt` — competent walk+dribble, `policy_shoot_rate = 0.000` (local optimum). The `train_production.py` patched defaults (stronger BC β, slower decay) are designed to push it out of this optimum. Eval target: `behavioral_evaluator ≥ 3/5`.

## Documentation Map

- `docs/README.md` — architecture overview index
- `docs/05-premisas-irrompibles.md` — hard constraints (field bounds, skill properties)
- `docs/12-wire-rl-realista-final.md` — complete wiring diagram (Python ↔ C++)
- `.docs/01-ppo-takeover-curriculum-handoff.md` — takeover curriculum rationale
- `.docs/09-production-selfplay-and-multiagent-handoff.md` — 4v4 multi-agent expansion plan
- `.docs/10-graph-coordination-post-production-roadmap.md` — current sprint plan
- `FOR_TRAIN_PRODUCTION.md` — production fine-tune quick-start and triage guide
