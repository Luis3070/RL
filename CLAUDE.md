# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## Repository layout transition

The repository moved from ~80 flat root modules to the `rlfootball` package. Older documents and notes still use the flat names (`train.py`, `config.py`, `from config import ...`).

- Translate old names through the "Module map" in `docs/REPO_LAYOUT.md` (old -> new path, one row per module). Rule: `<name>.py` -> `rlfootball/<sub>/<name>.py`; names did not change.
- Integration rules (where new code goes, PR checklist, porting work written against the old layout): `docs/PROJECT_STRUCTURE.md`.
- Entry points are modules: `python train.py` -> `python -m rlfootball.training.train`.

Where things go:

- Shared contracts (obs, action, gate, reward, config, backend): `rlfootball/core/`
- Stages and scenarios: `rlfootball/curriculum/`; teachers: `rlfootball/teachers/`; environments: `rlfootball/env/`
- Dataset builders: `rlfootball/datasets/`; trainers: `rlfootball/training/`; evaluators: `rlfootball/evaluation/`; probes: `rlfootball/diagnostics/`
- Launchers and recipes: `tools/{run,export,eval,dataset,paper,misc}/`; env setup: `scripts/`; policies: `models/<role>/`; tests: `tests/`; docs: `docs/NN-title.md` (+ index row in `docs/README.md`)
- Dependency direction: `core -> curriculum -> teachers -> env -> datasets/training/evaluation`; never import from a higher layer.

Rules:

- Run everything from the repository root (`python -m ...`, `pytest`, `tools/**/*.sh`).
- Never commit `runs/`, `.env`, datasets (`*.npz`), logs or credentials.
- No hardcoded absolute paths. Use `RL_REPO_DIR` and `SABANAHERONS_DIR` (read from the optional gitignored `.env`; template `.env.example`; shell launchers `source scripts/env.sh`, which is bash-only).
- Small Conventional Commit messages (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`). Do not add AI attribution or co-author lines to commits.

## Project summary

Python RL training system (MAPPO via `skrl`) that controls NAO soccer robots through B-Human's SimRobot simulator. The Python policy replaces B-Human's strategy layer, sending `SkillRequest` commands (stand/walk/shoot/pass/dribble/block/mark/observe) through the in-process `pybh` binding (C++ shared memory). B-Human handles all motion execution. The twin repos must stay in sync: this repo is Python; `SabanaHerons2026/` (sibling directory, or `SABANAHERONS_DIR`) is C++ and SimRobot. The old `bridge/` subprocess stack was removed; do not reintroduce it.

Required `SabanaHerons2026` branches: `rl-simrobot3d` for multi-agent training (bridge version 30), `hsl2026-ppo-team-v4` for deploying the ONNX policies. Do not edit the sibling repo from changes in this one.

## Environment setup

`environment.yml` is outdated (Python 3.10, no `onnx`/`onnxruntime`/`pytest`). Preferred: `scripts/setup_validation_env.sh` (creates `.venv`), then run commands through `scripts/run_with_validation_env.sh`, which sets `PYTHONPATH` to `$SABANAHERONS_DIR/Make/Python`. Use the Python version your `pybh` build targets.

`pybh` is a compiled extension built in `$SABANAHERONS_DIR/Make/Python` (`python3 setup.py build_ext --inplace`) and must be on `PYTHONPATH`. The SimRobot binary path is derived from `SABANAHERONS_DIR` in `rlfootball/core/config.py` (`SIMROBOT_BIN`, Linux build); on other platforms override it there.

## New environment variables

Read in code (verified with `rg`); all optional.

- `RL_SIMROBOT_BIN`: overrides the SimRobot binary path (`rlfootball/core/config.py`, default under `SABANAHERONS_DIR`).
- `RL_MA_DYNAMIC_WORLD_UPDATES`: `0`/`false`/`no` stops the env teleporting scripted opponents, so they run their own B-Human behaviour (`rlfootball/curriculum/curriculum.py`, default on).
- `RL_MA_MATCH_SUMMARY`: `1` enables the match-summary (eval) path of the multi-agent teacher dataset collector (`rlfootball/datasets/teacher_multiagent_dataset.py`, default `0`).
- `RL_BC_KL_SCALE`: weight of the KL-to-warm-start anchor in the BC replay loss, an ablation knob (`rlfootball/training/train.py`, default `1.0`).
- `RL_SEED`: seeds torch and numpy for independent ablation runs (`rlfootball/training/train.py`).
- `RL_MA_PROD_BC_WARM_START`: toggles the BC warm start in the multi-agent production fine-tune (`rlfootball/training/train_multiagent_production.py`, default on).

## Key commands

```bash
# Contract tests that need no simulator
python -m pytest -q tests/test_wire_audit.py tests/test_behavioral_stability.py
pytest                                       # full suite; needs a working pybh build

# Training: no CLI flags, configured by RL_* env vars (see _run_cfg_from_env in train.py)
RL_RUN_MODE=smoke RL_SMOKE_STEPS=4 python -m rlfootball.training.train
RL_RUN_MODE=train RL_TRAIN_STEPS=800000 python -m rlfootball.training.train_multiagent

# Production fine-tune and BC
python -m rlfootball.training.train_production --dataset <samples.npz> --checkpoint <ckpt.pt>
python -m rlfootball.training.train_multiagent_production --stage-id 7
python -m rlfootball.training.train_bc

# Final all-learned team: live matches, trainers (need runs/ inputs, pybh and SimRobot; docstrings have the flags)
python -m rlfootball.evaluation.run_v5_vs_bhuman_match --launch --seconds 600 --output runs/<batch>/match_1.json
python -m rlfootball.evaluation.run_probe_match --controller v5 --launch --output runs/<dir>/match.json
python -m rlfootball.training.train_dagger_v5   # then train_distill_hybrid, learned_coordinator, kickstart_ppo

# Paper evidence (run as plain scripts from the repo root; outputs in results/paper/)
python tools/eval/ablation_offline_eval.py
python tools/paper/summarize_v5_vs_bhuman_matches.py runs/paper_match_v5_vs_bhuman_live/batch_<stamp>
python tools/paper/paper_figures_final.py

# Datasets, evaluation, diagnostics
python -m rlfootball.datasets.teacher_multiagent_dataset
python -m rlfootball.evaluation.behavioral_evaluator
python -m rlfootball.diagnostics.diag_n6_slots
python -m tests.smoke_test_skill_requests --backend simrobot_visible --simrobot-profile 3d --launch-simrobot
```

The exact recipes for each policy version are in `tools/run/*.sh` (for example `run_v4_2.sh`, `train_merge_v5.sh`). Flags above were read from the source (argparse), not from `--help` runs; the default backend is `simrobot_visible`. Many commands require the simulator and could not be re-run after the refactor.

## Architecture

### Data flow

```
Python policy (rlfootball/training/train.py)
  -> StructuredSkillAction (rlfootball/core/action.py)
  -> PyBHBackend.step() (rlfootball/core/pybh_backend.py)
  -> pybh.rl_set_action()          <- shared memory boundary
  -> RLSkillProvider.cpp
  -> SkillRequest -> B-Human SkillBehaviorControl -> MotionEngine -> SimRobot
```

Observations travel the reverse path: `pybh.rl_get_obs()` -> `BackendObservation` -> `RealSimRobotEnv.step()` -> 47-float obs vector -> policy.

### Core files

| File | Role |
|------|------|
| `rlfootball/core/config.py` | Central constants: paths, field dims, `EnvConfig`, `TrainConfig`, `ActionConfig`, scene presets |
| `rlfootball/core/action.py` | `StructuredSkillAction`, `SkillType` enum (8 skills), `SKILL_ORDER`, masks, `ACTION_PARAM_MASKS` |
| `rlfootball/core/observation.py` | `ObsIndex` enum, `OBS_SIZE = 47` |
| `rlfootball/core/gate.py` | `SkillGate` (forbid-invalid masking, bakes arm bits into the obs) and `gate_skill_mask_from_obs()` |
| `rlfootball/core/pybh_backend.py` | `PyBHBackend`: SimRobot process and shared memory, `WorldSpec` for resets |
| `rlfootball/core/reward.py` | `RewardWeights`, `evaluate_step()`: dense shaping |
| `rlfootball/core/goalkeeper_action.py` | Goalkeeper action space and mask |
| `rlfootball/curriculum/curriculum.py` | `StageSpec`, `get_stage_spec()`, dynamic obstacle logic |
| `rlfootball/curriculum/production_curriculum.py` | Production scenarios shared by evaluator and BC dataset generation |
| `rlfootball/teachers/teacher_policy.py` | Single-agent heuristic teacher, `stage_aware_teacher_action()` |
| `rlfootball/teachers/multiagent_teacher.py` | Multi-agent teacher; `assign_roles` |
| `rlfootball/teachers/defender_policy.py`, `goalkeeper_teacher.py` | Defender and goalkeeper teachers |
| `rlfootball/env/environment.py` | `RealSimRobotEnv(ParallelEnv)`: PettingZoo env wrapping backend + reward + gate |
| `rlfootball/env/team_parallel_environment.py` | Team-coordinated multi-slot env for stage 6/7 MAPPO |
| `rlfootball/training/train.py` | MAPPO loop, `PolicyNetwork`, `ValueNetwork`; re-exports `gate_skill_mask_from_obs` |
| `rlfootball/training/train_production.py`, `train_multiagent_production.py` | Production fine-tune entry points; tunable BC/PPO knobs |
| `rlfootball/training/train_bc.py` | Behavioral Cloning warm start |
| `rlfootball/datasets/teacher_production_dataset.py` | Runs the teacher on hard scenarios, filters episodes, writes `samples.npz` |
| `rlfootball/evaluation/behavioral_evaluator.py` | Grades checkpoints per production scenario (pass/fail) |
| `tools/export/export_ppo_policy.py` (+ `_team`, goalkeeper) | ONNX export to `models/<role>/` |

### Observation vector (`OBS_SIZE = 47`)

- [0:3] Robot pose: `robot_x`, `robot_y`, `robot_theta`
- [3:13] Ball perception: relative pos, predicted end pos, velocity, time-since-seen, seen%, consistency flag
- [13:17] Scoring: `can_score_now`, `shot_quality_no_obstacles`, `shot_opening_with_obstacles`, `pass_options_count`
- [17:23] Proximity: nearest teammate/opponent/uncertain-obstacle distances (all directions and front)
- [23:26] Skill-gate bits (Design A): `shoot_armed`, `dribble_armed`, `shoot_arm_progress`
- [26:41] Multi-agent context: teammate relative positions, team ball estimate, teammate engagement flags, goal vector
- [41:44] Skill-arm extension: `pass_armed`, `observe_armed`, `pass_arm_progress` (zero by default)
- [44:47] Role one-hot: `is_striker`, `is_open_support`, `is_off_ball_support` (exactly one is 1.0 per RL agent; the goalkeeper reads none)

Older single-agent checkpoints used the 26-dim (and earlier 41-dim) layout; training pads them when warm-starting (`_load_state_dict_obs_padded`).

### Roles and gate contract

`gate_skill_mask_from_obs` (in `rlfootball/core/gate.py`) is a pure function of obs columns 3, 21, 23, 24, 41, 42, 44-46:

- Obs < 47 dims: single-agent contract. Shoot/dribble follow their armed flag; once finish-armed, stand/walk are masked.
- Obs >= 47 with a role bit: striker uses shoot/dribble/pass by armed flags; open support gets stand/walk (+ block when `dribble_armed`); off-ball support (defender) gets stand/walk, dribble when armed, block when the ball is in front (`ball_rel_x` < 1800 mm) or `dribble_armed`, mark when an opponent is in front (< 1500 mm), no shoot, no pass. Any non-striker may observe when `observe_armed`.
- Obs >= 47 without role bits: falls back to the single-agent contract.

### Curriculum stages

| Stage | Name | Notes |
|-------|------|-------|
| 0 | `stage_0_clean_handoff` (`stage_0a_walk_approach` is the walk-only variant) | Walk to ball, clean handoff |
| 1 | `stage_1_finish_after_handoff` | Full chain without opponents |
| 2 | `stage_2` | Static opponents |
| 3 | `stage_3` | Dynamic/scripted opponents |
| 4 | `stage_4_production` | Full production field (single-agent deployment target) |
| 6 | `multiagent_warmup_2v0` | Multi-agent |
| 7 | `multiagent_adversarial_3v3` | Multi-agent |
| 8 | `multiagent_defender_3v3` | Defender training |

Active skills per stage are in `StageSpec.active_skill_names` (`rlfootball/curriculum/curriculum.py`).

## Hard invariants - do not break

**Design A:** the forbid-invalid skill mask used at PPO update time is derived purely from the stored obs columns. The gate bits in the observation, the action mask in `env.step()`, and the teacher dataset labels must all agree. `gate_skill_mask_from_obs()` is the single source; do not compute the mask elsewhere.

**`OBS_SIZE = 47` and `SKILL_ORDER`** cannot change without invalidating loaded checkpoints (`.pt`) and the exported ONNX policies in `models/`.

**Max 4 visible SimRobot slots** (recorded in `docs/notes/10-graph-coordination-post-production-roadmap.md`): 6 slots caused Qt/OpenGL timeouts and crashed runs. `RLSharedState::MAX_PLAYERS` is 6 on the C++ side (`docs/22-handoff-optimizacion-training-32cores.md`).

**Shared-memory layout:** after any change to `RLSharedState`, `RLSkillProvider.cpp` or `Module.cpp` in `SabanaHerons2026/Src/`, rebuild `pybh` and `SimulatedNao`. A mismatched layout means `obs_ready` never becomes `True`. The segment name used here is `/pybh_rl_shared_state_v3`.

```bash
cd "$SABANAHERONS_DIR/Make/Python" && python3 setup.py build_ext --inplace
cmake --build "$SABANAHERONS_DIR/Build/Linux/CMake/Develop" --target SimulatedNao
```

## Data and policies

- `runs/` is gitignored (datasets, logs, checkpoints). Tracked deliverables are `models/<role>/ppo_*.onnx` + `ppo_*.json` (manifest `checkpoint` fields are repo-relative `runs/...` paths) and two defender checkpoints in `models/defender/checkpoints/`.
- Production warm-start checkpoints are looked up under `runs/production_finetune_v1/` (see `rlfootball/training/train_production.py`); they are not in the repository, so ask a teammate or retrain.
- Policy generations: striker v4.1/v4.2 and merged brain v5 (all three roles) in `models/team/`; goalkeeper variants in `models/goalkeeper/`.
- Final all-learned team and learned support coordinator: `models/team/ppo_team_hsl2026_final_alllearned.onnx` and `ppo_coord_hsl2026_final_learned_coordinator.onnx`. They have no `ppo_*.json` manifest.
- Paper evidence tables are in `results/paper/` (held-out ablation CSV; v5-vs-B-Human match CSVs), regenerated by `tools/eval/` and `tools/paper/`. The `runs/` inputs behind them are not tracked; see README "Latest progress: from v5 to an all-learned team".

## Removed or historical (do not follow old instructions)

- No `bridge/` subprocess stack: `pybh` runs in-process.
- Root-level `FOR_TRAIN_PRODUCTION.md` and the `.docs/` folder no longer exist; their material lives in `docs/` and `docs/notes/`.
- The "graph coordination (wrapper only, no retraining)" sprint is a historical plan: `docs/notes/10-graph-coordination-post-production-roadmap.md`. It is not an active constraint.
- Windows-specific setup from the old instructions was dropped; the supported flow is Linux.
- Old `train.py --backend ... --mode ...` flags do not exist; use `RL_*` environment variables.

## Documentation map

- `docs/README.md`: index of all documents (most are Spanish working logs)
- `docs/REPO_LAYOUT.md`: package layout and old -> new module table
- `docs/PROJECT_STRUCTURE.md`: structure, transition guide, integration rules
- `docs/05-premisas-irrompibles.md`: hard constraints (field bounds, skill properties)
- `docs/12-wire-rl-realista-final.md`: wiring diagram (Python <-> C++)
- `docs/37-striker-v2b-to-v4.2-and-cpp-transplant.md`, `docs/38-merged-striker-defender-and-naov6-transplant.md`: striker v4.2 and merged v5 policies
- `docs/PAPER_TRACEABILITY.md`: paper claims to evidence
