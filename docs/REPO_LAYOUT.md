# Repository layout and package map

Layout of the Python package that replaced the 80 flat root `*.py` modules.
Status: **implemented**. The module map below is also the migration table from
the old flat names (used by older docs and notes) to the new paths.

## Purpose

- One importable package, `rlfootball`, instead of 80 modules sharing the repo root.
- Tests live in `tests/`. Entry points run with `python -m rlfootball.<sub>.<module>`.
- Subpackages form an acyclic layering (see "Dependency analysis").

## Target tree

```text
rlfootball/
  core/         action, config, observation, reward, gate, pybh_backend, wire_audit, runtime   (leaf layer)
  curriculum/   stage specs and production scenarios
  teachers/     heuristic teachers and defender policy
  env/          environments and the skrl wrapper
  datasets/     teacher dataset builders, mergers, audits
  training/     BC / PPO / MAPPO trainers and repair trainers
  evaluation/   checkpoint, behavioral and teacher evaluators
  diagnostics/  probes, GUI checks, sweeps, visualisation
tests/          test_*.py, smoke_test_skill_requests.py
tools/  scripts/  docs/  models/  results/   (unchanged)
```

Each subpackage gets an empty `__init__.py`. Import rule: absolute only,
`from rlfootball.core.config import ENV`.

## Subpackages

| Subpackage | Modules | Allowed to import |
|---|---|---|
| `core` | 9 | (none) |
| `curriculum` | 2 | core |
| `teachers` | 4 | core, curriculum |
| `env` | 4 | core, curriculum, teachers |
| `datasets` | 12 | core, curriculum, teachers, env |
| `training` | 19 | core, curriculum, teachers, env, datasets |
| `evaluation` | 8 | all above except diagnostics |
| `diagnostics` | 14 | core, curriculum, teachers, env |
| `tests` | 8 | anything |

Deviation from the initial proposal (reasons):

- New `core` layer: `env <-> curriculum <-> teachers` was a 3-way cycle (see below).
  `core` holds the leaf modules every other layer needs.
- `wire_audit` moved from `diagnostics` to `core`: `train.py` imports it at top level
  and it only depends on `config`.
- `visible_parallel_runtime` is in `core` (depends only on `config`; used by env, train, datasets).
- `obs_invariants` moved from `env` to `diagnostics` (it is a CLI that imports `environment`).
- `run_goalkeeper_production`, `export_ppo_defender_obs26_skill_selector_v1` and
  `recover_defender_checkpoint_from_onnx` go to `training` (they orchestrate or post-process trainers);
  `compare_defender_checkpoints` goes to `evaluation`.

## Module map (80 modules)

Kind: `entry` = has `if __name__ == "__main__"`, `mod` = library, `test` = test.
LOC = lines. Imp = importers (root modules / tools+scripts / tests).

| Old (root) | New | Kind | LOC | Imp | Role |
|---|---|---|---|---|---|
| `action.py` | `rlfootball/core/action.py` | mod | 136 | 36/11/3 | Structured skill action space (SkillType, masks, clamping) |
| `config.py` | `rlfootball/core/config.py` | mod | 251 | 44/6/4 | Central configuration (scene, action, backend, env, train) |
| `gate.py` | `rlfootball/core/gate.py` | mod | 268 | 6/4/0 | Unified, stateful skill gate (dynamic forbid-invalid masking) |
| `goalkeeper_action.py` | `rlfootball/core/goalkeeper_action.py` | mod | 89 | 6/7/0 | Goalkeeper structured action space and mask |
| `observation.py` | `rlfootball/core/observation.py` | mod | 68 | 11/6/2 | ObsIndex: observation column layout |
| `pybh_backend.py` | `rlfootball/core/pybh_backend.py` | mod | 1638 | 25/9/5 | pybh/SimRobot backend and bridge settings |
| `reward.py` | `rlfootball/core/reward.py` | mod | 836 | 7/0/1 | Reward weights, step features and evaluation |
| `visible_parallel_runtime.py` | `rlfootball/core/visible_parallel_runtime.py` | mod | 365 | 5/0/0 | Xvfb/SimRobot process runtime for visible parallel slots |
| `wire_audit.py` | `rlfootball/core/wire_audit.py` | mod | 139 | 1/0/1 | Wire/observation metric audit |
| `curriculum.py` | `rlfootball/curriculum/curriculum.py` | mod | 1132 | 17/0/0 | Stage specs and dynamic world state for the curriculum |
| `production_curriculum.py` | `rlfootball/curriculum/production_curriculum.py` | mod | 867 | 4/2/0 | Shared production scenarios for both behavioral evaluation and BC dataset generation |
| `defender_policy.py` | `rlfootball/teachers/defender_policy.py` | mod | 172 | 5/0/0 | Defender actor, skill mask and teacher action |
| `goalkeeper_teacher.py` | `rlfootball/teachers/goalkeeper_teacher.py` | mod | 299 | 4/7/0 | Goalkeeper heuristic teacher (GKCognition) |
| `multiagent_teacher.py` | `rlfootball/teachers/multiagent_teacher.py` | mod | 1614 | 4/4/1 | Multi-agent heuristic teacher — MapleTeacherPolicy |
| `teacher_policy.py` | `rlfootball/teachers/teacher_policy.py` | mod | 204 | 6/0/0 | Stage-aware single-agent heuristic teacher |
| `environment.py` | `rlfootball/env/environment.py` | mod | 1917 | 11/0/2 | RealSimRobotEnv: main single-slot environment |
| `parallel_environment.py` | `rlfootball/env/parallel_environment.py` | mod | 185 | 3/0/0 | ParallelSlotSimRobotEnv: multi-slot environment |
| `team_parallel_environment.py` | `rlfootball/env/team_parallel_environment.py` | mod | 801 | 1/2/0 | Team-coordinated multi-slot env for stage 6/7 MAPPO training |
| `wrapper.py` | `rlfootball/env/wrapper.py` | mod | 238 | 1/0/0 | RealSimRobotEnvWrapper (skrl-facing wrapper) |
| `build_stage8_defense_dataset.py` | `rlfootball/datasets/build_stage8_defense_dataset.py` | entry | 268 | 0/0/0 | Build the stage-8 defense dataset from source datasets |
| `collect_preshoot_blocked_teacher_dataset.py` | `rlfootball/datasets/collect_preshoot_blocked_teacher_dataset.py` | entry | 48 | 0/0/0 | Targeted teacher collection for pre-shoot blocked cases |
| `generate_defender_dataset.py` | `rlfootball/datasets/generate_defender_dataset.py` | entry | 103 | 2/0/0 | Generate defender BC samples |
| `merge_goalkeeper_teacher_datasets.py` | `rlfootball/datasets/merge_goalkeeper_teacher_datasets.py` | entry | 116 | 0/0/0 | Merge goalkeeper teacher datasets |
| `merge_multiagent_teacher_datasets.py` | `rlfootball/datasets/merge_multiagent_teacher_datasets.py` | entry | 239 | 0/0/0 | Merge multi-agent teacher datasets |
| `merge_production_datasets.py` | `rlfootball/datasets/merge_production_datasets.py` | entry | 147 | 0/0/0 | Merge production BC datasets |
| `subsample_multiagent_attack_dataset.py` | `rlfootball/datasets/subsample_multiagent_attack_dataset.py` | entry | 124 | 0/0/0 | Subsample the multi-agent attack dataset |
| `teacher_dataset.py` | `rlfootball/datasets/teacher_dataset.py` | entry | 641 | 1/0/0 | Single-agent teacher BC dataset collection |
| `teacher_goalkeeper_dataset.py` | `rlfootball/datasets/teacher_goalkeeper_dataset.py` | entry | 502 | 1/0/0 | Goalkeeper teacher dataset collection |
| `teacher_multiagent_dataset.py` | `rlfootball/datasets/teacher_multiagent_dataset.py` | entry | 1758 | 0/0/0 | Multi-agent BC dataset collection over the team production scenarios |
| `teacher_multiagent_dataset_audit.py` | `rlfootball/datasets/teacher_multiagent_dataset_audit.py` | entry | 315 | 0/0/0 | Audit multi-agent teacher datasets |
| `teacher_production_dataset.py` | `rlfootball/datasets/teacher_production_dataset.py` | entry | 586 | 1/0/0 | Production BC dataset collection over the deployment-facing situations |
| `export_ppo_defender_obs26_skill_selector_v1.py` | `rlfootball/training/export_ppo_defender_obs26_skill_selector_v1.py` | entry | 77 | 0/0/0 | Export defender obs26 skill selector to ONNX/JSON |
| `recover_defender_checkpoint_from_onnx.py` | `rlfootball/training/recover_defender_checkpoint_from_onnx.py` | entry | 99 | 0/0/0 | Recover a defender checkpoint from ONNX |
| `run_goalkeeper_production.py` | `rlfootball/training/run_goalkeeper_production.py` | entry | 136 | 0/0/0 | Goalkeeper production pipeline runner |
| `train.py` | `rlfootball/training/train.py` | entry | 2472 | 11/5/1 | Main PPO/MAPPO training and shared policy/eval helpers |
| `train_bc.py` | `rlfootball/training/train_bc.py` | entry | 394 | 0/0/0 | Behavior-cloning trainer |
| `train_defender_bc.py` | `rlfootball/training/train_defender_bc.py` | entry | 178 | 0/0/0 | Defender BC trainer |
| `train_goalkeeper.py` | `rlfootball/training/train_goalkeeper.py` | entry | 288 | 0/0/0 | Goalkeeper RL training entrypoint — single-agent, stage 9 |
| `train_goalkeeper_bc.py` | `rlfootball/training/train_goalkeeper_bc.py` | entry | 292 | 2/0/0 | Goalkeeper BC trainer |
| `train_goalkeeper_ppo.py` | `rlfootball/training/train_goalkeeper_ppo.py` | entry | 457 | 2/3/0 | Goalkeeper PPO trainer |
| `train_goalkeeper_ppo_closedloop.py` | `rlfootball/training/train_goalkeeper_ppo_closedloop.py` | entry | 411 | 0/0/0 | Closed-loop GK PPO: optimize GOAL-ACTUALLY-PREVENTED, not teacher-skill-match |
| `train_multiagent.py` | `rlfootball/training/train_multiagent.py` | entry | 393 | 1/0/0 | Multi-agent MAPPO training entrypoint — sibling of `train.py` |
| `train_multiagent_production.py` | `rlfootball/training/train_multiagent_production.py` | entry | 367 | 0/2/0 | Multi-agent production fine-tune entrypoint — sibling of `train_production.py` |
| `train_param_repair.py` | `rlfootball/training/train_param_repair.py` | entry | 355 | 0/0/0 | Parameter-head repair trainer |
| `train_preshoot_blocked_gate_repair.py` | `rlfootball/training/train_preshoot_blocked_gate_repair.py` | entry | 147 | 0/0/0 | Pre-shoot blocked gate repair trainer |
| `train_preshoot_blocked_stability_repair.py` | `rlfootball/training/train_preshoot_blocked_stability_repair.py` | entry | 135 | 0/0/0 | Pre-shoot blocked stability repair trainer |
| `train_preshoot_repair.py` | `rlfootball/training/train_preshoot_repair.py` | entry | 115 | 0/0/0 | Pre-shoot repair trainer |
| `train_production.py` | `rlfootball/training/train_production.py` | entry | 286 | 4/0/0 | Production fine-tune entrypoint |
| `train_stability_repair.py` | `rlfootball/training/train_stability_repair.py` | entry | 68 | 0/0/0 | Stability repair trainer |
| `train_stand_repair.py` | `rlfootball/training/train_stand_repair.py` | entry | 283 | 1/0/0 | Stand/skill repair trainer |
| `behavioral_evaluator.py` | `rlfootball/evaluation/behavioral_evaluator.py` | entry | 397 | 1/0/0 | Scenario-based behavioral evaluation of checkpoints |
| `behavioral_stability.py` | `rlfootball/evaluation/behavioral_stability.py` | mod | 84 | 1/0/1 | Helpers: seed/scenario parsing and failure aggregation |
| `behavioral_stability_probe.py` | `rlfootball/evaluation/behavioral_stability_probe.py` | entry | 222 | 0/0/0 | Multi-seed stability probe over behavioral scenarios |
| `compare_defender_checkpoints.py` | `rlfootball/evaluation/compare_defender_checkpoints.py` | entry | 72 | 0/0/0 | Compare two defender checkpoints |
| `model_evaluator.py` | `rlfootball/evaluation/model_evaluator.py` | entry | 507 | 2/0/0 | Evaluate a trained checkpoint |
| `run_rl_vs_bhuman_3v3.py` | `rlfootball/evaluation/run_rl_vs_bhuman_3v3.py` | entry | 281 | 0/0/0 | Run RL policy vs B-Human 3v3 in SimRobot |
| `teacher_chain_verify.py` | `rlfootball/evaluation/teacher_chain_verify.py` | entry | 205 | 0/0/0 | Seed-stable verification that the gate-wired teacher completes the full |
| `teacher_evaluator.py` | `rlfootball/evaluation/teacher_evaluator.py` | entry | 138 | 0/0/0 | Evaluate the heuristic teacher |
| `action_sweep.py` | `rlfootball/diagnostics/action_sweep.py` | entry | 304 | 0/0/0 | Sweep action parameters and log robot response to CSV |
| `diag_n6_slots.py` | `rlfootball/diagnostics/diag_n6_slots.py` | entry | 377 | 0/0/0 | Per-slot sanity diagnostic for N=6 visible parallel training |
| `gui_visible_dynamic_world_check.py` | `rlfootball/diagnostics/gui_visible_dynamic_world_check.py` | entry | 91 | 0/0/0 | GUI check: dynamic world in visible SimRobot |
| `gui_visible_invalid_slot_check.py` | `rlfootball/diagnostics/gui_visible_invalid_slot_check.py` | entry | 63 | 0/0/0 | GUI check: invalid slot handling |
| `gui_visible_reset_extended_check.py` | `rlfootball/diagnostics/gui_visible_reset_extended_check.py` | entry | 65 | 0/0/0 | GUI check: extended reset |
| `gui_visible_suite.py` | `rlfootball/diagnostics/gui_visible_suite.py` | entry | 157 | 0/0/0 | Runs the visible-GUI checks together |
| `heuristic_agent_experiment.py` | `rlfootball/diagnostics/heuristic_agent_experiment.py` | entry | 307 | 0/0/0 | Heuristic-agent episodes for baseline comparison |
| `live_simrobot_probe.py` | `rlfootball/diagnostics/live_simrobot_probe.py` | entry | 375 | 0/0/0 | Live probe of a running SimRobot session |
| `live_simrobot_probe_extra.py` | `rlfootball/diagnostics/live_simrobot_probe_extra.py` | entry | 28 | 0/0/0 | Thin wrapper over live_simrobot_probe with extra scene args |
| `obs_invariants.py` | `rlfootball/diagnostics/obs_invariants.py` | entry | 260 | 0/0/0 | Observation invariant checks against the environment |
| `probe_protocol_experiment.py` | `rlfootball/diagnostics/probe_protocol_experiment.py` | entry | 290 | 0/0/0 | Experiment on probe protocol variants |
| `validate_simrobot_matrix.py` | `rlfootball/diagnostics/validate_simrobot_matrix.py` | entry | 370 | 0/0/0 | Skill-by-skill SimRobot validation matrix |
| `visible_probe_controlled.py` | `rlfootball/diagnostics/visible_probe_controlled.py` | entry | 147 | 0/0/0 | Controlled visible probe with heuristic actions |
| `visualize_simrobot_skills.py` | `rlfootball/diagnostics/visualize_simrobot_skills.py` | entry | 155 | 0/0/0 | Visualize skills in SimRobot |
| `smoke_test_skill_requests.py` | `tests/smoke_test_skill_requests.py` | test | 63 | 0/0/0 | Smoke test for skill requests |
| `test_behavioral_stability.py` | `tests/test_behavioral_stability.py` | test | 66 | 0/0/0 | Tests for behavioral_stability helpers |
| `test_mappo_observation_space.py` | `tests/test_mappo_observation_space.py` | test | 176 | 0/0/0 | Tests for the 47-dim MAPPO observation space |
| `test_multiagent_teacher.py` | `tests/test_multiagent_teacher.py` | test | 232 | 0/0/0 | Tests for the multi-agent teacher |
| `test_obs_migration_contract.py` | `tests/test_obs_migration_contract.py` | test | 159 | 0/0/0 | Tests for obs migration and gate mask contract |
| `test_observation_contract.py` | `tests/test_observation_contract.py` | test | 91 | 0/0/0 | Tests for the observation contract |
| `test_wire_audit.py` | `tests/test_wire_audit.py` | test | 99 | 0/0/0 | Tests for wire_audit |
| `test_world_spec_contract.py` | `tests/test_world_spec_contract.py` | test | 78 | 0/0/0 | Tests for world-spec contract |

## Running entry points

Run from the repo root with the venv active. Old command -> new command:

| Old | New |
|---|---|
| `python train.py` | `python -m rlfootball.training.train` |
| `python train_multiagent_production.py --stage-id 7` | `python -m rlfootball.training.train_multiagent_production --stage-id 7` |
| `python teacher_multiagent_dataset.py` | `python -m rlfootball.datasets.teacher_multiagent_dataset` |
| `python behavioral_evaluator.py` | `python -m rlfootball.evaluation.behavioral_evaluator` |
| `python diag_n6_slots.py` | `python -m rlfootball.diagnostics.diag_n6_slots` |
| `pytest test_x.py` | `pytest tests/test_x.py` |

`-m` needs the repo root on `sys.path` (it is when run from the root). Optionally add
a `pyproject.toml` and `pip install -e .` so the package also resolves from other directories.
No root launcher scripts are kept.

## Dependency analysis

Computed from AST imports (module level and function level) between root modules,
collapsed to subpackages (tests excluded as sources).

Initial proposal (env incl. config/reward/gate/backend/environment) had cycles:

| Cycle | Cause |
|---|---|
| `env <-> curriculum` | `curriculum` imports `config`, `reward`, `pybh_backend`; `environment` imports `curriculum`, `production_curriculum` |
| `env <-> teachers` | `teachers` import `action`, `config`, `gate`, `reward`, `pybh_backend`; `environment` (lazy) and `team_parallel_environment` import `multiagent_teacher` |
| `teachers <-> curriculum` | `teacher_policy` imports `curriculum.get_stage_spec` |

Resolution: split `env` into `core` (leaf contracts, backend) and `env` (the environments).
Resulting layering (edges point to dependencies):

```text
core  <-  curriculum  <-  teachers  <-  env  <-  datasets  <-  training  <-  evaluation
diagnostics -> core, curriculum, teachers, env     (nothing imports diagnostics)
```

Remaining back-edges in the layered graph (see "Deviations" for what was done):

| Edge | Where | Resolution |
|---|---|---|
| `datasets -> training` | `teacher_multiagent_dataset` imports `gate_skill_mask_from_obs` from `train` (3 lazy imports, lines 304, 908, 1281) | Move the function to `rlfootball/core/gate.py`; re-export from `train` so `tests` and `tools` keep working. This is the only code move needed beyond import rewriting. |
| `training -> datasets` (forward, fine) | `train_defender_bc`, `run_goalkeeper_production` | none needed |

Not removable by mapping alone: the `datasets -> training` edge. With the function moved
(or a lazy import tolerated), the graph is acyclic. No two subpackages remain mutually dependent.

Also note `evaluation -> training` (`train` helpers: `build_env`, `build_mappo`,
`_aggregate_eval_metrics`) and `train -> env` are one-directional; keep it so.

## Name collisions

Checked each of the 80 module names against `sys.stdlib_module_names` and the
`.venv` site-packages: **no collisions**. Generic names (`config`, `action`, `reward`,
`gate`, `wrapper`, `observation`, `environment`) become `rlfootball.<sub>.<name>`, so the
only risk is stale bare `import config` in code that relied on the repo root being on
`sys.path`: 27 such import lines for these names in `tools/` (21 files import root modules in total). After the move they raise
`ImportError` (they do not shadow anything), so they must be rewritten.

## Non-import couplings for the executor

| Coupling | Count | Fix |
|---|---|---|
| `Path(__file__)`-relative root lookups, root modules | 28 files / 30 lines | Replace with `ROOT_DIR` from `rlfootball.core.config`; `config.ROOT_DIR` itself becomes `Path(__file__).resolve().parents[2]` |
| `Path(__file__)...parents[2]` in `tools/` | 19 files | Unchanged location; keep, but drop their `sys.path` bootstrap |
| `sys.path` hacks, root modules | 11 files / 21 lines | Delete (package import replaces them); `pybh_backend`, `run_rl_vs_bhuman_3v3`, `diag_n6_slots` add the B-Human `Make/Python` dir: keep that part |
| `sys.path` hacks, `tools/` | 15 files / 29 lines | Keep a repo-root insert only if the package is not installed |
| String module names (`importlib.import_module`) | 2 (`train.py`: pandas, matplotlib; unrelated) | None |
| Subprocess running a script by path | 2 | `probe_protocol_experiment` re-runs itself via `__file__` (switch to `-m`); `run_goalkeeper_production` runs `tools/verify_goalkeeper_action_space_truth.py` (stale since Level A moved it to `tools/eval/`; fix path) |
| Root-relative string dirs (`"runs/"`, `"models/"`, `"docs/"`, `tools`) | 39 root files / 83 hits; 11 `tools` files / 23 | Resolve through `ROOT_DIR`, not cwd |
| `.sh` scripts naming a root module (`$PY X.py`) | 28 files / 50 lines | Rewrite to `$PY -m rlfootball.<sub>.<X>`; also `scripts/setup_validation_env.sh` (`diag_n6_slots.py`) |
| `.sh` with hard-coded `/home/<user>/...` PYTHONPATH | `tools/run/run_stability_chunked.sh` | Drop; run from repo root |
| `.vscode` | 1 file, 0 launch configs | `settings.json` has `extraPaths: ["."]`, still valid; no change required |
| Docs/README citing root files (`X.py`) | 41 md files / 628 mentions (README 39) | Update commands in README and live docs; `docs/notes/` is historical, rewrite only with a script mapping |
| `tools/**/*.py` importing root modules | 21 files | Rewrite imports to `rlfootball.*` |
| Tests | 8 files | Move to `tests/`, rewrite imports; `test_obs_migration_contract` imports from `train` |
| `environment.yml` | 0 refs | none |

## Migration notes

- Rule: `<name>.py` -> `rlfootball/<sub>/<name>.py`; tests -> `tests/<name>.py`. Names never change.
- Imports: `import X` / `from X import ...` -> `from rlfootball.<sub>.X import ...` (AST rewrite; function-level imports included).
- Commands: `python X.py` -> `python -m rlfootball.<sub>.X`.
- Use `git mv` so history follows; do the move in one commit per subpackage.
- Policy artifacts (`ppo_*.onnx/json`, C++ side) are loaded by filename from `models/`, not by module name: unaffected.
- `ROOT_DIR` depth changes from `.` to `rlfootball/core/`; every `ROOT = Path(__file__).resolve().parent` must be re-pointed.

## Deviations

Differences between the design above and what was implemented:

- `gate_skill_mask_from_obs` (with its eight `_*_SKILL_IDX` constants and two `_*_GATE_OBS_REQUIRED` constants) now lives in `rlfootball/core/gate.py`; `rlfootball/training/train.py` re-exports it. The three lazy imports in `teacher_multiagent_dataset` and the other importers still import it from `train`, so the `datasets -> training` back-edge remains as a function-level import.
- `tools/**` keep their `sys.path` insert of the repo root (they run as plain scripts, and the package is not installed). Root modules lose only the `sys.path` inserts that added the old flat directory; the B-Human `Make/Python` inserts in `pybh_backend`, `run_rl_vs_bhuman_3v3` and `diag_n6_slots` stay.
- Repo-root lookups use `Path(__file__).resolve().parents[2]` directly (not `ROOT_DIR` from `config`), to keep each file self-contained. `diag_n6_slots` now climbs three levels to reach the sibling `SabanaHerons2026`.
- Stale imports left over from the `tools/` regrouping were corrected (`tools.export_ppo_policy` -> `tools.export.export_ppo_policy`, and similar in `run_goalkeeper_production`, `train_stand_repair`, `train_param_repair`, `train_goalkeeper_ppo_closedloop`), plus the `tools/verify_goalkeeper_action_space_truth.py` path in `run_goalkeeper_production`.
- `probe_protocol_experiment` re-runs itself with `python -m rlfootball.diagnostics.probe_protocol_experiment` instead of the file path.
- Inline Python in four `.sh` heredocs now imports `rlfootball.core.*`.
- Smoke test: `python -m tests.smoke_test_skill_requests` (run from the repo root).
- `run_goalkeeper_production` and `train_goalkeeper` import `evaluate_step_gk` from `reward`, which does not define it. This predates the refactor and was left unchanged.
