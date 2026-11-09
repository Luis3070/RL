# Project structure and integration guide

## 1. Purpose and audience

This guide is for teammates and researchers who worked on the old flat repository layout (about 80 `*.py` files in the repository root) and now need to find things in the `rlfootball` package, port in-flight work, and add new work in the right place.
It states the rules; the full old-to-new name table and the design rationale live in [REPO_LAYOUT.md](REPO_LAYOUT.md).
It is also the reference behind the root [CLAUDE.md](../CLAUDE.md) used by Claude Code sessions.

## 2. The new structure

```text
.
├── README.md              entry point: architecture, setup, running, data policy
├── CLAUDE.md              project instructions for Claude Code sessions
├── .env.example           template for the optional, gitignored .env
├── environment.yml        conda environment (outdated, see section 6)
├── pytest.ini             pythonpath = ., testpaths = tests
├── rlfootball/            the importable Python package (absolute imports only)
│   ├── core/              leaf layer: action, config, observation, reward, gate,
│   │                      goalkeeper_action, pybh_backend, visible_parallel_runtime, wire_audit
│   ├── curriculum/        stage specs and production scenarios
│   ├── teachers/          rule-based teachers and the defender policy
│   ├── env/               environments and the skrl wrapper
│   ├── datasets/          teacher dataset collection, mergers, audits
│   ├── training/          BC / PPO / MAPPO trainers, production and repair trainers
│   ├── evaluation/        behavioral, model and teacher evaluators, stability, RL vs B-Human
│   └── diagnostics/       probes, visible-SimRobot checks, sweeps, visualisation
├── tools/                 launchers and utilities, grouped by purpose
│   ├── run/               training recipes and launchers (run_*.sh, train_merge_v5.sh)
│   ├── export/            ONNX export (export_ppo_policy.py, ...)
│   ├── eval/              evaluation, validation and parity probes, held-out and match statistics
│   ├── dataset/           dataset generation and audits
│   ├── paper/             paper figures and match-metric tables (writes to results/paper/)
│   └── misc/              small helpers
├── scripts/               env.sh (path setup), setup_validation_env.sh, run_with_validation_env.sh
├── models/                tracked policy deliverables: models/<role>/ppo_*.onnx + ppo_*.json
├── tests/                 test_*.py contract tests and smoke_test_skill_requests.py
├── docs/                  numbered documents (NN-title.md), index in docs/README.md
│   └── notes/             historical working notes
├── results/               small tracked result files
│   └── paper/             CSV tables for the revised paper (held-out ablation, match metrics)
└── runs/                  datasets, logs, checkpoints (gitignored, never committed)
```

Layer responsibilities:

| Subpackage | Responsibility | Allowed to import |
|---|---|---|
| `rlfootball/core` | Contracts every layer shares: 47-dim observation (`observation.py`), actions (`action.py`), config, reward, skill gate (`gate.py`), `pybh` backend, wire audit | nothing from the package |
| `rlfootball/curriculum` | Stage specs and production scenarios | `core` |
| `rlfootball/teachers` | Rule-based teachers (single-agent, multi-agent, goalkeeper, defender) | `core`, `curriculum` |
| `rlfootball/env` | Environments and the skrl wrapper | `core`, `curriculum`, `teachers` |
| `rlfootball/datasets` | Teacher dataset collection, merging, audits | `core`, `curriculum`, `teachers`, `env` |
| `rlfootball/training` | BC, PPO, MAPPO, production fine-tune, repair trainers | `core`, `curriculum`, `teachers`, `env`, `datasets` |
| `rlfootball/evaluation` | Evaluators, stability checks, RL vs B-Human 3v3 | everything above except `diagnostics` |
| `rlfootball/diagnostics` | Probes and GUI checks; nothing imports it | `core`, `curriculum`, `teachers`, `env` |

Dependency direction: `core -> curriculum -> teachers -> env -> datasets -> training -> evaluation`, with `diagnostics` hanging off `core/curriculum/teachers/env`.
A module may import only from layers to its left; never from a higher layer.
`tests/` may import anything.

## 3. Transition guide: old layout to new layout

### Translating commands and imports

| Old | New |
|---|---|
| `python train.py` | `python -m rlfootball.training.train` |
| `python behavioral_evaluator.py` | `python -m rlfootball.evaluation.behavioral_evaluator` |
| `from config import ENV` | `from rlfootball.core.config import ENV` |
| `import observation` / `from observation import ObsIndex` | `from rlfootball.core.observation import ObsIndex` |
| `tools/export_ppo_policy.py` | `tools/export/export_ppo_policy.py` |
| `pytest test_x.py` | `pytest tests/test_x.py` |

Rules (from [REPO_LAYOUT.md](REPO_LAYOUT.md), "Migration notes"):

- `<name>.py` became `rlfootball/<sub>/<name>.py`; names did not change. Tests moved to `tests/<name>.py`.
- Imports are absolute: `from rlfootball.<sub>.<name> import ...`.
- Run everything from the repository root. `python -m` needs the root on `sys.path`, and `pytest.ini` sets `pythonpath = .`.
- Old documents and `docs/notes/` still cite the flat names (`train.py`, `config.py`). The complete 80-row table is in [REPO_LAYOUT.md](REPO_LAYOUT.md), section "Module map"; the entry-point table is in "Running entry points".

Most common old names:

| Old | New |
|---|---|
| `action.py` | `rlfootball/core/action.py` |
| `config.py` | `rlfootball/core/config.py` |
| `observation.py` | `rlfootball/core/observation.py` |
| `gate.py` | `rlfootball/core/gate.py` |
| `pybh_backend.py` | `rlfootball/core/pybh_backend.py` |
| `reward.py` | `rlfootball/core/reward.py` |
| `curriculum.py` | `rlfootball/curriculum/curriculum.py` |
| `production_curriculum.py` | `rlfootball/curriculum/production_curriculum.py` |
| `multiagent_teacher.py` | `rlfootball/teachers/multiagent_teacher.py` |
| `environment.py` | `rlfootball/env/environment.py` |
| `teacher_multiagent_dataset.py` | `rlfootball/datasets/teacher_multiagent_dataset.py` |
| `train.py` | `rlfootball/training/train.py` |
| `train_multiagent_production.py` | `rlfootball/training/train_multiagent_production.py` |
| `behavioral_evaluator.py` | `rlfootball/evaluation/behavioral_evaluator.py` |
| `diag_n6_slots.py` | `rlfootball/diagnostics/diag_n6_slots.py` |
| `test_wire_audit.py` | `tests/test_wire_audit.py` |

### Porting work written against the old layout

1. Find where your branch diverged: `git merge-base <your-branch> <new-base>`. List what you changed: `git diff --stat <merge-base>..<your-branch>`.
2. Update from the new base with a rebase (`git rebase <new-base>`) or a merge (`git merge <new-base>`). Files were moved with `git mv`, so Git detects renames and applies your edits to the moved files; `git log --follow rlfootball/core/config.py` shows the pre-move history.
3. Resolve rename conflicts on the new path. If Git reports modify/delete on an old flat file, apply your change to the file's new location (table above) and `git add` it.
4. Rewrite leftover flat imports in your own code. Find them with `rg -n '^\s*(from|import) (config|action|observation|gate|reward|environment|train)\b' <your files>` and rewrite them to `rlfootball.<sub>.<name>` (use `sd` for mechanical replacements, then review the diff). A stale bare `import config` raises `ImportError`; it does not silently pick another module.
5. Rewrite shell launchers: `$PY X.py` becomes `$PY -m rlfootball.<sub>.X`, and hardcoded `/home/<user>/...` paths become `RL_REPO_DIR` / `SABANAHERONS_DIR` (see section 4).
6. Re-run the checks in section 5 and every `python -m` entry point your change touches.

Pickled files in `runs/`: training checkpoints are written by `rlfootball/training/train.py` as a plain dict (`timestep`, `stage_id`, `policy_state_dict`, `value_state_dict`, `eval_summary`), and the datasets are `.npz` arrays, so they do not normally reference module names. This was checked by reading the `torch.save` call, not by loading old files. The loader uses `torch.load(..., weights_only=False)`, which unpickles arbitrary objects: if a checkpoint from an older branch fails with `ModuleNotFoundError` naming a flat module (for example `config`), it contains a pickled class from the old layout. Re-export it from the old branch, or ask its author; there is no tested workaround in this repository.

## 4. Handling new integrations

### Where does new code go?

| You are adding | Put it in | Notes |
|---|---|---|
| A shared constant, observation column, action or gate rule | `rlfootball/core/` | Changing `OBS_SIZE`, `SKILL_ORDER` or `gate_skill_mask_from_obs` invalidates loaded checkpoints; document it |
| A reward term | `rlfootball/core/reward.py` | |
| A curriculum stage or production scenario | `rlfootball/curriculum/` | |
| A teacher / heuristic policy | `rlfootball/teachers/` | |
| A new environment or wrapper | `rlfootball/env/` | |
| Dataset collection, merging or audit | `rlfootball/datasets/` | |
| A training entry point | `rlfootball/training/` | Add `if __name__ == "__main__":`; run with `python -m rlfootball.training.<name>` |
| An evaluator or stability check | `rlfootball/evaluation/` | Same `python -m` rule |
| A probe, GUI check or sweep | `rlfootball/diagnostics/` | |
| A launcher or recipe | `tools/run/` (or `tools/eval/`, `tools/dataset/`, `tools/export/`, `tools/misc/`) | See below |
| A paper figure script or a match-metric summary | `tools/paper/`; small CSV outputs in `results/paper/` | Figures go to `results/paper/figures/` (not tracked) |
| A new exported policy | `models/<role>/` | See below |
| A test | `tests/` | See below |
| A document | `docs/NN-title.md` | See below |
| A dependency | `environment.yml` | Also note it in the README if it is not obvious |

Do not add new modules to the repository root or to `tools/` for logic that other code imports: library code belongs in `rlfootball/`, and `tools/` scripts import it.

### Launcher scripts

`tools/eval/` and `tools/paper/` scripts that import a sibling script do so by bare module name after inserting their own directory into `sys.path` (`paper_match_stats` and `select_rule_setting` import `analyze_full_probe`; `paper_figures_final` imports `paper_figures_icra`). Keep that convention inside the folder and run them as plain scripts, not with `-m`.

- Start with `source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/env.sh"` (adjust the number of `..` to the script depth; `tools/run/run_v4_2.sh` is the reference) and use `RL_REPO_DIR` and `SABANAHERONS_DIR` afterwards.
- Never hardcode `/home/<user>/...` paths. `scripts/env.sh` reads the optional `.env` (copy `.env.example`), falling back to the repository root and the sibling `../SabanaHerons2026`.
- `cd "$RL_REPO_DIR"`, then call `python -m rlfootball.<sub>.<module>`. `scripts/env.sh` is bash-only; do not source it from fish or zsh.
- Training entry points `train.py` and `train_multiagent.py` take no CLI flags; configure them with `RL_*` environment variables (see the README, "Running").

### New exported policy

Export with `tools/export/export_ppo_policy.py` (or `export_ppo_policy_team.py`, `export_goalkeeper_policy.py`), and put the `ppo_*.onnx` and its `ppo_*.json` manifest under `models/<role>/` (`striker`, `defender`, `goalkeeper`, `team`). The `checkpoint` field in the manifest must be a repo-relative `runs/...` path, not an absolute one. Add a row to the README policy table. Weights are tracked; the PyTorch checkpoints in `runs/` are not (the two defender checkpoints in `models/defender/checkpoints/` are the exception).

### Tests

- Put tests in `tests/test_*.py`; `pytest.ini` collects `tests/` with the root on `sys.path`.
- Prefer tests that do not need `pybh` (pure contract tests such as `tests/test_wire_audit.py`, `tests/test_behavioral_stability.py`). Tests that import the backend or environments need a working `pybh` build.
- `tests/smoke_test_skill_requests.py` is a simulator smoke test, run as `python -m tests.smoke_test_skill_requests`.

### Documents

Add `docs/NN-title.md` with the next free number (the highest is 42; 08 is absent and 25 and 35 are duplicated, so do not reuse them), write new documents in English, and add a row to the table in [README.md](README.md) (docs index). Working notes go in `docs/notes/`. Refer to modules by their new `rlfootball.*` paths.

### Dependency direction

- Never import from a higher layer (section 2). Move the shared piece down into `core` instead.
- The one known exception is `datasets -> training`: `teacher_multiagent_dataset` imports `gate_skill_mask_from_obs` from `rlfootball.training.train` (function-level imports). The function lives in `rlfootball/core/gate.py` and `train` re-exports it; see [REPO_LAYOUT.md](REPO_LAYOUT.md), "Deviations". New code should import it from `rlfootball.core.gate`.
- `evaluation -> training` and `training -> env` are one-directional; keep them so.

### Branch and PR workflow

1. Branch from `main`; one feature or fix per branch.
2. Commit in small, reviewable units with Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`), keeping tests and docs with the change.
3. Never commit `runs/`, `.env`, datasets (`*.npz`), logs (`*.log`), or credentials. They are gitignored; do not force-add them.
4. Open a PR only after the checklist in section 5 passes. In the description state what changed, which old names it touches, and what was run.

### Changes that touch the C++ repository `SabanaHerons2026`

- Multi-agent training in this repository requires branch `rl-simrobot3d` of `SabanaHerons2026`; deploying ONNX policies uses `hsl2026-ppo-team-v4`. Check the right branch is checked out before running.
- After changing `RLSharedState` (or `RLSkillProvider.cpp`, `Module.cpp`), rebuild `pybh` (`Make/Python`, `python3 setup.py build_ext --inplace`) and `SimulatedNao`; otherwise Python and SimRobot read different shared-memory layouts.
- Do not edit the sibling repository from a PR of this one. C++ changes go through their own branch and PR in `SabanaHerons2026`; mention the required `SabanaHerons2026` revision in this PR's description.

## 5. Verification checklist before opening a PR

- [ ] `python -m pytest -q tests/test_wire_audit.py tests/test_behavioral_stability.py` passes (pure tests, no simulator).
- [ ] `pytest` (full suite) passes if you have a working `pybh` build; otherwise state that you could not run it.
- [ ] `bash -n <script>` for every shell script you added or changed.
- [ ] Every `python -m rlfootball.<sub>.<module>` you cite or changed resolves to an existing file under `rlfootball/`.
- [ ] No bare flat imports (`rg -n '^\s*(from|import) (config|action|observation|gate|reward|environment)\b' rlfootball tools tests` prints nothing).
- [ ] No hardcoded `/home/...` paths in code, scripts or docs (`rg -n '/home/' <changed files>`).
- [ ] `git status --short` shows no `runs/`, `.env`, `*.npz` or `*.log`.
- [ ] New documents are indexed in [README.md](README.md) (docs index).

## 6. Known gaps and TODO

- No automated CI: checks are run by hand.
- `environment.yml` is outdated (Python 3.10, missing `onnx`, `onnxruntime`, `pytest`); see the README, "Environment setup".
- The full `pytest` suite and most `tools/` launchers need a built `pybh` and a SimRobot binary from `SabanaHerons2026`; they could not be run in a clean documentation environment.
- The package is not installed (`pyproject.toml` is not present), so entry points only resolve from the repository root; `tools/` scripts insert the repository root into `sys.path` themselves.
- `train_goalkeeper` imports `evaluate_step_gk` from `rlfootball/core/reward.py`, which does not define it (pre-existing, see REPO_LAYOUT.md "Deviations").
- Paper results not published here, missing figures and `references.bib`: see the TODO list in [PAPER_TRACEABILITY.md](PAPER_TRACEABILITY.md).
- The C++ repository URL is to be confirmed by the authors.
