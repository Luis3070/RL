# RL for role-conditioned football in humanoid robots (Python side)

Python reinforcement-learning code for the paper
**"A Teacher-Guided RL Pipeline for Role-Conditioned Football Cognition in Humanoid Robots"**
(NAO V6 in SimRobot, SabanaHerons / B-Human stack): a 47-dimensional observation,
a behavioral-cloning (BC) warm start, PPO with a KL anchor, and a role-conditioned
merged policy (striker, defender, goalkeeper).

Many documents under `docs/` are written in Spanish and are historical work logs.
See [docs/README.md](docs/README.md) for the index.

## Two-repository architecture

| Repository | Language | Role |
|---|---|---|
| This repository | Python | Environments, teachers, dataset generation, BC/PPO training, evaluation, ONNX export |
| `SabanaHerons2026` (B-Human fork; URL to be confirmed by the authors) | C++ | SimRobot, the `pybh` Python binding, `RLSharedState` bridge, native ONNX inference |

Python talks to B-Human in-process through `pybh` (see `rlfootball/core/pybh_backend.py`).
`rlfootball/core/config.py` expects `SabanaHerons2026` as a sibling directory of this repository.
The old `bridge/` subprocess stack was removed in commit `a1da9e9` ("pybh-clean: drop legacy subprocess stack").

Required `SabanaHerons2026` branches:

| Purpose | Branch |
|---|---|
| Multi-agent training in this repo | `rl-simrobot3d` (bridge version 30, multi-agent fields; as stated in `docs/notes/HANDOFF.md`) |
| Deploying the ONNX policies | `hsl2026-ppo-team-v4` (striker v4.2: [docs/37](docs/37-striker-v2b-to-v4.2-and-cpp-transplant.md); merged brain v5: [docs/38](docs/38-merged-striker-defender-and-naov6-transplant.md)) |

## Repository map

| Path | Purpose |
|---|---|
| `rlfootball/core/` | Configuration, observation (47-dim), action space, reward, skill gate, `pybh` backend, wire audit |
| `rlfootball/curriculum/` | Curriculum stage specs and production scenarios |
| `rlfootball/teachers/` | Rule-based teachers (single-agent, multi-agent, goalkeeper, defender) |
| `rlfootball/env/` | Environments and the skrl wrapper |
| `rlfootball/datasets/` | Teacher dataset collection, merging and audits |
| `rlfootball/training/` | PPO/MAPPO entry points (`train`, `train_multiagent`), production wrappers, BC and repair trainers |
| `rlfootball/evaluation/` | Behavioral, model and teacher evaluators, stability checks, RL vs B-Human 3v3 |
| `rlfootball/diagnostics/` | Visible-SimRobot checks, live probes, sweeps, visualisation |
| `tests/` | Contract tests and the bridge smoke test |
| `tools/` | Grouped by purpose: `run/` (launchers, `run_*.sh`), `export/` (ONNX export), `eval/` (evaluation, validation, probes), `dataset/` (dataset generation and audits), `misc/` |
| `scripts/` | Validation environment setup (`setup_validation_env.sh`, `run_with_validation_env.sh`) |
| `models/` | Tracked policy deliverables and manifests (see below) |
| `models/defender/checkpoints/` | Tracked defender PyTorch checkpoints |
| `docs/` | Documentation (index: [docs/README.md](docs/README.md)) |
| `docs/notes/` | Historical working notes (`HANDOFF.md`, `CODEX_OFFBALL_PERCEPTION_HANDOFF.md`, `cambios_para_produccion.md`, `bhuman-logic/`) (mostly Spanish) |
| `environment.yml` | Conda environment (outdated, see below) |

## Environment setup

`environment.yml` pins `python=3.10`, `pytorch-cuda=11.8` and installs `skrl`, `pettingzoo`, `gymnasium`.

Mismatches found:

| Item | Finding |
|---|---|
| Python version | `environment.yml` says 3.10; `scripts/setup_validation_env.sh` uses a `python3` venv at `.venv`, and the previous README referenced `cpython-312` `pybh` modules. Use the Python version your `pybh` build targets. |
| Extra packages | The code also needs `onnx` and `onnxruntime`, and `pytest` for tests; they are not listed in `environment.yml`. |
| `pettingzoo[all]` | Can fail to build (box2d needs swig); plain `pettingzoo` is enough. |

### Local paths (`.env`)

Launchers under `tools/` and `scripts/` and `rlfootball/core/config.py` read two variables:

| Variable | Meaning | Default when unset |
|---|---|---|
| `RL_REPO_DIR` | Absolute path of this repository | repository root, resolved from the script location |
| `SABANAHERONS_DIR` | Absolute path of the C++ B-Human fork `SabanaHerons2026` | sibling directory `../SabanaHerons2026` |

Both are optional. To override them, copy the template and edit it:

```bash
cp .env.example .env
```

`.env` is gitignored; variables already set in your shell take precedence over it.

Minimal setup, following `scripts/setup_validation_env.sh`:

```bash
scripts/setup_validation_env.sh                      # creates .venv, installs torch, skrl, pettingzoo, ...
scripts/run_with_validation_env.sh python3 -m rlfootball.diagnostics.diag_n6_slots   # sets PYTHONPATH to ../SabanaHerons2026/Make/Python
```

`pybh` must be built from `SabanaHerons2026` (`Make/Python`, `python3 setup.py build_ext --inplace`),
and `SimulatedNao` must be rebuilt after any change to `RLSharedState`; otherwise Python and SimRobot read different shared-memory layouts.
The `RLSharedState` segment name used by this repo is `/pybh_rl_shared_state_v3`.

## Running

Run every command from the repository root; entry points are modules (`python -m rlfootball.<subpackage>.<module>`), and the `tools/` and `scripts/` launchers also assume the repository root.
Older documents and notes use the pre-refactor flat names (`train.py`, `config.py`, ...); the old-to-new table is in [docs/REPO_LAYOUT.md](docs/REPO_LAYOUT.md).

`rlfootball/training/train.py` and `rlfootball/training/train_multiagent.py` have no command-line flags. They read a `ScriptRunConfig` and override it with `RL_*` environment variables (`_run_cfg_from_env` in `rlfootball/training/train.py`).

```bash
RL_RUN_MODE=smoke RL_SMOKE_STEPS=4 python -m rlfootball.training.train
RL_RUN_MODE=train RL_TRAIN_STEPS=800000 python -m rlfootball.training.train_multiagent
```

Variables seen in `rlfootball/training/train.py`: `RL_RUN_MODE`, `RL_TRAIN_STEPS`, `RL_SMOKE_STEPS`, `RL_STAGE_ID`, `RL_MAX_STAGE_ID`, `RL_LEARNER_DEVICE`, `RL_BC_WARM_START`, `RL_BC_CHECKPOINT`, `RL_BC_REPLAY`, `RL_BC_REPLAY_DATASET`, and others.
The default backend in `ScriptRunConfig` is `simrobot_visible`.

Other verified entry points (argparse):

| Command | Flags seen in the source |
|---|---|
| `python -m tests.smoke_test_skill_requests` | `--backend {simrobot,simrobot_visible}`, `--simrobot-profile {2d,3d}`, `--launch-simrobot`, `--simrobot-bin` |
| `python -m rlfootball.training.train_multiagent_production` | `--stage-id`; the `tools/run/*.sh` scripts also pass `--checkpoint` and `--dataset` |
| `python -m rlfootball.training.train_production` | `--dataset`, `--output-dataset`, `--checkpoint`, `--prepare-only` |
| `python -m rlfootball.training.train_defender_bc` | `--dataset`, `--steps`, `--batch-size`, `--lr`, `--init-checkpoint`, `--reference-checkpoint`, `--distill-weight`, `--output-dir`, ... |
| `pytest` (reads `pytest.ini`, runs `tests/`) | Contract tests; some may need `pybh` |

The `tools/run/*.sh` scripts record the exact recipes used for each policy version (for example `tools/run/run_v4_2.sh`, `tools/run/train_merge_v5.sh`).
These commands could not be executed during documentation work (the authors' `pybh` build was not usable in the documentation environment); flags are quoted from the source, not from `--help` output.

## Data and runs policy

- `runs/` is gitignored: datasets (`*.npz`), logs, telemetry and training checkpoints are not versioned.
- Tracked deliverables: `ppo_*.onnx` weights with their `ppo_*.json` manifests under `models/<role>/`, and `models/defender/checkpoints/checkpoint_stage8_step2500.pt` and `checkpoint_stage8_step10000.pt`.

| Policy | Files |
|---|---|
| Striker | `models/striker/ppo_striker_hsl2026.{onnx,json}` (+ `.onnx.data`) |
| Defender | `models/defender/ppo_defender_hsl2026.{onnx,json}` |
| Team (striker v4.1, v4.2) | `models/team/ppo_team_hsl2026.*`, `ppo_team_hsl2026_v4_1.*`, `ppo_team_hsl2026_v4_2.*` |
| Merged brain v5 | `models/team/ppo_team_hsl2026_v5_merged.{onnx,json}` |
| Goalkeeper | `models/goalkeeper/ppo_goalkeeper_hsl2026*.{onnx,json}` (base, `ppo`, `ppo_validated`, and the `gk_*` variants) |

## Project documents

| Document | Content |
|---|---|
| [docs/README.md](docs/README.md) | Index of all documentation |
| [docs/BRANCHES.md](docs/BRANCHES.md) | Branch registry and consolidation record |
| [docs/PAPER_TRACEABILITY.md](docs/PAPER_TRACEABILITY.md) | Paper claims mapped to repository evidence |

## Known gaps

- The corpus sizes and several result figures in the paper come from results produced outside this repository (see [docs/PAPER_TRACEABILITY.md](docs/PAPER_TRACEABILITY.md)). TODO: publish those results here.
- Paper figures and `references.bib` are not in this repository.
- `environment.yml` is out of date (Python version, missing `onnx`/`onnxruntime`/`pytest`).
- The C++ repository URL is to be confirmed by the authors.
- Most documentation is in Spanish and reflects work in progress.
- No automated CI; tests are run by hand and part of them require a working `pybh` build.
