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

Python talks to B-Human in-process through `pybh` (see `pybh_backend.py`).
`config.py` expects `SabanaHerons2026` as a sibling directory of this repository.
The old `bridge/` subprocess stack was removed in commit `a1da9e9` ("pybh-clean: drop legacy subprocess stack").

Required `SabanaHerons2026` branches:

| Purpose | Branch |
|---|---|
| Multi-agent training in this repo | `rl-simrobot3d` (bridge version 30, multi-agent fields; as stated in `HANDOFF.md`) |
| Deploying the ONNX policies | `hsl2026-ppo-team-v4` (striker v4.2: [docs/37](docs/37-striker-v2b-to-v4.2-and-cpp-transplant.md); merged brain v5: [docs/38](docs/38-merged-striker-defender-and-naov6-transplant.md)) |

## Repository map

| Path | Purpose |
|---|---|
| `train.py`, `train_multiagent.py` | Main PPO entry points (single-agent, multi-agent) |
| `train_production.py`, `train_multiagent_production.py` | Production wrappers with BC warm start and curriculum stages |
| `train_bc.py`, `train_defender_bc.py`, `train_goalkeeper*.py`, `train_*_repair.py` | BC and fine-tuning/repair trainers |
| `environment.py`, `parallel_environment.py`, `team_parallel_environment.py`, `wrapper.py` | Environments and wrappers |
| `pybh_backend.py`, `config.py` | `pybh` backend (SimRobot embedded/visible) and central configuration |
| `observation.py`, `action.py`, `goalkeeper_action.py`, `reward.py`, `gate.py` | Observation (47-dim), action space, reward, skill gating |
| `teacher_*.py`, `multiagent_teacher.py`, `goalkeeper_teacher.py`, `generate_defender_dataset.py`, `merge_*.py` | Rule-based teachers and dataset generation/merging |
| `behavioral_evaluator.py`, `model_evaluator.py`, `teacher_evaluator.py`, `behavioral_stability*.py` | Evaluation and stability checks |
| `test_*.py`, `smoke_test_skill_requests.py` | Contract tests and the bridge smoke test |
| `gui_visible_*.py`, `live_simrobot_probe*.py`, `visualize_simrobot_skills.py`, `run_rl_vs_bhuman_3v3.py` | Visible-SimRobot checks and RL vs B-Human runs |
| `tools/` | Run scripts (`run_*.sh`), ONNX export (`export_*.py`), offline evaluation, dataset audits |
| `scripts/` | Validation environment setup (`setup_validation_env.sh`, `run_with_validation_env.sh`) |
| `ppo_*.onnx`, `ppo_*.json` | Tracked policy deliverables and manifests (see below) |
| `pt_defender_47/` | Tracked defender PyTorch checkpoints |
| `docs/`, `.docs/` | Documentation (index: [docs/README.md](docs/README.md)) |
| `HANDOFF.md`, `CODEX_OFFBALL_PERCEPTION_HANDOFF.md`, `cambios_para_produccion.md` | Historical working notes (Spanish) |
| `environment.yml` | Conda environment (outdated, see below) |

## Environment setup

`environment.yml` pins `python=3.10`, `pytorch-cuda=11.8` and installs `skrl`, `pettingzoo`, `gymnasium`.

Mismatches found:

| Item | Finding |
|---|---|
| Python version | `environment.yml` says 3.10; `scripts/setup_validation_env.sh` uses a `python3` venv at `.venv`, and the previous README referenced `cpython-312` `pybh` modules. Use the Python version your `pybh` build targets. |
| Extra packages | The code also needs `onnx` and `onnxruntime`, and `pytest` for tests; they are not listed in `environment.yml`. |
| `pettingzoo[all]` | Can fail to build (box2d needs swig); plain `pettingzoo` is enough. |

Minimal setup, following `scripts/setup_validation_env.sh`:

```bash
scripts/setup_validation_env.sh                      # creates .venv, installs torch, skrl, pettingzoo, ...
scripts/run_with_validation_env.sh python3 diag_n6_slots.py   # sets PYTHONPATH to ../SabanaHerons2026/Make/Python
```

`pybh` must be built from `SabanaHerons2026` (`Make/Python`, `python3 setup.py build_ext --inplace`),
and `SimulatedNao` must be rebuilt after any change to `RLSharedState`; otherwise Python and SimRobot read different shared-memory layouts.
The `RLSharedState` segment name used by this repo is `/pybh_rl_shared_state_v3`.

## Running

`train.py` and `train_multiagent.py` have no command-line flags. They read a `ScriptRunConfig` and override it with `RL_*` environment variables (`_run_cfg_from_env` in `train.py`).

```bash
RL_RUN_MODE=smoke RL_SMOKE_STEPS=4 python train.py
RL_RUN_MODE=train RL_TRAIN_STEPS=800000 python train_multiagent.py
```

Variables seen in `train.py`: `RL_RUN_MODE`, `RL_TRAIN_STEPS`, `RL_SMOKE_STEPS`, `RL_STAGE_ID`, `RL_MAX_STAGE_ID`, `RL_LEARNER_DEVICE`, `RL_BC_WARM_START`, `RL_BC_CHECKPOINT`, `RL_BC_REPLAY`, `RL_BC_REPLAY_DATASET`, and others.
The default backend in `ScriptRunConfig` is `simrobot_visible`.

Other verified entry points (argparse):

| Command | Flags seen in the source |
|---|---|
| `python smoke_test_skill_requests.py` | `--backend {simrobot,simrobot_visible}`, `--simrobot-profile {2d,3d}`, `--launch-simrobot`, `--simrobot-bin` |
| `python train_multiagent_production.py` | `--stage-id`; the `tools/*.sh` scripts also pass `--checkpoint` and `--dataset` |
| `python train_production.py` | `--dataset`, `--output-dataset`, `--checkpoint`, `--prepare-only` |
| `python train_defender_bc.py` | `--dataset`, `--steps`, `--batch-size`, `--lr`, `--init-checkpoint`, `--reference-checkpoint`, `--distill-weight`, `--output-dir`, ... |
| `pytest test_*.py` | Contract tests; some may need `pybh` |

The `tools/run_*.sh` scripts record the exact recipes used for each policy version (for example `tools/run_v4_2.sh`, `tools/train_merge_v5.sh`).
These commands could not be executed during documentation work (the authors' `pybh` build was not usable in the documentation environment); flags are quoted from the source, not from `--help` output.

## Data and runs policy

- `runs/` is gitignored: datasets (`*.npz`), logs, telemetry and training checkpoints are not versioned.
- Tracked deliverables: root `ppo_*.onnx` weights with their `ppo_*.json` manifests, and `pt_defender_47/checkpoint_stage8_step2500.pt` and `checkpoint_stage8_step10000.pt`.

| Policy | Files |
|---|---|
| Striker | `ppo_striker_hsl2026.{onnx,json}` (+ `.onnx.data`) |
| Defender | `ppo_defender_hsl2026.{onnx,json}` |
| Team (striker v4.1, v4.2) | `ppo_team_hsl2026.*`, `ppo_team_hsl2026_v4_1.*`, `ppo_team_hsl2026_v4_2.*` |
| Merged brain v5 | `ppo_team_hsl2026_v5_merged.{onnx,json}` |
| Goalkeeper | `ppo_goalkeeper_hsl2026*.{onnx,json}` (base, `ppo`, `ppo_validated`, and the `gk_*` variants) |

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
