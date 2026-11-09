# RL for role-conditioned football in humanoid robots (Python side)

Python reinforcement-learning code for the paper
**"A Teacher-Guided RL Pipeline for Role-Conditioned Football Cognition in Humanoid Robots"**
(NAO V6 in SimRobot, SabanaHerons / B-Human stack): a 47-dimensional observation,
a behavioral-cloning (BC) warm start, PPO with a KL anchor, and a role-conditioned
merged policy (striker, defender, goalkeeper).

The C++ side of the system (B-Human fork, SimRobot, the `pybh` binding and native ONNX inference) lives in
[juanulloa2050/SabanaHerons2026](https://github.com/juanulloa2050/SabanaHerons2026).

Many documents under `docs/` are written in Spanish and are historical work logs.
See [docs/README.md](docs/README.md) for the index.

## Two-repository architecture

| Repository | Language | Role |
|---|---|---|
| This repository | Python | Environments, teachers, dataset generation, BC/PPO training, evaluation, ONNX export |
| [`SabanaHerons2026`](https://github.com/juanulloa2050/SabanaHerons2026) (B-Human fork) | C++ | SimRobot, the `pybh` Python binding, `RLSharedState` bridge, native ONNX inference |

Python talks to B-Human in-process through `pybh` (see `rlfootball/core/pybh_backend.py`).
`rlfootball/core/config.py` expects `SabanaHerons2026` as a sibling directory of this repository.
The in-process `pybh` binding is the only supported bridge; there is no subprocess stack.

Required `SabanaHerons2026` branches:

| Purpose | Branch |
|---|---|
| Multi-agent training in this repo | `rl-simrobot3d` (bridge version 30, multi-agent fields) |
| Deploying the ONNX policies | `hsl2026-ppo-team-v4` (striker v4.2: [docs/37](docs/37-striker-v2b-to-v4.2-and-cpp-transplant.md); merged brain v5: [docs/38](docs/38-merged-striker-defender-and-naov6-transplant.md)) |

## Repository map

| Path | Purpose |
|---|---|
| `rlfootball/core/` | Configuration, observation (47-dim), action space, reward, skill gate, `pybh` backend, wire audit |
| `rlfootball/curriculum/` | Curriculum stage specs and production scenarios |
| `rlfootball/teachers/` | Rule-based teachers (single-agent, multi-agent, goalkeeper, defender) |
| `rlfootball/env/` | Environments and the skrl wrapper |
| `rlfootball/datasets/` | Teacher dataset collection, merging and audits |
| `rlfootball/training/` | PPO/MAPPO entry points (`train`, `train_multiagent`), production wrappers, BC and repair trainers, and the all-learned team trainers (`train_dagger_v5`, `train_distill_hybrid`, `kickstart_ppo`, `learned_coordinator`) |
| `rlfootball/evaluation/` | Behavioral, model and teacher evaluators, stability checks, RL vs B-Human 3v3, live 4v4 match runners (`run_v5_vs_bhuman_match`, `run_teacher_vs_bhuman_match`, `run_probe_match`) |
| `rlfootball/diagnostics/` | Visible-SimRobot checks, live probes, sweeps, visualisation |
| `tests/` | Contract tests and the bridge smoke test |
| `tools/` | Grouped by purpose: `run/` (launchers, `run_*.sh`), `export/` (ONNX export), `eval/` (evaluation, validation, probes, held-out and match statistics), `dataset/` (dataset generation and audits), `paper/` (paper figures and match-metric tables), `misc/` |
| `results/` | Small tracked result files; `results/paper/` holds the held-out ablation and match-metric CSVs (see "Latest progress: from v5 to an all-learned team") |
| `scripts/` | Validation environment setup (`setup_validation_env.sh`, `run_with_validation_env.sh`) |
| `models/` | Tracked policy deliverables and manifests (see below) |
| `models/defender/checkpoints/` | Tracked defender PyTorch checkpoints |
| `docs/` | Documentation (index: [docs/README.md](docs/README.md)) |
| `docs/notes/` | Historical working notes (`HANDOFF.md`, `CODEX_OFFBALL_PERCEPTION_HANDOFF.md`, `cambios_para_produccion.md`, `bhuman-logic/`) (mostly Spanish) |
| `environment.yml` | Conda environment (see "Environment setup") |

## Environment setup

`environment.yml` creates a conda environment with PyTorch (CUDA 11.8), `skrl`, `pettingzoo` and `gymnasium`. In addition:

- Use the Python version your `pybh` build targets; `scripts/setup_validation_env.sh` creates a `python3` virtual environment in `.venv`.
- Install `onnx` and `onnxruntime` for export and inference, and `pytest` for the tests.
- Plain `pettingzoo` is enough; `pettingzoo[all]` also builds box2d, which needs swig.

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

## Data and runs policy

- `runs/` is gitignored: datasets (`*.npz`), logs, telemetry and training checkpoints are not versioned.
- Tracked deliverables: `ppo_*.onnx` weights with their `ppo_*.json` manifests under `models/<role>/`, and `models/defender/checkpoints/checkpoint_stage8_step2500.pt` and `checkpoint_stage8_step10000.pt`.

| Policy | Files |
|---|---|
| Striker | `models/striker/ppo_striker_hsl2026.{onnx,json}` (+ `.onnx.data`) |
| Defender | `models/defender/ppo_defender_hsl2026.{onnx,json}` |
| Team (striker v4.1, v4.2) | `models/team/ppo_team_hsl2026.*`, `ppo_team_hsl2026_v4_1.*`, `ppo_team_hsl2026_v4_2.*` |
| Merged brain v5 | `models/team/ppo_team_hsl2026_v5_merged.{onnx,json}` |
| Final all-learned team | `models/team/ppo_team_hsl2026_final_alllearned.onnx` |
| Learned support coordinator | `models/team/ppo_coord_hsl2026_final_learned_coordinator.onnx` |
| Goalkeeper | `models/goalkeeper/ppo_goalkeeper_hsl2026*.{onnx,json}` (base, `ppo`, `ppo_validated`, and the `gk_*` variants) |

## Datasets

Teacher datasets used for supervised policy learning (cited in the paper).

| Dataset | Purpose | Size / key positives |
| --- | --- | --- |
| Single-agent (26-dim) | Single-agent production replay: walk, dribble, shoot | 17,815 rows; shoot 4,212 |
| Multi-agent attack | Attack, pass, support | 105,624 rows; 45 kept episodes; pass 2,015, shoot 1,677, dribble 8,758, walk 38,255, stand 54,919 |
| Multi-agent cooperation and striker | Pass-rich attack overlay | 434,639 rows; pass 16,914, shoot 5,702 |
| Multi-agent defender | Defender, block and mark | 33,999 rows; 29 kept episodes; mark 673, block 2,817 |
| Final attack–defense merge | Striker–defender–support fine-tuning | 139,623 rows; 74 episodes; block 2,817, mark 673, pass 2,015, shoot 2,565 |

In the final merge corpus, `walk` and `stand` account for 85.94% of positive labels, while `pass`, `shoot`, and `mark` together account for 3.76%.

## Latest progress: from v5 to an all-learned team

Up to v5, every policy in this repository learned from teacher datasets collected in scripted scenarios: the rule-based teacher decides, the network imitates it, and PPO fine-tunes the result.
The latest step takes training into full matches.
The team now learns from live 4v4 games against a stock B-Human team, using its own observations (no oracle), and the outcome is a single network that plays all three field roles plus a small learned coordinator for the support robots.
This section explains how that was done and what the measurements show.

### How the new team is trained

The pipeline has four stages, each one a module in this repository:

1. **Probe matches.** `rlfootball/evaluation/run_probe_match.py` plays live 4v4 matches against real B-Human and records, for every step, the observation, the expert's choice (teacher plus B-Human's kick macro) and what B-Human itself would have requested.
2. **Imitation on the student's own states.** `rlfootball/training/train_dagger_v5.py` runs DAgger rounds, letting the student play and relabelling the states it actually visits, while a KL term keeps it close to v5. `rlfootball/training/train_distill_hybrid.py` then distils one policy from the learned striker and B-Human's support behaviour (skill cross-entropy, support-position regression, KL to the starting checkpoint).
3. **Reinforcement learning on real matches.** `rlfootball/training/kickstart_ppo.py` improves that policy with kickstarted PPO on batches of sampled live matches: the usual PPO-clip and GAE objective, plus a teacher cross-entropy term that fades out over iterations.
4. **A safety gate between iterations.** `tools/misc/kickstart_gate.py` only accepts a new candidate when its margin over a stock-B-Human control played at the same time, and its points per match, beat the baseline.

Alongside the policy, `rlfootball/training/learned_coordinator.py` trains a support coordinator that imitates where B-Human places its support robots.
It maps the same 47-dim observation to a field target, and the policy's own residual is applied on top, so the observation and action contracts do not change.

The two resulting networks are in `models/team/`:

| File | What it is |
|---|---|
| `models/team/ppo_team_hsl2026_final_alllearned.onnx` | The all-learned team policy: one network for striker, open support and off-ball support, with the same 47-dim contract as v5. `tools/eval/heldout_policy_compare.py` identifies the final team as `runs/kickstart/loop_20261006_224436/iter_12/policy.pt` |
| `models/team/ppo_coord_hsl2026_final_learned_coordinator.onnx` | The learned support coordinator: observation in, normalised field target (x/4500, y/3000) out |

### Why the warm start matters: held-out comparison

A natural question is whether all of this is needed, or whether PPO alone would get there.
To answer it, the stage-8 recipe was trained in several variants with independent seeds and scored on data none of them saw during training: the 2026-10-03 probe matches against real B-Human (37 matches according to `tools/eval/heldout_policy_compare.py`), where every field-robot step carries an expert label.
Each policy's choice is compared with that label under the Design-A gate mask.

The variants are:

- `FULL`: warm start + behaviour cloning + KL anchor.
- `NOKL`: the same without the KL anchor.
- `NOBC`: the same without behaviour cloning (KL only).
- `WSONLY`: warm start + PPO, with neither behaviour cloning nor KL.
- `PPO`: PPO alone, from random initialisation.
- `PPOLONG`: PPO alone, trained to 30,000 steps.

Each cell below is the mean ± sample standard deviation over seeds, taken at the variant's last checkpoint. The metrics are:

- **Agreement:** share of steps where the policy picks the expert's skill.
- **Balanced accuracy:** the mean recall over the seven skills.
- **Entropy:** how varied the chosen skills are. Close to 0 means the policy always does the same thing.

| Variant | Final step | Seeds | Agreement | Balanced acc. | Recall pass | Recall block | Recall mark | Pred. entropy (bits) |
|---|---:|---:|---|---|---|---|---|---|
| FULL | 10000 | 6 | 0.541 ± 0.048 | 0.519 ± 0.020 | 0.859 ± 0.011 | 0.680 ± 0.001 | 0.888 ± 0.021 | 1.741 ± 0.120 |
| NOKL | 10000 | 6 | 0.541 ± 0.079 | 0.590 ± 0.017 | 0.810 ± 0.025 | 0.682 ± 0.001 | 0.939 ± 0.016 | 1.739 ± 0.191 |
| NOBC | 10000 | 6 | 0.473 ± 0.064 | 0.508 ± 0.024 | 0.867 ± 0.009 | 0.848 ± 0.161 | 0.893 ± 0.038 | 1.750 ± 0.119 |
| WSONLY | 10000 | 6 | 0.530 ± 0.101 | 0.510 ± 0.024 | 0.869 ± 0.022 | 0.810 ± 0.015 | 0.870 ± 0.043 | 1.657 ± 0.219 |
| PPO | 10000 | 6 | 0.757 ± 0.217 | 0.142 ± 0.004 | 0.006 ± 0.010 | 0.000 ± 0.000 | 0.001 ± 0.003 | 0.209 ± 0.380 |
| PPOLONG | 30000 | 3 | 0.852 ± 0.001 | 0.143 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.010 ± 0.007 |

Source: [`results/paper/ablation_heldout_per_checkpoint.csv`](results/paper/ablation_heldout_per_checkpoint.csv), produced by `tools/eval/ablation_offline_eval.py`.

PPO alone looks good on raw agreement, but that is misleading.
Its entropy is close to zero and its balanced accuracy is about 0.14, which means it collapses onto the most frequent expert skill and practically never passes, blocks or marks.
Training it three times longer does not change that.
Every warm-started variant keeps a varied repertoire (1.66 to 1.75 bits) and a balanced accuracy between 0.51 and 0.59.
Among the warm-started variants the differences in agreement are within one standard deviation, so this comparison does not rank them; dropping the KL anchor (`NOKL`) gives the best balanced accuracy.
For reference, all warm-started variants start from the same checkpoint (agreement 0.531, entropy 1.739 bits), and the CSV also scores the v5 policy (agreement 0.615, balanced accuracy 0.537, entropy 1.525 bits).

### How v5 performed in live matches against B-Human

Before the new pipeline, the v5 policy was played in batches of live 600-second matches against real B-Human (`rlfootball/evaluation/run_v5_vs_bhuman_match.py`, with players 2 and 3 controlled by the policy).
These results describe **v5, not the final all-learned team**.
Goals use the corrected count, where a goal is confirmed only by the same robot that saw the ball cross the line (`tools/paper/recompute_goals_corrected.py` explains the bug this fixes).
Possession, recoveries and passes are estimated from ball and robot telemetry.
Each batch keeps the name it was recorded under.
Averages include stalled matches.

| Batch | Matches | W-D-L | Goals for-against | Stalled | Possession RL / opp. / loose+unknown (%) | Recoveries RL / opp. per match | Pass attempts per match (success) |
|---|---:|---|---|---:|---|---|---|
| `3v3` | 10 | 2-4-4 | 3-5 | 1 | 3.0 / 8.7 / 87.6 | 4.1 / 4.6 | 0.0 (0/0) |
| `4v4` | 10 | 1-7-2 | 2-4 | 0 | 3.4 / 7.4 / 88.5 | 7.5 / 8.0 | 4.9 (7/49) |
| `FINAL_reactive` | 10 | 1-9-0 | 1-0 | 0 | 2.3 / 30.0 / 65.3 | 14.4 / 14.3 | 4.9 (0/49) |
| `repair1_4v4` | 10 | 0-10-0 | 0-0 | 0 | 3.9 / 11.2 / 83.7 | 10.2 / 10.7 | 7.5 (15/75) |
| `repair2_4v4` | 10 | 2-7-1 | 5-3 | 0 | 4.2 / 12.1 / 82.9 | 9.1 / 9.1 | 4.9 (4/49) |
| `repair3_4v4` | 10 | 1-9-0 | 1-0 | 2 | 3.7 / 10.7 / 84.6 | 9.5 / 9.9 | 3.5 (3/35) |
| `repair4_4v4` | 10 | 0-10-0 | 0-0 | 0 | 3.0 / 11.7 / 83.9 | 8.2 / 8.8 | 4.5 (4/45) |
| `gt_repair2` | 5 | 0-5-0 | 0-0 | 2 | 2.3 / 33.8 / 61.6 | 16.2 / 15.8 | 4.2 (1/21) |
| `gt_repair4` | 5 | 1-4-0 | 1-0 | 0 | 2.4 / 30.9 / 63.7 | 18.2 / 18.0 | 7.2 (0/36) |

Source: `results/paper/table_match_metrics_v5_vs_real_bhuman_<batch>.csv` (one row per match), produced by `tools/paper/summarize_v5_vs_bhuman_matches.py`.

Most of these matches end in a draw, and possession looks low for both teams because the ball is classed as loose or unknown most of the time.
The batches differ in scene and settings, so compare them with care.
An earlier, aggregated comparison of v5 against the B-Human/Sabana-Herons heuristic baseline (15 episodes per side) is in [`table_match_metrics_v5_vs_bhuman.csv`](results/paper/table_match_metrics_v5_vs_bhuman.csv), produced by `tools/paper/generate_emergency_eval_figures.py`.
For example, it reports ball progression of 0.10 vs 0.20 m per episode and a possession proxy of 66.3% vs 72.7%.

### Running it yourself

Run everything from the repository root.
Inputs and outputs live under `runs/`; each module docstring lists what it reads.
The live-match commands need a built `pybh` and the SimRobot binary (`RL_SIMROBOT_BIN`, or the default under `SABANAHERONS_DIR`).

| Goal | Command |
|---|---|
| One live match, v5 vs B-Human | `python -m rlfootball.evaluation.run_v5_vs_bhuman_match --launch --seconds 600 --output runs/<batch>/match_1.json` |
| One live match, teacher vs B-Human | `python -m rlfootball.evaluation.run_teacher_vs_bhuman_match --launch --seconds 600 --output runs/<batch>/match_1.json` |
| Probe match (`--controller expert\|teacher\|v5\|bhuman\|dagger\|hybrid`) | `python -m rlfootball.evaluation.run_probe_match --controller v5 --launch --output runs/<dir>/match.json` |
| B-Human-labelled self-play dataset | `python tools/dataset/collect_bhuman_selfplay_dataset.py --launch --seconds 600 --out runs/bhuman_selfplay_dataset/match_1` |
| DAgger round | `python -m rlfootball.training.train_dagger_v5 --npz runs/.../V_*_a0.npz --out runs/dagger_v5/round1.pt` |
| Distil the hybrid team | `python -m rlfootball.training.train_distill_hybrid --json "runs/.../HD_*.json" --init runs/dagger_v5/round2.pt --out runs/distill_hybrid/round1.pt` |
| Learned coordinator | `python -m rlfootball.training.learned_coordinator --json "runs/.../*.json" --out runs/learned_coord/coord_v1.pt` |
| Kickstarted PPO iteration | `python -m rlfootball.training.kickstart_ppo --json "runs/kickstart/.../iter_003/*.json" --policy cur.pt --anchor runs/distill_hybrid/round3.pt --iteration 3 --out next.pt` |
| Held-out comparison table | `python tools/eval/ablation_offline_eval.py` (defaults: `--runs "runs/ablation_paper_*_s*"`, `--heldout "runs/full_probe/**/*.npz"`, `--out runs/paper_offline/ablation`) |
| Held-out comparison across policy generations | `python tools/eval/heldout_policy_compare.py` (output `runs/paper_offline/heldout_policies/`) |
| Match statistics with bootstrap CIs | `python tools/eval/paper_match_stats.py --arm "P=runs/paper_eval/block1/P_*.json" --arm "C=..." --compare P-C --out runs/paper_eval/stats` |
| Per-match metrics CSV | `python tools/paper/summarize_v5_vs_bhuman_matches.py runs/paper_match_v5_vs_bhuman_live/batch_<stamp>` (writes `results/paper/table_match_metrics_v5_vs_real_bhuman*.csv`) |
| Figures | `python tools/paper/paper_figures_final.py` (also `paper_figures_icra.py`, `paper_figures_tutor.py`, `generate_paper_figures.py`, `generate_emergency_eval_figures.py`); output in `results/paper/figures/` |

The other analysis scripts in `tools/eval/` (`analyze_full_probe.py`, `offline_heldout_eval.py`, `role_conditioning_analysis.py`, `mirror_detector.py`, `diagnose_localization.py`, `time_onnx_inference.py`, ...) and `tools/paper/` (`paper_skill_analyses.py`, `recompute_goals_corrected.py`) describe their inputs and outputs in their docstrings.

## Relation to the paper

| Paper element | Where it lives in this repository |
|---|---|
| 47-dimensional observation | `rlfootball/core/observation.py` |
| Skill action space and forbid-invalid skill gate | `rlfootball/core/action.py`, `rlfootball/core/gate.py` |
| Rule-based teachers | `rlfootball/teachers/` |
| Curriculum stages and production scenarios | `rlfootball/curriculum/` |
| Teacher datasets | `rlfootball/datasets/` (sizes in "Datasets") |
| Behavioral-cloning warm start | `rlfootball/training/train_bc.py` |
| PPO with KL anchor and BC replay | `rlfootball/training/train.py` (`BCReplayRegularizer`) |
| Role-conditioned merged policy (v5) | `models/team/ppo_team_hsl2026_v5_merged.onnx`, [docs/38](docs/38-merged-striker-defender-and-naov6-transplant.md) |
| All-learned team, held-out comparison and live matches | "Latest progress: from v5 to an all-learned team" |

[docs/PAPER_TRACEABILITY.md](docs/PAPER_TRACEABILITY.md) maps each claim in the paper to the file that supports it.

## Project documents

| Document | Content |
|---|---|
| [docs/README.md](docs/README.md) | Index of all documentation |
| [docs/PAPER_TRACEABILITY.md](docs/PAPER_TRACEABILITY.md) | Paper claims mapped to repository evidence |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Project structure, old-to-new transition guide and integration rules |
| [CLAUDE.md](CLAUDE.md) | Project instructions for Claude Code sessions |
