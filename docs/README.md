# Documentation index

Index of every file under `docs/` (working notes live in `docs/notes/`). Most documents are working logs written during development; many are in Spanish (column `Lang`).
Descriptions are derived from each file's title and opening lines.
Most documents refer to modules by their pre-refactor flat names (for example `train.py`); the old-to-new table is in [REPO_LAYOUT.md](REPO_LAYOUT.md).
Related: [root README](../README.md), [branch registry](BRANCHES.md), [paper traceability](PAPER_TRACEABILITY.md).

Topics: `bridge` (RL to B-Human/SimRobot wiring and environment), `striker`, `defender`, `goalkeeper`, `multiagent`, `audit` (validation, evaluation, stability).

## docs/

| File | Description | Topic | Lang |
|---|---|---|---|
| [00-estado-actual.md](00-estado-actual.md) | Current state: the project splits into two sibling repositories | bridge | ES |
| [01-flujo-natural-bhuman.md](01-flujo-natural-bhuman.md) | Natural B-Human flow: RL replaces only the high-level decision, not the walking engine | bridge | ES |
| [02-fisica-simrobot-2d-vs-3d.md](02-fisica-simrobot-2d-vs-3d.md) | SimRobot physics: the 2D and 3D simulation profiles | bridge | ES |
| [03-contrato-rl-env.md](03-contrato-rl-env.md) | RL contract, backend and environment (canonical action in `action.py`) | bridge | ES |
| [04-verificacion-operativa.md](04-verificacion-operativa.md) | Operational verification commands | audit | ES |
| [05-premisas-irrompibles.md](05-premisas-irrompibles.md) | Unbreakable premises: architecture constraints | bridge | ES |
| [06-simrobot-skill-matrix.md](06-simrobot-skill-matrix.md) | RL skill validation in SimRobot, 2D vs 3D comparison (2026-05-01) | audit | ES |
| [07-observation-space-realista-y-fallback-canary.md](07-observation-space-realista-y-fallback-canary.md) | Realistic observation space and canary fallback | bridge | ES |
| [09-implementacion-ppo-observation-space-23.md](09-implementacion-ppo-observation-space-23.md) | PPO implementation with the enlarged 23-dim observation space | striker | ES |
| [10-escenarios-rl-sin-oracle-world-model.md](10-escenarios-rl-sin-oracle-world-model.md) | RL scenarios without an oracle world model | bridge | ES |
| [11-rl1v0-3d-percepcion-natural.md](11-rl1v0-3d-percepcion-natural.md) | `RL1v0_3D` natural perception and diagnosis | bridge | ES |
| [12-wire-rl-realista-final.md](12-wire-rl-realista-final.md) | Final realistic RL to B-Human wiring | bridge | ES |
| [13-pruebas-gui-simrobot.md](13-pruebas-gui-simrobot.md) | 3D GUI test commands (contains absolute paths from the authors' machine) | audit | ES |
| [14-acceleration-pipeline.md](14-acceleration-pipeline.md) | Acceleration pipeline: single Python process, single learner | bridge | ES |
| [15-guia-escenas-y-escenarios-rl.md](15-guia-escenas-y-escenarios-rl.md) | Practical guide to building RL scenes and scenarios | bridge | ES |
| [16-curriculum-dinamico-balon-y-rivales.md](16-curriculum-dinamico-balon-y-rivales.md) | Dynamic curriculum: ball and opponents | bridge | ES |
| [17-handoff-visible-simrobot-gui.md](17-handoff-visible-simrobot-gui.md) | Handoff for the `simrobot_visible` GUI backend | bridge | ES |
| [18-validacion-sanidad-envs-n6.md](18-validacion-sanidad-envs-n6.md) | Sanity validation of N=6 parallel environments | audit | ES |
| [19-entorno-validacion.md](19-entorno-validacion.md) | Reusable validation environment | audit | ES |
| [20-validacion-n6-2026-05-15.md](20-validacion-n6-2026-05-15.md) | N=6 validation run of 2026-05-15 | audit | ES |
| [21-simrobot-estable-y-rl-vs-bhuman.md](21-simrobot-estable-y-rl-vs-bhuman.md) | Stable SimRobot and hybrid RL vs B-Human match | bridge | ES |
| [22-handoff-optimizacion-training-32cores.md](22-handoff-optimizacion-training-32cores.md) | Handoff: training efficiency on a 32-core host | bridge | ES |
| [23-plan-transplante-ppo-naov6-hsl2026.md](23-plan-transplante-ppo-naov6-hsl2026.md) | Plan to transplant PPO to NAO v6 for RoboCup HSL 2026 (cut-off 2026-06-08) | striker | ES |
| [24-multiagent-obs-extension-26-to-41-spec.md](24-multiagent-obs-extension-26-to-41-spec.md) | Spec: multi-agent observation extension from 26 to 41 dimensions | multiagent | EN |
| [25-guia-transplante-ppo-bhuman.md](25-guia-transplante-ppo-bhuman.md) | Guide to transplanting the RL policy into B-Human (inference pipeline) | striker | ES |
| [25-mappo-wire-findings.md](25-mappo-wire-findings.md) | MAPPO wire findings in SabanaHerons (actual 3D semantics) | multiagent | EN |
| [26-implementacion-ppo-a-transplantar.md](26-implementacion-ppo-a-transplantar.md) | Description of the PPO implementation to be transplanted (cut-off 2026-06-08) | striker | ES |
| [27-estabilidad-behavioral-evaluator.md](27-estabilidad-behavioral-evaluator.md) | Stability of the behavioral evaluator | audit | ES |
| [28-estado-final-ppo-sabanaherons.md](28-estado-final-ppo-sabanaherons.md) | Final PPO state in SabanaHerons (cut-off 2026-06-11) | striker | ES |
| [29-embedded-encoder-26-to-47-spec.md](29-embedded-encoder-26-to-47-spec.md) | Spec: embedded PPO encoder extension from obs26 to obs47 | multiagent | ES/EN |
| [30-defender-finetune-handoff.md](30-defender-finetune-handoff.md) | Defender 47-dim execution handoff (v3, 2026-06-13) | defender | ES |
| [31-stage8-defender-visible-debug.md](31-stage8-defender-visible-debug.md) | Stage 8 defender: visible SimRobot single process | defender | ES |
| [32-stage8-defender-checklist.md](32-stage8-defender-checklist.md) | Stage 8 defender: operational checklist | defender | ES |
| [33-stage8-defender-initial-train-eval.md](33-stage8-defender-initial-train-eval.md) | Stage 8 defender: evaluation of the initial training | defender | ES |
| [34-defender-v2-recovery-run-eval-validate.md](34-defender-v2-recovery-run-eval-validate.md) | Defender v2 (recovery + clear/pass): run, evaluate, validate | defender | ES |
| [35-dataset-v3-teammate-contingent-teacher.md](35-dataset-v3-teammate-contingent-teacher.md) | Dataset v3: teammate-contingent teacher and contrastive worlds | defender | ES |
| [35-defender-v7-block-pass-gate-fixes.md](35-defender-v7-block-pass-gate-fixes.md) | Defender v7: why `block`/`pass` were at 0 and how it was fixed (code superseded by the v4.2/v5 design) | defender | ES |
| [36-goalkeeper-action-space-sabana.md](36-goalkeeper-action-space-sabana.md) | Goalkeeper action space from Sabana Herons | goalkeeper | EN |
| [37-striker-v2b-to-v4.2-and-cpp-transplant.md](37-striker-v2b-to-v4.2-and-cpp-transplant.md) | Striker v2b to v4.2 and the C++ transplant of the striker agent | striker | EN |
| [38-merged-striker-defender-and-naov6-transplant.md](38-merged-striker-defender-and-naov6-transplant.md) | Merged striker+defender policy (v5) and its NAO v6 transplant | multiagent | EN |
| [39-goalkeeper-closedloop-and-naov6-transplant.md](39-goalkeeper-closedloop-and-naov6-transplant.md) | Goalkeeper closed-loop RL build and NAO v6 C++ transplant | goalkeeper | EN |
| [40-obs26-stability-repair-setup.md](40-obs26-stability-repair-setup.md) | Obs-26 setup to repair the embedded PPO (renumbered from 28) | audit | ES |
| [41-defender-obs26-pipeline-canonico.md](41-defender-obs26-pipeline-canonico.md) | Canonical obs-26 defender pipeline (renumbered from 29) | defender | ES |
| [42-recuperacion-defender-onnx-a-pt.md](42-recuperacion-defender-onnx-a-pt.md) | Short recovery of the defender `.pt` from its ONNX (renumbered from 30) | defender | ES |
| [README.md](README.md) | This index | - | EN |

Numbering notes: document 08 is absent; two files share number 25 and two share number 35.

## docs/notes/

| File | Description | Topic | Lang |
|---|---|---|---|
| [notes/09-production-selfplay-and-multiagent-handoff.md](notes/09-production-selfplay-and-multiagent-handoff.md) | Handoff: production self-play, multi-agent observation space and 4v4 wrapper-coordinated deployment | multiagent | EN |
| [notes/10-graph-coordination-post-production-roadmap.md](notes/10-graph-coordination-post-production-roadmap.md) | Graph coordination: one-week post-production sprint plan | multiagent | EN |
| [notes/bhuman-logic/Bhuman-deterministic-agent-map.md](notes/bhuman-logic/Bhuman-deterministic-agent-map.md) | Map of the B-Human deterministic agent code used as the base for the teacher policy | audit | ES |

## Other working notes (formerly at the repository root)

| File | Description | Lang |
|---|---|---|
| [notes/HANDOFF.md](notes/HANDOFF.md) | Defender v2 (doc 34) validation in SimRobot/Linux, 2026-06-15 | ES |
| [notes/CODEX_OFFBALL_PERCEPTION_HANDOFF.md](notes/CODEX_OFFBALL_PERCEPTION_HANDOFF.md) | Task note: fix off-ball defender opponent perception (head/gaze) in the multi-agent scene | EN |
| [notes/cambios_para_produccion.md](notes/cambios_para_produccion.md) | Historical note: changes to reach 4/5 evaluation criteria for the production PPO | ES |
