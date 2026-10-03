# Paper traceability

Maps the quantitative claims of the paper ("A Teacher-Guided RL Pipeline for Role-Conditioned Football Cognition in Humanoid Robots") to evidence in this repository.
Values were read from the paper source; "Where" lists only locations verified with `rg` on tracked files. Line numbers refer to the state of the repository at the time of writing.

Status: `traced` = value found in a tracked document or code; `partial` = some of the value found or only quoted as text; `not in repository` = not found.

Decision D1: the corpus-size numbers come from results produced outside this repository (a co-author's work) and were never committed. The paper values stand and are not corrected here.
Datasets and checkpoints live under `runs/`, which is gitignored, so no dataset or raw evaluation output is tracked for any row below.

## Claims

| # | Claim | Paper value | Where in this repo | Status | Note |
|---|---|---|---|---|---|
| 1 | Final fine-tuning corpus size | 139,623 samples (abstract, Table I) | `docs/38-merged-striker-defender-and-naov6-transplant.md` section 2, step 3 (text: "139 623 rows") | not in repository | Results produced outside this repository; paper values stand. Dataset `dataset_merged_v5_20260618` is not tracked. |
| 2 | Final corpus episodes | 74 total | none | not in repository | Results produced outside this repository; paper values stand. |
| 3 | Final corpus positives | block 2,817; mark 673; pass 2,015; shoot 2,565 | `docs/38`, section 2 (block 2817, mark 673, pass 2015) | partial | Shoot 2,565 not found. |
| 4 | Attack corpus size | 105,624 rows; 45 kept episodes | `docs/38`, section 2, step 3 (text: "105 624 rows") | not in repository | Results produced outside this repository; paper values stand. Kept-episode count not found. |
| 5 | Attack corpus positives | pass 2,015; shoot 1,677; dribble 8,758; walk 38,255; stand 54,919 | `docs/38`, section 2, step 3 (pass 2015, shoot 1677) | partial | Dribble, walk, stand counts not found. |
| 6 | Cooperation/striker corpus | 434,639 rows; pass 16,914; shoot 5,702 | none | not in repository | Results produced outside this repository; paper values stand. |
| 7 | Defender corpus | 33,999 rows; 29 kept episodes; mark 673; block 2,817 | `docs/38`, section 2, step 2 (text: "33 999 rows, mark 673 / block 2817") | not in repository | Results produced outside this repository; paper values stand. Kept-episode count not found. |
| 8 | Single-agent 26-dim dataset | 17,815 rows; shoot positives 4,212 | `docs/40-obs26-stability-repair-setup.md` line 144 ("positive=16402/17815") | partial | Row count appears for a production replay dataset; that it is the same dataset, and the 4,212 figure, are to be confirmed by the authors. |
| 9 | Skill agreement with pre-merge policy | 96.7% | `docs/38` section 3; computing script `tools/eval_merged_offline.py` | traced | Quoted in the document; raw evaluation output not tracked. |
| 10 | Pass / shoot / dribble recall | 100.0% / 99.9% / 96.4% | `docs/38` section 3 | traced | Same note as row 9. |
| 11 | Mark recall | 93.2% | `docs/38` section 3 | traced | Same note as row 9. |
| 12 | Defender score | about 0.76 | `docs/38` section 3 ("in-sim defender_score 0.76") | traced | Metric computed in `environment.py` / `train.py`; raw run output not tracked. |
| 13 | Goal-side rate | about 0.90 | `docs/38` section 3 ("goal_side 0.90") | traced | Abstract rounds to 0.90; table says about 0.90. |
| 14 | Runtime gate violations | zero | `docs/38` section 3 ("0 gate violations") | traced | Quoted in the document only. |
| 15 | Role flip effect | `mark` in about 61% of cases | `docs/38` section 1 ("mark 61%") | traced | Quoted in the document only. |
| 16 | Multi-agent run | 22 episodes | `docs/38` section 3 ("22 ep") | traced | The document also reports other metrics from that run; raw output not tracked. |
| 17 | Production stress robustness | 28 SimRobot episodes (figure) | none | not in repository | Figure and underlying run output not in the repository. Production scenarios are defined in `production_curriculum.py`. |
| 18 | BC replay coefficient schedule | beta_BC from 0.08 to 0.02 | `docs/38` section 2, step 4; `tools/train_merge_v5.sh` line 7 | traced | |
| 19 | Observation size and role block | 47 dimensions; role one-hot at indices 44-46 | `observation.py` (`OBS_SIZE` evaluates to 47; role block at indices 44-46) | traced | |
| 20 | Action space | 8 skills (+ 4 parameters) | `action.py` (`SkillType` has 8 members; `ACTION_PARAM_MASKS` has 4 entries per skill) | traced | |
| 21 | Positive-label share in final corpus | walk+stand 85.94%; pass+shoot+mark 3.76% | none | not in repository | Derived from the final corpus, which is not tracked. |
| 22 | Reward decomposition at 200k steps | good_walk about 96, walk_persistence about 63, good_dribble about 31, good_shoot 0, goal 0 (about 30x ratio) | none | not in repository | Run output not tracked. |
| 23 | Teacher heuristic constants | ball control below 350 mm; 7-frame hysteresis; 200 mm gap; pass overrides at 0.15 margin with 1.5 s penalty; sigma 800 mm | none | not in repository | These live in the C++ repository (`Behavior.cpp`, `PlayBall.cpp`), which is not part of this one; to be confirmed by the authors. `.docs/10-graph-coordination-post-production-roadmap.md` lists some similar constants for a Python port; it was not checked against the paper values. |

## Missing assets

| Asset | Status |
|---|---|
| Figures (`production_stress_metrics.png`, `teacherstability.png`, others) | Not in the repository |
| `references.bib` | Not in the repository |
| `tools/run_defv2_*.sh` (cited by working notes such as `HANDOFF.md`) | Found: `run_defv2_chunk.sh`, `run_defv2_finetune.sh`, `run_defv2_floor.sh`, `run_defv2_stability.sh`, `run_defv2_stability_seed67.sh` (checked with `fd 'run_defv2' tools`) |
| Datasets, run logs, training checkpoints | Under `runs/` (gitignored); not available from this repository |
| Corpus-generation results behind rows 1-7 | Produced outside this repository; to be confirmed by the authors |

## TODO

- [ ] Publish the results reported in the paper (rows marked `not in repository` or `partial`): the corpus datasets and their summaries (rows 1-8, 21), the raw evaluation outputs behind rows 9-16, the production stress run behind row 17, and the reward decomposition behind row 22. Owner: the co-author who produced them. Once published, update the `Where in this repo` and `Status` columns.
- [ ] Add the paper figures and `references.bib`, or link to where they are hosted.
