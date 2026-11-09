# Stage 8 Defender: checklist operativo

Estado al 2026-06-13: el pipeline visible stage 8 ya corre en modo
single-SimRobot, con escena multiagente corregida y `PYBH_RL_SHM_NAME` fresco
automatico. Se consolidaron las runs reales disponibles, incluyendo mas
`transition_midfield`, y se genero un dataset defensivo bootstrap auditable:
`runs/teacher_multiagent_dataset/dataset_defense_stage8_bootstrap_v2`.
Ese dataset ya fue mezclado con el dataset ofensivo stage 6/7 en
`runs/teacher_multiagent_dataset/dataset_attack_plus_defense_stage8_v1`.
El fine-tune inicial stage 8 termino en
`runs/defender_stage8_v4_defreward_single_dynamic`; el checkpoint recomendado
para continuar es `checkpoint_stage8_step4000.pt`.

## 0. Preparar shell

Estado: pendiente para cada corrida nueva.

```bash
cd /home/rafael/Documentos/HSL/RL
source .venv/bin/activate
unset PYBH_RL_SHM_NAME

export PYTHONPATH=/home/rafael/Documentos/HSL/SabanaHerons2026/Make/Python:/home/rafael/Documentos/HSL/RL
export LD_LIBRARY_PATH=/home/rafael/Documentos/HSL/SabanaHerons2026/Util/onnxruntime/lib/Linux:$LD_LIBRARY_PATH
export RL_GATE_SHOOT_OPEN_ENTER=0.30
export RL_GATE_SHOOT_OPEN_EXIT=0.20
```

Notas:

- Si vas a correr SimRobot visible desde Codex, debe ser fuera del sandbox.
- El log debe mostrar `shm=/pybh_rl_ma_teacher_stage8_...`.
- Si ves request IDs gigantes tipo `1070137xxx`, estas leyendo shared memory vieja
  o binarios desalineados.

## 1. Verificar binarios y bridge

Estado: hecho una vez en esta sesion, repetir si se recompila algo de
`RLSharedState`, `LocalConsole`, `Module.cpp`, SimRobot o PythonController.

```bash
cd /home/rafael/Documentos/HSL/SabanaHerons2026
env CCACHE_DIR=/tmp/ccache-hsl cmake --build Build/Linux/CMake/Develop --target SimRobot SimulatedNao -j2
env CCACHE_DIR=/tmp/ccache-hsl cmake --build Build/Linux/Python --target PythonController -j2
cp Build/Linux/Python/Release/controller.cpython-312-x86_64-linux-gnu.so Make/Python/pybh/controller.cpython-312-x86_64-linux-gnu.so
```

PASS esperado:

- `pybh.controller` tiene `rl_set_visible_action`.
- `Build/Linux/SimRobot/Develop/libSimulatedNao.so` y `SimRobot` son posteriores
  a cambios del bridge.
- No aparece `AttributeError: module 'pybh.controller' has no attribute ...`.

## 2. Dry-run sin SimRobot

Estado: PASS.

```bash
cd /home/rafael/Documentos/HSL/RL
RL_MA_TEACHER_STAGE=8 RL_MA_TEACHER_DRY_RUN=1 .venv/bin/python teacher_multiagent_dataset.py
```

PASS esperado:

```text
[ma-teacher dry-run] PASS stage=8 scenarios=4 active=(1, 2, 3)
[ma-teacher dry-run] PASS obs_size=47 role_mask_contract=true team_teacher_payload=true ready_for_simrobot_collection=true
```

Si esto falla, no correr SimRobot todavia. El problema esta en teacher, roles,
stage spec, obs contract o escenarios.

## 3. Smoke visible single-SimRobot

Estado: PASS de infraestructura. Con pocos steps puede escribir 0 filas por
filtros, eso es aceptable para este paso.

```bash
cd /home/rafael/Documentos/HSL/RL
unset PYBH_RL_SHM_NAME

PYTHONPATH=/home/rafael/Documentos/HSL/SabanaHerons2026/Make/Python:/home/rafael/Documentos/HSL/RL \
LD_LIBRARY_PATH=/home/rafael/Documentos/HSL/SabanaHerons2026/Util/onnxruntime/lib/Linux:$LD_LIBRARY_PATH \
RL_GATE_SHOOT_OPEN_ENTER=0.30 \
RL_GATE_SHOOT_OPEN_EXIT=0.20 \
RL_MA_TEACHER_STAGE=8 \
RL_MA_TEACHER_EPISODES=1 \
RL_MA_TEACHER_MAX_STEPS=80 \
RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=1 \
python teacher_multiagent_dataset.py
```

PASS de infraestructura:

- aparece `visible_parallel_slots=off shm=/pybh_rl_ma_teacher_stage8_...`;
- no aparece request ID gigante;
- no aparece assert de `Behavior.cpp`;
- no aparece `RL reset could not find teammate robot 1`;
- escribe `summary.json`.

Si `total_rows=0` pero los drops son `open_support_hold_too_short` u
`off_ball_hold_too_short`, el bridge esta bien; son filtros de calidad.

## 4. Smoke de escritura con filtros relajados

Estado: PASS. Evidencia: `dataset_stage8_20260612_211843`, `1611 rows`,
`obs_size=47`.

Este paso no produce dataset final. Solo confirma que el pipeline puede escribir
muestras.

```bash
cd /home/rafael/Documentos/HSL/RL
unset PYBH_RL_SHM_NAME

PYTHONPATH=/home/rafael/Documentos/HSL/SabanaHerons2026/Make/Python:/home/rafael/Documentos/HSL/RL \
LD_LIBRARY_PATH=/home/rafael/Documentos/HSL/SabanaHerons2026/Util/onnxruntime/lib/Linux:$LD_LIBRARY_PATH \
RL_GATE_SHOOT_OPEN_ENTER=0.30 \
RL_GATE_SHOOT_OPEN_EXIT=0.20 \
RL_MA_TEACHER_STAGE=8 \
RL_MA_TEACHER_EPISODES=1 \
RL_MA_TEACHER_MAX_STEPS=80 \
RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=1 \
RL_MA_TEACHER_MIN_OPEN_HOLD_STEPS=0 \
RL_MA_TEACHER_MIN_TRIANGLE_HOLD_STEPS=0 \
RL_MA_TEACHER_MAX_ROLE_SWAPS=999 \
RL_MA_TEACHER_COLLISION_SUSTAIN_STEPS=999 \
python teacher_multiagent_dataset.py
```

PASS esperado:

- `total_rows > 0`;
- `obs_size=47`;
- `positive_weight_gate_label_consistent=true`;
- `positive_weight_role_mask_consistent=true`.

## 5. Dataset defensivo real

Estado: ejecutado una vez, pero no aprobado como final.

Resultado actual:

- dataset: `runs/teacher_multiagent_dataset/dataset_stage8_20260612_212404`
- `total_rows=28078`
- `total_episodes=24`
- `obs_size=47`
- `positive_weight_gate_label_consistent=true`
- `positive_weight_role_mask_consistent=true`
- escenarios con episodios kept:
  - `defense_1v3_static_entry`: 4
  - `defense_2v3_dynamic`: 8
  - `defense_3v3_full`: 12
  - `transition_midfield`: 0

Problemas detectados en ese run:

- `transition_midfield` no conserva episodios;
- `block` existe pero es muy escaso: 67 filas `off_ball_support/block`;
- el audit defensivo marca `defender_block_frac_positive=false`;
- los drops principales son `teammate_collision`, `open_support_hold_too_short`
  y algunos `agent_out_of_bounds`.

Estos problemas fueron corregidos despues con:

- gate defensivo sintetico para `block` cuando el off-ball defender esta cerca
  de cover y hay amenaza frontal;
- `transition_midfield` movido a fase defensiva estable, cerca de medio campo
  pero sin cruzar la zona de histeresis;
- audit defensivo stage 8 midiendo filas `off_ball_support`, no x relativa del
  team ball como si fuera x absoluta.

Smoke posterior: `runs/teacher_multiagent_dataset/dataset_stage8_20260612_214346`
con `2007` filas, `transition_midfield=1/1`, `off_ball_support/block=510`,
`defender_block_frac=0.73`, `audit passed=true`.

Conclusion actual: regenerar dataset real con los mismos comandos chunked. No es
necesario probar 2 SimRobot todavia.

Ejecutar chunked. No usar filtros relajados.

```bash
cd /home/rafael/Documentos/HSL/RL
source .venv/bin/activate
unset PYBH_RL_SHM_NAME

export PYTHONPATH=/home/rafael/Documentos/HSL/SabanaHerons2026/Make/Python:/home/rafael/Documentos/HSL/RL
export LD_LIBRARY_PATH=/home/rafael/Documentos/HSL/SabanaHerons2026/Util/onnxruntime/lib/Linux:$LD_LIBRARY_PATH
export RL_GATE_SHOOT_OPEN_ENTER=0.30
export RL_GATE_SHOOT_OPEN_EXIT=0.20

for i in 1 2 3; do
  unset PYBH_RL_SHM_NAME
  RL_MA_TEACHER_STAGE=8 \
  RL_MA_TEACHER_EPISODES=8 \
  RL_MA_TEACHER_MAX_STEPS=600 \
  RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=6 \
  RL_MA_TEACHER_SEED_BASE=$((950000 + i * 10000)) \
  python teacher_multiagent_dataset.py
done
```

PASS por chunk:

- existe `summary.json`;
- `obs_size=47`;
- `total_rows > 0`;
- `positive_weight_gate_label_consistent=true`;
- `positive_weight_role_mask_consistent=true`;
- `drop_reasons` no dominado por `reset_ball_drift`, request failures o asserts.

Si un chunk queda sin `summary.json`, no cuenta. Borrarlo o ignorarlo.

Para correr solo un escenario:

```bash
RL_MA_TEACHER_STAGE=8 \
RL_MA_TEACHER_SCENARIOS=transition_midfield \
RL_MA_TEACHER_EPISODES=8 \
RL_MA_TEACHER_MAX_STEPS=500 \
RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=6 \
RL_MA_TEACHER_SEED_BASE=990000 \
python teacher_multiagent_dataset.py
```

PASS del filtro:

```text
[ma-teacher] 1 scenarios for stage 8:
  - transition_midfield (...)
```

Nota: el filtro `RL_MA_TEACHER_SCENARIOS` se aplica tanto en dry-run como en la
ruta principal.

## 6. Merge y audit del dataset defensivo

Estado: audit ejecutado sobre `dataset_stage8_20260612_212404`; FAIL por
cobertura, no por contrato. Luego smoke `dataset_stage8_20260612_214346` PASS
con los ajustes de cobertura.

Checks que pasan:

- `obs_size_47`;
- role-mask consistente;
- label consistency;
- teammate/team-ball/goal liveness;
- `striker_dribble_frac_min`;
- rows/episodes minimos.

Checks que fallaban despues de desactivar exigencias ofensivas de shoot/pass:

- `all_scenarios_kept`;
- `defender_block_frac_positive`.

Checks corregidos en smoke posterior:

- `all_scenarios_kept=true`;
- `defender_block_frac_positive=true`;
- `defense_phase_rows_present=true`;
- `passed=true`.

Cuando existan chunks buenos:

```bash
python merge_multiagent_teacher_datasets.py \
  runs/teacher_multiagent_dataset/dataset_stage8_<chunk1> \
  runs/teacher_multiagent_dataset/dataset_stage8_<chunk2> \
  runs/teacher_multiagent_dataset/dataset_stage8_<chunk3> \
  --output-dir runs/teacher_multiagent_dataset/dataset_defense_stage8_v1

python teacher_multiagent_dataset_audit.py \
  runs/teacher_multiagent_dataset/dataset_defense_stage8_v1 \
  --write
```

PASS esperado:

- audit sin bad rows de role-mask;
- `block` presente en muestras defensivas armadas;
- liveness razonable por rol;
- no exigir metricas ofensivas de shoot/pass como si fuera stage 7.

### Bootstrap desde runs existentes

Estado: PASS. Se genero la version final actual:

- `runs/teacher_multiagent_dataset/dataset_defense_stage8_bootstrap_v2`
- `source_count=8`
- `total_rows_raw=122090`
- `total_rows=133193`
- `total_episodes=100`
- `transition_midfield=47222 rows`
- `block=19979 rows`
- `defender_block_frac=0.38`
- audit `passed=true`

Distribucion por escenario:

- `defense_1v3_static_entry`: 26205 rows
- `defense_2v3_dynamic`: 24297 rows
- `defense_3v3_full`: 35469 rows
- `transition_midfield`: 47222 rows

Distribucion por skill:

- `walk`: 80950
- `stand`: 17585
- `dribble`: 14679
- `block`: 19979

Comando:

```bash
python build_stage8_defense_dataset.py \
  --output-dir runs/teacher_multiagent_dataset/dataset_defense_stage8_bootstrap_v2 \
  --min-transition-rows 30000 \
  --min-block-frac 0.15

python teacher_multiagent_dataset_audit.py \
  runs/teacher_multiagent_dataset/dataset_defense_stage8_bootstrap_v2 \
  --min-striker-shoot-armed 0 \
  --min-striker-shoot-frac 0 \
  --min-striker-pass-armed 0 \
  --write
```

Este dataset es sintetico en el sentido de bootstrap/oversampling de filas
existentes; no genera estados nuevos fuera de los runs.

## 7. Mezclar ataque + defensa

Estado: PASS.

El fine-tune no debe olvidar ataque. Mezclar dataset de ataque stage 7 con el
defensivo stage 8.

```bash
python merge_multiagent_teacher_datasets.py \
  runs/teacher_multiagent_dataset/dataset_multiagent_train_20260611_full_v1 \
  runs/teacher_multiagent_dataset/dataset_defense_stage8_bootstrap_v2 \
  --output-dir runs/teacher_multiagent_dataset/dataset_attack_plus_defense_stage8_v1
```

Resultado:

- `gate_config` compatible;
- `obs_size=47`;
- `stage_id=-1`, `stages=[6, 7, 8]`;
- `total_rows=488701`;
- `total_episodes=312`;
- `positive_weight_gate_label_consistent=true`;
- `positive_weight_role_mask_consistent=true`.

Distribucion de skills del dataset mixto:

- `walk`: 255687
- `stand`: 161139
- `dribble`: 36093
- `block`: 19979
- `pass`: 11808
- `shoot`: 3995

## 8. Fine-tune

Estado: PASS inicial.

Run inicial:

- `runs/defender_stage8_v4_defreward_single_dynamic`
- telemetria:
  `runs/defender_stage8_v4_defreward_single_dynamic/telemetry/multiagent_stage8_finetune_train_stage8_20260613_111459`
- dataset:
  `runs/teacher_multiagent_dataset/dataset_attack_plus_defense_stage8_v1/samples.npz`
- warm-start:
  `runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt`
- escena: `RLMAStage7_3D`
- jugadores RL visibles: `RL_MA_PROD_PLAYERS=1`
- steps: `12000`, eval/checkpoint cada `2000`

Checkpoint recomendado para continuar: `checkpoint_stage8_step4000.pt`.
Motivo: fue el unico punto con `wire=PASS`, `collision=0.000`,
`policy_incompat=0.001` y reward medio alto (`0.750`). `step2000` tuvo el mejor
`defender_score` bruto (`0.897`), pero con `collision=0.375`; `step12000`
recupero cobertura (`defender_score=0.852`) pero tambien subio a
`collision=0.500`.

Resumen de evaluacion:

| step | reward | defender_score | goal_side | cover | danger_cover | collision | wire |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 2000 | 0.231 | 0.897 | 1.000 | 0.872 | 0.865 | 0.375 | WARN |
| 4000 | 0.750 | 0.767 | 0.954 | 0.554 | 0.594 | 0.000 | PASS |
| 6000 | 0.490 | 0.822 | 1.000 | 0.716 | 0.730 | 0.375 | WARN |
| 8000 | 0.623 | 0.757 | 0.898 | 0.621 | 0.672 | 0.250 | PASS |
| 10000 | 0.729 | 0.655 | 0.867 | 0.464 | 0.513 | 0.375 | WARN |
| 12000 | 0.082 | 0.852 | 1.000 | 0.800 | 0.812 | 0.500 | WARN |

Comando usado para el run inicial:

```bash
cd /home/rafael/Documentos/HSL/RL
source .venv/bin/activate
unset PYBH_RL_SHM_NAME

export PYTHONPATH=/home/rafael/Documentos/HSL/RL:/home/rafael/Documentos/HSL/SabanaHerons2026/Make/Python
export LD_LIBRARY_PATH=/home/rafael/Documentos/HSL/SabanaHerons2026/Util/onnxruntime/lib/Linux:$LD_LIBRARY_PATH
export RL_SCENE_NAME=RLMAStage7_3D
export RL_MA_PROD_PLAYERS=1
export RL_MA_PROD_SIMROBOT_LAUNCH=1
RL_GATE_SHOOT_OPEN_ENTER=0.30 \
RL_GATE_SHOOT_OPEN_EXIT=0.20 \
RL_MA_PROD_SCENARIO_SET=defense \
RL_MA_PROD_EXPERIMENT_NAME=defender_stage8_v4_defreward_single_dynamic \
RL_MA_STAGE8_DYNAMIC_WORLD_UPDATES=1 \
RL_MA_STAGE8_OPP_SPEED_MM_S=120 \
RL_MA_STAGE8_BALL_SPEED_MM_S=40 \
RL_MA_PROD_TRAIN_STEPS=12000 \
RL_MA_PROD_MAX_EPISODE_STEPS=600 \
RL_MA_PROD_EVAL_INTERVAL=2000 \
RL_MA_PROD_CHECKPOINT_INTERVAL=2000 \
RL_MA_PROD_EVAL_EPISODES=8 \
RL_MA_PROD_BC_REPLAY_WEIGHT=0.15 \
RL_MA_PROD_BC_BETA_MAX=0.12 \
RL_MA_PROD_BC_BETA_MIN=0.04 \
RL_MA_PROD_BC_REPLAY_UPDATES=3 \
python train_multiagent_production.py \
  --stage-id 8 \
  --checkpoint runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt \
  --dataset runs/teacher_multiagent_dataset/dataset_attack_plus_defense_stage8_v1/samples.npz
```

PASS esperado para siguientes corridas:

- guarda checkpoints stage 8;
- no crashea por stage choices;
- mantiene floor de ataque;
- mejora metricas defensivas sin colisiones sistematicas con GK/supports.

Metricas stage 8 relevantes en el log `[eval]`:

- `defender_score`: agregado defensivo, no reward ofensivo heredado.
- `goal_side`: fraccion de fase defensiva en que el robot queda del lado de su
  arco respecto al balon.
- `cover`: fraccion de fase defensiva en el corredor arco-propio -> balon.
- `danger_cover`: cobertura del corredor cuando el balon esta en zona peligrosa.
- `gk_near`: cercania indebida al goalkeeper; debe tender a 0.
- `corridor_mm`: distancia media al corredor arco-propio -> balon; menor es mejor.
- `block` / `pass`: tasas de ejecucion de esas skills.
- `collision`, `stability`, `wire`: sanidad fisica/perceptual.

Para stage 8, `bc_ema` ahora usa `defender_score` como competencia principal en
vez de `clean_finish`, porque `finish/takeover` son metricas ofensivas heredadas.

Reward stage 8:

- Se conserva el reward base heredado para no romper la compatibilidad del
  checkpoint stage 7.
- Encima se agrega shaping defensivo por step: `defense_goal_side`,
  `defense_cover_corridor`, `defense_danger_cover`, `defense_block`,
  `defense_pass_clear`, `defense_threat_reduction`.
- Penaliza `defense_abandon`, `defense_far_corridor`, `defense_gk_near` y
  `defense_collision`.
- Estos terminos aparecen en `reward_terms_mean` y `reward_weighted_terms_mean`
  dentro de la telemetria/eval summary.

Nota operativa: stage 8 arranca con `RL_MA_PROD_PLAYERS=1` y
`RL_MA_PROD_SIMROBOT_LAUNCH=1` para usar un solo SimRobot visible. El modo
completo con tres slots se debe probar despues con `RL_MA_PROD_PLAYERS=1,2,3`,
cuando el fine-tune de un slot ya sea estable.

Mantener `RL_MA_STAGE8_DYNAMIC_WORLD_UPDATES=1`: stage 8 debe seguir siendo
dinamico. Para evitar movimientos raros cuando un oponente cae, el runtime C++
ignora dynamic teleports sobre robots que no estan upright, y el curriculum usa
velocidades conservadoras por defecto (`RL_MA_STAGE8_OPP_SPEED_MM_S=180`,
`RL_MA_STAGE8_BALL_SPEED_MM_S=60`). Si hace falta estresar mas, subir esas
velocidades gradualmente.

## 9. Evaluacion y export

Estado: pendiente sobre el checkpoint recomendado.

Antes de exportar:

- partir de `runs/defender_stage8_v4_defreward_single_dynamic/checkpoint_stage8_step4000.pt`
  salvo que una re-evaluacion multi-seed demuestre otro candidato mejor;
- correr evaluator ofensivo para confirmar que no se rompio ataque;
- correr evaluacion stage 8 defensiva;
- revisar `defender_cover_hold_rate`, `defender_gk_collision_rate`,
  `goals_conceded_rate` si se usa `team_parallel_environment.py`.

Export esperado:

```bash
python tools/export_ppo_policy_team.py \
  --checkpoint runs/<stage8_run>/checkpoint_*.pt \
  --output ppo_team_defender_stage8.onnx \
  --specialization defender
```

PASS esperado:

- ONNX exporta logits/params con contrato obs 47;
- manifest dice especializacion defender;
- no se reemplaza el checkpoint base stage 7.

## Punto actual

Estamos despues del primer fine-tune stage 8:

- dry-run stage 8: PASS;
- SimRobot visible single-process: PASS;
- smoke con filtros relajados: PASS, genero `1611` filas;
- dataset real anterior `dataset_stage8_20260612_212404`: completo, `28078`
  filas, pero FAIL por cobertura defensiva;
- smoke corregido `dataset_stage8_20260612_214346`: PASS de audit defensivo.
- consolidado defensivo bootstrap v2: PASS de audit;
- merge ataque+defensa: PASS.
- fine-tune inicial `defender_stage8_v4_defreward_single_dynamic`: PASS.

Siguiente accion concreta: evaluar/exportar desde
`runs/defender_stage8_v4_defreward_single_dynamic/checkpoint_stage8_step4000.pt`
o continuar fine-tuning desde ese checkpoint con eval multi-seed.
