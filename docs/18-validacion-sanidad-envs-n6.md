# Validacion De Sanidad De Environments N=6

## Objetivo

Dejar evidencia reproducible de que el pipeline `simrobot_visible` con `parallel_player_numbers=(1,2,3,4,5,6)` es sano en cuatro frentes:

- los 6 slots avanzan de forma independiente
- la pose no esta siendo sintetica de forma estructural
- la semantica del balon sigue siendo util para entrenar
- el observation space mantiene contrato y diversidad inter-slot

El objetivo no es solo ver que "arranca", sino producir datos exportables a documentacion y comparables entre corridas.

## Artefactos Que Deben Quedar

Cada validacion debe dejar como minimo:

- un directorio de `diag_n6_slots.py` en `runs/diag_n6_slots/...`
- un directorio de `visible_probe_controlled.py` en `runs/visible_probe_controlled/...`
- una corrida corta de `train.py` o smoke equivalente con metricas de wire audit
- una tabla resumen en la documentacion con fecha, commit y resultado `PASS/WARN/FAIL`

## Mapa De Scripts

Para evitar confusion, estos son los scripts que hoy se usan y su responsabilidad:

- `diag_n6_slots.py`
  Diagnostico principal de homogeneizacion N=6. Es el source of truth para
  sanidad por slot, features congeladas, diversidad inter-slot y rates de
  pose/balon.
- `probe_protocol_experiment.py --case visible_search`
  Probe controlado del balon usado en el informe actual. Es el source of truth
  para la semantica visible del balon cuando queremos una corrida manual y
  comparable.
- `visible_probe_controlled.py`
  Variante mas directa del probe visible. Sirve para exploracion local rapida,
  pero el informe de `2026-05-15` se cerro con `probe_protocol_experiment.py`.
- `train.py` en modo smoke
  Verifica el wire audit dentro del loop real de entrenamiento.
- `action_sweep.py`
  Auditoria del action space: clamp, masking, proyeccion y `requested` vs
  `executed`.
- `obs_invariants.py`
  Auditoria del obs space: shape, monotonia de frame, consistencia geometrica y
  consistencia semantica de flags del balon.
- `scripts/setup_validation_env.sh`
  Crea la `.venv` persistente e instala dependencias.
- `scripts/run_with_validation_env.sh`
  Wrapper obligatorio para correr probes con la `.venv`, `PYTHONPATH` de `pybh`
  y `Xvfb` del sistema.

## Protocolo Recomendado

### 1. Sanidad Basal De Los 6 Slots

Comando:

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 diag_n6_slots.py
```

Este script ya guarda artefactos en:

```text
runs/diag_n6_slots/diag_YYYYMMDD_HHMMSS/
```

Archivos esperados:

- `summary.json`
- `per_slot_summary.csv`
- `feature_stats.csv`
- `action_distribution.csv`

Que valida:

- `bridge_obs_ready` por slot
- `synthetic_pose_observation` por slot
- `synthetic_ball_observation` por slot
- `ball_export_fresh` y `corrected_exported_ball`
- `pose_alignment_warning`
- `ball_observation_stale`
- `ball_observation_collapsed`
- medias y desviaciones de las 23 features
- diversidad inter-slot
- avance de `frame` por slot

Nota:

- `diag_n6_slots.py` puntua la ventana estable despues de un warmup inicial
  alineado con `pybh_backend.py`
- eso evita castigar como fallo las correcciones transitorias permitidas en los
  primeros steps post-reset

### 2. Semantica Controlada Del Balon

Script recomendado para reporte reproducible:

- `probe_protocol_experiment.py --case visible_search`

Script alterno para exploracion local:

- `visible_probe_controlled.py`

Comando:

```bash
./scripts/run_with_validation_env.sh python3 probe_protocol_experiment.py --case visible_search
```

Este flujo guarda artefactos en:

```text
runs/probe_protocol_experiment/... o runs/probe_protocol_experiment/manual/...
```

Archivos esperados:

- `visible_search_steps.csv` o `steps.csv`
- `visible_search_summary.csv` o `summary.csv`

Que valida:

- tiempo hasta que el balon exportado vuelve a ser confiable despues del reset
- tiempo hasta que el balon natural vuelve a ser confiable
- `synthetic_ball_rate`
- `ball_stale_rate`
- `ball_export_fresh_rate`
- consistencia entre balon natural y balon exportado

### 3. Smoke Del Wire Audit En Flujo De Entrenamiento

Comando sugerido:

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
.venv/bin/python train.py --backend simrobot_visible --mode smoke --smoke-steps 4
```

Si el wrapper de entrenamiento cambia, el requisito sigue siendo el mismo: debe existir una corrida corta donde se vean metricas agregadas del wire audit y un estado final `PASS`, o en su defecto `WARN` con causa explicita.

Que valida:

- que la sanidad no solo existe en un script aislado, sino tambien dentro del loop real de entrenamiento
- que los indicadores agregados del balon se mantienen dentro de umbrales razonables

### 4. Barrido Del Action Space

Comando:

```bash
./scripts/run_with_validation_env.sh python3 action_sweep.py
```

Artefactos esperados:

- `runs/action_sweep/action_sweep_YYYYMMDD_HHMMSS/summary.json`
- `runs/action_sweep/action_sweep_YYYYMMDD_HHMMSS/action_sweep_cases.csv`
- `runs/action_sweep/action_sweep_YYYYMMDD_HHMMSS/action_sweep_steps.csv`

Que valida:

- `requested_skill_name` vs `executed_skill_name`
- `projection_applied`
- clamp de `target_x`, `target_y`, `target_theta` y `pass_target`
- skills enmascaradas que caen a `walk`
- casos rechazados antes del runtime por `skill_type` invalido
- efecto observable sobre pose tras algunos steps

### 5. Invariantes Del Obs Space

Comando:

```bash
./scripts/run_with_validation_env.sh python3 obs_invariants.py
```

Artefactos esperados:

- `runs/obs_invariants/obs_invariants_YYYYMMDD_HHMMSS/summary.json`
- `runs/obs_invariants/obs_invariants_YYYYMMDD_HHMMSS/obs_invariants_steps.csv`

Que valida:

- shape y finitud del vector de observacion
- monotonia de `bridge_frame`
- consistencia geometrica entre `robot_*`, `ball_rel_*` y `ball_*`
- consistencia entre `corrected_exported_ball`, `ball_export_fresh` y `ball_export_source`
- consistencia de `ball_visible_mask` y `ball_trusted_mask`
- rangos normalizados de features

## Criterios De Aceptacion

## 1. Homogeneizacion De Los 6 Envs

Aqui "homogeneizacion" no significa que los 6 envs den exactamente la misma observacion. Significa que:

- todos los slots responden
- todos usan el mismo contrato
- ninguno queda congelado, desfasado o sin progreso
- la diversidad entre slots proviene de dinamica/percepcion, no de bugs de wiring

Criterios:

- `obs_ready_rate >= 0.99` en los 6 slots
- `frame_delta_ms > 0` en los 6 slots
- no debe haber `NaN` en `feature_stats.csv`
- no debe haber features inesperadamente congeladas en todos los slots
- para las features de pose/balon, `inter_slot_std` no debe colapsar a cero

Interpretacion de timers:

- `time_since_ball_seen` puede quedarse casi plano cerca de `0` si la pelota se
  refresca continuamente
- `time_since_ball_disappeared` puede quedarse en `0` si no hubo episodios de
  desaparicion o stale ball durante la ventana puntuada

Evidencia a copiar al informe:

- tabla de `per_slot_summary.csv`
- bloque de `inter_slot_std` de `feature_stats.csv`

## 2. Pose

La pose visible debe venir del pipeline natural, no de una correccion estructural.

Criterios:

- `synthetic_pose_rate <= 0.02` por slot
- `pose_alignment_warning_rate <= 0.10` por slot
- `corrected_exported_robot_pose_rate` debe ser bajo o nulo en visible sano

Evidencia:

- columnas `synthetic_pose_rate`, `pose_alignment_warning_rate` y `corrected_exported_robot_pose_rate` de `per_slot_summary.csv`

Interpretacion:

- si `synthetic_pose_rate` sube de forma persistente, no hay que seguir entrenando hasta entender por que
- si la pose parece "estable" pero depende de correccion, el learner puede aprender sobre una localizacion artificial

## 3. Balon

El balon es la parte mas sensible porque puede verse sano numéricamente aun cuando semantica y perceptualmente ya no sirve.

Criterios minimos:

- `ball_export_fresh_rate >= 0.10` por slot
- `ball_collapsed_rate <= 0.10` por slot
- `ball_visible_rate >= 0.90` en smoke agregado
- `ball_rescued_rate <= 0.75` para no caer en `FAIL`
- `misleading_seen_synthetic_rate <= 0.75` para no caer en `FAIL`

Criterios recomendados:

- `ball_trusted_rate >= 0.55`
- `ball_rescued_rate <= 0.40`
- `ball_confidence_score_mean >= 0.50`

Evidencia:

- `visible_search_summary.csv` de `probe_protocol_experiment.py`
- metricas de wire audit del smoke
- en caso de duda, tambien `steps.csv` para mostrar el tiempo de recuperacion despues del reset

Interpretacion:

- `trusted` alto: el balon es util y natural buena parte del rollout
- `rescued` alto: el bridge esta sosteniendo demasiado la observacion
- `misleading_seen_synthetic` alto: el agente podria creer que ve una pelota real cuando no es asi

## 4. Observation Space

El contrato actual es de 23 features. La validacion debe responder dos preguntas:

- el shape y los indices siguen siendo los esperados
- las features criticas realmente varian

Criterios:

- `obs_shape == (23,)`
- las features ligadas a pose y balon no deben quedar congeladas en los 6 slots
- `ball_seen_percentage` y `time_since_ball_seen` deben variar

Evidencia:

- `feature_stats.csv`
- si hace falta, una pequeña tabla derivada con estas features:
  - `robot_x`
  - `robot_y`
  - `robot_theta`
  - `ball_rel_x`
  - `ball_rel_y`
  - `time_since_ball_seen`
  - `ball_seen_percentage`

## Estructura Del Informe En Documentacion

La recomendacion es dejar una seccion por corrida validada con este formato:

```text
Fecha:
Commit RL:
Commit SabanaHerons2026:
Backend:
Perfil:
Slots:
Resultado global:
```

Luego cuatro subsecciones cortas:

### 1. Homogeneizacion

- resumen de `obs_ready_rate`
- resumen de `frame_delta_ms`
- conclusiones sobre diversidad inter-slot

### 2. Pose

- `synthetic_pose_rate`
- `pose_alignment_warning_rate`
- conclusion

### 3. Balon

- `ball_export_fresh_rate`
- `ball_trusted_rate`
- `ball_rescued_rate`
- `misleading_seen_synthetic_rate`
- `ball_confidence_score_mean`
- conclusion

### 4. Obs Space

- shape
- features congeladas inesperadas
- conclusion

Cerrar con:

```text
Decision:
- PASS: apto para entrenamiento largo
- WARN: usable, pero dejar explicitado el riesgo
- FAIL: no usar para entrenamiento hasta corregir
```

## Tabla Minima Que Si Debe Quedar En La Doc

| player | obs_ready | synthetic_pose | synthetic_ball | ball_fresh | pose_warn | ball_stale | ball_collapsed | frame_delta_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p1 | ... | ... | ... | ... | ... | ... | ... | ... |
| p2 | ... | ... | ... | ... | ... | ... | ... | ... |
| p3 | ... | ... | ... | ... | ... | ... | ... | ... |
| p4 | ... | ... | ... | ... | ... | ... | ... | ... |
| p5 | ... | ... | ... | ... | ... | ... | ... | ... |
| p6 | ... | ... | ... | ... | ... | ... | ... | ... |

Y una tabla agregada del balon:

| metrica | valor | criterio | estado |
|---|---:|---:|---|
| ball_trusted_rate | ... | >= 0.55 | PASS/WARN |
| ball_rescued_rate | ... | <= 0.40 recomendado / <= 0.75 maximo | PASS/WARN/FAIL |
| misleading_seen_synthetic_rate | ... | <= 0.40 recomendado / <= 0.75 maximo | PASS/WARN/FAIL |
| ball_confidence_score_mean | ... | >= 0.50 | PASS/WARN |

## Decision Practica

Si solo puedes correr una validacion completa antes de documentar, la prioridad es:

1. `diag_n6_slots.py`
2. `visible_probe_controlled.py`
3. smoke corto de `train.py`

Con esos tres artefactos ya puedes defender con datos:

- que los 6 envs no estan duplicados ni congelados
- que la pose visible no depende de sintetico
- que el balon sigue siendo suficientemente natural para entrenar
- que el contrato del obs space sigue vivo bajo N=6
