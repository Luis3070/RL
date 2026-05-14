# Hallazgos de entrenamiento Stage 0 y problemas a atacar

Este documento resume el estado práctico del entrenamiento PPO Stage 0 después
de las corridas de sanidad recientes. La idea es dejar una bitácora útil para:

- continuar mañana sin perder contexto,
- explicar qué sí está funcionando,
- separar bugs de instrumentación de problemas de aprendizaje,
- y priorizar qué atacar primero.

Se apoya en:

- [01-ppo-takeover-curriculum-handoff.md](/home/luis/GitHub/RL/.docs/01-ppo-takeover-curriculum-handoff.md)
- [02-instrumentation-ppo-sanity.md](/home/luis/GitHub/RL/.docs/02-instrumentation-ppo-sanity.md)
- [04-implementation-changes-stage0-3.md](/home/luis/GitHub/RL/.docs/04-implementation-changes-stage0-3.md)
- [05-practical-training-runtime-handoff.md](/home/luis/GitHub/RL/.docs/05-practical-training-runtime-handoff.md)

## Corridas de referencia

### Sanidad train Stage 0

Ruta:

- [sanity_stage0_ide_train_stage0_20260514_001229](</home/luis/GitHub/RL/runs/pybh_mappo_takeover_ppo/telemetry/sanity_stage0_ide_train_stage0_20260514_001229>)

Archivos más importantes:

- [run_config.json](</home/luis/GitHub/RL/runs/pybh_mappo_takeover_ppo/telemetry/sanity_stage0_ide_train_stage0_20260514_001229/run_config.json>)
- [eval_summary.csv](</home/luis/GitHub/RL/runs/pybh_mappo_takeover_ppo/telemetry/sanity_stage0_ide_train_stage0_20260514_001229/csv/eval_summary.csv>)
- [eval_episodes.csv](</home/luis/GitHub/RL/runs/pybh_mappo_takeover_ppo/telemetry/sanity_stage0_ide_train_stage0_20260514_001229/csv/eval_episodes.csv>)
- [eval_episode_steps.csv](</home/luis/GitHub/RL/runs/pybh_mappo_takeover_ppo/telemetry/sanity_stage0_ide_train_stage0_20260514_001229/csv/eval_episode_steps.csv>)

### Smoke con telemetría sintética explícita

Ruta:

- [synthetic_cols_check_smoke_stage0_20260514_000419](</home/luis/GitHub/RL/runs/pybh_mappo_takeover_ppo/telemetry/synthetic_cols_check_smoke_stage0_20260514_000419>)

Archivo útil:

- [smoke_steps.csv](</home/luis/GitHub/RL/runs/pybh_mappo_takeover_ppo/telemetry/synthetic_cols_check_smoke_stage0_20260514_000419/csv/smoke_steps.csv>)

## Logros reales

### 1. El stack completo ya corre

Ya quedó validado que:

- `train.py` arranca,
- `MAPPO/PPO` se inicializa,
- el env interactúa con SimRobot 3D,
- se generan episodios,
- se corre evaluación periódica,
- se escriben CSV y PNG,
- y se guardan checkpoints.

Esto no era trivial y ya está resuelto.

### 2. El reward ya no está ciego

La corrida de Stage 0 muestra que:

- `episode_reward_mean` deja de ser puramente negativo,
- aparece `takeover_success_rate = 0.5`,
- la policy entra a `walk_zone` y `dribble_zone`,
- y se acumulan términos útiles de `good_walk`, `hold_takeover` y `good_dribble`.

Eso significa que:

- el `obs` crudo ya alimenta reward,
- el shaping sí distingue contexto,
- y el pipeline no está muerto.

### 3. La telemetría de sanidad quedó bastante más rica

Ya se guardan de forma explícita:

- `projection_applied`
- `synthetic_pose_observation`
- `synthetic_ball_observation`
- zonas `walk/dribble/shoot`
- reward por step
- reward terms agregados por episodio
- métricas agregadas por eval

Esto ya permite analizar no solo “cuánto reward hubo”, sino por qué.

### 4. Se localizó el bug real detrás de losses/entropy vacías

La causa encontrada fue:

- `skrl` escribe `tracking_data`
- luego limpia `tracking_data`
- y el código del proyecto estaba leyendo losses después de esa limpieza

Eso deja `policy_loss`, `value_loss` y `entropy` vacíos aunque haya habido update.

La corrección aplicada en [train.py](/home/luis/GitHub/RL/train.py):

- snapshot persistente de `tracking_data` antes de que `skrl` la vacíe
- extracción desde ese snapshot cuando el dict vivo ya está vacío

Importante:

- la corrida `20260514_001229` todavía salió con columnas vacías,
- porque ocurrió antes de que el arreglo del snapshot quedara listo.

## Problemas encontrados

## 1. `policy_loss`, `value_loss`, `entropy` salían vacías

Síntoma:

- en consola: `policy_loss=nan value_loss=nan entropy=nan`
- en `eval_summary.csv`: columnas vacías

Causa:

- bug de instrumentación, no necesariamente colapso numérico del optimizador

Estado:

- ya identificado
- ya parchado en [train.py](/home/luis/GitHub/RL/train.py)
- falta revalidación en una nueva corrida

Impacto:

- sin estas métricas, el análisis de estabilidad PPO queda cojo

Prioridad:

- alta

## 2. La observación sintética está sosteniendo el 100% del episodio

Señal observada en `eval_summary.csv`:

- `synthetic_pose_rate = 1.0`
- `synthetic_ball_rate = 1.0`

Señal observada en `eval_episode_steps.csv`:

- `synthetic_pose_observation = 1` en todos los steps
- `synthetic_ball_observation = 1` en todos los steps

Interpretación:

- el entrenamiento actual no está usando observación real estabilizada
- está entrenando completamente apoyado en la corrección sintética del backend

Impacto:

- la corrida sirve para validar integración y reward
- pero todavía no sirve como evidencia de aprendizaje robusto de producción

Prioridad:

- muy alta

## 3. La policy cruda sigue demasiado dependiente de `projection`

Resumen agregado de `eval_episode_steps.csv`:

- `projection_applied mean = 0.55234375`

Y por eval:

- step 256: `0.371875`
- step 512: `0.53125`
- step 768: `0.59375`
- step 1024: `0.7125`

Interpretación:

- la policy cruda no está decidiendo limpio
- el env corrige demasiadas acciones vía `preferred_skill`
- la dependencia de proyección incluso aumenta con el tiempo

Impacto:

- el reward puede subir sin que la cabeza de skill haya aprendido bien

Prioridad:

- muy alta

## 4. La policy no está emitiendo `dribble` de forma nativa

Conteo agregado de `policy_skill_name`:

- `stand`: 705
- `walk`: 514
- `shoot`: 59
- `dribble`: 0

Mientras que `executed_skill_name` sí muestra:

- `dribble`: 322

Interpretación:

- el `dribble` útil que aparece viene del env/proyección
- no de la cabeza cruda del actor

Impacto:

- la policy todavía no aprendió selección contextual real

Prioridad:

- alta

## 5. `shoot` sigue siendo casi puro ruido

Señales:

- `shoot_zone = 0.0`
- `shoot_in_shoot_zone_rate = 0.0`
- `good_shoot = 0`
- `bad_shoot_rate = 1.0`

Interpretación:

- `shoot` no está cerrado aún como conducta útil
- la Stage 0 actual parece aprender approach/takeover parcial, pero no cierre

Impacto:

- `clean_finish_rate` se queda en `0.0`

Prioridad:

- media-alta

## 6. `clean_*` sigue en cero a pesar de takeover parcial

En `eval_summary.csv`:

- `clean_walk_rate = 0.0`
- `clean_takeover_rate = 0.0`
- `clean_finish_rate = 0.0`

Pero a la vez:

- `takeover_success_rate = 0.5`

Interpretación:

- el agente logra comportamiento útil parcial
- pero no suficientemente limpio/estable para cumplir las definiciones operativas

Impacto:

- todavía no hay base para promoción ni para hablar de Stage 0 resuelta

Prioridad:

- alta

## Lo que nos dice la sanidad 001229

La corrida muestra un patrón bastante consistente:

- un episodio de eval no entra en takeover y se queda casi todo en `walk`
- el otro entra muy rápido a takeover y luego deriva a `dribble`
- el reward agregado sube con el tiempo
- pero el aprendizaje limpio no aparece todavía

Eso significa que:

- el reward ya tiene señal,
- el env ya produce comportamiento diferenciable,
- pero la policy todavía no internaliza la decisión correcta;
- la ejecución útil sigue apoyada por:
  - proyección,
  - y observación sintética persistente.

## Ataques recomendados

## Ataque 1. Revalidar losses/entropy inmediatamente

Objetivo:

- confirmar que el snapshot nuevo en [train.py](/home/luis/GitHub/RL/train.py)
  llena por fin:
  - `policy_loss`
  - `value_loss`
  - `entropy`

Acción:

- correr una sanity corta nueva
- revisar `eval_summary.csv`

Condición de éxito:

- columnas llenas con números finitos

## Ataque 2. Limitar la observación sintética a warmup/reset

Objetivo:

- que `synthetic_pose_rate` y `synthetic_ball_rate` dejen de ser `1.0`

Acción:

- recortar el uso de `synthetic_*` a una ventana corta post-reset
- o condicionar la estabilización a un número máximo de frames

Condición de éxito:

- `synthetic_*_rate` bajando de forma clara corrida a corrida

## Ataque 3. Reducir la dependencia de proyección

Objetivo:

- bajar `skill_projection_rate`

Acción sugerida:

- mantener el sesgo inicial a `walk`
- añadir penalización suave por `projection_applied`
- seguir observando si la policy deja de colapsar tanto a `stand`

Condición de éxito:

- `policy_skill_name=stand` baja
- `projection_applied` baja
- `executed_skill_name` se parece más a `policy_skill_name`

## Ataque 4. Mantener Stage 0 hasta limpiar takeover

No promover nada todavía.

Razón:

- `clean_takeover_rate` y `clean_finish_rate` siguen en cero
- no hay cierre de `shoot`
- la política todavía no es confiable por sí misma

## Resumen ejecutivo

Estado actual:

- integración: bastante bien
- runtime básico: funcional, aunque aún delicado en partes nativas
- reward: vivo
- curriculum Stage 0: corre
- telemetría: bastante útil
- PPO losses/entropy: bug identificado y parchado, falta revalidar
- aprendizaje limpio: todavía no

Juicio práctico:

- no estamos en “producción”
- sí estamos en una fase muy útil de depuración con comportamiento parcial real

La prioridad correcta no es abrir más stages.

La prioridad correcta es:

1. recuperar losses/entropy de forma verificable,
2. bajar `synthetic_*_rate`,
3. bajar `skill_projection_rate`,
4. recién después volver a juzgar la calidad del aprendizaje Stage 0.
