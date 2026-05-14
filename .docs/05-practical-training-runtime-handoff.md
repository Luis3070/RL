# Contexto práctico de runtime y entrenamiento PPO

Este documento resume lo que acabamos de dejar funcionando para que futuros
agentes de IA y personas del equipo puedan correr smoke tests y entrenamiento
sin tener que reconstruir toda la historia del repo.

Se apoya y complementa:

- [01-ppo-takeover-curriculum-handoff.md](/home/luis/GitHub/RL/.docs/01-ppo-takeover-curriculum-handoff.md)
- [02-instrumentation-ppo-sanity.md](/home/luis/GitHub/RL/.docs/02-instrumentation-ppo-sanity.md)
- [03-go-no-go-training-checklist.md](/home/luis/GitHub/RL/.docs/03-go-no-go-training-checklist.md)
- [04-implementation-changes-stage0-3.md](/home/luis/GitHub/RL/.docs/04-implementation-changes-stage0-3.md)

## Qué quedó decidido

### 1. El reward y el curriculum trabajan sobre observación cruda

Esto ya no es una preferencia, sino una regla operativa.

La fuente de verdad para:

- reward,
- métricas,
- gates,
- telemetría de sanidad,

es `BackendObservation` cruda, no el vector normalizado que consume PPO.

Razón:

- las heurísticas y probes que ya demostraron éxito sobre SimRobot 3D trabajan
  con valores crudos en mm, ms y ángulos reales;
- los thresholds del takeover también están definidos en esas unidades.

### 2. El entrenamiento corre con learner en GPU y env en CPU

Se dejó explícitamente esta separación:

- `SimRobot`, `B-Human`, `pybh`, bridge y stepping del env: CPU
- forward/backward de PPO/MAPPO: GPU (`cuda:0`)

Esto evita mezclar el runtime nativo del simulador con el device del learner.

### 3. Stage 4 sigue fuera del núcleo del sprint

Todo lo que quedó listo y validado está orientado a Stage 0-3.

Stage 4 no bloquea:

- smoke test,
- entrenamiento base,
- curriculum real hasta Stage 3,
- logging de sanidad.

## Qué archivos son ahora el punto de entrada práctico

### [train.py](/home/luis/GitHub/RL/train.py)

Es el entrypoint principal.

Ya no depende de `argparse`.

Se corre directo desde el IDE usando `IDE_RUN`, una `dataclass` con los
parámetros de la corrida.

Campos importantes:

- `mode`: `"smoke"` o `"train"`
- `learner_device`: `"cuda:0"` por defecto si CUDA existe
- `backend`: `"simrobot"`
- `simrobot_profile`: `"3d"`
- `stage_id`
- `max_stage_id`
- `smoke_steps`
- `train_steps`
- `save_telemetry`
- `save_plots`
- `telemetry_tag`

### [envSmoketest.py](/home/luis/GitHub/RL/envSmoketest.py)

Ya no es una prueba de CartPole.

Ahora es un smoke test real del stack:

- `train.main(...)`
- `mode="smoke"`
- SimRobot 3D
- learner en GPU
- env en CPU
- telemetría persistente

Esto permite darle Run directo en el IDE y verificar rápido que:

- el backend arranca,
- el env hace reset/step,
- el policy produce acciones,
- la telemetría cae a disco.

### [wrapper.py](/home/luis/GitHub/RL/wrapper.py)

Se volvió el punto práctico para recolectar episodios con telemetría legible.

Ahora `collect_episode(...)` puede devolver:

- reward acumulado,
- `reward_terms_sum`,
- `reward_weighted_terms_sum`,
- `episode_metrics`,
- `final_info`,
- y `step_records` si `capture_steps=True`.

Esto se usa en evaluación y en persistencia de métricas.

## Requisitos de entorno

El entorno canónico de trabajo es:

- `conda` env `RLNAO`

El stack quedó verificado sobre:

- Python 3.12
- `torch`
- `skrl`
- `gymnasium`
- `pandas`
- `matplotlib`

Además deben existir estas variables de entorno:

```bash
export LD_LIBRARY_PATH=/home/luis/GitHub/SabanaHerons2026/Util/onnxruntime/lib/Linux
export PYTHONPATH=/home/luis/GitHub/SabanaHerons2026/Make/Python:/home/luis/GitHub/RL
```

Sin eso, el binding `pybh` o los imports del repo pueden fallar.

## Runtime de SimRobot / pybh que ya quedó saneado

### 1. `pybh_backend.py` quedó como base funcional del runtime

Se corrigieron y estabilizaron varios puntos:

- arranque embebido de SimRobot 3D,
- override temporal de `settings.cfg` a `team=24`, `player=1`,
- restauración limpia al cerrar,
- soporte efectivo de:
  - `rl_reset_world_extended`
  - `rl_get_world_status`
  - `rl_set_dynamic_world`

### 2. Hubo que recompilar componentes nativos

Parte del problema anterior era que el binding Python y las librerías nativas
no estaban alineadas entre sí.

Se recompilaron:

- `PythonController`
- `SimulatedNao`
- `controller.so` de `Make/Python/pybh`

La lección práctica:

- si el stack vuelve a comportarse raro tras cambios en C++ o bridge,
  sospecha primero de binarios desalineados.

## Cómo corre ahora el smoke

El smoke validado de verdad es:

1. Run de [envSmoketest.py](/home/luis/GitHub/RL/envSmoketest.py)
2. `train.main(SMOKE_RUN)`
3. backend `simrobot`
4. perfil `3d`
5. env en CPU
6. learner en GPU

Se verificó que:

- el env resetea,
- el step corre,
- la policy produce skill,
- el info trae telemetría,
- y se escriben CSV/PNG.

## Cómo corre ahora el train

El camino validado es:

1. `train.main(IDE_RUN)` en modo `"train"`
2. rollout real con `skrl`/`MAPPO`
3. learner en GPU
4. env en CPU
5. evaluación al final del mini-run
6. checkpoint persistido

Se corrigió un bug importante de `skrl`:

- `write_interval="auto"` terminaba rompiendo `record_transition`
- se fijó un valor numérico seguro en la configuración del agente

## Separación GPU / CPU

Esta fue una parte importante del trabajo y conviene conservarla así.

### CPU

- `RealSimRobotEnv`
- `PyBHBackend`
- SimRobot
- B-Human
- `wrapped_env.step(...)`
- tensores que entran/salen del env

### GPU

- `PolicyNetwork`
- `ValueNetwork`
- `agent.act(...)`
- almacenamiento y updates de PPO/MAPPO

### Regla práctica

No mover el env entero a CUDA.

Lo correcto es:

- convertir observaciones/estados a GPU justo antes de `agent.act(...)`
- devolver acciones a CPU justo antes de `wrapped_env.step(...)`
- registrar transiciones en GPU para el learner

Eso ya quedó cableado en [train.py](/home/luis/GitHub/RL/train.py).

## Telemetría persistente que ya queda a disco

Cada corrida crea una sesión de telemetría dentro de:

```text
RL/runs/pybh_mappo_takeover_ppo/telemetry/<tag>_<mode>_stage<id>_<timestamp>/
```

### Archivos comunes

- `run_config.json`

### Smoke

- `csv/smoke_steps.csv`
- `plots/smoke_overview.png`

### Train / Eval

- `csv/eval_summary.csv`
- `csv/eval_episodes.csv`
- `csv/eval_episode_steps.csv`
- `plots/eval_metrics_overview.png`
- `plots/eval_reward_terms.png`

## Qué contiene cada archivo de telemetría

### `smoke_steps.csv`

Por step:

- reward
- `policy_skill_name`
- `executed_skill_name`
- `preferred_skill_name`
- `projection_applied`
- zonas (`walk/dribble/shoot/takeover`)
- `bridge_frame`
- distancia y ángulo al balón
- frescura de visión
- apertura de tiro
- obstáculo/oponente frontal
- reward terms
- weighted reward terms

### `eval_summary.csv`

Por evaluación:

- timestep
- stage
- `episode_reward_mean`
- `clean_takeover_rate`
- `clean_finish_rate`
- `thrash_rate`
- `collision_rate`
- `stability_score`
- reward term means
- losses/entropy si ya existen

Nota:

- en mini-runs muy cortos puede salir `NaN` en `policy_loss`, `value_loss`,
  `entropy`, simplemente porque aún no hubo update real.

### `eval_episodes.csv`

Por episodio de evaluación:

- steps
- mean reward
- `episode_metrics.*`
- sumas de reward terms
- sumas de weighted reward terms

### `eval_episode_steps.csv`

Por step dentro de evaluación:

- skill pedida / ejecutada
- zonas
- proyección
- frame del bridge
- señales crudas de percepción/contexto

## Qué mirar primero si algo sale mal

### Caso 1: el smoke ya no corre

Revisar en este orden:

1. `RLNAO` realmente activo
2. `PYTHONPATH`
3. `LD_LIBRARY_PATH`
4. binarios nativos alineados
5. [pybh_backend.py](/home/luis/GitHub/RL/pybh_backend.py)
6. Run de [envSmoketest.py](/home/luis/GitHub/RL/envSmoketest.py)

### Caso 2: corre pero no guarda telemetría

Revisar:

1. `save_telemetry=True`
2. `save_plots=True`
3. que exista la carpeta `runs/.../telemetry/...`
4. permisos de escritura en el repo
5. si `pandas` y `matplotlib` están instalados

### Caso 3: train corre pero no aprende nada

Mirar:

1. `eval_summary.csv`
2. `clean_takeover_rate`
3. `clean_finish_rate`
4. `thrash_rate`
5. `reward_terms_mean.*`
6. si el Stage 0 realmente está generando resets razonables

### Caso 4: los losses salen `NaN`

Primero distinguir:

- `NaN` en mini smoke/train muy corto: puede ser normal
- `NaN` después de varios rollouts completos: ahí sí investigar

Primero revisar:

1. si ya ocurrió un update real
2. `rollout_steps`
3. `eval_summary.csv`
4. si las recompensas están desbocadas

## Cómo se espera que alguien nuevo use esto

### Para verificar que todo respira

Abrir y correr:

- [envSmoketest.py](/home/luis/GitHub/RL/envSmoketest.py)

Luego inspeccionar:

- `smoke_steps.csv`
- `smoke_overview.png`

### Para entrenar

Abrir y correr:

- [train.py](/home/luis/GitHub/RL/train.py)

Modificar solo `IDE_RUN` si hace falta:

- `mode`
- `train_steps`
- `stage_id`
- `max_stage_id`
- `telemetry_tag`

### Para análisis posterior

Ir a la última carpeta de:

```text
RL/runs/pybh_mappo_takeover_ppo/telemetry/
```

y mirar primero:

- `run_config.json`
- `eval_summary.csv`
- `eval_metrics_overview.png`

## Riesgos abiertos

1. Stage 4 sigue fuera del núcleo y no debe mezclarse todavía con el pipeline
   base.
2. El smoke puede correr perfecto aunque la policy todavía no aprenda nada útil;
   por eso la telemetría es obligatoria.
3. Si se tocan librerías C++/binding, puede reaparecer desalineación binaria.
4. Si alguien vuelve a acoplar env y learner al mismo device, es fácil romper la
   estabilidad del stack.

## Resumen corto

El repo quedó en un estado mucho más práctico para entrenamiento:

- Run directo desde IDE
- sin `argparse`
- SimRobot 3D real
- reward sobre crudo
- learner en GPU, env en CPU
- smoke y mini-train verificados
- telemetría persistente en CSV/PNG

Ese es el baseline operativo a preservar.
