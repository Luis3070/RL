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
- preflight en subprocess para elegir `QT_QPA_PLATFORM` antes de cargar escena
  `3d` en el proceso principal,
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

### 3. Fix del wire de observación RL <-> SimRobot

Después de revalidar `simrobot` embebido y `simrobot_visible`, apareció un
problema más fino:

- `rl_set_action(...)` y `rl_get_obs(...)` sí funcionaban,
- pero `robot_x/y/theta` y `ball_x/y` exportados a Python podían quedar muy
  desalineados respecto al estado real del simulador después de un reset RL.

La causa práctica fue esta:

- `simrobot_robot_x/y/theta` salían del cuerpo real en SimRobot desde
  `LocalConsole`,
- mientras que `robot_x/y/theta` y `ball_x/y` salían de `theRobotPose` y
  `theFieldBall` en `SkillBehaviorControl`,
- y esas representaciones podían quedar stale después del teleport/reset RL.

Síntoma observado antes del fix:

- error de pose de ~`2.8 m` entre `robot_*` y `simrobot_robot_*`,
- `ball_seen_percentage = 0`,
- y colapso de `ball_x` al robot cuando se apagaba la corrección sintética
  del backend Python.

Fix aplicado en `SabanaHerons2026`:

- en el export de `SkillBehaviorControl` hacia `RLSharedState`,
- si la pose estimada de B-Human se desvía demasiado de `simRobotX/Y`, se
  exporta la pose real del simulador;
- si la pelota está stale, se exporta la pelota del último reset/dynamic world
  RL y se recalculan sus coordenadas relativas.
- además, el modo quedó explícito vía `PYBH_RL_OBS_MODE`:
  - `corrected` (default): aplica estas correcciones,
  - `estimated`: exporta la observación estimada de B-Human sin corregir.

Resultado de las sondas cortas:

- en `simrobot_visible`, `pose_err_mm` bajó de ~`2800` a `0`,
- en `simrobot` embebido, `pose_err_mm` quedó en el orden de `0-2 mm`,
- y `synthetic_pose_observation` / `synthetic_ball_observation` quedaron en `0`
  en los probes dirigidos del backend.

### 4. Impacto sobre entrenamiento

Este fix mejora el wire, pero no es neutral para aprendizaje.

Con esta exportación:

- RL deja de ver la estimación rota/stale de B-Human,
- y pasa a ver una observación corregida que está más cerca del estado real del
  simulador.

Eso significa:

- es bueno para depurar bridge, reward y reset,
- pero reduce realismo perceptual si se usa como observación final de training.

Interpretación operativa:

- para smoke tests, validación del wire y depuración Stage 0, el fix es útil;
- para entrenamiento orientado a transferencia, no conviene olvidar que esta
  observación ya no representa puramente “lo que cree B-Human”.

Recomendación:

- tratar este modo como `oracle/corrected observation`,
- y más adelante separar explícitamente:
  - observación estimada de B-Human,
  - observación real del simulador,
  - observación exportada a RL.
- la telemetría ahora también expone:
  - `obs_export_mode`,
  - `corrected_exported_robot_pose`,
  - `corrected_exported_ball`,
  - y las variantes `estimated_*` de pose/ball para auditoría.

### 5. Qué falta validar del wire

Después del fix, todavía no conviene dar el wire por cerrado. Falta validar:

- smoke y sanity completos de `train.py` en un entorno con `torch`, `skrl`,
  `pettingzoo` y `matplotlib`, para confirmar que `synthetic_*_rate` cae
  también en el pipeline completo y no solo en probes directos;
- que `pose_alignment_warning_rate`, `ball_stale_rate` y
  `ball_collapsed_rate` se mantengan bajos en episodios largos, no solo en
  resets cortos;
- que resets y dynamic world updates de Stage 2-3 no vuelvan a desalinear pose
  u observación;
- que `collision` siga siendo razonable después del cambio, porque parte de la
  observación consumida por reward ahora está más cerca de ground truth;
- que el comportamiento visible del robot siga siendo coherente con las actions
  RL en `simrobot_visible`, no solo con `obs_ready=1`;
- decidir si el export corregido debe quedar permanente para train o quedar
  detrás de un flag/configuración explícita.

### 6. Revalidación posterior del fix

Después de cerrar las inconsistencias internas del export corregido, se
revalidó lo siguiente:

- `SkillBehaviorControl` ahora recompone `ballRel*` y `ballEndRel*` con la pose
  realmente exportada cuando se corrige la pose del robot;
- `shotQualityNoObstacles`, `shotOpeningWithObstacles` y `canScoreNow` ahora se
  calculan con la pelota efectivamente exportada a RL, no con la pelota
  estimada stale;
- `eval_summary.csv` volvió a quedar alineado con la auditoría del wire y ahora
  agrega:
  - `pose_alignment_warning_rate`,
  - `ball_stale_rate`,
  - `ball_collapsed_rate`,
  - `corrected_exported_robot_pose_rate`,
  - `corrected_exported_ball_rate`.

Validación ejecutada:

- probe directo `simrobot` en perfil `2d`, fuera del sandbox:
  - `corrected`: `obs_ready=1`, `pose_alignment_error_mm=0`, error geométrico
    de `ballRel` = `0`, `corrected_exported_ball=1`;
  - `estimated`: `obs_ready=1`, `pose_alignment_error_mm≈2843.7 mm`,
    `corrected_exported_* = 0`.
- sanity corta de `train.py` en perfil `2d`, fuera del sandbox:
  - `corrected`: `reward_mean≈-0.333`, `stability≈0.407`,
    `pose_alignment_warning_rate=0`, `ball_stale_rate=0`,
    `ball_collapsed_rate=0`, `corrected_exported_ball_rate=1.0`;
  - `estimated`: `reward_mean≈-0.594`, `stability≈0.325`,
    `pose_alignment_warning_rate≈0.01875`, `ball_stale_rate=1.0`,
    `ball_collapsed_rate≈0.98125`, `corrected_exported_* = 0`.

Revalidación de runtime embebido ejecutada el 14 de mayo de 2026:

- se volvió a aislar el problema con la prueba mínima real:
  - importar `pybh.controller`;
  - llamar `pybh.simrobot_load_scene(...)`;
  - sin `env.reset()`, sin PPO, sin reward, sin curriculum.
- resultado mínimo:
  - `RL1v0.ros2d`: `ok=True`, `loaded=True`;
  - `RL1v0_3D.ros2`: caída del proceso durante `simrobot_load_scene(...)`;
  - `BH.ros2`: misma caída, así que no era un bug específico de escena RL.
- después se forzaron variantes de plataforma Qt:
  - `QT_QPA_PLATFORM=offscreen`: el `3d` seguía cayendo;
  - `QT_QPA_PLATFORM=xcb`: el `3d` cargaba bien en esta máquina cuando había
    `DISPLAY`.

Causa real confirmada:

- el host embebido estaba forzando `offscreen` por default;
- en este host, el core `3d` embebido no es estable por `offscreen`;
- con `DISPLAY=:0`, la ruta estable es `xcb` sobre X11/XWayland.

Fix aplicado:

- `SimRobotHost.cpp` ahora elige `QT_QPA_PLATFORM=xcb` si hay `DISPLAY`;
- `offscreen` queda como fallback solo cuando no hay display;
- se mantuvo el saneamiento previo de locale numérico y bootstrap OpenGL.

Estado actualizado:

- la prueba mínima vuelve a pasar:
  - `simrobot_load_scene(RL1v0.ros2d)`: PASS
  - `simrobot_load_scene(RL1v0_3D.ros2)`: PASS
  - `simrobot_load_scene(BH.ros2)`: PASS
- además volvió a pasar el smoke embebido alto:
  - `smoke_test_skill_requests.py --backend simrobot --simrobot-profile 2d`
  - `smoke_test_skill_requests.py --backend simrobot --simrobot-profile 3d`

Pendiente residual:

- el runtime `3d` embebido ya no está roto en carga de escena;
- lo que sigue siendo frágil es la capa de observación warmup, por eso
  `pybh_backend.py` endurece ahora el wait de reset y sintetiza pelota de
  warmup cuando el export corregido sale desalineado.

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
