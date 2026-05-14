# Cambios de implementación para PPO takeover Stage 0-3

Este documento registra de forma quirúrgica y detallada los cambios aplicados
para preparar el stack RL actual a entrenamiento PPO con:

- observación cruda como fuente de verdad para reward,
- curriculum Stage 0-3,
- telemetría de sanidad,
- y soporte de evaluación/promoción por stage.

Su propósito es servir como base de análisis de fallas si algo se rompe o si el
comportamiento no coincide con lo esperado.

## Resumen ejecutivo

Se aplicaron cambios en:

- [config.py](/home/luis/GitHub/RL/config.py)
- [curriculum.py](/home/luis/GitHub/RL/curriculum.py)
- [environment.py](/home/luis/GitHub/RL/environment.py)
- [pybh_backend.py](/home/luis/GitHub/RL/pybh_backend.py)
- [reward.py](/home/luis/GitHub/RL/reward.py)
- [train.py](/home/luis/GitHub/RL/train.py)
- [wrapper.py](/home/luis/GitHub/RL/wrapper.py)
- [test_observation_contract.py](/home/luis/GitHub/RL/test_observation_contract.py)

Objetivo técnico de los cambios:

1. dejar de calcular reward sobre observación normalizada,
2. pasar reward y métricas a datos crudos ya validados por heurísticas/probes,
3. conectar el soporte nuevo del bridge para resets extendidos y updates
   dinámicos,
4. introducir curriculum Stage 0-3 sin rediseñar el framework.

## Cambio 1: `reward.py` deja de ser smoke y pasa a semántica cruda

Archivo:

- [reward.py](/home/luis/GitHub/RL/reward.py)

### Antes

El reward anterior era mínimo:

- progreso a pelota,
- bonus por gol,
- step penalty.

No había:

- notion de takeover,
- notion de zonas walk/dribble/shoot,
- notion de thrash,
- notion de stage,
- ni compatibilidad skill-contexto.

### Después

Se reescribió `reward.py` para introducir:

#### 1. `RewardWeights`

Contiene pesos explícitos por término:

- `progress_ball`
- `align_ball`
- `enter_takeover`
- `hold_takeover`
- `good_walk`
- `bad_walk`
- `good_dribble`
- `bad_dribble`
- `good_shoot`
- `bad_shoot`
- `skill_switch`
- `thrash`
- `goal`
- `timeout`
- `collision`
- `clearance_gain`
- `corridor_keep`
- `ball_progress_post_takeover`
- `first_to_ball`
- `bad_duel`

#### 2. `TakeoverFeatures`

Derivadas operativas crudas:

- distancia al balón,
- ángulo al balón,
- frescura de visión,
- porcentaje visto,
- `score_now`,
- `shot_open`,
- presión frontal,
- obstáculo frontal.

#### 3. `StepContext`

Contexto crudo por step para evaluar reward.

Incluye:

- `prev_obs`
- `obs`
- `skill_name`
- `policy_skill_name`
- `prev_skill_name`
- `skill_history`
- `stage`
- flags de `goal`, `timeout`, `collision`, `corridor_open`, etc.

#### 4. Helpers semánticos

Se añadieron:

- `features_from_obs(...)`
- `is_takeover_envelope(...)`
- `is_walk_zone(...)`
- `is_shoot_zone(...)`
- `is_dribble_zone(...)`
- `preferred_skill(...)`
- `pattern_like(...)`
- `is_local_thrash(...)`

#### 5. `heuristic_bias(...)`

Se implementó el sesgo blando acordado:

- `walk` favorecido fuera de takeover,
- `shoot` favorecido solo en shoot zone,
- `dribble` favorecido en takeover sin tiro,
- presión frontal suma a `dribble`,
- anti-thrash básico.

#### 6. `evaluate_step(...)`

Se implementó el cálculo completo del reward por step sobre crudo.

Produce:

- `terms`
- `weighted_terms`
- `total_reward`
- zonas activas
- `preferred_skill_name`
- `heuristic_bias`

### Suposiciones importantes

- el reward se apoya en `BackendObservation` cruda,
- no depende del vector normalizado del PPO,
- Stage 4 queda cableado en estructura pero no se usa como núcleo del sprint.

### Punto de falla potencial

Si el reward parece incoherente, revisar primero:

- si `BackendObservation` trae bien los valores crudos,
- si los thresholds están en mm/ms reales,
- si `environment.py` arma bien `StepContext`.

## Cambio 2: creación de `curriculum.py`

Archivo:

- [curriculum.py](/home/luis/GitHub/RL/curriculum.py)

### Objetivo

Sacar del env la lógica de stage y dejar una capa explícita para:

- specs de stage,
- reward weights,
- masks,
- sampling de resets,
- dynamic world updates,
- gates de promoción.

### Contenido nuevo

#### 1. `StageSpec`

Contiene:

- `stage_id`
- `name`
- `description`
- `ball_radius_max_mm`
- `full_field`
- `obstacle_mode`
- `active_skill_names`
- `reward_weights`
- `max_episode_steps`
- `dynamic_update_interval_steps`

#### 2. `_STAGE_SPECS`

Se definieron Stage 0-3:

- Stage 0: corto, sin obstáculos
- Stage 1: más distancia, sin obstáculos
- Stage 2: obstáculos estáticos
- Stage 3: campo completo + obstáculos dinámicos scripted

#### 3. Máscaras

- `stage_skill_mask(stage_id)`

Devuelve activas solo:

- `walk`
- `dribble`
- `shoot`

y deja en `0`:

- `pass`
- `mark`
- `block`
- `observe`

#### 4. Samplers de reset

Se implementaron:

- `_sample_pose_ball_pair(...)`
- `_sample_full_field_pair(...)`
- `sample_stage_world_spec(...)`

Estos generan `WorldSpec` con:

- balón,
- robot propio,
- teammates de soporte,
- opponents estáticos o dinámicos según stage.

#### 5. Obstáculos scripted

Se añadieron:

- `_static_obstacles(...)`
- `_dynamic_obstacles(...)`
- `get_dynamic_world_update(...)`

Esto usa la infraestructura ya existente en `pybh_backend.py`.

#### 6. Gates de promoción

Se implementó:

- `promote_stage(...)`

Con reglas para Stage 0-3.

### Suposiciones importantes

- Stage 4 queda fuera del núcleo,
- Stage 3 usa dinámicos scripted simples, no duel fino,
- el scene switching usa `RL1v0_3D` para Stage 0-1 y `RLFull_3D` para Stage 2-3.

### Punto de falla potencial

Si Stage 2-3 no materializa obstáculos:

- revisar scene path,
- revisar slots de robots disponibles,
- revisar si `WorldSpec` se está pasando realmente al backend.

## Cambio 3: `pybh_backend.py` amplía la observación cruda

Archivo:

- [pybh_backend.py](/home/luis/GitHub/RL/pybh_backend.py)

### Antes

`BackendObservation` solo retenía el subconjunto usado por el env original.

### Después

Se ampliaron campos para consumir telemetría ya exportada por C++:

- `obstacle_count`
- `obstacle_teammate_count`
- `obstacle_opponent_count`
- `obstacle_uncertain_count`
- `obstacle_front_count`
- `nearest_obstacle_dist`
- `nearest_obstacle_angle`
- `nearest_front_obstacle_dist`
- `nearest_opponent_angle`
- `obstacle_arm_contact`
- `obstacle_foot_contact`
- `zweikampf_active`
- `zweikampf_call_count`
- `motion_obstacle_path_count`
- `walk_to_pose_path_count`
- `libwalk_active_obstacle_count`
- `libwalk_disable_obstacle_avoidance`
- `motion_engine_ground_contact`
- `motion_engine_fall_state`
- `motion_speed_x`
- `motion_speed_y`
- `motion_speed_rot`

También ya existía y se conservó:

- `simrobot_robot_x/y/theta`
- soporte de `WorldSpec`
- reset extendido
- dynamic world
- polling de estado visible.

### Punto de falla potencial

Si el reward no ve obstáculos o colisiones razonables:

- revisar que `Module.cpp` siga publicando esos campos,
- revisar que `BackendObservation.from_payload(...)` los parsee bien,
- revisar que el env los use en vez de ignorarlos.

## Cambio 4: `config.py` pasa a defaults de sprint real

Archivo:

- [config.py](/home/luis/GitHub/RL/config.py)

### Cambios

#### Backend

- `frames_per_step: 1 -> 3`
- `simrobot_profile: "2d" -> "3d"`

#### Env

- se añadió `stage_id`
- `active_skill_names` pasa a:
  - `("walk", "dribble", "shoot")`

#### Train

Se actualizaron defaults:

- `trainer_timesteps: 64 -> 20000`
- `rollout_steps: 16 -> 256`
- `learning_epochs: 2 -> 4`
- `mini_batches: 2 -> 4`
- `learning_rate = 1e-4`
- `discount_factor = 0.99`
- `gae_lambda = 0.95`
- `entropy_loss_scale = 0.001`
- `value_loss_scale = 1.0`
- `ratio_clip = 0.2`
- `grad_norm_clip = 0.5`
- `checkpoint_interval = 5000`
- `eval_interval = 5000`
- `eval_episodes = 4`
- `initial_stage_id = 0`
- `max_stage_id = 3`
- `auto_promote = True`

### Punto de falla potencial

Si el train sigue comportándose como smoke:

- revisar que `train.py` realmente use estos valores y no hardcodes viejos.

## Cambio 5: `environment.py` reescrito para sostener reward crudo + curriculum

Archivo:

- [environment.py](/home/luis/GitHub/RL/environment.py)

### Objetivo

Transformar el env en el punto de unión entre:

- `BackendObservation` cruda,
- reward semántico,
- curriculum por stage,
- telemetría por step y episodio.

### Cambios estructurales

#### 1. Nuevo estado interno

Se añadieron:

- `self.stage_spec`
- `self._last_backend_obs`
- `self._last_evaluation`
- `self._skill_history`
- `self._current_world_spec`
- `self._episode_tracker`

#### 2. Reset por curriculum

Antes:

- sample fijo de balón/robot.

Ahora:

- `sample_stage_world_spec(...)`
- `backend.reset(world_spec=...)`

#### 3. Proyección de acción

Se añadió:

- `project_action(...)`

Si la skill pedida no está activa:

- se proyecta a `preferred_skill(...)`,
- se construyen targets de fallback con `build_fallback_action_targets(...)`.

#### 4. Step con reward crudo

Antes:

- reward sobre `encoded`.

Ahora:

- construye `StepContext`,
- llama `evaluate_step(...)`,
- actualiza métricas,
- conserva `encoded` solo como input de policy.

#### 5. Dynamic world update

Se añadió:

- llamada a `get_dynamic_world_update(...)`
- y `backend.set_dynamic_world(...)`

para Stage 3.

#### 6. Derivaciones locales

Se añadieron:

- `_derive_collision(...)`
- `_derive_corridor_open(...)`

Importante:

- `collision` sigue siendo derivada heurística,
- no es todavía una señal de negocio perfecta.

#### 7. Tracking de episodio

Se añadió:

- `_reset_episode_tracker()`
- `_update_episode_tracker(...)`
- `_running_metrics_snapshot()`
- `_finalize_episode_metrics()`

Se acumulan:

- reward total,
- breakdown por términos,
- compatibilidad skill-zona,
- takeover,
- thrash,
- colisiones,
- clearance,
- corridor keep,
- progresión post takeover,
- skill projection rate.

#### 8. `info` rico

`_build_info(...)` ahora exporta:

- `raw_obs`
- `stage_id`, `stage_name`
- `policy_skill_name`
- `executed_skill_name`
- `projection_applied`
- `dynamic_update_applied`
- `reward_terms`
- `reward_weighted_terms`
- `preferred_skill_name`
- zonas
- snapshot de métricas
- y al cierre, `episode_metrics`

### Puntos de falla potencial

Si el env falla, revisar en este orden:

1. `sample_stage_world_spec(...)`
2. `backend.reset(world_spec=...)`
3. `project_action(...)`
4. `StepContext`
5. `evaluate_step(...)`
6. `_finalize_episode_metrics()`

### Riesgo conocido

Hay mucha más lógica dentro del env que antes. Eso mejora trazabilidad, pero si
algo sale raro, el sitio más probable del bug ahora es `environment.py`.

## Cambio 6: `wrapper.py` deja de resumir solo reward

Archivo:

- [wrapper.py](/home/luis/GitHub/RL/wrapper.py)

### Cambios

`collect_episode(...)` ahora acumula:

- `reward_terms_sum`
- `reward_weighted_terms_sum`
- `episode_metrics`
- `final_info`

Antes devolvía solo:

- `steps`
- `mean_reward`
- `total_rewards`

Ahora también devuelve telemetría útil para evaluación.

### Punto de falla potencial

Si la evaluación devuelve métricas vacías:

- revisar que `environment.py` esté rellenando `episode_metrics` al terminar.

## Cambio 7: `train.py` reescrito para evaluación, checkpoints y promoción

Archivo:

- [train.py](/home/luis/GitHub/RL/train.py)

### Cambios estructurales

#### 1. Config por stage

Se añadió:

- `make_env_config(stage_id, backend_cfg)`

para:

- scene path por stage,
- `active_skill_names`,
- `stage_id` dentro del env.

#### 2. `build_env(...)`

Ahora devuelve:

- `raw_env`
- `wrapped_env`
- `eval_wrapper`

#### 3. `build_mappo(...)`

Ahora usa `TrainConfig` real en vez de hardcodes smoke.

#### 4. `sample_smoke_action(...)`

Ahora proyecta la acción del agente usando el mismo criterio del env.

#### 5. Evaluación

Se añadieron:

- `_tensorize_eval_obs(...)`
- `_tensorize_eval_state(...)`
- `_aggregate_eval_metrics(...)`
- `evaluate_agent(...)`
- `print_eval_summary(...)`

Esto permite evaluación periódica con métricas de comportamiento reales.

#### 6. Checkpoints

Se añadió:

- `save_checkpoint(...)`

Guarda:

- timestep
- stage
- `policy_state_dict`
- `value_state_dict`
- `eval_summary`

#### 7. Promoción

Dentro de `run_training(...)`:

- eval periódica
- checkpoint periódica
- promoción automática hasta `max_stage_id`

### Suposiciones importantes

- se asume que `skrl` está disponible en el entorno real de training,
- la extracción de `policy_loss`, `value_loss`, `entropy` quedó best-effort.

### Punto de falla potencial

Si `train.py` falla:

1. revisar dependencias de `skrl`,
2. revisar compatibilidad del wrapper `pettingzoo`,
3. revisar la extracción de métricas del agente,
4. revisar si la promoción reabre envs correctamente.

## Cambio 8: `test_observation_contract.py`

Archivo:

- [test_observation_contract.py](/home/luis/GitHub/RL/test_observation_contract.py)

### Cambio

Se agregaron:

- `simrobot_robot_x`
- `simrobot_robot_y`
- `simrobot_robot_theta`

al constructor de `BackendObservation` del test.

Razón:

- el dataclass creció y el test debía seguir compilando.

## Verificación ejecutada

Se ejecutó:

```bash
python3 -m py_compile \
  config.py \
  curriculum.py \
  reward.py \
  pybh_backend.py \
  environment.py \
  wrapper.py \
  train.py \
  test_observation_contract.py \
  test_world_spec_contract.py
```

Resultado:

- sintaxis limpia.

## Lo que no se verificó todavía

No se verificó en esta sesión:

- entrenamiento completo de PPO,
- evaluación real con `skrl`,
- métricas reales de convergencia,
- estabilidad completa de Stage 3,
- telemetría final de promotion loop.

## Riesgos abiertos

### 1. `collision` sigue heurístico

Hoy se deriva a partir de:

- `fall_state`
- `ground_contact`
- `arm/foot obstacle contact`

Eso es útil para arrancar, pero no es una verdad canónica de colisión.

### 2. Stage 3 usa dinámicos scripted simples

Es suficiente para el sprint inicial, pero no representa aún un régimen rico de
oponentes reactivos.

### 3. Stage 4 queda fuera del núcleo

Se dejó la estructura preparada para términos de Stage 4, pero:

- no se cableó como requisito vital,
- no debe usarse para bloquear el sprint.

### 4. `environment.py` concentra mucha lógica

Eso es deliberado para destrabar el stack rápido, pero aumenta la carga de
depuración local si algo sale raro.

## Recomendación de análisis de fallas

Si algo falla en entrenamiento o comportamiento, revisar en este orden:

1. `raw_obs` exportada por el env
2. `StepContext`
3. `evaluate_step(...)`
4. `project_action(...)`
5. `sample_stage_world_spec(...)`
6. `episode_metrics`
7. `evaluate_agent(...)`
8. `promote_stage(...)`

## Resumen final

Los cambios mueven el sistema de:

- smoke reward sobre observación normalizada

hacia:

- reward semántico sobre observación cruda,
- curriculum real Stage 0-3,
- telemetría suficiente para sanidad,
- y evaluación usable para promoción.

Esa es la base nueva sobre la que deben analizarse fallas futuras.
