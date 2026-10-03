# Setup obs-26 para reparar PPO embebido

## Punto de partida

Esta rama local `obs26-stability-repair` reconstruye el estado usable para el
checkpoint vigente sin traer la ampliacion MAPPO de 41 observaciones.

Base:

```text
8043512 Merge remote-tracking branch 'origin/rl-simrobot3d' into rl-simrobot3d
```

Se trajeron encima, sin el commit `22a4631`:

```text
91f3885 Stabilize PPO baseline and update transplant docs
9a97634 Track PPO stability repair checkpoints
b2f62ae Create onnx from production_stability_repair_v1/checkpoint_stage4_step5000.pt
```

La razon es que `9a97634` contiene el checkpoint, pero su historia ya incluye
`22a4631 Obs space extension for multiagent (41 specs)`. SimRobot embebido y el
transplante actual esperan 26 observaciones, por eso no se debe usar el arbol
actual de `RL` para reentrenar este policy.

## Contrato verificado

```text
OBS_SIZE = 26
checkpoint_stage4_step5000.pt policy trunk.0.weight = (128, 26)
checkpoint_stage4_step5000.pt value  trunk.0.weight = (128, 26)
```

Las columnas son:

```text
robot_x, robot_y, robot_theta,
ball_rel_x, ball_rel_y, ball_end_rel_x, ball_end_rel_y,
ball_vel_x, ball_vel_y,
time_since_ball_seen, time_since_ball_disappeared,
ball_seen_percentage, ball_consistent_with_game_state,
can_score_now, shot_quality_no_obstacles, shot_opening_with_obstacles,
pass_options_count,
nearest_teammate_dist, nearest_opponent_dist, nearest_uncertain_obstacle_dist,
nearest_teammate_front_dist, nearest_opponent_front_dist, nearest_uncertain_front_dist,
shoot_armed, dribble_armed, shoot_arm_progress
```

## Como se entreno el checkpoint vigente

El checkpoint vigente:

```text
runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt
```

sale de `train_stability_repair.py`, que envuelve `train_production.py` con:

```text
RL_PROD_EXPERIMENT_NAME=production_stability_repair_v1
RL_PROD_TRAIN_STEPS=20000
RL_PROD_EVAL_INTERVAL=2500
RL_PROD_CHECKPOINT_INTERVAL=2500
RL_PROD_EVAL_EPISODES=8
RL_PROD_MAX_EPISODE_STEPS=1000
RL_PROD_SCENARIO_SET=eval_plus_topup
RL_PROD_VISIBLE_STEP_OBS_TIMEOUT_MS=1000
RL_VISIBLE_RESET_MAX_ATTEMPTS=3
RL_VISIBLE_BALL_SETTLE_TIMEOUT_MS=4000
RL_VISIBLE_BALL_SETTLE_MAX_REPLACES=5
RL_VISIBLE_RESET_UPRIGHT_STREAK=3
RL_VISIBLE_RESET_MAX_FALL_STATE=1
```

Pesos de escenarios:

```text
dynamic_full_field_pressure:6,
blocked_corridor_repair:8,
blocked_corridor_finish_topup:6,
dynamic_finish_pressure_topup:6,
finish_band_2p5_3m_static_lane:3,
finish_band_2p5_3m_clean:2,
finish_bridge_static_topup:3,
finish_bridge_clean_topup:2,
finish_near_arm_clean_topup:1,
preshoot_bad_angle_boundary:1
```

El warm-start por defecto de ese script fue:

```text
runs/production_readiness_v2/checkpoint_stage4_step10000.pt
```

Luego `checkpoint_stage4_step5000.pt` fue exportado a ONNX con:

```text
tools/export/export_ppo_policy.py
```

## Entorno local

Este worktree vive en:

```text
RL/.deps/obs26-base
```

El `.venv` local esta en:

```text
RL/.deps/obs26-base/.venv
```

El repo SabanaHerons usado por este entorno esta apuntado por symlink:

```text
RL/.deps/SabanaHerons2026 -> RL/.deps/SabanaHerons2026-rlsim3d-00ba473
```

y ese worktree esta en:

```text
00ba4734 Link RLSharedState into Nao target
```

## Comandos de verificacion ya corridos

```bash
.venv/bin/python -m py_compile observation.py train.py train_production.py \
  train_stability_repair.py behavioral_evaluator.py behavioral_stability.py \
  behavioral_stability_probe.py environment.py pybh_backend.py \
  production_curriculum.py curriculum.py gate.py action.py config.py

.venv/bin/python train_stability_repair.py --prepare-only
```

`--prepare-only` genero:

```text
runs/fine_tuning_production_dataset/production_replay_stage4/samples.npz
positive=16402/17815
```

## Comando seguro para continuar desde el checkpoint vigente

Para reparar el problema actual de `stand`, no relanzar desde el checkpoint
historico si se quiere conservar el comportamiento vigente. Usar:

```bash
cd /home/rafael/Documentos/Robocup2026/RL/.deps/obs26-base
PYTHONPATH=/home/rafael/Documentos/Robocup2026/RL/.deps/SabanaHerons2026/Make/Python \
.venv/bin/python train_stability_repair.py \
  --checkpoint runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt
```

Antes de correr entrenamiento largo, compilar el worktree SabanaHerons de
`00ba473` y validar una corrida corta visible/headless.

## Reparacion offline del stand

Diagnostico del problema observado en GameFast:

- En los logs embebidos, `shoot_armed=0` y `dribble_armed=0`, asi que el gate
  deja la decision reducida a `stand` vs `walk`.
- El replay positivo obs26 no etiqueta `stand`: en estados no-finish positivos,
  los labels son 100% `walk`.
- El checkpoint `checkpoint_stage4_step5000.pt` aun tenia zonas fuera de la
  distribucion del dataset donde `stand` y `walk` quedaban casi empatados.

Se agrego `train_stand_repair.py` para hacer una reparacion local de politica,
sin tocar el obs space y sin cambiar Sabana:

```bash
cd /home/rafael/Documentos/Robocup2026/RL/.deps/obs26-base
.venv/bin/python train_stand_repair.py
PYTHONPATH=/home/rafael/Documentos/Robocup2026/RL/.deps/obs26-base \
.venv/bin/python tools/export/export_ppo_policy.py \
  --checkpoint runs/production_stand_repair_v1/checkpoint_stage4_stand_repair.pt \
  --output runs/production_stand_repair_v1/ppo_striker_hsl2026_stand_repair.onnx
```

La reparacion congela trunk/parametros continuos y solo ajusta `skill_head` con:

- CE sobre el replay positivo stage 4.
- Estados sinteticos obs26 con `shoot_armed=0`, `dribble_armed=0` y target
  `walk`, parecidos al fallo online.
- Margen explicito `logit(walk) - logit(stand) >= 1.5`.
- KL suave contra el checkpoint original para no mover innecesariamente
  `dribble/shoot`.

Resultado offline del candidato:

```text
no_finish_positive:
  stand_rate 0.0053 -> 0.0000
  p_stand_mean 0.0494 -> 0.0013
  p_walk_mean  0.9506 -> 0.9987

synthetic_no_finish:
  stand_rate 0.4561 -> 0.0000
  p_stand_mean 0.4794 -> 0.0007
  p_walk_mean  0.5206 -> 0.9993
```

Artefactos generados:

```text
runs/production_stand_repair_v1/checkpoint_stage4_stand_repair.pt
runs/production_stand_repair_v1/ppo_striker_hsl2026_stand_repair.onnx
runs/production_stand_repair_v1/ppo_striker_hsl2026_stand_repair.onnx.data
```

Para probar en Sabana, exportar una variante compat con batch estatico/opset 11
y copiar el `.onnx` y el `.json` al directorio de
`Config/NeuralNets/RLPolicy/`. El export compat no requiere `.onnx.data`.

## Reparacion offline del param_head

Despues de reparar `stand`, GameFast mostro un segundo fallo: la policy elegia
`walk`, pero `param_head` emitia casi siempre:

```text
params[x,y,th,p] ~= -1,-1,1,-1
```

Eso no siempre es malo. Con la pelota lejos puede generar un `targetRel` grande
y caminar normalmente. El problema aparece cuando esa misma salida cae dentro
del umbral muerto de `WalkToPoint`:

```text
targetRel[x,y,th] ~= 0,0,0
```

En ese caso SimRobot recibe `walk`, pero la skill considera que ya llego y el
robot se queda quieto. Esto no debe arreglarse con un guard de C++; debe
corregirse en la cabeza continua de la policy.

Se agrego `train_param_repair.py` para partir del checkpoint con `stand`
reparado y ajustar solo `param_head`:

```bash
cd /home/rafael/Documentos/Robocup2026/RL/.deps/obs26-base
.venv/bin/python train_param_repair.py --steps 1600 --export-compat
```

La reparacion usa:

- BC sobre residuales positivos `walk`/`dribble` del replay stage 4.
- Estados sinteticos obs26 cercanos al caso muerto de `walk`.
- Labels continuas centradas en el anchor heuristico (`residual_y ~= 0`,
  `residual_theta ~= 0`) para evitar saturacion.
- Trunk y `skill_head` congelados, conservando el arreglo de `stand`.

Metricas del candidato:

```text
stand/no-finish:
  dataset_no_finish_positive stand_rate = 0.0000, p_walk_mean = 0.9987
  synthetic_no_finish        stand_rate = 0.0000, p_walk_mean = 0.9993

param/walk:
  synthetic_near_dead walk_dead_rate 0.0050 -> 0.0000
  synthetic_no_finish walk_dead_rate 0.0010 -> 0.0000
  walk_param_y_mean     -1.0000 -> -0.0233
  walk_param_theta_mean  0.9996 -> -0.0048
```

Artefactos generados:

```text
runs/production_param_repair_v1/checkpoint_stage4_param_repair.pt
runs/production_param_repair_v1/ppo_striker_hsl2026_param_repair_compat.onnx
runs/production_param_repair_v1/ppo_striker_hsl2026_param_repair_compat.json
```

Compatibilidad verificada del ONNX transplantado:

```text
ir_version = 6
opset ai.onnx = 11
input obs = [1, 26]
outputs = skill_logits [1, 8], param_mean [1, 4]
sha256 = 2870181793a4675a81ca1f3a2cef5dd8632292ebca06163ea0b071939cc76a4c
```

Nota: durante la prueba en Sabana conviene dejar la instrumentacion de logs
`params` y `targetRel`, pero sin fallback/guard de decoder, para validar que el
movimiento viene de la policy corregida.

## Behavioral evaluator del candidato transplantado

El checkpoint evaluado fue:

```text
runs/production_param_repair_v1/checkpoint_stage4_param_repair.pt
```

Comando usado:

```bash
cd /home/rafael/Documentos/Robocup2026/RL/.deps/obs26-base
.venv/bin/python behavioral_evaluator.py \
  --checkpoint runs/production_param_repair_v1/checkpoint_stage4_param_repair.pt \
  --episodes 4 \
  --seed 4200 \
  --max-episode-steps 1000
```

El resultado global fue `overall_pass=0`, con 4 de 5 situaciones aprobadas. El
fallo restante no fue el bloqueo en `stand`; fue seguridad de collision en
`preshoot_bad_angle_boundary`.

```text
summary = runs/behavioral_evaluator/behavior_20260610_192350/summary.csv

finish_band_2p5_3m_clean        pass=1 goal=0.250 finish=0.750 shoot_finish=0.500 collision=0.000 policy_stand=0.000
finish_band_2p5_3m_static_lane  pass=1 goal=0.500 finish=0.750 shoot_finish=0.750 collision=0.000 policy_stand=0.001
preshoot_bad_angle_boundary     pass=0 goal=0.000 finish=0.833 shoot_finish=0.667 collision=0.333 policy_stand=0.004 failures=collision_rate<=0.12
dynamic_full_field_pressure     pass=1 goal=0.000 finish=0.500 shoot_finish=0.000 collision=0.000 policy_stand=0.000
blocked_corridor_repair         pass=1 goal=0.167 finish=0.667 shoot_finish=0.333 collision=0.000 policy_stand=0.001
```

Lectura:

- La reparacion de `stand` funciona para el sintoma original: el
  `policy_stand_rate` queda practicamente en cero en el evaluator visible.
- El caso que aun falla es otro eje de comportamiento: mala orientacion cerca de
  zona de tiro produce colisiones. La siguiente iteracion debe entrenar
  terminacion/seguridad alrededor de `preshoot_bad_angle_boundary`, no volver a
  meter un fallback local en Sabana.
- En escenarios full-field dinamicos todavia hay poco tiro (`policy_shoot=0` en
  `dynamic_full_field_pressure`), asi que el candidato destraba el movimiento
  pero no cierra el rendimiento ofensivo.
