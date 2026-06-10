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
tools/export_ppo_policy.py
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
