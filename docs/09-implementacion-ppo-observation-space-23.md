# Implementacion PPO Observation Space 23

## Objetivo

Este documento describe la implementacion ya aplicada para ampliar el observation space de PPO desde el vector historico de 9 variables hacia un vector realista de 23 variables, alineado con los documentos:

- `docs/07-observation-space-realista-y-fallback-canary.md`
- `docs/12-wire-rl-realista-final.md`

El objetivo de esta fase no es resolver toda la politica futbolistica ni entrenar MAPPO. El objetivo es dejar PPO listo para consumir observaciones mas cercanas al flujo real de B-Human, sin romper el pipeline natural:

```text
RL
  -> SkillRequest
  -> SkillBehaviorControl
  -> MotionRequest
  -> MotionEngine
  -> SimRobot / robot real
```

La regla de arquitectura se mantiene:

```text
RL decide QUE skill usar.
B-Human decide COMO ejecutarla.
```

## Alcance implementado

El cambio implementado cubre cuatro capas:

1. Contrato de observacion en Python RL.
2. Bridge `pybh` para leer nuevas variables desde B-Human.
3. Publicacion de observaciones desde el flujo natural de B-Human.
4. Prueba visible con SimRobot 3D.

No se implemento:

- entrenamiento final de PPO,
- tuning de reward futbolistico,
- MAPPO,
- remocion completa de providers oraculares,
- accion continua de bajo nivel,
- reemplazo de locomocion o `MotionEngine`.

## Archivos RL modificados

### `observation.py`

Nuevo archivo que centraliza el contrato del vector:

- `ObsIndex`
- `OBS_FEATURE_NAMES`
- `OBS_SIZE`

Esto evita que `environment.py`, `reward.py` y tests dependan de indices hardcodeados.

### `environment.py`

Cambios principales:

- `observation_spaces["player_1"]` pasa de shape `(9,)` a `(23,)`.
- `state_space` pasa de shape `(9,)` a `(23,)`.
- Se agrega `state_spaces` para compatibilidad con wrappers que esperan espacios por agente.
- `_encode_obs(...)` codifica el nuevo vector normalizado.
- `info["obs_feature_names"]` expone los nombres de las 23 features.

### `pybh_backend.py`

`BackendObservation` fue ampliado para aceptar las nuevas variables expuestas por `pybh.rl_get_obs(...)`.

Tambien mantiene fallback defensivo:

- si `ball_rel_x` / `ball_rel_y` no llegan desde el bridge,
- se calculan desde `ball_x`, `ball_y`, `robot_x`, `robot_y`, `robot_theta`.

Ese fallback permite que herramientas viejas no fallen inmediatamente, pero el flujo correcto es recibir las variables desde B-Human.

### `reward.py`

`ball_distance(...)` ya no lee `obs[0]` y `obs[1]`.

Ahora usa:

- `ObsIndex.ball_rel_x`
- `ObsIndex.ball_rel_y`

Esto evita que el reward quede desalineado cuando cambia el layout del vector.

### `train.py`

Se agrego soporte operativo para:

- `--backend simrobot_visible`
- `--launch-simrobot`
- alias `--simrobot-launch`

Esto permite hacer smoke tests contra una ventana visible de SimRobot sin cambiar el codigo.

### `live_simrobot_probe.py`

Se mejoro la prueba manual visible:

- `--shoot-distance`
- `--shoot-hold-steps`
- conserva la ultima observacion valida si llega una muestra no lista,
- imprime `stale_frame` para diagnosticar problemas de handshake,
- permite confirmar visualmente que el robot camina y ejecuta skills desde el flujo natural.

### `test_observation_contract.py`

Nuevo test de contrato minimo:

- valida que `_encode_obs(...)` devuelve shape `(OBS_SIZE,)`,
- valida que flags booleanos se codifican como `1.0`,
- valida que `reward.ball_distance(...)` usa los indices correctos.

## Vector PPO 23

El vector final queda definido por `OBS_FEATURE_NAMES` en este orden:

| Indice | Feature | Fuente conceptual | Normalizacion Python |
| --- | --- | --- | --- |
| 0 | `robot_x` | `RobotPose.translation.x` | `/ ACTION.x_range` |
| 1 | `robot_y` | `RobotPose.translation.y` | `/ ACTION.y_range` |
| 2 | `robot_theta` | `RobotPose.rotation` | `/ pi` |
| 3 | `ball_rel_x` | `FieldBall.positionRelative.x` | `/ ACTION.x_range` |
| 4 | `ball_rel_y` | `FieldBall.positionRelative.y` | `/ ACTION.y_range` |
| 5 | `ball_end_rel_x` | `FieldBall.endPositionRelative.x` | `/ ACTION.x_range` |
| 6 | `ball_end_rel_y` | `FieldBall.endPositionRelative.y` | `/ ACTION.y_range` |
| 7 | `ball_vel_x` | `BallModel.estimate.velocity.x` | `/ 3000.0` |
| 8 | `ball_vel_y` | `BallModel.estimate.velocity.y` | `/ 3000.0` |
| 9 | `time_since_ball_seen` | `FieldBall.timeSinceBallWasSeen` | clamp `/ 5000.0` |
| 10 | `time_since_ball_disappeared` | `FieldBall.timeSinceBallDisappeared` | clamp `/ 5000.0` |
| 11 | `ball_seen_percentage` | `BallModel.seenPercentage` | clamp `/ 100.0` |
| 12 | `ball_consistent_with_game_state` | `FieldBall.ballPositionConsistentWithGameState` | bool a `0.0/1.0` |
| 13 | `can_score_now` | placeholder bridge | bool a `0.0/1.0` |
| 14 | `shot_quality_no_obstacles` | placeholder bridge | clamp `0.0..1.0` |
| 15 | `shot_opening_with_obstacles` | placeholder bridge | clamp `0.0..1.0` |
| 16 | `pass_options_count` | `TeamData.teammates.size()` | clamp `/ 5.0` |
| 17 | `nearest_teammate_dist` | resumen `ObstacleModel` | clamp `/ ACTION.x_range` |
| 18 | `nearest_opponent_dist` | resumen `ObstacleModel` | clamp `/ ACTION.x_range` |
| 19 | `nearest_uncertain_obstacle_dist` | resumen `ObstacleModel` | clamp `/ ACTION.x_range` |
| 20 | `nearest_teammate_front_dist` | resumen frontal `ObstacleModel` | clamp `/ ACTION.x_range` |
| 21 | `nearest_opponent_front_dist` | resumen frontal `ObstacleModel` | clamp `/ ACTION.x_range` |
| 22 | `nearest_uncertain_front_dist` | resumen frontal `ObstacleModel` | clamp `/ ACTION.x_range` |

## Variables placeholder

Tres features quedan cableadas pero todavia no tienen evaluador futbolistico real:

- `can_score_now`
- `shot_quality_no_obstacles`
- `shot_opening_with_obstacles`

Actualmente se publican como valores conservadores desde B-Human:

```text
can_score_now = false
shot_quality_no_obstacles = 0.0
shot_opening_with_obstacles = 0.0
```

Se dejaron en el contrato porque pertenecen al diseno PPO 23 acordado, pero no deben interpretarse como senales aprendibles de tiro hasta implementar el calculo ofensivo real.

## Bridge B-Human requerido

Para que PPO reciba estas 23 variables en SimRobot, el repo B-Human debe exponerlas por `RLSharedState`.

Cambios requeridos en `SabanaHerons2026`:

- `Src/Libs/Python/Controller/RLSharedState.h`
- `Src/Libs/Python/Controller/RLSharedState.cpp`
- `Src/Libs/Python/Controller/Module.cpp`
- `Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.h`
- `Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.cpp`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.cpp`

Puntos importantes:

- Se mantiene `PlaySoccer()` dentro de `SkillBehaviorControl`.
- No se puentea `MotionRequest` directamente desde Python.
- `SkillBehaviorControl` publica observaciones frescas por frame.
- `rl_get_obs(...)` espera hasta que `obsReady == true`, no solo hasta que el semaforo despierte.

El ultimo punto fue necesario porque habia senales viejas acumuladas en el semaforo. Sin esa correccion, el probe visible podia mostrar:

```text
obs_ready=0 stale_frame=...
```

aunque SimRobot estuviera avanzando.

## Flujo de prueba visible

Comando validado para inspeccion manual:

```bash
cd 
PYTHONPATH=/Make/Python python3 live_simrobot_probe.py \
  --launch \
  --policy heuristic \
  --simrobot-profile 3d \
  --steps 1000 \
  --shoot-distance 700 \
  --shoot-hold-steps 80
```

Resultado esperado:

- SimRobot abre la escena `RL1v0_3D.ros2`.
- El frame empieza a avanzar cuando termina de cargar la escena.
- El log no debe quedarse en `obs_ready=0 stale_frame=...`.
- El robot debe recibir acciones por `SkillRequest`.
- El movimiento visible debe seguir pasando por B-Human.

## Validaciones realizadas

Validaciones Python:

```bash
PYTHONPATH=/Make/Python python3 -m py_compile \
  observation.py pybh_backend.py environment.py reward.py train.py live_simrobot_probe.py test_observation_contract.py
```

```bash
PYTHONPATH=/Make/Python python3 -u -c \
  'import test_observation_contract as t; t.test_observation_contract_size_and_reward_indices(); print("observation_contract_ok")'
```

Validacion B-Human:

```bash
cmake --build /Build/Linux/CMake/Develop \
  --target PythonController SimulatedNao
```

Validacion embebida 3D:

- `rl_get_obs(...)` devuelve `obs_ready=True`.
- `frame` avanza.
- `robot_x` cambia.
- `skill_behavior_call_count` avanza.

Validacion manual visible:

- El probe visible ya no queda clavado en `stale_frame`.
- El robot se mueve desde la GUI de SimRobot.

## Riesgos restantes

### 1. Providers oraculares

Este cambio no elimina por completo los providers oraculares de los escenarios RL.

Reduce el dano porque el vector ya consume senales mas parecidas a B-Human real, pero si el escenario sigue inyectando `RobotPose`, `BallModel` u `ObstacleModel` oraculares, el entrenamiento aun puede sobreestimar desempeno.

### 2. Features ofensivas placeholder

Las features de tiro estan en el contrato pero todavia no aportan informacion real.

Antes de entrenar una politica que dependa de tiro, se debe implementar una estimacion real para:

- posibilidad de anotar,
- calidad de tiro sin obstaculos,
- apertura de tiro con obstaculos.

### 3. Normalizacion inicial

Las constantes:

- `BALL_VEL_NORM = 3000.0`
- `TIME_NORM_MS = 5000.0`
- `PASS_COUNT_NORM = 5.0`

son razonables para canary, pero deben revisarse con distribuciones reales antes de entrenamientos largos.

### 4. Compatibilidad de checkpoints

Cualquier checkpoint PPO entrenado con 9 variables queda incompatible con el nuevo `observation_space`.

El cambio es intencional y rompe compatibilidad hacia atras a nivel de modelo.

### 5. MAPPO no queda resuelto

El vector PPO 23 es una base reutilizable, pero MAPPO necesita estado multiagente adicional y contrato separado.

Este commit no implementa MAPPO 35.

## Criterio de listo para commit

Desde el punto de vista PPO, este cambio queda listo para commit cuando:

- `OBS_SIZE == 23`,
- `environment.py` expone observation/state spaces de 23,
- `reward.py` usa `ObsIndex`,
- `pybh_backend.py` lee las nuevas variables,
- el test de contrato pasa,
- el probe visible recibe observaciones frescas,
- y el bridge B-Human se commitea aparte sin cambios ajenos de `Config/Locations` o submodulos.
