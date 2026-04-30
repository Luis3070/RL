# Contrato RL, Backend y Entorno

## Contrato de accion

El contrato canonico esta en:

```text
RL/action.py
```

La accion plana tiene cinco campos:

```text
[skill_type, target_x, target_y, target_theta, pass_target]
```

Skills:

```text
0 stand
1 walk
2 shoot
3 pass
4 dribble
5 block
6 mark
7 observe
```

En Python, `pass` se llama `pass_` internamente para evitar la keyword del lenguaje, pero hacia `pybh` se envia como `"pass"`.

## Coordenadas

Regla de oro:

```text
SkillRequest::walkTo usa coordenadas globales de campo.
```

Por eso `adapt_action_to_backend(...)` deja sin transformar las acciones para:

```text
cpp
simrobot
simrobot_visible
```

Solo el backend legacy `python` convierte a coordenadas relativas.

## Bridge Python -> B-Human

Python llama:

```python
pybh.rl_set_action(
    player_number,
    skill,
    target_x,
    target_y,
    target_theta,
    pass_target,
)
```

El binding escribe en `RLSharedState`:

```text
skill
targetX
targetY
targetTheta
passTarget
obsReady = false
```

Tambien drena posts viejos de `obsSignal` para que la siguiente observacion corresponda al frame nuevo.

## Bridge B-Human -> Python

Python lee:

```python
pybh.rl_get_obs(player_number, timeout_ms)
```

Campos base:

```text
ball_x
ball_y
robot_x
robot_y
robot_theta
frame
obs_ready
sim_enabled
requested_skill
requested_pass_target
```

Campos de debug del flujo natural:

```text
provider_motion_request
provider_call_count
skill_behavior_skill_request
skill_behavior_motion_request
skill_behavior_call_count
motion_engine_input_request
motion_engine_effective_request
motion_engine_phase
motion_engine_fall_state
motion_speed_x
motion_speed_y
motion_speed_rot
joint_l_hip_pitch
joint_l_knee_pitch
joint_r_hip_pitch
joint_r_knee_pitch
```

## `RLSharedState`

Archivo:

```text
SabanaHerons2026/Src/Libs/Python/Controller/RLSharedState.h
```

Propiedades:

- memoria compartida POSIX;
- nombre actual: `/pybh_rl_shared_state_v2`;
- version actual: `12`;
- mutex compartido entre procesos;
- mutex robusto: si un proceso muere con el lock tomado, el siguiente proceso recupera con `EOWNERDEAD`;
- semaforo `obsSignal` para despertar a Python cuando hay observacion.

La robustez del mutex es importante porque los smoke tests con `timeout` pueden matar Python dentro de codigo nativo.

## Backends de `PyBHBackend`

### `mode="cpp"`

Usa:

```text
pybh.Controller()
scenario = 4v4_RL2D
RLSim2D enabled
```

Este modo no carga SimRobot. Es el camino rapido para train/smoke.

### `mode="simrobot"`

Usa SimRobot embebido:

```text
pybh.simrobot_load_scene(...)
pybh.simrobot_step(...)
pybh.simrobot_unload_scene()
```

Apaga `RLSim2D`.

El reset espera hasta ver la pose reseteada antes de devolver la primera observacion. Esto evita que el agente heuristico lea una observacion vieja `(0,0)` y dispare por error.

### `mode="simrobot_visible"`

Usa una instancia visible de SimRobot:

```text
pybh.rl_reset_world(...)
pybh.rl_set_action(...)
pybh.rl_get_obs(...)
```

El loop de SimRobot visible tiene que estar corriendo. En la ventana, presionar F5.

Este modo tambien apaga `RLSim2D`.

## `RealSimRobotEnv`

Archivo:

```text
RL/environment.py
```

Es un `ParallelEnv` monoagente:

```text
agent = player_1
```

Observacion de 9 floats:

```text
dx pelota-robot
dy pelota-robot
theta robot
ball_x
ball_y
robot_x
robot_y
frame normalizado
sim_enabled
```

Accion:

```text
Box(5,)
```

El entorno:

1. samplea posicion de pelota;
2. pone el robot detras de la pelota;
3. calcula orientacion inicial;
4. llama `backend.reset(...)`;
5. en cada step convierte la accion plana a `StructuredSkillAction`;
6. llama `backend.step(...)`;
7. calcula reward por progreso hacia la pelota y gol.

## Import order importante

Para scripts nuevos:

```python
from pybh_backend import PyBHBackend
import pybh.controller as pybh
```

No importar `pybh.controller` antes de que `pybh_backend` haga `os.chdir(BHUMAN_DIR)`, salvo que el script haga explicitamente:

```python
os.chdir(BHUMAN_DIR)
```

Algunos componentes nativos de B-Human esperan cwd en `SabanaHerons2026`.

