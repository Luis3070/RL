# Contrato RL, Backend y Entorno

## Contrato de acción

La acción canónica está en `RL/action.py`.

La acción plana tiene cinco campos:

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

`pass` se llama `pass_` solo dentro de Python para evitar la keyword.

## Coordenadas

Regla de oro:

```text
SkillRequest::walkTo usa coordenadas globales de campo.
```

Por eso `adapt_action_to_backend(...)` hoy deja pasar las acciones sin
transformación adicional.

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
requested_skill
requested_pass_target
```

Campos principales del estado actual:

```text
ball_rel_x
ball_rel_y
ball_end_rel_x
ball_end_rel_y
ball_vel_x
ball_vel_y
time_since_ball_seen
time_since_ball_disappeared
ball_seen_percentage
ball_consistent_with_game_state
can_score_now
shot_quality_no_obstacles
shot_opening_with_obstacles
pass_options_count
nearest_teammate_dist
nearest_opponent_dist
nearest_uncertain_obstacle_dist
nearest_teammate_front_dist
nearest_opponent_front_dist
nearest_uncertain_front_dist
```

## `RLSharedState`

Archivo:

```text
SabanaHerons2026/Src/Libs/Python/Controller/RLSharedState.h
```

Propiedades actuales:

- memoria compartida POSIX
- nombre actual: `/pybh_rl_shared_state_v3`
- versión actual: `25`
- mutex robusto compartido entre procesos
- semáforo `obsSignal`
- buffers para reset extendido de teammates y opponents

## Backends soportados por `PyBHBackend`

### `mode="simrobot"`

Usa SimRobot embebido:

```text
pybh.simrobot_load_scene(...)
pybh.simrobot_step(...)
pybh.simrobot_unload_scene()
```

El reset espera observación fresca alineada con la pose pedida.

### `mode="simrobot_visible"`

Usa una instancia visible de SimRobot:

```text
pybh.rl_reset_world_extended(...)
pybh.rl_set_action(...)
pybh.rl_get_obs(...)
```

Requiere una ventana viva.

## Contrato de mundo

`PyBHBackend.reset(...)` ya acepta:

```text
ball_pos
robot_pos
robot_theta
teammates
opponents
```

Los jugadores se expresan como:

```text
(number, x, y, theta, upright)
```

Reglas:

- números `1..20`
- no duplicados por lista
- el robot controlado no va en `teammates`
- `upright=False` no está soportado en las rutas activas

## `RealSimRobotEnv`

Archivo:

```text
RL/environment.py
```

Es un `ParallelEnv` monoagente con:

- `agent = player_1`
- `action_space = Box(5,)`
- observación de `23` features
- estado global igual a la observación actual

El vector de observación está definido en `RL/observation.py`.

## Import order importante

Para scripts nuevos:

```python
from pybh_backend import PyBHBackend
import pybh.controller as pybh
```

O explícitamente:

```python
import os
from config import BHUMAN_DIR
os.chdir(BHUMAN_DIR)
import pybh.controller as pybh
```
