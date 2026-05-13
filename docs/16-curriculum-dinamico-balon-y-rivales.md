# Curriculum dinamico: balon y rivales

Este handoff resume como extender el stack actual RL x B-Human para mover
balones y jugadores de forma dinamica, con foco en curriculum learning.

La idea central es separar dos casos:

1. cambios de mundo al inicio del episodio o del reset
2. cambios de mundo durante el episodio

El primer caso ya esta medio resuelto para balon + robot propio.
El segundo caso todavia necesita API nueva.

## Estado actual

Hoy existen dos rutas utiles para modificar el mundo desde Python:

1. `pybh.set_world_state(...)`
2. `pybh.rl_reset_world(...)`

Y existe una tercera ruta headless:

3. `pybh.rl_sim_reset(...)`

### `set_world_state(...)`

Esta funcion vive en:

- [/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/Module.cpp](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/Module.cpp:166)

Hoy solo inyecta:

- una pelota en `GroundTruthWorldState::balls`
- la pose propia en `GroundTruthWorldState::ownPose`

No llena:

- `ownTeamPlayers`
- `opponentTeamPlayers`

Eso significa que la representacion soporta rivales, pero la API expuesta
todavia no los manda.

### `rl_reset_world(...)`

Esta funcion tambien vive en:

- [/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/Module.cpp](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/Module.cpp:220)

Hoy solo escribe en `RLSharedState`:

- `resetBallX`
- `resetBallY`
- `resetRobotX`
- `resetRobotY`
- `resetRobotTheta`

Luego el reset visible de SimRobot se aplica en:

- [/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/SimulatedNao/LocalConsole.cpp](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/SimulatedNao/LocalConsole.cpp:333)

Y ahi solo se hace:

- `moveRobotPerTeam(...)`
- `moveBallPerTeam(...)`

O sea:

- en SimRobot visible hoy se puede mover el robot RL y la pelota
- no se pueden reposicionar rivales arbitrarios desde Python

### `rl_sim_reset(...)`

Sirve para el backend headless 2D:

- resetea balon + robot propio
- no modela todavia listas arbitrarias de rivales

### La representacion ya soporta rivales

La pieza importante es:

- [/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Representations/Infrastructure/GroundTruthWorldState.h](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Representations/Infrastructure/GroundTruthWorldState.h:19)

`GroundTruthWorldState` ya tiene:

- `ownTeamPlayers`
- `opponentTeamPlayers`
- `balls`
- `ownPose`

Y el pipeline oracle/perceptual ya consume esos jugadores para:

- `ObstacleModel`
- `GlobalOpponentsModel`
- perceptos de obstaculos

Eso es importante porque el cuello ya no es conceptual: el motor sabe manejar
jugadores en `GroundTruthWorldState`. Lo que falta es exponerlos desde el bridge.

## Que backend conviene para cada cosa

### Backend `python`

Es el mas facil para curriculum dinamico rico.

Ventajas:

- ya usa `GroundTruthWorldState`
- no depende de resets de GUI
- es el lugar mas directo para inyectar oponentes/companeros/varias pelotas

Desventajas:

- no replica el flujo visible completo de SimRobot
- es menos util para depuracion visual fina

### Backend `simrobot`

Es el correcto para depurar percepcion, avoidance y skills internas.

Ventajas:

- usa el pipeline real visible
- permite validar escenas, percepcion, world model y behavior

Desventajas:

- hoy el bridge visible solo soporta mover robot propio + balon
- para mover rivales hay que extender `RLSharedState` y `LocalConsole`

### Backend `cpp`

Es el mas rapido para experimentar curriculum base.

Ventajas:

- episodios baratos
- buena opcion para barridos iniciales de posiciones

Desventajas:

- menos fiel que SimRobot visible
- no es el mejor sitio para validar skills internas complejas

## Recomendacion de implementacion

### Fase 1: curriculum de reset

Objetivo:

- cambiar el mundo solo al inicio de cada episodio

Esto ya da mucho valor para:

- distancia robot-balon
- angulo hacia arco
- balon cerca/lejos
- balon lateral/centrado
- obstaculo fijo en la trayectoria
- numero de rivales discretizado por etapa

#### Opcion A: empezar por `python`

Extender `set_world_state(...)` para que reciba listas de jugadores.

API sugerida:

```python
pybh.set_world_state_extended(
    controller,
    robot_index,
    ball_x,
    ball_y,
    robot_x,
    robot_y,
    robot_theta,
    opponents=[(number, x, y, theta, upright), ...],
    teammates=[(number, x, y, theta, upright), ...],
)
```

Internamente deberia poblar:

- `ws.ownPose`
- `ws.balls`
- `ws.opponentTeamPlayers`
- `ws.ownTeamPlayers`

Esto es lo mas barato para tener curriculum real con rivales.

#### Opcion B: extender `rl_reset_world(...)` para SimRobot visible

Crear una variante mas rica, por ejemplo:

```python
pybh.rl_reset_world_extended(
    player_number,
    ball_x,
    ball_y,
    robot_x,
    robot_y,
    robot_theta,
    opponents=[...],
    teammates=[...],
)
```

Para eso hay que tocar al menos:

- `Src/Libs/Python/Controller/Module.cpp`
- `Src/Libs/Python/Controller/RLSharedState.*`
- `Src/Libs/SimulatedNao/LocalConsole.cpp`

Y ampliar el shared state con arreglos fijos o buffers serializados para:

- lista de rivales
- lista de companeros
- cantidad valida de cada lista

Luego `LocalConsole::applyPendingRLReset()` deberia:

1. mover el robot propio
2. mover la pelota
3. mover los rivales uno por uno
4. mover companeros extra si aplica
5. reactivar physics

### Fase 2: curriculum dinamico intra-episodio

Objetivo:

- mover rivales o balones durante el episodio

Ejemplos:

- rival que se cruza a mitad de la conduccion
- rival que persigue la pelota
- balon que aparece en distinta region tras un evento
- oponente que cambia de lateral segun etapa del curriculum

Esto no deberia montarse sobre una API llamada `reset`.

Lo correcto es una API separada, por ejemplo:

```python
pybh.rl_set_dynamic_world(
    player_number,
    opponents=[...],
    teammates=[...],
    balls=[...],
    own_pose=(x, y, theta) | None,
)
```

O, en backend `python`:

```python
pybh.set_world_state_extended(...)
```

llamada en cada frame antes de `controller.update()`.

### Fase 3: movimiento por politica de curriculum

Una vez exista la API de mundo extendido, el lado Python puede quedar con dos
funciones simples:

```python
def sample_curriculum_world(stage, rng) -> dict:
    ...

def apply_curriculum_world(backend, world_spec) -> None:
    ...
```

`sample_curriculum_world(...)` deberia producir una especificacion abstracta:

- pose del robot propio
- pose del balon
- lista de rivales
- lista de companeros
- metadatos del escenario

Y `apply_curriculum_world(...)` la traduce segun backend:

- `python`: `set_world_state_extended(...)`
- `simrobot`: `rl_reset_world_extended(...)` o `rl_set_dynamic_world(...)`
- `cpp`: funcion equivalente en `RLSim2D`

## Que debe tener una especificacion de curriculum

Minimo:

- `robot_pose = (x, y, theta)`
- `ball_pose = (x, y)`
- `opponents = [pose, ...]`

Recomendado:

- `teammates = [pose, ...]`
- `stage`
- `scenario_id`
- `flags`

Ejemplo:

```python
world_spec = {
    "robot_pose": (x, y, theta),
    "ball_pose": (bx, by),
    "opponents": [
        {"number": 21, "pose": (ox, oy, otheta), "upright": True},
    ],
    "teammates": [],
    "stage": 3,
    "flags": {
        "force_obstacle_in_path": True,
        "allow_dynamic_updates": False,
    },
}
```

## Propuesta de curriculum por etapas

### Etapa 0

- sin rivales
- balon siempre visible
- distancias cortas

### Etapa 1

- un rival fijo, fuera de la trayectoria
- solo para acostumbrar percepcion de obstaculos

### Etapa 2

- un rival en la trayectoria balon-arco
- offset lateral pequeno

### Etapa 3

- un rival en la trayectoria
- otro rival lateral
- angulos de llegada distintos

### Etapa 4

- rivales en posiciones que fuerzan avoidance real
- robot y balon con orientaciones variadas
- escenarios cerca de banda y cerca de arco

### Etapa 5

- rivales dinamicos durante el episodio
- curriculum ya no solo de reset sino de evento

## Que cambiar si solo queremos variar posiciones al reset

Si el objetivo inmediato es curriculum simple, no hace falta tocar todo.

Alcanza con:

1. definir `sample_curriculum_world(stage, rng)` en Python
2. para `simrobot`, seguir usando escenas base mas `rl_reset_world(...)`
3. para `python`, extender `set_world_state(...)` con rivales

Eso ya permite:

- entrenamiento barato sin editar escenas manualmente
- escenarios reproducibles por semilla
- barridos masivos de posiciones

## Que cambiar si queremos mover jugadores durante el episodio

Si el objetivo es mover rivales frame a frame, entonces si hace falta API nueva.

En visible SimRobot el cambio correcto es:

1. agregar datos dinamicos de rivales/companeros a `RLSharedState`
2. exponer una funcion nueva en `Module.cpp`
3. consumir esa informacion en `LocalConsole` antes del `update`
4. mover entidades con `SimulatedRobot`

En el backend `python`, la extension es mas simple porque `Controller::setWorldState`
ya acepta un `GroundTruthWorldState` completo.

## Que ya sabemos por la depuracion reciente

- la posicion de `robot1`, extras y pelota se define primero en la escena
- si Python llama `rl_reset_world(...)`, esa pose puede ser pisada
- `--no-reset` sirve para respetar la escena inicial
- los rivales ya aparecen en `ObstacleModel` y ya afectan avoidance
- por tanto, curriculum con rivales ya tiene sentido real en este stack

## Recomendacion concreta

Orden pragmatico:

1. implementar `set_world_state_extended(...)` en el backend `python`
2. crear `sample_curriculum_world(...)` y `apply_curriculum_world(...)` en `RL/`
3. entrenar/probar curriculum de reset con rivales
4. si hace falta validacion visual, extender luego `rl_reset_world_extended(...)`
   para SimRobot visible
5. dejar la API intra-episodio para una tercera iteracion

Ese orden minimiza riesgo y desbloquea rapido curriculum realista sin depender
primero de la GUI.
