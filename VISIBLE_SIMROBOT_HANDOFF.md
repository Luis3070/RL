# Handoff: SimRobot visible desde RL

> Nota: este documento queda como historico de handoff. El estado actual
> mantenido esta en [docs/README.md](./docs/README.md), especialmente en
> [docs/04-verificacion-operativa.md](./docs/04-verificacion-operativa.md).

## Contexto

Objetivo actual:

- mantener funcionando `simrobot` embebido/headless en 3D
- hacer que el entorno de `RL` también pueda correr contra un `SimRobot` visible externo, igual que antes se podía visualizar en 2D

Estado real al momento de este handoff:

- el backend 3D headless **sí funciona**
- el path nuevo `simrobot_visible` **compila y arranca**, pero el usuario reporta que **"no se mueve"**
- no cierro todavía la parte visible como resuelta

Branches activas:

- `RL`: `rl-simrobot3d`
- `SabanaHerons2026`: `rl-simrobot3d`


## Qué sí funciona ahora

### 1. Backend 3D embebido/headless

Esto ya estaba funcionando y sigue funcionando:

- `PyBHBackend(mode="simrobot", simrobot_profile="3d")`
- `reset(...)`
- `step(...)`
- escena `Config/Scenes/RL1v0.ros2`

Prueba que sí pasó:

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python .venv/bin/python -c '
import numpy as np
from pybh_backend import PyBHBackend
from action import StructuredSkillAction, SkillType
backend = PyBHBackend(mode="simrobot", simrobot_profile="3d", frames_per_step=1, obs_timeout_ms=2000)
obs = backend.reset(ball_pos=np.array([0.0, 0.0], dtype=np.float32), robot_pos=np.array([-300.0, 0.0], dtype=np.float32), robot_theta=0.0)
print("reset", obs)
obs2 = backend.step(StructuredSkillAction(skill_type=SkillType.stand))
print("step", obs2)
backend.close()
'
```

### 2. Infraestructura nueva para visible backend

Se agregó:

- `mode="simrobot_visible"` en `RL/pybh_backend.py`
- lanzamiento opcional del binario `SimRobot`
- `rl_reset_world(...)` en `pybh.controller`
- canal de reset remoto via `RLSharedState`
- aplicación del reset en `LocalConsole` dentro del simulador visible

Archivos tocados:

- `RL/config.py`
- `RL/environment.py`
- `RL/pybh_backend.py`
- `RL/train.py`
- `RL/smoke_test_skill_requests.py`
- `SabanaHerons2026/Src/Libs/Python/Controller/RLSharedState.h`
- `SabanaHerons2026/Src/Libs/Python/Controller/RLSharedState.cpp`
- `SabanaHerons2026/Src/Libs/Python/Controller/Module.cpp`
- `SabanaHerons2026/Src/Libs/SimulatedNao/LocalConsole.h`
- `SabanaHerons2026/Src/Libs/SimulatedNao/LocalConsole.cpp`

### 3. Verificaciones mecánicas que sí pasaron

- `python3 -m py_compile ...` en `RL`
- `cmake --build Build/Linux/CMake/Develop --target PythonController SimulatedNao`
- el símbolo Python `pybh.rl_reset_world` existe
- `PyBHBackend(mode="simrobot_visible", simrobot_profile="3d", simrobot_launch=False).start()/close()` funciona


## Qué está roto / síntoma reportado

El usuario lanzó el path visible y reportó:

- la ventana abre
- la simulación aparentemente no produce movimiento visible del robot

Texto del usuario:

- "No se mueve"

Importante: yo **no pude validar end-to-end con GUI real** desde este entorno, porque aquí no tengo una sesión visual usable para comprobar la ventana de `SimRobot`.


## Comandos de reproducción

### Visible desde `train.py`

Usar el `.venv`, no `python3` del sistema:

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python .venv/bin/python train.py \
  --backend simrobot_visible \
  --simrobot-profile 3d \
  --mode smoke \
  --smoke-steps 500 \
  --launch-simrobot
```

Después de abrirse la ventana:

- arrancar con `F5`

### Demo visual heurística existente

Esto ya existía como referencia conceptual para “ver un agente ir a la pelota y patear”:

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python .venv/bin/python live_simrobot_probe.py \
  --launch \
  --policy heuristic \
  --simrobot-profile 3d \
  --steps 1000
```


## Hipótesis más probable del problema

La hipótesis más fuerte no es un crash del bridge, sino una mala elección de demo.

### Hipótesis A: `train.py --mode smoke` no es una buena demo visual

En `train.py`, el smoke usa `sample_smoke_action(...)`, que toma la salida de una política `MAPPO` recién inicializada, no entrenada.

Eso significa:

- la acción puede caer cerca de cero
- `skill_type` puede redondear a `0`
- `0` en este stack es básicamente `stand`
- incluso si no da `stand`, puede producir objetivos débiles o poco visibles

Archivo relevante:

- `RL/train.py`

Este punto por sí solo puede explicar perfectamente el “no se mueve”.

### Hipótesis B: el path visible no está consumiendo bien el reset o la observación

Se agregó un reset remoto nuevo:

- Python llama `pybh.rl_reset_world(...)`
- `RLSharedState` guarda `resetPending`
- `LocalConsole::applyPendingRLReset()` hace `moveRobotPerTeam(...)` y `moveBallPerTeam(...)`

Si esto no se aplica en el orden correcto respecto a `update()`, el robot puede quedar quieto o el estado puede no reflejarse como se espera.

Archivos críticos:

- `SabanaHerons2026/Src/Libs/Python/Controller/Module.cpp`
- `SabanaHerons2026/Src/Libs/SimulatedNao/LocalConsole.cpp`

### Hipótesis C: el visible 3D necesita una ruta de demo separada del smoke RL

Aunque `simrobot_visible` sea correcto como backend, el smoke de entrenamiento no es buen sustituto de un “probe” visual.

Lo más probable es que convenga:

- mantener `train.py` para RL real
- tener un demo visible explícito con política heurística o fija


## Qué revisar primero

### 1. Confirmar si el problema es solo de `train.py`

Probar primero:

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python .venv/bin/python live_simrobot_probe.py \
  --launch \
  --policy heuristic \
  --simrobot-profile 3d \
  --steps 1000
```

Casos:

- si **esto sí mueve** el robot, entonces el bridge visible está bastante bien y el problema real está en `train.py --mode smoke`
- si **esto tampoco mueve** el robot, entonces hay un problema real en el path visible `RLSharedState -> LocalConsole -> SimulatedRobot`

### 2. Si el probe heurístico no mueve, instrumentar estos puntos

Puntos a instrumentar:

- `pybh.rl_set_action(...)` desde Python
- `RLSkillProvider::update(SkillRequest&)`
- `io.getSkill()`, `io.targetX`, `io.targetY`, `io.targetTheta`
- `LocalConsole::applyPendingRLReset()`
- `theGroundTruthWorldState` / `robotPose`

La pregunta concreta es:

- ¿están llegando acciones nuevas a `RLSkillProvider`?
- ¿el reset se aplica?
- ¿hay observaciones frescas (`obsReady`)?
- ¿la simulación visible realmente está corriendo después de `F5`?

### 3. Verificar observaciones en el modo visible

En `PyBHBackend.read_obs()`:

- `simrobot_visible` hace solo un intento práctico
- depende de que el simulador visible publique observación por su propio loop

Si hay timeout o si `frame` queda clavado, el problema no es la policy sino el bridge visible.


## Arreglo recomendado

### Opción 1: corregir la demo visible sin tocar el core RL

Esto es lo que yo haría primero.

Objetivo:

- demostrar movimiento visible confiable
- no depender de una política aleatoria sin entrenar

Plan:

1. agregar en `train.py` un modo de smoke especial para `simrobot_visible`
2. si `backend == "simrobot_visible"`, usar:
   - una política heurística simple, o
   - una secuencia fija (`walk`, `walk`, `shoot`, etc.)
3. dejar `MAPPO` smoke solo para pruebas RL headless, no como demo visual

Se puede incluso reutilizar la lógica de:

- `live_simrobot_probe.py`

Esto probablemente resolvería la percepción de “no se mueve” aunque el backend visible ya esté bien.

### Opción 2: separar explícitamente “demo visual” de “train RL”

Más limpio todavía:

1. dejar `train.py` como está para entrenamiento/smoke técnico
2. crear un script nuevo tipo:
   - `visible_rl_demo.py`
3. ese script:
   - arranca `simrobot_visible`
   - hace reset
   - usa política heurística
   - corre continuo hasta `Ctrl+C`

Esto evita mezclar una demo visual con una policy no entrenada.

### Opción 3: si el bridge visible está realmente roto

Si `live_simrobot_probe.py --policy heuristic --simrobot-profile 3d` tampoco mueve:

1. depurar `RLSkillProvider`
2. comprobar que `SkillRequest` está cambiando
3. comprobar que `LocalConsole` recibe/aplica el reset
4. verificar si el robot visible 3D necesita alguna llamada adicional tras `moveRobotPerTeam(...)`
5. revisar si hay que desactivar/reactivar física en otro punto, o resetear más estado del cuerpo/ball


## Mi lectura actual

La lectura más probable es:

- el backend headless 3D está bien
- el bridge visible base no necesariamente está roto
- el usuario probó una ruta (`train.py --mode smoke`) que no garantiza movimiento visible, porque usa una policy sin entrenar

Por eso, la primera verificación útil no es seguir parchando `LocalConsole`, sino probar el visible con la policy heurística de `live_simrobot_probe.py`.


## Cosas a no romper

- no revertir cambios ajenos en:
  - `SabanaHerons2026/Util/CompiledNN`
  - `SabanaHerons2026/Util/SimRobot`
- no commitear `Make/Python/pybh/controller.cpython-312-x86_64-linux-gnu.so`
  - ese binario ya causó rechazo por tamaño en GitHub
  - mantenerlo local solamente


## Estado del árbol al cerrar este handoff

### `RL`

Cambios sin commit:

- `config.py`
- `environment.py`
- `pybh_backend.py`
- `smoke_test_skill_requests.py`
- `train.py`

### `SabanaHerons2026`

Cambios sin commit:

- `Src/Libs/Python/Controller/Module.cpp`
- `Src/Libs/Python/Controller/RLSharedState.cpp`
- `Src/Libs/Python/Controller/RLSharedState.h`
- `Src/Libs/SimulatedNao/LocalConsole.cpp`
- `Src/Libs/SimulatedNao/LocalConsole.h`
- `Make/Python/pybh/controller.cpython-312-x86_64-linux-gnu.so` (local, **no commit**)

Cambios previos ajenos:

- `Util/CompiledNN`
- `Util/SimRobot`


## Siguiente paso recomendado para el agente que toma esto

1. correr `live_simrobot_probe.py --launch --policy heuristic --simrobot-profile 3d`
2. observar si el robot visible se mueve
3. si sí se mueve:
   - arreglar `train.py` para que `simrobot_visible` use una demo determinista y no una policy aleatoria
4. si no se mueve:
   - instrumentar `RLSkillProvider` y `LocalConsole::applyPendingRLReset()`
   - verificar flujo `rl_set_action -> SkillRequest -> movimiento visible`
