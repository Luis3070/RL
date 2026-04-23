# RL × B-Human con `pybh`

Esta rama (`pybh-clean`) ya no usa el bridge viejo por subprocess, JSON ni
sidecar. Ahora Python y B-Human viven en el mismo proceso: Python manda una
acción, B-Human corre Cognition y Python recibe la observación de vuelta.

La idea de esta rama no es "tener todo el stack RL final", sino dejar clara y
estable la integración con B-Human y con la simulación 2D.

## Estado actual

Hoy hay tres capas distintas y conviene separarlas bien:

1. `pybh` corre Cognition real de B-Human dentro del proceso Python.
2. El experimento principal usa un backend 2D headless en C++ (`RLSim2D`)
   dentro de `SabanaHerons2026`, inspirado en `SimulatedRobot2D`.
3. Además ya existe un host embebido de SimRobot 2D real dentro de `pybh` para
   cargar escenas `.ros2d`, hacer `step()` y leer/mover pelota y robots desde
   Python.

Eso significa:

- el cerebro del robot sí es B-Human real
- el experimento por defecto ya no depende de una física Python inventada
- ya podemos correr un smoke sobre escena real de SimRobot
- pero el loop RL todavía no está conectado por completo al body real del robot
  dentro del scene graph de SimRobot

## Qué sí funciona

- `ctrl.update()` corresponde a un frame real de Cognition.
- `rl_set_action()` y `rl_get_obs()` ya están sincronizados.
- El experimento heurístico corre múltiples episodios sin segfault.
- El backend C++ `RLSim2D` mueve robot y balón desde `SabanaHerons2026`.
- El host nuevo de SimRobot carga escenas reales 2D como `BH2D.ros2d` y
  `Game2D.ros2d`.
- Desde Python ya se puede:
  - cargar una escena real
  - avanzar la simulación
  - leer pose de robot
  - mover robot
  - leer estado de pelota
  - mover pelota y darle velocidad

## Qué no está cerrado todavía

La parte importante que sigue pendiente es esta:

- el modo por defecto (`sim=cpp`) usa `RLSim2D`, no el runtime completo de
  SimRobot
- el modo `--simrobot-real` sí usa escena real de SimRobot para pelota y scene
  graph
- pero el robot del experimento todavía no camina "desde adentro" del
  `ControllerRobot` de SimRobot; por ahora el experimento sigue reubicando el
  cuerpo con la cinemática portada

En corto: ya hay SimRobot real embebido, pero todavía no está conectado de
punta a punta al loop RL actual.

## Archivos importantes

| Archivo | Rol |
|---|---|
| `heuristic_agent_experiment.py` | experimento actual del agente heurístico |
| `heuristic_experiment_results.csv` | resultados de la última corrida |
| `environment.yml` | entorno para correr Python |

Del lado C++ en `SabanaHerons2026/`, lo más relevante ahora es:

- `Src/Libs/Python/Controller/Module.cpp`
- `Src/Libs/Python/Controller/PythonConsole.{h,cpp}`
- `Src/Libs/Python/Controller/RLSharedState.h`
- `Src/Libs/Python/Controller/RLSim2D.h`
- `Src/Libs/Python/Controller/SimRobotHost.{h,cpp}`
- `Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.cpp`

## Modos del experimento

`heuristic_agent_experiment.py` hoy tiene tres modos:

- default: backend C++ `RLSim2D`
- `--python-sim`: fallback viejo con física Python
- `--simrobot-real`: smoke sobre escena real de SimRobot 2D

Lo útil de cada uno:

- `sim=cpp`: rápido, estable y sirve para iterar
- `sim=python`: sólo fallback
- `sim=simrobot`: valida que el host embebido de SimRobot realmente carga y
  avanza una escena real

## Resultados actuales

### Backend C++ por defecto (`sim=cpp`)

Corrida validada:

```bash
python3 heuristic_agent_experiment.py --episodes 100 --no-plot
```

Resultado:

- `goal_rate = 0.630`
- `handoff_rate = 1.000`
- `shoot_rate = 1.000`
- `median_shoot_step = 44`
- `0.03s/episodio`

### SimRobot real (`--simrobot-real`)

Smoke validado:

```bash
python3 heuristic_agent_experiment.py --episodes 10 --no-plot --simrobot-real
```

Resultado:

- `goal_rate = 0.600`
- `handoff_rate = 1.000`
- `shoot_rate = 1.000`
- `median_shoot_step = 44`
- `3.29s/episodio`

Este modo es bastante más lento, que era esperable: ahí sí se está cargando una
escena real de SimRobot.

## Warning conocido

En el modo `--simrobot-real` aparece este warning:

```text
Error: Cognition: Module InitialToReadyHandler is unknown!
```

Eso no viene del `pybh` nuevo sino de `libSimulatedNao.so`, que está cargando
una config que pide `InitialToReadyHandler`, pero el plugin `SimulatedNao`
disponible en `Build/Linux/SimRobot/Develop` quedó desalineado con el source
actual. El experimento igual corre, pero el warning sigue ahí hasta recompilar
`SimulatedNao`.

## Cómo correrlo

### 1. Rebuild de `pybh`

```bash
cd /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python
python3 setup.py build_ext --inplace
cp build/lib.linux-x86_64-cpython-312/pybh/*.so ~/.local/lib/python3.12/site-packages/pybh/
```

Nota:

- `setup.py build_ext --inplace` todavía termina con un bug viejo de copiado
  al final
- aun así, el `.so` sí se genera bien en `build/lib.../pybh/`
- por eso se copia manualmente a `site-packages`

### 2. Correr el experimento

```bash
cd /home/rafael/Documentos/Robocup2026/RL

# backend C++ principal
python3 heuristic_agent_experiment.py --episodes 100 --no-plot

# fallback viejo
python3 heuristic_agent_experiment.py --episodes 100 --no-plot --python-sim

# smoke sobre SimRobot real
python3 heuristic_agent_experiment.py --episodes 10 --no-plot --simrobot-real
```

Importante:

- `--simrobot-real` necesita correr fuera del sandbox porque `SimulatedNao`
  abre UDP

## Qué sigue

Los siguientes pasos razonables son:

1. decidir si el entrenamiento va a montarse sobre `RLSim2D` o sobre el host
   real de SimRobot
2. conectar de verdad el robot del experimento al body real del scene graph
3. portar esto a un `gymnasium.Env`
4. recuperar wrapper multiagente y rewards del stack viejo
5. montar entrenamiento RL encima de esa base

## Referencia histórica

Si hace falta mirar el enfoque anterior:

- `main` y `rlsimrobot` conservan el stack viejo más grande, basado en sidecar
  y piezas más acopladas

La base más limpia hoy para seguir trabajando es esta rama.
