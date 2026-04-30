# Estado Actual

## Repos involucrados

El proyecto se divide en dos repos hermanos:

- `RL`: entorno, entrenamiento, heuristica, wrapper Python y scripts de prueba.
- `SabanaHerons2026`: B-Human, SimRobot, modulos C++ y binding `pybh.controller`.

La ruta esperada es:

```text
/home/rafael/Documentos/Robocup2026/RL
/home/rafael/Documentos/Robocup2026/SabanaHerons2026
```

## Objetivo actual

El objetivo ya no es mover el robot con un bypass de bajo nivel. El objetivo actual es:

```text
RL/Python
  -> RLSkillProvider
  -> SkillRequest
  -> SkillBehaviorControl
  -> MotionRequest
  -> MotionEngine / WalkingEngine
  -> JointRequest
  -> SimRobot
```

Esto aplica a:

- SimRobot 2D embebido.
- SimRobot 3D embebido.
- SimRobot visible, cuando la ventana esta corriendo con F5.
- Backend rapido `cpp`, usando `RLSim2D` como simulador barato, pero alimentado desde el `MotionRequest` final.

## Escenarios RL

Hay escenarios dedicados para no contaminar escenarios normales de B-Human:

```text
SabanaHerons2026/Config/Scenarios/4v4_RL2D
SabanaHerons2026/Config/Scenarios/4v4_RL3D
```

Ambos tienen:

```cfg
{representation = SkillRequest; provider = RLSkillProvider;}
{representation = StrategyStatus; provider = RLSkillProvider;}
{representation = MotionRequest; provider = SkillBehaviorControl;}
```

Y en `skillBehaviorControl.cfg`:

```cfg
options = [
  HandleRLRequest,
];

playingOptions = [];

useNewHandleCatchBallBehavior = true;
```

## Escenas RL

Las escenas principales son:

```text
SabanaHerons2026/Config/Scenes/RL1v0.ros2d
SabanaHerons2026/Config/Scenes/RL1v0_3D.ros2
```

`RL1v0.ros2d` apunta a `4v4_RL2D`.

`RL1v0_3D.ros2` apunta a `4v4_RL3D`.

## Configuracion Python

`RL/config.py` contiene:

- `SCENARIO = "4v4_RL2D"` para el backend `cpp`.
- preset `simrobot_profile="2d"` con escena `RL1v0.ros2d`.
- preset `simrobot_profile="3d"` con escena `RL1v0_3D.ros2`.
- `position_scale = -1.0 / 1000.0`.
- `theta_offset = -pi` para 2D.
- `theta_offset = -pi / 2` para 3D.

## Backends

`PyBHBackend` soporta:

- `cpp`: controlador B-Human con simulacion rapida `RLSim2D`.
- `python`: legacy, inyecta `GroundTruthWorldState` desde Python.
- `simrobot`: SimRobot embebido.
- `simrobot_visible`: SimRobot visible externo, controlado por memoria compartida.

Para `simrobot` y `simrobot_visible`, `pybh_backend.py` apaga explicitamente `RLSim2D`:

```python
pybh.rl_sim_set_enabled(PLAYER_NUMBER, False)
```

Eso evita mezclar el simulador rapido 2D con SimRobot real.

## Estado verificado

Verificado en esta sesion:

- `py_compile` de scripts principales de `RL`: OK.
- SimRobot 2D embebido con agente heuristico: OK.
- SimRobot 3D embebido con agente heuristico: OK en flujo natural.
- Escena 2D y escena 3D cargan.
- `controller.so` fue reconstruido y strippeado; queda alrededor de 74 MB.

Limitacion honesta:

- La verificacion visible con GUI requiere una sesion grafica y presionar F5 en SimRobot. Desde automatizacion se valido el motor embebido y las trazas nativas; para la ventana visible se debe correr `live_simrobot_probe.py` manualmente y observar el robot.

