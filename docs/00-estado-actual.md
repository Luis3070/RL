# Estado Actual

## Repos involucrados

El proyecto se divide en dos repos hermanos:

- `RL`: entorno, entrenamiento, heurísticas y scripts Python.
- `SabanaHerons2026`: B-Human, SimRobot, módulos C++ y binding `pybh.controller`.

La ruta esperada es:

```text
/home/rafael/Documentos/Robocup2026/RL
/home/rafael/Documentos/Robocup2026/SabanaHerons2026
```

## Objetivo actual

El objetivo activo es entrenar y depurar políticas que pidan `SkillRequest`
coherentes, manteniendo el movimiento natural de B-Human:

```text
RL / Python
  -> RLSharedState
  -> RLSkillProvider
  -> SkillRequest
  -> SkillBehaviorControl
  -> MotionRequest
  -> MotionEngine / WalkingEngine
  -> JointRequest
  -> SimRobot
```

Esto aplica a:

- SimRobot embebido perfil `2d`
- SimRobot embebido perfil `3d`
- SimRobot visible con `F5`

## Escenarios y escenas RL

Escenarios dedicados:

```text
SabanaHerons2026/Config/Scenarios/4v4_RL2D
SabanaHerons2026/Config/Scenarios/4v4_RL3D
```

Escenas principales:

```text
SabanaHerons2026/Config/Scenes/RL1v0.ros2d
SabanaHerons2026/Config/Scenes/RL1v0_3D.ros2
```

Los escenarios RL mantienen:

- `SkillRequest` provisto por `RLSkillProvider`
- `MotionRequest` provisto por `SkillBehaviorControl`
- `HandleRLRequest` como puerta de entrada del request externo

## Configuración Python

`RL/config.py` define hoy:

- `simrobot_profile="2d"` con escena `RL1v0.ros2d`
- `simrobot_profile="3d"` con escena `RL1v0_3D.ros2`
- `position_scale = -1.0 / 1000.0`
- `theta_offset = -pi` para `2d`
- `theta_offset = -pi / 2` para `3d`

## Superficie soportada

`PyBHBackend` soporta solo dos modos:

- `simrobot`: SimRobot embebido en el proceso Python
- `simrobot_visible`: SimRobot visible controlado por `RLSharedState`

Ambos consumen el mismo contrato de acción y el mismo payload de observación.

## Wire de mundo

El reset extendido de mundo ya soporta:

- pelota
- robot RL
- teammates
- opponents

Rutas disponibles:

- `PyBHBackend.reset(... teammates=..., opponents=...)` en modo embebido
- `pybh.rl_reset_world_extended(...)` para SimRobot visible

Además existe `pybh.set_world_state_extended(...)` como utilidad de controller.

## Wire dinámico

También existe actualización de mundo intra-episodio:

- `PyBHBackend.set_dynamic_world(...)` en modo embebido y visible
- `pybh.rl_set_dynamic_world(...)` para SimRobot visible
- `pybh.rl_get_world_status(...)` para esperar confirmación o error del request visible

En `simrobot_visible`, `LocalConsole` valida que los slots pedidos existan en la
escena antes de mover nada. Si falta un robot, el request falla con error claro
en vez de aplicar un reset parcial.

## Estado verificado

Verificado en esta sesión:

- `py_compile` de scripts principales de `RL`: OK
- rebuild de `pybh/controller`: OK
- el binding ya no expone funciones antiguas de simulación interna
- prueba mínima embebida `pybh.simrobot_load_scene(...)`:
  - `RL1v0.ros2d`: OK
  - `RL1v0_3D.ros2`: OK
  - `BH.ros2`: OK
- smoke embebido `simrobot` fuera del sandbox:
  - `2d`: `obs_ready=True`, `frame` avanza, pelota consistente con reset
  - `3d`: `obs_ready=True`, `frame` avanza, pelota consistente con reset

Limitación honesta:

- la validación visible con GUI sigue requiriendo una sesión gráfica y `F5`
