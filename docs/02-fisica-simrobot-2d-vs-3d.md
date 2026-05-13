# Física SimRobot: 2D vs 3D

## Dos perfiles reales

En el proyecto quedan dos perfiles de simulación soportados:

```text
SimRobot 2D
SimRobot 3D
```

## SimRobot 2D

Usa escenas `.ros2d`, por ejemplo:

```text
SabanaHerons2026/Config/Scenes/RL1v0.ros2d
```

Características:

- mundo planar
- pose del robot en 2D
- costo bajo y ciclos estables
- feedback geométrico muy útil para smoke y debug rápido

Es bueno para:

- validar el pipeline `SkillRequest -> MotionRequest`
- revisar resets y world state
- barridos rápidos de skills

## SimRobot 3D

Usa escenas `.ros2`, por ejemplo:

```text
SabanaHerons2026/Config/Scenes/RL1v0_3D.ros2
```

Características:

- cuerpo físico 3D
- articulaciones
- contacto pie-suelo
- inercia y caídas
- `MotionEngine` produciendo `JointRequest` real

Es el perfil correcto para:

- validar movimiento natural
- inspeccionar fases del motor
- revisar comportamiento visual y contacto

## Diferencias prácticas

| Aspecto | SimRobot 2D | SimRobot 3D |
| --- | --- | --- |
| Costo | bajo | alto |
| Cuerpo físico | planar | 3D completo |
| Joints relevantes | limitados | sí |
| Contacto pie-suelo | simplificado | sí |
| Ideal para smoke | sí | parcialmente |
| Ideal para validación visual final | parcialmente | sí |

## Qué mirar al validar

No basta con mirar solo `robot_x`.

Señales fuertes:

- `requested_skill`
- `skill_behavior_skill_request`
- `skill_behavior_motion_request`
- `motion_engine_input_request`
- `motion_engine_effective_request`
- `motion_engine_phase`
- joints de piernas

## Comparativo medido

El detalle cuantitativo vive en [06-simrobot-skill-matrix.md](./06-simrobot-skill-matrix.md).

Resumen de lectura:

- `2d` es más barato y da feedback planar más claro
- `3d` expone mejor el estado interno real del motor
- ambos sirven para validar el wire; `3d` manda para naturalidad
