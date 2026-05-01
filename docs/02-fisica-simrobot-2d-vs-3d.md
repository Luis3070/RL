# Fisica SimRobot: 2D vs 3D

## Tres modos distintos

En este proyecto hay tres cosas que pueden sonar parecidas, pero no lo son:

```text
cpp / RLSim2D
SimRobot 2D
SimRobot 3D
```

## `cpp` con `RLSim2D`

`RLSim2D` vive en:

```text
SabanaHerons2026/Src/Libs/Python/Controller/RLSim2D.h
```

No es SimRobot. Es una simulacion rapida dentro del proceso del controlador.

Caracteristicas:

- no hay cuerpo fisico 3D;
- no hay contacto real pie-suelo;
- no hay articulaciones fisicas;
- no hay caidas fisicas reales;
- el estado es una pose planar `(x, y, theta)`;
- la pelota se propaga con `BallPhysics`;
- la caminata se aproxima con limites de velocidad y paso inspirados en el 2D.

Uso correcto:

- entrenamientos rapidos;
- smoke tests;
- depuracion del contrato RL;
- verificar que el flujo `SkillRequest -> MotionRequest` no se rompe.

No sirve para validar:

- estabilidad de cuerpo 3D;
- naturalidad visual de los pasos;
- inercia real del robot;
- contactos de pies;
- caidas realistas.

## SimRobot 2D

SimRobot 2D usa una escena `.ros2d`, por ejemplo:

```text
SabanaHerons2026/Config/Scenes/RL1v0.ros2d
```

Caracteristicas:

- el mundo es planar;
- la pose del robot se actualiza en 2D;
- no hay dinamica completa de articulaciones 3D;
- el feedback de posicion se obtiene del mundo de SimRobot;
- es mucho mas barato y estable que 3D.

En el estado actual, la escena RL 2D usa:

```text
Config/Scenarios/4v4_RL2D
```

Y respeta el flujo:

```text
RLSkillProvider -> SkillRequest -> SkillBehaviorControl -> MotionRequest
```

En la verificacion embebida 2D, el robot avanzo de aproximadamente:

```text
(-900, 0) -> (-819, 132)
```

en seis acciones heuristicas, con:

```text
sim2d=False
skillctl=(skill=2,motion=4)
engine=(in=4,eff=4)
```

## SimRobot 3D

SimRobot 3D usa una escena `.ros2`, por ejemplo:

```text
SabanaHerons2026/Config/Scenes/RL1v0_3D.ros2
```

Caracteristicas:

- cuerpo fisico 3D;
- articulaciones;
- actuadores;
- contacto pie-suelo;
- inercia;
- caidas;
- `MotionEngine` produce `JointRequest`;
- SimRobot aplica esos joints al robot.

Este es el modo que importa para ver movimiento natural del NAO.

En 3D no se debe mover la raiz del robot manualmente para "ayudar". Si se fuerza la pose global, se ve como deslizamiento y se rompe la fisica.

## Diferencias practicas

| Aspecto | RLSim2D | SimRobot 2D | SimRobot 3D |
| --- | --- | --- | --- |
| Costo | muy bajo | bajo | alto |
| Cuerpo fisico | no | planar | 3D completo |
| Joints reales | no | limitado/no visual 3D | si |
| Contacto pie-suelo | no | simplificado | si |
| Caidas realistas | no | no como 3D | si |
| Ideal para train rapido | si | a veces | no inicialmente |
| Ideal para validar naturalidad visual | no | parcialmente | si |
| Fuente de movimiento | `MotionRequest` final aproximado | SimRobot 2D | `JointRequest` fisico |

## Por que 3D puede moverse lento en headless

En SimRobot 3D embebido, las primeras iteraciones pueden mostrar poco desplazamiento global aunque el motor este caminando:

- el robot puede estar girando primero;
- el motor puede estar en fases iniciales de caminata;
- el suelo/contacto/inercia retardan desplazamiento;
- el reset puede orientar el modelo con `theta_offset`;
- el feedback embebido no reemplaza observar la ventana.

Por eso la validacion minima de flujo no es solo `robot_x`:

```text
skill_behavior_motion_request = 4
motion_engine_input_request = 4
motion_engine_effective_request = 4
motion_engine_phase = walk phase
joint_* cambian con valores de caminata
```

Si esas trazas estan vivas, B-Human esta generando caminata natural.

## Comparativo medido en esta sesion

El detalle completo esta en [06-simrobot-skill-matrix.md](./06-simrobot-skill-matrix.md).

Resumen:

- `2d`: `8/8` skills validadas.
- `3d`: `8/8` skills validadas.
- `2d` RSS promedio del proceso: ~`179774 KB`.
- `3d` RSS promedio del proceso: ~`192248 KB`.
- `2d` step wall promedio: ~`324.6 ms`.
- `3d` step wall promedio: ~`236.5 ms`.

Interpretacion correcta de esos numeros:

- `3d` no necesariamente sale mas lento en esta muestra porque la medicion es del proceso Python completo, no de un benchmark aislado del motor fisico.
- `2d` paga mas overhead relativo en algunas skills porque el bridge y la observacion planar quedan dominando el costo total.
- el `cpu_ms` del proceso puede ser mayor que el `wall_ms` porque `B-Human` ejecuta varios threads y `time.process_time()` acumula CPU consumido por todos ellos.
- el primer reset de `2d` es mas caro porque incluye el costo frio de carga/estabilizacion de escena.
