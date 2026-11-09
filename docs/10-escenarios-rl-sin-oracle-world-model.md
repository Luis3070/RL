# Escenarios RL sin Oracle World Model

## Objetivo

Este documento registra la migracion inicial de los escenarios RL dedicados para dejar de alimentar PPO con representaciones principales desde `OracledWorldModelProvider`.

El cambio aplica a:

- `Config/Scenarios/4v4_RL2D/threads.cfg`
- `Config/Scenarios/4v4_RL3D/threads.cfg`

La razon es directa: despues de ampliar el observation space a 23 variables, el siguiente paso era que esas variables vinieran de providers naturales de B-Human y no de ground truth oracular.

## Cambios de provider

| Representacion | Antes | Ahora |
| --- | --- | --- |
| `BallModel` | `OracledWorldModelProvider` | `BallStateEstimator` |
| `GlobalOpponentsModel` | `OracledWorldModelProvider` | `GlobalOpponentsTracker` |
| `ObstacleModel` | `OracledWorldModelProvider` | `ObstacleModelProvider` |
| `RobotPose` | `OracledWorldModelProvider` | `SelfLocator` |
| `TeammatesBallModel` | `OracledWorldModelProvider` | `TeammatesBallModelProvider` |

`FieldBall` ya usaba `FieldBallProvider` y se mantiene asi.

`TeamData` ya usaba `TeamDataProvider` y se mantiene asi.

## Que significa para PPO

El agente deja de recibir un mundo geometricamente perfecto.

Ahora las features del vector PPO 23 dependen de:

- percepcion de pelota,
- estimacion de estado de pelota,
- localizacion propia,
- percepcion/modelado de obstaculos,
- trackers de oponentes y teammates.

Eso cambia la semantica de algunas senales:

- `ball_seen_percentage` puede ser `0`.
- `time_since_ball_seen` puede crecer.
- `ball_rel_x` / `ball_rel_y` pueden quedar stale o degradadas si la pelota no esta siendo observada.
- `RobotPose` puede tardar en estabilizarse.
- Obstaculos pueden aparecer como inciertos o no aparecer.

Ese comportamiento es deseado para sim2real. Una politica que solo funciona con `ball_seen_percentage=100` o pose perfecta no es una politica robusta.

## Validacion realizada

Se cargo `RL1v0.ros2d`, que usa `4v4_RL2D`.

Se cargo `RL1v0_3D.ros2`, que usa `4v4_RL3D`.

Ambas escenas cargaron correctamente con SimRobot embebido y devolvieron observaciones por `rl_get_obs(...)`.

Resultado esperado despues de quitar oracle:

- `obs_ready=True`
- el frame avanza
- las features pueden iniciar en valores pobres o stale
- `ball_seen_percentage` puede estar en `0`
- en 3D la pose/pelota se estabilizan despues de algunos frames

## Riesgos introducidos

### 1. Heuristicas antiguas pueden parecer peores

El probe heuristico asume que `ball_x`, `ball_y`, `robot_x`, `robot_y` son suficientes para decidir caminar o tirar.

Con providers naturales, esa suposicion ya no siempre es valida.

La politica futura debe usar tambien:

- frescura de pelota,
- porcentaje visto,
- consistencia con game state,
- distancia de obstaculos,
- degradacion temporal.

### 2. Entrenamiento puede necesitar warmup

Los primeros frames despues de cargar/resetear escena pueden tener observaciones pobres.

Para entrenamiento PPO puede ser necesario:

- ignorar frames iniciales,
- penalizar decisiones con pelota no confiable,
- agregar fallback B-Human cuando la pelota no esta vista,
- o iniciar episodios solo cuando la observacion pase un criterio minimo.

### 3. El reward actual sigue siendo limitado

El reward aun usa principalmente distancia relativa a la pelota y progreso.

Con observaciones realistas, el reward deberia distinguir entre:

- acercarse a una pelota confiable,
- perseguir una pelota stale,
- decidir buscar pelota,
- evitar obstaculos,
- preparar tiro cuando la ventana ofensiva sea real.

## Criterio de exito

Esta fase se considera correcta si:

- los escenarios RL ya no usan `OracledWorldModelProvider` para las representaciones que alimentan PPO,
- SimRobot carga las escenas RL,
- `rl_get_obs(...)` sigue entregando observaciones,
- el flujo natural `SkillRequest -> SkillBehaviorControl -> MotionRequest -> MotionEngine` se mantiene,
- y los problemas restantes se tratan como problemas de politica/reward/fallback, no como falta de realismo del observation space.

