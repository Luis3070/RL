# Wire RL realista final

## Objetivo

Este documento cierra la fase de wiring del entorno RL hacia B-Human con foco en:

- eliminar dependencias activas de `GroundTruth` del flujo RL,
- dejar una sola fuente canónica del payload de observación,
- y validar que las skills sigan entrando al pipeline natural.

## Estado final del flujo

El flujo activo queda así:

```text
Python
  -> RLSharedState
  -> RLSkillProvider
  -> SkillRequest
  -> SkillBehaviorControl
  -> MotionRequest
  -> MotionEngine
  -> SimRobot
```

Regla de arquitectura que queda vigente:

```text
RL decide qué skill pedir.
B-Human decide cómo ejecutarla.
```

## Cambios cerrados

### 1. `GroundTruth` removido del flujo RL activo

Se removieron dependencias activas a `GroundTruth` en:

- `RLSkillProvider`
- `SelfLocator`
- `RLNatural.con`
- `4v4_RL2D/threads.cfg`
- `4v4_RL3D/threads.cfg`

Eso deja los escenarios RL activos apoyados en providers naturales para:

- `RobotPose`
- `BallModel`
- `FieldBall`
- `ObstacleModel`
- `TeammatesBallModel`

## 2. Publicador canónico de observaciones

El payload observado por Python ahora se publica en un solo lugar:

- `SkillBehaviorControl`

`RLSkillProvider` queda reducido a su responsabilidad real:

- leer la acción RL desde `RLSharedState`
- producir `SkillRequest`
- producir `StrategyStatus`

Ya no mezcla bridge de acción con publicación del `obs`.

## 3. Features ofensivas básicas ya no quedan muertas

Las features:

- `can_score_now`
- `shot_quality_no_obstacles`
- `shot_opening_with_obstacles`

dejaron de publicarse como `false/0/0` en el flujo natural.

Ahora usan:

- `ExpectedGoals.xG(ballOnField)`
- `ExpectedGoals.getRating(ballOnField)`

con el gating mínimo:

```text
can_score_now = shot_opening_with_obstacles > 0.8
```

## Validación de skills

Se ejecutó la matriz de skills sobre `simrobot` embebido en perfiles `2d` y `3d`.

Resultado resumido:

- las 8 skills validaron en `2d` y `3d`
- `shoot` acepta `walkToBallAndKick` o `dribble`
- `pass` acepta `walkToBallAndKick` o `dribble`

Interpretación:

- el wire de RL hacia `SkillRequest` quedó correcto para todas las skills
- en B-Human, `shoot` puede degradar a `dribble` dentro de `KickAtGoal`
- en `RL1v0`, `pass` puede degradar a `dribble` porque no existe teammate válido y `PassToTeammate` aborta hacia `KickAtGoal`
- eso se documenta como comportamiento aguas abajo, no como falla del bridge

La matriz actualizada quedó en:

- `docs/06-simrobot-skill-matrix.md`

## Qué sigue fuera de esta fase

Esto ya no pertenece al alcance del wire, sino al equipo de RL o a una fase posterior:

- decidir si el backend `cpp` se usa solo para smoke o también para entrenamiento
- mejorar la semántica de `pass_options_count`
- refinar la semántica de ausencia de obstáculos
- redefinir validaciones largas para `shoot` y `pass` si se quiere medir ejecución completa y no solo activación del pipeline

## Conclusión

Para el flujo RL activo probado con el probe habitual:

- el wire ya es natural,
- el `obs` ya no depende de `GroundTruth`,
- la publicación de observaciones ya no está duplicada,
- y el bridge queda listo para ser consumido por entrenamiento RL sin seguir haciendo cambios estructurales del lado del wiring.
