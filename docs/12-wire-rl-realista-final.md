# Wire RL realista final

## Objetivo

Cerrar la fase de wiring RL -> B-Human con una superficie única y natural sobre
SimRobot real.

## Flujo activo

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

Regla vigente:

```text
RL decide qué skill pedir.
B-Human decide cómo ejecutarla.
```

## Cambios cerrados

### 1. Publicador canónico de observaciones

El payload que consume Python se publica en un solo lugar:

- `SkillBehaviorControl`

`RLSkillProvider` queda dedicado a:

- leer acción RL desde `RLSharedState`
- producir `SkillRequest`
- producir `StrategyStatus`

### 2. Features ofensivas mínimas vivas

El wire natural ya publica:

- `can_score_now`
- `shot_quality_no_obstacles`
- `shot_opening_with_obstacles`

### 3. Superficie soportada

El bridge activo expone:

- `simrobot`
- `simrobot_visible`
- reset extendido con teammates y opponents
- cambios dinámicos de mundo intra-episodio
- status explícito de requests visibles con `rl_get_world_status(...)`

### 4. Matriz de skills validada

Se validó el pipeline de las 8 skills públicas sobre perfiles `2d` y `3d`.

Interpretación:

- el bridge RL -> `SkillRequest` quedó correcto
- las degradaciones internas de `shoot` o `pass` son comportamiento aguas abajo

## Qué queda fuera de esta fase

Lo siguiente ya no es wiring puro:

- reward semántico de takeover
- curriculum por stages
- masks activas en entrenamiento
- telemetría de PPO
- evaluación corta con gates

## Conclusión

El wire ya está listo para que el equipo de RL trabaje encima sin seguir
tocando estructura base del bridge.
