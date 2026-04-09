# BHUMAN_BC_SCHEMA_v1 — Esquema fiel para dataset de Behavioral Cloning desde `bhuman_bridge.cpp`

> **Propósito**
>
> Definir un esquema de captura de datos para Behavioral Cloning que sea:
>
> - lo más fiel posible al funcionamiento real de BHuman
> - seguro para integrar en un proyecto BHuman limpio y ya probado en NAOs
> - implementable como **expansión de** `Src/Apps/RLBridge/bhuman_bridge.cpp`
> - no intrusivo sobre los módulos originales de decisión (`Behavior.cpp`, `PlayBall.cpp`, `StrategyBehaviorControl.cpp`, etc.)

---

## 1. Principio rector

Este esquema NO debe construirse:

- alterando la lógica central de BHuman
- modificando cómo BHuman decide `SkillRequest`
- insertando logging dentro de `StrategyBehaviorControl`
- insertando logging dentro de `PlayBall`
- insertando logging dentro de `PassToTeammate`
- creando atajos que mezclen percepción, decisión y outcome sin marcar fuentes

Este esquema SÍ debe construirse:

- como una **extensión del app artificial** `bhuman_bridge.cpp`
- como un observador pasivo de blackboard y snapshots
- como un serializador fiel de representaciones reales de BHuman
- como un pipeline doble:
  - `raw frame dataset`
  - `normalized event dataset`

---

## 2. Filosofía de diseño

Si el objetivo es hacer Behavioral Cloning serio, el dataset debe conservar tres cosas:

1. **Qué sabía BHuman**
2. **Qué decidió BHuman**
3. **Qué pasó después**

Por eso el diseño recomendado no es un único JSON "bonito", sino dos datasets coordinados:

1. `raw_bhuman_frame.jsonl`
- espejo fiel del frame de cognición y de las representaciones relevantes

2. `normalized_bhuman_event.jsonl`
- resumen por eventos semánticos derivados del raw

Ambos deben ir unidos por claves estables.

---

## 3. Por qué construirlo sobre `bhuman_bridge.cpp`

En un BHuman limpio y funcional, `bhuman_bridge.cpp` ya es la zona artificial creada para:

- observar decisiones
- serializar representaciones relevantes
- transportar información hacia el entorno RL

Eso lo convierte en el mejor lugar para incorporar el recorder del dataset BC, porque:

- ya es una frontera de integración
- ya trabaja con snapshots de cognición/motion
- ya lee representaciones como:
  - `SkillRequest`
  - `MotionRequest`
  - `MotionInfo`
  - `StrategyStatus`
  - `BehaviorStatus`
  - `ActivationGraph`
- evita contaminar el corazón del comportamiento original

### Decisión arquitectónica

El dataset recorder debe vivir:

- dentro de `bhuman_bridge.cpp`

o, si se quiere mejor separación interna:

- dentro de clases auxiliares nuevas declaradas y usadas por `bhuman_bridge.cpp`

Pero NO debe vivir:

- dentro de `Behavior.cpp`
- dentro de `StrategyBehaviorControl.cpp`
- dentro de `PlayBall.cpp`
- dentro de `PassToTeammate.cpp`

---

## 4. Objetivo funcional del recorder

La expansión de `bhuman_bridge.cpp` debe permitir:

1. capturar el estado real del frame de cognición
2. capturar la decisión real emitida por BHuman
3. capturar outcome posterior observable
4. generar un dataset raw fiel
5. generar un dataset de eventos ligado al raw

Sin:

- modificar protocolos internos de BHuman
- alterar providers del blackboard
- cambiar ownership de `SkillRequest`
- introducir nuevas dependencias en la cadena de decisión

---

## 5. Restricciones de seguridad e integración

La implementación debe respetar estas reglas:

### 5.1 Regla de no interferencia

El recorder:

- no provee representaciones nuevas
- no reemplaza providers
- no modifica `threads.cfg`
- no cambia la frecuencia de decisión del robot
- no modifica `SkillRequest`
- no modifica `MotionRequest`

### 5.2 Regla de lectura pasiva

El recorder solo puede:

- leer `DecisionSnapshot`
- leer blackboard en modo read-only
- usar counters de frame ya disponibles

### 5.3 Regla de serialización desacoplada

La escritura del dataset no debe:

- bloquear el ciclo de cognición
- depender de parsing pesado por frame
- convertir el bridge en un cuello de botella

Por eso la escritura debe ser:

- incremental
- append-only
- con buffers simples
- idealmente en formato `jsonl`

---

## 6. Arquitectura recomendada dentro de `bhuman_bridge.cpp`

Se recomienda estructurarlo así:

1. `BridgeTransport`
- responsabilidad existente del bridge
- comunicación y serialización operacional

2. `BHumanRawDatasetRecorder`
- construye y escribe `raw_bhuman_frame.jsonl`

3. `BHumanEventDatasetRecorder`
- detecta cambios semánticos
- construye y escribe `normalized_bhuman_event.jsonl`

4. `DatasetRecorderConfig`
- flags de activación
- rutas de salida
- frecuencia de captura
- modo raw/event/both

### Flujo recomendado

1. termina un frame de cognición
2. `bhuman_bridge.cpp` ya obtiene snapshot y/o acceso a blackboard
3. `BHumanRawDatasetRecorder` construye el registro raw
4. `BHumanEventDatasetRecorder` compara con estado previo
5. si hay transición relevante, emite evento
6. ambos escriben `jsonl`

---

## 7. Estructura del dataset

Se recomiendan exactamente dos archivos:

### 7.1 Raw

`raw_bhuman_frame.jsonl`

Una línea por frame de cognición.

### 7.2 Event

`normalized_bhuman_event.jsonl`

Una línea por evento semántico.

### 7.3 Regla de enlace obligatorio

El dataset de eventos debe ir siempre enlazado al raw mediante:

- `session_id`
- `robot_name`
- `player_number`
- `frame_number`
- `time_ms`
- `raw_frame_refs`
- `event_id`

El dataset de eventos nunca es autosuficiente.

---

## 8. Schema general del raw dataset

Cada línea de `raw_bhuman_frame.jsonl` debe tener esta estructura lógica:

```json
{
  "schema_version": "bhuman_bc_v1",
  "session_id": "2026-04-09_4v4_run_003",
  "robot_name": "Nao3",
  "player_number": 3,
  "frame_number": 182344,
  "time_ms": 6017352,
  "cycle_ms": 33,
  "scenario": "4v4_Scaled",
  "thread": "Cognition",
  "frame_context": {},
  "bhuman_inputs": {},
  "bhuman_outputs": {},
  "post_outcome": {}
}
```

---

## 9. Sección `frame_context`

Esta sección describe el contexto administrativo y temporal del frame.

### Campos obligatorios

- `schema_version`
- `session_id`
- `robot_name`
- `player_number`
- `frame_number`
- `time_ms`
- `cycle_ms`
- `scenario`
- `thread`

### Campos opcionales

- `match_id`
- `half`
- `team_color`
- `recording_mode`

### Ejemplo

```json
"frame_context": {
  "scenario": "4v4_Scaled",
  "thread": "Cognition",
  "recording_mode": "raw+event"
}
```

---

## 10. Sección `bhuman_inputs`

Esta sección debe reflejar lo mejor posible el estado cognitivo de BHuman
sin reinterpretarlo desde Python.

### 10.1 Reglas

- guardar la fuente real de cada grupo de campos
- no colapsar representaciones distintas
- no usar padding artificial de teammates
- no fabricar slots canónicos si no existen así en BHuman

### 10.2 Subbloques recomendados

1. `game_state`
2. `robot_pose`
3. `field_ball`
4. `ball_model`
5. `teammates_ball_model`
6. `team_data`
7. `global_opponents_model` o equivalente observado
8. `motion_info_before`
9. `strategy_status_before`
10. `behavior_status_before`
11. `activation_graph_before` si se quiere snapshot pre-output

---

## 11. Schema detallado de `game_state`

```json
"game_state": {
  "source": "GameState",
  "state": "playing",
  "player_state": "active",
  "game_controller_active": true
}
```

### Campos obligatorios

- `source`
- `state`
- `player_state`
- `game_controller_active`

---

## 12. Schema detallado de `robot_pose`

```json
"robot_pose": {
  "source": "RobotPose",
  "x_mm": 1200.0,
  "y_mm": -350.0,
  "theta_rad": 0.42
}
```

### Campos obligatorios

- `source`
- `x_mm`
- `y_mm`
- `theta_rad`

---

## 13. Schema detallado de balón

Se recomienda separar:

- `field_ball`
- `ball_model`
- `teammates_ball_model`

Porque BHuman no siempre decide con una sola noción de balón.

### 13.1 `field_ball`

```json
"field_ball": {
  "source": "FieldBall",
  "x_mm": 1620.0,
  "y_mm": -120.0,
  "vx_mm_s": 0.0,
  "vy_mm_s": 0.0
}
```

### 13.2 `ball_model`

```json
"ball_model": {
  "source": "BallModel",
  "x_rel_mm": 430.0,
  "y_rel_mm": 35.0
}
```

### 13.3 `teammates_ball_model`

```json
"teammates_ball_model": {
  "source": "TeammatesBallModel",
  "is_valid": true
}
```

---

## 14. Cómo guardar teammates de forma fiel a BHuman

Este es uno de los puntos más importantes.

### Regla principal

Los teammates NO deben guardarse como:

- slots artificiales fijos inventados por el bridge Python
- padding canónico
- home poses sintéticas

Los teammates deben guardarse como:

- los compañeros reales que BHuman tiene disponibles en ese frame
- identificados por `player_number`
- con la fuente real explícita
- con freshness/edad si existe

### Estructura recomendada

```json
"team_data": {
  "source": "TeamData",
  "teammates": [
    {
      "player_number": 1,
      "is_goalkeeper": false,
      "pose_x_mm": 300.0,
      "pose_y_mm": -900.0,
      "pose_theta_rad": 0.10,
      "status": "active",
      "time_since_last_packet_ms": 66
    },
    {
      "player_number": 2,
      "is_goalkeeper": false,
      "pose_x_mm": 900.0,
      "pose_y_mm": 250.0,
      "pose_theta_rad": -0.20,
      "status": "active",
      "time_since_last_packet_ms": 33
    }
  ]
}
```

### Campos recomendados por teammate

- `player_number`
- `is_goalkeeper`
- `pose_x_mm`
- `pose_y_mm`
- `pose_theta_rad`
- `status`
- `time_since_last_packet_ms`

### Campos opcionales

- `ball_x_mm`
- `ball_y_mm`
- `is_upright`
- `penalized`

### Si además se quiere capturar la visión táctica interna

Se puede añadir un bloque separado:

```json
"strategy_agents": {
  "source": "Behavior/Agent list",
  "agents": [
    {
      "player_number": 1,
      "role": "defenderL",
      "position": "defenderL",
      "is_active_candidate": false,
      "current_position_x_mm": 320.0,
      "current_position_y_mm": -880.0
    }
  ]
}
```

Pero esto debe ir separado de `team_data`.

---

## 15. Cómo guardar opponents

Los opponents deben guardarse con la representación real utilizada por el sistema.

```json
"global_opponents_model": {
  "source": "GlobalOpponentsModel",
  "players": [
    {
      "x_mm": 2100.0,
      "y_mm": 180.0
    }
  ]
}
```

Si hay campos extra disponibles y confiables:

- velocidad
- edad
- confianza

pueden añadirse, siempre marcando fuente.

---

## 16. `motion_info_before`

Esto captura el estado de ejecución inmediatamente relevante para la cognición.

```json
"motion_info_before": {
  "source": "MotionInfo",
  "executed_phase": "walk",
  "last_kick_type": "none",
  "last_kick_timestamp_ms": 0
}
```

### Campos recomendados

- `executed_phase`
- `last_kick_type`
- `last_kick_timestamp_ms`

---

## 17. `strategy_status_before`

```json
"strategy_status_before": {
  "source": "StrategyStatus",
  "accepted_tactic": "t121_4v4_scaled",
  "proposed_tactic": "t121_4v4_scaled",
  "role": "forwardL",
  "position": "forwardL",
  "proposed_set_play": "none",
  "accepted_set_play": "none",
  "set_play_step": -1,
  "accepted_mirror": false,
  "proposed_mirror": false
}
```

---

## 18. `behavior_status_before`

```json
"behavior_status_before": {
  "source": "BehaviorStatus",
  "speed_mm_s": 0.0,
  "pass_target": -1,
  "pass_origin": -1,
  "walking_to_x_mm": 0.0,
  "walking_to_y_mm": 0.0
}
```

---

## 19. Sección `bhuman_outputs`

Esta sección debe contener exactamente lo que BHuman decidió o publicó después de decidir.

### Subbloques recomendados

1. `skill_request`
2. `motion_request`
3. `strategy_status_after`
4. `behavior_status_after`
5. `activation_graph`

---

## 20. Schema de `skill_request`

```json
"skill_request": {
  "source": "SkillRequest",
  "skill": "pass",
  "pass_target": 2,
  "target_x_mm": 0.0,
  "target_y_mm": 0.0,
  "target_theta_rad": 0.0
}
```

### Campos obligatorios

- `source`
- `skill`

### Campos opcionales según skill

- `pass_target`
- `target_x_mm`
- `target_y_mm`
- `target_theta_rad`

### Regla importante

Para etiquetar `pass`, la fuente principal debe ser:

- `SkillRequest::pass`

No:

- una inferencia externa basada en movimiento

---

## 21. Schema de `motion_request`

```json
"motion_request": {
  "source": "MotionRequest",
  "motion": "walkToBallAndKick",
  "walk_speed_x": 0.0,
  "walk_speed_y": 0.0,
  "walk_speed_theta": 0.0,
  "kick_type": "forwardFastLeft",
  "kick_length_mm": 2500.0,
  "walk_target_x_mm": 1550.0,
  "walk_target_y_mm": -50.0,
  "walk_target_theta_rad": 0.12,
  "target_direction_rad": 0.08,
  "align_precisely": "notPrecise",
  "pre_step_allowed": true,
  "turn_kick_allowed": true
}
```

Esto es muy valioso para BC avanzado porque preserva:

- intención de ejecución
- kick type
- target físico
- velocidad de aproximación

---

## 22. `strategy_status_after`

```json
"strategy_status_after": {
  "source": "StrategyStatus",
  "accepted_tactic": "t121_4v4_scaled",
  "proposed_tactic": "t121_4v4_scaled",
  "role": "forwardL",
  "position": "forwardL"
}
```

Sirve para capturar:

- táctica aceptada
- rol
- posición táctica nominal

---

## 23. `behavior_status_after`

```json
"behavior_status_after": {
  "source": "BehaviorStatus",
  "state": "passing",
  "speed_mm_s": 180.0,
  "pass_target": 2,
  "pass_origin": 3,
  "walking_to_x_mm": 1500.0,
  "walking_to_y_mm": -80.0,
  "shooting_to_x_mm": null,
  "shooting_to_y_mm": null
}
```

---

## 24. `activation_graph`

```json
"activation_graph": {
  "source": "ActivationGraph",
  "nodes": [
    {
      "option": "Strategy",
      "depth": 0,
      "state": "playing",
      "option_time_ms": 1400,
      "state_time_ms": 240
    },
    {
      "option": "PlayBall",
      "depth": 1,
      "state": "pass",
      "option_time_ms": 350,
      "state_time_ms": 120
    },
    {
      "option": "PassToTeammate",
      "depth": 2,
      "state": "execute",
      "option_time_ms": 120,
      "state_time_ms": 85
    }
  ]
}
```

Esto es especialmente importante porque:

- da semántica temporal
- ayuda a construir eventos
- permite depurar decisiones sin invadir módulos internos

---

## 25. Sección `post_outcome`

Esta sección no debe inventar éxito semántico que BHuman no emitió.
Debe limitarse a outcomes observables y claramente marcados.

### Ejemplo

```json
"post_outcome": {
  "source": "bridge_observed",
  "kick_observed": false,
  "last_kick_timestamp_ms": 0,
  "last_kick_type": "none",
  "ball_progress_mm_100ms": 0.0,
  "ball_progress_mm_330ms": 420.0,
  "ball_progress_along_target_mm_330ms": 390.0,
  "team_possession_after_330ms": true,
  "goal_scored_within_2s": false
}
```

### Regla

Esto no reemplaza la decisión. Solo describe qué pasó después.

---

## 26. Schema general del event dataset

Cada línea de `normalized_bhuman_event.jsonl` debe resumir un cambio semántico.

### Ejemplo base

```json
{
  "schema_version": "bhuman_bc_event_v1",
  "event_id": "sess003_p3_f182344_pass_start",
  "session_id": "2026-04-09_4v4_run_003",
  "robot_name": "Nao3",
  "player_number": 3,
  "event_type": "skill_request_change",
  "event_name": "pass_start",
  "start_frame": 182344,
  "end_frame": 182361,
  "start_time_ms": 6017352,
  "end_time_ms": 6017913,
  "decision_frame_ref": 182344,
  "raw_frame_refs": [182344, 182345, 182346],
  "decision": {},
  "outcome_summary": {}
}
```

---

## 27. Eventos que vale la pena emitir

### Eventos mínimos

- `skill_request_change`
- `motion_request_change`
- `pass_target_change`
- `role_change`
- `tactic_change`
- `kick_observed`
- `macro_window_start`
- `macro_window_end`

### Regla

Los eventos deben derivarse del raw.
Nunca al revés.

---

## 28. Frecuencia de captura

### Recomendación principal

Capturar raw por frame de cognición:

- aproximadamente `33 ms`

### No recomendado como captura primaria

- `330 ms`

Porque puede borrar:

- el frame exacto donde cambia `SkillRequest`
- el nacimiento de `passTarget`
- el cambio `dribble -> shoot`
- el `kick timestamp`

### Sí recomendado

- capturar raw a `33 ms`
- hacer cribado y downsampling después

---

## 29. Cómo construirlo dentro de `bhuman_bridge.cpp`

## 29.1 Objetivo de implementación

Añadir recorder sin alterar la lógica existente de transporte.

### Recomendación

Crear clases auxiliares dentro del mismo app:

- `DatasetRecorderConfig`
- `BHumanRawDatasetRecorder`
- `BHumanEventDatasetRecorder`

### Campos mínimos de `DatasetRecorderConfig`

- `enabled`
- `write_raw`
- `write_events`
- `output_directory`
- `flush_every_n_frames`
- `record_cycle_ms`

---

## 29.2 Flujo recomendado de implementación

### Paso 1

Después de que el bridge ya tenga acceso a:

- `DecisionSnapshot`
- blackboard fallback

construir una estructura interna `RawFrameRecord`.

### Paso 2

Serializar esa estructura a JSONL sin reinterpretarla.

### Paso 3

Mantener en memoria el último `RawFrameRecord`.

### Paso 4

Comparar `RawFrameRecord actual` vs `RawFrameRecord previo`.

### Paso 5

Si hubo cambio semántico:

- emitir un `NormalizedEventRecord`

### Paso 6

Escribir ambos archivos:

- `raw_bhuman_frame.jsonl`
- `normalized_bhuman_event.jsonl`

---

## 30. Cómo evitar interferir con protocolos BHuman

La implementación debe seguir estas reglas:

### 30.1 No tocar `threads.cfg`

No hace falta añadir providers nuevos.

### 30.2 No tocar ownership de `SkillRequest`

`SkillRequest` debe seguir siendo provisto exactamente por quien ya lo provee.

### 30.3 No tocar módulos de decisión

No poner logging dentro de:

- `Behavior.cpp`
- `PlayBall.cpp`
- `StrategyBehaviorControl.cpp`
- `PassToTeammate.cpp`

### 30.4 No introducir RPCs o protocolos nuevos

El dataset se guarda localmente desde el bridge.

### 30.5 Mantener el recorder como observador

Su única responsabilidad es:

- capturar
- normalizar
- escribir

---

## 31. Qué parte de BHuman representa mejor "la cognición"

Si el objetivo del BC es capturar "lo mayor posible de BHuman que el agente vaya a hacer como cognición", las representaciones más valiosas son:

### Inputs

- `GameState`
- `RobotPose`
- `FieldBall`
- `BallModel`
- `TeammatesBallModel`
- `TeamData`
- `MotionInfo`
- `StrategyStatus` pre y post
- `BehaviorStatus` pre y post
- opponents disponibles

### Outputs

- `SkillRequest`
- `MotionRequest`
- `StrategyStatus`
- `BehaviorStatus`
- `ActivationGraph`

### Outcome observado

- `MotionInfo.lastKickTimestamp`
- `MotionInfo.lastKickType`
- progreso del balón
- continuidad de posesión
- gol posterior si ocurre

Eso cubre:

- pose
- rol
- táctica
- macro
- target
- receptor de pase
- transición temporal de decisión

---

## 32. Regla de fidelidad final

Si hay conflicto entre:

- un dataset más cómodo para RL

y

- una captura más fiel a BHuman

debe ganar la fidelidad a BHuman.

La captura raw debe ser:

- cruda
- versionada
- tipada
- con fuente explícita
- sin padding artificial
- sin reinterpretación temprana

La normalización para RL debe ocurrir después.

---

## 33. Recomendación final

La forma correcta de construir este esquema en un BHuman limpio y probado es:

1. tratar `bhuman_bridge.cpp` como perímetro artificial de integración
2. expandirlo con un recorder pasivo
3. capturar `raw frame` fiel a representaciones reales
4. derivar `event dataset` enlazado al raw
5. no tocar la lógica original de decisión táctica

Ese enfoque permite:

- preservar la robustez del BHuman base
- obtener un dataset serio para Behavioral Cloning
- mantener clara la frontera entre:
  - sistema original BHuman
  - integración RL
  - infraestructura de datos
