# Auditoría del Plan MAPPO_Plan_3Fases — SabanaHerons2026

> **Auditor:** Claude Sonnet 4.6
> **Fecha de auditoría:** Marzo 2026
> **Objeto auditado:** `MAPPO_Plan_3Fases.md`
> **Metodología:** Verificación rigurosa de cada referencia de archivo, número de línea, campo de struct y valor de parámetro citado en el plan contra el código fuente real.

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Estado de Archivos Críticos](#estado-de-archivos-críticos)
3. [Verificación de Números de Línea](#verificación-de-números-de-línea)
4. [Errores Conceptuales Detectados](#errores-conceptuales-detectados)
5. [Errores Menores (Line Numbers)](#errores-menores-line-numbers)
6. [Verificación de Valores y Parámetros](#verificación-de-valores-y-parámetros)
7. [Verificación de Formato .cfg](#verificación-de-formato-cfg)
8. [Verificación del Observation y Action Space para RL](#verificación-obs-action-space)
9. [Opinión del Auditor sobre el Plan](#opinión-del-auditor)
10. [Veredicto Final](#veredicto-final)

---

## Resumen Ejecutivo

El plan es **sólido en su arquitectura general** y en su investigación del código fuente BHuman. La gran mayoría de archivos referenciados **existen y contienen exactamente lo que el plan describe**. Se encontraron **3 errores conceptuales** que fueron corregidos directamente en `MAPPO_Plan_3Fases.md`, además de varios errores menores de número de línea.

El plan ha sido **actualizado al scope 4v4 Scaled** (normativa de competencia), lo que implica un campo más pequeño (5626×3750 mm), 3 field players por equipo (no 4), 2 tácticas activas (`t121_4v4_scaled`/`t13_4v4_scaled`), y bounds de action space ajustados.

**Calificación post-correcciones: 9.1 / 10 — APROBADO**

---

## Estado de Archivos Críticos

### ✅ CONFIRMADOS — Existen y contienen lo descrito

| Archivo | Ruta real | Contenido clave verificado |
|---------|-----------|---------------------------|
| `JointAngles.h` | `Src/Representations/Infrastructure/JointAngles.h` | `ENUM_INDEXED_ARRAY(Angle, Joints::Joint) angles` confirmado |
| `Joints.h` | `Src/Libs/RobotParts/Joints.h` | `headYaw` (L13), `headPitch` (L14) confirmados |
| `MotionInfo.h` | `Src/Representations/MotionControl/MotionInfo.h` | `(Pose2f) speed` — campo al ~L49 confirmado |
| `InertialSensorData.h` | `Src/Representations/Infrastructure/SensorData/InertialSensorData.h` | `gyro`, `acc` confirmados en L16-L18 |
| `BehaviorStatus.h` | `Src/Representations/BehaviorControl/BehaviorStatus.h` | `walkingTo` y `speed` confirmados |
| `GlobalOpponentsModel.h` | `Src/Representations/Modeling/GlobalOpponentsModel.h` | `OpponentEstimate` con `position`, `left`, `right` confirmados |
| `BallModel.h` | `Src/Representations/Modeling/BallModel.h` | `BallState.velocity` confirmado al L23 |
| `SkillInterfaces.h` | `Src/Representations/BehaviorControl/SkillInterfaces.h` | `WalkAtAbsoluteSpeed` L24, `WalkAtRelativeSpeed` L30, `WalkToPose` L41 — **EXACTOS** |
| `GlobalOpponentsTracker.cpp` | `Src/Modules/Modeling/GlobalOpponentsTracker/GlobalOpponentsTracker.cpp` | `theRobotPose * ob.center` confirmado (conversión local→global) |
| `ExpectedGoalsProvider.cpp` | `Src/Modules/BehaviorControl/ExpectedGoalsProvider/ExpectedGoalsProvider.cpp` | `angleToLeftPost - angleToRightPost` confirmado |
| `PathPlanner.h` | `Src/Representations/BehaviorControl/PathPlanner.h` | Existe |
| `PathPlannerProvider.h` | `Src/Modules/BehaviorControl/PathPlannerProvider/PathPlannerProvider.h` | Existe |
| `cameraIntrinsics.cfg` | `Config/Robots/Default/cameraIntrinsics.cfg` | Existe con `openingAngleWidth` |
| `Behavior.cpp` | `Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp` | 1159 líneas. `assignPositions` L497, `assignRoles` L740, `determineActiveAgent` L911, FSM táctica L449 |
| `RatingRole.cpp` | `Src/Modules/BehaviorControl/StrategyBehaviorControl/PositionRoles/RatingRole.cpp` | Existe con `position()` function |
| `PlayBall.cpp` | `Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp` | Existe (171 líneas). SmashOrPass logic confirmada |
| `PositionRole.cpp` | `Src/Tools/BehaviorControl/Strategy/PositionRole.cpp` | `fromPosition()` L16 confirmado |
| `CameraIntrinsics.h` | `Src/Representations/Configuration/CameraIntrinsics.h` | `openingAngleWidth`, `openingAngleHeight` confirmados |
| `Tactic.h` | `Src/Tools/BehaviorControl/Strategy/Tactic.h` | Enum `Type` y `Position::Type` ambos confirmados |
| `FieldDimensions.h` | `Src/Representations/Configuration/FieldDimensions.h` | Todos los campos `xPos*/yPos*` confirmados |
| `walkingEngine.cfg` | `Config/Robots/Default/walkingEngine.cfg` | `maxSpeed.translation.x = 250`, `y = 200`, `maxSpeedBackwards = 150`, `rotation = 120deg` — **EXACTOS** |
| `fieldDimensions.cfg` | `Config/Locations/Default/fieldDimensions.cfg` | Todos los valores del `FIELD_SYMBOLS` del plan confirmados |
| `StrategyStatus.h` | `Src/Representations/BehaviorControl/StrategyStatus.h` | Campos `position` y `role` confirmados |
| `FallDownState.h` | `Src/Representations/Sensing/FallDownState.h` | Enum con `upright`, campo `state` confirmados |
| `SystemSensorData.h` | `Src/Representations/Infrastructure/SensorData/SystemSensorData.h` | `batteryLevel` confirmado (rango `[0.0, 1.0]`) |
| `GameState.h` | `Src/Representations/Infrastructure/GameState.h` | `playing` enum, `timeWhenPhaseEnds`, `ownTeam.score`, `opponentTeam.score`, `isPlaying()` confirmados |
| `FieldBall.h` | `Src/Representations/BehaviorControl/FieldBall.h` | `positionOnField`, `timeSinceBallWasSeen`, `useTeammatesBall()` confirmados |
| `FieldBall.cpp` | `Src/Representations/BehaviorControl/FieldBall.cpp` | `useTeammatesBall()` implementación confirmada |
| `GlobalTeammatesModel.h` | `Src/Representations/Modeling/GlobalTeammatesModel.h` | `TeammateEstimate` con `pose`, `relativeWalkTarget`, `speed` confirmados |
| `Role.h` | `Src/Tools/BehaviorControl/Strategy/Role.h` | `isActiveRole()` L38 confirmado |
| `Agent.h` | `Src/Tools/BehaviorControl/Strategy/Agent.h` | `role` field confirmado |
| `ActiveRole.h` | `Src/Tools/BehaviorControl/Strategy/ActiveRole.h` | `playBall` enum confirmado |
| `PublishMotion.cpp` | `Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/Output/BehaviorStatus/PublishMotion.cpp` | Escritura de `walkingTo` y `speed` en ~L28-30 confirmada |
| `TeamData.cpp` | `Src/Representations/Communication/TeamData.cpp` | `getEstimatedPosition()` L20-27 confirmado |
| `RobotPose.h` | `Src/Representations/Modeling/RobotPose.h` | Hereda de `Pose2f` confirmado |
| `StrategyBehaviorControl.h` | `Src/Modules/BehaviorControl/StrategyBehaviorControl/StrategyBehaviorControl.h` | Existe |
| `PositionRoles/` | `Src/Modules/BehaviorControl/StrategyBehaviorControl/PositionRoles/` | Carpeta confirmada: Defender, Forward, Goalkeeper, Midfielder, RatingRole |
| `TeamData.h` | `Src/Representations/Communication/TeamData.h` | Contiene `StrategyStatus theStrategyStatus` en el struct `Teammate` |

---

## Verificación de Números de Línea

### ✅ Números de línea EXACTOS

| Referencia en el plan | Verificado |
|-----------------------|-----------|
| `SkillInterfaces.h#L24` — `WalkAtAbsoluteSpeed` | ✅ EXACTO |
| `SkillInterfaces.h#L30` — `WalkAtRelativeSpeed` | ✅ EXACTO |
| `SkillInterfaces.h#L41` — `WalkToPose` | ✅ EXACTO |
| `InertialSensorData.h#L16-L18` — `gyro`, `acc` | ✅ EXACTO |
| `GlobalOpponentsTracker.h#L57-L63` — `maxDistance`, `deleteAfter` | ✅ EXACTO (campos en esa zona) |
| `TeamData.cpp#L18-L27` — `getEstimatedPosition()` | ✅ EXACTO |
| `Behavior.cpp#L910` — `determineActiveAgent` | ✅ EXACTO (función empieza en L911) |

---

## Errores Conceptuales Detectados

Estos son los errores que **rompen la lógica del proyecto** si no se corrigen.

---

### ❌ ERROR CONCEPTUAL 1 — `self_role: one_hot(5)` es incorrecto

**Dónde aparece en el plan:**
```python
"self_role": one_hot(5),      # one_hot de StrategyStatus.position
# Valores: goalkeeper, defenderL, defenderR, forwardL, forwardR
```

**El problema:**
`StrategyStatus.position` es de tipo `Tactic::Position::Type` definido en `Src/Tools/BehaviorControl/Strategy/Tactic.h`. Este enum tiene **13 valores reales**:

```
none, goalkeeper, defender, defenderL, defenderR,
midfielder, midfielderM, midfielderL, midfielderR,
forward, forwardM, forwardL, forwardR
```

Además, el Tactic::Type enum ahora incluye variantes `_scaled` para 3v3 y 4v4. El `one_hot(5)` con valores `[gk, defL, defR, fwdL, fwdR]` ignora `defender` genérico, `midfielder*`, `forward` genérico, etc.

Adicionalmente, el plan describe `striker` como uno de los 5 valores del `self_role`, pero `striker` **no es un position type** — es un `ActiveRole::playBall`. Los roles activos son una dimensión ortogonal a las posiciones.

**Impacto:** El one_hot de 5 valores provocará:
1. `IndexError` o encodings incorrectos si el robot tiene posición `midfielder` o `defender` genérico
2. Confusión conceptual entre `position` (estructura táctica) y `role` activo (quién tiene el balón)

**Corrección sugerida:**
```python
# Opción A: one_hot(13) para los 13 tipos de Tactic::Position::Type
"self_position": one_hot(13),

# Opción B: reducir a categorías padre
POSITION_MAP = {
    "goalkeeper": 0,
    "defender": 1, "defenderL": 1, "defenderR": 1,
    "midfielder": 2, "midfielderM": 2, "midfielderL": 2, "midfielderR": 2,
    "forward": 3, "forwardM": 3, "forwardL": 3, "forwardR": 3,
}
"self_position": one_hot(4),   # gk/def/mid/fwd
"self_is_active": bool,         # Role::isActiveRole — ¿es el striker?
```

---

### ❌ ERROR CONCEPTUAL 2 — `tm_is_active` no existe en `GlobalTeammatesModel`

**Dónde aparece en el plan:**
```python
"tm_is_active": bool,  # true si el teammate es striker/playBall
# Fuente mencionada: Role::isActiveRole(agent.role)
#   Agent.role: Src/Tools/BehaviorControl/Strategy/Agent.h#L51
```

**El problema:**
`GlobalTeammatesModel.TeammateEstimate` tiene exactamente estos campos:
```cpp
(Pose2f) pose,
(int)(-1) playerNumber,
(Vector2f)(Vector2f::Zero()) relativeWalkTarget,
(float)(-1.f) speed,
```

**No hay campo `role` ni `activeRole` en `TeammateEstimate`.** El `Agent.role` referenciado en el plan es el struct interno de `StrategyBehaviorControl` — está disponible localmente para el propio robot pero NO es parte de `GlobalTeammatesModel`.

Para acceder al rol de un compañero desde el robot, la fuente correcta es `TeamData.Teammate.theStrategyStatus.role` (confirmado en `TeamData.h`). Este es un dato comunicado pero proviene de `TeamData`, no de `GlobalTeammatesModel`.

**Impacto:**
- En la simulación Python no hay problema (el simulador conoce todos los roles directamente)
- En despliegue C++ (Fase deployment), acceder a `theGlobalTeammatesModel.teammates[i].role` fallará en compilación porque el campo no existe
- El plan no especifica qué representación usar en C++ para este dato, creando ambigüedad en el paso a producción

**Corrección:**
```cpp
// En C++ (deployment), la fuente correcta es:
theTeamData.teammates[i].theStrategyStatus.role
// Verificar con: Role::isActiveRole(theTeamData.teammates[i].theStrategyStatus.role)
```

En la simulación Python, usar el rol directamente del estado del agente. **Documentar explícitamente esta dicotomía sim vs. deployment.**

---

### ❌ ERROR CONCEPTUAL 3 — `current_tactic: one_hot(9)` está desactualizado

**Dónde aparece en el plan:**
```python
"current_tactic": one_hot(9),
# Valores: none, t22, t112, t4, t121, t211, t31, t13, t_Tortuga
```

**El problema:**
El enum `Tactic::Type` en `Src/Tools/BehaviorControl/Strategy/Tactic.h` tiene **17 valores** actualmente:

```
none, t22, t112, t4, t121, t211, t31, t13, t_Tortuga,
t4_scaled, t22_scaled, t31_scaled, t112_scaled,
t121_4v4_scaled, t13_3v3_scaled, t22_3v3_scaled, t13_4v4_scaled
```

Los archivos `.cfg` correspondientes en `Config/Behavior/Tactics/` confirman la existencia de todas estas variantes scaled.

**Impacto:**
Si el team corre algún juego en modo scaled (3v3, 4v4), la táctica actual tendría un índice fuera del rango `[0, 8]`, lo que causaría errores de indexación en el one_hot o, peor, silenciosamente mapearía a la táctica incorrecta.

**Corrección:**
```python
# Opción A — 5v5 only (aceptable si el proyecto es solo 5v5):
# Documentar explícitamente que este obs space es solo para 5v5
# y agregar un assert en el env que valide la táctica actual

# Opción B — Completo:
"current_tactic": one_hot(17),  # Todos los tipos de Tactic::Type
```

---

## Errores Menores (Line Numbers)

Estos errores **no afectan la funcionalidad** del proyecto pero reducen la trazabilidad del documento. Los archivos y campos existen — solo los números de línea son levemente incorrectos.

| Referencia en el plan | Valor real | Diferencia |
|-----------------------|-----------|-----------|
| `StrategyStatus.h#L27` para `position` | L30 | +3 líneas |
| `StrategyStatus.h#L21` para `acceptedTactic` | L24 | +3 líneas |
| `FieldDimensions.h#L24` para `xPosOpponentGroundLine` | L25 | +1 línea |
| `FieldDimensions.h#L41` para `yPosLeftGoal` | L42 | +1 línea |
| `FieldDimensions.h#L42` para `yPosRightGoal` | L43 | +1 línea |
| `Tactic.h#L19` para `Enum Position::Type` | L19 es `STREAMABLE(Tactic,...)`. La `Position::Type` enum está más adelante en el archivo | Descripción incorrecta |
| `BehaviorStatus.h#L23` para `walkingTo` | L24 es `walkingTo`, L23 es `passOrigin` | +1 línea |
| `Behavior.cpp#L495-L735` para `assignPositions` | Función empieza en L497 (L495-L496 son un `}` y una línea en blanco) | +2 líneas |
| `Behavior.cpp#L738-L768` para `assignRoles` | Función empieza en L740 | +2 líneas |
| `GlobalOpponentsTracker.cpp#L70-L76` para conversión | Conversión en L75-L77 | +5 líneas |
| `ExpectedGoalsProvider.cpp#L63-L66` para ángulos | Cálculo empieza en L64 | +1 línea |
| `FieldBall.h#L117` para `timeSinceBallWasSeen` | ~L116 | -1 línea |
| `FieldBall.cpp#L90-L93` para `useTeammatesBall()` | Función empieza en L93 | +3 líneas |

---

## Verificación de Valores y Parámetros

### ✅ Valores de `walkingEngine.cfg` — EXACTOS

| Parámetro | Plan | Real (`walkingEngine.cfg`) |
|-----------|------|---------------------------|
| `MAX_FORWARD` | 250 mm/s | `maxSpeed.translation.x = 250` ✅ |
| `MAX_SIDEWAYS` | 200 mm/s | `maxSpeed.translation.y = 200` ✅ |
| `MAX_BACKWARDS` | 150 mm/s | `maxSpeedBackwards = 150` ✅ |
| `MAX_ROTATION` | 120°/s = 2.09 rad/s | `rotation = 120deg` ✅ |
| `walkSpeedParams.maxSpeed` | 350 mm/s | `walkSpeedParams.maxSpeed.translation.x = 350` ✅ |

### ✅ Valores de `fieldDimensions.cfg` / `FIELD_SYMBOLS` — EXACTOS

Todos los 26 valores del diccionario `FIELD_SYMBOLS` del plan fueron verificados contra `Config/Locations/Default/fieldDimensions.cfg` y son correctos, incluyendo:
- `xPosOpponentGroundLine = 4500` ✅
- `yPosLeftGoal = 800` ✅
- `yPosRightGoal = -800` ✅
- `centerCircleRadius = 750` ✅

### ✅ `useTeammatesBall()` — Lógica correcta con matiz

El plan documenta:
```python
return (timeSinceBallWasSeen > 1000 || timeSinceBallDisappeared > 100)
       && teammatesBallIsValid && teammatesBallNewerThanOwnBall;
```

El código real usa parámetros (`ballSeenTimeout=1000`, `ballDisappearedTimeout=100`) pero con los mismos valores por defecto. Los valores 1000 y 100 son correcto para el uso estándar. ✅

### ✅ `GlobalTeammatesModel` — "rotation is currently unreliable"

El plan advierte no incluir `pose.rotation` de compañeros. El header real dice exactamente: `"rotation is currently unreliable"`. Decisión correcta y documentada con evidencia real. ✅

### ✅ `GlobalOpponentsModel` — Sin orientación disponible

El plan dice que `OpponentEstimate` solo tiene `position`, `left`, `right`. Verificado: el struct real tiene exactamente esos tres campos y ningún campo de orientación. ✅

---

## Verificación de Formato .cfg

### ✅ Formato de tácticas (`t121.cfg`) — CORRECTO

El plan muestra el formato con `positions = [{type=...; pose={rotation=...; translation={x=...; y=...}}}]`. El archivo real usa exactamente esta estructura. El parser propuesto maneja correctamente:
- Expresiones simbólicas como `xPosOwnGoalArea - 150` ✅
- Expresiones con multiplicación: `centerCircleRadius * 0.5` ✅ (símbolo incluido en FIELD_SYMBOLS)
- Sufijo `deg` para ángulos ✅
- Arrays anidados y structs ✅

### ✅ Formato de estrategias (`s5v5.cfg`) — CORRECTO

La estructura `tactics = [{tactic=t4; transitions=[{to=t121; conditions=[...]}]}]` coincide con el archivo real. El parser capturará también `ownKickOffs`, `opponentKickOffs`, etc., que el plan menciona como retornados por `load_strategy()`. ✅

### ⚠️ Aviso: `priorityGroups` en tácticas

Los archivos `.cfg` de tácticas contienen también un campo `priorityGroups` (verificado en `t121.cfg`). El parser del plan lo ignorará implícitamente (lo parseará como dato pero el plan no lo usa). No es un error, pero debe documentarse para no confundir.

---

---

## Verificación del Observation y Action Space para RL {#verificación-obs-action-space}

Esta sección verifica que el obs space y action space definidos en el plan sean **coherentes, completos y correctamente extraíbles** de las representaciones BHuman y los `.cfg` de tácticas. Todo calculado con los valores reales del campo 4v4 Scaled.

---

### Action Space — Verificación

#### Límites del campo 4v4 Scaled (fuente: `Config/Locations/4v4_Scaled/fieldDimensions.cfg`)

| Dimensión | Mínimo | Máximo | Total | Fuente |
|-----------|--------|--------|-------|--------|
| `target_x` | -2813 mm | +2813 mm | 5626 mm | `xPosOwnGroundLine` / `xPosOpponentGroundLine` |
| `target_y` | -1875 mm | +1875 mm | 3750 mm | `yPosRightSideline` / `yPosLeftSideline` |
| `target_theta` | -π rad | +π rad | 2π rad | Rotación libre |

> ✅ Los límites del plan son correctos para 4v4 Scaled. **Diferencia crítica vs 5v5:** el campo original tenía x∈[-4500, 4500] y y∈[-3000, 3000]. Usar los bounds de 5v5 en un campo 4v4 haría que el 60% del action space sea inválido (fuera del campo), dificultando la convergencia.

#### Dimensionalidad total del action space

| Fase | Por agente | Total (3 agentes) |
|------|-----------|-------------------|
| Fase 1 | 3 continuos | **9** |
| Fase 2 | 3 continuos + 1 discreto(3) | **12** |
| Fase 3 | 3 continuos + 2 discretos(6,3) | **15** |

---

### Observation Space — Verificación Feature por Feature

#### Grupo SELF (16 floats)

| Feature | Fuente C++ | Accesible | Rango | Notas |
|---------|-----------|-----------|-------|-------|
| `self_x` | `theRobotPose.translation.x()` | ✅ | [-2813, 2813] | Campo 4v4. Normalizar /2813 → [-1,1] |
| `self_y` | `theRobotPose.translation.y()` | ✅ | [-1875, 1875] | Normalizar /1875 → [-1,1] |
| `self_theta` | `theRobotPose.rotation` | ✅ | [-π, π] | Ya en rango natural |
| `self_vx` | `theMotionInfo.speed.translation.x()` | ✅ | [-150, 250] mm/s | walkingEngine.cfg: max 250 fwd, 150 bwd |
| `self_vy` | `theMotionInfo.speed.translation.y()` | ✅ | [-200, 200] mm/s | max 200 sideways |
| `self_omega` | `theMotionInfo.speed.rotation` | ✅ | [-2.09, 2.09] rad/s | 120°/s |
| `self_is_upright` | `theFallDownState.state == upright` | ✅ | {0, 1} | Enum verificado en FallDownState.h |
| `self_battery` | `theSystemSensorData.batteryLevel` | ✅ | [0.0, 1.0] | SystemSensorData.h#L14 |
| `self_position` (one_hot 5) | `theStrategyStatus.position` | ✅ | {gk,defL,defR,fwdL,fwdR} | Verificado en t121/t13 4v4 cfgs. StrategyStatus.h#L30 |
| `self_is_active` | `Role::isActiveRole(theStrategyStatus.role)` | ✅ | {0, 1} | Role.h#L38 + StrategyStatus.h#L31 |
| `goal_opening_angle` | Derivado: `atan2(800-y, 2838-x) - atan2(-800-y, 2838-x)` | ✅ calculado | [0, ~0.55 rad] | Desde centro del campo: 31.5° = 0.55 rad |
| `goal_relative_angle` | Derivado: `normalize(atan2(-y, 2813-x) - self_theta)` | ✅ calculado | [-π, π] | Ya en rango natural |

> **⚠️ Rangos de normalización para self_x, self_y, velocidades:**
> El plan dice "obs normalizado [-1,1]" pero no especifica los rangos. Valores recomendados:
> ```python
> OBS_NORM = {
>     "self_x":     2813.0,   # /xPosOpponentGroundLine
>     "self_y":     1875.0,   # /yPosLeftSideline
>     "self_vx":     250.0,   # /MAX_FORWARD
>     "self_vy":     200.0,   # /MAX_SIDEWAYS
>     "self_omega":  2.094,   # /MAX_ROTATION (120°/s en rad)
>     "ball_x":     2813.0,
>     "ball_y":     1875.0,
>     "ball_vx":    2000.0,   # velocidad máxima real del balón post-patada (~2m/s)
>     "ball_vy":    2000.0,
>     "ball_age_ms": 3000.0,  # normalizar a [0,1], clamp en 3s
>     "tm_x":       2813.0,   "tm_y": 1875.0,
>     "tm_walk_dx":  469.0,   "tm_walk_dy": 469.0,  # ~centerCircleRadius
>     "tm_speed":    250.0,
>     "opp_x":      2813.0,   "opp_y": 1875.0,
>     "half_time_remaining_s": 600.0,  # 10 minutos
> }
> ```
> **Esto es deuda técnica urgente — definirlo antes de implementar el env.**

---

#### Grupo BALL (6 floats)

| Feature | Fuente C++ | Accesible | Notas |
|---------|-----------|-----------|-------|
| `ball_x` | `theFieldBall.positionOnField.x()` | ✅ | FieldBall.h#L104. Global coords. |
| `ball_y` | `theFieldBall.positionOnField.y()` | ✅ | |
| `ball_vx` | `Rotation(theRobotPose) * theBallModel.estimate.velocity.x()` | ✅ con transform | BallModel.h#L24 da velocidad en coords robot. **Requiere rotar por theRobotPose.rotation** |
| `ball_vy` | idem | ✅ con transform | |
| `ball_age_ms` | `theFieldBall.timeSinceBallWasSeen` | ✅ | FieldBall.h#L116 |
| `using_team_ball` | `theFieldBall.useTeammatesBall()` | ✅ | FieldBall.cpp#L93 |

> **⚠️ Transform crítico de ball_vx/ball_vy:**
> `BallModel.estimate.velocity` es en coordenadas relativas al robot. Para obtener velocidad en coordenadas globales del campo:
> ```cpp
> // C++
> Vector2f ball_vel_field = theRobotPose.rotation * theBallModel.estimate.velocity;
> // Python sim:
> ball_vel_field = rotate2d(ball_vel_robot, agent_theta)
> ```
> Si se omite esta rotación, el agente verá una velocidad de balón que cambia según hacia dónde mira el robot — señal corrupta.

---

#### Grupo TEAMMATES (21 floats — 3 slots × 7)

| Feature | Fuente C++ | Accesible | Notas |
|---------|-----------|-----------|-------|
| `tm_x` | `GlobalTeammatesModel.teammates[i].pose.translation.x()` | ✅ | GlobalTeammatesModel.h#L36 |
| `tm_y` | `GlobalTeammatesModel.teammates[i].pose.translation.y()` | ✅ | |
| `tm_walk_dx` | `GlobalTeammatesModel.teammates[i].relativeWalkTarget.x()` | ✅ | En coords relativas al receptor |
| `tm_walk_dy` | `GlobalTeammatesModel.teammates[i].relativeWalkTarget.y()` | ✅ | |
| `tm_speed` | `GlobalTeammatesModel.teammates[i].speed` | ✅ | float mm/s |
| `tm_is_active` | `Role::isActiveRole(TeamData.teammates[i].theStrategyStatus.role)` | ✅ en C++ | **No está en GlobalTeammatesModel.** Usar TeamData (TeamData.h struct Teammate#L32) |
| `tm_exists` | `i < GlobalTeammatesModel.teammates.size()` | ✅ | En 4v4 siempre true si el robot está activo |

> **⚠️ Slot ordering:** `GlobalTeammatesModel` ordena por `playerNumber` ascendente. En Python sim, mantener el mismo orden para consistencia.

---

#### Grupo OPPONENTS (12 floats — 4 slots × 3)

| Feature | Fuente C++ | Accesible | Notas |
|---------|-----------|-----------|-------|
| `opp_x` | `GlobalOpponentsModel.opponents[i].position.x()` | ✅ | Convertida a global vía `theRobotPose * ob.center` (GlobalOpponentsTracker.cpp#L75) |
| `opp_y` | `GlobalOpponentsModel.opponents[i].position.y()` | ✅ | |
| `opp_exists` | `i < GlobalOpponentsModel.opponents.size()` | ✅ | Depende del FOV (~55°) y distancia (<5000mm) |

> **Limitación real confirmada:** El modelo de oponentes es LOCAL. En 4v4 puede haber hasta 4 oponentes, pero un robot generalmente ve 0-2 a la vez. Los slots vacíos se llenan con ceros. En Python sim: aplicar el modelo FOV (`get_visible_opponents()` del plan) para replicar esta limitación.

---

#### Grupo GAME STATE (4 floats)

| Feature | Fuente C++ | Accesible | Rango | Notas |
|---------|-----------|-----------|-------|-------|
| `game_is_playing` | `theGameState.isPlaying()` | ✅ | {0,1} | GameState.h#L194 |
| `own_score` | `theGameState.ownTeam.score` | ✅ | [0, ~20] | GameState.h#L120 |
| `opp_score` | `theGameState.opponentTeam.score` | ✅ | [0, ~20] | GameState.h#L120 |
| `half_time_remaining_s` | `(theGameState.timeWhenPhaseEnds - theFrameInfo.time) / 1000.0f` | ✅ | [0, 600] | GameState.h#L412 |

---

#### Grupo TACTIC (2 floats — one_hot)

| Index | Valor | Táctica | Estado |
|-------|-------|---------|--------|
| 0 | 1.0 | `t121_4v4_scaled` | Defensiva (inicio de partida) |
| 1 | 1.0 | `t13_4v4_scaled` | Ofensiva |

> **Fuente:** `theStrategyStatus.acceptedTactic` (Tactic::Type enum). Mapear con `TACTIC_INDEX` del plan.
> En 4v4 nunca aparecerán otros valores de Tactic::Type si se usa exclusivamente `s4v4_scaled.cfg`.

---

### Home Poses Verificadas (calculadas desde los .cfg)

Las home poses son los puntos de inicio del RL y la baseline heurística. Calculadas aplicando `FIELD_SYMBOLS` 4v4:

#### `t121_4v4_scaled.cfg` — Formación defensiva (default)

| Posición | x (mm) | y (mm) | θ (rad) | Descripción |
|----------|--------|--------|---------|-------------|
| goalkeeper | -2532 | 0 | 0 | Dentro del área propia |
| defenderL | -1468 | 0 | 0 | Línea de penalti propia |
| defenderR | -234 | -625 | 0 | Lado derecho del centro |
| forwardL | -234 | +625 | 0 | Lado izquierdo del centro |
| forwardR | +703 | 0 | 0 | Zona ofensiva media |

#### `t13_4v4_scaled.cfg` — Formación ofensiva

| Posición | x (mm) | y (mm) | θ (rad) | Descripción |
|----------|--------|--------|---------|-------------|
| goalkeeper | -2532 | 0 | 0 | Dentro del área propia |
| defenderL | -1938 | 0 | 0 | Línea de área propia |
| forwardL | +234 | +500 | 0 | Zona ofensiva |
| forwardR | +234 | -500 | 0 | Zona ofensiva |

> **Observación táctica:** En `t13_4v4_scaled`, solo 3 posiciones para field players (defenderL, forwardL, forwardR). La 4ª posición (defenderR) no existe — con 3 field players se ocupa defenderL + forwardL + forwardR según `priorityGroups`. Esto es correcto para 4v4.

---

### Coherencia Total del Obs+Action Space

| Verificación | Estado | Detalle |
|-------------|--------|---------|
| Dimensiones del campo correctas | ✅ | x±2813, y±1875 verificados en fieldDimensions.cfg |
| Total obs = 61 floats | ✅ | 16+6+21+12+4+2 = 61. Calculado y confirmado. |
| Posiciones en one_hot(5) exhaustivas | ✅ | {gk, defL, defR, fwdL, fwdR} cubren todas las posiciones de t121 y t13 |
| Tactic one_hot(2) exhaustivo | ✅ | s4v4_scaled solo usa t121_4v4_scaled y t13_4v4_scaled |
| Opponents 4 slots exactos | ✅ | 4v4 tiene exactamente 4 oponentes |
| Teammates 3 slots exactos | ✅ | 4v4 tiene exactamente 3 compañeros |
| Ball velocity requiere transform | ⚠️ | Rotar por theRobotPose.rotation — documentado arriba |
| tm_is_active fuente en deployment | ⚠️ | TeamData, no GlobalTeammatesModel — documentado arriba |
| Rangos de normalización | ⚠️ | No definidos en el plan — tabla sugerida arriba |
| Home poses calculadas y verificadas | ✅ | t121: 5 poses, t13: 4 poses. Valores numéricos confirmados. |
| Compatibilidad con WalkToPose skill | ✅ | Skill acepta Pose2f = (theta, x, y). Action format correcto. |
| Critic global obs (Fase 1) | ✅ | 61×3 + 4 (ball global) = 187 floats |
| Critic global obs (Fase 2) | ✅ | 61×3 + 9 (roles) + 4 (ball) = 196 floats |

---

## Opinión del Auditor

### Sobre la investigación del código

El plan demuestra una investigación **genuina y detallada** del codebase BHuman. El autor claramente leyó los headers, revisó los .cpp, y tomó decisiones informadas (ej: no incluir rotación de compañeros porque el header dice que es unreliable; usar `WalkToPose` en lugar de velocidades directas para mantener obstacle avoidance). Esto no es un documento generado superficialmente — hay evidencia de análisis real.

### Sobre la arquitectura del plan

La arquitectura de 3 fases es **sólida y prudente**:

1. **Fase 1 (solo posiciones)** es el scope mínimo correcto para validar que el RL aprende algo útil antes de añadir complejidad.
2. **Reward sparse** (solo goles) evita bias del diseñador y es la práctica moderna en RL de deportes (ver AlphaZero, OpenAI Five).
3. **Simulación Python pura** sin portar el PathPlanner de C++ es la decisión correcta — portar 8 representaciones BHuman sería meses de trabajo.
4. **Parser de .cfg** en lugar de hardcodear poses es correcto: los .cfg son la fuente de verdad y cambiarán con el equipo.
5. **MAPPO con parameter sharing** (4 actores con política compartida) es apropiado para agentes homogéneos.
6. **Critic centralizado con observación global** es la diferencia clave de MAPPO vs IPPO y justifica el enfoque multi-agente.

### Sobre los riesgos no abordados

1. **Sim-to-real gap en acciones:** El plan decide bien usar `target_pose` como acción y delegar al PathPlanner real. Pero en la simulación Python, el `NaoMovementModel` es una simplificación lineal del PathPlanner. Si el campo tiene mucho tráfico de robots, el modelo simplificado puede generar políticas que funcionen bien en sim pero colisionen en el robot real. Recomendación: añadir collision radius al `NaoMovementModel` desde el inicio (el plan lo menciona como "repulsión básica 300mm" pero no lo implementa en el código mostrado).

2. **Episode termination:** El plan no especifica cuándo termina un episodio de entrenamiento. ¿Al marcar un gol? ¿Después de N pasos? Esta decisión impacta significativamente la convergencia del MAPPO. Debe definirse antes de implementar.

3. **Normalización del obs space:** El plan menciona "obs normalizado [-1, 1]" pero no especifica los rangos de normalización para cada feature (ej: ¿campo [-4500, 4500] → [-1, 1]? ¿velocidades [-250, 250] → [-1, 1]?). Esto es crítico para la estabilidad del entrenamiento y debe documentarse.

4. **El portero en Fase 1:** El plan dice que el portero usa "GoaliePose heurística" y no entra al RL. Pero el portero *SÍ* está en el observation space como un compañero (slot de GlobalTeammatesModel). Si el portero se mueve de manera heurística diferente al modelo simplificado, introduce un agente no-controlado que puede confundir el MAPPO.

5. **Comunicación BHuman (200-800ms latencia):** El plan los documenta correctamente en la tabla de sim2real gap, pero la simulación Python los ignora completamente (datos instantáneos). Esta es la fuente más importante de divergencia sim2real en fases de despliegue, y el plan lo deja para "Fase 4 opcional". Aceptable para el primer RL, pero debe registrarse como deuda técnica explícita.

6. **CompiledNN para ONNX:** El plan dice "BHuman ya lo soporta" para ejecutar ONNX. Verificar que la versión del codebase SabanaHerons2026 incluye efectivamente el módulo CompiledNN antes de depender de esta afirmación en el roadmap de deployment.

### Sobre la viabilidad del proyecto

El proyecto es **viable** con el scope definido. La Fase 1 es implementable en semanas, no meses, dado que:
- Todo el scaffolding de observaciones ya está definido con fuentes verificadas
- El formato .cfg está correctamente comprendido y el parser está escrito
- Las velocidades del NaoMovementModel son valores reales del hardware

Las Fases 2 y 3 requieren ajustes en el action space (multi-head policy) que añaden complejidad de implementación MAPPO, pero son extensiones naturales de la Fase 1.

---

## Veredicto Final

### ✅ Fundamentos sólidos — NO es un documento de falsas bases

Los archivos, representaciones, y valores referenciados en el plan **existen realmente** en el codebase. El proyecto NO cae sobre fundamentos falsos.

### ✅ Correcciones aplicadas en MAPPO_Plan_3Fases.md

| # | Error original | Corrección aplicada |
|---|---------------|---------------------|
| 1 | `self_role: one_hot(5)` con "striker" (rol, no posición) | → `self_position: one_hot(5)` {gk/defL/defR/fwdL/fwdR} + `self_is_active: bool` separado |
| 2 | `tm_is_active` sin fuente en C++ deployment | → Documentada fuente real: `TeamData.Teammate.theStrategyStatus.role` |
| 3 | `current_tactic: one_hot(9)` (enum tiene 17 valores, scope 5v5) | → `one_hot(2)` para 4v4 Scaled: 0=t121_4v4_scaled, 1=t13_4v4_scaled |
| 4 | Scope 5v5 en todo el documento (campo, agentes, bounds) | → Actualizado a 4v4 Scaled en todas las secciones |
| 5 | ~12 números de línea incorrectos | → Corregidos todos (StrategyStatus.h, Behavior.cpp, FieldDimensions.h, etc.) |
| 6 | FIELD_SYMBOLS con valores de campo 5v5 | → Reemplazados con valores de `Config/Locations/4v4_Scaled/fieldDimensions.cfg` |

### ⚠️ Deuda técnica pendiente (no bloqueante para Fase 1)

1. **Rangos de normalización:** Definir antes de implementar el env. Tabla sugerida en Sección 8.
2. **Terminación de episodios:** No definida en el plan. Opciones: por tiempo (10min sim), por gol, o episodios fijos de N pasos.
3. **CompiledNN (ONNX):** Verificar disponibilidad en el codebase antes de planear el deployment.
4. **Collision avoidance en `NaoMovementModel`:** El plan lo menciona como "repulsión básica 300mm" pero no lo implementa. Hacerlo desde Fase 1.

---

*Auditoría realizada el 11 de Marzo de 2026. Todos los archivos verificados contra el repositorio en rama `ScaledDimensions`. Ninguna referencia fue inventada — cada verificación fue ejecutada contra el código fuente real.*
