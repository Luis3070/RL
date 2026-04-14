# Analisis Unificado de CFGs para RL MAPPO — SabanaHerons2026

> **Fecha:** 21 de Marzo de 2026
> **Scope:** Campo 4v4 Scaled (5626 x 3750 mm) — 1 portero + 3 jugadores de campo
> **Objetivo:** Documento unico que consolida TODO el analisis de archivos .cfg relevantes para el proyecto RL MAPPO, unificando la informacion dispersa en `CFG_Analysis_4v4_MAPPO.md`, `CFG_ANALYSIS_COMPLETE_RL.md`, `RL_CFG_Analysis.md` y `DECISION_POINTS_CPP.md`.

---

## Tabla de Contenidos

1. [Jerarquia de CFGs y su Rol en el Sistema](#1-jerarquia)
2. [Capa 1: Strategy — FSM de Tacticas](#2-strategy)
3. [Capa 2: Tactics — Posicionamiento y Formaciones](#3-tactics)
4. [Capa 3: Actions — Shoot, Pass, Dribble, Zweikampf](#4-actions)
5. [Capa 4: Kick Info — Propiedades Fisicas de Pateos](#5-kickinfo)
6. [Capa 5: Walk Provider — Movimiento y Evasion](#6-walkprovider)
7. [Capa 6: Behavior Parameters — Configuracion Global](#7-behavior)
8. [Flujo de Decision C++: SmashOrPass](#8-smashOrPass)
9. [Tabla Maestra: Parametros Aprendibles vs Hard Constraints](#9-tabla-maestra)
10. [Impacto por Fase del Proyecto RL](#10-fases)
11. [Diagnostico: Estado Actual del Proyecto vs CFGs Reales](#11-diagnostico)

---

<a name="1-jerarquia"></a>
## 1. Jerarquia de CFGs y su Rol en el Sistema

```
CONFIG ROOT (BHuman)
│
├── CAPA 1: STRATEGY (Decision global — FSM)
│   └── Config/Behavior/Strategies/s4v4_scaled.cfg
│        ├── Define CUANDO cambiar de formacion
│        ├── 2 Estados: t121_4v4_scaled (defensa) <-> t13_4v4_scaled (ataque)
│        ├── Transiciones: ballX, tiempo, num jugadores
│        └── Set plays: kickoffs, penalties, free kicks, corners
│
├── CAPA 2: TACTICS (Posicionamiento)
│   ├── Config/Behavior/Tactics/t121_4v4_scaled.cfg
│   │    ├── Formacion 1-1-1 (1 defensor, 1 medio, 1 delantero)
│   │    ├── 5 posiciones (incl. portero)
│   │    └── Priority groups (asignacion de robots)
│   │
│   └── Config/Behavior/Tactics/t13_4v4_scaled.cfg
│        ├── Formacion 1-0-2 (1 defensor, 2 delanteros)
│        └── 4 posiciones (incl. portero, sin defenderR)
│
├── CAPA 3: ACTIONS (Comportamientos Elementales)
│   └── Config/Scenarios/4v4_Scaled/BehaviorControl/
│        ├── kickAtGoalImpl.cfg          → Disparar al arco (activo en partidos)
│        ├── passToTeammateImpl.cfg      → Pasar a companero (activo en partidos)
│        └── zweikampfImpl.cfg           → Duelo, dribble y pateo en presion (activo en partidos, pero NO es el unico skill)
│
├── CAPA 4: KICK INFO (Propiedades fisicas)
│   └── Config/Scenarios/4v4_Scaled/kickInfo.cfg
│        └── 23 tipos de pateo definidos; esta guia detalla sobre todo los 18 mas relevantes
│
├── CAPA 5: WALK PROVIDER (Movimiento)
│   └── Config/Scenarios/4v4_Scaled/libWalkProvider.cfg
│        └── Evasion de obstaculos, limites de campo, radios
│
└── CAPA 6: BEHAVIOR PARAMS (Global)
    ├── Config/Scenarios/4v4_Scaled/behaviorParameters.cfg
    ├── Config/Scenarios/4v4_Scaled/strategyBehaviorControl.cfg
    └── Config/Robots/Default/walkingEngine.cfg
```

**Codigo C++/headers criticos que consumen o enrutan estos CFGs:**
- `Src/Modules/BehaviorControl/StrategyBehaviorControl/StrategyBehaviorControl.cpp`
  - `update()` llama `theBehavior.update(...)` y produce `SkillRequest`
- `Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp` (1159 lineas)
  - L445-480: Evaluacion de condiciones de transicion de tactica
  - L495-735: `assignPositions()` — asignacion de posiciones
  - L738-768: `assignRoles()` — asignacion de roles
  - L910-1140: `determineActiveAgent()` — quien es striker
- `Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp`
  - L33-118: `smashOrPass()` — decision shoot vs pass
- `Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.h`
  - Define el contrato `SkillRequest` → `MotionRequest`/`BehaviorStatus`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.cpp`
  - L89: `PlaySoccer()`
  - L97: `executeRequest()`
  - L112: `thePlayBallSkill()`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/Options/PlaySoccer.h`
  - L15: `select_option(options)`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/Options/HandleGameState.h`
  - L28-29: `select_option(playingOptions)` o `executeRequest()`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/PlayBall.cpp`
  - Enruta `shoot/pass/dribble` hacia `KickAtGoal`, `PassToTeammate`, `GoToBallAndDribble`, `HandleBallAtOwnGoalPost` o `Zweikampf`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/KickAtGoal.cpp`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/PassToTeammate.cpp`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/Zweikampf.cpp`

---

<a name="2-strategy"></a>
## 2. Capa 1: Strategy — FSM de Tacticas

**Archivo:** `Config/Behavior/Strategies/s4v4_scaled.cfg`
**Codigo C++:** `Behavior.cpp` L445-480

### 2.1 Maquina de Estados

```
          ballX > 500mm
          sostenido 280ms
          >=3 field players
    ┌──────────────────────────┐
    │                          ▼
┌───────────────┐      ┌───────────────┐
│  t121_4v4     │      │  t13_4v4      │
│  (Defensa)    │      │  (Ataque)     │
│  Formacion    │      │  Formacion    │
│  1-1-1        │      │  1-0-2        │
└───────────────┘      └───────────────┘
    ▲                          │
    └──────────────────────────┘
          ballX < -500mm
          sostenido 280ms
                O
          <=2 field players
```

### 2.2 Parametros de Transicion

| Parametro | Valor | Tipo | Rango Viable RL | Prioridad |
|-----------|-------|------|-----------------|-----------|
| `numOfFieldPlayersGE` (def→atq) | 3 | INT | [1, 3] | CRITICA |
| `numOfFieldPlayersLE` (atq→def) | 2 | INT | [1, 3] | CRITICA |
| `ballXThreshold` (def→atq) | 500 | MM | [0, 1000] | CRITICA |
| `ballXThreshold` (atq→def) | -500 | MM | [-1000, 0] | CRITICA |
| `timeSinceBallAheadOfThresholdGE` | 280 | MS | [100, 500] | CRITICA |
| `timeSinceBallBehindThresholdGE` | 280 | MS | [100, 500] | CRITICA |

**Por que importa la hysteresis temporal (280ms):** Sin ella, si el balon cruza x=500 durante 50ms, el sistema cambiaria violentamente de formacion cada frame. El delay de 280ms confirma que la situacion es estable antes de cambiar.

### 2.3 Set Plays (Jugadas Especiales)

| Situacion | CFG Entry | Peso |
|-----------|-----------|------|
| Saque inicial propio | `attackKickOff_scaled` | 1 (fijo) |
| Defensa saque rival | `kiteKickOff_scaled` | 1 (fijo) |
| Penal propio | `theOneTrueOwnPenaltyKick_scaled` | 1 (fijo) |
| Defensa penal rival | `theOneTrueOpponentPenaltyKick_scaled` | 1 (fijo) |
| Esquina propia | `ownCornerKick_scaled` | 1 (fijo) |
| Saque de meta propio | `ownGoalKick_scaled` | 1 (fijo) |
| Defensa esquina rival | `opponentCornerKick_scaled` | 1 (fijo) |

**Para RL:** Los set plays son **NO aprendibles** (weight=1 fijo). RL debe detectar `game_state != PLAYING` y ceder control al set play hardcodeado.

---

<a name="3-tactics"></a>
## 3. Capa 2: Tactics — Posicionamiento y Formaciones

### 3.1 Tactica Defensiva: `t121_4v4_scaled.cfg` (Formacion 1-1-1)

```
          ┌─── Campo 4v4 Scaled (5626 x 3750 mm) ───┐
          │                                           │
  y=+1875 │  forwardL                                 │
          │  (-234.5, +625)                           │
          │            ○                              │
          │                                           │
          │  defenderL           forwardR             │
  y=0     │  (-1468, 0)         (+703.5, 0)   ← ARCO │
          │   ○                    ○           RIVAL  │
          │                                           │
          │  defenderR                                │
          │  (-234.5, -625)                           │
          │            ○                              │
  y=-1875 │                                           │
          └───────────────────────────────────────────┘
        x=-2813                                  x=+2813
```

| Posicion | Tipo | X (mm) | Y (mm) | Expresion Original |
|----------|------|--------|--------|--------------------|
| Portero | goalkeeper | -2532 | 0 | `xPosOwnGoalArea - 94` = -2438 - 94 |
| Defensor Izq | defenderL | -1468 | 0 | `xPosOwnPenaltyArea + 313` = -1781 + 313 |
| Defensor Der | defenderR | -234.5 | -625 | `-centerCircleRadius * 0.5` = -469 * 0.5 |
| Delantero Izq | forwardL | -234.5 | +625 | `-centerCircleRadius * 0.5` = -469 * 0.5 |
| Delantero Der | forwardR | +703.5 | 0 | `centerCircleRadius * 1.5` = 469 * 1.5 |

**Priority Groups (asignacion de robots por habilidad):**

| Prioridad | Posicion | Logica |
|-----------|----------|--------|
| 1 (mejor robot) | defenderR | Medio campo lado derecho |
| 2 | forwardR | Ataque central |
| 3 | forwardL | Medio campo lado izquierdo |
| 4 (peor robot) | defenderL | Defensa central |

### 3.2 Tactica Ofensiva: `t13_4v4_scaled.cfg` (Formacion 1-0-2)

```
          ┌─── Campo 4v4 Scaled (5626 x 3750 mm) ───┐
          │                                           │
  y=+1875 │                                           │
          │                                           │
          │               forwardL                    │
          │               (+234.5, +500)              │
          │  defenderL         ○              ← ARCO  │
  y=0     │  (-1938, 0)                       RIVAL   │
          │   ○                                       │
          │               forwardR                    │
          │               (+234.5, -500)              │
          │                    ○                      │
  y=-1875 │                                           │
          └───────────────────────────────────────────┘
        x=-2813                                  x=+2813
```

| Posicion | Tipo | X (mm) | Y (mm) | Expresion Original |
|----------|------|--------|--------|--------------------|
| Portero | goalkeeper | -2532 | 0 | `xPosOwnGoalArea - 94` = -2438 - 94 |
| Defensor | defenderL | -1938 | 0 | `xPosOwnGoalArea + 500` = -2438 + 500 |
| Delantero Izq | forwardL | +234.5 | +500 | `centerCircleRadius * 0.5` = 469 * 0.5 |
| Delantero Der | forwardR | +234.5 | -500 | `centerCircleRadius * 0.5` = 469 * 0.5 |

**SIN defenderR:** Formacion agresiva con 1 defensor y 2 delanteros.

### 3.3 Comparativa entre Tacticas

| Aspecto | t121 (Defensa) | t13 (Ataque) | Delta |
|---------|---------------|-------------|-------|
| Defensor mas adelantado X | -234.5mm (defR) | -1938mm (defL) | Retrocede 1703.5mm |
| Atacantes X | +703.5mm (fwdR, centro) | +234.5mm (fwdL/R, abiertos) | Retroceden 469mm |
| Atacantes Y | 0mm (unico, central) | ±500mm (2, abiertos) | Separacion lateral |
| Num defensores | 2 (defL + defR) | 1 (defL) | Pierde 1 |
| Num delanteros | 1 (fwdR) | 2 (fwdL + fwdR) | Gana 1 |

### 3.4 Parametros Aprendibles de Tacticas

| Parametro | t121 | t13 | Rango RL | Prioridad |
|-----------|------|-----|----------|-----------|
| striker_x | +703.5 | +234.5 | [-500, 2500] mm | CRITICA |
| striker_y | 0 | ±500 | [-1500, 1500] mm | CRITICA |
| defender_x | -1468 | -1938 | [-2500, -200] mm | CRITICA |
| defender_y | 0 | 0 | [-500, 500] mm | ALTA |
| priority_order | [1,2,3,4] | [1,2,3] | variable | ALTA |

---

<a name="4-actions"></a>
## 4. Capa 3: Actions — Shoot, Pass, Dribble, Zweikampf

### 4.1 kickAtGoalImpl.cfg — Disparar al Arco

**C++:** `KickAtGoal.cpp`

| Parametro | Valor | Funcion | Rango RL | Prioridad |
|-----------|-------|---------|----------|-----------|
| `hysteresisNumber` | 300ms | Inercia en decision (evita cambios rapidos) | [100, 500] | CRITICA |
| `hysteresisKickRangeExtension` | 500mm | Extension de rango por hysteresis | FIJO | - |
| `hysteresisAngle` | 10deg | Angulo de hysteresis | [5, 20] | MEDIA |
| `minOpeningAngle` | 6deg | Angulo MINIMO libre al arco para disparar | [2, 15] | CRITICA |
| `kickInaccuracy` | 10deg | Error simulado en tiro (ruido) | FIJO (no RL) | - |
| `hysteresisDecisionPenalty` | 500ms | Penalidad temporal por cambio de decision | FIJO | - |
| `walkForwardLongMalus` | -300 | Penalidad para pateos largos caminando | FIJO | - |

**Pateos permitidos:** `forwardFastLeftLong`, `forwardFastRightLong`, `walkForwardsLeftLong`, `walkForwardsRightLong`

**Rango de distancia:** Sin limite explicito (min=1.17549e-38, max=3.40282e+38 — IEEE float limits)

### 4.2 passToTeammateImpl.cfg — Pasar a Companero

**C++:** `PassToTeammate.cpp`

| Parametro | Valor | Funcion | Rango RL | Prioridad |
|-----------|-------|---------|----------|-----------|
| `minKickLengthForLong` | 2813mm | Si dist > 2813mm, usar pase LARGO | [1000, 4000] | CRITICA |
| `ratingThreshold` | 0.7 | Pase valido solo si rating >= 70% | [0.3, 0.9] | CRITICA |
| `passAheadDistance` | 0mm | No anticipar movimiento (patear directo) | [-500, 500] | BAJA |
| `minAngleOffset` | 7deg | Desviacion angular minima | [2, 15] | ALTA |
| `maxAngleOffset` | 25deg | Desviacion angular maxima | [10, 45] | ALTA |
| `maxAngleOffsetBehind` | 10deg | Max angulo si pase es hacia atras | FIJO | - |
| `maxPrecisionOffset` | 5deg | Precision maxima del pase | FIJO | - |
| `minTimeWaiting` | 3000ms | Espera minima al companero | [1000, 5000] | MEDIA |
| `maxTimeWaiting` | 8000ms | Espera maxima al companero | [5000, 15000] | MEDIA |
| `ignoreObstaclesThreshold` | 250mm | Ignorar obstaculos si < 250mm | FIJO | - |
| `lookAhead` | false | No predecir movimiento futuro | FIJO (por ahora) | - |
| `doSpecialKickChecks` | true | Chequeos especiales de pateo | FIJO | - |
| `waitingPositionOffset` | (-94, 0) mm | Offset de posicion de espera | FIJO | - |
| `targetOffsetToFieldBorder` | 500mm | Margen respecto al borde | FIJO | - |
| `kickRangeOffset` | 313mm | Offset de rango de pateo | FIJO | - |
| `averageKickExecutionTime` | 800ms | Tiempo promedio de ejecucion | FIJO | - |
| `ignoreDynamicObstaclesThreshold` | 500mm | Ignorar obstaculos dinamicos | FIJO | - |
| `minRating` | 0.01 | Rating minimo para considerar pase | FIJO | - |

**Pateos permitidos:** `forwardFastRightLong`, `forwardFastLeftLong`, `forwardFastRight`, `forwardFastLeft`, `forwardFastRightPass`, `forwardFastLeftPass`, `walkForwardsRightLong`, `walkForwardsLeftLong`, `walkForwardsRightAlternative`, `walkForwardsLeftAlternative`

### 4.3 zweikampfImpl.cfg — Sistema Completo de Duelo por Balon

**C++:** `Zweikampf.cpp` | **NOTA CRITICA:** `Zweikampf` SI es un skill real y activo en partidos 4v4, pero NO es el unico. La ruta real pasa por `SkillBehaviorControl.cpp` → `Skills/PlayBall.cpp`, que segun `SkillRequest` puede llamar `KickAtGoal`, `PassToTeammate`, `GoToBallAndDribble`, `HandleBallAtOwnGoalPost` o `Zweikampf`. `demoGoToBallAndKickImpl.cfg` NO se usa en partidos — solo en modo demo de exhibicion (AnyPlace Demo / LibDemo::soccer).

Zweikampf implementa un sistema de **decision de kick multi-criterio** con 6 subsistemas parametrizados. NO es simplemente "pelear por el balon" — define como el robot selecciona el mejor kick considerando obstaculos, angulos de gol, robo de balon y hysteresis.

#### Sub-sistema 1: SearchParameters (9 campos)

| Parametro | Valor | Funcion |
|-----------|-------|---------|
| `numOfAnglesNearBestDuelPose` | 3 | Angulos muestreados alrededor del mejor angulo previo |
| `rangeOfBestDuelPose` | 10deg | Rango de busqueda local alrededor del ultimo kick |
| `numOfOverallSearch` | 5 | Muestras adicionales fuera del rango local |
| `overallSearchRange` | 180deg | Busqueda en semicirculo |
| `bonusForStealBallDirectionAdjustmentSmall` | 0.01deg | Expansion minima para intento de robo |
| `bonusForStealBallDirectionAdjustmentBig` | 3deg | Expansion grande para intento de robo |
| `moreSearchAfterDoingNothing` | 50 | Angulos extra tras tiempo sin kick valido |
| `goalSectorWidth` | 1deg | Ancho minimo del sector del arco para considerar gol |
| `minMaxAngleAngleRange` | {min=4deg; max=10deg} | Rango permitido de ajuste de rotacion del robot |

#### Sub-sistema 2: DuelRatings (12 campos — sistema de puntuacion)

Los ratings son NEGATIVOS = MEJOR (menor es mas preferible):

| Parametro | Valor | Funcion |
|-----------|-------|---------|
| `ratingMinMaxDifference` | 0.3 | Malus por rotacion necesaria para el kick |
| `ratingBallLandsInOwnHalf` | 0.2 | Malus si el balon cae en campo propio |
| `ratingPoseBlockedSmallKickAngle` | 0.3 | Malus si obstaculizado en kicks de angulo pequeno |
| `ratingPoseBlocked` | 1.0 | Malus si pose de kick bloqueada por obstáculo |
| `ratingOpponentFaster` | 0.1 | Malus si el rival llega antes al balon |
| `ratingSameKick` | -0.1 | BONUS por repetir el mismo tipo de kick |
| `ratingSameKickAngle` | -0.1 | BONUS por repetir kick con angulo similar |
| `ratingStealBall` | -0.4 | BONUS por robar el balon al rival |
| `goalAreaLongKickRating` | -0.5 | BONUS para despejar cerca del arco propio |
| `ratingStealBallKick` | -0.9 | BONUS por kicks de tipo steal ball |
| `ratingStealBallKickBetterSide` | -0.05 | BONUS por elegir el mejor lado del forwardSteal |
| `ratingGoalShot` | -1.5 | MAXIMO BONUS: kick que llega al arco rival |

#### Sub-sistema 3: DuelTimings (6 campos)

| Parametro | Valor | Funcion |
|-----------|-------|---------|
| `maxTimeDoingNothing` | 1000ms | Tras este tiempo sin kick valido, Zweikampf se desactiva |
| `deactiveAfterDoingNothing` | 5000ms | Tiempo de espera antes de reactivarse tras desactivacion |
| `sidewardRestrictionTime` | 4000ms | Hysteresis para restriccion de kicks laterales |
| `noKickMinTime` | 200ms | Cooldown minimo entre kicks (no forwardSteal) |
| `noKickStealMinTime` | 250ms | Cooldown minimo entre kicks de robo |
| `ignoreSkillRequestTime` | 2000ms | Ignorar requests de pase si no puede ejecutarlos |

#### Sub-sistema 4: ObstacleHandling (14 campos — geometria de pie)

Este subsistema modela la geometria fisica de los pies del robot para detectar colisiones reales:

| Parametro | Valor | Funcion |
|-----------|-------|---------|
| `leftFootEdge` | {x=110; y=100} | Borde izquierdo del pie izquierdo (mm, frame local) |
| `rightFootEdge` | {x=110; y=-100} | Borde derecho del pie derecho |
| `leftFootKickEdge` | {x=170; y=100} | Borde pie izquierdo despues del kick |
| `rightFootKickEdge` | {x=170; y=-100} | Borde pie derecho despues del kick |
| `maxObstacleDistanceForWalkStealBallKick` | 100mm | Distancia max oponente-balon para bloquear forwardSteal |
| `maxObstacleDistanceForSideStealBallKick` | 450mm | Idem para kicks laterales de robo |
| `obstacleDistanceForBlockingPose` | 100mm | Proyectar obstaculopara evaluar si bloquea kick |
| `obstacleBlockingRadius` | 200mm | Radio de colision para evaluar foot edges |
| `obstacleBlockingRadiusSidewardsIncrease` | 30mm | Radio extra para kicks laterales |
| `noKickBlockingPose` | 180mm | Distancia al balon sin kick calculado |
| `obstacleMinDistanceForBackPass` | 600mm | Oponente debe estar a esta distancia para permitir pase atras |
| `obstacleShiftRange` | {min=-100; max=30} | Rango de desplazamiento de obstaculos cercanos al balon |
| `maxObstacleDistanceToBallForRiskyKicks` | 600mm | Distancia max para kicks de riesgo |
| `maxObstacleDistanceToForceForwardSteal` | 200mm | Forzar forwardSteal si rival esta muy cerca |

#### Sub-sistema 5: StealBallParameters (7 campos)

Define cuando se puede ejecutar el "forwardSteal" (kick en V para robar):

| Parametro | Valor | Funcion |
|-----------|-------|---------|
| `stealBallTypeTooMuchRotationMinValue` | 25deg | Rotacion maxima tolerable para steal kicks |
| `bonusForStealBallMinMaxAngle` | 120deg | Angulo de bonus para steal en rango optimo |
| `stealBallRange` | {min=88deg; max=95deg} | Rango de direcciones de campo con bonus de steal |
| `stealBallMinRobotRotation` | 70deg | Rotacion propia minima para bonus lateral |
| `stealBallMinPositionRotation` | 45deg | Angulo del balon en campo minimo para bonus lateral |
| `forwardStealBallOpponentPositionNormal` | {min=50deg; max=70deg} | Angulo rival relativo al balon para permitir forwardSteal |
| `forwardStealBallOpponentPositionMalus` | 40deg | Umbral reducido si ultimo kick no fue forwardSteal |

#### Sub-sistema 6: LastKickHysterese (2 campos)

| Parametro | Valor | Funcion |
|-----------|-------|---------|
| `kickLengthHysterese` | 500mm | Si rango del nuevo kick difiere < 500mm, mantener el anterior |
| `kickDirectionHysterese` | 20deg | Si direccion difiere < 20deg, mantener el anterior |

#### Parametros sueltos adicionales (9 campos)

| Parametro | Valor | Funcion |
|-----------|-------|---------|
| `smallerKickAnglePreference` | 15deg | Preferencia por kicks de angulo menor |
| `forwardStealPreferenceRange` | 10deg | Rango para seleccionar el mejor lado del forwardSteal |
| `replaceNormalToLongKickGoalShot` | 10deg | Umbral para reemplazar kick normal por long en gol |
| `sectorWheelObstacleBallRadiusFactor` | 3 | Factor de expansion de obstaculos por radio del balon |
| `allowPassAngle` | {min=-45deg; max=45deg} | Angulos de campo donde se permiten pases de skill request |
| `fieldBorderSafeDistance` | 625mm | Margen seguro respecto al borde para kicks con incertidumbre |
| `minForwardTurnRange` | 625mm | Rango minimo de kicks forward/turn fuera del area rival |
| `sidewardsRotationCorrectionBonus` | 5deg | Bonus de rotacion si ultimo kick fue lateral |
| `ballPositionInterpolationRange` | {min=250; max=500} | Interpolacion posicion percibida vs interceptada segun distancia |
| `lookActiveMinBallDistance` | 500mm | Distancia minima para usar lookActive con withBall=true |
| `replaceForwardWithLongGoalShot` | true | Reemplazar forward por long al disparar al gol |

**Pateos permitidos en duelo (8 tipos):**
`walkForwardsLeft`, `walkForwardsRight`, `walkForwardsLeftLong`, `walkForwardsRightLong`, `walkForwardsRightAlternative`, `walkForwardsLeftAlternative`, `walkForwardStealBallLeft`, `walkForwardStealBallRight`

**Dependencias de runtime (no documentadas previamente):**
- `REQUIRES(FieldRating)` — campo vectorial de calidad posicional
- `REQUIRES(WalkingEngineOutput)` — velocidad actual del robot
- `REQUIRES(SkillRequest)` — lee la intencion de PlayBall y puede sobreescribirla
- `REQUIRES(TeamData)` — posiciones de companeros para evaluar pases

**Rol en el proyecto RL por fase:**
- Fase 1: El curriculum de approach debe alinearse con el ejecutor real (`SkillBehaviorControl` + `Skills/PlayBall`); cuando hay duelo, Zweikampf opera sin cambios
- Fase 2: `noKickMinTime=200ms` y `fieldBorderSafeDistance=625mm` son los constraints de timing y posicion reales del approach (NO los de demoGoToBallAndKick)
- Fase 3: RL replica o reemplaza la logica completa del sistema de puntuacion de DuelRatings

---

<a name="5-kickinfo"></a>
## 5. Capa 4: Kick Info — Propiedades Fisicas de Pateos

**Archivo:** `Config/Scenarios/4v4_Scaled/kickInfo.cfg`
**Estructura:** Mapa con nombre (`kicks = { forwardFastRight = {...}; forwardFastLeft = {...}; ... };`) — NO es un array, es un dict de structs nombrados. Total real: 23 tipos de pateo. Esta guia resume en detalle los 18 de mayor relevancia operativa, pero el parser y la auditoria deben conservar cobertura de los 23.

### 5.1 Tabla de Pateos Principales

| Tipo de Pateo | Rango Min (mm) | Rango Max (mm) | Vel Min (mm/s) | Vel Max (mm/s) | Tiempo (ms) | Pierna | Uso |
|---------------|---------------|---------------|---------------|---------------|-------------|--------|-----|
| `walkForwardsRight` | 400 | 1000 | 489 | 774 | 500 | R | Pase CORTO |
| `walkForwardsLeft` | 400 | 1000 | 489 | 774 | 500 | L | Pase CORTO |
| `forwardFastRight` | 1600 | 4500 | 979 | 1643 | 1500 | R | Pase/Disparo MEDIO |
| `forwardFastLeft` | 1600 | 4500 | 979 | 1643 | 1500 | L | Pase/Disparo MEDIO |
| `forwardFastRightLong` | 5400 | 5400 | 1800 | 1800 | 1500 | R | Disparo LARGO |
| `forwardFastLeftLong` | 5400 | 5400 | 1800 | 1800 | 1500 | L | Disparo LARGO |
| `walkForwardsRightLong` | 3000 | 3500 | 1342 | 1449 | 500 | R | Pase LARGO caminando |
| `walkForwardsLeftLong` | 3000 | 3500 | 1342 | 1449 | 500 | L | Pase LARGO caminando |
| `walkForwardsRightAlternative` | 2000 | 2000 | 1095 | 1095 | 500 | R | Pase MEDIO alternativo |
| `walkForwardsLeftAlternative` | 2000 | 2000 | 1095 | 1095 | 500 | L | Pase MEDIO alternativo |
| `walkSidewardsRightFootToRight` | 1400 | 1400 | 917 | 917 | 250 | R | Pase LATERAL |
| `walkSidewardsLeftFootToLeft` | 1400 | 1400 | 917 | 917 | 250 | L | Pase LATERAL |
| `walkTurnRightFootToLeft` | 400 | 1700 | 490 | 1010 | 500 | R | Pase DIAGONAL |
| `walkTurnLeftFootToRight` | 400 | 1700 | 490 | 1010 | 500 | L | Pase DIAGONAL |

**Pateos presentes en el archivo real pero no detallados arriba por menor relevancia operativa:** `walkForwardsRightVeryLong`, `walkForwardsLeftVeryLong`, `walkTurnRightFootToLeftShifted`, `walkTurnLeftFootToRightShifted`, `newKick`

### 5.2 Hard Constraints de Pateo para RL

```
RESTRICCION ABSOLUTA: RL NO puede violar estos rangos
├── Distancia minima de pateo: 400mm
├── Distancia maxima de pateo: 5400mm
├── Velocidad maxima de balon: 1800 mm/s
├── Tiempo de ejecucion: 250-1500ms (robot esta "ocupado")
└── ballOffset: posicion relativa pie-balon para cada tipo
```

**Logica de seleccion de pateo:**
- Si distancia al target > `minKickLengthForLong` (2813mm) → pateo LARGO
- Si distancia < 2813mm → pateo CORTO o MEDIO
- C++ selecciona automaticamente el tipo optimo via `KickSelection`

---

<a name="6-walkprovider"></a>
## 6. Capa 5: Walk Provider — Movimiento y Evasion

**Archivo:** `Config/Scenarios/4v4_Scaled/libWalkProvider.cfg`

### 6.1 Velocidades Maximas (de walkingEngine.cfg)

| Parametro | configuratedParameters | walkSpeedParams | Usar en RL |
|-----------|------------------------|-----------------|------------|
| Forward (mm/s) | 250 | **350** | **250** (conservador — configuratedParameters) |
| Sideways (mm/s) | 200 | **270** | **200** |
| Backwards (mm/s) | 150 | **250** | **150** |
| Rotation (deg/s) | 120 | 120 | **120** |

**NOTA CRITICA:** `walkingEngine.cfg` tiene DOS bloques de velocidad: `configuratedParameters` (limites conservadores activos) y `walkSpeedParams` (limites maximos absolutos del hardware). El robot puede fisicamente alcanzar 350mm/s adelante. El parser debe leer `configuratedParameters` como el limite operativo del sim, pero documentar ambos para el sim2real.

### 6.2 Evasion de Obstaculos

| Tipo de Evasion | Distancia Inicio | Radio Min | Radio Max |
|-----------------|-----------------|-----------|-----------|
| Obstaculos generales | 480mm | 220mm | 600mm |
| Postes de arco | 400mm | 300mm | 1000mm |
| Balon (no pisar) | 465mm | 465mm | 765mm |
| Zona penal | 400mm | 300mm | 1000mm |
| Borde de campo | 400mm | 300mm | 1000mm |

### 6.3 Hard Constraints de Movimiento

| Restriccion | Valor | Tipo |
|-------------|-------|------|
| `fieldBorderExpansion` | 300mm | HARD — no salir del campo |
| `ballAvoidanceDistance` | 465mm | HARD — no pisar balon |
| Max walk speed | 250 mm/s | HARD — limite fisico |
| Max rotation speed | 120 deg/s | HARD — limite fisico |
| Max backwards speed | 150 mm/s | HARD — limite fisico |

---

<a name="7-behavior"></a>
## 7. Capa 6: Behavior Parameters — Configuracion Global

### 7.1 behaviorParameters.cfg

| Parametro | Valor | Funcion |
|-----------|-------|---------|
| `keeperJumpingOn` | true | Portero puede saltar |
| `penaltyStrikerWalkSpeed` | 0.5 | En penal: marcha al 50% |
| `penaltyStrikerAngleToLeftPostOffset` | 10deg | Offset angular al poste izquierdo en penal |
| `penaltyStrikerAngleToRightPostOffset` | 10deg | Offset angular al poste derecho en penal |
| `penaltyStrikerUseObstacles` | true | Considerar obstaculos en penales |
| `ballCatchMaxWalkDistance` | 1200mm | Distancia maxima para atrapar el balon caminando |
| `standRadius` | 60mm | Radio: portero "parado" — InterceptBall no actua |
| `walkRadius` | 125mm | Radio: portero "camina" hacia el balon |
| `genuflectRadius` | 200mm | Radio: portero ejecuta genuflexion lateral |
| `genuflectStandRadius` | 200mm | Radio: portero hace genuflexion y se queda parado — **critico para RL**: define la zona de intercepcion real del portero en Fase 1 |
| `jumpRadius` | 600mm | Radio maximo: portero salta para interceptar |

### 7.2 strategyBehaviorControl.cfg

```
strategy = s4v4_scaled;   ← Selecciona la estrategia activa
```

### 7.3 skillBehaviorControl.cfg — Opciones de Comportamiento

Opciones generales: `HandlePhysicalRobot`, `HandlePlayerState`, `HandleReturnFromSideline`, `HandlePenaltyShootout`, `HandlePenaltyKick`, `HandleRefereeSignal`, `HandleGameState`, `HandleIllegalAreas`

Opciones en juego: `HandleGoalkeeperCatchBall`, `HandleStrikerLostBall`, `HandleCatchBall`

**Rol operativo en partido (critico para el RL aunque no sea un knob aprendible):**
1. `PlaySoccer.h` selecciona `options`
2. `HandleGameState.h` decide si una `playingOption` intercepta el flujo; si no, llama `executeRequest()`
3. `SkillBehaviorControl.cpp` enruta `SkillRequest::shoot/pass/dribble` hacia `thePlayBallSkill()`
4. `SkillBehaviorControl/Skills/PlayBall.cpp` decide si ejecutar `KickAtGoal`, `PassToTeammate`, `GoToBallAndDribble`, `HandleBallAtOwnGoalPost` o `Zweikampf`

**Conclusion:** `skillBehaviorControl.cfg` no define acciones del RL, pero si define la puerta de entrada y el gateo de la ejecucion real. Debe auditarse junto con `SkillBehaviorControl.h/.cpp`, `PlaySoccer.h`, `HandleGameState.h` y `Skills/PlayBall.cpp`.

---

<a name="8-smashOrPass"></a>
## 8. Flujo de Decision C++: SmashOrPass

**Archivo:** `PlayBall.cpp` L33-118 | `PlayBall.h` L16-35

### 8.1 Parametros del CFG en PlayBall.h

```cpp
// PlayBall.h L16-35 — TODOS los parametros de decision
Parameters p {
    (bool)(false)   alwaysShoot,         // L18: Debug — forzar siempre disparo
    (bool)(false)   ignoreObstacles,     // L19: Switch legacy/smashOrPass
    (float)(0.8f)   shootThreshold,      // L20: Si xG > 0.8 → DISPARA ← APRENDIBLE
    (float)(1500.f) minPassDistance,     // L22: Pase minimo 1.5m ← APRENDIBLE
    (float)(4500.f) maxPassDistance,     // L23: Pase maximo 4.5m ← APRENDIBLE
    (float)(6000.f) maxFreeKickDistance, // L24: En tiros libres: 6m (HARD)
    (float)(9000.f) maxGoalKickDistance, // L25: En saques de meta: 9m (HARD)
    (float)(0.1f)   minPenalty,          // L26: Penalidad minima cambio ← APRENDIBLE
    (float)(0.8f)   maxPenalty,          // L27: Penalidad maxima cambio ← APRENDIBLE
    (float)(300.f)  ballDistMinPenalty,  // L28: Aplicar si balon > 300mm (HARD)
    (float)(200.f)  ballDistMaxPenalty,  // L29: Maximizar si balon < 200mm (HARD)
    (float)(0.01f)  minRating,           // L30: Rating minimo (HARD)
    (bool)(true)    lookAhead,           // L31: ← FLAG MUERTO — declarado pero sin if(p.lookAhead) en PlayBall.cpp. TODO nunca implementado.
    (int)(2000)     lookAheadTime,       // L32: ← FLAG MUERTO — idem L31
    (float)(500.f)  sampleDistance,      // L33: Distancia entre samples hacia arco ← APRENDIBLE
    (Rangei)(-1,2)  sampleSteps,         // L34: Pasos de muestreo min/max ← APRENDIBLE
};
```

> **NOTA:** Estos parametros NO tienen .cfg override en 4v4_Scaled. Son defaults del C++. Para Fase 2, RL inyecta valores alternativos directamente en los puntos de decision (ver Seccion 8.4).

### 8.2 Flujo Completo de Decision

```
FRAME N: STRIKER TIENE BALON
│
├─ PlayBall::smashOrPass()
│  │
│  ├─ L44: xG_shoot = expectedGoals.getRating(ballPos)
│  │  "Probabilidad de gol si disparo AHORA" → [0.0, 1.0]
│  │
│  ├─ L39-51: decisionPenalty = mapToRange(ballDist, 200, 300, 0.8, 0.1)
│  │  "Penalidad por cambiar de decision (hysteresis)"
│  │
│  ├─ L54-55: Si lastPassTarget > 0: maxValue -= decisionPenalty
│  │  "Si estaba pasando, penalizar cambio a disparo"
│  │
│  ├─ L56: if(maxValue > shootThreshold)  ← CFG: 0.8
│  │  │
│  │  ├─ SI (xG > 0.8): return shoot()
│  │  │
│  │  └─ NO (xG <= 0.8): Buscar pases...
│  │
│  ├─ L64-104: Para cada companero:
│  │  ├─ L66-67: Descartar portero
│  │  ├─ L70-72: Validar distancia [1500mm, 4500mm]
│  │  ├─ L77-90: Samplear posiciones hacia arco
│  │  ├─ L82-84: rating = PassEval × xG_companero × defenseRating
│  │  ├─ L95-98: Aplicar hysteresis si cambia target
│  │  └─ L99-103: Guardar si es mejor
│  │
│  ├─ L105: if(passTarget existe): return passTo(passTarget)
│  │
│  └─ L117: Si no hay pase valido: return shoot() (esperanza)
│
├─ Si shoot → KickAtGoal.cpp L64-83:
│  └─ updateKickDirectionAndKick()
│     └─ Seleccionar tipo de pateo (largo/corto/potente)
│
└─ Si pass → PassToTeammate.cpp L121+:
   └─ Calcular kickTarget, kickAngle, kickLength
```

### 8.3 Parametros Criticos por Decision

| Decision | Parametro | Valor Actual | Linea C++ | Rango RL |
|----------|-----------|-------------|-----------|----------|
| Disparar o Pasar? | `shootThreshold` | 0.8 | PlayBall.cpp:56 | [0.3, 0.9] |
| | `minPenalty` | 0.1 | PlayBall.h:26 | [0.0, 0.3] |
| | `maxPenalty` | 0.8 | PlayBall.h:27 | [0.3, 1.0] |
| | `ballDistMinPenalty` | 300mm | PlayBall.h:28 | FIJO |
| | `ballDistMaxPenalty` | 200mm | PlayBall.h:29 | FIJO |
| | `minRating` | 0.01 | PlayBall.h:30 | FIJO |
| A quien pasar? | `minPassDistance` | 1500mm | PlayBall.cpp:70 | [500, 3000] |
| | `maxPassDistance` | 4500mm | PlayBall.cpp:71 | [2000, 8000] |
| | `sampleDistance` | 500mm | PlayBall.h:33 | [200, 1000] |
| | `sampleSteps` | (-1, 2) | PlayBall.h:34 | Variable |
| | `lookAhead` | true | PlayBall.h:31 | FIJO (no impl.) |
| | `lookAheadTime` | 2000ms | PlayBall.h:32 | FIJO (no impl.) |
| Calidad del pase? | rating L82-84 | PassEval×xG×oppRating | PlayBall.cpp:82-84 | RL puede reemplazar |
| Donde pasar? | `passAheadDistance` | 0mm | PassToTeammate:62 | [-500, 500] |
| | `minAngleOffset` | 7deg | PassToTeammate:63 | [2, 20] |
| | `maxAngleOffset` | 25deg | PassToTeammate:64 | [10, 45] |
| | `ratingThreshold` | 0.7 | PassToTeammate:61 | [0.3, 0.9] |
| Cuando disparar? | `minOpeningAngle` | 6deg | KickAtGoal:45 | [2, 15] |
| | `hysteresisNumber` | 300ms | KickAtGoal:42 | [100, 500] |
| | `hysteresisDecisionPenalty` | 500ms | KickAtGoal:47 | FIJO |

### 8.4 Puntos de Inyeccion de RL por Fase

**Fase 1:** RL NO toca SmashOrPass. Solo posiciones.

**Fase 2:** RL ajusta parametros:
1. `PlayBall.cpp:56` → `shootThreshold` = `rl_predict_threshold(state)`
2. `PlayBall.cpp:70-72` → `minPassDistance`, `maxPassDistance` = `rl_predict_distances(state)`
3. `PlayBall.cpp:82-84` → `sampleValue` (rating del pase = PassEval × xG × defenseRating) = `rl_predict_pass_quality(state, teammate)`
4. `PlayBall.h:26-27` → `minPenalty`, `maxPenalty` = `rl_predict_hysteresis(state)` (penalidades por cambio de decision)
5. `PlayBall.h:33-34` → `sampleDistance`, `sampleSteps` = `rl_predict_sampling(state)` (granularidad del muestreo de pase)
6. `KickAtGoal.cpp:45` → `minOpeningAngle` = `rl_predict_angle(state)`
7. `PassToTeammate.cpp:61-63` → `ratingThreshold`, `passAheadDistance`, `minAngleOffset` = `rl_predict_pass_params(state)`

**Fase 3:** RL reemplaza la decision completa:
```cpp
int rl_action = rl_network.predict_action(state);
switch(rl_action) {
    case 0: return shoot();
    case 1: return passTo(teammate_0);
    case 2: return passTo(teammate_1);
    case 3: return passTo(teammate_2);
    case 4: return dribble();
}
```

---

<a name="9-tabla-maestra"></a>
## 9. Tabla Maestra: Parametros Aprendibles vs Hard Constraints

### 9.1 RL PUEDE APRENDER (variables por fase)

```python
learnable_params = {
    # STRATEGY
    "ballXThreshold":        (100, 1000),      # mm
    "timeSinceBallAhead":    (100, 500),        # ms
    "numFieldPlayersGE":     (1, 3),            # int

    # TACTICS
    "striker_x":             (-500, 2500),      # mm (t121=703.5, t13=234.5)
    "striker_y":             (-1500, 1500),      # mm
    "defender_x":            (-2500, -200),      # mm (t121=-1468, t13=-1938)

    # ACTIONS (Fase 2+) — PlayBall.h decision params
    "shootThreshold":        (0.3, 1.0),        # float [0,1] — PlayBall.h:20
    "minPassDistance":       (500, 3000),         # mm — PlayBall.h:22
    "maxPassDistance":       (2000, 8000),        # mm — PlayBall.h:23
    "minPenalty":            (0.0, 0.3),         # float — PlayBall.h:26 (hysteresis baja)
    "maxPenalty":            (0.3, 1.0),         # float — PlayBall.h:27 (hysteresis alta)
    "sampleDistance":        (200, 1000),         # mm — PlayBall.h:33 (granularidad muestreo)
    "sampleSteps_min":       (-3, 0),            # int — PlayBall.h:34 (pasos atras)
    "sampleSteps_max":       (1, 5),             # int — PlayBall.h:34 (pasos adelante)

    # ACTIONS (Fase 2+) — KickAtGoal/PassToTeammate CFG params
    "minOpeningAngle":       (2, 15),           # deg — KickAtGoal.cpp:45
    "minKickLengthForLong":  (1000, 4000),      # mm — passToTeammateImpl.cfg
    "ratingThreshold":       (0.3, 0.9),        # float [0,1] — PassToTeammate.cpp:61
    "hysteresis":            (100, 500),         # ms — kickAtGoalImpl.cfg
    "minAngleOffset":        (2, 15),            # deg — PassToTeammate.cpp:63
    "maxAngleOffset":        (10, 45),           # deg — PassToTeammate.cpp:64 (renombrado en .cfg)
    "minTimeWaiting":        (1000, 5000),       # ms — passToTeammateImpl.cfg
    "maxTimeWaiting":        (5000, 15000),      # ms — passToTeammateImpl.cfg
}
```

> **Nota de rigor:** Los rangos RL y el diccionario `learnable_params` de esta seccion son una propuesta de disenio para el bridge RL. La fuente de verdad verificada aqui son los archivos `.cfg` y `.cpp/.h`; los rangos de entrenamiento no existen hoy como contrato implementado en el repo.

### 9.2 RL DEBE RESPETAR (hard constraints inmutables)

```python
hard_constraints = {
    # CAMPO
    "field_x":              [-2813, 2813],       # mm
    "field_y":              [-1875, 1875],        # mm
    "fieldBorderExpansion": 300,                  # mm (no salir)

    # FISICO
    "max_walk_speed":       250,                  # mm/s
    "max_sideways_speed":   200,                  # mm/s
    "max_backwards_speed":  150,                  # mm/s
    "max_rotation_speed":   120,                  # deg/s = 2.09 rad/s
    "min_robot_distance":   200,                  # mm (evitar colision)
    "ball_avoidance":       465,                  # mm (no pisar balon)

    # PATEO
    "kick_range_min":       400,                  # mm
    "kick_range_max":       5400,                 # mm
    "ball_velocity_max":    1800,                 # mm/s

    # JUEGO
    "game_state":           "PLAYING",            # RL solo actua aqui
    "robot_upright":        True,                 # No puede ser striker si caido
    "ball_seen_max_age":    5000,                 # ms (max sin ver balon)
    "decision_hysteresis":  300,                  # ms minimo entre cambios
    # lookAhead: flag en PlayBall.h (bool)(true) pero SIN if(p.lookAhead) implementado.
    # Es un TODO comentado en PlayBall.cpp:91. No es un constraint, es codigo pendiente.
    # No incluir en action space ni en hard constraints.

    # COGNITION THREAD
    "cognition_cycle":      33,                   # ms (30 FPS)
    "rl_max_inference":     3,                    # ms (no bloquear thread)
}
```

### 9.3 RL DEBE RASTREAR (observables)

```python
observables = {
    "robot_positions":      (4, 2),   # (x, y) por robot
    "robot_velocities":     (4, 2),   # (vx, vy)
    "robot_orientations":   (4, 1),   # theta
    "robot_is_upright":     (4, 1),   # bool
    "ball_position":        (1, 2),   # (x, y)
    "ball_velocity":        (1, 2),   # (vx, vy)
    "ball_age_ms":          (1, 1),   # tiempo sin ver
    "using_team_ball":      (1, 1),   # bool
    "opponents_positions":  (4, 2),   # (x, y) — LOCAL y FOV-dependiente
    "current_tactic":       (1, 2),   # one_hot [t121, t13]
    "game_is_playing":      (1, 1),   # bool
    "own_score":            (1, 1),   # int
    "opp_score":            (1, 1),   # int
    "half_time_remaining":  (1, 1),   # seconds
}
```

---

<a name="10-fases"></a>
## 10. Impacto por Fase del Proyecto RL

### 10.1 Fase 1 — RL Solo Posiciones

| CFG | Tipo | Parametro | Como se Usa |
|-----|------|-----------|-------------|
| `fieldDimensions.cfg` | **MANDATORIO** | bounds, simbolos | Limites del campo, resolucion de expresiones |
| `t121_4v4_scaled.cfg` | **MANDATORIO** | positions[] | Home poses defensivas (baseline) |
| `t13_4v4_scaled.cfg` | **MANDATORIO** | positions[] | Home poses ofensivas (baseline) |
| `s4v4_scaled.cfg` | **MANDATORIO** | transitions[] | FSM de tacticas (mantenida intacta) |
| `walkingEngine.cfg` | **MANDATORIO** | maxSpeed | Limites de movimiento en sim |
| `libWalkProvider.cfg` | **MANDATORIO** | fieldBorderExpansion, avoidance | Restricciones de evasion |
| `strategyBehaviorControl.cfg` | SOPORTE | strategy = s4v4_scaled | Confirma estrategia activa |
| `behaviorParameters.cfg` | SOPORTE | standRadius, walkRadius | Comportamiento general portero/striker |
| `skillBehaviorControl.cfg` | SOPORTE | options, playingOptions | Gatea la ejecucion real de partido via `PlaySoccer`/`HandleGameState` |

**RL reemplaza:** `positions[].pose` de las tacticas
**RL mantiene intacto:** FSM, assignRoles, SmashOrPass, todos los skills

### 10.2 Fase 2 — RL Posiciones + Decisiones

| CFG (adicional a Fase 1) | Tipo | Parametro | Como se Usa |
|--------------------------|------|-----------|-------------|
| `kickAtGoalImpl.cfg` | **MANDATORIO** | minOpeningAngle, hysteresisNumber | RL ajusta thresholds de disparo |
| `passToTeammateImpl.cfg` | **MANDATORIO** | ratingThreshold, distances, angles | RL ajusta thresholds de pase |
| `PlayBall.h` (C++) | **MANDATORIO** | shootThreshold, minPenalty, maxPenalty, sampleDistance, sampleSteps | Defaults de decision — RL inyecta valores alternativos. `lookAhead`/`lookAheadTime` siguen siendo flags muertos (TODO sin implementar) — NO incluir en action space. |
| `kickInfo.cfg` | SOPORTE | ranges, velocities | Validar que pateos RL sean fisicamente posibles |

**RL reemplaza (adicionalmente):** assignRoles, determineActiveAgent (quien es striker)
**RL ajusta:** shootThreshold, minPassDistance, maxPassDistance, minPenalty, maxPenalty, sampleDistance, sampleSteps, minOpeningAngle, ratingThreshold, passAheadDistance, minAngleOffset
**Puntos de inyeccion C++ (7 totales):**
1. `PlayBall.cpp:56` — shootThreshold
2. `PlayBall.cpp:70-72` — minPassDistance, maxPassDistance
3. `PlayBall.cpp:82-84` — rating del pase (PassEval × xG × oppRating)
4. `PlayBall.h:26-27` — minPenalty, maxPenalty (hysteresis)
5. `PlayBall.h:33-34` — sampleDistance, sampleSteps (granularidad)
6. `KickAtGoal.cpp:45` — minOpeningAngle
7. `PassToTeammate.cpp:61-63` — ratingThreshold, passAheadDistance, minAngleOffset

### 10.3 Fase 3 — RL Suplanta Todo

| CFG (adicional a Fases 1+2) | Tipo | Parametro | Como se Usa |
|-----------------------------|------|-----------|-------------|
| `zweikampfImpl.cfg` | **MANDATORIO** | duelRatings, duelTimings, stealBallParameters, allowedKicks, fieldBorderSafeDistance | Modelo completo de duelo — RL lo replica en Fase 3. `noKickMinTime=200ms` y `fieldBorderSafeDistance=625mm` son los constraints reales de approach y timing (NO demoGoToBallAndKick). |
| `kickInfo.cfg` | **MANDATORIO** | ranges, velocities, executionTime | Validar acciones de pateo fisicamente |

**RL reemplaza:** FSM, posiciones, roles, SmashOrPass, seleccion de skill
**RL mantiene:** SkillBehaviorControl (gateo y ejecucion motora de `SkillRequest`), PathPlanner, percepciones

> **NOTA CRITICA:** En todas las fases, los CFGs de 4v4_Scaled son la UNICA fuente valida. NO mezclar con tacticas/enums heredados de otros formatos (Default 5v5, 3v3_Scaled, etc.). Las tacticas validas son EXCLUSIVAMENTE `t121_4v4_scaled` y `t13_4v4_scaled`; los set plays validos son exclusivamente los sufijados `_scaled`. Si el RL o el parser referencian tacticas sin sufijo `_4v4_scaled`, es un ERROR de contaminacion de formato.

---

<a name="11-diagnostico"></a>
## 11. Diagnostico: Estado Actual del Proyecto vs CFGs Reales

### 11.1 Brechas Criticas Identificadas

> **NOTA:** No existe ningun parser Python funcional en el repositorio actualmente. El BhumanParser debe construirse desde cero siguiendo la guia en `BHUMANPARSER_GUIDE.md`.

| ID | Severidad | Descripcion | Archivo Afectado |
|----|-----------|-------------|------------------|
| C1 | **CRITICO** | Campo entrenamiento usa 5v5 (4500/3000) en lugar de 4v4 (2813/1875) | config.py L23-26 |
| C2 | **CRITICO** | Action space derivado de campo 5v5 → 37% del rango es invalido | config.py L234-235 |
| A1 | ALTO | No existe parser Python integrado — environment.py no importa ningun parser | environment.py |
| A2 | ALTO | `_current_tactic` hardcodeado a "t4" — sin FSM, sin transiciones | environment.py L827 |
| A3 | ALTO | `TACTIC_NAMES` no contiene `t121_4v4_scaled` ni `t13_4v4_scaled` | config.py L251 |
| A4 | ALTO | Home poses hardcodeadas sin trazabilidad al .cfg real | environment.py L1178-1184 |
| A5 | ALTO | Roles nunca cambian — bloqueante para Fase 2 | environment.py L1130-1150 |
| M1 | MEDIO | FIELD_SYMBOLS deben venir del parser con valores 4v4 Scaled | config.py |
| M2 | MEDIO | TACTIC_NAMES duplicado — debe venir de una sola fuente (parser) | config.py |

### 11.2 Archivos .cfg Requeridos

Para cerrar TODAS las brechas, el BhumanParser necesita acceso a 13 archivos .cfg (11 operativos + 2 de control de behavior):

1. `Config/Locations/4v4_Scaled/fieldDimensions.cfg` → FIELD_SYMBOLS correctos
2. `Config/Behavior/Tactics/t121_4v4_scaled.cfg` → Home poses tactica defensiva
3. `Config/Behavior/Tactics/t13_4v4_scaled.cfg` → Home poses tactica ofensiva
4. `Config/Behavior/Strategies/s4v4_scaled.cfg` → FSM de transiciones
5. `Config/Scenarios/4v4_Scaled/BehaviorControl/kickAtGoalImpl.cfg` → Parametros disparo
6. `Config/Scenarios/4v4_Scaled/BehaviorControl/passToTeammateImpl.cfg` → Parametros pase
7. `Config/Scenarios/4v4_Scaled/BehaviorControl/zweikampfImpl.cfg` → Parametros duelo completo (6 subsistemas)
8. `Config/Scenarios/4v4_Scaled/kickInfo.cfg` → Propiedades de pateos
9. `Config/Scenarios/4v4_Scaled/libWalkProvider.cfg` → Evasion de obstaculos
10. `Config/Scenarios/4v4_Scaled/behaviorParameters.cfg` → Config global portero/striker
11. `Config/Robots/Default/walkingEngine.cfg` → Velocidades maximas (configuratedParameters + walkSpeedParams)
12. `Config/Scenarios/4v4_Scaled/strategyBehaviorControl.cfg` → Confirmar strategy activa
13. `Config/Scenarios/4v4_Scaled/skillBehaviorControl.cfg` → Gateo de opciones y playingOptions reales

> **ELIMINADO:** `demoGoToBallAndKickImpl.cfg` — no se invoca en partidos 4v4. Solo activo en modo exhibicion (LibDemo::soccer / AnyPlaceDemo), que no forma parte de `skillBehaviorControl.cfg` del escenario 4v4_Scaled. No debe formar parte del parser ni del AllConfigs.

**Archivos C++/headers criticos a auditar junto con esos CFGs:**
- `Src/Modules/BehaviorControl/StrategyBehaviorControl/StrategyBehaviorControl.cpp`
- `Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp`
- `Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.h`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.cpp`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/Options/PlaySoccer.h`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/Options/HandleGameState.h`
- `Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/PlayBall.cpp`

### 11.3 Matriz de Prioridad de Correccion

```
ORDEN DE CORRECCION:
1. [C1][C2] Campo → config.py usa valores de parser con FIELD_SYMBOLS 4v4
2. [A1] Construir BhumanParser e integrar en environment.py
3. [M1] FIELD_SYMBOLS desde parser (no hardcoded)
4. [A3][M2] Unificar TACTIC_NAMES desde parser (una sola fuente)
5. [A4] Home poses desde parser (no hardcoded)
6. [A2] FSM activa con transiciones reales desde s4v4_scaled.cfg
7. [A5] Roles dinamicos (preparar Fase 2)
```

---

## 12. Tabla de Simbolos del Campo 4v4 Scaled

Fuente: `Config/Locations/4v4_Scaled/fieldDimensions.cfg`

```python
FIELD_SYMBOLS_4V4 = {
    # Eje X (largo: 5626mm)
    "xPosOpponentFieldBorder":  3313,
    "xPosOpponentGoal":         3368,
    "xPosOpponentGoalPost":     2838,
    "xPosOpponentGroundLine":   2813,
    "xPosOpponentGoalArea":     2438,
    "xPosOpponentPenaltyMark":  2000,
    "xPosOpponentPenaltyArea":  1781,
    "xPosHalfWayLine":          0,
    "xPosOwnPenaltyArea":      -1781,
    "xPosOwnPenaltyMark":      -2000,
    "xPosOwnGoalArea":         -2438,
    "xPosOwnGroundLine":       -2813,
    "xPosOwnGoalPost":         -2838,
    "xPosOwnGoal":             -3368,
    "xPosOwnFieldBorder":      -3313,

    # Eje Y (ancho: 3750mm)
    "yPosLeftFieldBorder":      1960,
    "yPosLeftSideline":         1875,
    "yPosLeftPenaltyArea":      1250,
    "yPosLeftGoalArea":          688,
    "yPosLeftGoal":              800,
    "yPosRightGoal":            -800,
    "yPosRightGoalArea":        -688,
    "yPosRightPenaltyArea":    -1250,
    "yPosRightSideline":       -1875,
    "yPosRightFieldBorder":    -1960,

    # Otros
    "centerCircleRadius":        469,
    "fieldLinesWidth":            50,
    "goalPostRadius":             50,
    "penaltyMarkSize":           100,
}
```
> ESTADO DOCUMENTAL: `referencia`
>
> La copia preservada y citada de este documento vive en:
> [Documentacion/REFERENCIA/CONTEXTO BHUMAN/CFG_UNIFIED_ANALYSIS.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/CONTEXTO%20BHUMAN/CFG_UNIFIED_ANALYSIS.md)
> No usar este archivo como baseline por si solo.
