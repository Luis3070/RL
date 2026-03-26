# 🎯 PUNTOS DE DECISIÓN EN C++ - Dónde se Decide SHOOT/PASS

## Resumen Ejecutivo

```
STRIKER CON BALÓN (en campo):

┌─────────────────────────────────────────────────────────┐
│ DECIDE: ¿DISPARAR O PASAR?                              │
│ Archivo: PlayBall.cpp, líneas 33-118                    │
│ Función: smashOrPass()                                  │
│ Decisión: BASADA EN EXPECTED GOALS (probabilidad de gol) │
└─────────────────────────────────────────────────────────┘
         ↓
    ¿maxValue > shootThreshold?  (línea 56)
         ↓
    SÍ ─→ DISPARAR (línea 59)
    NO ↓
┌─────────────────────────────────────────────────────────┐
│ DECIDE: ¿A QUIÉN PASAR?                                 │
│ Archivo: PlayBall.cpp, líneas 62-104                    │
│ Función: smashOrPass(), bucle de compañeros             │
│ Decisión: BASADA EN RATING DE PASE + xG del compañero   │
└─────────────────────────────────────────────────────────┘
         ↓
    ¿passTarget existe?  (línea 105)
         ↓
    SÍ ─→ PASAR A passTarget (línea 108)
    NO ─→ DISPARAR (línea 117)
```

---

## 1. DECISIÓN: ¿CUÁNDO DISPARAR? (SHOOT vs PASS)

### Ubicación Precisa: `PlayBall.cpp`, líneas 33-60

```cpp
SkillRequest PlayBall::smashOrPass(const Agent& self, const Agents& teammates)
{
  if(p.alwaysShoot)
    return SkillRequest::Builder::shoot();

  // LÍNEA 39: Calcular penalidad por cambio (HYSTERESIS)
  const float decisionPenalty = mapToRange(
    self.ballPosition.norm(),
    p.ballDistMaxPenalty,      // ← CFG: 200mm
    p.ballDistMinPenalty,      // ← CFG: 300mm
    p.maxPenalty,              // ← CFG: 0.8
    p.minPenalty               // ← CFG: 0.1
  );

  // LÍNEA 43-44: CALCULAR PROBABILIDAD DE GÓL (EXPECTED GOALS)
  const Vector2f ballPosition = self.pose * self.ballPosition;
  float maxValue = theExpectedGoals.getRating(ballPosition);
  // ↑ maxValue = probabilidad de marcar si disparo AHORA
  //   Rango: [0.0, 1.0]

  // LÍNEA 45-50: Ajustar distancia de pase según situación
  float maxPassDistance = p.maxPassDistance;  // ← CFG: 4500mm
  if(theGameState.isFreeKick() && theGameState.isForOwnTeam())
  {
    maxValue = p.minRating;  // ← CFG: 0.01 (casi prohibir disparo en free kicks)
    maxPassDistance = theGameState.isGoalKick() ?
                      p.maxGoalKickDistance :     // ← CFG: 9000mm
                      p.maxFreeKickDistance;     // ← CFG: 6000mm
  }

  // LÍNEA 54-55: APLICAR HYSTERESIS (penalidad si cambio de decisión)
  if(lastPassTarget > 0)  // Si estaba pasando antes
    maxValue -= decisionPenalty;  // Penalizar cambio a disparo

  // ════════════════════════════════════════════════════════
  // LÍNEA 56: DECISIÓN CRÍTICA ✅
  // ════════════════════════════════════════════════════════
  if(maxValue > p.shootThreshold)  // ← CFG: 0.8
  {
    lastPassTarget = -1;
    return SkillRequest::Builder::shoot();  // ← DISPARA
  }
  // ════════════════════════════════════════════════════════
```

**En palabras simples:**
- **Línea 44**: Calcular xG (probabilidad % de gol) en posición actual
- **Línea 56**: Si xG > 80% (shootThreshold del CFG) → DISPARAR
- **Si xG < 80%**: Buscar pases

---

## 2. DECISIÓN: ¿A QUIÉN PASAR? (PASS TARGET)

### Ubicación Precisa: `PlayBall.cpp`, líneas 62-104

```cpp
SkillRequest PlayBall::smashOrPass(const Agent& self, const Agents& teammates)
{
  // ... (código anterior)

  // ════════════════════════════════════════════════════════
  // LÍNEA 62-104: BUSCAR MEJOR COMPAÑERO PARA PASAR
  // ════════════════════════════════════════════════════════

  const Agent* passTarget = nullptr;  // Variable donde guardamos al mejor compañero

  // LÍNEA 64: ITERAR TODOS LOS COMPAÑEROS
  for(const Agent* agent : teammates)
  {
    // LÍNEA 66-67: DESCARTAR PORTERO
    if(agent->position == Tactic::Position::Type::goalkeeper)
      continue;  // No pasar al portero

    Vector2f agentPosition = agent->currentPosition;
    const float passDistance = (agentPosition - ballPosition).norm();

    // LÍNEA 70-72: VALIDAR DISTANCIA DE PASE
    if(passDistance < p.minPassDistance ||     // ← CFG: 1500mm (mínimo)
       passDistance > maxPassDistance)          // ← CFG: 4500mm (máximo)
      continue;  // Compañero muy cerca o muy lejos → descartar

    // LÍNEA 74-90: EVALUAR POSICIONES DE PASE
    Vector2f bestAgentPosition = agentPosition;
    float agentValue = 0.f;

    // LÍNEA 77: Crear línea desde compañero hacia arco contrario
    const Geometry::Line agentGoalLine(
      agentPosition,
      (goalPosition - agentPosition).normalized(p.sampleDistance)
    );

    // LÍNEA 78-90: SAMPLEAR POSICIONES EN ESA LÍNEA
    for(int t = p.sampleSteps.min; t <= p.sampleSteps.max; t++)
    {
      const Vector2f samplePosition = agentGoalLine.base + t * agentGoalLine.direction;

      // ════════════════════════════════════════════════════════
      // LÍNEA 82-84: CALCULAR RATING DEL PASE
      // ════════════════════════════════════════════════════════
      const float sampleValue =
        thePassEvaluation.getRating(samplePosition) *      // ¿Puede llegar el balón?
        theExpectedGoals.getRating(samplePosition) *       // ¿Pode marcar compañero?
        theExpectedGoals.getOpponentRating(samplePosition);// ¿Defensa está fuerte?

      if(sampleValue > agentValue)
      {
        agentValue = sampleValue;  // Guardar mejor valor
        bestAgentPosition = samplePosition;  // Guardar mejor posición
      }
    }

    // LÍNEA 95-98: APLICAR HYSTERESIS (penalidad si cambio target)
    const bool applyDecisionPenalty = lastPassTarget && lastPassTarget != agent->number;
    if(applyDecisionPenalty)
      agentValue -= decisionPenalty;

    // LÍNEA 99-103: GUARDAR SI ES EL MEJOR PASE HASTA AHORA
    if(agentValue > maxValue)
    {
      maxValue = agentValue;  // Este pase es mejor que los anteriores
      passTarget = agent;     // ← GUARDAR ESTE COMPAÑERO COMO TARGET
    }
  }

  // ════════════════════════════════════════════════════════
  // LÍNEA 105-109: DECISIÓN FINAL
  // ════════════════════════════════════════════════════════
  if(passTarget)
  {
    lastPassTarget = passTarget->number;
    return SkillRequest::Builder::passTo(passTarget->number);  // ← PASAR A passTarget
  }
  else if(theGameState.isFreeKick() && theGameState.isForOwnTeam())
  {
    lastPassTarget = -1;
    return SkillRequest::Builder::passTo(0);  // En free kick, esperar
  }

  lastPassTarget = maxValue > p.minRating ? -1 : 0;
  return SkillRequest::Builder::shoot();  // ← DISPARA SI NO ENCUENTRA PASE
}
```

**En palabras simples:**

| Paso | Línea | Qué Hace | Parámetro CFG |
|------|-------|----------|---------------|
| 1 | 64 | Iterar cada compañero | N/A |
| 2 | 70-72 | Filtrar por distancia | `minPassDistance`, `maxPassDistance` |
| 3 | 77-90 | Samplear posiciones hacia arco | `sampleDistance`, `sampleSteps` |
| 4 | 82-84 | Calcular rating: **PassEval × xG × defenseRating** | N/A (modules) |
| 5 | 99-103 | Guardar el mejor rating | N/A |
| 6 | 105 | Si existe target → PASAR | N/A |
| 7 | 117 | Si no → DISPARAR | N/A |

---

## 3. DECISIÓN: ¿DÓNDE PASAR? (PASS POSITION)

### Ubicación: `PassToTeammate.cpp` (línea 121+) y `KickSelection`

**La posición donde pasa se determina en DOS partes:**

#### Parte A: Target Inicial (línea 82-84, PlayBall.cpp)
```cpp
// El target es: la posición del compañero que está libre
// Luego se ajusta por:
float passAheadDistance = p.passAheadDistance;  // ← CFG: 0mm (no adelantar)

// Esto se calcula en PassToTeammate.cpp (línea 121+)
setTeammateFromParameters(p);
```

#### Parte B: Ajuste de Pase (PassToTeammate.cpp)
```cpp
// PassToTeammate.cpp, línea ~250-300 (NO VISTO EN LECTURA)
// Calcula:
// - kickTarget (dónde patear hacia)
// - kickAngle (ángulo del pase)
// - kickLength (distancia)
//
// Basado en:
minAngleOffset = 7deg;      // ← CFG: desviación mínima
maxAngleOffset = 25deg;     // ← CFG: desviación máxima
ignoreObstaclesThreshold = 250;  // ← CFG: ignorar obstáculos si < 250mm
```

---

## 4. DECISIÓN: ¿QUÉ TIPO DE PATEO? (KICK TYPE)

### Ubicación: Determinado por `KickSelection` (parte de libDemoProvider)

**El tipo de pateo depende de:**

```cpp
// Basado en PassToTeammate.cpp línea 58:
allowedKicks = [
  forwardFastRightLong,    // ← Pase largo derecha (1800 mm/s, 5400mm)
  forwardFastLeftLong,     // ← Pase largo izquierda
  forwardFastRight, forwardFastLeft,
  forwardFastRightPass, forwardFastLeftPass,
  walkForwardsRightLong, walkForwardsLeftLong,
  walkForwardsRightAlternative, walkForwardsLeftAlternative
];

// LÓGICA: Si distancia > minKickLengthForLong (2813mm):
//   → Usar pase LARGO (forwardFastRightLong)
// Si distancia < minKickLengthForLong:
//   → Usar pase CORTO (walkForwardsRight)
```

**Selección en kickInfo.cfg:**

| Tipo Pateo | Rango (mm) | Velocidad (mm/s) | Duración (ms) | Cuándo Usar |
|------------|-----------|-----------------|---------------|------------|
| `walkForwardsRight` | 400-1000 | 489-774 | 500 | Pase CORTO |
| `forwardFastRight` | 1600-4500 | 979-1643 | 1500 | Pase MEDIO |
| `forwardFastRightLong` | 5400 | 1800 | 1500 | Pase LARGO |

---

## 5. DECISIÓN: ¿CUÁNDO DISPARAR? (SHOOT TIMING)

### Ubicación: `KickAtGoal.cpp` (línea 64-83) + `PlayBall.cpp` (línea 56)

**Cuando el striker decide DISPARAR (en línea 59 de PlayBall.cpp):**

```cpp
// PlayBall.cpp, línea 59:
return SkillRequest::Builder::shoot();  // ← Envía comando "shoot"

// Esto dispara el skill KickAtGoal (KickAtGoal.cpp, línea 64+):
void execute(const KickAtGoal&)
{
  // LÍNEA 66-76: DECIDIR QUÉ TIPO DE DISPARO
  updateKickDirectionAndKick();  // ← Calcula qué kick usar (ver kickInfo.cfg)

  const bool allowDirectKick = !(theGameState.isFreeKick() &&
                                 theGameState.isForOwnTeam());

  if(aimingAtGoal && allowDirectKick)
  {
    // LÍNEA 72-74: DISPARAR DIRECTO AL ARCO
    theGoToBallAndKickSkill({
      .targetDirection = kickDirectionRelative,
      .kickType = kickType,  // ← Tipo de pateo (forwardFastLeft, etc.)
      .alignPrecisely = ...
    });
  }
  else if(theGameState.isKickOff())
  {
    theDirectKickOffSkill();  // En saque inicial
  }
  else
  {
    theDribbleToGoalSkill();   // Si no puede disparar directo, driblear
  }
}
```

**Timing de disparo (línea 45 en kickAtGoalImpl.cfg):**

```cfg
minOpeningAngle = 6deg;  # ← SOLO dispara si hay 6° libres al arco
```

---

## 6. RESUMEN: FLUJO COMPLETO DE DECISIÓN

```
FRAME N: STRIKER TIENE BALÓN
│
├─ PlayBall::smashOrPass() EVALÚA:
│  │
│  ├─ Línea 44: xG_shoot = expectedGoals.getRating(ballPos)
│  │  "¿Qué probabilidad de gol si disparo AHORA?"
│  │  Rango: [0.0, 1.0]
│  │
│  ├─ Línea 56: ¿xG_shoot > shootThreshold (0.8)?
│  │  │
│  │  ├─ SÍ (xG_shoot > 0.8)
│  │  │  └─ Línea 59: return shoot()
│  │  │     └─ DECIDE: DISPARAR
│  │  │
│  │  └─ NO (xG_shoot ≤ 0.8)
│  │     └─ Línea 64-104: Buscar mejor pase
│  │        │
│  │        ├─ Para cada compañero:
│  │        │  ├─ Línea 70-72: ¿Distancia válida? (1500-4500mm)
│  │        │  ├─ Línea 77-90: Samplear posiciones hacia arco
│  │        │  ├─ Línea 82-84: Calcular rating del pase
│  │        │  │   = PassEval × xG_teammate × defenseRating
│  │        │  └─ Línea 99-103: Guardar mejor
│  │        │
│  │        ├─ Línea 105: ¿Existe passTarget?
│  │        │  │
│  │        │  ├─ SÍ
│  │        │  │  └─ Línea 108: return passTo(passTarget)
│  │        │  │     └─ DECIDE: PASAR
│  │        │  │
│  │        │  └─ NO
│  │        │     └─ Línea 117: return shoot()
│  │        │        └─ DECIDE: DISPARAR (esperanza)
│  │
│  └─ Resultado: SkillRequest = {shoot, passTo(ID), ...}
│
├─ KickAtGoal (si shoot)
│  └─ Línea 66: updateKickDirectionAndKick()
│     └─ DECIDE: qué tipo de pateo (largo/corto/potente)
│
└─ PassToTeammate (si pass)
   └─ DECIDE: dónde exactamente pasar
```

---

## 7. PARÁMETROS CRÍTICOS POR DECISIÓN

| Decisión | Parámetro CFG | Valor Actual | Línea C++ | Rango RL |
|----------|---------------|--------------|-----------|----------|
| **¿Disparar o Pasar?** | `shootThreshold` | 0.8 | PlayBall.cpp:56 | [0.3, 0.9] |
| | `passImprovement` | 0.15 | PlayBall.h:21 | [0.0, 0.5] |
| | `minPenalty` | 0.1 | PlayBall.h:30 | [0.0, 0.3] |
| | `maxPenalty` | 0.8 | PlayBall.h:27 | [0.3, 1.0] |
| **¿A quién pasar?** | `minPassDistance` | 1500 | PlayBall.cpp:70 | [500, 3000] |
| | `maxPassDistance` | 4500 | PlayBall.cpp:71 | [2000, 8000] |
| | `sampleDistance` | 500 | PlayBall.h:33 | [200, 1000] |
| | `sampleSteps.min/max` | [-1, 2] | PlayBall.h:34 | Variable |
| **¿Dónde pasar?** | `passAheadDistance` | 0 | PassToTeammate.cpp:62 | [-500, 500] |
| | `minAngleOffset` | 7° | PassToTeammate.cpp:63 | [2°, 20°] |
| | `maxAngleOffset` | 25° | PassToTeammate.cpp:64 | [10°, 45°] |
| **¿Cuándo disparar?** | `minOpeningAngle` | 6° | KickAtGoal.cpp:45 | [2°, 15°] |
| | `hysteresisNumber` | 300 | KickAtGoal.cpp:42 | [100, 500] |

---

## 8. PARA FASE 2 (RL PUEDE INFLUIR)

**Dónde RL puede "inyectar" su decisión:**

```cpp
// OPCIÓN 1: Reemplazar shootThreshold
// En lugar de: if(maxValue > 0.8)
// RL predice: if(maxValue > RL_predicted_threshold)

// OPCIÓN 2: Reemplazar rating del pase
// En lugar de: sampleValue = PassEval × xG × defenseRating
// RL predice: sampleValue = RL_predicted_pass_quality

// OPCIÓN 3: Reemplazar cálculo de distancia de pase
// En lugar de: if(passDistance < 1500 || > 4500)
// RL predice: if(passDistance < RL_min || > RL_max)

// OPCIÓN 4: Reemplazar penalidades de hysteresis
// En lugar de: decisionPenalty = mapToRange(...)
// RL predice: decisionPenalty = RL_predicted_penalty
```

**Punto de inyección en código:**

```cpp
// PlayBall.cpp, línea 56 (CRÍTICO):
// ORIGINAL:
if(maxValue > p.shootThreshold)  // ← CFG
  return SkillRequest::Builder::shoot();

// MODIFICADO para Fase 2 (con RL):
// OPCIÓN A: RL inyecta threshold
float rl_shoot_threshold = rl_network.predict_threshold(state);
if(maxValue > rl_shoot_threshold)  // ← RL
  return SkillRequest::Builder::shoot();

// OPCIÓN B: RL inyecta decisión directa
int rl_action = rl_network.predict_action(state);  // 0=shoot, 1=pass_to_0, ...
switch(rl_action) {
  case 0: return SkillRequest::Builder::shoot();
  case 1: return SkillRequest::Builder::passTo(0);
  ...
}
```

---

## 9. RESUMEN EJECUTIVO PARA RL

### Fase 1: RL Solo Posicionamiento
- RL: Predice posiciones de robots (x, y)
- C++: Toma TODAS las decisiones de shoot/pass
- CFG: Define parámetros fijos

### Fase 2: RL + Decisiones Básicas
- RL: Puede ajustar thresholds (shootThreshold, distancias, etc.)
- C++: Sigue lógica SmashOrPass pero con parámetros de RL
- CFG: Solo valores por defecto

**Puntos de inyección de RL en Fase 2:**

1. **PlayBall.cpp, línea 56**: `shootThreshold` → `rl_predict_threshold()`
2. **PlayBall.cpp, línea 70-72**: `minPassDistance`, `maxPassDistance` → `rl_predict_distances()`
3. **PlayBall.cpp, línea 82-84**: `sampleValue` rating → `rl_predict_pass_quality()`
4. **KickAtGoal.cpp, línea 45**: `minOpeningAngle` → `rl_predict_angle()`

---

