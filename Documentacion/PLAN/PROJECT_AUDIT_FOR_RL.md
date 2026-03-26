# 🔍 AUDITORÍA COMPLETA DEL PROYECTO PARA RL

## Executive Summary

Este documento identifica **TODAS** las restricciones, dependencias y consideraciones que afectan la implementación de RL MAPPO en el proyecto SabanaHerons2026.

---

## 1. ARQUITECTURA DE THREADS (CRÍTICO PARA TIMING)

### Thread Tree (threads.cfg, líneas 1-315)

```
Robot.start()
│
├─ Upper (Priority 0) ─────────────────────────────────────┐
│  Percepción cámara superior (HEAD camera)               │
│  - Detección de balón                                   │
│  - Detección de líneas                                  │
│  - Detección de postes                                  │
│  - Ejecución Unit: Perception                           │
│  Frame Rate: ~30 FPS (camera@30Hz)                      │
│                                                          │
├─ Lower (Priority 0) ────────────────────────────────────┐
│  Percepción cámara inferior (BODY camera)               │
│  - Detección alternativa de balón (BOP)                 │
│  - Detección de campo                                   │
│  - Ejecución Unit: Perception                           │
│  Frame Rate: ~30 FPS (camera@30Hz)                      │
│                                                          │
├─ COGNITION (Priority 1) ────────────────────────────────┤
│  🎯 DONDE ENTRA RL                                       │
│  - Decisiones de juego (PlayBall::smashOrPass)          │
│  - Localización del robot (SelfLocator)                 │
│  - Modelo del mundo (ObstacleModel, GlobalTeammatesModel)
│  - Evaluación de posiciones (FieldRating)               │
│  - Evaluación de pases (PassEvaluation)                 │
│  - Expected Goals (ExpectedGoalsProvider)               │
│  - Control de estrategia (StrategyBehaviorControl)      │
│  - Control de skills (SkillBehaviorControl)             │
│  - Ejecución Unit: Cognition                            │
│  Frame Rate: ~30 FPS (sincro con camera)               │
│  🔴 CRITICAL: Aquí es donde RL reemplaza PlayBall      │
│                                                          │
├─ Motion (Priority 20) ──────────────────────────────────┐
│  Ejecución de movimiento                                │
│  - Walking engine                                       │
│  - Kick engines (kick, walkKick, dribble)              │
│  - Head control                                         │
│  - Arm control                                          │
│  - Ejecución Unit: Motion                               │
│  Frame Rate: ~200 FPS (low-level control)              │
│  Sync: Recibe MotionRequest de Cognition               │
│                                                          │
├─ Audio (Priority 0) ────────────────────────────────────┐
│  Detección de silbatos (Whistle)                        │
│  Frame Rate: ~22 kHz (audio stream)                     │
│                                                          │
└─ Referee (Priority 0) ──────────────────────────────────┘
   Detección de gestos del árbitro
   Frame Rate: Variable (según timestamps)
```

### Timing Crítico

| Thread | Frame Rate | Cycle Time | Sincronización |
|--------|-----------|-----------|----------------|
| **Upper** | ~30 FPS | ~33 ms | Camera trigger |
| **Lower** | ~30 FPS | ~33 ms | Camera trigger |
| **Cognition** | ~30 FPS | ~33 ms | Camera @33ms ⚠️ |
| **Motion** | ~200 FPS | ~5 ms | Cognition output |

**🔴 CRÍTICO:** Cognition thread corre cada ~33ms (30 FPS). RL debe ejecutarse en < 33ms para no retrasar decisiones.

---

## 2. RESTRICCIONES DE HARDWARE (NAO Robot)

### Especificaciones Físicas

```
NAO H25 V5 (asumido):

Dimensiones:
  - Altura: ~580 mm
  - Peso: ~4.3 kg
  - Tamaño pie: ~130mm × 50mm

Sensores:
  - 2 cámaras (30 FPS cada una)
  - 1 IMU (inertial measurement unit)
  - 2 contactos en pies (foot bumper)
  - FSR (force sensitive resistors en pies)
  - Giróscopo + acelerómetro

Actuadores:
  - 25 motores (servos)
  - 1 motor/rueda en cada pie (walk control)
  - Brazos, cabeza, etc.

Batería:
  - ~48.6 Wh (4 células Li-ion)
  - Duración: ~45-60 minutos en juego
  - Consumo crítico: Walking engine consume ~50W
```

### Restricciones Críticas

| Restricción | Valor | Impacto en RL |
|------------|-------|--------------|
| **Velocidad máxima caminata** | ~250 mm/s | RL no puede esperar velocidades > 250mm/s |
| **Velocidad de giro** | ~180°/s | RL debe limitar cambios de dirección |
| **Aceleración máxima** | ~1-2 m/s² | RL debe generar acciones smooth |
| **Tiempo reacción** | ~100-200 ms | Latencia mínima request→ejecución |
| **Precisión de posición** | ±100-200 mm | RL debe respetar error de ±100-200mm |
| **Rango de visión** | ~60° horizontal | RL tiene visión limitada (no 360°) |
| **Energía batería** | ~45 min | RL no puede hacer acciones energéticamente costosas continuas |
| **Tiempo ciclo Cognition** | ~33 ms | RL debe ejecutarse en < 33 ms |

---

## 3. RESTRICCIONES DE TIEMPO REAL

### Cycle Time de Cognition Thread

```cpp
// threads.cfg línea 113-208
{
  name = Cognition;
  priority = 1;           // Priority > Motion (0) y Perception (0)
  executionUnit = Cognition;
  // 50+ representation providers
  // Cada uno se ejecuta secuencialmente
}

// Orden de ejecución (aproximado):
┌─ Cognition Cycle (33 ms) ────────────────────────┐
│                                                   │
│  1. FrameInfo (timestamp actual)         ~0.1 ms │
│  2. BallPercept (de Upper/Lower cameras) ~1 ms   │
│  3. RobotPose (SelfLocator)              ~5 ms   │
│  4. ObstacleModel                        ~2 ms   │
│  5. GlobalTeammatesModel                 ~1 ms   │
│  6. GlobalOpponentsModel                 ~1 ms   │
│  7. FieldBall (fusion de percepciones)   ~1 ms   │
│  8. ExpectedGoals                        ~1 ms   │
│  9. PassEvaluation                       ~1 ms   │
│ 10. StrategyBehaviorControl              ~2 ms   │  ← Aquí va RL (Fase 2)
│ 11. SkillBehaviorControl                 ~2 ms   │
│ 12. Motion request ready                 ~1 ms   │
│                                                   │
│ Total: ~20-25 ms (dejando ~5-10 ms buffer)      │
└───────────────────────────────────────────────────┘

⚠️ RL Predicción debe tomar < 2-3 ms (para no bloquear thread)
```

### Línea de Tiempo de Juego

```
FRAME:  1        2        3        4        5        6
TIME:   0ms    33ms     66ms     99ms    132ms    165ms

Upper:  📷      📷      📷      📷      📷      📷
Lower:  📷      📷      📷      📷      📷      📷
        │       │       │       │       │       │
        └──────────────┬───────────────┘
                    COGNITION (33ms de latencia desde lectura a decisión)
                    │
                    ├─ PlayBall::smashOrPass() ← RL entra aquí
                    ├─ ExpectedGoals calculation
                    ├─ Pass evaluation
                    └─ SkillBehaviorControl
                    │
                    └──────────────┬───────────────┘
                                MOTION (5ms)
                                │
                                ├─ WalkingEngine
                                ├─ KickEngine
                                └─ HeadControl
                                │
                                ▼ ROBOT EXECUTES
                    Robot starts moving ~70-80ms después de lectura
```

---

## 4. RESTRICCIONES DE COMUNICACIÓN

### Team Messages (TeamMessageHandler)

```cpp
// threads.cfg línea 193, 199, 206
{representation = ReceivedTeamMessages; provider = TeamMessageHandler;},
{representation = SentTeamMessage; provider = TeamMessageHandler;},
{representation = TeamData; provider = TeamDataProvider;},

// Estructura de mensaje (SPL protocol):
TeamMessage {
  // Cabecera
  magicNumber = 49;              // Identificar paquete válido
  teamNumber = 5;                // Sabana Herons
  robotNumber = 1-10;            // ID del robot

  // Información que se envía
  robotPose: (x, y, rotation);   // Posición del robot
  ballModel: (x, y, vx, vy);     // Dónde cree que está el balón
  playerState: (role, etc);      // Estado actual del jugador
  timeOfLastSharedBallPercept;   // Timestamp del balón

  // Frecuencia
  Send rate: ~10 Hz (cada 100ms)
  Receive rate: ~10 Hz
}
```

**Restricción:** Team messages se envían cada 100ms, RL recibe información de compañeros con ~100ms de latencia.

### Sincronización con Compañeros

```
Robot 1:    Robot 2:    Robot 3:    Robot 4:
Send msg    Send msg    Send msg    Send msg
  ↓           ↓           ↓           ↓
Recibe en: ~10-20ms de latencia promedio

💡 RL NO puede asumir sincronización perfecta
⚠️ RL debe usar información potencialmente desactualizada
```

---

## 5. RESTRICCIONES DEL JUEGO (RoboCup Rules)

### SPL Rules 2024 Aplicables

| Regla | Restricción | Impacto en RL |
|-------|------------|--------------|
| **Game Phases** | INITIAL→READY→SET→PLAYING | RL solo actúa en PLAYING |
| **Ball in/out** | Si balón sale del campo, reset | RL debe respetar límites campo |
| **Illegal Area** | Portero solo en zona penal | RL no puede enviar defender a zona penal |
| **Offside (5v5)** | No en SPL (H-League), pero pueden existir variantes | Depende del torneo |
| **Kick-off** | Equipo designado inicia con balón | RL must respect kick-off formation |
| **Free Kick** | Contrario a 500mm del balón | RL debe respetar distancia |
| **Penalty Kick** | Striker 500mm de línea penal | RL no controla penalties (hardcoded) |
| **Fallen Robot** | No puede defender si está caído | RL must check `isUpright` de robots |

### Estado del Juego (GameState)

```cpp
// GameStateProvider.cpp
enum GameState {
  initial,      // Prepregame
  ready,        // Antes de saque inicial
  set,          // Listos para jugar
  playing,      // Juego en acción
  finished,     // Partido terminado
};

// RL SOLO ENTRA EN "playing"
// En otros estados:
//  - initial: robots inicializan posiciones
//  - ready: posiciones de saque
//  - set: último chequeo
//  - finished: game over

⚠️ RL Must check: if(gameState != PLAYING) return default_behavior;
```

---

## 6. DEPENDENCIAS DE MÓDULOS CRÍTICOS

### Módulos que RL DEBE Respetar

```
┌─────────────────────────────────────────────┐
│ RL Output (Decisión de acción)              │
└─────────────────┬───────────────────────────┘
                  │
         SkillBehaviorControl.cpp
         │
         ├─ Valida que skill exista
         ├─ Chequea restricciones (robot caído, etc)
         ├─ Genera MotionRequest
         │
         └─────────────────────────────────────┐
                                               │
    ┌──────────────────────────────────────────┘
    │
    ▼ Motion Thread (~200 FPS)
    │
    ├─ WalkingEngine
    │  └─ Valida velocidades (max 250 mm/s)
    │  └─ Genera joint angles
    │
    ├─ KickEngine
    │  └─ Valida ángulos
    │  └─ Ejecuta motion de pateo
    │
    └─ Constraints:
       - No puede generar acciones imposibles
       - No puede ignorar campos de force
       - No puede forzar al robot fuera de rango de movimiento
```

### Módulos que Alimentan a RL (Inputs)

```
ENTRADA A RL:

┌─ Perception ─────────────┐
│  BallPercept            │
│  RobotPose              │
│  ObstaclesFieldPercept  │
└──────────┬──────────────┘
           │
        ┌──┴───────────────────────────────┐
        │                                  │
    FieldBall                        ObstacleModel
    (fusionado)                      (robots cercanos)
        │                                  │
        └──────────────┬───────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
    GlobalTeammatesModel      GlobalOpponentsModel
    (dónde están compañeros)   (dónde están rivales)
        │                             │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
  ExpectedGoals              PassEvaluation
  (prob de gol)              (prob de pase)
        │                             │
        └──────────────┬──────────────┘
                       │
                    RL INPUT
                       │
        ┌──────────────┴──────────────────────┐
        │                                     │
    PlayBall::smashOrPass()        [RL Replaces]
    (decide shoot/pass)             (Fase 2)
        │                                     │
        └─────────────┬───────────────────────┘
                      │
                SkillRequest
                (ejecutar)
```

**Orden de Dependencia:**
1. Perception (Upper/Lower) → BallPercept, ObstaclesPerceptorData
2. Localización (SelfLocator) → RobotPose
3. Modelado (ObstacleModel, GlobalTeammatesModel) → Posiciones
4. Evaluación (ExpectedGoals, PassEvaluation) → Ratings
5. **Decisión (PlayBall) → SkillRequest**  ← **RL ENTRA AQUÍ**
6. Ejecución (SkillBehaviorControl, Motion) → Robot moves

---

## 7. MODO DE EJECUCIÓN

### Simulación vs Real

```
┌────────────────────────────────────┐
│ Ejecución del Sistema              │
├────────────────────────────────────┤
│                                    │
│  SIMULADOR (SimulatedNao)          │  ROBOT REAL (Physical NAO)
│  ─────────────────────────         │  ──────────────────────
│  - SimulatedRobot3D                │  - Comunicación LoLA
│  - Physics engine                  │  - Sensores reales
│  - Ejecución 1:1 con Real          │  - Actuadores reales
│  - Comportamiento determinístico   │  - Stochastic (ruido)
│  - Replay de logs posible          │  - No determinístico
│                                    │
└────────────────────────────────────┘
```

**RL Considerations:**
- RL entrenado en simulación
- Transfer a real requiere domain randomization
- Ruido en sensores != simulación

---

## 8. RESTRICCIONES DE SOFTWARE

### Restricciones de C++ que NO puede violar RL

```cpp
// Hard constraints que RL DEBE respetar:

// 1. Rango de valores (de kickInfo.cfg)
float kick_range_min = 400;      // mm
float kick_range_max = 5400;     // mm
float ball_velocity_max = 1800;  // mm/s
ASSERT(predicted_kick_distance >= kick_range_min &&
       predicted_kick_distance <= kick_range_max);

// 2. Límites del campo
float field_x_min = -2813;       // mm (propia línea de gol)
float field_x_max = 2813;        // mm (arco rival)
float field_y_min = -1875;       // mm
float field_y_max = 1875;        // mm
ASSERT(robot_position.x >= field_x_min && robot_position.x <= field_x_max);

// 3. Distancia mínima entre robots
float min_robot_distance = 200;  // mm (evitar colisión)
ASSERT(distance_to_teammate >= min_robot_distance);

// 4. Ángulo de visión
float fov_horizontal = 60;       // grados (aprox)
float fov_vertical = 47;         // grados (aprox)
ASSERT(target_angle within FOV);

// 5. Velocidad máxima
float max_walk_speed = 250;      // mm/s
ASSERT(predicted_speed <= max_walk_speed);

// 6. Hysteresis (timing)
float min_decision_change_time = 300;  // ms
ASSERT(time_since_last_change >= min_decision_change_time);

// 7. Game state
ASSERT(game_state == PLAYING);   // RL no actúa si no estamos jugando

// 8. Robot upright
ASSERT(robot.isUpright);         // No puede ser striker si está caído

// 9. Ball visibility
ASSERT(time_since_ball_seen < 5000);  // Max 5s sin ver balón
```

---

## 9. LISTA CRÍTICA: LO QUE RL PUEDE Y NO PUEDE HACER

### ✅ RL PUEDE HACER (Fase 2)

```python
learnable_actions = [
    # Decisión nivel Strategy
    "change_tactic(t121 ↔ t13)",

    # Parámetros de decisión Shoot/Pass
    "adjust_shootThreshold",           # De 0.8 a algo aprendido
    "adjust_passImprovement",          # De 0.15 a algo aprendido

    # Parámetros de posición
    "adjust_minPassDistance",          # De 1500 a algo aprendido
    "adjust_maxPassDistance",          # De 4500 a algo aprendido

    # Parámetros de hiperesis
    "adjust_hysteresis_penalty",       # De 0.1-0.8 a algo aprendido
    "adjust_decision_timing",          # De 300ms a algo aprendido

    # Parámetros de visión
    "adjust_minOpeningAngle",          # De 6° a algo aprendido
    "adjust_minAngleOffset",           # De 7° a algo aprendido
    "adjust_maxAngleOffset",           # De 25° a algo aprendido
]
```

### ❌ RL NO PUEDE HACER

```python
forbidden_actions = [
    # Física del robot
    "ignore_max_walk_speed (250 mm/s)",
    "violate_field_boundaries",
    "cause_collisions (< 200mm entre robots)",

    # Restricciones del juego
    "act cuando game_state != PLAYING",
    "mandar striker a zona penal",
    "ignorar restricciones de kick-off",

    # Hardware
    "cambiar configuración de sensores/cámaras",
    "modificar timing de threads",
    "cambiar cycle time de Cognition",

    # Comunicación
    "ignorar team messages",
    "cambiar período de broadcast (100ms)",

    # Reglas RoboCup
    "violar reglas de la competencia",
    "cambiar magicNumber de equipo",
    "actuar en fase INITIAL/READY/SET",
]
```

---

## 10. CHECKLIST DE AUDITORÍA

### Pre-Integración de RL

- [ ] **Timing**
  - [ ] ¿RL se ejecuta en < 3ms?
  - [ ] ¿Respeta cycle time de 33ms de Cognition?
  - [ ] ¿No bloquea otros threads?

- [ ] **Hardware**
  - [ ] ¿Valida velocidades (max 250 mm/s)?
  - [ ] ¿Respeta rango de visión (~60° horizontal)?
  - [ ] ¿Considera batería (no acciones costosas continuas)?

- [ ] **Restricciones de Juego**
  - [ ] ¿Solo actúa en game_state = PLAYING?
  - [ ] ¿Respeta kick-off formations?
  - [ ] ¿Respeta zonas ilegales (area penal portero)?
  - [ ] ¿Valida "ball still in field"?

- [ ] **Dependencias Módulos**
  - [ ] ¿Usa FieldBall (fusionado) en lugar de BallPercept crudo?
  - [ ] ¿Usa GlobalTeammatesModel y GlobalOpponentsModel?
  - [ ] ¿Verifica ExpectedGoals válido?
  - [ ] ¿Verifica PassEvaluation válido?

- [ ] **Restricciones de Software**
  - [ ] ¿Valida rango de kicks (400-5400mm)?
  - [ ] ¿Valida posiciones dentro del campo?
  - [ ] ¿Respeta distancia mínima entre robots (200mm)?
  - [ ] ¿Chequea robot.isUpright?
  - [ ] ¿Chequea tiempo sin ver balón (< 5s)?

- [ ] **Robustez**
  - [ ] ¿Maneja game_state transitions?
  - [ ] ¿Maneja robot falling/standing up?
  - [ ] ¿Maneja balón fuera del campo?
  - [ ] ¿Maneja equipo dejado fuera (< 4 jugadores)?

---

## 11. TABLA RESUMEN: RESTRICCIONES CRÍTICAS

| Restricción | Valor | Tipo | Check |
|-------------|-------|------|-------|
| **Cognition Cycle Time** | 33 ms | Hard | RL must complete in < 3ms |
| **Max Walk Speed** | 250 mm/s | Hard | Check motion requests |
| **Field Boundaries** | ±2813, ±1875 mm | Hard | Validate positions |
| **Min Robot Distance** | 200 mm | Hard | Avoid collisions |
| **Min Ball Vision Time** | < 5000 ms | Hard | Can't be striker if lost ball |
| **Game State** | = PLAYING | Hard | Check before acting |
| **Robot Upright** | = True | Hard | Check before assign striker |
| **Kick Range** | 400-5400 mm | Hard | Respect KickInfo |
| **Team Message Rate** | 10 Hz (~100 ms) | Soft | Expect 100ms latency |
| **Robot Vision FOV** | ~60° horizontal | Soft | Limit visible opponents |
| **Decision Hysteresis** | >= 300 ms | Soft | Prevent rapid changes |

---

## 12. CONCLUSIÓN: CONSIDERACIONES FINALES

### Para Fase 1 (RL solo posicionamiento)
✅ RL predice (x, y) de cada robot
✅ C++ ejecuta todas las decisiones (shoot/pass)
✅ Nivel de integración: BAJO
⚠️ Restricciones: Mantener validaciones C++ intactas

### Para Fase 2 (RL influencia en decisiones)
✅ RL ajusta parámetros (shootThreshold, distancias, etc)
✅ C++ usa esos parámetros en PlayBall::smashOrPass
⚠️ Restricciones: MUCHAS (timing, juego, hardware)
🔴 Crítico: Validar hard constraints en cada ciclo

### Para Fase 3 (RL reemplaza SmashOrPass)
✅ RL predice directamente: {shoot, pass_to_X, dribble}
✅ C++ valida y ejecuta
❌ Muchas restricciones rompen
🔴 Muy complejo - probablemente no viable en Fase 2

**Recomendación:** Mantener Fase 2 conservador. Implementar como:
- RL ajusta 3-4 parámetros clave (shootThreshold, distancias)
- C++ valida restricciones hard
- Incrementar complejidad con paciencia

