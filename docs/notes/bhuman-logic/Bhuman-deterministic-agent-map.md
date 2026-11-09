# BHuman — Mapa del Sistema de Agentes Deterministas

> **Propósito:** Identificar exactamente qué código de BHuman implementa la lógica determinista que usaremos como base para el `MapleTeacherPolicy`.

---

## Resumen ejecutivo

BHuman es un agente **completamente determinista**: dado el mismo estado del mundo, siempre produce la misma acción. No hay red neuronal en el loop de decisión táctica — todo son reglas explícitas (`if/else`, umbrales numéricos, selección por costo mínimo). Esta es exactamente la misma filosofía de nuestro `teacher_policy.py`.

---

## Arquitectura del pipeline determinista

```
GameState + Sensors
        │
        ▼
┌─────────────────────────────────────────────┐
│  StrategyBehaviorControl.cpp (módulo raíz)  │
│  • Gestiona lista de agentes activos        │
│  • Actualiza estado propio y de compañeros  │
│  • Produce: SkillRequest + StrategyStatus   │
└─────────────────┬───────────────────────────┘
                  │ llama a
                  ▼
┌─────────────────────────────────────────────┐
│  Behavior::update()  [Behavior.cpp L245]    │
│  La función central del agente determinista │
│                                             │
│  Paso 1: Selección de SetPlay               │
│  Paso 2: Selección de Tactic                │ ← máquina de estados
│  Paso 3: assignPositions()                  │ ← asignación óptima
│  Paso 4: assignRoles()                      │ ← quién juega la pelota
│  Paso 5: execute()                          │ ← SkillRequest final
└─────────────────────────────────────────────┘
```

---

## Archivo 1: El cerebro — `Behavior.cpp`

**Ruta:** [`Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp`](file:///c:/Users/MSI/Desktop/Programacion/SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp)

Este es el **núcleo del agente determinista**. Contiene toda la lógica de toma de decisiones táctica multiagente.

### Función central: `Behavior::update()` — [L245](file:///c:/Users/MSI/Desktop/Programacion/SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp#L245-L532)

```
Behavior::update(strategy, self, agents)
├── Paso 1: Manejo de SetPlays (jugadas preparadas)
│   ├── isReady()  → proponer/aceptar SetPlay
│   ├── isSet()    → congelar SetPlay
│   └── isPlaying() → ejecutar/terminar SetPlay
│
├── Paso 2: Selección de Tactic [L462-L519]
│   ├── Máquina de estados por condiciones:
│   │   • numOfFieldPlayersLE/GE
│   │   • scoreDifference
│   │   • timeSinceBallAhead/Behind
│   └── Votación por mayoría entre agentes
│
├── Paso 3: assignPositions() [L614]
│   ├── Asigna goalkeeper fijo
│   ├── Construye matriz de costos (distancia agente↔posición)
│   └── Búsqueda óptima por permutaciones → asignación mínimo costo
│
├── Paso 4: assignRoles() [L875]
│   └── determineActiveAgent() → ¿quién va a jugar la pelota?
│       • Calcula TTRB (Time To Reach Ball) por agente
│       • Quién llegue antes = striker (playBall role)
│       • Resto = position roles (Defender, Midfielder, Forward)
│
└── Paso 5: execute() [L907]
    ├── isReady → walkTo(basePose)
    ├── isSet   → stand()
    ├── SetPlay activo → acción del SetPlay
    └── role activo → roles[agent.role]->execute(agent, teammates)
```

### `determineActiveAgent()` — [L1065](file:///c:/Users/MSI/Desktop/Programacion/SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp#L1065)

**Esta es la lógica de "striker assignment" de BHuman.**

Calcula el **TTRB (Time To Reach Reference Pose)** para cada agente:
- Considera pose actual, velocidad de caminata, si estaba caído (penalización)
- El agente con menor TTRB recibe el rol `ActiveRole::playBall`
- Incluye histéresis: quien ya tenía el rol recibe una ventaja de 1000ms

```cpp
// Pseudo-código del algoritmo
float ttrb = KickSelection::calcTTRP(
    agent.pose.inverse() * getReferencePoseOnField(ballOnField),
    walkSpeed  // (100deg, 260mm/s, 220mm/s)
);
if(wasActive) ttrb -= 1000.f;  // ← hysteresis / estabilidad de rol
if(!agent.isUpright) ttrb += 5000.f;  // ← penalización por caída
```

---

## Archivo 2: El rol activo — `PlayBall.cpp`

**Ruta:** [`Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp`](file:///c:/Users/MSI/Desktop/Programacion/SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp)

**¿Qué hace?** Una vez que un agente tiene el rol `playBall`, ejecuta el algoritmo `smashOrPass`:

```
smashOrPass(self, teammates)
├── Si xG_propio > shootThreshold (0.8) → SHOOT
├── Para cada compañero en rango [1500mm, 4500mm]:
│   │   sampleValue = P(pase llega) × xG(posición_compañero) × P(rival no intercepta)
│   └── Si sampleValue > xG_propio → PASS to teammate
└── Si ningún pase mejora → SHOOT (o DRIBBLE hacia mejor posición)
```

**Parámetros clave (todos umbrales fijos — 100% determinista):**
| Parámetro | Valor | Significado |
|-----------|-------|-------------|
| `shootThreshold` | 0.8 | xG mínimo para disparar directo |
| `minPassDistance` | 1500mm | Distancia mínima de pase |
| `maxPassDistance` | 4500mm | Distancia máxima de pase |
| `decisionPenalty` | 0.1-0.8 | Penalización por cambiar decisión (estabilidad) |

---

## Archivo 3: El módulo raíz — `StrategyBehaviorControl.cpp`

**Ruta:** [`Src/Modules/BehaviorControl/StrategyBehaviorControl/StrategyBehaviorControl.cpp`](file:///c:/Users/MSI/Desktop/Programacion/SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/StrategyBehaviorControl.cpp)

Gestiona el estado de los agentes del equipo. Las funciones clave:

- **`updateAgentBySelf()`** — actualiza estado propio (pose, ball, tactic, role)
- **`updateAgentByTeamMessage()`** — actualiza estado de compañeros por WiFi
- **`updateCurrentPosition()`** — propaga posición estimada al tiempo actual

**Detección de `disagreeOnBall`** [L222]:
```cpp
// Dos robots "no están de acuerdo en la pelota" si sus estimaciones
// difieren > 777mm (con histéresis de 222mm).
agent.disagreeOnBall = (itsBallOnField - myBallOnField).squaredNorm()
                       > sqr(777.f + (agent.disagreeOnBall ? 0.f : 222.f));
```
→ Agentes con `disagreeOnBall=true` son **excluidos** del cálculo de quién juega la pelota.

---

## Árbol completo de roles

```
Role (abstracto)
├── ActiveRole (el que tiene la pelota)
│   ├── PlayBall      ← smashOrPass / executeLegacy
│   ├── FreeKickWall  ← forman pared defensiva
│   └── ClosestToTeammatesBall
└── PositionRole (los que no tienen la pelota)
    ├── Goalkeeper    ← lógica de portero
    ├── Defender      ← posición defensiva
    ├── Midfielder    ← posición de mediocampo
    └── Forward       ← posición de delantero
```

**Archivos de PositionRoles:**
```
Src/Modules/BehaviorControl/StrategyBehaviorControl/PositionRoles/
├── Goalkeeper.cpp/.h
├── Defender.cpp/.h
├── Midfielder.cpp/.h
└── Forward.cpp/.h
```

---

## Estructura del Agent (`Agent.h`)

**Ruta:** [`Src/Tools/BehaviorControl/Strategy/Agent.h`](file:///c:/Users/MSI/Desktop/Programacion/SabanaHerons2026/Src/Tools/BehaviorControl/Strategy/Agent.h)

El `struct Agent` es el estado compartido de cada robot:

```cpp
struct Agent {
  int number;                    // Número de jugador
  bool isGoalkeeper;

  Pose2f pose;                   // Pose estimada
  Vector2f currentPosition;      // Posición propagada al "ahora"
  Vector2f ballPosition;         // Pelota relativa al robot
  bool disagreeOnBall;           // Descartado del cálculo de striker

  Tactic::Type proposedTactic;   // Voto de táctica
  Tactic::Type acceptedTactic;   // Táctica usada (mayoría)
  SetPlay::Type acceptedSetPlay; // Jugada preparada activa
  Tactic::Position::Type position; // Posición en la táctica (Midfielder, etc.)
  Role::Type role;               // Rol actual (PlayBall, Defender, etc.)
};
```

---

## Relación directa con nuestro `MapleTeacherPolicy`

| BHuman (C++ determinista) | Maple Teacher Policy (Python) | Nota |
|---|---|---|
| `determineActiveAgent()` | `is_striker` bit (obs[26]) | BHuman calcula quién es striker por TTRB; nosotros por distancia a la pelota |
| `smashOrPass()` con `xG > shootThreshold` | `shoot_armed` gate + `stage_aware_teacher_action` | Misma lógica: ¿tengo buen ángulo? disparo |
| `smashOrPass()` pase a compañero | `is_open_support` bit (obs[27]) | BHuman evalúa `passEvaluation * xG`; nosotros usamos `is_open_for_pass` |
| `PositionRole::execute()` para supporters | `is_off_ball_support` bit (obs[28]) | BHuman usa posiciones base de táctica; nosotros triangulación |
| Hysteresis en TTRB (1000ms) | Hysteresis de 5 frames en rol | Mismo principio anti-thrash |
| `disagreeOnBall` exclusion | Ninguno aún | Podríamos agregar si hay ruido en percepción |

---

## Anotaciones de revisión para el teacher multiagente

Estas notas fijan la interpretación correcta para usar BHuman como **teacher
determinista** del PPO multiagente, no como lógica táctica de producción.

### Pase en BHuman: la estrategia elige jugador, la skill resuelve geometría

`PlayBall::smashOrPass()` no entrega una trayectoria completa al layer táctico.
Evalúa compañeros, selecciona el mejor candidato y devuelve:

```cpp
SkillRequest::Builder::passTo(passTarget->number)
```

Es decir, el output táctico relevante para nuestro PPO es el **id del teammate**.
La geometría fina del pase queda dentro de `PassToTeammate`, que busca al teammate
por `playerNumber`, estima/lookahead de su pose, ajusta el punto de pase hacia el
lado útil para el gol, evita sectores bloqueados por obstáculos y decide el kick.

Confirmado en código:

- BHuman strategy elige el candidato y llama `passTo(passTarget->number)`:
  `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp:105-108`.
- `SkillRequest::Builder::passTo(int target)` solo setea `skill=pass` y
  `passTarget=target`:
  `SabanaHerons2026/Src/Representations/BehaviorControl/SkillRequest.cpp:38-43`.
- `SkillRequest.passTarget` está documentado como "the number of the passed-to
  player":
  `SabanaHerons2026/Src/Representations/BehaviorControl/SkillRequest.h:44-47`.
- `PassToTeammate` busca el teammate por `playerNumber`:
  `SabanaHerons2026/Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/PassToTeammate.cpp:331-342`.

Implicación para el teacher:

- `smash_or_pass()` debe elegir el mejor teammate candidato y emitir `pass_target`
  como id/slot estable que el env pueda mapear a `playerNumber`.
- No necesitamos enseñar al PPO la trayectoria completa del pase en los parámetros
  continuos; para `pass`, `ACTION_PARAM_MASKS` solo deja activo `pass_target`.
- Sí necesitamos que la selección del candidato sea buena: BHuman usa
  `passEvaluation × expectedGoals × opponentRating`, más `decisionPenalty`.

### Obs wire: los slots existen en el vector codificado, no como campos raw

Rafael sí cableó los slots de teammate:

- obs `26/27`: nearest teammate `(x_rel, y_rel)`
- obs `28/29`: second teammate
- obs `30/31`: third teammate

Pero esos nombres viven en `observation.py:ObsIndex` y en el vector codificado de
`environment._encode_obs()`. En `BackendObservation` los datos raw llegan como:

- `teammate_poses`
- `teammate_ages_ms`
- `teammate_activities`

Confirmado en código RL:

- `observation.py:36-41` define `nearest_teammate_x_rel` ...
  `third_teammate_y_rel` en obs `26-31`.
- `environment.py:784-810` construye esos slots desde `obs.teammate_poses`,
  filtra por edad, ordena por distancia y transforma a robot-frame normalizado.
- `pybh_backend.py:339-346` define los campos raw `teammate_poses`,
  `teammate_ages_ms`, `teammate_activities` y team-ball.
- `pybh_backend.py:432-438` parsea esos campos desde el payload de BHuman/pybh.

Implicación para el teacher:

- Si una helper recibe `BackendObservation`, debe construir slots desde
  `teammate_poses` y aplicar el mismo sort/stale-zero que `_encode_obs()`.
- Si una helper quiere leer `nearest_teammate_x_rel`, debe recibir el vector obs
  codificado o un paquete explícito del env, no `BackendObservation`.
- Cuidado con unidades: los slots codificados están normalizados en robot-frame;
  `TeammateSlot.x_field/y_field` deben estar en milímetros de campo.

### Qué bases cooperativas de BHuman ya captura el teacher

El `multiagent_teacher.py` actual captura la parte ofensiva mínima:

- rol activo / striker por cercanía a pelota con histéresis;
- rama striker que reutiliza `stage_aware_teacher_action()` para preservar el
  comportamiento single-agent;
- `smash_or_pass()` como versión sprint de `PlayBall::smashOrPass()`;
- exclusión del GK como candidato de pase;
- support slots inspirados en Forward/Midfielder;
- anti-thrash local vía role hysteresis, last-pass penalty y slot stability.

Esto es suficiente como esqueleto de teacher ofensivo de sprint, pero no es aún
un port completo de la cognición cooperativa de BHuman.

### Bases cooperativas de BHuman que faltan o están simplificadas

1. **TTRB real vs distancia a pelota.** BHuman no elige striker solo por distancia:
   calcula tiempo a una pose de referencia detrás de la pelota, incluye velocidad
   de caminata, ventaja al agente ya activo, penalización por caída, pérdida de
   pelota y filtros de `canBeActive`.

2. **Arbitraje de creencia de pelota.** BHuman usa `disagreeOnBall` para excluir
   agentes con una pelota incompatible. Nuestra obs tiene Block C/E para team-ball
   y consenso; el teacher debería usar esos datos para evitar roles venenosos
   cuando un robot ve una pelota distinta.

3. **Caso `closestToTeammatesBall`.** Si yo no veo pelota pero el team-ball es
   válido, BHuman puede asignar un rol activo de búsqueda/turn-to-ball. Este sprint
   solo deja `observe_armed` como bit futuro; no hay rama completa de búsqueda.

4. **Asignación óptima de posiciones.** BHuman `assignPositions()` minimiza una
   matriz de costos agente↔posición para una táctica. El teacher actual usa una
   grid fija y elige support por `forward_rating`; eso es útil, pero no reproduce
   la asignación global de formación.

5. **Pass evaluation completa.** BHuman evalúa target libre, línea de pase libre,
   target dentro de campo, línea de tiro posterior y rango. El teacher usa una
   proxy de `nearest_opponent_front_dist`, lo cual es compatible con la obs actual
   pero menos expresivo.

6. **Receive-pass signaling.** BHuman receivers pueden reaccionar a `passTarget` /
   `shootingTo` por BehaviorStatus. Nuestra obs actual no tiene “me están pasando”
   explícito; el receiver teacher solo camina al slot abierto.

7. **Defensa.** Defender, mark, block y recovery están fuera de este sprint. Para
   cooperación defensiva real se necesita §14.26: obs con oponentes, defender role
   y unmask de mark/block.

### Correcciones necesarias antes de usar el teacher para dataset

- `compute_open_lane_target()` debe devolver `(x, y)`, no `"walk"`/`"stand"`.
- `teammate_slots_from_obs()` debe construirse desde `teammate_poses` o cambiar
  su firma para recibir el obs codificado + metadata de slots.
- `decisionPenalty` debe crecer cuando la pelota está cerca, como en BHuman.
- `pass` labels deben estar alineadas con `pass_armed` y la máscara de obs `41`;
  no deben depender solo de `shoot_armed`.
- Role bits finales pertenecen a obs `44/45/46`, no `26/27/28`.

---

## Archivos a estudiar para el Maple Teacher Policy

En orden de prioridad:

1. **[`Behavior.cpp`](file:///c:/Users/MSI/Desktop/Programacion/SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp)** — `determineActiveAgent()` L1065 y `assignRoles()` L875
2. **[`PlayBall.cpp`](file:///c:/Users/MSI/Desktop/Programacion/SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp)** — `smashOrPass()` — toda la lógica de shoot/pass
3. **[`StrategyBehaviorControl.cpp`](file:///c:/Users/MSI/Desktop/Programacion/SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/StrategyBehaviorControl.cpp)** — L171 `updateAgentBySelf()`, L202 `updateAgentByTeamMessage()`
4. **[`Agent.h`](file:///c:/Users/MSI/Desktop/Programacion/SabanaHerons2026/Src/Tools/BehaviorControl/Strategy/Agent.h)** — definición del estado compartido
5. **`PositionRoles/`** — lógica de posicionamiento de supporters
