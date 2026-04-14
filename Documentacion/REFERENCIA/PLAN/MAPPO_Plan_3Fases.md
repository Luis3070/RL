# Plan de Integración RL-MAPPO en 3 Fases — SabanaHerons2026

> **Fecha:** Marzo 2026
> **Formato target:** **4v4 Scaled** (normativa competencia) — 1 portero + 3 jugadores de campo por equipo
> **Campo:** 5626 × 3750 mm (`xPosOpponentGroundLine = 2813`, `yPosLeftSideline = 1875`)
> **Estrategia activa:** `Config/Behavior/Strategies/s4v4_scaled.cfg`
> **Tácticas activas:** `t121_4v4_scaled` (defensiva), `t13_4v4_scaled` (ofensiva)
> **Objetivo:** Reemplazar progresivamente las heurísticas de posicionamiento, roles y estrategia con MAPPO.

---

## Tabla de Contenidos

1. [Decisiones de Scope](#decisiones-de-scope)
2. [Sobre confiar a ciegas en los datos (sin covarianzas)](#sobre-confiar-a-ciegas-en-los-datos)
3. [Fase 1 — RL solo define POSICIONES](#fase-1--rl-solo-define-posiciones)
4. [Fase 2 — RL define POSICIÓN + ROL](#fase-2--rl-define-posición--rol)
5. [Fase 3 — RL suplanta TODA táctica y estrategia](#fase-3--rl-suplanta-toda-táctica-y-estrategia)
6. [Comunicación .cfg ↔ Python: Parser y Bridge](#comunicación-cfg--python)
7. [Observation Space definitivo (todas las fases)](#observation-space-definitivo)
8. [Resumen de archivos a modificar por fase](#resumen-de-archivos-por-fase)

---

## Preguntas Técnicas Resueltas

### P1: Campo de visión (FOV) y ángulo de observación

**Propio robot:** Sí se puede saber hacia dónde mira.
- Cámaras: apertura `54.7° × 42.5°` (ambas cámaras) — [cameraIntrinsics.cfg](Config/Robots/Default/cameraIntrinsics.cfg)
- Head yaw/pitch disponibles vía `JointAngles.angles[Joints::headYaw]` y `[Joints::headPitch]` — [JointAngles.h L38](Src/Representations/Infrastructure/JointAngles.h#L38), [Joints.h L12-L13](Src/Libs/RobotParts/Joints.h#L12-L13)
- El ángulo de observación efectivo = `self_theta + headYaw` (aprox). Pero para el RL sim2D **no lo incluimos** porque en simulación no hay FOV limitado — todos los agentes "ven" todo el campo.

**Compañeros:** `GlobalTeammatesModel.TeammateEstimate.pose.rotation` existe pero el header dice explícitamente *"rotation is currently unreliable"*. **No incluir en obs** — meter datos poco fiables confundiría la policy.

**Oponentes:** `GlobalOpponentsModel.OpponentEstimate` NO tiene orientación. Solo `position`, `left`, `right` (bordes del obstáculo). **No disponible.**

> **Decisión para sim2D:** No incluir FOV/head angles. En simulación 2D todos ven todo. Si en el futuro se hace sim3D con percepción parcial, se puede añadir.

### P2: Velocidades y aceleraciones disponibles

| Entidad | vx, vy | ω (angular) | ax, ay | α (angular) | Fuente |
|---------|--------|-------------|--------|-------------|--------|
| **Self** | ✅ `MotionInfo.speed.translation` (mm/s) | ✅ `MotionInfo.speed.rotation` (rad/s) | ✅ `InertialSensorData.acc` (m/s²) | ✅ `InertialSensorData.gyro` (rad/s) | [MotionInfo.h L53](Src/Representations/MotionControl/MotionInfo.h#L53), [InertialSensorData.h L16-L18](Src/Representations/Infrastructure/SensorData/InertialSensorData.h#L16-L18) |
| **Compañeros** | ❌ Solo `speed` escalar (mm/s, forward) | ❌ | ❌ | ❌ | [BehaviorStatus.h L23](Src/Representations/BehaviorControl/BehaviorStatus.h#L23) |
| **Oponentes** | ❌ Nada | ❌ | ❌ | ❌ | [GlobalOpponentsModel.h](Src/Representations/Modeling/GlobalOpponentsModel.h) |
| **Balón** | ✅ `BallModel.estimate.velocity` (mm/s, relativo al robot) | N/A | ❌ | N/A | [BallModel.h L25](Src/Representations/Modeling/BallModel.h#L25) |

**Aceleraciones del self (IMU):** Existen pero son datos de nivel motor — el behavior de BHuman nunca las usa. En simulación 2D tampoco las simularemos. **No incluir en obs por ahora.**

**Velocidad del self:** `MotionInfo.speed` es un `Pose2f` con `(rotation, translation.x, translation.y)` = `(ω, vx, vy)`. **SÍ añadir al obs space.** Es información crítica — un agente que sabe que va a 250mm/s puede anticipar cuándo llegará a su target.

### P3: ¿Se puede controlar vx, vy y ω como action?

**SÍ.** BHuman tiene tres interfaces de caminar:

1. **`WalkAtAbsoluteSpeed(Pose2f speed)`** — velocidad directa en mm/s y rad/s — [SkillInterfaces.h L24](Src/Representations/BehaviorControl/SkillInterfaces.h#L24)
2. **`WalkAtRelativeSpeed(Pose2f speed)`** — ratio [0,1] del máximo — [SkillInterfaces.h L30](Src/Representations/BehaviorControl/SkillInterfaces.h#L30)
3. **`WalkToPose(Pose2f target, ...)`** — target pose con obstacle avoidance — [SkillInterfaces.h L41](Src/Representations/BehaviorControl/SkillInterfaces.h#L41)

**Pero para el RL Fase 1-2 mantenemos `target_pose` (x, y, θ) como action, NO velocidades directas.** ¿Por qué?
- El PathPlanner traduce target → velocidades considerando obstáculos. Si el RL diera velocidades directas, perdemos obstacle avoidance.
- Target pose es más natural para aprender estrategia ("ve a esta posición") vs velocidades ("muévete a 150mm/s hacia la derecha").
- Fase 3 podría explorar velocidad directa como acción si se quiere control fino.

### P4: Oponentes locales en coordenadas absolutas — limitación y estrategia

**¿Cómo funciona?** `GlobalOpponentsTracker` mantiene hipótesis EKF en coordenadas **relativas al robot**. Al llenar `GlobalOpponentsModel`, convierte a absolutas con `theRobotPose * ob.center` — [GlobalOpponentsTracker.cpp L70-L76](Src/Modules/Modeling/GlobalOpponentsTracker/GlobalOpponentsTracker.cpp#L70-L76). Cada robot solo ve lo que está en su FOV (~55° horizontal). No hay consenso entre robots.

**Limitaciones:**
- Un robot solo "sabe" de oponentes que ha visto recientemente
- Si un oponente está detrás del robot, desaparece del modelo
- No hay compartición de detecciones de oponentes entre teammates
- Depende de la calidad de `RobotPose` (localización) para la conversión a absolutas

**Estrategia para el primer entrenamiento:**

Desde Fase 1, la simulación replica la limitación real: **cada agente solo ve oponentes dentro de su FOV y la posición depende de su `RobotPose`**. Esto cierra brecha sim2real desde el inicio.

Implementación en el entorno sim2D:

```python
def get_visible_opponents(agent_pos, agent_theta, all_opponents,
                          fov_half=0.48,      # ~27.5° (mitad de 54.7°)
                          max_distance=5000,   # mm — GlobalOpponentsTracker.maxDistance
                          detection_prob=0.9): # probabilidad de detección si está en FOV
    """
    Simula GlobalOpponentsTracker: solo detecta oponentes dentro del FOV.
    agent_theta ya incluye la orientación del cuerpo (no head yaw, simplificado).
    """
    visible = []
    for opp in all_opponents:
        dx = opp[0] - agent_pos[0]
        dy = opp[1] - agent_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        angle_to_opp = normalize_angle(np.arctan2(dy, dx) - agent_theta)
        
        if abs(angle_to_opp) < fov_half and dist < max_distance:
            if np.random.random() < detection_prob:
                visible.append(opp)
    return visible
```

Características del modelo:
- **FOV simplificado:** cono de ±27.5° desde `self_theta` (apertura de cámara real = 54.7°). No modelamos head yaw por simplicidad — el robot "mira" hacia donde camina.
- **Distancia máxima:** 5000mm, consistente con `GlobalOpponentsTracker.maxDistance`.
- **Detección probabilística:** 90% si está en FOV (modela falsos negativos por oclusión, blur, etc.).
- **Persistencia temporal:** Un oponente detectado permanece en memoria por `deleteAfter` ms (~1000ms en config) antes de desaparecer. Esto replica el EKF con predicción por odometría.
- **Conversión local→global:** La posición observada depende de `RobotPose`. En sim, si añadimos ruido a la localización del agente, los oponentes también se desplazan — exactamente como en el robot real.

Esto obliga a la policy a aprender:
- La importancia de orientarse para "ver" oponentes cercanos
- Que oponentes detrás del robot desaparecen de la observación
- Robustez a información parcial sobre oponentes

### P5: Ángulo entre Nao y arco rival — ¿hay función en BHuman?

**SÍ, BHuman ya tiene esto.** En `ExpectedGoalsProvider` — [ExpectedGoalsProvider.cpp L63-L66](Src/Modules/BehaviorControl/ExpectedGoalsProvider/ExpectedGoalsProvider.cpp#L63-L66):

```cpp
const Angle angleToLeftPost = (leftGoalPost - pointOnField).angle() - leftAngleOffset;
const Angle angleToRightPost = (rightGoalPost - pointOnField).angle() + rightAngleOffset;
float openingAngle = angleToLeftPost - angleToRightPost;  // ángulo de apertura al gol
```

Además, `getOpeningAngle(pointOnField)` — [L100-L150](Src/Modules/BehaviorControl/ExpectedGoalsProvider/ExpectedGoalsProvider.cpp#L100-L150) — calcula el ángulo libre restando los sectores bloqueados por oponentes.

**Para la simulación Python, el cálculo es trivial:**

```python
import numpy as np

def goal_opening_angle(agent_pos, goal_left=(4500, 800), goal_right=(4500, -800)):
    """Ángulo de apertura al arco rival desde una posición."""
    angle_left = np.arctan2(goal_left[1] - agent_pos[1], goal_left[0] - agent_pos[0])
    angle_right = np.arctan2(goal_right[1] - agent_pos[1], goal_right[0] - agent_pos[0])
    return angle_left - angle_right  # radianes

def goal_center_angle(agent_pos, agent_theta, goal_center=(4500, 0)):
    """Ángulo RELATIVO del arco rival respecto a la orientación del agente."""
    dx = goal_center[0] - agent_pos[0]
    dy = goal_center[1] - agent_pos[1]
    absolute_angle = np.arctan2(dy, dx)
    return normalize_angle(absolute_angle - agent_theta)  # en [-pi, pi]
```

**¿Incluir en obs?** Sí, como features derivadas:
- `goal_opening_angle`: ángulo de apertura al arco (rad) — indica calidad de posición para disparar
- `goal_relative_angle`: ángulo del arco respecto a la orientación del agente — indica si está "mirando" al arco

Estas NO son mediciones nuevas, son derivadas de `(self_x, self_y, self_theta)` y las constantes del campo. Pero darlas pre-calculadas ayuda a la policy a converger más rápido.

### P6: Importar PathPlanner al entorno Python — análisis

**¿Qué hace el PathPlanner?** Recibe un target `Pose2f` y devuelve `ObstacleAvoidance` (vectores de evasión + path segments) — [PathPlanner.h L16-L26](Src/Representations/BehaviorControl/PathPlanner.h#L16-L26). Internamente usa un **grafo de visibilidad + A*** para navegar alrededor de obstáculos — [PathPlannerProvider.h L25-L59](Src/Modules/BehaviorControl/PathPlannerProvider/PathPlannerProvider.h#L25-L59).

**Dependencias:** `FieldBall, FieldDimensions, FrameInfo, GameState, IllegalAreas, ObstacleModel, RobotPose, StrategyStatus`.

**Opinión: excelente idea, pero no portar el PathPlanner de C++ a Python.**

Razones:
1. El PathPlanner depende de **8 representaciones BHuman** — portarlo implica portar todo el framework.
2. En simulación 2D, los agentes se mueven en línea recta hacia su target. No hay percepción parcial ni obstáculos complejos.
3. Lo que SÍ debemos implementar en Python es un **modelo de movimiento realista**:

```python
class NaoMovementModel:
    """Simula cómo un Nao se mueve hacia un target — reemplaza PathPlanner."""
    
    # Velocidades realistas (de walkingEngine.cfg Config/Robots/Default)
    MAX_FORWARD = 250.0    # mm/s
    MAX_SIDEWAYS = 200.0   # mm/s
    MAX_BACKWARDS = 150.0  # mm/s
    MAX_ROTATION = 2.09    # rad/s (120°/s)
    STEP_DURATION = 0.25   # s por paso
    
    def step(self, pos, theta, target_pos, target_theta, dt):
        """Mueve el robot hacia el target respetando límites de velocidad."""
        # Vector al target
        dx = target_pos[0] - pos[0]
        dy = target_pos[1] - pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Ángulo absoluto al target
        angle_to_target = np.arctan2(dy, dx)
        # Diferencia angular
        dtheta = normalize_angle(angle_to_target - theta)
        
        # Velocidad limitada
        if dist > 10:  # si no estamos ya en el target
            # Primero rotar hacia el target, luego avanzar
            if abs(dtheta) > 0.3:  # ~17°: rotar primero
                omega = np.clip(dtheta / dt, -self.MAX_ROTATION, self.MAX_ROTATION)
                vx, vy = 0, 0
            else:
                # Avanzar en dirección del target
                speed = min(dist / dt, self.MAX_FORWARD)
                vx = speed * np.cos(dtheta)  # en coords robot
                vy = speed * np.sin(dtheta)
                vx = np.clip(vx, -self.MAX_BACKWARDS, self.MAX_FORWARD)
                vy = np.clip(vy, -self.MAX_SIDEWAYS, self.MAX_SIDEWAYS)
                omega = np.clip(dtheta / dt, -self.MAX_ROTATION, self.MAX_ROTATION)
        else:
            vx, vy = 0, 0
            omega = np.clip(normalize_angle(target_theta - theta) / dt,
                           -self.MAX_ROTATION, self.MAX_ROTATION)
        
        # Actualizar pose
        new_theta = theta + omega * dt
        new_x = pos[0] + (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        new_y = pos[1] + (vx * np.sin(theta) + vy * np.cos(theta)) * dt
        
        return (new_x, new_y), new_theta
```

Esto **cierra brecha sim2real** porque:
- El agente no se teletransporta — tarda en llegar a su target
- Respeta velocidades máximas reales del Nao
- El agente aprende que cambiar de posición tiene costo temporal
- No necesita aprender a caminar (eso lo hace el WalkingEngine real)

**Collision avoidance simplificado:** Añadir repulsión entre agentes (distancia mínima ~300mm).

### P7: Velocidades realistas y conservadoras para el entrenamiento

De los .cfg del proyecto ([walkingEngine.cfg](Config/Robots/Default/walkingEngine.cfg)):

| Parámetro | Default | Nao específico | CompetitionWalk | **Recomendado para sim** |
|-----------|---------|---------------|-----------------|------------------------|
| Forward (mm/s) | 250 | 300 | 270 | **250** (conservador) |
| Sideways (mm/s) | 200 | 230 | 230 | **200** |
| Backwards (mm/s) | 150 | 150 | 150 | **150** |
| Rotation (°/s) | 120 | 120 | 120 | **120** |
| Step duration (s) | 0.25 | 0.25 | 0.25 | **0.25** |

**Recomendación: usar valores Default (conservadores).**

Por qué Default y no Nao:
- Default = 250mm/s forward. Nao = 300mm/s. La diferencia es 20%.
- En competencia real, el robot rara vez alcanza max speed por largos períodos (aceleración, giros, terreno).
- Valores conservadores → la policy aprendida funciona en cualquier robot, no solo en el Nao más rápido.
- Si la policy entrenada con 250mm/s funciona bien, al desplegarse en un Nao a 300mm/s solo se beneficia.

**Datos adicionales útiles:**
- `walkSpeedParams.maxSpeed` = 350mm/s forward (teórico, nunca se usa en comportamiento normal)
- `maxForwardAcceleration = 20.5mm` por step — el Nao NO acelera instantáneamente. En sim, modelar con un ramp-up simple.

---

## Decisiones de Scope

| Tema | Decisión | Razón |
|------|----------|-------|
| **IMU / GyroState** | ❌ Descartados del obs space | Ya integrados en el PathPlanner y WalkingEngine. El behavior nunca los ve. |
| **Oponentes en mensajes** | ❌ Fuera de scope por ahora | Se queda `GlobalOpponentsModel` local. No modificar `TeamMessageHandler`. |
| **Batería** | ✅ Incluir como feature | `batteryLevel` (0.0–1.0) de `SystemSensorData`. Sin reward shaping por ahora, solo observación. |
| **Temperatura** | ❌ Fuera de scope | No incluir `maxJointTemperatureStatus` ni `cpuTemperature`. |
| **Covarianzas** | ❌ No en el primer RL | Confiar a ciegas en los datos. Ver sección siguiente. |

---

## Sobre confiar a ciegas en los datos

### ¿Es válido un primer RL sin covarianzas?

**Sí, y es la decisión correcta para el primer iteration.**

Razones:

1. **Simplifica el obs space:** En vez de ~125+ floats con covarianzas, timestamps y confidence scores por slot, el obs space queda en ~60-80 floats limpios. Menos dimensiones = convergencia más rápida.

2. **El sistema BHuman ya filtra:** Todo lo que llega al behavior ya pasó por UKF (localización), multi-hypothesis Kalman (balón), y EKF (obstáculos). Los datos "crudos" del behavior son en realidad estimaciones filtradas de alta calidad. No es como confiar en sensores raw.

3. **El gap sim2real se manifiesta aquí.** Este es el punto clave: en simulación, los datos son perfectos (o con ruido controlado). En el robot real, los datos tienen incertidumbre no modelada. Al entrenar sin covarianzas, la policy asume que los datos son verdad. Cuando se despliega en el robot real, hay tres fuentes de discrepancia:

| Fuente de gap | En simulación | En robot real | Impacto |
|---------------|--------------|---------------|---------|
| `RobotPose` | Exacta o ruido gaussiano simple | UKF con multimodalidad, jumps, `okay/poor` quality | Posiciones absolutas (gol, compañeros) desplazadas |
| `BallModel` | Posición exacta + ruido | Kalman con oclusiones, `seenPercentage` variable, fallback a team ball | Balón "fantasma" o delayed |
| `GlobalOpponentsModel` | Posiciones conocidas | Solo ve oponentes en FOV, sin velocidad, sin covarianza | Oponentes desaparecen/aparecen |
| `GlobalTeammatesModel` | Posiciones exactas | Extrapolación con walk target, latencia 200-800ms | Compañeros desfasados |
| Comunicación | Instantánea | 200-800ms de latencia, budget limitado | Decisiones basadas en estado viejo |
| Caídas | Probabilísticas | Impredecibles, 3-7s recovery | Pérdida súbita de agente |

### Roadmap de incertidumbre

> **Nota:** La simulación de ruidos NO es parte de las 3 fases principales del proyecto. Es una fase opcional final (Fase 4) que solo tiene sentido si el RL ya funciona bien en simulación limpia. BHuman ya maneja todo el filtrado Kalman/UKF en sus módulos de percepción — no tenemos que replicar ese filtrado. Lo que eventualmente simularíamos son las *consecuencias observables* del ruido (posiciones desplazadas, ball ghost, oponentes que aparecen/desaparecen), no el ruido sensor crudo.

```
Fase 1-3: Datos limpios, sin ruido         ← PROYECTO ACTUAL
  │    Entrenar en sim2D perfecta
  │    Evaluar en sim2D
  │    Desplegar modelo en robot vía ONNX (si resultados buenos)
  │
Fase 4 (opcional, futura): Domain randomization + robustez
  │    Añadir ruido gaussiano a pose, ball, opponents en el env
  │    Randomizar latencia de comunicación (200-800ms)
  │    Randomizar caídas (prob = f(velocidad, giro))
  │    Entrenar con curriculum: empieza limpio → aumenta ruido
  │    Evaluar robustez con sweep de niveles de ruido
  │    Fine-tune con datos reales (si hay logging del robot)
```

**¿Es viable simular ruidos?** Sí, pero es complejo hacerlo bien. El riesgo es que el ruido simulado no represente la distribución real del robot, generando una policy que sea robusta al ruido *equivocado*. Por eso se deja para después de validar que el RL funciona en condiciones ideales.

---

## Fase 1 — RL solo define POSICIONES

### Qué se reemplaza

| Componente actual | Archivo | Líneas clave | Reemplazado por |
|---|---|---|---|
| Home poses estáticas | `Config/Behavior/Tactics/t*.cfg` | `positions = [...]` | Output de la policy MAPPO |
| `assignPositions()` cost matrix | [Behavior.cpp L497-L735](Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp#L497-L735) | Permutación exhaustiva euclidea | Asignación directa por la policy |
| Gradient ascent en RatingRole | [RatingRole.cpp L27](Src/Modules/BehaviorControl/StrategyBehaviorControl/PositionRoles/RatingRole.cpp#L27) | `position()` con rating iterativo | Posición ya óptima de la policy |

### Qué se mantiene intacto

- Toda la FSM de estrategia/táctica (`s4v4_scaled`: t121_4v4_scaled ↔ t13_4v4_scaled)
- `assignRoles()` con TTRB → determineActiveAgent()
- Todos los skills: KickAtGoal, PassToTeammate, DribbleToGoal, Zweikampf
- PathPlanner, FieldRating, ExpectedGoals, PassEvaluation
- GoaliePose (el portero NO entra al RL en esta fase)

### Action Space — Fase 1

```python
# Por cada field player (3 agentes en 4v4, excluir portero):
# El action se traduce a un Pose2f walkTarget que recibe el WalkToPose skill.
#   Skill: WalkToPose(target, speed) — SkillInterfaces.h L41
#          Src/Representations/BehaviorControl/SkillInterfaces.h#L41
#   Target final: MotionRequest.walkTarget (Pose2f) — MotionRequest.h L97
#          Src/Representations/MotionControl/MotionRequest.h#L97
#   Pose2f = { rotation: Angle (L90), translation: Vector2f (L91) }
#          Src/Libs/Math/Pose2f.h#L90-L91

# Límites del campo 4v4 Scaled (Config/Locations/4v4_Scaled/fieldDimensions.cfg):
#   xPosOwnGroundLine = -2813, xPosOpponentGroundLine = 2813
#   yPosRightSideline = -1875, yPosLeftSideline = 1875
action = {
    "target_x": continuous(-2813, 2813),   # mm — walkTarget.translation.x()
    "target_y": continuous(-1875, 1875),   # mm — walkTarget.translation.y()
    "target_theta": continuous(-pi, pi),   # rad — walkTarget.rotation
}
# Dimensión total: 3 continuos × 3 agentes = 9
```

> **¿Por qué θ desde Fase 1?** La orientación es crítica. Un defensor mirando hacia su propio arco no puede interceptar un pase. Un forward de espaldas al arco rival no puede rematar. En BHuman, las home poses en los .cfg ya incluyen `rotation` (ej: `rotation = 0deg`). El PathPlanner recibe un `Pose2f walkTarget` que incluye rotación — no solo posición. Si el RL solo da (x,y), el robot llegaría al punto sin saber hacia dónde mirar, y el PathPlanner por defecto caminaría con orientación arbitraria. Incluir θ permite que la policy aprenda comportamientos como "mirar al balón", "mirar al oponente más cercano", o "orientarse para recibir pase".

### Observation Space — Fase 1

```python
obs_per_agent = {
    # ═════ SELF (12 floats) ═════
    # Fuente: RobotPose (hereda Pose2f)
    #   Struct: Src/Representations/Modeling/RobotPose.h (hereda Pose2f)
    #   Pose2f.rotation  — Src/Libs/Math/Pose2f.h#L88
    #   Pose2f.translation — Src/Libs/Math/Pose2f.h#L89
    "self_x": float,              # theRobotPose.translation.x() — mm, campo
    "self_y": float,              # theRobotPose.translation.y() — mm, campo
    "self_theta": float,          # theRobotPose.rotation — rad

    # Fuente: MotionInfo.speed (Pose2f)
    #   Declaración: Src/Representations/MotionControl/MotionInfo.h#L49
    #   "(Pose2f) speed, /**< The current speed at which the robot is walking */"
    "self_vx": float,             # theMotionInfo.speed.translation.x() — mm/s
    "self_vy": float,             # theMotionInfo.speed.translation.y() — mm/s
    "self_omega": float,          # theMotionInfo.speed.rotation — rad/s

    # Fuente: FallDownState.state
    #   Declaración: Src/Representations/Sensing/FallDownState.h#L43
    #   Enum State (upright, fallen, ...): Src/Representations/Sensing/FallDownState.h#L21
    "self_is_upright": bool,      # theFallDownState.state == FallDownState::upright

    # Fuente: SystemSensorData.batteryLevel
    #   Declaración: Src/Representations/Infrastructure/SensorData/SystemSensorData.h#L12
    #   "(float)(SensorData::off) batteryLevel, /**< ... (in %). Range: [0.0, 1.0] */"
    "self_battery": float,        # theSystemSensorData.batteryLevel — [0.0, 1.0]

    # Fuente: StrategyStatus.position (Tactic::Position::Type)
    #   Declaración campo: Src/Representations/BehaviorControl/StrategyStatus.h#L30
    #   Enum Position::Type: Src/Tools/BehaviorControl/Strategy/Tactic.h (struct Position anidado)
    #   Valores usados en 4v4 Scaled (verificados en t121_4v4_scaled.cfg y t13_4v4_scaled.cfg):
    #     0=goalkeeper, 1=defenderL, 2=defenderR, 3=forwardL, 4=forwardR
    #   NOTA: "striker" NO es un Position::Type — es un ActiveRole (playBall). Ver self_is_active.
    "self_position": one_hot(5),  # one_hot de StrategyStatus.position (5 posiciones 4v4)

    # Fuente: StrategyStatus.role (Role::Type) + Role::isActiveRole()
    #   Método: Src/Tools/BehaviorControl/Strategy/Role.h#L38 (isActiveRole)
    #   Campo role: Src/Representations/BehaviorControl/StrategyStatus.h#L31
    #   ActiveRole playBall: Src/Tools/BehaviorControl/Strategy/ActiveRole.h#L20
    #   → true si Role::isActiveRole(theStrategyStatus.role) [es el striker/playBall]
    "self_is_active": bool,       # true si este agente es el striker activo

    # Derivado: ángulo de apertura al arco rival
    #   Postes del arco 4v4 Scaled:
    #     xPosOpponentGoalPost: Src/Representations/Configuration/FieldDimensions.h#L24 (=2838mm)
    #     yPosLeftGoal:  Src/Representations/Configuration/FieldDimensions.h#L42 (=800mm)
    #     yPosRightGoal: Src/Representations/Configuration/FieldDimensions.h#L43 (=-800mm)
    #   Cálculo: atan2(800 - y, 2838 - x) - atan2(-800 - y, 2838 - x)
    #   Referencia BHuman: ExpectedGoalsProvider.cpp L64-L68
    #     Src/Modules/BehaviorControl/ExpectedGoalsProvider/ExpectedGoalsProvider.cpp#L64
    "goal_opening_angle": float,  # rad — apertura angular al arco rival

    # Derivado: ángulo del centro del arco respecto a self_theta
    #   Cálculo: normalize(atan2(0 - y, 2813 - x) - self_theta)  [xPosOpponentGroundLine=2813]
    "goal_relative_angle": float, # rad — [-pi, pi]

    # ═════ BALL (6 floats) ═════
    # Fuente posición: FieldBall.positionOnField (Vector2f)
    #   Declaración: Src/Representations/BehaviorControl/FieldBall.h#L104
    #   "(Vector2f)(Vector2f::Zero()) positionOnField, /**< ball position in global field coordinates */"
    "ball_x": float,              # theFieldBall.positionOnField.x() — mm
    "ball_y": float,              # theFieldBall.positionOnField.y() — mm

    # Fuente velocidad: BallModel.estimate.velocity (Vector2f, robot-relative mm/s)
    #   BallState.velocity: Src/Representations/Modeling/BallModel.h#L23
    #   BallModel.estimate: Src/Representations/Modeling/BallModel.h#L38
    #   NOTA: velocity es en coordenadas relativas al robot. Para obtener en campo:
    #     ball_vel_field = theRobotPose.rotation.rotate(theBallModel.estimate.velocity)
    "ball_vx": float,             # velocidad campo mm/s (rotada por RobotPose)
    "ball_vy": float,             # velocidad campo mm/s (rotada por RobotPose)

    # Fuente: FieldBall.timeSinceBallWasSeen (int, ms)
    #   Declaración: Src/Representations/BehaviorControl/FieldBall.h#L116
    #   "(int)(0) timeSinceBallWasSeen,"
    #   Helper: ballWasSeen(X) es solo timeSinceBallWasSeen <= X
    "ball_age_ms": float,         # ms — continuo, no booleano

    # Fuente: FieldBall.useTeammatesBall() (método bool)
    #   Declaración: Src/Representations/BehaviorControl/FieldBall.h#L99
    #   Implementación: Src/Representations/BehaviorControl/FieldBall.cpp#L93
    #     return (timeSinceBallWasSeen > ballSeenTimeout      // default 1000ms
    #             || timeSinceBallDisappeared > ballDisappearedTimeout)  // default 100ms
    #            && teammatesBallIsValid && teammatesBallNewerThanOwnBall;
    "using_team_ball": bool,      # true = usando posición de equipo, no propia

    # ═════ TEAMMATES — 3 slots exactos en 4v4 (7 × 3 = 21 floats) ═════
    # En 4v4 hay exactamente 3 compañeros (incl. portero) → no hay padding.
    # Fuente: GlobalTeammatesModel.TeammateEstimate
    #   Struct: Src/Representations/Modeling/GlobalTeammatesModel.h#L30-L40
    #   Lista: Src/Representations/Modeling/GlobalTeammatesModel.h#L49
    #     "(std::vector<TeammateEstimate>) teammates,"

    # pose (Pose2f): Src/Representations/Modeling/GlobalTeammatesModel.h#L36
    #   NOTA: pose.rotation es "currently unreliable" según el header. NO se usa.
    "tm_x": float,               # .pose.translation.x() — mm, campo
    "tm_y": float,               # .pose.translation.y() — mm, campo

    # relativeWalkTarget (Vector2f): Src/Representations/Modeling/GlobalTeammatesModel.h#L38
    #   Origen: BehaviorStatus.walkingTo — Src/Representations/BehaviorControl/BehaviorStatus.h#L24
    #   Publicado por PublishMotion skill — Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/Output/BehaviorStatus/PublishMotion.cpp#L28
    #   Extrapolación: Teammate::getEstimatedPosition() — Src/Representations/Communication/TeamData.cpp#L20-L28
    "tm_walk_dx": float,         # .relativeWalkTarget.x() — robot-relative mm
    "tm_walk_dy": float,         # .relativeWalkTarget.y() — robot-relative mm

    # speed (float): Src/Representations/Modeling/GlobalTeammatesModel.h#L39
    #   Origen: BehaviorStatus.speed — Src/Representations/BehaviorControl/BehaviorStatus.h#L25
    #   Cálculo: speedRatio.x * maxWalkSpeed.x — PublishMotion.cpp#L31
    "tm_speed": float,           # mm/s, escalar (solo magnitud forward)

    # Striker: Role::isActiveRole(teammate.role)
    #   FUENTE EN SIM:   estado interno del agente simulado
    #   FUENTE EN C++:   TeamData.Teammate.theStrategyStatus.role
    #     TeamData.h: Src/Representations/Communication/TeamData.h (struct Teammate L32)
    #     theStrategyStatus.role: Src/Representations/BehaviorControl/StrategyStatus.h#L31
    #   Método check: Src/Tools/BehaviorControl/Strategy/Role.h#L38 (isActiveRole)
    #   ActiveRole playBall: Src/Tools/BehaviorControl/Strategy/ActiveRole.h#L20
    #   ADVERTENCIA: GlobalTeammatesModel.TeammateEstimate NO tiene campo role.
    #                En deployment usar TeamData, NO GlobalTeammatesModel.
    "tm_is_active": bool,        # true si el teammate es striker/playBall

    "tm_exists": bool,           # slot ocupado? (siempre true en 4v4 si el robot está activo)

    # ═════ OPPONENTS — 4 slots (3 × 4 = 12 floats) ═════
    # En 4v4 hay exactamente 4 oponentes (1 GK + 3 field) → 4 slots.
    # Fuente: GlobalOpponentsModel.OpponentEstimate
    #   Struct: Src/Representations/Modeling/GlobalOpponentsModel.h#L30-L35
    #   Lista: Src/Representations/Modeling/GlobalOpponentsModel.h#L42
    #     "(std::vector<OpponentEstimate>) opponents,"
    #
    # LOCAL y dependiente de FOV (como GlobalOpponentsTracker real).
    # Desde Fase 1 se simula la limitación: cada agente solo ve oponentes
    # dentro de su FOV (~55° horizontal) y hasta ~5000mm de distancia.
    #   FOV: CameraIntrinsics.openingAngleWidth = 54.7deg
    #     Src/Representations/Configuration/CameraIntrinsics.h#L26
    #     Config/Robots/Default/cameraIntrinsics.cfg
    #   maxDistance: GlobalOpponentsTracker param
    #     Src/Modules/Modeling/GlobalOpponentsTracker/GlobalOpponentsTracker.h#L57
    #   deleteAfter (persistencia EKF ~1000ms):
    #     Src/Modules/Modeling/GlobalOpponentsTracker/GlobalOpponentsTracker.h#L62
    # Conversión local→global: theRobotPose * ob.center
    #   GlobalOpponentsTracker.cpp L75-L77

    # position (Vector2f): Src/Representations/Modeling/GlobalOpponentsModel.h#L33
    "opp_x": float,              # .position.x() — mm, campo (convertida vía RobotPose)
    "opp_y": float,              # .position.y() — mm, campo
    "opp_exists": bool,          # slot ocupado?

    # ═════ GAME STATE (5 floats) ═════

    # Fuente: GameState.state == GameState::playing
    #   Campo state: Src/Representations/Infrastructure/GameState.h#L405
    #   Enum playing: Src/Representations/Infrastructure/GameState.h#L25
    #   Helper: GameState::isPlaying() — Src/Representations/Infrastructure/GameState.h#L191
    "game_is_playing": bool,

    # Fuente: GameState.ownTeam.score / GameState.opponentTeam.score
    #   Team.score: Src/Representations/Infrastructure/GameState.h#L120
    #     "(unsigned int)(0u) score, /**< The current score. */"
    #   ownTeam: Src/Representations/Infrastructure/GameState.h#L409
    #   opponentTeam: Src/Representations/Infrastructure/GameState.h#L410
    "own_score": int,             # theGameState.ownTeam.score
    "opp_score": int,             # theGameState.opponentTeam.score

    # Fuente: (GameState.timeWhenPhaseEnds - FrameInfo.time) / 1000
    #   timeWhenPhaseEnds: Src/Representations/Infrastructure/GameState.h#L404
    #     "(unsigned)(0) timeWhenPhaseEnds, /**< Time when the current phase is expected to end. */"
    "half_time_remaining_s": float,

    # Fuente: StrategyStatus.acceptedTactic (Tactic::Type)
    #   Declaración: Src/Representations/BehaviorControl/StrategyStatus.h#L24
    #   Enum Tactic::Type: Src/Tools/BehaviorControl/Strategy/Tactic.h (enum Type anidado)
    #   Valores usados en 4v4 Scaled (s4v4_scaled.cfg):
    #     0 = t121_4v4_scaled  (defensiva — estado inicial)
    #     1 = t13_4v4_scaled   (ofensiva)
    #   NOTA: Tactic::Type tiene 17 valores en total (incluyendo variantes scaled de otras
    #         categorías). Para 4v4 solo se usan 2. Codificar como índice local, no enum global.
    "current_tactic": one_hot(2),
}
# Total por agente: 16 (self) + 6 (ball) + 21 (tm×3) + 12 (opp×4) + 4 (game) + 2 (tactic) = 61 floats
```

### Reward — Fase 1

```python
reward = (
    + 10.0  * goal_scored
    - 10.0  * goal_conceded
)
# Reward SPARSE y COMPARTIDO (team reward) — MAPPO con shared reward.
# Sin reward shaping por decisión del revisor.
# La señal es solo goles a favor y goles en contra.
# Esto evita sesgos del diseñador y fuerza a la policy a descubrir
# por sí misma qué comportamientos conducen a goles.
```

### Paso a Paso — Fase 1

```
1. Construir entorno sim2D en Python
   ├── Parsear .cfg de tácticas → home poses iniciales (ver Sección 6)
   ├── Parsear .cfg de estrategias → FSM de transiciones
   ├── Implementar NaoMovementModel (ver P6) con velocidades realistas:
   │   ├── forward=250mm/s, sideways=200mm/s, backwards=150mm/s, rotation=120°/s
   │   ├── Aceleración gradual (no teletransporte)
   │   └── Collision avoidance básico (repulsión a 300mm)
   ├── Implementar lógica de skills simplificada:
   │   ├── Striker: si cerca del balón → dispara/pasa (basado en goal_opening_angle)
   │   ├── Otros: caminan a su target position (output del RL)
   │   └── Portero: GoaliePose heurística (no RL)
   ├── Oponentes: LOCAL y FOV-dependiente (ver P4):
   │   ├── Cono de visión ±27.5° desde self_theta
   │   ├── Distancia máxima 5000mm
   │   ├── Detección probabilística (90%)
   │   └── Persistencia temporal ~1000ms (simula EKF prediction)
   └── Implementar game state: kickoff, playing, goal, halftime

2. Implementar MAPPO
   ├── 3 actores (field players 4v4, excluir portero) con políticas compartidas (parameter sharing)
   ├── 1 critic centralizado con obs global (todas las poses + balón real)
   ├── Action space: 3 continuos por agente (target_x, target_y, target_theta)
   └── Obs space: 61 floats por agente

3. Entrenar
   ├── Curriculum: primero vs equipo estático → luego vs self-play
   ├── Episodios: 10 minutos de juego simulado
   ├── Normalización: obs normalizado [-1, 1], actions escaladas a campo
   └── Checkpoints: guardar cada 1M steps

4. Evaluar
   ├── Métricas: win rate, goals scored, ball possession %, field coverage
   ├── Comparar vs baseline: BHuman heurístico puro
   ├── Visualizar: heatmaps de posiciones aprendidas vs home poses estáticas
   └── Ablation: quitar features del obs → ver cuáles importan

5. Exportar policy → C++ (futuro, no en Fase 1)
   └── Convertir PyTorch → ONNX → CompiledNN (BHuman ya lo soporta)
```

---

## Fase 2 — RL define POSICIÓN + ROL

### Qué se reemplaza adicionalmente (sobre Fase 1)

| Componente | Archivo | Reemplazado por |
|---|---|---|
| `assignRoles()` | [Behavior.cpp L740-L770](Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp#L740-L770) | Output discreto de la policy |
| `determineActiveAgent()` + TTRB | [Behavior.cpp L911-L1140](Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp#L911-L1140) | La policy decide quién es striker |
| `PositionRole::fromPosition()` mapping fijo | [PositionRole.cpp L16-L38](Src/Tools/BehaviorControl/Strategy/PositionRole.cpp#L16-L38) | Rol directo del RL |

### Qué se mantiene

- FSM de táctica (transiciones entre formaciones)
- Skills de ejecución (KickAtGoal, Pass, Dribble, Walk, Zweikampf)
- PathPlanner, FieldRating, ExpectedGoals

### Action Space — Fase 2

```python
action = {
    "target_x": continuous(-2813, 2813),   # mm — campo 4v4 Scaled
    "target_y": continuous(-1875, 1875),   # mm — campo 4v4 Scaled
    "target_theta": continuous(-pi, pi),
    "role": discrete(3),   # defender / forward / striker  (sin midfielder en 4v4)
}
# Dimensión: 3 continuos + 1 discreto × 3 agentes
```

**Constraint crítico:** Máximo 1 striker en el equipo en todo momento. Implementar como:
- Softmax sobre los 3 logits de "striker" → el de mayor prob gana
- Penalty en reward si >1 agente elige striker simultáneamente

### Paso a Paso — Fase 2

```
1. Partir del entorno de Fase 1
   ├── Añadir roles como acción discreta
   ├── Implementar lógica de skills por rol:
   │   ├── Defender: BOb line (simplificada) + block
   │   ├── Midfielder: apoyo + pase
   │   ├── Forward: posicionamiento ofensivo + recibir pase
   │   ├── Striker: PlayBall (shoot/pass/dribble simplificado)
   │   └── Goalkeeper: GoaliePose (no RL, mismo que Fase 1)
   └── Constraint de unicidad de striker en el env

2. Modificar MAPPO
   ├── Action space híbrido: 2 continuos + 1 discreto (4 categorías)
   ├── Policy con doble cabeza: continuous head + discrete head
   └── Critic: misma estructura que Fase 1 + info de roles

3. Entrenar con curriculum
   ├── Pre-train: inicializar posiciones desde policy de Fase 1
   ├── Fase 2a: solo roles variables, posiciones fijas de Fase 1
   ├── Fase 2b: roles + posiciones variables
   └── Fase 2c: self-play competitivo

4. Evaluar
   ├── Comparar vs Fase 1 (RL posiciones + roles heurísticos)
   ├── Analizar: ¿el RL encuentra asignaciones de striker mejores que TTRB?
   ├── Metric nueva: role_switch_frequency (menos cambios = más estable)
   └── Buscar emergent behaviors: ¿rotaciones? ¿cobertura adaptiva?
```

---

## Fase 3 — RL suplanta TODA táctica y estrategia

### Qué se reemplaza (todo StrategyBehaviorControl)

| Componente | Archivo | Eliminado |
|---|---|---|
| FSM de tácticas | [Behavior.cpp L449-L487](Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp#L449-L487) | RL decide formación implícita |
| `assignPositions()` | [Behavior.cpp L497-L735](Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp#L497-L735) | Policy directa |
| `assignRoles()` | [Behavior.cpp L740-L770](Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp#L740-L770) | Policy directa |
| `determineActiveAgent()` | [Behavior.cpp L911-L1140](Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp#L911-L1140) | Policy directa |
| SmashOrPass | [PlayBall.cpp L38-L109](Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp#L38-L109) | Policy decide shoot/pass/dribble |
| Forward/Midfielder/Defender positioning | Toda la carpeta `PositionRoles/` | Policy directa |

### Qué se mantiene

- `SkillBehaviorControl` (ejecución motora de skills)
- PathPlanner (navegación local con obstáculos)
- Percepciones y modelos de mundo

### Action Space — Fase 3

```python
action = {
    "target_x": continuous(-2813, 2813),   # mm — campo 4v4 Scaled
    "target_y": continuous(-1875, 1875),   # mm — campo 4v4 Scaled
    "target_theta": continuous(-pi, pi),
    "skill": discrete(6),    # stand / walk / shoot / pass / dribble / block
    "pass_target": discrete(3),  # jersey index del receptor (3 compañeros en 4v4)
}
# Dimensión: 3 continuos + 2 discretos × 3 agentes
```

### Paso a Paso — Fase 3

```
1. Entorno completo
   ├── Implementar TODOS los skills como acciones del env
   │   ├── shoot: disparo con ángulo basado en target_theta
   │   ├── pass: pase a pass_target con rating de éxito basado en opps
   │   ├── dribble: avance con balón (velocidad reducida)
   │   ├── block: bloqueo defensivo (se posiciona entre opp y gol propio)
   │   ├── walk: caminar al target sin interactuar con balón
   │   └── stand: quedarse quieto (ahorro energía)
   ├── Eliminar FSM de tácticas del env
   └── Portero: considerar incluir en RL o mantener heurístico

2. MAPPO full
   ├── Action space: 3 continuos + 2 discretos
   ├── Multi-head policy: position head + skill head + pass head
   ├── Masking: pass_target solo activo si skill==pass
   └── Considerar population-based training (PBT) para hiperparámetros

3. Curriculum desde Fase 2
   ├── Inicializar policy de posiciones desde Fase 2
   ├── Fase 3a: skill fixed (solo walk) → aprende posiciones puras
   ├── Fase 3b: unlock shoot/pass para el striker
   ├── Fase 3c: unlock todos los skills
   └── Fase 3d: self-play completo con evaluación ELO

4. Evaluar
   ├── Win rate vs heurístico puro, vs Fase 1, vs Fase 2
   ├── Análisis táctico: ¿emergen formaciones? ¿pases triangulares?
   ├── Análisis de fallo: ¿cuándo la policy colapsa?
   └── Test de transferencia: ¿funciona contra equipos nuevos?
```

---

## Comunicación .cfg ↔ Python

### ¿Por qué parsear los .cfg y no clonarlos a mano?

**Se parsean, no se clonan.** Razones:

1. **Los .cfg son la fuente de verdad.** Si alguien modifica `t121.cfg` para mover un defensor 200mm a la izquierda, el parser lo refleja automáticamente. Un clone manual se desincroniza.
2. **Expresiones simbólicas.** Las poses usan expresiones como `xPosOwnPenaltyArea + 500` que dependen de `fieldDimensions.cfg`. Si se cambia el tamaño del campo (ej: campo reducido para pruebas), todas las poses cambian. Un clone tendría que recalcular a mano.
3. **Son pocos archivos** (4 estrategias + 8 tácticas + 1 fieldDimensions = 13 archivos). El parser es ~100 líneas de Python. El esfuerzo es mínimo.
4. **Reutilización.** El parser sirve para cargar home poses, FSM de transiciones, set plays, y cualquier .cfg futuro (ej: kick engine configs).
5. **Estructura limpia.** Clonar a mano implicaría crear diccionarios hardcodeados con 8×5 = 40 poses + 4 FSMs. El parser los genera automáticamente y es más mantenible.

Si por alguna razón el parser falla en un .cfg específico, siempre se puede tener un fallback con los valores hardcodeados (ya los tenemos documentados en este archivo).

### Formato de los .cfg

BHuman usa un formato propietario llamado **InExprMapFile**: key-value con expresiones aritméticas, enums sin comillas, sufijo `deg` para ángulos, y variables simbólicas de `FieldDimensions`.

Ejemplo de un tactic .cfg ([t121.cfg](Config/Behavior/Tactics/t121.cfg)):
```
positions = [
  { type = goalkeeper; pose = { rotation = 0deg; translation = { x = xPosOwnGoalArea - 150; y = 0; }; }; },
  { type = defenderL;  pose = { rotation = 0deg; translation = { x = xPosOwnPenaltyArea + 500; y = 0; }; }; },
  { type = defenderR;  pose = { rotation = 0deg; translation = { x = -375; y = -1000; }; }; },
  { type = forwardL;   pose = { rotation = 0deg; translation = { x = -375; y = 1000; }; }; },
  { type = forwardR;   pose = { rotation = 0deg; translation = { x = 1125; y = 0; }; }; },
];
```

Ejemplo de un strategy .cfg ([s5v5.cfg](Config/Behavior/Strategies/s5v5.cfg)):
```
tactics = [
  { tactic = t4;   transitions = [{ to = t121; conditions = [{ numOfFieldPlayersGE = 4; ballXThreshold = 1000; timeSinceBallAheadOfThresholdGE = 300; }]; }]; },
  { tactic = t121; transitions = [{ to = t4;   conditions = [{ numOfFieldPlayersLE = 4; }, { numOfFieldPlayersGE = 4; ballXThreshold = -1000; timeSinceBallBehindThresholdGE = 300; }]; }]; }
];
```

### Tabla de símbolos (FieldDimensions)

La tabla de símbolos necesaria para evaluar expresiones.
**Fuente: [fieldDimensions.cfg](Config/Locations/4v4_Scaled/fieldDimensions.cfg)** — campo 4v4 Scaled (5626 × 3750 mm).

```python
FIELD_SYMBOLS = {
    # ── Eje X (largo del campo: 5626mm total) ──
    "xPosOpponentFieldBorder":  3313,
    "xPosOpponentGoal":         3368,
    "xPosOpponentGoalPost":     2838,
    "xPosOpponentGroundLine":   2813,
    "xPosOpponentGoalArea":     2438,
    "xPosOpponentPenaltyMark":  2000,
    "xPosOpponentPenaltyArea":  1781,
    "xPosPenaltyStrikerStartPosition": 1781,  # = xPosOpponentPenaltyArea
    "xPosHalfWayLine":          0,
    "xPosOwnPenaltyArea":      -1781,
    "xPosOwnPenaltyMark":      -2000,
    "xPosOwnGoalArea":         -2438,
    "xPosOwnGroundLine":       -2813,
    "xPosOwnGoalPost":         -2838,
    "xPosOwnGoal":             -3368,
    "xPosOwnFieldBorder":      -3313,
    # ── Eje Y (ancho del campo: 3750mm total) ──
    "yPosLeftFieldBorder":      1960,
    "yPosLeftSideline":         1875,
    "yPosLeftPenaltyArea":      1250,
    "yPosLeftGoalArea":          688,
    "yPosLeftGoal":              800,   # postes del arco (mismo que 5v5)
    "yPosRightGoal":            -800,
    "yPosRightGoalArea":        -688,
    "yPosRightPenaltyArea":    -1250,
    "yPosRightSideline":       -1875,
    "yPosRightFieldBorder":    -1960,
    # ── Otros ──
    "centerCircleRadius":        469,   # más pequeño que 5v5 (750mm)
    "fieldLinesWidth":            50,
    "goalPostRadius":             50,
    "penaltyMarkSize":           100,
}
```

### Parser Python para .cfg

```python
"""
bhuman_cfg_parser.py — Parser de archivos .cfg de BHuman (InExprMapFile format)
Soporta: expresiones aritméticas, símbolos de FieldDimensions, arrays, structs anidados, deg suffix
"""

import re
import math
from typing import Any


class BHumanCfgParser:
    """Parser para el formato InExprMapFile de BHuman."""

    def __init__(self, symbols: dict[str, float] = None):
        self.symbols = symbols or FIELD_SYMBOLS
        self.pos = 0
        self.text = ""

    def parse_file(self, filepath: str) -> dict:
        with open(filepath, "r") as f:
            self.text = f.read()
        self.pos = 0
        return self._parse_map()

    def _skip_whitespace_and_comments(self):
        while self.pos < len(self.text):
            if self.text[self.pos:self.pos+2] == "//":
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self.pos += 1
            elif self.text[self.pos:self.pos+2] == "/*":
                end = self.text.find("*/", self.pos)
                self.pos = end + 2 if end != -1 else len(self.text)
            elif self.text[self.pos] in " \t\r\n":
                self.pos += 1
            else:
                break

    def _parse_map(self) -> dict:
        result = {}
        while self.pos < len(self.text):
            self._skip_whitespace_and_comments()
            if self.pos >= len(self.text) or self.text[self.pos] in "}]":
                break
            key = self._parse_identifier()
            self._skip_whitespace_and_comments()
            assert self.text[self.pos] == "=", f"Expected '=' after {key}, got '{self.text[self.pos]}'"
            self.pos += 1  # skip '='
            self._skip_whitespace_and_comments()
            value = self._parse_value()
            result[key] = value
            self._skip_whitespace_and_comments()
            if self.pos < len(self.text) and self.text[self.pos] == ";":
                self.pos += 1
        return result

    def _parse_identifier(self) -> str:
        self._skip_whitespace_and_comments()
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
            self.pos += 1
        return self.text[start:self.pos]

    def _parse_value(self) -> Any:
        self._skip_whitespace_and_comments()
        ch = self.text[self.pos]
        if ch == "[":
            return self._parse_array()
        elif ch == "{":
            return self._parse_struct()
        else:
            return self._parse_expression()

    def _parse_array(self) -> list:
        assert self.text[self.pos] == "["
        self.pos += 1
        result = []
        while True:
            self._skip_whitespace_and_comments()
            if self.text[self.pos] == "]":
                self.pos += 1
                return result
            value = self._parse_value()
            result.append(value)
            self._skip_whitespace_and_comments()
            if self.text[self.pos] == ",":
                self.pos += 1

    def _parse_struct(self) -> dict:
        assert self.text[self.pos] == "{"
        self.pos += 1
        result = self._parse_map()
        self._skip_whitespace_and_comments()
        assert self.text[self.pos] == "}", f"Expected '}}', got '{self.text[self.pos]}'"
        self.pos += 1
        return result

    def _parse_expression(self) -> Any:
        """Parse an expression that may be numeric, a symbol, or arithmetic."""
        self._skip_whitespace_and_comments()
        # Try to parse as arithmetic expression
        expr_start = self.pos
        # Collect tokens until we hit ; or , or } or ]
        depth = 0
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch in ";," and depth == 0:
                break
            self.pos += 1
        expr = self.text[expr_start:self.pos].strip()

        # Handle deg suffix: "5deg" → 5 * pi/180
        if expr.endswith("deg"):
            num_part = expr[:-3].strip()
            return self._eval_expr(num_part) * math.pi / 180.0

        # Try numeric evaluation
        return self._eval_expr(expr)

    def _eval_expr(self, expr: str) -> Any:
        """Evaluate a BHuman expression with symbols."""
        expr = expr.strip()
        if not expr:
            return 0.0

        # Replace symbols with values
        resolved = expr
        for name, value in sorted(self.symbols.items(), key=lambda x: -len(x[0])):
            resolved = resolved.replace(name, str(value))

        # Try numeric eval (safe: only arithmetic)
        try:
            return eval(resolved, {"__builtins__": {}}, {})
        except:
            # Return as string (enum identifier like "t4", "goalkeeper", etc.)
            return expr


# ─── Convenience functions ────────────────────────────────────────

def load_tactic(filepath: str) -> dict:
    """
    Load a tactic .cfg file → dict with 'positions' and 'priorityGroups'.
    
    Returns:
        {
            "positions": [
                {"type": "goalkeeper", "pose": {"rotation": 0.0, "translation": {"x": -4050, "y": 0}}},
                {"type": "defenderL", "pose": {"rotation": 0.0, "translation": {"x": -2350, "y": 0}}},
                ...
            ],
            "priorityGroups": [...]
        }
    """
    parser = BHumanCfgParser()
    return parser.parse_file(filepath)


def load_strategy(filepath: str) -> dict:
    """
    Load a strategy .cfg file → dict with 'tactics' (FSM) and set plays.
    
    Returns:
        {
            "tactics": [
                {
                    "tactic": "t4",
                    "transitions": [{"to": "t121", "conditions": [{...}]}]
                },
                ...
            ],
            "ownKickOffs": [...],
            ...
        }
    """
    parser = BHumanCfgParser()
    return parser.parse_file(filepath)


def extract_home_poses(tactic: dict) -> dict[str, tuple[float, float, float]]:
    """
    Extract home poses from a loaded tactic → {position_type: (x, y, theta)}.
    
    Returns:
        {
            "goalkeeper": (-4050, 0, 0.0),
            "defenderL": (-2350, 0, 0.0),
            ...
        }
    """
    poses = {}
    for pos in tactic["positions"]:
        ptype = pos["type"]
        pose = pos["pose"]
        x = pose["translation"]["x"]
        y = pose["translation"]["y"]
        theta = pose.get("rotation", 0.0)
        poses[ptype] = (float(x), float(y), float(theta))
    return poses


def build_fsm_from_strategy(strategy: dict) -> dict:
    """
    Build a tactic FSM from a loaded strategy.
    
    Returns:
        {
            "states": {
                "t4": {
                    "transitions": [
                        {"to": "t121", "conditions": [{"ballXThreshold": 1000, ...}]}
                    ]
                },
                "t121": {
                    "transitions": [{"to": "t4", "conditions": [...]}]
                }
            },
            "initial": "t4"
        }
    """
    states = {}
    initial = None
    for tstate in strategy["tactics"]:
        name = tstate["tactic"]
        if initial is None:
            initial = name
        states[name] = {"transitions": tstate.get("transitions", [])}
    return {"states": states, "initial": initial}
```

### Uso del Parser en el entorno RL

```python
import os
from bhuman_cfg_parser import load_tactic, load_strategy, extract_home_poses, build_fsm_from_strategy

CONFIG_ROOT = "Config/Behavior"
FIELD_CFG   = "Config/Locations/4v4_Scaled/fieldDimensions.cfg"  # fuente de FIELD_SYMBOLS

# ─── Cargar tácticas 4v4 Scaled ───
# Tácticas usadas en s4v4_scaled.cfg (verificado en Config/Behavior/Strategies/s4v4_scaled.cfg)
TACTICS = {}
for name in ["t121_4v4_scaled", "t13_4v4_scaled"]:
    tactic = load_tactic(os.path.join(CONFIG_ROOT, "Tactics", f"{name}.cfg"))
    TACTICS[name] = extract_home_poses(tactic)

# ─── Cargar estrategia 4v4 Scaled ───
STRATEGIES = {}
strategy = load_strategy(os.path.join(CONFIG_ROOT, "Strategies", "s4v4_scaled.cfg"))
STRATEGIES["s4v4_scaled"] = build_fsm_from_strategy(strategy)

# Mapeo de táctica → índice para el one_hot(2) del obs space:
TACTIC_INDEX = {
    "t121_4v4_scaled": 0,   # defensiva (estado inicial)
    "t13_4v4_scaled":  1,   # ofensiva
}

# ─── FSM runner (para el env de Fase 1) ───
class TacticFSM:
    """Ejecuta la FSM de tácticas como lo hace Behavior.cpp."""

    def __init__(self, strategy_name: str = "s4v4_scaled"):
        self.fsm = STRATEGIES[strategy_name]
        self.current = self.fsm["initial"]
    
    def step(self, ball_x: float, num_field_players: int, 
             time_ball_ahead_ms: int, time_ball_behind_ms: int) -> str:
        """Evalúa transiciones y retorna nombre de la táctica actual."""
        state = self.fsm["states"][self.current]
        for trans in state["transitions"]:
            for cond in trans["conditions"]:
                if self._check_condition(cond, ball_x, num_field_players,
                                         time_ball_ahead_ms, time_ball_behind_ms):
                    self.current = trans["to"]
                    return self.current
        return self.current
    
    def _check_condition(self, cond: dict, ball_x: float, num_fp: int,
                         time_ahead: int, time_behind: int) -> bool:
        """Evalúa UNA condición (OR entre condiciones en el .cfg)."""
        checks = []
        if "numOfFieldPlayersGE" in cond:
            checks.append(num_fp >= cond["numOfFieldPlayersGE"])
        if "numOfFieldPlayersLE" in cond:
            checks.append(num_fp <= cond["numOfFieldPlayersLE"])
        if "ballXThreshold" in cond:
            threshold = cond["ballXThreshold"]
            if "timeSinceBallAheadOfThresholdGE" in cond:
                checks.append(ball_x > threshold and time_ahead >= cond["timeSinceBallAheadOfThresholdGE"])
            if "timeSinceBallBehindThresholdGE" in cond:
                checks.append(ball_x < threshold and time_behind >= cond["timeSinceBallBehindThresholdGE"])
        if "ownScoreGE" in cond or "opponentScoreGE" in cond:
            pass  # Extender según necesidad
        return all(checks) if checks else False
    
    def get_home_poses(self) -> dict[str, tuple[float, float, float]]:
        """Retorna las home poses de la táctica actual."""
        return TACTICS[self.current]
```

### Arquitectura de Comunicación Completa

```
┌──────────────────────────────────────────────────────┐
│                    PYTHON (Training)                  │
│                                                      │
│  bhuman_cfg_parser.py                                │
│  ├── Parsea Config/Behavior/Strategies/s4v4_scaled.cfg │
│  ├── Parsea Config/Behavior/Tactics/t121_4v4_scaled.cfg │
│  ├── Parsea Config/Behavior/Tactics/t13_4v4_scaled.cfg │
│  └── Parsea Config/Locations/4v4_Scaled/fieldDimensions.cfg │
│                                                      │
│  TacticFSM                                           │
│  ├── Ejecuta FSM idéntica a Behavior.cpp L449-L487   │
│  ├── Input: ball_x, num_players, timers              │
│  └── Output: current_tactic, home_poses              │
│                                                      │
│  SoccerEnv (Gymnasium)                               │
│  ├── Fase 1: RL output = target_positions            │
│  │   └── FSM sigue corriendo, roles = heurísticos    │
│  ├── Fase 2: RL output = positions + roles           │
│  │   └── FSM sigue corriendo, skills = heurísticos   │
│  └── Fase 3: RL output = positions + roles + skills  │
│      └── FSM eliminada, RL controla todo              │
│                                                      │
│  MAPPO Trainer                                       │
│  ├── 3 actores field players 4v4 (parameter sharing) │
│  ├── 1 critic centralizado                           │
│  └── Shared team reward                              │
│                                                      │
│  Export: policy.onnx                                  │
└───────────────────┬──────────────────────────────────┘
                    │ (futuro)
                    ▼
┌──────────────────────────────────────────────────────┐
│                   C++ (Deployment)                    │
│                                                      │
│  Config/NeuralNets/MAPPO/policy.onnx                 │
│  ├── Cargada via CompiledNN (ya existe en BHuman)    │
│  └── Ejecutada en StrategyBehaviorControl            │
│                                                      │
│  Modificación en StrategyBehaviorControl:            │
│  ├── Si MAPPO habilitado:                            │
│  │   ├── Construir obs vector desde representaciones │
│  │   ├── Inferir policy.onnx → target positions      │
│  │   └── Escribir en agent.position (bypass FSM)     │
│  └── Si MAPPO deshabilitado:                         │
│      └── Código heurístico original intacto          │
└──────────────────────────────────────────────────────┘
```

---

## Observation Space definitivo (todas las fases)

### Fase 1 — Solo posiciones (4v4 Scaled)

| Grupo | Features | Dim |
|-------|----------|-----|
| Self pose | x, y, θ | 3 |
| Self velocity | vx, vy, ω | 3 |
| Self state | is_upright, battery_level | 2 |
| Self position | one_hot(5): gk/defL/defR/fwdL/fwdR | 5 |
| Self active role | is_active (bool) | 1 |
| Self goal info | goal_opening_angle, goal_relative_angle | 2 |
| Ball | x, y, vx, vy, ball_age_ms, using_team_ball | 6 |
| Teammates ×3 | x, y, walk_dx, walk_dy, speed, is_active, exists | 21 |
| Opponents ×4 | x, y, exists (LOCAL + FOV-dependiente) | 12 |
| Game | is_playing, own_score, opp_score, half_time_remaining_s | 4 |
| Tactic | one_hot(2): t121_4v4_scaled / t13_4v4_scaled | 2 |
| **Total** | | **61** |

> **Nota campo 4v4:** Las coordenadas absolutas tienen rango x ∈ [-2813, 2813], y ∈ [-1875, 1875].
> El arco rival está en `xPosOpponentGoalPost = 2838`, `yPosLeftGoal = ±800`.

### Fase 2 — Posiciones + Roles (4v4 Scaled)

Misma obs de Fase 1 + para el critic:
- Roles actuales de los 3 agentes RL: one_hot(3) × 3 = 9
- **Total actor:** 61, **Total critic:** 61×3 + 9 + balón real (4) = **196**

### Fase 3 — Todo (4v4 Scaled)

Misma obs de Fase 2 + para el actor:
- Acciones anteriores del equipo (embedding): +9
- Ball possession score: +1
- **Total actor:** ~71

---

## Resumen de archivos por fase

### Fase 1 — Solo Python (no tocar C++)

| Archivo Python a crear | Propósito |
|------------------------|-----------|
| `bhuman_cfg_parser.py` | Parser de .cfg → Python dicts |
| `soccer_env.py` | Entorno Gymnasium sim2D |
| `tactic_fsm.py` | FSM de tácticas (replica Behavior.cpp) |
| `mappo_trainer.py` | Entrenamiento MAPPO |
| `config.py` | Hiperparámetros, obs/action space definitions |

| Archivo .cfg que se LEE (no modifica) | Datos extraídos |
|----------------------------------------|-----------------|
| `Config/Behavior/Tactics/t121_4v4_scaled.cfg` | Home poses defensivas |
| `Config/Behavior/Tactics/t13_4v4_scaled.cfg` | Home poses ofensivas |
| `Config/Behavior/Strategies/s4v4_scaled.cfg` | FSM de transiciones t121↔t13 |
| `Config/Locations/4v4_Scaled/fieldDimensions.cfg` | Tabla de símbolos (FIELD_SYMBOLS) |

### Fase 2 — Solo Python (extiende Fase 1)

| Archivo Python a modificar | Cambio |
|---------------------------|--------|
| `soccer_env.py` | Action space + role assignment logic |
| `mappo_trainer.py` | Multi-head policy |

### Fase 3 — Solo Python (extiende Fase 2)

| Archivo Python a modificar | Cambio |
|---------------------------|--------|
| `soccer_env.py` | Skills como acciones, eliminar FSM |
| `mappo_trainer.py` | Action masking para pass_target |

### Deployment (futuro, post-entrenamiento)

| Archivo C++ a modificar | Cambio |
|-------------------------|--------|
| [StrategyBehaviorControl.h](Src/Modules/BehaviorControl/StrategyBehaviorControl/StrategyBehaviorControl.h) | Añadir `REQUIRES(MAPPOPolicy)` o cargar ONNX |
| [Behavior.cpp](Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp) | Flag `useMAPPO` que bypasea assignPositions/assignRoles |
| `Config/NeuralNets/MAPPO/` | Carpeta nueva para el modelo .onnx exportado |
> ESTADO DOCUMENTAL: `referencia`
>
> La copia preservada y citada de este documento vive en:
> [Documentacion/REFERENCIA/PLAN/MAPPO_Plan_3Fases.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/MAPPO_Plan_3Fases.md)
> No usar este archivo como baseline por si solo.
