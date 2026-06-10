# Guía de Transplante RL → BHuman
## Heurísticas, Enmascarado y Gate — Referencia Completa para Integración en Robot Físico

> **Propósito de este documento:** Permitir que cualquier persona del equipo replique en el código BHuman (C++) **exactamente** el mismo comportamiento de inferencia que produce los resultados vigentes del `behavioral_evaluator`. Si alguna de estas piezas se omite o se implementa incorrectamente, el agente puede dejar de cumplir los criterios de producción hoy revalidados.
>
> **Checkpoint de referencia vigente:** `runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt`  
> **Evaluación canónica vigente:** `runs/behavioral_evaluator/behavior_20260608_181747/summary.csv`  
> **Estado canónico vigente:** `3/5` escenarios aprobados (`finish_band_2p5_3m_clean`, `finish_band_2p5_3m_static_lane`, `dynamic_full_field_pressure`)  
> **Nota histórica:** `runs/production_readiness_v2/checkpoint_stage4_step10000.pt` y `behavior_20260605_022521` siguen siendo una referencia histórica útil, pero ya no deben tratarse como baseline vigente de transplante.

---

## Índice

1. [Visión general del pipeline de inferencia](#1-visión-general-del-pipeline-de-inferencia)
2. [Espacio de acción y SKILL_ORDER](#2-espacio-de-acción-y-skill_order)
3. [Espacio de observación](#3-espacio-de-observación)
4. [SkillGate — lógica completa](#4-skillgate--lógica-completa)
5. [Máscara de política en inferencia](#5-máscara-de-política-en-inferencia)
6. [Heurísticas de reward (solo referencia, NO van al robot)](#6-heurísticas-de-reward-solo-referencia-no-van-al-robot)
7. [Parámetros de acción por skill](#7-parámetros-de-acción-por-skill)
8. [Constantes críticas](#8-constantes-críticas)
9. [Orden de operaciones en cada paso](#9-orden-de-operaciones-en-cada-paso)
10. [Lista de verificación para el transplante](#10-lista-de-verificación-para-el-transplante)
11. [Qué NO reimplementar en el robot](#11-qué-no-reimplementar-en-el-robot)

---

## 1. Visión general del pipeline de inferencia

En cada paso del robot, el pipeline completo es:

```
Percepción BHuman
       │
       ▼
  BackendObservation          ← vector de 26 floats (ObsIndex)
       │
       ├──► SkillGate.step()  ← decide shoot_armed / dribble_armed
       │         │
       │         └──► obs[23], obs[24], obs[25] (baked en la observación)
       │
       ▼
  Red neuronal (checkpoint)
       │
       ▼
  skill_logits + param_mean
       │
       ├──► gate_skill_mask_from_obs()  ← enmascara logits
       │
       ▼
  argmax(masked_logits) → skill_id
  param_mean * ACTION_PARAM_MASK[skill] → target_x, target_y, target_theta
       │
       ▼
  BHuman skill executor
```

Las tres piezas que **deben** replicarse en BHuman son:
- La construcción del vector de observación (sección 3)
- La lógica del `SkillGate` con su estado (sección 4)
- La máscara de logits antes de hacer argmax (sección 5)

---

## 2. Espacio de acción y SKILL_ORDER

**Archivo fuente:** `action.py`

El modelo genera una acción discreta (skill) + 4 parámetros continuos. El índice de cada skill está fijo:

```
SKILL_ORDER = ("stand", "walk", "shoot", "pass", "dribble", "block", "mark", "observe")
índices:           0        1       2       3        4          5        6        7
```

**Skills activos en producción** (los únicos que la política puede escoger en Stage 4):
```
("stand", "walk", "shoot", "dribble")
```
`pass`, `block`, `mark`, `observe` están enmascarados a `-inf` por el stage mask durante entrenamiento y evaluación. En el robot, el stage mask de producción es Stage 4.

### Parámetros continuos por skill

```
acción = [skill_id, target_x, target_y, target_theta, pass_target]
```

Los parámetros usados por cada skill (1 = activo, 0 = fijo en cero):

| skill   | target_x | target_y | target_theta | pass_target |
|---------|----------|----------|--------------|-------------|
| stand   | 0        | 0        | 0            | 0           |
| walk    | 0        | **1**    | **1**        | 0           |
| shoot   | 0        | 0        | 0            | 0           |
| dribble | 0        | **1**    | 0            | 0           |

Los parámetros marcados con 0 **siempre son cero** en la salida del modelo (la red los aprende como cero). En BHuman, ignorar esas dimensiones y usar cero directamente.

### Rangos de parámetros continuos

Los parámetros se escalan internamente (ver `ActionConfig` en `config.py`):

```
target_x: residual_x_range = 0.0  → siempre 0 (no usar)
target_y: residual_y_range = 450.0 mm  (–450 a +450 mm relativo al anchor)
target_theta: residual_theta_range = 0.75 rad  (–0.75 a +0.75 rad)
pass_target: entero 0–4
```

El anchor para `walk` es `ball_x + (–210 mm), ball_y` (posición ideal de aproximación detrás del balón).

---

## 3. Espacio de observación

**Archivo fuente:** `observation.py`

Vector de 26 floats. **El orden y las unidades son críticos.** Una observación mal construida hace que el modelo se comporte de forma completamente errática.

| Índice | Nombre                          | Unidad / rango             | Fuente BHuman                              |
|--------|---------------------------------|----------------------------|--------------------------------------------|
| 0      | `robot_x`                       | mm, campo completo         | `RobotPose.translation.x`                  |
| 1      | `robot_y`                       | mm                         | `RobotPose.translation.y`                  |
| 2      | `robot_theta`                   | rad, [–π, π]               | `RobotPose.rotation`                       |
| 3      | `ball_rel_x`                    | mm relativo al robot       | `BallModel.estimate.position.x`            |
| 4      | `ball_rel_y`                    | mm relativo al robot       | `BallModel.estimate.position.y`            |
| 5      | `ball_end_rel_x`                | mm, posición final proyect.| derivado de velocidad                      |
| 6      | `ball_end_rel_y`                | mm                         | derivado de velocidad                      |
| 7      | `ball_vel_x`                    | mm/s                       | `BallModel.estimate.velocity.x`            |
| 8      | `ball_vel_y`                    | mm/s                       | `BallModel.estimate.velocity.y`            |
| 9      | `time_since_ball_seen`          | ms                         | `BallModel.timeWhenLastSeen` (delta)       |
| 10     | `time_since_ball_disappeared`   | ms                         | tiempo desde última desaparición           |
| 11     | `ball_seen_percentage`          | 0–100                      | porcentaje de frames con visión del balón  |
| 12     | `ball_consistent_with_game_state`| 0 o 1                     | consistencia con estado de juego           |
| 13     | `can_score_now`                 | 0 o 1                      | flag de oportunidad directa de gol         |
| 14     | `shot_quality_no_obstacles`     | [0, 1]                     | apertura geométrica sin obstáculos         |
| 15     | `shot_opening_with_obstacles`   | [0, 1]                     | apertura real considerando obstáculos      |
| 16     | `pass_options_count`            | entero                     | número de compañeros disponibles           |
| 17     | `nearest_teammate_dist`         | mm                         | distancia al compañero más cercano         |
| 18     | `nearest_opponent_dist`         | mm                         | distancia al oponente más cercano          |
| 19     | `nearest_uncertain_obstacle_dist`| mm                        | distancia al obstáculo incierto más cercano|
| 20     | `nearest_teammate_front_dist`   | mm                         | compañero más cercano al frente            |
| 21     | `nearest_opponent_front_dist`   | mm                         | oponente más cercano al frente             |
| 22     | `nearest_uncertain_front_dist`  | mm                         | obstáculo incierto más cercano al frente   |
| 23     | `shoot_armed`                   | 0.0 o 1.0                  | **salida del SkillGate** (ver sección 4)   |
| 24     | `dribble_armed`                 | 0.0 o 1.0                  | **salida del SkillGate** (ver sección 4)   |
| 25     | `shoot_arm_progress`            | [0, 1]                     | **salida del SkillGate** (ver sección 4)   |

> **Crítico:** Los índices 23, 24 y 25 son salidas del `SkillGate`, no percepciones directas. El gate se ejecuta **antes** de construir el vector de observación, y sus resultados se "hornean" (baked) en él. La red los usa para decidir si enmascarar shoot/dribble. Si estos tres valores se envían en cero siempre, el modelo nunca disparará ni driblarán limpiamente.

---

## 4. SkillGate — lógica completa

**Archivo fuente:** `gate.py`

El gate es un **objeto con estado** que debe vivir durante todo el episodio. Se resetea al inicio de cada episodio (kickoff, penalización, etc.).

### 4.1 Estado interno

```python
shoot_armed: bool = False
dribble_armed: bool = False
shoot_streak: int = 0    # contador de frames consecutivos cumpliendo condición de enter
dribble_streak: int = 0
```

### 4.2 Parámetros (GateConfig — valores de producción)

**Gate de shoot:**

| Parámetro               | Valor          | Descripción |
|-------------------------|----------------|-------------|
| `shoot_enter_ball_mm`   | 380.0 mm       | Distancia máxima robot–balón para entrar |
| `shoot_exit_ball_mm`    | 460.0 mm       | Distancia máxima para mantener armado |
| `shoot_enter_angle_rad` | 25.0° (0.436 rad) | Ángulo máximo robot–balón para entrar |
| `shoot_exit_angle_rad`  | 35.0° (0.611 rad) | Ángulo máximo para mantener armado |
| `shoot_goal_enter_mm`   | **3000.0 mm**  | Distancia balón–portería máxima para entrar |
| `shoot_goal_exit_mm`    | **3200.0 mm**  | Distancia balón–portería máxima para mantener |
| `shoot_open_enter`      | 0.6            | Apertura mínima del carril (sin obstáculos = armar) |
| `shoot_open_exit`       | 0.45           | Apertura mínima para mantener |
| `shoot_align_enter_rad` | 30.0° (0.524 rad) | Alineación máxima robot–portería para entrar |
| `shoot_align_exit_rad`  | 45.0° (0.785 rad) | Alineación máxima para mantener |
| `shoot_arm_frames`      | **3**          | Frames consecutivos requeridos para armar |

**Gate de dribble:**

| Parámetro                    | Valor             | Descripción |
|------------------------------|-------------------|-------------|
| `dribble_enter_ball_mm`      | 450.0 mm          | Distancia máxima robot–balón |
| `dribble_exit_ball_mm`       | 600.0 mm          | Distancia para mantener (latch amplio) |
| `dribble_enter_angle_rad`    | 35.0° (0.611 rad) | Ángulo máximo |
| `dribble_exit_angle_rad`     | 55.0° (0.960 rad) | Ángulo para mantener |
| `dribble_arm_frames`         | **1**             | 1 frame es suficiente para armar |
| `dribble_grace_enter_age_ms` | **1500.0 ms**     | Edad máxima del balón natural para entrar |
| `dribble_grace_hold_age_ms`  | **2500.0 ms**     | Edad máxima para mantener armado (dropout tolerance) |

**Percepción:**

| Parámetro            | Valor      | Descripción |
|----------------------|------------|-------------|
| `trusted_max_age_ms` | **600.0 ms** | Edad máxima del balón natural para considerarlo fresco (shoot) |

### 4.3 Algoritmo paso a paso

Ejecutar en **cada paso** antes de construir la observación:

```python
# Inputs del entorno:
nat_age_ms = time_since_natural_ball_seen  # ms desde último avistamiento NATURAL
d_ball = sqrt(ball_rel_x² + ball_rel_y²)  # mm
a_ball = abs(atan2(ball_rel_y, ball_rel_x))  # rad
shot_open = shot_opening_with_obstacles    # [0,1]
score_now = can_score_now                  # bool
goal_dist = sqrt((GOAL_X - ball_x)² + ball_y²)  # mm, GOAL_X = 4500.0

# Heading del robot hacia el centro de la portería contraria
goal_bearing = atan2(0.0 - robot_y, GOAL_X - robot_x)
align = abs(wrap_angle(robot_theta - goal_bearing))  # rad

# Freshness para shoot (estricta)
fresh = nat_age_ms <= 600.0  # trusted_max_age_ms

# Freshness para dribble (con gracia)
engage_enter_fresh = nat_age_ms <= 1500.0
engage_hold_fresh  = nat_age_ms <= 2500.0

# === SHOOT GATE ===
on_ball_enter = d_ball <= 380.0 and a_ball <= 0.436  # (25°)
on_ball_exit  = d_ball <= 460.0 and a_ball <= 0.611  # (35°)
lane_enter = shot_open >= 0.6 or score_now
lane_exit  = shot_open >= 0.45 or score_now
range_enter = goal_dist <= 3000.0
range_exit  = goal_dist <= 3200.0
align_enter = align <= 0.524   # (30°)
align_exit  = align <= 0.785   # (45°)

enter_cond = fresh and on_ball_enter and lane_enter and range_enter and align_enter
hold_cond  = fresh and on_ball_exit  and lane_exit  and range_exit  and align_exit

if not shoot_armed:
    if enter_cond:
        shoot_streak += 1
    else:
        shoot_streak = 0
    if shoot_streak >= 3:    # shoot_arm_frames
        shoot_armed = True
else:
    if not hold_cond:
        shoot_armed = False
        shoot_streak = 0

# === DRIBBLE GATE ===
db_enter = engage_enter_fresh and d_ball <= 450.0 and a_ball <= 0.611  # (35°)
db_hold  = engage_hold_fresh  and d_ball <= 600.0 and a_ball <= 0.960  # (55°)

if not dribble_armed:
    if db_enter:
        dribble_streak += 1
    else:
        dribble_streak = 0
    if dribble_streak >= 1:   # dribble_arm_frames
        dribble_armed = True
else:
    if not db_hold:
        dribble_armed = False
        dribble_streak = 0

# === SHOOT ARM PROGRESS ===
if shoot_armed:
    shoot_arm_progress = 1.0
else:
    shoot_arm_progress = min(1.0, shoot_streak / 3.0)

# Escribir en obs[23], obs[24], obs[25]
obs[23] = 1.0 if shoot_armed else 0.0
obs[24] = 1.0 if dribble_armed else 0.0
obs[25] = shoot_arm_progress
```

### 4.4 Diseño de la percepción: canal natural vs. exportado

El gate **distingue dos canales de balón**:

- **Canal natural** (`natural_time_since_ball_seen`): avistamiento visual directo de la cámara. Usado para armar shoot (fresco estricto ≤ 600ms) y dribble (con gracia ≤ 1500ms/2500ms).
- **Canal exportado/estimado**: el `BallModel` de BHuman que puede estar propagado o corregido. Se usa para `d_ball`, `a_ball`, `shot_open` y el resto de la observación.

En el robot físico esto implica mantener un timestamp separado del último frame donde la cámara detectó directamente el balón (no el modelo propagado).

### 4.5 Reset

Al inicio de cada episodio (estado `initial`, kickoff, penalización):
```python
shoot_armed = False
dribble_armed = False
shoot_streak = 0
dribble_streak = 0
```

---

## 5. Máscara de política en inferencia

**Archivo fuente:** `train.py`, función `gate_skill_mask_from_obs`

Esta es la pieza más crítica para el comportamiento en campo. Se aplica **después** de que la red genera logits, antes de hacer `argmax`.

### 5.1 Reglas de la máscara

```python
# Leer del vector de observación
shoot_armed  = obs[23] >= 0.5  # bool
dribble_armed = obs[24] >= 0.5  # bool
finish_armed  = shoot_armed or dribble_armed

# Máscara multiplicativa (1=permitido, 0=prohibido)
mask = [1.0] * len(SKILL_ORDER)  # todos permitidos por defecto

# shoot solo permitido cuando shoot_armed
mask[2] = 1.0 if shoot_armed else 0.0    # índice 2 = shoot

# dribble solo permitido cuando dribble_armed
mask[4] = 1.0 if dribble_armed else 0.0  # índice 4 = dribble

# stand prohibido cuando finish_armed (hay oportunidad de disparar o driblar)
mask[0] = 0.0 if finish_armed else 1.0   # índice 0 = stand

# walk prohibido cuando finish_armed
mask[1] = 0.0 if finish_armed else 1.0   # índice 1 = walk
```

### 5.2 Aplicar la máscara sobre los logits

```python
# logits: vector de 8 floats (uno por skill)
for i in range(len(logits)):
    if mask[i] <= 0.0:
        logits[i] = -1e9  # equivalente a probabilidad 0

skill_id = argmax(logits)
```

### 5.3 Por qué es crítico

Sin esta máscara:
- El agente elige `walk` o `stand` dentro del takeover envelope → `clean_takeover_rate ≈ 0`
- El agente puede disparar sin estar en posición → `bad_shoot_rate` alto
- El agente puede driblar estando lejos del balón → `bad_dribble_rate` alto

Con esta máscara, en el run de referencia se obtiene:
- `clean_takeover_rate ≥ 0.75` en los 4 escenarios que pasan
- `shoot_finish_rate ≥ 0.50` en finish_band_clean

### 5.4 Invariante de seguridad

La máscara **nunca puede bloquear todas las skills**. Cuando `finish_armed=False`, `walk` y `stand` están disponibles como fallback. Cuando `finish_armed=True`, al menos `shoot` o `dribble` están disponibles. Esta propiedad es garantizada por el diseño del gate: si `finish_armed=True` es porque exactamente una de {`shoot_armed`, `dribble_armed`} es True.

---

## 6. Heurísticas de reward (solo referencia, NO van al robot)

**Archivo fuente:** `reward.py`

Estas heurísticas se usaron **únicamente durante entrenamiento** para moldear el comportamiento de la política. En el robot físico no hay reward — solo inferencia. Sin embargo, documentarlas ayuda a entender qué comportamientos se reforzaron y por qué la política hace lo que hace.

### 6.1 Zonas geométricas de la heurística

Estas zonas definen qué skill se considera "correcto" en cada posición:

**Takeover envelope** (condición base para estar "en el balón"):
```
d_ball ≤ 450 mm
AND |a_ball| ≤ 35°
AND time_since_ball_seen ≤ 600 ms (trusted)
```

**Shoot zone** (condición para que shoot sea el skill correcto):
```
d_ball ≤ 380 mm
AND |a_ball| ≤ 25°
AND time_since_ball_seen ≤ 600 ms
AND (shot_opening_with_obstacles ≥ 0.6 OR can_score_now)
```

**Dribble zone**: `takeover_envelope AND NOT shoot_zone`

**Walk zone**: NOT takeover_envelope AND time_since_ball_seen ≤ 1000 ms

### 6.2 Tabla de heurísticas de reward

| Término | Qué castiga/premia | Efecto en comportamiento |
|---|---|---|
| `bad_walk` | walk dentro del takeover envelope | No caminar cuando ya está en el balón |
| `bad_dribble` | dribble fuera del takeover envelope | No driblar sin tener el balón |
| `bad_shoot` | shoot fuera de la shoot zone | No disparar sin estar bien posicionado |
| `idle_stand` | stand con balón visible y lejos | No pararse frente al balón |
| `walk_persistence` | walk consecutivo en walk zone | Premiar acercarse al balón consistentemente |
| `hold_takeover` | dribble/shoot en takeover con progreso goalward | Premiar avanzar hacia la portería |
| `skill_switch` | cambio de skill innecesario | Estabilidad en la elección de skill |
| `thrash` | patrón walk→dribble→walk o similar | No oscilar entre skills |
| `collision` | colisión con obstáculos | Evitar chocar |
| `missed_dribble_handoff` | no elegir dribble/shoot en stable takeover | Comprometerse con la oportunidad |
| `missed_shoot` | no elegir shoot en shoot zone | Disparar cuando la oportunidad está |
| `preferred_skill_match` | elegir el skill correcto para la zona | Refuerzo positivo de la zona correcta |

### 6.3 Constantes del reward que definen las zonas

```
BALL_TRUSTED_MAX_AGE_MS = 600.0 ms
BALL_VISIBLE_MAX_AGE_MS = 1200.0 ms
GOOD_DRIBBLE_FULL_STEPS = 8      # pasos antes de que el crédito tape
GOOD_DRIBBLE_TAPER_STEPS = 24.0
GOOD_DRIBBLE_FLOOR = 0.3
_APPROACH_OFFSET_X = -210.0 mm  # target de walk = ball_x - 210mm
_APPROACH_RADIUS_MM = 600.0 mm
```

---

## 7. Parámetros de acción por skill

Cuando `skill_id` está decidido, la red también genera `target_x`, `target_y`, `target_theta`. Los parámetros que usa cada skill (los demás son ignorados = cero):

**walk:**
```
target_y   → componente lateral del target de walk (residual sobre anchor)
target_theta → orientación deseada del robot

anchor = (ball_x + (−210 mm), ball_y)   ← posición ideal de aproximación
target_x siempre = 0 (residual_x_range = 0.0, siempre anchor)
```

**dribble:**
```
target_y → dirección lateral del dribble (residual sobre posición del balón)
target_x siempre = 0
target_theta siempre = 0 (la red no la usa)
```

**shoot:**
```
Todos los parámetros = 0. El kick se ejecuta sin target adicional.
La dirección la determina la orientación actual del robot.
```

**stand:**
```
Todos los parámetros = 0.
```

---

## 8. Constantes críticas

**Archivo fuente:** `config.py`

```
FIELD_X_HALF = 4500.0 mm     (mitad del campo en X)
FIELD_Y_HALF = 3000.0 mm     (mitad del campo en Y)
GOAL_X       = 4500.0 mm     (posición X de la portería contraria)
GOAL_Y_HALF  = 750.0 mm      (mitad del ancho de la portería)
DT           = 0.033 s/step  (33 ms por paso, ~30 Hz)
```

### Sistema de coordenadas

El origen es el centro del campo. El eje X apunta hacia la portería contraria (`GOAL_X = 4500`). El eje Y va de derecha a izquierda desde la perspectiva del robot. `robot_theta = 0` apunta en dirección +X (hacia la portería contraria).

**Atención:** En SimRobot hay un `theta_offset = -π` para el perfil 2D y `-π/2` para 3D. Al integrar en BHuman verificar que el sistema de coordenadas de `RobotPose` coincida con este convenio. Una inversión de signo en `robot_theta` hace que el gate de alineación falle completamente.

---

## 9. Orden de operaciones en cada paso

El orden es obligatorio. Cambiar el orden rompe la consistencia entre lo que entrenó y lo que corre el robot:

```
1. Leer percepción de BHuman:
   - RobotPose → robot_x, robot_y, robot_theta
   - BallModel → ball_rel_x/y, ball_vel_x/y, time_since_ball_seen, etc.
   - ObstacleModel → distancias a compañeros/oponentes
   - Flags de juego → can_score_now, shot_opening, etc.

2. Ejecutar SkillGate.step(obs_raw) con el estado persistente del gate:
   → produce shoot_armed, dribble_armed, shoot_arm_progress

3. Construir vector de observación completo (26 floats):
   → obs[0:23] = percepciones
   → obs[23] = shoot_armed (del gate)
   → obs[24] = dribble_armed (del gate)
   → obs[25] = shoot_arm_progress (del gate)

4. Forward pass de la red neuronal:
   → entrada: obs (26 floats)
   → salida: skill_logits (8 floats) + param_mean (4 floats)

5. Aplicar gate_skill_mask_from_obs(obs):
   → leer obs[23], obs[24]
   → enmascarar logits (ver sección 5)

6. skill_id = argmax(masked_logits)

7. Aplicar ACTION_PARAM_MASK[skill]:
   → params = param_mean * mask  (los parámetros inactivos = 0)

8. Enviar skill_id + params al ejecutor de skills de BHuman
```

---

## 10. Lista de verificación para el transplante

Antes de hacer pruebas con el robot físico, verificar cada uno de estos puntos:

### Observación
- [ ] El vector tiene exactamente 26 floats en el orden de `ObsIndex`
- [ ] `ball_rel_x/y` son coordenadas **relativas al robot** (no absolutas)
- [ ] `time_since_ball_seen` está en **milisegundos**
- [ ] `shot_opening_with_obstacles` está en [0, 1]
- [ ] `robot_theta` es en el sistema de coordenadas correcto (origen hacia portería contraria)
- [ ] `obs[23]`, `obs[24]`, `obs[25]` se llenan con la salida del gate (no ceros fijos)

### SkillGate
- [ ] El estado del gate (shoot_armed, dribble_armed, streaks) persiste entre pasos
- [ ] El gate se resetea al inicio de cada episodio de juego
- [ ] Se usa `natural_time_since_ball_seen` (canal natural) y NO el time del BallModel propagado
- [ ] `goal_dist` se calcula como distancia **del balón** a la portería contraria (no del robot)
- [ ] `align` se calcula como diferencia entre `robot_theta` y el bearing al centro de la portería
- [ ] `wrap_angle` está implementado: `(angle + π) % (2π) − π`
- [ ] Los 3 frames de debounce de shoot (`shoot_arm_frames = 3`) están implementados
- [ ] El dribble arma en 1 frame (`dribble_arm_frames = 1`)
- [ ] La gracia de dribble usa ventana asimétrica (1500ms enter, 2500ms hold)

### Máscara de logits
- [ ] La máscara se aplica **antes** del argmax, nunca después
- [ ] Cuando `shoot_armed=False`, el logit de shoot se lleva a −∞ (no a 0)
- [ ] Cuando `dribble_armed=False`, el logit de dribble se lleva a −∞
- [ ] Cuando `finish_armed=True`, los logits de stand (0) y walk (1) se llevan a −∞
- [ ] Cuando `finish_armed=False`, walk y stand están disponibles (fallback)

### Parámetros de acción
- [ ] `target_x` para walk y dribble es **siempre cero** (residual_x_range = 0)
- [ ] Los parámetros inactivos de cada skill se fijan en cero antes de enviarlos

### Sistema de coordenadas
- [ ] GOAL_X = 4500 mm (portería contraria)
- [ ] El campo es –4500 a +4500 en X, –3000 a +3000 en Y
- [ ] `robot_theta = 0` apunta hacia la portería contraria (+X)

---

## 11. Qué NO reimplementar en el robot

Las siguientes piezas son **solo para entrenamiento** y no deben incluirse en el código del robot:

- Toda la lógica de `reward.py` (heurísticas de reward, zonas geométricas del reward)
- El `behavioral_evaluator.py` y sus métricas
- La lógica de curriculum (`curriculum.py`, `train_production.py`)
- Los stage masks de entrenamiento (distintos al stage mask de producción Stage 4)
- La lógica de `heuristic_bias` del teacher
- Cualquier cosa relacionada con PPO, BC loss, o el loop de entrenamiento

Lo que sí va al robot es únicamente:
1. El vector de observación (sección 3)
2. El SkillGate con estado (sección 4)
3. Los pesos del checkpoint (la red neuronal)
4. La máscara de logits (sección 5)
5. El decodificador de parámetros por skill (sección 7)

---

*Documento actualizado tras la reparación de estabilidad de 2026-06-08. Baseline vigente: `production_stability_repair_v1/checkpoint_stage4_step5000.pt`. Actualizar si se modifica `gate.py`, `train.py` (función `gate_skill_mask_from_obs`), `observation.py`, `action.py`, `pybh_backend.py` o el checkpoint candidato oficial de transplante.*
