# Handoff Tecnico: PPO 23 vars + MAPPO 35 vars

## Objetivo de este documento

Este archivo resume las decisiones tecnicas ya conversadas para que otro Codex pueda continuar el trabajo sin perder contexto.

No es una especificacion final de implementacion. Es un handoff de arquitectura y alcance.

## Objetivo real del proyecto

El objetivo no es sacar una demo estetica ni entrenar fisicas desde RL.

El objetivo es:

- que un NAO v6 tome decisiones futbolisticas dinamicas,
- que RL elija que skill conviene,
- y que B-Human siga ejecutando esa skill con su stack natural.

La regla acordada es:

```text
RL decide QUE skill usar.
B-Human decide COMO ejecutarla.
```

Eso implica:

- RL no debe controlar joints,
- RL no debe controlar locomocion de bajo nivel,
- RL no debe reemplazar `MotionEngine`,
- RL no debe puentear `SkillBehaviorControl`.

## Pipeline actual que SI queremos mantener

Hoy, en los escenarios RL dedicados, el pipeline activo correcto es:

```text
RL
  -> SkillRequest
  -> SkillBehaviorControl
  -> MotionRequest
  -> MotionEngine
  -> joints
  -> SimRobot
```

Evidencia local:

- `SabanaHerons2026/Config/Scenarios/4v4_RL2D/threads.cfg`
- `SabanaHerons2026/Config/Scenarios/4v4_RL3D/threads.cfg`
- `SabanaHerons2026/Config/Scenarios/4v4_RL2D/skillBehaviorControl.cfg`
- `SabanaHerons2026/Config/Scenarios/4v4_RL3D/skillBehaviorControl.cfg`
- `SabanaHerons2026/Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.cpp`
- `SabanaHerons2026/Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.cpp`

Esto es positivo porque:

- conserva la arquitectura natural de B-Human,
- deja la ejecucion motora en B-Human,
- reduce el riesgo fisico,
- y es la base correcta para NAO v6 real.

## Estado actual del proyecto

La conclusion acordada es:

- el proyecto esta en **prototipo integrado en simulacion**,
- no en canary estable,
- no en preproduccion,
- y mucho menos en produccion fisica confiable.

### Lo que ya existe

- bridge RL <-> B-Human por `RLSharedState`
- `rl_set_action(...)`
- `rl_get_obs(...)`
- escenarios RL dedicados `2D` y `3D`
- `RLSkillProvider`
- `HandleRLRequest`
- `SkillBehaviorControl`
- soporte de skills validado a nivel de pipeline

### Lo que todavia no esta resuelto

- observation space realista para partido
- reward futbolistico serio
- evaluacion robusta sim2real
- estado multiagente honesto
- MAPPO realmente defendible

## Restricciones de tiempo ya acordadas

### PPO

Se acepta apuntar a un PPO de corto plazo para canary mode.

El alcance correcto es:

- skill selection,
- pocas skills,
- targets sintetizados por B-Human o por una capa determinista,
- entrenamiento rapido en simulacion,
- validacion cualitativa en `simrobot_visible`.

### MAPPO

El limite conversado para MAPPO es agresivo: `2-3 semanas`.

Eso obliga a:

- no reinventar el pipeline,
- no inventar un obs monstruoso,
- no meter acciones continuas libres,
- no intentar team strategy completa desde cero,
- reusar al maximo el mismo core del PPO.

## Decisiones de diseno ya cerradas

### 1. GameState

`GameState` importa, pero no entra al vector `obs`.

Se usa como:

- gating externo,
- construccion de `action_mask`,
- y fallback.

Regla acordada:

```text
Si no esta en playing, PPO y MAPPO no deciden.
Decide B-Human.
```

### 2. RobotPose

`RobotPose` debe respetarse en crudo desde B-Human:

- `robot_x`
- `robot_y`
- `robot_theta`

No normalizarlo en el bridge nativo.

Si alguien quiere normalizar para entrenamiento, eso puede hacerse afuera, en Python, no en `rl_get_obs()`.

### 3. Ball features

Las features del balon si son centrales para RL y deben salir de representaciones reales de B-Human:

- `FieldBall`
- `BallModel`

### 4. Obstaculos

`ObstacleModel` no es opcional conceptualmente.

RL necesita distinguir entre:

- teammate seguro,
- rival seguro,
- obstaculo incierto/random.

Clasificacion acordada desde `Obstacle::Type`:

- `teammate_sure`:
  - `teammate`
  - `fallenTeammate`
- `opponent_sure`:
  - `opponent`
  - `fallenOpponent`
- `uncertain_obstacle`:
  - `unknown`
  - `someRobot`
  - `fallenSomeRobot`

### 5. Derivaciones minimas

Se acordo transformar lo minimo posible.

Si una variable puede salir directo de B-Human, debe salir directo.

Solo se permiten derivaciones chicas y locales para:

- resumen ofensivo,
- resumen de obstaculos,
- slots de teammates para MAPPO.

### 6. Nada de overengineering

Se acordo explicitamente NO meter por ahora:

- `goal_lane_blocked`
- mapas densos
- sector wheels
- scores tacticos complicados
- covarianzas complejas de `RobotPose`
- schemas gigantes de observacion
- acciones continuas libres de campo completo

## Riesgo principal ya identificado

El problema mas serio del sistema actual no es la ejecucion motora.

El problema es el **observation space demasiado perfecto** por el uso de providers oraculares en escenarios RL.

Eso genera:

- reward enganoso,
- decisiones fragiles,
- politicas que parecen buenas en headless,
- y transferencias malas a 3D y a NAO fisico.

Documento relacionado:

- `RL/docs/07-observation-space-realista-y-fallback-canary.md`

## Action space acordado

### PPO canary

Se acuerda un action space discreto de skills:

```text
Discrete(7)
```

Con mapping:

0. `observe`
1. `walk`
2. `shoot`
3. `pass`
4. `dribble`
5. `block`
6. `mark`

### Politica operativa del canary

Aunque el espacio completo exista, para el canary temprano se recomienda habilitar primero:

- `observe`
- `walk`
- `shoot`

Opcional cuarto skill:

- `dribble`

### Regla de parametrizacion

RL no debe emitir targets continuos libres en la primera etapa.

La parametrizacion fina debe sintetizarse fuera de RL:

- `observe` -> scan o target predefinido
- `walk` -> target sintetizado por contexto
- `shoot` -> skill directa
- `pass` -> teammate elegible externo
- `dribble` -> modo discreto
- `block` -> punto sintetizado
- `mark` -> rival o zona sintetizada

### MAPPO

La extension sugerida despues es:

- `skill_id: Discrete(7)`
- `walk_target_mode: Discrete(...)`
- `pass_target_slot: Discrete(...)`
- `dribble_mode: Discrete(...)`

Pero no en la primera iteracion.

## PPO definitivo acordado: 23 variables

Este es el observation space acordado para PPO.

### Bloque RobotPose

1. `robot_x`
2. `robot_y`
3. `robot_theta`

Fuente:

- `RobotPose`

### Bloque ball state

4. `ball_rel_x`
5. `ball_rel_y`
6. `ball_end_rel_x`
7. `ball_end_rel_y`
8. `ball_vel_x`
9. `ball_vel_y`
10. `time_since_ball_seen`
11. `time_since_ball_disappeared`
12. `ball_seen_percentage`
13. `ball_consistent_with_game_state`

Fuentes:

- `FieldBall`
- `BallModel`

### Bloque ofensivo

14. `can_score_now`
15. `shot_quality_no_obstacles`
16. `shot_opening_with_obstacles`
17. `pass_options_count`

Fuentes esperadas:

- `KickAtGoal.cpp`
- `ExpectedGoals.h`
- `FieldRatingProvider.cpp`

Notas:

- `can_score_now` resume la idea de factibilidad ofensiva inmediata
- `shot_quality_no_obstacles` viene de `ExpectedGoals.xG(...)`
- `shot_opening_with_obstacles` viene de `ExpectedGoals.getRating(...)`
- `pass_options_count` sale de las opciones de pase posibles

### Bloque obstaculos

18. `nearest_teammate_dist`
19. `nearest_opponent_dist`
20. `nearest_uncertain_obstacle_dist`
21. `nearest_teammate_front_dist`
22. `nearest_opponent_front_dist`
23. `nearest_uncertain_front_dist`

Fuente:

- `ObstacleModel`

Reglas de derivacion acordadas:

- distancia = `norm(center)`
- frontal = `center.x > 0`

Sin mas semantica.

## MAPPO definitivo acordado: 35 variables por actor

MAPPO debe reusar el PPO completo y solo agregar contexto chico de teammates.

### Base

- las 23 variables del PPO

### Extension teammates

Se usaran 2 slots fijos de teammates relevantes:

- `slot_1`: mejor opcion de pase actual
- `slot_2`: mejor apoyo secundario o teammate mas relevante restante

### Variables extra

24. `mate_1_valid`
25. `mate_1_rel_x`
26. `mate_1_rel_y`
27. `mate_1_dist`
28. `mate_1_ball_recent`
29. `mate_1_pass_target_is_me`
30. `mate_2_valid`
31. `mate_2_rel_x`
32. `mate_2_rel_y`
33. `mate_2_dist`
34. `mate_2_ball_recent`
35. `mate_2_pass_target_is_me`

Fuente:

- `TeamData`
- `BehaviorStatus`
- `BallModel` de teammates si esta disponible en el team message

La idea es no meter todo el equipo completo en el actor.

## Critic de MAPPO acordado

No se quiere inventar otro schema global pesado.

La recomendacion acordada es:

```text
critic_state = concatenacion de las observaciones actor de todos los agentes,
en orden fijo de jersey number
```

Si falta un robot:

- se rellena con cero,
- o con flags `valid = 0`.

## Que cosas deben salir de B-Human y cuales pueden derivarse

### Deben salir directo de B-Human

- `RobotPose`
- `FieldBall`
- `BallModel`
- `ObstacleModel`
- `TeamData`
- `BehaviorStatus`

### Pueden derivarse minimamente

- `can_score_now`
- `shot_quality_no_obstacles`
- `shot_opening_with_obstacles`
- `pass_options_count`
- las 6 distancias de obstaculos
- los 2 teammate slots de MAPPO

## Que cosas quedan fuera por ahora

No meter en este alcance:

- `goal_lane_blocked`
- `sim_enabled`
- `frame` como feature tactica
- targets continuos libres
- reward shaping sofisticado por mapas
- roles complejos como vector grande
- estado completo de todo el equipo en el actor

## Estado de soporte de skills

Se acordo que el soporte de skills esta:

- implementado,
- validado a nivel de pipeline,
- usable para integrar RL,

pero eso NO significa todavia:

- validacion futbolistica completa,
- ni confiabilidad en partido real,
- ni validacion fisica final en NAO.

## Valor de simrobot_visible

`simrobot_visible` no se considera backend principal de entrenamiento.

Su valor esta en:

- auditar la verdad cualitativa de la politica,
- ver si la skill elegida tiene sentido en cuerpo y timing reales,
- detectar oscilaciones,
- detectar kicks mal temporizados,
- separar si el fallo viene de:
  - policy,
  - state,
  - abstraccion de accion,
  - o ejecucion de la skill.

## Recomendacion operativa final

### Para PPO

Ir con:

- observation space de 23 variables,
- accion discreta de skill,
- pocas skills activas al inicio,
- targets sintetizados,
- `GameState` solo como gating,
- validacion headless + `simrobot_visible`.

### Para MAPPO

Ir con:

- PPO base intacto,
- 2 teammate slots,
- critic centralizado por concatenacion,
- sin rediseñar todo el stack.

## Siguiente paso sugerido

Si otro Codex toma esto, el paso mas razonable es:

1. congelar este contrato como referencia
2. revisar que variables ya existen y cuales hay que exponer
3. separar:
   - campos directos,
   - derivados minimos,
   - gating externo,
   - `action_mask`
4. no tocar todavia la arquitectura `SkillRequest -> SkillBehaviorControl`

## Archivos clave para continuar

- `RL/docs/07-observation-space-realista-y-fallback-canary.md`
- `RL/environment.py`
- `RL/action.py`
- `RL/train.py`
- `SabanaHerons2026/Config/Scenarios/4v4_RL2D/threads.cfg`
- `SabanaHerons2026/Config/Scenarios/4v4_RL3D/threads.cfg`
- `SabanaHerons2026/Src/Libs/Python/Controller/Module.cpp`
- `SabanaHerons2026/Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.cpp`
- `SabanaHerons2026/Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.cpp`
- `SabanaHerons2026/Src/Representations/Modeling/RobotPose.h`
- `SabanaHerons2026/Src/Representations/BehaviorControl/FieldBall.h`
- `SabanaHerons2026/Src/Representations/Modeling/BallModel.h`
- `SabanaHerons2026/Src/Representations/Modeling/ObstacleModel.h`
- `SabanaHerons2026/Src/Representations/Communication/TeamData.h`
- `SabanaHerons2026/Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/KickAtGoal.cpp`
- `SabanaHerons2026/Src/Representations/BehaviorControl/ExpectedGoals.h`
- `SabanaHerons2026/Src/Modules/BehaviorControl/FieldRatingProvider/FieldRatingProvider.cpp`
