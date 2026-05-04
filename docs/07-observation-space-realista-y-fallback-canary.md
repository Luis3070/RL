# Observation Space Realista, Accion Minima y Guia de Fallback para Canary

## Objetivo del documento

Este documento existe para responder una pregunta concreta:

```text
Si el canary mode falla, como diagnosticamos si el problema fue:
- el observation space demasiado perfecto,
- el reward,
- la eleccion de accion,
- o una mala suposicion sobre lo que B-Human puede observar en partido real?
```

Tambien fija una direccion de arquitectura para los siguientes pasos:

- PPO primero, con foco en seleccion de skill futbolisticamente buena.
- B-Human sigue ejecutando la skill y conserva locomocion, control de balon, kick, cabeza, brazos y seguridad.
- MAPPO despues, cuando el estado y la evaluacion multiagente sean reales.

La regla central es:

```text
RL decide QUE hacer.
B-Human decide COMO ejecutarlo.
```

## Pipeline actual activo

Hoy, en los escenarios RL dedicados, el pipeline activo es:

```text
RL
  -> SkillRequest
  -> SkillBehaviorControl
  -> MotionRequest
  -> MotionEngine
  -> JointRequest
  -> SimRobot
```

Evidencia local:

- `SabanaHerons2026/Config/Scenarios/4v4_RL2D/threads.cfg`
- `SabanaHerons2026/Config/Scenarios/4v4_RL3D/threads.cfg`
- `SabanaHerons2026/Config/Scenarios/4v4_RL2D/skillBehaviorControl.cfg`
- `SabanaHerons2026/Config/Scenarios/4v4_RL3D/skillBehaviorControl.cfg`
- `SabanaHerons2026/Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.cpp`
- `SabanaHerons2026/Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.cpp`

Lo importante:

- `SkillRequest` lo provee `RLSkillProvider`.
- `MotionRequest` lo provee `SkillBehaviorControl`.
- `HandleRLRequest` ejecuta el request RL dentro del stack normal de skills.

Eso es correcto para este proyecto.

## Estado actual del observation space

El observation space actual del entrenamiento vive en `RL/environment.py`.

Hoy el agente recibe solo 9 floats:

1. `dx pelota - robot`
2. `dy pelota - robot`
3. `theta robot`
4. `ball_x`
5. `ball_y`
6. `robot_x`
7. `robot_y`
8. `frame normalizado`
9. `sim_enabled`

Esto tiene dos problemas al mismo tiempo:

1. Es demasiado perfecto.
2. Es demasiado pobre.

### Demasiado perfecto

En los escenarios RL actuales, varias representaciones de alto valor vienen de `OracledWorldModelProvider`:

- `BallModel`
- `GlobalOpponentsModel`
- `ObstacleModel`
- `RobotPose`
- `TeammatesBallModel`

Evidencia local:

- `SabanaHerons2026/Config/Scenarios/4v4_RL2D/threads.cfg`
- `SabanaHerons2026/Config/Scenarios/4v4_RL3D/threads.cfg`

Eso significa que el agente aprende con acceso a:

- pose propia casi perfecta,
- pose de la pelota casi perfecta,
- obstaculos ideales,
- estado de equipo ideal.

En partido real eso no existe asi.

### Demasiado pobre

Aunque el estado es perfecto geometricamente, no representa bien el contexto futbolistico. No incluye:

- frescura real de la pelota,
- calidad de localizacion,
- consistencia con game state,
- trayectorias de pelota y riesgo,
- obstaculos resumidos de forma util para decidir,
- informacion de teammates para pase,
- rol, contexto tactico o presion,
- senales de incertidumbre.

Por eso el agente hoy no aprende una "decision futbolistica buena". Aprende una reaccion geometrica simple sobre un mundo idealizado.

## Por que el estado actual es demasiado perfecto

### 1. La pelota esta "demasiado sabida"

En partido real, la pelota llega a behavior a traves de un pipeline de percepcion y modelado. En escenarios normales de B-Human:

- `BallModel` viene de `BallStateEstimator`
- `FieldBall` viene de `FieldBallProvider`

Evidencia local:

- `SabanaHerons2026/Config/Scenarios/DefaultScaled/threads.cfg`
- `SabanaHerons2026/Config/Scenarios/Default/threads.cfg`

Y esas representaciones reales tienen informacion que hoy PPO no usa:

- `BallModel.timeWhenLastSeen`
- `BallModel.timeWhenDisappeared`
- `BallModel.seenPercentage`
- `FieldBall.timeSinceBallWasSeen`
- `FieldBall.timeSinceBallDisappeared`
- `FieldBall.ballPositionConsistentWithGameState`
- `FieldBall.interceptBall`
- `FieldBall.recentBallEndPositionRelative`

El estado actual colapsa todo eso en una sola geometria ideal.

Consecuencia:

- el agente aprende umbrales demasiado finos para `shoot`, `pass` o `dribble`,
- pero no aprende cuando NO debe confiar en la pelota.

### 2. La localizacion esta "demasiado sabida"

En escenarios normales:

- `RobotPose` viene de `SelfLocator`

Evidencia local:

- `SabanaHerons2026/Config/Scenarios/DefaultScaled/threads.cfg`

`RobotPose` real trae senales de confianza:

- `quality`
- `covariance`
- `timestampLastJump`

Evidencia local:

- `SabanaHerons2026/Src/Representations/Modeling/RobotPose.h`

El estado actual usa `robot_x`, `robot_y`, `robot_theta` como si fueran verdad absoluta.

Consecuencia:

- la politica aprende decisiones que dependen de pose exacta,
- pero no aprende a degradarse cuando la localizacion es `okay` o `poor`.

### 3. Los obstaculos estan "demasiado sabidos"

En escenarios normales:

- `ObstacleModel` viene de `ObstacleModelProvider`

Evidencia local:

- `SabanaHerons2026/Config/Scenarios/DefaultScaled/threads.cfg`
- `SabanaHerons2026/Src/Representations/Modeling/ObstacleModel.h`

En los escenarios RL actuales, `ObstacleModel` es oracular.

Consecuencia:

- una politica de `pass`, `dribble`, `mark` o `block` aprenderia con lineas limpias y sin errores de percepcion,
- pero en partido real las ventanas de pase y los carriles libres son inciertos.

### 4. El contexto de equipo esta "demasiado sabidamente limpio"

En escenarios normales:

- `TeammatesBallModel` viene de `TeammatesBallModelProvider`
- `TeamData` viene de `TeamDataProvider`
- `StrategyStatus` viene de `StrategyBehaviorControl`

Evidencia local:

- `SabanaHerons2026/Config/Scenarios/DefaultScaled/threads.cfg`
- `SabanaHerons2026/Src/Representations/Communication/TeamData.h`
- `SabanaHerons2026/Src/Representations/BehaviorControl/StrategyStatus.h`

En el laboratorio RL actual:

- parte de esta informacion es oracular,
- y `StrategyStatus` desde RL se vacia.

Consecuencia:

- hoy no existe una base honesta para una politica colectiva definitiva.

## Consecuencias practicas del estado demasiado perfecto

### A. Falsos positivos en entrenamiento

La politica parece buena porque:

- siempre sabe donde esta la pelota,
- siempre sabe donde esta ella misma,
- no sufre latencia,
- no sufre perdida visual,
- no sufre ambiguedad de pose.

### B. Umbrales fragiles

La politica aprende cosas como:

- "si la pelota esta a esta distancia, shoot"
- "si el angulo es este, dribble"

pero esos umbrales se rompen al menor ruido.

### C. Reward alto, futbol malo

Puede subir el reward porque:

- se acerca rapido a la pelota,
- dispara cuando geometricamente parece correcto,

pero en verdad:

- la pelota no estaba controlada,
- habia un rival en la linea,
- la localizacion era mala,
- el kick era inoportuno.

### D. Fallas silenciosas en hardware

Este es el peor caso.

La politica no se cae ni crashea. Solo toma decisiones malas con mucha confianza:

- dispara tarde,
- dispara temprano,
- intenta pasar sin ventana,
- camina cuando deberia observar,
- marca mal porque no sabe filtrar incertidumbre.

## Si el canary falla, que sintomas apuntan a estado demasiado perfecto

### Sintoma 1: en sim gana, en visible se ve tonto

Ejemplo:

- reward alto en headless,
- pero en `simrobot_visible` el robot duda, oscila o dispara en mal timing.

Diagnostico probable:

- el estado tenia geometria suficiente para reward,
- pero no suficiente semantica para una decision de skill buena.

### Sintoma 2: `shoot` aparece demasiado pronto

Diagnostico probable:

- faltan senales de control de pelota,
- falta frescura de ball model,
- falta nocion de ventana de kick limpia.

### Sintoma 3: `pass` parece absurdo

Diagnostico probable:

- no hay teammate freshness,
- no hay resumen de carril libre,
- no hay nocion de companero util o no util.

### Sintoma 4: `observe` casi nunca sale o sale demasiado

Diagnostico probable:

- el agente no ve incertidumbre, entonces nunca observa,
- o el reward castiga tanto las decisiones activas que observa por defecto.

### Sintoma 5: en 3D el behavior entra, pero la eleccion sigue siendo mala

Diagnostico probable:

- el pipeline de ejecucion esta bien,
- el problema esta en el estado o en el action abstraction,
- no en locomocion ni en MotionEngine.

## Restriccion de diseno para corregir esto

Este documento asume una restriccion fuerte:

```text
El observation space nuevo solo puede usar senales que B-Human pueda exponer en partido real.
```

Eso excluye:

- ground truth puro,
- teleports del simulador,
- poses del mundo directamente leidas de SimRobot,
- cualquier target que solo exista porque el escenario es oracular.

## Representaciones B-Human que SI son validas para partido real

Las siguientes representaciones si tienen sentido como base:

### Estado de juego

- `GameState`
- `ExtendedGameState`

Fuente local:

- `SabanaHerons2026/Src/Representations/Infrastructure/GameState.h`

### Pelota propia y fusionada para behavior

- `BallModel`
- `FieldBall`

Fuente local:

- `SabanaHerons2026/Src/Representations/Modeling/BallModel.h`
- `SabanaHerons2026/Src/Representations/BehaviorControl/FieldBall.h`

### Localizacion propia

- `RobotPose`

Fuente local:

- `SabanaHerons2026/Src/Representations/Modeling/RobotPose.h`

### Obstaculos

- `ObstacleModel`

Fuente local:

- `SabanaHerons2026/Src/Representations/Modeling/ObstacleModel.h`

### Estado del propio comportamiento

- `BehaviorStatus`

Fuente local:

- `SabanaHerons2026/Src/Representations/BehaviorControl/BehaviorStatus.h`

### Estado de equipo

- `TeamData`
- `TeammatesBallModel`
- `StrategyStatus`

Fuente local:

- `SabanaHerons2026/Src/Representations/Communication/TeamData.h`
- `SabanaHerons2026/Src/Representations/BehaviorControl/StrategyStatus.h`

## Principios para el nuevo observation space

1. Toda variable debe existir o poder derivarse de representaciones reales de B-Human.
2. La politica debe ver incertidumbre, no solo geometria.
3. La politica debe decidir skills, no controlar movimiento fino.
4. Los targets finos deben sintetizarse fuera de RL siempre que sea posible.
5. Para canary PPO, menos grados de libertad es mejor.
6. Para MAPPO futuro, ampliar contexto de equipo, no empezar por acciones continuas grandes.

## Observation space nuevo para PPO de seleccion de skill

### Perfil objetivo

Este PPO no pretende reemplazar la estrategia completa del partido. Pretende decidir:

- observar,
- caminar,
- disparar,
- pasar,
- driblear,
- bloquear,
- marcar.

La ejecucion la sigue haciendo B-Human.

### Forma recomendada

Recomendacion:

```text
Vector tabular estructurado de ~32 a 48 features
```

No usar imagen cruda en este canary.
No usar acciones continuas sin una abstraccion de targets.

### Bloques del estado PPO

#### Bloque 1: contexto de juego

Fuente:

- `GameState`
- `ExtendedGameState`

Features recomendadas:

1. `is_playing`
2. `is_ready`
3. `is_set`
4. `is_restart_for_own_team`
5. `is_restart_for_opponent`
6. `is_penalized`
7. `is_goalkeeper`

Motivo:

- una skill buena depende del contexto reglamentario.
- no deberia `shoot` igual en `playing` que en `waitForOwnKickOff`.

#### Bloque 2: confianza de localizacion

Fuente:

- `RobotPose`

Features recomendadas:

8. `robot_theta_sin`
9. `robot_theta_cos`
10. `robot_x_norm`
11. `robot_y_norm`
12. `loc_quality_superb`
13. `loc_quality_okay`
14. `loc_quality_poor`
15. `loc_std_x_norm`
16. `loc_std_y_norm`
17. `loc_std_theta_norm`
18. `time_since_pose_jump_norm`

Motivo:

- la politica debe saber si puede confiar en posicion y orientacion.
- si `loc_quality` es mala, debe ser mas conservadora.

#### Bloque 3: pelota propia y fusionada

Fuente:

- `BallModel`
- `FieldBall`

Features recomendadas:

19. `ball_seen_recently`
20. `time_since_ball_seen_norm`
21. `time_since_ball_disappeared_norm`
22. `ball_rel_x_norm`
23. `ball_rel_y_norm`
24. `ball_end_rel_x_norm`
25. `ball_end_rel_y_norm`
26. `ball_vel_x_norm`
27. `ball_vel_y_norm`
28. `ball_seen_percentage_norm`
29. `ball_consistent_with_game_state`
30. `team_ball_valid`
31. `team_ball_newer_than_own`

Motivo:

- la politica no solo necesita donde esta la pelota.
- necesita saber si esa pelota es confiable y util para decidir una skill.

#### Bloque 4: geometria ofensiva minima

Fuente derivada:

- `FieldBall`
- `RobotPose`
- `FieldDimensions`

Features recomendadas:

32. `dist_ball_norm`
33. `angle_ball_norm`
34. `dist_to_opponent_goal_norm`
35. `angle_to_opponent_goal_norm`
36. `angle_ball_to_goal_norm`
37. `ball_towards_opponent_goal`

Motivo:

- `shoot` y `dribble` necesitan mas que distancia a pelota.
- necesitan una lectura de oportunidad ofensiva.

#### Bloque 5: obstaculos resumidos

Fuente:

- `ObstacleModel`

Features recomendadas:

38. `nearest_obstacle_front_dist_norm`
39. `nearest_obstacle_left_dist_norm`
40. `nearest_obstacle_right_dist_norm`
41. `front_lane_blocked`
42. `goal_lane_blocked`
43. `obstacle_count_front_norm`

Motivo:

- `pass`, `dribble`, `mark` y `block` son decisiones pobres sin nocion de congestion.

#### Bloque 6: equipo y pase

Fuente:

- `TeamData`
- `BehaviorStatus`
- `StrategyStatus`

Features recomendadas:

44. `teammate_count_norm`
45. `best_pass_candidate_dist_norm`
46. `best_pass_candidate_angle_norm`
47. `best_pass_candidate_freshness_norm`
48. `best_pass_lane_free`
49. `incoming_pass_to_me`

Motivo:

- `pass` no puede decidirse con un solo `pass_target` ciego.
- la politica necesita al menos un resumen de disponibilidad real.

### Que NO meter en este PPO canary

No meter todavia:

- imagenes,
- mapas densos,
- targets continuos en campo completo,
- poses perfectas de rivales desde el simulador,
- variables puramente de debug de SimRobot,
- `frame` como feature tactica,
- `sim_enabled`.

## Observation space recomendado para MAPPO

MAPPO no deberia nacer de un PPO con el mismo estado y solo mas robots. Necesita otro nivel de informacion.

### Base MAPPO

Conservar todos los bloques PPO, pero por agente.

### Extensiones minimas MAPPO

Agregar por slots de teammates:

1. pose relativa resumida de los `K` teammates mas relevantes
2. frescura de cada teammate
3. velocidad/objetivo estimado desde `BehaviorStatus`
4. estado de pase:
   - `passTarget`
   - `passOrigin`
5. rol o posicion tactica disponible
6. resumen local de rivales cercanos
7. distancia del teammate al balon
8. distancia del teammate al arco rival

### CTDE sugerido

Para MAPPO, el critic centralizado si deberia ver:

- resumen de todos los teammates activos,
- resumen de todos los obstaculos relevantes,
- game state,
- team ball,
- score y tiempo si se usa esa senal.

## Action space minimo recomendado

## Regla principal

No dejar que PPO emita `target_x`, `target_y`, `target_theta` libres en el primer canary.

Si RL decide skill y B-Human ejecuta, el primer canary debe usar:

- skill discreta,
- y parametros sintetizados por una capa determinista.

Eso reduce brutalmente la fragilidad.

## Action space minimo para PPO canary

Recomendacion:

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

### Como se parametriza cada skill sin volver a accion continua

La capa adaptadora debe sintetizar targets asi:

- `observe` -> punto de observacion o scan target predefinido
- `walk` -> target sintetizado segun contexto:
  - pelota actual
  - fin estimado de la pelota
  - pose de approach de tiro
  - pose de soporte
- `shoot` -> `SkillRequest::shoot()`
- `pass` -> mejor companero elegible segun filtro externo
- `dribble` -> direccion discretizada:
  - al arco
  - escape izquierda
  - escape derecha
- `block` -> target defensivo sintetizado sobre carril de pelota
- `mark` -> target sintetizado sobre rival peligroso o zona objetivo

Esto sigue siendo "RL decide la skill", pero evita que RL decida mal parametros finos todavia.

## Action space minimo para MAPPO

Cuando el estado de equipo exista de verdad, ampliar a:

```text
skill_id: Discrete(7)
walk_target_mode: Discrete(4 a 6)
pass_target_slot: Discrete(K + 1)
dribble_mode: Discrete(3)
```

Eso sigue siendo muchisimo mas estable que una `Box` continua grande.

## Tabla skill-by-skill: que senales necesita RL para decidir bien

| Skill | Senales minimas necesarias | Por que importan | Parametrizacion minima recomendada | Extension util para MAPPO |
| --- | --- | --- | --- | --- |
| `walk` | `ball_seen_recently`, `ball_rel_x/y`, `ball_end_rel_x/y`, `loc_quality`, `nearest_obstacle_*`, `front_lane_blocked` | caminar bien no es solo ir a la pelota; depende de confianza de pelota, pose y congestion | target sintetizado a `ball_now`, `ball_end` o `shooting_pose` | teammate support pose y presion rival |
| `shoot` | `dist_ball`, `angle_ball_to_goal`, `goal_lane_blocked`, `ball_seen_recently`, `ball_consistent_with_game_state`, `loc_quality` | disparar bien depende de control, ventana y contexto | sin target continuo; solo `shoot` cuando el gating externo lo permita | rol, score, mejor opcion entre shoot/pass |
| `pass` | `best_pass_candidate_*`, `best_pass_lane_free`, `team_ball_valid`, `ball_seen_recently`, `loc_quality` | pasar sin companero fresco ni linea libre es mala politica | target sintetizado al mejor companero elegible | eleccion explicita de companero via slot discreto |
| `dribble` | `dist_ball`, `angle_to_goal`, `goal_lane_blocked`, `nearest_obstacle_front`, `sideline_risk`, `ball_seen_recently` | dribble requiere control de pelota y carril util | modo discreto `goal`, `left_escape`, `right_escape` | coordinacion con support runner |
| `block` | `interceptBall`, `timeUntilIntersectsOwnYAxis`, `distanceToOwnPenaltyArea`, `isRollingTowardsOwnGoal`, `role`, `game_state` | bloquear es una decision temporal y defensiva, no solo espacial | target sintetizado al carril de intercepcion | asignacion colectiva de quien bloquea |
| `mark` | `nearest_dangerous_opponent`, `opponent_ball_relation`, `obstacle_density`, `role`, `team_shape` | marcar requiere nocion de amenaza y contexto colectivo | target sintetizado a rival/zona prioritaria | reparto multiagente de marcas |
| `observe` | `time_since_ball_seen`, `loc_quality`, `ball_disappeared`, `uncertainty flags`, `no_safe_lane` | observar debe aparecer cuando falta informacion o no hay accion segura | scan target fijo o zona de interes predefinida | observacion coordinada con teammates |

## Gating externo recomendado antes de llamar a la policy

Antes de preguntarle nada a PPO:

1. si el robot esta penalizado, no preguntar
2. si el game state no permite accion, no preguntar
3. si `loc_quality` es muy pobre, degradar a `observe` o heuristica
4. si la pelota no fue vista hace demasiado, degradar a `observe`
5. si el rol no es elegible para esa skill, enmascararla

Esto no contradice RL. Lo protege.

## Politica de skill masks recomendada

### Canary minimo de 3 dias

Habilitar solo:

- `observe`
- `walk`
- `shoot`

Opcional cuarto skill:

- `dribble`

No meter `pass`, `mark` o `block` en el primer canary salvo que ya existan senales fiables resumidas para ellos.

### PPO ampliado

Habilitar:

- `observe`
- `walk`
- `shoot`
- `pass`
- `dribble`

### MAPPO

Agregar:

- `block`
- `mark`

cuando exista:

- estado multiagente serio,
- criterio de asignacion,
- evaluacion colectiva.

## Como solucionaria el problema del estado demasiado perfecto

## Fase 1: dejar de entrenar con el vector actual como politica final

El vector actual puede servir para humo tecnico.
No debe servir como contrato final de decision.

## Fase 2: exponer representaciones reales de B-Human al bridge RL

Extender `rl_get_obs()` o crear un payload nuevo con:

- `GameState`
- `RobotPose` con calidad/covarianza
- `BallModel`
- `FieldBall`
- resumen de `ObstacleModel`
- resumen de `TeamData`
- resumen de `BehaviorStatus`

No hace falta exportar todo crudo.
Hace falta exportar un resumen estable y util.

## Fase 3: reemplazar geometria perfecta por geometria confiable

En vez de:

- `ball_x`, `ball_y` absolutos perfectos
- `robot_x`, `robot_y` absolutos perfectos

usar:

- posiciones relativas,
- medidas de frescura,
- medidas de confianza,
- resumentes tacticos derivados.

## Fase 4: mantener targets sintetizados fuera de RL

El primer PPO no debe aprender:

- a donde exactamente caminar en todo el campo,
- a que orientacion exacta disparar,
- a quien pasar en un espacio continuo libre.

Debe aprender:

- que skill conviene entre opciones discretas razonables.

## Fase 5: usar `simrobot_visible` como arbitro cualitativo

Si la politica da reward alto pero visualmente:

- oscila,
- dispara mal,
- observa cuando no debe,
- ignora rivales,

el problema casi seguro esta arriba:

- estado,
- reward,
- action abstraction,

no abajo:

- locomocion,
- walking engine,
- joints.

## Checklist de diagnostico si el canary falla

### Si falla por malas decisiones de `shoot`

Revisar:

- frescura del balon,
- control de balon,
- angulo libre al arco,
- gating de kickable range.

### Si falla por malas decisiones de `pass`

Revisar:

- freshness de teammates,
- ocupacion de carril,
- criterio externo de mejor receptor,
- action space demasiado libre.

### Si falla por caminar sin sentido

Revisar:

- si el target de `walk` se sintetiza bien,
- si falta distincion entre pelota actual y pelota futura,
- si la localizacion pobre no esta castigada.

### Si falla por observar demasiado

Revisar:

- reward demasiado defensivo,
- ausencia de hysteresis,
- umbral excesivo para confiar en pelota/localizacion.

### Si falla por no observar nunca

Revisar:

- entrenamiento demasiado oracular,
- falta de `time_since_ball_seen`,
- falta de `loc_quality`,
- falta de castigo por actuar con informacion vieja.

## Conclusiones operativas

1. El pipeline de ejecucion actual es correcto para este proyecto.
2. El gran problema actual no es `MotionEngine`.
3. El gran problema actual es que el estado mezcla:
   - informacion demasiado perfecta,
   - con semantica insuficiente.
4. Para un canary PPO en 3 dias, el camino mas estable es:
   - skill discreta,
   - pocas skills,
   - targets sintetizados,
   - observation space realista de B-Human.
5. MAPPO no debe arrancar hasta que el estado de equipo y la evaluacion colectiva sean honestos.
