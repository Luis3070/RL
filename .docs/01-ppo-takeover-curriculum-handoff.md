# PPO takeover handoff para RL sobre B-Human / SimRobot 3D

## Propósito

Este documento contextualiza y consolida la línea de trabajo definida para el
sprint PPO del stack `RL` actual. La meta no es rediseñar la arquitectura, sino
usar la integración ya validada entre Python y B-Human para entrenar una
política operativa sobre `simrobot` perfil `3d`, tratando `MAPPO(1 agente)` como
`PPO` práctico.

## Premisa central

La base útil que ya existe no es una política "inteligente" completa, sino una
propiedad del stack:

- si RL emite un `SkillRequest` coherente,
- en el momento correcto,
- dentro de una geometría compatible,
- y sin oscilación de intención,

entonces B-Human ejecuta razonablemente bien la skill.

Por eso el problema de aprendizaje del sprint no es descubrir locomoción ni
control fino desde cero. El problema es aprender a:

1. acercarse al balón con intención estable,
2. entrar al `takeover_envelope`,
3. sostener el takeover sin incoherencias,
4. cerrar con `shoot` o `dribble` según contexto.

## Definición operativa de takeover

Se adopta esta definición como verdad de trabajo:

`takeover = RL lleva la situación a un punto donde el SkillRequest que emite produce buena ejecución en B-Human`

Eso implica:

- RL sigue decidiendo la skill,
- B-Human sigue ejecutándola,
- el request RL debe ser coherente, oportuno, estable y compatible con el
  estado real del balón y del cuerpo.

Versión operacional:

- RL coloca al robot en una situación desde la cual su `SkillRequest` actual
  produce una ejecución limpia y consistente en B-Human,
- sin oscilación,
- sin cambio errático de intención,
- sin pedir una skill incompatible con la situación real.

## Restricciones cerradas para el sprint

- backend objetivo: `simrobot`
- perfil objetivo: `3d`
- mantener el action contract actual
- mantener fijo `obs space`
- mantener fijo `action space`
- solo cambian por stage:
  - distribución de resets
  - presencia de obstáculos
  - dificultad del escenario
  - masks
  - pesos del reward

## Action set PPO del sprint

Skills activas reales de Stage 0 a Stage 4:

- `walk`
- `dribble`
- `shoot`

Skills enmascaradas:

- `pass`
- `mark`
- `block`
- `observe`

El `Box` de acción se mantiene igual por integración. La regla operativa es:

- decodificar `skill_type` con el contrato actual,
- proyectarlo al subconjunto activo del stage.

## Qué ya aporta la heurística actual

La heurística existente no se toma como policy final, sino como sesgo útil de
encarrilamiento.

### `live_simrobot_probe.py`

La heurística del probe visual aporta dos ideas prácticas:

- confiabilidad mínima de pelota,
- fallback de búsqueda cuando la pelota no es confiable.

Su comportamiento base es:

- si la pelota no es confiable, busca hacia la pose de referencia;
- si la pelota es confiable:
  - camina hacia la pelota mirando a la pelota,
  - dispara al cruzar un umbral bruto.

Esto sirve como prior de búsqueda y approach, pero no como policy contextual
rica porque no usa `dribble` de forma útil.

### `heuristic_agent_experiment.py`

La heurística más cercana al takeover útil es la del `handoff offset`:

- define un offset útil detrás del balón,
- camina hacia una pose de handoff en lugar de solo perseguir la pelota,
- mantiene orientación hacia el balón,
- cierra cuando la geometría ya es compatible.

Esa idea es la semilla conceptual correcta para shaping:

- no basta con acercarse al balón,
- hay que acercarse de una forma que deje a B-Human en situación de ejecutar
  bien el request actual.

## Principio de aprendizaje adoptado

La política del sprint se guía por este principio:

- heurística para encarrillar,
- PPO para cerrar robustez y selección contextual.

En práctica:

- la heurística empuja,
- el reward refuerza,
- el gate filtra la promoción.

## Helpers fijos que definen la semántica del takeover

Los helpers no deben cambiar entre stages. Definen la semántica universal del
problema.

### Derivadas operativas

- `d_ball = ||ball_rel||`
- `a_ball = abs(atan2(ball_rel_y, ball_rel_x))`
- `t_seen = time_since_ball_seen`
- `seen_pct = ball_seen_percentage`
- `score_now = can_score_now`
- `shot_open = shot_opening_with_obstacles`
- `opp_front = nearest_opponent_front_dist`

### `takeover_envelope`

- `d_ball <= 450 mm`
- `a_ball <= 35 deg`
- `t_seen <= 400 ms`
- `seen_pct >= 0.4`

### `walk zone`

- no `takeover_envelope`
- `t_seen <= 1000 ms`
- `seen_pct >= 0.2`

### `shoot zone`

- `d_ball <= 320 mm`
- `a_ball <= 20 deg`
- `t_seen <= 300 ms`
- `seen_pct >= 0.5`
- `shot_open >= 0.7` o `score_now == 1`

### `dribble zone`

- dentro de `takeover_envelope`
- y no dentro de `shoot zone`

## Heurísticas blandas de encarrilamiento

Estas heurísticas no sustituyen la policy. Deben entrar como sesgo blando:

- favorecer `walk` si aún no está listo para takeover,
- premiar acercarse mirando y alineando hacia el balón,
- penalizar approach con heading incoherente,
- favorecer `dribble` cuando hay takeover pero no ventana clara de tiro,
- favorecer `dribble` con presión frontal moderada,
- favorecer `shoot` solo dentro de `shoot zone`,
- penalizar skills incompatibles con la zona,
- penalizar alternancias locales de intención.

Patrones explícitos de thrash:

- `walk -> shoot -> walk`
- `walk -> dribble -> walk`
- `shoot -> walk -> shoot`
- `dribble -> walk -> dribble`

## Curriculum reducido del sprint

### Stage 0

- balón a radio `<= 800 mm`
- sin obstáculos
- skills activas: `walk`, `dribble`, `shoot`
- objetivo:
  - approach limpio
  - takeover limpio
  - finalización limpia

### Stage 1

- balón a radio `<= 1600 mm`
- sin obstáculos
- skills activas: `walk`, `dribble`, `shoot`
- objetivo:
  - misma tarea que Stage 0
  - mayor exigencia temporal

### Stage 2

- balón a radio `<= 3200 mm`
- obstáculos estáticos
- skills activas: `walk`, `dribble`, `shoot`
- objetivo:
  - takeover limpio con clearance
  - sin colisión
  - `dribble` como escape útil

### Stage 3

- campo completo
- obstáculos dinámicos scripted
- skills activas: `walk`, `dribble`, `shoot`
- objetivo:
  - robustez temporal
  - evitar cierre de corredor
  - takeover limpio bajo presión

### Stage 4

- `1v1`
- rival scripted o congelado al inicio
- skills activas: `walk`, `dribble`, `shoot`
- objetivo:
  - llegar antes al balón
  - takeover limpio
  - progresar tras takeover
  - evitar duelos malos

## Reward adoptado

El reward se compone por step:

`R = sum(w_i * term_i)`

### Términos base

- `progress_ball`
- `align_ball`
- `enter_takeover`
- `hold_takeover`
- `good_walk`
- `bad_walk`
- `good_dribble`
- `bad_dribble`
- `good_shoot`
- `bad_shoot`
- `skill_switch`
- `thrash`
- `goal`
- `timeout`

### Términos adicionales por dificultad

Stage 2+:

- `collision`
- `clearance_gain`

Stage 3+:

- `corridor_keep`
- `ball_progress_post_takeover`

Stage 4:

- `first_to_ball`
- `bad_duel`

### Pseudocódigo operativo

El reward se concibe con esta lógica:

- calcular zonas a partir de balón, ángulo, frescura y apertura,
- medir progreso y alineación,
- detectar entrada y sostén de takeover,
- premiar skill compatible con la zona,
- castigar skill incompatible,
- castigar `thrash`,
- activar términos extra solo cuando el stage lo permita.

Detalles operativos fijados:

- `progress_ball = clip((d_prev - d_cur) / 250, -1, 1)`
- `align_ball = clip((abs(a_prev) - abs(a_cur)) / 15deg, -1, 1)`
- `enter_takeover = 1` al entrar por primera vez
- `hold_takeover = 1` al sostener takeover
- `thrash = 1` ante oscilación local de skills

### Regla práctica si hay que recortar

Si toca simplificar por tiempo, se preserva este núcleo:

1. `progress_ball`
2. `align_ball`
3. `enter_takeover`
4. `good/bad_shoot`
5. `good/bad_dribble`
6. `thrash`

## Conexión entre helpers, reward y curriculum

La semántica del takeover no cambia entre stages. Lo que cambia por stage es el
contexto.

### Los helpers quedan fijos

Viven en `reward.py` y definen:

- qué significa takeover,
- cuándo `walk` es compatible,
- cuándo `dribble` es compatible,
- cuándo `shoot` es compatible,
- cuándo hay thrash.

### El curriculum cambia el contexto

Vive en `curriculum.py` y define:

- radio o distribución de reset,
- presencia o ausencia de obstáculos,
- dificultad del escenario,
- máscaras activas,
- pesos del reward,
- gates de promoción.

En otras palabras:

- misma definición de "hacerlo bien",
- mundos cada vez más difíciles.

## Gates de promoción

### Stage 0

Promociona si:

- `clean_walk_rate >= 0.95`
- `clean_takeover_rate >= 0.95`
- `clean_finish_rate >= 0.95`
- `thrash_rate <= 0.05`

### Stage 1

Promociona si:

- `clean_walk_rate >= 0.90`
- `clean_takeover_rate >= 0.90`
- `clean_finish_rate >= 0.90`
- mejora en `mean_time_to_takeover`
- `thrash_rate <= 0.08`

### Stage 2

Promociona si:

- `clean_takeover_rate >= 0.85`
- `clean_finish_rate >= 0.85`
- `collision_rate <= 0.05`
- `obstacle_clearance_success >= 0.85`
- `thrash_rate <= 0.10`

### Stage 3

Promociona si:

- `clean_episode_rate >= 0.80`
- `collision_rate <= 0.08`
- `dynamic_obstacle_success > 0.50`
- `stability_score >= stage_2_stability_score`

### Stage 4

Promociona si:

- `win_rate > 0.55`
- `takeover_success_rate >= 0.75`
- `bad_duel_rate <= 0.20`
- `stability_score >= stage_3_stability_score`

## Métricas mínimas a loggear

- `clean_walk_rate`
- `clean_takeover_rate`
- `clean_finish_rate`
- `thrash_rate`
- `collision_rate`
- `mean_time_to_takeover`
- `obstacle_clearance_success`
- `dynamic_obstacle_success`
- `clean_episode_rate`
- `win_rate`
- `takeover_success_rate`
- `bad_duel_rate`
- `stability_score`

## Sanidad del entrenamiento

La lectura de sanidad del entrenamiento no debe depender solo del reward medio.
Hay que separar:

- métricas de comportamiento objetivo,
- métricas de compatibilidad skill-contexto,
- métricas específicas del stage,
- y métricas de salud PPO.

Regla práctica:

- si mejora optimización pero no mejora comportamiento en evaluación, no se está
  comprando comportamiento útil;
- si sube reward pero no suben `clean_takeover_rate` y `clean_finish_rate`,
  probablemente hay shaping inflado o reward hacking.

### Tabla de `must log` y `nice to have`

| Tipo | Métrica | Prioridad | Para qué sirve |
|---|---|---|---|
| comportamiento | `clean_walk_rate` | must log | validar approach limpio |
| comportamiento | `clean_takeover_rate` | must log | validar entrada y sostén de takeover |
| comportamiento | `clean_finish_rate` | must log | validar cierre útil |
| comportamiento | `thrash_rate` | must log | detectar oscilación de intención |
| comportamiento | `mean_time_to_takeover` | must log | medir eficiencia temporal |
| comportamiento | `takeover_success_rate` | must log | medir takeover útil a nivel episodio |
| compatibilidad | `walk_in_walk_zone_rate` | must log | ver si `walk` se usa cuando corresponde |
| compatibilidad | `dribble_in_dribble_zone_rate` | must log | ver si `dribble` se usa contextualmente |
| compatibilidad | `shoot_in_shoot_zone_rate` | must log | ver si `shoot` se usa bien gateado |
| compatibilidad | `bad_walk_rate` | must log | detectar `walk` tardío |
| compatibilidad | `bad_dribble_rate` | must log | detectar `dribble` absurdo |
| compatibilidad | `bad_shoot_rate` | must log | detectar `shoot` prematuro |
| stage | `collision_rate` | must log | obligatorio desde Stage 2 |
| stage | `obstacle_clearance_success` | must log | validar clearance útil en Stage 2 |
| stage | `dynamic_obstacle_success` | must log | validar robustez en Stage 3 |
| stage | `clean_episode_rate` | must log | criterio fuerte de estabilidad por episodio |
| stage | `win_rate` | must log | objetivo competitivo de Stage 4 |
| stage | `bad_duel_rate` | must log | validar que no entra a duelos malos |
| stage | `stability_score` | must log | continuidad inter-stage |
| optimización | `episode_reward_mean` | must log | referencia global, no criterio único |
| optimización | `policy_loss` | must log | detectar updates inestables |
| optimización | `value_loss` | must log | detectar critic ruidoso o roto |
| optimización | `entropy` | must log | vigilar colapso prematuro de exploración |
| compatibilidad | `skill_switch_rate` | nice to have | distinguir cambio útil de thrash |
| compatibilidad | `skill_distribution` | nice to have | detectar colapso a una sola skill |
| stage | `corridor_keep_rate` | nice to have | Stage 3+, mantener corredor útil |
| stage | `ball_progress_post_takeover_mean` | nice to have | Stage 3-4, medir valor tras takeover |
| stage | `first_to_ball_rate` | nice to have | Stage 4, ventaja de llegada |
| optimización | `clip_fraction` | nice to have | vigilar si PPO está pegando demasiado al clip |
| optimización | `approx_kl` | nice to have | medir drift de política entre updates |
| optimización | `grad_norm` | nice to have | detectar explosión o updates raros |

### Lectura práctica de la tabla

Señales de aprendizaje sano:

- sube `clean_takeover_rate`,
- sube `clean_finish_rate`,
- baja `thrash_rate`,
- baja `mean_time_to_takeover`,
- suben las tasas de skill usada en zona compatible,
- baja `collision_rate` cuando aparecen obstáculos,
- y esas mejoras se sostienen en evaluación corta, no solo en train.

Señales de inflación o reward hacking:

- sube `episode_reward_mean` pero no mejora `clean_finish_rate`,
- baja `thrash_rate` porque la policy colapsa a una sola skill,
- sube `bad_shoot_rate` o `bad_dribble_rate`,
- cae `entropy` demasiado pronto,
- mejora train pero no mejora eval,
- el agente evita riesgo sin acercarse al takeover real.

## Definiciones operativas mínimas de métricas

### `clean_walk`

- baja distancia al balón
- baja error angular al balón
- sin thrash

### `clean_takeover`

- entra a `takeover_envelope`
- lo sostiene por varios steps
- no hay request incompatible

### `clean_finish`

- `good_shoot` o `good_dribble`
- ejecución estable
- progreso final o gol

## Decisión de implementación para el sprint

Para minimizar dispersión y acelerar integración:

- helpers y reward viven en `reward.py`
- curriculum, masks, gates y stage config viven en `curriculum.py`

No se abre un módulo adicional solo para takeover salvo que el repo lo obligue.

## Resumen operativo final

La estrategia del sprint queda así:

1. mantener arquitectura actual RL x B-Human,
2. mantener `obs space` y `action space`,
3. entrenar sobre `simrobot` `3d`,
4. usar heurística como encarrilamiento blando,
5. definir takeover con helpers fijos,
6. reforzar esa semántica con reward por stage,
7. promover solo con gates medibles.

Principio final:

- heurística empuja,
- reward refuerza,
- gate filtra promoción.
