# Instrumentation para PPO takeover y métricas de sanidad

Este documento complementa:

- [01-ppo-takeover-curriculum-handoff.md](./01-ppo-takeover-curriculum-handoff.md)

Su objetivo es aterrizar:

1. necesidades de test y verificación antes y durante entrenamiento,
2. necesidades de implementación de telemetría,
3. qué señales del repo actual son confiables para construir reward y métricas,
4. análisis más profundo de las métricas de sanidad,
5. y decisiones de alcance para el sprint.

## Regla de instrumentación para este sprint

El reward debe trabajar con lo que ya funciona hoy en el repo y en las
visualizaciones heurísticas conectadas a `simrobot` `3d`.

Eso significa:

- priorizar señales ya publicadas por `rl_get_obs(...)`,
- priorizar señales ya observadas en `live_simrobot_probe.py`,
- priorizar señales ya usadas por las heurísticas que sabemos que caminan sobre
  SimRobot 3D,
- evitar depender, en la primera iteración, de variables nuevas que aún no estén
  instrumentadas de punta a punta.

## Señales confiables hoy para reward y sanidad

Estas señales ya están en el stack Python actual y forman la base segura del
reward:

- `ball_rel_x`
- `ball_rel_y`
- `time_since_ball_seen`
- `ball_seen_percentage`
- `can_score_now`
- `shot_opening_with_obstacles`
- `nearest_opponent_front_dist`
- `nearest_uncertain_front_dist`
- `requested_skill`
- `frame`
- posición del robot
- posición de la pelota

Además, el ecosistema de probes ya validó:

- detección funcional de obstáculos en scenario RL,
- avoidance activo en la cadena natural,
- integración real con `simrobot` `3d`,
- ejecución de `walk`, `dribble` y `shoot` vía `SkillRequest`.

## Señales que deben tratarse como derivadas o provisionales

Para este sprint, las siguientes señales no deben asumirse como oro si no están
ya publicadas o verificadas en el bridge actual:

- `collision`
- `corridor_open`
- `first_to_ball`
- `bad_duel`
- éxito exacto de clearance
- métricas finas de 1v1

La regla práctica es:

- si una señal no existe de forma fiable en el repo actual, se deriva en Python
  si es razonable;
- si no se puede derivar con suficiente estabilidad, se desactiva del reward del
  sprint inicial;
- no se bloquea el arranque de PPO por querer instrumentar Stage 4 demasiado
  pronto.

## Necesidades de test y verificación

Antes de iniciar un entrenamiento largo de PPO, hay que validar cuatro capas.

### 1. Verificación de observación

Objetivo:

- asegurar que el reward ve valores coherentes y no ruido mal escalado.

Checks mínimos:

- `ball_rel_x/y` cambian de forma consistente al mover robot o balón,
- `time_since_ball_seen` sube cuando la pelota deja de ser confiable,
- `ball_seen_percentage` responde al caso de visión real,
- `shot_opening_with_obstacles` cambia cuando hay congestión,
- `nearest_opponent_front_dist` y `nearest_uncertain_front_dist` responden con
  obstáculos delante.

Tests deseados:

- smoke test de lectura de `BackendObservation`,
- test de encoding si el env sigue normalizando,
- test de derivadas del reward con ejemplos sintéticos.

### 2. Verificación de reward

Objetivo:

- probar que el reward refuerza la conducta buscada y no una conducta espuria.

Checks mínimos:

- acercarse al balón mejora `progress_ball`,
- alinear al balón mejora `align_ball`,
- entrar a envelope dispara `enter_takeover`,
- pedir `shoot` lejos del balón dispara `bad_shoot`,
- pedir `dribble` fuera de envelope dispara `bad_dribble`,
- patrones `walk -> shoot -> walk` disparan `thrash`.

Tests deseados:

- tests deterministas de `is_walk_zone`, `is_dribble_zone`, `is_shoot_zone`,
- tests de reward por contexto manual,
- tests de historial de skills para thrash.

### 3. Verificación de masks y proyección

Objetivo:

- asegurar que PPO no se salga del subconjunto activo del sprint.

Checks mínimos:

- Stage 0-4 solo dejan activas `walk`, `dribble`, `shoot`,
- `pass`, `mark`, `block`, `observe` quedan enmascaradas,
- la proyección de `skill_type` funciona de forma determinista.

Tests deseados:

- test unitario de `stage_skill_mask`,
- test unitario de proyección desde skills inactivas.

### 4. Verificación de curriculum

Objetivo:

- asegurar que cada stage realmente cambia dificultad y no solo el nombre.

Checks mínimos:

- Stage 0 genera resets dentro de `<= 800 mm`,
- Stage 1 dentro de `<= 1600 mm`,
- Stage 2 dentro de `<= 3200 mm`,
- Stage 3 usa campo completo,
- los obstáculos aparecen según el stage esperado.

Tests deseados:

- test del sampler de resets por stage,
- smoke test de stage config y weights.

## Necesidades de telemetría

La telemetría del sprint debe servir para responder:

- ¿está aprendiendo el comportamiento correcto?
- ¿solo está inflando shaping?
- ¿el critic/policy están estables?
- ¿la promoción de stage es merecida?

### Telemetría por step

Conviene poder registrar o acumular:

- `skill`
- `prev_skill`
- `ball_dist_mm`
- `ball_angle_rad`
- `time_since_ball_seen_ms`
- `ball_seen_percentage`
- `shot_opening_with_obstacles`
- `nearest_opponent_front_dist_mm`
- `in_takeover_envelope`
- `in_walk_zone`
- `in_dribble_zone`
- `in_shoot_zone`
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

No hace falta persistir todo por siempre, pero sí acumularlo por episodio y por
ventana de evaluación.

### Telemetría por episodio

Debe existir un resumen por episodio con:

- `clean_walk`
- `clean_takeover`
- `clean_finish`
- `takeover_entered`
- `takeover_held_steps`
- `goal`
- `timeout`
- `collision_count`
- `thrash_count`
- `skill_counts`
- `time_to_takeover_steps`
- `mean_ball_progress`
- `mean_align_ball`

### Telemetría por ventana de evaluación

Cada evaluación corta debería emitir:

- `clean_walk_rate`
- `clean_takeover_rate`
- `clean_finish_rate`
- `thrash_rate`
- `collision_rate`
- `mean_time_to_takeover`
- `walk_in_walk_zone_rate`
- `dribble_in_dribble_zone_rate`
- `shoot_in_shoot_zone_rate`
- `bad_walk_rate`
- `bad_dribble_rate`
- `bad_shoot_rate`
- `episode_reward_mean`
- `policy_loss`
- `value_loss`
- `entropy`

## Diferencia entre proyectar a `preferred_skill` o a `walk`

Cuando `skill_type` sale fuera del subconjunto activo del sprint, hay dos
políticas de proyección posibles.

### Proyección a `walk`

Regla:

- toda skill inactiva cae a `walk`

Ventajas:

- implementación muy simple,
- comportamiento conservador,
- reduce efectos extraños al principio.

Desventajas:

- sesga fuertemente la distribución de acciones hacia `walk`,
- puede esconder errores reales de policy,
- castiga menos a una policy que insiste en skills inválidas,
- dificulta distinguir colapso real de una corrección artificial,
- si ya está en `shoot zone` o `dribble zone`, `walk` puede ser semánticamente
  incorrecto.

### Proyección a `preferred_skill`

Regla:

- si una skill sale inactiva, se reemplaza por la skill compatible con el
  contexto actual:
  - `shoot` si está en `shoot zone`
  - `dribble` si está en `dribble zone`
  - `walk` en otro caso

Ventajas:

- respeta mejor la semántica del takeover,
- produce una corrección contextual en vez de una corrección ciega,
- reduce castigos artificiales cuando el agente ya llegó a la zona correcta,
- preserva la lógica del sprint: `walk` fuera, `dribble` en takeover sin tiro,
  `shoot` al cerrar.

Desventajas:

- mete una heurística más fuerte en la interfaz de acción,
- puede ocultar un poco más que la policy original estaba proponiendo una skill
  inválida,
- complica ligeramente la atribución causal del reward.

### Decisión recomendada

Para este sprint, la proyección a `preferred_skill` es más consistente con el
objetivo operacional del takeover.

Razón:

- el sprint no busca aprender la diferencia entre ocho skills;
- busca aprender a estabilizar `walk`, `dribble` y `shoot` en el contexto
  correcto;
- una proyección contextual conserva mejor esa semántica.

Regla de sanidad:

- aun si se proyecta a `preferred_skill`, debe loggearse cuántas veces la policy
  pidió una skill inactiva, para no perder visibilidad del error original.

## Análisis profundo de métricas de sanidad

Las métricas deben leerse en grupos, no aisladas.

## 1. Métricas de comportamiento

### `clean_walk_rate`

Qué mide:

- approach limpio y estable antes de takeover.

Qué significa si sube:

- el agente se acerca mejor al balón sin oscilar.

Qué no garantiza:

- que sepa cerrar takeover o finalizar.

Riesgo de falsa mejora:

- la policy puede caminar mucho y parecer "limpia" sin realmente entrar a
  takeover.

### `clean_takeover_rate`

Qué mide:

- cuántos episodios o intentos realmente llegan a envelope y lo sostienen sin
  requests incompatibles.

Es la métrica más importante del sprint temprano.

Si esta no mejora, el reward puede estar inflado aunque el retorno suba.

### `clean_finish_rate`

Qué mide:

- cierre contextual correcto y estable.

Es la mejor defensa contra reward hacking de "llego cerca pero no cierro bien".

### `takeover_success_rate`

Qué mide:

- takeover útil a nivel episodio.

Conviene usarla como métrica de resumen más robusta que el reward agregado.

### `mean_time_to_takeover`

Qué mide:

- eficiencia temporal del approach.

Lectura correcta:

- solo importa si no empeora limpieza o estabilidad.

Si baja mucho pero también baja `clean_takeover_rate`, la policy puede estar
apresurando requests malos.

### `thrash_rate`

Qué mide:

- oscilación local de intención.

Es crítica porque el takeover buscado exige estabilidad.

Riesgo de interpretación:

- puede bajar por colapso a una sola skill.

Por eso siempre debe leerse junto con:

- `skill_distribution`
- tasas de compatibilidad por zona

## 2. Métricas de compatibilidad skill-contexto

### `walk_in_walk_zone_rate`

Qué mide:

- si `walk` se usa donde tiene sentido.

### `dribble_in_dribble_zone_rate`

Qué mide:

- si `dribble` aparece como skill de takeover útil y no como ruido.

### `shoot_in_shoot_zone_rate`

Qué mide:

- si `shoot` dejó de ser un reflejo prematuro.

Estas tres métricas son el mejor test de "selección contextual real".

### `bad_walk_rate`, `bad_dribble_rate`, `bad_shoot_rate`

Qué miden:

- requests incompatibles con la zona.

Son esenciales para detectar reward hacking. Ejemplos:

- `bad_shoot_rate` alta con reward creciente sugiere que el shaping de progreso
  está dominando al cierre correcto;
- `bad_dribble_rate` alta sugiere que el agente usa `dribble` como fallback
  incorrecto;
- `bad_walk_rate` alta sugiere que no aprende a cerrar.

## 3. Métricas por stage

### `collision_rate`

Se vuelve obligatoria desde Stage 2.

Hay que leerla siempre junto con:

- `clean_takeover_rate`
- `obstacle_clearance_success`

Si `collision_rate` baja pero también cae `clean_takeover_rate`, la política
puede estar evitando el problema en vez de resolverlo.

### `obstacle_clearance_success`

Mide si el agente atraviesa escenas con obstáculo sin arruinar el takeover.

No debe confundirse con "simplemente no chocar". El punto es conservar progreso
hacia takeover con clearance útil.

### `dynamic_obstacle_success`

En Stage 3 mide robustez bajo presión variable.

Debe leerse con `stability_score` y `thrash_rate`.

### `clean_episode_rate`

Resume si un episodio completo fue razonable:

- approach limpio,
- takeover limpio,
- cierre útil,
- sin colisión severa.

Es muy buena métrica de gate porque es difícil inflarla con un solo truco.

## 4. Métricas de optimización PPO

### `episode_reward_mean`

Sirve como referencia global, pero nunca debe ser criterio único.

### `policy_loss`

Ayuda a ver si las actualizaciones se vuelven erráticas.

### `value_loss`

Ayuda a ver si el critic está persiguiendo una señal demasiado ruidosa.

### `entropy`

Mide exploración.

Lectura sana:

- debe bajar gradualmente, no desplomarse de inmediato.

Si cae demasiado pronto:

- la policy puede colapsar a `walk`,
- o quedarse en una subestrategia cómoda.

### `clip_fraction`, `approx_kl`, `grad_norm`

Son útiles, pero secundarios para el sprint.

Sirven para confirmar que PPO no está actualizando de forma excesiva o
inestable.

## Patrones de diagnóstico

### Aprendizaje sano

- sube `clean_takeover_rate`
- sube `clean_finish_rate`
- bajan `bad_shoot_rate` y `bad_dribble_rate`
- baja `thrash_rate`
- mejora `mean_time_to_takeover` sin sacrificar limpieza
- en Stage 2+, baja `collision_rate` sin desplomar takeover
- las mejoras aparecen también en evaluación

### Inflación de shaping o reward hacking

- sube `episode_reward_mean` pero no `clean_finish_rate`
- baja `thrash_rate` por colapso a una sola skill
- `walk_in_walk_zone_rate` alta, pero `clean_takeover_rate` plana
- sube `progress_ball` medio, pero no entra a `takeover_envelope`
- baja `collision_rate` porque el agente evita avanzar
- train mejora, eval no

## Decisión de alcance: Stage 4 fuera del núcleo vital del sprint

Stage 4 no se toma como variable vital inmediata del sprint.

Razones:

- varias señales de 1v1 fino no están cerradas todavía,
- duel no está tan validado como avoidance,
- el bridge actual no da la misma flexibilidad para rival que para balón y
  robot propio,
- insistir en Stage 4 demasiado pronto puede contaminar reward, métricas y
  tiempos del sprint.

### Consecuencia práctica

Para el sprint inmediato:

- el núcleo vital es Stage 0 a Stage 3,
- Stage 4 puede quedar como placeholder o etapa posterior,
- las variables `first_to_ball`, `bad_duel`, `win_rate` y gates de Stage 4 no
  deben bloquear el arranque ni la validación inicial de PPO.

## Resumen operativo

Antes de entrenar PPO de verdad, la instrumentación debe dejarnos responder:

1. ¿las señales del reward son reales y confiables en SimRobot 3D?
2. ¿la policy está aprendiendo takeover o solo sumando shaping?
3. ¿la selección entre `walk`, `dribble` y `shoot` ya es contextual?
4. ¿las promociones de stage tienen evidencia de comportamiento y no solo de
   retorno?

Regla práctica final:

- primero instrumentar bien Stage 0-3,
- luego entrenar,
- luego usar métricas de sanidad para decidir si la policy mejora de verdad,
- dejar Stage 4 fuera del núcleo crítico del sprint inicial.
