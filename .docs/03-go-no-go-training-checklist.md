# Go / No-Go checklist para iniciar entrenamiento PPO

Este documento complementa:

- [01-ppo-takeover-curriculum-handoff.md](./01-ppo-takeover-curriculum-handoff.md)
- [02-instrumentation-ppo-sanity.md](./02-instrumentation-ppo-sanity.md)

Su objetivo es dar un filtro corto antes de lanzar entrenamiento real.

## Regla general

`GO`:

- las señales críticas existen,
- el reward puede calcularse con datos confiables,
- el stage actual se puede instanciar de verdad,
- y la telemetría mínima permite distinguir aprendizaje real de reward inflado.

`NO-GO`:

- falta una señal crítica,
- el stage no puede materializarse con el bridge actual,
- el reward depende de variables no instrumentadas,
- o no existe forma de validar sanidad en evaluación.

## GO mínimo para entrenar

### 1. Backend y escena

- [ ] `backend = simrobot`
- [ ] `profile = 3d`
- [ ] la escena RL carga y devuelve observaciones sin timeout

### 2. Señales mínimas del reward

- [ ] `ball_rel_x / ball_rel_y` responden coherentemente
- [ ] `time_since_ball_seen` responde coherentemente
- [ ] `ball_seen_percentage` responde coherentemente
- [ ] `can_score_now` existe y cambia
- [ ] `shot_opening_with_obstacles` existe y cambia
- [ ] `nearest_opponent_front_dist` o equivalente existe

### 3. Helpers y zonas

- [ ] `is_takeover_envelope` está implementado
- [ ] `is_walk_zone` está implementado
- [ ] `is_dribble_zone` está implementado
- [ ] `is_shoot_zone` está implementado
- [ ] hay tests o checks simples de estos helpers

### 4. Reward base

- [ ] existen `progress_ball` y `align_ball`
- [ ] existe `enter_takeover`
- [ ] existen `good/bad_dribble`
- [ ] existen `good/bad_shoot`
- [ ] existe `thrash`
- [ ] el reward funciona con señales ya validadas en SimRobot 3D

### 5. Stage actual realmente instanciable

- [ ] el sampler de reset del stage actual funciona
- [ ] las skills activas del sprint son solo `walk`, `dribble`, `shoot`
- [ ] `pass`, `mark`, `block`, `observe` quedan enmascaradas
- [ ] la proyección de `skill_type` al subset activo ya está resuelta

### 6. Telemetría mínima

- [ ] se loggea `clean_walk_rate`
- [ ] se loggea `clean_takeover_rate`
- [ ] se loggea `clean_finish_rate`
- [ ] se loggea `thrash_rate`
- [ ] se loggea `mean_time_to_takeover`
- [ ] se loggea `episode_reward_mean`
- [ ] se loggea `policy_loss`
- [ ] se loggea `value_loss`
- [ ] se loggea `entropy`

### 7. Evaluación corta

- [ ] existe evaluación corta periódica
- [ ] las métricas se observan también en eval, no solo en train
- [ ] el gate del stage actual está definido
- [ ] el stage no promociona si no cumple el gate

## NO-GO duro

Si se cumple cualquiera de estos puntos, no conviene lanzar entrenamiento largo.

### Señales críticas ausentes

- [ ] no hay lectura confiable de distancia/ángulo al balón
- [ ] no hay lectura confiable de frescura de visión
- [ ] `shot_opening_with_obstacles` no existe o no cambia
- [ ] el reward depende de una señal no publicada en el repo actual

### Stage no realizable

- [ ] el reset del stage no puede instanciar la distribución esperada
- [ ] el stage depende de obstáculos/rivales que el bridge actual no controla
- [ ] el stage depende de telemetría aún no implementada

### Reward no verificable

- [ ] no hay forma de probar que `good_shoot` o `bad_shoot` disparan bien
- [ ] no hay forma de detectar `thrash`
- [ ] el reward está escrito sobre observación mal escalada o ambigua

### Falta de sanidad

- [ ] no se puede distinguir train de eval
- [ ] no se loggean métricas de comportamiento
- [ ] solo se observa reward medio
- [ ] no hay criterio de promoción o no está instrumentado

## GO recomendado por etapas

### GO para Stage 0

- [ ] resets dentro de `<= 800 mm`
- [ ] reward base funcional
- [ ] masks activas correctas
- [ ] métricas de limpieza disponibles

### GO para Stage 1

- [ ] Stage 0 ya muestra takeover limpio repetible
- [ ] `mean_time_to_takeover` ya se puede medir bien
- [ ] el aumento de radio no rompe observación ni cierre

### GO para Stage 2

- [ ] obstáculos estáticos ya existen en escena o sampler usable
- [ ] `collision_rate` existe o se deriva de forma razonable
- [ ] `obstacle_clearance_success` ya puede medirse

### GO para Stage 3

- [ ] obstáculos dinámicos scripted existen de verdad
- [ ] `dynamic_obstacle_success` se puede medir
- [ ] `corridor_keep` ya es medible o está explícitamente desactivado

## No meter en GO crítico de este sprint

Estas variables no deben bloquear el arranque del PPO inicial:

- [ ] Stage 4 completo
- [ ] `1v1` fino
- [ ] `bad_duel` plenamente instrumentado
- [ ] `first_to_ball` robusto
- [ ] self-play

## Regla práctica final

Si hay duda:

1. primero verificar que Stage 0 entrena con métricas sanas,
2. luego Stage 1,
3. luego Stage 2,
4. luego Stage 3,
5. Stage 4 queda fuera del núcleo crítico del sprint.

Resumen corto:

- sin señales confiables: `NO-GO`
- sin métricas de comportamiento: `NO-GO`
- sin gates: `NO-GO`
- con reward verificable, stage instanciable y eval corta útil: `GO`
