# Observation Space Realista y Canary

## Objetivo

Este documento fija la lectura actual del estado que consume `RL/environment.py`
y cómo diagnosticar fallas de canary sin volver a simplificar el problema de
forma artificial.

La regla central sigue siendo:

```text
RL decide QUE hacer.
B-Human decide COMO ejecutarlo.
```

## Pipeline activo

```text
RL
  -> SkillRequest
  -> SkillBehaviorControl
  -> MotionRequest
  -> MotionEngine
  -> JointRequest
  -> SimRobot
```

## Estado actual del observation space

El vector activo tiene `23` features y está definido en `RL/observation.py`.

Orden actual:

1. `robot_x`
2. `robot_y`
3. `robot_theta`
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
14. `can_score_now`
15. `shot_quality_no_obstacles`
16. `shot_opening_with_obstacles`
17. `pass_options_count`
18. `nearest_teammate_dist`
19. `nearest_opponent_dist`
20. `nearest_uncertain_obstacle_dist`
21. `nearest_teammate_front_dist`
22. `nearest_opponent_front_dist`
23. `nearest_uncertain_front_dist`

## Qué resolvió esta ampliación

El estado dejó de ser un vector geométrico mínimo y ahora ya incluye:

- frescura de visión de pelota
- consistencia con game state
- velocidad y end position de pelota
- rating ofensivo básico
- resumen útil de obstáculos

Eso lo vuelve mucho más útil para PPO sobre `walk`, `dribble` y `shoot`.

## Qué todavía no resuelve

El vector actual todavía no incorpora por sí solo:

- masks por stage aplicadas en entrenamiento
- reward semántico de takeover
- gates de promoción
- telemetría de comportamiento
- curriculum con cambios dinámicos intra-episodio

## Cómo leer un fallo de canary

Si el canary falla, separar primero:

1. fallo del wire
2. fallo del estado
3. fallo del reward
4. fallo de la política

Indicadores rápidos:

- si `requested_skill` no coincide, el problema es del wire o de la decodificación
- si `obs_ready` no llega, el problema es del loop SimRobot/bridge
- si las features llegan pero la policy oscila, el problema suele ser reward o masks
- si las features no cambian con el escenario, falta instrumentación aguas abajo

## Dirección recomendada

Para PPO del sprint:

- mantener este vector de 23 features
- no volver al vector histórico corto
- agregar lógica de reward/curriculum encima, no rehacer el wire
