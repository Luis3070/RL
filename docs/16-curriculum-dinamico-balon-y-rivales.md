# Curriculum dinámico: balón y rivales

## Objetivo

Este documento resume el estado del control de mundo desde Python y qué falta
por encima del wire para soportar curriculum más rico.

La separación importante es:

1. cambios de mundo al reset
2. cambios de mundo durante el episodio

## Estado actual

### Reset extendido ya soportado

Hoy ya existen rutas útiles para fijar el mundo al inicio del episodio:

- `PyBHBackend.reset(... teammates=..., opponents=...)` en modo embebido
- `pybh.rl_reset_world_extended(...)` para SimRobot visible
- `pybh.set_world_state_extended(...)` como utilidad de controller

Eso ya permite definir:

- pelota
- robot RL
- teammates
- opponents

### Convención de jugadores

Los jugadores se expresan como:

```text
(number, x, y, theta, upright)
```

Reglas:

- `number` en `1..20`
- sin duplicados por lista
- el robot RL no va en `teammates`
- `upright=False` no está soportado en las rutas activas

### SimRobot embebido

En modo embebido, el backend Python traduce el world spec a teleports de:

- pelota
- robot propio
- teammates
- opponents

con la convención:

```text
robot1..robot20   teammates
robot21..robot40  opponents
```

### SimRobot visible

En modo visible, el reset extendido ya viaja por `RLSharedState` y
`LocalConsole::applyPendingRLReset()` reposiciona:

- robot propio
- pelota
- teammates
- opponents

## Qué ya quedó resuelto

Para curriculum por reset ya no falta el wire básico de mundo.

Con lo actual se puede instanciar:

- distintas distancias robot-balón
- distintos ángulos
- teammates u opponents fijos por episodio
- escenas con ocupación controlada al inicio

### API dinámica intra-episodio ya soportada

También existe ya una API canónica para cambiar mundo sin reset completo:

```text
rl_set_dynamic_world(...)
```

Superficie disponible:

- `PyBHBackend.set_dynamic_world(...)`
- `pybh.rl_set_dynamic_world(...)`
- `pybh.rl_get_world_status(...)`

Esto permite:

- mover pelota durante el episodio
- reposicionar el robot RL sin reiniciar `playing`
- actualizar subsets de teammates u opponents
- detectar errores explícitos si un slot no existe en la escena visible

## Qué sigue faltando

### 1. Integración con sampler de stages

El wire ya acepta el mundo; lo que falta es que el lado RL construya y use
samplers de stage sobre ese contrato.

### 2. Telemetría de colisión y éxito contextual

Para sacar provecho del curriculum con rivales faltan métricas del lado RL:

- `collision_rate`
- `obstacle_clearance_success`
- `dynamic_obstacle_success`

## Recomendación

Orden razonable:

1. usar ya el reset extendido para Stages 0-2
2. usar `set_dynamic_world(...)` solo cuando el stage realmente necesite eventos intra-episodio
3. cerrar samplers, masks y métricas del lado RL
