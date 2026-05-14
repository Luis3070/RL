# Checklist operativo del wire RL

Este checklist reemplaza la idea vaga de "el wire ya quedó" por una validación
operativa concreta, con puntos que hoy ya quedaron implementados en `RL/` y
otros que todavía requieren evidencia manual o runtime externo.

## Estado actual

### Implementado en este repo

- [x] `obs_mode` explícito en la configuración Python (`corrected` o
  `estimated`)
- [x] propagación de `obs_mode` al bridge vía `PYBH_RL_OBS_MODE`
- [x] telemetría de auditoría del wire en smoke y eval
- [x] veredicto automático `PASS/WARN/FAIL` para corridas
- [x] persistencia del veredicto en CSV (`smoke_summary.csv`,
  `eval_summary.csv`)
- [x] preflight embebido para `simrobot` `3d` antes de cargar la escena en el
  proceso principal
- [x] documentación alineada para no declarar cerrado el wire observacional

### Pendiente fuera de este repo o fuera del sandbox

- [ ] revalidación end-to-end de `simrobot_visible` `2d`
- [ ] revalidación end-to-end de `simrobot_visible` `3d`
- [ ] PASS documentado para reset extendido visible
- [ ] PASS documentado para dynamic world visible
- [x] revalidación del runtime embebido `3d` con prueba mínima de
  `simrobot_load_scene(...)` y smoke alto

### Evidencia ya reunida para `simrobot` embebido `3d`

- el 14 de mayo de 2026 se rehizo el aislamiento con la prueba mínima real:
  `import pybh.controller` + `pybh.simrobot_load_scene(scene)`
- con esa prueba:
  - `RL1v0.ros2d`: PASS
  - `RL1v0_3D.ros2`: crash si el host fuerza `offscreen`
  - `BH.ros2`: mismo crash, así que no era un bug específico de RL
- luego se probó el mismo `3d` forzando plataforma Qt:
  - `QT_QPA_PLATFORM=offscreen`: crash
  - `QT_QPA_PLATFORM=xcb`: PASS cuando hay `DISPLAY`
- el fix final fue mover esa decisión al host embebido:
  - con `DISPLAY`, usar `xcb`
  - sin `DISPLAY`, dejar `offscreen` como fallback

Lectura correcta:

- el problema real no estaba en PPO ni en `env.reset()`, sino en la carga
  mínima de escena embebida `3d`;
- el fallo tampoco era de escena RL, porque aparecía igual en `BH.ros2`;
- el punto de quiebre era la plataforma Qt elegida por defecto en el host;
- hoy la ruta mínima y el smoke alto vuelven a pasar en `2d` y `3d`.

## Gates automáticos del wire

Las corridas ahora calculan un estado de wire a partir de estas señales:

- `obs_mode_<configured>_rate >= 0.99`
- `synthetic_pose_rate <= 0.25`
- `synthetic_ball_rate <= 0.40`
- `pose_alignment_warning_rate <= 0.10`
- `ball_stale_rate <= 0.10`
- `ball_collapsed_rate <= 0.10`
- si `obs_mode="estimated"`:
  - `corrected_exported_robot_pose_rate <= 0.01`
  - `corrected_exported_ball_rate <= 0.01`

Advertencia no bloqueante:

- `collision_rate > 0.25` produce `WARN`, no `FAIL`

Interpretación:

- `PASS`: la corrida no mostró inconsistencias de wire con los thresholds
  actuales.
- `WARN`: el wire no falló por observación, pero hay una señal de riesgo a
  revisar.
- `FAIL`: la corrida no sirve como evidencia de wire estable.

## Uso recomendado

### Smoke

- correr `train.py` con `mode="smoke"`
- revisar `csv/smoke_summary.csv`
- exigir `wire_audit_status = PASS`

### Train corto / sanity

- correr `train.py` con `mode="train"`
- revisar `csv/eval_summary.csv`
- exigir `wire_audit_status != FAIL`
- si `obs_mode="estimated"`, revisar además que
  `corrected_exported_*_rate` siga en `0`

Nota operativa:

- en `simrobot_visible`, `obs_mode` solo puede imponerse desde Python si
  SimRobot fue lanzado por el propio backend;
- si la GUI ya estaba abierta aparte, el gate sigue siendo útil porque detecta
  el modo realmente observado aunque Python no pueda cambiarlo retroactivamente.
- en `simrobot` embebido `3d`, el backend sigue teniendo preflight defensivo,
  pero el fix determinante quedó en el host C++: ya no fuerza `offscreen` si
  existe `DISPLAY`.

## Siguiente cierre real

Para declarar el wire observacional realmente cerrado todavía falta:

1. correr la suite visible `2d` y `3d` y registrar PASS
2. repetir sanity corta con `obs_mode="corrected"` y `obs_mode="estimated"`
3. verificar que los `dynamic world updates` de Stage 2-3 no rompan la
   alineación
4. decidir qué modo quedará como default oficial de entrenamiento
