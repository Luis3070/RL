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
- [ ] revalidación del runtime embebido `3d` donde hoy puede caer en init
  gráfica / Mesa / OpenGL

### Evidencia ya reunida para `simrobot` embebido `3d`

- el 14 de mayo de 2026 se probó el preflight nuevo con estas plataformas Qt:
  `xcb`, `offscreen`, `minimalegl`, `eglfs`, `vnc`, `linuxfb`, `minimal`
- ninguna levantó la escena `3d` en este host:
  - `xcb`: no conecta a `DISPLAY :0`
  - `offscreen`: termina en `SIGSEGV`
  - `minimalegl`: no inicializa `EGL`
  - `eglfs`: no encuentra dispositivo `DRM`
  - `vnc` y `minimal`: no crean contexto OpenGL
  - `linuxfb`: no encuentra framebuffer válido
- además se reprodujo el bootstrap histórico para descartar regresión reciente:
  - con `DISPLAY`: `legacy-display-xcb`, abort `134`
  - sin `DISPLAY`: `legacy-host-default`, `SIGSEGV` `139`

Lectura correcta:

- el preflight nuevo no empeoró el caso `3d`; solo lo hace observable sin matar
  tan temprano el proceso principal;
- el pendiente real sigue siendo host/nativo y debe resolverse fuera de este
  repo o con cambios en el stack gráfico de `SabanaHerons2026`.

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
- en `simrobot` embebido `3d`, el backend ahora prueba plataformas Qt
  candidatas en un subprocess antes de cargar la escena real; eso reduce el
  riesgo de matar el proceso principal, pero no sustituye arreglos nativos si el
  core sigue cayendo en Mesa/OpenGL.

## Siguiente cierre real

Para declarar el wire observacional realmente cerrado todavía falta:

1. correr la suite visible `2d` y `3d` y registrar PASS
2. repetir sanity corta con `obs_mode="corrected"` y `obs_mode="estimated"`
3. verificar que los `dynamic world updates` de Stage 2-3 no rompan la
   alineación
4. decidir qué modo quedará como default oficial de entrenamiento
