# Estado actual: handoff visible 3D

Fecha de cierre parcial: 2026-05-13

## Resumen corto

El flujo `simrobot_visible` en `3d` quedó funcional para validación manual.

Estado actual:

- `reset_extended` visible `3d`: PASS
- `invalid_slot` visible `3d`: PASS
- `dynamic_world` visible `3d`: PASS

El backend embebido `simrobot` en `3d` no quedó cerrado y hoy sigue roto al
intentar cargar escena desde `pybh.simrobot_load_scene(...)`.

## Qué se resolvió

### 1. Validación visible real en GUI

Se creó una suite manual para `simrobot_visible`:

- `gui_visible_suite.py`

La suite corre tres bloques:

1. `reset_extended`
2. `invalid_slot`
3. `dynamic_world`

En `3d` los tres quedaron en `PASS`.

### 2. Falso positivo del reset visible

Antes, el backend Python podía devolver la última observación aunque el reset no
hubiera convergido a la pose pedida.

Eso ya no pasa:

- si no converge dentro del timeout, falla explícitamente
- el error reporta pose esperada, pose observada y distancia

### 3. Criterio correcto de validación para visible

La observación `robot_x / robot_y / robot_theta` publicada por el wire no es la
pose física real de SimRobot sino `RobotPose` estimada por B-Human.

Por eso se agregó telemetría explícita de pose física:

- `simrobot_robot_x`
- `simrobot_robot_y`
- `simrobot_robot_theta`

Ahora el reset visible se valida contra esa pose física real y no contra la
localización estimada.

### 4. Reset visible ya no se pisa por GameController

En `LocalConsole.cpp` se quitaron las llamadas extra a:

- `gamePhaseNormal()`
- `playing()`

después del teleport del reset visible.

Eso evitó que el reset pedido por Python quedara sobrescrito por lógica de
`SetupPoses` o transiciones del `GameController`.

## Qué sirve hoy

### Visible `3d`

Sirve para:

- validar reset extendido con teammates y opponents
- validar rechazo limpio de slots inválidos
- validar updates dinámicos intra-episodio
- inspeccionar el wire con GUI real
- usar la suite manual visible como referencia operativa

Comando principal:

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 gui_visible_suite.py --simrobot-profile 3d --launch-simrobot
```

### Telemetría ya disponible en `rl_get_obs(...)`

Ya están publicados:

- señales base:
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
  - `robot_x`
  - `robot_y`
  - `ball_x`
  - `ball_y`
- señales de pipeline:
  - `provider_*`
  - `skill_behavior_*`
  - `motion_engine_*`
  - `motion_speed_*`
  - `libwalk_*`
  - `zweikampf_*`
- pose física nueva:
  - `simrobot_robot_x`
  - `simrobot_robot_y`
  - `simrobot_robot_theta`

Eso significa:

- la telemetría cruda necesaria para el `doc 2` ya está
- varias métricas del sprint siguen siendo derivadas en Python, no señales
  directas del bridge

## Qué no sirve hoy

### Embebido `simrobot` `3d`

No quedó sano.

Comando que falla:

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 smoke_test_skill_requests.py --backend simrobot --simrobot-profile 3d
```

Falla actual observada:

- `simrobot_load_scene failed`
- errores de parseo de floats/units dentro de `RL1v0_3D.ros2` y `NaoV6H25.rsi2`
- al forzar locale `C`, también apareció `Segmentation fault`

Interpretación actual:

- la ruta embebida `simrobot` `3d` sigue rota en carga/inicialización de escena
- no es un problema del `simrobot_visible`
- no es un problema principal de telemetría

## Qué falta

### Pendiente técnico real

1. Depurar y corregir `simrobot` embebido `3d`
2. Volver a correr smoke embebido `3d`
3. Confirmar que la telemetría nueva no rompió el backend embebido una vez ese
   path vuelva a cargar escenas

### Pendiente documental

Conviene actualizar:

- `docs/17-handoff-visible-simrobot-gui.md`
- `docs/04-verificacion-operativa.md`

con:

- `PASS` visible `3d`
- aclaración `robot_est` vs `robot_sim`
- versión nueva del shared state

## Qué no falta por bridge base

No falta por wire visible:

- reset extendido visible
- request/ack visible por `world_status`
- invalid slot con error duro
- dynamic world visible
- telemetría cruda principal del stack

Lo que queda abierto ya no es “falta de API visible”, sino:

- saneamiento del backend embebido `simrobot` `3d`
- cierre documental

## Archivos clave tocados en este cierre

En `RL/`:

- `pybh_backend.py`
- `gui_visible_suite.py`
- `gui_visible_reset_extended_check.py`
- `gui_visible_invalid_slot_check.py`
- `gui_visible_dynamic_world_check.py`

En `SabanaHerons2026/`:

- `Src/Libs/SimulatedNao/LocalConsole.cpp`
- `Src/Libs/Python/Controller/RLSharedState.h`
- `Src/Libs/Python/Controller/RLSharedState.cpp`
- `Src/Libs/Python/Controller/Module.cpp`

