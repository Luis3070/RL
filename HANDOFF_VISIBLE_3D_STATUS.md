# Estado actual: handoff visible 3D

Fecha de cierre parcial: 2026-05-13

## Resumen corto

El flujo `simrobot_visible` en `3d` quedó funcional para validación manual.

Estado actual:

- `reset_extended` visible `3d`: PASS
- `invalid_slot` visible `3d`: PASS
- `dynamic_world` visible `3d`: PASS

El backend embebido `simrobot` en `3d` ya volvió a quedar funcional tanto en la
prueba mínima `pybh.simrobot_load_scene(...)` como en el smoke alto.

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

Quedó sano otra vez.

Prueba mínima que se usó para aislarlo:

```bash
cd /home/rafael/Documentos/Robocup2026/SabanaHerons2026
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 -c "import os; os.chdir('/home/rafael/Documentos/Robocup2026/SabanaHerons2026'); import pybh.controller as pybh; scene='/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Config/Scenes/RL1v0_3D.ros2'; ok=pybh.simrobot_load_scene(scene); print(ok, pybh.simrobot_is_loaded(), pybh.simrobot_last_error()); pybh.simrobot_unload_scene()"
```

Comando de smoke alto:

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 smoke_test_skill_requests.py --backend simrobot --simrobot-profile 3d
```

Resultado actual:

- `simrobot_load_scene(RL1v0_3D.ros2)`: PASS
- `simrobot_load_scene(BH.ros2)`: PASS
- `smoke_test_skill_requests.py --backend simrobot --simrobot-profile 3d`:
  PASS

Causa que se confirmó en el camino:

- el crash mínimo ya no venía del wire RL ni del reset;
- el problema estaba en el host embebido `3d` cuando forzaba
  `QT_QPA_PLATFORM=offscreen`;
- en esta máquina, con `DISPLAY=:0`, el core `3d` carga bien por `xcb` y se
  cae por `offscreen`.

Interpretación actual:

- la ruta embebida `simrobot` `3d` ya no está rota en
  carga/inicialización de escena;
- el fix real quedó en `SimRobotHost.cpp`: usar `xcb` con `DISPLAY` y dejar
  `offscreen` solo como fallback sin display;
- además se endureció el warmup del backend Python para no aceptar resets con
  observaciones `frame=0` o pelota desalineada.

## Qué falta

### Pendiente técnico real

1. Mantener documentada la prueba mínima `simrobot_load_scene(...)` como primer
   smoke cuando `3d` vuelva a romperse
2. Si aparece una regresión nueva en otra máquina, revisar primero
   `QT_QPA_PLATFORM` efectivo y presencia de `DISPLAY`
3. Seguir cerrando documentación operativa y probes de observación

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
