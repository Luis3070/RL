# Handoff: `simrobot_visible` GUI

## Resumen

El wire visible `RL <-> B-Human <-> SimRobot` ya no está en estado base; hoy
existe soporte real para:

- reset extendido con `teammates` y `opponents`
- updates dinámicos intra-episodio
- confirmación explícita de requests visibles
- error duro si un slot pedido no existe en la escena

La superficie principal está en:

- [pybh_backend.py](/pybh_backend.py)
- [Module.cpp](/Src/Libs/Python/Controller/Module.cpp)
- [LocalConsole.cpp](/Src/Libs/SimulatedNao/LocalConsole.cpp)
- [SimulatedRobot.cpp](/Src/Libs/SimulatedNao/SimulatedRobot.cpp)

## Qué hay implementado

### 1. Reset visible con request/ack

En modo `simrobot_visible`, `PyBHBackend.reset(...)` hace:

1. `rl_reset_world_extended(...)`
2. espera `world_status` por `request_id`
3. manda `stand`
4. espera una observación nueva alineada con la pose pedida

Eso está en [pybh_backend.py](/pybh_backend.py:367) y las esperas viven en [pybh_backend.py](/pybh_backend.py:613).

### 2. Dynamic world visible

Existe `PyBHBackend.set_dynamic_world(...)`, que en visible usa:

- `pybh.rl_set_dynamic_world(...)`
- `pybh.rl_get_world_status(...)`

Ver [pybh_backend.py](/pybh_backend.py:414).

### 3. Escenas visibles dedicadas

Se agregaron escenas específicas para visible con robots extra reales:

- [RLFull.ros2d](/Config/Scenes/RLFull.ros2d)
- [RLFull.con](/Config/Scenes/RLFull.con)
- [RLFull_3D.ros2](/Config/Scenes/RLFull_3D.ros2)
- [RLFull_3D.con](/Config/Scenes/RLFull_3D.con)

Y `simrobot_visible` ya usa esas escenas por defecto desde [config.py](/config.py:54).

## Problemas encontrados y resueltos

### 1. Backend rápido eliminado

Se eliminó `cpp/RLSim2D` de la superficie soportada. Ahora solo se mantienen:

- `simrobot`
- `simrobot_visible`

### 2. Corrupción de payload visible

Se observó basura en campos como `motion_speed_rot`. La causa más probable fue
desalineamiento entre:

- `pybh/controller.so` recompilado
- `libSimulatedNao.so` viejo en la GUI visible

Se recompiló `SimRobot`/`libSimulatedNao.so` y se limpió:

- `/dev/shm/pybh_rl_shared_state_v3`

### 3. Escenas mínimas sin robots extra

`RL1v0.ros2d` y `RL1v0_3D.ros2` solo tenían `robot1`, por eso los tests de
reset extendido fallaban con:

- `could not find teammate robot 2`
- `could not find opponent robot 1`

Eso motivó la creación de `RLFull*`.

### 4. Timeout aparente al probar GUI

No era un cuelgue real del wire. El problema era de UX:

- el script mandaba el reset antes de que la escena visible terminara de cargar
- el script no imprimía progreso mientras esperaba el `request_id` y la obs nueva

Los scripts de GUI ahora:

- esperan `Enter` después de que la escena visible ya está corriendo
- usan `timeout_ms=30000`
- imprimen progreso explícito
- permiten separar timeout de `request_id` y de observación posterior
- pueden loggear `world_status` periódicamente durante la espera

## Qué sigue fallando o no está cerrado

### 1. La experiencia visible sigue siendo lenta

Aunque el flujo ya no parece congelado, el visible sigue siendo más lento de lo
que sería deseable para pruebas manuales.

La razón principal es estructural: el reset visible espera dos hitos:

1. confirmación del `request_id`
2. una observación nueva con la pose correcta

Ver:

- [pybh_backend.py](/pybh_backend.py:367)
- [pybh_backend.py](/pybh_backend.py:636)

No está claro todavía si la lentitud residual es:

- solo latencia natural de GUI + frame loop
- o demasiado polling / timeout conservador en el backend Python

### 2. No hay prueba end-to-end cerrada documentada como “PASS”

Hasta ahora se confirmó:

- la GUI vuelve a abrir
- `simrobot_visible` usa escenas completas
- el script ya no se percibe congelado de la misma forma

Pero falta cerrar evidencia limpia de:

- reset extendido visible `2d`: PASS
- reset extendido visible `3d`: PASS
- dynamic world visible `2d`: PASS
- dynamic world visible `3d`: PASS

## Scripts de prueba que existen

Se agregaron estos scripts:

- [gui_visible_reset_extended_check.py](/gui_visible_reset_extended_check.py)
- [gui_visible_invalid_slot_check.py](/gui_visible_invalid_slot_check.py)
- [gui_visible_dynamic_world_check.py](/gui_visible_dynamic_world_check.py)
- [gui_visible_suite.py](/gui_visible_suite.py)

Estado de estos scripts:

- ya no asumen intervención manual para arrancar la simulación
- muestran progreso antes de esperar
- aceptan:
  - `--request-timeout-ms`
  - `--reset-obs-timeout-ms`
  - `--status-log-interval-s`
- sirven para validación manual, no automática

## Qué falta probar

### Checklist mínimo

1. `gui_visible_reset_extended_check.py --simrobot-profile 2d`
2. `gui_visible_reset_extended_check.py --simrobot-profile 3d`
3. `gui_visible_invalid_slot_check.py --simrobot-profile 2d`
4. `gui_visible_invalid_slot_check.py --simrobot-profile 3d`
5. `gui_visible_dynamic_world_check.py --simrobot-profile 2d`
6. `gui_visible_dynamic_world_check.py --simrobot-profile 3d`

### Criterios de aprobación

En reset extendido:

- `robot1`, teammates y opponents pedidos sí se recolocan
- la pelota sí se recoloca
- no aparece error
- el script imprime `reset ok`

En invalid slot:

- el script falla con error explícito
- la escena no queda parcialmente movida

En dynamic world:

- mover pelota solo mueve pelota
- mover rival solo mueve rival
- mover robot RL solo mueve robot RL
- no hay reset completo entre pasos

## Siguiente mejora recomendada

La mejora de mayor valor ahora no es más wire; es tooling de prueba visible.

Prioridad recomendada:

1. correr `gui_visible_suite.py` en `2d` y `3d` y registrar resultado
2. si sigue lento, perfilar si `obs_ready` visible está llegando tarde desde `LocalConsole`
3. si aún hace falta, separar más fino el timeout de:
   - aceptación del request
   - observación posterior al request

## Conclusión

El trabajo grueso del wire visible ya está hecho. Lo que queda no parece ser
falta de API, sino:

- cerrar validación manual completa en GUI
- mejorar la observabilidad y la percepción de latencia en los scripts
