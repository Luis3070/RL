# Premisas Irrompibles

Estas reglas no son estilo. Son restricciones de arquitectura para no volver a romper el movimiento natural.

## 1. RL no provee `MotionRequest` en escenarios naturales

En escenarios RL de SimRobot:

```cfg
{representation = MotionRequest; provider = SkillBehaviorControl;}
```

No volver a:

```cfg
{representation = MotionRequest; provider = RLSkillProvider;}
```

`RLSkillProvider` puede conservar codigo legacy/debug para `MotionRequest`, pero el grafo natural no debe mapearlo como provider.

## 2. El punto de entrada RL es `SkillRequest`

El contrato correcto:

```text
RLSkillProvider -> SkillRequest
SkillBehaviorControl -> MotionRequest
```

No saltarse `SkillBehaviorControl`.

## 3. `walkTo` usa coordenadas de campo

No convertir `walkTo` a coordenadas relativas para:

```text
cpp
simrobot
simrobot_visible
```

`SkillRequest::Builder::walkTo(...)` espera campo global.

## 4. No usar root-assist como solucion final

No activar por defecto:

```text
PYBH_SIMROBOT_RL_ROOT_ASSIST
PYBH_SIMROBOT_RL_ABSTRACT_MOTION
```

Si se usan para diagnostico, deben quedar apagados antes de validar.

Movimiento natural significa:

```text
MotionEngine -> JointRequest -> SimRobot
```

No:

```text
Python -> mover raiz del robot
```

## 5. SimRobot real debe tener `RLSim2D` apagado

Para:

```text
mode="simrobot"
mode="simrobot_visible"
```

debe ejecutarse:

```python
pybh.rl_sim_set_enabled(PLAYER_NUMBER, False)
```

`sim_enabled` en observaciones debe ser `False`.

Si `sim_enabled=True` mientras se cree estar usando SimRobot real, la prueba no es valida.

## 6. El reset debe devolver observacion fresca

Despues de reset en SimRobot embebido, no se puede devolver la primera observacion vieja que aparezca.

La observacion inicial debe estar cerca de la pose pedida.

Motivo: si el agente heuristico lee `robot=(0,0)` y `ball=(0,0)`, manda `shoot` inmediatamente y puede ensuciar el estado.

## 7. No importar `pybh.controller` desde cwd incorrecto

Scripts nuevos deben hacer una de estas dos cosas:

```python
from pybh_backend import PyBHBackend
import pybh.controller as pybh
```

o:

```python
import os
from config import BHUMAN_DIR
os.chdir(BHUMAN_DIR)
import pybh.controller as pybh
```

B-Human carga configuracion relativa al cwd. Importar `pybh.controller` demasiado temprano puede producir errores nativos dificiles de leer.

## 8. No commitear `.so` de mas de 100 MB

GitHub rechaza archivos sobre 100 MB.

Despues de compilar y copiar:

```bash
strip --strip-debug \
  Build/Linux/Python/Develop/controller.cpython-312-x86_64-linux-gnu.so \
  Make/Python/pybh/controller.cpython-312-x86_64-linux-gnu.so
```

Verificar:

```bash
ls -lh Make/Python/pybh/controller.cpython-312-x86_64-linux-gnu.so
```

## 9. Escenarios normales no son laboratorio RL

No romper `Default`, `4v4_Scaled`, `2D` u otros escenarios normales para probar RL.

Usar escenarios dedicados:

```text
4v4_RL2D
4v4_RL3D
```

## 10. La validacion minima debe mirar trazas, no solo posicion

Especialmente en 3D, la pose global puede moverse poco al inicio.

Validar:

```text
requested_skill
skill_behavior_skill_request
skill_behavior_motion_request
motion_engine_input_request
motion_engine_effective_request
motion_engine_phase
joints de piernas
```

Para caminata:

```text
requested_skill = walkTo
skill_behavior_skill_request = 2
skill_behavior_motion_request = 4
motion_engine_input_request = 4
motion_engine_effective_request = 4
```

## 11. Smoke visual no es entrenamiento

`live_simrobot_probe.py --policy heuristic` es demo/diagnostico visual.

`train.py --mode smoke` prueba el loop RL, pero una policy no entrenada puede mandar acciones casi quietas. No usarlo como prueba visual de movimiento natural.

## 12. Si se mata un proceso nativo, limpiar antes de diagnosticar

Si un proceso muere dentro de B-Human/SimRobot y las pruebas quedan inconsistentes:

```bash
rm -f /dev/shm/pybh_rl_shared_state_v2
```

Esto no borra codigo ni configuracion; solo borra estado temporal del bridge RL.

