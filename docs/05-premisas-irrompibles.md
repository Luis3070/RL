# Premisas Irrompibles

Estas reglas no son estilo. Son restricciones de arquitectura para no volver a
romper el flujo natural.

## 1. RL entra por `SkillRequest`

El contrato correcto:

```text
RLSkillProvider -> SkillRequest
SkillBehaviorControl -> MotionRequest
```

No volver a saltarse `SkillBehaviorControl`.

## 2. `walkTo` usa coordenadas de campo

No convertir `walkTo` a coordenadas relativas en las rutas activas.

`SkillRequest::Builder::walkTo(...)` espera campo global.

## 3. No usar root-assist como solución final

No activar por defecto:

```text
PYBH_SIMROBOT_RL_ROOT_ASSIST
PYBH_SIMROBOT_RL_ABSTRACT_MOTION
```

Movimiento natural significa:

```text
MotionEngine -> JointRequest -> SimRobot
```

## 4. El reset debe devolver observación fresca

Después de `reset`, la primera observación debe corresponder al mundo pedido.

No se debe devolver una observación vieja por rapidez.

## 5. No importar `pybh.controller` desde cwd incorrecto

Usar:

```python
from pybh_backend import PyBHBackend
import pybh.controller as pybh
```

o cambiar antes a `BHUMAN_DIR`.

## 6. Escenarios normales no son laboratorio RL

Usar escenarios dedicados:

```text
4v4_RL2D
4v4_RL3D
```

## 7. La validación mínima mira trazas, no solo posición

Mirar como mínimo:

```text
requested_skill
skill_behavior_skill_request
skill_behavior_motion_request
motion_engine_input_request
motion_engine_effective_request
motion_engine_phase
joints de piernas
```

## 8. La ventana visible requiere control exclusivo

Si se usa SimRobot visible:

- la ventana correcta debe estar corriendo
- hay que presionar `F5`
- no debe haber otra instancia consumiendo el mismo `RLSharedState`

## 9. Smoke visual no es entrenamiento

`live_simrobot_probe.py` y `heuristic_agent_experiment.py` sirven para
diagnóstico y sanity check, no como evidencia de aprendizaje PPO.

## 10. Si un proceso nativo muere, limpiar antes de diagnosticar

```bash
rm -f /dev/shm/pybh_rl_shared_state_v3
```

Eso solo borra estado temporal del bridge.
