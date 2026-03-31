# BHuman In Loop: Guia de Cambios para Opcion 1 y Opcion 2

Estado de referencia: 2026-03-30

Documento relacionado:

- [BHUMAN_IN_LOOP_COMPACT_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_COMPACT_GUIDE.md)
  Entrada corta para smokes canonicos, checklist y lectura minima.

Objetivo de este documento:

- dejar una guia operativa, puntual y archivo-por-archivo
- preservar el estado estable actual de BHuman-in-loop
- documentar exactamente que hay que cambiar si mas adelante se abre la opcion 2
- evitar que una futura refactorizacion rompa el loop actual por mezclar ambas rutas

## 1. Regla principal

Las dos opciones NO deben implementarse como un reemplazo destructivo una de la otra.

La forma correcta es:

- mantener la opcion 1 como baseline estable y entrenable
- abrir la opcion 2 como un modo alterno, explicitamente seleccionable
- no borrar la ruta actual hasta que la opcion 2 tenga smokes reales y entrenamiento corto validados

## 2. Escenario A: Opcion 1 estable actual

Descripcion:

- bridge persistente
- `scenario = "2D"`
- `location = "4v4_Scaled"`
- lectura principal desde `Cognition`
- `MotionInfo` sigue viniendo de `cognition_blackboard`
- el takeover real existe en el entorno
- la evidencia de ejecucion de kick para `shoot/pass` aun NO esta cerrada

### 2.1. Archivos fuente de verdad

1. [bridge_protocol.py](/j:/ROBOTICA/RL/bridge/bridge_protocol.py)
   Cambios/estado que deben mantenerse:
   - `BridgeConfig.scenario = "2D"` en la linea actual 220
   - `BridgeConfig.location = "4v4_Scaled"` en la linea actual 221
   - NO volver a `location = "Default"`
   - NO cambiar `scenario = "2D"` dentro de la opcion 1

2. [bhuman_bridge.cpp](/j:/ROBOTICA/SabanaHerons2026/Src/Apps/RLBridge/bhuman_bridge.cpp)
   Anchors actuales relevantes:
   - linea 830: deteccion de `motion_ = findMotionThread()`
   - linea 911: inyeccion de `MotionInfo` base hacia `Cognition`
   - linea 997: `debug_motion_sync`
   - linea 1165: `findMotionThread()`
   - linea 1429: `debug_provider_motion_info`
   - linea 1607: fallback `cognition_blackboard`
   - linea 1770: lectura opcional desde `motion_blackboard`

   Regla en opcion 1:
   - este archivo puede conservar la instrumentacion de `Motion`, pero NO se debe vender como fuente activa de evidencia mientras `scenario` siga en `2D`
   - el valor esperado hoy es `debug_motion_sync = "motion_unavailable"`
   - si `findMotionThread()` devuelve `nullptr`, eso es coherente con la arquitectura actual

3. [bhuman_bridge.cpp](/j:/ROBOTICA/RL/bridge/bhuman_bridge.cpp)
   Regla:
   - debe permanecer sincronizado con la copia en `SabanaHerons2026`
   - cualquier cambio en el bridge C++ real debe replicarse aqui

4. [bhuman_sidecar.py](/j:/ROBOTICA/RL/bhuman_sidecar.py)
   Anchors actuales relevantes:
   - linea 24: `last_kick_type` solo con evidencia real
   - linea 179: comentario de `MotionInfo evidence ONLY`
   - linea 478: parseo de `motion_request`
   - linea 524: parseo de `last_kick_type`

   Regla en opcion 1:
   - NO inventar `last_kick_type`
   - si el bridge no entrega evidencia real, el valor correcto sigue siendo `"unavailable"`
   - NO volver a inferir kick por segunda llamada, por skill elegido o por memoria local

5. [environment.py](/j:/ROBOTICA/RL/environment.py)
   Anchors actuales relevantes:
   - linea 616: activacion de takeover solo si `handoff_ready` y `sidecar_macro_type` es `playBall_*`
   - linea 1165: captura de `sidecar_macro_type`
   - linea 1181: takeover con efecto de fisica
   - linea 1636: `effective_macro_type`
   - linea 1651: `macro_type`
   - linea 1669: `sidecar_macro_type`

   Regla en opcion 1:
   - mantener takeover gatillado por macro real del sidecar, no por invento local
   - mantener `macro_type` y `takeover_macro_type` separados
   - NO endurecer `takeover_success` con evidencia de kick mientras la arquitectura siga en `scenario = "2D"`

6. [reward.py](/j:/ROBOTICA/RL/reward.py)
   Anchors actuales relevantes:
   - linea 144: `_handoff_alignment_reward`
   - lineas 361-404: integracion del shaping

   Regla:
   - la opcion 1 sigue orientada a superar fase 0 con `TkReady`
   - no reescalar reward para exigir `lastKick*` que hoy no existe de forma robusta

7. [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](/j:/ROBOTICA/RL/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
   Anchors actuales relevantes:
   - linea 21: existe takeover real con `completed=True`
   - linea 35: shaping denso de `handoff_alignment`
   - lineas 51-54: gaps abiertos de `MotionInfo` y fixtures positivas

### 2.2. Smokes que SI deben seguir siendo gating en opcion 1

Operativamente, el entrypoint recomendado ya no es correrlos uno por uno, sino:

```powershell
J:/CONDA/envs/RLRobocup/python.exe j:/ROBOTICA/RL/smoke_suite_option1.py --suite gating
```

- [smoke_test_environment_bridge.py](/j:/ROBOTICA/RL/smoke_test_environment_bridge.py)
- [smoke_test_semantic_strong.py](/j:/ROBOTICA/RL/smoke_test_semantic_strong.py)
- [smoke_test_real_takeover_completion.py](/j:/ROBOTICA/RL/smoke_test_real_takeover_completion.py)
- [smoke_test_sidecar.py](/j:/ROBOTICA/RL/smoke_test_sidecar.py)

### 2.3. Smokes/diagnosticos que NO deben venderse como cierre total de opcion 1

- [audit_motioninfo_takeover_evidence.py](/j:/ROBOTICA/RL/audit_motioninfo_takeover_evidence.py)
  Regla:
  - sigue siendo diagnostico
  - hoy puede quedar en `ATTENTION`

- [smoke_test_real_takeover_playball.py](/j:/ROBOTICA/RL/smoke_test_real_takeover_playball.py)
  Regla:
  - su validacion positiva de `pass` ya no debe asumirse estable automaticamente
  - si el fixture de `pass` deja de ser valida con `location = "4v4_Scaled"`, se debe tratar como deuda de fixture, no como smoke verde forzado

### 2.4. Que SI se puede afirmar al cerrar opcion 1

- el backend es real
- el transporte es real y estable
- el entorno usa BHuman real para la decision
- el takeover real existe y puede completarse en `env.step()`
- la fase 0 puede medirse honestamente con `TkReady`

### 2.5. Que NO se debe afirmar al cerrar opcion 1

- que `shoot/pass` ya tienen evidencia completa de ejecucion vía `MotionInfo`
- que `lastKickType` ya es robusto bajo kicks reales
- que el loop actual reproduce toda la cadena de motion de BHuman

## 3. Escenario B: Opcion 2 futura

Descripcion:

- mantener entrenamiento y API Python, pero abrir una ruta de bridge mas profunda
- objetivo: acercarse a ejecucion real de `MotionRequest` y `MotionInfo`
- justificacion: cuando haya mejor CPU y se acepte mayor costo por frame

La referencia documental principal es [BHUMAN_IN_LOOP_SIDECAR_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_SIDECAR_GUIDE.md), seccion `6.2 Opcion B: Headless Embebido`, y la ruta estable actual esta en [BHUMAN_PERSISTENT_BRIDGE_PLAN.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_PERSISTENT_BRIDGE_PLAN.md).

### 3.1. Principio de implementacion

NO reemplazar la opcion 1.

Hay que introducir un selector explicito de modo.

Propuesta recomendada:

- `persistent_2d_cognition` para la opcion 1
- `embedded_2d_motion` para la opcion 2

## 4. Cambios exactos requeridos para abrir opcion 2

### 4.1. [bridge_protocol.py](/j:/ROBOTICA/RL/bridge/bridge_protocol.py)

Agregar un campo nuevo en `BridgeConfig`:

- `bridge_mode: str = "persistent_2d_cognition"`

Actualizar el mensaje `init` para incluir:

- `bridge_mode`

Regla:

- opcion 1 debe seguir siendo el default
- opcion 2 debe activarse solo de forma explicita

Cambios puntuales:

1. Extender `BridgeConfig`.
2. Incluir `bridge_mode` en el JSON de `init`.
3. Exponer `bridge_mode` en logs/debug si el bridge responde `ready`.

### 4.2. [bhuman_sidecar.py](/j:/ROBOTICA/RL/bhuman_sidecar.py)

Cambios puntuales:

1. Permitir que `RealSidecarBackend` construya `BridgeConfig(bridge_mode=...)`.
2. Leer el modo desde variable de entorno o config central.
3. Exponer en `SidecarOutput` o `infos` una marca como:
   - `bridge_mode`
   - `sidecar_bridge_mode`

Regla:

- la logica de parseo semantico NO debe bifurcarse por completo
- solo cambian la procedencia y robustez de `MotionInfo`

### 4.3. [bhuman_bridge.cpp](/j:/ROBOTICA/SabanaHerons2026/Src/Apps/RLBridge/bhuman_bridge.cpp)

Este es el archivo central de la opcion 2.

Debe abrirse en dos caminos internos, no en uno solo:

1. Camino A:
   - `persistent_2d_cognition`
   - dejar el actual tal como esta

2. Camino B:
   - `embedded_2d_motion`
   - usar una ruta inspirada en:
     - `SimulatedRobot2D`
     - `LocalConsole`
   - referencias:
     - [LocalConsole.cpp](/j:/ROBOTICA/SabanaHerons2026/Src/Libs/SimulatedNao/LocalConsole.cpp)
     - [SimulatedRobot2D.cpp](/j:/ROBOTICA/SabanaHerons2026/Src/Libs/SimulatedNao/SimulatedRobot2D.cpp)

Cambios internos recomendados:

1. En `init(...)`
   - parsear `bridge_mode`
   - si el modo es opcion 2, inicializar la infraestructura embebida necesaria
   - NO eliminar la inicializacion actual

2. En `processFrame(...)`
   - opcion 1:
     - conservar la inyeccion actual de `GroundTruthWorldState` a `Cognition`
   - opcion 2:
     - inyectar frame
     - ejecutar la ruta que permita evolucionar `MotionRequest -> MotionInfo`
     - la referencia funcional es la secuencia de `LocalConsole::update()` y `LocalConsole::main()`

3. En `readDecisions(...)`
   - opcion 1:
     - seguir leyendo `MotionInfo` desde `cognition_blackboard` cuando sea lo unico disponible
   - opcion 2:
     - preferir `MotionInfo` de la ruta de ejecucion embebida
     - `last_kick_type` y `last_kick_timestamp_ms` deben salir de esa evidencia, no de la decision

4. Mantener y ampliar diagnosticos:
   - `debug_provider_motion_info`
   - `debug_motion_sync`
   - `debug_completed_motion_frames`
   - agregar si hace falta:
     - `debug_bridge_mode`
     - `debug_motion_info_source`

### 4.4. [bhuman_bridge.cpp](/j:/ROBOTICA/RL/bridge/bhuman_bridge.cpp)

Regla:

- copiar exactamente la misma logica que en `SabanaHerons2026`
- no dejar divergencia entre ambas copias

### 4.5. [environment.py](/j:/ROBOTICA/RL/environment.py)

La opcion 2 NO debe cambiar la semantica base del env inmediatamente.

Cambios permitidos solo despues de que el bridge embebido funcione:

1. endurecer el criterio de `macro_success` para `shoot/pass`
   - permitir usar `last_kick_timestamp`
   - permitir usar `motion_executed_phase`
   - solo si `bridge_mode == "embedded_2d_motion"`

2. agregar a `infos`:
   - `sidecar_bridge_mode`
   - `sidecar_motion_info_source`

Regla:

- no cambiar rewards ni curriculum antes de probar que la nueva evidencia existe

### 4.6. [smoke_test_environment_bridge.py](/j:/ROBOTICA/RL/smoke_test_environment_bridge.py)

Agregar verificaciones nuevas para opcion 2:

- `bridge_debug_motion_sync != "motion_unavailable"`
- `bridge_debug_provider_motion_info` coherente con el nuevo modo

Pero:

- esas verificaciones deben activarse solo cuando el bridge corra en opcion 2

### 4.7. [audit_motioninfo_takeover_evidence.py](/j:/ROBOTICA/RL/audit_motioninfo_takeover_evidence.py)

Este archivo debe migrar de diagnostico a gating SOLO en opcion 2.

Condicion de cierre en opcion 2:

- `shoot` con evidencia real de kick
- `pass` con evidencia real de kick
- `last_kick_type != "unavailable"`
- `last_kick_timestamp_ms > 0`

### 4.8. [smoke_test_real_takeover_playball.py](/j:/ROBOTICA/RL/smoke_test_real_takeover_playball.py)

Cambios requeridos en opcion 2:

- revalidar o reconstruir el fixture positiva de `pass`
- no reciclar un fixture vieja heredada de `location = Default`
- si no existe fixture robusta, dejar el test como diagnostico y NO como PASS artificial

### 4.9. [smoke_test_real_env_semantic_variants.py](/j:/ROBOTICA/RL/smoke_test_real_env_semantic_variants.py)

Objetivo en opcion 2:

- volver a buscar fixtures reales para:
  - `pass`
  - `shoot`
  - `dribble`
  - `duel`

Con diferencia importante:

- en opcion 2 ya no basta semantica del bridge
- se debe preferir fixture con evidencia de ejecucion real cuando aplique

## 5. Orden seguro de implementacion para opcion 2

1. Agregar `bridge_mode` en Python sin cambiar el default.
2. Hacer que el bridge C++ acepte `bridge_mode` sin cambiar comportamiento.
3. Implementar el camino nuevo en paralelo al actual.
4. Agregar diagnosticos del modo nuevo.
5. Crear smokes nuevos especificos para opcion 2.
6. Ejecutar smokes en serie, no en paralelo.
7. Solo despues endurecer criterios de `MotionInfo` en el entorno.

## 6. Checklist de decision entre ambas opciones

Usar opcion 1 si:

- la prioridad es entrenar ya
- la CPU actual es un cuello de botella
- el objetivo inmediato es cerrar fase 0 y avanzar curriculum
- se acepta que `shoot/pass` aun no tienen evidencia completa de `MotionInfo`

Usar opcion 2 si:

- la prioridad pasa a ser fidelidad motion-level
- ya existe hardware con mejor CPU
- se acepta mayor complejidad y menor throughput
- el proyecto necesita afirmar ejecucion real de kicks, no solo decision real

## 7. Regla final de mantenimiento

Cada vez que se toque cualquiera de estos archivos:

- [bridge_protocol.py](/j:/ROBOTICA/RL/bridge/bridge_protocol.py)
- [bhuman_sidecar.py](/j:/ROBOTICA/RL/bhuman_sidecar.py)
- [environment.py](/j:/ROBOTICA/RL/environment.py)
- [bhuman_bridge.cpp](/j:/ROBOTICA/SabanaHerons2026/Src/Apps/RLBridge/bhuman_bridge.cpp)
- [bhuman_bridge.cpp](/j:/ROBOTICA/RL/bridge/bhuman_bridge.cpp)

hay que responder explicitamente estas preguntas:

1. Estoy cambiando opcion 1, opcion 2, o ambas?
2. El cambio rompe el baseline entrenable actual?
3. El cambio inventa `MotionInfo` o la obtiene de ejecucion real?
4. El smoke que queda verde sigue siendo honesto bajo `4v4_Scaled`?

Si alguna respuesta no esta clara, NO mezclar ambas opciones en el mismo parche.
