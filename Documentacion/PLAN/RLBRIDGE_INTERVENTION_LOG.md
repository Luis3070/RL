# RLBridge Intervention Log

> Fecha: 25 de marzo de 2026  
> Scope: `bridge/bhuman_bridge.cpp`, build/integration con `C:/Users/Alegu/Sabana2026`, smoke tests manuales de `RLBridge.exe`

---

## 1. Problema recurrente observado

El fallo que se repitio varias veces fue un `access violation` en `RLBridge.exe` durante el flujo:

- `init`
- `frame`
- espera de respuesta de Cognition
- cierre o retorno de decisiones

Firma observada repetidamente:

- `0xc0000005`
- lectura de memoria nula
- ventana modal de Windows o `exit code -1073741819`

No era el mismo bug original de `DebugReceiver(..., nullptr, ...)`. Ese ya estaba corregido.  
La nueva familia de fallo aparecia despues de procesar un frame, no en startup puro.

---

## 2. Causa raiz confirmada

La causa raiz confirmada de esta iteracion fue el uso de `debug requests` del tipo:

- `representation:GameState`
- `representation:SkillRequest`
- `representation:MotionRequest`
- `representation:MotionInfo`
- `representation:StrategyStatus`
- `representation:BehaviorStatus`
- `representation:ActivationGraph`

como mecanismo de transporte para snapshots de decisiones.

### Evidencia

Con esos requests activos:

- el frame entraba a BHuman
- `GameStateProvider` podia reportar `GC Active: YES`
- pero el proceso moria antes de completar `readDecisions()`

Con esos requests desactivados:

- el bridge procesaba `frame`
- recibia `idLogResponse`
- recibia `idFrameFinished`
- devolvia JSON
- cerraba con `exit code 0`

Conclusion:

- la inyeccion 2D base del frame no era el problema
- el problema estaba en el camino de snapshot basado en `representation:*`

---

## 3. Cambios aplicados y dejados activos

### 3.1. Bridge estable sin `representation:*`

Se dejo desactivado el bloque de `representation:*` como via normal de snapshot.

Estado actual:

- el bridge ya no usa esos requests como fuente normal de lectura
- el bridge sigue usando el canal debug solo para:
  - inyectar frame
  - esperar `idLogResponse` / `idFrameFinished`

### 3.2. Crash handling util

Se agrego manejo explicito de excepcion no controlada en Windows para evitar depender solo de la ventana modal y dejar trazas por consola.

Objetivo:

- si vuelve a aparecer un crash nativo
- quede `exception code` y `address` en `stderr`

### 3.3. Lectura directa desde Blackboard de Cognition

Se implemento un fallback controlado:

- esperar `idLogResponse` / `idFrameFinished`
- si el snapshot del canal debug sigue vacio
- leer directamente del blackboard de `Cognition` con `cognition_->setGlobals()`

Esto permite extraer actualmente:

- `GameState`
- `SkillRequest`
- `MotionRequest`
- `MotionInfo`
- `StrategyStatus`
- `BehaviorStatus`
- `ActivationGraph`

La procedencia queda marcada en el JSON con:

- `debug_provider_* = "cognition_blackboard"`

### 3.4. Restauracion del arbol externo a estado compilable

Durante la investigacion se intento una ruta mas limpia:

- emitir snapshots propios desde `ModuleContainer`/`Cognition`
- usando un request dedicado `rlbridge:snapshot`

Ese experimento no se dejo activo en el arbol externo porque:

- obligaba a tocar `Framework`
- el target `Framework` presento acoplamiento incomodo de include paths al compilar representaciones desde `ModuleContainer.cpp`

Por eso, para no dejar `Sabana2026` roto:

- `Framework/ModuleContainer.cpp`
- `Framework/ModuleContainer.h`
- `Make/CMake/Framework.cmake`

se devolvieron a un estado limpio y compilable.

---

## 4. Estado tecnico alcanzado

### 4.1. Lo que ya funciona

- `init -> frame -> shutdown` estable
- `exit code 0`
- sin `access violation`
- lectura real de representaciones desde Blackboard de Cognition
- salida JSON enriquecida con:
  - `activation_graph`
  - `motion_request`
  - `motion_executed_phase`
  - `strategy_role`
  - `behavior_status`
  - etc.

### 4.2. Smoke valido de esta iteracion

Smoke usado:

- `bridge/run_rlbridge_frame_only.cmd`
- input: `bridge/rlbridge_smoke_input_frame_only.jsonl`

Resultado valido observado:

- `processFrame: cognition frame finished`
- `readDecisions: direct blackboard fallback`
- decision JSON devuelta
- `exit code 0`

---

## 5. Limitaciones actuales del bridge

Aunque el bridge ya extrae decisiones reales desde Blackboard, todavia no debe venderse como solucion final.

Limitaciones:

1. La lectura directa del blackboard de otro thread no es la arquitectura ideal.
2. Es util como bridge experimental estable, pero no como mecanismo final mas robusto del proyecto.
3. `GameState` / `PlayerState` siguen llegando como:
   - `beforeHalf`
   - `unstiff`
   en el smoke actual.
4. Por tanto, la ruta de activacion operacional de jugador/estado sigue incompleta para obtener semantica de partido realmente util.

---

## 6. Ruta recomendada para la siguiente intervencion

La siguiente ruta recomendada no es volver a `representation:*`.

La ruta correcta es:

1. Mantener desactivado `representation:*` en el bridge.
2. Mantener el fallback a Blackboard como camino estable de trabajo.
3. Investigar por que `GameState` y `PlayerState` siguen en:
   - `beforeHalf`
   - `unstiff`
4. Si se quiere un canal definitivo, retomarlo como:
   - snapshot dedicado por thread de Cognition
   - pero fuera de esta iteracion
   - y sin dejar `Framework` externo en estado no compilable

---

## 7. Archivos tocados en esta iteracion

En este repo:

- `bridge/bhuman_bridge.cpp`
- `bridge/ModuleContainer.external.cpp`
- `bridge/ModuleContainer.external.h`
- `bridge/Framework.external.cmake`

En arbol externo BHuman durante pruebas:

- `C:/Users/Alegu/Sabana2026/Src/Apps/RLBridge/bhuman_bridge.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Libs/Framework/ModuleContainer.cpp` (solo temporalmente, luego restaurado)
- `C:/Users/Alegu/Sabana2026/Src/Libs/Framework/ModuleContainer.h` (solo temporalmente, luego restaurado)
- `C:/Users/Alegu/Sabana2026/Make/CMake/Framework.cmake` (solo temporalmente, luego restaurado)

---

## 8. Regla operativa desde ahora

Hasta nueva intervencion:

- `representation:*` queda descartado como via normal del RL bridge
- el camino estable es:
  - `frame` por canal debug
  - `ack`/`FrameFinished` por `RLBridgeConsole`
  - lectura de decisiones desde Blackboard de Cognition

---

## 9. Smoke test operacional del environment

Para no depender solo del smoke manual de `RLBridge.exe`, se agrego un smoke end-to-end del lado Python:

- script: `smoke_test_environment_bridge.py`
- launcher simple: `bridge/run_env_bridge_smoke.cmd`

Objetivo:

- verificar que `NaoSoccerEnv` levanta el sidecar con backend `real`
- verificar que `env.step()` envia frames al bridge real
- verificar que el `info` del env ya expone metadata cruda del bridge
- verificar que `env.reset()` resetea el contador/frame state del bridge
- verificar que `env.close()` cierra el subprocess limpio

---

## 10. Causa raiz real del popup/modal repetido de `bhuman_bridge.exe`

En la iteracion posterior se confirmo que la familia de popup recurrente en Windows no era un bug nuevo del frame, sino una combinacion de decisiones del wrapper Python:

- intentar primero `pipe_process` en Windows
- caer despues a one-shot
- lanzar el one-shot separando `stdout` y `stderr`
- y ademas forzar `BHUMAN_ROOT`

El patron estable que si funcionaba era el del smoke manual:

- `.cmd`
- redireccion `2>&1`
- sin depender de la ruta de `pipe_process`

### Correccion aplicada

1. `bridge_protocol.py` usa ahora `oneshot_file_fallback` como ruta normal en Windows.
2. El one-shot ya no separa `stdout` y `stderr`; usa una sola salida combinada.
3. `bridge_protocol.py` ya no fuerza `BHUMAN_ROOT` al arrancar el bridge.
4. `bhuman_bridge.cpp` ya etiqueta correctamente los mensajes del receiver:
   - `idFrameBegin`
   - `idLogResponse`
   - `idFrameFinished`

### Resultado

- desaparece el popup recurrente en el camino normal del smoke
- `smoke_test_environment_bridge.py` pasa completo
- el bridge sigue devolviendo decision JSON real
- la metadata del bridge llega bien a `info`

---

## 11. Estado operativo actual

Estado validado al cierre de esta intervencion:

- `bridge/run_rlbridge_frame_only.cmd`: `PASS`
- `smoke_test_environment_bridge.py`: `PASS`
- backend del sidecar: `real`
- modo contractual de fallback: `False`
- transporte estable de Windows: `oneshot_file_fallback`

Lo que sigue abierto ya no es transporte ni crash modal, sino semantica operacional de BHuman:

- `GameState = beforeHalf`
- `PlayerState = unstiff`

Ese ya es el siguiente frente de trabajo.

Este smoke NO valida semantica futbolistica final. Valida comunicacion e integracion.

### Senales operacionales que debe comprobar

En `info` deben aparecer al menos:

- `sidecar_backend = "real"`
- `sidecar_contract_validation_mode = False`
- `takeover_simulator_active = False`
- `bridge_frame_id`
- `bridge_decision_time_us`
- `bridge_debug_frame_sync = "ack"`
- `bridge_debug_recent_messages` conteniendo `idFrameFinished`
- `bridge_debug_provider_* = "cognition_blackboard"`

### Criterio de aprobado

Se considera aprobada la comunicacion `RL <-> BHuman` a nivel environment si:

1. el env arranca con backend real
2. un `step()` produce `bridge_frame_id >= 1`
3. un segundo `step()` incrementa el `bridge_frame_id`
4. un `reset()` seguido de `step()` reinicia el `bridge_frame_id` a `1`
5. el `info` refleja providers reales del blackboard
6. el cierre del env no deja colgado el subprocess del bridge

---

## 10. Ajustes de integracion aplicados para soportar el smoke

Para soportar este smoke operacional se dejaron activos estos cambios en Python:

- `bhuman_sidecar.py`
  - preserva metadata de bridge real (`frame_id`, `decision_time_us`, `debug_*`)
- `environment.py`
  - hace `sidecar.reset()` en `env.reset()`
  - expone metadata de bridge en `info`
  - implementa `close()` para apagar el subprocess del sidecar
- `wrapper.py`
  - implementa `close()` delegando al env

Esto deja una base reproducible para futuras intervenciones:

- si el smoke de `RLBridge.exe` pasa pero el smoke del environment falla, el problema esta en la capa Python-sidecar-env
- si ambos fallan, el problema vuelve a estar en el bridge/binario o en el wiring con BHuman

---

## 11. Estado actual del smoke `environment <-> RLBridge`

Se agrego el validador:

- `smoke_test_environment_bridge.py`
- launcher: `bridge/run_env_bridge_smoke.cmd`

### Lo que ya valida bien

- resolucion del ejecutable real del bridge
- uso del environment Python con sidecar `real`
- propagacion de metadata del bridge a `info`
- `env.reset()` llama `sidecar.reset()`
- `env.close()` apaga el subprocess del sidecar
- el wrapper Python ya distingue:
  - `pipe_process`
  - `oneshot_file_fallback`

### Estado observado al 25 de marzo de 2026

Resultado actual del smoke:

- `FAIL`

Causa inmediata observada:

1. El transporte `pipe_process` llega a:
   - `ready`
   - primera linea diagnostica en `stdout`
   - luego timeout de frame
2. El wrapper hace fallback a `oneshot_file_fallback`
3. El bridge real (`RLBridge.exe` externo) sigue cayendo despues de:
   - `processFrame: frame sent`
4. La firma del fallo sigue siendo:
   - `0xc0000005`
   - access violation

### Conclusion operativa actual

El smoke del environment YA esta listo como herramienta de validacion.

Lo que todavia NO esta listo es la capa subyacente del bridge:

- hoy el test no demuestra comunicacion efectiva
- hoy demuestra, de forma reproducible, que el bridge real sigue rompiendose al procesar el frame dentro del loop Python

Eso significa que esta parte del proyecto queda mejor cerrada asi:

- el testing de environment ya existe
- el criterio de aprobado existe
- la salida `PASS/FAIL` es reproducible
- el siguiente bloqueo real ya no esta en el env sino en `RLBridge.exe`
  - inyeccion de frame por debug sender
  - sincronizacion por `idLogResponse` / `idFrameFinished`
  - lectura fallback desde Blackboard de Cognition

Si vuelve a aparecer un crash, revisar primero:

- si alguien reintrodujo `representation:*`
- si se altero el orden de inyeccion 2D
- si se volvio a tocar el wiring entre `Debug` y `RLBridgeConsole`

---

## 12. Cierre de la fase de transporte y warmup operacional

En la intervencion siguiente se cerro el bloqueo que quedaba entre "bridge estable" y "agente BHuman realmente operativo dentro del env".

### Causa raiz final de la semantica `beforeHalf / unstiff`

No bastaba con que el frame llegara a Cognition.

Habia dos problemas adicionales:

1. `GameControllerData` estaba dejando demasiados jugadores como activos.
   - BHuman terminaba viendo demasiados agentes en `StrategyBehaviorControl`.
   - eso contaminaba el preprocesamiento tactico y la activacion del rol propio.
2. Un salto directo a `PLAYING` no siempre despertaba correctamente la cadena operacional.
   - habia que bootstrapear la maquina interna de estado:
     - `initial`
     - `ready`
     - `set`
     - `playing`

### Correcciones aplicadas

1. `buildGameControllerData(...)` fue corregido para marcar como `PENALTY_NONE` solo a los jugadores realmente presentes en:
   - `own_team`
   - `opp_team`
   y dejar el resto como `PENALTY_SUBSTITUTE`.
2. `processFrame(...)` en `bridge/bhuman_bridge.cpp` ahora hace bootstrap interno cuando el frame pedido es operacional.
3. `reset()` vuelve a bajar el warmup interno para que el episodio siguiente arranque limpio.

### Resultado validado

Smoke manual del bridge:

- `bridge/run_rlbridge_frame_only.cmd`
- `exit code 0`
- JSON final con:
  - `debug_game_state = playing`
  - `debug_player_state = active`
  - `skill_request = walk`
  - `strategy_role = goalkeeper`
  - `current_tactic = t121`
  - `debug_bootstrap_steps = 4`

Smoke end-to-end del environment:

- `smoke_test_environment_bridge.py`
- resultado:
  - `PASS`
- evidencia:
  - `sidecar_backend = real`
  - `sidecar_contract_validation_mode = False`
  - `takeover_simulator_active = False`
  - `bridge_debug_frame_sync = ack`
  - `bridge_frame_id` incrementa correctamente
  - despues de `env.reset()`, el primer frame vuelve a `1`

### Estado final de esta fase

Lo que ya quedo cerrado:

- transporte estable de Windows
- `RLBridge.exe` estable en loop
- bridge real conectado al env Python
- bootstrap operacional suficiente para llegar a `playing / active`
- smoke manual del bridge pasando
- smoke end-to-end del environment pasando

Limpieza aplicada despues del cierre:

- se retiraron del arbol externo de BHuman las trazas temporales fuertes de:
  - `ModuleGraphRunner`
  - `StrategyBehaviorControl`
- en `GameStateProvider` se mantuvo solo el fix minimo funcional de `playerNumber`
- el smoke `environment <-> RLBridge` sigue pasando despues de esa limpieza

Lo que queda como siguiente fase:

- enriquecer la semantica util del agente BHuman dentro del env
- limpiar instrumentacion temporal del arbol externo de BHuman si ya no hace falta
- endurecer mas escenarios de smoke para roles/skills/macroestados especificos
