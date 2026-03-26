# Guia de Correccion para Sidecar y Loop BHuman-Aligned

> Fecha: 24 de marzo de 2026
> Objetivo: dejar una sola guia operativa para que la siguiente iteracion corrija los bloqueos reales del sidecar y del loop normal, y además abra una ruta concreta para llegar a `BHuman in loop` real.
> Scope principal: `bhuman_sidecar.py`, `environment.py`, `smoke_test_sidecar.py`, `config.py`, `reward.py`, `wrapper.py`, `train.py`.
> Scope secundario: `bhuman_cfg_parser.py` solo para conservar la carga real desde CFGs y, si hace falta, exponer mejor la procedencia del contrato.

Nota de continuidad:

- historial especifico del bridge: [RLBRIDGE_INTERVENTION_LOG.md](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/Documentacion/PLAN/RLBRIDGE_INTERVENTION_LOG.md)
- guia detallada de uso del env y arquitectura actual: [ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/Documentacion/PLAN/ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md)
- para `RLBridge`, la ruta estable actual ya no debe asumir `representation:*` como mecanismo seguro de snapshot
- la base operativa actual del bridge es `ack por debug channel + fallback directo al blackboard de Cognition`
- smoke test operacional de comunicacion env <-> bridge: `smoke_test_environment_bridge.py`
- estado actual de ese smoke: `PASS`
- bloqueo critico anterior ya resuelto:
  - en Windows, `pipe_process` no es la ruta estable para `RLBridge`
  - el transporte estable actual es `oneshot_file_fallback` via `.cmd` con `stdout + stderr` combinados
- `bridge_protocol.py` no debe volver a:
  - forzar `BHUMAN_ROOT` al lanzar el bridge
  - separar `stdout` y `stderr` en el transporte one-shot de Windows
- el siguiente foco ya no es transporte/env crash, sino semantica operacional de `GameState` / `PlayerState`

---

## 1. Veredicto rapido

Estado real del proyecto hoy:

- el parser ya puede cargar CFGs reales
- `config.py` ya resuelve `cfg_root` de forma interna con fallback repo-relative
- `wrapper.py` ya dejo de medir readiness con `takeover_count > 0`
- el sidecar ya puede seleccionar backend `real`
- el smoke del environment ya existe y ya detecta `backend_label = real`
- la metadata del bridge ya llega hasta `info` del env
- `environment.py` ya no usa takeover local simulado en modo normal
- `RLBridge.exe` ya no cae en el flujo normal del environment smoke
- el transporte estable en Windows ya quedo fijado como `oneshot_file_fallback`
- el env smoke ya pasa end-to-end con backend `real`
- `RLBridge` ya hace bootstrap interno de `initial -> ready -> set -> playing` cuando el env pide `playing`
- el smoke real ya reporta `debug_game_state = playing` y `debug_player_state = active`
- el smoke test del sidecar demuestra coherencia contractual y ausencia de leaks obvios, pero no cumplimiento del contrato fuerte de `BHuman in loop`

Conclusion operativa:

- no hay que rehacer parser ni contrato numerico desde cero
- no hay que seguir invirtiendo en parser/env solo para "arreglar transporte"; esa parte ya quedo funcional
- si hay que preservar el launcher estable de Windows:
  - `.cmd`
  - `2>&1`
  - `oneshot_file_fallback`
- el smoke `smoke_test_environment_bridge.py` ya es la herramienta correcta para validar si la comunicacion quedo operativa
- la siguiente iteracion debe enfocarse en semantica BHuman real, no en reabrir el bug de transporte
- la siguiente iteracion ya no debe enfocarse en "hacer que arranque", sino en enriquecer la semantica util del agente BHuman dentro del env

---

## 2. Fuentes de verdad obligatorias

Claude debe trabajar usando esta guia junto con:

- `Documentacion/PLAN/BHUMAN_IN_LOOP_SIDECAR_GUIDE.md`
- `Documentacion/PLAN/BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md`
- `Documentacion/PLAN/BHUMANPARSER_GUIDE.md`
- `Documentacion/CONTEXTO BHUMAN/BHUMAN_RL_CURRICULUM_CONTRACT.md`

Y para semantica real de BHuman debe mirar estos archivos del proyecto BHuman:

- `C:/Users/Alegu/Sabana2026/Src/Representations/BehaviorControl/SkillRequest.h`
- `C:/Users/Alegu/Sabana2026/Src/Representations/MotionControl/MotionRequest.h`
- `C:/Users/Alegu/Sabana2026/Src/Representations/BehaviorControl/ActivationGraph.h`
- `C:/Users/Alegu/Sabana2026/Src/Representations/MotionControl/MotionInfo.h`
- `C:/Users/Alegu/Sabana2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/PlayBall.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Representations/Infrastructure/GroundTruthWorldState.h`
- `C:/Users/Alegu/Sabana2026/Src/Modules/Modeling/OracledWorldModelProvider/OracledWorldModelProvider.cpp`

---

## 3. Reglas de trabajo para Claude

1. No re-auditar parser ni CFGs desde cero.
2. No simplificar `kickInfo` ni recortar patadas por intuicion.
3. No usar `.cpp/.h` como parser numerico principal.
4. Si usar `.cpp/.h` como fuente semantica obligatoria del bridge.
5. No declarar `BHuman in loop` mientras el modo normal siga usando el espejo Python o el takeover simulator local.
6. No dejar `skill_request = "duel"` como si fuera una interfaz real de BHuman.
7. No dejar `last_kick_type` como "ultimo kick elegido"; debe venir de evidencia de ejecucion o quedar ausente.
8. No expandir la observacion RL mas alla de `is_playing` por exceso de fidelidad.
9. Si el backend real del sidecar no esta disponible, el modo normal debe fallar fuerte o degradar a un modo explicitamente marcado, nunca caer en silencio a una emulacion Python.
10. Si `smoke_test_environment_bridge.py` vuelve a fallar en Windows, verificar primero que nadie haya reintroducido:
   - `pipe_process` como ruta normal
   - separacion de `stdout` y `stderr` en one-shot
   - inyeccion forzada de `BHUMAN_ROOT`

---

## 4. Hallazgos resueltos y hallazgos que no deben reabrirse

## H0. Resuelto: el crash prioritario de `RLBridge.exe` despues de `processFrame: frame sent`

Ubicacion:

- `bridge/bhuman_bridge.cpp`
- `bridge/bridge_protocol.py`
- `smoke_test_environment_bridge.py`
- `C:/Users/Alegu/Sabana2026/Src/Apps/RLBridge/bhuman_bridge.cpp`
- `C:/Users/Alegu/Sabana2026/Build/Windows/RLBridge/Release/RLBridge.exe`

Problema observado originalmente:

- el sidecar ya logra seleccionar backend `real`
- el validador `smoke_test_environment_bridge.py` ya alcanza:
  - `env.reset()`
  - primera llamada a `env.step()`
  - arranque del bridge real
- en Windows, el transporte por pipes era fragil
- el wrapper Python hacia fallback demasiado tarde
- el one-shot del wrapper separaba `stdout` y `stderr`
- ademas forzaba `BHUMAN_ROOT` al arrancar el bridge
- esa combinacion reintroducia la familia de crash/modal:
  - `0xc0000005`
  - ventana de Windows de `bhuman_bridge.exe`
  - o `exit code -1073741819`

Evidencia ya confirmada:

1. El smoke manual del bridge externo via `.cmd` y `2>&1` si era estable.
2. El fallo recurrente aparecia al lanzar el bridge desde `bridge_protocol.py`.
3. El wrapper Python estaba usando una ruta de lanzamiento distinta a la ruta manual estable.
4. El problema no era el frame conocido-bueno, sino el transporte de Windows usado por el wrapper.
5. El naming de `debug_recent_messages` tambien estaba incompleto y ocultaba parte del diagnostico.

Correccion aplicada:

1. `bridge_protocol.py` ahora usa en Windows `oneshot_file_fallback` como ruta normal.
2. El one-shot ya no separa `stdout` y `stderr`; usa `2>&1`, igual que el smoke manual estable.
3. El wrapper ya no fuerza `BHUMAN_ROOT` al lanzar el bridge.
4. `bhuman_bridge.cpp` ya etiqueta correctamente:
   - `idFrameBegin`
   - `idLogResponse`
   - `idFrameFinished`
5. `RLBridge.exe` externo fue recompilado con ese fix de observabilidad.
6. `smoke_test_environment_bridge.py` ya pasa end-to-end.

Estado aprobado actual:

1. `bridge/run_rlbridge_frame_only.cmd` devuelve decision JSON y `exit code 0`.
2. `smoke_test_environment_bridge.py` pasa.
3. `sidecar_backend = real`.
4. `sidecar_contract_validation_mode = False`.
5. `bridge_debug_recent_messages` contiene `idFrameFinished`.
6. No aparece la ventana modal de crash en el camino normal del smoke.
7. `debug_game_state = playing`.
8. `debug_player_state = active`.
9. `smoke_test_environment_bridge.py` confirma `sidecar_backend = real` y `sidecar_contract_validation_mode = False`.

Regla de no regresion:

- No reintroducir `pipe_process` como camino preferido en Windows.
- No reintroducir `stderr` separado en one-shot.
- No volver a forzar `BHUMAN_ROOT` salvo que exista una necesidad nueva y demostrada.

Prioridad: `RESUELTO / NO REABRIR SIN EVIDENCIA`

---

## H0.1. Resuelto: bootstrap operacional interno del bridge

Ubicacion:

- `bridge/bhuman_bridge.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Apps/RLBridge/bhuman_bridge.cpp`
- `smoke_test_environment_bridge.py`

Problema observado:

- aunque el transporte ya era estable, BHuman seguia respondiendo con estados pobres del tipo:
  - `beforeHalf`
  - `unstiff`
- el env enviaba un frame `playing`, pero BHuman no siempre quedaba operacional con un solo salto directo
- durante el diagnostico tambien se detecto que `GameControllerData` estaba dejando demasiados jugadores como activos, lo que inflaba `agents` y rompia la ruta de `StrategyBehaviorControl`

Correccion aplicada:

1. `buildGameControllerData(...)` ahora marca como `PENALTY_NONE` solo a los jugadores realmente presentes en `own_team` y `opp_team`; el resto quedan como `PENALTY_SUBSTITUTE`.
2. `processFrame(...)` ahora hace bootstrap interno cuando el env pide estados operacionales:
   - `ready` -> `initial, ready`
   - `set` -> `initial, ready, set`
   - `playing` -> `initial, ready, set, playing`
3. `reset()` baja `operationalWarmupDone_ = false` para que un episodio nuevo vuelva a inicializar correctamente el camino operacional.
4. una vez logrado el warmup, el bridge vuelve a procesar un frame por step normal.

Evidencia de aprobado:

1. el smoke manual del bridge devuelve JSON con:
   - `debug_game_state = playing`
   - `debug_player_state = active`
2. `smoke_test_environment_bridge.py` pasa.
3. el `info` del env expone:
   - `bridge_debug_frame_sync = ack`
   - `bridge_debug_provider_* = cognition_blackboard`
   - `bridge_frame_id` incremental
4. el primer frame despues de `env.reset()` vuelve a ser `1`.

Pendiente real despues de este fix:

- limpiar la instrumentacion temporal del arbol externo de BHuman cuando ya no haga falta
- pasar de "agente BHuman operacional en loop" a "semantica util cada vez mas rica" sin reabrir el transporte ni el warmup

---

## H1. El bridge real a BHuman todavia no existe

Ubicacion:

- `bhuman_sidecar.py:1-22`
- `bhuman_sidecar.py:180-286`
- `bhuman_sidecar.py:388-413`
- `bhuman_sidecar.py:420-456`
- `bhuman_sidecar.py:463-673`

Problema:

- `RealSidecarBackend` existe solo como placeholder
- no hay bridge real a BHuman C++
- no hay IPC/FFI/pybind11/backend externo operativo
- por tanto el modo `bhuman_sidecar` actual no es una ejecucion real de BHuman, sino `contract_validation_mode` apoyado en el stub Python

Por que rompe contrato:

- el proyecto necesita que la decision macro y sus evidencias bajen desde la cadena real de BHuman
- si Python vuelve a decidir localmente, la brecha semantica sigue viva aunque los nombres parezcan correctos

Correccion obligatoria:

1. Implementar de verdad `RealSidecarBackend`.
2. El backend real debe aceptar un schema tipo `GroundTruthFrame`.
3. El backend real debe llamar la cadena BHuman descrita en `BHUMAN_IN_LOOP_SIDECAR_GUIDE.md`:
   - `StrategyBehaviorControl`
   - `SkillBehaviorControl`
   - `PlayBall`
   - `ActivationGraph`
   - `MotionRequest`
   - `MotionInfo`
4. El modo normal `bhuman_sidecar` debe usar ese backend real cuando exista.
5. Si el backend real falla al inicializar, el modo debe quedar explicitamente marcado como no contractual.
6. El espejo Python solo puede sobrevivir como:
   - `python_stub_smoke_only`
   - pruebas de contrato
   - depuracion

Prioridad: `CRITICA`

---

## H2. `environment.py` ya no usa takeover local en modo normal, pero sigue sin tener ejecucion real de BHuman

Ubicacion:

- `environment.py:636-694`
- `environment.py:729-730`
- `environment.py:893-918`
- `environment.py:919-939`

Problema:

- en `bhuman_sidecar` ya no se usa `BHumanTakeoverSimulator`
- pero si el backend real no existe, el env solo recibe evidencia de decision
- eso significa que el macro puede existir sin ejecucion motora real de BHuman
- en ese estado el loop es util para validacion contractual, no para `BHuman in loop` real

Por que rompe contrato:

- el macro puede estar etiquetado por el sidecar
- pero la ejecucion del takeover sigue siendo una fisica local simplificada
- eso contamina reward, duration, success, end reasons y outcome del balon

Correccion obligatoria:

1. Mantener que en modo normal `bhuman_sidecar`, `environment.py` NO use `BHumanTakeoverSimulator`.
2. El takeover simulator debe seguir permitido solo en:
   - `python_stub_smoke_only`
   - tests explicitamente marcados
   - debug contractual
3. Cuando exista backend real, integrar las salidas reales del sidecar al ciclo de `env.step()` como fuente de verdad de:
   - `active_agent`
   - `skill_request`
   - `motion_request`
   - `activation_graph`
   - `last_kick_type`
4. Si el backend real del sidecar solo da decision y no world rollout, el entorno debe operar en uno de estos dos modos conservadores:
   - `shadow decision mode`: sidecar produce evidencias pero no se entrena macro success
   - `contract validation mode`: se valida semantica sin pretender ejecucion fisica real
5. Lo que NO puede seguir ocurriendo es:
   - sidecar para decidir
   - takeover simulator local para ejecutar
   - y luego llamarlo `BHuman in loop`

Prioridad: `CRITICA`

---

## H3. El smoke test del sidecar no demuestra carga real de CFGs

Ubicacion:

- `smoke_test_sidecar.py:50-52`
- `smoke_test_sidecar.py:98-105`
- `bhuman_cfg_parser.py:888-904`
- `config.py:34-55`

Problema:

- `_make_contract()` llama `load_contract()` sin `cfg_root`
- si no se pasa ruta, el parser puede usar defaults embebidos
- por tanto el smoke test del sidecar no obliga a leer CFGs reales aunque el parser ya soporte hacerlo

Por que rompe contrato:

- el test puede pasar aun si el sidecar nunca toca los CFGs reales de BHuman
- eso invalida la parte mas importante del smoke test de integracion

Correccion obligatoria:

1. Extraer un resolvedor comun de `cfg_root` reutilizable por:
   - `config.py`
   - `contract_smoke_test.py`
   - `smoke_test_sidecar.py`
2. El smoke test del sidecar debe cargar por defecto CFGs reales repo-relative.
3. Debe fallar si termina cayendo a defaults en una prueba marcada como "real sidecar integration".
4. Si se quiere conservar una suite con defaults, dejarla separada y nombrada como `stub/default contract tests`.
5. Idealmente exponer una senal de procedencia del contrato:
   - `contract_source = "cfg_files"` o equivalente

Prioridad: `ALTA`

---

## H4. El schema de salida del sidecar no coincide con BHuman real

Ubicacion:

- `bhuman_sidecar.py:41-73`
- `bhuman_sidecar.py:535-575`
- `bhuman_sidecar.py:637-672`

Problema:

- `skill_request` es un string simple e incluye `"duel"`
- `motion_request` es un string simplificado (`walk`, `kick`, `stand`, `specialAction`)
- `activation_graph` es `List[str]`
- todo eso se parece a BHuman, pero no coincide con sus representaciones reales

Semantica real de BHuman que debe respetarse:

- `SkillRequest` real NO tiene `duel`
- `duel` aparece como subestado de `PlayBall`, no como `SkillRequest::duel`
- `MotionRequest` real usa enums como:
  - `walkToBallAndKick`
  - `dribble`
  - `walkToPose`
  - etc.
- `ActivationGraph` real es una lista de nodos estructurados, no strings libres

Correccion obligatoria:

1. Redefinir la salida contractual del sidecar para que sea estructurada y cercana a BHuman.
2. `skill_request` debe reflejar el enum real de BHuman.
3. `macro_type = playBall_duel` debe salir de:
   - `ActivationGraph`
   - o de un campo explicito de subestado real equivalente
4. `duel` no debe seguir apareciendo como `skill_request`.
5. `motion_request` debe reflejar el enum/motion real o una proyeccion contractual fiel de ese enum.
6. `activation_graph` debe pasar de `List[str]` a nodos estructurados:
   - `option`
   - `depth`
   - `state`
   - `option_time`
   - `state_time`
   - `parameters`

Prioridad: `CRITICA`

---

## H5. `active_agent` y `last_kick_type` siguen siendo aproximados

Ubicacion:

- `bhuman_sidecar.py:388-413`
- `bhuman_sidecar.py:573`
- `bhuman_sidecar.py:670`
- `C:/Users/Alegu/Sabana2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Representations/MotionControl/MotionInfo.h`

Problema:

- el sidecar actual elige active agent por cercania simple al balon
- BHuman real usa una logica mas rica de asignacion
- `last_kick_type` hoy es el ultimo kick elegido por el decider
- pero en BHuman real `lastKickType` pertenece a `MotionInfo`, o sea, a ejecucion real de motion

Por que rompe contrato:

- el equipo RL puede coordinarse alrededor del jugador equivocado
- `last_kick_type` puede decir que hubo kick aunque solo hubo decision y no ejecucion

Correccion obligatoria:

1. En modo normal, `active_agent` debe venir del backend real o de una capa que reproduzca la logica real de `determineActiveAgent`, no del heuristico nearest-to-ball.
2. El heuristico actual solo puede vivir en stub mode.
3. `last_kick_type` debe:
   - venir de evidencia de motion real
   - o permanecer como `none/unavailable` si esa evidencia no existe aun
4. No poblar `last_kick_type` "despues de una segunda llamada" solo porque el decider ya se acuerdo del kick previo.

Prioridad: `ALTA`

---

## H6. El smoke test del sidecar era demasiado laxo y todavia debe usarse como prueba de integracion real, no solo de contrato

Ubicacion:

- `smoke_test_sidecar.py:151-153`
- `smoke_test_sidecar.py:168-171`
- `smoke_test_sidecar.py:185-190`
- `smoke_test_sidecar.py:242-246`
- `smoke_test_sidecar.py:297-300`
- `smoke_test_sidecar.py:350-353`
- `smoke_test_sidecar.py:360-379`

Problema:

- el escenario de pase acepta `pass` o `dribble`
- el escenario de duelo valida `skill_request == "duel"`
- `last_kick_type` se considera correcto tras una segunda llamada, no tras una ejecucion real
- "stub not used" solo verifica que el objeto sidecar exista y que el modo diga `bhuman_sidecar`
- no verifica que el modo normal este libre del takeover simulator local

Por que rompe contrato:

- el test puede pasar aunque la semantica principal siga mal
- eso genera falsa confianza

Correccion obligatoria:

1. Hacer fixtures mas deterministas para `shoot`, `pass`, `dribble`.
2. En el test de `duel`, validar:
   - `macro_type == playBall_duel`
   - `skill_request != "duel"`
   - `activation_graph` o `substate` indica `zweikampf`
3. En el test de `last_kick_type`, exigir evidencia de ejecucion o dejar el campo como no disponible.
4. En `stub not used`, verificar tambien que:
   - el takeover simulator no se usa en modo normal
   - el backend real del sidecar es el activo
5. Separar claramente:
   - tests de stub
   - tests de sidecar real

Prioridad: `ALTA`

---

## H7. `config.py` sigue comunicando de forma engañosa el modo normal

Ubicacion:

- `config.py:21-26`
- `config.py:34-62`

Problema:

- `EXECUTION_MODE = "bhuman_sidecar"` sigue comentado como si fuera siempre "fuente de verdad"
- hoy eso solo es cierto cuando existe backend real
- si no existe backend real, el sistema cae a `contract_validation_mode`
- entonces `config.py` puede inducir a pensar que ya hay `BHuman in loop` real cuando no es asi

Por que rompe contrato:

- no rompe ejecucion directamente
- pero si rompe trazabilidad conceptual y puede llevar a tomar resultados de entrenamiento como mas fieles de lo que realmente son

Correccion obligatoria:

1. Ajustar la documentacion inline de `EXECUTION_MODE`.
2. Aclarar en `config.py` que:
   - `bhuman_sidecar` = modo que intenta backend real
   - si backend real no existe, el modo efectivo puede ser `contract_validation`
3. Exponer una forma clara de leer el modo efectivo desde runtime o logs.
4. Evitar comentarios del tipo "fuente de verdad" si todavia no existe el bridge C++ real.

Prioridad: `MEDIA`

---

## H8. `reward.py` sigue pudiendo sobreinterpretar takeover como si fuera ejecucion real

Ubicacion:

- `reward.py:139-151`
- `reward.py:291-301`
- `wrapper.py:172-173`
- `train.py:1159`
- `environment.py:1395-1397`

Problema:

- el sistema ya expone `sidecar_backend` y `sidecar_contract_validation_mode`
- pero `reward.py` sigue premiando o castigando `takeover_success`
- mientras el backend siga en `contract_validation_mode`, ese `takeover_success` no equivale a ejecucion real de BHuman
- si no se corrige, el entrenamiento puede seguir asignando credito como si ya existiera `BHuman in loop`

Por que rompe contrato:

- mezcla validacion semantica con ejecucion real
- puede inflar confianza en `takeover_success_rate`
- mantiene un riesgo de sesgo en reward, wrapper y train aunque el sidecar ya sea honesto en su etiqueta

Correccion obligatoria:

1. Hacer que `reward.py` revise el backend efectivo del sidecar.
2. Si `sidecar_backend != "real"` o `sidecar_contract_validation_mode == true`, entonces:
   - no usar `takeover_success` como proxy de ejecucion real
   - o poner ese componente en cero
   - o degradarlo a una señal de validacion estructural, no de exito real
3. Hacer visible en logs y metrics cuando el entrenamiento ocurre sin backend real.
4. Revisar si `wrapper.py` y `train.py` deben etiquetar o separar:
   - `takeover_success_rate_contract_validation`
   - versus `takeover_success_rate_real`

Prioridad: `ALTA`

---

## 5. Lo que SI debe conservarse

No reescribir por gusto estas piezas:

- `bhuman_cfg_parser.py` ya soporta carga real desde CFGs
- `config.py` ya tiene una resolucion interna de `cfg_root`
- `wrapper.py` ya usa `episode_handoff_achieved`
- `EXECUTION_MODE` ya distingue entre modo stub y modo sidecar
- la idea de `GroundTruthFrame` como schema conceptual de entrada sigue siendo util
- la idea de `macro_type` contractual tambien sigue siendo util, pero debe apoyarse en evidencia real
- el `contract_validation_mode` es util y debe conservarse como modo honesto, pero no venderse como `BHuman in loop`

---

## 6. Orden obligatorio de correccion

1. `bhuman_sidecar.py`
   - implementar `RealSidecarBackend`
   - conservar stub Python solo como modo explicitamente no real
   - alinear output schema con BHuman
   - quitar `skill_request = "duel"`
   - quitar `last_kick_type` inventado

2. `environment.py`
   - mantener fuera el takeover simulator del modo normal
   - integrar el backend real cuando exista
   - no seguir mezclando decision sidecar + ejecucion local fake

3. `smoke_test_sidecar.py`
   - cargar CFGs reales de forma obligatoria
   - endurecer escenarios
   - verificar que el modo normal no usa stub ni takeover simulator local
   - distinguir claramente backend real vs `contract_validation`

4. `config.py`
   - si hace falta, extraer helper comun de `cfg_root`
   - mantener `.env` solo como override
   - aclarar el modo efectivo real vs `contract_validation`

5. `reward.py`
   - impedir que `takeover_success` se trate como ejecucion real cuando `sidecar_backend != real`

6. `wrapper.py`
   - tocarlo si hay que diferenciar metricas contractuales vs metricas reales

7. `train.py`
   - tocarlo solo si cambian nombres o disponibilidad de campos
   - no hacer refactors extras aqui
   - si hace falta, etiquetar cuando el entrenamiento corre sin backend real

---

## 7. Tabla ejecutable por archivo

| Archivo | Cambio minimo obligatorio | Criterio de aprobado |
| --- | --- | --- |
| `bhuman_sidecar.py` | Implementar `RealSidecarBackend`; conservar stub Python solo como modo no real; redefinir salida para que `skill_request`, `motion_request`, `activation_graph` y `last_kick_type` respeten BHuman real | El modo normal puede usar backend real; `skill_request` ya no contiene `duel`; `activation_graph` es estructurado; `last_kick_type` no es inventado |
| `environment.py` | Mantener fuera `BHumanTakeoverSimulator` del modo normal e integrar backend real cuando exista | El modo normal no ejecuta takeover local; si no hay backend real, el sistema entra en `contract_validation` o `shadow` explicito, no en fake execution silenciosa |
| `smoke_test_sidecar.py` | Forzar carga real de CFGs y endurecer pruebas de `pass`, `duel`, `last_kick_type`, `stub not used`, `contract_validation` | El test falla si usa defaults; `duel` se valida por macro/substate y no por `skill_request`; el test detecta uso del simulator local; el test distingue backend real vs `contract_validation` |
| `config.py` | Exponer o reutilizar resolvedor comun de `cfg_root` y aclarar semantica del modo efectivo | El sidecar smoke y el runtime encuentran CFGs reales sin depender de `.env`; la config ya no sugiere falsamente que siempre hay BHuman real |
| `reward.py` | Gatear reward de takeover segun backend real vs `contract_validation` | `takeover_success` deja de tratarse como exito real de BHuman cuando el backend no es real |
| `wrapper.py` | Mantener metrica de readiness alineada con `episode_handoff_achieved`; separar metricas contractuales vs reales si hace falta | No se vuelve a medir readiness con `takeover_count`; los dashboards ya no mezclan validacion contractual con ejecucion real |
| `train.py` | Ajuste minimo final, solo si cambian nombres/fields o si hay que etiquetar el backend efectivo | Sigue entrenando sin reintroducir metricas legacy ni falsas equivalencias de takeover |

---

## 8. Criterio global de aprobado

Esta iteracion solo se considera aprobada si se cumplen TODAS:

- `bhuman_sidecar` ya no es un espejo Python silencioso
- el modo normal ya no usa `BHumanTakeoverSimulator`
- el smoke test del sidecar carga CFGs reales y no defaults
- `skill_request` deja de modelar `duel` como si fuera un enum real
- `macro_type = playBall_duel` sale de `ActivationGraph` o subestado equivalente
- `active_agent` ya no depende del heuristico nearest-to-ball en modo normal
- `last_kick_type` solo existe con evidencia real o queda explicitamente no disponible
- el smoke test detecta cualquier recaida a stub/local simulator
- `config.py` ya no comunica `bhuman_sidecar` como si siempre fuera backend real
- `reward.py` ya no premia takeover como ejecucion real cuando el backend sigue en `contract_validation`

Si falta cualquiera de esos puntos, NO llamar a la version `BHuman in loop`.

---

## 9. Ruta minima para lograr `BHuman in loop` real

Claude debe implementar esta ruta minima:

1. backend real del sidecar
   - elegir un mecanismo:
     - IPC
     - FFI
     - pybind11
   - conectar con la cadena real de BHuman

2. entrada contractual al backend
   - partir de `GroundTruthFrame`
   - mapearlo al equivalente de `GroundTruthWorldState`
   - usar `OracledWorldModelProvider` o un adaptador equivalente

3. salida contractual del backend
   - `active_agent`
   - `skill_request`
   - `motion_request`
   - `activation_graph`
   - `last_kick_type`
   - `behavior_status`
   - `strategy_status`

4. integracion del env
   - el modo normal usa solo esas salidas
   - sin takeover simulator local

5. blindaje de entrenamiento
   - `config.py` deja claro el modo efectivo
   - `reward.py` y `wrapper.py` no mezclan validacion contractual con ejecucion real

Solo despues de eso se puede hablar seriamente de `BHuman in loop`.

---

## 10. Instruccion final para Claude

No re-audites todo el proyecto. Ejecuta exactamente esta correccion:

1. preserva parser, contrato numerico y `cfg_root`
2. implementa el bridge real del sidecar o deja el placeholder preparado de forma inequívoca
3. mantiene el stub Python solo como modo explicitamente no real
4. integra el backend real al env sin takeover simulator local
5. corrige `config.py` para que comunique correctamente el modo efectivo
6. corrige `reward.py` para que no premie takeover como si fuera ejecucion real cuando el backend no lo es
7. endurece el smoke test para que pruebe de verdad:
   - CFGs reales
   - schema realista
   - ausencia de stub
   - ausencia de simulator local
   - distincion entre backend real y `contract_validation`
8. toca `wrapper.py` y `train.py` solo al final y solo si el schema corregido lo exige

No optimices nada extra. No reescribas el parser por gusto. No intentes cerrar percepcion o locomocion real aqui. El objetivo de esta iteracion es cerrar la brecha semantica fuerte del sidecar, preparar el backend real y evitar que `config.py`, `reward.py`, `wrapper.py` o `train.py` vendan como real algo que aun esta en `contract_validation`.
