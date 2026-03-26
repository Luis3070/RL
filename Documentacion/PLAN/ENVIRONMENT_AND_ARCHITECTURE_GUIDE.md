# Guia de Uso del Environment y Arquitectura del Proyecto RL + BHuman

> Fecha: 25 de marzo de 2026  
> Estado del documento: vigente para el estado actual del repo  
> Objetivo: explicar de forma operativa y arquitectonica como usar el environment, como se conecta con BHuman, y que papel juega cada archivo principal del proyecto.

---

## 1. Proposito del proyecto

Este proyecto entrena una policy MARL tipo MAPPO para futbol robotico NAO 4v4, con una separacion explicita de responsabilidades:

- RL aprende principalmente:
  - posicionamiento
  - soporte
  - spacing
  - anti-blocking
  - llegada a la zona de handoff
- BHuman decide y produce la semantica de juego:
  - `SkillRequest`
  - `MotionRequest`
  - `ActivationGraph`
  - rol y estado tactico

La meta no es reemplazar BehaviorControl de BHuman, sino acoplar RL a ese ejecutor.

---

## 2. Vision arquitectonica general

La arquitectura actual se entiende mejor como 8 capas:

1. `bhuman_cfg_parser.py`
2. `config.py`
3. `environment.py`
4. `bhuman_sidecar.py`
5. `bridge/bridge_protocol.py`
6. `bridge/bhuman_bridge.cpp`
7. arbol externo de BHuman en `C:/Users/Alegu/Sabana2026`
8. `wrapper.py` + `reward.py` + `train.py`

Flujo general:

```text
CFGs BHuman
  -> bhuman_cfg_parser.py
  -> config.py
  -> NaoSoccerEnv (environment.py)
  -> BHumanDecisionSidecar (bhuman_sidecar.py)
  -> BHumanBridge subprocess (bridge_protocol.py)
  -> RLBridge.exe / bhuman_bridge.cpp
  -> Cognition de BHuman
  -> decisiones reales de BHuman
  -> info/reward/wrapper/train
```

---

## 3. Mapa de archivos principales

### 3.1. Contrato y configuracion

- `bhuman_cfg_parser.py`
  - parser de CFGs `.cfg`
  - produce `AllConfigs`
  - define el contrato Python tipado que consume el resto del stack

- `config.py`
  - carga `AllConfigs`
  - resuelve `cfg_root`
  - fija dimensiones, observaciones, acciones, curriculum y modo de ejecucion

### 3.2. Simulacion y bridge

- `environment.py`
  - define `NaoSoccerEnv`
  - mueve agentes RL
  - llama al sidecar
  - mantiene macroestado de handoff/takeover
  - computa `observations`, `infos`, reward inputs y terminacion

- `bhuman_sidecar.py`
  - adaptador entre el environment y BHuman
  - construye `GroundTruthFrame`
  - llama al backend real o al stub contractual
  - transforma la respuesta del bridge en `SidecarOutput`

- `bridge/bridge_protocol.py`
  - wrapper Python del subprocess `RLBridge.exe`
  - handshake `init`
  - envio de frames
  - `reset`
  - `shutdown`
  - parser del JSON devuelto por el bridge

- `bridge/bhuman_bridge.cpp`
  - app C++ que corre dentro del ecosistema BHuman
  - crea un robot BHuman headless
  - inyecta un frame 2D en Cognition
  - lee representaciones desde Blackboard de Cognition
  - devuelve JSON con decisiones reales

### 3.3. Aprendizaje y evaluacion

- `reward.py`
  - reward centrada en handoff/takeover/support
  - castiga menos la fantasia local y mas la calidad estructural
  - hace gating del takeover reward si el backend no es `real`

- `wrapper.py`
  - env adapter para MAPPO/skrl
  - agrega estado global para critic
  - maneja curriculum y metricas BHuman-aware

- `train.py`
  - loop de entrenamiento MAPPO
  - logging
  - metricas
  - curriculum advancement

### 3.4. Validacion

- `contract_smoke_test.py`
  - valida parser + contrato Python

- `smoke_test_sidecar.py`
  - valida sidecar y su schema

- `smoke_test_environment_bridge.py`
  - valida el loop real `environment -> sidecar -> bridge -> BHuman -> environment`

---

## 4. Fuente de verdad de cada capa

Esta parte es clave para no mezclar ownership:

### 4.1. `bhuman_cfg_parser.py`

Fuente de verdad de:

- campo 4v4 Scaled
- tacticas
- strategy FSM
- walk params
- kick/pass/shoot params
- timing contractual
- skill behavior control

No debe:

- inventar logica de juego
- simplificar `kickInfo`
- cortar patadas por intuicion

### 4.2. `config.py`

Fuente de verdad de:

- resolucion de `cfg_root`
- `EXECUTION_MODE`
- shapes de observacion/accion/critic
- curriculum
- constantes runtime derivadas del contrato

No debe:

- volverse una segunda fuente de verdad paralela al parser

### 4.3. `environment.py`

Fuente de verdad de:

- simulacion base del episodio
- movimiento RL de los agentes
- bookkeeping del episodio
- handoff manager
- reward inputs
- observaciones e `info`

No debe:

- inventar `shoot/pass/dribble/duel` cuando el sidecar esta activo
- reintroducir `BHumanTakeoverSimulator` en modo normal

### 4.4. `bhuman_sidecar.py`

Fuente de verdad de:

- traduccion `env state -> GroundTruthFrame`
- traduccion `bridge result -> SidecarOutput`
- procedencia del backend:
  - `real`
  - `contract_validation`
  - `python_stub`

No debe:

- ocultar fallback silenciosos
- reportar como `real` algo que no venga del bridge real

### 4.5. `bridge/bhuman_bridge.cpp`

Fuente de verdad de:

- integracion real con Cognition BHuman
- estado tactico real
- skill request real
- motion request real
- activation graph real

No debe:

- inventar decisions Python-style
- saltarse el warmup operacional si BHuman necesita secuencia `initial -> ready -> set -> playing`

---

## 5. Modos de ejecucion

El proyecto distingue entre modo solicitado y backend efectivo.

### 5.1. Modo solicitado

En `config.py`:

- `python_stub_smoke_only`
- `bhuman_sidecar`

### 5.2. Backend efectivo

Reportado por el sidecar:

- `real`
- `contract_validation`
- `python_stub`

### 5.3. Interpretacion correcta

#### `python_stub_smoke_only`

Uso:

- smoke tests locales
- depuracion rapida

Comportamiento:

- usa el stub Python
- puede usar `BHumanTakeoverSimulator`

#### `bhuman_sidecar` con backend `real`

Uso:

- modo normal actual del proyecto
- validacion y uso del agente BHuman in loop dentro del env

Comportamiento:

- sidecar llama al bridge real
- `takeover_simulator_active = False`
- el bridge procesa Cognition real

#### `bhuman_sidecar` con backend `contract_validation`

Uso:

- degradacion honesta cuando el bridge real no arranca

Comportamiento:

- usa el stub
- pero marcado explicitamente como no-real

Importante:

- los resultados de takeover no deben tomarse como evidencia real de BHuman

---

## 6. Como se usa el environment

La clase principal es:

- `environment.py` -> `NaoSoccerEnv`

Wrapper recomendado para entrenamiento:

- `wrapper.py` -> `NaoSoccerEnvWrapper`

### 6.1. Inicializacion

Constructor tipico:

```python
from environment import NaoSoccerEnv
from config import CURRICULUM

env = NaoSoccerEnv(stage_cfg=CURRICULUM[0], use_fov=True, seed=123)
```

Wrapper para MAPPO:

```python
from wrapper import NaoSoccerEnvWrapper

env = NaoSoccerEnvWrapper(stage_cfg=CURRICULUM[0], use_fov=True, seed=123)
```

### 6.2. Reset

`env.reset()` hace:

1. reinicia estado del juego
2. respawnea agentes y oponentes
3. respawnea la pelota
4. reinicia handoff/takeover bookkeeping
5. reinicia el sidecar/bridge real si esta activo

Esto es importante:

- el bridge real recibe `reset()`
- el contador de frames del bridge vuelve a 0
- el primer `step()` del nuevo episodio debe devolver `bridge_frame_id = 1`

### 6.3. Step

`env.step(actions)` sigue este flujo:

1. RL mueve agentes segun accion
2. se consulta el sidecar
3. el sidecar decide `active_agent` y macroestado
4. el handoff manager actualiza `handoff_ready` / `takeover_active`
5. oponentes se mueven
6. pelota y colisiones se actualizan
7. se revisan goles
8. se actualiza post-takeover
9. se computan rewards
10. se construyen observaciones e `infos`

### 6.4. Close

`env.close()`:

- apaga el subprocess del sidecar si el backend es real
- evita dejar `RLBridge.exe` colgado

El wrapper tambien delega esto.

---

## 7. Flujo detallado de `env.step()`

### 7.1. Movimiento RL

Cada agente:

- recibe accion continua
- la accion puede pasar por smoothing
- se decodifica a `target_pos`, `target_theta`
- `NaoMovementModel` actualiza pose y velocidad

Esto sigue siendo parte simulada del proyecto.

### 7.2. Consulta al sidecar

El environment construye implicitamente un `GroundTruthFrame` con:

- poses propias
- poses de oponentes
- estado de la pelota
- `is_playing`
- score

Ese frame se manda a:

- `BHumanDecisionSidecar.decide(...)`

### 7.3. Sidecar

Si el backend es `real`:

- llama a `BHumanBridge.step(...)`
- manda JSON al bridge C++

Si el backend no es real:

- usa el stub contractual

### 7.4. Bridge C++

`RLBridge.exe`:

1. recibe el frame
2. bootstrapea la secuencia interna si hace falta:
   - `initial`
   - `ready`
   - `set`
   - `playing`
3. inyecta representaciones 2D en Cognition
4. espera `FrameFinished`
5. lee Blackboard de Cognition
6. serializa decisiones a JSON

### 7.5. Respuesta al env

El env recibe de vuelta, via sidecar:

- `active_agent`
- `skill_request`
- `motion_request`
- `behavior_status`
- `strategy_status`
- `activation_graph`
- metadata del bridge

### 7.6. Handoff / takeover

El macro manager:

- usa `sidecar_macro_type`
- mantiene `handoff_ready`
- mantiene `takeover_active`
- registra eventos de takeover

Importante:

- en modo real ya no se usa `BHumanTakeoverSimulator`

### 7.7. Reward

`reward.py` usa:

- approach
- handoff readiness
- takeover outcome
- support
- anti-blocking

Y hace una distincion muy importante:

- si `sidecar_backend != "real"`, la parte de takeover reward se apaga

---

## 8. Como funciona el sidecar

### 8.1. Entrada

`GroundTruthFrame` contiene:

- `ball_pos`
- `ball_vel`
- `ball_seen_age_ms`
- `own_poses`
- `own_thetas`
- `own_upright`
- `own_roles`
- `opp_poses`
- `opp_upright`
- `is_playing`
- `own_score`
- `opp_score`

### 8.2. Salida

`SidecarOutput` contiene:

- `active_agent`
- `macro_type`
- `strategy_status`
- `current_tactic`
- `skill_request`
- `behavior_status`
- `motion_request`
- `kick_type`
- `kick_length`
- `last_kick_type`
- `activation_graph`
- `backend`
- `contract_validation_mode`
- metadata del bridge

### 8.3. Significado

Esto permite que el env no tenga que deducir por heuristica:

- quien es el activo
- que skill esta corriendo
- que tactic usa BHuman
- que macroestado se activo

---

## 9. Como funciona el bridge Python

Archivo:

- `bridge/bridge_protocol.py`

Responsabilidades:

- resolver el ejecutable de `RLBridge.exe`
- handshake de `init`
- construir `frame_to_json(...)`
- enviar `frame`
- leer respuesta JSON
- hacer `reset`
- hacer `shutdown`

### 9.1. Ruta estable en Windows

La ruta que quedo estable es:

- `oneshot_file_fallback`

Caracteristicas:

- usa `.cmd`
- usa `2>&1`
- evita el camino fragil por pipes como ruta principal

### 9.2. No reabrir estos bugs

No reintroducir:

- `pipe_process` como ruta preferida en Windows
- `stdout` y `stderr` separados en one-shot
- `BHUMAN_ROOT` forzado sin necesidad

---

## 10. Como funciona `RLBridge.exe`

Archivo:

- `bridge/bhuman_bridge.cpp`

Version efectiva compilada en:

- `C:/Users/Alegu/Sabana2026/Build/Windows/RLBridge/Release/RLBridge.exe`

### 10.1. Responsabilidades

- crear robot BHuman headless
- conectar `RLBridgeConsole`
- inyectar frame 2D
- activar `GameStateProvider`
- bootstrapear estado operacional cuando sea necesario
- leer:
  - `GameState`
  - `SkillRequest`
  - `MotionRequest`
  - `MotionInfo`
  - `StrategyStatus`
  - `BehaviorStatus`
  - `ActivationGraph`

### 10.2. Bootstrap interno

Si el env pide `playing`, el bridge hoy puede enviar internamente:

- `initial`
- `ready`
- `set`
- `playing`

Eso fue clave para salir de:

- `beforeHalf`
- `unstiff`

y llegar a:

- `playing`
- `active`

### 10.3. Estado actual

El loop real hoy ya valida:

- `debug_game_state = playing`
- `debug_player_state = active`

---

## 11. Como encaja BHuman real

El bridge corre sobre el arbol externo:

- `C:/Users/Alegu/Sabana2026`

Hoy se apoya en:

- `GameStateProvider`
- `StrategyBehaviorControl`
- `SkillBehaviorControl`
- Blackboard de `Cognition`

Lo que ya esta resuelto:

- transporte
- startup
- reset
- frame injection
- warmup operacional

Lo que sigue siendo siguiente fase:

- enriquecer mas escenarios
- validar mas macroestados especificos
- endurecer casos de pase, dribble, duel y role transitions

---

## 12. Reward y semantica de entrenamiento

### 12.1. Filosofia

La reward ya no debe premiar la fantasia local de la pelota como si fuera BHuman.

La prioridad actual es:

- llegar bien al handoff
- no bloquear
- posicionarse bien
- usar correctamente la evidencia del sidecar

### 12.2. Gating por backend

`reward.py` hace esto:

- si `sidecar_backend == "real"`:
  - takeover reward activa
- si `sidecar_backend != "real"`:
  - takeover reward se apaga

Esto evita contaminar la policy con exito ficticio.

---

## 13. Wrapper y entrenamiento

### 13.1. `wrapper.py`

Agrega:

- estado global para critic
- curriculum manager
- metrics aggregator

Metricas importantes:

- `takeover_ready_rate`
- `takeover_success_rate`
- `bhuman_exec_success_rate`
- `support_quality_rate`
- `anti_blocking_rate`

### 13.2. `train.py`

Se usa para:

- entrenamiento MAPPO
- logging CSV
- plots
- curriculum advancement

El loop de entrenamiento debe usar el wrapper, no el env crudo, salvo pruebas puntuales.

---

## 14. Smokes y validacion recomendada

### 14.1. Contrato

```bash
python contract_smoke_test.py
```

Valida:

- CFGs
- parser
- contrato Python

### 14.2. Sidecar

```bash
python smoke_test_sidecar.py
```

Valida:

- schema del sidecar
- backend real vs stub
- output estructurado

### 14.3. Environment + bridge real

```bash
python smoke_test_environment_bridge.py
```

Es la validacion mas importante del loop actual.

Debe confirmar:

- `sidecar_backend = real`
- `sidecar_contract_validation_mode = False`
- `takeover_simulator_active = False`
- `bridge_frame_id` incremental
- `bridge_debug_frame_sync = ack`
- `debug_game_state = playing`
- `debug_player_state = active`

### 14.4. Bridge manual

Script:

- `bridge/run_rlbridge_frame_only.cmd`

Sirve para validar el binario `RLBridge.exe` sin pasar por el env Python.

---

## 15. Uso practico recomendado

### 15.1. Para validar que todo sigue vivo

1. correr `smoke_test_environment_bridge.py`
2. si falla, correr `bridge/run_rlbridge_frame_only.cmd`
3. separar si el problema esta:
   - en bridge/BHuman
   - o en sidecar/env Python

### 15.2. Para entrenamiento

Usar:

- `NaoSoccerEnvWrapper`
- stages tempranos del curriculum
- backend real cuando sea posible

### 15.3. Para depuracion rapida

Usar:

- `python_stub_smoke_only`

pero recordando que eso no es evidencia real de BHuman.

---

## 16. Lo que hoy si significa “BHuman in loop”

Hoy ya es correcto decir que existe un agente BHuman in loop corriendo dentro del environment, en este sentido:

- el env usa backend `real`
- el sidecar llama al bridge real
- el bridge corre Cognition BHuman
- el env recibe decisiones reales desde Blackboard de BHuman
- el smoke end-to-end pasa con `playing / active`

Lo que no debe confundirse con esto:

- esto no significa que todo el sim2real gap haya desaparecido
- tampoco significa que ya toda la fisica mecanica del mundo sea NAO real

Lo que si significa:

- la semantica central de decision de BHuman ya esta entrando al loop del environment

---

## 17. Riesgos y limites actuales

1. Sigue existiendo simulacion de movimiento/ball physics del lado Python.
2. El entorno sigue siendo 2D y simplificado.
3. La parte operativa ya esta, pero no toda la riqueza semantica de juego esta explotada aun.
4. Hay que evitar reabrir bugs de transporte o de warmup por “limpiezas” apresuradas.

---

## 18. Estado final resumido

### Ya resuelto

- parser contractual
- cfg root estable
- sidecar con backend real
- bridge real estable
- bootstrap operacional del bridge
- env smoke pasando
- `playing / active` operativo

### Siguiente fase logica

- enriquecer escenarios de validacion
- documentar rutas de entrenamiento recomendadas por stage
- auditar que señales usar para fine-tuning o evaluacion real

---

## 19. Referencias internas recomendadas

- `Documentacion/PLAN/BHUMAN_IMPLEMENTATION_CORRECTION_GUIDE.md`
- `Documentacion/PLAN/RLBRIDGE_INTERVENTION_LOG.md`
- `Documentacion/PLAN/BHUMAN_IN_LOOP_SIDECAR_GUIDE.md`
- `Documentacion/PLAN/BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md`
- `Documentacion/PLAN/BHUMANPARSER_GUIDE.md`

