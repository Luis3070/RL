# Ground Truth: BHuman Integration & Technical Core

Este documento consolida el cimiento técnico de la integración RL + BHuman, unificando la arquitectura, los esquemas de datos y la guía del sidecar.

---

# SOURCE: ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md

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

---

# SOURCE: BHUMAN_BC_SCHEMA_v1.md

# BHUMAN_BC_SCHEMA_v1 — Esquema fiel para dataset de Behavioral Cloning desde `bhuman_bridge.cpp`

> **Propósito**
>
> Definir un esquema de captura de datos para Behavioral Cloning que sea:
>
> - lo más fiel posible al funcionamiento real de BHuman
> - seguro para integrar en un proyecto BHuman limpio y ya probado en NAOs
> - implementable como **expansión de** `Src/Apps/RLBridge/bhuman_bridge.cpp`
> - no intrusivo sobre los módulos originales de decisión (`Behavior.cpp`, `PlayBall.cpp`, `StrategyBehaviorControl.cpp`, etc.)

---

## 1. Principio rector

Este esquema NO debe construirse:

- alterando la lógica central de BHuman
- modificando cómo BHuman decide `SkillRequest`
- insertando logging dentro de `StrategyBehaviorControl`
- insertando logging dentro de `PlayBall`
- insertando logging dentro de `PassToTeammate`
- creando atajos que mezclen percepción, decisión y outcome sin marcar fuentes

Este esquema SÍ debe construirse:

- como una **extensión del app artificial** `bhuman_bridge.cpp`
- como un observador pasivo de blackboard y snapshots
- como un serializador fiel de representaciones reales de BHuman
- como un pipeline doble:
  - `raw frame dataset`
  - `normalized event dataset`

---

## 2. Filosofía de diseño

Si el objetivo es hacer Behavioral Cloning serio, el dataset debe conservar tres cosas:

1. **Qué sabía BHuman**
2. **Qué decidió BHuman**
3. **Qué pasó después**

Por eso el diseño recomendado no es un único JSON "bonito", sino dos datasets coordinados:

1. `raw_bhuman_frame.jsonl`
- espejo fiel del frame de cognición y de las representaciones relevantes

2. `normalized_bhuman_event.jsonl`
- resumen por eventos semánticos derivados del raw

Amos deben ir unidos por claves estables.

---

## 3. Por qué construirlo sobre `bhuman_bridge.cpp`

En un BHuman limpio y funcional, `bhuman_bridge.cpp` ya es la zona artificial creada para:

- observar decisiones
- serializar representaciones relevantes
- transportar información hacia el entorno RL

Eso lo convierte en el mejor lugar para incorporar el recorder del dataset BC, porque:

- ya es una frontera de integración
- ya trabaja con snapshots de cognición/motion
- ya lee representaciones como:
  - `SkillRequest`
  - `MotionRequest`
  - `MotionInfo`
  - `StrategyStatus`
  - `BehaviorStatus`
  - `ActivationGraph`
- evita contaminar el corazón del comportamiento original

### Decisión arquitectónica

El dataset recorder debe vivir:

- dentro de `bhuman_bridge.cpp`

o, si se quiere mejor separación interna:

- dentro de clases auxiliares nuevas declaradas y usadas por `bhuman_bridge.cpp`

Pero NO debe vivir:

- dentro de `Behavior.cpp`
- dentro de `StrategyBehaviorControl.cpp`
- dentro de `PlayBall.cpp`
- dentro de `PassToTeammate.cpp`

---

## 4. Objetivo funcional del recorder

La expansión de `bhuman_bridge.cpp` debe permitir:

1. capturar el estado real del frame de cognición
2. capturar la decisión real emitida por BHuman
3. capturar outcome posterior observable
4. generar un dataset raw fiel
5. generar un dataset de eventos ligado al raw

Sin:

- modificar protocolos internos de BHuman
- alterar providers del blackboard
- cambiar ownership de `SkillRequest`
- introducir nuevas dependencias en la cadena de decisión

---

## 5. Restricciones de seguridad e integración

La implementación debe respetar estas reglas:

### 5.1 Regla de no interferencia

El recorder:

- no provee representaciones nuevas
- no reemplaza providers
- no modifica `threads.cfg`
- no cambia la frecuencia de decisión del robot
- no modifica `SkillRequest`
- no modifica `MotionRequest`

### 5.2 Regla de lectura pasiva

El recorder solo puede:

- leer `DecisionSnapshot`
- leer blackboard en modo read-only
- usar counters de frame ya disponibles

### 5.3 Regla de serialización desacoplada

La escritura del dataset no debe:

- bloquear el ciclo de cognición
- depender de parsing pesado por frame
- convertir el bridge en un cuello de botella

Por eso la escritura debe ser:

- incremental
- append-only
- con buffers simples
- idealmente en formato `jsonl`

---

## 6. Arquitectura recomendada dentro de `bhuman_bridge.cpp`

Se recomienda estructurarlo así:

1. `BridgeTransport`
- responsabilidad existente del bridge
- comunicación y serialización operacional

2. `BHumanRawDatasetRecorder`
- construye y escribe `raw_bhuman_frame.jsonl`

3. `BHumanEventDatasetRecorder`
- detecta cambios semánticos
- construye y escribe `normalized_bhuman_event.jsonl`

4. `DatasetRecorderConfig`
- flags de activación
- rutas de salida
- frecuencia de captura
- modo raw/event/both

### Flujo recomendado

1. termina un frame de cognición
2. `bhuman_bridge.cpp` ya obtiene snapshot y/o acceso a blackboard
3. `BHumanRawDatasetRecorder` construye el registro raw
4. `BHumanEventDatasetRecorder` compara con estado previo
5. si hay transición relevante, emite evento
6. ambos escriben `jsonl`

---

## 7. Estructura del dataset

Se recomiendan exactamente dos archivos:

### 7.1 Raw

`raw_bhuman_frame.jsonl`

Una línea por frame de cognición.

### 7.2 Event

`normalized_bhuman_event.jsonl`

Una línea por evento semántico.

### 7.3 Regla de enlace obligatorio

El dataset de eventos debe ir siempre enlazado al raw mediante:

- `session_id`
- `robot_name`
- `player_number`
- `frame_number`
- `time_ms`
- `raw_frame_refs`
- `event_id`

El dataset de eventos nunca es autosuficiente.

---

## 8. Schema general del raw dataset

Cada línea de `raw_bhuman_frame.jsonl` debe tener esta estructura lógica:

```json
{
  "schema_version": "bhuman_bc_v1",
  "session_id": "2026-04-09_4v4_run_003",
  "robot_name": "Nao3",
  "player_number": 3,
  "frame_number": 182344,
  "time_ms": 6017352,
  "cycle_ms": 33,
  "scenario": "4v4_Scaled",
  "thread": "Cognition",
  "frame_context": {},
  "bhuman_inputs": {},
  "bhuman_outputs": {},
  "post_outcome": {}
}
```

---

## 9. Sección `frame_context`

Esta sección describe el contexto administrativo y temporal del frame.

### Campos obligatorios

- `schema_version`
- `session_id`
- `robot_name`
- `player_number`
- `frame_number`
- `time_ms`
- `cycle_ms`
- `scenario`
- `thread`

### Campos opcionales

- `match_id`
- `half`
- `team_color`
- `recording_mode`

### Ejemplo

```json
"frame_context": {
  "scenario": "4v4_Scaled",
  "thread": "Cognition",
  "recording_mode": "raw+event"
}
```

---

## 10. Sección `bhuman_inputs`

Esta sección debe reflejar lo mejor posible el estado cognitivo de BHuman
sin reinterpretarlo desde Python.

### 10.1 Reglas

- guardar la fuente real de cada grupo de campos
- no colapsar representaciones distintas
- no usar padding artificial de teammates
- no fabricar slots canónicos si no existen así en BHuman

### 10.2 Subbloques recomendados

1. `game_state`
2. `robot_pose`
3. `field_ball`
4. `ball_model`
5. `teammates_ball_model`
6. `team_data`
7. `global_opponents_model` o equivalente observado
8. `motion_info_before`
9. `strategy_status_before`
10. `behavior_status_before`
11. `activation_graph_before` si se quiere snapshot pre-output

---

## 11. Schema detallado de `game_state`

```json
"game_state": {
  "source": "GameState",
  "state": "playing",
  "player_state": "active",
  "game_controller_active": true
}
```

### Campos obligatorios

- `source`
- `state`
- `player_state`
- `game_controller_active`

---

## 12. Schema detallado de `robot_pose`

```json
"robot_pose": {
  "source": "RobotPose",
  "x_mm": 1200.0,
  "y_mm": -350.0,
  "theta_rad": 0.42
}
```

### Campos obligatorios

- `source`
- `x_mm`
- `y_mm`
- `theta_rad`

---

## 13. Schema detallado de balón

Se recomienda separar:

- `field_ball`
- `ball_model`
- `teammates_ball_model`

Porque BHuman no siempre decide con una sola noción de balón.

### 13.1 `field_ball`

```json
"field_ball": {
  "source": "FieldBall",
  "x_mm": 1620.0,
  "y_mm": -120.0,
  "vx_mm_s": 0.0,
  "vy_mm_s": 0.0
}
```

### 13.2 `ball_model`

```json
"ball_model": {
  "source": "BallModel",
  "x_rel_mm": 430.0,
  "y_rel_mm": 35.0
}
```

### 13.3 `teammates_ball_model`

```json
"teammates_ball_model": {
  "source": "TeammatesBallModel",
  "is_valid": true
}
```

---

## 14. Cómo guardar teammates de forma fiel a BHuman

Este es uno de los puntos más importantes.

### Regla principal

Los teammates NO deben guardarse como:

- slots artificiales fijos inventados por el bridge Python
- padding canónico
- home poses sintéticas

Los teammates deben guardarse como:

- los compañeros reales que BHuman tiene disponibles en ese frame
- identificados por `player_number`
- con la fuente real explícita
- con freshness/edad si existe

### Estructura recomendada

```json
"team_data": {
  "source": "TeamData",
  "teammates": [
    {
      "player_number": 1,
      "is_goalkeeper": false,
      "pose_x_mm": 300.0,
      "pose_y_mm": -900.0,
      "pose_theta_rad": 0.10,
      "status": "active",
      "time_since_last_packet_ms": 66
    },
    {
      "player_number": 2,
      "is_goalkeeper": false,
      "pose_x_mm": 900.0,
      "pose_y_mm": 250.0,
      "pose_theta_rad": -0.20,
      "status": "active",
      "time_since_last_packet_ms": 33
    }
  ]
}
```

### Campos recomendados por teammate

- `player_number`
- `is_goalkeeper`
- `pose_x_mm`
- `pose_y_mm`
- `pose_theta_rad`
- `status`
- `time_since_last_packet_ms`

### Campos opcionales

- `ball_x_mm`
- `ball_y_mm`
- `is_upright`
- `penalized`

### Si además se quiere capturar la visión táctica interna

Se puede añadir un bloque separado:

```json
"strategy_agents": {
  "source": "Behavior/Agent list",
  "agents": [
    {
      "player_number": 1,
      "role": "defenderL",
      "position": "defenderL",
      "is_active_candidate": false,
      "current_position_x_mm": 320.0,
      "current_position_y_mm": -880.0
    }
  ]
}
```

Pero esto debe ir separado de `team_data`.

---

## 15. Cómo guardar opponents

Los opponents deben guardarse con la representación real utilizada por el sistema.

```json
"global_opponents_model": {
  "source": "GlobalOpponentsModel",
  "players": [
    {
      "x_mm": 2100.0,
      "y_mm": 180.0
    }
  ]
}
```

Si hay campos extra disponibles y confiables:

- velocidad
- edad
- confianza

pueden añadirse, siempre marcando fuente.

---

## 16. `motion_info_before`

Esto captura el estado de ejecución inmediatamente relevante para la cognición.

```json
"motion_info_before": {
  "source": "MotionInfo",
  "executed_phase": "walk",
  "last_kick_type": "none",
  "last_kick_timestamp_ms": 0
}
```

### Campos recomendados

- `executed_phase`
- `last_kick_type`
- `last_kick_timestamp_ms`

---

## 17. `strategy_status_before`

```json
"strategy_status_before": {
  "source": "StrategyStatus",
  "accepted_tactic": "t121_4v4_scaled",
  "proposed_tactic": "t121_4v4_scaled",
  "role": "forwardL",
  "position": "forwardL",
  "proposed_set_play": "none",
  "accepted_set_play": "none",
  "set_play_step": -1,
  "accepted_mirror": false,
  "proposed_mirror": false
}
```

---

## 18. `behavior_status_before`

```json
"behavior_status_before": {
  "source": "BehaviorStatus",
  "speed_mm_s": 0.0,
  "pass_target": -1,
  "pass_origin": -1,
  "walking_to_x_mm": 0.0,
  "walking_to_y_mm": 0.0
}
```

---

## 19. Sección `bhuman_outputs`

Esta sección debe contener exactamente lo que BHuman decidió o publicó después de decidir.

### Subbloques recomendados

1. `skill_request`
2. `motion_request`
3. `strategy_status_after`
4. `behavior_status_after`
5. `activation_graph`

---

## 20. Schema de `skill_request`

```json
"skill_request": {
  "source": "SkillRequest",
  "skill": "pass",
  "pass_target": 2,
  "target_x_mm": 0.0,
  "target_y_mm": 0.0,
  "target_theta_rad": 0.0
}
```

### Campos obligatorios

- `source`
- `skill`

### Campos opcionales según skill

- `pass_target`
- `target_x_mm`
- `target_y_mm`
- `target_theta_rad`

### Regla importante

Para etiquetar `pass`, la fuente principal debe ser:

- `SkillRequest::pass`

No:

- una inferencia externa basada en movimiento

---

## 21. Schema de `motion_request`

```json
"motion_request": {
  "source": "MotionRequest",
  "motion": "walkToBallAndKick",
  "walk_speed_x": 0.0,
  "walk_speed_y": 0.0,
  "walk_speed_theta": 0.0,
  "kick_type": "forwardFastLeft",
  "kick_length_mm": 2500.0,
  "walk_target_x_mm": 1550.0,
  "walk_target_y_mm": -50.0,
  "walk_target_theta_rad": 0.12,
  "target_direction_rad": 0.08,
  "align_precisely": "notPrecise",
  "pre_step_allowed": true,
  "turn_kick_allowed": true
}
```

Esto es muy valioso para BC avanzado porque preserva:

- intención de ejecución
- kick type
- target físico
- velocidad de aproximación

---

## 22. `strategy_status_after`

```json
"strategy_status_after": {
  "source": "StrategyStatus",
  "accepted_tactic": "t121_4v4_scaled",
  "proposed_tactic": "t121_4v4_scaled",
  "role": "forwardL",
  "position": "forwardL"
}
```

Sirve para capturar:

- táctica aceptada
- rol
- posición táctica nominal

---

## 23. `behavior_status_after`

```json
"behavior_status_after": {
  "source": "BehaviorStatus",
  "state": "passing",
  "speed_mm_s": 180.0,
  "pass_target": 2,
  "pass_origin": 3,
  "walking_to_x_mm": 1500.0,
  "walking_to_y_mm": -80.0,
  "shooting_to_x_mm": null,
  "shooting_to_y_mm": null
}
```

---

## 24. `activation_graph`

```json
"activation_graph": {
  "source": "ActivationGraph",
  "nodes": [
    {
      "option": "Strategy",
      "depth": 0,
      "state": "playing",
      "option_time_ms": 1400,
      "state_time_ms": 240
    },
    {
      "option": "PlayBall",
      "depth": 1,
      "state": "pass",
      "option_time_ms": 350,
      "state_time_ms": 120
    },
    {
      "option": "PassToTeammate",
      "depth": 2,
      "state": "execute",
      "option_time_ms": 120,
      "state_time_ms": 85
    }
  ]
}
```

Esto es especialmente importante porque:

- da semántica temporal
- ayuda a construir eventos
- permite depurar decisiones sin invadir módulos internos

---

## 25. Sección `post_outcome`

Esta sección no debe inventar éxito semántico que BHuman no emitió.
Debe limitarse a outcomes observables y claramente marcados.

### Ejemplo

```json
"post_outcome": {
  "source": "bridge_observed",
  "kick_observed": false,
  "last_kick_timestamp_ms": 0,
  "last_kick_type": "none",
  "ball_progress_mm_100ms": 0.0,
  "ball_progress_mm_330ms": 420.0,
  "ball_progress_along_target_mm_330ms": 390.0,
  "team_possession_after_330ms": true,
  "goal_scored_within_2s": false
}
```

### Regla

Esto no reemplaza la decisión. Solo describe qué pasó después.

---

## 26. Schema general del event dataset

Cada línea de `normalized_bhuman_event.jsonl` debe resumir un cambio semántico.

### Ejemplo base

```json
{
  "schema_version": "bhuman_bc_event_v1",
  "event_id": "sess003_p3_f182344_pass_start",
  "session_id": "2026-04-09_4v4_run_003",
  "robot_name": "Nao3",
  "player_number": 3,
  "event_type": "skill_request_change",
  "event_name": "pass_start",
  "start_frame": 182344,
  "end_frame": 182361,
  "start_time_ms": 6017352,
  "end_time_ms": 6017913,
  "decision_frame_ref": 182344,
  "raw_frame_refs": [182344, 182345, 182346],
  "decision": {},
  "outcome_summary": {}
}
```

---

## 27. Eventos que vale la pena emitir

### Eventos mínimos

- `skill_request_change`
- `motion_request_change`
- `pass_target_change`
- `role_change`
- `tactic_change`
- `kick_observed`
- `macro_window_start`
- `macro_window_end`

### Regla

Los eventos deben derivarse del raw.
Nunca al revés.

---

## 28. Frecuencia de captura

### Recomendación principal

Capturar raw por frame de cognición:

- aproximadamente `33 ms`

---

# SOURCE: BHUMAN_IN_LOOP_SIDECAR_GUIDE.md

# Guia de Implementacion Sidecar BHuman In Loop

> Fecha: 24 de marzo de 2026
> Objetivo: definir una ruta concreta para integrar BHuman in loop mediante un sidecar de decision, respetando el contrato documental del proyecto RL y evitando una reimplementacion local de las decisiones `shoot/pass/dribble/duel`.
> Scope: integracion offline/experimental para entrenamiento, shadow mode y smoke tests de integracion. No es una guia para despliegue final de competicion.

Documento relacionado:

- [BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md)
  Define la secuencia recomendada: consolidar primero la opcion 1 entrenable y abrir la opcion 2 despues como modo alterno cuando exista mejor CPU.

---

## 1. Veredicto Ejecutivo

Si el sidecar se implementa bien, SI es suficientemente robusto para el scope actual del proyecto RL.

Eso significa:

- SI sirve para cerrar la brecha semantica de decision
- SI sirve para `BHuman in loop` en entrenamiento experimental, shadow mode y fine-tuning
- SI sirve para que el equipo RL aprenda a convivir con respuestas reales de BHuman
- NO sirve para cerrar por si solo percepcion real, locomocion real, latencia real ni fisica mecanica real
- NO debe venderse como reemplazo completo del runtime fisico de NAO

La meta de esta iteracion NO es "todo BHuman". La meta correcta es:

- meter la cadena real de decision de BHuman dentro del loop Python
- exportar las decisiones y macroestados reales de BHuman
- dejar de inventar `macro_type`, `shoot/pass/dribble` y striker activo desde `environment.py`

---

## 2. Criterio de Robustez del Sidecar

La siguiente tabla define cuando el sidecar puede considerarse lo suficientemente robusto y cuando ya no alcanza y conviene escalar a una opcion headless embebida.

| Criterio | Minimo para considerar el sidecar robusto | Senal de falla que empuja a headless |
| --- | --- | --- |
| Autoridad de decision | El sidecar corre `StrategyBehaviorControl -> SkillRequest -> SkillBehaviorControl -> MotionRequest` | Python sigue fijando `macro_type` o `shoot/pass/dribble` |
| Agente activo | El sidecar decide `active_agent` usando la logica real de `Behavior::determineActiveAgent()` | Python sigue usando heuristicas propias para striker/active player |
| Macroestado | `macro_type` se deriva de `SkillRequest` + `ActivationGraph` | `duel` sigue siendo inferido localmente o no existe |
| Fuente de verdad | El env consume salidas del sidecar como verdad operacional | El env sigue usando `BHumanTakeoverSimulator` en entrenamiento normal |
| Snapshot de entrada | Python manda snapshot de mundo por frame consistente con el contrato | Se mandan solo 2-3 flags y el sidecar "adivina" el resto |
| Team/ball semantics | Existen `FieldBall`, `TeammatesBallModel`, roles y team context suficientes | El sidecar no puede reproducir cambios de `pass` vs `shoot` porque falta contexto |
| Salida de motion | Se exporta `MotionRequest.motion`, `kickType`, `kickLength`, `BehaviorStatus`, `MotionInfo` relevante | Solo se exporta `skill_request` y se pierde la ejecucion concreta |
| Temporalidad | 1 `env.step()` corresponde a 1 decision frame del sidecar | El sidecar responde a otra granularidad y rompe el contrato temporal |
| Smoke tests | Hay smoke tests de integracion reales con sidecar | Solo existen tests unitarios del parser o mocks Python |
| Uso experimental | El sidecar funciona en shadow/assist mode | La unica forma de correrlo es como reemplazo total del runtime |

Regla practica:

- Si se cumplen las 10 condiciones, el sidecar es suficientemente robusto para el scope actual del RL.
- Si fallan `Autoridad de decision`, `Agente activo` o `Macroestado`, el sidecar NO es suficiente.
- Si el principal problema restante es fisica/percepcion/latencia, el sidecar sigue siendo valido; eso ya es otra capa del sim2real gap.

---

## 3. Delimitacion de Scope

Esta guia SI cubre:

- integracion semantica real de decisiones BHuman
- agente activo real
- `shoot/pass/dribble/duel` reales
- `SkillRequest`, `BehaviorStatus`, `MotionRequest`, `MotionInfo`, `ActivationGraph`
- smoke tests de integracion
- uso experimental en training/shadow mode

Esta guia NO cubre:

- percepcion visual real
- motores reales NAO
- sincronizacion completa con hardware
- despliegue final de competicion
- reemplazo total del ecosistema BHuman por Python

---

## 4. Cadena Real que el Sidecar Debe Envolver

La integracion debe respetar la cadena real de BHuman:

1. `StrategyBehaviorControl`
2. `Behavior::determineActiveAgent()`
3. `ActiveRoles/PlayBall`
4. `SkillRequest`
5. `SkillBehaviorControl`
6. `Skills/PlayBall`
7. `MotionRequest`
8. `MotionInfo`

Referencias concretas:

- `C:/Users/Alegu/Sabana2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/StrategyBehaviorControl.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Representations/BehaviorControl/SkillRequest.h`
- `C:/Users/Alegu/Sabana2026/Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/PlayBall.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Representations/MotionControl/MotionRequest.h`
- `C:/Users/Alegu/Sabana2026/Src/Representations/MotionControl/MotionInfo.h`
- `C:/Users/Alegu/Sabana2026/Src/Representations/BehaviorControl/ActivationGraph.h`

Importante:

- `duel` NO existe como `SkillRequest::duel`
- `duel` aparece como subestado interno de la skill `PlayBall`
- por eso el bridge no puede inferirse solo desde `SkillRequest`
- debe usar `ActivationGraph` o una exportacion equivalente del subestado activo

---

## 5. Hallazgo Clave para el Diseno

El proyecto BHuman ya contiene dos piezas extremadamente utiles:

1. `OracledWorldModelProvider`
2. `OracledWorldModelProvider`

Referencias:

- `C:/Users/Alegu/Sabana2026/Src/Representations/Infrastructure/GroundTruthWorldState.h`
- `C:/Users/Alegu/Sabana2026/Src/Modules/Modeling/OracledWorldModelProvider/OracledWorldModelProvider.h`
- `C:/Users/Alegu/Sabana2026/Src/Modules/Modeling/OracledWorldModelProvider/OracledWorldModelProvider.cpp`

Esto permite una estrategia mucho mejor que "Python reconstruye todo":

- Python manda un snapshot de mundo global por frame
- BHuman deriva desde ahi `RobotPose`, `BallModel`, `TeammatesBallModel`, `ObstacleModel`, `GlobalOpponentsModel`
- luego corre la decision real

Conclusion:

- NO conviene reimplementar localmente `BallModel`, `FieldBall`, `ObstacleModel` o `team ball` en el bridge Python
- SI conviene usar `GroundTruthWorldState` como molde del `input schema`

---

## 6. Arquitectura Recomendada

## 6.1 Opcion A: Decision Sidecar

Arquitectura recomendada para esta fase.

Flujo:

`Python env.step() -> snapshot de mundo -> sidecar C++ -> BHuman decision chain -> salida bridge -> Python actualiza estado/metricas`

Responsabilidades:

- Python:
  - mantiene el mundo del entrenamiento
  - ejecuta RL
  - envia snapshot por frame
  - consume decision real de BHuman
- Sidecar:
  - reconstruye representaciones BHuman minimas
  - decide agente activo
  - produce `SkillRequest`, `MotionRequest`, `BehaviorStatus`, `ActivationGraph`
  - devuelve macroestado fiel
- Env:
  - deja de inventar la decision de takeover
  - usa el sidecar como fuente de verdad en modo normal

## 6.2 Opcion B: Headless Embebido

Solo como backup.

Usar si:

- el sidecar no logra reproducir bien `active_agent`
- siguen apareciendo disonancias semanticas fuertes
- el wrapping de representaciones resulta demasiado fragmentado
- el proyecto necesita correr mas componentes del stack BHuman completos

Base tecnica potencial:

- `SimulatedRobot2D`
- `LocalConsole`
- `GroundTruthWorldState`
- `GameController`

Referencias:

- `C:/Users/Alegu/Sabana2026/Src/Libs/SimulatedNao/SimulatedRobot2D.h`
- `C:/Users/Alegu/Sabana2026/Src/Libs/SimulatedNao/SimulatedRobot2D.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Libs/SimulatedNao/LocalConsole.h`

Regla:

- NO aplicar ahora
- dejarlo documentado como plan B

---

## 7. Input Schema Propuesto Python -> Sidecar

El input del sidecar debe modelarse por frame.

Minimo obligatorio:

- `frame_index`
- `frame_time_ms`
- `is_playing`
- `own_player_number`
- `own_team_color`
- `own_pose`
- `own_upright`
- `ball_position_on_field`
- `ball_velocity_on_field`
- `own_team_players`
- `opponent_team_players`

Forma sugerida:

- usar un schema equivalente a `GroundTruthWorldState`
- agregar metadata minima que no esta ahi

Estructura minima sugerida:

```text
SidecarFrameInput
- frame_index
- frame_time_ms
- is_playing
- own_player_number
- own_pose
- own_upright
- own_team_players[number, pose, upright, active]
- opponent_team_players[number, pose, upright]
- balls[position, velocity]
- optional_context
  - previous_strategy_status
  - previous_motion_info
  - set_play_flags si fueran necesarios
```

Notas:

- Para la policy RL sigue bastando `is_playing`
- Si el sidecar necesita mas `GameState` interno para BHuman, puede recibirlo internamente sin exponerlo a la observacion RL
- No meter toda la FSM arbitral en la policy solo por "completitud"

---

## 8. Output Schema Propuesto Sidecar -> Python

Salida minima obligatoria:

- `active_agent`
- `strategy_role`
- `strategy_position`
- `skill_request.skill`
- `skill_request.pass_target`
- `behavior_status.pass_target`
- `behavior_status.shooting_to`
- `behavior_status.walking_to`
- `motion_request.motion`
- `motion_request.kick_type`
- `motion_request.kick_length`
- `motion_info.executed_phase`
- `motion_info.last_kick_type`
- `motion_info.last_kick_timestamp`
- `activation_graph`

Estructura sugerida:

```text
SidecarFrameOutput
- active_agent
- strategy_status
  - role
  - position
  - accepted_tactic
- skill_request
  - skill
  - target
  - pass_target
- behavior_status
  - pass_target
  - pass_origin
  - walking_to
  - speed
  - shooting_to
- motion_request
  - motion
  - kick_type
  - kick_length
  - target_direction
- motion_info
  - executed_phase
  - last_kick_type
  - last_kick_timestamp
- activation_graph
- diagnostics
  - used_team_ball
  - sidecar_latency_ms
```

Derivaciones que puede hacer Python a partir de esta salida:

- `macro_type`
- `takeover_active`
- `macro_success`
- `macro_end_reason`

Pero esas derivaciones ya no deben inventar la macroaccion, solo interpretarla.

---

## 9. Reglas de Macroestado

Reglas recomendadas:

- `playBall_pass` si `SkillRequest.skill == pass`
- `playBall_dribble` si `SkillRequest.skill == dribble`
- `playBall_shoot` si `SkillRequest.skill == shoot`
- `playBall_duel` si `ActivationGraph` indica estado `zweikampf`

No hacer:

- no fijar `macro_type = playBall_shoot` por defecto
- no inferir `duel` por distancia a obstaculos solo desde Python
- no convertir `motion_request.walkToBallAndKick` automaticamente en `shoot`

---

## 10. Integracion con el Proyecto RL

Cambios esperados por archivo del lado Python:

## 10.1 `environment.py`

Debe:

- dejar de ser la fuente de verdad de `shoot/pass/dribble/duel`
- usar sidecar en modo normal
- mantener stub local solo para smoke tests o modo explicitamente marcado
- consumir `active_agent` real del sidecar
- derivar `macro_type` desde sidecar, no desde heuristicas

No debe:

- ejecutar `BHumanTakeoverSimulator` en entrenamiento normal
- fijar siempre `playBall_shoot`
- usar fisica local como verdad de la decision macro

## 10.2 `wrapper.py`

Debe:

- medir `takeover_ready_rate` con handoff real alcanzado
- distinguir `handoff_ready` de `takeover_count`
- registrar metricas de integracion sidecar si aplica

## 10.3 `train.py`

Debe:

- permanecer casi intacto
- solo adaptarse a metricas corregidas
- no absorber logica del sidecar

## 10.4 `config.py`

Debe:

- seleccionar claramente el modo de ejecucion:
  - `python_stub_smoke_only`
  - `bhuman_sidecar`
- resolver internamente la ruta CFG repo-relative
- dejar `.env` como override, no como dependencia obligatoria

---

## 11. Smoke Tests de Integracion que Claude Debe Implementar

Claude debe crear smoke tests de integracion, no solo tests unitarios.

Nombre sugerido:

- `bhuman_sidecar_smoke_test.py`

Objetivo:

- demostrar que el sidecar corre la decision real de BHuman dentro del loop contractual

### 11.1 Test de arranque del sidecar

Debe verificar:

- el sidecar compila o inicia
- carga CFGs reales
- expone version/config activa
- responde a un frame vacio/minimo sin crash

### 11.2 Test de snapshot minimo

Dado un `GroundTruthWorldState` simple:

- 1 balon
- 4 propios
- 4 rivales
- `is_playing = true`

Debe verificar:

- se obtiene `active_agent`
- se obtiene `SkillRequest`
- se obtiene `MotionRequest`
- se obtiene `ActivationGraph`

### 11.3 Test de agente activo real

Escenario:

- dos propios cerca del balon con tiempos distintos plausibles

Debe verificar:

- el `active_agent` viene del sidecar
- Python no lo sobrescribe

### 11.4 Test de `shoot` vs `pass`

Escenario A:

- striker con buena apertura a porteria

Escenario B:

- porteria mal evaluada, companero bien posicionado

Debe verificar:

- A produce `shoot`
- B produce `pass`

### 11.5 Test de `duel`

Escenario:

- obstaculo cercano al balon en envelope compatible

Debe verificar:

- el macroestado se detecta como `playBall_duel`
- la fuente de verdad viene de `ActivationGraph` o export equivalente

### 11.6 Test de `dribble`

Escenario:

- situacion donde `KickAtGoal` cae a dribble o `SkillRequest` es `dribble`

Debe verificar:

- el macroestado no se deduce por heuristica Python
- el sidecar lo reporta consistentemente

### 11.7 Test de gating operativo

Debe verificar:

- si `is_playing = false`, Python no activa takeover normal
- si `is_playing = true`, el sidecar puede decidir normalmente

Nota:

- este gating es operativo
- no implica meter toda la FSM arbitral en la observacion RL

### 11.8 Test de salida de motion

Debe verificar:

- existe `motion_request.motion`
- existe `kick_type`
- existe `kick_length`
- existe `motion_info.last_kick_type`

### 11.9 Test de no regresion contractual

Debe verificar:

- parser y contract loader siguen pasando
- `kickInfo` sigue completo
- no se recorta el set de kicks
- `environment.py` ya no usa el stub local en modo normal

### 11.10 Test de budget temporal

Debe medir:

- latencia sidecar por frame
- latencia total bridge

Debe fallar o warning si:

- el sidecar supera el presupuesto contractual definido para Cognition segun la documentacion del RL

---

## 12. Orden Obligatorio de Implementacion para Claude

1. No tocar primero `train.py`
2. Crear interface clara del sidecar
3. Implementar snapshot de entrada por frame
4. Implementar salida real `SkillRequest/BehaviorStatus/MotionRequest/ActivationGraph`
5. Conectar `environment.py` al sidecar en modo normal
6. Dejar stub solo para smoke mode
7. Corregir `wrapper.py`
8. Agregar smoke tests de integracion
9. Validar budget temporal
10. Solo despues considerar ajustes menores de `train.py`

---

## 13. Restricciones Importantes para Claude

- NO reescribir parser ni simplificar `kickInfo`
- NO recortar las 23 patadas por intuicion
- NO usar `.cpp/.h` como parser numerico principal
- SI usar `.cpp/.h` como fuente semantica obligatoria
- NO mantener `macro_type` local inventado
- NO mantener `BHumanTakeoverSimulator` como fuente de verdad del entrenamiento normal
- NO meter toda la FSM arbitral en la observacion RL solo por exceso de fidelidad
- SI mantener `is_playing` como gating operativo para la policy

---

## 14. Criterio de Aprobado

Esta iteracion se considera aprobada solo si:

- el sidecar decide `active_agent`
- el sidecar produce `SkillRequest`
- el sidecar produce `MotionRequest`
- el sidecar permite derivar `macro_type` real
- `duel` deja de ser una inferencia Python local
- `environment.py` deja de inventar la decision macro en modo normal
- hay smoke tests de integracion reales
- el contrato parser/config/env sigue consistente

No se considera aprobada si:

- solo se agrego un wrapper cosmetico
- `macro_type` sigue clavado o inferido localmente
- el agente activo sigue saliendo de heuristicas Python
- el stub local sigue activo en entrenamiento normal

---

## 15. Prompt Corto Recomendado para Claude

```text
Lee y sigue obligatoriamente:
`Documentacion/PLAN/BHUMAN_IN_LOOP_SIDECAR_GUIDE.md`

Objetivo:
implementar un sidecar minimo pero real de decision BHuman in loop para el stack Python del proyecto RL.

Reglas:
- no reaudites todo desde cero
- no toques parser ni kickInfo salvo bug real
- no simplifiques las patadas
- usa `GroundTruthWorldState` y `OracledWorldModelProvider` como base conceptual del input
- el sidecar debe decidir `active_agent`, `SkillRequest`, `MotionRequest` y exponer `ActivationGraph`
- `environment.py` debe consumir el sidecar como fuente de verdad en modo normal
- el stub local debe quedar solo para smoke tests
- agrega smoke tests de integracion reales
- no toques `train.py` salvo adaptacion minima

Entrega final:
- cambios por archivo
- que criterio de aprobado cumple cada cambio
- que quedo pendiente
- si hubo bloqueo tecnico para sidecar vs headless
```

> ESTADO DOCUMENTAL: `referencia`
>
> La copia preservada y citada de este documento vive en:
> [Documentacion/REFERENCIA/PLAN/BHUMAN_IN_LOOP_SIDECAR_GUIDE.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/BHUMAN_IN_LOOP_SIDECAR_GUIDE.md)
> No usar este archivo como baseline por si solo.
