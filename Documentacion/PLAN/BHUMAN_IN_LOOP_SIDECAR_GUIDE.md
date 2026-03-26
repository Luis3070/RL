# Guia de Implementacion Sidecar BHuman In Loop

> Fecha: 24 de marzo de 2026
> Objetivo: definir una ruta concreta para integrar BHuman in loop mediante un sidecar de decision, respetando el contrato documental del proyecto RL y evitando una reimplementacion local de las decisiones `shoot/pass/dribble/duel`.
> Scope: integracion offline/experimental para entrenamiento, shadow mode y smoke tests de integracion. No es una guia para despliegue final de competicion.

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

1. `GroundTruthWorldState`
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

