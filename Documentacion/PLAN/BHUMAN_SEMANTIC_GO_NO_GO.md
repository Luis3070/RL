# BHuman Semantic Validation and GO / NO-GO

## Objetivo
Definir una compuerta clara antes de lanzar entrenamiento largo con `BHuman in loop`.

Esta guía no valida percepción física real ni robot NAO real. Valida:
- semántica del contrato
- consistencia de `macro_type`
- consistencia de `skill_request`
- `ActivationGraph` estructurado
- backend real dentro del environment
- ausencia de fallback silencioso
- estabilidad operacional mínima

## Smoke recomendado
Ejecutar primero:

```powershell
python smoke_test_environment_bridge.py
python smoke_test_semantic_strong.py
```

## Qué valida `smoke_test_semantic_strong.py`
### Capa 1: fixtures deterministas
- `shoot`
- `pass`
- `dribble`
- `duel`
- `is_playing` gating
- `kick metadata`
- `ActivationGraph` estructurado

### Capa 2: environment real
- `sidecar_backend = real`
- `sidecar_contract_validation_mode = False`
- `takeover_simulator_active = False`
- `bridge_debug_transport_mode = pipe_process`
- `bridge_debug_game_state = playing`
- `bridge_debug_requested_state = playing`
- `macro_type` válido
- `sidecar_skill_request` válido y nunca `duel`
- `active_agent_id` presente
- `ActivationGraph` con nodos estructurados

## GO
Se considera `GO` para entrenamiento largo si se cumplen todas:

1. `smoke_test_environment_bridge.py` pasa.
2. `smoke_test_semantic_strong.py` pasa.
3. El backend reportado en el env es `real`.
4. `contract_validation_mode` es siempre `False`.
5. El transporte reportado es `pipe_process`.
6. `takeover_simulator_active` es siempre `False`.
7. `macro_type` y `sidecar_skill_request` no violan el contrato.
8. El `ActivationGraph` llega estructurado y consistente.
9. No aparecen reconexiones repetidas del bridge durante warmup.
10. El watchdog del entrenamiento mantiene:
   - `stability_score >= 0.95`
   - `contract_validation_rate = 0.0`
   - `non_pipe_rate = 0.0`

## CONDITIONAL GO
Se puede considerar `GO condicional` para entrenamiento exploratorio corto si:

1. Los dos smokes pasan.
2. Hay desvíos menores de timing o memoria, pero no de backend/semántica.
3. El watchdog muestra:
   - `stability_score >= 0.85`
   - `contract_validation_rate = 0.0`
   - `non_pipe_rate = 0.0`
4. No hay pérdida de JSON, no hay timeouts y no hay caídas del bridge.

Esto sirve para:
- warmup
- entrenamiento corto
- tuning de reward
- validación de curriculum

No sirve todavía para declarar transplante robusto.

## NO-GO
Es `NO-GO` si ocurre cualquiera de estas:

1. El environment smoke falla.
2. El semantic smoke falla.
3. El backend cae a `contract_validation`.
4. El transporte cae a `oneshot` o algo distinto de `pipe_process`.
5. Se reactiva `takeover_simulator_active`.
6. `skill_request = duel`.
7. `macro_type = playBall_duel` sin `zweikampf` en `ActivationGraph`.
8. `macro_type` y `skill_request` se contradicen.
9. El watchdog muestra:
   - `contract_validation_rate > 0`
   - `non_pipe_rate > 0`
   - `stability_score < 0.85`
10. Hay reconexiones frecuentes del bridge o memory growth sostenido no explicado.

## Cómo leer el watchdog en entrenamiento
El `watchdog.png` y la consola ahora exponen:
- `episode_wall_s`
- `mean_env_step_ms`
- `mean_bridge_ms`
- `process_memory_mb`
- `stability_score`
- `bridge_reconnect_count`
- `contract_validation_rate`
- `non_pipe_rate`
- `not_playing_rate`

Interpretación rápida:
- `mean_bridge_ms` bajo y `mean_env_step_ms` razonable: buen loop persistente
- `process_memory_mb` estable: sin fuga evidente
- `stability_score` alto: loop sano
- `contract_validation_rate = 0`: backend real sostenido
- `non_pipe_rate = 0`: persistente real, no fallback

## Recomendación operativa
Antes de una corrida larga:

1. correr ambos smokes
2. correr una sesión corta de entrenamiento práctico
3. revisar `watchdog.png`
4. solo después lanzar entrenamiento largo
