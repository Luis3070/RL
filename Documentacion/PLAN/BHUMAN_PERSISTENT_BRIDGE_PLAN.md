# BHuman Persistent Bridge Plan

Documento relacionado:

- [BHUMAN_IN_LOOP_COMPACT_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_COMPACT_GUIDE.md)
  Entrada corta para el baseline actual, smokes canonicos y mapa minimo de lectura.

- [BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md)
  Este plan corresponde al baseline de la opcion 1. La guia relacionada explica como mantener esta ruta estable y como abrir mas adelante una opcion 2 sin romper el bridge persistente actual.

## Estado actual

Implementación ya aplicada parcialmente en el proyecto:

- `bridge_protocol.py` ya intenta `pipe_process` persistente por defecto también en Windows
- `oneshot_file_fallback` quedó como fallback explícito, no como camino principal
- el baseline actual ya viaja explícitamente como `bridge_mode = persistent_2d_cognition`
- `smoke_test_rlbridge_persistent_basic.py` ya valida:
  - startup persistente
  - 3 frames consecutivos
  - `reset()`
  - mismo proceso vivo
- `smoke_test_environment_bridge.py` ya valida:
  - backend real
  - `bridge_debug_transport_mode = pipe_process`
  - `sidecar_contract_validation_mode = False`

Métricas observadas tras aplicar esta fase:

- `smoke_test_environment_bridge.py`: ~`12-16 ms` por `env.step()`
- episodio quick debug stage 0 (`96` steps): ~`0.5 s`

Esto significa que la fase central del Plan B (bridge persistente funcional) ya quedó operativa.
Lo que sigue abierto es endurecimiento, semántica y robustez adicional, no el transporte base.

## 1. Propósito exacto

Este documento define, de forma operativa y detallada, cómo migrar el sistema actual de entrenamiento `RL <-> BHuman` desde el modo `oneshot` hacia un **bridge persistente**.

Objetivo principal:

- evitar arrancar y apagar `RLBridge.exe` en cada `env.step()`
- mantener a `BHuman` realmente dentro del loop
- reducir el costo por step desde cientos de milisegundos hacia decenas de milisegundos

Este documento está escrito para que:

- Claude
- Gemini
- o un desarrollador humano

puedan ejecutar la migración sin perder el hilo de la arquitectura ni introducir regresiones por interpretación ambigua.

---

## 2. Resumen ejecutivo

## Situación actual

En Windows, el código actual cae por defecto en `oneshot_file_fallback`.

Eso implica:

1. Python genera un archivo temporal con `init + frame`
2. lanza `RLBridge.exe`
3. BHuman crea robot/threads
4. BHuman hace bootstrap hasta estado operacional
5. procesa un solo frame
6. imprime la `decision`
7. hace shutdown completo
8. Python repite todo en el siguiente `step`

Resultado medido:

- `env_step_mean`: ~`560-590 ms`
- `bridge_decision_time_us`: ~`2200-2700 us` (`~2.2-2.7 ms`)

Diagnóstico:

- la decisión BHuman en sí **no** es el cuello de botella
- el costo enorme está en:
  - startup del proceso
  - bootstrap `initial -> ready -> set -> playing`
  - archivos temporales
  - parsing del `oneshot`
  - shutdown por step

## Objetivo del Plan B

Transformar el flujo a:

1. arrancar `RLBridge.exe` una sola vez
2. mantener el proceso vivo
3. enviar muchos `frame`
4. recibir muchas `decision`
5. usar `reset` entre episodios
6. hacer `shutdown` solo al final

---

## 3. Qué significa “frames en vivo”

No significa transmitir video, ni imágenes, ni una secuencia visual tipo cámara.

Significa transmitir **snapshots estructurados del mundo** por frame.

Cada `frame` representa el estado del mundo en ese instante:

- `frame_id`
- `timestamp_ms`
- `ball`
- `own_team`
- `opp_team`
- `game_state`
- `own_score`
- `opp_score`

Eso es exactamente lo que ya modela el contrato actual en [bridge_protocol.py](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bridge/bridge_protocol.py).

La diferencia entre modo actual y modo persistente no es el formato del frame.

La diferencia es el **lifecycle del proceso**:

- hoy: 1 proceso por 1 frame
- plan B: 1 proceso por muchos frames

---

## 4. Situación actual del código

## 4.1 Python

La clase principal es [BHumanBridge](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bridge/bridge_protocol.py#L260).

Puntos clave actuales:

- [start](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bridge/bridge_protocol.py#L291)
- [step](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bridge/bridge_protocol.py#L378)
- [reset](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bridge/bridge_protocol.py#L487)
- [shutdown](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bridge/bridge_protocol.py#L505)
- [_send_raw](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bridge/bridge_protocol.py#L547)
- [_recv](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bridge/bridge_protocol.py#L559)
- [_step_oneshot_file_transport](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bridge/bridge_protocol.py#L674)

Punto crítico:

En Windows, [start](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bridge/bridge_protocol.py#L291) fuerza `oneshot_file_fallback` por defecto. Ese es el cuello de botella estructural.

## 4.2 C++

El bridge real vive en:

- [bridge/bhuman_bridge.cpp](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bridge/bhuman_bridge.cpp)

Ese binario:

- ya es capaz de recibir `init`
- ya puede procesar `frame`
- ya devuelve `decision`
- ya puede hacer bootstrap hacia un estado operacional
- ya funciona dentro del environment

Pero aún está siendo usado con lifecycle incorrecto para entrenamiento intensivo:

- nacimiento por frame
- muerte por frame

## 4.3 Environment y sidecar

El flujo alto nivel ya está bien encaminado:

- [environment.py](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/environment.py)
- [bhuman_sidecar.py](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/bhuman_sidecar.py)

Esto es importante:

- no hace falta rediseñar el contrato del env
- no hace falta rediseñar el frame schema
- no hace falta tocar reward como paso inicial

El gran problema está en el transporte/lifecycle del bridge.

---

## 5. Estimación de tiempos esperados

Base observada:

- `env_step_mean ≈ 588 ms`
- `bridge_mean ≈ 2.5 ms`

Sobrecosto estructural aproximado:

- `588 - 2.5 = 585.5 ms`

### Escenario conservador

Suponiendo:

- IPC persistente moderado
- parseo
- sincronización
- reset limpio

Tiempo esperado por `step`:

- `25-50 ms`

Mejora:

- `~12x` a `~24x`

### Escenario realista

Suponiendo:

- stdio persistente estable
- bootstrap solo una vez
- frame loop limpio

Tiempo esperado por `step`:

- `12-25 ms`

Mejora:

- `~23x` a `~49x`

### Escenario optimista

Suponiendo:

- stdio persistente robusto
- sin reconexiones
- framing limpio

Tiempo esperado por `step`:

- `8-15 ms`

Mejora:

- `~39x` a `~73x`

### Impacto sobre episodio quick debug de 96 steps

Estado actual:

- `~57-58 s`

Plan B:

- conservador: `2.4-4.8 s`
- realista: `1.2-2.4 s`
- optimista: `0.8-1.4 s`

### Impacto sobre Stage 0 original de 455 steps

Estado actual:

- `~4.5 min`

Plan B:

- conservador: `11.4-22.8 s`
- realista: `5.5-11.4 s`
- optimista: `3.6-6.8 s`

Conclusión:

El plan B no es una optimización menor. Cambia completamente la factibilidad del entrenamiento.

---

## 6. Invariantes del diseño

Estas reglas no deben romperse durante la implementación.

1. El contrato `frame -> decision` debe mantenerse.
2. `BHuman` debe seguir siendo la fuente real de decisión.
3. `oneshot` no debe desaparecer al inicio; debe quedar como fallback.
4. `reset()` debe reiniciar episodio sin recrear proceso.
5. `shutdown()` debe ser limpio y explícito.
6. No debe introducirse un fallback silencioso a stub Python en entrenamiento.
7. Los smokes actuales no deben perderse; deben ampliarse.

---

## 7. Arquitectura objetivo final

```text
train.py
  -> wrapper.py
    -> environment.py
      -> bhuman_sidecar.py
        -> bridge_protocol.py
          -> RLBridge.exe (persistente)
            -> BHuman Cognition
              -> decision JSON
```

## Lifecycle esperado

### Inicio de entrenamiento

1. Python crea `BHumanBridge`
2. llama `start()`
3. `RLBridge.exe` arranca una sola vez
4. Python recibe `ready`

### Durante episodio

1. Python llama `step(frame)`
2. bridge envía línea JSON persistente
3. BHuman procesa frame
4. bridge devuelve `decision`

### Reset de episodio

1. Python llama `reset()`
2. BHuman limpia estado interno
3. se confirma `ack`
4. `frame_id` vuelve a `1`

### Fin de entrenamiento

1. Python llama `shutdown()`
2. `RLBridge.exe` termina limpio

---

## 8. Protocolo objetivo detallado

## Mensajes de entrada

### `init`

```json
{"type":"init","scenario":"2D","location":"4v4_Scaled","team_number":1,"player_number":1,"num_players":4,"bridge_mode":"persistent_2d_cognition"}
```

### `frame`

```json
{
  "type":"frame",
  "frame_id":123,
  "timestamp_ms":4059,
  "ball":{"x":100.0,"y":200.0,"vx":0.0,"vy":0.0},
  "own_team":[...],
  "opp_team":[...],
  "game_state":"playing",
  "own_score":0,
  "opp_score":0
}
```

### `reset`

```json
{"type":"reset"}
```

### `shutdown`

```json
{"type":"shutdown"}
```

## Respuestas de salida

### `ready`

```json
{"type":"ready","version":"1.0","bridge_mode":"persistent_2d_cognition","scenario":"2D","location":"4v4_Scaled"}
```

### `decision`

La decisión debe seguir exponiendo al menos:

- `type`
- `frame_id`
- `player_number`
- `skill_request`
- `motion_request`
- `activation_graph`
- `strategy_status`
- `behavior_status`
- `decision_time_us`
- debug provenance fields

### `ack`

```json
{"type":"ack","command":"reset"}
```

### `error`

```json
{"type":"error","message":"..."}
```

---

## 9. Trabajo exacto por archivo

## 9.1 `bridge/bridge_protocol.py`

### Estado actual

- ya tiene transporte persistente implementado parcialmente
- pero en Windows entra a `oneshot_file_fallback` por defecto

### Objetivo

Hacer que el modo normal en Windows sea:

1. intentar persistente
2. si falla, fallback explícito a `oneshot`

### Cambios concretos

1. Introducir una preferencia de transporte explícita:
   - `preferred_transport = "persistent"`
   - `fallback_transport = "oneshot"`

2. En `start()`:
   - no saltar inmediatamente a `oneshot` por `os.name == "nt"`
   - intentar `subprocess.Popen(... stdin=PIPE, stdout=PIPE, stderr=PIPE ...)`
   - enviar `init`
   - esperar `ready`

3. Si la negociación persistente falla:
   - loggear motivo
   - cambiar a `oneshot_file_fallback`
   - dejarlo explícito en metadata

4. En `step()`:
   - si `_transport_mode == "pipe_process"`, usar:
     - `_send_raw(frame_json)`
     - `_recv(timeout=...)`
   - solo caer a `oneshot` si el persistente falla de verdad

5. En `reset()`:
   - debe enviar `{"type":"reset"}`
   - esperar `ack`
   - poner `_frame_counter = 0`

6. En `shutdown()`:
   - debe mandar `shutdown` si el proceso sigue vivo
   - debe drenar/stopear correctamente

### Riesgos concretos en este archivo

- buffering de stdout en Windows
- lectura parcial de líneas
- `_recv_windows_line` devolviendo fragmentos corruptos
- deadlock si el proceso escribe stderr sin drenaje

### Invariante

No romper `oneshot_file_fallback` mientras se construye el persistente.

## 9.2 `bridge/bhuman_bridge.cpp`

### Estado actual

Ya sabe:

- inicializar
- procesar frame
- devolver decisión
- hacer bootstrap operacional

### Objetivo

Aceptar muchos frames en una misma vida del proceso.

### Cambios concretos

1. Verificar que el loop principal siga leyendo múltiples comandos hasta `shutdown`.
2. Verificar que `processFrame()` no dependa de proceso recién creado.
3. Hacer que `reset`:
   - limpie contadores
   - limpie estado por episodio
   - rearme bootstrap flags
   - no destruya robot/thread principal
4. Confirmar que `init` se rechaza o ignora si llega dos veces.

### Debe evitarse

- recrear `Robot` en cada `frame`
- reinicializar tiempo global por `frame`
- usar bootstrap completo por cada `step`

### Invariante

Una sola instancia de `RLBridge.exe` debe soportar:

- `init`
- muchos `frame`
- `reset`
- muchos `frame`
- `shutdown`

## 9.3 `bhuman_sidecar.py`

### Estado actual

Ya encapsula el backend real.

### Objetivo

Mantener una sola instancia viva de `BHumanBridge` durante el ciclo del env.

### Cambios concretos

1. `start()` solo una vez
2. `decide()` no recrea bridge
3. `reset()` resetea bridge persistente
4. `close()` hace `shutdown()`

### Invariante

No debe existir re-creación silenciosa del bridge por frame.

## 9.4 `environment.py`

### Estado actual

Llama sidecar por step y lo cierra en `env.close()`.

### Objetivo

Aprovechar el backend persistente sin cambiar contrato del env.

### Cambios concretos

1. `env.reset()` debe seguir llamando `sidecar.reset()`
2. `env.step()` debe seguir llamando `sidecar.decide()`
3. `env.close()` debe seguir llamando `sidecar.close()`

### Importante

Este archivo no debería ser el centro del cambio.
El trabajo grande está en transporte/bridge.

---

## 10. Orden estricto de implementación

## Fase 0 — Preparación

Antes de tocar nada:

1. confirmar que `oneshot` actual sigue funcionando
2. guardar smoke baseline
3. no mezclar cambios de reward/train aquí

## Fase 1 — Persistente mínimo en Python

Objetivo:

- lograr `start() -> ready` por stdio persistente en Windows

Aprobado si:

- el proceso queda vivo
- `_transport_mode == "pipe_process"`

## Fase 2 — Varios frames

Objetivo:

- mandar al menos 3 frames consecutivos al mismo proceso

Aprobado si:

- recibes 3 decisiones válidas
- el proceso sigue vivo al final

## Fase 3 — Reset persistente

Objetivo:

- correr 2 episodios en el mismo proceso

Aprobado si:

- `reset()` devuelve `ack`
- `frame_id` vuelve a `1`

## Fase 4 — Integración env

Objetivo:

- `smoke_test_environment_bridge.py` corre en persistente

Aprobado si:

- `debug_transport_mode == "pipe_process"`
- no hay fallback silencioso a `oneshot`

## Fase 5 — Entrenamiento quick debug

Objetivo:

- correr `train.py` con quick debug profile

Aprobado si:

- episodios cierran en tiempos razonables
- watchdog refleja caída grande de `env_step_mean`

---

## 11. Smoke tests exactos que deben existir

## Smoke A — Persistente mínimo

Archivo recomendado:

- `smoke_test_rlbridge_persistent_basic.py`

Secuencia:

1. crear `BHumanBridge`
2. `start()`
3. mandar 3 frames
4. `shutdown()`

Validar:

- `ready`
- 3 `decision`
- mismo proceso vivo

## Smoke B — Reset persistente

Archivo recomendado:

- `smoke_test_rlbridge_persistent_reset.py`

Secuencia:

1. `start()`
2. `frame`
3. `frame`
4. `reset()`
5. `frame`
6. `shutdown()`

Validar:

- `ack reset`
- `frame_id` reiniciado

## Smoke C — Env persistent bridge

Reusar:

- [smoke_test_environment_bridge.py](/c:/Users/Alegu/OneDrive/Desktop/ROBOTICA/ROBOCOP/RL/smoke_test_environment_bridge.py)

Añadir validaciones:

- `sidecar_backend == real`
- `sidecar_contract_validation_mode == False`
- `bridge_debug_transport_mode == pipe_process`
- `bridge_frame_id` incremental

## Smoke D — Fallback honesto

Objetivo:

- romper el persistente a propósito y verificar fallback

Validar:

- entra a `oneshot_file_fallback`
- lo reporta explícitamente
- no finge persistente

---

## 12. Instrumentación obligatoria

Durante la migración, deben existir logs/metadata suficientes para ver:

- `debug_transport_mode`
- `debug_bridge_mode`
- `debug_reconnect_count`
- `bridge_frame_id`
- `decision_time_us`
- si el backend cayó a fallback
- si el proceso fue recreado

Sin eso, la depuración se vuelve ciega.

---

## 13. Riesgos y mitigaciones

## Riesgo 1 — Deadlock de stdio en Windows

Síntoma:

- `start()` cuelga
- `_recv()` nunca devuelve

Mitigación:

- timeout estricto
- fallback a `oneshot`
- si reaparece, pasar a named pipes

## Riesgo 2 — `reset()` incompleto

Síntoma:

- segundo episodio hereda estado del primero
- `frame_id` no reinicia

Mitigación:

- smoke de reset dedicado
- limpiar flags de bootstrap

## Riesgo 3 — `frame` y `decision` se desincronizan

Síntoma:

- frame N recibe decisión de frame N-1

Mitigación:

- validar `frame_id` en respuesta
- fallar duro si no coincide

## Riesgo 4 — Persistente estable pero lento

Síntoma:

- sigue > `100 ms` por step

Mitigación:

- medir breakdown:
  - Python serialization
  - send/recv
  - BHuman decision
  - reset

---

## 14. Plan de backup si stdio persistente falla

Si `stdin/stdout` persistente no queda confiable en Windows:

1. mantener `RLBridge.exe` vivo
2. reemplazar solo el transporte por named pipe
3. conservar intacto:
   - contrato `frame`
   - contrato `decision`
   - lógica del env
   - sidecar

Esto es importante:

el plan B no depende conceptualmente de stdio.
Depende de mantener el proceso vivo.

---

## 15. Criterios de éxito finales

El Plan B se considera realmente logrado si:

1. `train.py` ya no usa `oneshot` como modo normal
2. `env_step_mean` cae por debajo de `100 ms`
3. quick debug episodes duran segundos, no minutos
4. el proceso del bridge sobrevive múltiples frames
5. el proceso del bridge sobrevive múltiples episodios con `reset`
6. `oneshot` queda como fallback explícito
7. los smokes pasan en persistente

---

## 16. Qué NO hacer

1. No tocar reward como primer paso del plan B.
2. No cambiar el schema de `frame` si no es estrictamente necesario.
3. No eliminar `oneshot` hasta que el persistente sea confiable.
4. No mezclar esta migración con refactors grandes del env.
5. No declarar éxito porque “BHuman responde”; el criterio real es persistencia estable.

---

## 17. Recomendación de ejecución

Implementar en este orden:

1. `bridge_protocol.py`
2. `bhuman_bridge.cpp`
3. smoke persistente básico
4. smoke persistente con reset
5. smoke env persistente
6. `train.py` quick debug

Ese orden minimiza riesgo y deja claro en qué capa falla si algo se rompe.
