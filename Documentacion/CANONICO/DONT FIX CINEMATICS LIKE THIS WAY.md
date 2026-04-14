# DONT FIX CINEMATICS LIKE THIS WAY

## Context

Durante la integración RL + BHuman en `2D`, aparecieron síntomas físicos molestos durante takeover:

- `PREP_SHOOT` entraba de forma ruidosa
- `BallTo` y `ExecTo` oscilaban demasiado
- el cuerpo parecía "levitar" o "teletransportarse" hacia el balón
- había jitter cerca del arco durante `shoot`

El objetivo legítimo era corregir esa ejecución corporal brusca sin romper la semántica real de BHuman.

## What We Tried

Intentamos arreglar el problema endureciendo la fase de ejecución fina:

- reglas más estrictas para alineación corporal
- reglas más estrictas para validar `kick_proxy`
- mayor dependencia de `HOLD`, `RELEASE` y `PROXY`
- entrada temprana a `_resolve_playball_body_target(...)`

En la práctica, eso hizo que la lógica estricta de kick secuestrara la aproximación desde demasiado lejos.

## What Went Wrong

El error no fue simplemente "amortiguar mal la cinemática".

El error real fue este:

1. BHuman todavía estaba en una fase natural de approach al balón.
2. El environment detectaba `playBall_shoot` o `shoot + walkToBallAndKick`.
3. Nuestra capa `2D` activaba demasiado pronto la geometría estricta de `PREP_SHOOT/HOLD/PROXY`.
4. El cuerpo dejaba de seguir el planner suave de BHuman.
5. La aproximación se volvía artificial, ruidosa y difícil de completar.

Eso disparó:

- más `skill_request_cleared`
- caída fuerte de `TkReady`
- caída fuerte de `TkSuccess`

## Root Cause

Queríamos arreglar la ejecución fina del takeover, pero terminamos rompiendo la fase de approach.

La física estricta no debe mandar mientras el robot todavía no tiene control geométrico real del balón.

## Anti-Pattern

No volver a hacer esto:

- endurecer `HOLD/RELEASE/PROXY` y al mismo tiempo
- permitir que esa lógica tome control desde muy lejos del balón

Eso convierte una corrección física legítima en una regresión semántica del takeover.

## Safer Rule

La regla correcta es:

- durante `approach`, respetar `motion_walk_target` y la aproximación natural de BHuman
- solo entrar al `body_target` estricto cuando exista control geométrico real del balón
- usar `HOLD/RELEASE/PROXY` como fase terminal de ejecución, no como planner de aproximación

## Practical Lesson

Si aparece un problema de "levitación", jitter o entrada brusca a `PREP_SHOOT`, no asumir que la solución es endurecer más la física local.

Primero verificar:

- si el cuerpo está entrando demasiado temprano al modo de ejecución
- si `BallTo/ExecTo` están secuestrando la aproximación
- si la capa `2D` está reemplazando prematuramente el movimiento natural de BHuman

## Final Summary

El problema original era físico.

La caída catastrófica de `TkReady` y `TkSuccess` ocurrió porque la solución aplicada invadió la fase de approach con reglas de ejecución terminal.

En corto:

`No arreglar cinemática brusca destruyendo la aproximación natural de BHuman.`
