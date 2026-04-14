# Environment, Reward And Curriculum Guide

Estado de referencia: 2026-04-13

Objetivo:

- consolidar la logica operacional del environment
- fijar reglas de reward sin romper el contrato con BHuman
- dejar el curriculum stage 0-4 como parte estable y auditable del proyecto

---

## 1. Principio rector

El environment no debe competir con BHuman.

Su trabajo es:

- entrenar approach y posicionamiento
- entregar takeover limpio
- medir honestamente acceptance y success
- mantener una ejecucion 2D util sin mentir sobre el origen de las decisiones

---

## 2. Metricas que no deben confundirse

- `handoff_reached`
  El RL llego a la condicion geometrica de oferta de takeover

- `TkReady`
  BHuman acepto takeover de forma operacionalmente real

- `playball_entry`
  BHuman entro a `playBall_*`

- `TkSuccess`
  El takeover termino con evidencia suficiente de ejecucion/progreso

Regla:

- nunca volver a colapsar `handoff_reached` y `TkReady` en una sola lectura

---

## 3. Reward: que debe optimizar

La cadena correcta es:

`approach limpio -> handoff_reached -> TkReady -> playball_entry -> TkSuccess`

La reward debe:

- ayudar a llegar al balon y a la pose de handoff
- no castigar al RL por microejecucion de BHuman cuando takeover ya es responsabilidad del ejecutor
- castigar fuerte errores reglamentarios como colisiones graves
- no premiar goles "regalados" por respawns malos

### 3.1 Reglas importantes

- si el agente recupera la pelota hacia atras en contexto defensivo, no castigarlo como regresion tactica
- si takeover ya esta activo, evitar cargar al RL con ruido de microejecucion de BHuman
- colision con obstaculo/oponente en contextos reglamentarios debe ser dura y explicita

---

## 4. Curriculum stage 0 a 4

### Stage 0

- focus: approach individual limpio
- sin obstaculos
- balon entre `800` y `1600 mm`

### Stage 1

- focus: approach individual con un obstaculo estatico
- un obstaculo estatico
- balon entre `1600` y `3200 mm`
- spawn del balon mas centrado para evitar demasiados episodios muertos en banda

### Stage 2

- se mantiene la dificultad base previa
- algunos episodios arrancan con balon dinamico

### Stage 3

- se mantiene la dificultad base previa
- algunos episodios arrancan con balon dinamico

### Stage 4

- se mantiene la dificultad base previa
- algunos episodios arrancan con balon dinamico

Regla:

- el porcentaje de balon dinamico crece con el stage
- no convertir stage 0-1 en un benchmark de ejecucion compleja de BHuman

---

## 5. Reglas de integridad para fixes fisicos

No hacer:

- endurecer `HOLD/RELEASE/PROXY` y a la vez secuestrar el approach desde lejos
- corregir jitter invadiendo la aproximacion natural de BHuman
- subir `TkSuccess` relajando tanto el proxy que `shoot` se vuelva pseudo-dribble

Si hacer:

- separar approach natural y fase terminal de kick
- usar telemetria real de BHuman primero
- medir el takeover con honestidad, no solo con overlap local

---

## 6. Reglas de escalado

Antes de agregar un stage nuevo o endurecer uno existente, validar siempre:

- `handoff_reached`
- `TkReady`
- `playball_entry`
- `TkSuccess`
- colisiones
- episodios muertos por pelota en banda

Un stage no escala bien si mejora una de esas y destruye las otras.

---

## 7. Archivos relacionados

- [BHUMAN_RL_CURRICULUM_CONTRACT.md](file:///c:/ALEJO/RL/Documentacion/CANONICO/BHUMAN_RL_CURRICULUM_CONTRACT.md)
- [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](file:///c:/ALEJO/RL/Documentacion/CANONICO/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
- [DONT FIX CINEMATICS LIKE THIS WAY.md](file:///c:/ALEJO/RL/Documentacion/CANONICO/DONT%20FIX%20CINEMATICS%20LIKE%20THIS%20WAY.md)

---

## 8. Motivacion de esta consolidacion

Este documento absorbe el nucleo operativo de:

- `Documentacion/REFERENCIA/PLAN/BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md`
- `Documentacion/REFERENCIA/PLAN/ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md`
- `Documentacion/REFERENCIA/PLAN/BHUMAN_IMPLEMENTATION_CORRECTION_GUIDE.md`

Los documentos anteriores pueden seguir existiendo como referencia detallada o historica, pero el baseline operativo del environment, reward y curriculum debe leerse aqui.
