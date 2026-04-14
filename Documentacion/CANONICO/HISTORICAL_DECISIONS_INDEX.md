# Historical Decisions Index

Estado de referencia: 2026-04-13

Objetivo:

- conservar trazabilidad sin permitir que planes viejos compitan con el baseline vigente
- dejar un mapa corto de que documentos viejos siguen siendo utiles y para que

---

## 1. Como leer este indice

Si un documento aparece aqui, no significa que sea inutil.

Significa:

- no debe guiar por si solo el estado actual del proyecto
- sigue siendo util como historial, postmortem o evidencia de una decision pasada

La fuente de verdad vigente sigue en:

- [BHUMAN_RL_CURRICULUM_CONTRACT.md](/c:/ALEJO/RL/Documentacion/CONTEXTO%20BHUMAN/BHUMAN_RL_CURRICULUM_CONTRACT.md)
- [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](/c:/ALEJO/RL/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
- [BHUMAN_IN_LOOP_COMPACT_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_COMPACT_GUIDE.md)
- [DOCUMENTATION_AND_SMOKE_GOVERNANCE.md](/c:/ALEJO/RL/Documentacion/PLAN/DOCUMENTATION_AND_SMOKE_GOVERNANCE.md)

---

## 2. Historicos importantes

### `Documentacion/HISTORICO/PLAN/pipeprocess.md`

Utilidad:

- entender los desafíos técnicos superados de `pipe_process`
- ver el histórico del diagnóstico de estabilidad que llevó al fix actual en Windows

No usar para:

- describir el baseline vigente

### `Documentacion/HISTORICO/PLAN/PROJECT_AUDIT_FOR_RL.md`

Utilidad:

- conservar auditorias antiguas del estado del proyecto

No usar para:

- decidir por si solo la arquitectura actual

### `Documentacion/HISTORICO/CRONOGRAMA/RL_TRANSPLANT_GUIDE_COMPLETE.md`
### `Documentacion/HISTORICO/CRONOGRAMA/TRANSPLANT_3_PHASES_GO_NO_GO_AND_CALENDAR.md`

Utilidad:

- historial de cronograma y expectativas originales

No usar para:

- describir el stage actual del proyecto sin contraste con checklist y guides vivos

### `DONT FIX CINEMATICS LIKE THIS WAY.md`

Caso especial:

- es historico como incidente
- pero sigue siendo una regla viva de seguridad

---

## 3. Regla para futuras limpiezas

Antes de borrar un documento historico, verificar si contiene:

- una leccion de regresion importante
- una decision de arquitectura dificil de redescubrir
- o una razon de por que una ruta anterior fracaso

Si la respuesta es si, archivar o indexar.
No borrar a ciegas.
