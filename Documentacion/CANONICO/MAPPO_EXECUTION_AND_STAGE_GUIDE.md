# MAPPO Execution And Stage Guide

Estado de referencia: 2026-04-13

Objetivo:

- concentrar en un solo documento la lectura ejecutiva del plan MAPPO
- preservar el enfoque de escalado por fases sin duplicar auditorias
- dejar claro que el exito del proyecto no es solo "meter goles", sino escalar sin romper el contrato con BHuman

---

## 1. Scope del proyecto

El proyecto busca entrenar agentes que:

- lleguen al balon
- entreguen takeover limpio a BHuman
- escalen desde approach individual hasta situaciones con mas complejidad

No busca, en esta fase:

- reemplazar la ejecucion fina de BHuman
- entrenar joints o control motor
- romper el contrato de decision real del ejecutor

---

## 2. Fases MAPPO

### Fase 1

Aprender approach y posicionamiento que hagan facil el takeover de BHuman.

### Fase 2

Escalar a situaciones con mayor complejidad local:

- obstaculos
- balon mas lejano
- episodios con mas necesidad de resolucion de approach

### Fase 3

Preparar el sistema para escenarios mas ricos y escalables, sin borrar el baseline estable de la opcion vigente.

---

## 3. Metricas de lectura sana

No usar una sola metrica como verdad absoluta.

La lectura correcta combina:

- `handoff_reached`
- `TkReady`
- `playball_entry`
- `TkSuccess`
- mezcla de macros (`shoot / duel / dribble / pass`)
- colisiones y episodios muertos

---

## 4. Regla de go / no-go por escalado

No avanzar solo porque:

- `goal` sube
- o porque una corrida aislada mete muchos goles

Si avanzar cuando:

- el approach mejora
- `handoff_reached` se sostiene
- `TkReady` no se degrada de forma estructural
- `TkSuccess` no cae por fixes fisicos mal puestos
- el agente no entra en loops contra obstaculos o bandas

---

## 5. Lecciones importantes ya confirmadas

- No arreglar cinematica destruyendo la aproximacion natural
- No convertir `TkReady` en un alias de `handoff_reached`
- No usar `goal` bruto para comparar corridas con distinto respawn
- No sobrerreaccionar a una corrida buena si abre una brecha sim2real

---

## 6. Regla de diseño para futuros agentes y asistentes

Si un fix mejora una corrida pero:

- hardcodea behavior
- reduce diversidad de takeover
- inventa evidencia semantica
- o abre una brecha sim2real obvia

entonces ese fix no es escalable aunque el numero suba temporalmente.

---

## 7. Archivos relacionados

- [BHUMAN_RL_CURRICULUM_CONTRACT.md](/c:/ALEJO/RL/Documentacion/CONTEXTO%20BHUMAN/BHUMAN_RL_CURRICULUM_CONTRACT.md)
- [ENV_REWARD_CURRICULUM_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/ENV_REWARD_CURRICULUM_GUIDE.md)
- [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](/c:/ALEJO/RL/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)

---

## 8. Motivacion de esta consolidacion

Este documento absorbe la lectura ejecutiva de:

- `Documentacion/REFERENCIA/PLAN/MAPPO_Plan_3Fases.md`
- `Documentacion/REFERENCIA/PLAN/MAPPO_Auditoria.md`
- `Documentacion/REFERENCIA/PLAN/FASE3ROADMAP.md`

Los documentos anteriores pueden seguir existiendo como apoyo historico o detallado, pero la lectura operativa del escalado MAPPO debe entrar por aqui.
