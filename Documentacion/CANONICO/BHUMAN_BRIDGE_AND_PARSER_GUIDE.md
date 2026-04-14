# BHuman Bridge And Parser Guide

Estado de referencia: 2026-04-13

Objetivo:

- consolidar en un solo lugar el contrato entre bridge C++, sidecar Python y environment
- evitar incoherencias de parser que abran brechas sim2real
- dejar claro que señales son reales de BHuman y cuales son derivadas por el entorno

---

## 1. Principio de integridad

La regla principal de esta capa es:

- no reinterpretar una senal de BHuman en una semantica que el contrato real no soporta

Ejemplos criticos:

- `BehaviorStatus.walkingTo` y `BehaviorStatus.shootingTo` son robot-relative
- `SkillRequest.target` es field-based, pero no siempre contiene una traslacion util para `shoot()` o `passTo()`
- `macro_type`, `TkReady` y `TkSuccess` no son salidas puras de BHuman; son metricas del proyecto construidas sobre telemetria real + logica operacional del env

---

## 2. Capas del pipeline

### 2.1 Bridge C++

Responsable de:

- exponer telemetria real desde BHuman
- conservar provenance del snapshot
- mantener el baseline `persistent_2d_cognition`
- no inventar evidencia de `MotionInfo` cuando el backend activo no la provee

### 2.2 Sidecar Python

Responsable de:

- parsear la salida del bridge
- normalizar nombres tacticos y de macros
- preservar la diferencia entre evidencia fuerte y señales incompletas

### 2.3 Environment

Responsable de:

- convertir telemetria real en una ejecucion operacional 2D
- construir `BallTo` y `ExecTo` sin traicionar la semantica del bridge
- medir `handoff_reached`, `TkReady` y `TkSuccess`

Regla:

- el environment puede derivar variables operacionales
- pero no debe reescribir la decision alta de BHuman por conveniencia local

---

## 3. Fuente de verdad por senal

### 3.1 Senales esencialmente reales de BHuman

- `skill_request`
- `motion_request`
- `active_agent`
- activaciones reales del grafo (`KickAtGoal`, `DribbleToGoal`, `Zweikampf`, etc.)
- provenance del bridge

### 3.2 Senales operacionales derivadas en Python

- `BallTo`
- `ExecTo`
- `takeover_rate`
- `TkReady`
- `TkSuccess`
- `HOLD / RELEASE / PROXY`

Estas ultimas no son "falsas", pero si pertenecen a la capa hibrida del proyecto.

---

## 4. Reglas de parser que no se deben romper

### 4.1 `BehaviorStatus`

- `walkingTo`: robot-relative
- `shootingTo`: robot-relative

Nunca tratarlas como field coordinates sin una conversion explicita.

### 4.2 `SkillRequest`

- `target` esta en campo
- `shoot()` no entrega una traslacion de destino util por si sola
- `passTo()` no equivale a un target de balon de campo listo para usar
- `dribbleTo()` usa sobre todo direccion/rotacion

Conclusion:

- no usar `SkillRequest.target` como reemplazo universal de `BehaviorStatus` o de los motion targets

---

## 5. Orden de confianza recomendado

Cuando varias senales compiten, la heuristica correcta es:

1. telemetria de motion/bridge explicitamente field-based
2. direcciones fuertes de motion + longitud util
3. activacion real del option/skill en BHuman
4. `BehaviorStatus` relativo como telemetria complementaria
5. inferencia local del environment solo como ultimo recurso

---

## 6. Reglas anti-brecha sim2real

No hacer:

- reinterpretar coordenadas relativas como absolutas
- promover `PREP_SHOOT` desde muy lejos del balon
- usar `SkillRequest.target` como target universal de tiro/pase
- inventar `last_kick_type`
- declarar takeover exitoso solo porque hubo overlap local sin evidencia de ejecucion

Si hacer:

- preservar la decision alta de BHuman
- retrasar la geometria estricta de ejecucion hasta control real del balon
- separar approach natural de BHuman vs fase terminal `HOLD/RELEASE/PROXY`

---

## 7. Archivos relacionados

- [BHUMAN_RL_CURRICULUM_CONTRACT.md](/c:/ALEJO/RL/Documentacion/CONTEXTO%20BHUMAN/BHUMAN_RL_CURRICULUM_CONTRACT.md)
- [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](/c:/ALEJO/RL/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
- [BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md)
- [DONT FIX CINEMATICS LIKE THIS WAY.md](/c:/ALEJO/RL/Documentacion/PLAN/DONT%20FIX%20CINEMATICS%20LIKE%20THIS%20WAY.md)

---

## 8. Motivacion de esta consolidacion

Este documento absorbe el nucleo util de:

- `Documentacion/REFERENCIA/PLAN/BHUMANPARSER_GUIDE.md`
- `Documentacion/REFERENCIA/PLAN/BHUMAN_IN_LOOP_SIDECAR_GUIDE.md`
- `Documentacion/REFERENCIA/PLAN/BHUMAN_PERSISTENT_BRIDGE_PLAN.md`
- `Documentacion/REFERENCIA/CONTEXTO BHUMAN/CFG_UNIFIED_ANALYSIS.md`
- `Documentacion/REFERENCIA/CONTEXTO BHUMAN/DECISION_POINTS_CPP.md`

Los documentos anteriores pueden seguir existiendo como referencia detallada, pero no deben competir con este resumen como contrato operativo de parser y bridge.
