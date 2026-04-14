# Gobernanza de Documentacion y Smokes

Estado de referencia: 2026-04-13

Objetivo:

- preservar integridad tecnica y trazabilidad
- reducir la cantidad de documentos que compiten como fuente de verdad
- mantener capacidad de escalado sin perder lecciones aprendidas
- distinguir con claridad entre baseline vigente, referencia profunda e historico

---

## 1. Principio rector

El proyecto debe crecer con dos propiedades al mismo tiempo:

- `integridad`: ninguna decision operativa debe contradecir el contrato real entre RL, environment y BHuman
- `capacidad de escalado`: una persona nueva debe poder entender, validar y extender el sistema sin leer demasiados `.md` ni correr una docena de smokes aislados

La regla practica es:

- pocos documentos canonicos y vivos
- pocos smokes oficiales
- documentos historicos y smokes viejos preservados, pero claramente degradados a soporte

---

## 2. Superficie oficial objetivo

### 2.1 Documentacion oficial: 10 MDs

1. [BHUMAN_RL_CURRICULUM_CONTRACT.md](/c:/ALEJO/RL/Documentacion/CONTEXTO%20BHUMAN/BHUMAN_RL_CURRICULUM_CONTRACT.md)
2. [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](/c:/ALEJO/RL/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
3. [BHUMAN_IN_LOOP_COMPACT_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_COMPACT_GUIDE.md)
4. [BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md)
5. [DOCUMENTATION_AND_SMOKE_GOVERNANCE.md](/c:/ALEJO/RL/Documentacion/PLAN/DOCUMENTATION_AND_SMOKE_GOVERNANCE.md)
6. [BHUMAN_BRIDGE_AND_PARSER_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/BHUMAN_BRIDGE_AND_PARSER_GUIDE.md)
7. [ENV_REWARD_CURRICULUM_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/ENV_REWARD_CURRICULUM_GUIDE.md)
8. [MAPPO_EXECUTION_AND_STAGE_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/MAPPO_EXECUTION_AND_STAGE_GUIDE.md)
9. [DONT FIX CINEMATICS LIKE THIS WAY.md](/c:/ALEJO/RL/Documentacion/PLAN/DONT%20FIX%20CINEMATICS%20LIKE%20THIS%20WAY.md)
10. [HISTORICAL_DECISIONS_INDEX.md](/c:/ALEJO/RL/Documentacion/PLAN/HISTORICAL_DECISIONS_INDEX.md)

### 2.2 Smokes oficiales: 4 archivos

1. [smoke_suite_option1.py](/c:/ALEJO/RL/smoke_suite_option1.py)
2. [smoke_core_bridge_semantics.py](/c:/ALEJO/RL/smoke_core_bridge_semantics.py)
3. [smoke_takeover_real.py](/c:/ALEJO/RL/smoke_takeover_real.py)
4. [smoke_diagnostic_extended.py](/c:/ALEJO/RL/smoke_diagnostic_extended.py)

---

## 3. Capas de documentacion

### 3.1 Canonica

Estos documentos deben mantenerse cortos, vivos y coherentes entre si:

- contrato RL <-> BHuman
- checklist vivo del baseline
- guia compacta
- change guide
- gobernanza
- guia consolidada de bridge/parser
- guia consolidada de env/reward/curriculum
- guia consolidada de MAPPO/stages

### 3.2 Historico / postmortem

Estos documentos no deben usarse para definir el baseline actual, pero si para evitar regresiones:

- [pipeprocess.md](/c:/ALEJO/RL/Documentacion/HISTORICO/PLAN/pipeprocess.md)
- [PROJECT_AUDIT_FOR_RL.md](/c:/ALEJO/RL/Documentacion/HISTORICO/PLAN/PROJECT_AUDIT_FOR_RL.md)
- [RL_TRANSPLANT_GUIDE_COMPLETE.md](/c:/ALEJO/RL/Documentacion/HISTORICO/CRONOGRAMA/RL_TRANSPLANT_GUIDE_COMPLETE.md)
- [TRANSPLANT_3_PHASES_GO_NO_GO_AND_CALENDAR.md](/c:/ALEJO/RL/Documentacion/HISTORICO/CRONOGRAMA/TRANSPLANT_3_PHASES_GO_NO_GO_AND_CALENDAR.md)

### 3.3 Referencia profunda preservada

Estos pueden seguir existiendo para lectura profunda, pero ya no deben competir con el canon:

- [MAPPO_Plan_3Fases.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/MAPPO_Plan_3Fases.md)
- [MAPPO_Auditoria.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/MAPPO_Auditoria.md)
- [BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md)
- [ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md)
- [BHUMAN_IN_LOOP_SIDECAR_GUIDE.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/BHUMAN_IN_LOOP_SIDECAR_GUIDE.md)
- [BHUMAN_IMPLEMENTATION_CORRECTION_GUIDE.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/BHUMAN_IMPLEMENTATION_CORRECTION_GUIDE.md)
- [RLBRIDGE_INTERVENTION_LOG.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/RLBRIDGE_INTERVENTION_LOG.md)
- [BHUMANPARSER_GUIDE.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/BHUMANPARSER_GUIDE.md)
- [BHUMAN_PERSISTENT_BRIDGE_PLAN.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/BHUMAN_PERSISTENT_BRIDGE_PLAN.md)
- [BHUMAN_SEMANTIC_GO_NO_GO.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/BHUMAN_SEMANTIC_GO_NO_GO.md)
- [CFG_UNIFIED_ANALYSIS.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/CONTEXTO%20BHUMAN/CFG_UNIFIED_ANALYSIS.md)
- [DECISION_POINTS_CPP.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/CONTEXTO%20BHUMAN/DECISION_POINTS_CPP.md)

---

## 4. Reglas de coherencia

Nunca debe quedar duplicado y divergente en varios documentos:

- definicion de `TkReady`
- definicion de `TkSuccess`
- diferencia entre `handoff_reached` y takeover real
- baseline de bridge (`persistent_2d_cognition`, `scenario = 2D`)
- contrato semantico de `BehaviorStatus` vs `SkillRequest`
- estado del curriculum stages 0 a 4
- smokes oficiales

Si una decision cambia el baseline, no debe quedar solo en un plan largo o postmortem. Debe actualizar al menos:

- checklist
- guia compacta
- y, si aplica, contrato o change guide

---

## 5. Reglas de smoke tests

### 5.1 Operacion diaria

La validacion normal del proyecto debe entrar por:

- [smoke_suite_option1.py](/c:/ALEJO/RL/smoke_suite_option1.py)

### 5.2 Smokes oficiales

- `core`
- `takeover_real`
- `diagnostic_extended`

### 5.3 Smokes viejos preservados

Los siguientes quedan como soporte interno de los wrappers oficiales y diagnostico puntual:

- `smoke_legacy/smoke_test_rlbridge_persistent_basic.py`
- `smoke_legacy/smoke_test_environment_bridge.py`
- `smoke_legacy/smoke_test_semantic_strong.py`
- `smoke_legacy/smoke_test_sidecar.py`
- `smoke_legacy/smoke_test_real_takeover_completion.py`
- `smoke_legacy/smoke_test_real_takeover_playball.py`
- `smoke_legacy/smoke_test_real_takeover_env_positive.py`
- `smoke_legacy/smoke_test_real_takeover_pass_completion.py`
- `smoke_legacy/smoke_test_real_takeover_shoot_completion.py`
- `smoke_legacy/smoke_test_rlbridge_persistent_stress.py`
- `smoke_legacy/audit_motioninfo_takeover_evidence.py`
- `smoke_legacy/smoke_test_real_env_semantic_variants.py`

Regla:

- no crear un smoke nuevo si la cobertura cabe en uno de los 4 oficiales
- no borrar un smoke viejo si todavia conserva una fixture dificil de redescubrir
- una fixture especializada de `pass` puede preservarse fuera del gating oficial
  si su propio contrato no es de completion estable
- un detector de drift semantico puede preservarse fuera del wrapper oficial
  aunque siga siendo valioso para auditoria puntual

---

## 6. Decision sobre `pipeprocess`

`pipeprocess` ya no debe tratarse como apuesta especulativa.

Estado correcto:

- el baseline vigente usa transporte real `pipe_process`
- `pipeprocess.md` queda como documento historico

---

## 7. Resultado buscado

Una persona nueva deberia poder:

1. leer 4 documentos primero y 3 despues si necesita profundidad
2. correr una sola suite canonica
3. entender donde esta el baseline real
4. distinguir facilmente entre:
   - contrato vigente
   - referencia profunda
   - historico

Si no se logra eso, la documentacion esta escalando mal aunque el codigo mejore.
