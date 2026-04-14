# BHuman In Loop: Guia Compacta

Estado de referencia: 2026-04-13

Objetivo:

- reducir la cantidad de smokes operativos a pocos entrypoints
- reducir la cantidad de documentos que hay que leer para operar el proyecto
- mantener trazabilidad sin perder integridad ni capacidad de escalado

Documento relacionado:

- [DOCUMENTATION_AND_SMOKE_GOVERNANCE.md](/c:/ALEJO/RL/Documentacion/PLAN/DOCUMENTATION_AND_SMOKE_GOVERNANCE.md)
  Define que es canonico, que es referencia y que es historico.

## 1. Lectura minima recomendada

Si alguien entra nuevo al proyecto, el orden corto recomendado es:

1. [BHUMAN_RL_CURRICULUM_CONTRACT.md](/c:/ALEJO/RL/Documentacion/CONTEXTO%20BHUMAN/BHUMAN_RL_CURRICULUM_CONTRACT.md)
2. [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](/c:/ALEJO/RL/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
3. [BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md)
4. [DOCUMENTATION_AND_SMOKE_GOVERNANCE.md](/c:/ALEJO/RL/Documentacion/PLAN/DOCUMENTATION_AND_SMOKE_GOVERNANCE.md)

Lectura tecnica recomendada despues:

5. [BHUMAN_BRIDGE_AND_PARSER_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/BHUMAN_BRIDGE_AND_PARSER_GUIDE.md)
6. [ENV_REWARD_CURRICULUM_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/ENV_REWARD_CURRICULUM_GUIDE.md)
7. [MAPPO_EXECUTION_AND_STAGE_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/MAPPO_EXECUTION_AND_STAGE_GUIDE.md)

## 2. Smokes canonicos

El proyecto ahora debe operar con 4 smokes oficiales:

1. [smoke_suite_option1.py](/c:/ALEJO/RL/smoke_suite_option1.py)
2. [smoke_core_bridge_semantics.py](/c:/ALEJO/RL/smoke_core_bridge_semantics.py)
3. [smoke_takeover_real.py](/c:/ALEJO/RL/smoke_takeover_real.py)
4. [smoke_diagnostic_extended.py](/c:/ALEJO/RL/smoke_diagnostic_extended.py)

Entry point recomendado:

```powershell
C:\Users\ingro\anaconda3\envs\RLRobocup\python.exe C:\ALEJO\RL\smoke_suite_option1.py --suite gating
```

Suites disponibles:

- `quick`
  Cubre el smoke `core`

- `gating`
  Cubre:
  - `core`
  - `takeover_real`

- `extended`
  Cubre:
  - `core`
  - `takeover_real`
  - `diagnostic_extended`

- `diagnostic`
  Solo `diagnostic_extended`

- `all`
  Equivale a `extended`

## 3. Como interpretar los archivos viejos

Los smokes viejos no se borran, pero pasan a ser infraestructura interna o soporte de fixtures.
Ahora viven bajo `C:\\ALEJO\\RL\\smoke_legacy\\`.

Los documentos viejos no se borran, pero deben leerse segun su rol:

- canonico
- referencia profunda
- historico / postmortem

Ver:

- [DOCUMENTATION_AND_SMOKE_GOVERNANCE.md](/c:/ALEJO/RL/Documentacion/PLAN/DOCUMENTATION_AND_SMOKE_GOVERNANCE.md)
- [HISTORICAL_DECISIONS_INDEX.md](/c:/ALEJO/RL/Documentacion/PLAN/HISTORICAL_DECISIONS_INDEX.md)

## 4. Superficie oficial del proyecto

### 4.1 Documentacion oficial de trabajo

- [BHUMAN_RL_CURRICULUM_CONTRACT.md](/c:/ALEJO/RL/Documentacion/CONTEXTO%20BHUMAN/BHUMAN_RL_CURRICULUM_CONTRACT.md)
- [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](/c:/ALEJO/RL/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
- [BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md)
- [BHUMAN_IN_LOOP_COMPACT_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_COMPACT_GUIDE.md)
- [DOCUMENTATION_AND_SMOKE_GOVERNANCE.md](/c:/ALEJO/RL/Documentacion/PLAN/DOCUMENTATION_AND_SMOKE_GOVERNANCE.md)
- [BHUMAN_BRIDGE_AND_PARSER_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/BHUMAN_BRIDGE_AND_PARSER_GUIDE.md)
- [ENV_REWARD_CURRICULUM_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/ENV_REWARD_CURRICULUM_GUIDE.md)
- [MAPPO_EXECUTION_AND_STAGE_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/MAPPO_EXECUTION_AND_STAGE_GUIDE.md)
- [DONT FIX CINEMATICS LIKE THIS WAY.md](/c:/ALEJO/RL/Documentacion/PLAN/DONT%20FIX%20CINEMATICS%20LIKE%20THIS%20WAY.md)
- [HISTORICAL_DECISIONS_INDEX.md](/c:/ALEJO/RL/Documentacion/PLAN/HISTORICAL_DECISIONS_INDEX.md)

### 4.2 Baseline operativo actual

- `bridge_mode = persistent_2d_cognition`
- `scenario = 2D`
- `location = 4v4_Scaled`
- transporte real `pipe_process`
- backend real dentro del env
- takeover real con efecto de fisica
- `MotionInfo` de `shoot/pass` aun NO es criterio de cierre de opcion 1

## 5. Regla de oro

Si una decision nueva no cabe de forma coherente dentro de esta guia, el checklist y el contrato, no esta lista para convertirse en baseline.
