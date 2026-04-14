# Handover Manifest for Gemini 3.1 Pro High

## Purpose

Este documento transfiere el estado operativo del proyecto `RL + BHuman in loop` a un nuevo agente de alta capacidad, minimizando redundancia y evitando repetir callejones sin salida ya explorados.

Punto de entrada recomendado antes de tocar código:

1. [README.md](C:\ALEJO\RL\Documentacion\CANONICO\README.md)
2. [BHUMAN_RL_CURRICULUM_CONTRACT.md](C:\ALEJO\RL\Documentacion\CANONICO\BHUMAN_RL_CURRICULUM_CONTRACT.md)
3. [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](C:\ALEJO\RL\Documentacion\CANONICO\PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
4. [BHUMAN_IN_LOOP_COMPACT_GUIDE.md](C:\ALEJO\RL\Documentacion\CANONICO\BHUMAN_IN_LOOP_COMPACT_GUIDE.md)
5. [DOCUMENTATION_AND_SMOKE_GOVERNANCE.md](C:\ALEJO\RL\Documentacion\CANONICO\DOCUMENTATION_AND_SMOKE_GOVERNANCE.md)

## User Profile & Expertise

El usuario es:

- Científico de Datos especializado en Reinforcement Learning
- Software Developer con background de ingeniería
- Muy competente en debugging avanzado de Python y C++
- Muy sensible a brechas sim2real, integridad semántica y trazabilidad de métricas

Expectativa de colaboración:

- respuestas honestas y auditables
- no ocultar incertidumbre
- no introducir fixes que “parezcan funcionar” si rompen el contrato con BHuman
- no introducir fixes que generen brecha sim2real con Bhuman
- preservar mejoras físicas ya logradas aunque el fix semántico sea difícil

## System Architecture Snapshot

Arquitectura resumida:

- `BHuman / RLBridge (C++)`
  - fuente primaria de telemetría y decisión semántica
  - expone `BehaviorStatus`, `SkillRequest`, `MotionRequest`, metadata de activaciones y estado operativo
- `Python env / wrapper / reward / train`
  - ejecuta entrenamiento MAPPO con `skrl`
  - interpreta la telemetría del bridge
  - mantiene una capa híbrida 2D para ejecutar el takeover
- `Sidecars / parser / bridge protocol`
  - traducen el output C++ a estructuras Python
  - sostienen contratos como `macro_type`, `skill_request`, `motion_request`, `last_kick_*`, `walkingTo`, `shootingTo`
- `CMake + RLBridge build`
  - la build de `RLBridge` es parte del flujo operativo
  - el baseline vigente usa bridge persistente y no debe volver a tratar `pipeprocess` como apuesta especulativa

Skeleton map:

```text
RL/
  train.py
  wrapper.py
  reward.py
  environment.py
  bhuman_sidecar.py
  bhuman_cfg_parser.py
  bridge/
  smoke_suite_option1.py
  smoke_core_bridge_semantics.py
  smoke_takeover_real.py
  smoke_diagnostic_extended.py
  Documentacion/
    CONTEXTO BHUMAN/
    PLAN/
    REFERENCIA/
    HISTORICO/

SabanaHerons2026/
  Src/Apps/RLBridge/bhuman_bridge.cpp
  Src/Representations/BehaviorControl/BehaviorStatus.h
  Src/Representations/BehaviorControl/SkillRequest.h
  Src/Representations/BehaviorControl/SkillRequest.cpp
```

Lectura conceptual del sistema:

- La decisión alta de BHuman es real
- La interpretación operacional del takeover en `environment.py` sigue siendo parcialmente híbrida
- `BallTo`, `ExecTo`, `TkReady`, `TkSuccess` no son salidas “puras” de BHuman; son métricas o targets operacionales derivados con distintos grados de dependencia del bridge

Referencias principales:

- [BHUMAN_BRIDGE_AND_PARSER_GUIDE.md](C:\ALEJO\RL\Documentacion\CANONICO\BHUMAN_BRIDGE_AND_PARSER_GUIDE.md)
- [ENV_REWARD_CURRICULUM_GUIDE.md](C:\ALEJO\RL\Documentacion\CANONICO\ENV_REWARD_CURRICULUM_GUIDE.md)
- [MAPPO_EXECUTION_AND_STAGE_GUIDE.md](C:\ALEJO\RL\Documentacion\CANONICO\MAPPO_EXECUTION_AND_STAGE_GUIDE.md)

## The Dead Ends (Soluciones Fallidas)

Estas estrategias ya se intentaron y no deben repetirse sin evidencia nueva fuerte.

### 1. Endurecer `HOLD / RELEASE / PROXY` demasiado pronto

Qué se hizo:

- se volvieron más estrictos los criterios de proxy kick
- se permitió que la fase `PREP_SHOOT` secuestrara la aproximación desde demasiado lejos

Qué rompió:

- caída catastrófica de `TkReady`
- caída fuerte de `TkSuccess`
- explosión de `skill_request_cleared`

Lección:

- no meter geometría de ejecución fina dentro de la fase de approach

Referencia:

- [DONT FIX CINEMATICS LIKE THIS WAY.md](C:\ALEJO\RL\Documentacion\CANONICO\DONT%20FIX%20CINEMATICS%20LIKE%20THIS%20WAY.md)

### 2. Reinterpretar mal señales del bridge

Qué se detectó:

- `BehaviorStatus.walkingTo` y `BehaviorStatus.shootingTo` son robot-relative
- `SkillRequest.target` es field coordinates, pero no sirve como target posicional universal para `shoot()` o `passTo()`

Qué habría roto:

- usar `_target_array` sobre `walkingTo/shootingTo`
- o asumir que `skill_target` resuelve el destino de `shoot/pass`

Lección:

- respetar el contrato C++ exacto antes de cambiar parseo

Referencia:

- [BHUMAN_BRIDGE_AND_PARSER_GUIDE.md](C:\ALEJO\RL\Documentacion\CANONICO\BHUMAN_BRIDGE_AND_PARSER_GUIDE.md)

### 3. Hardcodear corredores ofensivos o reproyectar comportamiento de BHuman

Qué se intentó:

- proyectar `BallTo` a conos/corredores “seguros”

Qué problema generó:

- abre brecha sim2real
- hace que Python imponga dirección operacional a una macro que BHuman debería decidir

Lección:

- sanear y validar targets sí
- reescribir el behavior de BHuman no

### 4. Aflojar demasiado `kick_proxy`

Qué pasó:

- `TkSuccess` subía artificialmente
- `shoot/pass` se convertían en mini-dribbles con pseudo-liberaciones

Lección:

- no confundir liberación débil con evidencia fuerte de kick

### 5. Castigar reward terminal varias veces por persistencia de estado

Qué pasó:

- episodios con reward exageradamente negativa
- una sola terminación (`timeout`, `skill_request_cleared`) seguía impactando frames siguientes

Lección:

- el reward debe usar eventos de cierre, no estados persistentes como si fueran eventos repetidos

## Current Pain Points

Los 4 bugs más críticos ahora mismo:

1. Múltiples shots / teletransportación al balón.
   - Patrón antinatural, a menudo en línea recta directa al arco. El agente pasa a teletransportarse al balon lo que lleva que lo dispare multipkes veces
   - Usualmente aparece junto a bucle de `HOLD`.

2. Colisión con obstáculo.
   - En stage 1 sigue ocurriendo.
   - Debe tratarse como falta grave de approach, con alto costo operativo.

3. Handoff demasiado lento en stage 1.
   - El agente tarda mucho en llegar a takeover útil.
   - La combinación balón + obstáculo + banda genera muchos steps muertos.

4. Rotaciones absurdas sobre su eje cuando BHuman manda `dribble`.
   - En algunos casos el balón termina teletransportado al frente del agente y vuelve a aparecer el problema 1.

## Current Diagnostic Hypotheses

Hipótesis más probables, ordenadas:

1. La transición `approach -> PREP_SHOOT/HOLD` todavía entra demasiado pronto en ciertos casos.
2. La interacción entre `BallTo`, `ExecTo` y `motion_walk_target` sigue siendo ruidosa en proximidad al arco y en banda.
3. Stage 1 sigue mezclando demasiada dificultad geométrica a la vez:
   - balón lejos
   - obstáculo
   - banda
   - handoff lento
4. El sesgo semántico de BHuman a `shoot` sigue existiendo en algunos setups, aunque parte del problema es nuestro executor 2D.

## What Changed Recently

Cambios recientes importantes que sí deben preservarse:

- separación explícita entre `handoff_reached` y `TkReady`
- `TkReady` quedó redefinido para significar aceptación real de takeover por BHuman, no solo envelope geométrico
- surface canónica de documentación reducida a 10 MDs
- surface canónica de smokes reducida a 4 entrypoints
- bridge persistente consolidado como baseline
- currículo stages 0-4 ya ajustado:
  - stage 0: `800-1600`
  - stage 1: `1600-3200` con obstáculo estático
  - stage 2-4 con probabilidad creciente de balón dinámico

## Knowledge Base Map

Existen 11 documentos útiles para teoría base, control, contrato y MARL. Clasificación sugerida:

### Master Docs (3)

1. [BHUMAN_RL_CURRICULUM_CONTRACT.md](C:\ALEJO\RL\Documentacion\CONTEXTO%20BHUMAN\BHUMAN_RL_CURRICULUM_CONTRACT.md)
   - contrato principal RL ↔ BHuman
2. [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](C:\ALEJO\RL\Documentacion\CANONICO\PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
   - checklist vivo del proyecto
3. [DOCUMENTATION_AND_SMOKE_GOVERNANCE.md](C:\ALEJO\RL\Documentacion\CANONICO\DOCUMENTATION_AND_SMOKE_GOVERNANCE.md)
   - gobierno documental y superficie oficial

### Specialization Docs (2)

4. [BHUMAN_BRIDGE_AND_PARSER_GUIDE.md](C:\ALEJO\RL\Documentacion\CANONICO\BHUMAN_BRIDGE_AND_PARSER_GUIDE.md)
   - contrato de parseo, bridge, sidecar, BehaviorStatus vs SkillRequest
5. [ENV_REWARD_CURRICULUM_GUIDE.md](C:\ALEJO\RL\Documentacion\CANONICO\ENV_REWARD_CURRICULUM_GUIDE.md)
   - reward, métricas, curriculum y semántica de `handoff_reached / TkReady / TkSuccess`

### Operational / Transfer Docs (6)

6. [README.md](C:\ALEJO\RL\Documentacion\CANONICO\README.md)
7. [BHUMAN_IN_LOOP_COMPACT_GUIDE.md](C:\ALEJO\RL\Documentacion\CANONICO\BHUMAN_IN_LOOP_COMPACT_GUIDE.md)
8. [MAPPO_EXECUTION_AND_STAGE_GUIDE.md](C:\ALEJO\RL\Documentacion\CANONICO\MAPPO_EXECUTION_AND_STAGE_GUIDE.md)
9. [DONT FIX CINEMATICS LIKE THIS WAY.md](C:\ALEJO\RL\Documentacion\CANONICO\DONT%20FIX%20CINEMATICS%20LIKE%20THIS%20WAY.md)
10. [HISTORICAL_DECISIONS_INDEX.md](C:\ALEJO\RL\Documentacion\CANONICO\HISTORICAL_DECISIONS_INDEX.md)
11. [RLBRIDGE_INTERVENTION_LOG.md](C:\ALEJO\RL\Documentacion\PLAN\RLBRIDGE_INTERVENTION_LOG.md)

## Official Smoke Surface

Solo 4 smokes oficiales:

1. [smoke_suite_option1.py](C:\ALEJO\RL\smoke_suite_option1.py)
2. [smoke_core_bridge_semantics.py](C:\ALEJO\RL\smoke_core_bridge_semantics.py)
3. [smoke_takeover_real.py](C:\ALEJO\RL\smoke_takeover_real.py)
4. [smoke_diagnostic_extended.py](C:\ALEJO\RL\smoke_diagnostic_extended.py)

Estado validado:

- `quick`: PASS
- `gating`: PASS
- `diagnostic`: PASS

## Recommended Reading Order for the Next Agent

1. [README.md](C:\ALEJO\RL\Documentacion\CANONICO\README.md)
2. [BHUMAN_RL_CURRICULUM_CONTRACT.md](C:\ALEJO\RL\Documentacion\CANONICO\BHUMAN_RL_CURRICULUM_CONTRACT.md)
3. [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](C:\ALEJO\RL\Documentacion\CANONICO\PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
4. [BHUMAN_BRIDGE_AND_PARSER_GUIDE.md](C:\ALEJO\RL\Documentacion\CANONICO\BHUMAN_BRIDGE_AND_PARSER_GUIDE.md)
5. [ENV_REWARD_CURRICULUM_GUIDE.md](C:\ALEJO\RL\Documentacion\CANONICO\ENV_REWARD_CURRICULUM_GUIDE.md)
6. [DONT FIX CINEMATICS LIKE THIS WAY.md](C:\ALEJO\RL\Documentacion\CANONICO\DONT%20FIX%20CINEMATICS%20LIKE%20THIS%20WAY.md)
7. Revisar `environment.py`, `reward.py`, `train.py`, `wrapper.py`, `bhuman_sidecar.py`, `bhuman_bridge.cpp`

## Final Advice to the Next Agent

- No confundas `TkReady` con `handoff_reached`
- No supongas que `BallTo` o `ExecTo` son verdad pura de BHuman
- No arregles ruido físico imponiendo comportamiento nuevo
- Si un fix mejora visualmente pero empeora `TkReady/TkSuccess`, sospecha que secuestraste la fase de approach
- Si una hipótesis contradice el contrato C++ documentado, el contrato gana
