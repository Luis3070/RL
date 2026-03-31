# BHuman In Loop: Guia Compacta

Estado de referencia: 2026-03-30

Objetivo:

- reducir la cantidad de smokes operativos a pocos entrypoints
- reducir la cantidad de documentos que hay que leer para operar el proyecto
- mantener los archivos detallados como referencia, sin perder trazabilidad

## 1. Lectura minima recomendada

Si alguien entra nuevo al proyecto, el orden corto recomendado es:

1. [MAPPO_Plan_3Fases.md](/j:/ROBOTICA/RL/Documentacion/PLAN/MAPPO_Plan_3Fases.md)
   Para entender el scope del proyecto y las fases del transplante.

2. [MAPPO_Auditoria.md](/j:/ROBOTICA/RL/Documentacion/PLAN/MAPPO_Auditoria.md)
   Para entender restricciones, reglas de validacion y auditoria.

3. [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](/j:/ROBOTICA/RL/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
   Para ver el estado real actual del loop BHuman-in-loop.

4. [BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md)
   Para saber exactamente que define la opcion 1 actual y que quedaria para la opcion 2.

Con esos 4 documentos ya se puede operar el proyecto sin navegar toda la carpeta `PLAN`.

## 2. Smokes canonicos

El entrypoint recomendado ya no debe ser correr muchos archivos sueltos.

Usar:

```powershell
J:/CONDA/envs/RLRobocup/python.exe j:/ROBOTICA/RL/smoke_suite_option1.py --suite gating
```

Suites disponibles:

- `quick`
  Cubre transporte persistente, integracion env-bridge y semantica fuerte.

- `gating`
  Es la suite recomendada antes de entrenamiento corto o mediano.
  Incluye:
  - transporte persistente
  - environment bridge
  - semantic strong
  - sidecar
  - completion real de takeover

- `extended`
  Agrega cobertura semantica y de alineacion adicional.

- `diagnostic`
  Solo auditorias no bloqueantes, como `MotionInfo`.

- `all`
  `extended + diagnostic`

## 3. Como interpretar los archivos de smoke actuales

Los archivos viejos no se borran, pero dejan de ser entrypoints principales.

### 3.1. Gating principal

- [smoke_test_rlbridge_persistent_basic.py](/j:/ROBOTICA/RL/smoke_test_rlbridge_persistent_basic.py)
- [smoke_test_environment_bridge.py](/j:/ROBOTICA/RL/smoke_test_environment_bridge.py)
- [smoke_test_semantic_strong.py](/j:/ROBOTICA/RL/smoke_test_semantic_strong.py)
- [smoke_test_sidecar.py](/j:/ROBOTICA/RL/smoke_test_sidecar.py)
- [smoke_test_real_takeover_completion.py](/j:/ROBOTICA/RL/smoke_test_real_takeover_completion.py)

### 3.2. Cobertura extendida

- [smoke_test_real_takeover_playball.py](/j:/ROBOTICA/RL/smoke_test_real_takeover_playball.py)
- [smoke_test_real_env_semantic_variants.py](/j:/ROBOTICA/RL/smoke_test_real_env_semantic_variants.py)

### 3.3. Diagnostico

- [audit_motioninfo_takeover_evidence.py](/j:/ROBOTICA/RL/audit_motioninfo_takeover_evidence.py)

### 3.4. Soporte / fixture especifica

- [smoke_test_real_takeover_env_positive.py](/j:/ROBOTICA/RL/smoke_test_real_takeover_env_positive.py)

Regla operativa:

- usar `smoke_suite_option1.py` para validacion normal
- correr archivos individuales solo cuando estes depurando un hueco puntual

## 4. Documentacion canonica vs referencia

### 4.1. Canonica

Estos son los documentos que deben mantenerse pequeños y actualizados:

- [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](/j:/ROBOTICA/RL/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
- [BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md)
- [BHUMAN_IN_LOOP_COMPACT_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_COMPACT_GUIDE.md)

### 4.2. Referencia detallada

Leer solo si hace falta profundidad en un tema concreto:

- [BHUMAN_PERSISTENT_BRIDGE_PLAN.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_PERSISTENT_BRIDGE_PLAN.md)
- [BHUMAN_SEMANTIC_GO_NO_GO.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_SEMANTIC_GO_NO_GO.md)
- [BHUMAN_IN_LOOP_SIDECAR_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_SIDECAR_GUIDE.md)
- [BHUMAN_IMPLEMENTATION_CORRECTION_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_IMPLEMENTATION_CORRECTION_GUIDE.md)
- [BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md)
- [ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md)
- [RLBRIDGE_INTERVENTION_LOG.md](/j:/ROBOTICA/RL/Documentacion/PLAN/RLBRIDGE_INTERVENTION_LOG.md)
- [BHUMANPARSER_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMANPARSER_GUIDE.md)
- [PROJECT_AUDIT_FOR_RL.md](/j:/ROBOTICA/RL/Documentacion/PLAN/PROJECT_AUDIT_FOR_RL.md)

## 5. Criterio de compresion recomendado

Para no volver a inflar el proyecto:

- no crear un smoke nuevo si la cobertura cabe claramente en una suite existente
- no crear un `.md` nuevo si el cambio cabe en checklist, guia compacta o change guide
- cuando un smoke nuevo nazca para depurar una fixture puntual, marcarlo como diagnostico o soporte
- cuando un documento nuevo nazca por una fase futura, enlazarlo desde esta guia compacta

## 6. Estado actual del baseline

El baseline operativo actual es:

- `bridge_mode = persistent_2d_cognition`
- `scenario = 2D`
- `location = 4v4_Scaled`
- transporte real `pipe_process`
- backend real dentro del env
- takeover real con efecto de fisica
- `MotionInfo` de `shoot/pass` aun NO es criterio de cierre de opcion 1

Ese baseline se valida con:

```powershell
J:/CONDA/envs/RLRobocup/python.exe j:/ROBOTICA/RL/smoke_suite_option1.py --suite gating
```
