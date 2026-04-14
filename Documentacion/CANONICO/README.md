# RL + BHuman In Loop

Punto de entrada corto para agentes y personas nuevas.

Si vas a leer el proyecto por primera vez, empieza aqui:

1. [BHUMAN_RL_CURRICULUM_CONTRACT.md](file:///c:/ALEJO/RL/Documentacion/CANONICO/BHUMAN_RL_CURRICULUM_CONTRACT.md)
2. [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](file:///c:/ALEJO/RL/Documentacion/CANONICO/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
3. [BHUMAN_IN_LOOP_COMPACT_GUIDE.md](file:///c:/ALEJO/RL/Documentacion/CANONICO/BHUMAN_IN_LOOP_COMPACT_GUIDE.md)
4. [DOCUMENTATION_AND_SMOKE_GOVERNANCE.md](file:///c:/ALEJO/RL/Documentacion/CANONICO/DOCUMENTATION_AND_SMOKE_GOVERNANCE.md)
5. [parser_mlops_audit.md](file:///C:/Users/ingro/.gemini/antigravity/brain/4551e3e4-c122-413c-a499-b7e438dcd9dd/parser_mlops_audit.md)

## Reglas de lectura

- la superficie canónica del proyecto es de `10 MDs` y `4 smokes`
- los archivos bajo `Documentacion/REFERENCIA/` son lectura profunda preservada
- los archivos bajo `Documentacion/HISTORICO/` son historial y postmortems
- los archivos bajo `smoke_legacy/` preservan fixtures y pruebas viejas útiles, pero no son la superficie operativa mínima

## Smokes oficiales

1. [smoke_suite_option1.py](/c:/ALEJO/RL/smoke_suite_option1.py)
2. [smoke_core_bridge_semantics.py](/c:/ALEJO/RL/smoke_core_bridge_semantics.py)
3. [smoke_takeover_real.py](/c:/ALEJO/RL/smoke_takeover_real.py)
4. [smoke_diagnostic_extended.py](/c:/ALEJO/RL/smoke_diagnostic_extended.py)

Entry point recomendado:

```powershell
C:\Users\ingro\anaconda3\envs\RLRobocup\python.exe C:\ALEJO\RL\smoke_suite_option1.py --suite gating
```

## Semántica mínima que no debe confundirse

- `handoff_reached`: el RL llegó a la condición geométrica de oferta
- `TkReady`: BHuman aceptó takeover de forma operacionalmente real
- `playball_entry`: BHuman entró a `playBall_*`
- `TkSuccess`: takeover terminó con evidencia suficiente de ejecución/progreso

## Regla de oro

Si un documento o smoke contradice:

- el contrato
- el checklist
- la guía compacta
- o la gobernanza

entonces no debe tratarse como baseline vigente.
