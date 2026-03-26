# trajectory_guidance

Subproyecto para generar trayectorias demostradas (`teacher trajectories`) y exportar snapshots listos para behavioral cloning y fine-tuning mixto.

## Qué hace

1. Usa el mismo `NaoSoccerEnvWrapper` del proyecto principal.
2. Captura el estado real del episodio recién reseteado.
3. Resuelve una trayectoria candidata con A* sobre un modelo simplificado pero **stage-aware**.
4. Garantiza diversidad por:
   - modos/primitive families (`straight`, `left_arc`, `right_arc`, `dribble_safe`, `dribble_aggressive`, `align_then_kick`)
   - `top_k` soluciones distintas
   - selección final con Gumbel-Softmax sobre score
5. Reproduce la trayectoria en el env real, exportando snapshots con la misma semántica de acción del actor.

## Estructura

```text
trajectory_guidance/
├── config/
├── heuristic/
├── solver/
├── evaluate_solver/
├── data/
├── integration/
└── generate_dataset.py
```

## Uso rápido

Desde la raíz del proyecto principal:

```bash
python -m trajectory_guidance.generate_dataset --stage 0 --episodes 50 --out demos_stage0.pkl
python -m trajectory_guidance.generate_dataset --stage 2 --episodes 50 --out demos_stage2.pkl
```

## Notas importantes

- El planner interno es aproximado. El juez final siempre es el rollout real sobre `NaoSoccerEnvWrapper`.
- El snapshot exporta `obs_local`, `state_global`, `teacher_action`, `next_obs_local`, `done`, `event`, `trajectory_id` y `quality_score`.
- Para stages 0–3, `teacher_action[2]` se deja en 0.0 porque `theta_free=False` y el env ya aplica `face-ball`.
