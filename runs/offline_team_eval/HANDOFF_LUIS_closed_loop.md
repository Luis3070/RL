# Handoff a Luis — confirmación closed-loop del V2 (v2b@7500)

**De:** Bryam (evaluación) · **Para:** Luis (sim en Linux) · **Fecha:** 2026-06-12

## Contexto
Evalué el multiagente **V2 / v2b@7500** offline sobre el ONNX de deploy
(`ppo_team_hsl2026.onnx`, 355k obs reales del corpus). Veredicto de los 3 ejes que
pidió el equipo: **🟢 estabilidad, 🟢 robustez, 🟢 diversidad** (Global VERDE).
Detalle en `runs/offline_team_eval/VERDICTO_v2b.md`.

Lo único que el offline **no puede medir** son las métricas de física en movimiento.
Te toca confirmarlas en tu Linux (donde `pybh` ya compila — en Windows el puente RL
no corre porque `RLSharedState` usa IPC POSIX, no portado).

## Lo que necesito que midas en sim (SimRobot 3D)
Checkpoint: `runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt`

| Métrica | Umbral verde | Por qué |
|---|---|---|
| **collision_rate** | ≤ 0.25 (idealmente < 0.15) | Tu sospecha #1 ("colisiones altas"). Es el riesgo principal para el físico. |
| **conversión a gol** (shoot_finish / team_goal) | ~0.25 o mejor | Confirmar que el ~25% se sostiene en cerrado. |
| **support_open_hold_rate / support_triangle_hold_rate** | ≥ ~0.6 | Resuelve el único 🟡 offline: los supports prefieren caminar a quedarse quietos; hay que ver si HOLDean o derivan. Los publica `team_parallel_environment.py` vía `LAST_TEAM_METRICS`. |

## Comando sugerido
```bash
# stability probe multi-seed (ajusta a tu harness de team si aplica):
PYTHONPATH=/home/luis/GitHub/SabanaHerons2026/Make/Python \
  tools/run_stability_chunked.sh \
  runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt
```
(`run_stability_chunked.sh` corre una sim fresca por seed; revisa `collision_mean` /
`collision_max` en el `[stability-summary]`. Para hold-rate de supports usa el path
del team env / `LAST_TEAM_METRICS`.)

## Decisión
- Si **collisions ≤ 0.25** y conversión + hold-rate ok → **🟢 luz verde para trasplante**.
- Si collisions altas → 🟡, ajustable (es el punto que ya veníamos viendo).
- Offline ya está verde, así que esto es **confirmación**, no evaluación desde cero.
```
