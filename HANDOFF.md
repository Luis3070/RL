# HANDOFF — Defender v2 (doc-34) validación SimRobot/Linux

**Fecha:** 2026-06-15 · **Branch:** `rl-v3a-multiagent-defender` · **Máquina:** CachyOS (i7-13620H, software-render llvmpipe)
**Twin repo:** `SabanaHerons2026 @ rl-simrobot3d` (bridge v30)

## Por qué este handoff

El commit `5fc3581` ("Defender v2 doc-34 D1–D5") solo validó la **lógica offline en Windows**
(faltaba `pybh`, no se pudo correr SimRobot). Esta sesión cierra esa brecha: corrí el
**pipeline completo doc-34 Pasos 1–9 en SimRobot/Linux** y produje un candidato exportado.

## Qué hice (Pasos 1–9)

| Paso | Qué | Resultado |
|------|-----|-----------|
| 1 | Dataset stage-8 del teacher (2 chunks, ~268k filas crudas) | OK |
| 2 | Merge + oversample defensivo → `dataset_defense_recovery_v2` (299.661 filas) | audit `passed`, obs_size 47, 5 escenarios |
| 3 | Mezcla 70:30 defensa:ataque → `dataset_defense_plus_attack_v2` (429.457 filas) | gate_config verificado |
| 4 | Fine-tune `defender_47_v2` 10k steps (warm-start + KL-anchor desde v2b@7500) | 5 checkpoints (0/2500/5000/7500/10000) |
| 5 | Eval stage-8 por checkpoint | **step2500 mejor** |
| 6 | Floor ofensivo (`behavioral_evaluator`, gate 0.30/0.20) | **4/5 PASS**, collision=0.000 en los 5 |
| 7 | Estabilidad multi-seed (6 seeds: 11 23 37 51 67 83) | 6/6 defensivo · 3/6 estricto (ver nota h) |
| 8 | Export ONNX per-role | `ppo_defender_hsl2026.onnx` (+ `.json`) |
| 9 | Semáforo offline (`offline_team_eval`) | STABILITY 🟢 / ROBUSTNESS 🟢 / DIVERSITY 🟡 |

### Cómo fue el fine-tune (Paso 4)
No es entrenamiento desde cero, es **afinado de un modelo ya competente**:
- **Warm-start** desde `runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt`.
- **KL-anchor** al mismo checkpoint → no se aleja del comportamiento ya validado.
- **BC + PPO** sobre dataset mezclado 70:30, stage 8, 10k steps, eval/checkpoint cada 2500.
- Recompensas de equipo: `TEAM_GOAL_BONUS=2.5`, `TEAM_GOAL_CONCEDED_PENALTY=2.5`.
- Rutas **absolutas** para `--checkpoint`/`--dataset` (el trainer hace `chdir` interno).

## Candidato

**`runs/defender_47_v2/checkpoint_stage8_step2500.pt`** → exportado a `ppo_defender_hsl2026.onnx`.

step2500 es el único bajo el ideal de colisión <0.15, con mejor cover/score. Confirma la
lección v2b ("el bueno es el intermedio"): los de 5000/7500 ya degradan (collision 0.50).

| step | score | cover | collision | wire |
|------|-------|-------|-----------|------|
| **2500** | **0.923** | **0.870** | **0.125** | PASS |
| 5000 | 0.795 | 0.690 | 0.500 | WARN |
| 7500 | 0.786 | 0.676 | 0.500 | WARN |
| 10000 | 0.798 | 0.705 | 0.250 | PASS |

## Rubric a–h (regla de decisión = (a)+(b)+(c)+(d) + recuperación (e))

- **a** Cover defensivo ≥0.60 → **0.870** ✅
- **b** Colisiones con GK propio = 0 → **0.000** ✅
- **c** collision_rate ≤0.25 → **0.125** ✅
- **d** Floor ofensivo ≥2/5 → **4/5 PASS** ✅
- **e** Recuperación visible → **conducir-fuera** dribble 0.292, rescued_ball +9.8 ✅ (⚠️ `policy_stand_rate=0.42` > ideal 0.20)
- **f** Despeje pase-salida → `executed_pass`/`block`=0 (dependiente del fix de gate; doc lo marca **no-bloqueante**) ⚠️
- **g** Goles vs baseline → **NO CORRIDO** (requiere run baseline separado; no es parte de la regla) ⬜
- **h** Estabilidad multi-seed → **6/6 defensivo**, 3/6 estricto ⚠️

**Nota (h):** las seeds que rompen (c) lo hacen *exclusivamente* en `preshoot_bad_angle_boundary`
(escenario ofensivo adversarial, OOD para un defensor). En todos los escenarios defensivos/neutrales
el `collision_rate=0.000` en las 6 seeds.

## Caveats honestos (no bloqueantes según la regla)
1. `policy_stand_rate=0.42` > ideal 0.20 — el defensor se queda quieto más de lo deseado.
2. `executed_pass`/`block` explícitos = 0 (pendientes del fix de gate de pase/despeje).
3. Estabilidad 3/6 si se cuenta el escenario ofensivo OOD `preshoot`.
4. Rubric (g) no corrido — comparación de goles vs trío sin defensor pendiente.

## Cambios en el repo (esta sesión)

**Modificados:**
- `train.py` — `evaluate_agent` tolerante a fallos de reset (reintenta+salta episodio si el
  getup tropieza por software-render flaky, en vez de tumbar el run).
- `tools/run_stability_chunked.sh` — `pkill -x` (exacto) en vez de `pkill -f` (peligroso).

**Nuevos:**
- `tools/run_defv2_finetune.sh` — launcher del Paso 4.
- `tools/run_defv2_floor.sh` — Paso 6 (`behavioral_evaluator` sobre step2500).
- `tools/run_defv2_stability.sh` / `run_defv2_stability_seed67.sh` — Paso 7 multi-seed.
- `tools/run_defv2_chunk.sh` — helper de chunking del dataset.
- `ppo_defender_hsl2026.onnx` + `.json` — artefacto de deploy (obs 47 / 8 skill logits / 4 params).
- `runs/defender_47_v2/VERDICT_defv2.md` — veredicto detallado.
- `.venv_env.sh` — setup de entorno local (paths máquina; **no commitear si es local-only**).

## Reproducir en esta máquina
```bash
source .venv_env.sh                         # PYTHONPATH, LD_LIBRARY_PATH, gate thresholds
bash tools/run_defv2_finetune.sh            # Paso 4 (10k steps)
bash tools/run_defv2_floor.sh               # Paso 6
bash tools/run_defv2_stability.sh runs/defender_47_v2/checkpoint_stage8_step2500.pt
```
Notas runtime: `RL_VISIBLE_RESET_MAX_ATTEMPTS=12`, `RL_VISIBLE_STEP_OBS_TIMEOUT_MS=2500`.
Seed 67 del probe crasheó una vez por freeze de SimRobot (infra, no policy) y se re-corrió
aislado → limpio 0.000.

## Pendientes / próximos pasos
- [ ] Rubric (g): correr baseline (trío v2b@7500 sin defensor, mismos seeds) y comparar goles encajados (~30 min).
- [ ] Fix de gate de pase/despeje para que `executed_pass`/`block` aparezcan en runtime.
- [ ] Reducir `policy_stand_rate` (0.42 → <0.20) en un futuro afinado.
- [ ] Commit/push de los cambios listados arriba (pendiente de decisión sobre `.venv_env.sh`).

— Ver detalle completo en `runs/defender_47_v2/VERDICT_defv2.md`.
