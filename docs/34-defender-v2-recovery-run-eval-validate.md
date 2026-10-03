# 34 — Defender v2 (recovery + clear/pass): correr, evaluar y validar

**Para:** Bryam · **De:** Luis· **Fecha:** 2026-06-13 · **Rama:** `rl-v3a-multiagent-defender`

## Qué cambió respecto al defender de doc-33

El defender de doc-33 (`checkpoint_stage8_step4000.pt`) **solo cubría posición**: cuando
recuperaba el balón no lo despejaba ni lo pasaba (`executed_pass_rate=0`,
`executed_block_rate=0`). Esta tanda añade conducta de balón al recuperar (lógica de
fútbol real: gano el balón → paso a un compañero adelantado o despejo fuera de la zona
de peligro):

- **D1** — control físico del balón promueve al agente a *striker* al instante (salta la
  histéresis de 7 frames). El que recupera juega el balón en vez de seguir bloqueando.
  (`multiagent_teacher.assign_roles`)
- **D2/D3** — striker en nuestro tercio defensivo: **pase seguro adelantado** → si no hay,
  **despeje** (chut up-field, lejos de nuestra portería) → si no, **saca el balón
  conduciendo** hacia mediocampo. (`multiagent_teacher.teacher_striker` +
  `defensive_clear_pass_target` + `defender_clear_armed`)
- **D4** — el dataset hornea `shoot_armed=True` en despejes (igual que el block hornea
  `dribble_armed`), así label/obs/máscara concuerdan (Design-A intacto).
  (`teacher_multiagent_dataset.py`)
- **D5** — escenario nuevo `defense_recovery_clear`: nuestro jugador gana el balón en el
  fondo con una salida adelantada disponible → genera las filas recuperación→despeje/pase
  que los escenarios posicionales nunca producían. (`production_curriculum.py`)

⚠️ **Caveat de runtime (igual que el block):** el *despeje-chut* se hornea en el dataset
y es legal en entrenamiento, pero el gate vivo no arma un chut profundo, así que **en el
robot el despeje-chut no es alcanzable todavía** — la política cae al **conducir-fuera**
(eso sí es alcanzable: `dribble_armed` es real al controlar). El pase de salida **sí** es
alcanzable en runtime. Hacer el despeje-chut alcanzable necesita el fix de gate
compartido (`gate.py` + encoder C++), el mismo pendiente del block. No bloquea esta
validación: validamos pase-de-salida + conducir-fuera como la conducta de despeje.

## Requisitos de entorno

- Pasos de **dataset/entrenamiento/sim**: Linux con `pybh` compilado
  (`PYTHONPATH=/home/luis/GitHub/SabanaHerons2026/Make/Python`), conda `RLNAO`, `DISPLAY`
  activo. UNA sola sesión de sims, máx 4 slots, todo *chunked* (un freeze = un chunk).
- Paso **offline** (semáforo): tu `offline_team_eval.py` sobre el ONNX exportado — corre
  en Windows sin pybh, como hiciste con v2b.
- Contrato de gate en TODO: `RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20`.

---

## Paso 1 — Generar el dataset stage-8 (chunked, sim)

Incluye automáticamente el escenario nuevo `defense_recovery_clear` (ya está en
`team_defense_scenarios` y en `DEFAULT_STAGE8_SCENARIO_WEIGHTS`).

```bash
export PYTHONPATH=/home/luis/GitHub/RL:/home/luis/GitHub/SabanaHerons2026/Make/Python
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
export RL_MA_TEACHER_COLLISION_SUSTAIN_STEPS=16

for i in 1 2 3; do
  RL_MA_TEACHER_STAGE=8 RL_MA_TEACHER_MAX_STEPS=600 \
  RL_MA_TEACHER_EPISODES=8 RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=6 \
  RL_MA_TEACHER_SEED_BASE=$((970000 + i*10000)) \
  PYBH_RL_SHM_NAME=/pybh_rl_defv2_chunk_${i} \
  python teacher_multiagent_dataset.py
  pkill -f SimRobot; pkill -f Xvfb; sleep 5
done
```

## Paso 2 — Merge + validación del corpus defensivo

`build_stage8_defense_dataset.py` junta los chunks (incluye parciales) y exige fracción
mínima de block/transición:

```bash
python build_stage8_defense_dataset.py \
  --root runs/teacher_multiagent_dataset \
  --output-dir runs/teacher_multiagent_dataset/dataset_defense_recovery_v2 \
  --min-block-frac 0.20
```

Audit defensivo (shoot/pass del striker NO aplican como mínimos; lo que importa es
liveness + block + recuperación):

```bash
python teacher_multiagent_dataset_audit.py \
  runs/teacher_multiagent_dataset/dataset_defense_recovery_v2 \
  --min-striker-shoot-frac 0 --min-striker-pass-armed 0 --write
```

**Qué mirar en el audit:** liveness verde, `role-mask bad rows = 0`,
`defender_block_frac_positive=true`, y — nuevo en esta tanda — que aparezcan filas de
**pase/clear del striker en fase defensiva** (las del escenario recovery). Si
`executed pass/clear` en recovery sale 0, los spawns no están generando recuperación —
revisar que el balón cae dentro de `RL_MA_BALL_CONTROL_RADIUS_MM` (350mm) del recuperador.

## Paso 3 — Mezcla 70:30 defensa:ataque (anti-olvido)

El corpus de ataque completo aplastaría la defensa; subsamplear ataque por **episodios
enteros** (no filas sueltas) al ~30%:

```bash
python subsample_multiagent_attack_dataset.py \
  runs/teacher_multiagent_dataset/dataset_multiagent_train_20260611_full_v1 \
  --output-dir runs/teacher_multiagent_dataset/dataset_attack_subsample_30pct \
  --defense-rows $(python - <<'PY'
import numpy as np; print(len(np.load("runs/teacher_multiagent_dataset/dataset_defense_recovery_v2/samples.npz")["weights"]))
PY
) --attack-fraction-of-defense 0.43

python merge_multiagent_teacher_datasets.py \
  runs/teacher_multiagent_dataset/dataset_defense_recovery_v2 \
  runs/teacher_multiagent_dataset/dataset_attack_subsample_30pct \
  --output-dir runs/teacher_multiagent_dataset/dataset_defense_plus_attack_v2
```
(El merge verifica `gate_config` — si no coincide, falla, que es lo correcto.)

## Paso 4 — Fine-tune del defender (sim, chunked si hace falta)

```bash
RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20 \
RL_MA_TEAM_GOAL_BONUS=2.5 RL_MA_TEAM_GOAL_CONCEDED_PENALTY=2.5 \
RL_MA_PROD_TRAIN_STEPS=10000 RL_MA_PROD_MAX_EPISODE_STEPS=600 \
RL_MA_PROD_EVAL_INTERVAL=2500 RL_MA_PROD_CHECKPOINT_INTERVAL=2500 \
RL_MA_PROD_SCENARIO_SET=defense RL_MA_PROD_EXPERIMENT_NAME=defender_47_v2 \
python train_multiagent_production.py --stage-id 8 \
  --checkpoint runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt \
  --dataset runs/teacher_multiagent_dataset/dataset_defense_plus_attack_v2/samples.npz
```
- `--checkpoint` = warm-start **y** KL anchor (v2b@7500 — mantiene el ataque vivo).
- Checkpoint cada 2500; **evaluar TODOS**, el bueno puede ser el de en medio
  (lección v2b: salió del 7500, no del último).

## Paso 5 — Evaluar (lee el print `[eval]` stage 8 + telemetría)

El trainer ya imprime, para stage 8: `defender_score`, `cover`, `danger_cover`,
`corridor_mm`, `collision`, y **ahora** `block` y `pass` ejecutados. Telemetría en
`runs/defender_47_v2/telemetry/.../csv/eval_summary.csv`.

## Paso 6 — Floor ofensivo (no debe romperse)

```bash
RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20 \
python behavioral_evaluator.py \
  --checkpoint runs/defender_47_v2/checkpoint_stage8_step<MEJOR>.pt --episodes 8
```

## Paso 7 — Estabilidad multi-seed (sim, chunked)

```bash
tools/run/run_stability_chunked.sh \
  runs/defender_47_v2/checkpoint_stage8_step<MEJOR>.pt 11 23 37 51 67 83
```

## Paso 8 — Exportar ONNX del defender (familia per-role)

```bash
python tools/export/export_ppo_policy_team.py \
  --checkpoint runs/defender_47_v2/checkpoint_stage8_step<MEJOR>.pt \
  --output models/defender/ppo_defender_hsl2026.onnx \
  --active-skills stand,walk,dribble,shoot,pass,block
```

## Paso 9 — Semáforo offline (tu harness, Windows)

Re-apunta tu `offline_team_eval.py` al nuevo `ppo_defender_hsl2026.onnx` + el corpus
`dataset_defense_plus_attack_v2`. Esperado: 🟢 estabilidad (0% NaN/ilegales,
determinista), 🟢 robustez al ruido. **Nuevo eje a reportar:** distribución de skills del
striker EN FASE DEFENSIVA — debe contener `pass` (salida) y, si el gate lo hornea, `shoot`
(despeje); `block` en el rol off_ball.

---

## Rubric de validación del defender v2 (todas)

| # | Check | Bar |
|---|---|---|
| a | Cover: defender en el cono balón→postes propios en fase defensiva | ≥ 0.60 |
| b | Colisiones con el GK propio | 0 |
| c | `collision_rate` por checkpoint | ≤ 0.25 (ideal < 0.15) |
| d | Floor ofensivo (`behavioral_evaluator`, gate 0.30/0.20) | ≥ 2/5 (diagnóstico: presser vivo) |
| e | **NUEVO — recuperación**: en el escenario recovery, al ganar el balón el agente ejecuta `pass` o conduce-fuera (no se queda en block/stand) | `executed_pass_rate` o conducir-fuera > 0; `policy_stand_rate` deep < 0.20 |
| f | **NUEVO — despeje**: pase de salida hacia compañero adelantado cuando hay lane libre | aparece en eval/telemetría |
| g | Goles encajados en `defense_3v3_full` vs baseline (trío v2b@7500 sin defender), mismos seeds | estrictamente menor |
| h | Estabilidad multi-seed (paso 7) | ≥ 4/6 seeds sostienen (a)+(c) |

**Regla de decisión:** (a)+(b)+(c)+(d) + recuperación visible (e) ⇒ candidato del defender;
exportar y meter a la familia per-role junto a `ppo_team_hsl2026.onnx`. El despeje-chut en
runtime queda pendiente del fix de gate (no bloquea: el conducir-fuera cubre la seguridad).
