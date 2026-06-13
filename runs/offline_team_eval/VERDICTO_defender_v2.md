# Veredicto — Defender v2 (doc-34, D1–D5)

**Rama:** `rl-v3a-multiagent-defender` · **Fecha:** 2026-06-13 · **Por:** Bryam (Claude)
**Alcance:** lo verificable en Windows. El semáforo del **modelo** (estabilidad/robustez/
diversidad) necesita el defender entrenado, que aún **no existe** → ver "Qué falta".

---

## TL;DR

🟢 **VERDE en lo que se puede validar sin el simulador.** La lógica nueva del *teacher*
defensivo de Rafa (D1–D5) está **correcta y bien cableada**: 21/21 checks en verde. No hay
amarillo ni rojo en el código.

⚠️ **El semáforo del modelo todavía no se puede dar.** El defender **no está entrenado**:
no hay dataset defensivo, ni checkpoint `defender_47_v2`, ni `ppo_defender_hsl2026.onnx`.
Generarlos (doc-34 Pasos 1–8) corre el *teacher* dentro de SimRobot y entrena con eval en
sim → **requiere Linux + pybh** (mismo bloqueo POSIX que con v2b; este Windows no puede
correr el loop vivo). Por eso esta tanda valida el **código que produce** al defender, no
al defender ya entrenado.

---

## Qué se validó en Windows (🟢 21/21)

Harness reproducible: **`tools/validate_defender_logic.py`**
(`.venv/Scripts/python.exe tools/validate_defender_logic.py`, exit 0). Stubbea las dos
barreras de carga de `pybh_backend` (`os.chdir` + `import pybh.controller`) y ejecuta la
lógica pura sobre observaciones de recuperación sintéticas — caza bugs de constante/signo/
campo **antes** de que Luis queme horas de sim.

| Bloque | Qué se comprobó | Resultado |
|---|---|---|
| **Constantes** | `BALL_CONTROL_RADIUS_MM=350`, `DEFENSIVE_CLEAR_X_MM=-1500` (tercio propio), `CLEAR_PASS_MIN_UPFIELD_MM=600`, EMA bajada a 0.15/0.10 (P5) | 🟢 |
| **D1** | control físico del balón promueve al recuperador a *striker* saltándose la histéresis; **sin** control, la histéresis mantiene al striker previo | 🟢 |
| **D2** | `defensive_clear_pass_target` elige el outlet más adelantado no-GK; nunca pasa al GK; lane bloqueado → sin pase; sin compañero adelantado → sin pase | 🟢 |
| **D3** | `defender_clear_armed` arma despeje **solo** si profundo + central + sin outlet; si hay pase seguro lo prefiere; no arma si poco profundo o abierto | 🟢 |
| **D2/D3 routing** | `teacher_striker` enruta recuperación → **pase** (con outlet+armado) / **despeje-shoot** (sin outlet, central, armado) / **conducir-fuera** (resto), y NO secuestra al striker en tercio de ataque | 🟢 |
| **D5** | `_defense_recovery_world` genera, en 200 semillas, balón en tercio propio con el recuperador dentro del radio de control; escenario `defense_recovery_clear` registrado | 🟢 |

**Design-A intacto:** D4 hornea `shoot_armed=True` en el despeje (espejo del block con
`dribble_armed`), así label/obs/máscara concuerdan. Verificado por lectura del diff
(`teacher_multiagent_dataset.py`): el baking ocurre **antes** de `build_team_gate_decision`,
de modo que el obs almacenado y el `gate_decision` que ve el teacher coinciden.

---

## Caveat de runtime (ya señalado en doc-34, no es defecto)

El **despeje-chut** se hornea en el dataset y es legal en entrenamiento, pero el gate vivo
no arma un chut profundo todavía → en el robot real el despeje cae al **conducir-fuera**
(eso sí alcanzable: `dribble_armed` es real al controlar). El **pase de salida sí** es
alcanzable. Hacerlo alcanzable necesita el fix de gate compartido (`gate.py` + encoder C++),
el mismo pendiente del block. **No bloquea**: conducir-fuera cubre la seguridad.

---

## Qué falta (necesita la Linux de Luis) — handoff

El semáforo del modelo que pidió el equipo (estabilidad/robustez/diversidad, como en v2b)
sale al final de doc-34. Orden exacto en **`docs/34-defender-v2-recovery-run-eval-validate.md`**:

1. **Pasos 1–3** (sim): generar dataset `defense_recovery_v2` (incluye el escenario nuevo),
   auditar, y mezclar 70:30 defensa:ataque (anti-olvido).
2. **Paso 4** (sim): fine-tune warm-start + KL-anchor de **v2b@7500**; checkpoint cada 2500,
   **evaluar TODOS** (lección v2b: el bueno salió del 7500, no del último).
3. **Pasos 5–7** (sim): eval stage-8 (`defender_score`/`cover`/`collision`/`block`/`pass`),
   floor ofensivo (`behavioral_evaluator ≥ 2/5`), estabilidad multi-seed (≥ 4/6).
4. **Paso 8**: exportar `ppo_defender_hsl2026.onnx`.
5. **Paso 9 (Windows, push-button):** apuntar **`tools/offline_team_eval.py`** (ya recuperado
   en esta rama) al `ppo_defender_hsl2026.onnx` + corpus `dataset_defense_plus_attack_v2`.
   Esperado 🟢 estabilidad/robustez; eje nuevo a reportar: skills del striker EN FASE
   DEFENSIVA debe contener `pass` (salida) y, si el gate lo hornea, `shoot` (despeje);
   `block` en off_ball.

**Regla de decisión (doc-34):** (a) cover ≥ 0.60 + (b) 0 colisiones con GK propio +
(c) `collision_rate ≤ 0.25` + (d) floor ofensivo ≥ 2/5 + recuperación visible (e) ⇒
candidato del defender → exportar y meter a la familia per-role junto a `ppo_team_hsl2026.onnx`.

---

## Archivos de esta entrega (en la rama)

- `tools/validate_defender_logic.py` — harness de validación Windows (lo de arriba).
- `tools/offline_team_eval.py` — semáforo offline para el Paso 9 (recuperado de v2b).
- `runs/offline_team_eval/VERDICTO_defender_v2.md` — este documento.
