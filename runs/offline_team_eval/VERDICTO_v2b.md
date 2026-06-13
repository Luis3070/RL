# Veredicto multiagente V2 (v2b@7500) — Semáforo

**Modelo:** `runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt`
**Artefacto evaluado:** `ppo_team_hsl2026.onnx` (el mismo ONNX que se trasplanta a C++)
**Corpus:** 355 508 observaciones reales de 47-dim del teacher multiagente (con etiquetas, role_ids, estado del gate)
**Harness:** `tools/offline_team_eval.py` (reproducible)
**Fecha:** 2026-06-12

---

## 🟢 VEREDICTO GLOBAL: VERDE (con 1 amarillo a confirmar en sim)

| Eje | Luz | Resumen |
|-----|-----|---------|
| **Estabilidad** | 🟢 VERDE | 0% NaN/Inf, 0% decisiones ilegales (enmascarado), determinista, jitter 0.75% |
| **Robustez** | 🟢 VERDE | Acuerdo 96.9% bajo ruido σ0.2, sano con teammates faltantes/OOD, pase robusto 99.4% |
| **Diversidad** | 🟢 VERDE | Striker fiel al teacher 96.1%, supports disciplinados, sin colapso de modo |

---

## Qué cubre esto (nivel función-de-política, sobre el ONNX de deploy)

Reproduce **bit-fiel** el camino de decisión que corre en C++:
- la máscara forbid-invalid de `train.gate_skill_mask_from_obs` (cols 23,24,41,42,44,45,46), y
- acción determinista = `argmax` sobre logits de skill enmascarados.

### 1. Estabilidad 🟢
- `nan_inf_rate = 0.0000` — ninguna observación produce salida no finita.
- `masked_choice_illegal_rate = 0.0000` — la decisión **desplegada** nunca cae en un skill prohibido por el gate. Design-A se sostiene end-to-end.
- `determinism_ok = True` — salida bit-idéntica entre corridas.
- `jitter_skill_flip_rate@σ0.05 = 0.0075` — sólo 0.75% de las decisiones cambian ante ruido pequeño.
- *(info)* `raw_head_wants_forbidden = 0.972` — **NO es defecto**: en MAPPO forbid-invalid la máscara se aplica al muestrear y al actualizar, así que la cabeza sin enmascarar nunca se entrena para respetar el gate. Lo que corre es la salida enmascarada, 100% legal.

### 2. Robustez 🟢
- Curva de ruido (acuerdo de skill vs limpio): σ0.05→0.997, σ0.1→0.99, **σ0.2→0.969**, σ0.5→0.86. Degradación suave, sin NaN.
- `missing_teammates_finite_rate = 1.0` — con `obs[26:38]` en cero (sin info de equipo) sigue finito y legal.
- **`pass_recall_noisy@σ0.1 = 0.994`** — la decisión de pase sobrevive al ruido: el "100% de pases cuando la posición lo permite" no es frágil.
- `ood_tail_finite_rate = 1.0` — sano en el 1% de observaciones más extremas.

### 3. Diversidad 🟢
- `top1_skill_rate = 0.804` (walk), `entropy = 0.727 nats` — repertorio real, sin colapso de modo.
- **Striker `fidelity = 0.961`** vs teacher, distribución casi idéntica (walk .70, dribble .15, pass .088, shoot .029).
- **`pass_recall(armed) = 0.996`** — cuando el gate habilita pase al striker, pasa el 99.6% de las veces. ✅ confirma el claim del equipo.
- `support_shoot_or_pass_rate = 0.0000` — los supports nunca disparan/pasan (disciplina de rol correcta).

---

## 🟡 Único amarillo: posicionamiento de supports (confirmar en SIM vivo)

- Fidelidad de supports vs teacher: open `0.385`, off_ball `0.423`.
- **Causa**: NO es violación de gate. El teacher etiqueta supports como `stand .66 / walk .34`; la política hace `stand .14 / walk .86`. Es un **sesgo a caminar en vez de quedarse quieto** (ambos legales).
- **Por qué importa**: offline no puede ver si el support *mantiene la posición* (hold) o *deriva*. Eso lo decide la cabeza de residuales + la dinámica cerrada.
- **Cómo cerrarlo**: medir `support_open_hold_rate` y `support_triangle_hold_rate` (los publica `team_parallel_environment.py` vía `LAST_TEAM_METRICS`) en el loop vivo. Si hold≥~0.6, el amarillo pasa a verde.

---

## ❗ Lo que este veredicto NO cubre (necesita SimRobot 3D vivo / pybh)

Métricas de loop cerrado, imposibles offline:
- **Tasa de colisiones** ← tu sospecha de "rango de colisiones muy alto". Es exactamente lo #1 a medir en sim.
- **Conversión a gol** (el ~25% que mencionaste).
- **Completitud de pase en movimiento** (offline sólo confirma la *decisión* de pasar, no que el balón llegue).
- **Hold-rate real de supports** (el amarillo de arriba).

**Estado del entorno en esta máquina (Windows):** se creó `.venv` con torch/skrl/onnxruntime y corre el ONNX perfecto. Pero `pybh` (extensión C++) **no está compilado** y `cl` (MSVC) no está en el PATH, así que el loop vivo de SimRobot no se puede lanzar aquí sin un build de Visual Studio. El build de Windows de B-Human sí existe (`Build/Windows/CMake/Develop`).

---

## Cómo reproducir / siguiente paso

```bash
# Offline (esta máquina, ya corre):
.venv/Scripts/python.exe tools/offline_team_eval.py

# Loop cerrado (Linux de Luis, donde pybh ya está compilado):
PYTHONPATH=/home/luis/GitHub/SabanaHerons2026/Make/Python \
  tools/run_stability_chunked.sh runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt
```

**Recomendación para el trasplante:** los ejes de función-de-política están **VERDES** — el modelo es estable, robusto y diverso a nivel de decisión, y el pase es sólido. El único pendiente antes de físico es confirmar en sim vivo: (1) colisiones, (2) conversión, (3) hold-rate de supports. Si el sim confirma colisiones aceptables, **luz verde para trasplante**.
