# HANDOFF → Trasplante del defensor a B-Human

Resumen para el chat que va a meter el policy entrenado dentro de **SabanaHerons2026 (C++/SimRobot)**.
Léelo completo antes de tocar el bridge. Lo escribió el chat que entrenó/validó el defensor.

---

## 0. TL;DR

- Artefacto a desplegar: **`ppo_defender_hsl2026.onnx`** (+ manifest `ppo_defender_hsl2026.json`), en la raíz del repo `RL/`.
- Es un actor PPO multi-agente **role-conditioned**, entrenado como defensor (doc-34 Defender v2).
- Checkpoint origen: `runs/defender_47_v2/checkpoint_stage8_step2500.pt` (el intermedio, el bueno).
- Twin repo debe estar en branch **`rl-simrobot3d` (bridge v30)**.
- ⚠️ **LO MÁS IMPORTANTE:** el ONNX emite **logits CRUDOS sin máscara**. El gate (Design-A) se
  aplica *fuera* de la red, en Python. Si lo trasplantas a C++ **tienes que reproducir la máscara
  del gate tú mismo antes del argmax**, o el robot elegirá skills ilegales. Detalle en §3.

---

## 1. Contrato de I/O del ONNX (verificado)

| | |
|---|---|
| Input | `obs` shape `[batch, 47]` float32 |
| Output 1 | `skill_logits` shape `[batch, 8]` (CRUDOS, sin máscara) |
| Output 2 | `param_mean` shape `[batch, 4]`, ya pasado por `tanh` → rango **[-1, 1]** |
| Inferencia | **determinista**: `argmax(skill_logits_enmascarados)` para skill; `param_mean` directo para params (sin samplear) |
| Normalización de obs | **ninguna** — la obs se alimenta cruda. No hay capa de running mean/std. |
| Hashes | `onnx_sha256=9f94f2…e331e`, `checkpoint_sha256=bac93e8…5746` |

`skill_order` (los 8 logits, en este orden exacto):
```
0 stand   1 walk   2 shoot   3 pass   4 dribble   5 block   6 mark   7 observe
```
`active_skills` (los que el entrenamiento dejó vivos): `stand, walk, dribble, shoot, pass, block`.
→ `mark` y `observe` nunca se activan; quedan enmascarados siempre por el stage mask.

Los 4 `param_mean` (tanh, [-1,1]) son: **`[target_x, target_y, target_theta, pass_target]`**.
Hay que **denormalizarlos** a coordenadas de campo / heading antes de mandarlos como `SkillRequest`.
La lógica de decode vive en `action.py` (`StructuredSkillAction.from_numpy` + `action_param_mask`).
`pass_target` es discreto (índice de compañero) aunque salga como float → redondear.

`param_mask` por skill: cada skill solo usa algunos params (p.ej. `stand`/`observe` ninguno).
Ver `action.py:action_param_mask(skill_name)` — replicar esa máscara para no mandar basura.

---

## 2. Layout de la observación de 47 dims (lo que C++ DEBE construir)

El bridge C++ (`RLSkillProvider.cpp` / `RLSharedState`) ya construye la obs que `pybh.rl_get_obs`
devuelve. **El orden y semántica deben ser EXACTAMENTE estos** (de `observation.py:ObsIndex`):

```
[0:3]   robot_x, robot_y, robot_theta
[3:13]  ball: rel pos, predicted end pos, velocity, time_since_seen, seen%, trust flags
[13:17] scoring: can_score_now, shot_quality_no_obs, shot_opening_with_obs, pass_options_count
[17:23] proximidad: nearest teammate/opponent/uncertain dist (front y all-dir)
[23:26] GATE bits (Design-A): shoot_armed, dribble_armed, shoot_arm_progress
[26:32] teammates rel: nearest_xy, second_xy, third_xy
[32:35] global_team_ball_x_rel, _y_rel, time_since_team_ball_update
[35:39] engaging flags: nearest/second/third_teammate_engaging_ball, team_ball_agrees_with_mine
[39:41] opponent_goal_x_rel, opponent_goal_y_rel
[41:44] GATE bits extendidos: pass_armed, observe_armed, pass_arm_progress
[44:47] ROLE one-hot: is_striker(44), is_open_support(45), is_off_ball_support(46)
```

Notas críticas:
- **Role one-hot [44:46]:** exactamente UNO en 1.0 por agente RL cada step (lo asigna el env con
  `assign_roles` + histéresis). El portero (GK) lleva los tres en 0. Para un **defensor**, el rol
  efectivo lo decide la lógica de asignación — hay que portar esa asignación o fijarla según el plan
  de despliegue. **Si los tres van en 0, el role-conditioning se cae** y el comportamiento no será
  el validado.
- **GATE bits [23:26] y [41:43]:** los pone el gate de B-Human/Python (shoot/dribble/pass armed).
  No son "lo que el robot quiere hacer", son "qué skills están armadas/permitidas ahora". El bridge
  v30 ya los calcula; verifica que sigan poblándose.
- Dims [41:43] y [44:46] eran cero en el checkpoint base de 26-dim; el warm-start hizo zero-pad
  26→47, por eso por defecto valen 0 y eso preserva comportamiento.

---

## 3. ⚠️ El gate NO está dentro del ONNX — hay que reproducirlo

En Python, el flujo real de inferencia es:

```
skill_logits_crudos  (del ONNX / red)
   → _masked_skill_logits():
        combined = active_skill_mask (stage)  AND  gate_skill_mask_from_obs(obs)
        skill_logits.masked_fill(combined <= 0, -1e9)
   → argmax  → skill elegido
```

`gate_skill_mask_from_obs()` (en `train.py`) es la **única fuente de verdad** de la máscara
(invariante Design-A). Deriva la máscara puramente de:
- `obs[23:26]` (shoot_armed, dribble_armed, shoot_arm_progress)
- `obs[41:43]` (pass_armed, observe_armed, pass_arm_progress)
- `obs[44:46]` (role one-hot → máscara condicionada por rol, §11.A.4 del doc)

Y `active_skill_mask` = los `active_skills` del stage (stand/walk/dribble/shoot/pass/block).

**Para el trasplante a C++:** antes del argmax debes poner a `-inf` (o muy negativo) los logits de
toda skill no permitida por (stage AND gate). Si el robot empieza a elegir `mark`/`observe` o a
shootear sin estar armado, es que falta esta máscara. **Es el bug #1 que vas a tener si lo olvidas.**

Config del gate de tiro usada en validación: `shoot_open_enter=0.30`, `shoot_open_exit=0.20`
(histéresis). Variables de entorno equivalentes: `RL_GATE_SHOOT_OPEN_ENTER` / `_EXIT`.

---

## 4. Mapeo skill → SkillRequest (B-Human)

El policy emite una `SkillType` (índice 0-7). En el lado Python, `action.py` →
`pybh_backend.py:step()` → `pybh.rl_set_action()` → `RLSkillProvider.cpp` → `SkillRequest`.
B-Human ejecuta TODA la motion (walk/dribble/shoot/pass). El policy solo decide *qué* skill y
*con qué params*. Revisa en `RLSkillProvider.cpp` cómo se traduce cada `SkillType` a su
`SkillRequest::Type` — ese mapeo ya existe en el bridge v30; el trasplante no debería inventarlo,
solo asegurarse de que los 8 índices del nuevo `skill_order` coinciden con ese switch.

⚠️ Verifica que el **orden de skills del bridge C++ == el `skill_order` del manifest**
(stand, walk, shoot, pass, dribble, block, mark, observe). Si el enum C++ tiene otro orden,
los índices del argmax apuntarán a la skill equivocada. Es el bug #2 más probable.

---

## 5. Qué hice para producir este artefacto (contexto, no acción)

Pipeline doc-34 Pasos 1–9 corrido en SimRobot/Linux (CachyOS, software-render):
fine-tune corto **warm-start + KL-anchor** desde el trío `v2b@7500`, BC+PPO sobre dataset
mezclado 70:30 defensa:ataque, 10k steps, elegí el checkpoint **step2500** (collision 0.125,
cover 0.870). Export per-role a ONNX. Semáforo offline: STABILITY🟢 ROBUSTNESS🟢 DIVERSITY🟡.
Detalle completo en **`HANDOFF.md`** y **`runs/defender_47_v2/VERDICT_defv2.md`**.

Caveats heredados que el robot mostrará en runtime:
- `policy_stand_rate=0.42` (se queda quieto más de lo ideal 0.20). Lo verás como un defensor algo pasivo.
- `executed_pass`/`block` explícitos = 0 en validación → **dependen de un fix de gate de pase/despeje
  pendiente**. Si quieres que el defensor pase/despeje activamente, ese gate es lo primero a revisar.
- Recuperación validada vía **conducir-fuera** (dribble), no vía despeje-chut.

---

## 6. Checklist para el chat de trasplante

1. [ ] Confirmar twin repo en `rl-simrobot3d` (bridge v30): `cd SabanaHerons2026 && git branch --show-current`.
2. [ ] Localizar dónde B-Human carga modelos ONNX (ya usa onnxruntime — ver `LD_LIBRARY_PATH` a `Util/onnxruntime`).
3. [ ] Copiar `ppo_defender_hsl2026.onnx` al destino que B-Human espere (config/path del módulo RL).
4. [ ] Verificar que la obs de 47 dims que arma C++ coincide bit a bit con §2 (orden + semántica).
5. [ ] **Reproducir la máscara del gate (§3) antes del argmax.** No saltar esto.
6. [ ] Verificar orden de skills C++ == manifest (§4).
7. [ ] Denormalizar los 4 params (tanh [-1,1] → campo) replicando `action.py`.
8. [ ] Poblar role one-hot [44:46] correctamente (o el comportamiento se degrada).
9. [ ] Smoke en SimRobot visible y observar: ¿elige solo skills legales? ¿defiende? ¿no choca con su GK?
10. [ ] Comparar contra la validación: cover defensivo, collision_rate, gk_near=0.

## 7. Archivos de referencia en RL/
- `ppo_defender_hsl2026.{onnx,json}` — artefacto + contrato.
- `observation.py` (`ObsIndex`, `OBS_SIZE=47`) — layout de obs, fuente de verdad.
- `action.py` (`SkillType`, `SKILL_ORDER`, `action_param_mask`, `from_numpy`) — skills + decode de params.
- `gate.py` + `train.py:gate_skill_mask_from_obs` — máscara del gate (Design-A).
- `pybh_backend.py`, `RLSkillProvider.cpp` (twin) — el puente que ya existe.
- `HANDOFF.md`, `runs/defender_47_v2/VERDICT_defv2.md` — qué se entrenó y por qué.
- `tools/export_ppo_policy_team.py` — cómo se exportó (por si hay que re-exportar).
