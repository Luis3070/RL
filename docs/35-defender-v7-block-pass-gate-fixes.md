# 35 — Defender v7: por qué `block`/`pass` salían en 0 y cómo se arregló

**Fecha:** 2026-06-16 · **Rama:** `rl-v3a-multiagent-defender`

Este doc es la guía de transplante para reproducir los dos fixes de esta tanda
(`executed_block_rate=0` y `executed_pass_rate=0` en stage 8, single-agent) en otro
checkout/branch sin tener que re-derivar el razonamiento. Cubre: por qué pasaba, qué
heurística/gate/máscara se tocó, y qué validar después de portarlo.

## Contexto: por qué stage 8 single-agent es un caso especial

Stage 8 (`multiagent_defender_3v3`) fue diseñado para un orquestador multi-agente
(`_team_hook` en `RealSimRobotEnv`, ver `gate.py:146` y `environment.py`) que llena los
6 floats de rol/gate de equipo (`obs[41:47]`: `pass_armed`, `observe_armed`,
`pass_arm_progress`, `is_striker`, `is_open_support`, `is_off_ball_support`) usando
lógica real de equipo (`multiagent_teacher.py`: `assign_roles`, `smash_or_pass`,
`build_team_gate_decision`). **Ese hook nunca se conectó en el pipeline de
entrenamiento/eval que corre con `RL_PARALLEL_PLAYER_NUMBERS=1`** (un solo robot RL,
los demás son cuerpos scripted/estáticos) — solo se usa para generar el dataset BC
offline (`teacher_multiagent_dataset.py`). En ese modo single-agent, `_team_hook is
None`, así que el único código que llena esos 6 floats es el `elif
self._episode_stage_id == 8:` que está en `environment.py` (reset path ~L357 y step
path ~L610). Si ese bloque no llena algo, **el default del dataclass se queda**
(`gate.py:148-153`, todo `False`/`0.0`), y la máscara de skills lo cierra
permanentemente — no hay ningún otro camino para que se abra.

`gate_skill_mask_from_obs()` en `train.py:18-128` es la fuente única de verdad de la
máscara (Design-A, ver `CLAUDE.md` [removed from this branch; available on rl-v4.2-multiagent-gk]). Las líneas relevantes para block/pass:

```python
has_role = is_striker | is_open | is_off_ball
...
mask[:, _BLOCK_SKILL_IDX] = ((~is_striker) & has_role).to(mask.dtype)
mask[:, _PASS_SKILL_IDX]  = (pass_armed & is_striker).to(mask.dtype)
```

Si `has_role` nunca es `True` → block siempre cerrado. Si `pass_armed` nunca es `True`
→ pass siempre cerrado. Los dos bugs de esta tanda son exactamente eso, con causas
distintas.

## Fix 1 — `block` en 0 (sesión anterior, ya validado, no toca esta tanda)

Causa: el bloque `elif self._episode_stage_id == 8:` no llenaba ningún role-bit
(`is_striker`/`is_open_support`/`is_off_ball_support`), así que `has_role` era siempre
`False` y `block` quedaba cerrado para siempre, sin importar lo que hiciera la
política. Fix: inyectar los role-bits mínimos directamente en `environment.py`, en
ambos paths (reset y step):

```python
elif self._episode_stage_id == 8:
    has_ball = self._gate_decision.shoot_armed or self._gate_decision.dribble_armed
    self._gate_decision = replace(
        self._gate_decision,
        is_striker=has_ball,
        is_off_ball_support=not has_ball,
        pass_armed=_stage8_pass_armed(obs, has_ball),  # añadido esta tanda, ver Fix 2
    )
```

Heurística: en single-agent no hay equipo real para asignar roles, así que se usa la
señal más simple que sí existe en el robot — si tiene el balón (`shoot_armed` o
`dribble_armed`, los dos bits que ya arma `SkillGate` con histéresis de frames) es
`striker`; si no, es `off_ball_support`. Esto basta para que `has_role` sea `True`
siempre y la máscara de `block` se abra cuando el robot no tiene el balón
(`(~is_striker) & has_role`). Confirmado estable en una corrida completa de 20k steps
(v7f): `block` se mantuvo en rango 0.66–0.97 todo el run, nunca volvió a colapsar.

**`is_open_support` se deja siempre en `False`** — no hay manera de detectarlo en
single-agent (requiere saber si un compañero tiene línea de pase libre, que es
información de equipo) y no es necesario: `has_role` ya queda cubierto por
`is_striker`/`is_off_ball_support`.

## Fix 2 — `pass` en 0 (esta tanda)

### Intento 1 (fallido): copiar la precondición de lane-clearance del teacher

La primera versión de `_stage8_pass_armed()` copió literal la precondición de
`multiagent_teacher.smash_or_pass` (lógica de ataque, no de recuperación):

```python
if obs.nearest_opponent_front_dist <= 800.0:  # OPPONENT_LANE_CLEARANCE_MM
    return False
```

Resultado: `pass=0.000` constante en todo el run v7f (20k steps), salvo un blip aislado
de 0.024 que no se repitió. **Causa raíz**: el escenario `defense_recovery_clear`
(`production_curriculum.py::_defense_recovery_world`, ~L468) coloca al presser
**deliberadamente a 250–600mm** del punto de recuperación del balón — es la mecánica
del escenario, "el balón se le acaba de robar a este atacante":

```python
presser_x = ball_x + float(rng.uniform(250.0, 600.0))
```

Eso cae siempre dentro del umbral de 800mm, así que la precondición rechazaba el pase
exactamente en el escenario diseñado para enseñar el despeje-por-pase. Lección para
transplantar: **no copiar preconditions del teacher multi-agente sin verificar contra
qué escenario corre el heurístico single-agent** — el contexto (ataque vs.
recuperación defensiva) cambia qué umbrales tienen sentido.

### Intento 2 (el que quedó): quitar la precondición de lane

`environment.py:185-224`, helper nuevo `_stage8_pass_armed()`:

```python
_STAGE8_TEAMMATE_FRESH_MAX_AGE_MS = 1000.0
_STAGE8_MIN_PASS_DISTANCE_MM = 400.0
_STAGE8_MAX_PASS_DISTANCE_MM = 4000.0
_STAGE8_GK_EXCLUDE_X_MM = -2000.0

def _stage8_pass_armed(obs: BackendObservation, has_ball: bool) -> bool:
    if not has_ball:
        return False
    self_x, self_y = float(obs.robot_x), float(obs.robot_y)
    for idx, pose in enumerate(obs.teammate_poses):
        age = float(obs.teammate_ages_ms[idx]) if idx < len(obs.teammate_ages_ms) else float("inf")
        if age > _STAGE8_TEAMMATE_FRESH_MAX_AGE_MS:
            continue
        x_field, y_field = float(pose[0]), float(pose[1])
        if x_field < _STAGE8_GK_EXCLUDE_X_MM:
            continue  # excluye al GK como destino de pase
        dist = math.hypot(x_field - self_x, y_field - self_y)
        if _STAGE8_MIN_PASS_DISTANCE_MM <= dist <= _STAGE8_MAX_PASS_DISTANCE_MM:
            return True
    return False
```

Heurística: arma `pass_armed` cuando el striker tiene el balón **y** existe al menos
un compañero "fresco" (visto hace <1s, mismo umbral que el teacher,
`TEAMMATE_FRESH_MAX_AGE_MS`) dentro de rango de pase razonable (400–4000mm, igual que
`multiagent_teacher.MIN/MAX_PASS_DISTANCE_MM`), excluyendo al portero como destino
(`GK_EXCLUDE_X_MM=-2000`, igual que el teacher). La selección del *target* concreto del
pase se la deja al propio head de `pass_target` de la política — esto solo decide si la
skill se desbloquea, no a quién pasa. **Deliberadamente sin precondición de lane**: que
el reward de PPO juzgue si pasar bajo presión fue buena idea, no el gate.

Resultado inmediato tras el fix: `pass_armed_rate` saltó de 0.000 a 0.063, y `pass`
(ejecutado) lo siguió de cerca (0.062) — confirmado con los diagnósticos nuevos (ver
abajo).

## Diagnóstico nuevo: `pass_armed_rate` / `striker_rate`

`policy_pass_rate` (ya existía) **no sirve para distinguir** "el gate nunca abre" de
"la política nunca elige pasar" — se captura *después* de aplicar la máscara (la
categórica ya se muestreó de logits enmascarados), así que `policy_pass_rate` ==
`executed_pass_rate` por construcción. Para medir la apertura del gate de forma
independiente se añadieron dos contadores nuevos:

- `environment.py` — `_reset_episode_tracker()` (~L988): `pass_armed_steps`,
  `striker_steps`.
- `environment.py` — `_update_episode_tracker()` (~L1110-1111): incrementan en cada
  step según `self._gate_decision.pass_armed` / `.is_striker`.
- `environment.py` — summary (~L1655-1656): `pass_armed_rate`, `striker_rate`.
- `train.py` — `_EVAL_METRIC_NAMES` y el print de eval stage-8 (~L1689) ahora muestran
  `pass=... policy_pass=... pass_armed=... striker=...` en la misma línea.

Con esto confirmado en todos los checkpoints leídos esta tanda: `pass` y `pass_armed`
se mueven casi idénticos siempre — **el gate es el cuello de botella, no la política**;
cuando el gate abre, la política casi siempre toma el pase.

## Validación: el problema del sample size en eval

`RL_EVAL_EPISODES=8` (el default de todas las corridas) es **ruidoso** — en una misma
corrida (`v7g_train3.log`) `block` osciló 0.887→0.771→0.666 en 3 checkpoints
consecutivos (step 2500→5000→7500) bajo el mismo régimen de `bc_beta`, lo cual por sí
solo no distingue "deriva real" de "ruido de 8 episodios".

Validación con `RL_EVAL_EPISODES=40` sobre el checkpoint
`runs/defender_47_v7g/checkpoint_stage8_step2500.pt` (que con 8 episodios había leído
`block=0.887 reward_mean=0.940 policy_incompat=0.364`):

| métrica | 8 episodios | 40 episodios |
|---|---|---|
| block | 0.887 | **0.681** |
| pass | 0.043 | 0.033 |
| pass_armed | 0.043 | 0.033 |
| striker | 0.111 | 0.318 |
| policy_incompat | 0.364 | **0.656** |
| reward_mean | 0.940 | **0.452** |
| defender_score | 0.697 | 0.694 |

Lectura: la muestra de 8 episodios fue optimista en `block`/`reward_mean`/
`policy_incompat` simultáneamente (no es un outlier aislado en una métrica). En cambio
`pass`/`pass_armed` y `defender_score` se mantuvieron estables entre ambas muestras —
son las métricas que se pueden leer con confianza incluso a 8 episodios; `block` y
`reward_mean` no. **Regla práctica para transplantar:** cualquier lectura de
`block`/`policy_incompat`/`reward_mean` con `RL_EVAL_EPISODES=8` debe tratarse como
señal direccional, no como número final — re-validar con ≥40 episodios antes de decidir
si un checkpoint es bueno.

## Transplante a BHuman (C++) — qué añade este doc sobre la guía 25/26

La guía base (`25-guia-transplante-ppo-bhuman.md`) cubre el contrato single-agent
(obs 0-25, gate de shoot/dribble, máscara de stand/walk/shoot/dribble). Esta tanda
añade un bloque de obs (41-46) y dos skills nuevas a la máscara (`block`, `pass`) que
el robot necesita para el rol de defensor. Lo de abajo es **delta sobre la guía 25**,
no la reemplaza — seguir aplicando las secciones 1-9 de la guía 25 tal cual.

### Observación — bloque nuevo `obs[41:47]`

| Índice | Nombre | Unidad/rango | Fuente |
|--------|--------|--------------|--------|
| 41 | `pass_armed` | 0.0 o 1.0 | **salida del gate** (heurística abajo) |
| 42 | `observe_armed` | 0.0 o 1.0 | no usado por el defensor single-robot — dejar siempre 0.0 |
| 43 | `pass_arm_progress` | [0,1] | no usado por esta heurística — dejar siempre 0.0 (no hay debounce de frames para pass_armed, arma en 1 paso) |
| 44 | `is_striker` | 0.0 o 1.0 | **salida del gate**: `shoot_armed OR dribble_armed` |
| 45 | `is_open_support` | 0.0 o 1.0 | siempre 0.0 — requiere info de equipo que el robot individual no tiene |
| 46 | `is_off_ball_support` | 0.0 o 1.0 | **salida del gate**: `NOT is_striker` |

Igual que en la guía 25 §3, estos 6 floats son salida del gate, no percepción directa
— se calculan **antes** de construir el vector de observación y se hornean en él.

### SkillGate — extensión de estado y algoritmo

No hay estado nuevo que persista entre frames para este bloque (a diferencia de
`shoot_streak`/`dribble_streak`, que sí son stateful) — `is_striker`/
`is_off_ball_support`/`pass_armed` se recalculan desde cero cada paso a partir de
`shoot_armed`/`dribble_armed` (que sí son stateful, ya cubiertos por la guía 25 §4) y
de la percepción de compañeros del frame actual:

```python
# Ejecutar DESPUÉS del gate de shoot/dribble de la guía 25 §4.3, mismo paso:
has_ball = shoot_armed or dribble_armed
is_striker = has_ball
is_off_ball_support = not has_ball
is_open_support = False   # fijo, no implementar de momento

# pass_armed: arma cuando el striker tiene el balón y existe un compañero
# "fresco" dentro de rango de pase razonable. Sin precondición de lane —
# ver razón en "Intento 1 (fallido)" arriba: la lane-clearance del teacher
# de ataque rechaza siempre el escenario de recuperación defensiva.
TEAMMATE_FRESH_MAX_AGE_MS = 1000.0
MIN_PASS_DISTANCE_MM = 400.0
MAX_PASS_DISTANCE_MM = 4000.0
GK_EXCLUDE_X_MM = -2000.0

pass_armed = False
if has_ball:
    for teammate in teammates_seen_this_frame:
        if teammate.age_ms > TEAMMATE_FRESH_MAX_AGE_MS:
            continue
        if teammate.field_x < GK_EXCLUDE_X_MM:
            continue  # no pasar al portero
        dist = hypot(teammate.field_x - robot_x, teammate.field_y - robot_y)
        if MIN_PASS_DISTANCE_MM <= dist <= MAX_PASS_DISTANCE_MM:
            pass_armed = True
            break

# Escribir en obs[41..46]
obs[41] = 1.0 if pass_armed else 0.0
obs[42] = 0.0   # observe_armed, no implementado
obs[43] = 0.0   # pass_arm_progress, no implementado
obs[44] = 1.0 if is_striker else 0.0
obs[45] = 0.0   # is_open_support, no implementado
obs[46] = 1.0 if is_off_ball_support else 0.0
```

`teammates_seen_this_frame` debe venir del `TeammateModel`/`ObstacleModel` de BHuman
con posición en coordenadas de campo absolutas (no relativas al robot, a diferencia de
`ball_rel_x/y`) y una edad en ms desde el último avistamiento — mismo concepto que
`time_since_ball_seen` pero para cada compañero.

### Máscara de logits — extensión sobre la guía 25 §5

```python
has_role = is_striker or is_off_ball_support   # is_open_support fijo en False, no aporta

# block: disponible para cualquier rol no-striker. Sin requerir dribble_armed —
# block es una postura de intercepción válida a cualquier distancia.
mask[5] = 1.0 if ((not is_striker) and has_role) else 0.0   # índice 5 = block

# pass: solo para el striker, y solo si el gate lo armó.
mask[3] = 1.0 if (pass_armed and is_striker) else 0.0        # índice 3 = pass

# mark, observe: no cubiertos por esta heurística — dejar enmascarados a 0
# (no implementar en el robot todavía).
mask[6] = 0.0   # mark
mask[7] = 0.0   # observe
```

`stand`/`walk` siguen las mismas reglas de la guía 25 §5.1 (prohibidos solo cuando
`finish_armed = shoot_armed or dribble_armed`); ese comportamiento no cambia.

**Invariante de seguridad (extiende guía 25 §5.4):** si `has_role` es siempre `True`
(que es el caso con esta heurística — todo robot es striker u off_ball_support, nunca
ninguno de los dos), entonces `block` o `walk`/`stand` siempre tienen al menos una
opción disponible para el no-striker. No hay combinación de bits que deje la máscara
completamente en cero.

### Checklist de verificación — bloque defensor (delta sobre guía 25 §10)

- [ ] `obs[41:47]` se llena con la heurística de arriba, **no con ceros fijos** (un
  checkpoint que vio estos bits reales en entrenamiento se comporta erráticamente si
  el robot los manda siempre en 0 — exactamente el bug de `executed_block_rate=0` /
  `executed_pass_rate=0` que motivó esta tanda).
- [ ] El cálculo de `is_striker`/`is_off_ball_support`/`pass_armed` se ejecuta **en el
  mismo paso**, después del gate de shoot/dribble y antes de construir el vector
  completo — igual orden que la guía 25 §9, insertando el paso nuevo entre el 2 y el 3.
- [ ] `is_open_support` y `observe_armed`/`pass_arm_progress` van fijos en `0.0` — no
  inventar una heurística para ellos sin antes confirmar que el checkpoint que se va a
  transplantar realmente los usa (si se entrenó con esta misma heurística single-agent,
  no los usa).
- [ ] La heurística de `pass_armed` usa posición de compañeros en **coordenadas de
  campo absolutas**, no relativas al robot — confundir esto produce distancias
  completamente erróneas y el pase nunca arma o arma siempre.
- [ ] `mark` (índice 6) y `observe` (índice 7) quedan enmascarados a 0 siempre — no
  están cubiertos por ninguna heurística de esta tanda.
- [ ] No copiar la precondición de lane-clearance (`nearest_opponent_front_dist`) de
  ningún heurístico de ataque al heurístico de `pass_armed` defensivo — ver "Intento 1
  (fallido)" arriba para el porqué exacto.
- [ ] Validar el checkpoint candidato con `RL_EVAL_EPISODES≥40` antes de fijarlo como
  baseline de transplante (ver tabla de la sección anterior) — un checkpoint elegido
  con 8 episodios puede estar sobreestimado en `block`/`reward_mean`.

### Qué NO va al robot (extiende guía 25 §11)

Igual que en la guía 25: nada de `reward.py`, `behavioral_evaluator.py`,
`curriculum.py`/`production_curriculum.py`, stage masks de entrenamiento, BC/PPO. Esto
incluye específicamente para esta tanda:
- El escenario `defense_recovery_clear` y el resto de `production_curriculum.py` — son
  generadores de mundos para entrenar/evaluar en sim, no lógica de inferencia.
- Los contadores de diagnóstico `pass_armed_rate`/`striker_rate` — son telemetría de
  entrenamiento (`environment.py` tracker), no tienen función en el robot.
- `multiagent_teacher.py` completo (`assign_roles`, `smash_or_pass`,
  `build_team_gate_decision`) — es el orquestador multi-agente que nunca se conectó en
  el pipeline single-robot; la heurística de esta tanda es un sustituto deliberadamente
  más simple, no una versión reducida de ese módulo.

Lo que sí va al robot, sumado a la lista de la guía 25 §11: el cálculo de
`is_striker`/`is_off_ball_support`/`pass_armed` (heurística de arriba) y las dos
entradas nuevas de máscara (`block`, `pass`).

## Estado de checkpoints al cierre de esta tanda

- `runs/defender_47_v7f/checkpoint_stage8_step20000.pt` — block fix validado en 20k
  steps completos, pass fix con la precondition rota (intento 1) — pass=0 todo el run.
- `runs/defender_47_v7g/checkpoint_stage8_step2500.pt` — ambos fixes activos
  (intento 2 del pass). Validado con 40 episodios: `block=0.681 pass=0.033
  pass_armed=0.033 defender_score=0.694 policy_incompat=0.656`. `policy_incompat` alto
  sugiere que conviene re-entrenar con `bc_beta` más alto que el 0.13 con el que
  terminó la corrida que generó este checkpoint, antes de considerarlo definitivo.
