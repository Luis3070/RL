# 30 — Defender 47-dim: handoff de ejecución (v3, 2026-06-13)

> **Ejecutor:** Codex (con Rafael). **Owner del contrato:** Luis + Claude (rama `rl-simrobot3d`).
> **Reemplaza** la v1 (2026-06-12) y la v2 de este doc. Cambio mayor de la v3
> (decisión Luis + Rafael, 2026-06-13): **ideología per-role** — NO un único
> cerebro para todos los roles, sino un ONNX independiente por rol (striker,
> defender, supporter): *mejor excelente en una cosa que mediocre en todas*.
> Este doc produce el **defender 47-dim especializado con su propio ONNX**,
> que defiende muy bien Y respeta el contrato multi-agente al 100%.
> Contexto previo que sigue vigente: existe el defender obs26 de Rafael
> (commit `fbe1e98`, rama `obs-26-final`); el candidato de ataque es
> **v2b@7500**; existe el harness de probe chunked.

---

## 0. Objetivo e ideología per-role

**Objetivo:** una política 47-dim ESPECIALIZADA en defensa — marca, bloquea,
despeja, cubre la línea balón–portería propia — exportada a su **propio ONNX**
(`ppo_defender_hsl2026.onnx`), hermana del ONNX de ataque ya exportado
(`ppo_team_hsl2026.onnx` = striker/supports, sha 7126a01c).

**Ideología (decisión 2026-06-13):** la familia de producción es **un ONNX por
rol** — striker (ya existe: v2b@7500), defender (este doc), supporter (misma
receta, después). Cada política se entrena y promociona por SU especialidad;
ya no exigimos que un solo cerebro lo haga todo.

**Lo que NO cambia con la ideología:** el contrato. TODAS las políticas de la
familia comparten el MISMO layout de 47 obs (§2), la MISMA firma de
entrada/salida del ONNX, el MISMO gate calibrado y la MISMA fuente única de
máscara (Design-A). El one-hot de rol (obs[44:46]) sigue siendo dinámico DENTRO
del episodio: cuando el defender es el más cercano al balón se convierte en
presser (`is_striker=1`) y debe presionar/despejar con competencia — la
especialización es del CHECKPOINT, no una máscara que congele el rol. Eso es lo
que permite intercambiar ONNX por rol en deploy sin tocar el encoder C++.

**Fuera de alcance de este doc:** CÓMO se despachan los tres ONNX en el robot
(switch por rol B-Human dinámico, histéresis del cambio, qué pasa en la
transición). Eso lo discuten Luis + Claude sobre el multi-agente actual cuando
el defender exista. Este doc sólo garantiza que nada de lo entrenado lo bloquee
(firma idéntica = el dispatch es decisión 100% del lado C++).

---

## 1. Punto de partida

### 1.1 Lo nuestro (rama `rl-simrobot3d`)

| Qué | Dónde |
|---|---|
| **Candidato actual (warm-start + KL anchor del defender)** | `runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt` (floor 4/5 incl. preshoot; ablation: lee teammates de verdad) |
| Corpus de ataque (replay anti-olvido, SIEMPRE en el fine-tune) | `runs/teacher_multiagent_dataset/dataset_multiagent_train_20260611_full_v1/samples.npz` (212 eps, 355k rows) |
| ONNX actual + manifest | `ppo_team_hsl2026.onnx` / `.json` (sha 7126a01c) |
| Pipeline de dataset | `teacher_multiagent_dataset.py` (+ `_audit.py`, `merge_multiagent_teacher_datasets.py`) |
| Teacher framework | `multiagent_teacher.py` |
| Trainers | `train_multiagent_production.py` → `train_multiagent.py` |
| Env de equipo (training) | `team_parallel_environment.py` + hooks en `environment.py` |
| Probe de estabilidad chunked | `tools/run/run_stability_chunked.sh` (un freeze = un seed perdido, no el run) |
| Spec encoder C++ 26→47 (prerequisito del deploy, no del fine-tune) | `docs/29-embedded-encoder-26-to-47-spec.md` |

### 1.2 Lo de Rafael (rama `obs-26-final`, commit `fbe1e98`)

`defender_policy.py` + `generate_defender_dataset.py` + `train_defender_bc.py` +
recovery ONNX→pt. El teacher heurístico **sí funciona** (pasa con traffic-clear,
bloquea con offset lateral, camina defensivo de cara al balón) y esos VALORES se
portan (§4.4). Su instinto de ESPECIALIZAR el defender era correcto — la
ideología per-role lo confirma. Lo que el salto a 47 obs arregla son **dos
defectos observados** y **tres violaciones de contrato** que NO se portan:

**Defectos (lo que 47 obs arregla):**

1. **Stand-freeze**: `teacher_action()` emite `stand` cuando
   `time_since_ball_seen > 0.45` o `ball_seen_percentage < 0.12`. En obs26 no hay
   alternativa: si el robot pierde el balón de vista, se congela. En 47 obs el
   **team ball (obs[32:34] + valid)** da una estimación del equipo → el defender
   camina al cover point con el balón del EQUIPO en vez de pararse. Esta es la
   razón #1 por la que el defender 47 será mejor; es **medible** (§9.d).
2. **Rol congelado por máscara** (distinto de "especializado"): su máscara
   propia prohíbe shoot/dribble SIEMPRE — cuando el defender recupera el balón
   y debería presionar/despejar como presser, no puede. En 47 la
   especialización vive en el CHECKPOINT (datos defensivos dominantes) pero el
   one-hot dinámico (obs[44:46]) le permite asumir presser dentro del episodio
   cuando es el más cercano al balón. Especialista ≠ incapaz del resto.

**Violaciones de contrato (NO portar, ni adaptadas):**

1. **Overload de los bits del gate**: su política lee `obs[shoot_armed]` como
   "pass_armed" y `obs[dribble_armed]` como "block-engage". En el contrato 47,
   obs[23] y obs[24] conservan su semántica de ataque (Design-A); el pass_armed
   real vive en **obs[41]** y el engage defensivo se deriva de `dribble_armed`
   SIN redefinirlo (§4.3 — "balón engageable" es exactamente lo que dribble_armed
   ya significa).
2. **Segunda fuente de máscara**: `defender_skill_mask_from_obs_{np,torch}` +
   `masked_defender_logits` duplican el gate. PROHIBIDO: la única fuente es
   `train.py:gate_skill_mask_from_obs()` (§3.2). Esos archivos se quedan en su
   rama.
3. **`DefenderActor` con trunk de 26**: no se porta; el fine-tune usa la red del
   candidato 47 tal cual (`train_multiagent_production.py` la carga).

---

## 2. Layout de obs 47 (referencia rápida — NO se toca)

| Índices | Contenido |
|---|---|
| 0–25 | Contrato single-agent legacy (pose, balón, scoring, proximidad, gate bits 23–25) |
| 26–31 | 3 teammates relativos (Block B, orden por proximidad) |
| 32–34 | Team ball: x, y, valid |
| 35–37 | Teammate engagement flags |
| 38 | Consensus |
| 39–40 | Goal vector |
| 41–43 | `pass_armed`, `observe_armed`, `pass_arm_progress` |
| 44–46 | Role one-hot: `is_striker`, `is_open_support`, `is_off_ball_support` |

`block` existe en `SKILL_ORDER` (índice 5, `action.py`) y el bridge lo ejecuta
(smoke de 8 skills) — el ataque nunca lo usó, el defender sí.

---

## 3. Contrato duro — no negociable

1. **OBS_SIZE = 47 y SKILL_ORDER no cambian.** Nada de un 4º bit de rol.
2. **Design-A**: la máscara forbid-invalid a tiempo de update PPO se deriva SOLO
   de columnas de obs, en UNA función: `train.py:gate_skill_mask_from_obs()`.
   Los labels del teacher deben ser gate-consistentes fila a fila (el audit lo
   verifica: `role-mask bad rows = 0`).
3. **Gate calibrado**: `RL_GATE_SHOOT_OPEN_ENTER=0.30` y
   `RL_GATE_SHOOT_OPEN_EXIT=0.20` exportados en TODO (generación, training,
   eval, probe). Están en el `gate_config` del dataset y el merge falla si no
   coinciden.
4. **Checkpoints inmutables**: `checkpoint_stage4_step5000.pt`,
   `runs/multiagent_stage7_v1/checkpoint_stage7_step10000.pt`,
   `runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt`.
5. **Gate de promoción = el rubric defensivo (§9), no el floor de ataque.**
   Con la ideología per-role, el defender ya NO tiene que ser buen striker.
   El floor single-agent (`behavioral_evaluator.py`, con los RL_GATE_* de
   arriba) baja a **diagnóstico con bar relajado ≥ 2/5**: mide que el
   checkpoint sigue sabiendo presionar/despejar cuando le toca presser
   (one-hot dinámico, §0) — no que siga siendo el finisher. Se corre igual en
   **CADA checkpoint, incluidos los intermedios** — lección v2: el pico puede
   estar a mitad de run; evaluar SOLO el último checkpoint es cómo se pierde
   el bueno. Si cae a 0-1/5, el presser interno está muerto → subir replay de
   ataque (§3.6), no aceptar el checkpoint.
6. **Corpus de ataque en el replay, con peso reducido.** Ya no es anti-olvido
   total (el striker tiene su propio ONNX): su función ahora es mantener la
   competencia de presser y estabilizar las codificaciones compartidas.
   Mezcla orientativa defensa:ataque ≈ 70:30 (la v2 usaba el ataque completo).
   Si el diagnóstico del §3.5 cae: subir el bucket de ataque, no alargar el PPO.
7. **Sims**: UNA sesión a la vez; máx 4 slots visibles; todo chunked (un freeze
   = un chunk perdido). `pkill -f SimRobot; pkill -f Xvfb` entre chunks y
   `PYBH_RL_SHM_NAME` fresco por chunk (patrón de `tools/run/run_stability_chunked.sh`).
8. **`reward.py`, `gate.py` (campos legacy), `train_production.py`: intocables.**
   La única edición permitida en `train.py` es la regla de block del §5.2, y
   requiere aprobación explícita de Luis antes de commitear.
9. **Commits**: sin push a remoto sin aprobación de Luis.

---

## 4. Diseño: fase defensiva sin tocar el contrato

La idea central: **los tres roles existentes se reutilizan; lo que cambia en fase
defensiva es el TARGET de coordinación de cada rol, no el contrato.** Los
teachers de soporte (`teacher_open_support`, `teacher_off_ball_support`) ya son
"camina/párate hacia `coordination_target`" — sirven casi sin cambios si el
target es defensivo.

### 4.1 Detección de fase (con histéresis — el campo medio no puede parpadear)

```
fase = "defense"  si team_ball_x < DEFENSE_PHASE_ENTER_X  (-300 mm)
fase = "attack"   si team_ball_x > DEFENSE_PHASE_EXIT_X   (+300 mm)
entre ambos: conserva la fase anterior (banda muerta de 600 mm)
```

Usar el team ball (obs[32:34]) si `valid`, si no el balón propio exportado.
La fase vive en `TeamCoordinationState` (campo nuevo `phase: str = "attack"`),
se resetea en `reset()`. Mismo patrón sticky que `forced_side`/spawn-side latch.

### 4.2 Asignación de roles en fase defensiva

Extender `assign_roles()` (`multiagent_teacher.py:162`) con branch defensivo
(o parámetro `phase`); la histéresis de 7 frames y `ROLE_MIN_DISTANCE_GAP_MM`
se conservan tal cual para el presser:

- **presser** (bit `is_striker`): el más cercano al balón — misma maquinaria de
  striker actual (candidato + streak + gap).
- **defender** (bit `is_off_ball_support`): de los restantes, el de **menor x**
  (el más profundo).
- **defense_supporter** (bit `is_open_support`): el otro.

Con 2 agentes de campo: presser + defender (sin defense_supporter), igual que
hoy el 2v0 da striker + open_support.

### 4.3 Targets y conducta por rol (fase defensiva)

**presser** → sin cambios: `teacher_striker()` ya persigue, recupera y conduce;
`smash_or_pass` da el pase de salida cuando `pass_armed` (que requiere
shoot|dribble armed — consistente). El "clear" emerge de la lógica existente.

**defender** → target = `defender_cover_target()`, port simplificado de
`Defender::position` (BHuman `PositionRoles/Defender.cpp:68-161`, caso
single-defender — `forward=false`, `radius=minRadius`):

```
circle_base   = (-4500, 0)                      # centro de nuestra ground line
radius        = DEFENDER_RADIUS_MM              # 1600 (≈ minRadius royale)
side          = sticky_left_right(ball)         # port de calcDefenderRoyaleIsLeft
                                                # (Defender.cpp:313-333): balón muy a la
                                                # izquierda→left, muy a la derecha→right,
                                                # si no conserva el lado anterior
goal_post     = (-4500, +800·side_sign)
gk_edge       = (gk_x, gk_y + GK_STAND_RADIUS_MM·side_sign)   # GK del obs/teammates;
                                                # si no hay GK detectable → centro de portería
cover_point   = midpoint(goal_post, gk_edge)    # el hueco que el GK NO cubre
target        = intersección de la línea (ball → cover_point) con el círculo
                (circle_base, radius), tomando la intersección más cercana al balón
theta         = atan2(ball - target)            # SIEMPRE de cara al balón (Defender.cpp:48)
```

Clips obligatorios: dentro del campo; **nunca dentro del área propia**
(si `target_x < -3900` y `|target_y| < 1100` → empujar a `x = -3650`). Esto es
el "no taparle la línea al GK" y la regla de 0 colisiones con el GK (§9.b).

Conducta (función nueva `teacher_defender()`, despachada cuando
`phase=="defense"` y rol off_ball; Design-A en cada rama):

1. `dribble_armed` y NO soy presser → **block**, `target_y` =
   `_signed_block_offset` portado de Rafael (`clip(ball_rel_y/900, ±1)`).
   Gate-consistente: block sólo se etiqueta con `dribble_armed=1` (§5.2).
2. Balón propio stale (`time_since_ball_seen` alto) PERO team ball válido →
   **walk al cover target calculado con el team ball**. (Fix del stand-freeze.
   Prohibido emitir stand aquí.)
3. Dentro de `HOLD_TOLERANCE_OFF_BALL_MM` del target → **stand** (de cara al
   balón). Si no → **walk** al target.

**defense_supporter** → target = `defense_support_target()`: mismo patrón que
`support_target()` (scoring de grid + `separation_penalty` + STAB_OFFSET +
`forced_side` latch) pero con grid defensivo:

```
DEFENSE_SLOT_CANDIDATE_GRID = (
    (-2200.0, +1100.0),   # segundo carril profundo
    (-1400.0, +900.0),    # medio propio, shading del carril lateral
    (-600.0,  +1300.0),   # salida de presión / receptor del clear
    (-1000.0, +400.0),    # carril central cuando el balón va por banda
)
```

Scoring: `lane_cover(cand) × separation_penalty(cand, [presser, defender])`,
donde `lane_cover` premia estar entre el balón y nuestra portería desplazado
al **segundo** carril (no en la misma línea que el defender — la
`separation_penalty` con σ=800 ya separa, mantenerla). Mirror por
`forced_side` igual que en ataque (lado pegado por episodio, NO por frame).
Conducta: `teacher_open_support()` sin cambios (walk/stand al target).

### 4.4 Qué se porta de `defender_policy.py` (valores, no archivos)

| De Rafael | A dónde | Nota |
|---|---|---|
| `_signed_block_offset` (`ball_y/900`) | param de block en `teacher_defender()` | tal cual |
| `_traffic_clear_np` umbrales (teammate 1200 / opponent 850 / uncertain 1000) | refuerzo opcional del pase de salida del presser dentro de `smash_or_pass` (sumar a `OPPONENT_LANE_CLEARANCE_MM` ya existente) | NO como gate paralelo |
| `_face_ball_theta` | ya cubierto: el cover target siempre orienta al balón | — |
| stand cuando balón perdido | **NO portar** — sustituido por la regla 2 de §4.3 | es el defecto |

### 4.5 Rewards de equipo

- Los **hold bonuses existentes** (`RL_MA_SUPPORT_OPEN_HOLD_BONUS`,
  `RL_MA_SUPPORT_TRIANGLE_HOLD_BONUS`) anclan al `coordination_target` — con
  targets defensivos recompensan mantener el cover SIN términos nuevos en
  `reward.py`.
- `RL_MA_TEAM_GOAL_BONUS=2.5` (NO 5.0 — lección v2: 5.0 compró goles vendiendo
  disciplina de colisión).
- Knob nuevo en el TEAM layer (`team_parallel_environment.py`, patrón idéntico
  al goal bonus): `RL_MA_TEAM_GOAL_CONCEDED_PENALTY=2.5` — resta cuando el
  episodio termina con gol ENCAJADO. Mismo lugar donde se detecta
  `env._episode_tracker.get("goal")`, signo según portería.

---

## 5. Cambios de código, archivo por archivo

1. **`multiagent_teacher.py`**: constantes (`DEFENSE_PHASE_ENTER_X=-300.0`,
   `DEFENSE_PHASE_EXIT_X=+300.0`, `DEFENDER_RADIUS_MM=1600.0`,
   `GK_STAND_RADIUS_MM=400.0`, `DEFENSE_SLOT_CANDIDATE_GRID`, todas
   env-tunables con prefijo `RL_MA_DEF_`); `defender_cover_target()`;
   `defense_support_target()`; branch defensivo en `assign_roles()`; campo
   `phase` en `TeamCoordinationState`; `teacher_defender()`; dispatch por fase
   en `stage_aware_team_teacher_action()`.
2. **`train.py`** ⚠️ única edición permitida, requiere OK de Luis: en
   `gate_skill_mask_from_obs()` path de equipo (líneas ~92-125), añadir
   `_BLOCK_SKILL_IDX` y la regla:
   ```python
   mask[:, _BLOCK_SKILL_IDX] = (dribble_armed & ~is_striker & has_role).to(mask.dtype)
   ```
   Sin esta línea, block queda en el 1.0 inicial de la máscara = permitido
   SIEMPRE que el stage lo active — PPO podría explorar block en pleno ataque
   y romper el floor. La regla lo confina a no-strikers con balón engageable.
   Nota Design-A: el teacher sólo etiqueta block bajo `dribble_armed` (§4.3.1)
   → labels y máscara coinciden.
3. **`curriculum.py`**: `get_stage_spec(8)` — copia del stage 7 con `block`
   añadido a los active skills.
4. **`production_curriculum.py`**: los 4 mundos del §6 +
   `team_defense_scenarios(episodes)` (mismo patrón que
   `team_production_scenarios`, stage_id=8).
5. **`environment.py`**: token `"defense"` en el factory de escenarios
   (patrón exacto del token `"team"`).
6. **`teacher_multiagent_dataset.py`**: aceptar `RL_MA_TEACHER_STAGE=8`
   (extender el branch de stage, ~10 líneas); cómputo de fase + targets
   defensivos en el bloque de coordinación (~L880-920, donde llama
   `compute_open_lane_target`/`compute_triangle_target` — añadir el branch
   `if phase == "defense"` que usa las funciones del §4.3).
7. **`team_parallel_environment.py`**: el MISMO branch de fase en su bloque de
   targets (~L290-310 — los dos call sites deben quedar idénticos);
   métricas nuevas a `LAST_TEAM_METRICS`: `defender_cover_hold_rate`,
   `defender_gk_collision_rate`, `defense_phase_step_frac`,
   `goals_conceded_rate`; knob `RL_MA_TEAM_GOAL_CONCEDED_PENALTY`.
8. **`train_multiagent_production.py`**: choices de stage → `(6, 7, 8)` +
   pesos de escenarios defense (~15 líneas, patrón stage-7).
9. **`teacher_multiagent_dataset_audit.py`**: en stage 8 ya se corre con
   `--min-striker-shoot-frac 0 --min-striker-pass-armed 0`; añadir check:
   `block_frac > 0` en filas con rol off_ball y fase defensiva, y
   `stand_frac < 0.5` en filas defender con team_ball válido (anti-freeze).

---

## 6. Escenarios stage 8 (geometría concreta)

Patrón invertido de los `_team_3v0_*`: el balón y los oponentes en NUESTRA
mitad avanzando hacia x=-4500. Los spawns deben INDUCIR la asignación de roles
del §4.2 (nearest→presser, deepest→defender). GK nuestro siempre en
`(-4300, 0)`. `obstacles="dynamic"` usa el BH default behaviour, que ya ataca
nuestra portería cuando tiene el balón — no hace falta scripting.

| Escenario | Peso | Geometría |
|---|---|---|
| `defense_1v3_static_entry` | 1× | Balón `x∈(-2200,-1200)`, `y∈(-1200,1200)`. 1 atacante ESTÁTICO en `(ball_x+400, ball_y)` mirando a nuestra portería. Nuestro trío: presser `(ball_x-700, ball_y±150)` de cara al balón; defender en la línea balón–portería a `x∈(-3400,-3000)`; defense_supporter `x∈(-1600,-1000)`, `y` al lado contrario del balón. Aprende posicionamiento + press sin presión real. |
| `defense_2v3_dynamic` | 2× | Igual pero 2 atacantes BH DINÁMICOS (`ball_x+400` y `ball_x+1100`, `y` opuesta). Balón `x∈(-2500,-800)`. |
| `defense_3v3_full` | 3× | Espejo del `team_3v3_full_dynamic`: full BH (3 field + GK rival en `(+4200,0)`) atacando con balón en `x∈(-3000,-1000)`. La distribución perceptual del demo en fase defensiva. |
| `transition_midfield` | 1.5× | Balón disputado `x∈(-500,+500)`, 2 oponentes dinámicos, spawns nuestros simétricos. Enseña el cambio de fase con histéresis (attack↔defense) — sin este escenario la política nunca ve la transición. |

Presupuesto de episodio: `RL_MA_TEACHER_MAX_STEPS=600`. Los guards existentes
(reset ball-drift, collision sustain, out-of-bounds) se quedan tal cual —
`agent_out_of_bounds`/`reset_ball_drift` en los drops = guards funcionando.

---

## 7. Dataset (comandos exactos)

```bash
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
export RL_MA_TEACHER_COLLISION_SUSTAIN_STEPS=16
export PYTHONPATH=

# Chunked: 3 chunks × 8 episodios. Un freeze = un chunk perdido.
for i in 1 2 3; do
  RL_MA_TEACHER_STAGE=8 RL_MA_TEACHER_MAX_STEPS=600 \
  RL_MA_TEACHER_EPISODES=8 RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=6 \
  RL_MA_TEACHER_SEED_BASE=$((950000 + i*10000)) \
  PYBH_RL_SHM_NAME=/pybh_rl_def_chunk_${i} \
  python teacher_multiagent_dataset.py
  pkill -f SimRobot; pkill -f Xvfb; sleep 5
done

python merge_multiagent_teacher_datasets.py <chunk1> <chunk2> <chunk3> \
  --output-dir runs/teacher_multiagent_dataset/dataset_defense_v1
python teacher_multiagent_dataset_audit.py \
  runs/teacher_multiagent_dataset/dataset_defense_v1 \
  --min-striker-shoot-frac 0 --min-striker-pass-armed 0 --write
```

Qué importa en el audit defensivo: liveness verde (teammate block,
team-ball≠goal-vector, goal vector vivo), `role-mask bad rows = 0`,
`block_frac > 0`, anti-freeze check del §5.9. ~25% keep es lo esperado
(caídas de sim). **Nunca declarar que el dataset existe sin verlo en disco.**

Merge defensa + ataque SUBSAMPLEADO (el merge verifica `gate_config`):

⚠️ El corpus de ataque completo (355k rows) aplastaría la defensa (~40-60k rows
esperadas de 3 chunks) y el resultado sería un striker re-entrenado, no un
defender. Antes del merge, crear un subsample de ataque para que la mezcla
final quede **defensa:ataque ≈ 70:30 en filas** (§3.6): script corto de numpy
sobre `samples.npz` que tome ~`0.43 × rows_defensa` filas de ataque,
**estratificado por episodio completo** (episodios enteros, no filas sueltas —
los residuals/EMA pierden sentido si se trocean), guardado con el mismo formato
y el mismo `gate_config` en su summary (si no, el merge falla — correcto).

```bash
# 1) subsample de ataque → runs/teacher_multiagent_dataset/dataset_attack_subsample_30pct
# 2) merge:
python merge_multiagent_teacher_datasets.py \
  runs/teacher_multiagent_dataset/dataset_defense_v1 \
  runs/teacher_multiagent_dataset/dataset_attack_subsample_30pct \
  --output-dir runs/teacher_multiagent_dataset/dataset_defense_70_attack_30_v1
```

---

## 8. Fine-tune (BC + PPO)

```bash
RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20 \
RL_MA_TEAM_GOAL_BONUS=2.5 \
RL_MA_TEAM_GOAL_CONCEDED_PENALTY=2.5 \
RL_MA_PROD_TRAIN_STEPS=10000 RL_MA_PROD_MAX_EPISODE_STEPS=600 \
RL_MA_PROD_EVAL_INTERVAL=2500 RL_MA_PROD_CHECKPOINT_INTERVAL=2500 \
RL_MA_PROD_SCENARIO_SET=defense RL_MA_PROD_EXPERIMENT_NAME=defender_47_v1 \
python train_multiagent_production.py --stage-id 8 \
  --checkpoint runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt \
  --dataset runs/teacher_multiagent_dataset/dataset_defense_70_attack_30_v1/samples.npz
```

- `--dataset` SIEMPRE explícito (el default coge el dir más nuevo = error conocido).
- `--checkpoint` = warm-start **y** KL anchor (auto en el trainer). v2b@7500 es
  la base correcta aunque el destino sea un especialista: ya lee teammates,
  ya respeta el gate, y el KL anchor evita el colapso de distribución en un
  fine-tune corto.
- Checkpoint cada 2500; en cada uno: **rubric defensivo (gate de promoción) +
  floor diagnóstico ≥2/5** — el bueno puede ser el de en medio (lección v2b:
  el candidato salió del 7500, no del 12500).

---

## 9. Validación (rubric del defender — todas obligatorias)

| # | Check | Bar |
|---|---|---|
| a | Cover: defender dentro del cono balón→postes propios, en fase defensiva | ≥ 60% de pasos |
| b | Colisiones con el GK propio | 0 |
| c | Floor single-agent DIAGNÓSTICO (`behavioral_evaluator.py`, RL_GATE_* exportados) — competencia de presser, no de finisher (§3.5) | ≥ 2/5 en CADA checkpoint |
| d | **Anti-freeze** (el defecto obs26, medible): episodios con balón propio stale + team ball válido → `policy_stand_rate` del defender | < 0.10 |
| e | Block engagement: ventanas defensivas con `dribble_armed=1` en rol off_ball donde ejecuta block | ≥ 50% |
| f | Goles encajados en `defense_3v3_full` vs baseline (trío v2b@7500 SIN fine-tune defensivo, mismos seeds) | estrictamente menor |
| g | Estabilidad multi-seed DEFENSIVA: eval `defense_3v3_full` chunked en los 6 stable seeds (11,23,37,51,67,83 — patrón de `tools/run/run_stability_chunked.sh`: shm fresco + pkill por seed); reportar goles encajados y cover por seed | (f) se sostiene en ≥ 4/6 seeds |
| h | Export: `tools/export/export_ppo_policy_team.py --output models/defender/ppo_defender_hsl2026.onnx` → ONNX propio + manifest (gate_config + role indices + `"specialization": "defender"`) | sha documentado |

El probe de ataque (`run_stability_chunked.sh` sobre los 5 escenarios
single-agent) ya NO es gate para el defender — correrlo una vez como
diagnóstico está bien, pero el especialista se juzga en (a)-(g).

(a), (d), (e) salen de las métricas nuevas de `LAST_TEAM_METRICS` (§5.7) — 10
episodios de eval stage-8 son suficientes.

---

## 10. Flujo de ramas (acordado con Rafael)

1. Rafael sigue iterando obs26 en `obs-26-final` — NO se mergea esa rama.
2. TODO el trabajo 47-dim de este doc se ejecuta **sobre `rl-simrobot3d`**.
3. De `obs-26-final` se portan **valores y heurísticas** (§4.4) a mano, nunca
   los archivos (`defender_policy.py`, `train_defender_bc.py`, recovery ONNX→pt
   se quedan allá; su `DefenderActor` y sus máscaras violan el contrato 47).
4. Ojo numeración de docs: su rama tiene `docs/29-defender-obs26-pipeline-canonico.md`
   y `docs/30-recuperacion-defender-onnx-a-pt.md` [renumbered in this branch to `docs/41-defender-obs26-pipeline-canonico.md` and `docs/42-recuperacion-defender-onnx-a-pt.md`] — nombres de archivo distintos
   a los nuestros (sin conflicto de merge), pero al citar "doc 29/30" decir SIEMPRE
   de qué rama.
5. Si el fine-tune 47 pasa el rubric completo (§9), `ppo_defender_hsl2026.onnx`
   entra a la **familia per-role** junto a `ppo_team_hsl2026.onnx` (striker/
   supports, v2b@7500). El de ataque NO se reemplaza. El supporter dedicado
   vendrá después con esta misma receta (escenarios + teacher de su
   especialidad, mismo contrato).
6. El dispatch de los ONNX por rol en el robot (cómo y cuándo se conmuta con el
   rol dinámico de B-Human) lo definen Luis + Claude después — NO implementarlo
   por iniciativa en este handoff. El deploy sigue bloqueado por el encoder C++
   26→47 (`docs/29-embedded-encoder-26-to-47-spec.md`, lado Rafael), que es
   compartido por TODA la familia (misma firma de entrada).
