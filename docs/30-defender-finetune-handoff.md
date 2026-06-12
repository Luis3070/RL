# 30 — Defender fine-tune handoff (Rafael, 2026-06-12)

Objetivo: un defensor (y defense-supporter) RL sobre el contrato 47-dim, fine-tuned
DESDE el checkpoint multi-agente, sin romper el ataque. Todo lo necesario está en
este commit; el dataset de ataque también (23M, `runs/teacher_multiagent_dataset/dataset_multiagent_train_20260611_full_v1/`).

## Artefactos base (este commit)

| Qué | Dónde |
|---|---|
| Checkpoint candidato (warm-start + KL anchor del defender) | `runs/multiagent_stage7_v1/checkpoint_stage7_step10000.pt` (floor 4/5, gates stage-7 pass) |
| Corpus de ataque (replay anti-olvido) | `runs/teacher_multiagent_dataset/dataset_multiagent_train_20260611_full_v1/samples.npz` |
| ONNX actual + manifest | `ppo_team_hsl2026.onnx` / `.json` |
| Pipeline de dataset multi-agente | `teacher_multiagent_dataset.py` (+ `_audit.py`, `merge_multiagent_teacher_datasets.py`) |
| Teacher framework | `multiagent_teacher.py` |
| Trainers | `train_multiagent_production.py` → `train_multiagent.py` |
| Env de equipo (training) | `team_parallel_environment.py` + hooks en `environment.py` |
| Spec encoder C++ 26→47 | `docs/29-embedded-encoder-26-to-47-spec.md` (prerequisito del deploy, no del fine-tune) |

## Diseño clave: el defender NO cambia el contrato de obs

El defender se mapea al bit `is_off_ball_support` (obs[46]) — el "rol de
posicionamiento profundo". NO se añade una 4ª dimensión de rol (eso invalidaría
checkpoints y dataset). La conducta defensiva viene de los DATOS (escenarios +
teacher defensivo), no de un bit nuevo. Defense-supporter → bit `is_open_support`.

## Secuencia (realista: dataset hoy, BC/PPO esta noche-mañana)

### Paso 1 — Escenarios defensivos (stage 8) en `production_curriculum.py`

Patrón a seguir: `_team_3v0_dynamic_world` pero invertido — oponentes spawn EN
NUESTRA mitad avanzando hacia nuestra portería con el balón; nuestro trío
defiende. 3-4 escenarios graduados (1 atacante estático → 2-3 dinámicos → full).
Registrarlos con `stage_id=8` en `team_production_scenarios` (o lista nueva
`team_defense_scenarios` + token "defense" en el factory de environment.py,
siguiendo el patrón del token "team").

⚠️ stage 8 necesita `get_stage_spec(8)` en `curriculum.py` (copiar stage 7;
active skills iguales) y el orquestador acepta `RL_MA_TEACHER_STAGE=8` (extender
el branch de stage en `teacher_multiagent_dataset.py`, ~10 líneas).

### Paso 2 — Teacher defensivo en `multiagent_teacher.py`

- `defender`: port de `Defender::position(side,...)` (BHuman
  `PositionRoles/Defender.cpp:25-49`, ya mapeado en §14.3 del roadmap): punto en
  la línea balón–portería propia a distancia fija, side-aware, GK-aware
  (no taparle la línea al GK).
- `defense_supporter`: segundo carril / medio campo (patrón `support_target` con
  grid defensivo, x negativos).
- Asignación de roles en fase defensiva: el más cercano al balón = presser
  (striker bit), el más profundo de los demás = defender (off_ball bit), el otro
  = defense_supporter (open bit). Extender `assign_roles` con un branch
  `ball_x < 0` o función nueva — mantener la histéresis de 7 frames.

### Paso 3 — Dataset (chunked, mismas reglas que el regen de ayer)

```bash
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
export RL_MA_TEACHER_COLLISION_SUSTAIN_STEPS=16
export PYTHONPATH=/path/to/RL
for i in 1 2 3; do
  RL_MA_TEACHER_STAGE=8 RL_MA_TEACHER_MAX_STEPS=600 \
  RL_MA_TEACHER_EPISODES=8 RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=6 \
  RL_MA_TEACHER_SEED_BASE=$((950000 + i*10000)) \
  python teacher_multiagent_dataset.py
done
# merge de chunks (gate_config debe coincidir — el merge lo verifica):
python merge_multiagent_teacher_datasets.py <chunk1> <chunk2> <chunk3> \
  --output-dir runs/teacher_multiagent_dataset/dataset_defense_v1
python teacher_multiagent_dataset_audit.py runs/teacher_multiagent_dataset/dataset_defense_v1 --write
```
Audit defensivo: shoot/pass fracs NO aplican (defensa casi no chuta) — usar
`--min-striker-shoot-frac 0 --min-striker-pass-armed 0`; lo que importa:
liveness verde, role-mask 0 bad rows, hold de defender presente.
~25% keep esperado (caídas de sim); `agent_out_of_bounds`/`reset_ball_drift`
en drops = guards funcionando, no bugs.

### Paso 4 — Fine-tune (BC+PPO)

Mezclar defensa + ataque para el replay (anti-olvido):
```bash
python merge_multiagent_teacher_datasets.py \
  runs/teacher_multiagent_dataset/dataset_multiagent_train_20260611_full_v1 \
  runs/teacher_multiagent_dataset/dataset_defense_v1 \
  --output-dir runs/teacher_multiagent_dataset/dataset_attack_plus_defense_v1

RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20 \
RL_MA_PROD_TRAIN_STEPS=5000 RL_MA_PROD_MAX_EPISODE_STEPS=600 \
RL_MA_PROD_SCENARIO_SET=defense RL_MA_PROD_EXPERIMENT_NAME=defender_v1 \
python train_multiagent_production.py --stage-id 8 \
  --checkpoint runs/multiagent_stage7_v1/checkpoint_stage7_step10000.pt \
  --dataset runs/teacher_multiagent_dataset/dataset_attack_plus_defense_v1/samples.npz
```
⚠️ `train_multiagent_production.py` valida stage ∈ (6,7) — extender choices a 8
+ añadir pesos de escenario defensivos (copiar el patrón stage-7). ~15 líneas.

### Reglas duras (idénticas a las nuestras)

- Floor single-agent en CADA checkpoint: `behavioral_evaluator.py` ≥ 3/5 (con
  los RL_GATE_* de arriba — son parte del contrato, están en el manifest del ONNX).
- El corpus de ataque SIEMPRE en el replay del fine-tune (si el floor cae o el
  ataque se olvida: subir peso del bucket de ataque, no alargar el PPO).
- `checkpoint_stage4_step5000.pt` y `checkpoint_stage7_step10000.pt`: inmutables.
- Lanzar UNA sola sesión de sims a la vez (3 slots + generación en paralelo =
  régimen de caídas).
- Conocido: freeze "stale visible frames" mata sesiones largas — chunked todo;
  un freeze = un chunk perdido, no el run.

### Validación del defender (no existe rubric todavía)

Mínimo viable: 10 episodios stage-8 eval — el defender debe (a) mantenerse entre
balón y portería propia ≥60% de pasos en fase defensiva, (b) 0 colisiones con el
GK, (c) floor de ataque intacto. Si montas `_pass_defense_positioning` en
`behavioral_evaluator.py`, mejor.
