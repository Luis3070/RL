# Pipeline canonico del defensor obs26

Fecha de corte: 2026-06-12  
Rol: `defender`  
Contrato: `ppo_defender_obs26_skill_selector_v1`  
Normalizacion: `ppo_obs_v26_hsl2026`

## Proposito

Este documento describe el pipeline que genera el policy defensor obs26 como si
el checkpoint fuente nunca se hubiera perdido. El objetivo es dejar claro que el
repositorio contiene la misma estructura necesaria para producir un selector de
skills compatible con el ONNX que quedo integrado en SabanaHerons.

El policy no controla joints ni reemplaza locomocion. Produce:

```text
obs26 -> skill_logits[8] + param_mean[4]
```

La salida se usa como selector de skills B-Human. Para esta primera version
defensiva solo se activan:

```text
stand, walk, pass, block
```

El orden completo de logits se conserva para compatibilidad:

```text
stand, walk, shoot, pass, dribble, block, mark, observe
```

## Artefacto ONNX de referencia

El artefacto integrado en SabanaHerons es:

```text
SabanaHerons2026/Config/NeuralNets/RLPolicy/ppo_defender_hsl2026_param_repair.onnx
SabanaHerons2026/Config/NeuralNets/RLPolicy/ppo_defender_hsl2026_param_repair.json
```

Propiedades verificadas:

```text
input:  obs [batch, 26]
output: skill_logits [batch, 8]
output: param_mean [batch, 4]
opset:  11
onnx sha256: cdc0d4071b4e86dbf1766dd3995dbfbe4fc590a5079ffd18db1db299d643a3a1
```

Los pesos densos del actor viven en inicializadores ONNX equivalentes a:

```text
w0 -> trunk.0.weight   [128, 26]
b0 -> trunk.0.bias     [128]
w2 -> trunk.2.weight   [128, 128]
b2 -> trunk.2.bias     [128]
ws -> skill_head.weight [8, 128]
bs -> skill_head.bias   [8]
wp -> param_head.weight [4, 128]
bp -> param_head.bias   [4]
```

El grafo ONNX tambien contiene logica de mascara para que la inferencia no sea
solo una red densa. Esa mascara es parte del contrato defensivo.

## Observaciones y gates

Se mantiene `OBS_SIZE = 26`. Las ultimas tres columnas reutilizan los indices
del striker, pero con significado defensivo:

```text
obs[23] = pass_armed
obs[24] = engage_armed
obs[25] = pass_arm_progress
```

Esto es intencional: permite conservar el contrato obs26 y el bridge existente,
sin traer la ampliacion MAPPO de mas de 40 observaciones.

## Mascara de skills

La mascara defensiva esta en `defender_policy.py`.

Reglas:

```text
base: stand, walk
si pass_armed y trafico frontal limpio: solo pass
si engage_armed y no hay commit de pass: stand, walk, block
si no hay pass_armed ni engage_armed: solo stand, walk
```

El pass sigue la misma idea critica del striker: cuando el gate esta armado y el
commit es valido, las demas skills quedan bloqueadas con logits muy bajos
(`-1e9`) antes del `argmax`.

La limpieza frontal usa umbrales conservadores:

```text
teammate_front >= 1200 mm
opponent_front >= 850 mm
uncertain_front >= 1000 mm
```

## Modelo PyTorch

La clase canonica esta en:

```text
defender_policy.py
```

Estructura del actor:

```text
Linear(26, 128)
ELU
Linear(128, 128)
ELU
skill_head: Linear(128, 8)
param_head: Linear(128, 4)
```

El `state_dict` se alinea con el formato historico del striker. Debe contener:

```text
log_std_parameter
param_low
param_high
active_skill_mask
trunk.0.weight
trunk.0.bias
trunk.2.weight
trunk.2.bias
skill_head.weight
skill_head.bias
param_head.weight
param_head.bias
```

Esto permite tratar el defensor como un checkpoint compatible con el tooling
obs26 existente, aunque la tarea sea defensiva.

## Dataset sintetico

El dataset se genera con:

```text
generate_defender_dataset.py
```

No depende de SimRobot real. Fabrica estados plausibles en obs26:

```text
normal
front_threat
clearance
stale
lost
box_danger
```

Cada muestra produce:

```text
observations: [N, 26]
actions:      [N, 5]
weights:      [N]
```

La accion tiene el mismo formato de alto nivel usado por el decoder:

```text
skill_id, target_x, target_y, target_theta, pass_target
```

El teacher esta en `defender_policy.teacher_action`:

```text
pass_armed + trafico limpio -> pass
engage_armed                -> block
balon viejo/no confiable    -> stand
caso normal                 -> walk
```

## Entrenamiento BC

El trainer canonico es:

```text
train_defender_bc.py
```

Entrenamiento desde cero:

```bash
.venv/bin/python train_defender_bc.py \
  --samples 60000 \
  --steps 2000 \
  --batch-size 512 \
  --device cpu \
  --output-dir runs/defender_bc/defender_bc_obs26
```

Entrenamiento conservador desde un checkpoint existente:

```bash
.venv/bin/python train_defender_bc.py \
  --samples 12000 \
  --steps 500 \
  --batch-size 512 \
  --device cpu \
  --output-dir runs/defender_bc/distilled_from_recovered_onnx_w20 \
  --init-checkpoint runs/defender_bc/recovered_from_sabana_onnx/checkpoint_defender_bc_recovered_full_from_onnx.pt \
  --reference-checkpoint runs/defender_bc/recovered_from_sabana_onnx/checkpoint_defender_bc_recovered_full_from_onnx.pt \
  --distill-weight 20.0 \
  --param-distill-weight 1.0
```

La perdida combina:

```text
cross entropy de skill contra teacher sintetico
smooth L1 de parametros activos
distillation opcional contra un checkpoint referencia
```

Con `--distill-weight` alto, el modelo aprende poco a poco sin alejarse del
comportamiento del checkpoint de referencia.

## Checkpoint esperado

El checkpoint entrenado debe parecerse al formato historico del striker:

```text
timestep
stage_id
policy_state_dict
value_state_dict
eval_summary
meta
optimizer_state_dict
```

El `optimizer_state_dict` solo existe en checkpoints generados por
`train_defender_bc.py`. Un checkpoint reconstruido desde ONNX no puede tener
optimizador real.

El `value_state_dict` se incluye para compatibilidad estructural con el pipeline
PPO, aunque el ONNX de inferencia solo use el actor.

## Export ONNX

El export canonico esta en:

```text
export_ppo_defender_obs26_skill_selector_v1.py
```

Ejemplo:

```bash
.venv/bin/python export_ppo_defender_obs26_skill_selector_v1.py \
  --checkpoint runs/defender_bc/distilled_from_recovered_onnx_w20/checkpoint_defender_bc_step500.pt \
  --output runs/defender_bc/distilled_from_recovered_onnx_w20/ppo_defender_distilled_w20.onnx
```

El export genera:

```text
ppo_defender_distilled_w20.onnx
ppo_defender_distilled_w20.json
```

El JSON documenta:

```text
obs_size_input = 26
skill_order completo
active_skills = stand, walk, pass, block
normalization_contract = ppo_obs_v26_hsl2026
pass_commit_mask
engage_mask
```

## Comparacion de checkpoints

Para medir que un nuevo `.pt` no rompio la salida del modelo anterior:

```bash
.venv/bin/python compare_defender_checkpoints.py \
  --reference runs/defender_bc/recovered_from_sabana_onnx/checkpoint_defender_bc_recovered_full_from_onnx.pt \
  --candidate runs/defender_bc/distilled_from_recovered_onnx_w20/checkpoint_defender_bc_step500.pt \
  --samples 8192 \
  --seed 260612 \
  --output runs/defender_bc/distilled_from_recovered_onnx_w20/compare_vs_recovered.json
```

Metricas relevantes:

```text
skill_argmax_match
logit_mse_clamped
logit_max_abs_clamped
param_mse
param_mean_abs
param_max_abs
```

Una corrida conservadora contra el checkpoint recuperado dio:

```text
skill_argmax_match:    0.981689453125
logit_mse_clamped:     0.0007899581687524915
logit_max_abs_clamped: 0.18683338165283203
param_mse:             0.00004650858318200335
param_mean_abs:        0.004682488739490509
param_max_abs:         0.03647419065237045
```

## Criterio de uso

Para replicar el comportamiento que ya estaba en SabanaHerons, usar el ONNX y
JSON integrados en:

```text
SabanaHerons2026/Config/NeuralNets/RLPolicy/
```

Para continuar entrenamiento o mejorar el defensor, partir del `.pt` entrenado
mas reciente, no del ONNX:

```text
runs/defender_bc/distilled_from_recovered_onnx_w20/checkpoint_defender_bc_step500.pt
```

Mantener la referencia al recuperado durante las primeras iteraciones permite
controlar cuanto se mueve el policy antes de promover un nuevo ONNX.
