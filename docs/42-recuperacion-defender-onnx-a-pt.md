# Recuperacion corta del defensor ONNX a PT

Este documento resume lo que realmente se hizo para recuperar el `.pt` del
defensor a partir del ONNX que ya estaba en SabanaHerons.

## Punto de partida

Solo estaba disponible el par:

```text
ppo_defender_hsl2026_param_repair.onnx
ppo_defender_hsl2026_param_repair.json
```

El manifest indicaba:

```text
obs26
active_skills = stand, walk, pass, block
obs[23] = pass_armed
obs[24] = engage_armed
obs[25] = pass_arm_progress
```

## Recuperacion

Se inspeccionaron los inicializadores del ONNX y se mapearon a las llaves del
actor PyTorch:

```text
w0, b0, w2, b2, ws, bs, wp, bp
```

Con eso se genero:

```text
runs/defender_bc/recovered_from_sabana_onnx/checkpoint_defender_bc_recovered_full_from_onnx.pt
```

Ese `.pt` recupera los pesos del actor. Tambien se agregaron las llaves extra
que tenia el striker (`log_std_parameter`, `param_low`, `param_high`,
`active_skill_mask`) y un `value_state_dict` inicializado para compatibilidad.

Limitacion: el critic real y el optimizador original no se pueden recuperar
desde ONNX, porque el ONNX solo exportaba inferencia del actor.

## Copia entrenada similar

Luego se entreno una copia partiendo del `.pt` recuperado y usando ese mismo
checkpoint como referencia de distillation:

```text
runs/defender_bc/distilled_from_recovered_onnx_w20/checkpoint_defender_bc_step500.pt
```

Ese checkpoint si tiene:

```text
optimizer_state_dict
policy_state_dict
value_state_dict
eval_summary
meta
```

La comparacion contra el recuperado dio:

```text
skill_argmax_match = 0.981689453125
param_mean_abs = 0.004682488739490509
```

## Como mejorarlo despues

No mejorar desde el ONNX. Mejorar desde:

```text
runs/defender_bc/distilled_from_recovered_onnx_w20/checkpoint_defender_bc_step500.pt
```

Mantener al principio una referencia conservadora:

```bash
--reference-checkpoint runs/defender_bc/recovered_from_sabana_onnx/checkpoint_defender_bc_recovered_full_from_onnx.pt
--distill-weight 5.0
```

Despues bajar gradualmente:

```text
20.0 -> casi copia el ONNX
5.0  -> permite mejoras controladas
1.0  -> permite moverse mas
0.0  -> entrenamiento libre contra dataset/teacher
```

La mejora real debe venir primero de mejores datos y evaluadores defensivos:
mas casos de pass armado, bloqueo frontal, amenaza al arco, balon viejo,
trafico y restricciones de no interferir con el goalkeeper.
