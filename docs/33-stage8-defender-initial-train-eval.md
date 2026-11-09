# Stage 8 Defender: evaluacion del train inicial

Fecha: 2026-06-13.

Run evaluado:
`runs/defender_stage8_v4_defreward_single_dynamic`

Telemetria:
`runs/defender_stage8_v4_defreward_single_dynamic/telemetry/multiagent_stage8_finetune_train_stage8_20260613_111459`

## Configuracion

- Stage: `8`
- Backend: `simrobot_visible`
- Escena: `RLMAStage7_3D`
- RL players: `1`
- Steps: `12000`
- Eval/checkpoint interval: `2000`
- Eval episodes: `8`
- Warm-start: `runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt`
- Dataset replay: `runs/teacher_multiagent_dataset/dataset_attack_plus_defense_stage8_v1/samples.npz`
- Dynamic world: activo
- Velocidades usadas: `RL_MA_STAGE8_OPP_SPEED_MM_S=120`,
  `RL_MA_STAGE8_BALL_SPEED_MM_S=40`

## Resultado por checkpoint

| checkpoint | reward | defender_score | goal_side | cover | danger_cover | corridor_mm | collision | stability | wire |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `checkpoint_stage8_step2000.pt` | 0.231 | 0.897 | 1.000 | 0.872 | 0.865 | 290 | 0.375 | 0.600 | WARN |
| `checkpoint_stage8_step4000.pt` | 0.750 | 0.767 | 0.954 | 0.554 | 0.594 | 340 | 0.000 | 0.666 | PASS |
| `checkpoint_stage8_step6000.pt` | 0.490 | 0.822 | 1.000 | 0.716 | 0.730 | 217 | 0.375 | 0.601 | WARN |
| `checkpoint_stage8_step8000.pt` | 0.623 | 0.757 | 0.898 | 0.621 | 0.672 | 252 | 0.250 | 0.619 | PASS |
| `checkpoint_stage8_step10000.pt` | 0.729 | 0.655 | 0.867 | 0.464 | 0.513 | 518 | 0.375 | 0.616 | WARN |
| `checkpoint_stage8_step12000.pt` | 0.082 | 0.852 | 1.000 | 0.800 | 0.812 | 279 | 0.500 | 0.650 | WARN |

## Recomendacion

Usar `checkpoint_stage8_step4000.pt` como candidato inicial. No es el checkpoint
con mayor `defender_score`, pero es el mejor punto operativo porque:

- fue `wire=PASS`;
- tuvo `collision=0.000`;
- tuvo `policy_incompat=0.001`;
- mantuvo reward positivo en todos los escenarios evaluados;
- no muestra el deterioro por colisiones que aparece en `step6000`,
  `step10000` y `step12000`.

`checkpoint_stage8_step8000.pt` queda como candidato secundario: tambien pasa
wire audit y tiene mas cobertura que `step4000`, pero ya toca el limite de
colisiones (`0.250`) y tiene menor `defender_score`.

## Lectura tecnica

El entrenamiento inicial ya aprende senales defensivas medibles, pero todavia no
conviene declararlo final. El `defender_score` alto de `step2000` y `step12000`
viene con colisiones, lo que en SimRobot visible suele significar que el modelo
esta encontrando posiciones defensivas pero no siempre de forma fisicamente
estable. Para continuar, conviene arrancar desde `step4000` y empujar cobertura
defensiva sin sacrificar collision/wire audit.

En este run no se observa uso de `pass` ni `block` ejecutados en el agregado
final (`executed_pass_rate=0.000`, `executed_block_rate=0.000`). Eso es
esperado para el primer fine-tune si los gates no abren durante las ventanas de
evaluacion; no debe interpretarse como que el contrato no soporte esas skills.
La siguiente fase debe validar escenarios donde el gate defensivo de block/pass
se active con mas frecuencia.

## Siguiente fase

1. Re-evaluar `checkpoint_stage8_step4000.pt` multi-seed.
2. Validar que el floor ofensivo stage 7 no haya caido.
3. Continuar fine-tune desde `step4000` con mas presion de cobertura defensiva,
   manteniendo `collision_rate <= 0.25` y preferiblemente `wire=PASS`.
4. Solo exportar ONNX cuando un checkpoint mantenga estabilidad y cobertura en
   evaluaciones repetidas.
