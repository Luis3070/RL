# 27 — Estabilidad del Behavioral Evaluator

## Objetivo

Evitar dos errores de lectura:

1. tomar un `4/5` histórico aislado como si fuera un baseline robusto;
2. declarar una regresión del PPO cuando en realidad lo que varía es el runtime
   de `simrobot_visible` en escenarios sensibles a colisión.

El `behavioral_evaluator.py` original sigue siendo el **gate canónico** de una
sola corrida. No se toca. Este documento define el complemento que faltaba:
un **gate de estabilidad multi-seed**.

## Problema detectado

La referencia histórica `behavior_20260605_022521` reportó `4/5` con:

- `finish_band_2p5_3m_clean` -> `PASS`
- `finish_band_2p5_3m_static_lane` -> `PASS`
- `dynamic_full_field_pressure` -> `PASS`
- `blocked_corridor_repair` -> `PASS`
- `preshoot_bad_angle_boundary` -> `FAIL`

Al revalidarlo en esta máquina con el mismo checkpoint y el mismo `seed=4200`,
el resultado cayó a `2/5`:

- `finish_band_2p5_3m_clean` -> `PASS`
- `finish_band_2p5_3m_static_lane` -> `PASS`
- `dynamic_full_field_pressure` -> `FAIL`
- `blocked_corridor_repair` -> `FAIL`
- `preshoot_bad_angle_boundary` -> `FAIL`

La diferencia nueva no estuvo en el finish limpio ni en el static lane, sino en
los escenarios con tráfico:

- `dynamic_full_field_pressure`
- `blocked_corridor_repair`

En ambos, el criterio bloqueante fue `collision_rate<=0.12`.

## Lectura correcta

Eso no prueba automáticamente que el PPO “se dañó”.

Lo que sí prueba es que el gate actual era demasiado frágil para escenarios
dependientes de:

- resets visibles;
- convergencia de pose;
- timing de la escena;
- jitter físico del entorno;
- densidad de obstáculos.

Con 6 u 8 episodios agregados en una sola corrida, el resultado puede moverse
demasiado por seed.

## Estado actual tras la reparación

La revalidación posterior sí dejó un checkpoint mejor candidato:

- `runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt`

Con ese checkpoint:

- el `behavioral_evaluator` canónico volvió a cruzar el gate mínimo con `3/5` en `runs/behavioral_evaluator/behavior_20260608_181747/summary.csv`;
- `dynamic_full_field_pressure` volvió a `PASS`;
- `preshoot_bad_angle_boundary` siguió siendo el fail conocido por `collision_rate`;
- la batería multi-seed parcial en `runs/behavioral_stability/stability_20260608_175828/per_rep_summary.csv` mostró mejora clara en `dynamic` y mejora parcial en `blocked`.

## Criterio nuevo

Se agrega `behavioral_stability_probe.py`, que mantiene el mismo policy path y
los mismos criterios por escenario, pero cambia la forma de muestrear:

1. usa un **seed set fijo y nombrado**;
2. corre varias réplicas del mismo escenario;
3. separa `world_seed` de `reset_seed`;
4. reporta `pass_rate`, no solo `pass/fail` binario de una corrida;
5. guarda CSV por episodio y por réplica para reproducir outliers.

## Seed set estable

Por defecto:

```text
11,23,37,51,67,83
```

La idea no es que esos números tengan magia. La idea es que:

- el set sea fijo;
- sea pequeño pero suficiente para distribución;
- se pueda rerunear tal cual en otra máquina;
- se use siempre el mismo conjunto al comparar checkpoints.

## Diseño de reproducibilidad

Para cada `stable_seed` y cada episodio:

- `world_seed = stable_seed * 100000 + situation_index * 1000 + episode_index`
- `reset_seed = stable_seed * 100000 + 50000 + situation_index * 1000 + episode_index`

Eso evita un problema clásico del evaluator viejo:

- antes, el RNG del mundo avanzaba en cadena y dependía del orden de iteración;
- ahora, cada episodio queda direccionable de forma independiente.

Si sale un outlier en colisión, ya no hay que “adivinar” cómo repetirlo.

## Qué mirar para `dynamic` y `blocked`

Los campos principales son:

- `pass_rate`
- `collision_rate_mean`
- `collision_rate_max`
- `clean_episode_rate_mean`
- `dynamic_obstacle_success_mean`
- `corridor_keep_rate_mean`
- `top_failures`

Interpretación:

- si `pass_rate` es bajo y `top_failures` está dominado por `collision_rate<=0.12`,
  el problema sigue siendo robustez frente a tráfico, no definición del gate;
- si `pass_rate` es alto pero hay 1 seed malo, estás viendo un outlier, no una
  regresión general;
- si `pass_rate` es bajo y además caen `corridor_keep_rate` y
  `dynamic_obstacle_success`, el policy realmente está entrando peor al carril.

## Comando recomendado

Corrida de estabilidad enfocada en los dos escenarios que hoy son dudosos:

```bash
env PYBH_RL_SHM_NAME=/pybh_rl_stability \
xvfb-run -a ./scripts/run_with_validation_env.sh \
python3 behavioral_stability_probe.py \
  --checkpoint runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt \
  --episodes 4 \
  --scenarios dynamic_full_field_pressure,blocked_corridor_repair \
  --seed-list 11,23,37,51,67,83 \
  --pass-rate-threshold 0.75 \
  --max-episode-steps 1000
```

## Recomendación operativa

Para transplante no basta con decir:

```text
hubo una vez un 4/5
```

La lectura sana pasa a ser:

1. `behavioral_evaluator.py` en seed canónico `4200` sobre el checkpoint candidato vigente;
2. `behavioral_stability_probe.py` sobre seed set fijo;
3. si `dynamic`/`blocked` son inestables, no se declara baseline congelado;
4. primero se estabiliza el behavior gate y luego se transplanta.

## Qué NO hace este probe

- no cambia el checkpoint;
- no afloja los thresholds de escenario;
- no maquilla fallos de colisión;
- no reemplaza el gate canónico histórico.

Solo evita que una sola corrida afortunada se confunda con estabilidad real.
