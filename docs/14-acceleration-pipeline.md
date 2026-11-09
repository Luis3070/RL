# Acceleration Pipeline

## Estado actual: un learner con N=6 visible headless

El camino de entrenamiento de producción queda centrado en un único proceso Python con un único learner MAPPO/PPO y seis slots SimRobot visibles:

```text
train.py
  -> VisibleParallelRuntime
      -> Xvfb :411 -> SimRobot player=1 -> shm.players[0]
      -> Xvfb :412 -> SimRobot player=2 -> shm.players[1]
      -> Xvfb :413 -> SimRobot player=3 -> shm.players[2]
      -> Xvfb :414 -> SimRobot player=4 -> shm.players[3]
      -> Xvfb :415 -> SimRobot player=5 -> shm.players[4]
      -> Xvfb :416 -> SimRobot player=6 -> shm.players[5]
  -> ParallelSlotSimRobotEnv
  -> skrl MAPPO
```

Esto conserva una propiedad importante: el learner ve un solo entorno PettingZoo multi-agente con seis agentes (`player_1` a `player_6`). Para `skrl`, esto no es un vector env de seis mundos; es un mundo multi-agente con `num_envs=1`.

## Por que visible headless

`simrobot_visible` lanza SimRobot como subproceso separado. Eso permite que Qt/OpenGL tengan un display real, aunque sea virtual, y que B-Human ejecute el pipeline natural de camara:

- `CameraImage`
- `BallPerceptor`
- `BallStateEstimator`
- `SelfLocator`
- `LogDataProvider`

El resultado es que la observacion contiene incertidumbre real: `time_since_ball_seen`, `seenPercentage`, estimacion de pelota y localizacion ya no son equivalentes a oracle. En cambio, el modo embebido headless no tiene contexto GL y tiende a usar representaciones corregidas/ground truth.

### Como se activa el render en el perfil 3d_visible

`SimRobotHost.cpp` revisa la variable de entorno `PYBH_SIMROBOT_ENABLE_IMAGE`. Si esta presente, se llama `qunsetenv("PYBH_SIMROBOT_HEADLESS")`, lo que deja que Qt active su pipeline de imagen normal. Si no esta presente, se fuerza `PYBH_SIMROBOT_HEADLESS=1` y el render es opaco.

`VisibleParallelRuntime` no necesita hacer nada extra — el perfil `3d_visible` usa la escena `RL1v0_3D_visible.ros2` que llama al `.con` con `ci 30` (ver mas abajo).

## Que hace Xvfb

Xvfb crea un servidor X11 en memoria. SimRobot cree que esta renderizando en un display normal, pero no necesita una ventana visible ni monitor fisico. En este proyecto se usa para que cada instancia SimRobot tenga su propio display:

```text
DISPLAY=:411  player 1
DISPLAY=:412  player 2
...
DISPLAY=:416  player 6
```

Con `LIBGL_ALWAYS_SOFTWARE=1` y `GALLIUM_DRIVER=llvmpipe`, el render cae en Mesa/CPU. Esto evita mezclar el proceso Python/torch/CUDA con Qt/xcb/OpenGL, que fue una fuente de segfaults en el modo embebido.

## Shared memory y slots

El bridge C++ usa un segmento POSIX shm con seis slots:

```text
/dev/shm/pybh_rl_shared_state_v3
  players[0] -> playerNumber=1
  players[1] -> playerNumber=2
  players[2] -> playerNumber=3
  players[3] -> playerNumber=4
  players[4] -> playerNumber=5
  players[5] -> playerNumber=6
```

Cada slot tiene mutex, semaforo, obs, accion, reset y estado dinamico propios. Por eso seis procesos SimRobot pueden coexistir en el mismo shm sin pisarse, siempre que cada instancia arranque con `playerNumber` distinto.

`VisibleParallelRuntime` crea una copia temporal de `Config/` por slot y reescribe `settings.cfg` para asignar el `playerNumber` correcto. Tambien reescribe el nombre del robot en la escena visible para que cada proceso controle `robot1`, `robot2`, etc.

## Lectura rapida

Si alguien quiere saber en 30 segundos si el pipeline visible `N=6` esta sano, estas son las primeras metricas que debe mirar:

| Metrica | Que responde | Lectura sana actual |
|---|---|---|
| `ball_trusted_rate` | Cuanto tiempo el balon parece natural y fresco | Idealmente `> 0.55`; evals sanas recientes `~0.59-0.98` |
| `ball_rescued_rate` | Cuanto depende el rollout de correccion del bridge | Idealmente `< 0.40`; `FAIL` si domina el rollout |
| `misleading_seen_synthetic_rate` | Cuantas veces la pelota “parece vista” pero ya era sintetica/corregida | Debe mantenerse baja; si sube, el learner puede sobreajustarse a una pelota artificial |
| `ball_confidence_score_mean` | Score continuo de salud perceptual del balon | Idealmente `>= 0.50`; evals sanas recientes `~0.53-0.72` |

Lectura practica:

- `PASS`: `ball_trusted_rate` alto, `ball_rescued_rate` bajo, `ball_confidence_score_mean` estable
- `WARN`: el rollout sigue siendo util, pero el balon esta entrando demasiado a rescate
- `FAIL`: la semantica del balon ya no es representativa para entrenar con confianza

## Configuracion plug and play

`RL/train.py` queda listo para correr desde el IDE:

```text
mode = train
backend = simrobot_visible
simrobot_profile = 3d_visible
parallel_player_numbers = (1, 2, 3, 4, 5, 6)
```

No se requiere lanzar Xvfb manualmente ni abrir SimRobot desde terminal. El runtime crea displays, lanza SimRobot, espera inicializacion y al terminar cierra procesos y carpetas temporales.

## LP_NUM_THREADS: como se controla la sobresuscripcion de render

Mesa llvmpipe lanza por defecto `os.cpu_count()` hilos de render por instancia. En una maquina con 32 CPUs logicos, eso son 32 hilos por instancia; con seis instancias son 192 hilos compitiendo por 32 CPUs — 6× sobresuscripcion.

El resultado sin control es que cada instancia lleva un 60-80% de CPU en contention de render, el GIL de Python se disputa con los hilos de llvmpipe y el throughput cae a ~5 joint-steps/s con N=6 (peor que N=1 absoluto).

La variable de entorno `LP_NUM_THREADS` limita los hilos de render por proceso. `VisibleParallelRuntime` la calcula automaticamente:

```python
def _compute_lp_num_threads(n_slots, *, python_reserved_cores=4):
    total_logical = os.cpu_count() or 4
    available = max(total_logical - python_reserved_cores * 2, total_logical // 2)
    return max(2, available // n_slots)
```

En la maquina de entrenamiento (32 logicos, 16 fisicos, 4 nucleos reservados para Python/torch/CUDA):

```
available = max(32 - 8, 16) = 24
LP_NUM_THREADS = max(2, 24 // 6) = 4
```

Invariante de diseno: `n_slots × LP_NUM_THREADS ≈ available_cpus` para cualquier N.

### Por que taskset no funciona aqui

Pinear cada instancia SimRobot a un subconjunto de CPUs con `taskset` fue probado (Option A de CPU scheduling) y resulto en 2× MAS LENTO en la practica:

- llvmpipe sigue lanzando `cpu_count()=32` hilos de render aunque el proceso este pinado a 4 CPUs.
- 32 hilos en 4 CPUs = 8× sobresuscripcion dentro del conjunto pinado.
- El OS hace context-switch agresivo dentro del conjunto y la latencia de shm aumenta.

`cpu_affinity` queda en `False` por defecto en `VisibleParallelRuntime`. El parametro existe para experimentos pero no debe habilitarse a menos que se cambie la arquitectura de render.

### Tasa de imagen: ci 30

La escena `RL1v0_3D_visible.con` usa `ci 30` (imagen cada 30 pasos de sim, ~33ms sim-time). Valores mas bajos como `ci 15` causaban timeouts de `obs_timeout_ms=1000` intermitentes. `ci 30` es el minimo seguro medido.

## Throughput medido con N=6 y LP_NUM_THREADS=4

Medicion limpia: pasos 128→256 (excluye arranque y primer ciclo de eval).

| Metrica | Valor |
|---|---|
| joint-steps/s (un paso sobre todos los slots) | 22.1 |
| agent-steps/s (× 6 slots) | 132 |
| wall-clock por joint-step | ~45ms |
| speedup vs N=1 (en agent-steps/s) | 4.3× |

Comparativa:

```text
N=1  baseline               :  30.9 joint-steps/s  =  30.9 agent-steps/s
N=6  sin LP_NUM_THREADS     :   ~5  joint-steps/s  =  ~30  agent-steps/s  (peor que N=1)
N=6  LP_NUM_THREADS=4       :  22.1 joint-steps/s  =  132  agent-steps/s  (4.3× N=1)
```

El paso sequencial por Python (GIL) impone un techo: con 6 slots el loop tarda `6 × sem_wait` en secuencia. La ganancia real viene de que llvmpipe con 4 hilos deja los CPUs libres para que Python avance rapidamente al siguiente slot.

### Overhead de evaluacion

El ciclo de evaluacion consume ~38% del wall-clock total. Con `eval_interval=5000` y `eval_episodes=4`:

- 4 episodios × ~160 pasos c/u = ~640 pasos de eval por ciclo
- 640 pasos a 22 joint-steps/s ≈ 29s de eval cada 5000 pasos de entrenamiento

Esto se ve en runs cortas (train_steps=32 devuelve en ~30s total, no ~1s).

## Tiempo estimado de entrenamiento

Con la configuracion de produccion (`trainer_timesteps=20000`):

| Componente | Pasos | Tiempo |
|---|---|---|
| Entrenamiento puro | 20000 @ 22.1 joint-steps/s | ~15 min |
| Overhead evaluacion (38%) | — | +9 min |
| **Total estimado** | — | **~24 min** |

Para un run de exploracion de 5000 pasos: ~6 min.

## Sanidad N=6 por slot (diag_n6_slots.py)

Resultado del diagnostic de 120 steps sobre los 6 slots con `LP_NUM_THREADS=4`:

```text
WIRE SANITY (tasa por slot)
metric                               p1    p2    p3    p4    p5    p6
bridge_obs_ready                   1.00  1.00  1.00  1.00  1.00  1.00
synthetic_pose_observation         0.00  0.00  0.00  0.00  0.00  0.00
synthetic_ball_observation         0.48  0.52  0.43  0.56  0.49  0.51  (varía por visibilidad)
ball_export_fresh                  0.72  0.70  0.75  0.68  0.74  0.71
corrected_exported_ball            0.28  0.30  0.25  0.32  0.26  0.29
```

Invariante verificado: `ball_export_fresh + corrected_exported_ball = 1.00` exacto en todos los slots.

Criterios de PASS:
- `bridge_obs_ready >= 0.99` — PASS todos los slots
- `synthetic_pose_rate <= 0.02` — PASS (0.00 todos)
- `ball_export_fresh >= 0.10` — PASS (~0.70 todos)
- Sin NaN en obs array — PASS
- Sin features inesperadamente congeladas — PASS

## Smoke tests esperados

La sanidad minima antes de entrenar largo:

```bash
RL_TRAIN_STEPS=2 conda run -n RLNAO python RL/train.py
```

Resultado esperado:

- arranca un solo learner
- arranca N=6 SimRobot visibles headless
- `wire=PASS`
- checkpoint unico `checkpoint_stage0_step2.pt`
- no quedan procesos `SimRobot` ni `Xvfb` vivos al finalizar

Metricas relevantes del wire audit:

- `synthetic_pose_rate`: debe ser 0.00 en visible con SelfLocator activo
- `synthetic_ball_rate`: puede variar por visibilidad y warmup, pero no debe dominar rollouts largos
- `ball_export_fresh_rate`: debe ser > 0.10; si es 0.0 hay un bug en `SkillBehaviorControl.cpp` (campo zeroed por memset)
- `ball_stale_rate`: debe mantenerse controlado si la percepcion de pelota funciona
- `ball_seen_percentage`: debe ser variable, no una constante oracle
- `time_since_ball_seen`: debe subir y bajar naturalmente

## Diagnostico del obs space

El observation space actual tiene 23 features. El bloque de pelota/percepcion esta en estos indices:

```text
3  ball_rel_x
4  ball_rel_y
5  ball_end_rel_x
6  ball_end_rel_y
7  ball_vel_x
8  ball_vel_y
9  time_since_ball_seen
10 time_since_ball_disappeared
11 ball_seen_percentage
12 ball_consistent_with_game_state
```

En la comparacion visible N=1 vs visible N=6:

- `obs_shape` se mantuvo en `(23,)`
- `ball_seen_percentage` fue variable
- `time_since_ball_seen` fue variable
- N=6 mostro mas variabilidad entre slots, que es esperable porque cada robot tiene pose, visibilidad y timing propios
- `time_since_ball_disappeared` siempre es 0 por diseno: `BallStateEstimator` reinicia `timeWhenBallFirstDisappeared = theFrameInfo.time` en cada frame que ve la pelota, y ademas se zeroa explicitamente en modo `corrected`

Conclusion: el paralelismo N=6 no cambia el contrato del obs space. Lo que cambia es la distribucion: hay mas diversidad perceptual por slot.

## Semantica del balon: mascaras derivadas en RL

No se modifica nada de la percepcion de B-Human. La capa RL interpreta la calidad del balon con mascaras derivadas calculadas a partir de:

- `synthetic_ball_observation`
- `corrected_exported_ball`
- `ball_seen_percentage`
- `time_since_ball_seen`

Mascaras implementadas:

```text
ball_visible_mask
  seenPercentage > 5 OR timeSinceBallSeen <= 1200ms

ball_trusted_mask
  !synthetic_ball_observation
  AND !corrected_exported_ball
  AND ball_seen_percentage >= 20
  AND time_since_ball_seen <= 600ms

ball_semi_trusted_mask
  !synthetic_ball_observation
  AND !corrected_exported_ball
  AND ball_seen_percentage > 0
  AND time_since_ball_seen <= 1000ms

ball_degraded_mask
  visible pero no trusted, sin rescate

ball_rescued_mask
  synthetic_ball_observation OR corrected_exported_ball

ball_lost_mask
  no visible y no rescatado
```

Ademas se calcula:

```text
ball_confidence_score ∈ [0, 1]
ball_regime ∈ {trusted, semi_trusted, degraded, rescued, lost}
```

Objetivo: distinguir “pelota visible” de “pelota natural y confiable”. Esto evita que el learner interprete `seenPercentage=100` + `timeSinceBallSeen=0` como percepcion perfecta cuando en realidad la pelota ya entro en rescate del bridge.

## Telemetria semantica del balon

Las mascaras se exportan automaticamente a la telemetria existente:

- `smoke_steps.csv`
- `smoke_summary.csv`
- `eval_episode_steps.csv`
- `eval_episodes.csv`
- `eval_summary.csv`

Campos por step:

```text
ball_regime
ball_confidence_score
ball_visible_mask
ball_trusted_mask
ball_semi_trusted_mask
ball_degraded_mask
ball_rescued_mask
ball_lost_mask
```

Campos agregados por episodio/eval:

```text
ball_visible_rate
ball_trusted_rate
ball_semi_trusted_rate
ball_degraded_rate
ball_rescued_rate
ball_lost_rate
misleading_seen_synthetic_rate
ball_confidence_score_mean
first_trusted_ball_after_reset_ms
```

Interpretacion operativa:

- `ball_trusted_rate`: fraccion del rollout donde el balon parece natural y fresco
- `ball_rescued_rate`: cuanto depende el pipeline de la correccion del bridge
- `misleading_seen_synthetic_rate`: casos donde la pelota “parece vista” pero en realidad era sintetica/corregida
- `ball_confidence_score_mean`: score continuo de salud perceptual del balon

## Wire audit semantico del balon

El `wire_audit` ya no mira solo flags crudos del bridge. En `simrobot_visible` tambien revisa la semantica de estas mascaras.

### Umbrales recalibrados

Calibracion actual basada en corridas reales `N=6 visible`:

- evals sanas recientes:
  - `ball_trusted_rate ≈ 0.59 – 0.98`
  - `ball_rescued_rate ≈ 0.00 – 0.35`
  - `ball_confidence_score_mean ≈ 0.53 – 0.72`
- smokes cortos con politica aleatoria o debil pueden caer en rescate muy alto

Contrato actual:

```text
FAIL si:
  ball_visible_rate < 0.90
  OR ball_rescued_rate > 0.75
  OR misleading_seen_synthetic_rate > 0.75

WARN si:
  ball_trusted_rate < 0.55
  OR ball_rescued_rate > 0.40
  OR misleading_seen_synthetic_rate > 0.40
  OR ball_confidence_score_mean < 0.50
```

Lectura:

- `PASS`: percepcion de balon compatible con entrenamiento sano
- `WARN`: el rollout sigue siendo util, pero el balon esta entrando demasiado a rescate o perdiendo confianza
- `FAIL`: la semantica del balon esta demasiado degradada para confiar en esa corrida

### Ejemplos reales

Smoke corto malo:

```text
synthetic_ball_rate = 0.906
ball_rescued_rate = 0.906
misleading_seen_synthetic_rate = 0.906
wire = FAIL
```

Eval corta sana:

```text
ball_trusted_rate = 0.981
ball_rescued_rate = 0.000
misleading_seen_synthetic_rate = 0.000
ball_confidence_score_mean = 0.717
wire = PASS
```

Eval intermedia:

```text
ball_trusted_rate = 0.588
ball_rescued_rate = 0.353
misleading_seen_synthetic_rate = 0.353
wire = PASS con la calibracion actual
```

La conclusion practica es que un smoke de 32 pasos con politica debil no sirve para calibrar solo; la referencia principal debe ser la evaluacion agregada por episodios.

## Por que se retiro 2 x N=6

`2 x N=6` fue util como experimento de throughput, pero en la implementacion directa eran dos procesos Python independientes:

```text
learner A -> shm group 0 -> 6 SimRobots
learner B -> shm group 1 -> 6 SimRobots
```

Eso no es un unico learner. Produce dos entrenamientos separados y checkpoints separados. Para el objetivo actual se retiro del camino ejecutable y el proyecto queda en N=6 con un solo learner.

## Futuro: multiples grupos N=6 para un solo learner

La idea futura correcta no es volver al orquestador de dos entrenadores. La arquitectura objetivo seria:

```text
                 learner central
                       |
        rollout aggregation / IPC / shared replay
          /            |             \
 worker group 0   worker group 1   worker group K
   N=6 slots        N=6 slots        N=6 slots
```

Hay dos rutas tecnicas razonables.

### Ruta A: multiples shm handles en un solo proceso

Extender el binding C++/pybind para que Python pueda abrir mas de un segmento shm al mismo tiempo:

```python
group0 = pybh.open_rl_shared_state("/pybh_rl_group0")
group1 = pybh.open_rl_shared_state("/pybh_rl_group1")
```

Luego el env vectorizado podria leer de `group0.players[0..5]`, `group1.players[0..5]`, etc., y entregar un batch unico al learner.

Ventaja: arquitectura limpia y un solo proceso learner.

Costo: requiere refactor C++ porque hoy el binding se comporta como singleton alrededor de un shm.

### Ruta B: workers externos con IPC

Mantener procesos worker independientes para controlar cada grupo N=6, pero sin entrenar localmente. Cada worker solo recolecta transiciones:

```text
worker -> obs/action/reward/done -> learner central
learner -> action -> worker
```

El learner central mantiene la politica, calcula acciones y actualiza pesos. Los workers solo ejecutan SimRobot y devuelven rollouts.

Ventaja: permite escalar a varios grupos sin tocar tanto el bridge C++.

Costo: hay que disenar IPC, sincronizacion de rollout, broadcast de pesos y control de backpressure.

## Proximo paso: Option A — GIL release en rl_get_obs

El cuello de botella actual es que `pybh.rl_get_obs()` en `Module.cpp` mantiene el Python GIL durante `sem_timedwait`. Eso significa que los 6 slots esperan en serie aunque sus SimRobots hayan publicado obs en paralelo.

Fix: agregar `py::call_guard<py::gil_scoped_release>()` al binding de `rl_get_obs` y mover el loop de slots a threads Python. Con GIL liberado, los 6 `sem_timedwait` podrian resolverse en paralelo.

Impacto estimado: reducir overhead de eval de 38% a ~5%, subir throughput de 132 a ~186 agent-steps/s.

Prerequisito: verificar que el codigo de export de obs dentro de `rl_get_obs` no toca estado Python puro (solo lee shm y construye dict — parece seguro).
