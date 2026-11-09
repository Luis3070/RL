# Handoff: Optimizacion De Eficiencia De Training En Host De 32 Cores

Fecha: 2026-05-19

## Objetivo

Evaluar metodos para mejorar la eficiencia del entrenamiento RL sin:

- bajar `ci 30`
- bajar `eval_episodes`
- degradar la sanidad del contrato de `obs` y `action`

Repos inspeccionados:

- `RL/`
- `../SabanaHerons2026/`

La conclusion corta es esta:

1. El cuello de botella actual sigue siendo el tiempo de espera de observacion visible, no `torch`, no `reward`, ni el encoder Python.
2. La mejora mas importante para un host de `32 cores` no es “bajar mas cosas”, sino corregir como se asigna `LP_NUM_THREADS` y sincronizar mejor la fase de frame entre slots.
3. El siguiente salto importante no esta en micro-optimizaciones de Python, sino en dos cambios estructurales:
   - migrar de `MAPPO + PettingZoo multi-agent artificial` a `PPO vectorizado de un solo agente`
   - escalar mas alla de `N=6` con varios grupos de slots, porque hoy el bridge tiene un limite duro de `6`

## Estado Actual Del Stack

### 1. Que cuenta como “step” hoy

El loop principal de training hace un `wrapped_env.step(...)` por iteracion en `train.py`, asi que:

- `1 train step = 1 joint-step`
- con `6` slots activos, `1 train step = 6 agent-steps`

Evidencia:

- `train.py`: `for timestep in range(timesteps)` y luego `wrapped_env.step(...)`
- `parallel_environment.py`: `num_envs = 1`

Eso explica por que `5000 steps en 10 min` equivale a `8.33 joint-steps/s`, no a `5000 env-steps/s`.

### 2. Donde se va el tiempo hoy

El profiler ya integrado en `environment.py` separa:

- `backend`
- `reward`
- `encode`
- `other`

En la corrida manual observada hoy, el costo dominante estuvo en `backend`, con slots lentos alrededor de `7.6-9.5 steps/s` y un slot rapido cerca de `35 steps/s`. Eso apunta a:

- espera de observacion visible
- variacion de fase entre slots
- contencion de render / CPU

No apunta a:

- reward
- encoding
- forward de la policy

### 3. `rl_get_obs` ya libera el GIL

Una nota importante: la recomendacion vieja de “agregar `gil_scoped_release()` a `rl_get_obs`” ya no es una accion pendiente. En el codebase actual, `rl_get_obs` ya libera el GIL durante el wait bloqueante y solo lo recupera para construir el `py::dict`.

Evidencia:

- `../SabanaHerons2026/Src/Libs/Python/Controller/Module.cpp`

Esto importa porque parte de la documentacion vieja en `docs/14-acceleration-pipeline.md` quedo desactualizada en ese punto. El cuello no desaparecio, pero ya no es “falta liberar el GIL”; ahora el foco debe estar en render-thread budgeting, phase alignment y arquitectura del env.

### 4. El training paralelo actual es “falso multi-agent”

Hoy el paralelismo se modela como un `PettingZoo ParallelEnv` con `possible_agents = player_1..player_6`, pero en realidad son `6` replicas independientes del mismo problema monoagente.

Señales de esto:

- `parallel_environment.py` expone `num_envs = 1`
- `parallel_environment.py` construye `state()` como el promedio de estados de los slots
- `train.py` crea un `MAPPO` con modelos compartidos por agente y un `RandomMemory` por slot-agente

Eso funciona, pero no es la representacion mas eficiente ni la mas limpia del problema.

### 5. El bridge visible sigue limitado a `6` slots por grupo

El shared state C++ tiene un limite duro de `6` players:

- `../SabanaHerons2026/Src/Libs/Python/Controller/RLSharedState.h`

Entonces, en la arquitectura actual:

- un solo grupo visible no puede pasar de `N=6`
- para usar bien una maquina de `32 cores`, el siguiente escalon serio es `2 x N=6` o un refactor del bridge

### 6. La escena visible corre un pipeline perceptual real y pesado

La escena `RL1v0_3D_visible.ros2` usa el escenario `4v4_RL3D`, cuyo `threads.cfg` incluye, entre otros:

- `BallPerceptor`
- `SelfLocator`
- `ObstacleModel`
- `ExpectedGoals`
- `PathPlanner`
- `SkillBehaviorControl`
- `Motion`

Eso es bueno para sanidad del contrato RL, porque el obs no viene de un oracle. Pero tambien significa que cualquier mejora seria debe respetar que este stack es costoso por diseño.

## Ranking De Metodos Evaluados

## Metodo 1: Corregir `LP_NUM_THREADS` Para Que Use Nucleos Fisicos, No Logicos

### Diagnostico

La heuristica actual de `VisibleParallelRuntime` usa `os.cpu_count()`:

- `visible_parallel_runtime.py`

Eso fue calibrado en una maquina donde el documento hablaba de `32 logical CPUs`, no de `32 physical cores`.

En un host moderno de `32 cores / 64 threads`, la heuristica actual tenderia a sobreasignar llvmpipe:

- `total_logical = 64`
- `available = max(64 - 8, 32) = 56`
- `LP_NUM_THREADS = 56 // 6 = 9`

Eso implicaria:

- `6 * 9 = 54` threads de render

Para llvmpipe, eso es agresivo. Render por software suele escalar mucho mejor con presupuesto por nucleo fisico que por SMT/hyperthread.

### Evaluacion

Este es, probablemente, el ajuste con mejor relacion impacto/riesgo para tu host objetivo.

No cambia:

- `obs`
- `action`
- `ci30`
- `eval_episodes`

Solo cambia cuantas hebras de render compiten por CPU.

### Recomendacion

Cambiar `_compute_lp_num_threads(...)` para que estime presupuesto sobre nucleos fisicos, no logicos.

Opciones razonables:

1. Heuristica simple:
   - si `os.cpu_count()` es par y la topologia parece SMT simetrico, usar `physical = logical // 2`
2. Heuristica robusta:
   - leer `/sys/devices/system/cpu/*/topology/thread_siblings_list`
   - contar siblings unicos
   - derivar `physical_cores`

### Punto de partida para benchmark en host de 32 cores

Si el host es:

- `32 physical / 64 logical`: empezar probando `LP_NUM_THREADS` manual en `{4, 5, 6}`
- `16 physical / 32 logical`: empezar en `{3, 4, 5}`

No empezaria en `9` por slot salvo que los perfiles reales lo justifiquen.

### Impacto esperado

Alto en throughput de entorno visible.

### Riesgo para sanidad

Bajo.

## Metodo 2: Activar `RL_PHASE_SYNC` Y Corregir El Periodo De Frame A ~33 ms

### Diagnostico

`parallel_environment.py` ya tiene soporte para phase sync:

- `RL_PHASE_SYNC`
- `RL_PHASE_T_FRAME_MS`
- `RL_PHASE_T_LEAD_MS`

Pero el default actual del frame period es `30 ms`, mientras que en el lado SimRobot visible, el comentario real del `.con` deja claro que con `ci 30` el delay efectivo es aproximadamente `33 ms`:

- `../SabanaHerons2026/Config/Scenes/RL1v0_3D_visible.con`
- `../SabanaHerons2026/Src/Libs/SimulatedNao/LocalConsole.cpp`

Eso significa que la sincronizacion ya existe, pero arranca con un periodo por defecto ligeramente optimista.

### Evaluacion

El log observado hoy muestra mucha dispersion entre slots:

- algunos alrededor de `8-9 steps/s`
- uno alrededor de `35 steps/s`

Esa dispersion es consistente con slots cayendo en fases distintas del ciclo de camara.

### Recomendacion

Activar phase sync por defecto para visible parallel training y ajustar:

- `RL_PHASE_T_FRAME_MS=33`
- `RL_PHASE_T_LEAD_MS` en rango `2-4`

No haria estimacion dinamica agresiva del frame period a partir del return time. La observacion del comentario de `parallel_environment.py` sigue siendo correcta: el retorno total del step no es el mismo objeto que el periodo real de camara.

### Impacto esperado

Medio a alto.

El throughput promedio puede subir moderadamente, pero el mayor beneficio esperado es:

- reducir variabilidad entre slots
- acercar el throughput conjunto al de los slots rapidos

### Riesgo para sanidad

Bajo.

No cambia el contrato; solo cambia cuando despachas la accion dentro del ciclo visible.

## Metodo 3: Mantener `N=6` Como Baseline De Un Grupo, Pero Cambiar La Heuristica De Escalado Para Host De 32 Cores

### Diagnostico

Con la arquitectura actual, un grupo visible esta limitado a `6` slots por `RLSharedState::MAX_PLAYERS`.

Por eso, en una maquina grande, el problema ya no es “subir de 4 a 6 slots”, sino:

- usar mejor esos `6`
- decidir si vale la pena escalar a mas de un grupo

### Evaluacion

Para un solo grupo:

- `N=6` sigue siendo la configuracion natural
- el margen principal esta en `LP_NUM_THREADS`, `phase sync` y pipeline RL

Para usar mejor `32 cores`, el escalado real aparece al pasar de:

- `1 x N=6`

a:

- `2 x N=6`

pero eso ya no cabe en la arquitectura actual sin refactor.

### Recomendacion

No perder tiempo intentando “N=8” dentro del mismo grupo actual sin antes resolver el limite de shared state.

Si el objetivo es throughput de muestras por segundo en ese host, el orden correcto es:

1. optimizar muy bien `1 x N=6`
2. luego evaluar `2 x N=6`

### Impacto esperado

Alto, pero solo como parte de una estrategia de segundo escalon.

### Riesgo para sanidad

Bajo si se mantiene `N=6` por grupo; medio/alto si se fuerza un `N>6` dentro del bridge actual.

## Metodo 4: Migrar De `MAPPO + PettingZoo` A `PPO Vectorizado Monoagente`

### Diagnostico

El problema actual no es verdaderamente multi-agent durante training:

- cada slot es una replica independiente
- comparten policy
- no hay interaccion estrategica entre “agentes” del batch
- el `state()` usado por MAPPO es un promedio de slots

Eso sugiere que el stack actual esta pagando overhead de framework:

- diccionarios por agente
- memoria por agente
- bookkeeping de MAPPO
- wrappers PettingZoo

sin una ganancia clara de modelado.

### Evaluacion

Este es un cambio de arquitectura con upside real:

- menos overhead Python por step
- batch natural de `num_envs = 6`
- contabilidad mas limpia de throughput
- PPO vectorizado encaja mejor con replicas independientes

Ademas, conceptualmente es mas correcto que tratar replicas independientes como “equipo multi-agent”.

### Recomendacion

Mediano plazo:

1. construir un env vectorizado monoagente con batch `N=6`
2. usar PPO o una variante single-agent vectorized
3. dejar MAPPO solo si mas adelante se entrena control coordinado real entre robots

### Impacto esperado

Medio.

No espero que esto gane lo mismo que `LP_NUM_THREADS`, pero si espero:

- menos overhead de wrapper
- menos churn de diccionarios / tensores
- pipeline mas simple para escalar a varios grupos

### Riesgo para sanidad

Bajo para `obs/action`; medio para entrenamiento por migracion de algoritmo y checkpoints.

## Metodo 5: Crear Un `threads.cfg` Dedicado Para RL Fast-Natural

### Diagnostico

La escena visible actual hereda `4v4_RL3D/threads.cfg`, que trae bastante infraestructura cognitiva y perceptual. Parte es necesaria para mantener un obs natural:

- `BallPerceptor`
- `SelfLocator`
- `ObstacleModel`
- `ExpectedGoals`
- `PathPlanner`
- `SkillBehaviorControl`

Pero no todo parece estrictamente necesario para el contrato RL actual.

Ejemplos de modulos candidatos a poda o revisión:

- `AutomaticCameraCalibrator`
- parte de `TeamMessageHandler`
- `BHumanMessageOutputGenerator`
- trackers globales que no alimenten directo el obs exportado

### Cuidado Importante

No conviene podar a ciegas:

- `SkillBehaviorControl.cpp` exporta `passOptionsCount` a partir de `theTeamData.teammates.size()`
- varios features de obstaculos dependen del pipeline perceptual completo
- el action path usa el comportamiento normal de B-Human y no se debe “acelerar” rompiendo la ruta natural

### Recomendacion

Crear una variante nueva, no tocar directamente `4v4_RL3D`:

- `4v4_RL3D_RLFastNatural`

Proceso recomendado:

1. clonar `threads.cfg`
2. quitar un pequeño grupo de providers
3. correr:
   - `diag_n6_slots.py`
   - `obs_invariants.py`
   - `action_sweep.py`
4. aceptar el cambio solo si:
   - `bridge_obs_ready` sigue sano
   - no aparecen `NaN`
   - `ball_export_fresh` se mantiene
   - no cambia el contrato exportado

### Impacto esperado

Medio a alto, dependiendo de cuanta carga no esencial siga viva hoy.

### Riesgo para sanidad

Medio.

Es una buena ruta, pero requiere disciplina de validacion.

## Metodo 6: Rediseñar `dynamic world updates` Para No Hacer Un Segundo Round-Trip Visible Cada 4 Steps

### Diagnostico

En `stage 3` y `stage 4`, el curriculum mete obstaculos dinamicos cada `4` steps:

- `curriculum.py`

Y en `environment.py`, cada vez que toca update:

1. se hace `backend.step(...)`
2. luego `backend.set_dynamic_world(...)`
3. luego `read_obs()` otra vez

Eso significa que en esos steps estas pagando dos viajes visibles:

- uno para el step normal
- otro para aplicar mundo dinamico y volver a leer obs

### Evaluacion

Esto no afecta la corrida corta de `stage=-1`, pero si afecta el throughput de etapas productivas.

### Opciones

#### Opcion A: Cola de world update para el siguiente step

Se envia el world update al final del step actual, pero no se fuerza `read_obs()` inmediatamente. El obs actualizado llega en el siguiente step.

Ventaja:

- elimina el segundo read en caliente

Desventaja:

- cambia la temporalidad del MDP

#### Opcion B: Unificar accion y world update en un solo request

Agregar una operacion de bridge tipo:

- `rl_set_action_and_dynamic_world(...)`

o una cola de world update consumida antes del siguiente frame, de forma que Python solo espere una vez.

Ventaja:

- mantiene mejor la semantica actual

Desventaja:

- requiere trabajo C++ y Python coordinado

### Recomendacion

Si el throughput de `stage 3/4` resulta claramente peor que `stage -1/0/1/2`, esta es una de las mejores rutas estructurales.

Yo priorizaria la opcion B.

### Impacto esperado

Medio.

### Riesgo para sanidad

Medio.

Hay que validar que el reward y el obs sigan representando la transicion correcta.

## Metodo 7: Escalar A `2 x N=6` En Un Host De 32 Cores

### Diagnostico

El host de `32 cores` probablemente tenga margen para mas de `6` slots efectivos. Pero hoy el codebase no soporta, en un solo learner, multiples grupos de shared memory de forma limpia.

El motivo:

- `RLSharedState` es singleton por proceso
- el shm se selecciona por una sola variable de entorno
- `MAX_PLAYERS = 6`

### Opciones Reales

#### Ruta A: Multiples SHM en un solo proceso

Extender el binding para abrir multiples handles:

- `group0`
- `group1`

Esto es limpio, pero exige refactor serio del bridge C++.

#### Ruta B: Workers externos con IPC

Mantener varios grupos `N=6` como workers y un learner central aparte.

Esto es mas realista como siguiente paso de escalado.

### Evaluacion

Para una maquina de `32 cores`, este es el camino con mayor upside absoluto de muestras por segundo, pero no es el primer cambio a hacer.

Orden correcto:

1. exprimir `1 x N=6`
2. luego pasar a `2 x N=6`

### Impacto esperado

Muy alto.

### Riesgo para sanidad

Bajo para `obs/action` por grupo; alto en complejidad de arquitectura.

## Metodo 8: Reducir Overhead De Wall-Clock Del Learner, Evaluacion Y Telemetria

Este metodo no aumenta el throughput puro del env, pero si puede bajar tiempo total de entrenamiento.

### Subopcion A: Usar GPU para el learner

La corrida manual observada hoy fue con `learner_device=cpu`.

Si el host de entrenamiento tiene GPU utilizable:

- usar `cuda`

Esto no arregla el wait de `read_obs`, pero si reduce:

- costo de updates PPO
- costo de forward cuando el batch crezca

### Subopcion B: Aumentar `rollout_steps`

Hoy el default es `256`.

En simuladores caros, suele ser razonable probar:

- `512`
- `1024`

Eso reduce la frecuencia de updates del learner por muestra recolectada.

No cambia `obs/action`, pero si cambia dinamica de optimizacion. Hay que medir sample efficiency, no solo wall-clock.

### Subopcion C: Subir `eval_interval` y `checkpoint_interval`

No se tocan `eval_episodes`, pero si la frecuencia:

- menos evaluaciones por unidad de wall-clock
- menos serializacion de checkpoints

Esto es especialmente razonable en corridas largas donde el KPI principal es avance de training, no telemetria fina cada `5000` steps.

### Subopcion D: Desactivar `save_plots` en corridas largas

Las graficas y parte de la telemetria usan `pandas/matplotlib` dentro del loop de evaluacion. Eso no afecta sanidad del env y puede recortar wall-clock innecesario.

### Impacto esperado

Medio en wall-clock total.

### Riesgo para sanidad

Bajo.

### Riesgo para aprendizaje

Bajo a medio, dependiendo de `rollout_steps`.

## Metodos Que No Recomiendo Como Primera Ola

### 1. Bajar `ci30` o volver a `ci60`

Rechazado.

Ya hay evidencia de que `ci60` rompe la sanidad del balon y de que `ci15` generaba timeouts intermitentes.

### 2. Activar `cpu_affinity` por defecto

No recomendado.

La propia documentacion y el runtime actual lo dejan apagado porque llvmpipe suele empeorar con pinning estricto.

### 3. Usar escenas oracle o `Fast.con` como reemplazo productivo

No recomendado para training principal.

Sirve para experimentos o diagnósticos, pero rompe el objetivo de preservar sanidad natural de `obs/action`.

### 4. `root assist`

No recomendado para la ruta principal.

Es una palanca de simulacion que cambia la semantica efectiva de accion / locomocion y ya no seria una mejora “limpia” de eficiencia.

## Recomendacion Prioritizada

## Ola 1: Bajo Riesgo, Alto Retorno

1. Hacer `LP_NUM_THREADS` fisico-aware.
2. Activar `RL_PHASE_SYNC=1`.
3. Cambiar default de `RL_PHASE_T_FRAME_MS` de `30` a `33`.
4. Probar `RL_PHASE_T_LEAD_MS` en `2-4`.
5. Correr learner en GPU si el host la tiene.
6. Subir `eval_interval`, `checkpoint_interval` y apagar plots en corridas largas.

## Ola 2: Arquitectura Del Pipeline RL

1. Migrar a env vectorizado monoagente.
2. Medir PPO vectorizado vs MAPPO actual.
3. Validar sample efficiency y throughput conjunto.

## Ola 3: C++ / Escalado Estructural

1. Crear `threads.cfg` RL-fast-natural podado.
2. Coalescer `dynamic world update` con el path normal del step.
3. Diseñar `2 x N=6` con workers o multi-shm.

## Plan De Benchmark Para El Host De 32 Cores

## Fase A: Baseline Sano

Correr:

- `diag_n6_slots.py`
- `train.py` con `RL_PROFILE_STEP=1`

Metrica a guardar:

- `joint-steps/s`
- `agent-steps/s`
- `backend_prof.read_obs` por slot
- dispersion entre slots

## Fase B: Sweep De Render Budget

Probar:

- `LP_NUM_THREADS = 4, 5, 6`

Si el host es `32 physical / 64 logical`, no arrancaria por arriba de `6` sin evidencia.

## Fase C: Sweep De Phase Sync

Probar:

- `RL_PHASE_SYNC = 0/1`
- `RL_PHASE_T_FRAME_MS = 33`
- `RL_PHASE_T_LEAD_MS = 2, 3, 4`

KPI:

- throughput medio
- varianza entre slots

## Fase D: Wall-Clock Total

Con la mejor combinacion de B y C:

- mantener `eval_episodes`
- probar `eval_interval`
- probar `checkpoint_interval`
- probar `save_plots=0`
- probar `learner_device=cuda`

## Fase E: Arquitectura

Comparar:

- stack actual `MAPPO + PettingZoo`
- `PPO vectorizado`

Y medir:

- joint-steps/s
- agent-steps/s
- wall-clock por update
- reward/estabilidad por muestra

## Conclusion

Si el entrenamiento se va a correr en un PC de `32 cores`, mi recomendacion principal no es tocar percepcion ni bajar fidelidad. Es esta:

1. corregir primero la asignacion de threads de llvmpipe para nucleos fisicos
2. sincronizar mejor la fase de frame visible
3. despues simplificar la arquitectura RL para que trate los `6` slots como replicas vectorizadas, no como multi-agent artificial
4. si hace falta mas throughput, escalar a varios grupos `N=6`

El mayor riesgo tecnico en este momento no es “quedarnos cortos de ideas”, sino optimizar el lugar equivocado. En el codebase actual, el mayor retorno sigue estando en:

- budget de render CPU
- phase alignment
- arquitectura del env / learner

No en micro-optimizaciones de reward ni en bajar la fidelidad perceptual.
