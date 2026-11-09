# Stage 8 Defender: SimRobot visible single-process

## Contexto

Stage 8 es el curriculum defensivo con 47 observaciones para tres field players
RL y goalkeeper B-Human. Aunque el objetivo final es volver a correr varios
SimRobot visibles en paralelo, la ruta de depuracion estable queda en un solo
SimRobot visible. Primero se valida escena, reset, teacher y dataset; solo
despues se aumenta el numero de slots.

Archivos relevantes:

- `teacher_multiagent_dataset.py`
- `visible_parallel_runtime.py`
- `curriculum.py`
- `production_curriculum.py`
- `multiagent_teacher.py`
- `SabanaHerons2026/Config/Scenes/RLMAStage6_3D.*`
- `SabanaHerons2026/Config/Scenes/RLMAStage7_3D.*`

## Regla Qt / Sandbox

Todo lo que abra Qt, Xvfb o SimRobot visible debe correr fuera del sandbox de
Codex. Dentro del sandbox, Xvfb puede fallar con errores como:

```text
qt.qpa.xcb: could not connect to display
Cannot establish any listening sockets
Owner of /tmp/.X11-unix should be set to root
```

Si aparece eso, no es un bug del teacher ni de `RLSharedState`; es el entorno de
ejecucion. Repetir el smoke fuera del sandbox.

## Binarios alineados

Si aparece un request visible corrupto, por ejemplo:

```text
Visible SimRobot world request 1070137351 failed
```

o IDs enormes despues del primer request, revisar que estos binarios hayan sido
recompilados en la misma rama:

- `Make/Python/pybh/controller*.so`
- `Build/Linux/SimRobot/Develop/libSimulatedNao.so`
- `Build/Linux/SimRobot/Develop/SimRobot`

Comandos usados:

```bash
cd 
env CCACHE_DIR=/tmp/ccache-hsl cmake --build Build/Linux/CMake/Develop --target SimRobot SimulatedNao -j2
env CCACHE_DIR=/tmp/ccache-hsl cmake --build Build/Linux/Python --target PythonController -j2
cp Build/Linux/Python/Release/controller.cpython-312-x86_64-linux-gnu.so Make/Python/pybh/controller.cpython-312-x86_64-linux-gnu.so
```

Si hay procesos viejos, cerrar SimRobot/Xvfb y usar un `PYBH_RL_SHM_NAME` nuevo.
Limpiar `/dev/shm/pybh_rl_shared_state_v3` tambien ayuda cuando se esta usando el
nombre default.

`teacher_multiagent_dataset.py` ahora crea automaticamente un nombre fresco en
`simrobot_visible` si no existe `PYBH_RL_SHM_NAME`, tanto en single-SimRobot como
en parallel slots. En el log debe aparecer algo como:

```text
[ma-teacher] visible_parallel_slots=off shm=/pybh_rl_ma_teacher_stage8_145931
```

Si no aparece una linea `shm=...`, revisar que se este ejecutando el script
actual y no una copia vieja.

## Escenas Stage 8

`RLMAStage7_3D` es la escena visible usada para stages 7 y 8. Debe tener:

- escenario `4v4_RL3D`;
- robots reales `robot1`, `robot2`, `robot3`, `robot4`;
- oponentes reales `robot21`, `robot22`, `robot23`;
- `.con` con `call Includes/RLNatural`;
- `ci 30`;
- `gc playing`.

No usar `3v3_Scaled` para esta ruta visible. En las pruebas caia con:

```text
Behavior.cpp:559: ASSERT(remainingAgents.size() <= positionSubsets->size()) failed
```

El include `Includes/RLNatural` y `ci 30` deben mantenerse alineados con la
escena visible estable `RL1v0_3D_visible`.

## Single SimRobot por defecto

Para stage 8, `teacher_multiagent_dataset.py` deja por defecto:

```text
RL_MA_TEACHER_VISIBLE_PARALLEL_SLOTS=0
```

Eso significa:

- un solo proceso SimRobot visible;
- player 1 hace el reset fisico de todo el mundo;
- players 2 y 3 leen/actuan por el mismo `RLSharedState`;
- el filtro `reset_ball_drift` solo valida la pelota del primary.

Esto evita depurar a la vez tres problemas distintos: Qt, escena y convergencia
fisica independiente por slot.

## Comando base

```bash
cd 
source .venv/bin/activate

PYTHONPATH=/Make/Python: \
LD_LIBRARY_PATH=/Util/onnxruntime/lib/Linux:$LD_LIBRARY_PATH \
RL_GATE_SHOOT_OPEN_ENTER=0.30 \
RL_GATE_SHOOT_OPEN_EXIT=0.20 \
RL_MA_TEACHER_STAGE=8 \
PYBH_RL_SHM_NAME=/pybh_rl_stage8_single_1 \
python teacher_multiagent_dataset.py
```

Para correr solo un escenario, usar `RL_MA_TEACHER_SCENARIOS`. Esto filtra la
lista real de escenarios antes de arrancar SimRobot:

```bash
RL_MA_TEACHER_STAGE=8 \
RL_MA_TEACHER_SCENARIOS=transition_midfield \
RL_MA_TEACHER_EPISODES=8 \
RL_MA_TEACHER_MAX_STEPS=500 \
RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=6 \
python teacher_multiagent_dataset.py
```

El log correcto debe decir:

```text
[ma-teacher] 1 scenarios for stage 8:
  - transition_midfield (...)
```

Para smoke rapido:

```bash
PYTHONPATH=/Make/Python: \
LD_LIBRARY_PATH=/Util/onnxruntime/lib/Linux:$LD_LIBRARY_PATH \
RL_GATE_SHOOT_OPEN_ENTER=0.30 \
RL_GATE_SHOOT_OPEN_EXIT=0.20 \
RL_MA_TEACHER_STAGE=8 \
RL_MA_TEACHER_EPISODES=1 \
RL_MA_TEACHER_MAX_STEPS=80 \
RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=1 \
PYBH_RL_SHM_NAME=/pybh_rl_stage8_smoke_1 \
python teacher_multiagent_dataset.py
```

Para validar que el pipeline escribe filas sin que los filtros de calidad tapen
el resultado:

```bash
PYTHONPATH=/Make/Python: \
LD_LIBRARY_PATH=/Util/onnxruntime/lib/Linux:$LD_LIBRARY_PATH \
RL_GATE_SHOOT_OPEN_ENTER=0.30 \
RL_GATE_SHOOT_OPEN_EXIT=0.20 \
RL_MA_TEACHER_STAGE=8 \
RL_MA_TEACHER_EPISODES=1 \
RL_MA_TEACHER_MAX_STEPS=80 \
RL_MA_TEACHER_MAX_ATTEMPT_MULTIPLIER=1 \
RL_MA_TEACHER_MIN_OPEN_HOLD_STEPS=0 \
RL_MA_TEACHER_MIN_TRIANGLE_HOLD_STEPS=0 \
RL_MA_TEACHER_MAX_ROLE_SWAPS=999 \
RL_MA_TEACHER_COLLISION_SUSTAIN_STEPS=999 \
PYBH_RL_SHM_NAME=/pybh_rl_stage8_relaxed_1 \
python teacher_multiagent_dataset.py
```

Evidencia de validacion local:

- salida: `runs/teacher_multiagent_dataset/dataset_stage8_20260612_211843`
- filas: `1611`
- `obs_size=47`
- `positive_weight_gate_label_consistent=true`
- `positive_weight_role_mask_consistent=true`

## Interpretacion de fallos

`rl_set_visible_action` no existe:

- El binding Python esta viejo.
- Recompilar `PythonController` y copiar el `.so` a `Make/Python/pybh`.

Request visible con ID gigante:

- Probable desalineacion `controller.so` vs `libSimulatedNao.so`, o shared memory
  viejo.
- Recompilar SimRobot/SimulatedNao/PythonController y cambiar `PYBH_RL_SHM_NAME`.
- En stage 8 single-SimRobot, el script actual debe autogenerar ese nombre; si el
  log no muestra `visible_parallel_slots=off shm=/pybh_rl_ma_teacher_stage8_...`,
  se esta corriendo codigo viejo o un wrapper que pisa el entorno.

`Timed out waiting for visible SimRobot world request 1`:

- Si ocurre dentro del sandbox, primero sospechar Qt/Xvfb.
- Revisar logs en `/tmp/pybh_visible_parallel_*/player_*/simrobot.log`.
- Si el log muestra assert de Behavior, revisar escenario/tactica de la escena.

`RL reset could not find teammate robot 1` en modo parallel:

- La escena multi-robot fue renombrada como si fuera 1v0.
- `visible_parallel_runtime.py` no debe renombrar `robot1` cuando la escena ya
  contiene `robot2`, `robot3` o `robot4`.

`reset_ball_drift` con single-SimRobot:

- No validar la pelota de todos los agentes al reset.
- En single-SimRobot hay una sola pelota fisica; validar solo la observacion del
  primary.

`*_hold_too_short`, `role_thrash`, `teammate_collision`:

- Ya no son fallos de infraestructura.
- Son filtros de calidad del dataset. Para smoke se pueden relajar; para dataset
  real deben revisarse segun el objetivo defensivo.

`stale visible frames` despues de un assert de WalkingEngine:

- El SimRobot visible puede dejar de producir frames para un jugador.
- El collector actual descarta ese episodio como `backend_step_error` y continua.
- Tambien escribe `samples.partial.npz` y `summary.partial.json` despues de cada
  escenario, asi que una caida ya no pierde todo el chunk.

## Bootstrap desde runs utiles

Si ya existen runs completos/parciales utiles, se puede construir un dataset
stage 8 balanceado sin volver a simular todo:

```bash
python build_stage8_defense_dataset.py \
  --output-dir runs/teacher_multiagent_dataset/dataset_defense_stage8_bootstrap_v1 \
  --min-transition-rows 9000 \
  --min-block-frac 0.15

python teacher_multiagent_dataset_audit.py \
  runs/teacher_multiagent_dataset/dataset_defense_stage8_bootstrap_v1 \
  --min-striker-shoot-armed 0 \
  --min-striker-shoot-frac 0 \
  --min-striker-pass-armed 0 \
  --write
```

Este builder usa `samples.npz/summary.json` y, si existen, tambien
`samples.partial.npz/summary.partial.json`. No inventa observaciones nuevas;
hace bootstrap/oversampling de filas existentes para cubrir `transition_midfield`
y `block`.

## Cuando volver a parallel slots

Solo volver a:

```text
RL_MA_TEACHER_VISIBLE_PARALLEL_SLOTS=1
```

cuando single-SimRobot cumpla:

- genera filas con filtros razonables;
- los roles defensivos no hacen thrash excesivo;
- no hay colisiones sistematicas entre supports;
- `block` aparece en muestras defensivas armadas;
- los resets no dependen de relajar `reset_ball_drift`.

Al volver a parallel, usar un `display_base` nuevo y un `PYBH_RL_SHM_NAME` unico.
Si falla, depurar primero logs de `/tmp/pybh_visible_parallel_*` antes de tocar el
teacher.
