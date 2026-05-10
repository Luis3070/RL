# RL × B-Human con `pybh`

> Estado actual: la documentacion mantenida de la integracion natural RL/B-Human esta en
> [docs/README.md](./docs/README.md). Los documentos `BHUMAN_NATURAL_SKILLS_PLAN.md`
> y `VISIBLE_SIMROBOT_HANDOFF.md` quedan como contexto historico de investigacion.

Este repo quedó en un punto bastante más serio que el `pybh-clean` inicial.
Ya no estamos sólo en “tengo un script heurístico que llama a B-Human”, sino en
una base donde:

- Python manda acciones RL directo a B-Human dentro del mismo proceso.
- el bridge ya entiende todas las variantes públicas de `SkillRequest`.
- hay dos backends útiles:
  - `cpp` con `RLSim2D` para iterar rápido
  - `simrobot` con SimRobot 2D embebido real
- `train.py` ya levanta un agente `MAPPO` real con `skrl`
- el smoke y el train corto ya corren tanto en `cpp` como en `simrobot`

No está terminado el stack final de entrenamiento completo, pero sí quedó
cerrada esta fase: bridge, backend real y loop RL mínimo funcional.

## Qué problema resolvimos

La rama `pybh-clean` estaba muy reducida. Al empezar, en `RL/` prácticamente
sólo existían:

- `heuristic_agent_experiment.py`
- `README.md`
- `environment.yml`
- un CSV de resultados

No estaban los módulos grandes heredados de `main`:

- `train.py`
- `environment.py`
- `wrapper.py`
- `reward.py`
- `config.py`

Además, el bridge C++ sólo entendía una parte de `SkillRequest`. En la práctica,
el camino RL -> B-Human estaba limitado a:

- `stand`
- `walkTo`
- `shoot`
- `dribble`

Y aunque ya existía el host embebido de SimRobot, el path real de `simrobot`
había quedado roto: Python podía cargar la escena, mover pelota/robot y llamar
`rl_set_action()`, pero no salía `obs_ready` por `rl_get_obs()`.

## Qué se hizo en esta fase

Se hicieron dos grupos grandes de cambios:

1. cambios en este repo `RL/`
2. cambios en el repo hermano `SabanaHerons2026/`

La idea fue alinear ambos lados:

- del lado Python: una API RL coherente
- del lado C++: un bridge que de verdad consuma esa API

## Cambios en `RL/`

### Archivos creados

Se agregaron estos archivos:

- [action.py](/home/rafael/Documentos/Robocup2026/RL/action.py)
- [config.py](/home/rafael/Documentos/Robocup2026/RL/config.py)
- [pybh_backend.py](/home/rafael/Documentos/Robocup2026/RL/pybh_backend.py)
- [environment.py](/home/rafael/Documentos/Robocup2026/RL/environment.py)
- [reward.py](/home/rafael/Documentos/Robocup2026/RL/reward.py)
- [wrapper.py](/home/rafael/Documentos/Robocup2026/RL/wrapper.py)
- [train.py](/home/rafael/Documentos/Robocup2026/RL/train.py)
- [smoke_test_skill_requests.py](/home/rafael/Documentos/Robocup2026/RL/smoke_test_skill_requests.py)

### Archivos modificados

Se tocaron estos archivos existentes:

- [heuristic_agent_experiment.py](/home/rafael/Documentos/Robocup2026/RL/heuristic_agent_experiment.py)
- [heuristic_experiment_results.csv](/home/rafael/Documentos/Robocup2026/RL/heuristic_experiment_results.csv)

### 1. Acción canónica RL

En [action.py](/home/rafael/Documentos/Robocup2026/RL/action.py) quedó definida
la acción estructurada común para todo el stack:

- `skill_type`
- `target_x`
- `target_y`
- `target_theta`
- `pass_target`

Orden de skills:

1. `stand`
2. `walk`
3. `shoot`
4. `pass`
5. `dribble`
6. `block`
7. `mark`
8. `observe`

Detalles prácticos:

- internamente existe `SkillType.pass_` porque `pass` es keyword en Python
- hacia afuera siempre se expone como `"pass"`
- hay helpers para:
  - convertir dict -> acción
  - convertir `np.ndarray` -> acción
  - convertir acción -> `np.ndarray`
  - clamplear rangos
  - construir máscara de skills activas

### 2. Configuración mínima del stack

En [config.py](/home/rafael/Documentos/Robocup2026/RL/config.py) quedó la
config central para esta fase:

- rutas a `SabanaHerons2026`
- escena de SimRobot
- robot name
- tamaños y rangos de acción
- skills activas por defecto para entrenar:
  - `stand`
  - `walk`
  - `shoot`
  - `pass`
  - `dribble`
- configuración del backend
- configuración mínima de train/smoke

No es todavía el `config.py` grande de `main`. Es un config minimalista para
esta base nueva.

### 3. Backend Python sobre `pybh`

En [pybh_backend.py](/home/rafael/Documentos/Robocup2026/RL/pybh_backend.py)
se encapsuló el acceso a `pybh`.

Tiene tres modos:

- `simrobot`
- `cpp`
- `python`

Qué hace:

- `start()`: levanta `pybh.Controller()` o carga escena SimRobot
- `reset()`: reinicia pelota/robot según el backend
- `step()`: manda acción RL y avanza uno o más frames
- `read_obs()`: consume `pybh.rl_get_obs(...)`
- `close()`: apaga controller o descarga la escena

Detalles importantes:

- traduce `walk` a `"walkTo"` sólo en la frontera con `pybh`
- manda `pass_target` si la firma de `pybh.rl_set_action(...)` lo soporta
- usa `BackendObservation` para parsear respuesta del bridge
- también recoge:
  - `requested_skill`
  - `requested_pass_target`
  Eso sirve para verificar qué quedó realmente latcheado en el bridge.

### 4. Entorno RL

En [environment.py](/home/rafael/Documentos/Robocup2026/RL/environment.py) se
construyó un `ParallelEnv` minimalista.

Importante: sigue siendo un entorno chico y monoagente. No pretende todavía ser
el reemplazo final del entorno grande de `main`.

Qué tiene:

- agente único: `player_1`
- `observation_space`: vector de 9 floats
- `action_space`: `Box(5,)`
- `state_space`: igual al estado global mínimo
- método `state()` para que `skrl` pueda consumirlo

Por qué el `action_space` es plano y no `Dict`:

- `skrl`/`PettingZooWrapper` trabaja mucho más simple con `Box`
- la semántica estructurada se conserva igual, sólo que la decodificación queda
  dentro del env

Observación actual:

- delta robot-pelota
- pose robot
- pose pelota
- frame del bridge
- flag de backend

Todavía falta:

- multiagente real
- CTDE real
- observación rica
- curriculum
- métricas B-Human-aware completas

### 5. Reward

[reward.py](/home/rafael/Documentos/Robocup2026/RL/reward.py) tiene un reward
muy simple para esta fase:

- progreso hacia la pelota
- bonus por gol
- penalización por step

Esto no pretende sustituir el shaping sofisticado del `reward.py` de `main`.
Sirve para tener el loop vivo y comprobable.

### 6. Wrapper

[wrapper.py](/home/rafael/Documentos/Robocup2026/RL/wrapper.py) hace una capa
ligera para trabajar con acciones estructuradas o tensores planos, y expone
`collect_episode(...)`.

De nuevo: es una versión chica, no el wrapper completo heredado de `main`.

### 7. `heuristic_agent_experiment.py`

Se migró [heuristic_agent_experiment.py](/home/rafael/Documentos/Robocup2026/RL/heuristic_agent_experiment.py)
para que use la misma acción canónica.

Antes tenía su propio contrato informal.
Ahora:

- `_heuristic_action(...)` devuelve `StructuredSkillAction`
- el `walkTo` del bridge se obtiene vía traducción `walk -> walkTo`
- ya no hay dos APIs RL distintas dentro del repo

### 8. `train.py`

El `train.py` actual ya no es un stub. Tampoco es todavía el `train.py` final
de `main`, pero ya usa `MAPPO` real con `skrl`.

Qué hace hoy:

- crea un `PolicyNetwork` con `GaussianMixin`
- crea un `ValueNetwork` con `DeterministicMixin`
- monta `RandomMemory`
- monta `MAPPO`
- ejecuta:
  - `mode=smoke`
  - `mode=train`

Qué NO hace todavía:

- no recupera el logger grande de `main`
- no recupera curriculum
- no tiene critic CTDE real multiagente
- no tiene la capa completa de métricas B-Human-aware

Importante:

- se probó `SequentialTrainer`, pero en esta versión de `skrl` el camino de
  “un solo `MultiAgent`” cae por el código single-agent y rompe con el retorno
  tipo dict de `PettingZoo`
- por eso se dejó un loop corto propio en `train.py`
- el agente sí es `MAPPO` real; lo que no usamos es el trainer de librería

### 9. `smoke_test_skill_requests.py`

Este archivo existe para validar el bridge sin meter todavía toda la pila de
entrenamiento.

Qué hace:

- resetea la escena por cada skill
- manda una variante por vez
- imprime:
  - `obs_ready`
  - `frame`
  - `requested_skill`
  - `requested_pass_target`
  - pose robot
  - pose pelota

Eso sirve para separar muy rápido:

- “el bridge no acepta la acción”
- “la acción sí llegó, pero la semántica no hizo nada”
- “el runtime real de SimRobot está roto”

## Cambios en `SabanaHerons2026/`

La otra mitad del trabajo se hizo en el repo hermano:

- `/home/rafael/Documentos/Robocup2026/SabanaHerons2026`

### Archivos tocados

- [Module.cpp](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/Module.cpp)
- [RLSharedState.h](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/RLSharedState.h)
- [RLSharedState.cpp](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/RLSharedState.cpp)
- [RLSim2D.h](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/RLSim2D.h)
- [RLSkillProvider.cpp](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.cpp)

### 1. Shared state RL

En `RLSharedState` se agregó:

- `passTarget`

Y se versionó la shared memory:

- nombre: `"/pybh_rl_shared_state_v2"`
- versión: `3`

Esto fue importante porque el layout compartido ya no era compatible con el
viejo.

### 2. Python binding `rl_set_action`

En `Module.cpp` se extendió la firma:

de:

- `rl_set_action(player, skill, x, y, theta)`

a:

- `rl_set_action(player, skill, x, y, theta, pass_target)`

También `rl_get_obs()` ahora expone:

- `requested_skill`
- `requested_pass_target`

Eso es oro para debug, porque deja ver si Python realmente escribió lo que uno
cree que escribió.

### 3. `RLSkillProvider`

Acá estuvo la parte semántica importante.

Antes el bridge sólo manejaba una parte chica de `SkillRequest`.
Ahora se soportan las 8 variantes públicas:

- `stand`
- `walk`
- `shoot`
- `pass`
- `dribble`
- `block`
- `mark`
- `observe`

Mapeos principales:

- `walk` / `walkTo` -> `SkillRequest::Builder::walkTo(...)`
- `shoot` -> `SkillRequest::Builder::shoot()`
- `pass` -> `SkillRequest::Builder::passTo(passTarget)`
- `dribble` -> `SkillRequest::Builder::dribbleTo(theta)`
- `block` -> `SkillRequest::Builder::block(Vector2f(x, y))`
- `mark` -> `SkillRequest::Builder::mark(Vector2f(x, y))`
- `observe` -> `SkillRequest::Builder::observe(Vector2f(x, y))`

También hay un fallback de `MotionRequest`:

- `block` / `mark` / `observe` se degradan a `walkToPose`
- `pass` y `shoot` se degradan a `walkToBallAndKick`

Ojo con esto:

- ese fallback no es la “idea final del diseño”
- la interfaz canónica sigue siendo `SkillRequest`
- el fallback existe para no dejar muerto el otro camino de la representación

### 4. `RLSim2D`

En el simulador headless rápido:

- `block`
- `mark`
- `observe`

se tratan como `walk`

y:

- `pass`

se trata como un kick simplificado

Eso no es semántica final. Es una implementación mínima para smoke y test de
transporte del bridge.

## El bug más feo que apareció

El bug más tramposo de esta fase fue el de `simrobot` real.

Síntoma:

- `simrobot_load_scene(...)` cargaba bien
- se podía mover pelota y robot desde Python
- `rl_set_action(...)` cambiaba `requested_skill`
- pero `rl_get_obs(...)` nunca devolvía `obs_ready=True`
- el robot de la escena tampoco reaccionaba a la acción RL

La causa:

- `pybh.controller` sí se recompiló con el `RLSharedState` nuevo
- `libSimulatedNao.so` seguía viejo
- resultado: Python escribía una versión de shared memory y SimRobot leía otra

En castellano: parecía un bug de lógica, pero era un desalineamiento binario.

## Cómo compilar TODO

Acá van los pasos completos, ya aterrizados a esta máquina.

### 1. Rebuild de `pybh`

Directorio:

```bash
cd /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python
```

Comando:

```bash
python3 setup.py build_ext --inplace
```

Importante:

- ahora `setup.py` sí termina bien y copia los módulos compilados al paquete
  fuente `pybh/`
- los artefactos principales quedan disponibles tanto en `build/lib...` como en
  `Make/Python/pybh/`
- los módulos que interesan son:

```bash
/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python/pybh/controller.cpython-312-x86_64-linux-gnu.so
/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python/pybh/logs.cpython-312-x86_64-linux-gnu.so
```

Error real que apareció varias veces:

- `cmake --build ... --target PythonController SimulatedNao B-Human` podía decir
  `ninja: no work to do`
- pero Python seguía importando un `controller...so` viejo desde
  `Make/Python/pybh/`
- en ese caso el problema no era el `PYTHONPATH` en sí, sino que el artefacto
  cargado por ese `PYTHONPATH` estaba desalineado con `Build/Linux/...`

Chequeo rápido:

```bash
ls -lh \
  /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Build/Linux/Python/Develop/controller.cpython-312-x86_64-linux-gnu.so \
  /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python/pybh/controller.cpython-312-x86_64-linux-gnu.so
```

Si `Build/Linux/...` tiene timestamp nuevo y `Make/Python/pybh/...` quedó viejo,
hay que correr otra vez:

```bash
cd /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python
python3 setup.py build_ext --inplace
```

Eso simplifica bastante el uso: ya no hace falta apuntar el `PYTHONPATH` al
subdirectorio `build/lib...`; basta con apuntarlo a `Make/Python`.

### 2. Rebuild de `SimulatedNao`

Este paso es obligatorio si cambiaste algo en:

- `RLSharedState`
- `RLSkillProvider`
- o cualquier cosa del camino RL dentro de SimRobot

Comando:

```bash
cmake --build /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Build/Linux/CMake/Develop --target SimulatedNao
```

Si no haces esto, te puede pasar lo que pasó acá:

- Python ve una estructura de shared memory
- `libSimulatedNao.so` ve otra
- y entonces `simrobot` parece “muerto”

### 3. Crear el entorno Python local

En este repo:

```bash
cd /home/rafael/Documentos/Robocup2026/RL
python3 -m venv .venv
```

### 4. Instalar dependencias Python

```bash
.venv/bin/pip install torch skrl pettingzoo
```

Notas:

- se usó `venv` porque el Python del sistema estaba protegido por PEP 668
- `torch` que bajó acá fue `2.11.0+cu130`
- `skrl` quedó en `2.0.0`

### 5. Usar `PYTHONPATH` correcto

Mientras no instales `pybh` en `site-packages`, corre así:

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python
```

Eso hace que Python vea `pybh.controller` y `pybh.logs` recién compilados.

## Cómo hacerlo funcionar TODO

Acá van los flujos útiles, de menor a mayor complejidad.

### 1. Smoke del bridge en backend rápido `cpp`

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python python3 smoke_test_skill_requests.py --backend cpp
```

Qué valida:

- que el bridge reciba las 8 variantes
- que `requested_skill` y `requested_pass_target` estén bien
- que no crashee el path rápido

### 2. Smoke del bridge en `simrobot` real

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python python3 smoke_test_skill_requests.py --backend simrobot
```

Qué valida:

- que la escena real carga
- que `obs_ready` vuelve a salir
- que el bridge y `SimulatedNao` ya están recompilados y alineados

Importante:

- este modo necesita ejecutarse fuera de entornos muy restringidos si el
  plugin intenta abrir sockets UDP

### 3. Experimento heurístico rápido en `cpp`

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python python3 heuristic_agent_experiment.py --episodes 10 --no-plot
```

Útil para:

- iteración rápida
- validar que el flujo heurístico no se rompió

### 4. Experimento heurístico real en `simrobot`

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python python3 heuristic_agent_experiment.py --episodes 1 --no-plot --simrobot-real
```

Útil para:

- validar el path embebido real completo
- confirmar que ya sale observación RL

### 5. Smoke RL con `MAPPO` sobre backend `cpp`

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python .venv/bin/python train.py --backend cpp --mode smoke --smoke-steps 8
```

Útil para:

- validar env + wrapper + `MAPPO` + bridge rápido

### 6. Train corto con `MAPPO` sobre backend `cpp`

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python .venv/bin/python train.py --backend cpp --mode train --train-steps 16
```

Qué valida:

- `act`
- `record_transition`
- `post_interaction`
- updates del agente

### 7. Smoke RL con `MAPPO` sobre `simrobot` real

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python .venv/bin/python train.py --backend simrobot --mode smoke --smoke-steps 4
```

### 8. Train corto con `MAPPO` sobre `simrobot` real

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python .venv/bin/python train.py --backend simrobot --mode train --train-steps 8
```

Este es el smoke más importante de esta fase, porque junta:

- agente `MAPPO`
- bridge real
- SimRobot embebido real

## Visualización en vivo en SimRobot

Estado actual:

- el backend `simrobot` embebido de [pybh_backend.py](/home/rafael/Documentos/Robocup2026/RL/pybh_backend.py) corre en modo headless
- eso significa que la escena real `.ros2d` sí avanza, pero no aparece una ventana
- en este repo no encontré escenas `.ros`; hoy el path listo es 2D, no 3D
- la convención Python del stack quedó fijada así:
  `walk.target_x/target_y/target_theta` se expresan en coordenadas globales de campo
  y [pybh_backend.py](/home/rafael/Documentos/Robocup2026/RL/pybh_backend.py) hace la conversión al frame que consume el backend real

Ruta práctica para ver el agente vivo:

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python python3 live_simrobot_probe.py --launch --policy heuristic
```

Por defecto ahora corre en modo continuo hasta `Ctrl+C`.
También usa la política heurística por defecto.
Si quieres limitarlo:

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python python3 live_simrobot_probe.py --launch --steps 500
```

Si quieres volver al patrón fijo de depuración:

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python python3 live_simrobot_probe.py --launch --policy fixed
```

Ese script:

- abre `Build/Linux/SimRobot/Develop/SimRobot`
- carga `Config/Scenes/RL1v0.ros2d`
- manda acciones RL por shared memory con `rl_set_action(...)`
- lee observaciones con `rl_get_obs(...)`

Uso esperado:

- al abrirse SimRobot, arrancar la simulación con `F5`
- si `RLSkillProvider` está cargado correctamente, el robot visible debe empezar a reaccionar en vivo

Limitación actual:

- este probe visible no hace `reset()` del mundo como el backend embebido
- sirve para validar visualización/control en vivo, no todavía para episodios RL completos en UI

## Estado funcional actual

### Qué sí funciona

- `pybh` corre Cognition real en proceso Python
- `rl_set_action()` y `rl_get_obs()` funcionan
- el bridge ya soporta las 8 variantes públicas de `SkillRequest`
- `pass_target` ya viaja hasta C++
- `RLSim2D` sigue sirviendo como backend rápido
- el host embebido de SimRobot carga escena real
- `simrobot` volvió a emitir observación RL
- `train.py` ya levanta `MAPPO` real
- se puede hacer smoke y train corto tanto en `cpp` como en `simrobot`

### Qué sigue faltando

Esta fase no cierra todavía el stack final. Falta:

- restaurar entorno multiagente real
- restaurar CTDE real
- recuperar el reward grande y las métricas B-Human-aware de `main`
- recuperar curriculum manager
- recuperar logging serio
- refinar semántica real de:
  - `pass`
  - `block`
  - `mark`
  - `observe`

En particular:

- hoy `block` / `mark` / `observe` están soportadas de punta a punta en bridge
- pero su semántica operativa sigue siendo mínima, pensada para smoke

## Qué archivo mirar para cada cosa

Si quieres entrar rápido a un tema:

- acción RL: [action.py](/home/rafael/Documentos/Robocup2026/RL/action.py)
- config local: [config.py](/home/rafael/Documentos/Robocup2026/RL/config.py)
- backend `pybh`: [pybh_backend.py](/home/rafael/Documentos/Robocup2026/RL/pybh_backend.py)
- entorno RL: [environment.py](/home/rafael/Documentos/Robocup2026/RL/environment.py)
- smoke de skills: [smoke_test_skill_requests.py](/home/rafael/Documentos/Robocup2026/RL/smoke_test_skill_requests.py)
- heurístico: [heuristic_agent_experiment.py](/home/rafael/Documentos/Robocup2026/RL/heuristic_agent_experiment.py)
- train RL: [train.py](/home/rafael/Documentos/Robocup2026/RL/train.py)

Del lado C++:

- binding Python: [Module.cpp](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/Module.cpp)
- shared state: [RLSharedState.h](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/RLSharedState.h)
- sim headless rápido: [RLSim2D.h](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/RLSim2D.h)
- host embebido real: [SimRobotHost.cpp](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Libs/Python/Controller/SimRobotHost.cpp)
- traducción RL -> `SkillRequest`: [RLSkillProvider.cpp](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.cpp)

## Cierre de fase

Si hay que resumir esta fase en una frase:

“dejamos de tener un script suelto hablando con B-Human y pasamos a tener una
superficie RL unificada, con bridge completo de `SkillRequest`, backend rápido,
backend real de SimRobot rehabilitado y un `MAPPO` mínimo corriendo encima”.

Eso todavía no es el sistema final, pero ya es una base honesta para construir
la siguiente etapa sin parches raros ni contratos duplicados.
