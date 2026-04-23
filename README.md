# RL × B-Human — integración pybh / SimRobot

Rama `pybh-clean`. Stack nuevo, en proceso, que reemplaza el approach anterior
basado en subprocess + JSON (`bhuman_sidecar`, `bridge/bhuman_bridge.exe`). Todo
lo superado quedó en la rama `rlsimrobot` y en `main`.

## Qué hay aquí

| Archivo                              | Rol                                              |
|--------------------------------------|--------------------------------------------------|
| `heuristic_agent_experiment.py`      | Experimento activo: agente heurístico ↔ pybh     |
| `heuristic_experiment_results.csv`   | Output del experimento                           |
| `environment.yml`                    | Spec del env conda                               |
| `trajectory_guidance_project/`       | Subproyecto paralelo (guía de trayectorias)      |

El resto del código vive en `SabanaHerons2026/` (C++). Lo que tocamos:

- `Src/Libs/Python/Controller/Module.cpp` — bindings Python (`rl_set_action`, `rl_get_obs`, `set_world_state`)
- `Src/Libs/Python/Controller/PythonConsole.{h,cpp}` — thread que inyecta el frame de Cognition cuando Python llama `ctrl.update()`
- `Src/Libs/Python/Controller/RLSharedState.h` — singleton header-only con `RLPlayerIO` por jugador (mutex + semáforo + buffers de acción/obs)
- `Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.cpp` — módulo BHuman que lee la acción RL y emite `SkillRequest`, y escribe la observación de vuelta
- `Config/Scenarios/4v4_Scaled/threads.cfg` — thread `Cognition` cambiado a execution unit `Cognition2D`

## Arquitectura (alto nivel)

```
  Python (main thread)                    BHuman (robot thread)
  ───────────────────────                 ────────────────────────
  set_world_state(ctrl, …)   ───►  PythonConsole::setWorldState
  rl_set_action(n, skill, …) ───►  RLPlayerIO.skill/target
  ctrl.update()              ───►  trigger() + wait(updatedSignal)
                                   └─► PythonConsole::main
                                       · injectFrame (idFrameBegin…idFrameFinished)
                                       · Cognition2D::beforeFrame → true
                                       · ModuleContainer ejecuta módulos
                                       · RLSkillProvider::update
                                         - lee io.skill/target → SkillRequest
                                         - escribe io.ball*/robot* → obs
                                         - io.obsSignal.post()
  rl_get_obs(n)              ◄───  wait(obsSignal) + lectura bajo mutex
```

Comunicación in-process, sin JSON, sin pipes. Un sólo proceso Python con
N jugadores; cada uno tiene su `RLPlayerIO` indexado por `player_number`.

## Bugs corregidos en esta rama

1. **Cognition no ejecutaba módulos después del init.** Causa: `Cognition::beforeFrame()` exige `UpperFrameInfo`/`LowerFrameInfo` con timestamp nuevo, y pybh no corre los threads de cámaras Upper/Lower. Fix: `threads.cfg` → `executionUnit = Cognition2D` (su `beforeFrame()` sólo requiere `LogDataProvider::isFrameDataComplete()`, que sí se activa con `idFrameFinished`).
2. **`ctrl.update()` retornaba antes del frame real.** Causa: `updatedSignal` quedaba posteado por el `main()` que se ejecuta automáticamente al arrancar el thread. Fix: `PythonConsole` añade `std::atomic<bool> updateRequested`; `main()` sólo hace `updatedSignal.post()` si ese flag está activo.
3. **`rl_get_obs` devolvía `obs_ready=False` aunque `RLSkillProvider` ya había corrido.** Causa paralela a la anterior: `obsSignal` tenía un post residual del ciclo de init. Fix: `rl_set_action` drena el semáforo (`while(io.obsSignal.tryWait()) {}`) antes de marcar `obsReady=false`, garantizando que el próximo `wait()` sólo despierte por la observación fresca.

Los 3 bugs comparten raíz: el thread arranca una vez antes de que Python pida nada, y los semáforos quedaban en estado "ya disparado" al primer uso real.

## Ventajas del approach pybh vs. el anterior (sidecar/subprocess)

- **In-process, 0 IPC.** Sin serialización JSON, sin stdin/stdout, sin `tcp`. La acción y la observación son escrituras a memoria protegidas por mutex.
- **Latencia determinista.** Una llamada `ctrl.update()` equivale a un frame de Cognition exacto. El ciclo acción→obs queda bajo 1 ms en vez de los ~10-30 ms del pipe JSON.
- **Escala a N jugadores triviales.** `RLPlayerIO` indexado por `player_number` — MARL sin multiplexar sockets.
- **Debugging unificado.** Un sólo proceso; gdb/valgrind sobre Python cubre también el C++ de BHuman.
- **Reutiliza Cognition real.** Todo el stack de BHuman (módulos, blackboard, skill graph) corre sin stubs. Lo que entrene aquí es transferible a robot real con mínimas adaptaciones.
- **Menos superficie de código.** Eliminó `bhuman_bridge.cpp`, `bridge_protocol.py`, `bhuman_sidecar.py` y toda la capa de parseo/handshake.

## Desventajas / costos del cambio

- **Build acoplado a BHuman.** Editar pybh exige recompilar SabanaHerons2026 (`python3 setup.py build_ext --inplace` en `Make/Python`). Ciclo de iteración más largo que editar Python puro.
- **Un crash en C++ tumba el proceso Python.** Antes un segfault del sidecar lo reiniciabas; ahora hay que relanzar el entrenamiento entero.
- **Lock-in con la API interna de BHuman.** `RLSkillProvider` depende de `SkillRequest::Builder`, `GroundTruthWorldState`, `GameState`. Un refactor upstream obliga a tocar pybh.
- **Mocks de frame incompletos.** `PythonConsole::injectFrame` sólo inyecta lo mínimo (`FrameInfo`, `GroundTruthWorldState`, `FallDownState`, etc.). Módulos que dependan de representaciones no inyectadas pueden no funcionar — hay que ampliar `injectFrame` caso por caso.
- **Execution unit `Cognition2D` en lugar de `Cognition`.** Se pierden los gates de visión (Upper/Lower). Para entrenamiento sintético no importa, pero cualquier módulo que los asuma dejará de correr.
- **Cwd obligatorio en `SabanaHerons2026/`.** `File::getBHDir()` resuelve rutas relativas al directorio de trabajo. Los scripts tienen que hacer `os.chdir` antes del `import pybh.controller`.

## Estado actual

- ✅ Round-trip validado: `set_world_state` → `ctrl.update()` → `rl_get_obs` devuelve `obs_ready=True` y valores coherentes.
- ✅ Loop multi-frame estable (smoke test con 5 frames consecutivos, sin leaks de semáforo).
- ✅ Diagnósticos `fprintf` retirados de `Module.cpp`, `PythonConsole.cpp`, `RLSkillProvider.cpp`, `Debug.cpp`, `ModuleContainer.cpp`.
- ⚠️ `heuristic_agent_experiment.py` con múltiples episodios: segfault en el 2º episodio. Por investigar — no es bug del bridge per se (el smoke frame-a-frame sí funciona), probablemente estado residual en el Controller entre episodios o liberación indebida al reiniciar.

## Lo que falta

**Corto plazo (próxima iteración):**
1. Reproducir y arreglar el segfault multi-episodio del heurístico.
2. Reescribir `environment.py` como `gymnasium.Env` que envuelva pybh (reemplazo directo del viejo `NaoSoccerEnv`).
3. Wrapper multi-agente (PettingZoo ParallelEnv) con un `Controller` compartido y N `player_number`.
4. Reward y terminación: portar la lógica útil del viejo `reward.py` al env nuevo.

**Mediano plazo:**
5. Loop MAPPO sobre el env nuevo (reemplazo de `train.py`).
6. Espacio de observación completo (los ~61 floats del plan canónico — ver rama `rlsimrobot` para la enumeración previa).
7. Tests de estabilidad largos (10k+ frames) y perfilado (si cuello de botella está en Python, C++ o sync).
8. Curriculum stage 0 → stage 1 (handoff → takeover) con evaluación automática.

**Abiertos / riesgos:**
- Decidir si mantener `Cognition2D` o ampliar `Cognition` con inyección sintética de `UpperFrameInfo`/`LowerFrameInfo`.
- Integración con GameController real (ahora se usa un `GameControllerData` vacío en `injectFrame`).
- Sim2Real: SimRobot vs. robot físico, tolerancia a ruido sensorial.

## Cómo correr

```bash
# 1. Build de pybh (una vez, y tras cualquier cambio en C++)
cd /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python
python3 setup.py build_ext --inplace
cp build/lib.linux-x86_64-cpython-312/pybh/*.so ~/.local/lib/python3.12/site-packages/pybh/

# 2. Smoke test mínimo
cd /home/rafael/Documentos/Robocup2026/RL
python3 heuristic_agent_experiment.py --episodes 1 --no-plot
```

## Referencias

- Memoria del proyecto: `mem-search` / `get_observations` — IDs clave 115, 117-123, 125, 129-131, 135
- Rama `rlsimrobot`: snapshot con todo el stack viejo (sidecar, smoke tests, docs antiguas) si hay que rescatar algo.
