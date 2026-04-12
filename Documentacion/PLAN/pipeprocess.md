# Plan de Diagnóstico, Fix e Integración de `pipe_process` en Windows

## 1. Contexto y objetivo

El transporte `pipe_process` (stdin/stdout persistente) reduciría el tiempo por step de **~56 ms** a **~2-5 ms**, pero fue desactivado en Windows por causar `access violation (0xc0000005)` dentro de `RLBridge.exe`.

Este plan define:
1. Cómo diagnosticar cuál de las 5 causas raíz posibles provoca el crash
2. El fix exacto para cada causa, archivo por archivo
3. Un pipeline de testing progresivo que valida cada fix antes del siguiente
4. Integración final sin romper la ruta estable `oneshot_file_fallback`

---

## 2. Arquitectura de referencia (para entender el crash)

```
Python (main thread)                        C++ RLBridge.exe
┌─────────────────────────┐                ┌──────────────────────────────────┐
│ bridge_protocol.py      │                │ main()                           │
│                         │                │   └─ while(getline(cin, line))   │
│ start():                │   stdin PIPE   │        ├─ "init" → bridge.init() │
│   Popen(PIPE,PIPE,PIPE) │ ──────────────>│        │    └─ Robot::start()    │
│   _send(init)           │                │        │        ├─ Cognition thr  │
│   _recv() → "ready"     │   stdout PIPE  │        │        ├─ Motion thread  │
│                         │ <──────────────│        │        └─ Debug thread   │
│ step():                 │   stdin PIPE   │        ├─ "frame" → processFrame()│
│   _send_raw(frame_json) │ ──────────────>│        │    ├─ sender.bin(...)    │
│   _recv() → "decision"  │   stdout PIPE  │        │    ├─ sender.send(true) │
│                         │ <──────────────│        │    ├─ waitForCognition() │
│ _stderr_reader_thread   │   stderr PIPE  │        │    ├─ readDecisions()   │
│   (drena stderr async)  │ <──────────────│        │    └─ cout << JSON      │
└─────────────────────────┘                │        └─ "shutdown" → break     │
                                           └──────────────────────────────────┘
```

**Punto crítico:** En modo `pipe_process`, stdout/stderr de `RLBridge.exe` son pipes anónimos
del kernel de Windows, **no una consola real**. Todos los hilos del framework BHuman comparten
esos handles implícitamente a través de la librería CRT de MSVC.

---

## 3. Resumen del estado actual del código

### Lado Python — [bridge_protocol.py](file:///c:/ALEJO/RL/bridge/bridge_protocol.py)

| Componente | Estado | Líneas clave |
|---|---|---|
| `start()` con pipes | Implementado, funciona parcialmente | L316-405 |
| Reader threads (stdout/stderr) | Implementados con `threading.Thread` | L626-685 |
| Fallback a oneshot en handshake | Implementado | L367-377 |
| Fallback a oneshot en step | Implementado | L458-496 |
| `CREATE_NO_WINDOW` flag | Activa en Windows | L344-345 |
| `bufsize=0` (unbuffered) | Activo | L342 |

### Lado C++ — [bhuman_bridge.cpp](file:///c:/ALEJO/RL/bridge/bhuman_bridge.cpp)

| Componente | Estado | Líneas clave |
|---|---|---|
| Main loop `getline(cin, line)` | Funcional | L1931 |
| `cout << json << endl; cout.flush()` | Sin mutex | L1978, L1984, L1993, L2006 |
| `cerr << ...` (logs) | Concurrente desde main + hilos BHuman | L816-850, L1038-1072 |
| `SetErrorMode` + crash handler | Activo | L1922-1924 |
| `Robot::start()` (lanza hilos) | Funcional | L821 |
| `readDecisions()` con `setGlobals()` cross-thread | Funcional pero sin sync | L1602-1767 |

### Smoke test existente — [smoke_test_rlbridge_persistent_basic.py](file:///c:/ALEJO/RL/smoke_test_rlbridge_persistent_basic.py)

Ya existe y valida exactamente lo que necesitamos: 3 frames + reset + 1 frame, todo
con `preferred_transport="pipe_process"` y `allow_oneshot_fallback=False`.

---

## 4. Diagnóstico: 5 causas raíz posibles

El crash `0xc0000005` al usar pipes en Windows puede venir de **una o varias** de estas
fuentes. Cada una tiene un mecanismo diferente y un fix distinto. Es obligatorio diagnosticar
cuál(es) están activas antes de declarar un fix.

---

### Causa A: Escritura concurrente a `stdout` sin mutex

**Mecanismo:**
`BridgeRunner::processFrame()` escribe a `std::cout` desde el hilo principal. Pero el framework
BHuman puede escribir a `stdout` internamente (logging del framework, asserts, o salida del
`Communication` thread) desde sus propios hilos (Cognition, Motion, Debug). El MSVC CRT
sincroniza parcialmente `std::cout` con un lock nivel-stream, pero no protege contra
corrupciones del `streambuf` interno cuando hay múltiples escritores concurrentes en un pipe
anónimo sin consola real.

**Evidencia esperada si es esta causa:**
- El crash ocurre **durante o justo después** de `cout << jsonSerialize(result)` (L1984)
- El stack trace apunta a `_write` o `_flushall` del CRT
- Ocurre de forma intermitente, no en el primer frame

**Diagnóstico:**

Agregar un `std::mutex` temporal alrededor de cada escritura a `cout` y `cerr` y ver si
el crash desaparece.

**Fix definitivo:**

Archivo: `bridge/bhuman_bridge.cpp` (y la copia en `SabanaHerons2026`)

```cpp
// Añadir al inicio del archivo, después de los includes (línea ~57):
namespace {
    std::mutex g_ioMutex;  // Protege stdout/stderr contra escritura concurrente
}

// Reemplazar CADA escritura a cout/cerr con bloques protegidos.
// Ejemplo en main() línea 1978:
{
    std::lock_guard<std::mutex> lock(g_ioMutex);
    std::cout << jsonSerialize(JsonValue(response)) << std::endl;
    std::cout.flush();
}

// Ejemplo en main() línea 1926:
{
    std::lock_guard<std::mutex> lock(g_ioMutex);
    std::cerr << "[bridge] BHuman Bridge v1.0 starting" << std::endl;
}
```

**Líneas exactas a proteger en `main()` (L1920-2013):**
- L1926: cerr del startup
- L1941-1942: cout del error JSON
- L1978-1979: cout del ready response
- L1984-1985: cout del frame decision
- L1993-1994: cout del reset ack
- L2006-2007: cout del error desconocido
- L2011: cerr del exit

**Líneas a proteger en `BridgeRunner` (métodos que escriben a cerr):**
- L815-850: init() cerr logs
- L1038-1072: shutdown() cerr logs
- L1021-1025: reset() cerr logs
- L127-137: `verboseBridgeLog()` — usar el mismo mutex

**Cambio en `verboseBridgeLog` (L129-137):**
```cpp
template<typename... Ts>
void verboseBridgeLog(const Ts&... parts)
{
    if constexpr(kVerboseBridgeLogs)
    {
        std::lock_guard<std::mutex> lock(g_ioMutex);
        (std::cerr << ... << parts);
        std::cerr << std::endl;
    }
}
```

**Volumen del cambio:** ~30 sitios en `bhuman_bridge.cpp`. No cambia lógica, solo envuelve I/O.

---

### Causa B: `CREATE_NO_WINDOW` elimina la consola y BHuman asume que existe

**Mecanismo:**
En `bridge_protocol.py` línea 344-345, se usa `subprocess.CREATE_NO_WINDOW`. Esto no crea
**ningún** recurso de consola para el proceso hijo. El framework BHuman o el CRT de MSVC
pueden invocar internamente funciones Win32 como:
- `GetConsoleScreenBufferInfo()`
- `SetConsoleCtrlHandler()`
- `GetConsoleWindow()`
- `WriteConsoleW()` (versus `WriteFile()` para pipes)

Si cualquiera de estas se llama con un handle nulo o inválido, crash.

**Evidencia esperada si es esta causa:**
- El crash ocurre **durante `init()`** (L764-857), antes de procesar el primer frame
- Ocurre siempre, reproduciblmente
- El proceso muere con `0xc0000005` en una dirección dentro de `kernel32.dll` o `ucrtbase.dll`

**Diagnóstico:**

Cambiar `CREATE_NO_WINDOW` por `CREATE_NEW_CONSOLE` en Python y ver si el crash desaparece.

**Fix definitivo:**

Archivo: `bridge/bridge_protocol.py`

```python
# Línea 344-345, REEMPLAZAR:
if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
    startup_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

# CON:
if os.name == "nt":
    # CREATE_NEW_CONSOLE asigna una consola real (oculta) al proceso hijo.
    # Esto evita que el framework BHuman crashee al intentar usar APIs
    # de consola Win32 con handles nulos. La consola no aparece al usuario
    # porque el proceso no tiene ventana visible, pero sí existe como recurso
    # del kernel.
    # 
    # NO usar CREATE_NO_WINDOW: BHuman asume que hay una consola válida.
    startup_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
```

**Nota:** Si la solución real es que BHuman necesita una consola pero NO queremos que el
usuario vea una ventana nueva, usar en cambio:
```python
# Alternativa: consola existente pero oculta (STARTUPINFO)
si = subprocess.STARTUPINFO()
si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
si.wShowWindow = 0  # SW_HIDE
startup_kwargs["startupinfo"] = si
# NO usar creationflags en este caso — hereda la consola del padre
```

**Volumen:** 3-5 líneas en `bridge_protocol.py`.

---

### Causa C: Buffer de pipe lleno causa deadlock → timeout → kill → crash

**Mecanismo:**
Los pipes anónimos de Windows tienen un buffer de **4 KB a 64 KB** por defecto (depende
de la versión del OS). BHuman, durante `init()`, ejecuta `Robot::start()` que:
1. Inicializa el framework
2. Carga módulos (decenas de cerr logs)
3. Los hilos arrancan y generan logs adicionales

Si `stderr` se llena antes de que Python lo drene, la siguiente escritura a `cerr` **bloquea**
el hilo que la intenta. Si ese hilo es el main del bridge, nunca llega a enviar `ready` por
`stdout`. Python ve un timeout, mata el proceso con `kill()`, y la terminación abrupta de
BHuman con hilos activos genera el access violation.

**Evidencia esperada si es esta causa:**
- El `start()` de Python hace timeout en `_recv()` (L363)
- `poll_code` es `None` (proceso sigue vivo pero colgado)
- stderr contiene logs parciales que se cortan a mitad de frase
- El crash ocurre cuando Python llama `_process.kill()`

**Diagnóstico:**

Aumentar el buffer del pipe de stderr a 1 MB y ver si el handshake completa.

**Fix definitivo:**

Archivo: `bridge/bridge_protocol.py`

Opción 1 — Redirigir stderr a un archivo en lugar de pipe:
```python
# En start(), línea 336-343, REEMPLAZAR:
startup_kwargs: Dict[str, Any] = {
    "args": [str(exe)],
    "stdin": subprocess.PIPE,
    "stdout": subprocess.PIPE,
    "stderr": subprocess.PIPE,
    "env": env,
    "bufsize": 0,
}

# CON:
# Redirigir stderr a un archivo para evitar que el pipe se llene.
# stdout queda como pipe para el protocolo JSON.
self._stderr_file = tempfile.NamedTemporaryFile(
    prefix="rlbridge_stderr_", suffix=".log",
    dir=_oneshot_runtime_dir(), delete=False, mode="w",
)
startup_kwargs: Dict[str, Any] = {
    "args": [str(exe)],
    "stdin": subprocess.PIPE,
    "stdout": subprocess.PIPE,
    "stderr": self._stderr_file,  # No pipe, archivo directo
    "env": env,
    "bufsize": 0,
}
```

Opción 2 — Stderr mergeado con stdout (como en one-shot estable):
```python
startup_kwargs: Dict[str, Any] = {
    "args": [str(exe)],
    "stdin": subprocess.PIPE,
    "stdout": subprocess.PIPE,
    "stderr": subprocess.STDOUT,  # 2>&1 — merged con stdout
    "env": env,
    "bufsize": 0,
}
```

> [!WARNING]
> La Opción 2 requiere que el reader de stdout distinga entre JSON y logs de texto.
> Esto ya está implementado en `_recv()` (L614-616) que hace `skip non-JSON lines`.

**Si se usa Opción 2**, también hay que ajustar:
- `_start_reader_threads()` (L626-650): no crear stderr reader thread si stderr es `STDOUT`
- `_read_stderr()` (L726-729): retornar "(merged into stdout)" si no hay deque

**Volumen:** 5-15 líneas en `bridge_protocol.py` + posible eliminación del stderr reader.

---

### Causa D: `readDecisions()` cross-thread blackboard sin sincronización real

**Mecanismo:**
En `readDecisions()` L1602-1767, el bridge llama `cognition_->setGlobals()` y luego lee
directamente del `Blackboard::getInstance()`. Esto cambia el thread-local storage del
framework BHuman para apuntar al blackboard de Cognition **desde el hilo main**. Si
Cognition está escribiendo al blackboard en ese mismo instante (modificando una representación
en su propio hilo), hay una race condition. En modo one-shot el proceso muere inmediatamente
después de esta lectura, así que la ventana temporal de la race es mínima. Pero en modo
persistente, esta función se llama cientos de veces por segundo, aumentando masivamente
la probabilidad de que la race produzca un null dereference.

**Evidencia esperada si es esta causa:**
- El crash ocurre DESPUÉS de que los primeros frames funcionen (no en init)
- El crash es en `readDecisions()`, no en I/O
- El stack trace apunta a `Blackboard::getInstance()` o a un `static_cast` de una
  representación
- El crash se correlaciona con el momento exacto en que Cognition está corriendo su frame

**Diagnóstico:**

Reemplazar temporalmente `readDecisions()` con una versión que solo usa el snapshot del
`RLBridgeConsole` (que SÍ está protegido por mutex) y nunca hace fallback al blackboard
directo.

**Fix definitivo:**

Archivo: `bridge/bhuman_bridge.cpp`

```cpp
// En readDecisions(), línea 1602, REEMPLAZAR la condición del fallback:
if(!snapshotHadDecisionData && cognition_)
{
    // ... direct blackboard fallback (ELIMINAR este bloque completo)
}

// CON:
if(!snapshotHadDecisionData)
{
    // En modo persistente, el snapshot del RLBridgeConsole siempre debe
    // tener datos porque waitForCognitionFrameFinished() ya esperó a que
    // Cognition terminara y emitiera sus representaciones.
    // Si no los tiene, es un error de wiring, no una señal para hacer
    // lectura cross-thread insegura.
    result["debug_provider_game_state"] = JsonValue("snapshot_empty");
    result["debug_provider_skill_request"] = JsonValue("snapshot_empty");
    result["debug_provider_motion_request"] = JsonValue("snapshot_empty");
    result["debug_provider_motion_info"] = JsonValue("snapshot_empty");
    result["debug_provider_activation_graph"] = JsonValue("snapshot_empty");
    result["debug_provider_behavior_status"] = JsonValue("snapshot_empty");
    result["debug_provider_strategy_status"] = JsonValue("snapshot_empty");
    std::lock_guard<std::mutex> lock(g_ioMutex);
    std::cerr << "[bridge] readDecisions: snapshot empty — skipping unsafe"
                 " blackboard fallback in persistent mode" << std::endl;
}
```

**Nota:** Este fix solo aplica si el snapshot del debug channel SÍ está poblado (que
es el caso actual — las representaciones llegan por el canal `handleMessage` del
`RLBridgeConsole`). Si el snapshot no se llena, hay que verificar que BHuman esté
configurado para emitir las representaciones necesarias como mensajes debug.

**Volumen:** ~20 líneas eliminadas, ~15 nuevas.

---

### Causa E: Terminación abrupta del proceso cuando Python cierra pipes

**Mecanismo:**
Cuando Python llama `shutdown()` o cuando ocurre un error, cierra los pipes. En `main()`
de C++ (L1931), `std::getline(std::cin, line)` recibe EOF y el bucle termina. Pero los hilos
de BHuman (Cognition, Motion, Debug) siguen corriendo. Si `main()` retorna y se ejecutan
los destructores globales (del CRT, de estáticos de BHuman) mientras esos hilos aún están
activos, cualquier acceso a memoria liberada produce `0xc0000005`.

**Evidencia esperada si es esta causa:**
- El crash ocurre durante el **shutdown o después del último frame**
- stderr muestra `[bridge] shutdown: begin` pero no `[bridge] Shutdown` (final)
- El crash es en destructores o `atexit` handlers

**Diagnóstico:**

Agregar `std::cerr << "[bridge] main: loop exited"` justo después del while loop (L2009)
y verificar si el crash ocurre después de ese punto.

**Fix definitivo:**

Archivo: `bridge/bhuman_bridge.cpp`

```cpp
// En main(), DESPUÉS del while loop (línea 2009), REEMPLAZAR:
    }

    std::cerr << "[bridge] Exiting" << std::endl;
    return 0;
}

// CON:
    }

    // Shutdown cooperativo ANTES de salir del main.
    // Sin esto, los destructores globales de C++ se ejecutan mientras
    // los hilos de BHuman aún están activos, causando access violation.
    {
        std::lock_guard<std::mutex> lock(g_ioMutex);
        std::cerr << "[bridge] main: loop exited, starting cooperative shutdown"
                  << std::endl;
    }
    bridge.shutdown();  // Esto ya existe si se recibió "shutdown", pero puede
                        // no ejecutarse si el loop terminó por EOF (pipe roto).
    {
        std::lock_guard<std::mutex> lock(g_ioMutex);
        std::cerr << "[bridge] Exiting" << std::endl;
    }
    return 0;
}
```

Adicionalmente, robustecer la condición del loop:
```cpp
// Línea 1931, REEMPLAZAR:
while (std::getline(std::cin, line))

// CON:
while (std::cin.good() && std::getline(std::cin, line))
```

Y agregar un guard en el destructor de BridgeRunner:
```cpp
// Línea 757, REEMPLAZAR:
~BridgeRunner() { shutdown(); }

// CON:
~BridgeRunner()
{
    try { shutdown(); }
    catch (...) {}
}
```

**Volumen:** ~10 líneas en `bhuman_bridge.cpp`.

---

## 5. Orden de diagnóstico recomendado

El diagnóstico debe hacerse **uno a la vez**, en el orden que maximiza la probabilidad
de encontrar la causa con el mínimo esfuerzo. El orden está basado en la historia del
proyecto (el crash original fue durante frames, no durante init).

```
┌─────────────────────────────────────────────────────────────┐
│ Paso 1: Aplicar Fix E (shutdown cooperativo)                │
│         Esto es SIEMPRE necesario, sin importar la causa.   │
│         Si no se hace, los tests pueden dar falsos          │
│         positivos (el crash en shutdown oculta un fix real).│
│                                                             │
│ Paso 2: Aplicar Fix A (mutex de I/O)                        │
│         Si el problema persiste → Causa A descartada.       │
│         Si se resuelve → Causa A confirmada. FIN.           │
│                                                             │
│ Paso 3: Cambiar CREATE_NO_WINDOW → CREATE_NEW_CONSOLE       │
│         Si el problema persiste → Causa B descartada.       │
│         Si se resuelve → Causa B confirmada. FIN.           │
│                                                             │
│ Paso 4: Stderr a archivo (no pipe)                          │
│         Si el problema persiste → Causa C descartada.       │
│         Si se resuelve → Causa C confirmada. FIN.           │
│                                                             │
│ Paso 5: Eliminar blackboard fallback directo                │
│         Si el problema persiste → Combinación de causas.    │
│         Si se resuelve → Causa D confirmada. FIN.           │
└─────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Aplicar los 5 fixes juntos es seguro y recomendado.** No se contradicen entre sí.
> El orden de arriba es para diagnóstico aislado si se quiere entender cuál era la causa.
> Si el objetivo es solo que funcione, aplicar los 5 y correr los smokes.

---

## 6. Pipeline de testing

### Fase T0 — Test manual de la reproducción del crash

**Objetivo:** Confirmar que el crash sigue reproduciéndose antes de aplicar fixes.

**Comando:**
```powershell
python smoke_test_rlbridge_persistent_basic.py
```

**Resultado esperado:** FAIL con access violation o timeout.

**Si pasa:** El crash ya fue resuelto por otro cambio reciente. Documentar y continuar
a Fase T3.

---

### Fase T1 — Aplicar todos los fixes y re-testear

**Archivos a modificar:**

1. `bridge/bhuman_bridge.cpp` (copia local del repo RL)
   - Agregar `g_ioMutex` global
   - Proteger todas las escrituras a cout/cerr
   - Robustecer el exit del main loop
   - Agregar shutdown cooperativo post-loop
   - Eliminar fallback directo a blackboard

2. `SabanaHerons2026/Src/Apps/RLBridge/bhuman_bridge.cpp` (árbol externo compilable)
   - Mismos cambios que arriba (debe ser idéntico)

3. `bridge/bridge_protocol.py`
   - Cambiar `CREATE_NO_WINDOW` → `CREATE_NEW_CONSOLE` (o `STARTUPINFO` oculto)
   - Cambiar `stderr=subprocess.PIPE` → `stderr=subprocess.STDOUT`
   - Eliminar stderr reader thread cuando stderr es merged
   - Ajustar `_read_stderr()` para caso merged

**Recompilar:**
```powershell
# Desde el directorio de SabanaHerons2026:
cmake --build Build/Windows/RLBridge --config Release
```

**Test:**
```powershell
python smoke_test_rlbridge_persistent_basic.py
```

**Resultado esperado:** PASS con transport = pipe_process.

---

### Fase T2 — Validar que one-shot no se rompió

**Objetivo:** Confirmar que los cambios no rompieron el fallback.

**Comando:**
```powershell
python smoke_test_environment_bridge.py
```

**Resultado esperado:** PASS (puede usar oneshot o pipe, ambos son válidos).

Adicionalmente, verificar la suite completa:
```powershell
python smoke_suite_option1.py --suite gating
```

---

### Fase T3 — Crear smoke test de estrés para pipe persistente

**Nuevo archivo:** `smoke_test_rlbridge_persistent_stress.py`

```python
"""
smoke_test_rlbridge_persistent_stress.py -- stress test for persistent transport.

Validates that pipe_process survives:
  - 100 consecutive frames
  - 5 reset cycles
  - concurrent decision traffic
  - no memory growth or crash
"""

from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from bridge.bridge_protocol import BHumanBridge, BridgeConfig


def _assert(condition: bool, msg: str):
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def _build_teams():
    own_team = [
        {"number": 1, "x": 0.0, "y": 0.0, "theta": 0.0, "upright": True},
        {"number": 2, "x": -800.0, "y": 500.0, "theta": 0.0, "upright": True},
        {"number": 3, "x": -1200.0, "y": -500.0, "theta": 0.0, "upright": True},
        {"number": 4, "x": -1800.0, "y": 0.0, "theta": 0.0, "upright": True},
    ]
    opp_team = [
        {"number": 1, "x": 1200.0, "y": 0.0, "theta": np.pi, "upright": True},
        {"number": 2, "x": 1500.0, "y": 700.0, "theta": np.pi, "upright": True},
        {"number": 3, "x": 1500.0, "y": -700.0, "theta": np.pi, "upright": True},
        {"number": 4, "x": 2200.0, "y": 0.0, "theta": np.pi, "upright": True},
    ]
    return own_team, opp_team


def main():
    print("[TEST] RLBridge persistent transport STRESS test")

    cfg = BridgeConfig(
        preferred_transport="pipe_process",
        allow_oneshot_fallback=False,
    )
    _assert(Path(cfg.bridge_executable).exists(),
            f"bridge executable exists at {cfg.bridge_executable}")

    bridge = BHumanBridge(cfg)
    own_team, opp_team = _build_teams()
    ball_pos = np.array([200.0, 0.0], dtype=np.float32)
    ball_vel = np.zeros(2, dtype=np.float32)

    FRAMES_PER_EPISODE = 100
    NUM_EPISODES = 5

    try:
        bridge.start()
        _assert(bridge.is_running, "bridge started")
        _assert(bridge._transport_mode == "pipe_process",
                f"transport = pipe_process (got {bridge._transport_mode})")

        all_latencies = []

        for episode in range(NUM_EPISODES):
            print(f"\n[TEST] Episode {episode + 1}/{NUM_EPISODES}")
            if episode > 0:
                bridge.reset()
                _assert(bridge.is_running, "bridge alive after reset")

            episode_latencies = []
            for frame_idx in range(FRAMES_PER_EPISODE):
                t0 = time.perf_counter()
                result = bridge.step(
                    ball_pos=ball_pos, ball_vel=ball_vel,
                    own_team=own_team, opp_team=opp_team,
                    game_state="playing", own_score=0, opp_score=0,
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0
                episode_latencies.append(dt_ms)

                _assert(result.get("type") == "decision",
                        f"ep{episode+1} frame{frame_idx+1}: got decision")
                _assert(result.get("debug_transport_mode") == "pipe_process",
                        f"ep{episode+1} frame{frame_idx+1}: still pipe_process")

                expected_fid = frame_idx + 1
                _assert(result.get("frame_id") == expected_fid,
                        f"ep{episode+1}: frame_id={expected_fid}"
                        f" (got {result.get('frame_id')})")

            mean_ms = sum(episode_latencies) / len(episode_latencies)
            p99_ms = sorted(episode_latencies)[int(0.99 * len(episode_latencies))]
            all_latencies.extend(episode_latencies)
            print(f"  INFO: ep{episode+1}: mean={mean_ms:.1f}ms, "
                  f"p99={p99_ms:.1f}ms, frames={len(episode_latencies)}")

        total_mean = sum(all_latencies) / len(all_latencies)
        total_frames = len(all_latencies)
        _assert(total_mean < 50.0,
                f"overall mean latency < 50ms (got {total_mean:.1f}ms)")
        _assert(bridge.is_running, "bridge still alive at end of stress test")

        print(f"\nRESULT: PASS")
        print(f"  {total_frames} frames across {NUM_EPISODES} episodes.")
        print(f"  Mean latency: {total_mean:.1f}ms")
        print(f"  Pipe process transport survived stress test.")

    finally:
        bridge.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nRESULT: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        raise SystemExit(1)
```

---

### Fase T4 — Integración con environment

**Objetivo:** Validar que `NaoSoccerEnv` funciona con pipe_process.

No hay que cambiar environment.py. Solo hay que cambiar `runtime_paths.py` para que
`pipe_process` sea la preferencia en Windows cuando el binario existe:

```python
# runtime_paths.py, línea 94-104, ya es así:
if os.name == "nt":
    bhuman_root = resolve_bhuman_root()
    built_bridge_candidates = (...)
    if any(candidate.exists() for candidate in built_bridge_candidates):
        return "pipe_process"  # YA DEVUELVE pipe_process si el binario existe
    return "oneshot_file_fallback"
```

**Test:**
```powershell
# Forzar pipe_process para el smoke del environment:
set BHUMAN_BRIDGE_TRANSPORT=pipe_process
python smoke_test_environment_bridge.py
```

**Resultado esperado:** PASS con `bridge_debug_transport_mode = pipe_process`.

---

### Fase T5 — Training loop quick debug

**Test:**
```powershell
set BHUMAN_BRIDGE_TRANSPORT=pipe_process
python train.py --quick-debug
```

**Métricas a observar en el watchdog:**
- `env_step_mean`: debe caer drásticamente (de ~56ms a ~5-15ms)
- `bridge_decision_time_us`: debe mantenerse en ~2000-3000us
- No debe haber crashes ni fallback silencioso a oneshot

---

## 7. Plan de integración final

### Si los 5 fixes resuelven el crash:

1. **Mantener `oneshot_file_fallback` como fallback.** No eliminarlo.
   Los cambios en `bridge_protocol.py` ya hacen esto (L367-377, L458-496).

2. **No cambiar el `.env` por defecto.** El usuario puede forzar pipe con:
   ```
   BHUMAN_BRIDGE_TRANSPORT=pipe_process
   ```

3. **Actualizar `runtime_paths.py`** para que `pipe_process` sea el default
   en Windows cuando el binario compilado existe (esto ya es así en L94-103).

4. **Actualizar `BHUMAN_PERSISTENT_BRIDGE_PLAN.md`** con las métricas nuevas.

5. **Actualizar `RLBRIDGE_INTERVENTION_LOG.md`** con la causa raíz confirmada.

6. **Actualizar `ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md` §9.2** para quitar
   la restricción de "no reintroducir pipe_process" y reemplazarla con la
   nota de que pipe_process requiere los 5 fixes aplicados.

### Respuesta a las 4 preguntas obligatorias (§7 de la guía):

1. **¿Estoy cambiando Opción 1, 2, o ambas?** → Solo Opción 1 (transporte)
2. **¿Rompe el baseline?** → No. Oneshot sigue siendo fallback
3. **¿Inventa MotionInfo?** → No. No toca la lectura de representaciones
4. **¿El smoke sigue siendo honesto?** → Sí. Los smokes existentes no cambian

---

## 8. Resumen de archivos a modificar

| Archivo | Cambio | Causa que resuelve |
|---|---|---|
| `bridge/bhuman_bridge.cpp` | Mutex I/O global | A |
| `bridge/bhuman_bridge.cpp` | Shutdown cooperativo post-loop | E |
| `bridge/bhuman_bridge.cpp` | `cin.good()` guard | E |
| `bridge/bhuman_bridge.cpp` | Eliminar blackboard fallback directo | D |
| `SabanaHerons2026/.../bhuman_bridge.cpp` | Mismos cambios (copia sincronizada) | A, D, E |
| `bridge/bridge_protocol.py` | `CREATE_NO_WINDOW` → `CREATE_NEW_CONSOLE` | B |
| `bridge/bridge_protocol.py` | `stderr=STDOUT` (merged) | C |
| `bridge/bridge_protocol.py` | Ajustar reader threads | C |
| `smoke_test_rlbridge_persistent_stress.py` | **[NUEVO]** stress test | testing |

---

## 9. Tiempos esperados post-fix

| Métrica | Oneshot actual | Pipe post-fix (R5 5000) | Pipe post-fix (R9 275HX) |
|---|---|---|---|
| step latency | ~56 ms | ~5-12 ms | ~3-8 ms |
| 96-step episode | ~5.4 s | ~0.5-1.2 s | ~0.3-0.8 s |
| 455-step Stage 0 | ~25.5 s | ~2.3-5.5 s | ~1.4-3.6 s |
| 1M steps training | ~31 h | ~2.8-6.7 h | ~1.7-4.4 h |

> [!IMPORTANT]
> La mayor ganancia viene del fix del pipe, no del hardware.
> El Ryzen 9 AI 275HX sobre pipe persistente sería ~**8-18× más rápido** que el
> Ryzen 5 5000 sobre oneshot actual.

---

## 10. Qué hacer si las 5 hipótesis están equivocadas

Si se aplican los 5 fixes del §4 y el crash persiste, las hipótesis A-E no eran
la causa raíz (o lo son parcialmente). Esta sección define:

1. **6 causas adicionales** que no están cubiertas arriba
2. **4 herramientas de diagnóstico de escalación** para obtener evidencia real
3. **3 tests de reproducción mínima** que eliminan variables

---

### 10.1. Causas adicionales no cubiertas en las hipótesis A-E

---

#### Causa F: Variables `static` locales en el JSON parser son thread-unsafe

**Problema descubierto en el código:**

En `bhuman_bridge.cpp` líneas 187-193, 201, la clase `JsonValue` tiene:

```cpp
const std::string& asString() const {
    static std::string empty;        // ← COMPARTIDO entre hilos
    return isString() ? std::get<std::string>(data) : empty;
}
const JsonArray& asArray() const {
    static JsonArray empty;          // ← COMPARTIDO entre hilos
    return isArray() ? std::get<JsonArray>(data) : empty;
}
const JsonObject& asObject() const {
    static JsonObject empty;         // ← COMPARTIDO entre hilos
    return isObject() ? std::get<JsonObject>(data) : empty;
}
const JsonValue& operator[](const std::string& key) const {
    static JsonValue null;           // ← COMPARTIDO entre hilos
    // ...
}
const JsonValue& operator[](size_t idx) const {
    static JsonValue null;           // ← COMPARTIDO entre hilos
    // ...
}
```

En C++11+, la inicialización de `static` locales es thread-safe, pero **el acceso
posterior no lo es**. Si un hilo del framework BHuman de alguna forma invoca código
que usa `JsonValue` (poco probable pero posible si hay logging con JSON), dos hilos
podrían estar leyendo/escribiendo la misma referencia `static`.

**Cómo diagnosticarlo:**
- Reemplazar todas las variables `static` locales con variables de instancia o
  con `thread_local static`:
  ```cpp
  const std::string& asString() const {
      thread_local std::string empty;  // Cada hilo tiene su propia copia
      return isString() ? std::get<std::string>(data) : empty;
  }
  ```

**Probabilidad:** Baja (solo si BHuman invoca código que toca JsonValue).

---

#### Causa G: El CRT de MSVC usa `_isatty()` para decidir el modo de buffering

**Mecanismo:**
Cuando `stdout` es un pipe en Windows, el CRT de MSVC cambia `std::cout` a buffering
completo (no line-buffered). Esto significa que `std::endl` puede no vaciar el buffer
inmediatamente. El `flush()` explícito lo fuerza, pero si hay código interno de BHuman
que escribe a `stdout` o `stderr` sin `flush()`, los datos se acumulan y se envían en
bloques parciales. Python, leyendo desde el otro lado del pipe con un reader thread,
puede recibir JSON cortado a mitad de línea.

Si `_recv()` del lado Python procesa JSON truncado y lo descarta, pero el bridge C++
ya avanzó su estado interno, se produce una desincronización de protocolo que puede
llevar a comportamiento indefinido en frames posteriores.

**Cómo diagnosticarlo:**
- En `bhuman_bridge.cpp`, agregar después de `main()` (línea 1921):
  ```cpp
  // Forzar stdout completamente unbuffered programáticamente
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
  ```
- Verificar si esto cambia el comportamiento.

**Probabilidad:** Media. El `cout.flush()` explícito debería ser suficiente, pero
`setvbuf` afecta también el nivel C de stdio que BHuman podría usar internamente.

---

#### Causa H: El framework BHuman usa `ASSERT()` o `OUTPUT_ERROR()` con formateo de consola

**Mecanismo:**
El framework BHuman define macros como `ASSERT()`, `OUTPUT_ERROR()`, `OUTPUT_WARNING()`
que internamente pueden llamar a funciones de formateo de error que asumen una consola
Win32 activa. En modo headless (sin ventana), estas macros pueden intentar abrir una
ventana de diálogo de error (`MessageBox`) o acceder a un handle de consola nulo.

En modo one-shot esto no se observa porque el proceso muere antes de que se triggereen
las condiciones de assert. En modo persistente, con múltiples frames, la probabilidad
de que un assert interno falle se multiplica.

**Cómo diagnosticarlo:**
- Buscar en el output de `stderr` mensajes que empiecen con `ASSERT` o `OUTPUT_ERROR`
  antes del crash
- En `bhuman_bridge.cpp`, definir antes de cualquier include:
  ```cpp
  #define NDEBUG  // Desactiva asserts de C stdlib
  ```
- O redirigir el handler de assert de BHuman:
  ```cpp
  #ifdef WINDOWS
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  #endif
  ```

**Probabilidad:** Media-Alta. BHuman es un framework con muchos `ASSERT()`.

---

#### Causa I: Stack overflow por la secuencia de bootstrap de 4 estados

**Mecanismo:**
En `processFrame()` (L903-985), cuando `operationalWarmupDone_` es `false`, se envían
**4 frames de bootstrap** (initial → ready → set → playing). Cada uno invoca:
- `bridgeSender.bin()` × 11 representaciones
- `bridgeSender.send(true)`
- `waitForCognitionFrameFinished()` con timeout de 1000ms

Si el stack estándar de un hilo de Windows es de 1 MB y la serialización BHuman
acumula objetos grandes en el stack (como `GroundTruthWorldState` con todos los
jugadores), 4 iteraciones × 11 representaciones puede acercarse al límite,
especialmente si BHuman internamente hace recursión profunda durante el
módulo graph runner.

**Cómo diagnosticarlo:**
- Compilar con `/STACK:4194304` (4 MB) y ver si desaparece:
  ```cmake
  # En el CMakeLists.txt de RLBridge:
  set_target_properties(RLBridge PROPERTIES LINK_FLAGS "/STACK:4194304")
  ```
- O agregar al `main()`:
  ```cpp
  #ifdef WINDOWS
  // Verificar espacio de stack restante
  ULONG_PTR lowLimit, highLimit;
  GetCurrentThreadStackLimits(&lowLimit, &highLimit);
  std::cerr << "[bridge] Stack: " << (highLimit - lowLimit) / 1024
            << "KB total" << std::endl;
  #endif
  ```

**Probabilidad:** Baja-Media. Depende de la profundidad del módulo graph.

---

#### Causa J: El antivirus o Windows Defender interfiere con el pipe I/O

**Mecanismo:**
Windows Defender y otros antivirus pueden interceptar y escanear I/O de procesos
que se comunican por pipes. Si el antivirus demora o bloquea una escritura/lectura:
- Python puede ver un timeout y matar el proceso
- El proceso C++ puede recibir un error de I/O inesperado

Esto NO causa un access violation directamente, pero sí puede causar la secuencia
deadlock → kill → crash (similar a Causa C pero diferente raíz).

**Cómo diagnosticarlo:**
- Agregar el directorio del proyecto a las exclusiones de Windows Defender:
  ```powershell
  Add-MpPreference -ExclusionPath "C:\ALEJO"
  Add-MpPreference -ExclusionProcess "RLBridge.exe"
  ```
- Temporalmente desactivar "Real-Time Protection" y re-testear
- Usar Process Monitor (ver §10.2) para ver si hay delays en operaciones I/O

**Probabilidad:** Baja (no explicaría un access violation, sí un timeout).

---

#### Causa K: Heap corruption latente en el framework BHuman

**Mecanismo:**
Si existe un bug de memoria (buffer overrun, use-after-free, double free) en el código
de BHuman que solo se manifiesta en modo persistente (porque en one-shot el proceso muere
antes de que el heap corrupto se reutilice), el access violation aparecería en un lugar
completamente diferente del código que causó la corrupción.

Este es el peor escenario: la dirección del crash (`ExceptionAddress`) no tiene relación
directa con la causa real.

**Cómo diagnosticarlo:**
- Compilar con Address Sanitizer (ASan):
  ```cmake
  # En CMakeLists.txt de RLBridge:
  target_compile_options(RLBridge PRIVATE /fsanitize=address)
  target_link_options(RLBridge PRIVATE /fsanitize=address)
  ```
- O usar Application Verifier (ver §10.2)
- El output de ASan mostrará el sitio ORIGINAL de la corrupción, no el crash

**Probabilidad:** Media. Dependería de bugs preexistentes en BHuman.

---

### 10.2. Herramientas de diagnóstico de escalación

Si las causas A-K no se confirman con los tests propuestos, usar estas herramientas
para obtener evidencia real del sistema operativo.

---

#### Herramienta 1: WinDbg (Debugger de Windows)

**Para qué:** Obtener el **stack trace real** del crash, con nombre de función y
línea exacta. Sin esto, todas las hipótesis son especulaciones.

**Cómo usarlo:**

1. Instalar WinDbg Preview desde Microsoft Store
2. Modificar `bridge_protocol.py` para no matar el proceso al crash:
   ```python
   # Temporalmente en start(), agregar después de Popen:
   print(f"RLBridge PID: {self._process.pid}")
   input("Attach WinDbg and press Enter...")
   ```
3. Abrir WinDbg → File → Attach to Process → PID del bridge
4. En WinDbg: `g` (go) para continuar la ejecución
5. Cuando crashee, WinDbg atrapa la excepción. Ejecutar:
   ```
   !analyze -v
   k
   .ecxr
   k
   ```
6. Copiar el output completo — contiene:
   - Módulo exacto donde ocurrió el crash
   - Nombre de función
   - Offset del crash dentro de la función
   - Stack completo de la cadena de llamadas

**Output esperado:** Algo como:
```
EXCEPTION_RECORD: 0xc0000005 -- access violation
READ_ADDRESS: 0x0000000000000000

STACK:
RLBridge!BridgeRunner::readDecisions+0x1a4
RLBridge!BridgeRunner::processFrame+0x3f2
RLBridge!main+0x285
```

**Esto confirmaría O descartaría** todas las hipótesis a la vez.

---

#### Herramienta 2: Application Verifier (AppVerif)

**Para qué:** Detectar heap corruption, handle leaks, y race conditions que
no se manifiestan directamente como el crash observado.

**Cómo usarlo:**

1. Instalar Application Verifier desde Windows SDK
2. Ejecutar `appverif.exe` → agregar `RLBridge.exe`
3. Activar las siguientes verificaciones:
   - Heap (detecta buffer overruns, use-after-free)
   - Handles (detecta handle leaks y doble-cierre)
   - Locks (detecta deadlocks y lock ordering)
4. Correr el smoke test — AppVerif hará que el crash sea más determinístico
   y mostrará la causa raíz

**Output esperado:** Un `VerifierStop` con el site de la primera violación,
que puede ser diferente del crash observable.

---

#### Herramienta 3: Process Monitor (Sysinternals ProcMon)

**Para qué:** Ver TODA la actividad I/O del proceso segundo a segundo, incluyendo
operaciones de archivo, registro, red, y pipes. Detecta:
- Pipes que se llenan
- Escrituras que fallan
- DLLs que se cargan tarde
- Accesos a rutas que no existen

**Cómo usarlo:**

1. Descargar Process Monitor de Sysinternals
2. Filtrar por `Process Name = RLBridge.exe`
3. Correr el smoke test
4. Buscar operaciones con resultado `BUFFER OVERFLOW`, `ACCESS DENIED`,
   `NAME NOT FOUND` justo antes del crash

**Este test es especialmente útil** si la causa es J (antivirus) o una
dependencia de DLL que falla.

---

#### Herramienta 4: Crash dump automático (sin debugger)

**Para qué:** Capturar un minidump cuando el crash ocurre, sin necesidad
de tener WinDbg abierto en el momento.

**Cómo configurarlo:**

```powershell
# Registrar RLBridge.exe para generar crash dumps automáticos:
New-Item -Path "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\RLBridge.exe" -Force
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\RLBridge.exe" `
    -Name "DumpFolder" -Value "C:\ALEJO\RL\bridge\crashdumps" -Type String
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\RLBridge.exe" `
    -Name "DumpType" -Value 2 -Type DWord  # Full dump
New-Item -Path "C:\ALEJO\RL\bridge\crashdumps" -ItemType Directory -Force
```

Después del crash, abrir el `.dmp` con WinDbg o Visual Studio para análisis.

---

### 10.3. Tests de reproducción mínima (eliminan variables)

Si las herramientas de §10.2 no están disponibles o son lentas, estos tests
ayudan a aislar el problema quitando capas.

---

#### Test M1: Pipe persistente desde cmd.exe puro (sin Python)

**Objetivo:** ¿El crash es del bridge C++ o del wiring con Python?

Crear archivo `bridge/test_pipe_persistent.cmd`:
```cmd
@echo off
echo [TEST] Pipe persistente desde cmd.exe

echo {"type":"init","scenario":"2D","location":"4v4_Scaled","team_number":1,"player_number":1,"num_players":4,"bridge_mode":"persistent_2d_cognition"} > %TEMP%\rlbridge_pipe_test.jsonl
echo {"type":"frame","frame_id":1,"timestamp_ms":33,"ball":{"x":200,"y":0,"vx":0,"vy":0},"own_team":[{"number":1,"x":0,"y":0,"theta":0,"upright":true},{"number":2,"x":-800,"y":500,"theta":0,"upright":true},{"number":3,"x":-1200,"y":-500,"theta":0,"upright":true},{"number":4,"x":-1800,"y":0,"theta":0,"upright":true}],"opp_team":[{"number":1,"x":1200,"y":0,"theta":3.14,"upright":true},{"number":2,"x":1500,"y":700,"theta":3.14,"upright":true},{"number":3,"x":1500,"y":-700,"theta":3.14,"upright":true},{"number":4,"x":2200,"y":0,"theta":3.14,"upright":true}],"game_state":"playing","own_score":0,"opp_score":0} >> %TEMP%\rlbridge_pipe_test.jsonl
echo {"type":"frame","frame_id":2,"timestamp_ms":66,"ball":{"x":200,"y":0,"vx":0,"vy":0},"own_team":[{"number":1,"x":0,"y":0,"theta":0,"upright":true},{"number":2,"x":-800,"y":500,"theta":0,"upright":true},{"number":3,"x":-1200,"y":-500,"theta":0,"upright":true},{"number":4,"x":-1800,"y":0,"theta":0,"upright":true}],"opp_team":[{"number":1,"x":1200,"y":0,"theta":3.14,"upright":true},{"number":2,"x":1500,"y":700,"theta":3.14,"upright":true},{"number":3,"x":1500,"y":-700,"theta":3.14,"upright":true},{"number":4,"x":2200,"y":0,"theta":3.14,"upright":true}],"game_state":"playing","own_score":0,"opp_score":0} >> %TEMP%\rlbridge_pipe_test.jsonl
echo {"type":"frame","frame_id":3,"timestamp_ms":99,"ball":{"x":200,"y":0,"vx":0,"vy":0},"own_team":[{"number":1,"x":0,"y":0,"theta":0,"upright":true},{"number":2,"x":-800,"y":500,"theta":0,"upright":true},{"number":3,"x":-1200,"y":-500,"theta":0,"upright":true},{"number":4,"x":-1800,"y":0,"theta":0,"upright":true}],"opp_team":[{"number":1,"x":1200,"y":0,"theta":3.14,"upright":true},{"number":2,"x":1500,"y":700,"theta":3.14,"upright":true},{"number":3,"x":1500,"y":-700,"theta":3.14,"upright":true},{"number":4,"x":2200,"y":0,"theta":3.14,"upright":true}],"game_state":"playing","own_score":0,"opp_score":0} >> %TEMP%\rlbridge_pipe_test.jsonl
echo {"type":"shutdown"} >> %TEMP%\rlbridge_pipe_test.jsonl

REM Ejecutar con PIPE real (no redirección de archivo):
type %TEMP%\rlbridge_pipe_test.jsonl | RLBridge.exe

echo Exit code: %ERRORLEVEL%
```

**Interpretación:**
- Si pasa con `exit code 0` → **El crash es culpa de Python**, no del C++.
  Investigar `Popen`, `bufsize`, `creationflags`, o los reader threads.
- Si crashea también aquí → **El crash es del C++**, proceder con WinDbg.

**Nota:** El `|` (pipe) de cmd.exe crea un pipe anónimo real, idéntico al de Python.

---

#### Test M2: Pipe persistente pero con un solo frame (mínimo absoluto)

**Objetivo:** ¿El crash ocurre en el primer frame o solo después de varios?

```cmd
@echo off
echo {"type":"init","scenario":"2D","location":"4v4_Scaled","team_number":1,"player_number":1,"num_players":4,"bridge_mode":"persistent_2d_cognition"} > %TEMP%\rlbridge_min.jsonl
echo {"type":"frame","frame_id":1,"timestamp_ms":33,"ball":{"x":200,"y":0,"vx":0,"vy":0},"own_team":[{"number":1,"x":0,"y":0,"theta":0,"upright":true}],"opp_team":[{"number":1,"x":1200,"y":0,"theta":3.14,"upright":true}],"game_state":"playing","own_score":0,"opp_score":0} >> %TEMP%\rlbridge_min.jsonl
echo {"type":"shutdown"} >> %TEMP%\rlbridge_min.jsonl

type %TEMP%\rlbridge_min.jsonl | RLBridge.exe
echo Exit code: %ERRORLEVEL%
```

**Interpretación:**
- Si pasa → El crash ocurre con frames múltiples (Causa A o D más probable)
- Si crashea → El crash ocurre en el primer frame (Causa B, H, I, o interno)

---

#### Test M3: Bridge C++ con stdin redirigido desde archivo (no pipe)

**Objetivo:** ¿El problema es los PIPES o simplemente el modo persistente?

```cmd
@echo off
REM Mismos datos que M1 pero con redirección de archivo, no pipe
RLBridge.exe < %TEMP%\rlbridge_pipe_test.jsonl > %TEMP%\rlbridge_out.txt 2>&1
echo Exit code: %ERRORLEVEL%
type %TEMP%\rlbridge_out.txt
```

**Interpretación:**
- Si M1 crashea pero M3 pasa → **El problema ES el pipe** (vs. archivo). Causa C.
- Si ambos crashean → El problema es el modo persistente en sí (no el transporte).
  Investigar el C++ (Causa D, H, I, K).
- Si M1 pasa y M3 pasa → El problema es específico de Python (wiring de `Popen`).

---

### 10.4. Árbol de decisión completo

```
¿El crash se reproduce HOY?
├─ NO → Documentar, seguir a T3 (stress test)
└─ SÍ
   │
   ├─ Aplicar los 5 fixes (A-E) juntos
   │  ├─ ¿Crash desaparece? → ÉXITO. Aislar cuál fix fue el clave (opcional)
   │  └─ ¿Crash persiste?
   │     │
   │     ├─ Correr Test M1 (pipe desde cmd.exe puro)
   │     │  ├─ PASA → Problema en Python. Investigar:
   │     │  │         - reader threads
   │     │  │         - bufsize
   │     │  │         - creationflags
   │     │  │         - timing de _recv()
   │     │  └─ FALLA → Problema en C++. Correr Test M3
   │     │     ├─ M3 PASA → Problema específico de PIPE. Verificar:
   │     │     │             - Causa G (_isatty/buffering)
   │     │     │             - Causa J (antivirus)
   │     │     │             - setvbuf(stdout, NULL, _IONBF, 0)
   │     │     └─ M3 FALLA → Problema del modo persistente. Usar:
   │     │                    - WinDbg (Herramienta 1) para stack trace
   │     │                    - ASan (Causa K) para heap corruption
   │     │                    - Test M2 para aislar primer frame vs. multi
   │     │
   │     ├─ Test M2 (un solo frame)
   │     │  ├─ PASA → Problema acumulativo (frame N>1). Causas: A, D, F, K
   │     │  └─ FALLA → Problema en init/bootstrap. Causas: B, H, I
   │     │
   │     └─ Si nada de lo anterior funciona:
   │        → WinDbg para stack trace REAL
   │        → El stack trace descarta/confirma TODAS las hipótesis a la vez
   │        → Actualizar este plan con la causa descubierta
```

---

### 10.5. Protocolo de documentación cuando se encuentra la causa real

Cuando se identifique la causa raíz (por cualquier vía), documentar:

1. **En `RLBRIDGE_INTERVENTION_LOG.md`:**
   - Causa raíz exacta (cuál de las A-K, o una nueva)
   - El stack trace o evidencia que la confirmó
   - El fix aplicado
   - Los tests que validan el fix

2. **En `ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md` §9.2:**
   - Actualizar la restricción: ya no es "no reintroducir pipe_process"
   - Sino: "pipe_process requiere [los fixes específicos aplicados]"

3. **En `BHUMAN_PERSISTENT_BRIDGE_PLAN.md` §1:**
   - Actualizar métricas con los tiempos reales post-fix
   - Marcar las fases como completadas

> [!CAUTION]
> Si la causa resulta ser Causa K (heap corruption en BHuman), el fix puede requerir
> modificar código del framework BHuman en `SabanaHerons2026/Src/Libs/`, lo cual es
> significativamente más invasivo y requiere coordinación con el equipo de BHuman.
> En ese caso, mantener `oneshot_file_fallback` como ruta principal hasta que el fix
> en BHuman se valide exhaustivamente.
