# Plan de Diagnostico, Fix e Integracion de `pipe_process` en Windows

> Estado documental: historico
>
> Este archivo ya no describe el baseline vigente del proyecto.
> El transporte `pipe_process` forma parte del baseline actual y este documento
> debe leerse como postmortem / plan de diagnostico de la fase en que aun era
> una ruta inestable.
>
> Fuentes de verdad actuales:
>
> - [PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md](/c:/ALEJO/RL/PROJECT_CHECKLIST_BHUMAN_IN_LOOP.md)
> - [BHUMAN_IN_LOOP_COMPACT_GUIDE.md](/c:/ALEJO/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_COMPACT_GUIDE.md)
> - [DOCUMENTATION_AND_SMOKE_GOVERNANCE.md](/c:/ALEJO/RL/Documentacion/PLAN/DOCUMENTATION_AND_SMOKE_GOVERNANCE.md)
## 1. Contexto y objetivo

El transporte `pipe_process` (stdin/stdout persistente) reducirÃ­a el tiempo por step de **~56 ms** a **~2-5 ms**, pero fue desactivado en Windows por causar `access violation (0xc0000005)` dentro de `RLBridge.exe`.

Este plan define:
1. CÃ³mo diagnosticar cuÃ¡l de las 5 causas raÃ­z posibles provoca el crash
2. El fix exacto para cada causa, archivo por archivo
3. Un pipeline de testing progresivo que valida cada fix antes del siguiente
4. IntegraciÃ³n final sin romper la ruta estable `oneshot_file_fallback`

---

## 2. Arquitectura de referencia (para entender el crash)

```
Python (main thread)                        C++ RLBridge.exe
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ bridge_protocol.py      â”‚                â”‚ main()                           â”‚
â”‚                         â”‚                â”‚   â””â”€ while(getline(cin, line))   â”‚
â”‚ start():                â”‚   stdin PIPE   â”‚        â”œâ”€ "init" â†’ bridge.init() â”‚
â”‚   Popen(PIPE,PIPE,PIPE) â”‚ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€>â”‚        â”‚    â””â”€ Robot::start()    â”‚
â”‚   _send(init)           â”‚                â”‚        â”‚        â”œâ”€ Cognition thr  â”‚
â”‚   _recv() â†’ "ready"     â”‚   stdout PIPE  â”‚        â”‚        â”œâ”€ Motion thread  â”‚
â”‚                         â”‚ <â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚        â”‚        â””â”€ Debug thread   â”‚
â”‚ step():                 â”‚   stdin PIPE   â”‚        â”œâ”€ "frame" â†’ processFrame()â”‚
â”‚   _send_raw(frame_json) â”‚ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€>â”‚        â”‚    â”œâ”€ sender.bin(...)    â”‚
â”‚   _recv() â†’ "decision"  â”‚   stdout PIPE  â”‚        â”‚    â”œâ”€ sender.send(true) â”‚
â”‚                         â”‚ <â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚        â”‚    â”œâ”€ waitForCognition() â”‚
â”‚ _stderr_reader_thread   â”‚   stderr PIPE  â”‚        â”‚    â”œâ”€ readDecisions()   â”‚
â”‚   (drena stderr async)  â”‚ <â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚        â”‚    â””â”€ cout << JSON      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                â”‚        â””â”€ "shutdown" â†’ break     â”‚
                                           â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**Punto crÃ­tico:** En modo `pipe_process`, stdout/stderr de `RLBridge.exe` son pipes anÃ³nimos
del kernel de Windows, **no una consola real**. Todos los hilos del framework BHuman comparten
esos handles implÃ­citamente a travÃ©s de la librerÃ­a CRT de MSVC.

---

## 3. Resumen del estado actual del cÃ³digo

### Lado Python â€” [bridge_protocol.py](file:///c:/ALEJO/RL/bridge/bridge_protocol.py)

| Componente | Estado | LÃ­neas clave |
|---|---|---|
| `start()` con pipes | Implementado, funciona parcialmente | L316-405 |
| Reader threads (stdout/stderr) | Implementados con `threading.Thread` | L626-685 |
| Fallback a oneshot en handshake | Implementado | L367-377 |
| Fallback a oneshot en step | Implementado | L458-496 |
| `CREATE_NO_WINDOW` flag | Activa en Windows | L344-345 |
| `bufsize=0` (unbuffered) | Activo | L342 |

### Lado C++ â€” [bhuman_bridge.cpp](file:///c:/ALEJO/RL/bridge/bhuman_bridge.cpp)

| Componente | Estado | LÃ­neas clave |
|---|---|---|
| Main loop `getline(cin, line)` | Funcional | L1931 |
| `cout << json << endl; cout.flush()` | Sin mutex | L1978, L1984, L1993, L2006 |
| `cerr << ...` (logs) | Concurrente desde main + hilos BHuman | L816-850, L1038-1072 |
| `SetErrorMode` + crash handler | Activo | L1922-1924 |
| `Robot::start()` (lanza hilos) | Funcional | L821 |
| `readDecisions()` con `setGlobals()` cross-thread | Funcional pero sin sync | L1602-1767 |

### Smoke test existente â€” [smoke_test_rlbridge_persistent_basic.py](file:///c:/ALEJO/RL/smoke_test_rlbridge_persistent_basic.py)

Ya existe y valida exactamente lo que necesitamos: 3 frames + reset + 1 frame, todo
con `preferred_transport="pipe_process"` y `allow_oneshot_fallback=False`.

---

## 4. DiagnÃ³stico: 5 causas raÃ­z posibles

El crash `0xc0000005` al usar pipes en Windows puede venir de **una o varias** de estas
fuentes. Cada una tiene un mecanismo diferente y un fix distinto. Es obligatorio diagnosticar
cuÃ¡l(es) estÃ¡n activas antes de declarar un fix.

---

### Causa A: Escritura concurrente a `stdout` sin mutex

**Mecanismo:**
`BridgeRunner::processFrame()` escribe a `std::cout` desde el hilo principal. Pero el framework
BHuman puede escribir a `stdout` internamente (logging del framework, asserts, o salida del
`Communication` thread) desde sus propios hilos (Cognition, Motion, Debug). El MSVC CRT
sincroniza parcialmente `std::cout` con un lock nivel-stream, pero no protege contra
corrupciones del `streambuf` interno cuando hay mÃºltiples escritores concurrentes en un pipe
anÃ³nimo sin consola real.

**Evidencia esperada si es esta causa:**
- El crash ocurre **durante o justo despuÃ©s** de `cout << jsonSerialize(result)` (L1984)
- El stack trace apunta a `_write` o `_flushall` del CRT
- Ocurre de forma intermitente, no en el primer frame

**DiagnÃ³stico:**

Agregar un `std::mutex` temporal alrededor de cada escritura a `cout` y `cerr` y ver si
el crash desaparece.

**Fix definitivo:**

Archivo: `bridge/bhuman_bridge.cpp` (y la copia en `SabanaHerons2026`)

```cpp
// AÃ±adir al inicio del archivo, despuÃ©s de los includes (lÃ­nea ~57):
namespace {
    std::mutex g_ioMutex;  // Protege stdout/stderr contra escritura concurrente
}

// Reemplazar CADA escritura a cout/cerr con bloques protegidos.
// Ejemplo en main() lÃ­nea 1978:
{
    std::lock_guard<std::mutex> lock(g_ioMutex);
    std::cout << jsonSerialize(JsonValue(response)) << std::endl;
    std::cout.flush();
}

// Ejemplo en main() lÃ­nea 1926:
{
    std::lock_guard<std::mutex> lock(g_ioMutex);
    std::cerr << "[bridge] BHuman Bridge v1.0 starting" << std::endl;
}
```

**LÃ­neas exactas a proteger en `main()` (L1920-2013):**
- L1926: cerr del startup
- L1941-1942: cout del error JSON
- L1978-1979: cout del ready response
- L1984-1985: cout del frame decision
- L1993-1994: cout del reset ack
- L2006-2007: cout del error desconocido
- L2011: cerr del exit

**LÃ­neas a proteger en `BridgeRunner` (mÃ©todos que escriben a cerr):**
- L815-850: init() cerr logs
- L1038-1072: shutdown() cerr logs
- L1021-1025: reset() cerr logs
- L127-137: `verboseBridgeLog()` â€” usar el mismo mutex

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

**Volumen del cambio:** ~30 sitios en `bhuman_bridge.cpp`. No cambia lÃ³gica, solo envuelve I/O.

---

### Causa B: `CREATE_NO_WINDOW` elimina la consola y BHuman asume que existe

**Mecanismo:**
En `bridge_protocol.py` lÃ­nea 344-345, se usa `subprocess.CREATE_NO_WINDOW`. Esto no crea
**ningÃºn** recurso de consola para el proceso hijo. El framework BHuman o el CRT de MSVC
pueden invocar internamente funciones Win32 como:
- `GetConsoleScreenBufferInfo()`
- `SetConsoleCtrlHandler()`
- `GetConsoleWindow()`
- `WriteConsoleW()` (versus `WriteFile()` para pipes)

Si cualquiera de estas se llama con un handle nulo o invÃ¡lido, crash.

**Evidencia esperada si es esta causa:**
- El crash ocurre **durante `init()`** (L764-857), antes de procesar el primer frame
- Ocurre siempre, reproduciblmente
- El proceso muere con `0xc0000005` en una direcciÃ³n dentro de `kernel32.dll` o `ucrtbase.dll`

**DiagnÃ³stico:**

Cambiar `CREATE_NO_WINDOW` por `CREATE_NEW_CONSOLE` en Python y ver si el crash desaparece.

**Fix definitivo:**

Archivo: `bridge/bridge_protocol.py`

```python
# LÃ­nea 344-345, REEMPLAZAR:
if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
    startup_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

# CON:
if os.name == "nt":
    # CREATE_NEW_CONSOLE asigna una consola real (oculta) al proceso hijo.
    # Esto evita que el framework BHuman crashee al intentar usar APIs
    # de consola Win32 con handles nulos. La consola no aparece al usuario
    # porque el proceso no tiene ventana visible, pero sÃ­ existe como recurso
    # del kernel.
    # 
    # NO usar CREATE_NO_WINDOW: BHuman asume que hay una consola vÃ¡lida.
    startup_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
```

**Nota:** Si la soluciÃ³n real es que BHuman necesita una consola pero NO queremos que el
usuario vea una ventana nueva, usar en cambio:
```python
# Alternativa: consola existente pero oculta (STARTUPINFO)
si = subprocess.STARTUPINFO()
si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
si.wShowWindow = 0  # SW_HIDE
startup_kwargs["startupinfo"] = si
# NO usar creationflags en este caso â€” hereda la consola del padre
```

**Volumen:** 3-5 lÃ­neas en `bridge_protocol.py`.

---

### Causa C: Buffer de pipe lleno causa deadlock â†’ timeout â†’ kill â†’ crash

**Mecanismo:**
Los pipes anÃ³nimos de Windows tienen un buffer de **4 KB a 64 KB** por defecto (depende
de la versiÃ³n del OS). BHuman, durante `init()`, ejecuta `Robot::start()` que:
1. Inicializa el framework
2. Carga mÃ³dulos (decenas de cerr logs)
3. Los hilos arrancan y generan logs adicionales

Si `stderr` se llena antes de que Python lo drene, la siguiente escritura a `cerr` **bloquea**
el hilo que la intenta. Si ese hilo es el main del bridge, nunca llega a enviar `ready` por
`stdout`. Python ve un timeout, mata el proceso con `kill()`, y la terminaciÃ³n abrupta de
BHuman con hilos activos genera el access violation.

**Evidencia esperada si es esta causa:**
- El `start()` de Python hace timeout en `_recv()` (L363)
- `poll_code` es `None` (proceso sigue vivo pero colgado)
- stderr contiene logs parciales que se cortan a mitad de frase
- El crash ocurre cuando Python llama `_process.kill()`

**DiagnÃ³stico:**

Aumentar el buffer del pipe de stderr a 1 MB y ver si el handshake completa.

**Fix definitivo:**

Archivo: `bridge/bridge_protocol.py`

OpciÃ³n 1 â€” Redirigir stderr a un archivo en lugar de pipe:
```python
# En start(), lÃ­nea 336-343, REEMPLAZAR:
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

OpciÃ³n 2 â€” Stderr mergeado con stdout (como en one-shot estable):
```python
startup_kwargs: Dict[str, Any] = {
    "args": [str(exe)],
    "stdin": subprocess.PIPE,
    "stdout": subprocess.PIPE,
    "stderr": subprocess.STDOUT,  # 2>&1 â€” merged con stdout
    "env": env,
    "bufsize": 0,
}
```

> [!WARNING]
> La OpciÃ³n 2 requiere que el reader de stdout distinga entre JSON y logs de texto.
> Esto ya estÃ¡ implementado en `_recv()` (L614-616) que hace `skip non-JSON lines`.

**Si se usa OpciÃ³n 2**, tambiÃ©n hay que ajustar:
- `_start_reader_threads()` (L626-650): no crear stderr reader thread si stderr es `STDOUT`
- `_read_stderr()` (L726-729): retornar "(merged into stdout)" si no hay deque

**Volumen:** 5-15 lÃ­neas en `bridge_protocol.py` + posible eliminaciÃ³n del stderr reader.

---

### Causa D: `readDecisions()` cross-thread blackboard sin sincronizaciÃ³n real

**Mecanismo:**
En `readDecisions()` L1602-1767, el bridge llama `cognition_->setGlobals()` y luego lee
directamente del `Blackboard::getInstance()`. Esto cambia el thread-local storage del
framework BHuman para apuntar al blackboard de Cognition **desde el hilo main**. Si
Cognition estÃ¡ escribiendo al blackboard en ese mismo instante (modificando una representaciÃ³n
en su propio hilo), hay una race condition. En modo one-shot el proceso muere inmediatamente
despuÃ©s de esta lectura, asÃ­ que la ventana temporal de la race es mÃ­nima. Pero en modo
persistente, esta funciÃ³n se llama cientos de veces por segundo, aumentando masivamente
la probabilidad de que la race produzca un null dereference.

**Evidencia esperada si es esta causa:**
- El crash ocurre DESPUÃ‰S de que los primeros frames funcionen (no en init)
- El crash es en `readDecisions()`, no en I/O
- El stack trace apunta a `Blackboard::getInstance()` o a un `static_cast` de una
  representaciÃ³n
- El crash se correlaciona con el momento exacto en que Cognition estÃ¡ corriendo su frame

**DiagnÃ³stico:**

Reemplazar temporalmente `readDecisions()` con una versiÃ³n que solo usa el snapshot del
`RLBridgeConsole` (que SÃ estÃ¡ protegido por mutex) y nunca hace fallback al blackboard
directo.

**Fix definitivo:**

Archivo: `bridge/bhuman_bridge.cpp`

```cpp
// En readDecisions(), lÃ­nea 1602, REEMPLAZAR la condiciÃ³n del fallback:
if(!snapshotHadDecisionData && cognition_)
{
    // ... direct blackboard fallback (ELIMINAR este bloque completo)
}

// CON:
if(!snapshotHadDecisionData)
{
    // En modo persistente, el snapshot del RLBridgeConsole siempre debe
    // tener datos porque waitForCognitionFrameFinished() ya esperÃ³ a que
    // Cognition terminara y emitiera sus representaciones.
    // Si no los tiene, es un error de wiring, no una seÃ±al para hacer
    // lectura cross-thread insegura.
    result["debug_provider_game_state"] = JsonValue("snapshot_empty");
    result["debug_provider_skill_request"] = JsonValue("snapshot_empty");
    result["debug_provider_motion_request"] = JsonValue("snapshot_empty");
    result["debug_provider_motion_info"] = JsonValue("snapshot_empty");
    result["debug_provider_activation_graph"] = JsonValue("snapshot_empty");
    result["debug_provider_behavior_status"] = JsonValue("snapshot_empty");
    result["debug_provider_strategy_status"] = JsonValue("snapshot_empty");
    std::lock_guard<std::mutex> lock(g_ioMutex);
    std::cerr << "[bridge] readDecisions: snapshot empty â€” skipping unsafe"
                 " blackboard fallback in persistent mode" << std::endl;
}
```

**Nota:** Este fix solo aplica si el snapshot del debug channel SÃ estÃ¡ poblado (que
es el caso actual â€” las representaciones llegan por el canal `handleMessage` del
`RLBridgeConsole`). Si el snapshot no se llena, hay que verificar que BHuman estÃ©
configurado para emitir las representaciones necesarias como mensajes debug.

**Volumen:** ~20 lÃ­neas eliminadas, ~15 nuevas.

---

### Causa E: TerminaciÃ³n abrupta del proceso cuando Python cierra pipes

**Mecanismo:**
Cuando Python llama `shutdown()` o cuando ocurre un error, cierra los pipes. En `main()`
de C++ (L1931), `std::getline(std::cin, line)` recibe EOF y el bucle termina. Pero los hilos
de BHuman (Cognition, Motion, Debug) siguen corriendo. Si `main()` retorna y se ejecutan
los destructores globales (del CRT, de estÃ¡ticos de BHuman) mientras esos hilos aÃºn estÃ¡n
activos, cualquier acceso a memoria liberada produce `0xc0000005`.

**Evidencia esperada si es esta causa:**
- El crash ocurre durante el **shutdown o despuÃ©s del Ãºltimo frame**
- stderr muestra `[bridge] shutdown: begin` pero no `[bridge] Shutdown` (final)
- El crash es en destructores o `atexit` handlers

**DiagnÃ³stico:**

Agregar `std::cerr << "[bridge] main: loop exited"` justo despuÃ©s del while loop (L2009)
y verificar si el crash ocurre despuÃ©s de ese punto.

**Fix definitivo:**

Archivo: `bridge/bhuman_bridge.cpp`

```cpp
// En main(), DESPUÃ‰S del while loop (lÃ­nea 2009), REEMPLAZAR:
    }

    std::cerr << "[bridge] Exiting" << std::endl;
    return 0;
}

// CON:
    }

    // Shutdown cooperativo ANTES de salir del main.
    // Sin esto, los destructores globales de C++ se ejecutan mientras
    // los hilos de BHuman aÃºn estÃ¡n activos, causando access violation.
    {
        std::lock_guard<std::mutex> lock(g_ioMutex);
        std::cerr << "[bridge] main: loop exited, starting cooperative shutdown"
                  << std::endl;
    }
    bridge.shutdown();  // Esto ya existe si se recibiÃ³ "shutdown", pero puede
                        // no ejecutarse si el loop terminÃ³ por EOF (pipe roto).
    {
        std::lock_guard<std::mutex> lock(g_ioMutex);
        std::cerr << "[bridge] Exiting" << std::endl;
    }
    return 0;
}
```

Adicionalmente, robustecer la condiciÃ³n del loop:
```cpp
// LÃ­nea 1931, REEMPLAZAR:
while (std::getline(std::cin, line))

// CON:
while (std::cin.good() && std::getline(std::cin, line))
```

Y agregar un guard en el destructor de BridgeRunner:
```cpp
// LÃ­nea 757, REEMPLAZAR:
~BridgeRunner() { shutdown(); }

// CON:
~BridgeRunner()
{
    try { shutdown(); }
    catch (...) {}
}
```

**Volumen:** ~10 lÃ­neas en `bhuman_bridge.cpp`.

---

## 5. Orden de diagnÃ³stico recomendado

El diagnÃ³stico debe hacerse **uno a la vez**, en el orden que maximiza la probabilidad
de encontrar la causa con el mÃ­nimo esfuerzo. El orden estÃ¡ basado en la historia del
proyecto (el crash original fue durante frames, no durante init).

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Paso 1: Aplicar Fix E (shutdown cooperativo)                â”‚
â”‚         Esto es SIEMPRE necesario, sin importar la causa.   â”‚
â”‚         Si no se hace, los tests pueden dar falsos          â”‚
â”‚         positivos (el crash en shutdown oculta un fix real).â”‚
â”‚                                                             â”‚
â”‚ Paso 2: Aplicar Fix A (mutex de I/O)                        â”‚
â”‚         Si el problema persiste â†’ Causa A descartada.       â”‚
â”‚         Si se resuelve â†’ Causa A confirmada. FIN.           â”‚
â”‚                                                             â”‚
â”‚ Paso 3: Cambiar CREATE_NO_WINDOW â†’ CREATE_NEW_CONSOLE       â”‚
â”‚         Si el problema persiste â†’ Causa B descartada.       â”‚
â”‚         Si se resuelve â†’ Causa B confirmada. FIN.           â”‚
â”‚                                                             â”‚
â”‚ Paso 4: Stderr a archivo (no pipe)                          â”‚
â”‚         Si el problema persiste â†’ Causa C descartada.       â”‚
â”‚         Si se resuelve â†’ Causa C confirmada. FIN.           â”‚
â”‚                                                             â”‚
â”‚ Paso 5: Eliminar blackboard fallback directo                â”‚
â”‚         Si el problema persiste â†’ CombinaciÃ³n de causas.    â”‚
â”‚         Si se resuelve â†’ Causa D confirmada. FIN.           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

> [!IMPORTANT]
> **Aplicar los 5 fixes juntos es seguro y recomendado.** No se contradicen entre sÃ­.
> El orden de arriba es para diagnÃ³stico aislado si se quiere entender cuÃ¡l era la causa.
> Si el objetivo es solo que funcione, aplicar los 5 y correr los smokes.

---

## 6. Pipeline de testing

### Fase T0 â€” Test manual de la reproducciÃ³n del crash

**Objetivo:** Confirmar que el crash sigue reproduciÃ©ndose antes de aplicar fixes.

**Comando:**
```powershell
python smoke_test_rlbridge_persistent_basic.py
```

**Resultado esperado:** FAIL con access violation o timeout.

**Si pasa:** El crash ya fue resuelto por otro cambio reciente. Documentar y continuar
a Fase T3.

---

### Fase T1 â€” Aplicar todos los fixes y re-testear

**Archivos a modificar:**

1. `bridge/bhuman_bridge.cpp` (copia local del repo RL)
   - Agregar `g_ioMutex` global
   - Proteger todas las escrituras a cout/cerr
   - Robustecer el exit del main loop
   - Agregar shutdown cooperativo post-loop
   - Eliminar fallback directo a blackboard

2. `SabanaHerons2026/Src/Apps/RLBridge/bhuman_bridge.cpp` (Ã¡rbol externo compilable)
   - Mismos cambios que arriba (debe ser idÃ©ntico)

3. `bridge/bridge_protocol.py`
   - Cambiar `CREATE_NO_WINDOW` â†’ `CREATE_NEW_CONSOLE` (o `STARTUPINFO` oculto)
   - Cambiar `stderr=subprocess.PIPE` â†’ `stderr=subprocess.STDOUT`
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

### Fase T2 â€” Validar que one-shot no se rompiÃ³

**Objetivo:** Confirmar que los cambios no rompieron el fallback.

**Comando:**
```powershell
python smoke_test_environment_bridge.py
```

**Resultado esperado:** PASS (puede usar oneshot o pipe, ambos son vÃ¡lidos).

Adicionalmente, verificar la suite completa:
```powershell
python smoke_suite_option1.py --suite gating
```

---

### Fase T3 â€” Crear smoke test de estrÃ©s para pipe persistente

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

### Fase T4 â€” IntegraciÃ³n con environment

**Objetivo:** Validar que `NaoSoccerEnv` funciona con pipe_process.

No hay que cambiar environment.py. Solo hay que cambiar `runtime_paths.py` para que
`pipe_process` sea la preferencia en Windows cuando el binario existe:

```python
# runtime_paths.py, lÃ­nea 94-104, ya es asÃ­:
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

### Fase T5 â€” Training loop quick debug

**Test:**
```powershell
set BHUMAN_BRIDGE_TRANSPORT=pipe_process
python train.py --quick-debug
```

**MÃ©tricas a observar en el watchdog:**
- `env_step_mean`: debe caer drÃ¡sticamente (de ~56ms a ~5-15ms)
- `bridge_decision_time_us`: debe mantenerse en ~2000-3000us
- No debe haber crashes ni fallback silencioso a oneshot

---

## 7. Plan de integraciÃ³n final

### Si los 5 fixes resuelven el crash:

1. **Mantener `oneshot_file_fallback` como fallback.** No eliminarlo.
   Los cambios en `bridge_protocol.py` ya hacen esto (L367-377, L458-496).

2. **No cambiar el `.env` por defecto.** El usuario puede forzar pipe con:
   ```
   BHUMAN_BRIDGE_TRANSPORT=pipe_process
   ```

3. **Actualizar `runtime_paths.py`** para que `pipe_process` sea el default
   en Windows cuando el binario compilado existe (esto ya es asÃ­ en L94-103).

4. **Actualizar `BHUMAN_PERSISTENT_BRIDGE_PLAN.md`** con las mÃ©tricas nuevas.

5. **Actualizar `RLBRIDGE_INTERVENTION_LOG.md`** con la causa raÃ­z confirmada.

6. **Actualizar `ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md` Â§9.2** para quitar
   la restricciÃ³n de "no reintroducir pipe_process" y reemplazarla con la
   nota de que pipe_process requiere los 5 fixes aplicados.

### Respuesta a las 4 preguntas obligatorias (Â§7 de la guÃ­a):

1. **Â¿Estoy cambiando OpciÃ³n 1, 2, o ambas?** â†’ Solo OpciÃ³n 1 (transporte)
2. **Â¿Rompe el baseline?** â†’ No. Oneshot sigue siendo fallback
3. **Â¿Inventa MotionInfo?** â†’ No. No toca la lectura de representaciones
4. **Â¿El smoke sigue siendo honesto?** â†’ SÃ­. Los smokes existentes no cambian

---

## 8. Resumen de archivos a modificar

| Archivo | Cambio | Causa que resuelve |
|---|---|---|
| `bridge/bhuman_bridge.cpp` | Mutex I/O global | A |
| `bridge/bhuman_bridge.cpp` | Shutdown cooperativo post-loop | E |
| `bridge/bhuman_bridge.cpp` | `cin.good()` guard | E |
| `bridge/bhuman_bridge.cpp` | Eliminar blackboard fallback directo | D |
| `SabanaHerons2026/.../bhuman_bridge.cpp` | Mismos cambios (copia sincronizada) | A, D, E |
| `bridge/bridge_protocol.py` | `CREATE_NO_WINDOW` â†’ `CREATE_NEW_CONSOLE` | B |
| `bridge/bridge_protocol.py` | `stderr=STDOUT` (merged) | C |
| `bridge/bridge_protocol.py` | Ajustar reader threads | C |
| `smoke_test_rlbridge_persistent_stress.py` | **[NUEVO]** stress test | testing |

---

## 9. Tiempos esperados post-fix

| MÃ©trica | Oneshot actual | Pipe post-fix (R5 5000) | Pipe post-fix (R9 275HX) |
|---|---|---|---|
| step latency | ~56 ms | ~5-12 ms | ~3-8 ms |
| 96-step episode | ~5.4 s | ~0.5-1.2 s | ~0.3-0.8 s |
| 455-step Stage 0 | ~25.5 s | ~2.3-5.5 s | ~1.4-3.6 s |
| 1M steps training | ~31 h | ~2.8-6.7 h | ~1.7-4.4 h |

> [!IMPORTANT]
> La mayor ganancia viene del fix del pipe, no del hardware.
> El Ryzen 9 AI 275HX sobre pipe persistente serÃ­a ~**8-18Ã— mÃ¡s rÃ¡pido** que el
> Ryzen 5 5000 sobre oneshot actual.

---

## 10. QuÃ© hacer si las 5 hipÃ³tesis estÃ¡n equivocadas

Si se aplican los 5 fixes del Â§4 y el crash persiste, las hipÃ³tesis A-E no eran
la causa raÃ­z (o lo son parcialmente). Esta secciÃ³n define:

1. **6 causas adicionales** que no estÃ¡n cubiertas arriba
2. **4 herramientas de diagnÃ³stico de escalaciÃ³n** para obtener evidencia real
3. **3 tests de reproducciÃ³n mÃ­nima** que eliminan variables

---

### 10.1. Causas adicionales no cubiertas en las hipÃ³tesis A-E

---

#### Causa F: Variables `static` locales en el JSON parser son thread-unsafe

**Problema descubierto en el cÃ³digo:**

En `bhuman_bridge.cpp` lÃ­neas 187-193, 201, la clase `JsonValue` tiene:

```cpp
const std::string& asString() const {
    static std::string empty;        // â† COMPARTIDO entre hilos
    return isString() ? std::get<std::string>(data) : empty;
}
const JsonArray& asArray() const {
    static JsonArray empty;          // â† COMPARTIDO entre hilos
    return isArray() ? std::get<JsonArray>(data) : empty;
}
const JsonObject& asObject() const {
    static JsonObject empty;         // â† COMPARTIDO entre hilos
    return isObject() ? std::get<JsonObject>(data) : empty;
}
const JsonValue& operator[](const std::string& key) const {
    static JsonValue null;           // â† COMPARTIDO entre hilos
    // ...
}
const JsonValue& operator[](size_t idx) const {
    static JsonValue null;           // â† COMPARTIDO entre hilos
    // ...
}
```

En C++11+, la inicializaciÃ³n de `static` locales es thread-safe, pero **el acceso
posterior no lo es**. Si un hilo del framework BHuman de alguna forma invoca cÃ³digo
que usa `JsonValue` (poco probable pero posible si hay logging con JSON), dos hilos
podrÃ­an estar leyendo/escribiendo la misma referencia `static`.

**CÃ³mo diagnosticarlo:**
- Reemplazar todas las variables `static` locales con variables de instancia o
  con `thread_local static`:
  ```cpp
  const std::string& asString() const {
      thread_local std::string empty;  // Cada hilo tiene su propia copia
      return isString() ? std::get<std::string>(data) : empty;
  }
  ```

**Probabilidad:** Baja (solo si BHuman invoca cÃ³digo que toca JsonValue).

---

#### Causa G: El CRT de MSVC usa `_isatty()` para decidir el modo de buffering

**Mecanismo:**
Cuando `stdout` es un pipe en Windows, el CRT de MSVC cambia `std::cout` a buffering
completo (no line-buffered). Esto significa que `std::endl` puede no vaciar el buffer
inmediatamente. El `flush()` explÃ­cito lo fuerza, pero si hay cÃ³digo interno de BHuman
que escribe a `stdout` o `stderr` sin `flush()`, los datos se acumulan y se envÃ­an en
bloques parciales. Python, leyendo desde el otro lado del pipe con un reader thread,
puede recibir JSON cortado a mitad de lÃ­nea.

Si `_recv()` del lado Python procesa JSON truncado y lo descarta, pero el bridge C++
ya avanzÃ³ su estado interno, se produce una desincronizaciÃ³n de protocolo que puede
llevar a comportamiento indefinido en frames posteriores.

**CÃ³mo diagnosticarlo:**
- En `bhuman_bridge.cpp`, agregar despuÃ©s de `main()` (lÃ­nea 1921):
  ```cpp
  // Forzar stdout completamente unbuffered programÃ¡ticamente
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
  ```
- Verificar si esto cambia el comportamiento.

**Probabilidad:** Media. El `cout.flush()` explÃ­cito deberÃ­a ser suficiente, pero
`setvbuf` afecta tambiÃ©n el nivel C de stdio que BHuman podrÃ­a usar internamente.

---

#### Causa H: El framework BHuman usa `ASSERT()` o `OUTPUT_ERROR()` con formateo de consola

**Mecanismo:**
El framework BHuman define macros como `ASSERT()`, `OUTPUT_ERROR()`, `OUTPUT_WARNING()`
que internamente pueden llamar a funciones de formateo de error que asumen una consola
Win32 activa. En modo headless (sin ventana), estas macros pueden intentar abrir una
ventana de diÃ¡logo de error (`MessageBox`) o acceder a un handle de consola nulo.

En modo one-shot esto no se observa porque el proceso muere antes de que se triggereen
las condiciones de assert. En modo persistente, con mÃºltiples frames, la probabilidad
de que un assert interno falle se multiplica.

**CÃ³mo diagnosticarlo:**
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
En `processFrame()` (L903-985), cuando `operationalWarmupDone_` es `false`, se envÃ­an
**4 frames de bootstrap** (initial â†’ ready â†’ set â†’ playing). Cada uno invoca:
- `bridgeSender.bin()` Ã— 11 representaciones
- `bridgeSender.send(true)`
- `waitForCognitionFrameFinished()` con timeout de 1000ms

Si el stack estÃ¡ndar de un hilo de Windows es de 1 MB y la serializaciÃ³n BHuman
acumula objetos grandes en el stack (como `GroundTruthWorldState` con todos los
jugadores), 4 iteraciones Ã— 11 representaciones puede acercarse al lÃ­mite,
especialmente si BHuman internamente hace recursiÃ³n profunda durante el
mÃ³dulo graph runner.

**CÃ³mo diagnosticarlo:**
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

**Probabilidad:** Baja-Media. Depende de la profundidad del mÃ³dulo graph.

---

#### Causa J: El antivirus o Windows Defender interfiere con el pipe I/O

**Mecanismo:**
Windows Defender y otros antivirus pueden interceptar y escanear I/O de procesos
que se comunican por pipes. Si el antivirus demora o bloquea una escritura/lectura:
- Python puede ver un timeout y matar el proceso
- El proceso C++ puede recibir un error de I/O inesperado

Esto NO causa un access violation directamente, pero sÃ­ puede causar la secuencia
deadlock â†’ kill â†’ crash (similar a Causa C pero diferente raÃ­z).

**CÃ³mo diagnosticarlo:**
- Agregar el directorio del proyecto a las exclusiones de Windows Defender:
  ```powershell
  Add-MpPreference -ExclusionPath "C:\ALEJO"
  Add-MpPreference -ExclusionProcess "RLBridge.exe"
  ```
- Temporalmente desactivar "Real-Time Protection" y re-testear
- Usar Process Monitor (ver Â§10.2) para ver si hay delays en operaciones I/O

**Probabilidad:** Baja (no explicarÃ­a un access violation, sÃ­ un timeout).

---

#### Causa K: Heap corruption latente en el framework BHuman

**Mecanismo:**
Si existe un bug de memoria (buffer overrun, use-after-free, double free) en el cÃ³digo
de BHuman que solo se manifiesta en modo persistente (porque en one-shot el proceso muere
antes de que el heap corrupto se reutilice), el access violation aparecerÃ­a en un lugar
completamente diferente del cÃ³digo que causÃ³ la corrupciÃ³n.

Este es el peor escenario: la direcciÃ³n del crash (`ExceptionAddress`) no tiene relaciÃ³n
directa con la causa real.

**CÃ³mo diagnosticarlo:**
- Compilar con Address Sanitizer (ASan):
  ```cmake
  # En CMakeLists.txt de RLBridge:
  target_compile_options(RLBridge PRIVATE /fsanitize=address)
  target_link_options(RLBridge PRIVATE /fsanitize=address)
  ```
- O usar Application Verifier (ver Â§10.2)
- El output de ASan mostrarÃ¡ el sitio ORIGINAL de la corrupciÃ³n, no el crash

**Probabilidad:** Media. DependerÃ­a de bugs preexistentes en BHuman.

---

### 10.2. Herramientas de diagnÃ³stico de escalaciÃ³n

Si las causas A-K no se confirman con los tests propuestos, usar estas herramientas
para obtener evidencia real del sistema operativo.

---

#### Herramienta 1: WinDbg (Debugger de Windows)

**Para quÃ©:** Obtener el **stack trace real** del crash, con nombre de funciÃ³n y
lÃ­nea exacta. Sin esto, todas las hipÃ³tesis son especulaciones.

**CÃ³mo usarlo:**

1. Instalar WinDbg Preview desde Microsoft Store
2. Modificar `bridge_protocol.py` para no matar el proceso al crash:
   ```python
   # Temporalmente en start(), agregar despuÃ©s de Popen:
   print(f"RLBridge PID: {self._process.pid}")
   input("Attach WinDbg and press Enter...")
   ```
3. Abrir WinDbg â†’ File â†’ Attach to Process â†’ PID del bridge
4. En WinDbg: `g` (go) para continuar la ejecuciÃ³n
5. Cuando crashee, WinDbg atrapa la excepciÃ³n. Ejecutar:
   ```
   !analyze -v
   k
   .ecxr
   k
   ```
6. Copiar el output completo â€” contiene:
   - MÃ³dulo exacto donde ocurriÃ³ el crash
   - Nombre de funciÃ³n
   - Offset del crash dentro de la funciÃ³n
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

**Esto confirmarÃ­a O descartarÃ­a** todas las hipÃ³tesis a la vez.

---

#### Herramienta 2: Application Verifier (AppVerif)

**Para quÃ©:** Detectar heap corruption, handle leaks, y race conditions que
no se manifiestan directamente como el crash observado.

**CÃ³mo usarlo:**

1. Instalar Application Verifier desde Windows SDK
2. Ejecutar `appverif.exe` â†’ agregar `RLBridge.exe`
3. Activar las siguientes verificaciones:
   - Heap (detecta buffer overruns, use-after-free)
   - Handles (detecta handle leaks y doble-cierre)
   - Locks (detecta deadlocks y lock ordering)
4. Correr el smoke test â€” AppVerif harÃ¡ que el crash sea mÃ¡s determinÃ­stico
   y mostrarÃ¡ la causa raÃ­z

**Output esperado:** Un `VerifierStop` con el site de la primera violaciÃ³n,
que puede ser diferente del crash observable.

---

#### Herramienta 3: Process Monitor (Sysinternals ProcMon)

**Para quÃ©:** Ver TODA la actividad I/O del proceso segundo a segundo, incluyendo
operaciones de archivo, registro, red, y pipes. Detecta:
- Pipes que se llenan
- Escrituras que fallan
- DLLs que se cargan tarde
- Accesos a rutas que no existen

**CÃ³mo usarlo:**

1. Descargar Process Monitor de Sysinternals
2. Filtrar por `Process Name = RLBridge.exe`
3. Correr el smoke test
4. Buscar operaciones con resultado `BUFFER OVERFLOW`, `ACCESS DENIED`,
   `NAME NOT FOUND` justo antes del crash

**Este test es especialmente Ãºtil** si la causa es J (antivirus) o una
dependencia de DLL que falla.

---

#### Herramienta 4: Crash dump automÃ¡tico (sin debugger)

**Para quÃ©:** Capturar un minidump cuando el crash ocurre, sin necesidad
de tener WinDbg abierto en el momento.

**CÃ³mo configurarlo:**

```powershell
# Registrar RLBridge.exe para generar crash dumps automÃ¡ticos:
New-Item -Path "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\RLBridge.exe" -Force
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\RLBridge.exe" `
    -Name "DumpFolder" -Value "C:\ALEJO\RL\bridge\crashdumps" -Type String
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\RLBridge.exe" `
    -Name "DumpType" -Value 2 -Type DWord  # Full dump
New-Item -Path "C:\ALEJO\RL\bridge\crashdumps" -ItemType Directory -Force
```

DespuÃ©s del crash, abrir el `.dmp` con WinDbg o Visual Studio para anÃ¡lisis.

---

### 10.3. Tests de reproducciÃ³n mÃ­nima (eliminan variables)

Si las herramientas de Â§10.2 no estÃ¡n disponibles o son lentas, estos tests
ayudan a aislar el problema quitando capas.

---

#### Test M1: Pipe persistente desde cmd.exe puro (sin Python)

**Objetivo:** Â¿El crash es del bridge C++ o del wiring con Python?

Crear archivo `bridge/test_pipe_persistent.cmd`:
```cmd
@echo off
echo [TEST] Pipe persistente desde cmd.exe

echo {"type":"init","scenario":"2D","location":"4v4_Scaled","team_number":1,"player_number":1,"num_players":4,"bridge_mode":"persistent_2d_cognition"} > %TEMP%\rlbridge_pipe_test.jsonl
echo {"type":"frame","frame_id":1,"timestamp_ms":33,"ball":{"x":200,"y":0,"vx":0,"vy":0},"own_team":[{"number":1,"x":0,"y":0,"theta":0,"upright":true},{"number":2,"x":-800,"y":500,"theta":0,"upright":true},{"number":3,"x":-1200,"y":-500,"theta":0,"upright":true},{"number":4,"x":-1800,"y":0,"theta":0,"upright":true}],"opp_team":[{"number":1,"x":1200,"y":0,"theta":3.14,"upright":true},{"number":2,"x":1500,"y":700,"theta":3.14,"upright":true},{"number":3,"x":1500,"y":-700,"theta":3.14,"upright":true},{"number":4,"x":2200,"y":0,"theta":3.14,"upright":true}],"game_state":"playing","own_score":0,"opp_score":0} >> %TEMP%\rlbridge_pipe_test.jsonl
echo {"type":"frame","frame_id":2,"timestamp_ms":66,"ball":{"x":200,"y":0,"vx":0,"vy":0},"own_team":[{"number":1,"x":0,"y":0,"theta":0,"upright":true},{"number":2,"x":-800,"y":500,"theta":0,"upright":true},{"number":3,"x":-1200,"y":-500,"theta":0,"upright":true},{"number":4,"x":-1800,"y":0,"theta":0,"upright":true}],"opp_team":[{"number":1,"x":1200,"y":0,"theta":3.14,"upright":true},{"number":2,"x":1500,"y":700,"theta":3.14,"upright":true},{"number":3,"x":1500,"y":-700,"theta":3.14,"upright":true},{"number":4,"x":2200,"y":0,"theta":3.14,"upright":true}],"game_state":"playing","own_score":0,"opp_score":0} >> %TEMP%\rlbridge_pipe_test.jsonl
echo {"type":"frame","frame_id":3,"timestamp_ms":99,"ball":{"x":200,"y":0,"vx":0,"vy":0},"own_team":[{"number":1,"x":0,"y":0,"theta":0,"upright":true},{"number":2,"x":-800,"y":500,"theta":0,"upright":true},{"number":3,"x":-1200,"y":-500,"theta":0,"upright":true},{"number":4,"x":-1800,"y":0,"theta":0,"upright":true}],"opp_team":[{"number":1,"x":1200,"y":0,"theta":3.14,"upright":true},{"number":2,"x":1500,"y":700,"theta":3.14,"upright":true},{"number":3,"x":1500,"y":-700,"theta":3.14,"upright":true},{"number":4,"x":2200,"y":0,"theta":3.14,"upright":true}],"game_state":"playing","own_score":0,"opp_score":0} >> %TEMP%\rlbridge_pipe_test.jsonl
echo {"type":"shutdown"} >> %TEMP%\rlbridge_pipe_test.jsonl

REM Ejecutar con PIPE real (no redirecciÃ³n de archivo):
type %TEMP%\rlbridge_pipe_test.jsonl | RLBridge.exe

echo Exit code: %ERRORLEVEL%
```

**InterpretaciÃ³n:**
- Si pasa con `exit code 0` â†’ **El crash es culpa de Python**, no del C++.
  Investigar `Popen`, `bufsize`, `creationflags`, o los reader threads.
- Si crashea tambiÃ©n aquÃ­ â†’ **El crash es del C++**, proceder con WinDbg.

**Nota:** El `|` (pipe) de cmd.exe crea un pipe anÃ³nimo real, idÃ©ntico al de Python.

---

#### Test M2: Pipe persistente pero con un solo frame (mÃ­nimo absoluto)

**Objetivo:** Â¿El crash ocurre en el primer frame o solo despuÃ©s de varios?

```cmd
@echo off
echo {"type":"init","scenario":"2D","location":"4v4_Scaled","team_number":1,"player_number":1,"num_players":4,"bridge_mode":"persistent_2d_cognition"} > %TEMP%\rlbridge_min.jsonl
echo {"type":"frame","frame_id":1,"timestamp_ms":33,"ball":{"x":200,"y":0,"vx":0,"vy":0},"own_team":[{"number":1,"x":0,"y":0,"theta":0,"upright":true}],"opp_team":[{"number":1,"x":1200,"y":0,"theta":3.14,"upright":true}],"game_state":"playing","own_score":0,"opp_score":0} >> %TEMP%\rlbridge_min.jsonl
echo {"type":"shutdown"} >> %TEMP%\rlbridge_min.jsonl

type %TEMP%\rlbridge_min.jsonl | RLBridge.exe
echo Exit code: %ERRORLEVEL%
```

**InterpretaciÃ³n:**
- Si pasa â†’ El crash ocurre con frames mÃºltiples (Causa A o D mÃ¡s probable)
- Si crashea â†’ El crash ocurre en el primer frame (Causa B, H, I, o interno)

---

#### Test M3: Bridge C++ con stdin redirigido desde archivo (no pipe)

**Objetivo:** Â¿El problema es los PIPES o simplemente el modo persistente?

```cmd
@echo off
REM Mismos datos que M1 pero con redirecciÃ³n de archivo, no pipe
RLBridge.exe < %TEMP%\rlbridge_pipe_test.jsonl > %TEMP%\rlbridge_out.txt 2>&1
echo Exit code: %ERRORLEVEL%
type %TEMP%\rlbridge_out.txt
```

**InterpretaciÃ³n:**
- Si M1 crashea pero M3 pasa â†’ **El problema ES el pipe** (vs. archivo). Causa C.
- Si ambos crashean â†’ El problema es el modo persistente en sÃ­ (no el transporte).
  Investigar el C++ (Causa D, H, I, K).
- Si M1 pasa y M3 pasa â†’ El problema es especÃ­fico de Python (wiring de `Popen`).

---

### 10.4. Ãrbol de decisiÃ³n completo

```
Â¿El crash se reproduce HOY?
â”œâ”€ NO â†’ Documentar, seguir a T3 (stress test)
â””â”€ SÃ
   â”‚
   â”œâ”€ Aplicar los 5 fixes (A-E) juntos
   â”‚  â”œâ”€ Â¿Crash desaparece? â†’ Ã‰XITO. Aislar cuÃ¡l fix fue el clave (opcional)
   â”‚  â””â”€ Â¿Crash persiste?
   â”‚     â”‚
   â”‚     â”œâ”€ Correr Test M1 (pipe desde cmd.exe puro)
   â”‚     â”‚  â”œâ”€ PASA â†’ Problema en Python. Investigar:
   â”‚     â”‚  â”‚         - reader threads
   â”‚     â”‚  â”‚         - bufsize
   â”‚     â”‚  â”‚         - creationflags
   â”‚     â”‚  â”‚         - timing de _recv()
   â”‚     â”‚  â””â”€ FALLA â†’ Problema en C++. Correr Test M3
   â”‚     â”‚     â”œâ”€ M3 PASA â†’ Problema especÃ­fico de PIPE. Verificar:
   â”‚     â”‚     â”‚             - Causa G (_isatty/buffering)
   â”‚     â”‚     â”‚             - Causa J (antivirus)
   â”‚     â”‚     â”‚             - setvbuf(stdout, NULL, _IONBF, 0)
   â”‚     â”‚     â””â”€ M3 FALLA â†’ Problema del modo persistente. Usar:
   â”‚     â”‚                    - WinDbg (Herramienta 1) para stack trace
   â”‚     â”‚                    - ASan (Causa K) para heap corruption
   â”‚     â”‚                    - Test M2 para aislar primer frame vs. multi
   â”‚     â”‚
   â”‚     â”œâ”€ Test M2 (un solo frame)
   â”‚     â”‚  â”œâ”€ PASA â†’ Problema acumulativo (frame N>1). Causas: A, D, F, K
   â”‚     â”‚  â””â”€ FALLA â†’ Problema en init/bootstrap. Causas: B, H, I
   â”‚     â”‚
   â”‚     â””â”€ Si nada de lo anterior funciona:
   â”‚        â†’ WinDbg para stack trace REAL
   â”‚        â†’ El stack trace descarta/confirma TODAS las hipÃ³tesis a la vez
   â”‚        â†’ Actualizar este plan con la causa descubierta
```

---

### 10.5. Protocolo de documentaciÃ³n cuando se encuentra la causa real

Cuando se identifique la causa raÃ­z (por cualquier vÃ­a), documentar:

1. **En `RLBRIDGE_INTERVENTION_LOG.md`:**
   - Causa raÃ­z exacta (cuÃ¡l de las A-K, o una nueva)
   - El stack trace o evidencia que la confirmÃ³
   - El fix aplicado
   - Los tests que validan el fix

2. **En `ENVIRONMENT_AND_ARCHITECTURE_GUIDE.md` Â§9.2:**
   - Actualizar la restricciÃ³n: ya no es "no reintroducir pipe_process"
   - Sino: "pipe_process requiere [los fixes especÃ­ficos aplicados]"

3. **En `BHUMAN_PERSISTENT_BRIDGE_PLAN.md` Â§1:**
   - Actualizar mÃ©tricas con los tiempos reales post-fix
   - Marcar las fases como completadas

> [!CAUTION]
> Si la causa resulta ser Causa K (heap corruption en BHuman), el fix puede requerir
> modificar cÃ³digo del framework BHuman en `SabanaHerons2026/Src/Libs/`, lo cual es
> significativamente mÃ¡s invasivo y requiere coordinaciÃ³n con el equipo de BHuman.
> En ese caso, mantener `oneshot_file_fallback` como ruta principal hasta que el fix
> en BHuman se valide exhaustivamente.

