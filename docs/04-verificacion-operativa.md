# Verificacion Operativa

## Rebuild

Desde `SabanaHerons2026`:

```bash
cmake --build Build/Linux/CMake/Develop --target SimulatedNao PythonController B-Human
```

Copiar el binding:

```bash
cp Build/Linux/Python/Develop/controller.cpython-312-x86_64-linux-gnu.so \
  Make/Python/pybh/controller.cpython-312-x86_64-linux-gnu.so
```

Reducir debug symbols para no superar el limite de GitHub:

```bash
strip --strip-debug \
  Build/Linux/Python/Develop/controller.cpython-312-x86_64-linux-gnu.so \
  Make/Python/pybh/controller.cpython-312-x86_64-linux-gnu.so
```

El `.so` actual queda alrededor de 74 MB.

## Verificar Python

Desde `RL`:

```bash
.venv/bin/python -m py_compile \
  config.py \
  pybh_backend.py \
  live_simrobot_probe.py \
  environment.py \
  smoke_test_skill_requests.py
```

## Smoke test rapido `cpp`

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
.venv/bin/python -u - <<'PY'
import numpy as np
from pybh_backend import PyBHBackend
from action import StructuredSkillAction, SkillType

b = PyBHBackend(mode="cpp", frames_per_step=1, obs_timeout_ms=2000)
try:
    print(b.reset(
        ball_pos=np.array([0.0, 0.0], dtype=np.float32),
        robot_pos=np.array([-900.0, 0.0], dtype=np.float32),
        robot_theta=0.0,
    ))
    print(b.step(StructuredSkillAction(
        skill_type=SkillType.walk,
        target_x=0.0,
        target_y=0.0,
        target_theta=0.0,
    )))
finally:
    b.close()
PY
```

Resultado esperado:

- `sim_enabled=True`;
- el robot cambia pose;
- no hay timeout en `rl_get_obs`.

## Verificacion SimRobot embebido 2D/3D con heuristica

Este comando es reproducible sin GUI:

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
.venv/bin/python -u - <<'PY'
import math
import numpy as np
from pybh_backend import PyBHBackend
import pybh.controller as pybh
from action import StructuredSkillAction, SkillType

def run(profile: str) -> None:
    b = PyBHBackend(mode="simrobot", simrobot_profile=profile, frames_per_step=20, obs_timeout_ms=2500)
    try:
        obs = b.reset(
            ball_pos=np.array([0.0, 0.0], dtype=np.float32),
            robot_pos=np.array([-900.0, 0.0], dtype=np.float32),
            robot_theta=0.0,
        )
        print(f"PROFILE {profile} RESET robot=({obs.robot_x:.1f},{obs.robot_y:.1f},{obs.robot_theta:.2f})")
        for i in range(6):
            dx = obs.ball_x - obs.robot_x
            dy = obs.ball_y - obs.robot_y
            if math.hypot(dx, dy) <= 260.0:
                action = StructuredSkillAction(skill_type=SkillType.shoot)
            else:
                action = StructuredSkillAction(
                    skill_type=SkillType.walk,
                    target_x=obs.ball_x,
                    target_y=obs.ball_y,
                    target_theta=math.atan2(dy, dx),
                )
            obs = b.step(action)
            payload = pybh.rl_get_obs(1, 1)
            print(
                profile,
                i + 1,
                payload.get("requested_skill"),
                payload.get("skill_behavior_skill_request"),
                payload.get("skill_behavior_motion_request"),
                payload.get("motion_engine_input_request"),
                payload.get("motion_engine_effective_request"),
                payload.get("motion_engine_phase"),
                payload.get("sim_enabled"),
            )
    finally:
        b.close()

run("2d")
run("3d")
PY
```

Resultado observado en esta sesion:

```text
PROFILE 2d RESET robot=(-900.0,-0.0,0.00) ball=(-0.0,-0.0) sim_enabled=False
PROFILE 2d STEP 1 skill=walkTo ... skillctl=(skill=2,motion=4) engine=(in=4,eff=4,phase=1)
PROFILE 2d STEP 6 skill=walkTo robot=(-819.1,131.9,1.24) ... skillctl=(skill=2,motion=4) engine=(in=4,eff=4,phase=1)

PROFILE 3d RESET robot=(-900.0,-0.0,-1.57) ball=(-0.0,-0.0) sim_enabled=False
PROFILE 3d STEP 1 skill=walkTo ... skillctl=(skill=2,motion=4) engine=(in=4,eff=4,phase=1)
PROFILE 3d STEP 6 skill=walkTo ... skillctl=(skill=2,motion=4) engine=(in=4,eff=4,phase=2)
```

Interpretacion:

- `sim_enabled=False`: no se esta usando `RLSim2D`.
- `skill=2`: `SkillRequest::walk`.
- `motion=4`: `MotionRequest::walkToPose`.
- `engine in/eff=4`: `MotionEngine` recibio y mantuvo `walkToPose`.
- `phase=1/2`: motor de movimiento activo.
- joints de piernas cambian: hay salida de caminata, no root-assist.

## Verificacion visible

Para ver el agente en la ventana:

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
.venv/bin/python live_simrobot_probe.py \
  --launch \
  --policy heuristic \
  --simrobot-profile 3d \
  --steps 1000
```

Luego en SimRobot:

```text
F5
```

Tambien funciona con:

```bash
--simrobot-profile 2d
```

La salida esperada debe contener:

```text
action=walkToField(...)
skillctl=(skill=2,motion=4,...)
engine=(in=4,eff=4,...)
```

Si `obs_ready=0` o `frame` no avanza:

- la ventana no esta corriendo;
- falta F5;
- el proceso visible no cargo la escena esperada;
- hay otra instancia de SimRobot consumiendo el mismo estado compartido.

## Limpieza de memoria compartida

Solo si un proceso fue matado de forma dura y las pruebas quedan raras:

```bash
rm -f /dev/shm/pybh_rl_shared_state_v2
```

Con version 12, el mutex es robusto y normalmente no deberia hacer falta.

## Matriz completa de skills

Hay un harness reproducible para validar todas las skills y generar comparativo cuantitativo entre `2d` y `3d`:

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
.venv/bin/python validate_simrobot_matrix.py \
  --steps-per-skill 20 \
  --output docs/06-simrobot-skill-matrix.md
```

Ese script:

- valida `stand`, `walk`, `shoot`, `pass`, `dribble`, `block`, `mark`, `observe`;
- comprueba que `requested_skill`, `SkillBehaviorControl` y `MotionEngine` queden consistentes;
- registra `reset_wall_ms`, `step_wall_ms`, `step_cpu_ms` y `rss_kb`;
- deja el reporte final en [06-simrobot-skill-matrix.md](./06-simrobot-skill-matrix.md).
