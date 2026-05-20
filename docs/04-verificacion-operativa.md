# Verificación Operativa

## Rebuild del binding

Desde `SabanaHerons2026/Make/Python`:

```bash
python3 setup.py build_ext --inplace
```

Ese es el camino más confiable porque deja actualizado el módulo que consume el
`PYTHONPATH` de `RL`.

## Verificar Python

Desde `RL`:

```bash
python3 -m py_compile \
  config.py \
  pybh_backend.py \
  environment.py \
  smoke_test_skill_requests.py \
  heuristic_agent_experiment.py \
  validate_simrobot_matrix.py
```

## Verificar superficie expuesta

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 - <<'PY'
import pybh.controller as pybh
print("has_reset_world_extended", hasattr(pybh, "rl_reset_world_extended"))
PY
```

Esperado:

- `has_reset_world_extended = True`

## Smoke embebido `2d` / `3d`

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 smoke_test_skill_requests.py --backend simrobot --simrobot-profile 2d
```

Y:

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 smoke_test_skill_requests.py --backend simrobot --simrobot-profile 3d
```

Esperado:

- `obs_ready=1`
- `frame` avanza
- `requested_skill` coincide con la skill emitida

## Smoke visible

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 smoke_test_skill_requests.py --backend simrobot_visible --simrobot-profile 3d --launch-simrobot
```

## Verificación heurística

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 heuristic_agent_experiment.py --episodes 3 --no-plot --backend simrobot --simrobot-profile 2d
```

## Matriz de skills

```bash
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 validate_simrobot_matrix.py \
  --steps-per-skill 20 \
  --output docs/06-simrobot-skill-matrix.md
```

## Limpieza de shared memory

Solo si un proceso nativo murió mal:

```bash
rm -f /dev/shm/pybh_rl_shared_state_v3
```

La versión actual del bloque compartido es `25`.
